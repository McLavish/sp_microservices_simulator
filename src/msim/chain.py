from collections.abc import Callable
from dataclasses import dataclass, field

from msim.caller import ClientRuntime
from msim.core import DropReason, RootRequest, TimeDuration, TimePoint
from msim.service import (
    AttemptControl,
    AttemptDoneCallback,
    ServiceConfig,
    ServiceRuntime,
    ServiceTime,
)
from msim.simulator import Event, Simulator


@dataclass
class DownstreamEdge:
    client: ClientRuntime
    target: "ServiceNodeRuntime"
    # Call wave. A node's edges with the SAME wave are launched together (the
    # concurrent fan-out); waves run in ascending order, and a wave starts only
    # once every call of the previous one has succeeded. So a handler that calls
    # B, then C, then D is three edges on waves 0, 1, 2, and one that makes four
    # concurrent calls followed by three more is waves 0 and 1. Only the order
    # of the values matters (gaps are fine), and when every edge of a node
    # carries the same value -- the default -- the node behaves exactly as it
    # did before waves existed: one concurrent wave.
    wave: int = 0

    def __post_init__(self) -> None:
        # bool is an int: `wave=True` would silently mean "wave 1".
        if isinstance(self.wave, bool) or not isinstance(self.wave, int) or self.wave < 0:
            raise ValueError(f"wave must be an int >= 0, got {self.wave!r}")


class _FanOutAttempt:
    """One logical attempt at a `ServiceNodeRuntime`, as an explicit state machine.

    The lifecycle is: local service stage -> (optionally) fan out to the
    downstream edges, one call wave after the other -> settle. It settles
    exactly once, through `_settle`, on whichever of these happens first:

    * the local stage expires against the deadline, or the local service fails;
    * the node has no downstreams and the local stage succeeds;
    * every downstream of the last wave succeeds (AND semantics);
    * any downstream fails (short-circuit, but only after every edge of its wave
      is launched so downstream load does not depend on edge order); the waves
      that had not started are never launched;
    * the deadline guard fires;
    * an upstream cancels this node.

    This was a nest of seven closures over eight mutable locals, where the
    lifetime rules were implicit in definition order — and every bug found in it
    was a state-ordering mistake. Here the state is named, inspectable, and each
    transition is one method.
    """

    def __init__(
        self,
        node: "ServiceNodeRuntime",
        svc: ServiceRuntime,
        sim: Simulator,
        deadline: TimePoint | None,
        finalize: Callable[[bool, TimeDuration, DropReason], None],
        begin_at: TimePoint,
        release_worker: Callable[[], None],
        register_cancel: Callable[[Callable[[], None] | None], None],
    ) -> None:
        self.node = node
        self.svc = svc
        self.sim = sim
        self.deadline = deadline
        self.begin_at = begin_at
        self._finalize = finalize
        self._release_worker = release_worker
        self._register_cancel = register_cancel

        #: The single exactly-once guard. Every settling path sets it via _commit.
        self.settled = False
        self._local_event: Event | None = None
        self._deadline_event: Event | None = None
        #: The node's edges grouped by `DownstreamEdge.wave`, ascending, and the
        #: wave currently in flight. One group unless the edges carry more than
        #: one distinct wave value.
        self._waves: list[list[DownstreamEdge]] = []
        self._wave_index = 0
        #: The pending start of the next wave (scheduled, not yet run).
        self._wave_event: Event | None = None
        self._remaining = 0
        #: True only inside the launch loop, so a synchronous rejection defers
        #: its short-circuit until every edge of the wave has been launched.
        self._launching = False
        self._sync_failure: RootRequest | None = None
        self._launched: list[RootRequest] = []

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Register teardown, then schedule the local service stage.

        Registration happens FIRST so the node is cancellable throughout its
        whole lifecycle — during local service, and for leaf nodes that never
        reach the fan-out block at all.
        """
        self._register_cancel(self.cancel)
        base_time = self.svc._sample_service_time(self.begin_at, self.sim, self.node.service_time)
        end_local = (
            min(self.begin_at + base_time, self.deadline)
            if self.deadline is not None
            else self.begin_at + base_time
        )
        self._local_event = self.sim.schedule(end_local, self._on_local_done)

    def _commit(self) -> None:
        """Settle the node and revoke its teardown, together.

        These are inseparable: an upstream cancel racing our own finalize must
        see an unregistered teardown, so `AttemptControl.cancel()` truthfully
        returns False and the caller keeps the real outcome. A teardown left
        registered but guarded by `settled` swallows the cancel while reporting
        success, which rewrites genuine failures as CANCELLED.
        """
        self.settled = True
        self._register_cancel(None)

    def _elapsed(self) -> TimeDuration:
        return self.sim.timestep - self.begin_at

    def _settle(self, success: bool, drop: DropReason, keep: RootRequest | None = None) -> None:
        """The one settling path: every transition above ends here.

        Full teardown in a fixed order — drop the pending local completion,
        drop the pending start of the next wave, drop the deadline guard,
        abandon downstream calls nobody is waiting for (all but `keep`, the one
        that reported the outcome), then finalize. Each step is a no-op on the
        paths that do not need it, so there is no reason for a caller to skip
        one: a settle that skipped the sibling teardown would leave abandoned
        calls consuming downstream capacity and recording outcomes of their own.
        """
        self._commit()
        if self._local_event is not None:
            self._local_event.cancelled = True  # still in the local stage
            self._local_event = None
        if self._wave_event is not None:
            self._wave_event.cancelled = True  # between two waves: the next never starts
            self._wave_event = None
        self._cancel_deadline_guard()
        self._abandon_siblings(except_root=keep)
        self._finalize(success, self._elapsed(), drop)

    # -- cancellation --------------------------------------------------------

    def cancel(self) -> None:
        """Abandon this node from above; unwinds the subtree if it fans out."""
        if self.settled:
            return
        self._settle(False, DropReason.CANCELLED)

    def _cancel_deadline_guard(self) -> None:
        if self._deadline_event is not None:
            self._deadline_event.cancelled = True
            self._deadline_event = None

    def _abandon_siblings(self, except_root: RootRequest | None = None) -> None:
        """Tear down downstream calls nobody is waiting for any more.

        No-op unless this node models cancellation propagation. Each root's own
        `cancel` recurses: cancelling a downstream that is itself a fan-out node
        runs *its* teardown, so a whole subtree unwinds — but only through
        descendants that also set the flag (each node models its own
        architecture).
        """
        if not self.node.propagate_cancellation:
            return
        for root in self._launched:
            if root is not except_root and not root.done and root.cancel is not None:
                root.cancel()

    # -- stage 1: local service ---------------------------------------------

    def _on_local_done(self) -> None:
        self._local_event = None
        if self.settled:
            return  # cancelled from above while we were doing local work

        if self.deadline is not None and self.sim.timestep >= self.deadline:
            self._settle(False, DropReason.DEADLINE)
            return

        if self.svc._fails_now(self.sim.timestep, self.sim):
            self._settle(False, DropReason.SERVER_FAILURE)
            return

        if not self.node.downstreams:
            self._settle(True, DropReason.NONE)  # leaf: behave like a plain finish
            return

        self._launch_fanout()

    # -- stage 2: fan out ----------------------------------------------------

    def _launch_fanout(self) -> None:
        # Grouped per attempt, from a snapshot: `downstreams` is a public,
        # mutable list and an edge's `wave` can be reassigned. A dict keeps the
        # list order inside each wave, so a node whose edges share one wave
        # value launches them in exactly the order it always did.
        waves: dict[int, list[DownstreamEdge]] = {}
        for edge in list(self.node.downstreams):
            waves.setdefault(edge.wave, []).append(edge)
        self._waves = [waves[wave] for wave in sorted(waves)]
        self._wave_index = 0

        # Guard before launching, so a downstream that completes synchronously
        # cannot outlive the deadline unnoticed. Armed ONCE, for all the waves:
        # being scheduled before any wave event exists is also what lets it win
        # the tie when a wave completes at the deadline instant.
        if self.deadline is not None:
            self._deadline_event = self.sim.schedule(int(self.deadline), self._on_deadline)

        self._launch_wave()

        # After the FIRST wave only, and never from `_launch_wave`: the worker
        # is handed back once (a second release raises), and handing it back
        # resumes this service's queue, whose next attempt draws its service
        # time -- so where this sits in the sequence is part of the RNG order.
        if not self.settled and not self.node.hold_worker_during_fanout:
            self._release_worker()

    def _launch_wave(self) -> None:
        """Launch every edge of the current wave, concurrently."""
        edges = self._waves[self._wave_index]
        self._remaining = len(edges)

        self._launching = True
        for edge in edges:
            self._launched.append(
                edge.client.start_request(
                    self.sim,
                    on_done=self._on_downstream_done,
                    # The same absolute deadline for every wave: a later wave
                    # simply has less of it left.
                    override_global_deadline=(
                        self.deadline if self.node.propagate_deadline else None
                    ),
                )
            )
        self._launching = False

        # A synchronous rejection during the loop was deferred so that every
        # edge of the wave still got launched; act on it now.
        if self._sync_failure is not None and not self.settled:
            self._fail_from(self._sync_failure)

    def _on_next_wave(self) -> None:
        self._wave_event = None
        if self.settled:
            return  # settled between the two waves (deadline, cancel from above)
        self._launch_wave()

    def _on_downstream_done(self, root: RootRequest) -> None:
        if self.settled:
            return
        if any(attempt.success for attempt in root.attempts):
            self._remaining -= 1
            if self._remaining == 0:
                if self._wave_index + 1 == len(self._waves):
                    self._settle(True, DropReason.NONE)
                    return
                # The wave is complete and another follows. Start it through
                # the event heap, at this same instant, rather than from inside
                # this completion callback: the callee that just reported has
                # freed its worker but not yet resumed its queue, so a call
                # submitted synchronously here would start ahead of work that
                # was queued first (the callback contract in
                # `ServiceRuntime.submit_attempt`). Retries take the same hop.
                self._wave_index += 1
                self._wave_event = self.sim.schedule(self.sim.timestep, self._on_next_wave)
            return
        if self._launching:
            # Defer: launching the remaining edges first keeps downstream load
            # independent of edge order.
            if self._sync_failure is None:
                self._sync_failure = root
            return
        self._fail_from(root)  # AND semantics: first failure short-circuits

    def _fail_from(self, root: RootRequest) -> None:
        # `_settle` latches `settled` first, so the sibling teardown it runs
        # re-enters _on_downstream_done harmlessly.
        self._settle(False, self._drop_reason_of(root), keep=root)

    @staticmethod
    def _drop_reason_of(root: RootRequest) -> DropReason:
        if not root.attempts:
            return DropReason.SERVER_FAILURE
        # Append order is completion order (ties on end time are ambiguous).
        drop = root.attempts[-1].drop_reason
        return DropReason.SERVER_FAILURE if drop == DropReason.NONE else drop

    def _on_deadline(self) -> None:
        if self.settled:
            return
        self._settle(False, DropReason.DEADLINE)


@dataclass
class ServiceNodeRuntime:
    """A logical service node that may fan out to downstream services.

    Each node wraps a `ServiceRuntime` for its own queue/latency/failures,
    and holds a list of downstream edges, each with its own `ClientRuntime` (retry/timeout)
    and a target `ServiceNodeRuntime`.
    """

    service: ServiceRuntime
    downstreams: list[DownstreamEdge] = field(default_factory=list)
    # Worker-occupancy model during fan-out:
    #   False (default) -> tandem-queue: release the local worker once local work is
    #     done and downstream calls are launched, so the worker is free while waiting.
    #   True -> synchronous thread-per-request: hold the worker for the full subtree
    #     (local work + downstream fan-out wait), so capacity is consumed down the tree.
    hold_worker_during_fanout: bool = False
    # Cancellation model for the fan-out, mirroring hold_worker_during_fanout as a
    # switch between two real architectures:
    #   False (default) -> no propagation. When AND semantics short-circuit, the
    #     sibling calls already in flight are abandoned but keep running to
    #     completion, consuming downstream capacity and recording their own
    #     outcomes. Models a stack without cancellation plumbing (plain HTTP/1.1
    #     clients, older Java stacks) — and preserves every pre-existing result.
    #   True -> Go-style propagation. Short-circuiting cancels the outstanding
    #     siblings: pending retries are dropped, in-flight attempts abandon their
    #     worker immediately, and the cancel recurses into sub-fan-outs (through
    #     descendants that also set the flag). Those roots are recorded with
    #     DropReason.CANCELLED.
    propagate_cancellation: bool = False
    # Deadline model for the fan-out, the modelling-choice ablation "no
    # propagated deadline":
    #   True (default) -> this node passes its own attempt deadline down each
    #     edge, so a downstream call can never outlive the caller's attempt:
    #     the downstream client runs under min(its own controller, that
    #     deadline) and books a DEADLINE drop of its own when it expires.
    #     Models a stack that plumbs the caller's deadline through the wire
    #     (Go contexts, gRPC's grpc-timeout header) — the current behaviour.
    #   False -> the deadline stops here. Downstream calls run under their own
    #     controller timeouts only, so they keep running (and consuming
    #     downstream capacity) after this node's attempt has already been
    #     abandoned upstream. Models a stack with no deadline propagation.
    #     This node still enforces its own deadline on itself, so the caller's
    #     drop is unchanged; only the downstream's view of it disappears.
    propagate_deadline: bool = True
    # Service time of THIS node's local stage, when it differs from the
    # station's:
    #   None (default) -> the local stage draws from the station's own
    #     distribution (`service.cfg.latency_median` / `latency_lognorm_sigma`),
    #     as every node always did.
    #   a ServiceTime -> it draws from that instead. Several nodes can share one
    #     `ServiceRuntime` -- one per handler of a service, the way a caller
    #     reaches different methods of it -- and each can then have the local
    #     work its own method does, while workers, queue, latency injections,
    #     partial failures and the 100 us floor stay the station's. Still one
    #     draw per attempt.
    # Appended last so positional construction keeps its meaning.
    service_time: ServiceTime | None = None

    @property
    def cfg(self) -> ServiceConfig:
        return self.service.cfg

    @property
    def local_service_time(self) -> ServiceTime:
        """The distribution this node's local stage actually draws from.

        `cfg` is the STATION's config, shared by every node over it, so
        `cfg.latency_median` is not this node's service time once an override is
        set. Read the effective pair here when estimating E[S] per handler.
        """
        if self.service_time is not None:
            return self.service_time
        return ServiceTime(
            median=self.service.cfg.latency_median, sigma=self.service.cfg.latency_lognorm_sigma
        )

    def add_downstream(self, edge: DownstreamEdge) -> None:
        self.downstreams.append(edge)

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: AttemptDoneCallback,
        attempt_deadline: TimePoint | None = None,
    ) -> AttemptControl:
        """Submit a logical attempt to this node.

        Runs local service time, then calls the downstreams wave by wave
        (``DownstreamEdge.wave``: concurrently within a wave, the next wave
        once the previous one has succeeded; one wave by default), with AND
        semantics throughout, and completes only when the last wave finishes.
        The local worker is released after the local stage unless
        ``hold_worker_during_fanout`` is set, in which case it is held for the
        whole subtree (synchronous thread-per-request model).
        """

        def behavior(
            svc: ServiceRuntime,
            sim2: Simulator,
            deadline: TimePoint | None,
            finalize: Callable[[bool, TimeDuration, DropReason], None],
            begin_at: TimePoint,
            release_worker: Callable[[], None],
            register_cancel: Callable[[Callable[[], None] | None], None],
        ) -> None:
            _FanOutAttempt(
                node=self,
                svc=svc,
                sim=sim2,
                deadline=deadline,
                finalize=finalize,
                begin_at=begin_at,
                release_worker=release_worker,
                register_cancel=register_cancel,
            ).start()

        # Propagate the cancellation handle up: without it the caller holds no
        # way to abandon this node mid-subtree.
        return self.service.submit_attempt(
            sim,
            on_done,
            attempt_deadline=attempt_deadline,
            behavior=behavior,
        )
