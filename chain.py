from collections.abc import Callable
from dataclasses import dataclass, field

from caller import ClientRuntime
from core import DropReason, RootRequest, TimeDuration, TimePoint
from service import AttemptControl, ServiceConfig, ServiceRuntime
from simulator import Event, Simulator


@dataclass
class DownstreamEdge:
    client: ClientRuntime
    target: "ServiceNodeRuntime"


class _FanOutAttempt:
    """One logical attempt at a `ServiceNodeRuntime`, as an explicit state machine.

    The lifecycle is: local service stage -> (optionally) fan out to every
    downstream edge -> settle. It settles exactly once, through `_commit`, on
    whichever of these happens first:

    * the local stage expires against the deadline, or the local service fails;
    * the node has no downstreams and the local stage succeeds;
    * every downstream succeeds (AND semantics);
    * any downstream fails (short-circuit, but only after every edge is launched
      so downstream load does not depend on edge order);
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
        self._remaining = 0
        #: True only inside the launch loop, so a synchronous rejection defers
        #: its short-circuit until every edge has been launched.
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
        base_time = self.svc._sample_service_time(self.begin_at, self.sim)
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

    def _settle(self, success: bool, drop: DropReason) -> None:
        self._commit()
        self._cancel_deadline_guard()
        self._finalize(success, self._elapsed(), drop)

    # -- cancellation --------------------------------------------------------

    def cancel(self) -> None:
        """Abandon this node from above; unwinds the subtree if it fans out."""
        if self.settled:
            return
        self._commit()
        if self._local_event is not None:
            self._local_event.cancelled = True  # still in the local stage
        self._cancel_deadline_guard()
        self._abandon_siblings()
        self._finalize(False, self._elapsed(), DropReason.CANCELLED)

    def _cancel_deadline_guard(self) -> None:
        if self._deadline_event is not None:
            self._deadline_event.cancelled = True

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
        edges = list(self.node.downstreams)
        self._remaining = len(edges)

        # Guard before launching, so a downstream that completes synchronously
        # cannot outlive the deadline unnoticed.
        if self.deadline is not None:
            self._deadline_event = self.sim.schedule(int(self.deadline), self._on_deadline)

        self._launching = True
        for edge in edges:
            self._launched.append(
                edge.client.start_request(
                    self.sim,
                    on_done=self._on_downstream_done,
                    override_global_deadline=self.deadline,
                )
            )
        self._launching = False

        # A synchronous rejection during the loop was deferred so that every
        # edge still got launched; act on it now.
        if self._sync_failure is not None and not self.settled:
            self._fail_from(self._sync_failure)

        if not self.settled and not self.node.hold_worker_during_fanout:
            self._release_worker()

    def _on_downstream_done(self, root: RootRequest) -> None:
        if self.settled:
            return
        if any(attempt.success for attempt in root.attempts):
            self._remaining -= 1
            if self._remaining == 0:
                self._settle(True, DropReason.NONE)
            return
        if self._launching:
            # Defer: launching the remaining edges first keeps downstream load
            # independent of edge order.
            if self._sync_failure is None:
                self._sync_failure = root
            return
        self._fail_from(root)  # AND semantics: first failure short-circuits

    def _fail_from(self, root: RootRequest) -> None:
        self._commit()
        self._cancel_deadline_guard()
        # `settled` is already set, so the cancellations below re-enter
        # _on_downstream_done harmlessly.
        self._abandon_siblings(except_root=root)
        self._finalize(False, self._elapsed(), self._drop_reason_of(root))

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
        self._commit()
        self._abandon_siblings()
        self._finalize(False, self._elapsed(), DropReason.DEADLINE)


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

    @property
    def cfg(self) -> ServiceConfig:
        return self.service.cfg

    def add_downstream(self, edge: DownstreamEdge) -> None:
        self.downstreams.append(edge)

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: Callable[..., None],
        attempt_deadline: TimePoint | None = None,
    ) -> AttemptControl:
        """Submit a logical attempt to this node.

        Runs local service time, fans out to downstreams concurrently (AND
        semantics), and completes only when the fan-out finishes. The local
        worker is released after the local stage unless
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
