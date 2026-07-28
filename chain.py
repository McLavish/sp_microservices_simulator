from collections.abc import Callable
from dataclasses import dataclass, field

from caller import ClientRuntime
from core import DropReason, TimeDuration, TimePoint
from service import AttemptControl, ServiceConfig, ServiceRuntime
from simulator import Event, Simulator


@dataclass
class DownstreamEdge:
    client: ClientRuntime
    target: "ServiceNodeRuntime"


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
    #     worker immediately, and the cancel recurses into sub-fan-outs. Those
    #     roots are recorded with DropReason.CANCELLED.
    propagate_cancellation: bool = False

    @property
    def cfg(self) -> ServiceConfig:
        return self.service.cfg

    def add_downstream(self, edge: DownstreamEdge) -> None:
        self.downstreams.append(edge)

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: Callable[[bool, TimeDuration, DropReason, int], None],
        attempt_deadline: TimePoint | None = None,
    ) -> AttemptControl:
        """Submit a logical attempt to this node.

        Uses a custom behavior that runs local service time, fans out to downstreams
        concurrently (AND semantics), and completes only when the fanout finishes.
        The local worker is released after the local stage unless
        ``hold_worker_during_fanout`` is set, in which case it is held for the whole
        subtree (synchronous thread-per-request model).
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
            # Stage 1: local service time
            base_time = svc._sample_service_time(begin_at, sim2)
            end_local = (
                min(begin_at + base_time, deadline)
                if deadline is not None
                else begin_at + base_time
            )

            # Node-level teardown, installed BEFORE any stage runs so this node is
            # cancellable throughout its whole lifecycle — including during local
            # service, and including leaf nodes that never reach the fan-out
            # block. `settled` is the single exactly-once guard for the node:
            # every finalize path sets it, and the cancel path checks it.
            settled = False
            local_event: Event | None = None
            teardown_fanout: Callable[[], None] | None = None

            def commit() -> None:
                """Mark the node settled and revoke its teardown.

                These must happen together: an upstream cancel racing our own
                finalize has to see an unregistered teardown (so cancel() returns
                False and the caller keeps the real outcome), not a registered
                one guarded by `settled` that swallows the request while
                reporting success.
                """
                nonlocal settled
                settled = True
                register_cancel(None)

            def cancel_node() -> None:
                nonlocal settled
                if settled:
                    return
                commit()
                if local_event is not None:
                    local_event.cancelled = True  # still doing local work
                if teardown_fanout is not None:
                    teardown_fanout()  # drop the deadline guard, unwind children
                finalize(False, sim2.timestep - begin_at, DropReason.CANCELLED)

            register_cancel(cancel_node)

            def after_local():
                nonlocal settled, teardown_fanout
                if settled:
                    return  # cancelled from above while we were doing local work

                # If we expired during local processing, drop by deadline
                if deadline is not None and sim2.timestep >= deadline:
                    commit()
                    finalize(False, sim2.timestep - begin_at, DropReason.DEADLINE)
                    return

                local_drop = (
                    DropReason.SERVER_FAILURE
                    if svc._fails_now(sim2.timestep, sim2)
                    else DropReason.NONE
                )
                if local_drop != DropReason.NONE:
                    commit()
                    finalize(False, sim2.timestep - begin_at, local_drop)
                    return

                # No downstreams: behave like default finish.
                if not self.downstreams:
                    commit()
                    finalize(True, sim2.timestep - begin_at, DropReason.NONE)
                    return

                downstream_edges = list(self.downstreams)

                # AND semantics across all downstream edges: all must succeed
                # eventually. `settled` is the node-level exactly-once guard,
                # shared with the cancel path installed before stage 1.
                remaining = len(downstream_edges)

                def downstream_drop_reason(_root) -> DropReason:
                    if not _root.attempts:
                        return DropReason.SERVER_FAILURE
                    # Append order is completion order (ties on end time are ambiguous).
                    return _root.attempts[-1].drop_reason

                deadline_event = None

                def cancel_deadline():
                    if deadline_event is not None:
                        deadline_event.cancelled = True

                def maybe_finish():
                    nonlocal remaining
                    if settled:
                        return
                    if remaining == 0:
                        # All downstreams completed
                        commit()
                        cancel_deadline()
                        finalize(True, sim2.timestep - begin_at, DropReason.NONE)

                # Deadline guard: if deadline elapses before downstreams complete, drop
                if deadline is not None:

                    def on_deadline():
                        if not settled:
                            commit()
                            abandon_siblings()
                            finalize(False, sim2.timestep - begin_at, DropReason.DEADLINE)

                    # ensure we schedule at int(deadline)
                    deadline_event = sim2.schedule(int(deadline), on_deadline)

                launching = False
                sync_failure_root = None
                launched_roots: list = []

                def abandon_siblings(except_root=None):
                    """Tear down downstream calls nobody is waiting for any more.

                    No-op unless this node models cancellation propagation. The
                    root's own `cancel` recurses: cancelling a downstream that is
                    itself a fan-out node runs *its* registered teardown, so a
                    whole subtree unwinds.
                    """
                    if not self.propagate_cancellation:
                        return
                    for _r in launched_roots:
                        if _r is not except_root and not _r.done and _r.cancel is not None:
                            _r.cancel()

                # From here on, a cancel from above must also unwind the fan-out.
                def _teardown_fanout() -> None:
                    cancel_deadline()
                    abandon_siblings()

                teardown_fanout = _teardown_fanout

                def fail_with(_root):
                    commit()
                    cancel_deadline()
                    # `settled` is already set, so the cancellations below re-enter
                    # on_downstream_done harmlessly.
                    abandon_siblings(except_root=_root)
                    drop = downstream_drop_reason(_root)
                    if drop == DropReason.NONE:
                        drop = DropReason.SERVER_FAILURE
                    finalize(False, sim2.timestep - begin_at, drop)

                def on_downstream_done(_root):
                    nonlocal remaining, sync_failure_root
                    if settled:
                        return
                    # Determine success from root attempts
                    succeeded = any(a.success for a in _root.attempts)
                    if not succeeded:
                        if launching:
                            # A synchronous rejection (queue-full / circuit-open)
                            # during the launch loop: remember it, but launch the
                            # remaining edges first so downstream load does not
                            # depend on edge order.
                            if sync_failure_root is None:
                                sync_failure_root = _root
                            return
                        # First failure short-circuits the parent (AND semantics)
                        fail_with(_root)
                        return
                    remaining -= 1
                    maybe_finish()

                # Launch ALL downstreams concurrently; AND semantics still
                # short-circuit the parent, but only after every edge is launched.
                launching = True
                for edge in downstream_edges:
                    launched_roots.append(
                        edge.client.start_request(
                            sim2,
                            on_done=on_downstream_done,
                            override_global_deadline=deadline,
                        )
                    )
                launching = False
                if sync_failure_root is not None and not settled:
                    fail_with(sync_failure_root)

                if not settled and not self.hold_worker_during_fanout:
                    release_worker()

            local_event = sim2.schedule(end_local, after_local)

        # Enqueue with custom behavior. Propagate the cancellation handle up:
        # without it the caller holds no way to abandon this node mid-subtree.
        return self.service.submit_attempt(
            sim,
            on_done,
            attempt_deadline=attempt_deadline,
            behavior=behavior,
        )
