from collections.abc import Callable
from dataclasses import dataclass, field

from caller import ClientRuntime
from core import DropReason, TimeDuration, TimePoint
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator


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
    ) -> None:
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
        ) -> None:
            # Stage 1: local service time
            base_time = svc._sample_service_time(begin_at, sim2)
            end_local = (
                min(begin_at + base_time, deadline)
                if deadline is not None
                else begin_at + base_time
            )

            def after_local():
                # If we expired during local processing, drop by deadline
                if deadline is not None and sim2.timestep >= deadline:
                    finalize(False, sim2.timestep - begin_at, DropReason.DEADLINE)
                    return

                local_drop = (
                    DropReason.SERVER_FAILURE
                    if svc._fails_now(sim2.timestep, sim2)
                    else DropReason.NONE
                )
                if local_drop != DropReason.NONE:
                    finalize(False, sim2.timestep - begin_at, local_drop)
                    return

                # No downstreams: behave like default finish.
                if not self.downstreams:
                    finalize(True, sim2.timestep - begin_at, DropReason.NONE)
                    return

                downstream_edges = list(self.downstreams)

                # AND semantics across all downstream edges: all must succeed eventually.
                done = False
                remaining = len(downstream_edges)

                def downstream_drop_reason(_root) -> DropReason:
                    if not _root.attempts:
                        return DropReason.SERVER_FAILURE
                    last_attempt = max(_root.attempts, key=lambda a: a.interval.end)
                    return last_attempt.drop_reason

                deadline_event = None

                def cancel_deadline():
                    if deadline_event is not None:
                        deadline_event.cancelled = True

                def maybe_finish():
                    nonlocal done, remaining
                    if done:
                        return
                    if remaining == 0:
                        # All downstreams completed
                        done = True
                        cancel_deadline()
                        finalize(True, sim2.timestep - begin_at, DropReason.NONE)

                # Deadline guard: if deadline elapses before downstreams complete, drop
                if deadline is not None:

                    def on_deadline():
                        nonlocal done
                        if not done:
                            done = True
                            finalize(False, sim2.timestep - begin_at, DropReason.DEADLINE)

                    # ensure we schedule at int(deadline)
                    deadline_event = sim2.schedule(int(deadline), on_deadline)

                launching = False
                sync_failure_root = None

                def fail_with(_root):
                    nonlocal done
                    done = True
                    cancel_deadline()
                    drop = downstream_drop_reason(_root)
                    if drop == DropReason.NONE:
                        drop = DropReason.SERVER_FAILURE
                    finalize(False, sim2.timestep - begin_at, drop)

                def on_downstream_done(_root):
                    nonlocal remaining, sync_failure_root
                    if done:
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
                    edge.client.start_request(
                        sim2,
                        on_done=on_downstream_done,
                        override_global_deadline=deadline,
                    )
                launching = False
                if sync_failure_root is not None and not done:
                    fail_with(sync_failure_root)

                if not done and not self.hold_worker_during_fanout:
                    release_worker()

            sim2.schedule(end_local, after_local)

        # Enqueue with custom behavior
        self.service.submit_attempt(
            sim,
            on_done,
            attempt_deadline=attempt_deadline,
            behavior=behavior,
        )
