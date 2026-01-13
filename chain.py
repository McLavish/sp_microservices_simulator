from dataclasses import dataclass, field
from typing import Callable, List, Optional
from core import TimePoint, TimeDuration, DropReason
from simulator import Simulator
from service import ServiceRuntime, ServiceConfig
from caller import ClientRuntime


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
    downstreams: List[DownstreamEdge] = field(default_factory=list)

    @property
    def cfg(self) -> ServiceConfig:
        return self.service.cfg

    def add_downstream(self, edge: DownstreamEdge) -> None:
        self.downstreams.append(edge)

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: Callable[[bool, TimeDuration, DropReason, int], None],
        attempt_deadline: Optional[TimePoint] = None,
    ) -> None:
        """Submit a logical attempt to this node.

        Uses a custom behavior to keep the worker busy across local work and downstream calls,
        and completes only when the fanout (if any) finishes.
        """

        def behavior(
            svc: ServiceRuntime,
            sim2: Simulator,
            deadline: Optional[TimePoint],
            finalize: Callable[[bool, TimeDuration, DropReason], None],
            begin_at: TimePoint,
            release_worker: Callable[[], None],
        ) -> None:
            # Stage 1: local service time
            base_time = svc._sample_service_time(begin_at, sim2)
            end_local = (
                min(begin_at + base_time, deadline) if deadline is not None else begin_at + base_time
            )

            def after_local():
                # If we expired during local processing, drop by deadline
                if deadline is not None and sim2.timestep >= deadline:
                    finalize(False, sim2.timestep - begin_at, DropReason.DEADLINE)
                    return

                # No downstreams: behave like default finish (including partial failure)
                if not self.downstreams:
                    drop = DropReason.SERVER_FAILURE if svc._fails_now(sim2.timestep, sim2) else DropReason.NONE
                    finalize(drop == DropReason.NONE, sim2.timestep - begin_at, drop)
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

                def maybe_finish():
                    nonlocal done, remaining
                    if done:
                        return
                    if remaining == 0:
                        # All downstreams completed
                        done = True
                        finalize(True, sim2.timestep - begin_at, DropReason.NONE)

                # Deadline guard: if deadline elapses before downstreams complete, drop
                if deadline is not None:
                    def on_deadline():
                        nonlocal done
                        if not done:
                            done = True
                            finalize(False, sim2.timestep - begin_at, DropReason.DEADLINE)

                    # ensure we schedule at int(deadline)
                    sim2.schedule(int(deadline), on_deadline)

                # Launch downstreams concurrently
                for edge in downstream_edges:
                    def _mk_cb():
                        # Create a distinct callback capturing this invocation
                        def _cb(_root):
                            nonlocal remaining, done
                            if done:
                                return
                            # Determine success from root attempts
                            succeeded = any(a.success for a in _root.attempts)
                            if not succeeded:
                                # We can finish immediately on first failure (AND semantics)
                                done = True
                                drop = downstream_drop_reason(_root)
                                if drop == DropReason.NONE:
                                    drop = DropReason.SERVER_FAILURE
                                finalize(False, sim2.timestep - begin_at, drop)
                                return
                            remaining -= 1
                            maybe_finish()

                        return _cb

                    cb = _mk_cb()
                    edge.client.start_request(
                        sim2,
                        on_done=cb,
                        override_global_deadline=deadline,
                    )

                if not done:
                    release_worker()

            sim2.schedule(end_local, after_local)

        # Enqueue with custom behavior
        self.service.submit_attempt(
            sim,
            on_done,
            attempt_deadline=attempt_deadline,
            behavior=behavior,
        )
