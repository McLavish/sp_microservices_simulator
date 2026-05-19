from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Callable

from core import (
    Request,
    RootRequest,
    DropReason,
    TimeInterval,
    TimePoint,
    TimeDuration,
)
from retry_policies import RetryPolicy, RetryContext
from service import ServiceRuntime
from simulator import Simulator
from metrics import Metrics
from timeout_controllers import TimeoutController


@dataclass(frozen=True)
class ClientConfig:
    retry: RetryPolicy
    timeout_controller: TimeoutController


@dataclass
class AttemptCtx:
    root: RootRequest
    req: Request
    last_delay: TimeDuration = 0
    total_delay: TimeDuration = 0


@dataclass
class ClientRuntime:
    cfg: ClientConfig
    service: ServiceRuntime

    roots: List[RootRequest] = field(default_factory=list)
    attempts_total: int = 0

    def _min_deadline(
        self, d1: Optional[TimePoint], d2: Optional[TimePoint]
    ) -> Optional[TimePoint]:
        if d1 is None:
            return d2
        if d2 is None:
            return d1
        return min(d1, d2)

    def _update_policy_results(
        self, policy: RetryPolicy, success: bool, now: TimePoint
    ) -> None:
        """Update result tracking for policies that support it (e.g., circuit breakers, retry budget)."""

        if hasattr(policy, "add_result"):
            policy.add_result(success, now)

        # Update underlying strategy if it exists
        if hasattr(policy, "underlying_strategy"):
            self._update_policy_results(policy.underlying_strategy, success, now)

    def _allow_policy_request(self, policy: RetryPolicy, now: TimePoint) -> bool:
        if hasattr(policy, "allow_request") and not policy.allow_request(now):
            return False
        if hasattr(policy, "underlying_strategy"):
            return self._allow_policy_request(policy.underlying_strategy, now)
        return True

    def _complete_root(self, root: RootRequest) -> None:
        root.done = True
        self.roots.append(root)
        if getattr(root, "on_done", None):
            root.on_done(root)  # type: ignore[arg-type]

    def start_request(
        self,
        sim: Simulator,
        on_done: Optional[Callable[[RootRequest], None]] = None,
        override_global_deadline: Optional[TimePoint] = None,
    ):
        root_req = RootRequest()

        # Compute global deadline: min(override, controller)
        global_timeout = self.cfg.timeout_controller.get_global_timeout()
        computed_deadline: Optional[TimePoint] = None
        if global_timeout is not None:
            computed_deadline = sim.timestep + global_timeout
        if override_global_deadline is not None:
            computed_deadline = (
                override_global_deadline
                if computed_deadline is None
                else min(computed_deadline, override_global_deadline)
            )
        root_req.global_deadline = computed_deadline
        root_req.on_done = on_done

        if not self._allow_policy_request(self.cfg.retry, sim.timestep):
            self.attempts_total += 1
            req = Request(
                service=self.service.cfg.name,
                interval=TimeInterval(begin=sim.timestep, end=sim.timestep),
                deadline=computed_deadline,
                success=False,
                drop_reason=DropReason.CIRCUIT_OPEN,
                queue_size_at_end=0,
            )
            root_req.add_attempt(req)
            self._complete_root(root_req)
            return

        # Start first attempt
        # Create dummy request that will be replaced in _start_attempt
        dummy_req = Request(
            service="",
            interval=TimeInterval(begin=sim.timestep, end=sim.timestep),
            deadline=None,
        )
        initial_ctx = AttemptCtx(
            root=root_req, req=dummy_req, last_delay=0, total_delay=0
        )
        self._start_attempt(sim, initial_ctx)

    def _start_attempt(self, sim: Simulator, ctx: AttemptCtx):
        if ctx.root.done:
            raise RuntimeError("Late start after root is done")

        self.attempts_total += 1
        attempt_timeout = self.cfg.timeout_controller.get_attempt_timeout()

        req_deadline = self._min_deadline(
            ctx.root.global_deadline,
            (sim.timestep + attempt_timeout) if attempt_timeout else None,
        )
        req = Request(
            service=self.service.cfg.name,
            interval=TimeInterval(begin=sim.timestep, end=sim.timestep),
            deadline=req_deadline,
        )

        ctx.req = req

        on_done = partial(self._on_attempt_done, ctx, sim)
        self.service.submit_attempt(sim, on_done, attempt_deadline=req_deadline)

    def _on_attempt_done(
        self,
        ctx: AttemptCtx,
        sim: Simulator,
        success: bool,
        svc_time: TimeDuration,  # unused for now
        drop_reason: DropReason,
        queue_size: int,
    ):
        if ctx.root.done:
            raise RuntimeError("Root request was already done!")

        end_time = sim.timestep

        # Half-open semantics: success must finish strictly before deadline.
        within_deadline = (ctx.req.deadline is None) or (end_time < ctx.req.deadline)
        if not within_deadline and drop_reason == DropReason.NONE:
            drop_reason = DropReason.DEADLINE
        success_effective = (
            success and within_deadline and drop_reason == DropReason.NONE
        )

        # Update request state
        ctx.req.interval.end = end_time
        ctx.req.success = success_effective
        ctx.req.drop_reason = drop_reason
        ctx.req.queue_size_at_end = queue_size

        # Log the request
        ctx.root.add_attempt(ctx.req)

        # Update circuit breaker state if policy supports it
        self._update_policy_results(self.cfg.retry, success_effective, sim.timestep)

        # Record timeout controller results
        request_latency = ctx.req.interval.end - ctx.req.interval.begin
        self.cfg.timeout_controller.record_result(success_effective, request_latency)

        # always fail on first attempt
        # if ctx.root.attempt_count() == 1:
        #     ctx.req.success = False

        if ctx.req.success:
            self._complete_root(ctx.root)
            return

        # Build retry context
        retry_context = RetryContext(
            attempt=ctx.root.attempt_count(),
            now=sim.timestep,
        )

        should_retry, delay = self.cfg.retry.next_delay(retry_context)
        next_start = sim.timestep + delay
        if (not should_retry) or (
            ctx.root.global_deadline is not None
            and next_start >= ctx.root.global_deadline
        ):
            self._complete_root(ctx.root)
            return

        sim.schedule(next_start, partial(self._start_attempt, sim, ctx))

    def metrics(self) -> Metrics:
        return Metrics(roots=self.roots, attempts_total=self.attempts_total)
