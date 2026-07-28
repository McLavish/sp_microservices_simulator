from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Protocol, runtime_checkable

from core import (
    DropReason,
    Request,
    RootRequest,
    TimeDuration,
    TimeInterval,
    TimePoint,
)
from metrics import Metrics
from retry_policies import RetryContext, RetryPolicy
from service import AttemptControl, ServiceConfig
from simulator import Event, Simulator
from timeout_controllers import TimeoutController


@runtime_checkable
class _SupportsAddResult(Protocol):
    def add_result(self, success: bool, now: TimePoint) -> None: ...


@runtime_checkable
class _SupportsUnderlying(Protocol):
    underlying_strategy: RetryPolicy


class ServiceLike(Protocol):
    """Structural type for anything ClientRuntime can submit attempts to.

    Implemented by `ServiceRuntime` (single service) and `ServiceNodeRuntime`
    (a chain node that fans out downstream).
    """

    cfg: ServiceConfig

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: Callable[[bool, TimeDuration, DropReason, int], None],
        attempt_deadline: TimePoint | None = None,
    ) -> AttemptControl: ...


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
    # Handle on the attempt currently at the service (None while backing off).
    control: "AttemptControl | None" = None
    # Scheduled retry event, so cancellation can drop a pending backoff.
    retry_event: Event | None = None
    # The context that authorised the pending retry. A retry is charged when
    # `next_delay` grants it, so abandoning it before it runs must refund
    # through the same rollback path a deadline veto uses.
    pending_retry_context: RetryContext | None = None
    # Set before tearing down, so _on_attempt_done completes the root instead
    # of scheduling the next retry.
    cancelling: bool = False


@dataclass
class ClientRuntime:
    cfg: ClientConfig
    service: ServiceLike

    roots: list[RootRequest] = field(default_factory=list)
    attempts_total: int = 0

    def _min_deadline(self, d1: TimePoint | None, d2: TimePoint | None) -> TimePoint | None:
        if d1 is None:
            return d2
        if d2 is None:
            return d1
        return min(d1, d2)

    def _update_policy_results(self, policy: RetryPolicy, success: bool, now: TimePoint) -> None:
        """Update result tracking for policies that support it (e.g., circuit breakers, retry budget)."""

        if isinstance(policy, _SupportsAddResult):
            policy.add_result(success, now)

        if isinstance(policy, _SupportsUnderlying):
            self._update_policy_results(policy.underlying_strategy, success, now)

    def _notify_request_start(self, policy: RetryPolicy, now: TimePoint) -> None:
        """Announce an admitted root request to every layer of the policy chain."""
        policy.on_request_start(now)
        if isinstance(policy, _SupportsUnderlying):
            self._notify_request_start(policy.underlying_strategy, now)

    def _rollback_policy(self, policy: RetryPolicy, context: RetryContext) -> None:
        """Refund a retry that ``next_delay`` authorised but we never issued.

        Mirrors `_update_policy_results`' recursion: every layer that charged
        during `next_delay` sits on the `underlying_strategy` chain, so every
        layer has to be given the chance to refund.
        """
        policy.rollback(context)
        if isinstance(policy, _SupportsUnderlying):
            self._rollback_policy(policy.underlying_strategy, context)

    def _allow_policy_request(self, policy: RetryPolicy, now: TimePoint) -> bool:
        if not policy.allow_request(now):
            return False
        if isinstance(policy, _SupportsUnderlying):
            return self._allow_policy_request(policy.underlying_strategy, now)
        return True

    def _complete_root(self, root: RootRequest, now: TimePoint) -> None:
        root.done = True
        root.cancel = None  # a completed root is no longer cancellable
        # When the root actually finished. Usually the final attempt's end, but
        # a root cancelled while backing off has no attempt of its own to carry
        # the timestamp, and bucketing it at the previous attempt's end would
        # place it in the wrong window.
        root.completed_at = now
        self.roots.append(root)
        if root.on_done is not None:
            root.on_done(root)

    def _cancel_root(self, ctx: AttemptCtx, sim: Simulator) -> None:
        """Abandon this root: no further attempts, and give back in-flight work.

        Idempotent and safe to call from a completion callback — a root that
        has already finished is left alone.
        """
        if ctx.root.done or ctx.cancelling:
            return
        now = sim.timestep
        ctx.cancelling = True
        ctx.root.cancelled = True

        if ctx.retry_event is not None:
            # Waiting on backoff: drop the pending retry. No attempt is at the
            # service, so there is nothing to mark; the root's `cancelled` flag
            # carries the classification.
            ctx.retry_event.cancelled = True
            ctx.retry_event = None
            # That retry was already charged when the policy granted it. Refund
            # it — a retry that never runs must not consume budget, whether the
            # deadline vetoed it or an upstream cancelled it.
            if ctx.pending_retry_context is not None:
                self._rollback_policy(self.cfg.retry, ctx.pending_retry_context)
                ctx.pending_retry_context = None
            self._complete_root(ctx.root, now)
            return

        control = ctx.control
        if control is not None and control.cancel():
            # cancel() drove the service's teardown, which called back into
            # _on_attempt_done and completed the root already.
            return
        # An attempt is outstanding but not cancellable (a behavior that never
        # registered a teardown), or it is completing at this very instant.
        # Leave the root open: _on_attempt_done sees `cancelling` and closes it
        # without retrying. Completing it here would strand that callback and
        # trip the "Root request was already done!" guard.

    def start_request(
        self,
        sim: Simulator,
        on_done: Callable[[RootRequest], None] | None = None,
        override_global_deadline: TimePoint | None = None,
    ):
        root_req = RootRequest()

        # One budget deposit per admitted request (Finagle's RetryFilter deposits
        # before attempting, not per outcome), so retry budgets are a ratio of
        # requests rather than of successes.
        self._notify_request_start(self.cfg.retry, sim.timestep)

        # Compute global deadline: min(override, controller)
        global_timeout = self.cfg.timeout_controller.get_global_timeout()
        computed_deadline = self._min_deadline(
            (sim.timestep + global_timeout) if global_timeout is not None else None,
            override_global_deadline,
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
            self._complete_root(root_req, sim.timestep)
            return root_req

        # Start first attempt
        # Create dummy request that will be replaced in _start_attempt
        dummy_req = Request(
            service="",
            interval=TimeInterval(begin=sim.timestep, end=sim.timestep),
            deadline=None,
        )
        initial_ctx = AttemptCtx(root=root_req, req=dummy_req, last_delay=0, total_delay=0)
        root_req.cancel = partial(self._cancel_root, initial_ctx, sim)
        self._start_attempt(sim, initial_ctx)
        return root_req

    def _start_attempt(self, sim: Simulator, ctx: AttemptCtx):
        if ctx.root.done:
            raise RuntimeError("Late start after root is done")

        self.attempts_total += 1
        attempt_timeout = self.cfg.timeout_controller.get_attempt_timeout()

        req_deadline = self._min_deadline(
            ctx.root.global_deadline,
            (sim.timestep + attempt_timeout) if attempt_timeout is not None else None,
        )
        req = Request(
            service=self.service.cfg.name,
            interval=TimeInterval(begin=sim.timestep, end=sim.timestep),
            deadline=req_deadline,
        )

        ctx.req = req

        ctx.retry_event = None
        ctx.pending_retry_context = None
        on_done = partial(self._on_attempt_done, ctx, sim)
        ctx.control = self.service.submit_attempt(sim, on_done, attempt_deadline=req_deadline)

    def _on_attempt_done(
        self,
        ctx: AttemptCtx,
        sim: Simulator,
        success: bool,
        svc_time: TimeDuration,
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
        success_effective = success and within_deadline and drop_reason == DropReason.NONE

        # Update request state
        ctx.req.interval.end = end_time
        ctx.req.success = success_effective
        ctx.req.drop_reason = drop_reason
        ctx.req.queue_size_at_end = queue_size
        # svc_time == 0 marks paths where service never began (queue-full,
        # deadline-at-dequeue); real service times are floored at 100 us.
        ctx.req.service_time_ns = svc_time if svc_time > 0 else None

        # Log the request
        ctx.root.add_attempt(ctx.req)

        ctx.control = None  # nothing outstanding at the service any more

        # Cancelled by an upstream fan-out: abandoned, NOT failed. Return before
        # any feedback, because this attempt is not evidence about the service.
        # Feeding it to the breaker would trip it on work the service never got
        # to finish, and feeding it to the adaptive timeout controller would
        # count a truncated latency as an observation.
        if ctx.cancelling or drop_reason == DropReason.CANCELLED:
            ctx.root.cancelled = True
            self._complete_root(ctx.root, sim.timestep)
            return

        # Update circuit breaker state if policy supports it
        self._update_policy_results(self.cfg.retry, success_effective, sim.timestep)

        # Record timeout controller results
        request_latency = ctx.req.interval.end - ctx.req.interval.begin
        self.cfg.timeout_controller.record_result(success_effective, request_latency)

        if ctx.req.success:
            self._complete_root(ctx.root, sim.timestep)
            return

        # Build retry context
        retry_context = RetryContext(
            attempt=ctx.root.attempt_count(),
            now=sim.timestep,
        )

        decision = self.cfg.retry.next_delay(retry_context)
        next_start = sim.timestep + decision.delay
        deadline_veto = (
            ctx.root.global_deadline is not None and next_start >= ctx.root.global_deadline
        )
        if (not decision.allow) or deadline_veto:
            # `next_delay` is side-effecting for throttling policies. When the
            # policy authorised the retry but the deadline vetoes it, refund the
            # charge — otherwise a retry that never happens still consumes
            # budget/tokens. (`allow=False` charged nothing: every throttling
            # policy withdraws only on the path that returns allow=True.)
            if decision.allow:
                self._rollback_policy(self.cfg.retry, retry_context)
            self._complete_root(ctx.root, sim.timestep)
            return

        ctx.pending_retry_context = retry_context
        ctx.retry_event = sim.schedule(next_start, partial(self._start_attempt, sim, ctx))

    def metrics(self) -> Metrics:
        return Metrics(roots=self.roots, attempts_total=self.attempts_total)
