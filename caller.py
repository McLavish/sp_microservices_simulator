from collections.abc import Callable, Iterator
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


#: The service -> client completion callback: (success, occupancy, drop reason,
#: queue size, occupancy_censored). Spelled out rather than `Callable[..., None]`
#: because the last argument carries statistical semantics — it marks the
#: occupancy sample as a LOWER BOUND for the censored MLE — so a producer that
#: omits it silently reports a truncated sample as a complete observation.
AttemptDoneCallback = Callable[[bool, TimeDuration, DropReason, int, bool], None]


class ServiceLike(Protocol):
    """Structural type for anything ClientRuntime can submit attempts to.

    Implemented by `ServiceRuntime` (single service) and `ServiceNodeRuntime`
    (a chain node that fans out downstream).
    """

    cfg: ServiceConfig

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: AttemptDoneCallback,
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
    # Set when a cancel could NOT take effect because the attempt had already
    # committed to an outcome. Distinct from `cancelling`, which would rewrite
    # that outcome as CANCELLED: this one leaves the attempt alone and only
    # forbids further ones. Survives the `cancelling` reset in `_cancel_root`.
    abandoned: bool = False


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

    def _policy_chain(self, policy: RetryPolicy) -> Iterator[RetryPolicy]:
        """Every layer of a wrapping policy, outermost first.

        One walker for all four traversals. They previously each re-implemented
        this recursion, so a wrapper that one of them reached and another missed
        would silently see inconsistent notifications, results or refunds.
        """
        current: RetryPolicy | None = policy
        while current is not None:
            yield current
            current = (
                current.underlying_strategy if isinstance(current, _SupportsUnderlying) else None
            )

    def _update_policy_results(self, policy: RetryPolicy, success: bool, now: TimePoint) -> None:
        """Record an attempt outcome on every layer that tracks results."""
        for layer in self._policy_chain(policy):
            if isinstance(layer, _SupportsAddResult):
                layer.add_result(success, now)

    def _notify_request_start(self, policy: RetryPolicy, now: TimePoint) -> None:
        """Announce an admitted root request to every layer of the policy chain."""
        for layer in self._policy_chain(policy):
            layer.on_request_start(now)

    def _rollback_policy(
        self, policy: RetryPolicy, context: RetryContext, deferred: bool = False
    ) -> None:
        """Refund a retry that ``next_delay`` authorised but we never issued.

        Every layer that charged during `next_delay` sits on the chain, so every
        layer gets the chance to refund.

        ``deferred=True`` marks a refund issued long after the charge (a cancel
        arriving mid-backoff). `RetryPolicy.rollback` is specified as undoing the
        *immediately preceding* `next_delay`, so layers that keep single-slot
        undo state are skipped: with several roots sharing one policy, a late
        refund would rewind a different root's live reservation. Skipping leaves
        the charge standing, which is conservative — it over-counts rather than
        handing the same slot out twice.
        """
        for layer in self._policy_chain(policy):
            if not deferred or layer.supports_deferred_rollback:
                layer.rollback(context)

    def _allow_policy_request(self, policy: RetryPolicy, now: TimePoint) -> bool:
        """Admission requires EVERY layer to allow the request."""
        return all(layer.allow_request(now) for layer in self._policy_chain(policy))

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
        has already finished, or is already being abandoned, is left alone.
        """
        if ctx.root.done or ctx.cancelling or ctx.abandoned:
            return
        now = sim.timestep
        # Tentative: these only stand if something was still cancellable. A node
        # that has already committed to a real outcome (its finalize is in
        # flight) must keep that outcome — marking the root cancelled regardless
        # rewrote genuine failures as CANCELLED and hid them from the breaker.
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
                self._rollback_policy(self.cfg.retry, ctx.pending_retry_context, deferred=True)
                ctx.pending_retry_context = None
            self._complete_root(ctx.root, now)
            return

        control = ctx.control
        if control is not None and control.cancel():
            # cancel() drove the service's teardown, which called back into
            # _on_attempt_done and completed the root already.
            return
        # Nothing was cancellable: either no teardown was registered, or the
        # attempt has already committed to an outcome (its finalize is running
        # right now, reachable re-entrantly through a sibling's teardown). Undo
        # the tentative marks so that outcome is reported as itself rather than
        # being rewritten as CANCELLED, and leave the root open —
        # _on_attempt_done will close it. Completing it here would strand that
        # callback and trip the "Root request was already done!" guard.
        ctx.cancelling = False
        ctx.root.cancelled = False
        # The abandonment itself still stands, though: without this the retry
        # branch of _on_attempt_done would launch a fresh attempt into a subtree
        # nobody is waiting on any more — the load propagate_cancellation exists
        # to eliminate — and book the root by whatever that late attempt did.
        ctx.abandoned = True

    def start_request(
        self,
        sim: Simulator,
        on_done: Callable[[RootRequest], None] | None = None,
        override_global_deadline: TimePoint | None = None,
    ):
        root_req = RootRequest()

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
                # The request never reached the service, so it observed no
                # queue. Reporting 0 injected a fabricated "queue is empty"
                # sample into queue_avg_at_attempt_end at exactly the moments
                # the breaker is shedding hardest, inverting the signal.
                # None is the same convention service_time_ns already uses here.
                queue_size_at_end=None,
            )
            root_req.add_attempt(req)
            self._complete_root(root_req, sim.timestep)
            return root_req

        # One budget deposit per ADMITTED request (Finagle's RetryFilter
        # deposits per request that enters the filter, not per outcome), so
        # retry budgets are a ratio of requests rather than of successes. Below
        # the gate, so a request an outer circuit breaker sheds does not fund
        # retries for the few requests that do get through.
        self._notify_request_start(self.cfg.retry, sim.timestep)

        # Start first attempt
        # Create dummy request that will be replaced in _start_attempt
        dummy_req = Request(
            service="",
            interval=TimeInterval(begin=sim.timestep, end=sim.timestep),
            deadline=None,
        )
        initial_ctx = AttemptCtx(root=root_req, req=dummy_req)
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
        occupancy_censored: bool,
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
        ctx.req.service_time_censored = occupancy_censored

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

        # An upstream abandoned this root while the attempt above was already
        # committing to its outcome, so the cancel could not take effect. That
        # outcome is genuine and has just been reported as itself (and fed to
        # the policy and the timeout controller — the service really produced
        # it), but the root is abandoned: no further attempts, and it is
        # classified as cancelled rather than by the reason of an attempt whose
        # caller had already given up.
        if ctx.abandoned:
            ctx.root.cancelled = True
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
