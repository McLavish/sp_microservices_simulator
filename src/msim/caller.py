from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Protocol

from msim.core import (
    DropReason,
    Request,
    RequestContext,
    RootRequest,
    TimeDuration,
    TimeInterval,
    TimePoint,
)
from msim.metrics import Metrics
from msim.retry_policies import RetryContext, RetryPolicy, policy_layers
from msim.service import AttemptControl, AttemptDoneCallback, ServiceConfig
from msim.simulator import Event, Simulator
from msim.timeout_controllers import TimeoutController


class ServiceLike(Protocol):
    """Structural type for anything ClientRuntime can submit attempts to.

    Implemented by `ServiceRuntime` (single service) and `ServiceNodeRuntime`
    (a chain node that fans out downstream).

    `ctx` is the request context of the attempt (`core.RequestContext`). A client
    passes it ONLY for a request that carries one, so a target that never serves
    such requests is called with the three arguments it always was.
    """

    cfg: ServiceConfig

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: AttemptDoneCallback,
        attempt_deadline: TimePoint | None = None,
        *,
        ctx: RequestContext | None = None,
    ) -> AttemptControl: ...


@dataclass(frozen=True)
class ClientConfig:
    retry: RetryPolicy
    timeout_controller: TimeoutController
    # Who this client's requests belong to. A client with a tenant gives every
    # request it STARTS a `core.RequestContext`, which then travels with every
    # attempt that request causes, down the whole call tree. None (the default):
    # no context is created and nothing downstream is handed one. It is the ROOT
    # client's tenant that counts: a request that arrives with a context keeps it.
    tenant: str | None = None


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
    # The request context of the attempt in flight (of the last one issued, while
    # backing off): the request's context with `attempt` and `is_retry` set for
    # this client's attempt. None for a request that carries no context.
    request_ctx: RequestContext | None = None


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

    # The four traversals below all reach the layers of a composed policy through
    # `retry_policies.policy_layers`, the one walk: a wrapper is either seen by all
    # of them or by none.

    def _update_policy_results(
        self, policy: RetryPolicy, context: RetryContext, success: bool
    ) -> None:
        """Tell every layer of the policy how an attempt ended.

        EVERY layer, whichever branch the retry decision will take, so the walk
        is deliberately not given `context` (with one it follows the deciding
        branch alone, which is what a refund needs). The context is what each
        layer is TOLD: `RetryPolicy.record_attempt`, whose default feeds
        `add_result(success, now)` to the layers that have one.
        """
        for layer in policy_layers(policy):
            layer.record_attempt(context, success)

    def _notify_request_start(self, policy: RetryPolicy, now: TimePoint) -> None:
        """Announce an admitted root request to every layer of the policy chain."""
        for layer in policy_layers(policy):
            layer.on_request_start(now)

    def _rollback_policy(
        self, policy: RetryPolicy, context: RetryContext, deferred: bool = False
    ) -> None:
        """Refund a retry that ``next_delay`` authorised but we never issued.

        Every layer that charged during `next_delay` sits on the path the
        decision took, so every one of them gets the chance to refund.

        ``deferred=True`` marks a refund issued long after the charge (a cancel
        arriving mid-backoff). `RetryPolicy.rollback` is specified as undoing the
        *immediately preceding* `next_delay`, so layers that keep single-slot
        undo state are skipped: with several roots sharing one policy, a late
        refund would rewind a different root's live reservation. Skipping leaves
        the charge standing, which is conservative — it over-counts rather than
        handing the same slot out twice.
        """
        # Only the layers that took part in THIS decision: a policy that chose a
        # branch by the failure's cause is refunded in that branch alone.
        for layer in policy_layers(policy, context):
            if not deferred or layer.supports_deferred_rollback:
                layer.rollback(context)

    def _allow_policy_request(self, policy: RetryPolicy, now: TimePoint) -> bool:
        """Admission requires EVERY layer to allow the request."""
        return all(layer.allow_request(now) for layer in policy_layers(policy))

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
        *,
        ctx: RequestContext | None = None,
    ):
        """Start one request; `ctx` is the context it arrives with, if any.

        A request that arrives with a context keeps it (a chain node hands its
        own attempt's context to every downstream call). One that arrives with
        none gets a fresh context when this client has a `tenant`, and none
        otherwise.
        """
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
        if ctx is None and self.cfg.tenant is not None:
            ctx = RequestContext(tenant=self.cfg.tenant)
        initial_ctx = AttemptCtx(root=root_req, req=dummy_req, request_ctx=ctx)
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
        if ctx.request_ctx is None:
            # No request context: the call every target has always received.
            ctx.control = self.service.submit_attempt(sim, on_done, attempt_deadline=req_deadline)
            return
        # The two hop-relative fields, for THIS attempt of THIS client. Attempts
        # are recorded on the root as they end and never overlap, so the next
        # one's number is the count so far plus one. The flag is sticky: once
        # True -- here or at a hop above -- it stays True.
        number = ctx.root.attempt_count() + 1
        ctx.request_ctx = replace(
            ctx.request_ctx, attempt=number, is_retry=ctx.request_ctx.is_retry or number > 1
        )
        ctx.control = self.service.submit_attempt(
            sim, on_done, attempt_deadline=req_deadline, ctx=ctx.request_ctx
        )

    def _on_attempt_done(
        self,
        ctx: AttemptCtx,
        sim: Simulator,
        success: bool,
        svc_time: TimeDuration,
        drop_reason: DropReason,
        queue_size: int | None,
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
        # deadline while queued); real service times are floored at 100 us.
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

        # One context for this attempt's outcome. Every policy layer is told
        # with it, and for a failed attempt the retry decision is asked with it
        # and, should the retry never be issued, refunded with it.
        retry_context = RetryContext(
            attempt=ctx.root.attempt_count(),
            now=sim.timestep,
            # The cause as booked above: DEADLINE for an attempt that ran out of
            # time, whatever the service reported; NONE for a success.
            drop_reason=drop_reason,
            request=ctx.request_ctx,
        )

        # Update circuit breaker state if policy supports it
        self._update_policy_results(self.cfg.retry, retry_context, success_effective)

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
