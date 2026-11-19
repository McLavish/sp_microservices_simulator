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


#: The values of `ClientConfig.server_deadline`.
SERVER_DEADLINE_MODES = ("attempt", "global", "none")


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
    # What the target is told about this client's deadlines:
    #   "attempt" (default) -> the attempt's own deadline, min(global deadline,
    #     now + attempt timeout). The server enforces it: it reports a waiting
    #     attempt at that instant and cuts one in service, freeing the worker.
    #     A stack that sends its deadline with the call (gRPC's grpc-timeout).
    #   "global" -> only the request's global deadline (None when it has none).
    #     The attempt timeout stays with the client, which enforces it with a
    #     timer of its own and walks away; the server keeps working on the
    #     attempt -- an ORPHAN -- until it finishes or the global deadline cuts
    #     it. A client-side timeout the server never hears about.
    #   "none" -> nothing: the client enforces every deadline alone.
    server_deadline: str = "attempt"

    def __post_init__(self) -> None:
        if self.server_deadline not in SERVER_DEADLINE_MODES:
            raise ValueError(
                f"server_deadline must be one of {', '.join(SERVER_DEADLINE_MODES)}; "
                f"got {self.server_deadline!r}"
            )


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

        # What the target is told (`ClientConfig.server_deadline`). When that is
        # later than the deadline this client works to -- or nothing at all --
        # the client has to enforce its own deadline: a timer, armed BEFORE the
        # attempt is submitted so that it precedes whatever the target schedules
        # and therefore wins a tie at the deadline instant, as the half-open rule
        # requires. An attempt told its real deadline needs none (the server
        # reports at that instant itself), which is every attempt by default.
        mode = self.cfg.server_deadline
        if mode == "attempt":
            disclosed = req_deadline
        else:
            disclosed = ctx.root.global_deadline if mode == "global" else None
            if req_deadline is not None and (disclosed is None or req_deadline < disclosed):
                on_done = self._race_own_deadline(sim, on_done, req_deadline)

        if ctx.request_ctx is None:
            # No request context: the call every target has always received.
            ctx.control = self.service.submit_attempt(sim, on_done, attempt_deadline=disclosed)
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
            sim, on_done, attempt_deadline=disclosed, ctx=ctx.request_ctx
        )

    def _race_own_deadline(
        self, sim: Simulator, deliver: AttemptDoneCallback, deadline: TimePoint
    ) -> AttemptDoneCallback:
        """Race one attempt's answer against this client's own deadline.

        Returns the callback to hand to the target instead of `deliver`. Whichever
        comes first -- the target's report or the deadline -- is delivered, once;
        the other is ignored. When the deadline wins, the client books a
        `DEADLINE` drop that carries no queue size and no occupancy (it walked
        away without learning either) and moves on: feedback, retry decision,
        the next attempt. The attempt itself is NOT withdrawn. It stays at the
        target as an orphan, holding its worker or its queue slot until the
        target is done with it, and its eventual answer goes nowhere -- no
        record, no policy feedback. Nothing holds a handle on an orphan, so a
        later cancellation of the request cannot reach it either.

        A report that arrives first disarms the timer, and that includes the
        `CANCELLED` report of an attempt an upstream cancels.
        """
        settled = False

        def on_done(
            success: bool,
            svc_time: TimeDuration,
            drop_reason: DropReason,
            queue_size: int | None,
            occupancy_censored: bool,
        ) -> None:
            nonlocal settled
            if settled:
                return  # the orphan's answer: the client left at its deadline
            settled = True
            timer.cancelled = True
            deliver(success, svc_time, drop_reason, queue_size, occupancy_censored)

        def on_deadline() -> None:
            nonlocal settled
            if settled:
                return
            settled = True
            deliver(False, 0, DropReason.DEADLINE, None, False)

        # `max`: a deadline that is already behind us cannot be scheduled; the
        # timer then fires now, after a target that answers synchronously.
        timer = sim.schedule(max(deadline, sim.timestep), on_deadline)
        return on_done

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


class _LayerControl(AttemptControl):
    """Cancellation handle for an attempt a `RetryLayer` is serving.

    Like `transport._LinkControl`: what cancelling means depends on where the
    inner request is, so `cancel()` asks the attempt instead of running a bound
    teardown.
    """

    __slots__ = ("_attempt",)

    def __init__(self, attempt: "_LayerAttempt") -> None:
        super().__init__()
        self._attempt = attempt

    def cancel(self) -> bool:
        if self._settled:
            return False
        return self._attempt.cancel()


class _LayerAttempt:
    """One attempt of the caller in front of a `RetryLayer`: one inner REQUEST."""

    def __init__(self, on_done: AttemptDoneCallback) -> None:
        self._on_done = on_done
        self.control = _LayerControl(self)
        #: The inner request. None until `start_request` has returned, by which
        #: time it may already be over.
        self.root: RootRequest | None = None
        #: A cancellation took effect: the inner request ends as CANCELLED.
        self._cancelled = False

    def finished(self, root: RootRequest) -> None:
        """The inner client's `on_done`: report the request as one attempt."""
        self.control._settle()
        last = root.attempts[-1]
        if last.success:
            drop = DropReason.NONE
        elif self._cancelled:
            drop = DropReason.CANCELLED
        else:
            # The reason the request itself ended with. An inner root can carry
            # the `cancelled` flag here without a cancellation having taken
            # effect (the attempt had already committed when it was asked to
            # stop): its outcome is genuine and is reported as itself.
            drop = last.drop_reason
            if drop in (DropReason.NONE, DropReason.CANCELLED):
                drop = DropReason.SERVER_FAILURE
        self._on_done(
            last.success,
            last.service_time_ns or 0,
            drop,
            last.queue_size_at_end,
            last.service_time_censored,
        )

    def cancel(self) -> bool:
        root = self.root
        if root is None or root.done or root.cancel is None:
            return False
        self._cancelled = True
        root.cancel()
        if root.done:
            # Torn down: a pending retry was dropped or the attempt in flight
            # was abandoned, and `finished` has already reported CANCELLED.
            return True
        # Nothing was cancellable: the attempt in flight has committed to an
        # outcome. The inner client will issue no further attempt, and that
        # outcome arrives through `finished` as what it is.
        self._cancelled = False
        return False


@dataclass
class RetryLayer:
    """A retry loop as a target: a `ClientRuntime` that another client calls.

    For two retry loops on one hop, each with its own policy and its own
    timeouts: a caller that retries, in front of a service (or a sidecar, or a
    gateway) that retries the same call itself.

        inner = ClientRuntime(cfg=service_side_cfg, service=service)
        outer = ClientRuntime(cfg=caller_cfg, service=RetryLayer(inner))

    One attempt of the caller is one whole REQUEST of the inner client, which
    makes as many attempts at its own target as its policy allows. The caller
    sees one answer -- the outcome of the inner request's last attempt, with
    that attempt's occupancy, queue size and censoring -- when the inner client
    has finished, and the caller's attempt deadline is the inner request's
    global deadline (tightened by the inner client's own, if it has one).

    The two loops keep separate books. The inner client's records are attempts
    at the real target, under the target's name; the caller's records are
    booked under the LAYER's name (`name`, by default `<target>.retries`), so a
    consumer that counts attempts per service does not count a call twice.

    Cancelling an attempt of the layer cancels the inner request: a pending
    retry is dropped, an attempt in flight is abandoned, and `cancel()` returns
    True with `CANCELLED` already reported. If the attempt in flight has
    committed to an outcome, `cancel()` returns False, the inner client stops
    retrying, and that outcome is reported as itself.
    """

    client: ClientRuntime
    name: str | None = None
    cfg: ServiceConfig = field(init=False)

    def __post_init__(self) -> None:
        target = self.client.service.cfg
        self.cfg = replace(target, name=self.name or f"{target.name}.retries")

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: AttemptDoneCallback,
        attempt_deadline: TimePoint | None = None,
        *,
        ctx: RequestContext | None = None,
    ) -> AttemptControl:
        # The handle exists BEFORE the inner request starts: the request can be
        # over, and reported, by the time `start_request` returns.
        attempt = _LayerAttempt(on_done)
        if ctx is None:
            attempt.root = self.client.start_request(
                sim, on_done=attempt.finished, override_global_deadline=attempt_deadline
            )
        else:
            attempt.root = self.client.start_request(
                sim, on_done=attempt.finished, override_global_deadline=attempt_deadline, ctx=ctx
            )
        return attempt.control
