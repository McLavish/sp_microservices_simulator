"""Policy feedback carries the attempt (`RetryPolicy.record_attempt`).

Every layer of a client's policy is told how each attempt ended. `add_result`
says only "success or not, and when"; `record_attempt` adds the context of the
attempt -- its number, why it ended, the request it belongs to -- for a policy
whose bookkeeping depends on WHICH attempt it was. The default hands the outcome
to `add_result`, so every policy written before the hook is fed exactly as it
was.

Also here: `RetryBudgetPolicy.min_retries_per_sec`, the time-based floor under
the budget's ratio.
"""

import unittest

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import DropReason, RequestContext, TimeInterval, ms_to_ns, s_to_ns
from msim.faults import PartialFailure
from msim.retry_policies import (
    CountBasedCircuitBreakerPolicy,
    DelayDecision,
    FixedBackoffRetryPolicy,
    RetryBudgetPolicy,
    RetryByCausePolicy,
    RetryContext,
    RetryPolicy,
)
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController


def _service(*, fail_until_ms: float | None = None, median_ms: float = 10) -> ServiceRuntime:
    failures = []
    if fail_until_ms is not None:
        failures.append(PartialFailure(TimeInterval(0, ms_to_ns(fail_until_ms)), p_fail=1.0))
    return ServiceRuntime(
        cfg=ServiceConfig(
            name="svc",
            latency_median=ms_to_ns(median_ms),
            latency_lognorm_sigma=0.0,
            workers=8,
            partial_failures=failures,
        )
    ).bind()


def _client(service, policy: RetryPolicy, *, tenant: str | None = None, attempt_ms: float = 1000):
    return ClientRuntime(
        cfg=ClientConfig(
            retry=policy,
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(attempt_ms)
            ),
            tenant=tenant,
        ),
        service=service,
    )


class _Told(FixedBackoffRetryPolicy):
    """A backoff that records what it is told and what it is asked."""

    def __init__(self, max_attempts: int) -> None:
        super().__init__(max_attempts=max_attempts, delay=0)
        self.told: list[tuple[RetryContext, bool]] = []
        self.asked: list[RetryContext] = []
        self.refunded: list[RetryContext] = []

    def record_attempt(self, context: RetryContext, success: bool) -> None:
        self.told.append((context, success))

    def next_delay(self, context: RetryContext) -> DelayDecision:
        self.asked.append(context)
        return super().next_delay(context)

    def rollback(self, context: RetryContext) -> None:
        self.refunded.append(context)


class TestRecordAttempt(unittest.TestCase):
    def test_every_attempt_is_reported_with_its_number_and_cause(self):
        # Attempts end at 10, 20 (inside the fault) and 30 ms (after it).
        policy = _Told(max_attempts=4)
        sim = Simulator(seed=1)
        _client(_service(fail_until_ms=25), policy).start_request(sim)
        sim.run()

        self.assertEqual(
            [(c.attempt, c.drop_reason, ok) for c, ok in policy.told],
            [
                (1, DropReason.SERVER_FAILURE, False),
                (2, DropReason.SERVER_FAILURE, False),
                (3, DropReason.NONE, True),
            ],
        )
        self.assertEqual([c.now for c, _ in policy.told], sorted(c.now for c, _ in policy.told))

    def test_the_cause_is_the_one_the_client_booked(self):
        """An attempt cut by the client's deadline is told as DEADLINE."""
        policy = _Told(max_attempts=1)
        sim = Simulator(seed=1)
        _client(_service(median_ms=50), policy, attempt_ms=20).start_request(sim)
        sim.run()
        self.assertEqual(
            [(c.attempt, c.drop_reason, ok) for c, ok in policy.told],
            [(1, DropReason.DEADLINE, False)],
        )

    def test_the_decision_is_asked_with_the_same_context_it_was_told_with(self):
        policy = _Told(max_attempts=3)
        sim = Simulator(seed=1)
        _client(_service(fail_until_ms=1000), policy).start_request(sim)
        sim.run()

        self.assertEqual(len(policy.told), 3)
        self.assertEqual(len(policy.asked), 3)  # asked after every failure, the last one too
        for (told, _), asked in zip(policy.told, policy.asked, strict=True):
            self.assertIs(told, asked)

    def test_a_vetoed_retry_is_refunded_with_that_context_too(self):
        """The global deadline vetoes the retry the policy granted."""
        policy = _Told(max_attempts=3)
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=policy,
                timeout_controller=StaticTimeoutController(
                    global_timeout=ms_to_ns(10), attempt_timeout=ms_to_ns(1000)
                ),
            ),
            service=_service(fail_until_ms=1000, median_ms=10),
        )
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()
        # One attempt, cut at the 10 ms global deadline; the retry would start at
        # the deadline, so it is never issued.
        self.assertEqual(len(policy.told), 1)
        self.assertEqual(len(policy.refunded), 1)
        self.assertIs(policy.refunded[0], policy.told[0][0])

    def test_the_request_context_comes_with_it(self):
        policy = _Told(max_attempts=2)
        sim = Simulator(seed=1)
        _client(_service(fail_until_ms=15), policy, tenant="acme").start_request(sim)
        sim.run()
        self.assertEqual(
            [c.request for c, _ in policy.told],
            [
                RequestContext(tenant="acme", attempt=1, is_retry=False),
                RequestContext(tenant="acme", attempt=2, is_retry=True),
            ],
        )

    def test_a_cancelled_attempt_is_not_reported(self):
        """A sibling fails at 10 ms and the 50 ms call is cancelled: no evidence."""
        slow_policy = _Told(max_attempts=3)
        fast_policy = _Told(max_attempts=1)
        parent = ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="P", latency_median=ms_to_ns(1), latency_lognorm_sigma=0.0, workers=4
                )
            ).bind(),
            propagate_cancellation=True,
        )
        fast = _client(_service(fail_until_ms=1000, median_ms=10), fast_policy)
        slow = _client(_service(median_ms=50), slow_policy)
        parent.add_downstream(DownstreamEdge(client=fast, target=parent))
        parent.add_downstream(DownstreamEdge(client=slow, target=parent))
        sim = Simulator(seed=1)
        _client(parent, FixedBackoffRetryPolicy(max_attempts=1, delay=0)).start_request(sim)
        sim.run()

        self.assertEqual(len(fast_policy.told), 1)
        self.assertEqual(slow.roots[0].attempts[-1].drop_reason, DropReason.CANCELLED)
        self.assertEqual(slow_policy.told, [])


class TestTheDefaultFeedsAddResult(unittest.TestCase):
    def test_a_policy_with_add_result_is_fed_as_before(self):
        seen: list[tuple[bool, int]] = []

        class Windowed(FixedBackoffRetryPolicy):
            def add_result(self, success, now):
                seen.append((success, now))

        sim = Simulator(seed=1)
        _client(_service(fail_until_ms=15), Windowed(max_attempts=3, delay=0)).start_request(sim)
        sim.run()
        self.assertEqual([ok for ok, _ in seen], [False, True])
        self.assertEqual(len({now for _, now in seen}), 2)

    def test_a_policy_without_one_needs_nothing(self):
        policy = FixedBackoffRetryPolicy(max_attempts=2, delay=0)
        policy.record_attempt(RetryContext(attempt=1, now=0), False)  # no error, no effect

    def test_every_layer_of_a_stack_is_told_once_per_attempt(self):
        inner = _Told(max_attempts=3)
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(3, 3),
            success_threshold_ratio=(1, 1),
            underlying_strategy=inner,
            half_open_delay=s_to_ns(100),
        )
        budget = RetryBudgetPolicy(budget_ratio=1.0, max_retries=5, underlying_strategy=breaker)
        sim = Simulator(seed=1)
        _client(_service(fail_until_ms=1000), budget).start_request(sim)
        sim.run()
        # Three failed attempts: the breaker (through add_result) opened on the
        # third, the innermost layer was told of each.
        self.assertEqual([c.attempt for c, _ in inner.told], [1, 2, 3])
        self.assertEqual(breaker.get_state().name, "OPEN")

    def test_every_branch_is_told_and_a_shared_layer_only_once(self):
        shared = _Told(max_attempts=3)
        other = _Told(max_attempts=3)
        by_cause = RetryByCausePolicy(
            {
                DropReason.SERVER_FAILURE: shared,
                DropReason.QUEUE_FULL: shared,
                DropReason.DEADLINE: other,
            }
        )
        sim = Simulator(seed=1)
        _client(_service(fail_until_ms=15), by_cause).start_request(sim)
        sim.run()
        # Two attempts (failure, success). Both branches hear of both, and the
        # policy that sits under two causes hears of each once.
        self.assertEqual([c.attempt for c, _ in shared.told], [1, 2])
        self.assertEqual([c.attempt for c, _ in other.told], [1, 2])
        # Only the deciding branch was ASKED.
        self.assertEqual(len(shared.asked), 1)
        self.assertEqual(other.asked, [])


class _FirstAttemptFunded(RetryPolicy):
    """Retries paid for by first attempts that succeeded: what the hook is for."""

    def __init__(self, inner: RetryPolicy, per_success: float) -> None:
        self.underlying_strategy = inner
        self.per_success = per_success
        self.balance = 0.0

    def record_attempt(self, context: RetryContext, success: bool) -> None:
        if success and context.attempt == 1:
            self.balance += self.per_success

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if self.balance < 1.0:
            return DelayDecision(False, 0)
        decision = self.underlying_strategy.next_delay(context)
        if decision.allow:
            self.balance -= 1.0
        return decision


class TestAPolicyThatNeedsTheAttemptNumber(unittest.TestCase):
    def test_only_first_attempt_successes_fund_retries(self):
        # The service fails during [100 ms, 200 ms) and is healthy around it.
        service = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(1),
                latency_lognorm_sigma=0.0,
                workers=64,
                partial_failures=[
                    PartialFailure(TimeInterval(ms_to_ns(100), ms_to_ns(200)), p_fail=1.0)
                ],
            )
        ).bind()
        policy = _FirstAttemptFunded(FixedBackoffRetryPolicy(max_attempts=2, delay=0), 0.5)
        client = _client(service, policy)
        sim = Simulator(seed=1)
        # 10 healthy requests fund 5 retries; then 20 requests fail.
        for i in range(10):
            sim.schedule(ms_to_ns(5 * i), lambda: client.start_request(sim))
        for i in range(20):
            sim.schedule(ms_to_ns(100 + 2 * i), lambda: client.start_request(sim))
        sim.run()

        retried = [root for root in client.roots if len(root.attempts) == 2]
        self.assertEqual(len(client.roots), 30)
        self.assertEqual(len(retried), 5)
        # A retry that succeeded would be attempt 2 and fund nothing; here none did.
        self.assertEqual(policy.balance, 0.0)


class TestBudgetTimeFloor(unittest.TestCase):
    def _budget(self, **kwargs) -> RetryBudgetPolicy:
        return RetryBudgetPolicy(
            budget_ratio=0.1,
            max_retries=2,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=0),
            **kwargs,
        )

    def _drain(self, budget: RetryBudgetPolicy, now: int = 0) -> int:
        granted = 0
        while budget.next_delay(RetryContext(attempt=1, now=now)).allow:
            granted += 1
        return granted

    def test_off_by_default(self):
        budget = self._budget()
        self.assertEqual(self._drain(budget), 2)
        # An hour later nothing has come back: only requests fund this budget.
        self.assertEqual(self._drain(budget, now=s_to_ns(3600)), 0)

    def test_retries_accrue_with_time(self):
        budget = self._budget(min_retries_per_sec=2)
        self.assertEqual(self._drain(budget), 2)  # the full bucket
        self.assertEqual(self._drain(budget, now=ms_to_ns(499)), 0)  # 0.998 of a retry
        self.assertEqual(self._drain(budget, now=ms_to_ns(500)), 1)
        self.assertEqual(self._drain(budget, now=ms_to_ns(999)), 0)
        self.assertEqual(self._drain(budget, now=ms_to_ns(1000)), 1)

    def test_the_cap_still_holds(self):
        budget = self._budget(min_retries_per_sec=2)
        self._drain(budget)
        self.assertEqual(self._drain(budget, now=s_to_ns(60)), 2)  # max_retries, not 120

    def test_the_accrual_is_exact_however_often_it_is_consulted(self):
        for step_ms in (1, 7, 50, 333):
            with self.subTest(step_ms=step_ms):
                budget = self._budget(min_retries_per_sec=3)
                self._drain(budget)
                granted = sum(
                    self._drain(budget, now=now)
                    for now in range(ms_to_ns(step_ms), s_to_ns(10) + 1, ms_to_ns(step_ms))
                )
                # 3 per second for the part of the 10 s that was looked at.
                looked_at_ms = (10_000 // step_ms) * step_ms
                self.assertEqual(granted, 3 * looked_at_ms // 1000)

    def test_it_adds_to_the_deposits(self):
        budget = self._budget(min_retries_per_sec=1)
        self._drain(budget)
        for _ in range(5):  # 5 requests at 10 % are half a retry
            budget.on_request_start(ms_to_ns(100))
        self.assertEqual(self._drain(budget, now=ms_to_ns(400)), 0)  # 0.5 + 0.4
        self.assertEqual(self._drain(budget, now=ms_to_ns(500)), 1)  # 0.5 + 0.5

    def test_a_refund_gives_back_exactly_one_retry(self):
        budget = self._budget(min_retries_per_sec=1)
        self._drain(budget)
        context = RetryContext(attempt=1, now=s_to_ns(1))
        self.assertTrue(budget.next_delay(context).allow)
        budget.rollback(context)
        self.assertEqual(self._drain(budget, now=s_to_ns(1)), 1)

    def test_the_rate_must_be_positive_and_finite(self):
        for bad in (0, -1.0, float("nan"), float("inf"), 1e-12):
            with self.subTest(rate=bad), self.assertRaises(ValueError):
                self._budget(min_retries_per_sec=bad)


if __name__ == "__main__":
    unittest.main()
