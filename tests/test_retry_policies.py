import random
import unittest

from caller import ClientConfig, ClientRuntime
from core import DropReason, TimeInterval, s_to_ns
from faults import PartialFailure
from retry_policies import (
    BurstyRateLimiterPolicy,
    CBState,
    CountBasedCircuitBreakerPolicy,
    DelayDecision,
    ExponentialBackoffRetryPolicy,
    ExponentialBackoffWithJitterRetryPolicy,
    FixedBackoffRetryPolicy,
    FixedWindowBurstyLimiterPolicy,
    JitterMode,
    LeakyRateLimiterPolicy,
    NoRetryPolicy,
    RetryBudgetPolicy,
    RetryContext,
    RetryPolicy,
    TimeBasedCircuitBreakerPolicy,
)
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController


class TestRetryPolicies(unittest.TestCase):
    def test_fixed_backoff_max_attempts(self):
        policy = FixedBackoffRetryPolicy(max_attempts=2, delay=10)
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 10))
        self.assertEqual(policy.next_delay(RetryContext(attempt=2, now=0)), (False, 0))

    def test_leaky_rate_limiter_max_attempts(self):
        policy = LeakyRateLimiterPolicy(max_requests=10, max_attempts=2, period=100)
        allow1, _ = policy.next_delay(RetryContext(attempt=1, now=0))
        allow2, _ = policy.next_delay(RetryContext(attempt=2, now=0))
        self.assertTrue(allow1)
        self.assertFalse(allow2)

    def test_bursty_rate_limiter_refills_by_period(self):
        policy = BurstyRateLimiterPolicy(
            max_requests=2,
            refill_rate=2,
            period=100,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=0),
        )

        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 0))
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 0))
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=1)), (False, 0))
        self.assertAlmostEqual(policy._tokens, 0.02)

        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=50)), (True, 0))
        self.assertAlmostEqual(policy._tokens, 0.0)

    def test_bursty_rate_limiter_rejects_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "max_requests"):
            BurstyRateLimiterPolicy(
                max_requests=0,
                refill_rate=1,
                period=100,
                underlying_strategy=NoRetryPolicy(),
            )
        with self.assertRaisesRegex(ValueError, "period"):
            BurstyRateLimiterPolicy(
                max_requests=1,
                refill_rate=1,
                period=0,
                underlying_strategy=NoRetryPolicy(),
            )

    # --- pure retry policies -------------------------------------------------

    def test_fixed_backoff_returns_configured_delay(self):
        policy = FixedBackoffRetryPolicy(max_attempts=3, delay=25)
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 25))
        self.assertEqual(policy.next_delay(RetryContext(attempt=2, now=0)), (True, 25))
        self.assertEqual(policy.next_delay(RetryContext(attempt=3, now=0)), (False, 0))

    def test_no_retry_always_denies(self):
        policy = NoRetryPolicy()
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (False, 0))
        self.assertEqual(policy.next_delay(RetryContext(attempt=5, now=0)), (False, 0))

    def test_exponential_backoff_doubles_and_caps(self):
        policy = ExponentialBackoffRetryPolicy(max_attempts=5, initial_delay=10, max_delay=50)
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 10))  # 10*2^0
        self.assertEqual(policy.next_delay(RetryContext(attempt=2, now=0)), (True, 20))  # 10*2^1
        self.assertEqual(policy.next_delay(RetryContext(attempt=3, now=0)), (True, 40))  # 10*2^2
        self.assertEqual(policy.next_delay(RetryContext(attempt=4, now=0)), (True, 50))  # 80 -> cap
        self.assertEqual(policy.next_delay(RetryContext(attempt=5, now=0)), (False, 0))  # exhausted

    def test_exponential_jitter_full_mode_bounds(self):
        policy = ExponentialBackoffWithJitterRetryPolicy(
            max_attempts=4,
            initial_delay=100,
            max_delay=10_000,
            rng=random.Random(0),
            jitter_mode=JitterMode.FULL,
        )
        for attempt in range(1, 4):
            exp = min(100 * (2 ** (attempt - 1)), 10_000)
            allow, delay = policy.next_delay(RetryContext(attempt=attempt, now=0))
            self.assertTrue(allow)
            self.assertGreaterEqual(delay, 0)
            self.assertLessEqual(delay, exp)
        self.assertEqual(policy.next_delay(RetryContext(attempt=4, now=0)), (False, 0))

    def test_exponential_jitter_equal_mode_bounds(self):
        policy = ExponentialBackoffWithJitterRetryPolicy(
            max_attempts=4,
            initial_delay=100,
            max_delay=10_000,
            rng=random.Random(0),
            jitter_mode=JitterMode.EQUAL,
        )
        for attempt in range(1, 4):
            exp = min(100 * (2 ** (attempt - 1)), 10_000)
            allow, delay = policy.next_delay(RetryContext(attempt=attempt, now=0))
            self.assertTrue(allow)
            self.assertGreaterEqual(delay, exp // 2)  # EQUAL jitter floors at exp/2
            self.assertLessEqual(delay, exp)

    def test_exponential_jitter_is_deterministic_under_seed(self):
        def run():
            policy = ExponentialBackoffWithJitterRetryPolicy(
                max_attempts=5,
                initial_delay=100,
                max_delay=10_000,
                rng=random.Random(42),
            )
            return [policy.next_delay(RetryContext(attempt=i, now=0)) for i in range(1, 5)]

        self.assertEqual(run(), run())

    def test_exponential_jitter_decorrelated_not_implemented(self):
        policy = ExponentialBackoffWithJitterRetryPolicy(
            max_attempts=3,
            initial_delay=100,
            max_delay=10_000,
            rng=random.Random(0),
            jitter_mode=JitterMode.DECORRELATED,
        )
        with self.assertRaises(NotImplementedError):
            policy.next_delay(RetryContext(attempt=1, now=0))

    # --- circuit breakers ----------------------------------------------------

    def test_count_circuit_breaker_lifecycle(self):
        policy = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(2, 2),
            success_threshold_ratio=(2, 2),
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=5),
            half_open_delay=100,
        )
        # closed: delegates to the underlying policy
        self.assertEqual(policy.get_state(), CBState.CLOSED)
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 5))

        # filling the failure window opens the circuit
        policy.add_result(False, now=0)
        self.assertEqual(policy.get_state(), CBState.CLOSED)
        policy.add_result(False, now=1)
        self.assertEqual(policy.get_state(), CBState.OPEN)

        # open: denies retries and blocks requests before the half-open delay
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=1)), (False, 0))
        self.assertFalse(policy.allow_request(now=50))

        # after the half-open delay it flips to half-open and admits a probe
        self.assertTrue(policy.allow_request(now=101))
        self.assertEqual(policy.get_state(), CBState.HALF_OPEN)

        # a full batch of successful probes closes the circuit again
        policy.add_result(True, now=101)
        self.assertEqual(policy.get_state(), CBState.HALF_OPEN)
        policy.add_result(True, now=102)
        self.assertEqual(policy.get_state(), CBState.CLOSED)

    def test_count_circuit_breaker_failed_probe_reopens(self):
        policy = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(1, 1),
            success_threshold_ratio=(2, 2),
            underlying_strategy=NoRetryPolicy(),
            half_open_delay=10,
        )
        policy.add_result(False, now=0)  # window maxlen 1 -> opens immediately
        self.assertEqual(policy.get_state(), CBState.OPEN)
        self.assertTrue(policy.allow_request(now=20))  # -> half-open
        policy.add_result(True, now=20)
        policy.add_result(False, now=21)  # 1 success out of 2 -> reopen
        self.assertEqual(policy.get_state(), CBState.OPEN)

    def test_time_circuit_breaker_min_requests_gating(self):
        policy = TimeBasedCircuitBreakerPolicy(
            failure_threshold_rate=0.5,
            success_threshold_rate=0.5,
            min_requests=3,
            window_duration=1000,
            underlying_strategy=NoRetryPolicy(),
            half_open_delay=100,
        )
        # 100% failure rate but too few samples -> stays closed
        policy.add_result(False, now=0)
        policy.add_result(False, now=1)
        self.assertEqual(policy.get_state(), CBState.CLOSED)
        # crossing min_requests with rate over threshold opens it
        policy.add_result(False, now=2)
        self.assertEqual(policy.get_state(), CBState.OPEN)

    def test_time_circuit_breaker_evicts_stale_results(self):
        policy = TimeBasedCircuitBreakerPolicy(
            failure_threshold_rate=0.5,
            success_threshold_rate=0.5,
            min_requests=2,
            window_duration=100,
            underlying_strategy=NoRetryPolicy(),
            half_open_delay=10,
        )
        policy.add_result(False, now=0)
        # the gap evicts the first result, so only one sample remains -> closed
        policy.add_result(False, now=200)
        self.assertEqual(policy.get_state(), CBState.CLOSED)

    def test_time_circuit_breaker_recovery(self):
        policy = TimeBasedCircuitBreakerPolicy(
            failure_threshold_rate=0.5,
            success_threshold_rate=0.5,
            min_requests=2,
            window_duration=10_000,
            underlying_strategy=NoRetryPolicy(),
            half_open_delay=50,
        )
        policy.add_result(False, now=0)
        policy.add_result(False, now=1)
        self.assertEqual(policy.get_state(), CBState.OPEN)
        self.assertFalse(policy.allow_request(now=10))
        self.assertTrue(policy.allow_request(now=51))
        self.assertEqual(policy.get_state(), CBState.HALF_OPEN)
        policy.add_result(True, now=51)
        policy.add_result(True, now=52)
        self.assertEqual(policy.get_state(), CBState.CLOSED)

    def test_time_circuit_breaker_resets_window_on_close(self):
        policy = TimeBasedCircuitBreakerPolicy(
            failure_threshold_rate=0.5,
            success_threshold_rate=0.5,
            min_requests=2,
            window_duration=10_000,
            underlying_strategy=NoRetryPolicy(),
            half_open_delay=50,
        )
        policy.add_result(False, now=0)
        policy.add_result(False, now=1)
        self.assertTrue(policy.allow_request(now=51))
        policy.add_result(True, now=51)
        policy.add_result(True, now=52)
        self.assertEqual(policy.get_state(), CBState.CLOSED)
        # The half-open probes must not seed the closed window: two fresh
        # failures alone (100% rate over min_requests) re-open the circuit.
        policy.add_result(False, now=53)
        policy.add_result(False, now=54)
        self.assertEqual(policy.get_state(), CBState.OPEN)

    # --- rate limiters / budget ----------------------------------------------

    def test_leaky_rate_limiter_spacing(self):
        # period / max_requests = 25 ns between admitted requests
        policy = LeakyRateLimiterPolicy(max_requests=4, max_attempts=10, period=100)
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 0))
        self.assertEqual(policy.next_delay(RetryContext(attempt=2, now=0)), (True, 25))
        self.assertEqual(policy.next_delay(RetryContext(attempt=3, now=0)), (True, 50))
        # once wall-clock passes the reserved slot, no extra delay is added
        self.assertEqual(policy.next_delay(RetryContext(attempt=4, now=1000)), (True, 0))

    def test_leaky_rate_limiter_keeps_the_fractional_remainder(self):
        """Rounding the reserved slot every step dropped the same remainder each
        time, so the achieved spacing was floor(period / max_requests) and the
        bucket ran permanently fast: 3 per 100 ns admitted a 4th at 99 ns."""
        policy = LeakyRateLimiterPolicy(max_requests=3, max_attempts=10, period=100)
        delays = [policy.next_delay(RetryContext(attempt=i, now=0)).delay for i in range(1, 5)]
        self.assertEqual(delays, [0, 34, 67, 100])
        self.assertEqual(sum(1 for d in delays if d < 100), policy.max_requests)

        # The exact-half case rounded to 2 ns instead of 2.5 ns, a permanent
        # 25% rate inflation.
        halved = LeakyRateLimiterPolicy(max_requests=2, max_attempts=10, period=5)
        self.assertEqual(
            [halved.next_delay(RetryContext(attempt=i, now=0)).delay for i in range(1, 4)],
            [0, 3, 5],
        )

    def test_retry_budget_depletes_and_refills(self):
        policy = RetryBudgetPolicy(
            budget_ratio=0.1,  # deposit = 10
            max_retries=2,  # max_tokens = 200, retry_cost = 100
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=100, delay=7),
        )
        self.assertEqual(policy.get_token_balance(), 200)
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 7))  # 200 -> 100
        self.assertEqual(policy.next_delay(RetryContext(attempt=2, now=0)), (True, 7))  # 100 -> 0
        self.assertEqual(policy.next_delay(RetryContext(attempt=3, now=0)), (False, 0))  # depleted

        # Finagle semantics: the denominator is REQUESTS, so a request refills
        # the budget regardless of how it turns out. (Depositing only on success
        # would tighten the budget during exactly the failure storms it exists
        # to damp.)
        for _ in range(10):  # 10 requests * 10 deposit = 100 = one retry
            policy.on_request_start(now=0)
        self.assertEqual(policy.get_token_balance(), 100)
        self.assertEqual(policy.next_delay(RetryContext(attempt=4, now=0)), (True, 7))

    def test_retry_budget_deposits_regardless_of_outcome(self):
        policy = RetryBudgetPolicy(
            budget_ratio=0.1,
            max_retries=2,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=100, delay=7),
        )
        while policy.can_retry():  # drain
            policy.next_delay(RetryContext(attempt=1, now=0))
        self.assertEqual(policy.get_token_balance(), 0)
        # A request that will ultimately fail still deposits.
        for _ in range(10):
            policy.on_request_start(now=0)
        self.assertTrue(policy.can_retry())

    def test_retry_budget_does_not_consume_when_underlying_denies(self):
        policy = RetryBudgetPolicy(
            budget_ratio=0.1,
            max_retries=2,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=1, delay=7),  # always denies
        )
        before = policy.get_token_balance()
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (False, 0))
        self.assertEqual(policy.get_token_balance(), before)

    def test_bursty_rate_limiter_bursts_then_denies(self):
        policy = BurstyRateLimiterPolicy(
            max_requests=3,
            refill_rate=1,
            period=100,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=100, delay=0),
        )
        for _ in range(3):  # burst capacity = 3
            self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 0))
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (False, 0))

    def test_bursty_rate_limiter_refill_caps_at_capacity(self):
        policy = BurstyRateLimiterPolicy(
            max_requests=2,
            refill_rate=2,
            period=100,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=100, delay=0),
        )
        policy.next_delay(RetryContext(attempt=1, now=0))  # drain both tokens
        policy.next_delay(RetryContext(attempt=1, now=0))
        # a long gap would over-refill, but tokens are capped at max_requests=2
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=10_000)), (True, 0))
        self.assertAlmostEqual(policy._tokens, 1.0)  # capped to 2, then consumed 1

    def test_bursty_rate_limiter_does_not_consume_when_underlying_denies(self):
        policy = BurstyRateLimiterPolicy(
            max_requests=2,
            refill_rate=1,
            period=100,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=1, delay=0),  # denies
        )
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (False, 0))
        self.assertEqual(policy._tokens, 2.0)

    def test_fixed_window_bursty_limits_per_window(self):
        policy = FixedWindowBurstyLimiterPolicy(
            max_requests=2,
            period=100,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=100, delay=0),
        )
        # window [0, 100): 2 admitted, 3rd denied
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 0))
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=10)), (True, 0))
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=20)), (False, 0))
        # the next window resets the quota
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=100)), (True, 0))

    def test_fixed_window_bursty_next_refill_time(self):
        policy = FixedWindowBurstyLimiterPolicy(
            max_requests=1,
            period=100,
            underlying_strategy=NoRetryPolicy(),
        )
        self.assertEqual(policy.get_next_refill_time(now=20), 100)
        self.assertEqual(policy.get_next_refill_time(now=150), 200)

    def test_open_circuit_blocks_new_root_request(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=1,
                latency_lognorm_sigma=0.1,
                workers=1,
                queue_capacity=None,
                partial_failures=[
                    PartialFailure(
                        duration=TimeInterval(begin=0, end=s_to_ns(1)),
                        p_fail=1.0,
                    )
                ],
            )
        ).bind()
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(1, 1),
            success_threshold_ratio=(1, 1),
            underlying_strategy=NoRetryPolicy(),
            half_open_delay=s_to_ns(1),
        )
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=breaker,
                timeout_controller=StaticTimeoutController(),
            ),
            service=svc,
        )

        client.start_request(sim)
        sim.run()
        self.assertEqual(client.roots[-1].attempts[-1].drop_reason, DropReason.SERVER_FAILURE)

        client.start_request(sim)

        self.assertEqual(client.attempts_total, 2)
        self.assertEqual(len(client.roots), 2)
        self.assertEqual(client.roots[-1].attempts[-1].drop_reason, DropReason.CIRCUIT_OPEN)

    def test_completion_callback_exceptions_propagate(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=1,
                latency_lognorm_sigma=0.1,
                workers=1,
                queue_capacity=None,
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(),
            ),
            service=svc,
        )

        def on_done(_root):
            raise RuntimeError("callback failed")

        client.start_request(sim, on_done=on_done)

        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            sim.run()


class TestTimeBasedBreakerCounters(unittest.TestCase):
    def test_incremental_counters_match_full_rescan(self):
        rng = random.Random(7)
        policy = TimeBasedCircuitBreakerPolicy(
            failure_threshold_rate=0.5,
            success_threshold_rate=0.6,
            min_requests=5,
            window_duration=s_to_ns(2),
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=3, delay=0),
            half_open_delay=s_to_ns(1),
        )
        now = 0
        for _ in range(3000):
            now += rng.randint(0, s_to_ns(0.01))
            policy.allow_request(now)
            policy.add_result(rng.random() < 0.6, now)
            self.assertEqual(
                policy._failures_in_window,
                sum(1 for _, ok in policy._window if not ok),
            )


class TestRetryBudgetExactArithmetic(unittest.TestCase):
    def _drained(self, ratio):
        policy = RetryBudgetPolicy(
            budget_ratio=ratio,
            max_retries=1,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=0),
        )
        while policy.can_retry():  # drain the initial burst
            policy.next_delay(RetryContext(attempt=1, now=0))
        return policy

    def test_one_third_ratio_earns_a_retry_after_three_requests(self):
        policy = self._drained(1 / 3)
        for _ in range(3):
            policy.on_request_start(now=0)
        self.assertTrue(policy.can_retry())  # float accumulation gave 99.99999999999999

    def test_one_seventh_ratio_earns_a_retry_after_seven_requests(self):
        policy = self._drained(1 / 7)
        for _ in range(6):
            policy.on_request_start(now=0)
        self.assertFalse(policy.can_retry())
        policy.on_request_start(now=0)
        self.assertTrue(policy.can_retry())

    def test_shipped_ratios_unchanged(self):
        for ratio, needed in ((0.05, 20), (0.1, 10), (0.2, 5)):
            policy = self._drained(ratio)
            for _ in range(needed - 1):
                policy.on_request_start(now=0)
            self.assertFalse(policy.can_retry())
            policy.on_request_start(now=0)
            self.assertTrue(policy.can_retry())

    def test_tiny_ratio_is_still_a_ratio(self):
        """A ratio below ~1/20000 used to rationalize to a zero deposit, so the
        budget never refilled: 1e-5, 1e-7 and 1e-9 all became "max_retries
        retries for the whole run", silently and identically."""
        policy = self._drained(1e-5)
        self.assertEqual((policy._deposit, policy._retry_cost), (1, 100_000))
        for _ in range(99_999):
            policy.on_request_start(now=0)
        self.assertFalse(policy.can_retry())
        policy.on_request_start(now=0)
        self.assertTrue(policy.can_retry())

    def test_unrepresentable_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "budget_ratio"):
            RetryBudgetPolicy(
                budget_ratio=1e-12, max_retries=1, underlying_strategy=NoRetryPolicy()
            )


class TestAdmissionGateOrdering(unittest.TestCase):
    """Finagle's RetryFilter deposits per request that ENTERS the filter.

    `_notify_request_start` therefore has to run below the admission gate:
    a request an open breaker sheds never becomes a retryable request, so
    funding retries with it hands the few admitted requests the whole
    shed storm's worth of budget at exactly the moment it should bind.
    """

    def test_shed_request_does_not_deposit_into_a_wrapped_budget(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=1,
                latency_lognorm_sigma=0.1,
                workers=4,
                partial_failures=[PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=1.0)],
            )
        ).bind()
        budget = RetryBudgetPolicy(
            budget_ratio=0.1,
            max_retries=1,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=2, delay=0),
        )
        while budget.can_retry():  # drain, so deposits are visible below the cap
            budget.next_delay(RetryContext(attempt=1, now=0))
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(1, 1),
            success_threshold_ratio=(1, 1),
            underlying_strategy=budget,
            half_open_delay=s_to_ns(1000),  # stays open for the whole test
        )
        client = ClientRuntime(
            cfg=ClientConfig(retry=breaker, timeout_controller=StaticTimeoutController()),
            service=svc,
        )

        client.start_request(sim)
        sim.run()
        self.assertEqual(breaker.get_state(), CBState.OPEN)
        after_admitted = budget._tokens

        for _ in range(50):
            client.start_request(sim)
        self.assertEqual(
            [r.attempts[-1].drop_reason for r in client.roots[1:]],
            [DropReason.CIRCUIT_OPEN] * 50,
        )
        self.assertEqual(budget._tokens, after_admitted)


class TestBreakerConfigValidation(unittest.TestCase):
    def test_count_breaker_rejects_impossible_ratios(self):
        def build(failure=(1, 1), success=(1, 1), half_open_delay=0):
            return CountBasedCircuitBreakerPolicy(
                failure_threshold_ratio=failure,
                success_threshold_ratio=success,
                underlying_strategy=NoRetryPolicy(),
                half_open_delay=half_open_delay,
            )

        with self.assertRaisesRegex(ValueError, "failure_threshold_ratio"):
            build(failure=(3, 2))
        with self.assertRaisesRegex(ValueError, "failure_threshold_ratio"):
            build(failure=(0, 0))
        with self.assertRaisesRegex(ValueError, "success_threshold_ratio"):
            build(success=(-1, 2))
        with self.assertRaisesRegex(ValueError, "half_open_delay"):
            build(half_open_delay=-1)

    def test_time_breaker_rejects_invalid_config(self):
        def build(
            failure_threshold_rate=0.5,
            success_threshold_rate=0.5,
            min_requests=2,
            window_duration=1000,
            half_open_delay=10,
            half_open_min_requests=1,
        ):
            return TimeBasedCircuitBreakerPolicy(
                failure_threshold_rate=failure_threshold_rate,
                success_threshold_rate=success_threshold_rate,
                min_requests=min_requests,
                window_duration=window_duration,
                underlying_strategy=NoRetryPolicy(),
                half_open_delay=half_open_delay,
                half_open_min_requests=half_open_min_requests,
            )

        # A negative window evicts every prior result, so the breaker could
        # never reach min_requests and silently never opened.
        with self.assertRaisesRegex(ValueError, "window_duration"):
            build(window_duration=-1)
        with self.assertRaisesRegex(ValueError, "failure_threshold_rate"):
            build(failure_threshold_rate=1.5)
        with self.assertRaisesRegex(ValueError, "success_threshold_rate"):
            build(success_threshold_rate=-0.1)
        with self.assertRaisesRegex(ValueError, "min_requests"):
            build(min_requests=0)
        with self.assertRaisesRegex(ValueError, "half_open_min_requests"):
            build(half_open_min_requests=0)
        with self.assertRaisesRegex(ValueError, "half_open_delay"):
            build(half_open_delay=-1)


class TestDeferredRollbackIsOptIn(unittest.TestCase):
    """A policy that charges in `next_delay` must not be refunded late by
    default: the deferred path (a cancel arriving mid-backoff) can rewind a
    reservation that belongs to a different live root."""

    def test_base_default_is_fail_safe(self):
        self.assertFalse(RetryPolicy().supports_deferred_rollback)
        for policy in (
            LeakyRateLimiterPolicy(max_requests=1, max_attempts=2, period=100),
            RetryBudgetPolicy(budget_ratio=0.1, max_retries=1, underlying_strategy=NoRetryPolicy()),
            BurstyRateLimiterPolicy(
                max_requests=1, refill_rate=1, period=100, underlying_strategy=NoRetryPolicy()
            ),
            FixedWindowBurstyLimiterPolicy(
                max_requests=1, period=100, underlying_strategy=NoRetryPolicy()
            ),
        ):
            with self.subTest(policy=type(policy).__name__):
                self.assertFalse(policy.supports_deferred_rollback)

    def test_a_new_side_effecting_policy_is_skipped_without_opting_in(self):
        class ConcurrencyLimiterPolicy(RetryPolicy):
            """Reserves a permit in next_delay, returns it in rollback."""

            def __init__(self):
                self.refunds = 0

            def next_delay(self, context):
                return DelayDecision(True, 0)

            def rollback(self, context):
                self.refunds += 1

        policy = ConcurrencyLimiterPolicy()
        client = ClientRuntime(
            cfg=ClientConfig(retry=policy, timeout_controller=StaticTimeoutController()),
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="svc", latency_median=1, latency_lognorm_sigma=0.1, workers=1
                )
            ).bind(),
        )
        context = RetryContext(attempt=1, now=0)
        client._rollback_policy(policy, context, deferred=True)
        self.assertEqual(policy.refunds, 0)
        client._rollback_policy(policy, context)
        self.assertEqual(policy.refunds, 1)


class TestBackoffValidation(unittest.TestCase):
    def test_negative_fixed_backoff_rejected(self):
        with self.assertRaises(ValueError):
            FixedBackoffRetryPolicy(max_attempts=2, delay=-1)

    def test_nonpositive_attempts_rejected(self):
        with self.assertRaises(ValueError):
            FixedBackoffRetryPolicy(max_attempts=0, delay=0)

    def test_negative_exponential_delays_rejected(self):
        with self.assertRaises(ValueError):
            ExponentialBackoffRetryPolicy(max_attempts=2, initial_delay=-5, max_delay=10)

    def test_negative_jitter_delays_rejected(self):
        with self.assertRaises(ValueError):
            ExponentialBackoffWithJitterRetryPolicy(
                max_attempts=2, initial_delay=-5, max_delay=10, rng=random.Random(1)
            )


class TestVetoedRetriesAreRefunded(unittest.TestCase):
    """A retry the global deadline vetoes must not be charged.

    `ClientRuntime._on_attempt_done` asks the policy for a delay — which is
    side-effecting for every throttling policy — and then discards the decision
    when `next_start >= global_deadline`. Without `RetryPolicy.rollback` those
    never-issued retries still consumed budget/tokens, silently tightening the
    effective constraint.
    """

    def _run(self, policy, attempt_timeout_ms=30, global_timeout_ms=30):
        from core import ms_to_ns

        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(10),
                latency_lognorm_sigma=1e-6,
                workers=4,
                partial_failures=[
                    PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=1.0),
                ],
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=policy,
                timeout_controller=StaticTimeoutController(
                    global_timeout=ms_to_ns(global_timeout_ms),
                    attempt_timeout=ms_to_ns(attempt_timeout_ms),
                ),
            ),
            service=svc,
        )
        client.start_request(sim)
        sim.run()
        root = client.roots[0]
        return len(root.attempts) - 1  # retries actually issued

    def test_retry_budget_charges_only_issued_retries(self):
        from core import ms_to_ns

        policy = RetryBudgetPolicy(
            budget_ratio=0.1,
            max_retries=5,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=ms_to_ns(50)),
        )
        before = policy._tokens
        issued = self._run(policy)
        charged = (before - policy._tokens) // policy._retry_cost
        # The backoff (50ms) overshoots the 30ms global deadline, so the retry
        # is authorised then vetoed: zero issued, and so zero charged.
        self.assertEqual(issued, 0)
        self.assertEqual(charged, 0)
        self.assertEqual(policy._tokens, before)

    def test_token_bucket_limiters_charge_only_issued_retries(self):
        from core import ms_to_ns

        for policy in (
            BurstyRateLimiterPolicy(
                max_requests=5,
                refill_rate=0.0,
                period=s_to_ns(1),
                underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=ms_to_ns(50)),
            ),
            FixedWindowBurstyLimiterPolicy(
                max_requests=5,
                period=s_to_ns(1),
                underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=ms_to_ns(50)),
            ),
        ):
            with self.subTest(policy=type(policy).__name__):
                before = policy._tokens
                issued = self._run(policy)
                self.assertEqual(before - policy._tokens, issued)

    def test_leaky_limiter_releases_vetoed_slot(self):
        policy = LeakyRateLimiterPolicy(max_requests=10, max_attempts=10, period=s_to_ns(1))
        # `next_delay` DOES reserve a slot here (leaky delay is 0 on an idle
        # bucket, so _next_at briefly advances to 105 ms), but the retry is never
        # issued: the 5 ms global deadline vetoes it and `rollback` puts _next_at
        # back. The invariant is that _next_at advances once per ISSUED retry,
        # never for a vetoed one.
        issued = self._run(policy, attempt_timeout_ms=5, global_timeout_ms=5)
        self.assertEqual(issued, 0)
        self.assertIsNone(policy._next_at)

    def test_rollback_reaches_wrapped_layers(self):
        from core import ms_to_ns

        inner = FixedWindowBurstyLimiterPolicy(
            max_requests=5,
            period=s_to_ns(1),
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=ms_to_ns(50)),
        )
        outer = RetryBudgetPolicy(budget_ratio=0.1, max_retries=5, underlying_strategy=inner)
        outer_before, inner_before = outer._tokens, inner._tokens
        issued = self._run(outer)
        self.assertEqual(issued, 0)
        # Both layers charged during next_delay; both must refund.
        self.assertEqual(outer._tokens, outer_before)
        self.assertEqual(inner._tokens, inner_before)


class TestPolicyFeedbackReachesWrappedLayers(unittest.TestCase):
    """Results and request starts must reach EVERY layer, not just the outermost.

    `_update_policy_results` and `_notify_request_start` walk the chain, and
    CLAUDE.md warns that circuit breakers and retry budgets update *only*
    through those paths. Every other test drives an `add_result`-bearing policy
    as the top-level `ClientConfig.retry`, where reaching the outermost layer
    alone is enough — so a walk collapsed to `policy` itself changed nothing.
    """

    def _client(self, policy, p_fail: float):
        failures = [PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=p_fail)] if p_fail else []
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=1,
                latency_lognorm_sigma=0.1,
                workers=4,
                partial_failures=failures,
            )
        ).bind()
        return ClientRuntime(
            cfg=ClientConfig(retry=policy, timeout_controller=StaticTimeoutController()),
            service=svc,
        )

    def test_attempt_results_reach_a_wrapped_breaker(self):
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(2, 2),
            success_threshold_ratio=(1, 1),
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=3, delay=0),
            half_open_delay=s_to_ns(1000),  # stays open once tripped
        )
        # RetryBudgetPolicy has no add_result of its own, so the breaker only
        # ever sees an outcome if the walk descends into underlying_strategy.
        client = self._client(
            RetryBudgetPolicy(budget_ratio=1.0, max_retries=5, underlying_strategy=breaker),
            p_fail=1.0,
        )
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()

        self.assertEqual(len(client.roots[0].attempts), 2)  # both failures fed in
        self.assertEqual(breaker.get_state(), CBState.OPEN)

    def test_request_starts_reach_a_wrapped_budget(self):
        budget = RetryBudgetPolicy(
            budget_ratio=0.1,
            max_retries=1,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=0),
        )
        while budget.can_retry():  # drain, so deposits are visible below the cap
            budget.next_delay(RetryContext(attempt=1, now=0))
        self.assertEqual(budget._tokens, 0)
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(1, 1),
            success_threshold_ratio=(1, 1),
            underlying_strategy=budget,
            half_open_delay=0,
        )
        client = self._client(breaker, p_fail=0.0)  # healthy: no retries, no withdrawals
        sim = Simulator(seed=1)
        for _ in range(3):
            client.start_request(sim)
            sim.run()

        self.assertEqual([root.attempts[-1].success for root in client.roots], [True] * 3)
        # One deposit per admitted root request, through the wrapper.
        self.assertEqual(budget._tokens, 3 * budget._deposit)


if __name__ == "__main__":
    unittest.main()
