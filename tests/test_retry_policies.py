import random
import unittest

from caller import ClientConfig, ClientRuntime
from core import DropReason, TimeInterval, s_to_ns
from faults import PartialFailure
from retry_policies import (
    BurstyRateLimiterPolicy,
    CBState,
    CountBasedCircuitBreakerPolicy,
    ExponentialBackoffRetryPolicy,
    ExponentialBackoffWithJitterRetryPolicy,
    FixedBackoffRetryPolicy,
    FixedWindowBurstyLimiterPolicy,
    JitterMode,
    LeakyRateLimiterPolicy,
    NoRetryPolicy,
    RetryBudgetPolicy,
    RetryContext,
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

    def test_retry_budget_depletes_and_refills(self):
        policy = RetryBudgetPolicy(
            budget_ratio=0.1,  # success_award = 10
            max_retries=2,  # max_tokens = 200, retry_cost = 100
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=100, delay=7),
        )
        self.assertEqual(policy.get_token_balance(), 200)
        self.assertEqual(policy.next_delay(RetryContext(attempt=1, now=0)), (True, 7))  # 200 -> 100
        self.assertEqual(policy.next_delay(RetryContext(attempt=2, now=0)), (True, 7))  # 100 -> 0
        self.assertEqual(policy.next_delay(RetryContext(attempt=3, now=0)), (False, 0))  # depleted

        policy.add_result(False, now=0)  # failures do not refill
        self.assertEqual(policy.get_token_balance(), 0)

        for _ in range(10):  # 10 successes * 10 award = 100 = one retry
            policy.add_result(True, now=0)
        self.assertEqual(policy.get_token_balance(), 100)
        self.assertEqual(policy.next_delay(RetryContext(attempt=4, now=0)), (True, 7))

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

    def test_one_third_ratio_earns_a_retry_after_three_successes(self):
        policy = self._drained(1 / 3)
        for _ in range(3):
            policy.add_result(True, now=0)
        self.assertTrue(policy.can_retry())  # float accumulation gave 99.99999999999999

    def test_one_seventh_ratio_earns_a_retry_after_seven_successes(self):
        policy = self._drained(1 / 7)
        for _ in range(6):
            policy.add_result(True, now=0)
        self.assertFalse(policy.can_retry())
        policy.add_result(True, now=0)
        self.assertTrue(policy.can_retry())

    def test_shipped_ratios_unchanged(self):
        for ratio, needed in ((0.05, 20), (0.1, 10), (0.2, 5)):
            policy = self._drained(ratio)
            for _ in range(needed - 1):
                policy.add_result(True, now=0)
            self.assertFalse(policy.can_retry())
            policy.add_result(True, now=0)
            self.assertTrue(policy.can_retry())


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


if __name__ == "__main__":
    unittest.main()
