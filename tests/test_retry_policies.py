import unittest

from caller import ClientConfig, ClientRuntime
from core import DropReason, TimeInterval, s_to_ns
from faults import PartialFailure
from retry_policies import (
    BurstyRateLimiterPolicy,
    CountBasedCircuitBreakerPolicy,
    FixedBackoffRetryPolicy,
    LeakyRateLimiterPolicy,
    NoRetryPolicy,
    RetryContext,
)
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController


class TestRetryPolicies(unittest.TestCase):
    def test_fixed_backoff_max_attempts(self):
        policy = FixedBackoffRetryPolicy(max_attempts=2, delay=10)
        self.assertEqual(policy.next_delay(RetryContext(attempt=1)), (True, 10))
        self.assertEqual(policy.next_delay(RetryContext(attempt=2)), (False, 0))

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


if __name__ == "__main__":
    unittest.main()
