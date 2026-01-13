import unittest

from retry_policies import (
    FixedBackoffRetryPolicy,
    LeakyRateLimiterPolicy,
    RetryContext,
)


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


if __name__ == "__main__":
    unittest.main()
