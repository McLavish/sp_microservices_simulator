"""What a controller may change on a client between two events, and when it takes effect.

`docs/request-handling.md` ("Changing a policy while a run is in progress")
names four settings that can be re-tuned while a run is paused. The three that
are plain attributes are pinned here, on a station whose timing is known to the
millisecond; the fourth, `RetryBudgetPolicy.set_budget_ratio`, has its
arithmetic in `test_retry_policies.py` and a seeded run in
`test_trace_goldens.py`.
"""

import unittest

from msim.caller import ClientConfig, ClientRuntime
from msim.core import DropReason, RootRequest, TimeInterval, ms_to_ns, ns_to_ms, s_to_ns
from msim.faults import PartialFailure
from msim.retry_policies import FixedBackoffRetryPolicy
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController


class TestChangesBetweenEvents(unittest.TestCase):
    """One request at t = 0 on a station that takes 10 ms, to the nanosecond or nearly."""

    def _client(self, *, p_fail: float, attempt_ms: float | None, global_ms: float | None = None):
        self.sim = Simulator(seed=5)
        station = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(10),
                latency_lognorm_sigma=0.0,  # floored to 1e-6: 10 ms give or take a few ns
                workers=4,
                partial_failures=[PartialFailure(TimeInterval(0, s_to_ns(10)), p_fail=p_fail)],
            )
        ).bind()
        self.backoff = FixedBackoffRetryPolicy(max_attempts=3, delay=ms_to_ns(20))
        self.timeouts = StaticTimeoutController(
            global_timeout=None if global_ms is None else ms_to_ns(global_ms),
            attempt_timeout=None if attempt_ms is None else ms_to_ns(attempt_ms),
        )
        return ClientRuntime(
            cfg=ClientConfig(retry=self.backoff, timeout_controller=self.timeouts),
            service=station,
        )

    def _starts_ms(self, root: RootRequest) -> list[int]:
        return [round(ns_to_ms(attempt.interval.begin)) for attempt in root.attempts]

    def test_left_alone_the_request_makes_three_attempts(self):
        client = self._client(p_fail=1.0, attempt_ms=None)
        root = client.start_request(self.sim)
        self.sim.run()
        self.assertEqual(self._starts_ms(root), [0, 30, 60])
        self.assertEqual({a.drop_reason for a in root.attempts}, {DropReason.SERVER_FAILURE})

    def test_max_attempts_applies_at_the_next_decision_of_a_request_in_flight(self):
        client = self._client(p_fail=1.0, attempt_ms=None)
        root = client.start_request(self.sim)
        self.sim.run(until=ms_to_ns(35))  # the second attempt is at the station
        self.backoff.max_attempts = 2
        self.sim.run()
        self.assertEqual(self._starts_ms(root), [0, 30])  # asked at 40 ms: no third

    def test_a_retry_already_scheduled_stays_scheduled(self):
        client = self._client(p_fail=1.0, attempt_ms=None)
        root = client.start_request(self.sim)
        self.sim.run(until=ms_to_ns(45))  # the third attempt was granted at 40 ms, for 60 ms
        self.backoff.max_attempts = 1
        self.backoff.delay = ms_to_ns(500)
        self.sim.run()
        self.assertEqual(self._starts_ms(root), [0, 30, 60])
        later = client.start_request(self.sim)  # a new request finds the new settings
        self.sim.run()
        self.assertEqual(len(later.attempts), 1)

    def test_delay_applies_at_the_next_decision(self):
        client = self._client(p_fail=1.0, attempt_ms=None)
        root = client.start_request(self.sim)
        self.sim.run(until=ms_to_ns(35))
        self.backoff.delay = ms_to_ns(50)
        self.sim.run()
        self.assertEqual(self._starts_ms(root), [0, 30, 90])  # 40 ms + the new 50 ms

    def test_attempt_timeout_applies_to_attempts_that_start_afterwards(self):
        client = self._client(p_fail=0.0, attempt_ms=5)
        root = client.start_request(self.sim)
        self.sim.run(until=ms_to_ns(2))  # the first attempt is in service, deadline 5 ms
        self.timeouts.attempt_timeout = ms_to_ns(50)
        self.sim.run()
        first, second = root.attempts
        self.assertEqual(first.drop_reason, DropReason.DEADLINE)  # kept its 5 ms
        self.assertEqual(first.interval.end, ms_to_ns(5))
        self.assertEqual(second.deadline, ms_to_ns(25 + 50))  # started at 5 + 20 under 50 ms
        self.assertTrue(second.success)

    def test_global_timeout_applies_to_requests_that_start_afterwards(self):
        client = self._client(p_fail=1.0, attempt_ms=None, global_ms=1000)
        first = client.start_request(self.sim)
        self.timeouts.global_timeout = ms_to_ns(25)
        second = client.start_request(self.sim)
        self.sim.run()
        self.assertEqual(first.global_deadline, ms_to_ns(1000))
        self.assertEqual(len(first.attempts), 3)
        self.assertEqual(second.global_deadline, ms_to_ns(25))
        self.assertEqual(len(second.attempts), 1)  # the retry at 30 ms is past its deadline


if __name__ == "__main__":
    unittest.main()
