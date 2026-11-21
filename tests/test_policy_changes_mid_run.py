"""What a controller may change on a client between two events, and when it takes effect.

`docs/request-handling.md` ("Changing a policy while a run is in progress")
names the settings that can be re-tuned while a run is paused. The three of a
client that are plain attributes are pinned here, on a station whose timing is
known to the millisecond, and so is the one of a limiter
(`TenantRetryQuota.set_share`); `RetryBudgetPolicy.set_budget_ratio` has its
arithmetic in `test_retry_policies.py` and a seeded run in
`test_trace_goldens.py`.
"""

import unittest

from msim.admission import AdmissionGate, TenantRetryQuota
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


class TestALimitersShareBetweenEvents(unittest.TestCase):
    """One tenant whose every call fails, behind a gate with a retry quota.

    A request every 10 ms from t = 0, up to three attempts each, no backoff: an
    attempt ends 10 ms after it started and the next one asks at once.
    """

    def _run(self, change_at_ms: float | None = None, share: float = 0.0):
        sim = Simulator(seed=5)
        station = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(10),
                latency_lognorm_sigma=0.0,
                workers=64,
                partial_failures=[PartialFailure(TimeInterval(0, s_to_ns(10)), p_fail=1.0)],
            )
        ).bind()
        quota = TenantRetryQuota(share=1.0)
        gate = AdmissionGate(station, [quota])
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=3, delay=0),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=None
                ),
                tenant="a",
            ),
            service=gate,
        )
        roots: list[RootRequest] = []
        for index in range(20):  # at 0, 10, ..., 190 ms
            sim.schedule(ms_to_ns(10 * index), lambda: roots.append(client.start_request(sim)))
        if change_at_ms is not None:
            sim.run(until=ms_to_ns(change_at_ms))
            quota.set_share("a", share)
        sim.run()
        return roots, gate, sim

    def _retries(self, roots: list[RootRequest], *, since_ms: float = 0.0):
        """`(asked at, ms; how it ended)` of every retry from `since_ms` on."""
        return sorted(
            (round(ns_to_ms(attempt.interval.begin)), attempt.drop_reason.name)
            for root in roots
            for attempt in root.attempts[1:]
            if attempt.interval.begin >= ms_to_ns(since_ms)
        )

    def test_a_share_of_one_lets_in_as_many_retries_as_first_attempts(self):
        roots, gate, _ = self._run()
        counts = gate.counts["a"]
        self.assertEqual(counts.attempts - counts.retries_asked, 20)  # the first attempts
        self.assertEqual(counts.retries_asked, counts.retries_admitted + counts.rejected)
        self.assertGreater(counts.rejected, 0)  # a third attempt is one retry too many
        self.assertLessEqual(counts.retries_admitted, 20)
        self.assertEqual(gate.outstanding, 0)

    def test_a_new_share_applies_to_the_next_retry_that_asks(self):
        untouched, _, sim = self._run()
        closed, gate, closed_sim = self._run(95, share=0.0)
        # Up to 95 ms the two runs are one run ...
        early = [r for r in self._retries(untouched) if r[0] < 95]
        self.assertEqual([r for r in self._retries(closed) if r[0] < 95], early)
        self.assertIn("SERVER_FAILURE", {reason for _, reason in early})  # some were let in
        # ... and from then on every retry that asks is refused, the first at 100 ms.
        late = self._retries(closed, since_ms=95)
        self.assertEqual(late[0][0], 100)
        self.assertEqual({reason for _, reason in late}, {"REJECTED"})
        # The retry let in at 90 ms was at the station when the share changed, and stayed.
        self.assertIn((90, "SERVER_FAILURE"), early)
        self.assertEqual(gate.outstanding, 0)
        self.assertLess(closed_sim._seq, sim._seq)  # fewer attempts, nothing else

    def test_setting_the_share_a_tenant_has_changes_nothing(self):
        untouched, _, sim = self._run()
        same, _, same_sim = self._run(95, share=1.0)
        self.assertEqual(self._retries(same), self._retries(untouched))
        self.assertEqual(same_sim._seq, sim._seq)
        self.assertEqual(same_sim.rng().getstate(), sim.rng().getstate())


if __name__ == "__main__":
    unittest.main()
