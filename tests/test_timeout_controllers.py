import unittest
from functools import partial

from caller import ClientConfig, ClientRuntime
from core import TimeInterval, ms_to_ns, s_to_ns
from faults import LatencyInjection
from retry_policies import FixedBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import AdaptiveTimeoutController
from workload import Workload


def _feed(ctrl: AdaptiveTimeoutController, latency: int, n: int, success: bool = True):
    for _ in range(n):
        ctrl.record_result(success, latency)


class TestAdaptiveTimeoutController(unittest.TestCase):
    def test_returns_initial_timeouts_before_min_samples(self):
        ctrl = AdaptiveTimeoutController(
            initial_global_timeout=s_to_ns(1), initial_attempt_timeout=ms_to_ns(100)
        )
        _feed(ctrl, ms_to_ns(10), n=ctrl.min_samples - 1)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(100))
        self.assertEqual(ctrl.get_global_timeout(), s_to_ns(1))

        # One more sample crosses the min_samples threshold and adaptation kicks in.
        _feed(ctrl, ms_to_ns(10), n=1)
        self.assertEqual(ctrl.get_attempt_timeout(), 2 * ms_to_ns(10))

    def test_adapts_without_initial_attempt_timeout(self):
        ctrl = AdaptiveTimeoutController()
        self.assertIsNone(ctrl.get_attempt_timeout())
        _feed(ctrl, ms_to_ns(10), n=ctrl.min_samples)
        self.assertEqual(ctrl.get_attempt_timeout(), 2 * ms_to_ns(10))

    def test_adapts_to_percentile_with_safety_margin(self):
        ctrl = AdaptiveTimeoutController(safety_margin=3.0)
        _feed(ctrl, ms_to_ns(20), n=20)
        # Constant latencies make every percentile exact: 20ms * 3.0.
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(60))

    def test_window_tracks_latency_shift_both_directions(self):
        ctrl = AdaptiveTimeoutController()
        _feed(ctrl, ms_to_ns(10), n=ctrl.window_size)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(20))

        # Fully displace the window with slower latencies: timeout rises.
        _feed(ctrl, ms_to_ns(50), n=ctrl.window_size)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(100))

        # And back: the sliding window forgets the spike.
        _feed(ctrl, ms_to_ns(10), n=ctrl.window_size)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(20))

    def test_min_timeout_clamp(self):
        ctrl = AdaptiveTimeoutController()
        # 0.1ms latencies would yield a 0.2ms timeout; the 1ms floor wins.
        _feed(ctrl, 100_000, n=20)
        self.assertEqual(ctrl.get_attempt_timeout(), ctrl.min_timeout)

    def test_max_timeout_clamp(self):
        ctrl = AdaptiveTimeoutController(max_timeout=ms_to_ns(30))
        _feed(ctrl, ms_to_ns(50), n=20)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(30))

    def test_failure_latencies_do_not_feed_percentile_window(self):
        ctrl = AdaptiveTimeoutController()
        _feed(ctrl, ms_to_ns(10), n=10)
        # Failed attempts report huge latencies (e.g. deadline drops at attempt
        # end); if they entered the window the timeout would explode. Success
        # rate stays at the 0.5 boundary, so the inflation path stays off.
        _feed(ctrl, s_to_ns(1), n=10, success=False)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(20))

    def test_low_success_rate_inflates_timeout(self):
        ctrl = AdaptiveTimeoutController()
        _feed(ctrl, ms_to_ns(10), n=10)
        # 10 successes / 25 results = 0.4 < 0.5: timeout = 20ms * (1/0.4).
        _feed(ctrl, 0, n=15, success=False)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(50))

    def test_inflation_capped_at_10x(self):
        ctrl = AdaptiveTimeoutController()
        _feed(ctrl, ms_to_ns(10), n=10)
        # Enough failures to push every success out of the success-rate window:
        # rate 0.0 is floored at 0.1, so inflation tops out at 10x.
        _feed(ctrl, 0, n=ctrl.success_rate_window + 10, success=False)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(200))

    def test_max_timeout_caps_inflated_timeout(self):
        ctrl = AdaptiveTimeoutController(max_timeout=ms_to_ns(30))
        _feed(ctrl, ms_to_ns(10), n=20)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(20))
        # 20 successes / 50 results = 0.4: inflation asks for 50ms, but the
        # configured max_timeout is a hard cap and must win.
        _feed(ctrl, 0, n=30, success=False)
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(30))

    def test_global_timeout_never_adapts(self):
        ctrl = AdaptiveTimeoutController(
            initial_global_timeout=s_to_ns(1), initial_attempt_timeout=ms_to_ns(100)
        )
        _feed(ctrl, ms_to_ns(50), n=100)
        _feed(ctrl, 0, n=100, success=False)
        self.assertEqual(ctrl.get_global_timeout(), s_to_ns(1))


class TestAdaptiveTimeoutConfigValidation(unittest.TestCase):
    """`initial_attempt_timeout` is served verbatim until enough SUCCESSFUL
    samples accumulate, so an initial outside `[min_timeout, max_timeout]`
    escaped bounds every adaptation path applies. Rejected at construction
    rather than clamped: clamping would override an explicit setting."""

    def test_initial_above_max_timeout_rejected(self):
        with self.assertRaisesRegex(ValueError, "initial_attempt_timeout"):
            AdaptiveTimeoutController(
                initial_attempt_timeout=ms_to_ns(100), max_timeout=ms_to_ns(30)
            )

    def test_initial_below_min_timeout_rejected(self):
        with self.assertRaisesRegex(ValueError, "initial_attempt_timeout"):
            AdaptiveTimeoutController(initial_attempt_timeout=ms_to_ns(0.05))

    def test_contradictory_bounds_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_timeout"):
            AdaptiveTimeoutController(min_timeout=ms_to_ns(10), max_timeout=ms_to_ns(5))
        with self.assertRaisesRegex(ValueError, "min_timeout"):
            AdaptiveTimeoutController(min_timeout=-1)

    def test_initial_inside_the_bounds_is_served_unchanged(self):
        ctrl = AdaptiveTimeoutController(
            initial_attempt_timeout=ms_to_ns(20), min_timeout=ms_to_ns(1), max_timeout=ms_to_ns(30)
        )
        self.assertEqual(ctrl.get_attempt_timeout(), ms_to_ns(20))


class TestAdaptiveTimeoutScenario(unittest.TestCase):
    def _run_scenario(self, seed: int):
        """Single service with a 10x latency injection from t=10s to t=20s.

        The adaptive attempt timeout starts loose (500ms), tightens to track
        the healthy p95 (~35ms), gets trapped when injected latencies exceed
        it (all attempts deadline-drop, so no new successes enter the window),
        escapes via the low-success-rate inflation path, and relaxes back
        after the fault clears.
        """
        sim = Simulator(seed=seed)
        svc_cfg = ServiceConfig(
            "svc",
            ms_to_ns(10),
            0.3,
            workers=8,
            queue_capacity=100,
            latency_injections=[
                LatencyInjection(
                    duration=TimeInterval(begin=s_to_ns(10), end=s_to_ns(20)),
                    multiplier=10,
                )
            ],
        )
        svc = ServiceRuntime(cfg=svc_cfg).bind()
        ctrl = AdaptiveTimeoutController(
            initial_global_timeout=None, initial_attempt_timeout=ms_to_ns(500)
        )
        client = ClientRuntime(
            ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=3, delay=ms_to_ns(5)),
                timeout_controller=ctrl,
            ),
            service=svc,
        )

        probes: dict[str, int | None] = {}

        def probe(label: str):
            probes[label] = ctrl.get_attempt_timeout()

        sim.schedule(s_to_ns(9.5), partial(probe, "baseline"))
        sim.schedule(s_to_ns(19.5), partial(probe, "injection"))
        sim.schedule(s_to_ns(29.5), partial(probe, "recovered"))

        wl = Workload(base_rps=50.0, duration_s=30, load_spikes=[])
        wl.drive(sim, lambda s: client.start_request(s))
        sim.run(until=s_to_ns(30))
        sim.run()

        return probes, client.metrics().summary()

    def test_timeout_tracks_injected_latency(self):
        probes, summary = self._run_scenario(seed=42)
        baseline = probes["baseline"]
        injection = probes["injection"]
        recovered = probes["recovered"]
        assert baseline is not None and injection is not None and recovered is not None

        # Healthy phase: adapted well below the loose 500ms initial value.
        self.assertLess(baseline, ms_to_ns(100))
        self.assertGreaterEqual(baseline, ms_to_ns(1))

        # Injection phase: the timeout followed the 10x latency upward.
        self.assertGreater(injection, 2 * baseline)

        # Recovery phase: the window flushed and the timeout came back down.
        self.assertLess(recovered, injection)
        self.assertLess(recovered, ms_to_ns(100))

        # The escape hatch worked: requests kept succeeding overall instead of
        # death-spiraling on a stale tight timeout.
        self.assertGreater(summary.succeeded / summary.total, 0.7)
        # The trap was real, though: the injection did cause deadline drops.
        self.assertGreater(summary.dropped_deadline, 0)


if __name__ == "__main__":
    unittest.main()
