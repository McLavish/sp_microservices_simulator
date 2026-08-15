import unittest

from core import TimeInterval, s_to_ns
from faults import LoadSpike
from simulator import Simulator
from workload import Workload


class TestWorkload(unittest.TestCase):
    def test_zero_rps_no_arrivals(self):
        sim = Simulator(seed=1)
        arrivals = []

        def on_arrival(_sim):
            arrivals.append(_sim.timestep)

        wl = Workload(base_rps=0.0, duration_s=2, load_spikes=[])
        wl.drive(sim, on_arrival)

        sim.run(until=s_to_ns(wl.duration_s))
        sim.run()

        self.assertEqual(arrivals, [])

    def test_positive_rps_emits_arrivals(self):
        sim = Simulator(seed=1)
        arrivals = []

        def on_arrival(_sim):
            arrivals.append(_sim.timestep)

        wl = Workload(base_rps=5.0, duration_s=1, load_spikes=[])
        wl.drive(sim, on_arrival)

        sim.run(until=s_to_ns(wl.duration_s))
        sim.run()

        self.assertGreaterEqual(len(arrivals), 1)


class TestExactNHPP(unittest.TestCase):
    def _arrival_times(self, seed, base_rps, spikes, duration_s=1):
        sim = Simulator(seed=1)
        times = []
        wl = Workload(base_rps=base_rps, duration_s=duration_s, load_spikes=spikes, rng_seed=seed)
        wl.drive(sim, lambda s: times.append(s.timestep))
        sim.run()
        return times

    def test_first_arrival_is_never_at_time_zero(self):
        for seed in range(50):
            times = self._arrival_times(seed, base_rps=100.0, spikes=[])
            self.assertGreater(times[0], 0)

    def test_short_spike_is_not_skipped(self):
        # Review scenario: 1 rps base, x100 on [0.1, 0.2) s. Correct NHPP mean
        # over [0,1): 0.9*1 + 0.1*100 = 10.9 arrivals, ~10 inside the spike.
        spike = LoadSpike(
            duration=TimeInterval(begin=s_to_ns(0.1), end=s_to_ns(0.2)), rps_multiplier=100.0
        )
        counts, in_spike = [], []
        for seed in range(300):
            times = self._arrival_times(seed, base_rps=1.0, spikes=[spike])
            counts.append(len(times))
            in_spike.append(sum(1 for t in times if s_to_ns(0.1) <= t < s_to_ns(0.2)))
        mean = sum(counts) / len(counts)
        self.assertGreater(mean, 9.5)
        self.assertLess(mean, 12.5)
        # P(zero spike arrivals) = exp(-10) under the correct process.
        self.assertGreaterEqual(sum(1 for n in in_spike if n > 0) / len(in_spike), 0.99)

    def test_constant_rate_total_count_is_poisson_scale(self):
        times = self._arrival_times(seed=5, base_rps=100.0, spikes=[], duration_s=10)
        self.assertGreater(len(times), 850)
        self.assertLess(len(times), 1150)

    def test_zero_rate_workload_emits_nothing(self):
        times = self._arrival_times(seed=5, base_rps=0.0, spikes=[])
        self.assertEqual(times, [])

    def test_stop_halts_future_arrivals(self):
        sim = Simulator(seed=1)
        times = []
        wl = Workload(base_rps=100.0, duration_s=10, load_spikes=[], rng_seed=1)
        wl.drive(sim, lambda s: times.append(s.timestep))
        sim.run(until=s_to_ns(1))
        wl.stop()
        count_at_stop = len(times)
        sim.run()
        self.assertEqual(len(times), count_at_stop)


class TestFractionalDuration(unittest.TestCase):
    def _times(self, duration_s):
        sim = Simulator(seed=1)
        times = []
        wl = Workload(base_rps=200.0, duration_s=duration_s, load_spikes=[], rng_seed=3)
        wl.drive(sim, lambda s: times.append(s.timestep))
        sim.run()
        return times

    def test_no_arrivals_start_after_fractional_horizon(self):
        times = self._times(1.6)
        self.assertGreater(len(times), 0)
        self.assertLess(max(times), s_to_ns(1.6))

    def test_sub_second_duration_still_generates_traffic(self):
        self.assertGreater(len(self._times(0.4)), 0)


class TestWorkloadSeeding(unittest.TestCase):
    def _arrivals(self, rng_seed):
        sim = Simulator(seed=1)
        times = []
        wl = Workload(base_rps=180.0, duration_s=1, load_spikes=[], rng_seed=rng_seed)
        wl.drive(sim, lambda s: times.append(s.timestep))
        sim.run()
        return times

    def test_distinct_seeds_give_distinct_streams(self):
        self.assertNotEqual(self._arrivals(0), self._arrivals(1))

    def test_default_seed_is_zero_and_deterministic(self):
        # Documented: rng_seed=None falls back to 0 — two unseeded workloads
        # emit identical streams, which is why concurrent workloads must pass
        # explicit distinct seeds (see experiments/multichain_retry_storm.py).
        self.assertEqual(self._arrivals(None), self._arrivals(0))


class TestLoadSpikeValidation(unittest.TestCase):
    """A non-finite multiplier used to survive construction and then delete the
    spike it configured.

    `nan < 0` is False, so NaN passed the sign check; `rps_at` then computed
    `base * nan = nan` and `max(0.0, nan)` returned 0.0, so the window produced
    ZERO arrivals while the spike was still exported and shaded on every plot —
    verbatim the failure the negative-multiplier message exists to prevent.
    `inf` went the other way: infinite exposure put every arrival on `drive`'s
    1 ns interarrival clamp.
    """

    def test_non_finite_multiplier_is_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(rps_multiplier=bad), self.assertRaises(ValueError):
                LoadSpike(TimeInterval(s_to_ns(1), s_to_ns(2)), bad)

    def test_finite_multiplier_still_scales_the_rate(self):
        wl = Workload(
            base_rps=100.0,
            duration_s=3,
            load_spikes=[LoadSpike(TimeInterval(s_to_ns(1), s_to_ns(2)), 2.0)],
        )
        self.assertAlmostEqual(wl.rps_at(s_to_ns(1.5)), 200.0)
        self.assertAlmostEqual(wl.rps_at(s_to_ns(2.5)), 100.0)


if __name__ == "__main__":
    unittest.main()
