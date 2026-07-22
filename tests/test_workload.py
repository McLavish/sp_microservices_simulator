import unittest

from core import s_to_ns
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


if __name__ == "__main__":
    unittest.main()
