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


if __name__ == "__main__":
    unittest.main()
