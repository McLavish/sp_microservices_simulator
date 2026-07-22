import unittest

from core import s_to_ns
from simulator import Simulator


class TestSimulatorTimeValidation(unittest.TestCase):
    def test_scheduling_in_the_past_raises(self):
        sim = Simulator(seed=1)
        sim.schedule(s_to_ns(5), lambda: sim.schedule(s_to_ns(1), lambda: None))
        with self.assertRaises(ValueError):
            sim.run()

    def test_run_until_the_past_raises(self):
        sim = Simulator(seed=1)
        sim.run(until=s_to_ns(10))
        with self.assertRaises(ValueError):
            sim.run(until=s_to_ns(3))

    def test_scheduling_at_now_is_allowed(self):
        sim = Simulator(seed=1)
        fired = []
        sim.schedule(s_to_ns(5), lambda: sim.schedule(sim.timestep, lambda: fired.append(1)))
        sim.run()
        self.assertEqual(fired, [1])


if __name__ == "__main__":
    unittest.main()
