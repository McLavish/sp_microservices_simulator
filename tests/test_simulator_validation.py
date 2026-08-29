import unittest

from msim.core import s_to_ns
from msim.simulator import Simulator


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

    def test_nested_run_does_not_rewind_time(self):
        # A nested sim.run() inside an event can advance past the outer
        # run(until=...); the outer run must not drag the clock backwards.
        sim = Simulator(seed=1)
        fired = []

        def outer_event():
            sim.schedule(s_to_ns(7), lambda: fired.append(sim.timestep))
            sim.run()  # nested drain advances to t=7

        sim.schedule(s_to_ns(1), outer_event)
        sim.run(until=s_to_ns(5))
        self.assertEqual(fired, [s_to_ns(7)])
        self.assertEqual(sim.timestep, s_to_ns(7))  # was rewound to 5

    def test_scheduling_at_now_is_allowed(self):
        sim = Simulator(seed=1)
        fired = []
        sim.schedule(s_to_ns(5), lambda: sim.schedule(sim.timestep, lambda: fired.append(1)))
        sim.run()
        self.assertEqual(fired, [1])


if __name__ == "__main__":
    unittest.main()
