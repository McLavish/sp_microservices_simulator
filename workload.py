from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Optional
import random

from simulator.core import TimePoint, TimeDuration, s_to_ns
from simulator.faults import LoadSpike
from simulator.simulator import Simulator
from simulator.fault_events import FaultEventsTracker


@dataclass
class Workload:
    base_rps: float
    duration_s: int
    load_spikes: List[LoadSpike]
    rng_seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.rng_seed if self.rng_seed is not None else 0)

    def register_fault_events(self, tracker: FaultEventsTracker):
        """Register all load spike events with the tracker"""
        for load_spike in self.load_spikes:
            tracker.add_load_spike(load_spike.duration, load_spike.rps_multiplier)

    def rps_at(self, time: TimePoint) -> float:
        rps = self.base_rps
        for spike in self.load_spikes:
            if spike.duration.contains(time):
                rps *= spike.rps_multiplier
        return max(0.0, rps)

    def _sample_interarrival(self, time: TimePoint) -> TimeDuration:
        rps = self.rps_at(time)
        if rps <= 0.0:
            # No load right now - push a bit forward.
            return s_to_ns(0.5)
        # Use workload's own RNG so arrivals are identical across experiments
        return s_to_ns(self._rng.expovariate(rps))

    def drive(self, sim: Simulator, on_arrival: Callable[[Simulator], None]):
        end_at = s_to_ns(self.duration_s)

        def tick():
            if sim.timestep >= end_at:
                return
            on_arrival(sim)
            ia = max(1, self._sample_interarrival(sim.timestep))
            sim.schedule(sim.timestep + ia, tick)

        # start at t=0
        sim.schedule(0, tick)
