import random
from collections.abc import Callable
from dataclasses import dataclass, field

from core import TimeDuration, TimePoint, s_to_ns
from fault_events import FaultEventsTracker
from faults import LoadSpike
from simulator import Simulator


@dataclass
class Workload:
    base_rps: float
    duration_s: int
    load_spikes: list[LoadSpike]
    rng_seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        # rng_seed=None falls back to seed 0 for cross-experiment determinism; pass an int explicitly to vary.
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

    def _sample_interarrival(self, time: TimePoint, rps: float | None = None) -> TimeDuration:
        rps = self.rps_at(time) if rps is None else rps
        if rps <= 0.0:
            # No load right now - push a bit forward.
            # RNG is intentionally NOT consumed here; arrivals stay identical across experiments only when RPS-positive intervals match.
            return s_to_ns(0.5)
        return s_to_ns(self._rng.expovariate(rps))

    def drive(self, sim: Simulator, on_arrival: Callable[[Simulator], None]):
        end_at = s_to_ns(self.duration_s)

        def tick():
            if sim.timestep >= end_at:
                return
            rps = self.rps_at(sim.timestep)
            if rps > 0.0:
                on_arrival(sim)
            ia = max(1, self._sample_interarrival(sim.timestep, rps))
            sim.schedule(sim.timestep + ia, tick)

        # start at t=0
        sim.schedule(0, tick)
