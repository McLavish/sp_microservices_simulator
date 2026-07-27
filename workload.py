import random
from collections.abc import Callable
from dataclasses import dataclass, field

from core import TimePoint, ns_to_s, s_to_ns
from fault_events import FaultEventsTracker
from faults import LoadSpike
from simulator import Event, Simulator

# Domain separation between the arrival process and every other RNG in the
# simulation. Callers routinely pass the SAME integer to `Simulator(seed=...)`
# and `Workload(rng_seed=...)`; without an offset both would drive the identical
# Mersenne-Twister stream, so arrival times would be correlated with service
# times and failure rolls instead of independent of them. Python seeds MT via
# init_by_array, which scrambles thoroughly, so a fixed offset is enough to
# decorrelate (same convention as the bystander workload in rl/env.py).
_ARRIVAL_SEED_OFFSET = 100_003


@dataclass
class Workload:
    base_rps: float
    duration_s: float
    load_spikes: list[LoadSpike]
    rng_seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _next_event: Event | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        # rng_seed=None falls back to seed 0 for cross-experiment determinism; pass an int explicitly to vary.
        base = self.rng_seed if self.rng_seed is not None else 0
        self._rng = random.Random(base + _ARRIVAL_SEED_OFFSET)

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

    def _next_boundary(self, t: TimePoint, end_at: TimePoint) -> TimePoint:
        """Next spike start/end after t (the rate is constant in between)."""
        nxt = end_at
        for spike in self.load_spikes:
            for edge in (spike.duration.begin, spike.duration.end):
                if t < edge < nxt:
                    nxt = edge
        return nxt

    def _next_arrival_after(self, t: TimePoint, end_at: TimePoint) -> TimePoint | None:
        """Exact next-arrival draw for the piecewise-constant rate.

        One Exp(1) budget is consumed across rate segments (spike boundaries):
        a segment of rate r and length dt supplies r*dt of exposure, and
        memorylessness makes segment-wise consumption exact — equivalent to
        re-drawing at every boundary, without needing the max-rate bound that
        thinning would. Zero-rate segments supply no exposure and are skipped
        (consuming no RNG beyond the one draw, so arrivals stay identical
        across experiments whose RPS-positive intervals match). Returns None
        when no arrival lands before end_at.
        """
        e = self._rng.expovariate(1.0)
        while t < end_at:
            rate = self.rps_at(t)
            boundary = self._next_boundary(t, end_at)
            if rate <= 0.0:
                t = boundary
                continue
            exposure = rate * ns_to_s(boundary - t)
            if e <= exposure:
                return t + s_to_ns(e / rate)
            e -= exposure
            t = boundary
        return None

    def drive(self, sim: Simulator, on_arrival: Callable[[Simulator], None]):
        end_at = s_to_ns(self.duration_s)

        def schedule_next(from_t: TimePoint):
            nxt = self._next_arrival_after(from_t, end_at)
            if nxt is not None:
                # Keep the historical 1 ns minimum interarrival clamp.
                self._next_event = sim.schedule(max(nxt, from_t + 1), arrive)
            else:
                self._next_event = None

        def arrive():
            if sim.timestep >= end_at:
                return
            on_arrival(sim)
            schedule_next(sim.timestep)

        schedule_next(0)

    def stop(self) -> None:
        """Cancel the pending arrival: no further arrivals are generated.

        Used by the RL env at the episode horizon so the terminal drain only
        completes in-flight work instead of admitting post-horizon traffic
        when the workload's own duration outlives the episode.
        """
        if self._next_event is not None:
            self._next_event.cancelled = True
            self._next_event = None
