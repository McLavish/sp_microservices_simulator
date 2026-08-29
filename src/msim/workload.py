import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from msim.core import TimePoint, ns_to_s, s_to_ns
from msim.fault_events import FaultEventsTracker
from msim.faults import LoadSpike
from msim.simulator import Event, Simulator

# Domain separation between the arrival process and every other RNG in the
# simulation. Callers routinely pass the SAME integer to `Simulator(seed=...)`
# and `Workload(rng_seed=...)`; without an offset both would drive the identical
# Mersenne-Twister stream, so arrival times would be correlated with service
# times and failure rolls instead of independent of them. Python seeds MT via
# init_by_array, which scrambles thoroughly, so a fixed offset is enough to
# decorrelate.
_ARRIVAL_SEED_OFFSET = 100_003


@dataclass
class Workload:
    base_rps: float
    duration_s: float
    load_spikes: list[LoadSpike]
    rng_seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _next_event: Event | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        if not math.isfinite(self.base_rps) or self.base_rps < 0:
            raise ValueError(f"base_rps must be finite and >= 0, got {self.base_rps}")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError(f"duration_s must be finite and >= 0, got {self.duration_s}")
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
        self._stopped = False
        end_at = s_to_ns(self.duration_s)

        def schedule_next(from_t: TimePoint):
            if self._stopped:
                # stop() may be called from inside on_arrival, i.e. while the
                # current arrival event is executing. Cancelling that event is a
                # no-op (it has already been popped), so the halt has to be a
                # latch that also suppresses re-arming.
                self._next_event = None
                return
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

        Lets a caller close the arrival process early — e.g. to drain a run at a
        chosen horizon so the tail only completes in-flight work instead of
        admitting fresh traffic when the workload's own duration outlives it.
        """
        self._stopped = True
        if self._next_event is not None:
            self._next_event.cancelled = True
            self._next_event = None
