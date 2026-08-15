import math
from dataclasses import dataclass

from core import TimeDuration, TimeInterval, TimePoint


def _require_finite(name: str, value: float) -> None:
    """Reject NaN/inf before the range checks, which silently accept both.

    `nan < 0` and `nan < 1` are False, so a non-finite value passes every
    ordering guard below and only shows up as corrupted physics: NaN collapses
    to zero through `max()` (a blacked-out load spike, a latency injection that
    makes the service *faster*), inf either saturates the 1 ns interarrival
    clamp or overflows `Simulator.schedule`'s `int(at)` mid-run. In every case
    the fault is still exported and shaded on the plots. Matches the
    `math.isfinite` guards `Workload.__post_init__` applies to base_rps and
    duration_s.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


@dataclass
class LoadSpike:
    duration: TimeInterval
    rps_multiplier: float

    def __post_init__(self):
        _require_finite("rps_multiplier", self.rps_multiplier)
        if self.rps_multiplier < 0:
            raise ValueError(
                f"rps_multiplier must be >= 0, got {self.rps_multiplier}; a negative "
                "multiplier clamps the arrival rate to zero, producing a traffic "
                "blackout that is still exported and plotted as a load spike"
            )


@dataclass
class LatencyInjection:
    duration: TimeInterval
    add_latency: TimeDuration = 0  # additive latency in ns
    multiplier: int = 1  # multiplicative inflation

    def __post_init__(self):
        _require_finite("latency multiplier", self.multiplier)
        _require_finite("add_latency", self.add_latency)
        if self.multiplier < 1:
            raise ValueError(f"latency multiplier must be >= 1, got {self.multiplier}")
        if self.add_latency < 0:
            raise ValueError(f"add_latency must be >= 0 ns, got {self.add_latency}")

    def active(self, t: TimePoint) -> bool:
        return self.duration.contains(t)


@dataclass
class PartialFailure:
    duration: TimeInterval
    p_fail: float

    def __post_init__(self):
        if not 0 <= self.p_fail <= 1:
            raise ValueError(
                f"p_fail must be a fraction in [0, 1], got {self.p_fail}"
                + (" — did you pass a percentage?" if self.p_fail > 1 else "")
            )

    def active(self, t: TimePoint) -> bool:
        return self.duration.contains(t)
