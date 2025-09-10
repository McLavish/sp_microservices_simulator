from dataclasses import dataclass
from core import TimeDuration, TimePoint, TimeInterval


@dataclass
class LoadSpike:
    duration: TimeInterval
    rps_multiplier: float


@dataclass
class LatencyInjection:
    duration: TimeInterval
    add_latency: TimeDuration = 0  # additive latency in ns
    multiplier: int = 1  # multiplicative inflation

    def active(self, t: TimePoint) -> bool:
        return self.duration.contains(t)


@dataclass
class PartialFailure:
    duration: TimeInterval
    p_fail: float

    def active(self, t: TimePoint) -> bool:
        return self.duration.contains(t)
