from dataclasses import dataclass

from core import TimeDuration, TimeInterval, TimePoint


@dataclass
class LoadSpike:
    duration: TimeInterval
    rps_multiplier: float

    def __post_init__(self):
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
