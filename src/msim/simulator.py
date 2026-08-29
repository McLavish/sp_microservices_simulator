import heapq
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from msim.core import TimePoint


@dataclass(order=True)
class Event:
    at: TimePoint
    seq: int
    action: Callable[[], None] = field(compare=False)
    cancelled: bool = field(default=False, compare=False)


# https://en.wikipedia.org/wiki/M/G/k_queue#Steady_state_distribution
# https://www.youtube.com/watch?v=dNj-T4j5N2M
# We have a M/G/c/K queue
class Simulator:
    def __init__(self, seed: int = 1):
        self.timestep: TimePoint = 0
        self._eventheap: list[Event] = []
        self._seq = 0
        self._rng = random.Random(seed)

    def rng(self) -> random.Random:
        return self._rng

    def schedule(self, at: TimePoint, fn: Callable[[], None]):
        at = int(at)
        if at < self.timestep:
            raise ValueError(
                f"cannot schedule an event at {at} ns: simulator time is already "
                f"{self.timestep} ns (negative delay or misconfigured deadline?)"
            )
        self._seq += 1
        ev = Event(at=at, seq=self._seq, action=fn)
        heapq.heappush(self._eventheap, ev)
        return ev

    def run(self, until: TimePoint | None = None):
        if until is not None and int(until) < self.timestep:
            raise ValueError(
                f"cannot run until {int(until)} ns: simulator time is already {self.timestep} ns"
            )
        heap = self._eventheap
        while heap and (until is None or heap[0].at <= until):
            ev = heapq.heappop(heap)
            if ev.cancelled:
                continue
            self.timestep = ev.at
            ev.action()
        if until is not None:
            # max: a nested run() inside an event may already have advanced
            # the clock past `until`; never move time backwards.
            self.timestep = max(self.timestep, int(until))
