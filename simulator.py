import heapq
import random
from dataclasses import dataclass, field
from typing import Callable, Optional
from core import TimePoint


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
        self._seq += 1
        ev = Event(at=int(at), seq=self._seq, action=fn)
        heapq.heappush(self._eventheap, ev)
        return ev

    def run(self, until: Optional[TimePoint] = None):
        heap = self._eventheap
        while heap and (until is None or heap[0].at <= until):
            ev = heapq.heappop(heap)
            if ev.cancelled:
                continue
            self.timestep = ev.at
            ev.action()
        if until is not None:
            self.timestep = int(until)
