from dataclasses import dataclass, field
from typing import List, Optional

from enum import Enum, auto

TimePoint = int  # nanoseconds
TimeDuration = int  # nanoseconds

S_TO_NS: TimeDuration = 1_000_000_000
MS_TO_NS: TimeDuration = 1_000_000


def s_to_ns(s: float) -> TimeDuration:
    return int(round(s * S_TO_NS))


def ms_to_ns(ms: float) -> TimeDuration:
    return int(round(ms * MS_TO_NS))


def ns_to_ms(ns: TimeDuration) -> float:
    return ns / MS_TO_NS


def ns_to_s(ns: TimeDuration) -> float:
    return ns / S_TO_NS


# Half-open interval [begin, end)
@dataclass
class TimeInterval:
    begin: TimePoint
    end: TimePoint

    def duration(self) -> TimeDuration:
        return self.end - self.begin

    def contains(self, timepoint: TimePoint) -> bool:
        return self.begin <= timepoint < self.end


class DropReason(Enum):
    NONE = 0
    QUEUE_FULL = auto()
    DEADLINE = auto()
    SERVER_FAILURE = auto()
    CIRCUIT_OPEN = auto()  # unused for now


@dataclass
class Request:
    service: str  # name of the service that handled the request
    interval: TimeInterval  # indicates the lifespan of the request
    deadline: Optional[TimePoint] = None  # maybe refactor into a policy
    valid: bool = True  # unused for now. TODO: add invalid requests
    success: bool = False
    drop_reason: DropReason = DropReason.NONE
    # Captured at attempt end by the service: current queue length
    queue_size_at_end: Optional[int] = None


@dataclass
class RootRequest:
    """Represents a client request that may consist of multiple attempts."""

    attempts: List[Request] = field(default_factory=list)
    global_deadline: Optional[TimePoint] = None
    done: bool = False  # indicated if the request was handled

    def add_attempt(self, attempt: Request):
        self.attempts.append(attempt)

    def attempt_count(self) -> int:
        return len(self.attempts)


@dataclass
class Summary:
    total: int
    succeeded: int
    dropped_queue: int
    dropped_deadline: int
    dropped_server_failure: int
    p50: float
    p90: float
    p95: float
    p99: float
    p99_9: float
    p99_99: float
    mean: float
    max: float
    retries_per_root: float
    attempts_total: int
