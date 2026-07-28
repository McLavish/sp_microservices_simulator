from collections.abc import Callable
from dataclasses import dataclass, field
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

    def __post_init__(self):
        if self.end < self.begin:
            raise ValueError(
                f"TimeInterval end ({self.end}) must be >= begin ({self.begin}); "
                "contains() is False everywhere on an inverted interval, so any "
                "fault built on one is silently inert while still being exported"
            )

    def duration(self) -> TimeDuration:
        return self.end - self.begin

    def contains(self, timepoint: TimePoint) -> bool:
        return self.begin <= timepoint < self.end


class DropReason(Enum):
    NONE = 0
    QUEUE_FULL = auto()
    DEADLINE = auto()
    SERVER_FAILURE = auto()
    CIRCUIT_OPEN = auto()
    # Abandoned, not failed: an upstream fan-out short-circuited and cancelled
    # this call while it was still outstanding. Only produced when a
    # ServiceNodeRuntime has propagate_cancellation=True.
    CANCELLED = auto()


@dataclass
class Request:
    service: str  # name of the service that handled the request
    interval: TimeInterval  # indicates the lifespan of the request
    deadline: TimePoint | None = None  # maybe refactor into a policy
    valid: bool = True  # unused for now. TODO: add invalid requests
    success: bool = False
    drop_reason: DropReason = DropReason.NONE
    # Captured at attempt end by the service: current queue length
    queue_size_at_end: int | None = None
    # Queue-exclusive processing duration reported by the service (for chain
    # nodes: local stage + downstream wait, still excluding the local queue).
    # None when service never began (queue-full / deadline-at-dequeue drops).
    service_time_ns: TimeDuration | None = None


@dataclass
class RootRequest:
    """Represents a client request that may consist of multiple attempts."""

    attempts: list[Request] = field(default_factory=list)
    global_deadline: TimePoint | None = None
    done: bool = False  # indicated if the request was handled
    # Optional: callback invoked exactly once when the root completes
    # Not used by metrics; safe to leave None in existing flows
    on_done: Callable[..., object] | None = None
    # Set when an upstream cancelled this root (fan-out short-circuit with
    # propagate_cancellation). Metrics classify on this flag *before* falling
    # back to the final attempt's drop_reason, so a root cancelled while
    # waiting on backoff — with no attempt of its own to mark — is still
    # counted as cancelled without fabricating a synthetic attempt.
    cancelled: bool = False
    # Installed by ClientRuntime.start_request; cancels this root's in-flight
    # attempt and any pending retry. None once the root completes.
    cancel: Callable[[], None] | None = None
    # When the root finished. Usually equals the final attempt's end, but a root
    # cancelled while backing off has no attempt carrying that instant, and
    # bucketing it at the previous attempt's end would report it in the wrong
    # window. Consumers should prefer this over `attempts[-1].interval.end`.
    completed_at: TimePoint | None = None

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
    dropped_circuit_open: int
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
    # Abandoned by an upstream cancellation. Deliberately its own bucket and
    # NOT folded into the other dropped_* counts: cancelled work did not fail,
    # and charging it to server/deadline failures would count one root cause
    # twice (once where it failed, once at every sibling it cancelled).
    # Declared last so existing positional construction keeps working.
    dropped_cancelled: int = 0
