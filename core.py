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


class RootOutcome(Enum):
    """How a completed root request ended.

    Single source of truth: `Metrics.summary`, `Metrics.export_csv` and
    `MicroservicesEnv._summarize_roots` all classify the same populations, and
    they previously each re-implemented the rules. They drifted — the cancelled
    test ended up before the drop-reason chain in two of them and after it in
    the third, so the same root was booked as cancelled in the CSV and as a
    deadline drop in the RL reward.
    """

    SUCCESS = auto()
    QUEUE_FULL = auto()
    DEADLINE = auto()
    SERVER_FAILURE = auto()
    CIRCUIT_OPEN = auto()
    CANCELLED = auto()


_OUTCOME_BY_DROP_REASON: "dict[DropReason, RootOutcome]" = {
    DropReason.QUEUE_FULL: RootOutcome.QUEUE_FULL,
    DropReason.DEADLINE: RootOutcome.DEADLINE,
    DropReason.CIRCUIT_OPEN: RootOutcome.CIRCUIT_OPEN,
    DropReason.CANCELLED: RootOutcome.CANCELLED,
    DropReason.SERVER_FAILURE: RootOutcome.SERVER_FAILURE,
}


def classify_root(root: "RootRequest") -> RootOutcome:
    """Classify a completed root. Order matters and is load-bearing.

    `cancelled` is tested BEFORE the final attempt's drop reason: a root
    cancelled while waiting on backoff has no attempt of its own carrying
    CANCELLED, so it still holds the previous attempt's reason (usually
    DEADLINE). Testing the reason first books abandoned work as a failure.
    """
    if any(attempt.success for attempt in root.attempts):
        return RootOutcome.SUCCESS
    if root.cancelled:
        return RootOutcome.CANCELLED
    if not root.attempts:
        return RootOutcome.SERVER_FAILURE
    # Append order is completion order (ties on end time are ambiguous).
    return _OUTCOME_BY_DROP_REASON.get(root.attempts[-1].drop_reason, RootOutcome.SERVER_FAILURE)


def root_completed_at(root: "RootRequest") -> "TimePoint":
    """When the root finished — the instant every end-bucketed metric uses.

    Prefer the recorded `completed_at`: a root cancelled while backing off
    finishes strictly after its last attempt ended, and bucketing it at that
    attempt's end put it in the wrong window (or outside the bucket range
    entirely, where it was silently dropped from the CSV).
    """
    if root.completed_at is not None:
        return root.completed_at
    return max(attempt.interval.end for attempt in root.attempts)


def root_started_at(root: "RootRequest") -> "TimePoint":
    return min(attempt.interval.begin for attempt in root.attempts)


def root_latency(root: "RootRequest") -> "TimeDuration":
    """End-to-end latency across every attempt of the root."""
    return root_completed_at(root) - root_started_at(root)


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
    # WORKER OCCUPANCY: how long a worker of the handling service was actually
    # held. Queue-exclusive, and for a tandem chain node (the default
    # hold_worker_during_fanout=False) it covers the local stage only — the
    # downstream wait happens after the worker is handed back. With
    # hold_worker_during_fanout=True it covers the whole subtree, because the
    # worker really is held throughout. End-to-end node latency is `interval`.
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
