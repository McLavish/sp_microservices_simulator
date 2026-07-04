import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

from core import TimeDuration, TimePoint


@dataclass
class RetryContext:
    attempt: int  # attempt starts at 1 (the original request is attempt 1)
    now: TimePoint  # current simulation time; required (rate limiters need it)
    # last_delay: TimeDuration
    # total_delay: TimeDuration
    # last_result: Optional[bool] = None


class DelayDecision(NamedTuple):
    allow: bool
    delay: TimeDuration


class RetryPolicy:
    def allow_request(self, now: TimePoint) -> bool:
        return True

    def next_delay(self, context: RetryContext) -> DelayDecision:
        raise NotImplementedError


@dataclass
class FixedBackoffRetryPolicy(RetryPolicy):
    max_attempts: int  # total attempts including the initial try
    delay: TimeDuration

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if context.attempt < self.max_attempts:
            return DelayDecision(True, self.delay)
        return DelayDecision(False, 0)


@dataclass
class NoRetryPolicy(RetryPolicy):
    def next_delay(self, context: RetryContext) -> DelayDecision:
        return DelayDecision(False, 0)


@dataclass
class ExponentialBackoffRetryPolicy(RetryPolicy):
    max_attempts: int  # total attempts including the initial try
    initial_delay: TimeDuration
    max_delay: TimeDuration

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if context.attempt < self.max_attempts:
            delay = min(self.initial_delay * (2 ** (context.attempt - 1)), self.max_delay)
            return DelayDecision(True, delay)
        return DelayDecision(False, 0)


class JitterMode(Enum):
    FULL = 0
    EQUAL = 1
    DECORRELATED = 2


@dataclass
class ExponentialBackoffWithJitterRetryPolicy(RetryPolicy):
    max_attempts: int  # total attempts including the initial try
    initial_delay: TimeDuration
    max_delay: TimeDuration
    rng: random.Random  # pass in the simulator's rng
    jitter_mode: JitterMode = JitterMode.FULL
    # _prev: TimeDuration = 0

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if context.attempt < self.max_attempts:
            exp_delay = min(self.initial_delay * (2 ** (context.attempt - 1)), self.max_delay)
            if self.jitter_mode == JitterMode.FULL:
                delay = self.rng.uniform(0, exp_delay)
            elif self.jitter_mode == JitterMode.EQUAL:
                delay = exp_delay / 2 + self.rng.uniform(0, exp_delay / 2)
            elif self.jitter_mode == JitterMode.DECORRELATED:
                raise NotImplementedError("Decorrelated jitter not implemented")
            else:
                raise ValueError("Unknown jitter mode")
            return DelayDecision(True, int(delay))
        return DelayDecision(False, 0)


class CBState(Enum):
    CLOSED = 0
    OPEN = 1
    # Trial stage after the open delay. NOTE: the transitions work and are tested
    # (open -> half-open -> closed/open), but probe admission is *not* capped:
    # allow_request admits every request while half-open instead of permitting a
    # fixed number of trial calls and rejecting the rest until they complete, the
    # way resilience4j / failsafe-go / gobreaker do. So half-open currently offers
    # no load protection on the recovering service. See allow_request below.
    HALF_OPEN = 2


@dataclass
class CountBasedCircuitBreakerPolicy(RetryPolicy):
    # e.g. if you have 5 failures out of 10 calls, the circuit opens
    failure_threshold_ratio: tuple[int, int]  # (failures, total)
    # e.g. if you have 3 successes out of 5 calls, the circuit closes
    success_threshold_ratio: tuple[int, int]  # (successes, total)
    underlying_strategy: RetryPolicy
    half_open_delay: TimeDuration

    _state: CBState = CBState.CLOSED
    _closed_window: deque[bool] = field(init=False)
    _open_window: deque[bool] = field(init=False)
    _opened_at: TimePoint | None = None

    def __post_init__(self):
        f, ftot = self.failure_threshold_ratio
        s, stot = self.success_threshold_ratio
        assert 0 <= f <= ftot and ftot > 0
        assert 0 <= s <= stot and stot > 0
        # Note: deques with maxlen will drop old entries when new ones are added
        self._closed_window = deque(maxlen=ftot)
        self._open_window = deque(maxlen=stot)

    def get_state(self) -> CBState:
        return self._state

    def add_result(self, success: bool, now: TimePoint) -> None:
        if self._state == CBState.CLOSED:
            self._closed_window.append(success)
            if len(self._closed_window) == self._closed_window.maxlen:
                failures = sum(1 for ok in self._closed_window if not ok)
                if failures >= self.failure_threshold_ratio[0]:
                    self._state = CBState.OPEN
                    self._open_window.clear()
                    self._opened_at = now
        elif self._state == CBState.OPEN:
            return
        elif self._state == CBState.HALF_OPEN:
            self._open_window.append(success)
            if len(self._open_window) == self._open_window.maxlen:
                successes = sum(1 for ok in self._open_window if ok)
                if successes >= self.success_threshold_ratio[0]:
                    self._state = CBState.CLOSED
                    self._closed_window.clear()
                    self._opened_at = None
                else:
                    self._state = CBState.OPEN
                    self._open_window.clear()
                    self._opened_at = now

    def allow_request(self, now: TimePoint) -> bool:
        # NOTE: half-open admits *every* request (no probe cap) — see CBState.HALF_OPEN.
        if self._state != CBState.OPEN:
            return True
        if (
            self._opened_at is None
            or self.half_open_delay == 0
            or now - self._opened_at >= self.half_open_delay
        ):
            self._state = CBState.HALF_OPEN
            self._open_window.clear()
            return True
        return False

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if self._state != CBState.CLOSED:
            return DelayDecision(False, 0)
        return self.underlying_strategy.next_delay(context)


@dataclass
class TimeBasedCircuitBreakerPolicy(RetryPolicy):
    failure_threshold_rate: float  # percentage of failures to open the circuit
    success_threshold_rate: float  # percentage of successes to close the circuit
    min_requests: int  # Minimum requests in the window to evaluate
    window_duration: TimeDuration
    underlying_strategy: RetryPolicy
    half_open_delay: TimeDuration

    _window: deque[tuple[TimePoint, bool]] = field(default_factory=deque)  # (timestamp, success)
    _state: CBState = CBState.CLOSED
    _opened_at: TimePoint | None = None

    def _evict_old(self, now: TimePoint) -> None:
        while self._window and (now - self._window[0][0]) > self.window_duration:
            self._window.popleft()

    def get_state(self) -> CBState:
        return self._state

    def add_result(self, success: bool, now: TimePoint) -> None:
        self._evict_old(now)
        if self._state != CBState.OPEN:
            self._window.append((now, success))

        total = len(self._window)
        failures = sum(1 for _, ok in self._window if not ok)
        rate = failures / total if total > 0 else 0.0

        if self._state == CBState.CLOSED:
            if total >= self.min_requests and rate > self.failure_threshold_rate:
                self._state = CBState.OPEN
                self._opened_at = now
        elif self._state == CBState.OPEN:
            return
        elif self._state == CBState.HALF_OPEN:
            if total >= self.min_requests and rate < (1 - self.success_threshold_rate):
                self._state = CBState.CLOSED
                self._opened_at = None
                # Fresh metrics after recovery (resilience4j resets the sliding
                # window on transition to CLOSED); otherwise half-open probe
                # results would seed the closed-state failure rate.
                self._window.clear()
            elif total >= self.min_requests:
                self._state = CBState.OPEN
                self._opened_at = now
                self._window.clear()

    def allow_request(self, now: TimePoint) -> bool:
        # NOTE: half-open admits *every* request (no probe cap) — see CBState.HALF_OPEN.
        if self._state != CBState.OPEN:
            return True
        if (
            self._opened_at is None
            or self.half_open_delay == 0
            or now - self._opened_at >= self.half_open_delay
        ):
            self._state = CBState.HALF_OPEN
            self._window.clear()
            return True
        return False

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if self._state != CBState.CLOSED:
            return DelayDecision(False, 0)

        return self.underlying_strategy.next_delay(context)


# By leaky we mean the leaky bucket as a queue version.
@dataclass
class LeakyRateLimiterPolicy(RetryPolicy):
    max_requests: int  # number of max requests in the period
    max_attempts: int  # total attempts including the initial try
    period: TimeDuration

    _next_at: TimePoint | None = field(default=None, init=False)

    def add_request(self, now: TimePoint) -> TimeDuration:
        start_at = max(now, self._next_at or now)
        delay = max(0, start_at - now)

        self._next_at = round(start_at + self.period / self.max_requests)
        return delay

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if context.attempt >= self.max_attempts:
            return DelayDecision(False, 0)
        delay = self.add_request(context.now)
        return DelayDecision(True, delay)


@dataclass
class RetryBudgetPolicy(RetryPolicy):
    budget_ratio: float  # retries can be at most this ratio of successes (e.g., 0.1 for 10%)
    max_retries: int  # maximum consecutive retries before throttling
    underlying_strategy: RetryPolicy

    # Derived token values, calculated from budget_ratio and max_retries
    _max_tokens: int = field(init=False)
    _success_award: float = field(init=False)
    _retry_cost: int = field(init=False)

    _tokens: float = field(init=False)

    def __post_init__(self):
        # Derive token values from budget_ratio and max_retries
        # If budget_ratio = 0.1 (10%), then 10 successes should allow 1 retry
        # So success_award = 10, retry_cost = 100
        self._retry_cost = 100  # Base cost for a retry
        self._success_award = self._retry_cost * self.budget_ratio
        self._max_tokens = self._retry_cost * self.max_retries

        self._tokens = self._max_tokens  # start with full budget

    def add_result(self, success: bool, now: TimePoint | None = None) -> None:
        # Add tokens for a successful request
        if success:
            self._tokens = min(self._max_tokens, self._tokens + self._success_award)

    def can_retry(self) -> bool:
        return self._tokens >= self._retry_cost

    def get_token_balance(self) -> float:
        return self._tokens

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if self.can_retry():
            can_retry, delay = self.underlying_strategy.next_delay(context)
            if can_retry:
                self._tokens -= self._retry_cost
                return DelayDecision(True, delay)
            return DelayDecision(False, 0)
        return DelayDecision(False, 0)


@dataclass
class BurstyRateLimiterPolicy(RetryPolicy):
    max_requests: int  # maximum requests in the bucket
    refill_rate: float  # requests added per period
    period: TimeDuration  # time period for refill calculations
    underlying_strategy: RetryPolicy

    _tokens: float = field(init=False)
    _last_refill: TimePoint | None = field(default=None, init=False)

    def __post_init__(self):
        if self.max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if self.refill_rate < 0:
            raise ValueError("refill_rate must be >= 0")
        if self.period <= 0:
            raise ValueError("period must be > 0")
        self._tokens = float(self.max_requests)

    def _refill_tokens(self, now: TimePoint) -> None:
        if self._last_refill is None:
            self._last_refill = now
            return

        time_elapsed = max(0, now - self._last_refill)
        tokens_to_add = (time_elapsed * self.refill_rate) / self.period
        self._tokens = min(self.max_requests, self._tokens + tokens_to_add)
        self._last_refill = now

    def can_retry(self, now: TimePoint) -> bool:
        self._refill_tokens(now)
        return self._tokens >= 1.0

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if self.can_retry(context.now):
            can_retry, delay = self.underlying_strategy.next_delay(context)
            if can_retry:
                self._tokens -= 1
                return DelayDecision(True, delay)
            return DelayDecision(False, 0)
        return DelayDecision(False, 0)


@dataclass
class FixedWindowBurstyLimiterPolicy(RetryPolicy):
    max_requests: int  # maximum executions per period
    period: TimeDuration
    underlying_strategy: RetryPolicy

    _tokens: int = field(init=False)
    _window_start: TimePoint | None = field(default=None, init=False)

    def __post_init__(self):
        self._tokens = self.max_requests  # start with full tokens

    def _get_current_window_start(self, now: TimePoint) -> TimePoint:
        return (now // self.period) * self.period

    def _refill_tokens(self, now: TimePoint) -> None:
        current_window_start = self._get_current_window_start(now)

        if self._window_start is None or current_window_start > self._window_start:
            # new window, reset tokens
            self._window_start = current_window_start
            self._tokens = self.max_requests

    def can_retry(self, now: TimePoint) -> bool:
        self._refill_tokens(now)
        return self._tokens > 0

    def get_next_refill_time(self, now: TimePoint) -> TimePoint:
        current_window_start = self._get_current_window_start(now)
        return current_window_start + self.period

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if self.can_retry(context.now):
            can_retry, delay = self.underlying_strategy.next_delay(context)
            if can_retry:
                self._tokens -= 1
                return DelayDecision(True, delay)
            return DelayDecision(False, 0)
        return DelayDecision(False, 0)


# class HedgePolicy:
