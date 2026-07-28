import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
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

    def on_request_start(self, now: TimePoint) -> None:
        """Called once per root request, when the client admits it.

        Separate from ``add_result`` (once per *attempt*) so a policy can key
        off request arrivals rather than attempt outcomes — which is what a
        Finagle-style retry budget needs, since its ratio is retries per
        *request*. Default is a no-op.
        """
        return

    #: Whether `rollback` is safe to apply LATER than the `next_delay` it undoes.
    #: Default FALSE for any policy that overrides `rollback`: every throttling
    #: policy here holds either single-slot undo state (the leaky bucket rewinds
    #: whichever charge was most recent, not the one being undone) or a CAPPED,
    #: refilling balance (a refund arriving after an intervening refill or
    #: deposit clamps at the cap and thereby MINTS quota that was never spent).
    #: A side-effect-free policy leaves this True because its no-op rollback is
    #: trivially order-independent.
    supports_deferred_rollback: bool = True

    def rollback(self, context: RetryContext) -> None:
        """Undo the side effects of the immediately preceding ``next_delay``.

        ``next_delay`` is side-effecting for every throttling policy (a retry
        budget withdraws tokens, a rate limiter spends one, a leaky bucket
        reserves the next slot). The client may still decline to issue the retry
        it just asked for — the global deadline can veto it — and without a
        refund those never-issued retries are charged anyway.

        The client calls this only when ``next_delay`` returned ``allow=True``
        and the retry was then vetoed, and it recurses through
        ``underlying_strategy`` so every layer that charged also refunds.
        Default is a no-op, so side-effect-free policies need not implement it.
        Cf. Finagle's ``RetryBudget.tryWithdraw()``, which avoids the problem by
        only withdrawing when the caller commits to the retry.
        """
        return


@dataclass
class FixedBackoffRetryPolicy(RetryPolicy):
    max_attempts: int  # total attempts including the initial try
    delay: TimeDuration

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.delay < 0:
            raise ValueError("delay must be >= 0 ns")

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

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be >= 0 ns")

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

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be >= 0 ns")

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
    # Probes needed to decide in HALF_OPEN. Deliberately smaller than
    # min_requests: an open circuit suppresses traffic, so requiring the
    # closed-state quorum can strand the breaker in HALF_OPEN forever.
    half_open_min_requests: int = 1

    _window: deque[tuple[TimePoint, bool]] = field(default_factory=deque)  # (timestamp, success)
    _failures_in_window: int = field(default=0, init=False)
    _state: CBState = CBState.CLOSED
    _opened_at: TimePoint | None = None

    def _evict_old(self, now: TimePoint) -> None:
        while self._window and (now - self._window[0][0]) > self.window_duration:
            _, ok = self._window.popleft()
            if not ok:
                self._failures_in_window -= 1

    def _clear_window(self) -> None:
        self._window.clear()
        self._failures_in_window = 0

    def get_state(self) -> CBState:
        return self._state

    def add_result(self, success: bool, now: TimePoint) -> None:
        self._evict_old(now)
        if self._state != CBState.OPEN:
            self._window.append((now, success))
            if not success:
                self._failures_in_window += 1

        # Incremental counters instead of a full window rescan per result
        # (resilience4j's time-based sliding window keeps aggregates the same
        # way); the old sum() made every result O(window) -> O(n^2) overall.
        total = len(self._window)
        failures = self._failures_in_window
        rate = failures / total if total > 0 else 0.0

        if self._state == CBState.CLOSED:
            if total >= self.min_requests and rate > self.failure_threshold_rate:
                self._state = CBState.OPEN
                self._opened_at = now
        elif self._state == CBState.OPEN:
            return
        elif self._state == CBState.HALF_OPEN:
            # Half-open needs its own, smaller quorum. Gating on the CLOSED
            # window's min_requests means a service whose throughput has dropped
            # below min_requests / window_duration -- which is exactly what an
            # open circuit causes -- can never accumulate enough probes to
            # decide, and stays HALF_OPEN indefinitely.
            probes = max(1, min(self.min_requests, self.half_open_min_requests))
            if total >= probes and rate < (1 - self.success_threshold_rate):
                self._state = CBState.CLOSED
                self._opened_at = None
                # Fresh metrics after recovery (resilience4j resets the sliding
                # window on transition to CLOSED); otherwise half-open probe
                # results would seed the closed-state failure rate.
                self._clear_window()
            elif total >= probes:
                self._state = CBState.OPEN
                self._opened_at = now
                self._clear_window()

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
            self._clear_window()
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

    # Undo state is a single previous value, so a refund is only correct
    # immediately after the charge it undoes — see supports_deferred_rollback.
    supports_deferred_rollback: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.max_requests < 1:
            raise ValueError(f"max_requests must be >= 1, got {self.max_requests}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.period <= 0:
            raise ValueError(f"period must be > 0 ns, got {self.period}")

    _next_at: TimePoint | None = field(default=None, init=False)
    # Previous _next_at, so a vetoed retry can release the slot it reserved.
    _prev_next_at: TimePoint | None = field(default=None, init=False)
    _slot_reserved: bool = field(default=False, init=False)

    def add_request(self, now: TimePoint) -> TimeDuration:
        start_at = max(now, self._next_at or now)
        delay = max(0, start_at - now)

        self._prev_next_at = self._next_at
        self._slot_reserved = True
        self._next_at = round(start_at + self.period / self.max_requests)
        return delay

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if context.attempt >= self.max_attempts:
            return DelayDecision(False, 0)
        delay = self.add_request(context.now)
        return DelayDecision(True, delay)

    def rollback(self, context: RetryContext) -> None:
        if not self._slot_reserved:
            return
        self._next_at = self._prev_next_at
        self._prev_next_at = None
        self._slot_reserved = False


@dataclass
class RetryBudgetPolicy(RetryPolicy):
    """Finagle-style retry budget: retries capped at `budget_ratio` of REQUESTS.

    One deposit per admitted root request (`on_request_start`), one withdrawal
    per retry (`next_delay`), refunded if the client then declines to issue it
    (`rollback`) — the same deposit/tryWithdraw shape as Finagle's `RetryBudget`.

    Deliberate divergence from Finagle: deposits here never expire. Finagle
    ages them out on a TTL (default 60 s) so a long-idle client cannot bank
    unlimited retry credit; this implementation caps the balance at
    `max_retries` worth of tokens instead, which bounds the burst but not the
    staleness. Adequate for episode-length simulations; would need the TTL to
    model a long-lived client faithfully.

    Note the denominator is *requests*, not *successes*. Depositing only on
    success would enforce `retries <= ratio x successes`, which is strictly
    tighter during exactly the failure storms a budget exists to damp, and does
    not match the retries-per-root constraint the RL cost measures.
    """

    budget_ratio: float  # retries can be at most this ratio of requests (e.g., 0.1 for 10%)
    max_retries: int  # maximum consecutive retries before throttling
    underlying_strategy: RetryPolicy

    # Integer token bucket in units of 1/denominator of the (rationalized)
    # ratio: one retry costs `denominator` tokens, each request deposits
    # `numerator`. Exact arithmetic — no float drift (cf. Finagle's integer
    # RetryBudget deposit/withdraw); limit_denominator(10_000) recovers
    # intended ratios like 1/3 from their float representations.
    _retry_cost: int = field(init=False)
    _deposit: int = field(init=False)
    _max_tokens: int = field(init=False)
    _tokens: int = field(init=False)

    def __post_init__(self):
        if not 0 < self.budget_ratio <= 1:
            raise ValueError("budget_ratio must be in (0, 1]")
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        ratio = Fraction(self.budget_ratio).limit_denominator(10_000)
        self._retry_cost = ratio.denominator
        self._deposit = ratio.numerator
        self._max_tokens = self._retry_cost * self.max_retries
        self._tokens = self._max_tokens  # start with full budget

    def on_request_start(self, now: TimePoint) -> None:
        # One deposit per admitted request, regardless of how it turns out.
        self._tokens = min(self._max_tokens, self._tokens + self._deposit)

    def can_retry(self) -> bool:
        return self._tokens >= self._retry_cost

    def get_token_balance(self) -> float:
        # Backward-compatible scale: 100 tokens == one retry, as before.
        return self._tokens / self._retry_cost * 100.0

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if self.can_retry():
            can_retry, delay = self.underlying_strategy.next_delay(context)
            if can_retry:
                self._tokens -= self._retry_cost
                return DelayDecision(True, delay)
            return DelayDecision(False, 0)
        return DelayDecision(False, 0)

    # Capped balance: a refund after an intervening deposit clamps at the cap
    # and mints quota. Only the immediate (deadline-veto) refund is exact.
    supports_deferred_rollback: bool = field(default=False, init=False, repr=False)

    def rollback(self, context: RetryContext) -> None:
        # Refund a withdrawal for a retry the client then declined to issue.
        self._tokens = min(self._max_tokens, self._tokens + self._retry_cost)


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

    # Capped, continuously refilling balance: see supports_deferred_rollback.
    supports_deferred_rollback: bool = field(default=False, init=False, repr=False)

    def rollback(self, context: RetryContext) -> None:
        self._tokens = min(float(self.max_requests), self._tokens + 1.0)


@dataclass
class FixedWindowBurstyLimiterPolicy(RetryPolicy):
    max_requests: int  # maximum executions per period
    period: TimeDuration
    underlying_strategy: RetryPolicy

    _tokens: int = field(init=False)
    _window_start: TimePoint | None = field(default=None, init=False)

    def __post_init__(self):
        if self.max_requests < 1:
            raise ValueError(f"max_requests must be >= 1, got {self.max_requests}")
        if self.period <= 0:
            raise ValueError(f"period must be > 0 ns, got {self.period}")
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

    # Window resets make a late refund unsound: see supports_deferred_rollback.
    supports_deferred_rollback: bool = field(default=False, init=False, repr=False)

    def rollback(self, context: RetryContext) -> None:
        # Refunds land in the window the charge was made in: the client vetoes
        # at the same simulated instant, so no window rollover intervenes.
        self._tokens = min(self.max_requests, self._tokens + 1)


# class HedgePolicy:
