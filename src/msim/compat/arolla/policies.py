"""Arolla's client-side retry controls, as layers of an msim retry policy.

An Arolla client can wrap its backoff in a circuit breaker and in a retry
budget. They are not the breakers and the budget of `msim.retry_policies`:

* they gate RETRIES only -- a first attempt is always sent, so a "breaker" here
  never sheds a request the way `CountBasedCircuitBreakerPolicy` does;
* the budget is funded by first attempts that SUCCEED (and by time), where
  `RetryBudgetPolicy` is funded by every admitted request.

Each class below is the rule their code means, written against msim's policy
protocol: told of every attempt through `record_attempt`, asked through
`next_delay`, layered through `underlying_strategy`. Balances and thresholds are
exact. Where their implementation and their intent part, the intent is kept and
the difference is named in the class.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction

from msim.core import S_TO_NS, TimeDuration, TimePoint
from msim.retry_policies import DelayDecision, RetryContext, RetryPolicy

_MAX_DENOMINATOR = 1_000_000_000


def _ratio(value: float, name: str) -> Fraction:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return Fraction(value).limit_denominator(_MAX_DENOMINATOR)


@dataclass
class RetryBank:
    """The balance behind an Arolla client retry budget.

    Separate from the policy layer so that several clients can draw on ONE
    balance (`shared_budget_id`) while each keeps its own backoff: build one
    bank and hand it to every client's `FirstAttemptFundedBudget`.

    The bank starts full with `max_retries` retries. Each first attempt that
    succeeds deposits `budget_ratio` of a retry, `min_retries_per_sec` retries
    per second accrue with time, and a retry costs one; the balance never
    exceeds `max_retries`.
    """

    budget_ratio: float
    max_retries: int
    min_retries_per_sec: float = 10

    _cost: int = field(init=False, repr=False)
    _deposit: int = field(init=False, repr=False)
    _cap: int = field(init=False, repr=False)
    _tokens: int = field(init=False, repr=False)
    _rate: Fraction = field(init=False, repr=False)  # tokens per nanosecond
    _carry: Fraction = field(default=Fraction(0), init=False, repr=False)
    _last: TimePoint | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        ratio = _ratio(self.budget_ratio, "budget_ratio")
        if isinstance(self.max_retries, bool) or self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries!r}")
        if not math.isfinite(self.min_retries_per_sec) or self.min_retries_per_sec < 0:
            raise ValueError(f"min_retries_per_sec must be >= 0, got {self.min_retries_per_sec!r}")
        # Integer tokens in units of 1/denominator of a retry: exact deposits.
        self._cost = ratio.denominator
        self._deposit = ratio.numerator
        self._cap = self._cost * self.max_retries
        self._tokens = self._cap
        rate = Fraction(self.min_retries_per_sec).limit_denominator(_MAX_DENOMINATOR)
        self._rate = rate * self._cost / S_TO_NS

    @property
    def balance(self) -> float:
        """Retries in the bank, as of the last time it was brought up to date."""
        return self._tokens / self._cost

    def accrue(self, now: TimePoint) -> None:
        """Credit what `min_retries_per_sec` has earned since the last look."""
        last, self._last = self._last, now
        if last is None or now <= last or self._rate == 0:
            return
        earned = self._carry + self._rate * (now - last)
        whole = earned.numerator // earned.denominator
        if self._tokens + whole >= self._cap:
            self._tokens = self._cap
            self._carry = Fraction(0)
        else:
            self._tokens += whole
            self._carry = earned - whole

    def deposit(self) -> None:
        self._tokens = min(self._cap, self._tokens + self._deposit)

    def can_pay(self) -> bool:
        return self._tokens >= self._cost

    def pay(self) -> None:
        self._tokens -= self._cost

    def refund(self) -> None:
        self._tokens = min(self._cap, self._tokens + self._cost)


@dataclass
class FirstAttemptFundedBudget(RetryPolicy):
    """Arolla's client `retry_budget`: retries paid for by first-try successes.

    A retry is allowed while the bank holds one. Only an attempt that is the
    FIRST of its request and succeeds makes a deposit, so the allowance follows
    the traffic that works without retrying; a retry that succeeds earns nothing.

    Differences from their `RetryBudgetPolicy`, all in the direction of what it
    is meant to do:

    * a deposit is `budget_ratio` exactly (theirs was cut to whole hundredths);
    * a retry the client then does not issue (the global deadline vetoes it) is
      refunded, as msim's policy protocol requires (theirs keeps the charge);
    * a budget shared by id keeps its own `min_retries_per_sec` and each client
      its own backoff (theirs hands every sharing client the FIRST client's
      whole policy object, and builds it with the default refill rate).
    """

    bank: RetryBank
    underlying_strategy: RetryPolicy

    def record_attempt(self, context: RetryContext, success: bool) -> None:
        if success and context.attempt == 1:
            self.bank.deposit()

    def next_delay(self, context: RetryContext) -> DelayDecision:
        self.bank.accrue(context.now)
        if not self.bank.can_pay():
            return DelayDecision(False, 0)
        decision = self.underlying_strategy.next_delay(context)
        if not decision.allow:
            return DelayDecision(False, 0)
        self.bank.pay()
        return decision

    def rollback(self, context: RetryContext) -> None:
        self.bank.refund()


class _Window:
    """Outcomes of the last `duration`, with a running count of the failures."""

    __slots__ = ("duration", "failed", "outcomes")

    def __init__(self, duration: TimeDuration) -> None:
        self.duration = duration
        self.outcomes: deque[tuple[TimePoint, bool]] = deque()
        self.failed = 0

    def add(self, now: TimePoint, success: bool) -> None:
        self.outcomes.append((now, success))
        if not success:
            self.failed += 1
        self.evict(now)

    def evict(self, now: TimePoint) -> None:
        while self.outcomes and now - self.outcomes[0][0] > self.duration:
            _, ok = self.outcomes.popleft()
            if not ok:
                self.failed -= 1

    def clear(self) -> None:
        self.outcomes.clear()
        self.failed = 0

    def __len__(self) -> int:
        return len(self.outcomes)

    def failing(self, threshold: Fraction, quorum: int) -> bool:
        """At least `quorum` outcomes, of which `threshold` or more failed."""
        total = len(self.outcomes)
        return (
            total >= quorum and self.failed * threshold.denominator >= threshold.numerator * total
        )


def _check_window(failure_threshold: float, window: TimeDuration, min_window_size: int) -> Fraction:
    if window <= 0:
        raise ValueError(f"window must be > 0 ns, got {window!r}")
    if isinstance(min_window_size, bool) or min_window_size < 1:
        raise ValueError(f"min_window_size must be >= 1, got {min_window_size!r}")
    return _ratio(failure_threshold, "failure_threshold")


@dataclass
class WindowRetryGate(RetryPolicy):
    """Arolla's client `retry_circuit_breaker`: no retries while calls keep failing.

    The gate holds the outcomes of this client's attempts over the last
    `window`. A retry is refused when the window holds at least
    `min_window_size` outcomes and `failure_threshold` or more of them failed.
    There is no state beyond the window: the gate reopens as failures age out of
    it or successes dilute them. Requests themselves are never held back.
    """

    underlying_strategy: RetryPolicy
    failure_threshold: float
    window: TimeDuration
    min_window_size: int

    _threshold: Fraction = field(init=False, repr=False)
    _outcomes: _Window = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._threshold = _check_window(self.failure_threshold, self.window, self.min_window_size)
        self._outcomes = _Window(self.window)

    def record_attempt(self, context: RetryContext, success: bool) -> None:
        self._outcomes.add(context.now, success)

    def next_delay(self, context: RetryContext) -> DelayDecision:
        self._outcomes.evict(context.now)
        if self._outcomes.failing(self._threshold, self.min_window_size):
            return DelayDecision(False, 0)
        return self.underlying_strategy.next_delay(context)


class BreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ThreeStateRetryBreaker(RetryPolicy):
    """Arolla's client `time_based` breaker: closed, open, half-open, on retries only.

    Like `WindowRetryGate` it watches this client's attempts over a window and
    refuses retries, never requests. Unlike it, it remembers:

    * CLOSED: retries pass. A retry asked for while the window holds at least
      `min_window_size` outcomes with `failure_threshold` or more failed opens
      the breaker instead.
    * OPEN: no retries for `open_wait`. The first retry asked for after that
      puts the breaker HALF_OPEN and passes.
    * HALF_OPEN: retries pass while fewer than `min_window_size` outcomes have
      been seen since. Then the next retry asked for decides: open again if the
      failures are at the threshold, closed otherwise.

    The window is emptied at every change of state, and the state only moves
    when a retry is asked for (a client whose calls all succeed asks for none).
    """

    underlying_strategy: RetryPolicy
    failure_threshold: float
    window: TimeDuration
    min_window_size: int
    open_wait: TimeDuration

    state: BreakerState = field(default=BreakerState.CLOSED, init=False)
    _threshold: Fraction = field(init=False, repr=False)
    _outcomes: _Window = field(init=False, repr=False)
    _opened_at: TimePoint = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._threshold = _check_window(self.failure_threshold, self.window, self.min_window_size)
        if self.open_wait < 0:
            raise ValueError(f"open_wait must be >= 0 ns, got {self.open_wait!r}")
        self._outcomes = _Window(self.window)

    def _become(self, state: BreakerState, now: TimePoint) -> None:
        self.state = state
        if state is BreakerState.OPEN:
            self._opened_at = now
        self._outcomes.clear()

    def record_attempt(self, context: RetryContext, success: bool) -> None:
        self._outcomes.add(context.now, success)

    def next_delay(self, context: RetryContext) -> DelayDecision:
        now = context.now
        failing = self._outcomes.failing(self._threshold, self.min_window_size)
        if self.state is BreakerState.CLOSED:
            if failing:
                self._become(BreakerState.OPEN, now)
                return DelayDecision(False, 0)
        elif self.state is BreakerState.OPEN:
            if now - self._opened_at < self.open_wait:
                return DelayDecision(False, 0)
            self._become(BreakerState.HALF_OPEN, now)
        elif len(self._outcomes) >= self.min_window_size:
            if failing:
                self._become(BreakerState.OPEN, now)
                return DelayDecision(False, 0)
            self._become(BreakerState.CLOSED, now)
        return self.underlying_strategy.next_delay(context)
