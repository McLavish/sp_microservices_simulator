"""Admission control: which attempts a server takes.

Everything else in the package that sheds load does it for a reason of its own:
a station refuses an attempt when its queue is full, a client's circuit breaker
stops sending. Admission control is a POLICY decision, taken on the server's
side, about one attempt at the moment it arrives: let it in, or turn it away
before it costs a queue slot. The interesting attempts are retries. A server
cannot stop its callers from retrying, but it can decline the retries it is
sent, and it sees what no single caller sees: all of them at once.

Three pieces:

* an `AdmissionController` is one rule ("retries at 30 per second", "retries up
  to a tenth of the goodput", "no retries while a third of the answers fail");
* an `AdmissionGate` is a `caller.ServiceLike` decorator, like
  `transport.TransportLink`, that puts an ordered list of controllers in front
  of a target. A refused attempt is reported at once as `DropReason.REJECTED`
  and never reaches the target;
* a `ControlledRetryPolicy` applies the same controllers on the CALLER's side,
  as a layer of its retry policy: a client that holds itself to a budget.

    budget = TokenBucketBudget(rate=30, burst=10)
    gated = AdmissionGate(service, [budget])
    client = ClientRuntime(cfg=ClientConfig(..., tenant="a"), service=gated)

A gate decides by the attempt's `core.RequestContext` (is it a retry, whose is
it), so the requests that reach one must carry a context: give the client that
starts them a `ClientConfig.tenant`.

`GoodputCoupledBudget` is the retry admission rule of Arolla (ETH EASL,
https://github.com/eth-easl/Arolla); the algorithm is theirs, ported here.
"""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from msim.caller import ServiceLike
from msim.core import S_TO_NS, DropReason, RequestContext, TimeDuration, TimePoint
from msim.retry_policies import DelayDecision, RetryContext, RetryPolicy
from msim.service import AttemptControl, AttemptDoneCallback, ServiceConfig
from msim.simulator import Simulator

#: The values of `AdmissionGate.retry_scope`.
RETRY_SCOPES = ("hop", "request")


@dataclass(frozen=True)
class AdmissionRequest:
    """One attempt asking to be let in, as the controllers see it."""

    now: TimePoint
    #: Whose request it is (`RequestContext.tenant`); None when it has no tenant.
    tenant: str | None
    #: The attempt's number at the hop that issued it.
    attempt: int
    #: Whether this is a retry, in the asker's sense: for a gate, by its
    #: `retry_scope`; for a `ControlledRetryPolicy`, always (it is only asked
    #: about retries).
    is_retry: bool
    #: Attempts the asking gate has let in and not yet seen end, this one not
    #: included, and how many of those are retries. None when the asker cannot
    #: know (a retry policy sees one client, not the server's load).
    outstanding: int | None = None
    outstanding_retries: int | None = None


@dataclass(frozen=True)
class AdmissionOutcome:
    """How an attempt that was let in ended, as its target reported it."""

    now: TimePoint
    tenant: str | None
    attempt: int
    is_retry: bool
    #: Served successfully, within the deadline the target was told.
    success: bool
    drop_reason: DropReason


class AdmissionController:
    """One rule about which attempts are let in.

    A controller is asked in TWO phases, because a gate may hold several: every
    controller is asked `check` first, and only when all of them agree is each
    told to `commit`. `check` must therefore not spend anything (bringing a
    time-based balance up to date is fine: asking twice at one instant changes
    nothing). A refusal by a later controller then costs an earlier one none of
    its quota.

    `refund` undoes one `commit`, for an asker that was granted an attempt and
    did not make it (a client whose retry the global deadline then vetoed). A
    gate never needs it: what it admits, it forwards.

    `observe` is told how every admitted attempt ended (not the cancelled ones:
    abandoned work is not evidence about the server).
    """

    def check(self, request: AdmissionRequest) -> bool:
        raise NotImplementedError

    def commit(self, request: AdmissionRequest) -> None:
        return

    def refund(self, request: AdmissionRequest) -> None:
        return

    def observe(self, outcome: AdmissionOutcome) -> None:
        return


class RetryController(AdmissionController):
    """A controller that only ever refuses RETRIES.

    A first attempt passes untouched and is never charged; subclasses say what a
    retry needs (`admits_retry`), what it costs (`spend`) and how to give that
    back (`give_back`).
    """

    def check(self, request: AdmissionRequest) -> bool:
        return (not request.is_retry) or self.admits_retry(request)

    def commit(self, request: AdmissionRequest) -> None:
        if request.is_retry:
            self.spend(request)

    def refund(self, request: AdmissionRequest) -> None:
        if request.is_retry:
            self.give_back(request)

    def admits_retry(self, request: AdmissionRequest) -> bool:
        raise NotImplementedError

    def spend(self, request: AdmissionRequest) -> None:
        return

    def give_back(self, request: AdmissionRequest) -> None:
        return


# A token balance is an integer number of 1e-18 tokens, and a rate an integer
# number of 1e-9 tokens per second, so `rate * elapsed_ns` is already in balance
# units and a refill is exact however often the balance is looked at. (A float
# balance refilled in many small steps drifts, and one truncated per step never
# refills at all.) The only rounding is of the configured rate, to 1e-9 per second.
_TOKEN = 10**18
_RATE_SCALE = 10**9

#: Bound for turning a configured ratio or threshold into an exact fraction.
_MAX_DENOMINATOR = 1_000_000_000


def _rate_units(rate: float, name: str) -> int:
    if not math.isfinite(rate) or rate < 0:
        raise ValueError(f"{name} must be a finite rate >= 0 per second, got {rate!r}")
    return round(Fraction(rate) * _RATE_SCALE)


def _fraction(value: float, name: str, *, low: float = 0.0, high: float = 1.0) -> Fraction:
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low:g}, {high:g}], got {value!r}")
    return Fraction(value).limit_denominator(_MAX_DENOMINATOR)


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")


def _positive_duration(value: TimeDuration, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a duration > 0 ns, got {value!r}")


@dataclass
class TokenBucketBudget(RetryController):
    """Retries at a fixed rate, whoever sends them.

    A bucket of `burst` tokens, full at the start and refilled at `rate` tokens
    per second; a retry takes one. The server's answer to "how many retries can
    I afford": a number that does not depend on what the callers do.
    """

    rate: float  # retries per second
    burst: int  # bucket size, and the opening balance

    _rate: int = field(init=False, repr=False)
    _cap: int = field(init=False, repr=False)
    _tokens: int = field(init=False, repr=False)
    _last: TimePoint | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rate = _rate_units(self.rate, "rate")
        _positive_int(self.burst, "burst")
        self._cap = self.burst * _TOKEN
        self._tokens = self._cap

    def _refill(self, now: TimePoint) -> None:
        last, self._last = self._last, now
        if last is not None and now > last:
            self._tokens = min(self._cap, self._tokens + self._rate * (now - last))

    @property
    def balance(self) -> float:
        """Tokens in the bucket as of the last time it was looked at."""
        return self._tokens / _TOKEN

    def admits_retry(self, request: AdmissionRequest) -> bool:
        self._refill(request.now)
        return self._tokens >= _TOKEN

    def spend(self, request: AdmissionRequest) -> None:
        self._tokens -= _TOKEN

    def give_back(self, request: AdmissionRequest) -> None:
        self._tokens = min(self._cap, self._tokens + _TOKEN)


@dataclass
class AimdBudget(RetryController):
    """A token bucket whose refill rate follows the server's health.

    Additive increase, multiplicative decrease on the retry RATE: at the end of
    each `window`, if more than `failure_threshold` of the attempts that ended
    in it failed, the rate is multiplied by `decrease_factor`; otherwise
    `additive_step` retries per second are added. The rate stays within
    `[min_rate, max_rate]`. A window with no attempt changes nothing.

    A window ends at the first look at the controller (an outcome or a retry
    asking) at least `window` after it began, and the next one begins there.
    The bucket is refilled at the rate in force when it is looked at.
    """

    min_rate: float  # retries per second
    max_rate: float
    initial_rate: float
    burst: int = 10
    additive_step: float = 1.0  # retries per second, per healthy window
    decrease_factor: float = 0.5
    window: TimeDuration = S_TO_NS
    failure_threshold: float = 0.1

    _min: int = field(init=False, repr=False)
    _max: int = field(init=False, repr=False)
    _rate: int = field(init=False, repr=False)
    _step: int = field(init=False, repr=False)
    _factor: Fraction = field(init=False, repr=False)
    _threshold: Fraction = field(init=False, repr=False)
    _cap: int = field(init=False, repr=False)
    _tokens: int = field(init=False, repr=False)
    _last: TimePoint | None = field(default=None, init=False, repr=False)
    _window_start: TimePoint | None = field(default=None, init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)
    _failed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._min = _rate_units(self.min_rate, "min_rate")
        self._max = _rate_units(self.max_rate, "max_rate")
        self._rate = _rate_units(self.initial_rate, "initial_rate")
        self._step = _rate_units(self.additive_step, "additive_step")
        if not self._min <= self._max:
            raise ValueError(
                f"min_rate ({self.min_rate}) must not exceed max_rate ({self.max_rate})"
            )
        if not self._min <= self._rate <= self._max:
            raise ValueError(
                f"initial_rate ({self.initial_rate}) must be within "
                f"[min_rate, max_rate] = [{self.min_rate}, {self.max_rate}]"
            )
        if not math.isfinite(self.decrease_factor) or not 0 < self.decrease_factor < 1:
            raise ValueError(f"decrease_factor must be in (0, 1), got {self.decrease_factor!r}")
        self._factor = Fraction(self.decrease_factor).limit_denominator(_MAX_DENOMINATOR)
        self._threshold = _fraction(self.failure_threshold, "failure_threshold")
        _positive_int(self.burst, "burst")
        _positive_duration(self.window, "window")
        self._cap = self.burst * _TOKEN
        self._tokens = self._cap

    @property
    def current_rate(self) -> float:
        """Retries per second the bucket is refilled at right now."""
        return self._rate / _RATE_SCALE

    @property
    def balance(self) -> float:
        return self._tokens / _TOKEN

    def _roll(self, now: TimePoint) -> None:
        if self._window_start is None:
            self._window_start = now
            return
        if now - self._window_start < self.window:
            return
        if self._seen > 0:
            # failed / seen > threshold, without the division.
            if self._failed * self._threshold.denominator > self._threshold.numerator * self._seen:
                self._rate = max(self._min, round(self._rate * self._factor))
            else:
                self._rate = min(self._max, self._rate + self._step)
        self._window_start = now
        self._seen = 0
        self._failed = 0

    def _refill(self, now: TimePoint) -> None:
        self._roll(now)
        last, self._last = self._last, now
        if last is not None and now > last:
            self._tokens = min(self._cap, self._tokens + self._rate * (now - last))

    def observe(self, outcome: AdmissionOutcome) -> None:
        self._roll(outcome.now)
        self._seen += 1
        if not outcome.success:
            self._failed += 1

    def admits_retry(self, request: AdmissionRequest) -> bool:
        self._refill(request.now)
        return self._tokens >= _TOKEN

    def spend(self, request: AdmissionRequest) -> None:
        self._tokens -= _TOKEN

    def give_back(self, request: AdmissionRequest) -> None:
        self._tokens = min(self._cap, self._tokens + _TOKEN)


@dataclass
class SuccessFundedBudget(RetryController):
    """Retries paid for by successes.

    Every attempt that ends in success deposits `ratio` of a retry, up to
    `max_retries` retries in the bank; a retry costs one. The budget starts
    full. While the server answers, a tenth (say) of its traffic may be retries;
    when it stops answering the deposits stop and so, soon, do the retries.

    Not `retry_policies.RetryBudgetPolicy`, which is funded by REQUESTS and so
    keeps paying out during an outage.
    """

    ratio: float  # retries earned per success
    max_retries: int  # cap of the balance, and the opening balance

    _cost: int = field(init=False, repr=False)
    _deposit: int = field(init=False, repr=False)
    _cap: int = field(init=False, repr=False)
    _tokens: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ratio = _fraction(self.ratio, "ratio")
        _positive_int(self.max_retries, "max_retries")
        # Integer tokens in units of 1/denominator of a retry: exact deposits.
        self._cost = ratio.denominator
        self._deposit = ratio.numerator
        self._cap = self._cost * self.max_retries
        self._tokens = self._cap

    @property
    def balance(self) -> float:
        """Retries in the bank."""
        return self._tokens / self._cost

    def observe(self, outcome: AdmissionOutcome) -> None:
        if outcome.success:
            self._tokens = min(self._cap, self._tokens + self._deposit)

    def admits_retry(self, request: AdmissionRequest) -> bool:
        return self._tokens >= self._cost

    def spend(self, request: AdmissionRequest) -> None:
        self._tokens -= self._cost

    def give_back(self, request: AdmissionRequest) -> None:
        self._tokens = min(self._cap, self._tokens + self._cost)


@dataclass
class _QuotaState:
    #: When each first attempt, and each retry, of the last window was let in.
    firsts: deque[TimePoint] = field(default_factory=deque)
    retries: deque[TimePoint] = field(default_factory=deque)


@dataclass
class TenantRetryQuota(RetryController):
    """Retries as a share of the first attempts, tenant by tenant.

    A retry of a tenant is let in while its retries let in during the last
    `window` are fewer than `share x n`, where `n` is the number of ITS first
    attempts let in during that window. Both are counted and the share is an
    exact fraction, so there is no balance to drift and none to spend down: a
    share of 0 lets no retry in, from the next one on, and a share of 3 never
    refuses a caller that tries four times. The count is compared with the
    product as it is, fractions included, the way `GoodputCoupledBudget`
    compares it with its allowance: 3.5 lets a fourth retry in.

    Each tenant has the share `set_share` gave it, or `share`. A share can be
    changed while a run is in progress: it is read at each retry that asks, and
    changing it draws nothing and schedules nothing.

    `floor` (default 0) is a number of retries per tenant and window that are
    let in whatever the share: a probe. With a share of 0 no retry is let in,
    and whoever sets the shares from what the retries do (`rl.limiter`) can no
    longer see whether a retry would succeed again; a floor of 1 shows it once
    a window, at the price of one retry.

    For a gate, which shows it the first attempts too. Behind a
    `ControlledRetryPolicy`, which is only asked about retries, it would count
    no first attempt and refuse every retry.
    """

    share: float = 0.1  # retries per first attempt, for a tenant with no share of its own
    window: TimeDuration = S_TO_NS
    floor: int = 0  # retries per tenant and window let in whatever the share

    _default: Fraction = field(init=False, repr=False)
    _shares: dict[str | None, Fraction] = field(default_factory=dict, init=False, repr=False)
    _tenants: dict[str | None, _QuotaState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._default = self._exact(self.share)
        _positive_duration(self.window, "window")
        if isinstance(self.floor, bool) or not isinstance(self.floor, int) or self.floor < 0:
            raise ValueError(f"floor is a whole number of retries >= 0, got {self.floor!r}")

    @staticmethod
    def _exact(share: float) -> Fraction:
        if isinstance(share, bool) or not math.isfinite(share) or share < 0:
            raise ValueError(f"a share must be a finite number >= 0, got {share!r}")
        return Fraction(share).limit_denominator(_MAX_DENOMINATOR)

    def set_share(self, tenant: str | None, share: float) -> None:
        """Give `tenant` a share of its own, from its next retry on."""
        self._shares[tenant] = self._exact(share)  # raises before anything is touched

    def share_of(self, tenant: str | None) -> float:
        """Retries per first attempt `tenant` may have."""
        return float(self._shares.get(tenant, self._default))

    def _state(self, tenant: str | None, now: TimePoint) -> _QuotaState:
        """The tenant's counts, with what has left the window dropped."""
        state = self._tenants.get(tenant)
        if state is None:
            state = self._tenants[tenant] = _QuotaState()
        cutoff = now - self.window
        for times in (state.firsts, state.retries):
            while times and times[0] < cutoff:
                times.popleft()
        return state

    def in_window(self, tenant: str | None, now: TimePoint) -> tuple[int, int]:
        """`(first attempts, retries)` of `tenant` let in during the window ending now."""
        state = self._state(tenant, now)
        return len(state.firsts), len(state.retries)

    def commit(self, request: AdmissionRequest) -> None:
        state = self._state(request.tenant, request.now)
        (state.retries if request.is_retry else state.firsts).append(request.now)

    def admits_retry(self, request: AdmissionRequest) -> bool:
        state = self._state(request.tenant, request.now)
        share = self._shares.get(request.tenant, self._default)
        if len(state.retries) < self.floor:
            return True  # the probe
        # len(retries) < share x len(firsts), in whole numbers.
        return len(state.retries) * share.denominator < share.numerator * len(state.firsts)

    def give_back(self, request: AdmissionRequest) -> None:
        retries = self._state(request.tenant, request.now).retries
        if retries:
            retries.pop()


#: What a retry deep in a chain costs in `MixedRefillBudget`, by its attempt number.
_DEEP_COSTS = {
    "linear": lambda attempt: attempt,
    "triangular": lambda attempt: attempt * (attempt - 1) // 2,
    "quadratic": lambda attempt: attempt * attempt,
}


@dataclass
class MixedRefillBudget(RetryController):
    """One bucket of retries, refilled by successes AND by time (Arolla's prototype).

    The retry admission of Arolla's Kubernetes prototype (ETH EASL,
    https://github.com/eth-easl/Arolla, the Envoy filter `prototype/arolla-filter`,
    "Algorithm 1" of their paper), ported: a bucket of `capacity` tokens, full at
    the start; every attempt that ends in success deposits `r` of a token, and
    `t` tokens arrive per second whatever happens; a retry takes one. One bucket
    for everybody: no tenant has an account of its own. (`GoodputCoupledBudget`
    is the rule of their SIMULATOR, which is another one: per tenant, a count
    against a smoothed goodput.)

    `deep_from` and `deep_cost` are their second filter (`arolla-filter-fairness`):
    a retry whose attempt number is above `deep_from` costs more than one token
    (`"linear"`: its attempt number; `"triangular"`: attempt x (attempt - 1) / 2;
    `"quadratic"`: attempt squared), so that a caller deep in a chain of retries
    pays for draining the bucket it shares. `None` (default): every retry costs one.

    The balance is exact (integer token units); `r` and `capacity` are rounded
    to 1e-18 of a token and `t` to 1e-9 of a token per second.
    """

    r: float = 0.1  # tokens per success
    capacity: float = 20.0  # bucket size, and the opening balance
    t: float = 1.0  # tokens per second
    deep_from: int | None = None
    deep_cost: str = "linear"

    _deposit: int = field(init=False, repr=False)
    _rate: int = field(init=False, repr=False)
    _cap: int = field(init=False, repr=False)
    _tokens: int = field(init=False, repr=False)
    _last: TimePoint | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("r", "capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        if self.capacity < 1:
            raise ValueError(f"capacity must hold at least one retry, got {self.capacity!r}")
        if self.deep_from is not None:
            _positive_int(self.deep_from, "deep_from")
        if self.deep_cost not in _DEEP_COSTS:
            raise ValueError(
                f"deep_cost must be one of {', '.join(_DEEP_COSTS)}; got {self.deep_cost!r}"
            )
        self._rate = _rate_units(self.t, "t")
        self._deposit = round(Fraction(self.r).limit_denominator(_MAX_DENOMINATOR) * _TOKEN)
        self._cap = round(Fraction(self.capacity).limit_denominator(_MAX_DENOMINATOR) * _TOKEN)
        self._tokens = self._cap

    def _refill(self, now: TimePoint) -> None:
        last, self._last = self._last, now
        if last is not None and now > last:
            self._tokens = min(self._cap, self._tokens + self._rate * (now - last))

    @property
    def balance(self) -> float:
        """Tokens in the bucket as of the last time it was looked at."""
        return self._tokens / _TOKEN

    def _cost(self, request: AdmissionRequest) -> int:
        if self.deep_from is None or request.attempt <= self.deep_from:
            return _TOKEN
        return _DEEP_COSTS[self.deep_cost](request.attempt) * _TOKEN

    def observe(self, outcome: AdmissionOutcome) -> None:
        if outcome.success:
            self._refill(outcome.now)
            self._tokens = min(self._cap, self._tokens + self._deposit)

    def admits_retry(self, request: AdmissionRequest) -> bool:
        self._refill(request.now)
        return self._tokens >= self._cost(request)

    def spend(self, request: AdmissionRequest) -> None:
        self._tokens -= self._cost(request)

    def give_back(self, request: AdmissionRequest) -> None:
        self._tokens = min(self._cap, self._tokens + self._cost(request))


@dataclass
class FailureRateGate(RetryController):
    """No retries while the server is failing: a breaker on retries alone.

    The gate watches the outcomes of the last `window`. Closed, it admits every
    retry, and opens when at least `min_requests` attempts ended in the window
    and MORE than `failure_threshold` of them failed. Open, it refuses every
    retry, and closes again once `open_delay` has passed and the window holds at
    least `min_requests` outcomes of which `success_threshold` or more succeeded
    (default: `1 - failure_threshold`). First attempts are never refused, so the
    window keeps filling while the gate is open; there is no half-open stage.

    Unlike the circuit breakers in `retry_policies`, this one sits at the
    server and sees every caller's outcomes.
    """

    failure_threshold: float
    min_requests: int
    window: TimeDuration
    open_delay: TimeDuration
    success_threshold: float | None = None

    _fail: Fraction = field(init=False, repr=False)
    _ok: Fraction = field(init=False, repr=False)
    _outcomes: deque[tuple[TimePoint, bool]] = field(default_factory=deque, init=False, repr=False)
    _failures: int = field(default=0, init=False, repr=False)
    _opened_at: TimePoint | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._fail = _fraction(self.failure_threshold, "failure_threshold")
        if self.success_threshold is None:
            self._ok = 1 - self._fail
        else:
            self._ok = _fraction(self.success_threshold, "success_threshold")
        _positive_int(self.min_requests, "min_requests")
        _positive_duration(self.window, "window")
        if isinstance(self.open_delay, bool) or self.open_delay < 0:
            raise ValueError(f"open_delay must be >= 0 ns, got {self.open_delay!r}")

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None

    def observe(self, outcome: AdmissionOutcome) -> None:
        now = outcome.now
        while self._outcomes and now - self._outcomes[0][0] > self.window:
            _, ok = self._outcomes.popleft()
            if not ok:
                self._failures -= 1
        self._outcomes.append((now, outcome.success))
        if not outcome.success:
            self._failures += 1

        total = len(self._outcomes)
        if total < self.min_requests:
            return
        if self._opened_at is None:
            # failures / total > failure_threshold
            if self._failures * self._fail.denominator > self._fail.numerator * total:
                self._opened_at = now
        elif now - self._opened_at >= self.open_delay:
            # successes / total >= success_threshold
            successes = total - self._failures
            if successes * self._ok.denominator >= self._ok.numerator * total:
                self._opened_at = None

    def admits_retry(self, request: AdmissionRequest) -> bool:
        return self._opened_at is None


@dataclass
class _GoodputState:
    """What `GoodputCoupledBudget` keeps for one tenant."""

    goodput_rate: float = 0.0  # successes per second, smoothed
    request_rate: float = 0.0  # attempts per second, smoothed
    goodput: int = 0  # successes in the current window
    requests: int = 0  # attempts ended in the current window
    window_start: TimePoint | None = None
    retries: deque[TimePoint] = field(default_factory=deque)  # when each was admitted
    retry_success: float = 0.5  # smoothed success rate of the tenant's retries
    initialized: bool = False


#: Smoothing of the request rate behind `max_retry_ratio` (new-sample weight).
_REQUEST_RATE_WEIGHT = 0.3


@dataclass
class GoodputCoupledBudget(RetryController):
    """Retries in proportion to the work that succeeds, per tenant (Arolla).

    A tenant may have at most `alpha x goodput` retries in any `window`, where
    its goodput is the rate of its attempts that SUCCEED here, smoothed over
    windows: quickly downwards (`beta_down`) and slowly upwards (`beta_up`). So
    a tenant's retries are cut as soon as its requests stop succeeding and come
    back only as they succeed again, and one tenant's storm cannot spend
    another's allowance. Until a tenant's first window has ended it is allowed
    one retry per window.

    The retries in the window are COUNTED and compared with the allowance as it
    is, fractions included: a retry is admitted while the count is below it. So
    an allowance of 2.5 admits three retries, and a tenant whose goodput has
    collapsed -- the estimate decays towards zero and does not reach it -- still
    gets the first retry of every window. With many tenants that floor adds up:
    it is one retry per tenant per window, whatever the server's state.

    Two optional refinements:

    * `success_rate_threshold`: refuse a tenant's retries while the smoothed
      success rate of ITS retries (weight `success_rate_beta` on each new one)
      is below the threshold: retrying is not helping;
    * `max_retry_ratio`: also cap the retries in a window at this fraction of
      the tenant's smoothed attempt rate, which holds back the first burst
      after a fault, when the goodput estimate is still the healthy one.

    The algorithm is Arolla's (ETH EASL, https://github.com/eth-easl/Arolla,
    `GoodputCoupledRetryBudget`), ported: the goodput is theirs to couple
    retries to. Rates are smoothed estimates and are kept as floats, as there;
    windows and counts are exact.
    """

    alpha: float = 0.1
    beta_down: float = 0.3
    beta_up: float = 0.05
    window: TimeDuration = S_TO_NS
    success_rate_threshold: float | None = None
    success_rate_beta: float = 0.1
    max_retry_ratio: float | None = None

    _tenants: dict[str | None, _GoodputState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _fraction(self.alpha, "alpha")
        for name in ("beta_down", "beta_up", "success_rate_beta"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")
        _positive_duration(self.window, "window")
        if self.success_rate_threshold is not None:
            _fraction(self.success_rate_threshold, "success_rate_threshold")
        if self.max_retry_ratio is not None and not (
            math.isfinite(self.max_retry_ratio) and 0 < self.max_retry_ratio <= 1
        ):
            raise ValueError(f"max_retry_ratio must be in (0, 1], got {self.max_retry_ratio!r}")

    def state(self, tenant: str | None) -> _GoodputState:
        """The state kept for `tenant` (created on first use)."""
        state = self._tenants.get(tenant)
        if state is None:
            state = self._tenants[tenant] = _GoodputState()
        return state

    def _advance(self, state: _GoodputState, now: TimePoint) -> None:
        """Close every window that has ended, folding its counts into the rates."""
        if state.window_start is None:
            state.window_start = now
            return
        seconds = self.window / S_TO_NS
        while now - state.window_start >= self.window:
            goodput = state.goodput / seconds
            requests = state.requests / seconds
            if not state.initialized:
                # The first window is taken as it is: starting the estimate at
                # zero would lock a new tenant out of every retry.
                state.goodput_rate = goodput
                state.request_rate = requests
                state.initialized = True
            else:
                beta = self.beta_down if goodput < state.goodput_rate else self.beta_up
                state.goodput_rate = (1 - beta) * state.goodput_rate + beta * goodput
                state.request_rate = (
                    1 - _REQUEST_RATE_WEIGHT
                ) * state.request_rate + _REQUEST_RATE_WEIGHT * requests
            state.goodput = 0
            state.requests = 0
            state.window_start += self.window

    def observe(self, outcome: AdmissionOutcome) -> None:
        state = self.state(outcome.tenant)
        self._advance(state, outcome.now)
        state.requests += 1
        if outcome.success:
            state.goodput += 1
        if outcome.is_retry:
            state.retry_success = (
                1 - self.success_rate_beta
            ) * state.retry_success + self.success_rate_beta * (1.0 if outcome.success else 0.0)

    def allowance(self, tenant: str | None) -> float:
        """Retries the tenant may have in one window, by the current estimates."""
        state = self.state(tenant)
        seconds = self.window / S_TO_NS
        allowance = self.alpha * state.goodput_rate * seconds
        if self.max_retry_ratio is not None and state.request_rate > 0:
            allowance = min(allowance, self.max_retry_ratio * state.request_rate * seconds)
        if not state.initialized and allowance == 0:
            allowance = 1.0
        return allowance

    def admits_retry(self, request: AdmissionRequest) -> bool:
        state = self.state(request.tenant)
        self._advance(state, request.now)
        cutoff = request.now - self.window
        while state.retries and state.retries[0] < cutoff:
            state.retries.popleft()
        if len(state.retries) >= self.allowance(request.tenant):
            return False
        return not (
            self.success_rate_threshold is not None
            and state.initialized
            and state.retry_success < self.success_rate_threshold
        )

    def spend(self, request: AdmissionRequest) -> None:
        self.state(request.tenant).retries.append(request.now)

    def give_back(self, request: AdmissionRequest) -> None:
        retries = self.state(request.tenant).retries
        if retries:
            retries.pop()


@dataclass
class ConcurrencyRetryBudget(RetryController):
    """Retries as a share of what is in flight (Envoy's retry budget).

    A retry is admitted while the retries in flight are fewer than
    `budget_percent` of ALL the attempts in flight, and never fewer than
    `min_retry_concurrency` are allowed. It needs no rate and no window: when
    the server slows down, attempts pile up in flight and so does the retry
    allowance, but only in proportion.

    The counts are the asking gate's own (`AdmissionRequest.outstanding`), so
    this controller works at a gate and not in a `ControlledRetryPolicy`.
    """

    budget_percent: float = 20.0
    min_retry_concurrency: int = 3

    _share: Fraction = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._share = _fraction(self.budget_percent, "budget_percent", high=100.0) / 100
        if (
            isinstance(self.min_retry_concurrency, bool)
            or not isinstance(self.min_retry_concurrency, int)
            or self.min_retry_concurrency < 0
        ):
            raise ValueError(
                f"min_retry_concurrency must be an int >= 0, got {self.min_retry_concurrency!r}"
            )

    def admits_retry(self, request: AdmissionRequest) -> bool:
        if request.outstanding is None or request.outstanding_retries is None:
            raise ValueError(
                "ConcurrencyRetryBudget needs the number of attempts in flight, which only "
                "an AdmissionGate knows; it cannot be used in a ControlledRetryPolicy"
            )
        # retries < max(minimum, share x outstanding), without the division.
        if request.outstanding_retries < self.min_retry_concurrency:
            return True
        return (
            request.outstanding_retries * self._share.denominator
            < self._share.numerator * request.outstanding
        )


@dataclass
class AdmissionCounts:
    """What a gate saw of one tenant."""

    attempts: int = 0  # attempts that arrived at the gate
    rejected: int = 0  # ... that it refused
    retries_asked: int = 0  # retries that arrived (by the gate's scope)
    retries_admitted: int = 0  # ... that it let in


@dataclass
class AdmissionGate:
    """A `ServiceLike` decorator that puts admission controllers in front of a target.

    Drop-in for the target it wraps, as `transport.TransportLink` is: a client,
    an edge's client or a link submits its attempts to the gate, and attempts
    stay booked under the target's name. Each attempt is put to every controller
    (`check`); if all agree each is told to `commit` and the attempt is
    forwarded, with the gate watching how it ends (`observe`). If one refuses,
    the attempt is reported at once as `DropReason.REJECTED` and the target
    never hears of it -- no queue slot, no worker, no `admitted_total`.

    `retry_scope` says which attempts count as retries here:

    * `"hop"` (default): a retry OF THE CALLER IN FRONT OF THE GATE
      (`RequestContext.attempt > 1`). A first call made on behalf of a retry
      further up is a first call.
    * `"request"`: anything caused by a retry anywhere above
      (`RequestContext.is_retry`), so the first downstream call of a retried
      upstream attempt is a retry too.

    The gate keeps its own account of what it has let in and not yet seen end
    (`outstanding`, `outstanding_retries`) rather than reading the target's: a
    station's `in_flight` excludes what waits in its queue and includes work for
    callers that have left, and a chain node holds no worker while it waits on
    its downstreams. Controllers that need a load figure are given these.

    Every outcome the target reports is observed, including the answer to an
    attempt its caller has stopped waiting for. A cancelled attempt is not
    observed (it says nothing about the server) but does leave the account.
    """

    target: ServiceLike
    controllers: Sequence[AdmissionController]
    retry_scope: str = "hop"

    #: Attempts let in and not yet ended, and how many of them are retries.
    outstanding: int = field(default=0, init=False)
    outstanding_retries: int = field(default=0, init=False)
    #: Per tenant (None for requests without one).
    counts: dict[str | None, AdmissionCounts] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.retry_scope not in RETRY_SCOPES:
            raise ValueError(
                f"retry_scope must be one of {', '.join(RETRY_SCOPES)}; got {self.retry_scope!r}"
            )
        self.controllers = tuple(self.controllers)
        for controller in self.controllers:
            if not isinstance(controller, AdmissionController):
                raise ValueError(f"a controller must be an AdmissionController, got {controller!r}")

    @property
    def cfg(self) -> ServiceConfig:
        # Not a service of its own: attempts stay booked under the target's name.
        return self.target.cfg

    def counts_for(self, tenant: str | None) -> AdmissionCounts:
        counts = self.counts.get(tenant)
        if counts is None:
            counts = self.counts[tenant] = AdmissionCounts()
        return counts

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: AttemptDoneCallback,
        attempt_deadline: TimePoint | None = None,
        *,
        ctx: RequestContext | None = None,
    ) -> AttemptControl:
        if ctx is None:
            raise ValueError(
                f"the admission gate in front of {self.cfg.name!r} was handed an attempt "
                "without a request context, so it cannot tell a retry from a first try; "
                "give the client that starts these requests a ClientConfig.tenant"
            )
        is_retry = ctx.attempt > 1 if self.retry_scope == "hop" else ctx.is_retry
        request = AdmissionRequest(
            now=sim.timestep,
            tenant=ctx.tenant,
            attempt=ctx.attempt,
            is_retry=is_retry,
            outstanding=self.outstanding,
            outstanding_retries=self.outstanding_retries,
        )
        counts = self.counts_for(ctx.tenant)
        counts.attempts += 1
        if is_retry:
            counts.retries_asked += 1

        # Two phases: nobody spends until everybody has agreed.
        if not all(controller.check(request) for controller in self.controllers):
            counts.rejected += 1
            # Refused on arrival. Reported synchronously, like a full queue, on
            # a handle that is already settled: there is nothing to cancel. No
            # queue size: the attempt never saw the queue.
            control = AttemptControl()
            control._settle()
            on_done(False, 0, DropReason.REJECTED, None, False)
            return control
        for controller in self.controllers:
            controller.commit(request)

        # The books are closed BEFORE forwarding: a target may report from
        # inside submit_attempt (a full queue), and that report must find the
        # attempt already counted.
        if is_retry:
            counts.retries_admitted += 1
            self.outstanding_retries += 1
        self.outstanding += 1
        reported = False

        def observed(
            success: bool,
            occupancy: TimeDuration,
            drop: DropReason,
            queue_size: int | None,
            occupancy_censored: bool,
        ) -> None:
            nonlocal reported
            if reported:
                raise RuntimeError(
                    f"the target behind the admission gate of {self.cfg.name!r} reported "
                    "one attempt more than once"
                )
            reported = True
            self.outstanding -= 1
            if is_retry:
                self.outstanding_retries -= 1
            if drop is not DropReason.CANCELLED:
                outcome = AdmissionOutcome(
                    now=sim.timestep,
                    tenant=ctx.tenant,
                    attempt=ctx.attempt,
                    is_retry=is_retry,
                    success=success and drop is DropReason.NONE,
                    drop_reason=drop,
                )
                for controller in self.controllers:
                    controller.observe(outcome)
            on_done(success, occupancy, drop, queue_size, occupancy_censored)

        # The target's own handle: the gate adds no stage in which an attempt
        # could be cancelled, and a cancellation reports through `observed`.
        return self.target.submit_attempt(sim, observed, attempt_deadline=attempt_deadline, ctx=ctx)


@dataclass
class ControlledRetryPolicy(RetryPolicy):
    """Admission controllers applied by the CALLER, to its own retries.

    The same rules as at a gate, on the other side of the wire: a client (or a
    service retrying its own handler) that holds its retries to a budget or
    stops them while its calls fail. A layer of the client's retry policy like
    any other throttle: it is told every attempt's outcome, a retry needs every
    controller's agreement AND the wrapped policy's, the controllers are charged
    only for a retry that is granted, and refunded if it is then not issued.

    What it cannot do is see the server: it knows this client's attempts only,
    so controllers that need the load in flight (`ConcurrencyRetryBudget`)
    refuse to work here, and several clients must share the controller OBJECT to
    share a budget.
    """

    controllers: Sequence[AdmissionController]
    underlying_strategy: RetryPolicy

    def __post_init__(self) -> None:
        self.controllers = tuple(self.controllers)
        for controller in self.controllers:
            if not isinstance(controller, AdmissionController):
                raise ValueError(f"a controller must be an AdmissionController, got {controller!r}")

    @staticmethod
    def _tenant(context: RetryContext) -> str | None:
        return None if context.request is None else context.request.tenant

    def _retry_of(self, context: RetryContext) -> AdmissionRequest:
        # The retry being asked for: the attempt AFTER the one that just ended.
        return AdmissionRequest(
            now=context.now,
            tenant=self._tenant(context),
            attempt=context.attempt + 1,
            is_retry=True,
        )

    def record_attempt(self, context: RetryContext, success: bool) -> None:
        reason = context.drop_reason
        if reason is None:
            reason = DropReason.NONE if success else DropReason.SERVER_FAILURE
        outcome = AdmissionOutcome(
            now=context.now,
            tenant=self._tenant(context),
            attempt=context.attempt,
            is_retry=context.attempt > 1,
            success=success,
            drop_reason=reason,
        )
        for controller in self.controllers:
            controller.observe(outcome)

    def next_delay(self, context: RetryContext) -> DelayDecision:
        request = self._retry_of(context)
        if not all(controller.check(request) for controller in self.controllers):
            return DelayDecision(False, 0)
        decision = self.underlying_strategy.next_delay(context)
        if not decision.allow:
            return DelayDecision(False, 0)
        for controller in self.controllers:
            controller.commit(request)
        return decision

    def rollback(self, context: RetryContext) -> None:
        request = self._retry_of(context)
        for controller in self.controllers:
            controller.refund(request)
