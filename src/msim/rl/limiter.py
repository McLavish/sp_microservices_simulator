"""A server-side agent: the retry quota of an admission gate, tenant by tenant.

`RetryGrid` puts an agent on a group of clients. `QuotaGrid` puts one in front
of a service instead, as its limiter. The clients keep whatever retry policy
they have; what the agent chooses, window after window, is how many of each
tenant's retries the service lets in (`admission.TenantRetryQuota`, one share
per tenant).

What it is shown is what a limiter can count where it stands, per tenant and
window, and nothing else: the attempts that ended at the service, those that
succeeded, the retries that asked to be let in and those that were. These are
the quantities Arolla's rule is driven by (`GoodputCoupledBudget`: a tenant's
request rate, its goodput, its retries in the window, the success of its
retries). No queue length, no latency, nothing from inside a client, and no
word of a fault.

    quota = QuotaGrid()
    agent = quota.install(gate, tenants, clients, rates)   # while building an episode
    scenario.knobs    = quota.knobs(tenants)
    scenario.features = quota.features(tenants)
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from msim.admission import (
    AdmissionController,
    AdmissionGate,
    AdmissionOutcome,
    AdmissionRequest,
    TenantRetryQuota,
)
from msim.caller import ClientRuntime
from msim.core import s_to_ns
from msim.rl.scenario import Knob

#: What is measured for every tenant, in this order (`<tenant>:<measure>`).
MEASURES = (
    "ended",  # attempts that ended at the service, over ATTEMPTS x the rate expected
    "goodput",  # ... of them the successes, over GOODPUT x the rate expected
    "retries_asked",  # retries that asked to be let in, over ATTEMPTS x the rate expected
    "retries_admitted",  # ... and those let in
    "retry_success",  # share of the retries that ended which succeeded
)
#: The scales, in requests of the tenant when all is well. A tenant that tries
#: four times sends up to four attempts per request; goodput cannot pass the
#: requests that arrive, and twice the expected rate leaves room for a heavy load.
ATTEMPTS = 4.0
GOODPUT = 2.0

_PREFIX = "quota:"


class _Seen:
    """What passed the gate for one tenant since it was last read."""

    __slots__ = (
        "ended",
        "firsts",
        "goodput",
        "retries_admitted",
        "retries_asked",
        "retries_ended",
        "retries_ok",
    )

    def __init__(self) -> None:
        self.firsts = 0  # first attempts let in: requests that began
        self.ended = 0  # attempts that ended at the service
        self.goodput = 0  # ... successfully
        self.retries_asked = 0
        self.retries_admitted = 0
        self.retries_ended = 0
        self.retries_ok = 0


class _GateTap(AdmissionController):
    """Counts what a gate sees, tenant by tenant, and decides nothing.

    It stands FIRST among the gate's controllers: a gate stops asking at the
    first refusal, and a retry that is refused has still asked.
    """

    def __init__(self) -> None:
        self.seen: dict[str | None, _Seen] = {}

    def of(self, tenant: str | None) -> _Seen:
        seen = self.seen.get(tenant)
        if seen is None:
            seen = self.seen[tenant] = _Seen()
        return seen

    def take(self, tenant: str | None) -> _Seen:
        """The tenant's counts since the last call; they start again at zero."""
        return self.seen.pop(tenant, None) or _Seen()

    def check(self, request: AdmissionRequest) -> bool:
        if request.is_retry:
            self.of(request.tenant).retries_asked += 1
        return True

    def commit(self, request: AdmissionRequest) -> None:
        seen = self.of(request.tenant)
        if request.is_retry:
            seen.retries_admitted += 1
        else:
            seen.firsts += 1

    def observe(self, outcome: AdmissionOutcome) -> None:
        seen = self.of(outcome.tenant)
        seen.ended += 1
        seen.goodput += outcome.success
        if outcome.is_retry:
            seen.retries_ended += 1
            seen.retries_ok += outcome.success


class LimiterKnobs:
    """The quota of each tenant in one episode, and what the limiter counts.

    Made by `QuotaGrid.install`. A share takes effect at the tenant's next retry
    (`docs/request-handling.md`, "Changing a policy while a run is in
    progress"); setting one draws nothing and schedules nothing.
    """

    def __init__(
        self,
        quota: TenantRetryQuota,
        tap: _GateTap,
        tenants: Sequence[str],
        clients: Sequence[ClientRuntime],
        rates: Mapping[str, float],
        rest: float | None = None,
    ) -> None:
        self.quota = quota
        self._tap = tap
        #: The share every tenant has while no retry asks (`QuotaGrid.rest`), or `None`.
        self._rest = rest
        self._asked = 1  # retries that asked in the last window measured (none measured yet)
        self.tenants = tuple(tenants)
        #: Every client in front of the gate: the requests the quota governs.
        self.clients = list(clients)
        #: Requests per second each tenant sends when all is well: the scale of
        #: what is measured, the same in every episode.
        self.rates = {tenant: float(rates[tenant]) for tenant in self.tenants}

    def apply(self, values: Mapping[str, float]) -> Mapping[str, float] | None:
        """Give each named tenant (`quota:<tenant>`) its share.

        With `QuotaGrid(rest=True)`, when no retry asked in the window last
        measured, every tenant gets the default share instead, whatever is
        given, and that is what is returned (the shares in force).
        """
        for name in values:
            tenant = name.removeprefix(_PREFIX)
            if not name.startswith(_PREFIX) or tenant not in self.rates:
                raise KeyError(
                    f"no quota named {name!r}; there are "
                    f"{[_PREFIX + tenant for tenant in self.tenants]}"
                )
        resting = self._rest is not None and self._asked == 0
        if resting:
            values = {_PREFIX + tenant: self._rest for tenant in self.tenants}
        for name, share in values.items():
            self.quota.set_share(name.removeprefix(_PREFIX), share)
        return dict(values) if resting else None

    def measure(self, window_s: float) -> tuple[int, list[float]]:
        """Requests that began in the window, and `MEASURES` for every tenant; resets the counts."""
        started = 0
        self._asked = 0
        values: list[float] = []
        for tenant in self.tenants:
            seen = self._tap.take(tenant)
            started += seen.firsts
            self._asked += seen.retries_asked
            expected = self.rates[tenant] * window_s
            per_request = 1.0 / expected if expected > 0 else 0.0  # a warm-up of no length
            values += [
                seen.ended * per_request / ATTEMPTS,
                seen.goodput * per_request / GOODPUT,
                seen.retries_asked * per_request / ATTEMPTS,
                seen.retries_admitted * per_request / ATTEMPTS,
                seen.retries_ok / seen.retries_ended if seen.retries_ended else 0.0,
            ]
        return started, values


@dataclass(frozen=True)
class QuotaGrid:
    """The shares a limiter's agent chooses from, for every tenant alike.

    A share is retries per first attempt of the tenant, counted over `window_s`.
    The last level is out of reach of a caller that tries four times: it refuses
    nothing. The default is what holds until the agent's first action, the
    warm-up included, and 0 leaves the service to its first attempts.

    `floor` (default 0) is `TenantRetryQuota.floor`: retries per tenant and
    window that are let in whatever the level. An agent whose only view of a
    failure is what its retries do keeps a level above zero to go on seeing,
    and pays for it in retries that cannot succeed; with a floor of 1 the
    lowest level is no longer blind.

    `rest` (default `False`): while no retry asks, the quota is at its default.
    When the last window had no retry asking, of any tenant, every tenant's
    share is the default whatever the agent chose, and the agent is shown that
    level. The level an agent holds while nothing fails earns nothing either
    way, so what it learns to hold is arbitrary, it stays there (its own level
    is part of what it is shown), and a failure then starts from wherever that
    was. With `rest` every failure starts from the default.
    """

    shares: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 3.0)
    default_share: float = 0.0
    window_s: float = 1.0
    floor: int = 0
    rest: bool = False

    def __post_init__(self) -> None:
        if self.default_share not in self.shares:
            raise ValueError(
                f"default_share = {self.default_share!r} is not one of its levels {self.shares}"
            )
        if any(share < 0 for share in self.shares) or self.window_s <= 0:
            raise ValueError("shares are >= 0 and the window is > 0")

    def knobs(self, tenants: Sequence[str]) -> tuple[Knob, ...]:
        """One knob per tenant, in the order given (which is the order of the action)."""
        levels = tuple(float(share) for share in self.shares)
        default = self.shares.index(self.default_share)
        return tuple(Knob(_PREFIX + tenant, levels, default) for tenant in tenants)

    def features(self, tenants: Sequence[str]) -> tuple[str, ...]:
        """The names of what `LimiterKnobs.measure` returns."""
        return tuple(f"{tenant}:{measure}" for tenant in tenants for measure in MEASURES)

    def install(
        self,
        gate: AdmissionGate,
        tenants: Sequence[str],
        clients: Sequence[ClientRuntime],
        rates: Mapping[str, float],
    ) -> LimiterKnobs:
        """Make the gate's limiter this quota, at the default share for every tenant.

        Call it while the simulation is being built, before any request starts.
        Whatever controllers the gate had are REPLACED: the agent is the
        limiter. `clients` are the clients in front of the gate and `rates` the
        requests per second each tenant sends when all is well.
        """
        if not tenants:
            raise ValueError("a limiter needs at least one tenant")
        missing = [tenant for tenant in tenants if tenant not in rates]
        if missing:
            raise ValueError(f"no rate given for {missing}")
        quota = TenantRetryQuota(
            share=self.default_share, window=s_to_ns(self.window_s), floor=self.floor
        )
        tap = _GateTap()
        gate.controllers = (tap, quota)
        rest = self.default_share if self.rest else None
        return LimiterKnobs(quota, tap, tenants, clients, rates, rest=rest)
