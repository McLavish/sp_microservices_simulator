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
word of a fault. They are shown as RATIOS of one another (`MEASURES`), never
against a rate the tenant "should" send: nothing about a deployment has to be
known to put such a limiter in front of it.

`QuotaGrid(in_flight=True)` adds the one thing more a proxy can count without
being told anything: the attempts it has let in and not yet seen end. It is
shown as how much longer an attempt is in flight than while nothing fails
(`IN_FLIGHT`), which is where a limiter sees that the service has no room left
for the retries it lets in.

    quota = QuotaGrid()
    agent = quota.install(gate, tenants, clients)   # while building an episode
    scenario.knobs    = quota.knobs(tenants)
    scenario.features = quota.features(tenants)
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from msim.admission import (
    AdmissionController,
    AdmissionGate,
    AdmissionOutcome,
    AdmissionRequest,
    RetryController,
    TenantRetryQuota,
)
from msim.caller import ClientRuntime
from msim.core import s_to_ns
from msim.rl.scenario import Knob

#: What is measured for every tenant, in this order (`<tenant>:<measure>`):
#: ratios of the window's own counts, so that nothing about the deployment (no
#: rate a tenant sends "when all is well") is configured.
MEASURES = (
    "succeeded",  # share of the attempts that ended which succeeded
    "first_success",  # ... of the FIRST attempts that ended
    "asked_per_first",  # retries that asked per first attempt let in, over ATTEMPTS - 1
    "admitted_share",  # share of the retries that asked which were let in
    "retry_success",  # share of the retries that ended which succeeded
)
#: Measured once for the whole gate, after every tenant's entries, with
#: `QuotaGrid(in_flight=True)`: 1 - (time in flight at rest) / (time in flight
#: now), so 0 while an attempt is in flight no longer than at rest and towards 1
#: as attempts pile up. The time in flight is attempts in flight, averaged over
#: the window, per attempt let in during it (what a proxy has without timing a
#: single request), and "at rest" is the lowest it was in a window in which no
#: retry asked, allowed to drift up by `DRIFT` a window.
IN_FLIGHT = "in_flight"
DRIFT = 0.02
#: A window with fewer attempts let in than this does not move the level at rest.
AT_REST = 10
#: `QuotaGrid(in_flight_cap=c)` never refuses a retry while fewer attempts than this are in flight.
CAP_FLOOR = 8.0
#: The attempts of a request the scale of `asked_per_first` assumes: a caller
#: that tries four times asks up to three retries per first attempt.
ATTEMPTS = 4.0

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
        self.in_flight = 0  # attempts let in whose end has not been seen
        self._flight_ns = 0  # ... summed over time since it was last read (attempt x ns)
        self._at = 0  # the instant the sum is up to

    def _advance(self, now: int) -> None:
        self._flight_ns += self.in_flight * (now - self._at)
        self._at = now

    def take_flight(self, now: int) -> int:
        """Attempts in flight summed over time (attempt x ns) since the last call, up to `now`."""
        self._advance(max(now, self._at))
        flight, self._flight_ns = self._flight_ns, 0
        return flight

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
        self._advance(request.now)
        if request.outstanding is not None:
            # The gate's own count: a cancelled attempt leaves it without being observed.
            self.in_flight = request.outstanding
        self.in_flight += 1
        seen = self.of(request.tenant)
        if request.is_retry:
            seen.retries_admitted += 1
        else:
            seen.firsts += 1

    def observe(self, outcome: AdmissionOutcome) -> None:
        self._advance(outcome.now)
        self.in_flight -= 1
        seen = self.of(outcome.tenant)
        seen.ended += 1
        seen.goodput += outcome.success
        if outcome.is_retry:
            seen.retries_ended += 1
            seen.retries_ok += outcome.success


class _FlightCap(RetryController):
    """Refuses a retry while the gate has more than `limit` attempts in flight.

    The one rule here that acts within a window: a quota is a count per window,
    and a service that is pushed past what it serves is past it within
    milliseconds. `limit` is set by `LimiterKnobs.measure` (`QuotaGrid.in_flight_cap`);
    `None`, until there is a window at rest to compare with, refuses nothing.
    """

    def __init__(self) -> None:
        self.limit: float | None = None
        self.refused = 0  # retries it refused since the episode began

    def admits_retry(self, request: AdmissionRequest) -> bool:
        if self.limit is None or request.outstanding is None:
            return True
        if request.outstanding <= self.limit:
            return True
        self.refused += 1
        return False


class _AttemptBound(RetryController):
    """Refuses a retry beyond the `max_attempt`-th attempt of its request (`QuotaGrid.max_attempt`)."""

    def __init__(self, max_attempt: int) -> None:
        self.max_attempt = max_attempt

    def admits_retry(self, request: AdmissionRequest) -> bool:
        return request.attempt <= self.max_attempt


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
        rest: float | None = None,
        in_flight: bool = False,
        ladder: Sequence[float] = (),
        step_up: int = 0,
        start: float = 0.0,
        hold_above: float | None = None,
        cap: _FlightCap | None = None,
        cap_times: float = 0.0,
        asked_per_failure: float | None = None,
        confirm: bool = False,
        await_rest: bool = False,
    ) -> None:
        self.quota = quota
        self._tap = tap
        #: `QuotaGrid.in_flight_cap`: the controller that refuses retries while the
        #: service is crowded, and how many times the attempts in flight at rest it allows.
        self.cap = cap
        self._cap_times = cap_times
        #: `QuotaGrid.asked_per_failure`: the retries a tenant is SHOWN to have asked
        #: are at most this many per first attempt of its that failed in the window.
        self._asked_per_failure = asked_per_failure
        #: `QuotaGrid.step_up`: the levels of `ladder` a share may rise per window (0: any);
        #: `QuotaGrid.hold_above`: no rise at all while `IN_FLIGHT` is above it. Each
        #: tenant's place on the ladder now (it starts at `start`) is kept for both.
        self._ladder = tuple(ladder)
        self._start = self._ladder.index(start) if self._ladder else 0
        self._step_up = step_up
        self._hold_above = hold_above
        #: `QuotaGrid.confirm`: no share rises until retries have asked in two windows running.
        self._confirm = confirm
        #: `QuotaGrid.await_rest`: every share at its default until there has been a window at rest.
        self._await_rest = await_rest
        self._asked_before = 0  # retries that asked in the window before the last one measured
        self._bounded = bool(step_up) or hold_above is not None or confirm
        self._place = (
            {tenant: self._ladder.index(start) for tenant in tenants} if self._bounded else {}
        )
        self._entry = 0.0  # `IN_FLIGHT` of the last window measured
        #: Whether `IN_FLIGHT` is measured (`QuotaGrid.in_flight`).
        self._in_flight = in_flight
        self._now = 0  # the end of the last window measured, ns
        self._flight_at_rest: float | None = None  # ns in flight per attempt, while nothing fails
        #: The share every tenant has while no retry asks (`QuotaGrid.rest`), or `None`.
        self._rest = rest
        self._asked = 1  # retries that asked in the last window measured (none measured yet)
        self.tenants = tuple(tenants)
        #: Every client in front of the gate: the requests the quota governs.
        self.clients = list(clients)

    def apply(self, values: Mapping[str, float]) -> Mapping[str, float] | None:
        """Give each named tenant (`quota:<tenant>`) its share.

        With `QuotaGrid(rest=True)`, when no retry asked in the window last
        measured, every tenant gets the default share instead, whatever is
        given. With `QuotaGrid(step_up=n)` a share rises by at most `n` levels
        of the ladder from what the tenant had, and with `hold_above` not at
        all while attempts are in flight that much longer than at rest. In
        these cases the shares in force are returned.
        """
        for name in values:
            tenant = name.removeprefix(_PREFIX)
            if not name.startswith(_PREFIX) or tenant not in self.tenants:
                raise KeyError(
                    f"no quota named {name!r}; there are "
                    f"{[_PREFIX + tenant for tenant in self.tenants]}"
                )
        rest = self._rest
        resting = rest is not None and self._asked == 0
        waiting = self._await_rest and self._flight_at_rest is None
        if waiting:
            # Nothing to compare the attempts in flight with yet: the default, as at the start.
            resting = True
            values = {_PREFIX + tenant: self._ladder[self._start] for tenant in self.tenants}
        elif rest is not None and self._asked == 0:
            values = {_PREFIX + tenant: rest for tenant in self.tenants}
        elif self._bounded:
            # No further up than `step_up` levels from where the tenant stands,
            # and not up at all while the service is crowded.
            held = (self._hold_above is not None and self._entry > self._hold_above) or (
                self._confirm and self._asked_before == 0
            )
            rise = 0 if held else (self._step_up or len(self._ladder))
            values = {
                name: self._ladder[
                    min(self._ladder.index(share), self._place[name.removeprefix(_PREFIX)] + rise)
                ]
                for name, share in values.items()
            }
        for name, share in values.items():
            tenant = name.removeprefix(_PREFIX)
            self.quota.set_share(tenant, share)
            if self._bounded:
                self._place[tenant] = self._ladder.index(share)
        return dict(values) if resting or self._bounded else None

    def measure(self, window_s: float) -> tuple[int, list[float]]:
        """Requests that began in the window, and `MEASURES` for every tenant; resets the counts.

        With `QuotaGrid(in_flight=True)` one more value follows them, `IN_FLIGHT`.
        """
        started = 0
        admitted = 0
        self._asked_before = (
            self._asked if self._now else 0
        )  # nothing was measured before the first
        self._asked = 0
        values: list[float] = []
        for tenant in self.tenants:
            seen = self._tap.take(tenant)
            started += seen.firsts
            admitted += seen.firsts + seen.retries_admitted
            self._asked += seen.retries_asked
            firsts_ended = seen.ended - seen.retries_ended
            firsts_ok = seen.goodput - seen.retries_ok
            asked: float = seen.retries_asked
            if self._asked_per_failure is not None:
                asked = min(asked, self._asked_per_failure * max(firsts_ended - firsts_ok, 0))
            values += [
                seen.goodput / seen.ended if seen.ended else 0.0,
                firsts_ok / firsts_ended if firsts_ended > 0 else 0.0,
                asked / ((ATTEMPTS - 1.0) * seen.firsts) if seen.firsts else float(asked > 0),
                seen.retries_admitted / seen.retries_asked if seen.retries_asked else 0.0,
                seen.retries_ok / seen.retries_ended if seen.retries_ended else 0.0,
            ]
        self._now += s_to_ns(window_s)
        self._entry = self._crowding(self._tap.take_flight(self._now), admitted)
        if self.cap is not None and window_s > 0 and self._flight_at_rest is not None:
            # What the attempts let in at this rate have in flight when each takes
            # the time it takes at rest (rate x time), times `in_flight_cap`.
            at_rest = admitted / s_to_ns(window_s) * self._flight_at_rest
            self.cap.limit = max(CAP_FLOOR, self._cap_times * at_rest)
        if self._in_flight:
            values.append(self._entry)
        return started, values

    def _crowding(self, flight_ns: int, admitted: int) -> float:
        """`IN_FLIGHT` for a window: its attempts in flight over time, and those let in."""
        in_flight = flight_ns / max(admitted, 1)  # ns in flight per attempt let in (Little's law)
        if in_flight > 0.0 and self._asked == 0 and admitted >= AT_REST:
            at_rest = self._flight_at_rest
            self._flight_at_rest = (
                in_flight if at_rest is None else min(in_flight, at_rest * (1.0 + DRIFT))
            )
        if self._flight_at_rest is None or in_flight <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self._flight_at_rest / in_flight)


def hand_rule(
    feature_names: Sequence[str],
    levels: int,
    *,
    alive: float = 0.05,
    down_above: float = 0.5,
    hold_above: float | None = None,
) -> Callable[[Any], list[int]]:
    """A limiter's policy written by hand, for the question what a learned one adds.

    Three lines on what a `QuotaGrid(in_flight=True)` agent is shown, per
    tenant: level 0 while fewer than `alive` of the tenant's attempts that
    ended succeeded; a level down while `IN_FLIGHT` is above `down_above`;
    otherwise the top level (held where it is above `hold_above`, if given).
    It is meant to run inside the same rules as a learned policy
    (`QuotaGrid(step_up=1, in_flight_cap=..., confirm=True, rest=True)`), which
    make "the top level" a climb of a level a window.

    `feature_names` and `levels` are the environment's (`RetryEnv.feature_names`,
    the number of levels of a knob); the result maps an observation to an
    action, as `rl.evaluate.rollout` takes it.
    """
    names = list(feature_names)
    if IN_FLIGHT not in names:
        raise ValueError(
            "the hand-written rule needs the in-flight entry: QuotaGrid(in_flight=True)"
        )
    crowding = names.index(IN_FLIGHT)
    tenants = [name.removeprefix("knob:" + _PREFIX) for name in names if name.startswith("knob:")]
    where = [
        (names.index(f"{tenant}:succeeded"), names.index(f"knob:{_PREFIX}{tenant}"))
        for tenant in tenants
    ]
    top = levels - 1

    def policy(observation: Any) -> list[int]:  # an array or a sequence of the entries
        action = []
        for succeeded_at, knob in where:
            level = round(float(observation[knob]) * top)
            succeeded = float(observation[succeeded_at])
            if succeeded < alive:
                action.append(0)
            elif float(observation[crowding]) > down_above:
                action.append(max(level - 1, 0))
            elif hold_above is not None and float(observation[crowding]) > hold_above:
                action.append(level)
            else:
                action.append(top)
        return action

    return policy


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

    `in_flight` (default `False`) shows the agent one entry more, for the gate
    as a whole: `IN_FLIGHT`, how much longer an attempt is in flight than while
    nothing fails. Counts tell a limiter whether retries succeed, not whether
    the service has room for them, and without this entry the room is whatever
    the scenarios it was trained on had.

    `step_up` (default 0: any) is how many levels a tenant's share may rise
    from one window to the next; it may fall as far as the agent likes. A
    limiter that lets in more than the service has room for learns of it a
    window later, and a deployed service was seen to stay congested for the
    rest of a failure after one such window: with `step_up=1` the agent climbs
    the ladder a level at a time and is shown each level's effect before the
    next. The agent is shown the level in force, as with `rest`.

    `hold_above` (default `None`: no such rule) is a value of `IN_FLIGHT`
    above which no share rises, whatever the agent says: while attempts are in
    flight that much longer than at rest, the quota holds or falls. It is
    measured whether or not the agent is shown it.

    `in_flight_cap` (default `None`: no such rule) is the bound that acts
    WITHIN a window: a retry is refused while the gate has more attempts in
    flight than `in_flight_cap` times what the attempts it let in during the
    last window have in flight at rest (their rate times the time one is in
    flight while nothing fails), and never while fewer than `CAP_FLOOR` are.
    By Little's law that holds the time an attempt is in flight to about
    `in_flight_cap` times its time at rest, whatever share the agent chose: the
    agent decides whether retries are worth letting in, this decides whether
    the service has room for them now. Until there has been a window at rest
    it refuses nothing.

    `asked_per_failure` (default `None`: as counted) bounds what the agent is
    SHOWN of the retries that asked: at most that many per first attempt of
    the tenant that failed in the window. The scales assume a caller that
    tries `ATTEMPTS` times; one that tries eleven times asks for ten retries
    per failed request, which an agent trained on the former reads as a far
    worse failure than it is, and closes for everybody. With
    `asked_per_failure=ATTEMPTS - 1` a caller cannot move the quota by asking
    more often. Nothing but that entry changes (what rests, what is refused
    and the counts in `info` are the real ones).

    `max_attempt` (default `None`: any) refuses a retry beyond that attempt of
    its request, whatever the share: with 4, a caller gets the three retries
    the scales assume and no more, so one that retries ten times is let in no
    more often per request than one that retries three times.

    `confirm` (default `False`): no share rises until retries have asked in
    two windows running. A failure begins inside a window, and that window
    looks like a partial failure whatever follows; an agent that opens on it
    lets a share of the retries through at the start of a total failure. With
    `confirm` it acts on a failure once it has seen a whole window of it.

    `await_rest` (default `False`): every share stays at its default until
    there has been a window at rest (one in which no retry asked), that is
    until the limiter has a time in flight to compare with. Before that the
    in-flight entry reads 0 and `in_flight_cap` refuses nothing, so a limiter
    that starts in the middle of trouble is blind to how crowded the service
    is: on a deployment one that started while its service was still cold, at
    a high load, opened level by level into the congestion and kept it going
    for the whole run. With `await_rest` it lets in its probes until it has
    seen the service well once.
    """

    shares: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 3.0)
    default_share: float = 0.0
    window_s: float = 1.0
    floor: int = 0
    rest: bool = False
    in_flight: bool = False
    step_up: int = 0
    hold_above: float | None = None
    in_flight_cap: float | None = None
    asked_per_failure: float | None = None
    max_attempt: int | None = None
    confirm: bool = False
    await_rest: bool = False

    def __post_init__(self) -> None:
        if self.default_share not in self.shares:
            raise ValueError(
                f"default_share = {self.default_share!r} is not one of its levels {self.shares}"
            )
        if self.step_up < 0:
            raise ValueError(f"step_up is a number of levels >= 0, got {self.step_up}")
        if self.hold_above is not None and not 0.0 <= self.hold_above <= 1.0:
            raise ValueError(f"hold_above is a value of the in-flight entry, got {self.hold_above}")
        if self.max_attempt is not None and self.max_attempt < 2:
            raise ValueError(f"max_attempt is an attempt number >= 2, got {self.max_attempt}")
        if self.in_flight_cap is not None and self.in_flight_cap < 1.0:
            raise ValueError(
                f"in_flight_cap is a multiple >= 1 of the attempts in flight at rest, "
                f"got {self.in_flight_cap}"
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
        per_tenant = tuple(f"{tenant}:{measure}" for tenant in tenants for measure in MEASURES)
        return per_tenant + ((IN_FLIGHT,) if self.in_flight else ())

    def install(
        self,
        gate: AdmissionGate,
        tenants: Sequence[str],
        clients: Sequence[ClientRuntime],
    ) -> LimiterKnobs:
        """Make the gate's limiter this quota, at the default share for every tenant.

        Call it while the simulation is being built, before any request starts.
        Whatever controllers the gate had are REPLACED: the agent is the
        limiter. `clients` are the clients in front of the gate. Nothing else
        is asked for: what the agent is shown has no unit.
        """
        if not tenants:
            raise ValueError("a limiter needs at least one tenant")
        quota = TenantRetryQuota(
            share=self.default_share, window=s_to_ns(self.window_s), floor=self.floor
        )
        tap = _GateTap()
        cap = _FlightCap() if self.in_flight_cap is not None else None
        bound = _AttemptBound(self.max_attempt) if self.max_attempt is not None else None
        # The tap first (a refused retry has still asked), the quota last.
        gate.controllers = tuple(
            controller for controller in (tap, bound, cap, quota) if controller is not None
        )
        rest = self.default_share if self.rest else None
        return LimiterKnobs(
            quota,
            tap,
            tenants,
            clients,
            rest=rest,
            in_flight=self.in_flight,
            ladder=self.shares,
            step_up=self.step_up,
            start=self.default_share,
            hold_above=self.hold_above,
            cap=cap,
            cap_times=self.in_flight_cap or 0.0,
            asked_per_failure=self.asked_per_failure,
            confirm=self.confirm,
            await_rest=self.await_rest,
        )
