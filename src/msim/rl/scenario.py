"""What a scenario gives the environment, and the settings an agent turns.

A `Scenario` describes a situation and is picklable, so that an environment can
be created in another process. For each episode it builds a fresh simulation
(`build(seed) -> Episode`); it also says, before anything is built, which
settings the agent chooses and among which values (`knobs`), and how long an
episode runs (`horizon_s`).

The stock settings are the retry settings of a group of clients, `RetryGrid`:
max attempts, attempt timeout, backoff and retry budget ratio. `RetryGrid.install`
gives each of those clients the policy stack the settings act on, and returns the
episode's `RetryKnobs`: the hands that apply a setting to all of them, and the
eyes, a `Window` of what those clients measured since it was last read. Nothing
here reads the state of a service: the agent sees what its clients see.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from msim.caller import ClientRuntime
from msim.core import DropReason, TimeDuration, TimePoint, ms_to_ns
from msim.retry_policies import (
    DelayDecision,
    FixedBackoffRetryPolicy,
    RetryBudgetPolicy,
    RetryContext,
    RetryPolicy,
)
from msim.service import ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.workload import Workload


@dataclass(frozen=True)
class Knob:
    """One setting the agent picks a level of at every step."""

    name: str
    #: The levels, in the order of the action's index. One level: a fixed setting,
    #: which is not part of the action.
    values: tuple[float, ...]
    #: Index of the level in force at reset and through the warm-up.
    default: int = 0

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"knob {self.name!r} needs at least one value")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"knob {self.name!r} lists a value twice: {self.values}")
        if not 0 <= self.default < len(self.values):
            raise ValueError(f"knob {self.name!r}: default index {self.default} is out of range")


class Window:
    """What the agent's clients measured since this was last reset.

    Counted as it happens, per attempt, by the layers `RetryGrid.install` puts
    on each client: an attempt is known the moment it ends, not when the request
    it belongs to has finished retrying.
    """

    __slots__ = (
        "attempts",
        "deadline",
        "latencies",
        "ok",
        "other",
        "queue_full",
        "requests",
        "retries",
        "retry_ok",
        "server_failure",
    )

    def __init__(self) -> None:
        self.latencies: list[TimeDuration] = []
        self.reset()

    def reset(self) -> None:
        self.requests = 0  # requests the clients started
        self.attempts = 0  # attempts that ended (an abandoned one is not counted)
        self.ok = 0  # ... successfully
        self.retries = 0  # ... that were not a request's first
        self.retry_ok = 0  # ... of those, successfully
        # Failed attempts by cause, as the client books it.
        self.deadline = 0
        self.queue_full = 0
        self.server_failure = 0
        self.other = 0  # refused by admission control
        #: Duration of each successful attempt, in ns. A new list: the old one
        #: may still be in the hands of whoever read it.
        self.latencies = []


class _Tap(RetryPolicy):
    """A policy layer that decides nothing and counts what its client sees.

    Outermost on an agent client: the client tells every layer of each request
    it starts and each attempt that ends, so this one hears of them too. It
    passes the retry decision through and has nothing to refund.
    """

    def __init__(self, inner: RetryPolicy, window: Window) -> None:
        self.underlying_strategy = inner
        self._window = window

    def next_delay(self, context: RetryContext) -> DelayDecision:
        return self.underlying_strategy.next_delay(context)

    def on_request_start(self, now: TimePoint) -> None:
        self._window.requests += 1

    def record_attempt(self, context: RetryContext, success: bool) -> None:
        window = self._window
        window.attempts += 1
        retry = context.attempt > 1
        if retry:
            window.retries += 1
        if success:
            window.ok += 1
            if retry:
                window.retry_ok += 1
            return
        reason = context.drop_reason
        if reason is DropReason.DEADLINE:
            window.deadline += 1
        elif reason is DropReason.QUEUE_FULL:
            window.queue_full += 1
        elif reason is DropReason.SERVER_FAILURE:
            window.server_failure += 1
        else:
            window.other += 1


@dataclass
class _TimedTimeouts(StaticTimeoutController):
    """A static timeout controller that also notes how long successful attempts took."""

    window: Window | None = None

    def record_result(self, success: bool, latency: TimeDuration) -> None:
        if success and self.window is not None:
            self.window.latencies.append(latency)


class RetryKnobs:
    """The retry settings of the agent's clients in one episode, and what they measure.

    Made by `RetryGrid.install`. `apply` sets a value on every client alike: one
    policy rolled out to all of them. The four settings are documented in
    `docs/request-handling.md` ("Changing a policy while a run is in progress"):
    each takes effect at the next decision it governs, and none draws from the
    simulator's RNG or schedules an event.
    """

    def __init__(
        self,
        clients: Sequence[ClientRuntime],
        window: Window,
        backoffs: Sequence[FixedBackoffRetryPolicy],
        budgets: Sequence[RetryBudgetPolicy],
        timeouts: Sequence[StaticTimeoutController],
        *,
        agent_rps: float,
        latency_scale: TimeDuration,
        max_attempts: int,
    ) -> None:
        self.clients = list(clients)
        self.window = window
        self._backoffs = list(backoffs)
        self._budgets = list(budgets)
        self._timeouts = list(timeouts)
        #: Requests per second the clients start when all is well.
        self.agent_rps = agent_rps
        #: What a latency is measured against: the longest an attempt may take.
        self.latency_scale = latency_scale
        #: Requests the clients can have in flight at worst: every one of them
        #: on its last attempt, each attempt running into its timeout.
        self.in_flight_scale = max(1.0, agent_rps * latency_scale / 1e9 * max_attempts)
        self._setters = {
            "max_attempts": self._set_max_attempts,
            "attempt_timeout_ms": self._set_attempt_timeout_ms,
            "backoff_ms": self._set_backoff_ms,
            "budget_ratio": self._set_budget_ratio,
        }

    def apply(self, values: Mapping[str, float]) -> None:
        """Put each named setting to its value, on every client."""
        for name, value in values.items():
            setter = self._setters.get(name)
            if setter is None:
                raise KeyError(
                    f"no retry setting named {name!r}; there are {sorted(self._setters)}"
                )
            setter(value)

    def _set_max_attempts(self, value: float) -> None:
        for backoff in self._backoffs:
            backoff.max_attempts = int(value)

    def _set_backoff_ms(self, value: float) -> None:
        delay = ms_to_ns(value)
        for backoff in self._backoffs:
            backoff.delay = delay

    def _set_attempt_timeout_ms(self, value: float) -> None:
        timeout = ms_to_ns(value)
        for controller in self._timeouts:
            controller.attempt_timeout = timeout

    def _set_budget_ratio(self, value: float) -> None:
        for budget in self._budgets:
            budget.set_budget_ratio(value)

    def budget_balance(self) -> float:
        """How full the clients' retry budgets are, 0 to 1 (1 when they have none)."""
        if not self._budgets:
            return 1.0
        full = 100.0 * len(self._budgets)  # get_token_balance: 100 is one retry
        return sum(b.get_token_balance() / b.max_retries for b in self._budgets) / full


@dataclass(frozen=True)
class RetryGrid:
    """The retry settings an agent chooses among, for a group of clients.

    Each tuple lists the levels of one setting; one level fixes it. `None`
    leaves the setting out: no attempt timeout knob (every client keeps the
    timeout it has), or no retry budget at all. The `default_*` values are the
    setting in force at reset and through the warm-up, the deployment the agent
    takes over; each must be one of its setting's levels.
    """

    max_attempts: tuple[int, ...] = (1, 2, 3, 4)
    attempt_timeout_ms: tuple[float, ...] | None = (50.0, 100.0, 200.0)
    backoff_ms: tuple[float, ...] = (0.0, 10.0, 50.0)
    budget_ratio: tuple[float, ...] | None = (0.05, 0.1, 0.2, 0.5, 1.0)
    #: The budget's cap, in retries (`RetryBudgetPolicy.max_retries`). Not a knob.
    budget_max_retries: int = 20
    default_max_attempts: int = 3
    default_attempt_timeout_ms: float | None = 100.0
    default_backoff_ms: float = 10.0
    default_budget_ratio: float | None = 0.2

    def __post_init__(self) -> None:
        self.knobs()  # validates every level list and default
        if any(int(n) != n or n < 1 for n in self.max_attempts):
            raise ValueError(f"max_attempts levels must be whole numbers >= 1: {self.max_attempts}")
        if self.attempt_timeout_ms is not None and min(self.attempt_timeout_ms) <= 0:
            raise ValueError(f"attempt_timeout_ms levels must be > 0: {self.attempt_timeout_ms}")
        if min(self.backoff_ms) < 0:
            raise ValueError(f"backoff_ms levels must be >= 0: {self.backoff_ms}")
        if self.budget_ratio is not None and not all(0 < r <= 1 for r in self.budget_ratio):
            raise ValueError(f"budget_ratio levels must be in (0, 1]: {self.budget_ratio}")
        if self.budget_max_retries < 1:
            raise ValueError(f"budget_max_retries must be >= 1, got {self.budget_max_retries}")

    def _levels(self) -> list[tuple[str, tuple[float, ...], float | None]]:
        levels: list[tuple[str, tuple[float, ...] | None, float | None]] = [
            ("max_attempts", self.max_attempts, self.default_max_attempts),
            ("attempt_timeout_ms", self.attempt_timeout_ms, self.default_attempt_timeout_ms),
            ("backoff_ms", self.backoff_ms, self.default_backoff_ms),
            ("budget_ratio", self.budget_ratio, self.default_budget_ratio),
        ]
        return [(name, values, default) for name, values, default in levels if values is not None]

    def knobs(self) -> tuple[Knob, ...]:
        """The settings as knobs, in a fixed order (which is the order of the action)."""
        knobs = []
        for name, values, default in self._levels():
            if default not in values:
                raise ValueError(f"default_{name} = {default!r} is not one of its levels {values}")
            knobs.append(Knob(name, tuple(float(v) for v in values), values.index(default)))
        return tuple(knobs)

    def defaults(self) -> dict[str, float]:
        """The default setting, by knob name."""
        return {knob.name: knob.values[knob.default] for knob in self.knobs()}

    def install(self, clients: Sequence[ClientRuntime], *, agent_rps: float) -> RetryKnobs:
        """Give `clients` the policy stack the settings act on, at the default setting.

        Call it while the simulation is being built, before any of them has
        started a request. Each client's retry policy is REPLACED by a fixed
        backoff, under a retry budget when the grid has one; its timeout
        controller, which must be a `StaticTimeoutController`, is replaced by
        one with the same timeouts. Tenant, `server_deadline` and the rest of
        its `ClientConfig` stay. `agent_rps` is the request rate the clients
        have together when all is well, the scale of what they report.
        """
        if not clients:
            raise ValueError("the agent needs at least one client")
        if not math.isfinite(agent_rps) or agent_rps <= 0:
            raise ValueError(f"agent_rps must be a rate > 0, got {agent_rps}")
        defaults = self.defaults()
        window = Window()
        backoffs: list[FixedBackoffRetryPolicy] = []
        budgets: list[RetryBudgetPolicy] = []
        timeouts: list[StaticTimeoutController] = []
        for client in clients:
            old = client.cfg.timeout_controller
            if not isinstance(old, StaticTimeoutController):
                raise TypeError(
                    "an agent client needs a StaticTimeoutController, "
                    f"this one has a {type(old).__name__}"
                )
            backoff = FixedBackoffRetryPolicy(
                max_attempts=int(defaults["max_attempts"]),
                delay=ms_to_ns(defaults["backoff_ms"]),
            )
            policy: RetryPolicy = backoff
            if self.budget_ratio is not None:
                budget = RetryBudgetPolicy(
                    budget_ratio=defaults["budget_ratio"],
                    max_retries=self.budget_max_retries,
                    underlying_strategy=backoff,
                )
                budgets.append(budget)
                policy = budget
            attempt_timeout = old.attempt_timeout
            if "attempt_timeout_ms" in defaults:
                attempt_timeout = ms_to_ns(defaults["attempt_timeout_ms"])
            controller = _TimedTimeouts(
                global_timeout=old.global_timeout, attempt_timeout=attempt_timeout, window=window
            )
            client.cfg = replace(
                client.cfg, retry=_Tap(policy, window), timeout_controller=controller
            )
            backoffs.append(backoff)
            timeouts.append(controller)
        return RetryKnobs(
            clients,
            window,
            backoffs,
            budgets,
            timeouts,
            agent_rps=agent_rps,
            latency_scale=self._latency_scale(timeouts),
            max_attempts=max(self.max_attempts),
        )

    def _latency_scale(self, timeouts: Sequence[StaticTimeoutController]) -> TimeDuration:
        """The longest an attempt of these clients may take, as far as that is known."""
        if self.attempt_timeout_ms is not None:
            return ms_to_ns(max(self.attempt_timeout_ms))
        bounds = [t.attempt_timeout or t.global_timeout for t in timeouts]
        known = [bound for bound in bounds if bound]
        return max(known) if known else ms_to_ns(1000)


class Agent(Protocol):
    """The agent's side of an episode, as the environment uses it.

    `RetryKnobs` is the stock one: the settings of a group of clients, measured
    at those clients and shown as `env.FEATURES`. An agent that sits elsewhere
    (the limiter in front of a service: `rl/limiter.py`) also offers
    `measure(window_s)`, which returns the requests that began in the window and
    the entries its scenario names in `features`, each in [0, 1].
    """

    @property
    def clients(self) -> Sequence[ClientRuntime]:
        """The clients whose requests the agent's settings govern."""
        ...

    def apply(self, values: Mapping[str, float]) -> Mapping[str, float] | None:
        """Put each named setting to its value.

        An agent that does not always take what it is given (a limiter whose
        quota rests while nobody retries, `QuotaGrid(rest=True)`) returns the
        settings in force, and the environment shows the agent those.
        """
        ...


@dataclass
class Episode:
    """One fresh simulation, wired and ready to run, as the environment needs to see it."""

    sim: Simulator
    #: Every arrival process, already driving its client.
    workloads: list[Workload]
    #: The agent's side: its settings and what it measures (`RetryKnobs` for an
    #: agent on clients).
    agent: Agent
    #: The clients whose finished requests are the reward: the scenario's roots.
    reward_clients: list[ClientRuntime]
    #: Every client of the simulation. The environment drops the records it has
    #: read from all of them, so one left out only costs memory and time.
    clients: list[ClientRuntime]
    #: Requests per second expected on the reward clients: the scale of the reward.
    expected_rps: float
    #: When the faults are, in seconds, for the episode summary.
    fault_windows: list[tuple[float, float]] = field(default_factory=list)
    #: The stations by name. For `info` only: the agent does not see them.
    services: dict[str, ServiceRuntime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.expected_rps) or self.expected_rps <= 0:
            raise ValueError(f"expected_rps must be a rate > 0, got {self.expected_rps}")
        if not self.reward_clients:
            raise ValueError("an episode needs at least one reward client")
        known = {id(client) for client in self.clients}
        for client in [*self.reward_clients, *self.agent.clients]:
            if id(client) not in known:
                raise ValueError(
                    "`clients` must list every client, the agent's and the reward's too"
                )


class Scenario(Protocol):
    """A situation an agent is put in, episode after episode. Must be picklable."""

    @property
    def knobs(self) -> tuple[Knob, ...]:
        """The settings the agent chooses, known before anything is built."""
        ...

    @property
    def horizon_s(self) -> float:
        """How much simulated time one episode covers, warm-up included."""
        ...

    def build(self, seed: int) -> Episode:
        """A fresh simulation for one episode, everything in it drawn from `seed`."""
        ...
