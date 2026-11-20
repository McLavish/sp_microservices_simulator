"""The scenarios that ship: one service, and two chains that share a bottleneck.

Both put the agent in front of ONE fault per episode, of a kind and at a time it
is not told: the station starts failing a share of its calls, or gets slower, or
the load rises. The kinds pull the retry policy in opposite directions, which is
what makes them a control problem. Measured with fixed settings (the numbers are
in `docs/rl.md`, and `tests/test_rl_scenarios.py` pins the property):

* when calls FAIL, every retry is a fresh chance: more attempts win;
* when the station is SLOW or the load SPIKES, attempts time out and a retry is
  one more call on a station that is already behind: fewer attempts win.

No fixed setting is best in both, so an agent that reads its clients' outcomes
(server failures or deadlines?) has something to gain.

An episode's fault, and the load, are drawn from the episode seed with a
generator of their own: the same seed gives the same episode, and changing what
the agent does changes neither the fault nor the arrivals.
"""

import random
from dataclasses import dataclass

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import TimeInterval, ms_to_ns, s_to_ns
from msim.faults import LatencyInjection, LoadSpike, PartialFailure
from msim.retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy
from msim.rl.scenario import Episode, Knob, RetryGrid
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.workload import Workload

#: The kinds of fault, by what `Fault.strength` means for each.
KINDS = ("failure", "slowdown", "spike")


@dataclass(frozen=True)
class Fault:
    """One fault: what happens, when, and how hard."""

    kind: str  # "failure" | "slowdown" | "spike"
    begin_s: float
    end_s: float
    #: failure: the share of calls the station fails. slowdown: milliseconds
    #: added to every service time. spike: what the arrival rate is multiplied by.
    strength: float

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"a fault is one of {', '.join(KINDS)}; got {self.kind!r}")
        if not 0 <= self.begin_s < self.end_s:
            raise ValueError(f"a fault needs 0 <= begin < end, got {self.begin_s} and {self.end_s}")

    @property
    def window(self) -> TimeInterval:
        return TimeInterval(s_to_ns(self.begin_s), s_to_ns(self.end_s))

    def on_station(self) -> tuple[list[LatencyInjection], list[PartialFailure]]:
        """The fault as the station's injections (both empty for a spike)."""
        if self.kind == "failure":
            return [], [PartialFailure(self.window, p_fail=self.strength)]
        if self.kind == "slowdown":
            return [LatencyInjection(self.window, add_latency=ms_to_ns(self.strength))], []
        return [], []

    def on_load(self) -> list[LoadSpike]:
        """The fault as the workloads' spikes (empty unless it is one)."""
        if self.kind == "spike":
            return [LoadSpike(self.window, rps_multiplier=self.strength)]
        return []


@dataclass(frozen=True)
class FaultRanges:
    """What an episode's fault is drawn from: a kind, then a time and a strength.

    `kinds` may repeat a kind to make it likelier, and may contain "none" for an
    episode in which nothing goes wrong. Each range is `(low, high)`, drawn
    uniformly.
    """

    kinds: tuple[str, ...] = (*KINDS, "none")
    onset_s: tuple[float, float] = (10.0, 35.0)
    length_s: tuple[float, float] = (10.0, 20.0)
    failure: tuple[float, float] = (0.2, 0.6)
    slowdown: tuple[float, float] = (15.0, 30.0)
    spike: tuple[float, float] = (1.6, 2.2)

    def __post_init__(self) -> None:
        if not self.kinds or any(kind not in (*KINDS, "none") for kind in self.kinds):
            raise ValueError(f"kinds are among {', '.join(KINDS)} and none; got {self.kinds}")

    def draw(self, rng: random.Random) -> Fault | None:
        kind = rng.choice(self.kinds)
        if kind == "none":
            return None
        begin = rng.uniform(*self.onset_s)
        length = rng.uniform(*self.length_s)
        return Fault(kind, begin, begin + length, rng.uniform(*getattr(self, kind)))

    def latest_end_s(self) -> float:
        """When the latest fault these ranges can draw is over (0 when they draw none)."""
        if all(kind == "none" for kind in self.kinds):
            return 0.0
        return self.onset_s[1] + self.length_s[1]


def _draws(scenario: object, seed: int) -> random.Random:
    """The generator an episode's fault and load come from: the scenario's own.

    Seeded with a string, so it shares nothing with the simulator's generator
    (`seed`) or the arrivals' (`seed` plus an offset), and gives the same
    episode on every machine.
    """
    return random.Random(f"{type(scenario).__name__}/{seed}")


def _check(scenario, faults: FaultRanges, fault: Fault | None) -> None:
    latest = faults.latest_end_s() if fault is None else fault.end_s
    if latest > scenario.horizon_s:
        raise ValueError(
            f"a fault could last until {latest} s, past the horizon of {scenario.horizon_s} s"
        )
    if not 0 < scenario.load[0] <= scenario.load[1]:
        raise ValueError(f"load is a range of factors > 0, got {scenario.load}")


def _relay(attempt_timeout_ms: float, attempts: int, backoff_ms: float) -> ClientConfig:
    """A client the agent does not drive: a fixed number of attempts, a fixed timeout."""
    retry = (
        NoRetryPolicy()
        if attempts <= 1
        else FixedBackoffRetryPolicy(max_attempts=attempts, delay=ms_to_ns(backoff_ms))
    )
    return ClientConfig(
        retry=retry,
        timeout_controller=StaticTimeoutController(
            global_timeout=None, attempt_timeout=ms_to_ns(attempt_timeout_ms)
        ),
    )


@dataclass(frozen=True)
class SingleService:
    """One client, driven by the agent, on one station.

    400 requests per second on 16 workers of 20 ms: a little under 60% of what
    the station can serve, so a slowdown or a spike takes it past capacity and
    retries decide whether it comes back.
    """

    rps: float = 400.0
    #: The episode's load is `rps` times a factor drawn from this range.
    load: tuple[float, float] = (0.9, 1.1)
    workers: int = 16
    latency_ms: float = 20.0
    latency_sigma: float = 0.5
    queue_capacity: int | None = 200
    grid: RetryGrid = RetryGrid()
    faults: FaultRanges = FaultRanges()
    #: A fault to use in every episode, instead of drawing one.
    fault: Fault | None = None
    horizon_s: float = 65.0

    def __post_init__(self) -> None:
        _check(self, self.faults, self.fault)

    @property
    def knobs(self) -> tuple[Knob, ...]:
        return self.grid.knobs()

    def build(self, seed: int) -> Episode:
        draws = _draws(self, seed)
        fault = self.fault or self.faults.draw(draws)
        rps = self.rps * draws.uniform(*self.load)
        injections, failures = fault.on_station() if fault else ([], [])

        sim = Simulator(seed=seed)
        station = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(self.latency_ms),
                latency_lognorm_sigma=self.latency_sigma,
                workers=self.workers,
                queue_capacity=self.queue_capacity,
                latency_injections=injections,
                partial_failures=failures,
            )
        ).bind()
        client = ClientRuntime(cfg=_relay(100.0, 1, 0.0), service=station)
        agent = self.grid.install([client], agent_rps=rps)
        workload = Workload(
            base_rps=rps,
            duration_s=self.horizon_s,
            load_spikes=fault.on_load() if fault else [],
            rng_seed=seed,
        )
        workload.drive(sim, client.start_request)
        return Episode(
            sim=sim,
            workloads=[workload],
            agent=agent,
            reward_clients=[client],
            clients=[client],
            expected_rps=rps,
            fault_windows=[(fault.begin_s, fault.end_s)] if fault else [],
            services={"svc": station},
        )


@dataclass(frozen=True)
class RetryStorm:
    """Two chains, A -> B -> C and X -> Y -> C, that share the bottleneck C.

    The agent drives BOTH callers of C (B's and Y's client): one policy on every
    caller of the bottleneck. The reward is the goodput of both chains, measured
    where the requests enter. The fault is on C, or a spike on both chains.

    The layout of `examples/multichain_retry_storm.py`, with other numbers. With
    the example's, nothing the callers of C do changes the outcome: the layers
    above them retry as well, so a failed call is recovered upstream or the
    storm is fed from upstream, and its fault leaves C beyond help. Here only
    the callers of C retry (`upstream_attempts = 1`) and C runs at about 60% of
    what it can serve, so their policy decides. `upstream_attempts > 1` brings
    the layered storm back, with retries at every hop.
    """

    #: Requests per second entering EACH chain.
    rps: float = 60.0
    load: tuple[float, float] = (0.9, 1.1)
    bottleneck_workers: int = 5
    bottleneck_queue: int | None = 100
    #: Attempts of the clients above the callers of C (the chains' entry and A's, X's).
    upstream_attempts: int = 1
    grid: RetryGrid = RetryGrid(backoff_ms=(0.0, 5.0, 20.0), default_backoff_ms=5.0)
    faults: FaultRanges = FaultRanges(failure=(0.3, 0.7), slowdown=(8.0, 20.0), spike=(1.3, 1.8))
    fault: Fault | None = None
    horizon_s: float = 65.0

    def __post_init__(self) -> None:
        _check(self, self.faults, self.fault)
        if self.upstream_attempts < 1:
            raise ValueError(f"upstream_attempts must be >= 1, got {self.upstream_attempts}")

    @property
    def knobs(self) -> tuple[Knob, ...]:
        return self.grid.knobs()

    def build(self, seed: int) -> Episode:
        draws = _draws(self, seed)
        fault = self.fault or self.faults.draw(draws)
        rps = self.rps * draws.uniform(*self.load)
        injections, failures = fault.on_station() if fault else ([], [])

        sim = Simulator(seed=seed)

        def node(name: str, latency_ms: float, sigma: float, workers: int, queue, **faulty):
            return ServiceNodeRuntime(
                service=ServiceRuntime(
                    cfg=ServiceConfig(
                        name=name,
                        latency_median=ms_to_ns(latency_ms),
                        latency_lognorm_sigma=sigma,
                        workers=workers,
                        queue_capacity=queue,
                        **faulty,
                    )
                ).bind()
            )

        c = node(
            "C",
            20,
            0.7,
            self.bottleneck_workers,
            self.bottleneck_queue,
            latency_injections=injections,
            partial_failures=failures,
        )
        a, b, x, y = (node(name, 12, 0.4, 16, 600) for name in "ABXY")

        # The agent's clients: the two callers of C. `install` gives them their policy.
        callers = [ClientRuntime(cfg=_relay(100.0, 1, 0.0), service=c) for _ in range(2)]
        agent = self.grid.install(callers, agent_rps=2 * rps)
        b.add_downstream(DownstreamEdge(client=callers[0], target=c))
        y.add_downstream(DownstreamEdge(client=callers[1], target=c))

        relays = []
        for parent, child in ((a, b), (x, y)):
            relay = ClientRuntime(cfg=_relay(450.0, self.upstream_attempts, 20.0), service=child)
            parent.add_downstream(DownstreamEdge(client=relay, target=child))
            relays.append(relay)

        entries, workloads = [], []
        for index, first in enumerate((a, x)):
            entry = ClientRuntime(cfg=_relay(900.0, self.upstream_attempts, 30.0), service=first)
            workload = Workload(
                base_rps=rps,
                duration_s=self.horizon_s,
                load_spikes=fault.on_load() if fault else [],
                rng_seed=seed + index,
            )
            workload.drive(sim, entry.start_request)
            entries.append(entry)
            workloads.append(workload)

        return Episode(
            sim=sim,
            workloads=workloads,
            agent=agent,
            reward_clients=entries,
            clients=[*entries, *relays, *callers],
            expected_rps=2 * rps,
            fault_windows=[(fault.begin_s, fault.end_s)] if fault else [],
            services={n.service.cfg.name: n.service for n in (a, b, c, x, y)},
        )
