"""An Arolla scenario file as a situation for the agent.

    scenario = ArollaScenario.from_file(".../s03__thundering_herd.yaml", agent="three_retries")
    env = RetryEnv(scenario)

Arolla (https://github.com/eth-easl/Arolla) writes a scenario as arms laid side
by side: one client entry per retry policy, often each on a service of its own,
all hit by the same fault. `msim.compat.arolla` builds such a file as it stands.
Here the agent takes ONE arm over:

* from the file comes the situation: the services, their faults and server-side
  retry limiters, the load, the timeouts, the replicas;
* the agent's clients get the settings' policy stack in place of the policy the
  file gives them (`RetryGrid.install`), and are named by `agent`: client
  entries of the file (all replicas of each), or services that retry their own
  calls (their retry loop is taken over);
* only the part of the file those clients can interact with is simulated: the
  services they reach, and every other client of those services. The other arms
  are not built. `agent_part` is that cut.

The reward is counted on every client that is left, the agent's or not. The
default levels come from the file (`file_grid`), so the default setting is the
arm as the file has it wherever a fixed backoff can say it.

Needs PyYAML to read a file: the `arolla` extra.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from msim.admission import ControlledRetryPolicy
from msim.caller import ClientRuntime
from msim.compat.arolla import Experiment, UnsupportedScenario, parse, read, schema
from msim.compat.arolla import build as build_file
from msim.compat.arolla.importer import IMPLICIT_CLIENT
from msim.compat.arolla.importer import Scenario as Built
from msim.retry_policies import RetryAllowancePolicy, policy_layers
from msim.rl.scenario import Episode, Knob, RetryGrid
from msim.rl.summary import of_records


def _entries(experiment: Experiment) -> tuple[schema.Client, ...]:
    """The client entries, the implicit one of a top-level `workload` included."""
    if experiment.clients:
        return experiment.clients
    assert experiment.workload is not None
    return (schema.Client(name=IMPLICIT_CLIENT, workload=experiment.workload),)


def _reach(experiment: Experiment, service: str) -> set[str]:
    """`service` and everything it calls, directly or not."""
    by_name = {s.name: s for s in experiment.services}
    seen: set[str] = set()
    pending = [service]
    while pending:
        name = pending.pop()
        if name not in seen:
            seen.add(name)
            pending.extend(d.service for d in by_name[name].dependencies)
    return seen


def _target(experiment: Experiment, entry: schema.Client) -> str:
    return entry.target_service or experiment.entry.name


def agent_part(experiment: Experiment, agent: Sequence[str]) -> Experiment:
    """The part of a scenario the agent's clients can interact with.

    The services the agent's clients reach, every service that reaches one of
    those, what THEY reach, and so on; and the client entries that call any of
    them. For a file of arms on separate services that is the agent's arm; for
    tenants on one shared service it is the whole file. What is cut shares no
    station with what is kept, so it could only have changed the kept part
    through the simulator's one random stream.
    """
    entries = _entries(experiment)
    targets = {entry.name: _target(experiment, entry) for entry in entries}
    services = {s.name for s in experiment.services}
    keep: set[str] = set()
    for name in agent:
        if name in targets:
            keep |= _reach(experiment, targets[name])
        elif name in services:
            keep |= _reach(experiment, name)
        else:
            raise ValueError(
                f"{name!r} is neither a client entry {sorted(targets)} "
                f"nor a service {sorted(services)} of {experiment.name!r}"
            )
    grown = True
    while grown:
        grown = False
        for start in (*targets.values(), *services):
            reach = _reach(experiment, start)
            if reach & keep and not reach <= keep:
                keep |= reach
                grown = True
    if not experiment.clients and experiment.entry.name not in keep:
        raise ValueError(
            f"the only client of {experiment.name!r} calls {experiment.entry.name!r}, "
            f"which {list(agent)} never interact with"
        )
    clients = tuple(
        # The target is written out: "the first service" may not be the first any more.
        replace(entry, target_service=targets[entry.name])
        for entry in experiment.clients
        if targets[entry.name] in keep
    )
    return replace(
        experiment,
        services=tuple(s for s in experiment.services if s.name in keep),
        clients=clients,
    )


def _file_policy(
    experiment: Experiment, name: str
) -> tuple[schema.Retry | None, schema.Timeout | None]:
    """The retry policy and timeouts the file gives the agent called `name`."""
    for entry in _entries(experiment):
        if entry.name == name:
            service = experiment.service(_target(experiment, entry))
            # A client entry without a policy of its own relays the service's.
            return entry.retry or service.retry, entry.timeout or service.timeout
    service = experiment.service(name)
    return service.retry, service.timeout


def file_grid(experiment: Experiment, agent: Sequence[str]) -> RetryGrid:
    """Levels for the agent's settings, with the file's own values as the defaults.

    Max attempts and backoff: the usual levels plus what the file says. Attempt
    timeout: half, once and twice the file's, when the agent's clients share
    one; no timeout knob when they differ or one has none (each keeps its own).
    A retry budget at its most permissive, since the arms taken over have none.
    """
    retries, timeouts = zip(*(_file_policy(experiment, name) for name in agent), strict=True)
    attempts = {
        1 if retry is None or retry.type == "none" else retry.max_attempts for retry in retries
    }
    delays = {
        0.0 if retry is None else float(retry.delay_ms or retry.initial_delay_ms or 0.0)
        for retry in retries
    }
    own = {None if timeout is None else timeout.attempt_ms for timeout in timeouts}
    shared = own.pop() if len(own) == 1 else None
    return RetryGrid(
        max_attempts=tuple(sorted({1, 2, 3, 4} | attempts)),
        default_max_attempts=max(attempts),
        backoff_ms=tuple(sorted({0.0, 10.0, 50.0} | delays)),
        default_backoff_ms=min(delays),
        attempt_timeout_ms=None if shared is None else (shared / 2, float(shared), shared * 2.0),
        default_attempt_timeout_ms=None if shared is None else float(shared),
        default_budget_ratio=1.0,
    )


@dataclass(frozen=True)
class ArollaScenario:
    """One arm of an Arolla scenario, handed to the agent. Make it with `from_file`."""

    #: The scenario, already cut down to the agent's part.
    experiment: Experiment
    #: Who the agent drives: client entries, or services that retry their own calls.
    agent: tuple[str, ...]
    grid: RetryGrid

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        agent: str | Sequence[str],
        *,
        point: int = 0,
        grid: RetryGrid | None = None,
    ) -> "ArollaScenario":
        """The scenario of `path` (its sweep point `point`) with `agent` taken over.

        `agent` names one or several client entries or services. `grid`
        replaces the levels `file_grid` would derive.
        """
        names = _names(agent)
        experiment = agent_part(parse(read(path)[point].config), names)
        scenario = cls(experiment, names, grid or file_grid(experiment, names))
        scenario.build(
            experiment.seed
        )  # a scenario that cannot be built fails here, not in a worker
        return scenario

    @property
    def knobs(self) -> tuple[Knob, ...]:
        return self.grid.knobs()

    @property
    def horizon_s(self) -> float:
        """The length the file gives its workloads."""
        return horizon_s(self.experiment)

    def fault_windows(self) -> list[tuple[float, float]]:
        """When the file's faults are: failures, injected latency, load spikes."""
        return fault_windows(self.experiment)

    def build(self, seed: int) -> Episode:
        experiment = self.experiment
        built = build_file(experiment, seed=seed, arrivals=seed)
        runtimes = [rt for name in self.agent for rt in _runtimes(experiment, built, name)]
        for runtime in runtimes:
            for layer in policy_layers(runtime.cfg.retry):
                if isinstance(layer, ControlledRetryPolicy | RetryAllowancePolicy):
                    raise UnsupportedScenario(
                        f"an agent client of {experiment.name!r} carries more than a retry policy "
                        f"in its policy stack ({type(layer).__name__}: the service's own limiter, "
                        "or a retry allowance); taking the stack over would drop it"
                    )
        rps = sum(_rps(experiment, name) for name in self.agent)
        agent = self.grid.install(runtimes, agent_rps=rps)
        for client in built.clients:
            client.workload.drive(built.sim, client.runtime.start_request)
        roots = [client.runtime for client in built.clients]
        return Episode(
            sim=built.sim,
            workloads=[client.workload for client in built.clients],
            agent=agent,
            reward_clients=roots,
            clients=[*roots, *built.loops.values()],
            expected_rps=expected_rps(experiment),
            fault_windows=fault_windows(experiment),
            services=dict(built.stations),
        )


def horizon_s(experiment: Experiment) -> float:
    """The length the file gives its workloads."""
    return float(max(entry.workload.duration_s for entry in _entries(experiment)))


def expected_rps(experiment: Experiment) -> float:
    """Requests per second the file's clients send together."""
    return sum(entry.workload.base_rps * entry.replicas for entry in _entries(experiment))


def fault_windows(experiment: Experiment) -> list[tuple[float, float]]:
    """When the file's faults are: failures, injected latency, load spikes."""
    windows = {
        (f.start_s, f.end_s)
        for service in experiment.services
        for f in service.partial_failures
        if f.p_fail > 0
    }
    windows |= {
        (f.start_s, f.end_s)
        for service in experiment.services
        for f in service.latency_injections
        if f.add_latency_ms > 0 or f.multiplier > 1
    }
    windows |= {
        (spike.start_s, spike.end_s)
        for entry in _entries(experiment)
        for spike in entry.workload.load_spikes
        if spike.rps_multiplier != 1
    }
    return sorted(windows)


def _rps(experiment: Experiment, name: str) -> float:
    """Requests per second the client entry or service loop called `name` starts when all is well."""
    entries = _entries(experiment)
    own = [entry for entry in entries if entry.name == name]
    if not own:  # a service's loop: one call for each request that reaches the service
        own = [e for e in entries if name in _reach(experiment, _target(experiment, e))]
    return sum(entry.workload.base_rps * entry.replicas for entry in own)


def _runtimes(experiment: Experiment, built: Built, name: str) -> list[ClientRuntime]:
    """The clients that the entry or service called `name` is, in the built scenario."""
    replicas = [client.runtime for client in built.group(name)]
    if replicas:
        return replicas
    loop = built.loops.get(name)
    if loop is None:
        raise UnsupportedScenario(
            f"{name!r} is a service with no retry loop of its own in {experiment.name!r}: "
            "its callers hold the policy. Name a client entry instead"
        )
    return [loop]


def as_written(
    path: str | Path,
    seeds: Iterable[int],
    *,
    part_of: str | Sequence[str] | None = None,
    effort: str | Sequence[str] | None = None,
    point: int = 0,
    warmup_s: float = 5.0,
    step_s: float = 1.0,
) -> list[dict[str, float]]:
    """The file run as it is written, with nobody taken over: one summary per seed.

    What a learned policy on this file is put next to: the file's own clients,
    behind whatever the file puts in front of its services. Each episode is
    built and seeded as `ArollaScenario` does it and scored by the same
    definitions (`msim.rl.summary`), on every client of the file.

    `part_of` keeps only the part of the file the named clients interact with,
    as `from_file` does for an agent. `effort` names the client entries or
    service loops whose attempts per request and retry efficiency are reported
    (every client entry unless given).
    """
    experiment = parse(read(path)[point].config)
    if part_of is not None:
        experiment = agent_part(experiment, _names(part_of))
    episodes = []
    for seed in seeds:
        built = build_file(experiment, seed=seed, arrivals=seed)
        built.run()
        spent = None
        if effort is not None:
            spent = [rt for name in _names(effort) for rt in _runtimes(experiment, built, name)]
        scored = of_records(
            [client.runtime for client in built.clients],
            expected_rps=expected_rps(experiment),
            fault_windows=fault_windows(experiment),
            horizon_s=horizon_s(experiment),
            warmup_s=warmup_s,
            step_s=step_s,
            effort_clients=spent,
        )
        episodes.append({"seed": float(seed), **scored})
    return episodes


def _names(names: str | Sequence[str]) -> tuple[str, ...]:
    return (names,) if isinstance(names, str) else tuple(names)
