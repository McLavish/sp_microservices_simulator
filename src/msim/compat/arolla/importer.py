"""Build an msim scenario from an Arolla scenario.

    scenario = build(parse(point.config))
    scenario.run()
    for client in scenario.clients: ...client.runtime.metrics()...

What their YAML means is written out here once, as the mapping to msim's parts.

**Services.** A service is a station (`ServiceRuntime`) with their latency,
workers, queue and faults. Its limiter, if it has one, becomes one admission
controller (`msim.admission`), used in the two places their runner uses it:

* in front of the station, as an `AdmissionGate`, for callers that retry
  themselves: the gate refuses the caller's retries on arrival;
* inside the service's OWN retry loop, as a `ControlledRetryPolicy`, when the
  service retries its handler itself.

**Clients.** A client entry with `replicas: n` is `n` clients named `name.i`,
each its own tenant. There are two kinds, as in their runner:

* a client WITH a `retry` policy runs it: its backoff, wrapped in its circuit
  breaker and its retry budget (`policies.py`). Its `timeout.attempt_ms` is a
  timer of its own that the server never hears about, so the client is built
  with `server_deadline="global"` and the attempts it gives up on live on at the
  server.
* a client WITHOUT one (and the single implicit client of a top-level
  `workload`) only relays: the service's own retry policy and timeout decide.
  It is built as a client that carries the service's policy and tells the
  server its deadline.

**Arrivals.** Every client has its own arrival stream, derived from the
workload's `rng_seed`, the position of its entry among the entries that target
the same service, and its replica index, so arms laid out on separate services
see the same arrivals (`arrival_seed`).
"""

import hashlib
from dataclasses import dataclass, field

from msim import faults
from msim.admission import (
    AdmissionController,
    AdmissionGate,
    AimdBudget,
    ControlledRetryPolicy,
    FailureRateGate,
    GoodputCoupledBudget,
    SuccessFundedBudget,
    TokenBucketBudget,
)
from msim.caller import ClientConfig, ClientRuntime, ServiceLike
from msim.compat.arolla import schema
from msim.compat.arolla.policies import (
    FirstAttemptFundedBudget,
    RetryBank,
    ThreeStateRetryBreaker,
    WindowRetryGate,
)
from msim.core import TimeInterval, ms_to_ns, s_to_ns
from msim.retry_policies import (
    ExponentialBackoffRetryPolicy,
    ExponentialBackoffWithJitterRetryPolicy,
    FixedBackoffRetryPolicy,
    JitterMode,
    NoRetryPolicy,
    RetryPolicy,
)
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.workload import Workload


class UnsupportedScenario(NotImplementedError):
    """A valid Arolla scenario that uses something this importer does not map yet."""


#: The name of the single client of a scenario written with a top-level `workload`.
IMPLICIT_CLIENT = "client"


@dataclass
class ImportedClient:
    """One client of the scenario (one replica of a client entry)."""

    name: str  # the entry's name, or `name.i` for a replica
    group: str  # the entry's name
    target: str  # the service it calls
    runtime: ClientRuntime
    workload: Workload


@dataclass
class Scenario:
    """An Arolla scenario, built and ready to run."""

    experiment: schema.Experiment
    sim: Simulator
    clients: list[ImportedClient]
    stations: dict[str, ServiceRuntime]
    #: The admission gate in front of a service, for the services that have one.
    gates: dict[str, AdmissionGate] = field(default_factory=dict)
    #: Each service's limiter as a controller, for the services that have one.
    controllers: dict[str, AdmissionController] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.experiment.name

    def run(self) -> None:
        """Drive every client's workload and run until nothing is left to do."""
        for client in self.clients:
            client.workload.drive(self.sim, client.runtime.start_request)
        self.sim.run()

    def group(self, name: str) -> list[ImportedClient]:
        return [client for client in self.clients if client.group == name]


def arrival_seed(rng_seed: int | None, position: int = 0, replica: int = 0) -> int | None:
    """Seed of one client's arrival stream (the rule of Arolla's builder).

    `position` is the rank of the client's entry among the entries that target
    the same service, `replica` its index among the entry's replicas. The first
    client of a service keeps `rng_seed`; entries on different services get the
    same streams.
    """
    if position == 0 and replica == 0:
        return rng_seed
    base = rng_seed if rng_seed is not None else 0
    digest = hashlib.md5(f"{base}:arrivals:{position}:{replica}".encode()).hexdigest()
    return int(digest, 16) & 0xFFFFFFFF


def _backoff(retry: schema.Retry | None, sim: Simulator) -> RetryPolicy:
    if retry is None or retry.type == "none":
        return NoRetryPolicy()
    if retry.type == "fixed":
        assert retry.delay_ms is not None
        return FixedBackoffRetryPolicy(
            max_attempts=retry.max_attempts, delay=ms_to_ns(retry.delay_ms)
        )
    assert retry.initial_delay_ms is not None and retry.max_delay_ms is not None
    if retry.type == "exponential":
        return ExponentialBackoffRetryPolicy(
            max_attempts=retry.max_attempts,
            initial_delay=ms_to_ns(retry.initial_delay_ms),
            max_delay=ms_to_ns(retry.max_delay_ms),
        )
    return ExponentialBackoffWithJitterRetryPolicy(
        max_attempts=retry.max_attempts,
        initial_delay=ms_to_ns(retry.initial_delay_ms),
        max_delay=ms_to_ns(retry.max_delay_ms),
        rng=sim.rng(),
        jitter_mode=JitterMode.EQUAL if retry.jitter_mode == "equal" else JitterMode.FULL,
    )


def _controller(service: schema.Service) -> AdmissionController | None:
    """The service's limiter as an admission controller, if it is one."""
    limiter = service.limiter
    if limiter is None:
        return None
    key, cfg = limiter
    if key == "arolla_retry_budget":
        return GoodputCoupledBudget(
            alpha=cfg.alpha,
            beta_down=cfg.beta_down,
            beta_up=cfg.beta_up,
            window=ms_to_ns(cfg.window_ms),
            success_rate_threshold=cfg.success_rate_threshold,
            success_rate_beta=cfg.success_rate_beta,
            max_retry_ratio=cfg.max_retry_ratio,
        )
    if key == "aimd_global_retry_budget":
        return AimdBudget(
            min_rate=cfg.min_rps,
            max_rate=cfg.max_rps,
            initial_rate=cfg.initial_rps,
            burst=cfg.max_burst,
            additive_step=cfg.additive_step,
            decrease_factor=cfg.decrease_factor,
            window=ms_to_ns(cfg.window_ms),
            failure_threshold=cfg.failure_threshold,
        )
    if key == "global_retry_budget":
        return TokenBucketBudget(rate=cfg.target_rps, burst=cfg.max_burst)
    if key == "retry_budget":
        return SuccessFundedBudget(ratio=cfg.budget_ratio, max_retries=cfg.max_retries)
    if key == "circuit_breaker" and cfg.type == "time_based":
        return FailureRateGate(
            failure_threshold=cfg.failure_threshold,
            success_threshold=cfg.success_threshold,
            min_requests=cfg.min_requests,
            window=ms_to_ns(cfg.window_duration_ms),
            open_delay=ms_to_ns(cfg.half_open_delay_ms),
        )
    what = f"circuit_breaker of type {cfg.type}" if key == "circuit_breaker" else key
    raise UnsupportedScenario(
        f"service {service.name!r}: a {what} on a service is not mapped yet "
        "(their runner applies it to the service's own retries only)"
    )


def _client_policy(
    client: schema.Client, sim: Simulator, banks: dict[str, RetryBank]
) -> RetryPolicy:
    """A client's own policy stack: backoff, then its breaker, then its budget."""
    policy = _backoff(client.retry, sim)
    breaker = client.circuit_breaker
    if breaker is not None and breaker.type != "count_based":  # count_based: ignored, warned
        window = ms_to_ns(breaker.window_duration_ms or 5000)
        quorum = breaker.min_requests or 100
        if breaker.type == "retry_circuit_breaker":
            policy = WindowRetryGate(
                underlying_strategy=policy,
                failure_threshold=breaker.failure_threshold,
                window=window,
                min_window_size=quorum,
            )
        else:
            policy = ThreeStateRetryBreaker(
                underlying_strategy=policy,
                failure_threshold=breaker.failure_threshold,
                window=window,
                min_window_size=quorum,
                open_wait=ms_to_ns(breaker.half_open_delay_ms),
            )
    budget = client.retry_budget
    if budget is not None:
        bank = banks.get(budget.shared_budget_id) if budget.shared_budget_id else None
        if bank is None:
            bank = RetryBank(
                budget_ratio=budget.budget_ratio,
                max_retries=budget.max_retries,
                min_retries_per_sec=budget.min_retries_per_sec,
            )
            if budget.shared_budget_id:
                banks[budget.shared_budget_id] = bank
        policy = FirstAttemptFundedBudget(bank=bank, underlying_strategy=policy)
    return policy


def _ms(value: float | None) -> int | None:
    return None if value is None else ms_to_ns(value)


def _station(service: schema.Service) -> ServiceRuntime:
    return ServiceRuntime(
        cfg=ServiceConfig(
            name=service.name,
            latency_median=ms_to_ns(service.latency.median_ms),
            latency_lognorm_sigma=service.latency.lognorm_sigma,
            workers=service.workers,
            queue_capacity=service.queue_capacity,
            latency_injections=[
                faults.LatencyInjection(
                    TimeInterval(s_to_ns(f.start_s), s_to_ns(f.end_s)),
                    add_latency=ms_to_ns(f.add_latency_ms),
                    multiplier=f.multiplier,
                )
                for f in service.latency_injections
            ],
            partial_failures=[
                faults.PartialFailure(
                    TimeInterval(s_to_ns(f.start_s), s_to_ns(f.end_s)), p_fail=f.p_fail
                )
                for f in service.partial_failures
            ],
        )
    ).bind()


def build(
    experiment: schema.Experiment,
    *,
    seed: int | None = None,
    arrivals: int | None = None,
    duration_s: float | None = None,
) -> Scenario:
    """Build the msim objects of one concrete Arolla scenario.

    `seed` replaces the scenario's simulator seed and `arrivals` every
    workload's `rng_seed` (the per-client streams are still derived from it),
    which is how one scenario is run many times. `duration_s` cuts every
    workload short, for a scenario too large to run whole.
    """
    for service in experiment.services:
        if service.dependencies:
            raise UnsupportedScenario(
                f"service {service.name!r} has dependencies, which are not mapped yet"
            )

    sim = Simulator(seed=experiment.seed if seed is None else seed)
    stations = {service.name: _station(service) for service in experiment.services}
    controllers = {}
    for service in experiment.services:
        controller = _controller(service)
        if controller is not None:
            controllers[service.name] = controller
    # What a caller with retries of its own submits to: the gate, where there is one.
    gates = {
        name: AdmissionGate(stations[name], [controller])
        for name, controller in controllers.items()
    }

    if experiment.clients:
        entries = list(experiment.clients)
    else:
        assert experiment.workload is not None
        entries = [schema.Client(name=IMPLICIT_CLIENT, workload=experiment.workload)]

    banks: dict[str, RetryBank] = {}
    position_on: dict[str, int] = {}
    clients: list[ImportedClient] = []
    for entry in entries:
        service = experiment.service(entry.target_service or experiment.entry.name)
        position = position_on.get(service.name, 0)
        position_on[service.name] = position + 1
        if entry.e2e_retry_budget is not None:
            raise UnsupportedScenario(f"client {entry.name!r}: e2e_retry_budget is not mapped yet")
        for replica in range(entry.replicas):
            name = entry.name if entry.replicas <= 1 else f"{entry.name}.{replica}"
            runtime = _client(entry, name, service, sim, stations, gates, controllers, banks)
            spec = entry.workload
            base_seed = spec.rng_seed if arrivals is None else arrivals
            workload = Workload(
                base_rps=spec.base_rps,
                duration_s=spec.duration_s
                if duration_s is None
                else min(duration_s, spec.duration_s),
                load_spikes=[
                    faults.LoadSpike(
                        TimeInterval(s_to_ns(s.start_s), s_to_ns(s.end_s)),
                        rps_multiplier=s.rps_multiplier,
                    )
                    for s in spec.load_spikes
                ],
                rng_seed=arrival_seed(base_seed, position, replica),
            )
            clients.append(
                ImportedClient(
                    name=name,
                    group=entry.name,
                    target=service.name,
                    runtime=runtime,
                    workload=workload,
                )
            )
    return Scenario(
        experiment=experiment,
        sim=sim,
        clients=clients,
        stations=stations,
        gates=gates,
        controllers=controllers,
    )


def _client(
    entry: schema.Client,
    name: str,
    service: schema.Service,
    sim: Simulator,
    stations: dict[str, ServiceRuntime],
    gates: dict[str, AdmissionGate],
    controllers: dict[str, AdmissionController],
    banks: dict[str, RetryBank],
) -> ClientRuntime:
    service_timeout = service.timeout or schema.Timeout()
    if entry.retry is None:
        # A relay: the service's own retry loop and timeouts are the client's.
        # Its attempts ARE the service's attempts, so they go straight to the
        # station; the limiter gates them from inside the loop.
        if entry.timeout is not None and entry.timeout.attempt_ms is not None:
            raise UnsupportedScenario(
                f"client {entry.name!r}: an attempt timeout on a client without a retry "
                "policy (a second timeout around the service's own) is not mapped yet"
            )
        policy = _backoff(service.retry, sim)
        controller = controllers.get(service.name)
        if controller is not None:
            policy = ControlledRetryPolicy([controller], policy)
        global_ms = (
            entry.timeout.global_ms if entry.timeout is not None else service_timeout.global_ms
        )
        target: ServiceLike = stations[service.name]
        return ClientRuntime(
            cfg=ClientConfig(
                retry=policy,
                timeout_controller=StaticTimeoutController(
                    global_timeout=_ms(global_ms), attempt_timeout=_ms(service_timeout.attempt_ms)
                ),
                tenant=name,
            ),
            service=target,
        )

    # A client with a policy of its own.
    if service.retry is not None and service.retry.type != "none":
        raise UnsupportedScenario(
            f"client {entry.name!r} has a retry policy and so has the service it calls "
            f"({service.name!r}): two retry loops on one hop are not mapped yet"
        )
    target = gates.get(service.name, stations[service.name])
    policy = _client_policy(entry, sim, banks)
    if entry.timeout is None:
        # No timeout of its own: the request's global timeout is the service's,
        # and the service's attempt timeout bounds each call, enforced there.
        timeouts = StaticTimeoutController(
            global_timeout=_ms(service_timeout.global_ms),
            attempt_timeout=_ms(service_timeout.attempt_ms),
        )
        mode = "attempt"
    elif entry.timeout.attempt_ms is None:
        timeouts = StaticTimeoutController(
            global_timeout=_ms(entry.timeout.global_ms),
            attempt_timeout=_ms(service_timeout.attempt_ms),
        )
        mode = "attempt"
    else:
        if service_timeout.attempt_ms is not None:
            raise UnsupportedScenario(
                f"client {entry.name!r} and service {service.name!r} both set an attempt "
                "timeout: two timeouts on one hop are not mapped yet"
            )
        # The client's own timer: the server is told the global deadline only.
        timeouts = StaticTimeoutController(
            global_timeout=_ms(entry.timeout.global_ms),
            attempt_timeout=_ms(entry.timeout.attempt_ms),
        )
        mode = "global"
    return ClientRuntime(
        cfg=ClientConfig(
            retry=policy, timeout_controller=timeouts, tenant=name, server_deadline=mode
        ),
        service=target,
    )
