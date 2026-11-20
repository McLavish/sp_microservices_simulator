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

**Dependencies.** A service with `dependencies` is a chain node that holds its
worker from the start, makes its calls first and does its own work last
(`local_work="last"`): `sequential` dependencies are one call wave each,
`parallel` ones share a wave that the service waits out (`wave_join="wait_all"`),
and an `optional` dependency is an optional edge. What a service calls is the
dependency's own retry LOOP -- a client that carries the dependency's retry
policy and attempt timeout, one per service, shared by all its callers -- so the
caller gets one answer per call, when the dependency has finished retrying.

**Two loops on one hop.** A client with a retry policy in front of a service
that retries too, or that has an attempt timeout of its own, calls that same
loop through a `RetryLayer`: the client's policy and timer on the outside, the
service's on the inside.

**Admission.** A service's limiter refuses the RETRIES it is sent: a call whose
caller is on its second or later attempt. For a client that retries and calls a
plain service, that is an `AdmissionGate` in front of the station. Where the
service runs a loop of its own (it retries, has its own attempt timeout, or is
called by another service) the call is put to the limiter as it arrives at the
loop (`AdmittingLoop`), and the loop reports every attempt it makes to the same
controller.

**The rest.** A `count_based` breaker, a `retry_circuit_breaker` and a rate
limiter on a service gate that service's OWN retries only (`policies.py`). A
client's `e2e_retry_budget` is a retry allowance on its requests
(`ClientConfig.retry_allowance`) that every service's loop draws on; the client
itself, as in their runner, does not.

**Arrivals.** Every client has its own arrival stream, derived from the
workload's `rng_seed`, the position of its entry among the entries that target
the same service, and its replica index, so arms laid out on separate services
see the same arrivals (`arrival_seed`).
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from msim import faults
from msim.admission import (
    AdmissionController,
    AdmissionCounts,
    AdmissionGate,
    AdmissionRequest,
    AimdBudget,
    ControlledRetryPolicy,
    FailureRateGate,
    GoodputCoupledBudget,
    SuccessFundedBudget,
    TokenBucketBudget,
)
from msim.caller import ClientConfig, ClientRuntime, RetryLayer, ServiceLike
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.compat.arolla import schema
from msim.compat.arolla.policies import (
    CountRetryBreaker,
    FirstAttemptFundedBudget,
    LeakyRetryGate,
    RetryBank,
    ThreeStateRetryBreaker,
    WindowRetryGate,
)
from msim.core import (
    DropReason,
    Request,
    RequestContext,
    RootRequest,
    TimeInterval,
    TimePoint,
    ms_to_ns,
    s_to_ns,
)
from msim.retry_policies import (
    BurstyRateLimiterPolicy,
    ExponentialBackoffRetryPolicy,
    ExponentialBackoffWithJitterRetryPolicy,
    FixedBackoffRetryPolicy,
    FixedWindowBurstyLimiterPolicy,
    JitterMode,
    NoRetryPolicy,
    RetryAllowancePolicy,
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
    #: The chain node of every service that calls others or is called by one.
    nodes: dict[str, ServiceNodeRuntime] = field(default_factory=dict)
    #: A service's own retry loop as the client that runs it, where something
    #: calls the service through it (another service, or a client that retries).
    loops: dict[str, ClientRuntime] = field(default_factory=dict)

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

    def admission_counts(self, client: ImportedClient) -> AdmissionCounts | None:
        """What the limiter of the service a client calls counted of its calls.

        The retries the client sent and how many were let in, wherever the
        mapping put the door: the gate in front of a plain service, or the
        service's own loop where it has one. `None` when the service has no
        limiter. A relay's counts are empty: its retries are the service's own,
        gated inside the loop and never put to the limiter on arrival.
        """
        if client.target not in self.controllers:
            return None
        door: object = client.runtime.service
        if isinstance(door, RetryLayer):
            door = door.client  # the service's own loop, which admits each call
        if isinstance(door, AdmissionGate | AdmittingLoop):
            return door.counts.get(client.name) or AdmissionCounts()
        return AdmissionCounts()


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
    # A count_based breaker, a retry_circuit_breaker and the rate limiters gate
    # the service's own retries only: `_own_retry_gate`.
    return None


def _own_retry_gate(service: schema.Service, policy: RetryPolicy) -> RetryPolicy:
    """`policy` inside the service's retry-only limiter, when it has one."""
    limiter = service.limiter
    if limiter is None:
        return policy
    key, cfg = limiter
    if key == "rate_limiter":
        period = ms_to_ns(cfg.period_ms)
        if cfg.type == "leaky":
            return LeakyRetryGate(
                underlying_strategy=policy,
                max_requests=cfg.max_requests,
                max_attempts=cfg.max_attempts or cfg.max_requests,
                period=period,
            )
        if cfg.type == "bursty":
            return BurstyRateLimiterPolicy(
                max_requests=cfg.max_requests,
                refill_rate=cfg.refill_rate or 1,
                period=period,
                underlying_strategy=policy,
            )
        return FixedWindowBurstyLimiterPolicy(
            max_requests=cfg.max_requests, period=period, underlying_strategy=policy
        )
    if key == "circuit_breaker" and cfg.type == "count_based":
        return CountRetryBreaker(
            underlying_strategy=policy,
            failure_threshold=cfg.failure_threshold,
            failure_window=cfg.failure_window_size,
            success_threshold=cfg.success_threshold,
            success_window=cfg.success_window_size,
            open_delay=ms_to_ns(cfg.half_open_delay_ms),
        )
    if key == "circuit_breaker" and cfg.type == "retry_circuit_breaker":
        return WindowRetryGate(
            underlying_strategy=policy,
            failure_threshold=cfg.failure_threshold,
            window=ms_to_ns(cfg.window_duration_ms),
            min_window_size=cfg.min_requests or 100,
        )
    return policy


@dataclass
class AdmittingLoop(ClientRuntime):
    """A service's retry loop that puts each CALL to the service's limiter first.

    Arolla's services check their limiter when a call arrives, before the
    service's own retry loop starts, and only for a call that is itself a retry
    of its caller. A refused call never reaches the loop: the caller gets one
    failed attempt, `REJECTED`. `AdmissionGate` does this in front of a plain
    service; this is the same decision in front of a LOOP, where the thing that
    arrives is a request and not an attempt. The loop reports its attempts to
    the same controllers from inside (`ControlledRetryPolicy`), so nothing is
    observed here.
    """

    controllers: Sequence[AdmissionController] = ()
    #: Per tenant: the calls that arrived as retries, and how many were let in.
    counts: dict[str | None, AdmissionCounts] = field(default_factory=dict)

    def start_request(
        self,
        sim: Simulator,
        on_done: Callable[[RootRequest], None] | None = None,
        override_global_deadline: TimePoint | None = None,
        *,
        ctx: RequestContext | None = None,
    ):
        if ctx is None:
            return super().start_request(sim, on_done, override_global_deadline)
        counts = self.counts.get(ctx.tenant)
        if counts is None:
            counts = self.counts[ctx.tenant] = AdmissionCounts()
        counts.attempts += 1
        if ctx.attempt > 1:  # the caller's own attempt is a retry
            counts.retries_asked += 1
            request = AdmissionRequest(
                now=sim.timestep, tenant=ctx.tenant, attempt=ctx.attempt, is_retry=True
            )
            if not all(controller.check(request) for controller in self.controllers):
                counts.rejected += 1
                # Refused on arrival: one attempt that never reached anything, as
                # a client books a request its own breaker sheds.
                root = RootRequest(global_deadline=override_global_deadline, on_done=on_done)
                self.attempts_total += 1
                root.add_attempt(
                    Request(
                        service=self.service.cfg.name,
                        interval=TimeInterval(begin=sim.timestep, end=sim.timestep),
                        deadline=override_global_deadline,
                        success=False,
                        drop_reason=DropReason.REJECTED,
                        queue_size_at_end=None,
                    )
                )
                self._complete_root(root, sim.timestep)
                return root
            for controller in self.controllers:
                controller.commit(request)
            counts.retries_admitted += 1
        return super().start_request(sim, on_done, override_global_deadline, ctx=ctx)


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
    builder = _Builder(experiment, Simulator(seed=experiment.seed if seed is None else seed))

    if experiment.clients:
        entries = list(experiment.clients)
    else:
        assert experiment.workload is not None
        entries = [schema.Client(name=IMPLICIT_CLIENT, workload=experiment.workload)]

    position_on: dict[str, int] = {}
    clients: list[ImportedClient] = []
    for entry in entries:
        service = experiment.service(entry.target_service or experiment.entry.name)
        position = position_on.get(service.name, 0)
        position_on[service.name] = position + 1
        for replica in range(entry.replicas):
            name = entry.name if entry.replicas <= 1 else f"{entry.name}.{replica}"
            runtime = builder.client(entry, name, service)
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
        sim=builder.sim,
        clients=clients,
        stations=builder.stations,
        gates=builder.gates,
        controllers=builder.controllers,
        nodes=builder.nodes,
        loops=builder.loops,
    )


class _Builder:
    """The services of one scenario, built once and shared by everything that calls them."""

    def __init__(self, experiment: schema.Experiment, sim: Simulator) -> None:
        self.experiment = experiment
        self.sim = sim
        self.stations = {service.name: _station(service) for service in experiment.services}
        self.controllers: dict[str, AdmissionController] = {}
        for service in experiment.services:
            controller = _controller(service)
            if controller is not None:
                self.controllers[service.name] = controller
        #: Whether requests carry a retry allowance for the services' loops to
        #: draw on (some client sets `e2e_retry_budget`).
        self.allowance = any(c.e2e_retry_budget is not None for c in experiment.clients)
        self.banks: dict[str, RetryBank] = {}
        #: A service's own retry loop, as the client that runs it. One per
        #: service, shared by every caller: the loop is the service's.
        self.loops: dict[str, ClientRuntime] = {}
        self.layers: dict[str, RetryLayer] = {}
        #: The chain node of every service that calls others or is called by one.
        self.nodes: dict[str, ServiceNodeRuntime] = {}
        self._graph()
        #: What a client with retries of its own submits to, where the service
        #: has a limiter and no loop of its own: the gate that turns its retries away.
        self.gates = {
            service.name: AdmissionGate(self.target(service), [self.controllers[service.name]])
            for service in experiment.services
            if service.name in self.controllers
        }

    def _graph(self) -> None:
        """Build a node for every service in a call graph, callees before callers."""
        services = {service.name: service for service in self.experiment.services}
        called = {d.service for service in services.values() for d in service.dependencies}
        pending = {name for name, service in services.items() if service.dependencies} | called
        while pending:
            ready = sorted(
                name
                for name in pending
                if all(d.service in self.nodes for d in services[name].dependencies)
            )
            for name in ready:
                service = services[name]
                if not service.dependencies:
                    self.nodes[name] = ServiceNodeRuntime(service=self.stations[name])
                    continue
                # Their dependency model: the worker is held from the start, the
                # dependencies are called first -- one after the other, or all
                # together with the service waiting for every answer -- and the
                # service's own work comes last.
                parallel = service.dependency_call_pattern == "parallel"
                node = ServiceNodeRuntime(
                    service=self.stations[name],
                    hold_worker_during_fanout=True,
                    local_work="last",
                    wave_join="wait_all" if parallel else "short_circuit",
                )
                for index, dependency in enumerate(service.dependencies):
                    node.add_downstream(
                        DownstreamEdge(
                            client=self.loop(services[dependency.service]),
                            target=self.nodes[dependency.service],
                            wave=0 if parallel else index,
                            optional=dependency.optional,
                        )
                    )
                self.nodes[name] = node
            pending -= set(ready)

    def target(self, service: schema.Service) -> ServiceLike:
        """What one attempt AT the service is submitted to: its node, or its station."""
        return self.nodes.get(service.name, self.stations[service.name])

    def loop_policy(self, service: schema.Service) -> RetryPolicy:
        """The service's own retry policy, gated by its limiter from inside."""
        policy = _backoff(service.retry, self.sim)
        controller = self.controllers.get(service.name)
        if controller is not None:
            policy = ControlledRetryPolicy([controller], policy)
        elif service.retry is not None and service.retry.type != "none":
            # A limiter that gates only the service's own retries: nothing to
            # gate on a service that does not retry (the schema says so).
            policy = _own_retry_gate(service, policy)
        if self.allowance:
            # Their end-to-end budget: every retry a service issues takes a unit
            # of the request's allowance, and none is issued without one.
            policy = RetryAllowancePolicy(policy)
        return policy

    def loop(self, service: schema.Service) -> ClientRuntime:
        """The service's own retry loop: what another service, or a layer, calls."""
        loop = self.loops.get(service.name)
        if loop is None:
            timeout = service.timeout or schema.Timeout()
            cfg = ClientConfig(
                retry=self.loop_policy(service),
                # The caller's deadline is the loop's global deadline; the
                # service's attempt timeout bounds each attempt, enforced there.
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=_ms(timeout.attempt_ms)
                ),
            )
            controller = self.controllers.get(service.name)
            if controller is None:
                loop = ClientRuntime(cfg=cfg, service=self.target(service))
            else:
                loop = AdmittingLoop(
                    cfg=cfg, service=self.target(service), controllers=[controller]
                )
            self.loops[service.name] = loop
        return loop

    def layer(self, service: schema.Service) -> RetryLayer:
        layer = self.layers.get(service.name)
        if layer is None:
            layer = self.layers[service.name] = RetryLayer(self.loop(service))
        return layer

    def client(self, entry: schema.Client, name: str, service: schema.Service) -> ClientRuntime:
        service_timeout = service.timeout or schema.Timeout()
        own_attempt_ms = entry.timeout.attempt_ms if entry.timeout is not None else None
        if entry.retry is None and own_attempt_ms is None:
            # A relay: the service's own retry loop and timeouts are the client's.
            # Its attempts ARE the service's attempts, so they go straight to the
            # service; the limiter gates them from inside the loop.
            global_ms = (
                entry.timeout.global_ms if entry.timeout is not None else service_timeout.global_ms
            )
            return ClientRuntime(
                cfg=ClientConfig(
                    retry=self.loop_policy(service),
                    timeout_controller=StaticTimeoutController(
                        global_timeout=_ms(global_ms),
                        attempt_timeout=_ms(service_timeout.attempt_ms),
                    ),
                    tenant=name,
                    retry_allowance=entry.e2e_retry_budget,
                ),
                service=self.target(service),
            )

        # A client with a policy of its own (or a timer of its own and no
        # policy, which is a client that does not retry: the schema says so).
        policy = _client_policy(entry, self.sim, self.banks)
        global_ms = (
            entry.timeout.global_ms if entry.timeout is not None else service_timeout.global_ms
        )
        service_retries = service.retry is not None and service.retry.type != "none"
        if service_retries or (
            own_attempt_ms is not None and service_timeout.attempt_ms is not None
        ):
            # Two retry loops, or two attempt timeouts, on one hop: the client
            # calls the service's own loop, which keeps its policy and its
            # timeout (and its limiter, at the door), and sees one answer per call.
            target: ServiceLike = self.layer(service)
            attempt_ms = own_attempt_ms
        else:
            target = self.gates.get(service.name, self.target(service))
            # Without a timer of its own, the service's attempt timeout bounds
            # each call, and the server enforces it.
            attempt_ms = (
                own_attempt_ms if own_attempt_ms is not None else service_timeout.attempt_ms
            )
        return ClientRuntime(
            cfg=ClientConfig(
                retry=policy,
                timeout_controller=StaticTimeoutController(
                    global_timeout=_ms(global_ms), attempt_timeout=_ms(attempt_ms)
                ),
                tenant=name,
                # The client's own timer is the client's: the server hears of
                # the global deadline only.
                server_deadline="attempt" if own_attempt_ms is None else "global",
                retry_allowance=entry.e2e_retry_budget,
            ),
            service=target,
        )
