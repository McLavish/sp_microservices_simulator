"""Trace goldens: seeded runs pinned attempt by attempt.

`test_golden_and_invariants.py` pins counts and two-decimal percentiles. That
cannot see a change that leaves the aggregates alone: an extra scheduled event,
an extra draw from the simulator's RNG, two events at one instant swapping
places, a policy layer fed twice. Everything downstream of this package compares
simulator output byte for byte against stored runs, so those are exactly the
changes that must not happen by accident.

Each scenario here is therefore pinned three ways:

* a digest of EVERY attempt record of every client (and of each root's own
  completion), in completion order;
* the number of events the run scheduled (`Simulator._seq`);
* a digest of the simulator RNG's final state, i.e. how many draws were taken.

`examples.default` is pinned by the bytes of the files it writes, since those
files are what the downstream gates compare against.

Brittle BY DESIGN, like the goldens next door: a diff means the event order, the
RNG consumption or a modelling rule changed. Establish why before touching an
expected value. A new mechanism must leave every value here untouched as long as
it is not switched on.
"""

import hashlib
import os
import runpy
import unittest
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pytest

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import TimeInterval, ms_to_ns, s_to_ns
from msim.faults import LatencyInjection, LoadSpike, PartialFailure
from msim.retry_policies import (
    BurstyRateLimiterPolicy,
    CountBasedCircuitBreakerPolicy,
    ExponentialBackoffWithJitterRetryPolicy,
    FixedBackoffRetryPolicy,
    FixedWindowBurstyLimiterPolicy,
    JitterMode,
    LeakyRateLimiterPolicy,
    RetryBudgetPolicy,
    TimeBasedCircuitBreakerPolicy,
)
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import AdaptiveTimeoutController, StaticTimeoutController
from msim.transport import TransportDelay, TransportLink
from msim.workload import Workload

Trace = tuple[str, int, str]


def _trace(
    sim: Simulator,
    clients: Iterable[ClientRuntime],
    services: Iterable[ServiceRuntime],
    extra: Iterable[object] = (),
) -> Trace:
    """`(attempt digest, events scheduled, RNG-state digest)` of a finished run."""
    digest = hashlib.sha256()
    for client in clients:
        for root in client.roots:
            digest.update(repr((root.cancelled, root.completed_at, root.global_deadline)).encode())
            for attempt in root.attempts:
                digest.update(
                    repr(
                        (
                            attempt.service,
                            attempt.interval.begin,
                            attempt.interval.end,
                            attempt.deadline,
                            attempt.success,
                            attempt.drop_reason.name,
                            attempt.queue_size_at_end,
                            attempt.service_time_ns,
                            attempt.service_time_censored,
                        )
                    ).encode()
                )
        digest.update(repr(("attempts_total", client.attempts_total)).encode())
    for service in services:
        digest.update(
            repr(
                (service.cfg.name, service.admitted_total, service.in_flight, service.queue_len())
            ).encode()
        )
    for item in extra:
        digest.update(repr(item).encode())
    rng_state = hashlib.sha256(repr(sim.rng().getstate()).encode()).hexdigest()
    return digest.hexdigest(), sim._seq, rng_state


def _reasons(clients: Iterable[ClientRuntime]) -> Counter:
    return Counter(
        attempt.drop_reason.name
        for client in clients
        for root in client.roots
        for attempt in root.attempts
    )


def _static(attempt_ms: float, global_ms: float | None = None) -> StaticTimeoutController:
    return StaticTimeoutController(
        global_timeout=None if global_ms is None else ms_to_ns(global_ms),
        attempt_timeout=ms_to_ns(attempt_ms),
    )


def _fixed(max_attempts: int, delay_ms: float) -> FixedBackoffRetryPolicy:
    return FixedBackoffRetryPolicy(max_attempts=max_attempts, delay=ms_to_ns(delay_ms))


# --------------------------------------------------------------- scenarios


def _single_service(seed: int = 1234):
    """`test_golden_and_invariants._single_service_run`, with the simulator kept."""
    sim = Simulator(seed=seed)
    svc = ServiceRuntime(
        cfg=ServiceConfig(
            name="svc-A",
            latency_median=ms_to_ns(10),
            latency_lognorm_sigma=0.5,
            workers=2,
            queue_capacity=100,
        )
    ).bind()
    client = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(2, 20), timeout_controller=_static(200, global_ms=500)),
        service=svc,
    )
    Workload(base_rps=180.0, duration_s=10.0, load_spikes=[], rng_seed=seed).drive(
        sim, client.start_request
    )
    sim.run()
    return sim, [client], [svc]


def _chain(seed: int = 1234):
    """`test_golden_and_invariants._chain_run`: A -> B -> C."""
    sim = Simulator(seed=seed)

    def node(name: str, median_ms: float, workers: int) -> ServiceNodeRuntime:
        return ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name=name,
                    latency_median=ms_to_ns(median_ms),
                    latency_lognorm_sigma=0.4,
                    workers=workers,
                    queue_capacity=200,
                )
            ).bind()
        )

    a, b, c = node("A", 5, 8), node("B", 8, 6), node("C", 12, 4)
    edges = []
    for parent, child in ((a, b), (b, c)):
        client = ClientRuntime(
            cfg=ClientConfig(retry=_fixed(2, 15), timeout_controller=_static(150)),
            service=child,
        )
        parent.add_downstream(DownstreamEdge(client=client, target=child))
        edges.append(client)
    top = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(2, 20), timeout_controller=_static(400)), service=a
    )
    Workload(base_rps=120.0, duration_s=6.0, load_spikes=[], rng_seed=seed).drive(
        sim, top.start_request
    )
    sim.run()
    return sim, [top, *edges], [a.service, b.service, c.service]


def _fanout(seed: int = 1234):
    """`test_golden_and_invariants._fanout_run`: many completions share an instant."""
    sim = Simulator(seed=seed)

    def node(name: str, median_ms: float, workers: int) -> ServiceNodeRuntime:
        return ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name=name,
                    latency_median=ms_to_ns(median_ms),
                    latency_lognorm_sigma=1e-6,
                    workers=workers,
                    queue_capacity=3,
                )
            ).bind()
        )

    a = node("A", 2, 4)
    edges, leaves = [], []
    for index in range(3):
        leaf = node(f"L{index}", 10, 1)
        leaf.service.cfg.partial_failures.append(
            PartialFailure(TimeInterval(s_to_ns(2), s_to_ns(4)), p_fail=0.3)
        )
        client = ClientRuntime(
            cfg=ClientConfig(retry=_fixed(2, 5), timeout_controller=_static(60)), service=leaf
        )
        a.add_downstream(DownstreamEdge(client=client, target=leaf))
        edges.append(client)
        leaves.append(leaf.service)
    top = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(2, 10), timeout_controller=_static(300)), service=a
    )
    Workload(base_rps=150.0, duration_s=6.0, load_spikes=[], rng_seed=seed).drive(
        sim, top.start_request
    )
    sim.run()
    return sim, [top, *edges], [a.service, *leaves]


def _chain_switches(seed: int = 77):
    """Every per-node switch at its non-default value, plus a transport link.

    A holds its worker through the fan-out and cancels abandoned siblings; B
    does not pass its deadline down; C sits behind a link with a random delay;
    queues are small enough to shed and both fault kinds are active. So the
    trace contains every drop reason a chain can produce.
    """
    sim = Simulator(seed=seed)

    def service(name: str, median_ms: float, sigma: float, workers: int, capacity: int, **faults):
        return ServiceRuntime(
            cfg=ServiceConfig(
                name=name,
                latency_median=ms_to_ns(median_ms),
                latency_lognorm_sigma=sigma,
                workers=workers,
                queue_capacity=capacity,
                **faults,
            )
        ).bind()

    a = ServiceNodeRuntime(
        service=service("A", 3, 0.3, 6, 40),
        hold_worker_during_fanout=True,
        propagate_cancellation=True,
    )
    b = ServiceNodeRuntime(
        service=service("B", 4, 0.4, 4, 30),
        propagate_cancellation=True,
        propagate_deadline=False,
    )
    c = ServiceNodeRuntime(
        service=service(
            "C",
            6,
            0.6,
            2,
            2,
            partial_failures=[PartialFailure(TimeInterval(s_to_ns(2), s_to_ns(4)), p_fail=0.4)],
        )
    )
    d = ServiceNodeRuntime(
        service=service(
            "D",
            8,
            0.5,
            3,
            4,
            latency_injections=[
                LatencyInjection(TimeInterval(s_to_ns(3), s_to_ns(5)), add_latency=ms_to_ns(20))
            ],
        )
    )
    a_to_b = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(2, 5), timeout_controller=_static(60)), service=b
    )
    a_to_c = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(3, 3), timeout_controller=_static(40)),
        service=TransportLink(c, TransportDelay(median=ms_to_ns(1), sigma=0.3)),
    )
    b_to_d = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(2, 4), timeout_controller=_static(35)), service=d
    )
    a.add_downstream(DownstreamEdge(client=a_to_b, target=b))
    a.add_downstream(DownstreamEdge(client=a_to_c, target=c))
    b.add_downstream(DownstreamEdge(client=b_to_d, target=d))
    top = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(2, 10), timeout_controller=_static(120, global_ms=400)),
        service=a,
    )
    spike = LoadSpike(TimeInterval(s_to_ns(1), s_to_ns(2)), rps_multiplier=3.0)
    Workload(base_rps=140.0, duration_s=6.0, load_spikes=[spike], rng_seed=seed).drive(
        sim, top.start_request
    )
    sim.run()
    return sim, [top, a_to_b, a_to_c, b_to_d], [a.service, b.service, c.service, d.service]


def _retry_storm(seed: int = 4242):
    """`examples.multichain_retry_storm` in miniature: A -> B -> C and X -> Y -> C.

    Same topology, policies and fault as the shipped example (three retry
    layers of 3, 3 and 5 attempts over a shared bottleneck that slows down and
    fails), on a 30 s horizon instead of 120 s so it can run on every commit.
    The full-size example is pinned by the recorded Multichain cells that
    `harness ablate --stage verify` rebuilds.
    """
    sim = Simulator(seed=seed)
    outage = TimeInterval(s_to_ns(10), s_to_ns(16))

    def relay(name: str) -> ServiceNodeRuntime:
        return ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name=name,
                    latency_median=ms_to_ns(12),
                    latency_lognorm_sigma=0.4,
                    workers=48,
                    queue_capacity=2000,
                )
            ).bind()
        )

    c = ServiceNodeRuntime(
        service=ServiceRuntime(
            cfg=ServiceConfig(
                name="svc-C",
                latency_median=ms_to_ns(20),
                latency_lognorm_sigma=0.7,
                workers=24,
                queue_capacity=800,
                latency_injections=[LatencyInjection(outage, add_latency=ms_to_ns(30))],
                partial_failures=[PartialFailure(outage, p_fail=0.4)],
            )
        ).bind()
    )
    a, b, x, y = relay("svc-A"), relay("svc-B"), relay("svc-X"), relay("svc-Y")
    b_to_c = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(5, 5), timeout_controller=_static(60)), service=c
    )
    y_to_c = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(5, 5), timeout_controller=_static(60)), service=c
    )
    a_to_b = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(3, 20), timeout_controller=_static(120)), service=b
    )
    x_to_y = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(3, 20), timeout_controller=_static(120)), service=y
    )
    b.add_downstream(DownstreamEdge(client=b_to_c, target=c))
    y.add_downstream(DownstreamEdge(client=y_to_c, target=c))
    a.add_downstream(DownstreamEdge(client=a_to_b, target=b))
    x.add_downstream(DownstreamEdge(client=x_to_y, target=y))
    top_a = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(3, 30), timeout_controller=_static(200)), service=a
    )
    top_x = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(3, 30), timeout_controller=_static(200)), service=x
    )
    spike = LoadSpike(TimeInterval(s_to_ns(12), s_to_ns(18)), rps_multiplier=1.5)
    Workload(base_rps=180.0, duration_s=30.0, load_spikes=[spike], rng_seed=0).drive(
        sim, top_a.start_request
    )
    Workload(base_rps=180.0, duration_s=30.0, load_spikes=[spike], rng_seed=1).drive(
        sim, top_x.start_request
    )
    sim.run()
    clients = [top_a, top_x, a_to_b, x_to_y, b_to_c, y_to_c]
    return sim, clients, [node.service for node in (a, b, c, x, y)]


def _call_waves(seed: int = 2024):
    """Sequential and staged downstream calls (`DownstreamEdge.wave`), under load.

    A calls {B, C} concurrently and then D; B calls E and then F. C sits behind
    a transport link, queues are small, F fails part of the time and D slows
    down, so waves are cut short by failures, deadlines and full queues as well
    as completing normally.
    """
    sim = Simulator(seed=seed)

    def node(name: str, median_ms: float, workers: int, capacity: int, **faults):
        return ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name=name,
                    latency_median=ms_to_ns(median_ms),
                    latency_lognorm_sigma=0.5,
                    workers=workers,
                    queue_capacity=capacity,
                    **faults,
                )
            ).bind()
        )

    a, b, c = node("A", 2, 8, 40), node("B", 3, 6, 30), node("C", 6, 3, 6)
    d = node(
        "D",
        5,
        3,
        6,
        latency_injections=[
            LatencyInjection(TimeInterval(s_to_ns(3), s_to_ns(5)), add_latency=ms_to_ns(25))
        ],
    )
    e = node("E", 4, 3, 6)
    f = node(
        "F",
        4,
        3,
        6,
        partial_failures=[PartialFailure(TimeInterval(s_to_ns(1), s_to_ns(3)), p_fail=0.4)],
    )

    def edge(parent, child, wave: int, attempt_ms: float, target=None) -> ClientRuntime:
        client = ClientRuntime(
            cfg=ClientConfig(retry=_fixed(2, 4), timeout_controller=_static(attempt_ms)),
            service=target or child,
        )
        parent.add_downstream(DownstreamEdge(client=client, target=child, wave=wave))
        return client

    a_to_b = edge(a, b, 0, 70)
    a_to_c = edge(a, c, 0, 50, target=TransportLink(c, TransportDelay(ms_to_ns(1), 0.3)))
    a_to_d = edge(a, d, 1, 40)
    b_to_e = edge(b, e, 0, 30)
    b_to_f = edge(b, f, 1, 30)
    top = ClientRuntime(
        cfg=ClientConfig(retry=_fixed(2, 10), timeout_controller=_static(150, global_ms=400)),
        service=a,
    )
    spike = LoadSpike(TimeInterval(s_to_ns(4), s_to_ns(5)), rps_multiplier=2.0)
    Workload(base_rps=150.0, duration_s=6.0, load_spikes=[spike], rng_seed=seed).drive(
        sim, top.start_request
    )
    sim.run()
    clients = [top, a_to_b, a_to_c, a_to_d, b_to_e, b_to_f]
    return sim, clients, [n.service for n in (a, b, c, d, e, f)]


def _wrapped_policies(seed: int = 4321):
    """Two clients with deep policy stacks on one faulted service.

    Client 1: count breaker > retry budget > fixed-window limiter > jittered
    backoff drawing from the simulator's own RNG, under a global timeout short
    enough to veto granted retries (so refunds happen). Client 2: time breaker >
    token bucket > leaky bucket, with an adaptive timeout controller. Their
    final internal state is part of the digest: it is what a change to the
    client's walk over policy layers would move first.
    """
    sim = Simulator(seed=seed)
    svc = ServiceRuntime(
        cfg=ServiceConfig(
            name="svc-P",
            latency_median=ms_to_ns(12),
            latency_lognorm_sigma=0.6,
            workers=4,
            queue_capacity=6,
            latency_injections=[
                LatencyInjection(TimeInterval(s_to_ns(3), s_to_ns(5)), add_latency=ms_to_ns(25))
            ],
            partial_failures=[PartialFailure(TimeInterval(s_to_ns(6), s_to_ns(9)), p_fail=0.5)],
        )
    ).bind()

    jitter = ExponentialBackoffWithJitterRetryPolicy(
        max_attempts=4,
        initial_delay=ms_to_ns(10),
        max_delay=ms_to_ns(80),
        rng=sim.rng(),
        jitter_mode=JitterMode.FULL,
    )
    window = FixedWindowBurstyLimiterPolicy(
        max_requests=40, period=s_to_ns(1), underlying_strategy=jitter
    )
    budget = RetryBudgetPolicy(budget_ratio=0.2, max_retries=5, underlying_strategy=window)
    count_breaker = CountBasedCircuitBreakerPolicy(
        failure_threshold_ratio=(6, 10),
        success_threshold_ratio=(3, 5),
        underlying_strategy=budget,
        half_open_delay=ms_to_ns(500),
    )
    first = ClientRuntime(
        cfg=ClientConfig(retry=count_breaker, timeout_controller=_static(60, global_ms=250)),
        service=svc,
    )

    leaky = LeakyRateLimiterPolicy(max_requests=30, max_attempts=3, period=s_to_ns(1))
    bucket = BurstyRateLimiterPolicy(
        max_requests=20, refill_rate=10.0, period=s_to_ns(1), underlying_strategy=leaky
    )
    time_breaker = TimeBasedCircuitBreakerPolicy(
        failure_threshold_rate=0.6,
        success_threshold_rate=0.5,
        min_requests=10,
        window_duration=s_to_ns(1),
        underlying_strategy=bucket,
        half_open_delay=ms_to_ns(400),
        half_open_min_requests=3,
    )
    adaptive = AdaptiveTimeoutController(
        initial_global_timeout=ms_to_ns(300), initial_attempt_timeout=ms_to_ns(70)
    )
    second = ClientRuntime(
        cfg=ClientConfig(retry=time_breaker, timeout_controller=adaptive), service=svc
    )

    Workload(base_rps=110.0, duration_s=12.0, load_spikes=[], rng_seed=seed).drive(
        sim, first.start_request
    )
    Workload(base_rps=90.0, duration_s=12.0, load_spikes=[], rng_seed=seed + 1).drive(
        sim, second.start_request
    )
    sim.run()
    policy_state = (
        count_breaker.get_state().name,
        budget._tokens,
        window._tokens,
        time_breaker.get_state().name,
        bucket._tokens,
        leaky._next_at,
        adaptive.get_attempt_timeout(),
    )
    return sim, [first, second], [svc], policy_state


# Captured from the implementation at 3935139 (before any new mechanism).
_SINGLE_SERVICE: Trace = (
    "64face9c1a40b95771b55635d89403977b1aa424abf6b95f924a3d0e1d9cb769",
    6726,
    "bcb234375c3f94f570760e229ee5cce4d5bf46970df62a976ad53bce0cf45786",
)
_CHAIN: Trace = (
    "8c87017690494e15d0658d81f58f7c4a920be8554d5ee7921b4d353e5bad36db",
    4129,
    "787681e44b73121731b8c533f21b365780d7367f6430c62317f332d356f01cc5",
)
_FANOUT: Trace = (
    "64e0f9136df4b0c8e8649aec1df3c9fd5ae276d2ba09039cf2b42631930a7586",
    11203,
    "f5dda689eb8a01fb123b50b121b111a04b0dc13f9563c7e26e26d77e439868f4",
)
_CHAIN_SWITCHES: Trace = (
    "4fef729e87fbc8091563b23477aa2822df254542d195bb03bbc794227275d1bb",
    16989,
    "fb7d20c012351678fc468c4a4239029479486f6bff3d2c5aaaad5d4d5095d345",
)
_RETRY_STORM: Trace = (
    "f286c424758deaa36b423cca8d304dbf67364b1ad798328565599a620aa3d8f0",
    169793,
    "84cfe5f9590d6d9023acd03edcb3a8602ae424fc82f442bb3af89daa40b13093",
)
_WRAPPED_POLICIES: Trace = (
    "4cfc9f5512e84cd91ab533335ff2432be7d4ccc47c6f95cefe44994c4bd7c194",
    5416,
    "406ed7d7d422da9ca0dd7fc470cbdc9831a3b051eed46a5a5b8acc69f3843fab",
)


# One per mechanism added since, captured with the mechanism switched ON.
_CALL_WAVES: Trace = (
    "1ea7a1a18cc29afcd6028bf8c8b1d10f0f6ee89b182fa94f4e92f230eb9daf70",
    23822,
    "ced390460ed52428bd31da7f0e1cd51d157421913c49c6b95cfe32ff9d9898a7",
)


class TestTraceGoldens(unittest.TestCase):
    """If one of these fails, a seeded run no longer unfolds the way it did."""

    def test_call_waves_trace(self):
        sim, clients, services = _call_waves()
        reasons = _reasons(clients)
        for reason in ("NONE", "QUEUE_FULL", "DEADLINE", "SERVER_FAILURE"):
            self.assertGreater(reasons[reason], 0, reason)
        self.assertEqual(_trace(sim, clients, services), _CALL_WAVES)

    def test_single_service_trace(self):
        sim, clients, services = _single_service()
        self.assertEqual(_trace(sim, clients, services), _SINGLE_SERVICE)

    def test_chain_trace(self):
        sim, clients, services = _chain()
        self.assertEqual(_trace(sim, clients, services), _CHAIN)

    def test_fanout_trace(self):
        sim, clients, services = _fanout()
        self.assertEqual(_trace(sim, clients, services), _FANOUT)

    def test_chain_switches_trace(self):
        sim, clients, services = _chain_switches()
        # The scenario is only worth pinning while it exercises every path.
        reasons = _reasons(clients)
        for reason in ("NONE", "QUEUE_FULL", "DEADLINE", "SERVER_FAILURE", "CANCELLED"):
            self.assertGreater(reasons[reason], 0, reason)
        self.assertEqual(_trace(sim, clients, services), _CHAIN_SWITCHES)

    def test_retry_storm_trace(self):
        sim, clients, services = _retry_storm()
        self.assertEqual(_trace(sim, clients, services), _RETRY_STORM)

    def test_wrapped_policy_trace(self):
        sim, clients, services, policy_state = _wrapped_policies()
        reasons = _reasons(clients)
        for reason in ("NONE", "QUEUE_FULL", "DEADLINE", "SERVER_FAILURE", "CIRCUIT_OPEN"):
            self.assertGreater(reasons[reason], 0, reason)
        self.assertEqual(_trace(sim, clients, services, extra=policy_state), _WRAPPED_POLICIES)


# ------------------------------------------------------- shipped examples

_EXAMPLE_FILES: dict[str, dict[str, str]] = {
    "examples.default": {
        "fault_events.json": "e1d00f8f52ee4823ad43d2059c607f5932995b3ecf56c1e7952f573607e28cc2",
        "output.csv": "a05e4736f0537157c32ed3fa42b6b907a15d0701ee563644ee4fcf72cdb5f106",
    },
}


def _written_files(module: str, cwd: Path) -> dict[str, str]:
    """Run a demo the way a reader does and digest every file it wrote."""
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        os.chdir(previous)
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(cwd.iterdir())
        if path.is_file()
    }


@pytest.mark.parametrize("module", sorted(_EXAMPLE_FILES))
def test_shipped_example_writes_the_same_bytes(module, tmp_path):
    """The files of the example the matched builder is checked against.

    `harness k0b` (microservices-system) asserts that its builder reproduces
    this example's files byte for byte, so a change to them is a change to
    every stored simulation that gate stands for. The other example it checks,
    `examples.multichain_retry_storm`, takes a minute to run; `_retry_storm`
    above pins the same topology on a shorter horizon.
    """
    assert _written_files(module, tmp_path) == _EXAMPLE_FILES[module]
