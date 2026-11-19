"""A service-time distribution per handler node (`ServiceNodeRuntime.service_time`).

Several chain nodes can share one `ServiceRuntime` -- one per handler of a
service. With an override each draws its local work from its own lognormal,
while the station (workers, queue, latency injections, partial failures, the
100 us floor) stays shared. The tests pin that split, and that a node WITHOUT
an override is exactly what it was.
"""

import math
import unittest

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import DropReason, TimeInterval, ms_to_ns
from msim.faults import LatencyInjection, PartialFailure
from msim.retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy
from msim.service import MIN_LOGNORM_SIGMA, ServiceConfig, ServiceRuntime, ServiceTime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.workload import Workload

US = 1_000


def _station(median_ms: float = 20.0, *, sigma: float = 0.0, workers: int = 8, **kwargs):
    return ServiceRuntime(
        cfg=ServiceConfig(
            name="svc",
            latency_median=ms_to_ns(median_ms),
            latency_lognorm_sigma=sigma,
            workers=workers,
            **kwargs,
        )
    ).bind()


def _client(target, attempt_ms: float = 1000.0) -> ClientRuntime:
    return ClientRuntime(
        cfg=ClientConfig(
            retry=NoRetryPolicy(),
            timeout_controller=StaticTimeoutController(None, ms_to_ns(attempt_ms)),
        ),
        service=target,
    )


def _one_attempt(client: ClientRuntime):
    (root,) = client.roots
    (attempt,) = root.attempts
    return attempt


def _records(clients) -> list[tuple]:
    return [
        (
            a.service,
            a.interval.begin,
            a.interval.end,
            a.success,
            a.drop_reason,
            a.queue_size_at_end,
            a.service_time_ns,
        )
        for client in clients
        for root in client.roots
        for a in root.attempts
    ]


class TestServiceTimeValue(unittest.TestCase):
    def test_service_time_validation_matches_service_config(self):
        for median, sigma in ((0, 0.5), (-5, 0.5), (ms_to_ns(1), -0.1)):
            with self.subTest(median=median, sigma=sigma):
                with self.assertRaises(ValueError):
                    ServiceTime(median=median, sigma=sigma)
                with self.assertRaises(ValueError):
                    ServiceConfig("svc", median, sigma, workers=1)

    def test_non_finite_values_are_rejected(self):
        for median, sigma in (
            (math.nan, 0.5),
            (math.inf, 0.5),
            (ms_to_ns(1), math.nan),
            (ms_to_ns(1), math.inf),
        ):
            with self.subTest(median=median, sigma=sigma), self.assertRaises(ValueError):
                ServiceTime(median=median, sigma=sigma)  # ty: ignore[invalid-argument-type]

    def test_sigma_is_floored_like_the_station_config(self):
        self.assertEqual(ServiceTime(ms_to_ns(1), 0.0).sigma, MIN_LOGNORM_SIGMA)
        self.assertEqual(
            ServiceTime(ms_to_ns(1), 0.0).sigma,
            ServiceConfig("svc", ms_to_ns(1), 0.0, workers=1).latency_lognorm_sigma,
        )
        self.assertEqual(ServiceTime(ms_to_ns(1), 0.4).sigma, 0.4)


class TestNodeServiceTime(unittest.TestCase):
    def test_default_is_none_and_samples_the_station_config(self):
        station = _station(20)
        node = ServiceNodeRuntime(service=station)
        self.assertIsNone(node.service_time)
        sim = Simulator(seed=1)
        client = _client(node)
        client.start_request(sim)
        sim.run()
        self.assertAlmostEqual(_one_attempt(client).service_time_ns, ms_to_ns(20), delta=US)

    def test_local_service_time_reports_the_effective_distribution(self):
        station = _station(20, sigma=0.5)
        plain = ServiceNodeRuntime(service=station)
        fast = ServiceNodeRuntime(service=station, service_time=ServiceTime(ms_to_ns(2), 0.1))
        self.assertEqual(plain.local_service_time, ServiceTime(ms_to_ns(20), 0.5))
        self.assertEqual(fast.local_service_time, ServiceTime(ms_to_ns(2), 0.1))
        # `cfg` stays the station's for both: it is what attempts are booked under.
        self.assertIs(fast.cfg, station.cfg)

    def test_override_changes_only_that_nodes_local_stage(self):
        sim = Simulator(seed=1)
        station = _station(20)
        read = ServiceNodeRuntime(service=station, service_time=ServiceTime(ms_to_ns(1), 0.0))
        write = ServiceNodeRuntime(service=station, service_time=ServiceTime(ms_to_ns(6), 0.0))
        plain = ServiceNodeRuntime(service=station)
        clients = [_client(node) for node in (read, write, plain)]
        for client in clients:
            client.start_request(sim)
        sim.run()

        durations = [_one_attempt(client).service_time_ns for client in clients]
        for got, expected_ms in zip(durations, (1, 6, 20), strict=True):
            self.assertAlmostEqual(got, ms_to_ns(expected_ms), delta=US)
        # All three are attempts at the one station.
        self.assertEqual({_one_attempt(c).service for c in clients}, {"svc"})
        self.assertEqual(station.admitted_total, 3)

    def test_shared_station_still_owns_workers_and_queue(self):
        sim = Simulator(seed=1)
        station = _station(20, workers=1, queue_capacity=1)
        slow = ServiceNodeRuntime(service=station, service_time=ServiceTime(ms_to_ns(10), 0.0))
        fast = ServiceNodeRuntime(service=station, service_time=ServiceTime(ms_to_ns(1), 0.0))
        first, second, third = _client(slow), _client(fast), _client(fast)
        first.start_request(sim)  # takes the only worker for 10 ms
        second.start_request(sim)  # queues behind it, although it is another node
        third.start_request(sim)  # the one waiting slot is taken: shed
        self.assertEqual((station.in_flight, station.queue_len()), (1, 1))
        sim.run()

        self.assertEqual(_one_attempt(third).drop_reason, DropReason.QUEUE_FULL)
        queued = _one_attempt(second)
        self.assertTrue(queued.success)
        # 10 ms waiting for the other node's work, then its own 1 ms.
        self.assertAlmostEqual(queued.interval.end, ms_to_ns(11), delta=US)
        self.assertAlmostEqual(queued.service_time_ns, ms_to_ns(1), delta=US)

    def test_station_faults_and_floor_apply_to_an_override(self):
        window = TimeInterval(ms_to_ns(100), ms_to_ns(200))
        station = _station(
            20,
            latency_injections=[LatencyInjection(window, add_latency=ms_to_ns(30))],
            partial_failures=[PartialFailure(window, p_fail=1.0)],
        )
        node = ServiceNodeRuntime(service=station, service_time=ServiceTime(ms_to_ns(2), 0.0))
        tiny = ServiceNodeRuntime(service=station, service_time=ServiceTime(10 * US, 0.0))
        sim = Simulator(seed=1)
        outside, inside, floored = _client(node), _client(node), _client(tiny)
        outside.start_request(sim)
        floored.start_request(sim)
        sim.schedule(ms_to_ns(110), lambda: inside.start_request(sim))
        sim.run()

        self.assertTrue(_one_attempt(outside).success)
        self.assertAlmostEqual(_one_attempt(outside).service_time_ns, ms_to_ns(2), delta=US)
        # Inside the window: the station's +30 ms and its failure roll both hit.
        hit = _one_attempt(inside)
        self.assertAlmostEqual(hit.service_time_ns, ms_to_ns(2 + 30), delta=US)
        self.assertEqual(hit.drop_reason, DropReason.SERVER_FAILURE)
        # A 10 us override is floored at the station's 100 us, exactly.
        self.assertEqual(_one_attempt(floored).service_time_ns, 100 * US)

    def test_override_applies_to_the_local_stage_of_a_node_that_fans_out(self):
        sim = Simulator(seed=1)
        leaf = ServiceNodeRuntime(service=_station(5))
        parent = ServiceNodeRuntime(
            service=_station(20), service_time=ServiceTime(ms_to_ns(3), 0.0)
        )
        edge = _client(leaf)
        parent.add_downstream(DownstreamEdge(client=edge, target=leaf))
        top = _client(parent)
        top.start_request(sim)
        sim.run()

        self.assertAlmostEqual(_one_attempt(edge).interval.begin, ms_to_ns(3), delta=US)
        self.assertAlmostEqual(_one_attempt(top).interval.end, ms_to_ns(3 + 5), delta=US)
        self.assertAlmostEqual(_one_attempt(top).service_time_ns, ms_to_ns(3), delta=US)


def _loaded_run(override: bool, *, other: ServiceTime | None = None):
    """Two handler nodes over one contended station, with retries and a fault."""
    sim = Simulator(seed=99)
    station = ServiceRuntime(
        cfg=ServiceConfig(
            name="svc",
            latency_median=ms_to_ns(8),
            latency_lognorm_sigma=0.5,
            workers=3,
            queue_capacity=10,
            partial_failures=[PartialFailure(TimeInterval(ms_to_ns(500), ms_to_ns(1500)), 0.3)],
        )
    ).bind()
    same = ServiceTime(station.cfg.latency_median, station.cfg.latency_lognorm_sigma)
    nodes = [
        ServiceNodeRuntime(service=station, service_time=(other or same) if override else None)
        for _ in range(2)
    ]
    clients = [
        ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(5)),
                timeout_controller=StaticTimeoutController(None, ms_to_ns(40)),
            ),
            service=node,
        )
        for node in nodes
    ]
    for index, client in enumerate(clients):
        Workload(base_rps=150.0, duration_s=3.0, load_spikes=[], rng_seed=index).drive(
            sim, client.start_request
        )
    sim.run()
    return _records(clients), sim._seq, sim.rng().getstate()


class TestDefaultPathIsUnchanged(unittest.TestCase):
    def test_override_equal_to_station_config_reproduces_the_run(self):
        # Same distribution through the override as through the station config:
        # every attempt record, the event count and the RNG state must agree, so
        # the override is ONE draw taken at the same place.
        self.assertEqual(_loaded_run(override=False), _loaded_run(override=True))

    def test_a_different_override_changes_the_run(self):
        base = _loaded_run(override=False)
        other = _loaded_run(override=True, other=ServiceTime(ms_to_ns(2), 0.2))
        self.assertNotEqual(base[0], other[0])

    def test_override_consumes_the_same_rng_stream(self):
        # One request at a time, so the number of draws cannot depend on timing:
        # a service-time draw and a failure roll per attempt, whatever the
        # distribution's parameters.
        def final_state(service_time: ServiceTime | None):
            sim = Simulator(seed=5)
            node = ServiceNodeRuntime(service=_station(20, sigma=0.5), service_time=service_time)
            client = _client(node)
            for index in range(50):
                sim.schedule(ms_to_ns(1000 * index), lambda: client.start_request(sim))
            sim.run()
            self.assertEqual(len(client.roots), 50)
            return sim.rng().getstate()

        self.assertEqual(final_state(None), final_state(ServiceTime(ms_to_ns(3), 1.2)))

    def test_leaf_node_matches_plain_service_runtime(self):
        def run(wrap: bool):
            sim = Simulator(seed=7)
            station = _station(8, sigma=0.5, workers=2, queue_capacity=5)
            target = ServiceNodeRuntime(service=station) if wrap else station
            client = ClientRuntime(
                cfg=ClientConfig(
                    retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(5)),
                    timeout_controller=StaticTimeoutController(None, ms_to_ns(30)),
                ),
                service=target,
            )
            Workload(base_rps=200.0, duration_s=3.0, load_spikes=[], rng_seed=7).drive(
                sim, client.start_request
            )
            sim.run()
            return _records([client]), sim.rng().getstate()

        self.assertEqual(run(wrap=False), run(wrap=True))
