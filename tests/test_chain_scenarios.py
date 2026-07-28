"""End-to-end chain scenarios (tandem fan-out semantics).

These were previously a script under experiments/test_chain_modes.py that mixed
asserts with print output. They are characterization tests of the chain/fan-out
behaviour: AND-fan-out success monotonicity, failure/deadline propagation, and
queue-pressure drops. The tandem-vs-hold-worker occupancy model is covered
separately by tests/test_chain_worker_model.py.
"""

import unittest

from caller import ClientConfig, ClientRuntime
from chain import DownstreamEdge, ServiceNodeRuntime
from core import DropReason, TimeInterval, ms_to_ns, s_to_ns
from faults import LatencyInjection, LoadSpike, PartialFailure
from retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload


def _client_cfg(attempt_timeout_ms: int, max_attempts: int, delay_ms: int) -> ClientConfig:
    return ClientConfig(
        retry=FixedBackoffRetryPolicy(max_attempts=max_attempts, delay=ms_to_ns(delay_ms)),
        timeout_controller=StaticTimeoutController(
            global_timeout=None, attempt_timeout=ms_to_ns(attempt_timeout_ms)
        ),
    )


def _run_fanout_scenario(seed: int, top_attempt_timeout_ms: int = 200):
    """A -> B -> {C, E} with tandem semantics; C fails partially during an outage.

    Returns the per-client metrics summaries for (top A, B, C, E).
    """
    sim = Simulator(seed=seed)

    a_cfg = ServiceConfig("svc-A", ms_to_ns(5), 0.4, workers=16, queue_capacity=100)
    b_cfg = ServiceConfig("svc-B", ms_to_ns(10), 0.5, workers=16, queue_capacity=100)
    c_cfg = ServiceConfig(
        "svc-C",
        ms_to_ns(15),
        0.6,
        workers=16,
        queue_capacity=100,
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(20), end=s_to_ns(40)), p_fail=0.2)
        ],
    )
    e_cfg = ServiceConfig(
        "svc-E",
        ms_to_ns(8),
        0.5,
        workers=12,
        queue_capacity=80,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(30), end=s_to_ns(45)),
                add_latency=ms_to_ns(10),
            )
        ],
    )

    a_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=a_cfg).bind())
    b_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=b_cfg).bind())
    c_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=c_cfg).bind())
    e_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=e_cfg).bind())

    b_to_c = ClientRuntime(_client_cfg(80, 2, 40), service=c_node)
    b_to_e = ClientRuntime(_client_cfg(80, 2, 40), service=e_node)
    a_to_b = ClientRuntime(_client_cfg(120, 2, 60), service=b_node)

    b_node.add_downstream(DownstreamEdge(client=b_to_c, target=c_node))
    b_node.add_downstream(DownstreamEdge(client=b_to_e, target=e_node))
    a_node.add_downstream(DownstreamEdge(client=a_to_b, target=b_node))

    top_client = ClientRuntime(
        ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(100)),
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(top_attempt_timeout_ms)
            ),
        ),
        service=a_node,
    )

    wl = Workload(
        base_rps=300.0,
        duration_s=60,
        load_spikes=[
            LoadSpike(duration=TimeInterval(begin=s_to_ns(25), end=s_to_ns(35)), rps_multiplier=1.8)
        ],
    )
    wl.drive(sim, lambda s: top_client.start_request(s))
    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    return {
        "A": top_client.metrics().summary(),
        "B": a_to_b.metrics().summary(),
        "C": b_to_c.metrics().summary(),
        "E": b_to_e.metrics().summary(),
    }


class TestChainScenarios(unittest.TestCase):
    def test_tandem_fanout_behavior(self):
        r = _run_fanout_scenario(seed=1234)

        # AND-fanout: successes are monotonic down the chain.
        self.assertEqual(r["A"].succeeded, r["B"].succeeded)
        self.assertLessEqual(r["B"].succeeded, min(r["C"].succeeded, r["E"].succeeded))

        # Partial failures on C are observable at C and propagate to the root.
        self.assertGreater(r["C"].dropped_server_failure, 0)
        self.assertGreater(r["A"].dropped_server_failure, 0)

        # The caller retries under downstream failure.
        self.assertGreater(r["A"].retries_per_root, 0)

    def test_deadline_propagation(self):
        # A tight top-level attempt timeout forces deadline drops upstream.
        r = _run_fanout_scenario(seed=4321, top_attempt_timeout_ms=30)
        self.assertGreater(r["A"].dropped_deadline, 0)
        self.assertGreater(r["B"].dropped_deadline, 0)

    def test_queue_pressure_drops(self):
        sim = Simulator(seed=7)

        a_cfg = ServiceConfig("svc-A", ms_to_ns(3), 0.4, workers=2, queue_capacity=5)
        b_cfg = ServiceConfig("svc-B", ms_to_ns(5), 0.5, workers=2, queue_capacity=5)
        c_cfg = ServiceConfig("svc-C", ms_to_ns(7), 0.3, workers=2, queue_capacity=5)

        a_node = ServiceNodeRuntime(service=ServiceRuntime(a_cfg).bind())
        b_node = ServiceNodeRuntime(service=ServiceRuntime(b_cfg).bind())
        c_node = ServiceNodeRuntime(service=ServiceRuntime(c_cfg).bind())

        a_to_b = ClientRuntime(_client_cfg(20, 1, 10), service=b_node)
        b_to_c = ClientRuntime(_client_cfg(25, 1, 10), service=c_node)
        a_node.add_downstream(DownstreamEdge(client=a_to_b, target=b_node))
        b_node.add_downstream(DownstreamEdge(client=b_to_c, target=c_node))

        top_client = ClientRuntime(_client_cfg(30, 1, 10), service=a_node)

        wl = Workload(base_rps=200.0, duration_s=10, load_spikes=[])
        wl.drive(sim, lambda s: top_client.start_request(s))
        sim.run(until=s_to_ns(10))
        sim.run()

        a = top_client.metrics().summary()
        b = a_to_b.metrics().summary()
        self.assertTrue(a.dropped_queue > 0 or b.dropped_queue > 0)
        self.assertGreater(a.succeeded, 0)

    def test_deeper_chain_monotonic(self):
        sim = Simulator(seed=99)

        a_node = ServiceNodeRuntime(
            ServiceRuntime(
                ServiceConfig("svc-A", ms_to_ns(3), 0.3, workers=8, queue_capacity=50)
            ).bind()
        )
        b_node = ServiceNodeRuntime(
            ServiceRuntime(
                ServiceConfig("svc-B", ms_to_ns(4), 0.3, workers=8, queue_capacity=50)
            ).bind()
        )
        c_node = ServiceNodeRuntime(
            ServiceRuntime(
                ServiceConfig("svc-C", ms_to_ns(5), 0.3, workers=8, queue_capacity=50)
            ).bind()
        )
        d_node = ServiceNodeRuntime(
            ServiceRuntime(
                ServiceConfig("svc-D", ms_to_ns(6), 0.3, workers=8, queue_capacity=50)
            ).bind()
        )

        c_to_d = ClientRuntime(_client_cfg(80, 1, 10), service=d_node)
        b_to_c = ClientRuntime(_client_cfg(80, 1, 10), service=c_node)
        a_to_b = ClientRuntime(_client_cfg(100, 1, 10), service=b_node)
        c_node.add_downstream(DownstreamEdge(client=c_to_d, target=d_node))
        b_node.add_downstream(DownstreamEdge(client=b_to_c, target=c_node))
        a_node.add_downstream(DownstreamEdge(client=a_to_b, target=b_node))

        top_client = ClientRuntime(_client_cfg(120, 1, 10), service=a_node)

        wl = Workload(base_rps=100.0, duration_s=10, load_spikes=[])
        wl.drive(sim, lambda s: top_client.start_request(s))
        sim.run(until=s_to_ns(10))
        sim.run()

        a = top_client.metrics().summary()
        b = a_to_b.metrics().summary()
        c = b_to_c.metrics().summary()
        d = c_to_d.metrics().summary()
        self.assertEqual(a.succeeded, b.succeeded)
        self.assertEqual(b.succeeded, c.succeeded)
        self.assertLessEqual(c.succeeded, d.succeeded)


class TestFanoutLaunchesAllEdges(unittest.TestCase):
    def _run(self, reject_first: bool):
        sim = Simulator(seed=1)

        def leaf(name, queue_capacity=None):
            return ServiceNodeRuntime(
                service=ServiceRuntime(
                    cfg=ServiceConfig(
                        name=name,
                        latency_median=ms_to_ns(10),
                        latency_lognorm_sigma=1e-6,
                        workers=1,
                        queue_capacity=queue_capacity,
                    )
                ).bind()
            )

        reject = leaf("svc-reject", queue_capacity=0)
        ok = leaf("svc-ok")
        # Occupy svc-reject's single worker so a fan-out submit hits the full
        # (capacity-0) queue and fails synchronously.
        reject.service.submit_attempt(sim, lambda *a: None)

        parent = ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="svc-parent",
                    latency_median=ms_to_ns(5),
                    latency_lognorm_sigma=1e-6,
                    workers=4,
                )
            ).bind()
        )

        def edge(node):
            return DownstreamEdge(
                client=ClientRuntime(
                    cfg=ClientConfig(
                        retry=NoRetryPolicy(),
                        timeout_controller=StaticTimeoutController(
                            global_timeout=None, attempt_timeout=ms_to_ns(200)
                        ),
                    ),
                    service=node,
                ),
                target=node,
            )

        edges = [edge(reject), edge(ok)] if reject_first else [edge(ok), edge(reject)]
        for e in edges:
            parent.add_downstream(e)
        results = []
        parent.submit_attempt(
            sim, lambda success, t, drop, q, censored=False: results.append((success, drop))
        )
        sim.run()
        ok_edge = edges[1] if reject_first else edges[0]
        return results, len(ok_edge.client.roots)

    def test_second_edge_still_receives_load_when_first_rejects_synchronously(self):
        results, ok_calls = self._run(reject_first=True)
        self.assertEqual(ok_calls, 1)  # was 0: the loop broke before launching it
        self.assertEqual(results[0][0], False)  # AND semantics: parent still fails
        self.assertEqual(results[0][1], DropReason.QUEUE_FULL)

    def test_edge_order_does_not_change_the_load_pattern(self):
        _, ok_calls = self._run(reject_first=False)
        self.assertEqual(ok_calls, 1)


if __name__ == "__main__":
    unittest.main()
