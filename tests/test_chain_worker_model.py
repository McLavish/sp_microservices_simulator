import unittest

from caller import ClientConfig, ClientRuntime
from chain import DownstreamEdge, ServiceNodeRuntime
from core import ms_to_ns
from retry_policies import NoRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController


def _build_fanout(hold_worker: bool):
    """Parent node (1 worker, fast local) fanning out to one slow downstream.

    Local stage takes ~10ms; the downstream call takes ~200ms, so there is a wide
    window during which the parent is waiting on its fan-out. Returns
    (parent_node, downstream_node, downstream_client).
    """
    parent_cfg = ServiceConfig(
        name="parent",
        latency_median=ms_to_ns(10),
        latency_lognorm_sigma=1e-6,
        workers=1,
        queue_capacity=None,
    )
    downstream_cfg = ServiceConfig(
        name="downstream",
        latency_median=ms_to_ns(200),
        latency_lognorm_sigma=1e-6,
        workers=4,
        queue_capacity=None,
    )

    parent = ServiceNodeRuntime(
        service=ServiceRuntime(cfg=parent_cfg).bind(),
        hold_worker_during_fanout=hold_worker,
    )
    downstream = ServiceNodeRuntime(service=ServiceRuntime(cfg=downstream_cfg).bind())

    client = ClientRuntime(
        cfg=ClientConfig(
            retry=NoRetryPolicy(),
            timeout_controller=StaticTimeoutController(global_timeout=None, attempt_timeout=None),
        ),
        service=downstream,
    )
    parent.add_downstream(DownstreamEdge(client=client, target=downstream))
    return parent, downstream, client


class TestChainWorkerModel(unittest.TestCase):
    def test_default_is_release(self):
        """Tandem-queue (release) model is the default, per CLAUDE.md."""
        node = ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="n",
                    latency_median=ms_to_ns(1),
                    latency_lognorm_sigma=1e-6,
                    workers=1,
                )
            ).bind()
        )
        self.assertFalse(node.hold_worker_during_fanout)

    def test_release_mode_frees_worker_during_fanout(self):
        sim = Simulator(seed=1)
        parent, downstream, _client = _build_fanout(hold_worker=False)

        probe: dict[str, int] = {}

        def take_probe():
            probe["parent_in_flight"] = parent.service.in_flight
            probe["downstream_in_flight"] = downstream.service.in_flight

        ends: list[tuple[bool, int]] = []

        def on_done(success, _svc_time, _drop_reason, _q):
            ends.append((success, sim.timestep))

        # Probe mid fan-out: local stage done (~10ms), downstream still running (~210ms).
        sim.schedule(ms_to_ns(100), take_probe)
        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertEqual(probe["downstream_in_flight"], 1)
        # Tandem model: the parent worker is freed while it waits on the fan-out.
        self.assertEqual(probe["parent_in_flight"], 0)
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0][0])
        self.assertGreater(ends[0][1], ms_to_ns(100))

    def test_hold_mode_occupies_worker_during_fanout(self):
        sim = Simulator(seed=1)
        parent, downstream, _client = _build_fanout(hold_worker=True)

        probe: dict[str, int] = {}

        def take_probe():
            probe["parent_in_flight"] = parent.service.in_flight
            probe["downstream_in_flight"] = downstream.service.in_flight

        ends: list[tuple[bool, int]] = []

        def on_done(success, _svc_time, _drop_reason, _q):
            ends.append((success, sim.timestep))

        sim.schedule(ms_to_ns(100), take_probe)
        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertEqual(probe["downstream_in_flight"], 1)
        # Thread-per-request model: the parent worker stays occupied for the whole
        # downstream wait, so capacity is consumed all the way down the tree.
        self.assertEqual(probe["parent_in_flight"], 1)
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0][0])
        self.assertGreater(ends[0][1], ms_to_ns(100))

    def test_release_mode_allows_concurrent_requests(self):
        sim = Simulator(seed=1)
        parent, _downstream, _client = _build_fanout(hold_worker=False)

        ends: list[tuple[bool, int]] = []

        def on_done(success, _svc_time, _drop_reason, _q):
            ends.append((success, sim.timestep))

        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertEqual(len(ends), 2)
        self.assertTrue(all(success for success, _ in ends))
        # With the worker freed during fan-out, the second request's local stage
        # overlaps the first's downstream wait, so both finish around ~210-220ms.
        self.assertLess(max(t for _, t in ends), ms_to_ns(300))

    def test_hold_mode_serializes_requests(self):
        sim = Simulator(seed=1)
        parent, _downstream, _client = _build_fanout(hold_worker=True)

        ends: list[tuple[bool, int]] = []

        def on_done(success, _svc_time, _drop_reason, _q):
            ends.append((success, sim.timestep))

        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertEqual(len(ends), 2)
        self.assertTrue(all(success for success, _ in ends))
        # The single worker is held through the entire subtree, so the second
        # request cannot start until the first fully completes (~210ms), pushing
        # its completion out to ~420ms.
        self.assertGreater(max(t for _, t in ends), ms_to_ns(350))


if __name__ == "__main__":
    unittest.main()
