import unittest

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import DropReason, TimeInterval, s_to_ns
from msim.faults import PartialFailure
from msim.retry_policies import NoRetryPolicy
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController


class TestChainFailureReason(unittest.TestCase):
    def test_downstream_deadline_propagates(self):
        sim = Simulator(seed=1)

        a_cfg = ServiceConfig(
            name="svc-A",
            latency_median=1,
            latency_lognorm_sigma=0.1,
            workers=1,
            queue_capacity=None,
        )
        b_cfg = ServiceConfig(
            name="svc-B",
            latency_median=1,
            latency_lognorm_sigma=0.1,
            workers=1,
            queue_capacity=None,
        )

        a_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=a_cfg).bind())
        b_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=b_cfg).bind())

        b_client_cfg = ClientConfig(
            retry=NoRetryPolicy(),
            timeout_controller=StaticTimeoutController(global_timeout=None, attempt_timeout=1),
        )
        b_client = ClientRuntime(cfg=b_client_cfg, service=b_node)

        a_node.add_downstream(DownstreamEdge(client=b_client, target=b_node))

        results = {}

        def on_done(success, _svc_time, drop_reason, _queue_size, _censored=False):
            results["success"] = success
            results["drop_reason"] = drop_reason

        a_node.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertFalse(results.get("success", True))
        self.assertEqual(results.get("drop_reason"), DropReason.DEADLINE)

    def test_local_failure_with_downstream_propagates(self):
        sim = Simulator(seed=1)

        a_cfg = ServiceConfig(
            name="svc-A",
            latency_median=1,
            latency_lognorm_sigma=0.1,
            workers=1,
            queue_capacity=None,
            partial_failures=[
                PartialFailure(
                    duration=TimeInterval(begin=0, end=s_to_ns(1)),
                    p_fail=1.0,
                )
            ],
        )
        b_cfg = ServiceConfig(
            name="svc-B",
            latency_median=1,
            latency_lognorm_sigma=0.1,
            workers=1,
            queue_capacity=None,
        )

        a_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=a_cfg).bind())
        b_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=b_cfg).bind())
        b_client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(),
            ),
            service=b_node,
        )
        a_node.add_downstream(DownstreamEdge(client=b_client, target=b_node))

        results = {}

        def on_done(success, _svc_time, drop_reason, _queue_size, _censored=False):
            results["success"] = success
            results["drop_reason"] = drop_reason

        a_node.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertFalse(results.get("success", True))
        self.assertEqual(results.get("drop_reason"), DropReason.SERVER_FAILURE)
        self.assertEqual(b_client.attempts_total, 0)


if __name__ == "__main__":
    unittest.main()
