import unittest

from chain import DownstreamEdge, ServiceNodeRuntime
from caller import ClientConfig, ClientRuntime
from core import DropReason
from retry_policies import NoRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController


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
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=1
            ),
        )
        b_client = ClientRuntime(cfg=b_client_cfg, service=b_node)  # type: ignore[arg-type]

        a_node.add_downstream(DownstreamEdge(client=b_client, target=b_node))

        results = {}

        def on_done(success, _svc_time, drop_reason, _queue_size):
            results["success"] = success
            results["drop_reason"] = drop_reason

        a_node.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertFalse(results.get("success", True))
        self.assertEqual(results.get("drop_reason"), DropReason.DEADLINE)


if __name__ == "__main__":
    unittest.main()
