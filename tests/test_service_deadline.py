import unittest

from caller import ClientConfig, ClientRuntime
from core import DropReason, ms_to_ns
from retry_policies import FixedBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController


class TestServiceDeadline(unittest.TestCase):
    def test_submit_attempt_expired_does_not_raise(self):
        sim = Simulator(seed=1)
        cfg = ServiceConfig(
            name="svc",
            latency_median=1,
            latency_lognorm_sigma=0.1,
            workers=1,
            queue_capacity=None,
        )
        svc = ServiceRuntime(cfg=cfg).bind()
        results = []

        def on_done(success, _svc_time, drop_reason, _queue_size):
            results.append((success, drop_reason))

        svc.submit_attempt(sim, on_done, attempt_deadline=0)

        self.assertEqual(results, [(False, DropReason.DEADLINE)])


class TestServiceTimeRecording(unittest.TestCase):
    def test_attempt_interval_includes_queue_wait_but_service_time_does_not(self):
        sim = Simulator(seed=1)
        cfg = ServiceConfig(
            name="svc", latency_median=ms_to_ns(20), latency_lognorm_sigma=1e-6, workers=1
        )
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=1, delay=0),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=None
                ),
            ),
            service=ServiceRuntime(cfg=cfg).bind(),
        )
        client.start_request(sim)
        client.start_request(sim)  # queues behind the first (workers=1)
        sim.run()
        attempts = sorted(
            (a for root in client.roots for a in root.attempts),
            key=lambda a: a.interval.end,
        )
        first, second = attempts
        tol = ms_to_ns(2)
        self.assertLess(abs(first.service_time_ns - ms_to_ns(20)), tol)
        self.assertLess(abs(second.service_time_ns - ms_to_ns(20)), tol)
        # The second attempt's measured interval contains ~20 ms of queue wait.
        self.assertGreater(second.interval.end - second.interval.begin, ms_to_ns(35))


if __name__ == "__main__":
    unittest.main()
