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
        assert first.service_time_ns is not None and second.service_time_ns is not None
        tol = ms_to_ns(2)
        self.assertLess(abs(first.service_time_ns - ms_to_ns(20)), tol)
        self.assertLess(abs(second.service_time_ns - ms_to_ns(20)), tol)
        # The second attempt's measured interval contains ~20 ms of queue wait.
        self.assertGreater(second.interval.end - second.interval.begin, ms_to_ns(35))

    def test_timeout_records_occupancy_not_latent_service_time(self):
        # A 100 ms latent draw cut by a 10 ms attempt deadline frees the worker
        # at 10 ms; the recorded service time must be the ~10 ms of actual
        # occupancy, not the latent 100 ms (which would corrupt utilization /
        # Erlang-C moments downstream).
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(100), latency_lognorm_sigma=1e-6, workers=1
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=1, delay=0),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(10)
                ),
            ),
            service=svc,
        )
        client.start_request(sim)
        sim.run()
        (attempt,) = [a for root in client.roots for a in root.attempts]
        assert attempt.service_time_ns is not None
        self.assertEqual(attempt.drop_reason, DropReason.DEADLINE)
        self.assertLess(attempt.service_time_ns, ms_to_ns(11))
        self.assertEqual(svc.in_flight, 0)  # worker freed at the deadline


if __name__ == "__main__":
    unittest.main()
