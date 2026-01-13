import unittest

from core import DropReason
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator


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


if __name__ == "__main__":
    unittest.main()
