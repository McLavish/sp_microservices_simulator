import unittest

from core import DropReason, Request, RootRequest, TimeInterval
from metrics import Metrics


class TestMetricsSummary(unittest.TestCase):
    def test_summary_uses_final_attempt_drop_reason(self):
        root = RootRequest(
            attempts=[
                Request(
                    service="svc",
                    interval=TimeInterval(begin=0, end=10),
                    success=False,
                    drop_reason=DropReason.QUEUE_FULL,
                ),
                Request(
                    service="svc",
                    interval=TimeInterval(begin=10, end=20),
                    success=False,
                    drop_reason=DropReason.DEADLINE,
                ),
            ]
        )
        metrics = Metrics(roots=[root], attempts_total=2)
        summary = metrics.summary()

        self.assertEqual(summary.dropped_deadline, 1)
        self.assertEqual(summary.dropped_queue, 0)
        self.assertEqual(summary.dropped_server_failure, 0)

    def test_summary_counts_success_once(self):
        root = RootRequest(
            attempts=[
                Request(
                    service="svc",
                    interval=TimeInterval(begin=0, end=10),
                    success=True,
                    drop_reason=DropReason.NONE,
                )
            ]
        )
        metrics = Metrics(roots=[root], attempts_total=1)
        summary = metrics.summary()

        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.dropped_deadline, 0)
        self.assertEqual(summary.dropped_queue, 0)
        self.assertEqual(summary.dropped_server_failure, 0)


if __name__ == "__main__":
    unittest.main()
