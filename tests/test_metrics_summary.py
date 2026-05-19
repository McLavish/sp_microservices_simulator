import unittest
import csv
import tempfile
from pathlib import Path

from core import DropReason, Request, RootRequest, TimeInterval, s_to_ns
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

    def test_export_csv_empty_metrics_writes_header_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.csv"

            Metrics(roots=[], attempts_total=0).export_csv(str(path))

            with path.open(newline="") as f:
                rows = list(csv.reader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "timepoint")

    def test_export_csv_uses_absolute_bucket_timepoints(self):
        root = RootRequest(
            attempts=[
                Request(
                    service="svc",
                    interval=TimeInterval(begin=s_to_ns(5), end=s_to_ns(5.1)),
                    success=True,
                    drop_reason=DropReason.NONE,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "offset.csv"

            Metrics(roots=[root], attempts_total=1).export_csv(
                str(path), granularity_s=1.0
            )

            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["timepoint"], "5.000")


if __name__ == "__main__":
    unittest.main()
