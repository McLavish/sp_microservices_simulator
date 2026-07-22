import csv
import tempfile
import unittest
from pathlib import Path

from core import DropReason, Request, RootRequest, TimeInterval, ms_to_ns, s_to_ns
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

            Metrics(roots=[root], attempts_total=1).export_csv(str(path), granularity_s=1.0)

            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["timepoint"], "5.000")


class TestExportGranularity(unittest.TestCase):
    def _metrics(self):
        root = RootRequest()
        root.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=int(ms_to_ns(0.2)), end=int(ms_to_ns(0.9))),
                success=True,
            )
        )
        root2 = RootRequest()
        root2.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=int(ms_to_ns(1.4)), end=int(ms_to_ns(2.1))),
                success=True,
            )
        )
        return Metrics(roots=[root, root2], attempts_total=2)

    def test_sub_nanosecond_granularity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            self._metrics().export_csv(str(Path(tmp) / "out.csv"), granularity_s=1e-10)

    def test_sub_millisecond_buckets_keep_distinct_timepoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "out.csv")
            self._metrics().export_csv(path, granularity_s=0.0005)
            import pandas as pd

            df = pd.read_csv(path)
            self.assertEqual(df["timepoint"].nunique(), len(df))


if __name__ == "__main__":
    unittest.main()
