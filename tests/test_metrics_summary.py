import csv
import tempfile
import unittest
from pathlib import Path

from core import (
    FAILURE_OUTCOMES,
    DropReason,
    Request,
    RootOutcome,
    RootRequest,
    TimeInterval,
    ms_to_ns,
    require_outcome_coverage,
    s_to_ns,
)
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


class TestSummaryPopulation(unittest.TestCase):
    """Every Summary field is computed over the roots being summarized.

    `attempts_total` used to be a pass-through of whatever the constructor was
    handed, and the three construction sites hand it different populations: a
    client's LIFETIME counter (bumped at attempt start, while roots are
    appended at completion), the attempts on these roots, or a sum of lifetime
    counters across clients. A consumer computing amplification as
    `attempts_total / total` therefore got `retries_per_root + 1` from one site
    and an in-flight-inflated number from another, while `print_stats` printed
    the two side by side as if they agreed.
    """

    def _root(self, attempts: int) -> RootRequest:
        return RootRequest(
            attempts=[
                Request(
                    service="svc",
                    interval=TimeInterval(begin=i * 10, end=i * 10 + 10),
                    success=(i == attempts - 1),
                    drop_reason=DropReason.NONE if i == attempts - 1 else DropReason.DEADLINE,
                )
                for i in range(attempts)
            ]
        )

    def test_attempts_total_counts_the_summarized_roots_not_the_client_counter(self):
        # 17 attempts started lifetime-wide; 2 recorded on the one root here.
        metrics = Metrics(roots=[self._root(2)], attempts_total=17)
        summary = metrics.summary()

        self.assertEqual(summary.attempts_total, 2)
        self.assertAlmostEqual(summary.retries_per_root, 1.0)

    def test_both_populations_stay_readable_under_distinct_names(self):
        metrics = Metrics(roots=[self._root(2)], attempts_total=17)

        self.assertEqual(metrics.attempts_on_roots(), 2)
        self.assertEqual(metrics.client_attempts_total(), 17)


class TestTiedFinalAttempts(unittest.TestCase):
    def test_tied_final_attempts_use_completion_order(self):
        # Zero-backoff retry rejected synchronously: SERVER_FAILURE at t, then
        # QUEUE_FULL with begin == end == t. Both attempts end at the same
        # nanosecond; the true final outcome is the last APPENDED attempt, and
        # summary and CSV export must agree on it.
        root = RootRequest(
            attempts=[
                Request(
                    service="svc",
                    interval=TimeInterval(begin=0, end=s_to_ns(1)),
                    success=False,
                    drop_reason=DropReason.SERVER_FAILURE,
                ),
                Request(
                    service="svc",
                    interval=TimeInterval(begin=s_to_ns(1), end=s_to_ns(1)),
                    success=False,
                    drop_reason=DropReason.QUEUE_FULL,
                ),
            ]
        )
        metrics = Metrics(roots=[root], attempts_total=2)
        summary = metrics.summary()
        self.assertEqual(summary.dropped_queue, 1)
        self.assertEqual(summary.dropped_server_failure, 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "tied.csv")
            metrics.export_csv(path)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(sum(int(r["failure_queue_full"]) for r in rows), 1)
        self.assertEqual(sum(int(r["failure_server"]) for r in rows), 0)


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

    def test_non_decimal_granularity_keeps_exact_timepoints(self):
        # 1.5 ms buckets need FOUR decimals; rounding to three would serialize
        # alternating 1/2 ms gaps and corrupt downstream width inference.
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "out.csv")
            self._metrics().export_csv(path, granularity_s=0.0015)
            import pandas as pd

            df = pd.read_csv(path)
            self.assertEqual(df["timepoint"].nunique(), len(df))
            widths = df["timepoint"].diff().dropna().round(9).unique()
            self.assertEqual(list(widths), [0.0015])

    def test_sub_millisecond_buckets_keep_distinct_timepoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "out.csv")
            self._metrics().export_csv(path, granularity_s=0.0005)
            import pandas as pd

            df = pd.read_csv(path)
            self.assertEqual(df["timepoint"].nunique(), len(df))


class TestOutcomeCoverageGuard(unittest.TestCase):
    """A consumer table missing a non-success outcome must fail loudly.

    The guard is what turns "new RootOutcome silently booked as a server drop"
    into an import-time error, so it must reject an incomplete mapping and not
    compare a table against itself.
    """

    def test_complete_mapping_passes(self):
        require_outcome_coverage(FAILURE_OUTCOMES, owner="test")

    def test_missing_outcome_is_named(self):
        incomplete = dict(FAILURE_OUTCOMES)
        del incomplete[RootOutcome.CANCELLED]
        with self.assertRaises(RuntimeError) as ctx:
            require_outcome_coverage(incomplete, owner="tests.incomplete_table")
        self.assertIn("CANCELLED", str(ctx.exception))
        self.assertIn("tests.incomplete_table", str(ctx.exception))

    def test_success_needs_no_entry(self):
        no_success = {k: v for k, v in FAILURE_OUTCOMES.items() if k is not RootOutcome.SUCCESS}
        require_outcome_coverage(no_success, owner="test")


if __name__ == "__main__":
    unittest.main()
