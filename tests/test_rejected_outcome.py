"""`DropReason.REJECTED`: an outcome most scenarios cannot produce.

It is booked like every other failure -- by the request's FINAL attempt, in the
summary and in the CSV alike -- but its CSV column is optional: absent from the
file of a run in which nothing was rejected, so that every scenario without
admission control exports exactly what it always did.
"""

import contextlib
import io
import tempfile
import unittest
from enum import Enum, auto
from pathlib import Path
from unittest import mock

import pandas as pd

from msim import core
from msim.core import (
    FAILURE_OUTCOMES,
    DropReason,
    Request,
    RootOutcome,
    RootRequest,
    TimeInterval,
    classify_root,
    ms_to_ns,
    require_drop_reason_coverage,
)
from msim.metrics import Metrics

FIXED_HEADER = [
    "timepoint",
    "root_requests",
    "retries",
    "success_root",
    "completed",
    "failure_root",
    "failure_retry",
    "failure_queue_full",
    "failure_deadline",
    "failure_server",
    "failure_circuit_open",
    "failure_cancelled",
    "total_request",
    "total_failure",
    "p50",
    "p90",
    "p95",
    "p99",
    "p99.9",
    "p99.99",
    "min",
    "max",
    "queue_avg_at_attempt_end",
]


def _root(*reasons: DropReason) -> RootRequest:
    """A finished root with one attempt per reason, 10 ms apart; NONE succeeds."""
    root = RootRequest()
    for i, reason in enumerate(reasons):
        root.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=ms_to_ns(10 * i), end=ms_to_ns(10 * i + 5)),
                success=reason is DropReason.NONE,
                drop_reason=reason,
            )
        )
    root.done = True
    root.completed_at = root.attempts[-1].interval.end
    return root


def _export(roots: list[RootRequest], **kwargs) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "out.csv")
        Metrics(roots=roots, attempts_total=sum(len(r.attempts) for r in roots)).export_csv(
            path, **kwargs
        )
        return pd.read_csv(path)


WITHOUT = [
    _root(DropReason.NONE),
    _root(DropReason.SERVER_FAILURE, DropReason.NONE),
    _root(DropReason.SERVER_FAILURE, DropReason.DEADLINE),
    _root(DropReason.QUEUE_FULL),
]
WITH = [
    *WITHOUT,
    _root(DropReason.SERVER_FAILURE, DropReason.REJECTED),  # the retry was refused
    _root(DropReason.REJECTED),  # refused outright
    _root(DropReason.REJECTED, DropReason.NONE),  # refused, then served
]


class TestRejectedIsBookedByTheFinalAttempt(unittest.TestCase):
    def test_classification(self):
        self.assertIs(classify_root(_root(DropReason.REJECTED)), RootOutcome.REJECTED)
        # A request whose last retry was refused is a rejected request, whatever
        # its earlier attempts failed with ...
        self.assertIs(
            classify_root(_root(DropReason.SERVER_FAILURE, DropReason.REJECTED)),
            RootOutcome.REJECTED,
        )
        # ... and one that was refused first and then failed otherwise is not.
        self.assertIs(
            classify_root(_root(DropReason.REJECTED, DropReason.DEADLINE)), RootOutcome.DEADLINE
        )
        self.assertIs(
            classify_root(_root(DropReason.REJECTED, DropReason.NONE)), RootOutcome.SUCCESS
        )

    def test_summary(self):
        summary = Metrics(roots=WITH, attempts_total=0).summary()
        self.assertEqual(summary.dropped_rejected, 2)
        self.assertEqual(summary.dropped_server_failure, 0)  # not booked as the server's
        self.assertEqual(summary.succeeded, 3)
        counted = summary.succeeded + sum(
            getattr(summary, reporting.summary_field) for reporting in FAILURE_OUTCOMES.values()
        )
        self.assertEqual(counted, summary.total)

    def test_summary_without_it_is_zero(self):
        self.assertEqual(Metrics(roots=WITHOUT, attempts_total=0).summary().dropped_rejected, 0)


class TestTheCsvColumnIsOptional(unittest.TestCase):
    def test_no_rejection_no_column(self):
        """The header of a run without rejections is the fixed one, to the name."""
        df = _export(WITHOUT)
        self.assertEqual(list(df.columns), FIXED_HEADER)

    def test_a_rejection_appends_the_column_after_the_fixed_header(self):
        df = _export(WITH)
        self.assertEqual(list(df.columns), [*FIXED_HEADER, "failure_rejected"])
        self.assertEqual(int(df["failure_rejected"].sum()), 2)

    def test_the_fixed_columns_do_not_depend_on_the_optional_one(self):
        """Same roots, with and without the forced column: identical fixed block."""
        plain = _export(WITHOUT)
        forced = _export(WITHOUT, include_optional=True)
        self.assertEqual(list(forced.columns), [*FIXED_HEADER, "failure_rejected"])
        self.assertEqual(int(forced["failure_rejected"].sum()), 0)
        pd.testing.assert_frame_equal(forced[FIXED_HEADER], plain)

    def test_forcing_gives_runs_with_and_without_one_header(self):
        self.assertEqual(
            list(_export(WITHOUT, include_optional=True).columns),
            list(_export(WITH, include_optional=True).columns),
        )

    def test_an_empty_export_follows_the_same_rule(self):
        self.assertEqual(list(_export([]).columns), FIXED_HEADER)
        self.assertEqual(
            list(_export([], include_optional=True).columns), [*FIXED_HEADER, "failure_rejected"]
        )

    def test_the_reasons_sum_to_the_failed_roots(self):
        df = _export(WITH)
        reasons = [reporting.csv_column for reporting in FAILURE_OUTCOMES.values()]
        self.assertEqual(int(df[reasons].sum().sum()), int(df["failure_root"].sum()))
        summary = Metrics(roots=WITH, attempts_total=0).summary()
        for reporting in FAILURE_OUTCOMES.values():
            with self.subTest(column=reporting.csv_column):
                self.assertEqual(
                    int(df[reporting.csv_column].sum()), getattr(summary, reporting.summary_field)
                )

    def test_a_rejected_retry_is_a_failed_retry(self):
        # WITHOUT has one failed retry (the DEADLINE one); WITH adds the refused one.
        self.assertEqual(int(_export(WITHOUT)["failure_retry"].sum()), 1)
        self.assertEqual(int(_export(WITH)["failure_retry"].sum()), 2)


class TestPrintStats(unittest.TestCase):
    def _printed(self, roots: list[RootRequest]) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            Metrics(roots=roots, attempts_total=0).print_stats()
        return out.getvalue()

    def test_an_optional_outcome_that_did_not_occur_has_no_line(self):
        printed = self._printed(WITHOUT)
        self.assertNotIn("Rejected", printed)
        # The fixed outcomes are printed even at zero, as before.
        self.assertIn("Dropped (Circuit Open): 0", printed)
        self.assertIn("Cancelled (Abandoned Upstream): 0", printed)

    def test_an_optional_outcome_that_occurred_is_printed(self):
        self.assertIn("Rejected (Admission): 2", self._printed(WITH))


class _NewReason(Enum):
    THROTTLED = auto()


class _ExtendedDropReason:
    """`core.DropReason` with one member added (an enum with members cannot be subclassed)."""

    NONE = DropReason.NONE

    def __iter__(self):
        yield from DropReason
        yield _NewReason.THROTTLED


class TestDropReasonCoverageGuard(unittest.TestCase):
    """A drop reason with no outcome would be booked as a server failure, silently."""

    def test_the_core_table_is_complete(self):
        require_drop_reason_coverage(core._OUTCOME_BY_DROP_REASON, owner="test")

    def test_a_missing_reason_is_named(self):
        incomplete = dict(core._OUTCOME_BY_DROP_REASON)
        del incomplete[DropReason.REJECTED]
        with self.assertRaises(RuntimeError) as ctx:
            require_drop_reason_coverage(incomplete, owner="tests.incomplete_table")
        self.assertIn("REJECTED", str(ctx.exception))
        self.assertIn("tests.incomplete_table", str(ctx.exception))

    def test_none_needs_no_entry(self):
        self.assertNotIn(DropReason.NONE, core._OUTCOME_BY_DROP_REASON)

    def test_a_new_enum_member_fails_the_core_table(self):
        with (
            mock.patch.object(core, "DropReason", _ExtendedDropReason()),
            self.assertRaises(RuntimeError) as ctx,
        ):
            require_drop_reason_coverage(
                core._OUTCOME_BY_DROP_REASON, owner="core._OUTCOME_BY_DROP_REASON"
            )
        self.assertIn("THROTTLED", str(ctx.exception))
        self.assertNotIn("DEADLINE", str(ctx.exception))

    def test_every_outcome_a_reason_maps_to_is_reported(self):
        for reason, outcome in core._OUTCOME_BY_DROP_REASON.items():
            with self.subTest(reason=reason.name):
                self.assertIn(outcome, FAILURE_OUTCOMES)


if __name__ == "__main__":
    unittest.main()
