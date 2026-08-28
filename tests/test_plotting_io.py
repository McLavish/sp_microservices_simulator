import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: these tests render real figures

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from metrics import Metrics  # noqa: E402
from plotting.qps_over_time import _bucket_width_s  # noqa: E402


class TestQpsBucketWidth(unittest.TestCase):
    def test_width_inferred_from_timepoints(self):
        df = pd.DataFrame({"timepoint": [0.0, 0.5, 1.0, 1.5]})
        self.assertAlmostEqual(_bucket_width_s(df), 0.5)

    def test_single_row_defaults_to_one_second(self):
        df = pd.DataFrame({"timepoint": [0.0]})
        self.assertEqual(_bucket_width_s(df), 1.0)


class TestLoadMetricsTrim(unittest.TestCase):
    def test_completion_only_final_bucket_is_kept(self):
        df = pd.DataFrame(
            {
                "timepoint": [0.0, 1.0, 2.0],
                "root_requests": [5, 0, 0],
                "completed": [0, 5, 0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.csv")
            df.to_csv(path, index=False)
            from plotting.utils.io import load_metrics

            out = load_metrics(path)
        # The idle bucket at t=2 is trimmed; the completion-only bucket at t=1
        # carries real latency/failure data and must survive.
        self.assertEqual(list(out["timepoint"]), [0.0, 1.0])

    def test_root_completion_only_final_bucket_is_kept(self):
        """`completed` alone does not cover the trim rule. A root cancelled while
        backing off completes AFTER its last attempt ended, so its bucket has
        `completed == 0` and only a root-completion column set — the row
        metrics.py widened its bucket range to preserve."""
        from plotting.utils.io import load_metrics

        for column in ("success_root", "failure_root"):
            with self.subTest(column=column):
                df = pd.DataFrame(
                    {
                        "timepoint": [0.0, 1.0, 2.0],
                        "root_requests": [1, 0, 0],
                        "completed": [1, 0, 0],
                        "success_root": [0, 0, 0],
                        "failure_root": [0, 0, 0],
                    }
                )
                df.loc[2, column] = 1
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "t.csv")
                    df.to_csv(path, index=False)
                    out = load_metrics(path)
                self.assertEqual(list(out["timepoint"]), [0.0, 1.0, 2.0])


class TestHeaderOnlyCsv(unittest.TestCase):
    def _header_only_csv(self, tmp):
        path = os.path.join(tmp, "empty.csv")
        Metrics(roots=[], attempts_total=0).export_csv(path)  # writes header only
        return path

    def test_require_rows_raises_system_exit(self):
        from plotting.utils.io import load_metrics, require_rows

        with tempfile.TemporaryDirectory() as tmp:
            path = self._header_only_csv(tmp)
            df = load_metrics(path)
            with self.assertRaises(SystemExit):
                require_rows(df, path)

    def test_failures_over_time_exits_cleanly_not_typeerror(self):
        import plotting.failures_over_time as failures_over_time

        with tempfile.TemporaryDirectory() as tmp:
            path = self._header_only_csv(tmp)
            with self.assertRaises(SystemExit):
                failures_over_time.main(path, str(Path(tmp) / "out.png"))


def _legend_labels(ax) -> list[str]:
    """Legend entry texts for `ax`; empty when it has no legend at all."""
    legend = ax.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def _stall_frame() -> pd.DataFrame:
    """Three buckets; nothing completes in the middle one (all roots in flight)."""
    return pd.DataFrame(
        {
            "timepoint": [0.0, 1.0, 2.0],
            "root_requests": [4, 4, 4],
            "retries": [0, 2, 2],
            "success_root": [1, 0, 4],
            "failure_root": [1, 0, 0],
            "failure_deadline": [1, 0, 0],
            "failure_server": [0, 0, 0],
        }
    )


class TestFailureReasonPercentages(unittest.TestCase):
    def test_zero_completion_bucket_stays_nan(self):
        from plotting.failures_over_time import failure_reason_percentages

        cols, pct = failure_reason_percentages(_stall_frame())
        self.assertEqual(cols, ["failure_deadline", "failure_server"])
        self.assertAlmostEqual(pct["failure_deadline"].iloc[0], 50.0)
        # t=1 completed nothing: a 0% reading would paint the stalled window as
        # the healthiest part of the figure.
        self.assertTrue(np.isnan(pct["failure_deadline"].iloc[1]))
        self.assertTrue(np.isnan(pct["failure_server"].iloc[1]))
        # t=2 completed four roots, none of them failures: a real 0%.
        self.assertAlmostEqual(pct["failure_deadline"].iloc[2], 0.0)

    def test_missing_reason_column_is_skipped(self):
        from plotting.failures_over_time import failure_reason_percentages

        df = _stall_frame().drop(columns=["failure_server"])
        cols, pct = failure_reason_percentages(df)
        self.assertEqual(cols, ["failure_deadline"])
        self.assertEqual(list(pct.columns), ["failure_deadline"])

    def _render(self, df, tmp):
        import plotting.failures_over_time as failures_over_time

        csv_path = os.path.join(tmp, "metrics.csv")
        df.to_csv(csv_path, index=False)
        out = os.path.join(tmp, "out.png")
        plt.close("all")
        failures_over_time.main(csv_path, out)  # stackplot must accept nan
        self.assertTrue(os.path.exists(out))
        return _legend_labels(plt.gcf().axes[0])

    def test_stalled_bucket_is_labelled_no_completions(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = self._render(_stall_frame(), tmp)
        plt.close("all")
        self.assertIn("no completions", labels)

    def test_fully_completing_run_has_no_no_data_band(self):
        df = _stall_frame()
        df.loc[1, "success_root"] = 4  # nothing is in flight across a bucket edge
        with tempfile.TemporaryDirectory() as tmp:
            labels = self._render(df, tmp)
        plt.close("all")
        self.assertNotIn("no completions", labels)


def _fault_event(event_type, params, start=1.0, end=2.0):
    return {
        "event_type": event_type,
        "start_time_s": start,
        "end_time_s": end,
        "parameters": params,
    }


class TestFaultEventOverlay(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close("all")

    def test_unknown_event_type_raises(self):
        from plotting.utils.fault_events import plot_fault_events

        with self.assertRaises(SystemExit) as caught:
            plot_fault_events([_fault_event("thread_pool_exhaustion", {})], ax=self.ax)
        self.assertIn("thread_pool_exhaustion", str(caught.exception))

    def test_renamed_parameter_key_raises(self):
        from plotting.utils.fault_events import plot_fault_events

        # Producer-side rename of failure_rate: used to render "Partial Failure (0%)".
        with self.assertRaises(SystemExit) as caught:
            plot_fault_events([_fault_event("partial_failure", {"p_fail": 0.5})], ax=self.ax)
        self.assertIn("failure_rate", str(caught.exception))

    def test_identical_events_share_one_legend_entry(self):
        from plotting.utils.fault_events import plot_fault_events

        events = [
            _fault_event("load_spike", {"rps_multiplier": 3.0}, 1.0, 2.0),
            _fault_event("load_spike", {"rps_multiplier": 3.0}, 4.0, 5.0),
            _fault_event("load_spike", {"rps_multiplier": 2.5}, 7.0, 8.0),
        ]
        plot_fault_events(events, ax=self.ax)
        labels = [text.get_text() for text in self.ax.legend().get_texts()]
        self.assertEqual(labels.count("Load Spike (x3.0)"), 1)
        self.assertEqual(labels.count("Load Spike (x2.5)"), 1)
        self.assertEqual(len(self.ax.patches), 3)  # every window is still drawn

    def test_label_false_draws_the_span_without_a_legend_entry(self):
        from plotting.utils.fault_events import plot_fault_events

        self.ax.plot([0, 1], [0, 1], label="series")
        plot_fault_events(
            [_fault_event("partial_failure", {"failure_rate": 0.5})], ax=self.ax, label=False
        )
        labels = [text.get_text() for text in self.ax.legend().get_texts()]
        self.assertEqual(labels, ["series"])
        self.assertEqual(len(self.ax.patches), 1)

    def test_failures_plot_names_the_window_in_both_legends(self):
        import plotting.failures_over_time as failures_over_time

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "stall.csv")
            _stall_frame().to_csv(csv_path, index=False)
            json_path = os.path.join(tmp, "faults.json")
            with open(json_path, "w") as f:
                json.dump([_fault_event("partial_failure", {"failure_rate": 0.5})], f)

            plt.close("all")
            failures_over_time.main(csv_path, os.path.join(tmp, "out.png"), json_path)
            fig = plt.gcf()
            for ax in fig.axes:
                self.assertIn("Partial Failure (50%)", _legend_labels(ax))
            plt.close("all")


class TestLatencyDistributionArguments(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_non_positive_median_exits(self):
        from plotting.latency_distribution import plot_lognormal_latency

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.png")
            for median in (0.0, -5.0, float("nan")):
                with self.assertRaises(SystemExit):
                    plot_lognormal_latency(median, 0.5, out)
            self.assertFalse(os.path.exists(out))

    def test_non_positive_sigma_exits(self):
        from plotting.latency_distribution import plot_lognormal_latency

        # sigma=0 is the silent variant: scipy returns nan without a warning.
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.png")
            with self.assertRaises(SystemExit):
                plot_lognormal_latency(20.0, 0.0, out)
            self.assertFalse(os.path.exists(out))

    def test_scale_is_the_median(self):
        from plotting.latency_distribution import plot_lognormal_latency

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.png")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                plot_lognormal_latency(20.0, 0.5, out)
            self.assertIn("p50 (median): 20.00 ms", printed.getvalue())
            self.assertTrue(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
