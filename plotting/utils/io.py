"""Shared helpers for the CSV-driven plotting scripts."""

import argparse

import pandas as pd


def load_metrics(csv_file: str) -> pd.DataFrame:
    """Load a metrics CSV, sorted by time and trimmed of the idle tail.

    Trailing buckets with NO activity at all (no starts and no completions)
    are dropped. Buckets that only contain completions are kept: a request
    starting in one bucket and completing in the next carries real
    completion/latency data even when nothing new started there — trimming by
    ``root_requests`` alone (the previous rule) silently discarded it.
    """
    df = pd.read_csv(csv_file).sort_values(by="timepoint").reset_index(drop=True)
    if "root_requests" in df.columns and len(df):
        activity = df["root_requests"].to_numpy()
        # `completed` is attempt-end bucketed and `success_root`/`failure_root`
        # are ROOT-completion bucketed; a root cancelled while backing off
        # completes after its last attempt ended, so it can land in a bucket
        # where completed == 0 but failure_root == 1. Counting only the first
        # two would trim exactly the rows metrics.py widened the range to keep.
        for column in ("completed", "success_root", "failure_root"):
            if column in df.columns:
                activity = activity + df[column].to_numpy()
        last = len(activity) - 1
        while last >= 0 and activity[last] == 0:
            last -= 1
        df = df.iloc[: last + 1]
    return df


def require_rows(df: pd.DataFrame, csv_file: str) -> None:
    """Exit with a clear message on header-only CSVs (a legitimate
    Metrics.export_csv product when an experiment completed zero attempts)."""
    if df.empty:
        raise SystemExit(f"no data rows in {csv_file} (the experiment completed zero attempts)")


def build_plot_parser(description: str, default_output: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("csv_file", help="Input CSV file path")
    parser.add_argument("--output", "-o", default=default_output, help="Output plot filename")
    parser.add_argument("--fault-events", help="JSON file containing fault events (optional)")
    return parser
