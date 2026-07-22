"""Shared helpers for the CSV-driven plotting scripts."""

import argparse

import pandas as pd


def load_metrics(csv_file: str) -> pd.DataFrame:
    """Load a metrics CSV, sorted by time and trimmed to the experiment window.

    Experiments drain in-flight work after the measured window (``sim.run()``
    with no ``until``), which appends trailing buckets in which no new roots
    started (``root_requests == 0``). Those drain-phase rows are dropped so the
    plot ends at the experiment's end time. This replaces the old positional
    ``df[:-1]`` slice, which only ever removed a single trailing bucket.
    """
    df = pd.read_csv(csv_file).sort_values(by="timepoint").reset_index(drop=True)
    if "root_requests" in df.columns and len(df):
        started = df["root_requests"].to_numpy()
        last = len(started) - 1
        while last >= 0 and started[last] == 0:
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
