import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from msim.core import FAILURE_OUTCOMES
from msim.plotting.utils.fault_events import load_fault_events, plot_fault_events
from msim.plotting.utils.io import build_plot_parser, load_metrics, require_rows

# Derived from the authoritative outcome table so a new RootOutcome shows up
# here (stack + legend follow the CSV column order) without a manual edit.
REASON_COLUMNS = [reporting.csv_column for reporting in FAILURE_OUTCOMES.values()]


def completed_roots(df):
    """Roots completing in each bucket — the denominator of the reason mix."""
    return df["success_root"] + df["failure_root"]


def failure_reason_percentages(df):
    """Failure reasons as a % of the roots completing in each bucket.

    Buckets where NOTHING completed stay nan rather than 0%: a stalled window
    (every root still in flight) would otherwise render as the healthiest part
    of the figure. That is the same convention metrics.py keeps for empty
    buckets, and the one the amplification series below already follows.
    `stackplot` accepts nan and draws a gap — which on a stacked area is the
    same blank as a genuine 0%, hence the labelled no-data band in `main`.

    Returns the reason columns actually present and their percentages: a CSV
    written before a reason column existed is zero, not a crash, so older
    exports stay plottable.
    """
    reason_cols = [c for c in REASON_COLUMNS if c in df.columns]
    completed = completed_roots(df).replace(0, np.nan)
    return reason_cols, df[reason_cols].div(completed, axis=0) * 100.0


def main(csv_file, output_file, fault_events_file=None):
    df = load_metrics(csv_file)
    require_rows(df, csv_file)

    t = df["timepoint"].astype(float).to_numpy()

    reason_cols, reasons_pct = failure_reason_percentages(df)

    # Request amplification % (extra requests due to retries relative to root)
    root = df["root_requests"].replace(0, np.nan)
    amplification_pct = (df["retries"] / root) * 100.0

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]}
    )

    # Labels are DERIVED from reason_cols: matplotlib pulls labels lazily, so a
    # hand-written list that is shorter than the series list silently renders an
    # unlabelled band instead of raising.
    reason_labels = [c.removeprefix("failure_") for c in reason_cols]
    ax_top.stackplot(
        t,
        *[reasons_pct[c].to_numpy() for c in reason_cols],
        labels=reason_labels,
    )

    # Buckets with no completions are nan above, i.e. blank — indistinguishable
    # from a genuine 0% on a stacked area. Shade them so a stall reads as
    # "no data", not as a flawless window.
    no_completions = completed_roots(df).to_numpy() == 0
    if no_completions.any():
        ax_top.fill_between(
            t,
            0,
            100,
            where=no_completions,
            step="post",
            color="grey",
            alpha=0.25,
            hatch="//",
            label="no completions",
        )

    # Fault events (top) — BEFORE the legend, otherwise the shaded windows are
    # drawn but never named in it.
    fault_events = load_fault_events(fault_events_file)
    plot_fault_events(fault_events, ax=ax_top)

    ax_top.legend(loc="upper left", ncol=len(reason_labels), frameon=True)
    ax_top.set_ylabel("Failure rate (%)")
    ax_top.set_xlim(left=0)
    ax_top.set_ylim(0, 100)
    ax_top.yaxis.set_major_formatter(PercentFormatter(100))
    ax_top.grid(True, which="both")
    ax_top.set_title("Failures (%) over Time by Reason Breakdown")

    # BOTTOM: request amplification %
    ax_bottom.plot(t, amplification_pct, label="Request amplification")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Amplification (%)")
    ax_bottom.set_xlim(left=0)

    ax_bottom.yaxis.set_major_formatter(PercentFormatter(100))
    ax_bottom.grid(True, which="both")
    ax_bottom.set_title("Request Amplification Over Time")

    # Fault events (bottom), again before the legend that has to name them.
    plot_fault_events(fault_events, ax=ax_bottom)
    ax_bottom.legend(loc="upper left", ncol=1, frameon=True)

    plt.tight_layout()
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = build_plot_parser(
        "Plot failure reason breakdown (top) and request amplification % (bottom) over time from CSV",
        "reasons_and_request_amplification.png",
    )
    args = parser.parse_args()
    main(args.csv_file, args.output, args.fault_events)
