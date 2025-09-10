import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from plotting.utils.fault_events import load_fault_events, plot_fault_events
from fault_events import FaultEventsTracker


def main(csv_file, output_file, fault_events_file=None):
    df = pd.read_csv(csv_file).sort_values(by="timepoint")

    df = df[:-1]

    t = df["timepoint"].astype(float).to_numpy()

    # Per-window completed requests
    completed = (df["success_root"] + df["failure_root"]).replace(0, np.nan)

    # Breakdown by reason (deadline, server, queue_full) as % of completed
    reason_cols = ["failure_deadline", "failure_server", "failure_queue_full"]
    reasons_pct = df[reason_cols].div(completed, axis=0) * 100.0
    reasons_pct = reasons_pct.fillna(0.0)

    # NEW: request amplification % (extra requests due to retries relative to root)
    # amplification_pct = (retries / root_requests) * 100
    root = df["root_requests"].replace(0, np.nan)
    retries = df["retries"]
    amplification_pct = (retries / root) * 100.0

    # Make the reason breakdown the TOP plot, and amplification the BOTTOM plot
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]}
    )

    # TOP: stacked breakdown by reason (as % of completed)
    ax_top.stackplot(
        t,
        *[reasons_pct[c].to_numpy() for c in reason_cols],
        labels=["deadline", "server", "queue_full"],
    )
    ax_top.legend(loc="upper left", ncol=3, frameon=True)
    ax_top.set_ylabel("Failure rate (%)")
    ax_top.set_xlim(left=0)
    ax_top.set_ylim(0, 100)
    ax_top.yaxis.set_major_formatter(PercentFormatter(100))
    ax_top.grid(True, which="both")
    ax_top.set_title("Failures (%) over Time by Reason Breakdown")

    # Fault events (top)
    if fault_events_file:
        fault_events = load_fault_events(fault_events_file)
        plt.sca(ax_top)
        plot_fault_events(fault_events)

    # BOTTOM: request amplification %
    ax_bottom.plot(t, amplification_pct, label="Request amplification")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Amplification (%)")
    ax_bottom.set_xlim(left=0)
    # Let y autoscale since amplification can exceed 100% in extreme cases
    ax_bottom.yaxis.set_major_formatter(PercentFormatter(100))
    ax_bottom.grid(True, which="both")
    ax_bottom.set_title("Request Amplification Over Time")
    ax_bottom.legend(loc="upper left", ncol=1, frameon=True)

    # Fault events (bottom)
    if fault_events_file:
        plt.sca(ax_bottom)
        plot_fault_events(fault_events)

    plt.tight_layout()
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot failure reason breakdown (top) and request amplification % (bottom) over time from CSV"
    )
    parser.add_argument("csv_file", help="Input CSV file path")
    parser.add_argument(
        "--output",
        "-o",
        default="reasons_and_request_amplification.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--fault-events", help="JSON file containing fault events (optional)"
    )
    args = parser.parse_args()
    main(args.csv_file, args.output, args.fault_events)
