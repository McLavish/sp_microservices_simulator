import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from plotting.utils.fault_events import load_fault_events, plot_fault_events
from plotting.utils.io import build_plot_parser, load_metrics, require_rows


def main(csv_file, output_file, fault_events_file=None):
    df = load_metrics(csv_file)
    require_rows(df, csv_file)

    t = df["timepoint"].astype(float).to_numpy()

    # Per-window completed requests
    completed = (df["success_root"] + df["failure_root"]).replace(0, np.nan)

    # Breakdown by reason as % of completed
    reason_cols = [
        "failure_deadline",
        "failure_server",
        "failure_queue_full",
        "failure_circuit_open",
        "failure_cancelled",
    ]
    reasons_pct = (df[reason_cols].div(completed, axis=0) * 100.0).fillna(0.0)

    # Request amplification % (extra requests due to retries relative to root)
    root = df["root_requests"].replace(0, np.nan)
    amplification_pct = (df["retries"] / root) * 100.0

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]}
    )

    ax_top.stackplot(
        t,
        *[reasons_pct[c].to_numpy() for c in reason_cols],
        labels=["deadline", "server", "queue_full", "circuit_open"],
    )
    ax_top.legend(loc="upper left", ncol=4, frameon=True)
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
    parser = build_plot_parser(
        "Plot failure reason breakdown (top) and request amplification % (bottom) over time from CSV",
        "reasons_and_request_amplification.png",
    )
    args = parser.parse_args()
    main(args.csv_file, args.output, args.fault_events)
