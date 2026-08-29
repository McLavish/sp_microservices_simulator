import matplotlib.pyplot as plt

from msim.plotting.utils.fault_events import load_fault_events, plot_fault_events
from msim.plotting.utils.io import build_plot_parser, load_metrics, require_rows


def _bucket_width_s(df) -> float:
    """Bucket width inferred from consecutive timepoints (default 1.0 s)."""
    if len(df) > 1:
        width = float(df["timepoint"].diff().median())
        if width > 0:
            return width
    print("warning: single-bucket CSV, cannot infer bucket width; assuming 1.0 s (QPS = count)")
    return 1.0


def main(csv_file, output_file, fault_events_file=None):
    df = load_metrics(csv_file)
    require_rows(df, csv_file)

    # Counts per bucket divided by the bucket width: the y axis is a true
    # rate (QPS) at any export granularity, not just the default 1 s.
    width = _bucket_width_s(df)
    plt.figure(figsize=(10, 6))
    plt.plot(df["timepoint"], df["root_requests"] / width, label="Root Requests", linewidth=2)
    plt.plot(df["timepoint"], df["retries"] / width, label="Retries", linewidth=2)
    plt.plot(df["timepoint"], df["failure_root"] / width, label="Failures", linewidth=2)

    plot_fault_events(load_fault_events(fault_events_file), ax=plt.gca())

    plt.xlabel("Time (s)")
    plt.ylabel("QPS")
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.legend()

    plt.title("QPS over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    parser = build_plot_parser("Plot QPS over time from CSV", "qps_over_time.png")
    args = parser.parse_args()
    main(args.csv_file, args.output, args.fault_events)
