import matplotlib.pyplot as plt

from plotting.utils.fault_events import load_fault_events, plot_fault_events
from plotting.utils.io import build_plot_parser, load_metrics


def main(csv_file, output_file, fault_events_file=None):
    df = load_metrics(csv_file)

    plt.figure(figsize=(10, 6))
    plt.plot(df["timepoint"], df["root_requests"], label="Root Requests", linewidth=2)
    plt.plot(df["timepoint"], df["retries"], label="Retries", linewidth=2)
    plt.plot(df["timepoint"], df["failure_root"], label="Failures", linewidth=2)

    if fault_events_file:
        fault_events = load_fault_events(fault_events_file)
        plot_fault_events(fault_events)

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
