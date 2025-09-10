import pandas as pd
import matplotlib.pyplot as plt
import argparse

from plotting.utils.fault_events import load_fault_events, plot_fault_events


def main(csv_file, output_file, fault_events_file=None):
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="timepoint")

    # remove the last point (no data)
    df = df[:-1]

    plt.figure(figsize=(10, 6))
    plt.plot(
        df["timepoint"],
        df["root_requests"],
        label="Root Requests",
        # marker="o",
        # markersize=5,
        linewidth=2,
    )

    plt.plot(
        df["timepoint"],
        df["retries"],
        label="Retries",
        # marker="^",
        # markersize=5,
        linewidth=2,
    )

    plt.plot(
        df["timepoint"],
        df["failure_root"],
        label="Failures",
        linewidth=2,
    )

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
    parser = argparse.ArgumentParser(description="Plot QPS over time from CSV")
    parser.add_argument("csv_file", help="Input CSV file path")
    parser.add_argument(
        "--output", "-o", default="qps_over_time.png", help="Output plot filename"
    )
    parser.add_argument(
        "--fault-events", help="JSON file containing fault events (optional)"
    )
    args = parser.parse_args()
    main(args.csv_file, args.output, args.fault_events)
