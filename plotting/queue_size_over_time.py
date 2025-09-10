import argparse
import pandas as pd
import matplotlib.pyplot as plt

from plotting.utils.fault_events import load_fault_events, plot_fault_events


def main(csv_file, output_file, fault_events_file=None):
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="timepoint")

    df = df[:-1]

    t = df["timepoint"].astype(float).to_numpy()

    q = pd.to_numeric(df["queue_avg_at_attempt_end"], errors="coerce").to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(t, q, label="Avg queue size (attempt end)", linewidth=2)

    if fault_events_file:
        fault_events = load_fault_events(fault_events_file)
        plot_fault_events(fault_events)

    plt.xlabel("Time (s)")
    plt.ylabel("Avg Queue Size")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.title("Queue Size over Time")
    plt.tight_layout()
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot queue size over time from CSV")
    parser.add_argument("csv_file", help="Input CSV file path")
    parser.add_argument(
        "--output",
        "-o",
        default="queue_size_over_time.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--fault-events", help="JSON file containing fault events (optional)"
    )
    args = parser.parse_args()
    main(args.csv_file, args.output, args.fault_events)
