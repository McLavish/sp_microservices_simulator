import argparse
import pandas as pd
import matplotlib.pyplot as plt
from plotting.utils.fault_events import load_fault_events, plot_fault_events


def main(csv_file, output_file, fault_events_file=None):
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="timepoint")

    df = df[:-1]

    t = df["timepoint"].astype(float)

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["p50"], label="p50")
    plt.plot(t, df["p90"], label="p90")
    plt.plot(t, df["p95"], label="p95")
    plt.plot(t, df["p99"], label="p99")
    plt.plot(t, df["Max"], label="max")

    if fault_events_file:
        fault_events = load_fault_events(fault_events_file)
        plot_fault_events(fault_events)

    plt.xlabel("Time (s)")
    plt.ylabel("Latency (ms)")
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # plt.legend(title="Latency stats")
    plt.legend()

    plt.title("Latency over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latencies over time from CSV")
    parser.add_argument("csv_file", help="Input CSV file path")
    parser.add_argument(
        "--output", "-o", default="latencies_over_time.png", help="Output plot filename"
    )
    parser.add_argument(
        "--fault-events", help="JSON file containing fault events (optional)"
    )
    args = parser.parse_args()
    main(args.csv_file, args.output, args.fault_events)
