import matplotlib.pyplot as plt

from plotting.utils.fault_events import load_fault_events, plot_fault_events
from plotting.utils.io import build_plot_parser, load_metrics, require_rows


def main(csv_file, output_file, fault_events_file=None):
    df = load_metrics(csv_file)
    require_rows(df, csv_file)

    t = df["timepoint"].astype(float)

    plt.figure(figsize=(9, 5))
    for col in ("p50", "p90", "p95", "p99", "max"):
        plt.plot(t, df[col], label=col)

    if fault_events_file:
        fault_events = load_fault_events(fault_events_file)
        plot_fault_events(fault_events)

    plt.xlabel("Time (s)")
    plt.ylabel("Latency (ms)")
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.legend()

    plt.title("Latency over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = build_plot_parser("Plot latencies over time from CSV", "latencies_over_time.png")
    args = parser.parse_args()
    main(args.csv_file, args.output, args.fault_events)
