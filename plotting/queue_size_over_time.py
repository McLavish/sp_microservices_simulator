import matplotlib.pyplot as plt
import pandas as pd

from plotting.utils.fault_events import load_fault_events, plot_fault_events
from plotting.utils.io import build_plot_parser, load_metrics, require_rows


def main(csv_file, output_file, fault_events_file=None):
    df = load_metrics(csv_file)
    require_rows(df, csv_file)

    t = df["timepoint"].astype(float).to_numpy()

    q = pd.to_numeric(df["queue_avg_at_attempt_end"], errors="coerce").to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(t, q, label="Avg queue size (attempt end)", linewidth=2)

    plot_fault_events(load_fault_events(fault_events_file), ax=plt.gca())

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
    parser = build_plot_parser("Plot queue size over time from CSV", "queue_size_over_time.png")
    args = parser.parse_args()
    main(args.csv_file, args.output, args.fault_events)
