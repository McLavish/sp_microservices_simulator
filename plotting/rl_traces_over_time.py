"""Plot per-step policy traces from rl.experiments.compare_policies.

Top panel: the attempt-timeout knob each policy chose over the episode.
Bottom panel: the per-window success rate. The fault-events overlay only lines
up for the fixed-window (`basic`) scenario; randomized scenarios draw different
windows per seed and export no fault JSON.
"""

import matplotlib.pyplot as plt
import pandas as pd

from plotting.utils.fault_events import load_fault_events, plot_fault_events
from plotting.utils.io import build_plot_parser, require_rows


def _step_seconds(df: pd.DataFrame) -> float:
    if "step_duration_s" in df.columns:
        return float(df["step_duration_s"].iloc[0])
    print("warning: trace CSV has no step_duration_s column (legacy); assuming 1.0 s/step")
    return 1.0


def _select_seed(df: pd.DataFrame, seed):
    """Rows for `seed`, or a message naming the seeds the CSV does carry.

    `require_rows` runs before the filter, so it cannot catch a selector that
    matched nothing: unguarded, a mistyped --seed either died with a raw pandas
    IndexError in `_step_seconds` or wrote a blank figure and exited 0.
    """
    if seed is None:
        seed = int(df["seed"].iloc[0])
    selected = df[df["seed"] == seed]
    if selected.empty:
        available = ", ".join(str(s) for s in sorted(df["seed"].unique()))
        raise SystemExit(f"no rows for --seed {seed} in the CSV (seeds present: {available})")
    return seed, selected


def _select_policies(df: pd.DataFrame, policies, seed):
    available = sorted(df["policy"].unique())
    if not policies:
        return available
    missing = [name for name in policies if name not in available]
    if missing:
        raise SystemExit(
            f"no rows for --policies {', '.join(missing)} at seed {seed} "
            f"(policies present: {', '.join(available)})"
        )
    return policies


def main(csv_file, output_file, policies=None, seed=None, fault_events_file=None):
    df = pd.read_csv(csv_file)
    require_rows(df, csv_file)
    seed, df = _select_seed(df, seed)
    names = _select_policies(df, policies, seed)
    step_s = _step_seconds(df)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for name in names:
        sub = df[df["policy"] == name].sort_values("step")
        ax_top.step(sub["step"] * step_s, sub["attempt_timeout_ms"], where="post", label=name)
        ax_bottom.plot(sub["step"] * step_s, sub["success_rate"], label=name)

    ax_top.set_ylabel("Attempt timeout (ms)")
    ax_top.set_ylim(bottom=0)
    ax_top.grid(True)
    ax_top.set_title(f"Policy knob and success rate over time (seed {seed})")

    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Success rate")
    ax_bottom.set_ylim(0, 1.05)
    ax_bottom.grid(True)

    # Overlay before the legend so the shaded windows are named in it. Only the
    # top panel labels them: the bottom panel carries no legend of its own and
    # the axes share an x scale, so one entry describes both.
    fault_events = load_fault_events(fault_events_file)
    plot_fault_events(fault_events, ax=ax_top)
    plot_fault_events(fault_events, ax=ax_bottom, label=False)

    ax_top.legend(loc="upper left", ncol=2, frameon=True)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    parser = build_plot_parser(
        "Plot per-step policy traces (knob choices + success rate) from compare_policies",
        "rl_traces_over_time.png",
    )
    parser.add_argument("--seed", type=int, default=None, help="Eval seed to plot (default: first)")
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help="Comma-separated policy names to include (default: all)",
    )
    args = parser.parse_args()
    selected = [p.strip() for p in args.policies.split(",")] if args.policies else None
    main(args.csv_file, args.output, selected, args.seed, args.fault_events)
