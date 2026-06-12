"""Goodput-vs-amplification Pareto plot from rl.experiments.constrained_retry.

X: mean edge retry amplification (constraint cost). Y: mean base objective J.
One marker per (arm, budget); dashed vertical lines mark the budgets.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

ARM_STYLES = {
    "unconstrained_ppo": {"marker": "s", "color": "tab:red"},
    "constrained_ppo": {"marker": "o", "color": "tab:blue"},
    "retry_budget": {"marker": "^", "color": "tab:green"},
}


def main(csv_file, output_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(9, 6))
    for arm, style in ARM_STYLES.items():
        sub = df[df["arm"] == arm]
        if sub.empty:
            continue
        plt.errorbar(
            sub["cost_mean"],
            sub["J_mean"],
            yerr=sub["J_std"],
            linestyle="none",
            capsize=3,
            label=arm,
            marker=style["marker"],
            color=style["color"],
        )
        for _, row in sub.iterrows():
            if pd.notna(row["budget"]) and str(row["budget"]) != "":
                plt.annotate(
                    f"b={row['budget']}",
                    (row["cost_mean"], row["J_mean"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )

    budgets = sorted(
        {float(b) for b in df["budget"].dropna() if str(b) != ""},
    )
    for budget in budgets:
        plt.axvline(budget, color="gray", linestyle="--", alpha=0.4)

    plt.xlabel("Edge retry amplification (cost, retries per edge root)")
    plt.ylabel("Base objective J (mean per episode)")
    plt.title("Goodput vs retry amplification: constrained-RL comparison")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pareto plot for the constrained-retry study")
    parser.add_argument("csv_file", help="rl_constrained_comparison.csv path")
    parser.add_argument("--output", "-o", default="rl_pareto.png")
    args = parser.parse_args()
    main(args.csv_file, args.output)
