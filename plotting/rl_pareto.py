"""Goodput-vs-amplification Pareto plot from rl.experiments.constrained_retry.

X: mean edge retry amplification (constraint cost). Y: mean base objective J.
One marker per (arm, budget); dashed vertical lines mark the budgets. When the
CSV carries bystander data (multichain scenario), a second panel shows the
bystander chain's success rate against the same cost axis — the externality
the retry budget exists to bound.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

ARM_STYLES = {
    "unconstrained_ppo": {"marker": "s", "color": "tab:red"},
    "constrained_ppo": {"marker": "o", "color": "tab:blue"},
    "retry_budget": {"marker": "^", "color": "tab:green"},
}


def _draw_panel(ax, df, y_col, y_err_col, ylabel, title):
    for arm, style in ARM_STYLES.items():
        sub = df[df["arm"] == arm]
        sub = sub[pd.notna(sub[y_col])]
        if sub.empty:
            continue
        ax.errorbar(
            sub["cost_mean"],
            sub[y_col],
            yerr=sub[y_err_col] if y_err_col else None,
            linestyle="none",
            capsize=3,
            label=arm,
            marker=style["marker"],
            color=style["color"],
        )
        for _, row in sub.iterrows():
            if pd.notna(row["budget"]) and str(row["budget"]) != "":
                ax.annotate(
                    f"b={row['budget']}",
                    (row["cost_mean"], row[y_col]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )

    budgets = sorted(
        {float(b) for b in df["budget"].dropna() if str(b) != ""},
    )
    for budget in budgets:
        ax.axvline(budget, color="gray", linestyle="--", alpha=0.4)

    ax.set_xlabel("Edge retry amplification (cost, retries per edge root)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend()


def main(csv_file, output_file):
    df = pd.read_csv(csv_file)

    has_bystander = (
        "bystander_success_rate" in df.columns and df["bystander_success_rate"].notna().any()
    )
    if has_bystander:
        fig, (ax_j, ax_bystander) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax_j = plt.subplots(figsize=(9, 6))

    _draw_panel(
        ax_j,
        df,
        y_col="J_mean",
        y_err_col="J_std",
        ylabel="Base objective J (mean per episode)",
        title="Goodput vs retry amplification",
    )
    if has_bystander:
        _draw_panel(
            ax_bystander,
            df,
            y_col="bystander_success_rate",
            y_err_col=None,
            ylabel="Bystander chain success rate",
            title="Externality: bystander success vs amplification",
        )

    fig.suptitle("Constrained-RL comparison")
    fig.tight_layout()
    fig.savefig(output_file)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pareto plot for the constrained-retry study")
    parser.add_argument("csv_file", help="rl_constrained_comparison CSV path")
    parser.add_argument("--output", "-o", default="rl_pareto.png")
    args = parser.parse_args()
    main(args.csv_file, args.output)
