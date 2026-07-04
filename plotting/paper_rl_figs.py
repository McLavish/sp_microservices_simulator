"""Print-styled PDF figures for the constrained-retry paper section.

Consumes the CSVs written by rl.experiments.constrained_retry:
  - comparison CSV (one row per arm/budget) -> Pareto figure
    (J vs cost, bystander success vs cost);
  - dual-ascent curve CSVs (timesteps, eval_J, eval_cost, lambda)
    -> training-dynamics figure.

Figures are sized for a two-column conference page (column width 3.29 in)
and saved as vector PDFs.
"""

import argparse
import glob

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

COLUMN_WIDTH_IN = 3.29

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "axes.grid": True,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
}

# Categorical slots (validated colourblind-safe order): blue, aqua, yellow.
C_BLUE = "#2a78d6"
C_AQUA = "#1baf7a"
C_YELLOW = "#eda100"

ARM_STYLES = {
    "constrained_ppo": {"marker": "o", "color": C_BLUE, "label": "constrained PPO"},
    "retry_budget": {"marker": "^", "color": C_AQUA, "label": "retry-budget policy"},
    "unconstrained_ppo": {"marker": "s", "color": C_YELLOW, "label": "unconstrained PPO"},
}


def plot_pareto(comparison_csv: str, output: str) -> None:
    df = pd.read_csv(comparison_csv)
    fig, (ax_j, ax_by) = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH_IN, 3.1), sharex=True, height_ratios=[1.2, 1]
    )

    budgets = sorted({float(b) for b in df["budget"].dropna()})
    for ax in (ax_j, ax_by):
        for budget in budgets:
            ax.axvline(budget, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)

    for arm, style in ARM_STYLES.items():
        sub = df[df["arm"] == arm]
        ax_j.errorbar(
            sub["cost_mean"],
            sub["J_mean"],
            yerr=sub["J_std"],
            linestyle="none",
            capsize=2,
            elinewidth=0.7,
            markersize=4.5,
            marker=style["marker"],
            color=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=style["label"],
        )
        by = sub[pd.notna(sub["bystander_success_rate"])]
        ax_by.plot(
            by["cost_mean"],
            by["bystander_success_rate"],
            linestyle="none",
            markersize=4.5,
            marker=style["marker"],
            color=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.4,
        )

    for _, row in df[df["arm"] == "constrained_ppo"].iterrows():
        if pd.notna(row["budget"]):
            ax_j.annotate(
                f"$b$={row['budget']:g}",
                (row["cost_mean"], row["J_mean"]),
                textcoords="offset points",
                xytext=(4, -9),
                fontsize=6.5,
            )

    ax_j.set_ylabel("objective $J$")
    ax_by.set_ylabel("bystander success rate")
    ax_by.set_xlabel("edge retry amplification (retries per edge request)")
    ax_j.legend(loc="lower right", handletextpad=0.3)
    fig.align_ylabels()
    fig.savefig(output)
    plt.close(fig)


def plot_dual_ascent(curve_prefix: str, output: str) -> None:
    files = sorted(glob.glob(f"{curve_prefix}_b*.csv"))
    if not files:
        raise SystemExit(f"no curve CSVs match {curve_prefix}_b*.csv")

    budget_colors = [C_BLUE, C_AQUA, C_YELLOW]
    fig, (ax_cost, ax_lambda) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 2.9), sharex=True)

    for i, path in enumerate(files):
        budget = float(path.rsplit("_b", 1)[1].removesuffix(".csv"))
        df = pd.read_csv(path)
        color = budget_colors[i % len(budget_colors)]
        steps = df["timesteps"] / 1000.0
        ax_cost.plot(steps, df["eval_cost"], color=color, label=f"$b$={budget:g}")
        ax_cost.axhline(budget, color=color, linestyle=":", linewidth=0.7, alpha=0.8)
        ax_lambda.plot(steps, df["lambda"], color=color)
        ax_lambda.annotate(
            f"$b$={budget:g}",
            (steps.iloc[-1], df["lambda"].iloc[-1]),
            textcoords="offset points",
            xytext=(-2, 4),
            ha="right",
            fontsize=6.5,
            color=color,
        )

    ax_cost.set_ylabel("evaluation cost")
    ax_lambda.set_ylabel(r"multiplier $\lambda$")
    ax_lambda.set_xlabel("training timesteps ($\\times 10^3$)")
    ax_cost.legend(loc="upper left", handletextpad=0.4, ncols=3, columnspacing=0.9)
    fig.align_ylabels()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", default="rl_constrained_comparison_multichain.csv")
    parser.add_argument("--curve-prefix", default="rl_constrained_curve_multichain")
    parser.add_argument("--pareto-output", default="paper_rl_pareto.pdf")
    parser.add_argument("--curves-output", default="paper_rl_dual_ascent.pdf")
    args = parser.parse_args()

    plt.rcParams.update(PAPER_RC)
    plot_pareto(args.comparison_csv, args.pareto_output)
    plot_dual_ascent(args.curve_prefix, args.curves_output)
    print(f"wrote {args.pareto_output} and {args.curves_output}")


if __name__ == "__main__":
    main()
