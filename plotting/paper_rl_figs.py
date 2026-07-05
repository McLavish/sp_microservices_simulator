"""Print-styled PDF figures for the constrained-retry paper section.

Consumes the outputs of rl.experiments.constrained_retry across one or more
training runs (comparison CSVs and dual-ascent curve CSVs):
  - Pareto figure: unique behavioural outcomes (J vs cost, bystander vs
    cost), with coincident run outcomes merged and counted;
  - training-dynamics figure: evaluation-cost and multiplier trajectories
    for every (run, budget), showing the attractor bifurcation.

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
    "retry_budget": {"marker": "^", "color": C_AQUA, "label": "token retry budget"},
    "unconstrained_ppo": {"marker": "s", "color": C_YELLOW, "label": "unconstrained PPO"},
}

BUDGET_COLORS = {0.05: C_BLUE, 0.1: C_AQUA, 0.2: C_YELLOW}


def _load_runs(comparison_csvs: list[str]) -> pd.DataFrame:
    frames = []
    for path in comparison_csvs:
        df = pd.read_csv(path)
        df["run"] = path
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_pareto(comparison_csvs: list[str], output: str) -> None:
    df = _load_runs(comparison_csvs)
    fig, (ax_j, ax_by) = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH_IN, 3.2), sharex=True, height_ratios=[1.2, 1]
    )

    budgets = sorted({float(b) for b in df["budget"].dropna()})
    for ax in (ax_j, ax_by):
        for budget in budgets:
            ax.axvline(budget, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)

    for arm, style in ARM_STYLES.items():
        sub = df[df["arm"] == arm].copy()
        # merge coincident outcomes across runs/budgets; count multiplicity
        sub["key"] = list(zip(sub["cost_mean"].round(4), sub["J_mean"].round(4), strict=True))
        merged = (
            sub.groupby("key")
            .agg(
                cost=("cost_mean", "first"),
                J=("J_mean", "first"),
                J_std=("J_std", "first"),
                byst=("bystander_success_rate", "first"),
                n=("key", "size"),
            )
            .reset_index(drop=True)
        )
        ax_j.errorbar(
            merged["cost"],
            merged["J"],
            yerr=merged["J_std"],
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
        ax_by.plot(
            merged["cost"],
            merged["byst"],
            linestyle="none",
            markersize=4.5,
            marker=style["marker"],
            color=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.4,
        )
        if arm == "constrained_ppo":
            for _, row in merged.iterrows():
                ax_j.annotate(
                    f"$\\times${int(row['n'])}",
                    (row["cost"], row["J"]),
                    textcoords="offset points",
                    xytext=(5, -9),
                    fontsize=6.5,
                )

    ax_j.set_ylabel("objective $J$")
    ax_by.set_ylabel("bystander success rate")
    ax_by.set_xlabel("constraint cost (edge retries per request)")
    ax_j.legend(loc="lower right", handletextpad=0.3)
    fig.align_ylabels()
    fig.savefig(output)
    plt.close(fig)


def plot_dual_ascent(curve_prefixes: list[str], output: str) -> None:
    """Cost and multiplier trajectories for every (run, budget) pair."""
    fig, (ax_cost, ax_lambda) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 3.0), sharex=True)

    seen_budgets: set[float] = set()
    for prefix in curve_prefixes:
        for path in sorted(glob.glob(f"{prefix}_b*.csv")):
            budget = float(path.rsplit("_b", 1)[1].removesuffix(".csv"))
            color = BUDGET_COLORS.get(budget, "gray")
            df = pd.read_csv(path)
            steps = df["timesteps"] / 1000.0
            label = f"$b$={budget:g}" if budget not in seen_budgets else None
            seen_budgets.add(budget)
            ax_cost.plot(steps, df["eval_cost"], color=color, alpha=0.8, label=label)
            ax_lambda.plot(steps, df["lambda"], color=color, alpha=0.8)

    for budget in sorted(seen_budgets):
        ax_cost.axhline(
            budget, color=BUDGET_COLORS.get(budget, "gray"), linestyle=":", linewidth=0.7
        )

    ax_cost.set_ylabel("evaluation cost")
    ax_lambda.set_ylabel(r"multiplier $\lambda$")
    ax_lambda.set_xlabel("training timesteps ($\\times 10^3$)")
    ax_cost.legend(loc="upper right", handletextpad=0.4, ncols=3, columnspacing=0.9)
    fig.align_ylabels()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-csvs",
        nargs="+",
        default=["rl_constrained_comparison_multichain.csv"],
        help="one or more comparison CSVs (multiple training runs)",
    )
    parser.add_argument(
        "--curve-prefixes",
        nargs="+",
        default=["rl_constrained_curve_multichain"],
        help="one or more curve-file prefixes (multiple training runs)",
    )
    parser.add_argument("--pareto-output", default="paper_rl_pareto.pdf")
    parser.add_argument("--curves-output", default="paper_rl_dual_ascent.pdf")
    args = parser.parse_args()

    plt.rcParams.update(PAPER_RC)
    plot_pareto(args.comparison_csvs, args.pareto_output)
    plot_dual_ascent(args.curve_prefixes, args.curves_output)
    print(f"wrote {args.pareto_output} and {args.curves_output}")


if __name__ == "__main__":
    main()
