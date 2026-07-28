"""Three-arm constrained-retry comparison on the chain or multichain scenario.

Arms, per retry budget b:
  A. unconstrained PPO  — trained with the default hand-weighted reward
     (budget-independent, trained once);
  B. constrained PPO    — Lagrangian PPO with edge-amplification budget b;
  C. retry_budget       — best fixed knobs (oracle grid search) on the env
     whose B->C edge is wrapped in the hand-coded RetryBudgetPolicy(b).

All arms are evaluated on the same seeds and reported on the base objective J
(reward with retry/cost penalties zeroed) plus the edge-amplification cost, so
training-weight differences cannot skew the comparison. On the multichain
scenario the bystander chain's success rate is reported alongside, gauging the
externality each arm's retries impose on the shared bottleneck. Output: one
CSV row per (arm, budget), plus per-budget dual-ascent curves.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import ttest_rel

from rl.cli import parse_float_list, parse_int_list
from rl.constrained import (
    ConstrainedTrainConfig,
    objective_from_summary,
    train_constrained_ppo,
)
from rl.env import (
    ActionSpec,
    EnvComponents,
    EnvConfig,
    MicroservicesEnv,
    build_chain_env,
    build_multichain_env,
)
from rl.policies.action_space import ActionGrid
from rl.policies.fixed import FixedPolicy
from rl.sb3_support import import_sb3
from rl.spec import RewardWeights

PPO, _, _, _, _SB3_IMPORT_ERROR = import_sb3()

BASE_WEIGHTS = RewardWeights()

# Both builders share the (seed, duration, *, retry_budget_ratio) signature.
SCENARIO_TO_BUILDER = {
    "chain": build_chain_env,
    "multichain": build_multichain_env,
}


def _make_env(
    build: Callable[[int | None], EnvComponents],
    action_spec: ActionSpec,
    episode_duration_s: float,
) -> MicroservicesEnv:
    return MicroservicesEnv(
        EnvConfig(
            build=build,
            action_spec=action_spec,
            step_duration_s=1.0,
            episode_duration_s=episode_duration_s,
            reward_weights=BASE_WEIGHTS,
        )
    )


def _eval_arm(
    env: MicroservicesEnv,
    policy,
    seeds: Sequence[int],
    budget: float | None,
    keep_per_seed: bool = False,
) -> dict:
    """Evaluate one arm on the base objective J plus the constraint cost.

    ``keep_per_seed`` adds the raw per-seed objectives under ``J_per_seed``
    (calibration pairs them across arms; the CSV paths leave it off).
    """
    objectives: list[float] = []
    costs: list[float] = []
    successes: list[float] = []
    p95s: list[float] = []
    successes_horizon: list[float] = []
    p95s_horizon: list[float] = []
    violations: list[float] = []
    bystander_successes: list[float] = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        policy.reset(seed=seed)
        terminated = truncated = False
        episode_objective = 0.0
        episode_costs: list[float] = []
        while not (terminated or truncated):
            action = policy.act(obs, info)
            obs, _reward, terminated, truncated, info = env.step(action)
            summary = info["summary"]
            episode_objective += objective_from_summary(summary, BASE_WEIGHTS)
            episode_costs.append(summary["cost"])
        cost_mean = float(np.mean(episode_costs)) if episode_costs else 0.0
        objectives.append(episode_objective)
        costs.append(cost_mean)
        if budget is not None:
            violations.append(1.0 if cost_mean > budget else 0.0)
        metrics = env.metrics_summary()
        if metrics:
            successes.append(metrics.succeeded / metrics.total if metrics.total else 0.0)
            p95s.append(metrics.p95)
        horizon = env.in_horizon_summary()
        if horizon:
            successes_horizon.append(horizon.succeeded / horizon.total if horizon.total else 0.0)
            p95s_horizon.append(horizon.p95)
        # Score the externality on the SAME population as the objective J, which
        # only sees in-horizon roots. The full population includes the drained
        # tail, which completes after the agent's chain has gone quiet and so
        # reads systematically better than what J was actually measured against.
        bystander = env.bystander_in_horizon_summary()
        if bystander is not None:
            bystander_successes.append(
                bystander.succeeded / bystander.total if bystander.total else 0.0
            )

    result: dict[str, object] = {
        "J_mean": float(np.mean(objectives)),
        "J_std": float(np.std(objectives)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "p95_ms": float(np.mean(p95s)) if p95s else 0.0,
        # Full-population above (incl. drained roots); reward-visible below.
        "success_rate_in_horizon": float(np.mean(successes_horizon)) if successes_horizon else 0.0,
        "p95_ms_in_horizon": float(np.mean(p95s_horizon)) if p95s_horizon else 0.0,
        "cost_mean": float(np.mean(costs)),
        "violation_rate": float(np.mean(violations)) if violations else "",
        # Empty when the scenario has no bystander chain (CSV-shape convention
        # shared with violation_rate).
        "bystander_success_rate": (
            float(np.mean(bystander_successes)) if bystander_successes else ""
        ),
    }
    if keep_per_seed:
        result["J_per_seed"] = objectives
    return result


def _best_fixed_policy(
    env: MicroservicesEnv, grid: ActionGrid, train_seeds: Sequence[int]
) -> FixedPolicy:
    best_j = -float("inf")
    best_idx = grid.indices[0]
    for action_idx in grid.indices:
        metrics = _eval_arm(env, FixedPolicy(action=action_idx), train_seeds, budget=None)
        if metrics["J_mean"] > best_j:
            best_j = metrics["J_mean"]
            best_idx = action_idx
    return FixedPolicy(action=best_idx, name="retry_budget")


def _run_calibration(
    env: MicroservicesEnv,
    grid: ActionGrid,
    oracle_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    budgets: Sequence[float],
) -> bool:
    """Check the scenario supports a binding retry budget before any training.

    Sweeps every fixed action on the same J metric arm C's oracle uses, then
    verdicts: retries must be valuable at the unconstrained optimum, the
    smallest budget must bind there, budgets must trade J for cost, and (when
    a bystander chain exists) the agent's retries must measurably hurt it.
    Returns True when every applicable criterion passes.
    """
    print(f"[calibrate] sweeping {len(grid.indices)} fixed actions on seeds {list(oracle_seeds)}")
    sweep: list[tuple[tuple[int, ...], dict]] = []
    for action_idx in grid.indices:
        metrics = _eval_arm(env, FixedPolicy(action=action_idx), oracle_seeds, budget=None)
        sweep.append((action_idx, metrics))
        values = grid.values_for(action_idx)
        bystander = metrics["bystander_success_rate"]
        bystander_str = f"{bystander:.3f}" if isinstance(bystander, float) else "-"
        print(
            f"  timeout={values.attempt_timeout_ms:>3}ms attempts={values.max_attempts} "
            f"backoff={values.backoff_ms:>2}ms  J={metrics['J_mean']:7.3f} "
            f"cost={metrics['cost_mean']:5.3f} success={metrics['success_rate']:.3f} "
            f"bystander={bystander_str}"
        )

    def re_eval(action_idx: tuple[int, ...]) -> dict:
        return _eval_arm(
            env, FixedPolicy(action=action_idx), eval_seeds, budget=None, keep_per_seed=True
        )

    best_idx, _ = max(sweep, key=lambda item: item[1]["J_mean"])
    noretry_sweep = [item for item in sweep if grid.values_for(item[0]).max_attempts == 1]
    best_noretry_idx, _ = max(noretry_sweep, key=lambda item: item[1]["J_mean"])

    best = re_eval(best_idx)
    best_values = grid.values_for(best_idx)
    best_noretry = re_eval(best_noretry_idx)
    min_budget = min(budgets)

    # Feasible optimum per budget, selected on the sweep's cost (as arm C's
    # oracle would see it), re-evaluated on the eval seeds for tighter stats.
    feasible: dict[float, dict] = {}
    for budget in sorted(budgets):
        candidates = [item for item in sweep if item[1]["cost_mean"] <= budget]
        idx, _ = max(candidates, key=lambda item: item[1]["J_mean"])
        feasible[budget] = re_eval(idx)

    print(
        f"[calibrate] best action: timeout={best_values.attempt_timeout_ms}ms "
        f"attempts={best_values.max_attempts} backoff={best_values.backoff_ms}ms "
        f"J={best['J_mean']:.3f}±{best['J_std']:.3f} cost={best['cost_mean']:.3f}"
    )
    for budget in sorted(budgets):
        row = feasible[budget]
        print(
            f"[calibrate] best under cost<={budget}: J={row['J_mean']:.3f} "
            f"cost={row['cost_mean']:.3f}"
        )

    results: list[tuple[str, bool, str]] = []
    results.append(
        (
            "(a) retries valuable: unconstrained optimum has max_attempts >= 2",
            best_values.max_attempts >= 2,
            "raise p_fail range or blip length in build_multichain_env",
        )
    )
    results.append(
        (
            f"(b) smallest budget binds: best cost {best['cost_mean']:.3f} > {min_budget}",
            best["cost_mean"] > min_budget,
            "raise p_fail range or blip length in build_multichain_env",
        )
    )
    ordered_j = [feasible[b]["J_mean"] for b in sorted(budgets)] + [best["J_mean"]]
    # Every arm runs the same eval seeds, so the seed-to-seed blip-difficulty
    # variance is shared and cancels in a paired comparison — hence a paired
    # t-test on per-seed J, not the pooled stds.
    gap = float(np.mean(best["J_per_seed"])) - float(np.mean(feasible[min_budget]["J_per_seed"]))
    paired = ttest_rel(
        best["J_per_seed"], feasible[min_budget]["J_per_seed"], alternative="greater"
    )
    results.append(
        (
            f"(c) budgets separate: J nondecreasing in budget and paired "
            f"gap {gap:.3f} > 0 (t-test p={paired.pvalue:.3f})",
            all(a <= b + 1e-9 for a, b in zip(ordered_j, ordered_j[1:], strict=False))
            and paired.pvalue < 0.05,
            "widen the p_fail range so the cost spectrum spreads",
        )
    )
    if isinstance(best["bystander_success_rate"], float) and isinstance(
        best_noretry["bystander_success_rate"], float
    ):
        externality = best_noretry["bystander_success_rate"] - best["bystander_success_rate"]
        results.append(
            (
                f"(d) externality real: bystander damage {externality:.3f} >= 0.02",
                externality >= 0.02,
                "drop svc-C workers or raise per-chain base_rps",
            )
        )
    else:
        print("[calibrate] (d) externality: SKIP — scenario has no bystander chain")
    results.append(
        (
            f"(e) baseline healthy: no-retry success {best_noretry['success_rate']:.3f} >= 0.85",
            best_noretry["success_rate"] >= 0.85,
            "raise svc-C workers (undersized even without amplification)",
        )
    )

    all_passed = True
    for label, passed, remedy in results:
        verdict = "PASS" if passed else f"FAIL — remedy: {remedy}"
        print(f"[calibrate] {label}: {verdict}")
        all_passed = all_passed and passed
    print(f"[calibrate] overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIO_TO_BUILDER), default="chain")
    parser.add_argument("--budgets", type=parse_float_list, default=[0.05, 0.1, 0.2])
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--chunk-timesteps", type=int, default=5_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--episode-duration-s", type=float, default=10.0)
    parser.add_argument("--lambda-lr", type=float, default=0.5)
    parser.add_argument("--lambda-init", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train-seed", type=int, default=123)
    parser.add_argument("--oracle-seeds", type=parse_int_list, default="123,456,789")
    parser.add_argument(
        "--eval-seeds",
        type=parse_int_list,
        default="321,654,987,1320,1653,1986,2319,2652,2985,3318",
    )
    parser.add_argument("--attempt-timeouts-ms", type=parse_int_list, default=[10, 30, 50, 80])
    parser.add_argument("--max-attempts", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--backoff-ms", type=parse_int_list, default=[0, 20, 50])
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="default: rl_constrained_comparison_<scenario>.csv",
    )
    parser.add_argument(
        "--curve-prefix", type=str, default=None, help="default: rl_constrained_curve_<scenario>"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="grid-sweep fixed policies and verify the scenario supports a "
        "binding retry budget, then exit without training",
    )
    args = parser.parse_args()

    builder = SCENARIO_TO_BUILDER[args.scenario]
    output_csv = args.output_csv or f"rl_constrained_comparison_{args.scenario}.csv"
    curve_prefix = args.curve_prefix or f"rl_constrained_curve_{args.scenario}"

    action_spec = ActionSpec(
        attempt_timeout_ms=list(args.attempt_timeouts_ms),
        max_attempts=list(args.max_attempts),
        backoff_ms=list(args.backoff_ms),
    )
    grid = ActionGrid(action_spec)
    duration = float(args.episode_duration_s)

    def plain_build(seed):
        return builder(seed, duration)

    plain_env = _make_env(plain_build, action_spec, args.episode_duration_s)

    if args.calibrate:
        # The gate's verdict is the point of the mode: propagate it as the exit
        # status instead of always reporting success.
        passed = _run_calibration(plain_env, grid, args.oracle_seeds, args.eval_seeds, args.budgets)
        raise SystemExit(0 if passed else 1)

    if PPO is None:
        raise ImportError(
            "stable-baselines3 is required for this experiment"
        ) from _SB3_IMPORT_ERROR

    rows: list[dict] = []

    def common_train_kwargs() -> dict:
        return {
            "action_spec": action_spec,
            "episode_duration_s": args.episode_duration_s,
            "timesteps": args.timesteps,
            "chunk_timesteps": args.chunk_timesteps,
            "n_envs": args.n_envs,
            "seed": args.train_seed,
            "eval_seed": args.train_seed + 10_000,
            "device": args.device,
        }

    # Arm A: unconstrained PPO with the default hand-weighted reward, trained
    # through the same loop (lambda frozen at 0, hand retries penalty kept).
    print("[arm A] training unconstrained PPO...")
    unconstrained = train_constrained_ppo(
        ConstrainedTrainConfig(
            build=plain_build,
            lambda_init=0.0,
            lambda_lr=0.0,
            zero_retry_penalty=False,
            learning_curve_csv=f"{curve_prefix}_unconstrained.csv",
            name="unconstrained_ppo",
            **common_train_kwargs(),
        )
    )
    metrics = _eval_arm(plain_env, unconstrained, args.eval_seeds, budget=None)
    rows.append({"arm": "unconstrained_ppo", "budget": "", **metrics})
    print(
        f"[arm A] J={metrics['J_mean']:.3f} cost={metrics['cost_mean']:.3f} "
        f"bystander={metrics['bystander_success_rate']}"
    )

    for budget in args.budgets:
        # Arm B: Lagrangian PPO with edge-amplification budget.
        print(f"[arm B] training constrained PPO, budget={budget}...")
        constrained = train_constrained_ppo(
            ConstrainedTrainConfig(
                build=plain_build,
                budget=budget,
                lambda_init=args.lambda_init,
                lambda_lr=args.lambda_lr,
                learning_curve_csv=f"{curve_prefix}_b{budget}.csv",
                **common_train_kwargs(),
            )
        )
        metrics = _eval_arm(plain_env, constrained, args.eval_seeds, budget=budget)
        rows.append({"arm": "constrained_ppo", "budget": budget, **metrics})
        print(
            f"[arm B] b={budget} J={metrics['J_mean']:.3f} cost={metrics['cost_mean']:.3f} "
            f"violations={metrics['violation_rate']} "
            f"bystander={metrics['bystander_success_rate']}"
        )

        # Arm C: hand-coded RetryBudgetPolicy on the edge, best fixed knobs.
        def budget_build(seed, _b=budget):
            return builder(seed, duration, retry_budget_ratio=_b)

        budget_env = _make_env(budget_build, action_spec, args.episode_duration_s)
        oracle = _best_fixed_policy(budget_env, grid, args.oracle_seeds)
        metrics = _eval_arm(budget_env, oracle, args.eval_seeds, budget=budget)
        rows.append({"arm": "retry_budget", "budget": budget, **metrics})
        print(
            f"[arm C] b={budget} J={metrics['J_mean']:.3f} cost={metrics['cost_mean']:.3f} "
            f"violations={metrics['violation_rate']} "
            f"bystander={metrics['bystander_success_rate']}"
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote comparison to {output_csv}")


if __name__ == "__main__":
    main()
