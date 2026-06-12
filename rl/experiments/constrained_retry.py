"""Three-arm constrained-retry comparison on the chain scenario.

Arms, per retry budget b:
  A. unconstrained PPO  — trained with the default hand-weighted reward
     (budget-independent, trained once);
  B. constrained PPO    — Lagrangian PPO with edge-amplification budget b;
  C. retry_budget       — best fixed knobs (oracle grid search) on the chain
     env whose B->C edge is wrapped in the hand-coded RetryBudgetPolicy(b).

All arms are evaluated on the same seeds and reported on the base objective J
(reward with retry/cost penalties zeroed) plus the edge-amplification cost, so
training-weight differences cannot skew the comparison. Output: one CSV row
per (arm, budget), plus per-budget dual-ascent curves.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence

import numpy as np

from rl.cli import parse_float_list, parse_int_list
from rl.constrained import (
    ConstrainedTrainConfig,
    objective_from_summary,
    train_constrained_ppo,
)
from rl.env import ActionSpec, EnvComponents, EnvConfig, MicroservicesEnv, build_chain_env
from rl.policies.action_space import ActionGrid
from rl.policies.fixed import FixedPolicy
from rl.sb3_support import import_sb3
from rl.spec import RewardWeights

PPO, _, _, _SB3_IMPORT_ERROR = import_sb3()

BASE_WEIGHTS = RewardWeights()


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
) -> dict:
    """Evaluate one arm on the base objective J plus the constraint cost."""
    objectives: list[float] = []
    costs: list[float] = []
    successes: list[float] = []
    p95s: list[float] = []
    violations: list[float] = []

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

    return {
        "J_mean": float(np.mean(objectives)),
        "J_std": float(np.std(objectives)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "p95_ms": float(np.mean(p95s)) if p95s else 0.0,
        "cost_mean": float(np.mean(costs)),
        "violation_rate": float(np.mean(violations)) if violations else "",
    }


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


def main() -> None:
    if PPO is None:
        raise ImportError(
            "stable-baselines3 is required for this experiment"
        ) from _SB3_IMPORT_ERROR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=parse_float_list, default=[0.05, 0.1, 0.2])
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--chunk-timesteps", type=int, default=5_000)
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
    parser.add_argument("--output-csv", type=str, default="rl_constrained_comparison.csv")
    parser.add_argument("--curve-prefix", type=str, default="rl_constrained_curve")
    args = parser.parse_args()

    action_spec = ActionSpec(
        attempt_timeout_ms=list(args.attempt_timeouts_ms),
        max_attempts=list(args.max_attempts),
        backoff_ms=list(args.backoff_ms),
    )
    grid = ActionGrid(action_spec)
    duration = int(round(args.episode_duration_s))

    def plain_build(seed):
        return build_chain_env(seed, duration)

    plain_env = _make_env(plain_build, action_spec, args.episode_duration_s)
    rows: list[dict] = []

    def common_train_kwargs() -> dict:
        return {
            "action_spec": action_spec,
            "episode_duration_s": args.episode_duration_s,
            "timesteps": args.timesteps,
            "chunk_timesteps": args.chunk_timesteps,
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
            learning_curve_csv=f"{args.curve_prefix}_unconstrained.csv",
            name="unconstrained_ppo",
            **common_train_kwargs(),
        )
    )
    metrics = _eval_arm(plain_env, unconstrained, args.eval_seeds, budget=None)
    rows.append({"arm": "unconstrained_ppo", "budget": "", **metrics})
    print(f"[arm A] J={metrics['J_mean']:.3f} cost={metrics['cost_mean']:.3f}")

    for budget in args.budgets:
        # Arm B: Lagrangian PPO with edge-amplification budget.
        print(f"[arm B] training constrained PPO, budget={budget}...")
        constrained = train_constrained_ppo(
            ConstrainedTrainConfig(
                build=plain_build,
                budget=budget,
                lambda_init=args.lambda_init,
                lambda_lr=args.lambda_lr,
                learning_curve_csv=f"{args.curve_prefix}_b{budget}.csv",
                **common_train_kwargs(),
            )
        )
        metrics = _eval_arm(plain_env, constrained, args.eval_seeds, budget=budget)
        rows.append({"arm": "constrained_ppo", "budget": budget, **metrics})
        print(
            f"[arm B] b={budget} J={metrics['J_mean']:.3f} cost={metrics['cost_mean']:.3f} "
            f"violations={metrics['violation_rate']}"
        )

        # Arm C: hand-coded RetryBudgetPolicy on the edge, best fixed knobs.
        def budget_build(seed, _b=budget):
            return build_chain_env(seed, duration, retry_budget_ratio=_b)

        budget_env = _make_env(budget_build, action_spec, args.episode_duration_s)
        oracle = _best_fixed_policy(budget_env, grid, args.oracle_seeds)
        metrics = _eval_arm(budget_env, oracle, args.eval_seeds, budget=budget)
        rows.append({"arm": "retry_budget", "budget": budget, **metrics})
        print(
            f"[arm C] b={budget} J={metrics['J_mean']:.3f} cost={metrics['cost_mean']:.3f} "
            f"violations={metrics['violation_rate']}"
        )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote comparison to {args.output_csv}")


if __name__ == "__main__":
    main()
