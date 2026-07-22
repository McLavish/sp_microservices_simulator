"""Lagrangian PPO: constrained training under a retry-amplification budget.

The env's scalar reward already has the shape of a Lagrangian — this module
makes that explicit. ``RewardWeights.cost`` acts as the multiplier lambda on
the constraint cost (the cost client's retry amplification, see
``MicroservicesEnv._window_cost``), and is updated by projected dual ascent
between training chunks:

    lambda <- max(0, lambda + lr * (cost - budget))

During constrained training the hand-tuned ``retries`` penalty is zeroed so
the learned multiplier is the only retry pressure. Evaluation always reports
the *base objective* J (reward with retry and cost penalties zeroed) plus the
cost, so all arms are comparable regardless of their training weights.

Caveat: the constraint is enforced in expectation over eval episodes, not per
episode or per window — report violation rates alongside the means.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from rl.env import EnvComponents, EnvConfig, MicroservicesEnv
from rl.policies.sb3 import SB3Policy
from rl.sb3_support import import_sb3
from rl.spec import ActionSpec, RewardWeights

PPO, make_vec_env, _, SubprocVecEnv, _SB3_IMPORT_ERROR = import_sb3()


@dataclass
class ConstrainedTrainConfig:
    build: Callable[[int | None], EnvComponents]
    action_spec: ActionSpec
    budget: float = 0.1
    lambda_init: float = 0.0
    lambda_lr: float = 0.5
    step_duration_s: float = 1.0
    episode_duration_s: float = 10.0
    timesteps: int = 30_000
    chunk_timesteps: int = 5_000
    n_envs: int = 8
    eval_episodes: int = 3
    eval_seed: int = 10_000
    seed: int = 123
    n_steps: int = 64
    batch_size: int = 64
    gamma: float = 0.95
    device: str = "auto"
    base_weights: RewardWeights = field(default_factory=RewardWeights)
    # False trains an unconstrained arm through the same loop (keep the hand
    # retries penalty; pair with lambda_lr=0 so lambda stays at lambda_init).
    zero_retry_penalty: bool = True
    learning_curve_csv: str | None = None
    name: str = "constrained_ppo"


def dual_ascent_update(lam: float, cost: float, budget: float, lr: float) -> float:
    """One projected dual-ascent step on the multiplier."""
    return max(0.0, lam + lr * (cost - budget))


def objective_from_summary(summary: dict, weights: RewardWeights) -> float:
    """Base objective J: the env reward with retry and cost penalties zeroed.

    Used for evaluation of every arm so the comparison scale does not depend
    on each arm's training weights.
    """
    return float(
        summary["success_rate"] * weights.success
        - summary["deadline_drop_rate"] * weights.deadline_drop
        - summary["queue_drop_rate"] * weights.queue_drop
        - summary["server_drop_rate"] * weights.server_drop
        - summary["circuit_open_drop_rate"] * weights.circuit_open_drop
        - summary["p95_latency_ms"] * weights.p95_latency_ms
    )


def evaluate_objective_and_cost(
    model, env: MicroservicesEnv, episodes: int, weights: RewardWeights
) -> tuple[float, float]:
    """Mean episode objective J and mean per-window cost over `episodes`.

    Resets unseeded so the env's (pre-seeded) RNG yields a reproducible
    scenario sequence.
    """
    objectives = []
    costs = []
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        episode_objective = 0.0
        episode_costs = []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            summary = info["summary"]
            episode_objective += objective_from_summary(summary, weights)
            episode_costs.append(summary["cost"])
        objectives.append(episode_objective)
        costs.append(float(np.mean(episode_costs)) if episode_costs else 0.0)
    return float(np.mean(objectives)), float(np.mean(costs))


def train_constrained_ppo(cfg: ConstrainedTrainConfig) -> SB3Policy:
    if PPO is None or make_vec_env is None:
        raise ImportError(
            "stable-baselines3 is required for constrained training"
        ) from _SB3_IMPORT_ERROR

    weights = replace(
        cfg.base_weights,
        retries=0.0 if cfg.zero_retry_penalty else cfg.base_weights.retries,
        cost=cfg.lambda_init,
    )

    def make_env() -> MicroservicesEnv:
        # Each env gets its own weights copy: lambda updates are broadcast via
        # env_method (works across SubprocVecEnv process boundaries, where a
        # shared mutable object would not).
        return MicroservicesEnv(
            EnvConfig(
                build=cfg.build,
                action_spec=cfg.action_spec,
                step_duration_s=cfg.step_duration_s,
                episode_duration_s=cfg.episode_duration_s,
                reward_weights=replace(weights),
            )
        )

    vec_env = make_vec_env(
        make_env,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv if cfg.n_envs > 1 else None,
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        seed=cfg.seed,
        device=cfg.device,
    )

    eval_env = make_env()
    eval_env.reset(seed=cfg.eval_seed)

    lam = cfg.lambda_init
    curve: list[tuple[int, float, float, float]] = []
    trained = 0
    while trained < cfg.timesteps:
        chunk = min(cfg.chunk_timesteps, cfg.timesteps - trained)
        # SB3 semantics: with reset_num_timesteps=False, learn() trains
        # `total_timesteps` ADDITIONAL steps (_setup_learn adds the current
        # num_timesteps). Pass the per-chunk increment; the counter, schedules,
        # and in-flight episodes still continue across chunks.
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        trained = int(model.num_timesteps)  # true cumulative; curve rows stay honest
        eval_objective, eval_cost = evaluate_objective_and_cost(
            model, eval_env, cfg.eval_episodes, cfg.base_weights
        )
        lam = dual_ascent_update(lam, eval_cost, cfg.budget, cfg.lambda_lr)
        vec_env.env_method("set_cost_weight", lam)
        curve.append((trained, eval_objective, eval_cost, lam))

    if cfg.learning_curve_csv and curve:
        with open(cfg.learning_curve_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "eval_J", "eval_cost", "lambda"])
            writer.writerows(curve)

    return SB3Policy(model=model, deterministic=True, name=cfg.name)
