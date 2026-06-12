import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core import Summary
from rl.cli import parse_int_list, parse_optional_int_list
from rl.env import (
    ActionSpec,
    EnvConfig,
    MicroservicesEnv,
    RewardWeights,
    build_faulty_single_service_env,
    build_single_service_env,
)
from rl.policies.action_space import ActionGrid
from rl.policies.aimd import AIMDPolicy
from rl.policies.bayesian import BayesianPolicy
from rl.policies.fixed import FixedPolicy, FixedValuePolicy
from rl.policies.model_based import ModelBasedPolicy
from rl.policies.sb3 import SB3Policy

try:
    from stable_baselines3 import PPO  # ty: ignore[unresolved-import]
    from stable_baselines3.common.env_util import make_vec_env  # ty: ignore[unresolved-import]
    from stable_baselines3.common.evaluation import (  # ty: ignore[unresolved-import]
        evaluate_policy as sb3_evaluate,
    )
except Exception:  # pragma: no cover
    PPO = None
    make_vec_env = None
    sb3_evaluate = None


@dataclass
class ExperimentConfig:
    attempt_timeouts_ms: list[int]
    max_attempts: list[int]
    backoff_ms: list[int]
    global_timeouts_ms: list[int | None] | None
    step_duration_s: float
    episode_duration_s: float
    reward_weights: RewardWeights
    train_seeds: list[int]
    eval_seeds: list[int]
    ppo_timesteps: int
    ppo_eval_interval: int
    ppo_eval_episodes: int
    ppo_patience: int
    ppo_min_delta: float
    ppo_n_steps: int
    ppo_batch_size: int
    ppo_gamma: float
    model_samples: int
    model_buffer: int
    model_queue_delay_mode: str
    model_queue_workers: int
    model_queue_max_utilization: float
    model_queue_moment_samples: int
    model_queue_use_retries: bool
    bayes_buffer: int
    aimd_success_target: float
    aimd_deadline_target: float
    aimd_timeout_increase_step: int
    aimd_timeout_decrease_factor: float
    aimd_max_attempts_step: int
    aimd_backoff_step: int
    aimd_adjust_max_attempts: bool
    aimd_adjust_backoff: bool
    aimd_ema_alpha: float
    aimd_min_completed: int
    fault_scenario: str


def _build_env(cfg: ExperimentConfig) -> MicroservicesEnv:
    duration_s = int(round(cfg.episode_duration_s)) if cfg.episode_duration_s else None
    if cfg.fault_scenario == "basic":

        def build_fn(seed):
            return build_faulty_single_service_env(seed, duration_s=duration_s)
    else:

        def build_fn(seed):
            return build_single_service_env(seed, duration_s=duration_s)

    action_spec = ActionSpec(
        attempt_timeout_ms=cfg.attempt_timeouts_ms,
        max_attempts=cfg.max_attempts,
        backoff_ms=cfg.backoff_ms,
        global_timeout_ms=cfg.global_timeouts_ms,
    )
    env_cfg = EnvConfig(
        build=build_fn,
        action_spec=action_spec,
        step_duration_s=cfg.step_duration_s,
        episode_duration_s=cfg.episode_duration_s,
        reward_weights=cfg.reward_weights,
        emit_latency_samples=True,
    )
    return MicroservicesEnv(env_cfg)


def _episode(env: MicroservicesEnv, policy, seed: int) -> tuple[float, Summary | None]:
    obs, info = env.reset(seed=seed)
    policy.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        policy.update(obs, reward, terminated or truncated, info)

    summary = env.metrics_summary()
    return total_reward, summary


def _evaluate(env: MicroservicesEnv, policy, seeds: Sequence[int]) -> dict:
    rewards = []
    successes = []
    p95s = []
    retries = []

    for seed in seeds:
        total_reward, summary = _episode(env, policy, seed)
        rewards.append(total_reward)
        if summary:
            successes.append(summary.succeeded / summary.total if summary.total else 0.0)
            p95s.append(summary.p95)
            retries.append(summary.retries_per_root)

    def avg(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "reward_mean": avg(rewards),
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "success_rate": avg(successes),
        "p95_ms": avg(p95s),
        "retries_per_root": avg(retries),
    }


def _train_ppo(cfg: ExperimentConfig) -> SB3Policy:
    if PPO is None or make_vec_env is None or sb3_evaluate is None:
        raise ImportError("stable-baselines3 is required for PPO baseline")

    def env_factory():
        return _build_env(cfg)

    vec_env = make_vec_env(env_factory, n_envs=1, seed=cfg.train_seeds[0])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        gamma=cfg.ppo_gamma,
        seed=cfg.train_seeds[0],
    )

    best_mean = -float("inf")
    no_improve = 0
    trained = 0
    while trained < cfg.ppo_timesteps:
        step = min(cfg.ppo_eval_interval, cfg.ppo_timesteps - trained)
        model.learn(total_timesteps=step)
        trained += step
        mean_reward, _ = sb3_evaluate(model, _build_env(cfg), n_eval_episodes=cfg.ppo_eval_episodes)
        if mean_reward > best_mean + cfg.ppo_min_delta:
            best_mean = mean_reward
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.ppo_patience:
            break

    return SB3Policy(model=model, deterministic=True, name="ppo")


def _best_fixed_action(cfg: ExperimentConfig, grid: ActionGrid) -> tuple[int, ...]:
    env = _build_env(cfg)
    best_reward = -float("inf")
    best_idx = grid.indices[0]
    for action_idx in grid.indices:
        policy = FixedPolicy(action=action_idx, name="oracle_fixed")
        metrics = _evaluate(env, policy, cfg.train_seeds)
        reward = metrics["reward_mean"]
        if reward > best_reward:
            best_reward = reward
            best_idx = action_idx
    return best_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policy baselines.")
    parser.add_argument("--attempt-timeouts-ms", type=parse_int_list, default=[10, 30, 50, 80])
    parser.add_argument("--max-attempts", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--backoff-ms", type=parse_int_list, default=[0, 20, 50])
    parser.add_argument("--global-timeouts-ms", type=parse_optional_int_list, default=None)
    parser.add_argument("--step-duration-s", type=float, default=1.0)
    parser.add_argument("--episode-duration-s", type=float, default=10.0)
    parser.add_argument("--train-seeds", type=parse_int_list, default="123,456,789")
    parser.add_argument("--eval-seeds", type=parse_int_list, default="321,654,987")
    parser.add_argument(
        "--policies",
        type=str,
        default="fixed,aimd,model_based,model_based_queue,bayesian,ppo,oracle_fixed",
    )
    parser.add_argument("--ppo-timesteps", type=int, default=5000)
    parser.add_argument("--ppo-eval-interval", type=int, default=1000)
    parser.add_argument("--ppo-eval-episodes", type=int, default=3)
    parser.add_argument("--ppo-patience", type=int, default=3)
    parser.add_argument("--ppo-min-delta", type=float, default=0.1)
    parser.add_argument("--ppo-n-steps", type=int, default=64)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--ppo-gamma", type=float, default=0.95)
    parser.add_argument("--model-samples", type=int, default=500)
    parser.add_argument("--model-buffer", type=int, default=2000)
    parser.add_argument(
        "--model-queue-delay-mode",
        choices=["deterministic", "exponential"],
        default="deterministic",
    )
    parser.add_argument("--model-queue-workers", type=int, default=16)
    parser.add_argument("--model-queue-max-utilization", type=float, default=0.95)
    parser.add_argument("--model-queue-moment-samples", type=int, default=500)
    parser.add_argument("--model-queue-use-retries", action="store_true")
    parser.add_argument("--bayes-buffer", type=int, default=2000)
    parser.add_argument("--aimd-success-target", type=float, default=0.98)
    parser.add_argument("--aimd-deadline-target", type=float, default=0.01)
    parser.add_argument("--aimd-timeout-increase-step", type=int, default=1)
    parser.add_argument("--aimd-timeout-decrease-factor", type=float, default=0.5)
    parser.add_argument("--aimd-max-attempts-step", type=int, default=1)
    parser.add_argument("--aimd-backoff-step", type=int, default=1)
    parser.add_argument("--aimd-adjust-max-attempts", action="store_true")
    parser.add_argument("--aimd-adjust-backoff", action="store_true")
    parser.add_argument("--aimd-ema-alpha", type=float, default=0.2)
    parser.add_argument("--aimd-min-completed", type=int, default=1)
    parser.add_argument("--fault-scenario", choices=["none", "basic"], default="none")
    parser.add_argument("--reward-success", type=float, default=1.0)
    parser.add_argument("--reward-deadline", type=float, default=1.0)
    parser.add_argument("--reward-queue", type=float, default=1.0)
    parser.add_argument("--reward-server", type=float, default=1.0)
    parser.add_argument("--reward-p95", type=float, default=0.001)
    parser.add_argument("--reward-retries", type=float, default=0.1)
    args = parser.parse_args()

    reward_weights = RewardWeights(
        success=args.reward_success,
        deadline_drop=args.reward_deadline,
        queue_drop=args.reward_queue,
        server_drop=args.reward_server,
        p95_latency_ms=args.reward_p95,
        retries=args.reward_retries,
    )

    cfg = ExperimentConfig(
        attempt_timeouts_ms=list(args.attempt_timeouts_ms),
        max_attempts=list(args.max_attempts),
        backoff_ms=list(args.backoff_ms),
        global_timeouts_ms=args.global_timeouts_ms,
        step_duration_s=args.step_duration_s,
        episode_duration_s=args.episode_duration_s,
        reward_weights=reward_weights,
        train_seeds=list(args.train_seeds),
        eval_seeds=list(args.eval_seeds),
        ppo_timesteps=args.ppo_timesteps,
        ppo_eval_interval=args.ppo_eval_interval,
        ppo_eval_episodes=args.ppo_eval_episodes,
        ppo_patience=args.ppo_patience,
        ppo_min_delta=args.ppo_min_delta,
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_gamma=args.ppo_gamma,
        model_samples=args.model_samples,
        model_buffer=args.model_buffer,
        model_queue_delay_mode=args.model_queue_delay_mode,
        model_queue_workers=args.model_queue_workers,
        model_queue_max_utilization=args.model_queue_max_utilization,
        model_queue_moment_samples=args.model_queue_moment_samples,
        model_queue_use_retries=args.model_queue_use_retries,
        bayes_buffer=args.bayes_buffer,
        aimd_success_target=args.aimd_success_target,
        aimd_deadline_target=args.aimd_deadline_target,
        aimd_timeout_increase_step=args.aimd_timeout_increase_step,
        aimd_timeout_decrease_factor=args.aimd_timeout_decrease_factor,
        aimd_max_attempts_step=args.aimd_max_attempts_step,
        aimd_backoff_step=args.aimd_backoff_step,
        aimd_adjust_max_attempts=args.aimd_adjust_max_attempts,
        aimd_adjust_backoff=args.aimd_adjust_backoff,
        aimd_ema_alpha=args.aimd_ema_alpha,
        aimd_min_completed=args.aimd_min_completed,
        fault_scenario=args.fault_scenario,
    )

    action_spec = ActionSpec(
        attempt_timeout_ms=cfg.attempt_timeouts_ms,
        max_attempts=cfg.max_attempts,
        backoff_ms=cfg.backoff_ms,
        global_timeout_ms=cfg.global_timeouts_ms,
    )
    env = _build_env(cfg)
    grid = ActionGrid(action_spec)

    policy_names = [p.strip() for p in args.policies.split(",") if p.strip()]
    policies = []

    if "fixed" in policy_names:
        fixed_values = FixedValuePolicy(
            grid=grid,
            attempt_timeout_ms=cfg.attempt_timeouts_ms[0],
            max_attempts=cfg.max_attempts[0],
            backoff_ms=cfg.backoff_ms[0],
        )
        policies.append(fixed_values)

    if "aimd" in policy_names:
        aimd = AIMDPolicy(
            grid=grid,
            initial_timeout_ms=cfg.attempt_timeouts_ms[0],
            initial_max_attempts=cfg.max_attempts[0],
            initial_backoff_ms=cfg.backoff_ms[0],
            timeout_increase_step=cfg.aimd_timeout_increase_step,
            timeout_decrease_factor=cfg.aimd_timeout_decrease_factor,
            max_attempts_step=cfg.aimd_max_attempts_step,
            backoff_step=cfg.aimd_backoff_step,
            adjust_max_attempts=cfg.aimd_adjust_max_attempts,
            adjust_backoff=cfg.aimd_adjust_backoff,
            ema_alpha=cfg.aimd_ema_alpha,
            success_target=cfg.aimd_success_target,
            deadline_target=cfg.aimd_deadline_target,
            min_completed=cfg.aimd_min_completed,
        )
        policies.append(aimd)

    if "model_based" in policy_names:
        model_based = ModelBasedPolicy(
            grid=grid,
            reward_weights=cfg.reward_weights,
            sample_size=cfg.model_samples,
            buffer_size=cfg.model_buffer,
            name="model_based",
        )
        policies.append(model_based)

    if "model_based_queue" in policy_names:
        model_based_queue = ModelBasedPolicy(
            grid=grid,
            reward_weights=cfg.reward_weights,
            sample_size=cfg.model_samples,
            buffer_size=cfg.model_buffer,
            include_queue_delay=True,
            queue_delay_mode=cfg.model_queue_delay_mode,
            queue_workers=cfg.model_queue_workers,
            queue_max_utilization=cfg.model_queue_max_utilization,
            queue_moment_samples=cfg.model_queue_moment_samples,
            use_retries_in_lambda=cfg.model_queue_use_retries,
            step_duration_s=cfg.step_duration_s,
            name="model_based_queue",
        )
        policies.append(model_based_queue)

    if "bayesian" in policy_names:
        bayes = BayesianPolicy(
            grid=grid,
            reward_weights=cfg.reward_weights,
            sample_size=cfg.model_samples,
            buffer_size=cfg.bayes_buffer,
            name="bayesian",
        )
        policies.append(bayes)

    if "ppo" in policy_names:
        policies.append(_train_ppo(cfg))

    if "oracle_fixed" in policy_names:
        best_idx = _best_fixed_action(cfg, grid)
        policies.append(FixedPolicy(action=best_idx, name="oracle_fixed"))

    print("=== Evaluation ===")
    for policy in policies:
        metrics = _evaluate(env, policy, cfg.eval_seeds)
        print(
            f"{policy.name:14s} reward={metrics['reward_mean']:.4f} "
            f"+/-{metrics['reward_std']:.4f} "
            f"success={metrics['success_rate']:.3f} "
            f"p95={metrics['p95_ms']:.2f} "
            f"retries={metrics['retries_per_root']:.3f}"
        )


if __name__ == "__main__":
    main()
