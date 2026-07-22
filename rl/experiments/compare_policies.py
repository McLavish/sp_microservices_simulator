import argparse
import csv
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core import Summary
from rl.cli import parse_int_list, parse_optional_int_list
from rl.constrained import ConstrainedTrainConfig, train_constrained_ppo
from rl.env import (
    SCENARIO_BUILDERS,
    ActionSpec,
    EnvConfig,
    MicroservicesEnv,
    RewardWeights,
)
from rl.policies.action_space import ActionGrid
from rl.policies.aimd import AIMDPolicy
from rl.policies.bayesian import BayesianPolicy
from rl.policies.fixed import FixedPolicy, FixedValuePolicy
from rl.policies.model_based import ModelBasedPolicy
from rl.policies.sb3 import SB3Policy
from rl.sb3_support import import_sb3

PPO, make_vec_env, _sb3_evaluate, _SubprocVecEnv, _ = import_sb3()


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
    builder = SCENARIO_BUILDERS[cfg.fault_scenario]

    def build_fn(seed):
        return builder(seed, duration_s)

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


def _episode(
    env: MicroservicesEnv,
    policy,
    seed: int,
    trace_rows: list[dict] | None = None,
) -> tuple[float, Summary | None, float]:
    obs, info = env.reset(seed=seed)
    policy.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    step_idx = 0
    window_costs: list[float] = []
    while not (terminated or truncated):
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        summary = info["summary"]
        window_costs.append(summary["cost"])
        if trace_rows is not None:
            applied = info["action"]
            trace_rows.append(
                {
                    "policy": policy.name,
                    "seed": seed,
                    "step": step_idx,
                    "attempt_timeout_ms": applied["attempt_timeout_ms"],
                    "max_attempts": applied["max_attempts"],
                    "backoff_ms": applied["backoff_ms"],
                    "global_timeout_ms": applied["global_timeout_ms"],
                    "reward": float(reward),
                    "completed": summary["completed"],
                    "success_rate": summary["success_rate"],
                    "deadline_drop_rate": summary["deadline_drop_rate"],
                    "queue_drop_rate": summary["queue_drop_rate"],
                    "server_drop_rate": summary["server_drop_rate"],
                    "p95_latency_ms": summary["p95_latency_ms"],
                    "retries_per_root": summary["retries_per_root"],
                    "cost": summary["cost"],
                }
            )
        step_idx += 1
        policy.update(obs, reward, terminated or truncated, info)

    cost_mean = float(np.mean(window_costs)) if window_costs else 0.0
    return total_reward, env.metrics_summary(), env.in_horizon_summary(), cost_mean


def _evaluate(
    env: MicroservicesEnv,
    policy,
    seeds: Sequence[int],
    trace_rows: list[dict] | None = None,
) -> dict:
    rewards = []
    successes = []
    p95s = []
    successes_horizon = []
    p95s_horizon = []
    retries = []
    costs = []

    for seed in seeds:
        total_reward, summary, horizon, cost_mean = _episode(
            env, policy, seed, trace_rows=trace_rows
        )
        rewards.append(total_reward)
        costs.append(cost_mean)
        if summary:
            successes.append(summary.succeeded / summary.total if summary.total else 0.0)
            p95s.append(summary.p95)
            retries.append(summary.retries_per_root)
        if horizon:
            successes_horizon.append(horizon.succeeded / horizon.total if horizon.total else 0.0)
            p95s_horizon.append(horizon.p95)

    def avg(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    # success_rate / p95_ms keep their historical full-population meaning
    # (incl. drained roots); the *_in_horizon columns cover exactly the roots
    # the reward saw.
    return {
        "reward_mean": avg(rewards),
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "success_rate": avg(successes),
        "p95_ms": avg(p95s),
        "success_rate_in_horizon": avg(successes_horizon),
        "p95_ms_in_horizon": avg(p95s_horizon),
        "retries_per_root": avg(retries),
        "cost_mean": avg(costs),
    }


def _train_ppo(cfg: ExperimentConfig, learning_curve_csv: str | None = None) -> SB3Policy:
    if PPO is None or make_vec_env is None:
        raise ImportError("stable-baselines3 is required for PPO baseline")
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        StopTrainingOnNoModelImprovement,
    )
    from stable_baselines3.common.monitor import Monitor

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

    # Seed the early-stopping eval env once: its unseeded resets then draw a
    # reproducible scenario sequence from the gym RNG instead of fresh entropy.
    eval_env = Monitor(_build_env(cfg))
    eval_env.reset(seed=cfg.train_seeds[0] + 10_000)

    # EvalCallback only keeps its evaluation history when log_path is set; the
    # npz it writes there is discarded in favor of the CSV below.
    with tempfile.TemporaryDirectory() as npz_dir:
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=cfg.ppo_eval_episodes,
            eval_freq=cfg.ppo_eval_interval,
            log_path=npz_dir,
            callback_after_eval=StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=cfg.ppo_patience
            ),
            verbose=0,
        )
        model.learn(total_timesteps=cfg.ppo_timesteps, callback=eval_callback)

    if learning_curve_csv and len(eval_callback.evaluations_timesteps):
        with open(learning_curve_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "eval_reward"])
            writer.writerows(
                (steps, float(np.mean(rewards)))
                for steps, rewards in zip(
                    eval_callback.evaluations_timesteps,
                    eval_callback.evaluations_results,
                    strict=True,
                )
            )
        print(f"Wrote PPO learning curve to {learning_curve_csv}")

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
    parser.add_argument(
        "--eval-seeds",
        type=parse_int_list,
        default="321,654,987,1320,1653,1986,2319,2652,2985,3318",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="fixed,aimd,model_based,model_based_queue,bayesian,ppo,oracle_fixed",
    )
    parser.add_argument("--ppo-timesteps", type=int, default=5000)
    parser.add_argument("--ppo-eval-interval", type=int, default=1000)
    parser.add_argument("--ppo-eval-episodes", type=int, default=3)
    parser.add_argument("--ppo-patience", type=int, default=3)
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
    parser.add_argument("--fault-scenario", choices=sorted(SCENARIO_BUILDERS), default="none")
    parser.add_argument("--reward-success", type=float, default=1.0)
    parser.add_argument("--reward-deadline", type=float, default=1.0)
    parser.add_argument("--reward-queue", type=float, default=1.0)
    parser.add_argument("--reward-server", type=float, default=1.0)
    parser.add_argument("--reward-p95", type=float, default=0.001)
    parser.add_argument("--reward-retries", type=float, default=0.1)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="rl_compare_policies.csv",
        help="Per-policy evaluation results (repo convention: CSV at repo root).",
    )
    parser.add_argument(
        "--traces-csv",
        type=str,
        default="rl_policy_traces.csv",
        help="Per-step evaluation traces (action/reward/rates per policy and seed).",
    )
    parser.add_argument(
        "--ppo-learning-curve-csv",
        type=str,
        default="rl_ppo_learning_curve.csv",
        help="PPO early-stopping eval reward per timestep budget.",
    )
    parser.add_argument(
        "--ppo-save-path",
        type=str,
        default=None,
        help="Save the trained PPO model (e.g. to reuse across fault scenarios).",
    )
    parser.add_argument(
        "--ppo-load-path",
        type=str,
        default=None,
        help="Load a trained PPO model instead of training (cross-scenario eval).",
    )
    parser.add_argument("--retry-budget", type=float, default=0.1)
    parser.add_argument("--lambda-lr", type=float, default=0.5)
    parser.add_argument("--lambda-init", type=float, default=0.0)
    parser.add_argument(
        "--constrained-curve-csv",
        type=str,
        default="rl_constrained_curve.csv",
        help="Constrained-PPO dual-ascent trajectory (timesteps, eval_J, eval_cost, lambda).",
    )
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
        if args.ppo_load_path:
            if PPO is None:
                raise ImportError("stable-baselines3 is required for PPO baseline")
            model = PPO.load(args.ppo_load_path)
            policies.append(SB3Policy(model=model, deterministic=True, name="ppo"))
        else:
            ppo_policy = _train_ppo(cfg, learning_curve_csv=args.ppo_learning_curve_csv)
            if args.ppo_save_path:
                ppo_policy.model.save(args.ppo_save_path)
                print(f"Saved PPO model to {args.ppo_save_path}")
            policies.append(ppo_policy)

    if "constrained_ppo" in policy_names:
        duration_s = int(round(cfg.episode_duration_s)) if cfg.episode_duration_s else None
        builder = SCENARIO_BUILDERS[cfg.fault_scenario]
        constrained_cfg = ConstrainedTrainConfig(
            build=lambda seed: builder(seed, duration_s),
            action_spec=action_spec,
            budget=args.retry_budget,
            lambda_init=args.lambda_init,
            lambda_lr=args.lambda_lr,
            step_duration_s=cfg.step_duration_s,
            episode_duration_s=cfg.episode_duration_s,
            timesteps=cfg.ppo_timesteps,
            chunk_timesteps=cfg.ppo_eval_interval,
            eval_episodes=cfg.ppo_eval_episodes,
            eval_seed=cfg.train_seeds[0] + 10_000,
            seed=cfg.train_seeds[0],
            n_steps=cfg.ppo_n_steps,
            batch_size=cfg.ppo_batch_size,
            gamma=cfg.ppo_gamma,
            base_weights=cfg.reward_weights,
            learning_curve_csv=args.constrained_curve_csv,
        )
        policies.append(train_constrained_ppo(constrained_cfg))

    if "oracle_fixed" in policy_names:
        best_idx = _best_fixed_action(cfg, grid)
        policies.append(FixedPolicy(action=best_idx, name="oracle_fixed"))

    print("=== Evaluation ===")
    results = []
    trace_rows: list[dict] = []
    for policy in policies:
        metrics = _evaluate(env, policy, cfg.eval_seeds, trace_rows=trace_rows)
        results.append({"policy": policy.name, **metrics})
        print(
            f"{policy.name:16s} reward={metrics['reward_mean']:.4f} "
            f"+/-{metrics['reward_std']:.4f} "
            f"success={metrics['success_rate']:.3f} "
            f"p95={metrics['p95_ms']:.2f} "
            f"retries={metrics['retries_per_root']:.3f} "
            f"cost={metrics['cost_mean']:.3f}"
        )

    if args.output_csv and results:
        fieldnames = list(results[0].keys())
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote results to {args.output_csv}")

    if args.traces_csv and trace_rows:
        with open(args.traces_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trace_rows)
        print(f"Wrote per-step traces to {args.traces_csv}")


if __name__ == "__main__":
    main()
