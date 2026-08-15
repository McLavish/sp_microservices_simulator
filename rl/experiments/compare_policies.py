import argparse
import csv
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rl.cli import parse_int_list, parse_optional_int_list
from rl.constrained import ConstrainedTrainConfig, train_constrained_ppo
from rl.env import (
    SCENARIO_BUILDERS,
    ActionSpec,
    EnvConfig,
    MicroservicesEnv,
    RewardWeights,
)
from rl.model_io import check_action_spec, save_action_spec
from rl.policies.action_space import ActionGrid
from rl.policies.aimd import AIMDPolicy
from rl.policies.bayesian import BayesianPolicy
from rl.policies.fixed import FixedPolicy, FixedValuePolicy
from rl.policies.model_based import ModelBasedPolicy
from rl.policies.sb3 import SB3Policy
from rl.rollout import EpisodeResult, mean_ignoring_nan, policy_actor, run_episode
from rl.sb3_support import import_sb3

_SB3 = import_sb3()
PPO, make_vec_env = _SB3.ppo, _SB3.make_vec_env
_SB3_IMPORT_ERROR = _SB3.import_error


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
    duration_s = cfg.episode_duration_s if cfg.episode_duration_s else None
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
) -> EpisodeResult:
    policy.reset(seed=seed)
    result = run_episode(
        env,
        policy_actor(policy),
        seed=seed,
        on_step=lambda obs, reward, done, info: policy.update(obs, reward, done, info),
    )
    if trace_rows is not None:
        # Trace rows are this caller's business; the loop and the aggregations
        # that must agree across callers live in rl.rollout.
        trace_rows.extend(
            {
                "policy": policy.name,
                "seed": seed,
                "step": window.step,
                "step_duration_s": env.step_duration_s,
                "attempt_timeout_ms": window.action_values["attempt_timeout_ms"],
                "max_attempts": window.action_values["max_attempts"],
                "backoff_ms": window.action_values["backoff_ms"],
                "global_timeout_ms": window.action_values["global_timeout_ms"],
                "reward": window.reward,
                "completed": window.summary["completed"],
                "success_rate": window.summary["success_rate"],
                "deadline_drop_rate": window.summary["deadline_drop_rate"],
                "queue_drop_rate": window.summary["queue_drop_rate"],
                "server_drop_rate": window.summary["server_drop_rate"],
                "p95_latency_ms": window.summary["p95_latency_ms"],
                "retries_per_root": window.summary["retries_per_root"],
                "cost": window.summary["cost"],
            }
            for window in result.windows
        )
    # Return the shared result itself: flattening it into a positional tuple is
    # what forced this module to re-implement EpisodeResult.success_rate inline
    # and put bystander_success_rate()/drained_summary out of reach.
    return result


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
        result = _episode(env, policy, seed, trace_rows=trace_rows)
        rewards.append(result.total_reward)
        costs.append(result.pooled_cost)
        if result.metrics:
            # Through the shared accessor, so this path and _eval_arm cannot
            # drift on what "success rate over a population" means.
            successes.append(result.success_rate(in_horizon=False))
            p95s.append(result.metrics.p95)
            retries.append(result.metrics.retries_per_root)
        if result.in_horizon:
            successes_horizon.append(result.success_rate(in_horizon=True))
            p95s_horizon.append(result.in_horizon.p95)

    def avg(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    # success_rate / p95_ms keep their historical full-population meaning
    # (incl. drained roots); the *_in_horizon columns cover exactly the roots
    # the reward saw. Percentiles go through mean_ignoring_nan: a seed with no
    # successful root reports p95 as nan by design, and plain averaging would
    # void the cell for every other seed too.
    return {
        "reward_mean": avg(rewards),
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "success_rate": avg(successes),
        "p95_ms": mean_ignoring_nan(p95s),
        "success_rate_in_horizon": avg(successes_horizon),
        "p95_ms_in_horizon": mean_ignoring_nan(p95s_horizon),
        "retries_per_root": avg(retries),
        "cost_mean": avg(costs),
    }


def _train_ppo(cfg: ExperimentConfig, learning_curve_csv: str | None = None) -> SB3Policy:
    if PPO is None or make_vec_env is None:
        # Chained: import_sb3 catches bare Exception, so a broken torch/sb3
        # install is otherwise indistinguishable from a missing one and the
        # user is sent to reinstall a package they already have.
        raise ImportError("stable-baselines3 is required for PPO baseline") from _SB3_IMPORT_ERROR
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
        # min_evals keeps patience from firing in the opening evals of a long
        # run. StopTrainingOnNoModelImprovement compares against a running max
        # with min_evals=0 by default, so a budget raised without also raising
        # the eval interval stopped after patience+1 evals - the documented
        # 100k command trained 5000 steps, a 20x silent truncation.
        expected_evals = max(1, cfg.ppo_timesteps // max(1, cfg.ppo_eval_interval))
        min_evals = max(cfg.ppo_patience + 1, expected_evals // 4)
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=cfg.ppo_eval_episodes,
            eval_freq=cfg.ppo_eval_interval,
            log_path=npz_dir,
            # Checkpoint the best model: without a save path SB3 only tracks
            # best_mean_reward, so "early stopping" returned the LAST weights -
            # by construction several non-improving evals past the best it saw.
            best_model_save_path=npz_dir,
            callback_after_eval=StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=cfg.ppo_patience,
                min_evals=min_evals,
            ),
            verbose=0,
        )
        model.learn(total_timesteps=cfg.ppo_timesteps, callback=eval_callback)

        best_path = os.path.join(npz_dir, "best_model.zip")
        if os.path.exists(best_path):
            # Reload inside the tempdir, before it is cleaned up.
            model = PPO.load(best_path, env=vec_env)

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


def main(argv: Sequence[str] | None = None) -> None:
    """`argv` defaults to sys.argv; taking it explicitly makes the entry
    point callable from a test without patching sys.argv."""
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
    parser.add_argument("--bayes-buffer", type=int, default=2000)
    parser.add_argument("--aimd-success-target", type=float, default=0.95)
    parser.add_argument("--aimd-deadline-target", type=float, default=0.05)
    parser.add_argument("--aimd-timeout-increase-step", type=int, default=1)
    parser.add_argument("--aimd-timeout-decrease-factor", type=float, default=0.5)
    parser.add_argument("--aimd-max-attempts-step", type=int, default=1)
    parser.add_argument("--aimd-backoff-step", type=int, default=1)
    parser.add_argument("--aimd-adjust-max-attempts", action="store_true")
    parser.add_argument("--aimd-adjust-backoff", action="store_true")
    parser.add_argument("--aimd-ema-alpha", type=float, default=0.5)
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
    args = parser.parse_args(argv)

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
    known = {
        "fixed",
        "aimd",
        "model_based",
        "model_based_queue",
        "bayesian",
        "ppo",
        # "sb3" is deliberately absent: it was accepted here but only "ppo" was
        # ever instantiated, so `--policies sb3` exited 0 with no rows.
        "oracle_fixed",
        "constrained_ppo",
    }
    unknown = [n for n in policy_names if n not in known]
    if unknown:
        # Silently dropping a typo produced a comparison table quietly missing
        # the arm the caller asked for.
        raise SystemExit(
            f"unknown --policies entries: {', '.join(unknown)}. Known: {', '.join(sorted(known))}"
        )
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
                raise ImportError(
                    "stable-baselines3 is required for PPO baseline"
                ) from _SB3_IMPORT_ERROR
            check_action_spec(args.ppo_load_path, action_spec, cfg.step_duration_s)
            model = PPO.load(args.ppo_load_path)
            policies.append(SB3Policy(model=model, deterministic=True, name="ppo"))
        else:
            ppo_policy = _train_ppo(cfg, learning_curve_csv=args.ppo_learning_curve_csv)
            if args.ppo_save_path:
                ppo_policy.model.save(args.ppo_save_path)
                save_action_spec(
                    args.ppo_save_path, action_spec, cfg.step_duration_s, cfg.fault_scenario
                )
                print(f"Saved PPO model to {args.ppo_save_path}")
            policies.append(ppo_policy)

    if "constrained_ppo" in policy_names:
        duration_s = cfg.episode_duration_s if cfg.episode_duration_s else None
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
