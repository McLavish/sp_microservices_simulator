from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass

from rl.cli import parse_int_list, parse_optional_int_list
from rl.env import SCENARIO_BUILDERS, ActionSpec, EnvConfig, MicroservicesEnv
from rl.model_io import check_action_spec, save_action_spec
from rl.rollout import model_actor, run_episode
from rl.sb3_support import import_sb3
from rl.spec import (
    DEFAULT_ATTEMPT_TIMEOUTS_MS,
    DEFAULT_BACKOFF_MS,
    DEFAULT_MAX_ATTEMPTS,
)

_SB3 = import_sb3()
PPO, make_vec_env, SubprocVecEnv = _SB3.ppo, _SB3.make_vec_env, _SB3.subproc_vec_env
_SB3_IMPORT_ERROR = _SB3.import_error


@dataclass
class TrainConfig:
    timesteps: int
    seed: int
    n_envs: int
    step_duration_s: float
    episode_duration_s: float
    attempt_timeouts_ms: list[int]
    max_attempts: list[int]
    backoff_ms: list[int]
    global_timeouts_ms: list[int | None] | None
    n_steps: int
    batch_size: int
    gamma: float
    save_path: str | None
    load_path: str | None
    eval_episodes: int
    fault_scenario: str = "none"
    device: str = "auto"


def _action_spec(cfg: TrainConfig) -> ActionSpec:
    return ActionSpec(
        attempt_timeout_ms=cfg.attempt_timeouts_ms,
        max_attempts=cfg.max_attempts,
        backoff_ms=cfg.backoff_ms,
        global_timeout_ms=cfg.global_timeouts_ms,
    )


def _make_env(cfg: TrainConfig) -> MicroservicesEnv:
    builder = SCENARIO_BUILDERS[cfg.fault_scenario]
    duration_s = cfg.episode_duration_s if cfg.episode_duration_s else None

    def build_fn(seed):
        return builder(seed, duration_s)

    action_spec = _action_spec(cfg)
    env_cfg = EnvConfig(
        build=build_fn,
        action_spec=action_spec,
        step_duration_s=cfg.step_duration_s,
        episode_duration_s=cfg.episode_duration_s,
    )
    return MicroservicesEnv(env_cfg)


def _evaluate(model, cfg: TrainConfig) -> None:
    total_rewards: list[float] = []
    for ep in range(cfg.eval_episodes):
        env = _make_env(cfg)
        result = run_episode(env, model_actor(model), seed=cfg.seed + 1000 + ep)
        total_rewards.append(result.total_reward)
        # The per-window summary is NOT an episode aggregate — it is recomputed
        # each step over that window's completions — so report the episode-level
        # figures explicitly and label the window as a window.
        last = result.windows[-1] if result.windows else None
        episode_success = result.success_rate(in_horizon=True)
        print(
            f"[eval] episode={ep} reward={result.total_reward:.4f} "
            f"success_rate_in_horizon={episode_success:.4f} "
            f"cost_pooled={result.pooled_cost:.4f} "
            f"action={last.action_values if last else {}} "
            f"last_window_summary={last.summary if last else {}}"
        )

    if total_rewards:
        avg = sum(total_rewards) / len(total_rewards)
        print(f"[eval] avg_reward={avg:.4f} over {len(total_rewards)} episode(s)")


def train(cfg: TrainConfig) -> None:
    if PPO is None or make_vec_env is None:
        raise ImportError("stable-baselines3 is required for training") from _SB3_IMPORT_ERROR

    def env_factory():
        return _make_env(cfg)

    vec_env = make_vec_env(
        env_factory,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        # Parallel rollout collection: the simulator dominates wall-clock, so
        # worker processes are the only meaningful speedup for tiny MLPs.
        vec_env_cls=SubprocVecEnv if cfg.n_envs > 1 else None,
    )

    if cfg.load_path:
        check_action_spec(cfg.load_path, _action_spec(cfg), cfg.step_duration_s)
        model = PPO.load(cfg.load_path, env=vec_env, device=cfg.device)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            seed=cfg.seed,
            device=cfg.device,
        )

    if cfg.timesteps > 0:
        model.learn(total_timesteps=cfg.timesteps)

    if cfg.save_path:
        save_dir = os.path.dirname(cfg.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        model.save(cfg.save_path)
        save_action_spec(cfg.save_path, _action_spec(cfg), cfg.step_duration_s, cfg.fault_scenario)
        print(f"[train] saved model to {cfg.save_path}")

    if cfg.eval_episodes > 0:
        _evaluate(model, cfg)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Stable-Baselines3 agent on MicroservicesEnv."
    )
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--step-duration-s", type=float, default=1.0)
    parser.add_argument("--episode-duration-s", type=float, default=10.0)
    parser.add_argument(
        "--attempt-timeouts-ms", type=parse_int_list, default=list(DEFAULT_ATTEMPT_TIMEOUTS_MS)
    )
    parser.add_argument("--max-attempts", type=parse_int_list, default=list(DEFAULT_MAX_ATTEMPTS))
    parser.add_argument("--backoff-ms", type=parse_int_list, default=list(DEFAULT_BACKOFF_MS))
    parser.add_argument("--global-timeouts-ms", type=parse_optional_int_list, default=None)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--fault-scenario", choices=sorted(SCENARIO_BUILDERS), default="none")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='PPO device ("auto", "cpu", "cuda"); SB3 recommends cpu for small MLPs.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """`argv` defaults to sys.argv; taking it explicitly makes the entry
    point callable from a test without patching sys.argv."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        step_duration_s=args.step_duration_s,
        episode_duration_s=args.episode_duration_s,
        attempt_timeouts_ms=list(args.attempt_timeouts_ms),
        max_attempts=list(args.max_attempts),
        backoff_ms=list(args.backoff_ms),
        global_timeouts_ms=args.global_timeouts_ms,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        save_path=args.save_path,
        load_path=args.load_path,
        eval_episodes=args.eval_episodes,
        fault_scenario=args.fault_scenario,
        device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
