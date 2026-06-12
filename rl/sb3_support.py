"""Shared optional-import shim for stable-baselines3 (the opt-in `rl` group)."""

from __future__ import annotations

from typing import Any


def import_sb3() -> tuple[Any, Any, Any, Any, Exception | None]:
    """Returns (PPO, make_vec_env, evaluate_policy, SubprocVecEnv, import_error)."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        return None, None, None, None, exc
    return PPO, make_vec_env, evaluate_policy, SubprocVecEnv, None
