"""Shared optional-import shim for stable-baselines3 (the opt-in `rl` group)."""

from __future__ import annotations

from typing import Any


def import_sb3() -> tuple[Any, Any, Any, Exception | None]:
    """Returns (PPO, make_vec_env, evaluate_policy, import_error); Nones when missing."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.evaluation import evaluate_policy
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        return None, None, None, exc
    return PPO, make_vec_env, evaluate_policy, None
