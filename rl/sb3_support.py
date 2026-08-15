"""Shared optional-import shim for stable-baselines3 (the opt-in `rl` group)."""

from __future__ import annotations

from typing import Any, NamedTuple


class SB3Imports(NamedTuple):
    """Named slots so call sites bind by attribute, never by tuple position.

    Every slot is `Any` — the modules are absent without the `rl` group — so a
    positional unpack that drifted out of order type-checks fine and only
    surfaces as a wrong argument deep inside SB3.
    """

    ppo: Any
    make_vec_env: Any
    subproc_vec_env: Any
    import_error: Exception | None


def import_sb3() -> SB3Imports:
    """Import the SB3 pieces the RL entry points need, or report why it failed.

    `import_error` is the caught exception: the `except` is deliberately broad
    (a torch/CUDA ABI failure inside an *installed* stable-baselines3 looks
    nothing like a missing package), so callers must chain it — `raise
    ImportError(...) from import_error` — or the user is sent to reinstall a
    package that is already there.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        return SB3Imports(None, None, None, exc)
    return SB3Imports(PPO, make_vec_env, SubprocVecEnv, None)
