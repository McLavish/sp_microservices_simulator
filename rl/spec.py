"""Shared, dependency-light RL specs.

These pure data types (and the action coercion helper) are imported by both
``rl.env`` and ``rl.policies``. Keeping them here avoids the import cycle that
would otherwise form between the environment and the policy/action-grid modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardWeights:
    success: float = 1.0
    deadline_drop: float = 1.0
    queue_drop: float = 1.0
    server_drop: float = 1.0
    circuit_open_drop: float = 1.0
    p95_latency_ms: float = 0.001
    retries: float = 0.1


@dataclass
class ActionSpec:
    attempt_timeout_ms: list[int]
    max_attempts: list[int]
    backoff_ms: list[int]
    global_timeout_ms: list[int | None] | None = None


def coerce_action(action: Any) -> tuple[int, ...]:
    if isinstance(action, np.ndarray):
        action = action.tolist()
    if isinstance(action, (list, tuple)):
        return tuple(int(a) for a in action)
    return (int(action),)
