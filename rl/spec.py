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
    # Weight on the constraint cost (cost-client retry amplification). Acts as
    # the Lagrange multiplier in constrained training; 0 disables the term.
    cost: float = 0.0


@dataclass
class ActionSpec:
    attempt_timeout_ms: list[int]
    max_attempts: list[int]
    backoff_ms: list[int]
    global_timeout_ms: list[int | None] | None = None

    def __post_init__(self) -> None:
        """Reject grids that would fail late, deep inside a rollout.

        An empty grid gives a zero-sized action dimension; a timeout of 0 makes
        the deadline arithmetic divide by zero; a negative backoff schedules an
        event in the past; max_attempts=0 still issues the initial attempt, so
        the knob silently means nothing. AIMD additionally assumes the timeout
        grid is ascending (its multiplicative decrease snaps to a lower index).
        """
        for name, values, minimum in (
            ("attempt_timeout_ms", self.attempt_timeout_ms, 1),
            ("max_attempts", self.max_attempts, 1),
            ("backoff_ms", self.backoff_ms, 0),
        ):
            if not values:
                raise ValueError(f"ActionSpec.{name} must not be empty")
            if any(v < minimum for v in values):
                raise ValueError(f"ActionSpec.{name} values must be >= {minimum}, got {values}")
            if list(values) != sorted(values):
                raise ValueError(
                    f"ActionSpec.{name} must be ascending (AIMD snaps to adjacent "
                    f"indices and assumes monotonic values), got {values}"
                )
        if self.global_timeout_ms is not None:
            if not self.global_timeout_ms:
                raise ValueError("ActionSpec.global_timeout_ms must not be empty when given")
            numeric = [v for v in self.global_timeout_ms if v is not None]
            if any(v < 1 for v in numeric):
                raise ValueError(
                    f"ActionSpec.global_timeout_ms values must be >= 1, got {self.global_timeout_ms}"
                )


@dataclass(frozen=True)
class ActionValues:
    """Decoded knob values for one action (indices resolved against ActionSpec)."""

    attempt_timeout_ms: int
    max_attempts: int
    backoff_ms: int
    global_timeout_ms: int | None


def coerce_action(action: Any) -> tuple[int, ...]:
    if isinstance(action, np.ndarray):
        action = action.tolist()
    if isinstance(action, (list, tuple)):
        return tuple(int(a) for a in action)
    return (int(action),)
