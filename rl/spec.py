"""Shared, dependency-light RL specs.

These pure data types (and the action coercion helper) are imported by both
``rl.env`` and ``rl.policies``. Keeping them here avoids the import cycle that
would otherwise form between the environment and the policy/action-grid modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

# Default action grid for every RL entry point (`rl.train_sb3`,
# `rl.experiments.compare_policies`, `rl.experiments.constrained_retry`). One
# definition rather than three copies: `rl.model_io.check_action_spec` refuses
# to load a saved model into a differing grid, so changing the grid in one entry
# point hard-failed the documented save-then-load workflow of another. Parsers
# materialise their own list, so no parser can mutate the shared default.
DEFAULT_ATTEMPT_TIMEOUTS_MS: Final[tuple[int, ...]] = (10, 30, 50, 80)
DEFAULT_MAX_ATTEMPTS: Final[tuple[int, ...]] = (1, 2, 3)
DEFAULT_BACKOFF_MS: Final[tuple[int, ...]] = (0, 20, 50)

# Seed lists in the form argparse stores them: strings, so the `parse_int_list`
# converter runs on the default exactly as it does on a value from the CLI.
# Shared because separately reported experiments are only comparable when their
# arms are evaluated on the same seeds (`--train-seeds` in compare_policies is
# the same population as `--oracle-seeds` in constrained_retry).
DEFAULT_TRAIN_SEEDS: Final[str] = "123,456,789"
DEFAULT_EVAL_SEEDS: Final[str] = "321,654,987,1320,1653,1986,2319,2652,2985,3318"


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


def linear_reward(
    summary: Mapping[str, float], weights: RewardWeights, *, cost_term: float = 0.0
) -> float:
    """The weighted linear form over a window summary — the one copy.

    Both `MicroservicesEnv._reward_from_summary` (the training signal) and
    `EpisodeResult.objective` (the base objective J every arm is scored and
    constrained on) are this expression; J is documented as "the reward with the
    retry and cost penalties zeroed" and gets there by passing zeroed weights.
    Written out twice, that relation held only by convention: a term added to
    the reward would leave J silently meaning something else while the
    experiment CSVs, the Pareto plot and the dual-ascent update kept reporting
    it as the reward-with-penalties-zeroed.

    `cost_term` is a parameter rather than a summary key because the constraint
    cost is not a plain rate: the env penalises the linearized constraint
    `cost_retries - cost_budget * cost_roots` (or, with no budget configured, a
    normalised retry count) — see `MicroservicesEnv._cost_penalty`.
    """
    return float(
        summary["success_rate"] * weights.success
        - summary["deadline_drop_rate"] * weights.deadline_drop
        - summary["queue_drop_rate"] * weights.queue_drop
        - summary["server_drop_rate"] * weights.server_drop
        - summary["circuit_open_drop_rate"] * weights.circuit_open_drop
        - summary["p95_latency_ms"] * weights.p95_latency_ms
        - summary["retries_per_root"] * weights.retries
        - cost_term * weights.cost
    )


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
