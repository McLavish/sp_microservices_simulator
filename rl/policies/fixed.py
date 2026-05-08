from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from rl.policies.action_space import ActionGrid


@dataclass
class FixedPolicy:
    action: Tuple[int, ...]
    name: str = "fixed"

    def reset(self, seed: Optional[int] = None) -> None:
        return

    def act(self, obs: np.ndarray, info: dict) -> Tuple[int, ...]:
        return self.action

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return


@dataclass
class FixedValuePolicy:
    grid: ActionGrid
    attempt_timeout_ms: int
    max_attempts: int
    backoff_ms: int
    global_timeout_ms: Optional[int] = None
    name: str = "fixed_values"

    def __post_init__(self):
        self._action = self.grid.nearest_indices(
            self.attempt_timeout_ms,
            self.max_attempts,
            self.backoff_ms,
            self.global_timeout_ms,
        )

    def reset(self, seed: Optional[int] = None) -> None:
        return

    def act(self, obs: np.ndarray, info: dict) -> Tuple[int, ...]:
        return self._action

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return
