from dataclasses import dataclass

import numpy as np

from rl.policies.action_space import UNSPECIFIED, ActionGrid


@dataclass
class FixedPolicy:
    action: tuple[int, ...]
    name: str = "fixed"

    def reset(self, seed: int | None = None) -> None:
        return

    def act(self, obs: np.ndarray, info: dict) -> tuple[int, ...]:
        return self.action

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return


@dataclass
class FixedValuePolicy:
    grid: ActionGrid
    attempt_timeout_ms: int
    max_attempts: int
    backoff_ms: int
    global_timeout_ms: int | None = None
    name: str = "fixed_values"

    def __post_init__(self):
        # `None` here means "this policy has no global-timeout opinion", NOT
        # "select the spec's None entry" — passing it through literally made a
        # numeric-only grid such as [100, 200] raise.
        self._action = self.grid.nearest_indices(
            self.attempt_timeout_ms,
            self.max_attempts,
            self.backoff_ms,
            UNSPECIFIED if self.global_timeout_ms is None else self.global_timeout_ms,
        )

    def reset(self, seed: int | None = None) -> None:
        return

    def act(self, obs: np.ndarray, info: dict) -> tuple[int, ...]:
        return self._action

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return
