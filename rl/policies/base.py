from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, obs: np.ndarray, info: dict) -> tuple[int, ...]: ...

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None: ...
