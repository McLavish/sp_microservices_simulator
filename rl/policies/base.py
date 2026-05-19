from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np


class Policy(Protocol):
    name: str

    def reset(self, seed: Optional[int] = None) -> None:
        ...

    def act(self, obs: np.ndarray, info: dict) -> Tuple[int, ...]:
        ...

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        ...
