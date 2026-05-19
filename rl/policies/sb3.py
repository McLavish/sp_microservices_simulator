from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from rl.policies.action_space import coerce_action

try:
    from stable_baselines3.common.base_class import BaseAlgorithm
except Exception:  # pragma: no cover - optional dependency
    BaseAlgorithm = None


@dataclass
class SB3Policy:
    model: "BaseAlgorithm"
    deterministic: bool = True
    name: str = "sb3"

    def reset(self, seed: Optional[int] = None) -> None:
        return

    def act(self, obs: np.ndarray, info: dict) -> Tuple[int, ...]:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return coerce_action(action)

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return
