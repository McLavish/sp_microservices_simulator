from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from rl.env import RewardWeights
from rl.models.lognormal import LogNormalModel
from rl.policies.action_space import ActionGrid
from rl.policies.model_based import _simulate_action, LatencyBuffer


@dataclass
class NormalInverseGamma:
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 2.0
    beta0: float = 2.0

    def posterior(self, data: np.ndarray):
        if data.size == 0:
            return self.mu0, self.kappa0, self.alpha0, self.beta0

        n = data.size
        mean = float(np.mean(data))
        ssd = float(np.sum((data - mean) ** 2))

        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * mean) / kappa_n
        alpha_n = self.alpha0 + n / 2.0
        beta_n = (
            self.beta0
            + 0.5 * ssd
            + (self.kappa0 * n * (mean - self.mu0) ** 2) / (2.0 * kappa_n)
        )
        return mu_n, kappa_n, alpha_n, beta_n

    def sample(self, rng: np.random.Generator, data: np.ndarray) -> Tuple[float, float]:
        mu_n, kappa_n, alpha_n, beta_n = self.posterior(data)
        sigma2 = 1.0 / rng.gamma(shape=alpha_n, scale=1.0 / beta_n)
        mu = rng.normal(loc=mu_n, scale=np.sqrt(sigma2 / kappa_n))
        sigma = float(np.sqrt(sigma2))
        return float(mu), float(max(sigma, 1e-6))


@dataclass
class BayesianPolicy:
    grid: ActionGrid
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    sample_size: int = 500
    buffer_size: int = 2000
    use_censored: bool = False
    prior: NormalInverseGamma = field(default_factory=NormalInverseGamma)
    name: str = "bayesian"

    def __post_init__(self):
        self._buffer = LatencyBuffer(self.buffer_size)
        self._rng = np.random.default_rng(123)

    def reset(self, seed: Optional[int] = None) -> None:
        self._buffer = LatencyBuffer(self.buffer_size)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict) -> Tuple[int, ...]:
        self._buffer.ingest_info(info, self.use_censored)
        data = self._buffer.data()
        if data.size == 0:
            return self.grid.indices[0]

        log_data = np.log(data)
        mu, sigma = self.prior.sample(self._rng, log_data)
        model = LogNormalModel(mu=mu, sigma=sigma)

        best_idx = self.grid.indices[0]
        best_value = -float("inf")
        for action_idx, action_vals in zip(self.grid.indices, self.grid.values):
            reward, _ = _simulate_action(
                model,
                action_vals,
                self.reward_weights,
                self._rng,
                self.sample_size,
                self.use_censored,
                0.0,
                "deterministic",
            )
            if reward > best_value:
                best_value = reward
                best_idx = action_idx

        return best_idx

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return
