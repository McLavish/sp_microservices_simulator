from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rl.models.lognormal import LogNormalModel
from rl.policies.action_space import ActionGrid
from rl.policies.model_based import LatencyBuffer, _action_cost_key, _simulate_action
from rl.spec import RewardWeights


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
            self.beta0 + 0.5 * ssd + (self.kappa0 * n * (mean - self.mu0) ** 2) / (2.0 * kappa_n)
        )
        return mu_n, kappa_n, alpha_n, beta_n

    def sample(self, rng: np.random.Generator, data: np.ndarray) -> tuple[float, float]:
        mu_n, kappa_n, alpha_n, beta_n = self.posterior(data)
        sigma2 = 1.0 / rng.gamma(shape=alpha_n, scale=1.0 / beta_n)
        mu = rng.normal(loc=mu_n, scale=np.sqrt(sigma2 / kappa_n))
        sigma = float(np.sqrt(sigma2))
        return float(mu), float(max(sigma, 1e-6))


@dataclass
class BayesianPolicy:
    """Thompson-sampling policy with a Normal-Inverse-Gamma prior on log-latency.

    Known limitation: the conjugate update needs exact observations, so
    timeout-censored attempts are dropped from the posterior. Under frequent
    timeouts the posterior is biased optimistic (compare `LogNormalModel.
    fit_censored`, which the model-based policy uses instead).
    """

    grid: ActionGrid
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    sample_size: int = 500
    buffer_size: int = 2000
    prior: NormalInverseGamma = field(default_factory=NormalInverseGamma)
    name: str = "bayesian"

    def __post_init__(self):
        self._buffer = LatencyBuffer(self.buffer_size)
        self._rng = np.random.default_rng(123)

    def reset(self, seed: int | None = None) -> None:
        self._buffer = LatencyBuffer(self.buffer_size)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict) -> tuple[int, ...]:
        self._buffer.ingest_info(info)
        data = self._buffer.data()
        data = data[data > 0]  # defense in depth: log(0) would collapse the posterior
        if data.size == 0:
            return self.grid.indices[0]

        log_data = np.log(data)
        mu, sigma = self.prior.sample(self._rng, log_data)
        model = LogNormalModel(mu=mu, sigma=sigma)

        # Common random numbers: score every candidate against the SAME draws.
        # With independent sequential streams the sampling error dominates the
        # true spread among near-tied actions, so the argmax depended on the
        # order the grid happened to be enumerated in — reversing the same
        # physical grid changed the selected action. Also removes the
        # winner's-curse bias in taking a max over noisy estimates.
        state = self._rng.bit_generator.state
        best_idx = self.grid.indices[0]
        best_value = -float("inf")
        best_key: tuple = ()
        for action_idx, action_vals in zip(self.grid.indices, self.grid.values, strict=False):
            self._rng.bit_generator.state = state
            reward = _simulate_action(
                model,
                action_vals,
                self.reward_weights,
                self._rng,
                self.sample_size,
                0.0,
                "deterministic",
            )
            if reward > best_value or (
                reward == best_value and _action_cost_key(action_vals) < best_key
            ):
                best_value = reward
                best_idx = action_idx
                best_key = _action_cost_key(action_vals)

        # Advance once so consecutive steps do not reuse the same draws.
        self._rng.bit_generator.state = state
        self._rng.random()
        return best_idx

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return
