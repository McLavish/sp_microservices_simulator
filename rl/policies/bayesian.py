from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rl.models.lognormal import LogNormalModel
from rl.policies.action_space import ActionGrid
from rl.policies.model_based import LatencyBuffer, _argmax_over_grid
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
            # Cold start: no posterior update yet, so start mid-grid like the
            # env's own warm-up action instead of at the pathological corner.
            return self.grid.midpoint_indices()

        log_data = np.log(data)
        mu, sigma = self.prior.sample(self._rng, log_data)
        model = LogNormalModel(mu=mu, sigma=sigma)

        # Same common-random-numbers scoring as the model-based policy; the
        # Thompson draw above is what makes this one Bayesian. There is no
        # queueing term here, hence the (0.0, "deterministic") arguments.
        return _argmax_over_grid(
            self.grid,
            model,
            self.reward_weights,
            self._rng,
            self.sample_size,
            0.0,
            "deterministic",
        )

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return
