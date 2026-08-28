import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import CensoredData, FitError, lognorm

from rl.models.utils import ensure_positive


@dataclass
class LogNormalModel:
    mu: float = 0.0
    sigma: float = 1.0
    min_sigma: float = 1e-6

    def fit(self, samples: np.ndarray) -> "LogNormalModel":
        data = ensure_positive(samples)
        logx = np.log(data)
        self.mu = float(np.mean(logx))
        sigma = float(np.std(logx, ddof=1)) if logx.size > 1 else self.min_sigma
        self.sigma = max(sigma, self.min_sigma)
        return self

    def fit_censored(self, samples: np.ndarray, censored: np.ndarray) -> "LogNormalModel":
        """Right-censored MLE: censored entries are timeout-truncated, i.e. the
        true latency exceeds the recorded value."""
        values = np.asarray(samples, dtype=np.float64)
        flags = np.asarray(censored, dtype=bool)
        positive = values > 0
        values, flags = values[positive], flags[positive]
        uncensored = values[~flags]
        right = values[flags]
        if uncensored.size == 0:
            raise ValueError("Censored MLE requires at least one uncensored sample")
        if right.size == 0:
            return self.fit(uncensored)
        try:
            shape, _loc, scale = lognorm.fit(
                CensoredData(uncensored=uncensored, right=right), floc=0
            )
        except FitError:
            # Numerical MLE failed to converge; fall back to the (optimistically
            # biased) uncensored-only fit rather than crashing a long experiment.
            return self.fit(uncensored)
        self.sigma = max(float(shape), self.min_sigma)
        self.mu = float(math.log(scale))
        return self

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return lognorm.logpdf(x, s=self.sigma, scale=math.exp(self.mu))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return lognorm.cdf(x, s=self.sigma, scale=math.exp(self.mu))

    def quantile(self, q: float) -> float:
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return math.inf
        return float(lognorm.ppf(q, s=self.sigma, scale=math.exp(self.mu)))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.lognormal(mean=self.mu, sigma=self.sigma, size=size)

    def params(self) -> dict[str, float]:
        return {"mu": self.mu, "sigma": self.sigma}

    def log_likelihood(self, samples: np.ndarray) -> float:
        data = ensure_positive(samples)
        return float(np.sum(self.logpdf(data)))
