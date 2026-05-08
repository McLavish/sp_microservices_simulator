import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from rl.models.utils import ensure_positive, norm_cdf, norm_ppf, lognormal_logpdf


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

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        logx = np.log(x, where=x > 0, out=np.full_like(x, -math.inf))
        out = lognormal_logpdf(logx, self.mu, self.sigma)
        out = np.where(x > 0, out, -math.inf)
        return out

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        logx = np.log(x, where=x > 0, out=np.full_like(x, -math.inf))
        z = (logx - self.mu) / self.sigma
        out = np.where(x > 0, norm_cdf(z), 0.0)
        return out

    def quantile(self, q: float) -> float:
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return math.inf
        return float(math.exp(self.mu + self.sigma * norm_ppf(q)))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.lognormal(mean=self.mu, sigma=self.sigma, size=size)

    def params(self) -> Dict[str, float]:
        return {"mu": self.mu, "sigma": self.sigma}

    def log_likelihood(self, samples: np.ndarray) -> float:
        data = ensure_positive(samples)
        return float(np.sum(self.logpdf(data)))
