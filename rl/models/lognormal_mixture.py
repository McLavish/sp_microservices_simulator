import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from rl.models.utils import (
    ensure_positive,
    lognormal_logpdf,
    logsumexp,
    norm_cdf,
)


@dataclass
class LogNormalMixtureModel:
    n_components: int = 2
    max_iter: int = 100
    tol: float = 1e-4
    min_sigma: float = 1e-6
    weights: Optional[np.ndarray] = field(default=None, init=False)
    mus: Optional[np.ndarray] = field(default=None, init=False)
    sigmas: Optional[np.ndarray] = field(default=None, init=False)
    _min_sample: float = field(default=0.0, init=False)
    _max_sample: float = field(default=0.0, init=False)

    def fit(self, samples: np.ndarray) -> "LogNormalMixtureModel":
        data = ensure_positive(samples)
        self._min_sample = float(np.min(data))
        self._max_sample = float(np.max(data))
        logx = np.log(data)
        n = logx.size
        if n < self.n_components:
            raise ValueError("Not enough samples for mixture components")

        qs = np.linspace(0.2, 0.8, self.n_components)
        mus = np.quantile(logx, qs)
        sigma_init = float(np.std(logx, ddof=1)) if n > 1 else 1.0
        sigmas = np.full(self.n_components, max(sigma_init, self.min_sigma))
        weights = np.full(self.n_components, 1.0 / self.n_components)

        prev_ll = -math.inf
        for _ in range(self.max_iter):
            log_probs = []
            for j in range(self.n_components):
                comp_logpdf = (
                    -0.5 * ((logx - mus[j]) / sigmas[j]) ** 2
                    - math.log(sigmas[j] * math.sqrt(2.0 * math.pi))
                )
                log_probs.append(np.log(weights[j]) + comp_logpdf)
            log_probs = np.vstack(log_probs).T
            log_sum = logsumexp(log_probs, axis=1)
            ll = float(np.sum(log_sum))

            resp = np.exp(log_probs - log_sum[:, None])
            weights = np.mean(resp, axis=0)
            weights = weights / np.sum(weights)

            for j in range(self.n_components):
                r = resp[:, j]
                r_sum = np.sum(r)
                if r_sum <= 0:
                    continue
                mus[j] = np.sum(r * logx) / r_sum
                var = np.sum(r * (logx - mus[j]) ** 2) / r_sum
                sigmas[j] = max(math.sqrt(var), self.min_sigma)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.weights = weights
        self.mus = mus
        self.sigmas = sigmas
        return self

    def _logpdf_components(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        logx = np.log(x, where=x > 0, out=np.full_like(x, -math.inf))
        logpdf = []
        for w, mu, sigma in zip(self.weights, self.mus, self.sigmas):
            logpdf.append(np.log(w) + lognormal_logpdf(logx, float(mu), float(sigma)))
        return np.vstack(logpdf).T

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.full_like(x, -math.inf, dtype=np.float64)
        mask = x > 0
        if not np.any(mask):
            return out
        log_comp = self._logpdf_components(x[mask])
        out[mask] = logsumexp(log_comp, axis=1)
        return out

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float64)
        mask = x > 0
        if not np.any(mask):
            return out
        logx = np.log(x[mask])
        z = (logx[:, None] - self.mus[None, :]) / self.sigmas[None, :]
        cdfs = norm_cdf(z)
        out[mask] = np.sum(cdfs * self.weights[None, :], axis=1)
        return out

    def quantile(self, q: float) -> float:
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return math.inf

        lo = max(self._min_sample * 0.5, 1e-12)
        hi = max(self._max_sample * 2.0, lo * 10.0)

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.cdf(np.array([mid]))[0] < q:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        choices = rng.choice(self.n_components, size=size, p=self.weights)
        out = np.empty(size, dtype=np.float64)
        for j in range(self.n_components):
            mask = choices == j
            count = int(np.sum(mask))
            if count:
                out[mask] = rng.lognormal(
                    mean=float(self.mus[j]), sigma=float(self.sigmas[j]), size=count
                )
        return out

    def params(self) -> Dict[str, float]:
        params = {}
        for i, (w, mu, sigma) in enumerate(zip(self.weights, self.mus, self.sigmas)):
            params[f"w{i}"] = float(w)
            params[f"mu{i}"] = float(mu)
            params[f"sigma{i}"] = float(sigma)
        return params

    def log_likelihood(self, samples: np.ndarray) -> float:
        data = ensure_positive(samples)
        return float(np.sum(self.logpdf(data)))
