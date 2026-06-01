import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from rl.models.utils import ensure_positive


@dataclass
class LogNormalMixtureModel:
    n_components: int = 2
    max_iter: int = 100
    tol: float = 1e-4
    min_sigma: float = 1e-6
    weights: np.ndarray | None = field(default=None, init=False)
    mus: np.ndarray | None = field(default=None, init=False)
    sigmas: np.ndarray | None = field(default=None, init=False)
    _min_sample: float = field(default=0.0, init=False)
    _max_sample: float = field(default=0.0, init=False)
    _gmm: GaussianMixture | None = field(default=None, init=False, repr=False)

    def fit(self, samples: np.ndarray) -> "LogNormalMixtureModel":
        data = ensure_positive(samples)
        self._min_sample = float(np.min(data))
        self._max_sample = float(np.max(data))
        logx = np.log(data)
        if logx.size < self.n_components:
            raise ValueError("Not enough samples for mixture components")

        gmm = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=0,
        )
        gmm.fit(logx.reshape(-1, 1))

        self.weights = np.asarray(gmm.weights_, dtype=np.float64)
        self.mus = np.asarray(gmm.means_, dtype=np.float64).ravel()
        covariances = np.asarray(gmm.covariances_, dtype=np.float64).ravel()
        self.sigmas = np.maximum(np.sqrt(covariances), self.min_sigma)
        self._gmm = gmm
        return self

    def _fitted_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (weights, mus, sigmas), narrowing the Optional fields after fit()."""
        if self.weights is None or self.mus is None or self.sigmas is None:
            raise RuntimeError("LogNormalMixtureModel must be fit() before use")
        return self.weights, self.mus, self.sigmas

    def _fitted_gmm(self) -> GaussianMixture:
        """Return the fitted sklearn model, narrowing the Optional field after fit()."""
        if self._gmm is None:
            raise RuntimeError("LogNormalMixtureModel must be fit() before use")
        return self._gmm

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        gmm = self._fitted_gmm()
        x = np.asarray(x, dtype=np.float64)
        out = np.full_like(x, -math.inf, dtype=np.float64)
        mask = x > 0
        if not np.any(mask):
            return out
        logx = np.log(x[mask])
        out[mask] = gmm.score_samples(logx.reshape(-1, 1)) - logx
        return out

    def cdf(self, x: np.ndarray) -> np.ndarray:
        weights, mus, sigmas = self._fitted_params()
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float64)
        mask = x > 0
        if not np.any(mask):
            return out
        logx = np.log(x[mask])
        z = (logx[:, None] - mus[None, :]) / sigmas[None, :]
        cdfs = norm.cdf(z)
        out[mask] = np.sum(cdfs * weights[None, :], axis=1)
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
        weights, mus, sigmas = self._fitted_params()
        choices = rng.choice(self.n_components, size=size, p=weights)
        out = np.empty(size, dtype=np.float64)
        for j in range(self.n_components):
            mask = choices == j
            count = int(np.sum(mask))
            if count:
                out[mask] = rng.lognormal(mean=float(mus[j]), sigma=float(sigmas[j]), size=count)
        return out

    def params(self) -> dict[str, float]:
        weights, mus, sigmas = self._fitted_params()
        params = {}
        for i, (w, mu, sigma) in enumerate(zip(weights, mus, sigmas, strict=False)):
            params[f"w{i}"] = float(w)
            params[f"mu{i}"] = float(mu)
            params[f"sigma{i}"] = float(sigma)
        return params

    def log_likelihood(self, samples: np.ndarray) -> float:
        data = ensure_positive(samples)
        return float(np.sum(self.logpdf(data)))
