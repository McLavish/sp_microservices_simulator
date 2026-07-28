import math
from dataclasses import dataclass, field

import numpy as np

from rl.models.lognormal import LogNormalModel
from rl.models.utils import ensure_positive


def _log_or_neg_inf(values: np.ndarray) -> np.ndarray:
    """Elementwise log, with -inf where the value is non-positive."""
    out = np.full_like(values, -math.inf, dtype=np.float64)
    np.log(values, where=values > 0, out=out)
    return out


@dataclass
class EVTTailModel:
    """Lognormal body with a Pareto tail above a quantile threshold.

    The tail index is the Hill estimator (MLE for a pure Pareto tail), not a
    generalized-Pareto peaks-over-threshold fit — despite the EVT name, this
    assumes the tail is exactly power-law above ``u``.
    """

    threshold_quantile: float = 0.95
    min_alpha: float = 0.1
    max_alpha: float = 1e6
    body: LogNormalModel = field(default_factory=LogNormalModel)
    u: float = field(default=0.0, init=False)
    alpha: float = field(default=1.0, init=False)
    p_u: float = field(default=0.0, init=False)
    _body_cdf_u: float = field(default=1.0, init=False)

    def fit(self, samples: np.ndarray) -> "EVTTailModel":
        data = ensure_positive(samples)
        self.u = float(np.quantile(data, self.threshold_quantile))
        body_samples = data[data <= self.u]
        tail_samples = data[data > self.u]

        self.p_u = float(body_samples.size / data.size)
        if body_samples.size == 0:
            body_samples = data
            self.p_u = 0.0

        # `body_samples` is a right-TRUNCATED sample (everything above u was
        # removed), so an ordinary MLE biases both parameters toward zero and
        # every body quantile comes out low — the p_u/_body_cdf_u
        # renormalization fixes the mass, not the shape. Treating the tail
        # observations as right-censored at u recovers an unbiased body.
        fit_censored = getattr(self.body, "fit_censored", None)
        if callable(fit_censored) and tail_samples.size > 0:
            # Right-censored AT u: the observation for an exceedance is the
            # THRESHOLD ("we know only that it exceeded u"), not its actual
            # value. Passing the actual values leaks tail information into the
            # body fit (measured sigma 0.8384 vs 0.8004 against a true 0.8).
            censored_at_u = np.where(data > self.u, self.u, data)
            self.body = fit_censored(censored_at_u, data > self.u)
        else:
            self.body.fit(body_samples)
        self._body_cdf_u = float(self.body.cdf(np.array([self.u]))[0])
        if self._body_cdf_u <= 0:
            self._body_cdf_u = 1e-12

        if tail_samples.size > 0:
            ratios = tail_samples / self.u
            log_ratios = np.log(ratios)
            denom = float(np.sum(log_ratios))
            # Samples clustered just above the threshold drive the Hill
            # estimator arbitrarily high (observed alpha ~ 4e5) — a signal the
            # tail is not power-law. The log-space density handles those values
            # fine, so max_alpha is only a guard against inf/nan from a
            # degenerate denominator, deliberately far above any plausible fit
            # rather than a modelling constraint that would distort the tail.
            alpha = tail_samples.size / denom if denom > 0 else self.max_alpha
            self.alpha = float(min(max(alpha, self.min_alpha), self.max_alpha))
        else:
            self.alpha = 10.0

        return self

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.full_like(x, -math.inf, dtype=np.float64)
        mask = x > 0
        if not np.any(mask):
            return out

        idx = np.where(mask)[0]
        x_pos = x[mask]
        body_mask = x_pos <= self.u
        tail_mask = ~body_mask

        if np.any(body_mask):
            body_pdf = np.exp(self.body.logpdf(x_pos[body_mask]))
            scale = self.p_u / self._body_cdf_u if self.p_u > 0 else 0.0
            out[idx[body_mask]] = _log_or_neg_inf(body_pdf * scale)

        if np.any(tail_mask):
            x_tail = x_pos[tail_mask]
            scale = 1.0 - self.p_u
            if scale <= 0.0:
                out[idx[tail_mask]] = -math.inf
            else:
                # Evaluate in LOG space. Done in probability space, u**alpha is a
                # Python float power that RAISES OverflowError once alpha is
                # large (tightly-clustered samples drive the Hill estimator up),
                # and even below that it multiplies ~1e213 by ~1e-215 instead of
                # computing the density.
                out[idx[tail_mask]] = (
                    math.log(scale)
                    + math.log(self.alpha)
                    + self.alpha * math.log(self.u)
                    - (self.alpha + 1.0) * np.log(x_tail)
                )

        return out

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float64)
        mask = x > 0
        if not np.any(mask):
            return out

        idx = np.where(mask)[0]
        x_pos = x[mask]
        body_mask = x_pos <= self.u
        tail_mask = ~body_mask

        if np.any(body_mask):
            body_cdf = self.body.cdf(x_pos[body_mask])
            scale = self.p_u / self._body_cdf_u if self.p_u > 0 else 0.0
            out[idx[body_mask]] = body_cdf * scale

        if np.any(tail_mask):
            x_tail = x_pos[tail_mask]
            tail_cdf = 1.0 - (self.u / x_tail) ** self.alpha
            out[idx[tail_mask]] = self.p_u + (1.0 - self.p_u) * tail_cdf

        return out

    def quantile(self, q: float) -> float:
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return math.inf

        if q <= self.p_u and self.p_u > 0:
            scaled_q = q * self._body_cdf_u / self.p_u
            return float(self.body.quantile(min(max(scaled_q, 1e-9), 1.0 - 1e-9)))

        tail_q = (q - self.p_u) / max(1.0 - self.p_u, 1e-12)
        return float(self.u / (1.0 - tail_q) ** (1.0 / self.alpha))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        u = rng.random(size)
        out = np.empty(size, dtype=np.float64)
        body_mask = u < self.p_u
        tail_mask = ~body_mask

        if np.any(body_mask):
            needed = int(np.sum(body_mask))
            collected = []
            while len(collected) < needed:
                samples = self.body.sample(rng, needed)
                kept = samples[samples <= self.u]
                collected.extend(kept.tolist())
            out[body_mask] = np.array(collected[:needed], dtype=np.float64)

        if np.any(tail_mask):
            r = rng.random(int(np.sum(tail_mask)))
            out[tail_mask] = self.u / (1.0 - r) ** (1.0 / self.alpha)

        return out

    def params(self) -> dict[str, float]:
        return {
            "u": self.u,
            "alpha": self.alpha,
            "p_u": self.p_u,
            **self.body.params(),
        }

    def log_likelihood(self, samples: np.ndarray) -> float:
        data = ensure_positive(samples)
        return float(np.sum(self.logpdf(data)))
