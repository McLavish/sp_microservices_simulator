import math
from typing import Iterable

import numpy as np
from scipy.stats import norm

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(z: np.ndarray) -> np.ndarray:
    return norm.cdf(np.asarray(z, dtype=np.float64))


def norm_ppf(p: float) -> float:
    return float(norm.ppf(p))


def logsumexp(log_values: np.ndarray, axis: int = -1) -> np.ndarray:
    max_val = np.max(log_values, axis=axis, keepdims=True)
    shifted = log_values - max_val
    return (max_val + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))).squeeze(axis)


def ensure_positive(samples: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(samples), dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        raise ValueError("No positive samples provided")
    return arr


def lognormal_logpdf(logx: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (
        -0.5 * ((logx - mu) / sigma) ** 2
        - logx
        - math.log(sigma * _SQRT_2PI)
    )
