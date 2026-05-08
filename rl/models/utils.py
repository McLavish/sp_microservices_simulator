import math
from typing import Iterable

import numpy as np

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(z / _SQRT_2))


def norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    # Acklam's inverse normal approximation.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4])
            * r
            + 1.0
        )
    )


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
