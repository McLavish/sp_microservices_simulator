from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class LatencyModel(Protocol):
    def fit(self, samples: np.ndarray) -> LatencyModel: ...

    def logpdf(self, x: np.ndarray) -> np.ndarray: ...

    def cdf(self, x: np.ndarray) -> np.ndarray: ...

    def quantile(self, q: float) -> float: ...

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray: ...

    def params(self) -> dict[str, float]: ...

    def log_likelihood(self, samples: np.ndarray) -> float: ...


@dataclass
class ModelReport:
    name: str
    params: dict[str, float]
    nll: float
    p95_error: float
    p99_error: float
