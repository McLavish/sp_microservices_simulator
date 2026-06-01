from collections.abc import Iterable

import numpy as np


def ensure_positive(samples: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(samples), dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        raise ValueError("No positive samples provided")
    return arr
