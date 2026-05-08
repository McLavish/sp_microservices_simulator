from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from rl.env import ActionSpec


@dataclass(frozen=True)
class ActionValues:
    attempt_timeout_ms: int
    max_attempts: int
    backoff_ms: int
    global_timeout_ms: Optional[int]


class ActionGrid:
    def __init__(self, spec: ActionSpec):
        self._spec = spec
        self._indices: List[Tuple[int, ...]] = []
        self._values: List[ActionValues] = []
        self._build()

    @property
    def spec(self) -> ActionSpec:
        return self._spec

    @property
    def indices(self) -> List[Tuple[int, ...]]:
        return self._indices

    @property
    def values(self) -> List[ActionValues]:
        return self._values

    def _build(self) -> None:
        for i, timeout in enumerate(self._spec.attempt_timeout_ms):
            for j, max_attempts in enumerate(self._spec.max_attempts):
                for k, backoff in enumerate(self._spec.backoff_ms):
                    if self._spec.global_timeout_ms is None:
                        indices = (i, j, k)
                        values = ActionValues(timeout, max_attempts, backoff, None)
                        self._indices.append(indices)
                        self._values.append(values)
                    else:
                        for g, global_timeout in enumerate(self._spec.global_timeout_ms):
                            indices = (i, j, k, g)
                            values = ActionValues(
                                timeout, max_attempts, backoff, global_timeout
                            )
                            self._indices.append(indices)
                            self._values.append(values)

    def values_for(self, indices: Sequence[int]) -> ActionValues:
        idx = tuple(int(i) for i in indices)
        if self._spec.global_timeout_ms is None and len(idx) != 3:
            raise ValueError("Expected 3 indices for action")
        if self._spec.global_timeout_ms is not None and len(idx) != 4:
            raise ValueError("Expected 4 indices for action")
        i, j, k = idx[:3]
        g_idx = idx[3] if len(idx) > 3 else None
        global_timeout = (
            None
            if g_idx is None
            else self._spec.global_timeout_ms[g_idx]
        )
        return ActionValues(
            self._spec.attempt_timeout_ms[i],
            self._spec.max_attempts[j],
            self._spec.backoff_ms[k],
            global_timeout,
        )

    def nearest_indices(
        self,
        attempt_timeout_ms: int,
        max_attempts: int,
        backoff_ms: int,
        global_timeout_ms: Optional[int] = None,
    ) -> Tuple[int, ...]:
        def nearest(value: int, candidates: List[int]) -> int:
            return int(min(range(len(candidates)), key=lambda i: abs(candidates[i] - value)))

        i = nearest(attempt_timeout_ms, self._spec.attempt_timeout_ms)
        j = nearest(max_attempts, self._spec.max_attempts)
        k = nearest(backoff_ms, self._spec.backoff_ms)
        if self._spec.global_timeout_ms is None:
            return (i, j, k)

        if global_timeout_ms is None:
            g = 0
        else:
            gt_candidates = [
                v if v is not None else 0 for v in self._spec.global_timeout_ms
            ]
            g = nearest(global_timeout_ms, gt_candidates)
        return (i, j, k, g)
