from collections.abc import Sequence

from rl.spec import ActionSpec, ActionValues, coerce_action


class _Unspecified:
    """Sentinel: the caller expressed no preference for a knob.

    Distinct from `None`, which for `global_timeout_ms` means "explicitly no
    global timeout" and selects the spec's None entry.
    """

    __slots__ = ()


UNSPECIFIED = _Unspecified()


__all__ = ["ActionGrid", "ActionValues", "coerce_action"]


class ActionGrid:
    def __init__(self, spec: ActionSpec):
        self._spec = spec
        self._indices: list[tuple[int, ...]] = []
        self._values: list[ActionValues] = []
        self._build()

    @property
    def spec(self) -> ActionSpec:
        return self._spec

    @property
    def indices(self) -> list[tuple[int, ...]]:
        return self._indices

    @property
    def values(self) -> list[ActionValues]:
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
                            values = ActionValues(timeout, max_attempts, backoff, global_timeout)
                            self._indices.append(indices)
                            self._values.append(values)

    def midpoint_indices(self) -> tuple[int, ...]:
        """Mid-grid action: the neutral choice when no data supports one yet.

        Mirrors `MicroservicesEnv`'s warm-up action (`rl/env.py`): index 0 on
        every knob is the grid CORNER — shortest timeout, fewest attempts, no
        backoff — which measured ~11% success, so a policy that has nothing to
        fit must not emit it.
        """
        sizes = [
            len(self._spec.attempt_timeout_ms),
            len(self._spec.max_attempts),
            len(self._spec.backoff_ms),
        ]
        if self._spec.global_timeout_ms is not None:
            sizes.append(len(self._spec.global_timeout_ms))
        return tuple(n // 2 for n in sizes)

    def values_for(self, indices: Sequence[int]) -> ActionValues:
        idx = tuple(int(i) for i in indices)
        if self._spec.global_timeout_ms is None and len(idx) != 3:
            raise ValueError("Expected 3 indices for action")
        if self._spec.global_timeout_ms is not None and len(idx) != 4:
            raise ValueError("Expected 4 indices for action")
        i, j, k = idx[:3]
        g_idx = idx[3] if len(idx) > 3 else None
        global_timeout_list = self._spec.global_timeout_ms
        global_timeout = (
            None if g_idx is None or global_timeout_list is None else global_timeout_list[g_idx]
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
        global_timeout_ms: int | None | _Unspecified = UNSPECIFIED,
    ) -> tuple[int, ...]:
        """Map knob values onto grid indices.

        `global_timeout_ms` distinguishes three cases: omitted (the caller has no
        opinion — pick the spec's own default), explicitly `None` (select the
        spec's no-global-timeout entry, which must exist), or a number (nearest
        numeric entry). Conflating the first two made every policy that defaults
        the argument — FixedValuePolicy, AIMDPolicy — raise on any spec whose
        global-timeout grid is entirely numeric.
        """

        def nearest(value: int, candidates: list[int]) -> int:
            return int(min(range(len(candidates)), key=lambda i: abs(candidates[i] - value)))

        i = nearest(attempt_timeout_ms, self._spec.attempt_timeout_ms)
        j = nearest(max_attempts, self._spec.max_attempts)
        k = nearest(backoff_ms, self._spec.backoff_ms)
        if self._spec.global_timeout_ms is None:
            return (i, j, k)

        if isinstance(global_timeout_ms, _Unspecified):
            # No opinion: prefer the spec's None entry if it has one, otherwise
            # fall back to its first (numeric) entry rather than failing.
            g = (
                self._spec.global_timeout_ms.index(None)
                if None in self._spec.global_timeout_ms
                else 0
            )
        elif global_timeout_ms is None:
            if None not in self._spec.global_timeout_ms:
                raise ValueError(
                    "requested global_timeout_ms=None but the ActionSpec has no None entry"
                )
            g = self._spec.global_timeout_ms.index(None)
        else:
            numeric = [
                (idx, v) for idx, v in enumerate(self._spec.global_timeout_ms) if v is not None
            ]
            if not numeric:
                raise ValueError(
                    f"requested global_timeout_ms={global_timeout_ms} but the "
                    "ActionSpec contains only None"
                )
            g = min(numeric, key=lambda iv: abs(iv[1] - global_timeout_ms))[0]
        return (i, j, k, g)
