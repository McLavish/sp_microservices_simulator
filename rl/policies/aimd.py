from dataclasses import dataclass

import numpy as np

from rl.policies.action_space import ActionGrid


@dataclass
class AIMDPolicy:
    """Additive-increase / multiplicative-decrease control of the attempt timeout.

    Unhealthy windows (deadline rate or success rate past target) bump the
    timeout up one grid index (additive increase); healthy windows multiply the
    timeout value by ``timeout_decrease_factor`` and snap to the nearest grid
    index (multiplicative decrease), the same shape as TCP congestion control
    and Netflix's adaptive concurrency limits. The optional max-attempts /
    backoff adjustments remain additive in both directions.
    """

    grid: ActionGrid
    initial_timeout_ms: int | None = None
    initial_max_attempts: int | None = None
    initial_backoff_ms: int | None = None
    initial_global_timeout_ms: int | None = None
    initial_action: tuple[int, ...] | None = None
    timeout_increase_step: int = 1
    timeout_decrease_factor: float = 0.5
    max_attempts_step: int = 1
    backoff_step: int = 1
    adjust_max_attempts: bool = False
    adjust_backoff: bool = False
    ema_alpha: float = 0.2
    success_target: float = 0.98
    deadline_target: float = 0.01
    min_completed: int = 1
    name: str = "aimd"

    def __post_init__(self):
        if not 0.0 < self.timeout_decrease_factor < 1.0:
            raise ValueError("timeout_decrease_factor must be in (0, 1)")
        self._has_global = self.grid.spec.global_timeout_ms is not None
        self._ema_success = None
        self._ema_deadline = None
        self._init_state()

    def reset(self, seed: int | None = None) -> None:
        self._ema_success = None
        self._ema_deadline = None
        self._init_state()

    def _init_state(self) -> None:
        idx = self._init_indices()
        self._timeout_idx, self._max_attempts_idx, self._backoff_idx = idx[0], idx[1], idx[2]
        # Global timeout is not adapted by AIMD; it is held at its initial index so
        # the emitted action matches the (3- or 4-dim) MultiDiscrete action space.
        self._global_idx = idx[3] if len(idx) > 3 else None
        self._action = self._build_action()

    def act(self, obs: np.ndarray, info: dict) -> tuple[int, ...]:
        summary = info.get("summary")
        if summary and summary.get("completed", 0) >= self.min_completed:
            deadline_rate = summary.get("deadline_drop_rate", 0.0)
            success_rate = summary.get("success_rate", 0.0)
            self._ema_deadline = self._update_ema(self._ema_deadline, deadline_rate)
            self._ema_success = self._update_ema(self._ema_success, success_rate)

            should_increase = (
                self._ema_deadline > self.deadline_target or self._ema_success < self.success_target
            )

            if should_increase:
                self._timeout_idx = min(
                    self._timeout_idx + self.timeout_increase_step,
                    self._max_timeout_idx(),
                )
                if self.adjust_max_attempts:
                    self._max_attempts_idx = min(
                        self._max_attempts_idx + self.max_attempts_step,
                        self._max_attempts_max_idx(),
                    )
                if self.adjust_backoff:
                    self._backoff_idx = min(
                        self._backoff_idx + self.backoff_step,
                        self._backoff_max_idx(),
                    )
            else:
                self._timeout_idx = self._decreased_timeout_idx()
                if self.adjust_max_attempts:
                    self._max_attempts_idx = max(self._max_attempts_idx - self.max_attempts_step, 0)
                if self.adjust_backoff:
                    self._backoff_idx = max(self._backoff_idx - self.backoff_step, 0)

            self._action = self._build_action()
        return self._action

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return

    def _decreased_timeout_idx(self) -> int:
        """Multiplicative decrease in value space, snapped to the grid.

        Guarantees at least one index of progress when rounding would stall
        above the minimum.
        """
        candidates = self.grid.spec.attempt_timeout_ms
        target = candidates[self._timeout_idx] * self.timeout_decrease_factor
        idx = min(range(len(candidates)), key=lambda i: abs(candidates[i] - target))
        if idx >= self._timeout_idx and self._timeout_idx > 0:
            idx = self._timeout_idx - 1
        return idx

    def _update_ema(self, current: float | None, value: float) -> float:
        if current is None:
            return value
        return (1.0 - self.ema_alpha) * current + self.ema_alpha * value

    def _init_indices(self) -> tuple[int, ...]:
        expected = 4 if self._has_global else 3
        if self.initial_action is not None:
            idx = tuple(int(v) for v in self.initial_action)
            if len(idx) != expected:
                raise ValueError(f"AIMDPolicy initial_action must be {expected} indices")
            return idx

        timeout_ms = self.initial_timeout_ms
        if timeout_ms is None:
            timeout_ms = self.grid.spec.attempt_timeout_ms[0]
        max_attempts = self.initial_max_attempts
        if max_attempts is None:
            max_attempts = self.grid.spec.max_attempts[0]
        backoff_ms = self.initial_backoff_ms
        if backoff_ms is None:
            backoff_ms = self.grid.spec.backoff_ms[0]

        return self.grid.nearest_indices(
            timeout_ms, max_attempts, backoff_ms, self.initial_global_timeout_ms
        )

    def _build_action(self) -> tuple[int, ...]:
        if self._global_idx is not None:
            return (self._timeout_idx, self._max_attempts_idx, self._backoff_idx, self._global_idx)
        return (self._timeout_idx, self._max_attempts_idx, self._backoff_idx)

    def _max_timeout_idx(self) -> int:
        return len(self.grid.spec.attempt_timeout_ms) - 1

    def _max_attempts_max_idx(self) -> int:
        return len(self.grid.spec.max_attempts) - 1

    def _backoff_max_idx(self) -> int:
        return len(self.grid.spec.backoff_ms) - 1
