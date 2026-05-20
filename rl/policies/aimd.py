from dataclasses import dataclass

import numpy as np

from rl.policies.action_space import ActionGrid


@dataclass
class AIMDPolicy:
    grid: ActionGrid
    initial_timeout_ms: int | None = None
    initial_max_attempts: int | None = None
    initial_backoff_ms: int | None = None
    initial_action: tuple[int, ...] | None = None
    timeout_increase_step: int = 1
    timeout_decrease_step: int = 1
    max_attempts_step: int = 1
    backoff_step: int = 1
    adjust_max_attempts: bool = False
    adjust_backoff: bool = False
    ema_alpha: float = 0.2
    success_target: float = 0.98
    deadline_target: float = 0.01
    min_completed: int = 1
    name: str = "aimd_v2"

    def __post_init__(self):
        self._ema_success = None
        self._ema_deadline = None
        self._timeout_idx, self._max_attempts_idx, self._backoff_idx = self._init_indices()
        self._action = self._build_action()

    def reset(self, seed: int | None = None) -> None:
        self._ema_success = None
        self._ema_deadline = None
        self._timeout_idx, self._max_attempts_idx, self._backoff_idx = self._init_indices()
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
                self._timeout_idx = max(self._timeout_idx - self.timeout_decrease_step, 0)
                if self.adjust_max_attempts:
                    self._max_attempts_idx = max(self._max_attempts_idx - self.max_attempts_step, 0)
                if self.adjust_backoff:
                    self._backoff_idx = max(self._backoff_idx - self.backoff_step, 0)

            self._action = self._build_action()
        return self._action

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return

    def _update_ema(self, current: float | None, value: float) -> float:
        if current is None:
            return value
        return (1.0 - self.ema_alpha) * current + self.ema_alpha * value

    def _init_indices(self) -> tuple[int, int, int]:
        if self.initial_action is not None:
            idx = tuple(int(v) for v in self.initial_action)
            if len(idx) != 3:
                raise ValueError("AIMDPolicy initial_action must be 3 indices")
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

        idx = self.grid.nearest_indices(timeout_ms, max_attempts, backoff_ms)
        return idx[0], idx[1], idx[2]

    def _build_action(self) -> tuple[int, int, int]:
        return (self._timeout_idx, self._max_attempts_idx, self._backoff_idx)

    def _max_timeout_idx(self) -> int:
        return len(self.grid.spec.attempt_timeout_ms) - 1

    def _max_attempts_max_idx(self) -> int:
        return len(self.grid.spec.max_attempts) - 1

    def _backoff_max_idx(self) -> int:
        return len(self.grid.spec.backoff_ms) - 1
