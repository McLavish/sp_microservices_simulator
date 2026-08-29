from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from msim.core import TimeDuration


class TimeoutController(ABC):
    @abstractmethod
    def get_global_timeout(self) -> TimeDuration | None:
        pass

    @abstractmethod
    def get_attempt_timeout(self) -> TimeDuration | None:
        pass

    @abstractmethod
    def record_result(self, success: bool, latency: TimeDuration) -> None:
        pass


@dataclass
class StaticTimeoutController(TimeoutController):
    """Static timeouts"""

    global_timeout: TimeDuration | None = None
    attempt_timeout: TimeDuration | None = None

    def get_global_timeout(self) -> TimeDuration | None:
        return self.global_timeout

    def get_attempt_timeout(self) -> TimeDuration | None:
        return self.attempt_timeout

    def record_result(self, success: bool, latency: TimeDuration) -> None:
        pass


@dataclass
class AdaptiveTimeoutController(TimeoutController):
    """Adaptive timeout controller that adjusts based on latency percentiles and success rates"""

    # Configuration
    initial_global_timeout: TimeDuration | None = None
    initial_attempt_timeout: TimeDuration | None = None

    # Adaptation parameters
    target_percentile: float = 0.95  # Use p95 latency for timeout calculation
    safety_margin: float = 2.0  # Multiply percentile by this factor
    min_samples: int = 10  # Minimum samples before adapting
    window_size: int = 100  # Number of recent results to consider
    min_timeout: TimeDuration = 1_000_000  # 1ms minimum
    max_timeout: TimeDuration | None = None  # No maximum by default

    # Success rate tracking
    success_rate_window: int = 50
    min_success_rate: float = 0.5  # If success rate drops below this, increase timeout

    # Internal state
    _latencies: deque[TimeDuration] = field(default_factory=deque, init=False)
    _successes: deque[bool] = field(default_factory=deque, init=False)
    # Global timeout is fixed by design: it bounds the entire request
    # budget for the caller's deadline contract. Only attempt timeout
    # adapts to observed latency.
    _current_global_timeout: TimeDuration | None = field(default=None, init=False)
    _current_attempt_timeout: TimeDuration | None = field(default=None, init=False)

    def __post_init__(self):
        # Reject a self-contradictory configuration instead of silently serving
        # or silently clamping it. `initial_attempt_timeout` is handed out
        # verbatim until enough SUCCESSFUL samples accumulate, so an initial
        # outside the bounds escapes them for that whole window — while every
        # adaptation path applies them. Clamping instead would override an
        # operator's explicit setting, so this validates at construction, the
        # way resilience4j checks its timeout config.
        if self.min_timeout < 0:
            raise ValueError(f"min_timeout must be >= 0 ns, got {self.min_timeout}")
        if self.max_timeout is not None and self.max_timeout < self.min_timeout:
            raise ValueError(
                f"max_timeout ({self.max_timeout} ns) must be >= "
                f"min_timeout ({self.min_timeout} ns)"
            )
        if self.initial_attempt_timeout is not None:
            above_max = (
                self.max_timeout is not None and self.initial_attempt_timeout > self.max_timeout
            )
            if self.initial_attempt_timeout < self.min_timeout or above_max:
                raise ValueError(
                    f"initial_attempt_timeout ({self.initial_attempt_timeout} ns) must lie within "
                    f"[{self.min_timeout}, {self.max_timeout}] ns"
                )
        self._latencies = deque(maxlen=self.window_size)
        self._successes = deque(maxlen=self.success_rate_window)
        self._current_global_timeout = self.initial_global_timeout
        self._current_attempt_timeout = self.initial_attempt_timeout

    def _calculate_percentile_timeout(self) -> TimeDuration | None:
        """Calculate timeout based on the target latency percentile (numpy)."""
        if len(self._latencies) < self.min_samples:
            return None

        percentile_latency = float(
            np.quantile(np.fromiter(self._latencies, dtype=np.float64), self.target_percentile)
        )
        adapted_timeout = int(percentile_latency * self.safety_margin)

        # Apply min/max constraints
        adapted_timeout = max(adapted_timeout, self.min_timeout)
        if self.max_timeout is not None:
            adapted_timeout = min(adapted_timeout, self.max_timeout)

        return adapted_timeout

    def _get_success_rate(self) -> float:
        if not self._successes:
            return 1.0
        return sum(self._successes) / len(self._successes)

    def _update_timeouts(self) -> None:
        """Update current timeout values based on collected metrics"""
        percentile_timeout = self._calculate_percentile_timeout()
        success_rate = self._get_success_rate()

        if percentile_timeout is None:
            # Not enough SUCCESSFUL samples to estimate a percentile. That is
            # exactly the state a too-short timeout produces, and the old code
            # returned here — so the timeout froze at its initial value and the
            # controller could never escape. Inflate off the current value
            # instead, which is the one signal available.
            if (
                success_rate < self.min_success_rate
                and self._successes
                and self.initial_attempt_timeout is not None
            ):
                # Inflate off the INITIAL timeout, never off the already-inflated
                # current value: compounding turned 10 ms into 100 s in four
                # windows, and max_timeout is unbounded by default. This is a
                # bounded cold-start nudge (at most 10x, the success_rate floor),
                # not a control loop.
                inflated = int(self.initial_attempt_timeout / max(success_rate, 0.1))
                if self.max_timeout is not None:
                    inflated = min(inflated, self.max_timeout)
                self._current_attempt_timeout = max(inflated, self.min_timeout)
            return

        if percentile_timeout is not None:
            # If success rate is low, increase timeout
            if success_rate < self.min_success_rate:
                success_adjustment = 1.0 / max(success_rate, 0.1)
                adjusted_timeout = int(percentile_timeout * success_adjustment)
                # max_timeout is a hard cap; inflation must not escape it
                if self.max_timeout is not None:
                    adjusted_timeout = min(adjusted_timeout, self.max_timeout)
            else:
                adjusted_timeout = percentile_timeout

            # Update attempt timeout
            self._current_attempt_timeout = adjusted_timeout

    def get_global_timeout(self) -> TimeDuration | None:
        return self._current_global_timeout

    def get_attempt_timeout(self) -> TimeDuration | None:
        return self._current_attempt_timeout

    def record_result(self, success: bool, latency: TimeDuration) -> None:
        """Record result and update timeout values.

        Only successful latencies feed the percentile window: drops at
        submit/dequeue carry near-zero latency and would bias the
        percentile downward, shrinking the timeout and amplifying drops.
        The success-rate window still records every outcome so the
        low-success-rate inflation path remains active.
        """
        if success:
            self._latencies.append(latency)
        self._successes.append(success)
        self._update_timeouts()
