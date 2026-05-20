from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rl.env import RewardWeights
from rl.models.base import LatencyModel
from rl.models.lognormal import LogNormalModel
from rl.policies.action_space import ActionGrid, ActionValues


@dataclass
class LatencyBuffer:
    max_samples: int
    samples: list[float] = field(default_factory=list)
    _array_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    def add(self, values: list[float]) -> None:
        if not values:
            return
        self.samples.extend(values)
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples :]
        self._array_cache = None

    def ingest_info(self, info: dict, use_censored: bool) -> None:
        latencies = info.get("attempt_latencies_ms", [])
        if not latencies:
            return
        if use_censored:
            self.add([float(v) for v in latencies])
            return
        censored = info.get("attempt_censored", [])
        filtered = [
            float(v) for v, is_censored in zip(latencies, censored, strict=False) if not is_censored
        ]
        self.add(filtered)

    def data(self) -> np.ndarray:
        if self._array_cache is None:
            self._array_cache = (
                np.array([], dtype=np.float64)
                if not self.samples
                else np.asarray(self.samples, dtype=np.float64)
            )
        return self._array_cache


def _simulate_action(
    model: LatencyModel,
    action: ActionValues,
    weights: RewardWeights,
    rng: np.random.Generator,
    n_roots: int,
    use_censored: bool,
    queue_delay_ms: float,
    queue_delay_mode: str,
) -> tuple[float, list[float]]:
    successes = 0
    retries = 0
    latencies = []
    observed = []

    timeout = float(action.attempt_timeout_ms)
    backoff = float(action.backoff_ms)
    max_attempts = int(action.max_attempts)
    global_timeout = None if action.global_timeout_ms is None else float(action.global_timeout_ms)

    for _ in range(n_roots):
        elapsed = 0.0
        success = False
        attempts = 0
        for attempt in range(max_attempts):
            attempts += 1
            sample = float(model.sample(rng, 1)[0])
            if queue_delay_ms > 0:
                if queue_delay_mode == "exponential":
                    q_delay = float(rng.exponential(queue_delay_ms))
                else:
                    q_delay = queue_delay_ms
            else:
                q_delay = 0.0
            total_sample = sample + q_delay
            attempt_latency = min(total_sample, timeout)
            elapsed += attempt_latency

            timed_out = total_sample > timeout
            if timed_out:
                if use_censored:
                    observed.append(attempt_latency)
            else:
                observed.append(attempt_latency)
                success = True
                latencies.append(elapsed)
                break

            if attempt < max_attempts - 1:
                elapsed += backoff

            if global_timeout is not None and elapsed >= global_timeout:
                break

        retries += max(0, attempts - 1)
        if success:
            successes += 1

    success_rate = successes / n_roots if n_roots else 0.0
    deadline_rate = 1.0 - success_rate
    retries_per_root = retries / n_roots if n_roots else 0.0
    p95 = float(np.quantile(latencies, 0.95)) if latencies else 0.0

    reward = (
        success_rate * weights.success
        - deadline_rate * weights.deadline_drop
        - p95 * weights.p95_latency_ms
        - retries_per_root * weights.retries
    )
    return reward, observed


@dataclass
class ModelBasedPolicy:
    grid: ActionGrid
    model_factory: Callable[[], LatencyModel] = LogNormalModel
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    sample_size: int = 500
    buffer_size: int = 2000
    use_censored: bool = False
    horizon: int = 1
    gamma: float = 0.95
    include_queue_delay: bool = False
    queue_delay_mode: str = "deterministic"
    queue_workers: int = 16
    queue_max_utilization: float = 0.95
    queue_moment_samples: int = 500
    use_retries_in_lambda: bool = True
    step_duration_s: float = 1.0
    name: str = "model_based"

    def __post_init__(self):
        self._buffer = LatencyBuffer(self.buffer_size)
        self._rng = np.random.default_rng(123)

    def reset(self, seed: int | None = None) -> None:
        self._buffer = LatencyBuffer(self.buffer_size)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict) -> tuple[int, ...]:
        self._buffer.ingest_info(info, self.use_censored)
        data = self._buffer.data()
        if data.size == 0:
            return self.grid.indices[0]

        base_model = self.model_factory().fit(data)
        queue_delay = self._estimate_queue_delay(info, base_model)
        best_idx = self.grid.indices[0]
        best_value = -float("inf")

        for action_idx, action_vals in zip(self.grid.indices, self.grid.values, strict=False):
            value = self._evaluate_action(base_model, action_vals, queue_delay)
            if value > best_value:
                best_value = value
                best_idx = action_idx

        return best_idx

    def _evaluate_action(
        self, model: LatencyModel, action_vals: ActionValues, queue_delay: float
    ) -> float:
        if self.horizon <= 1:
            reward, _ = _simulate_action(
                model,
                action_vals,
                self.reward_weights,
                self._rng,
                self.sample_size,
                self.use_censored,
                queue_delay,
                self.queue_delay_mode,
            )
            return reward

        total = 0.0
        temp_model = copy.deepcopy(model)
        buffer = LatencyBuffer(self.buffer_size)
        buffer.add(list(self._buffer.samples))

        for h in range(self.horizon):
            reward, observed = _simulate_action(
                temp_model,
                action_vals,
                self.reward_weights,
                self._rng,
                self.sample_size,
                self.use_censored,
                queue_delay,
                self.queue_delay_mode,
            )
            total += (self.gamma**h) * reward
            if observed:
                buffer.add(observed)
                temp_model = self.model_factory().fit(buffer.data())

        return total

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return

    def _estimate_queue_delay(self, info: dict, model: LatencyModel) -> float:
        if not self.include_queue_delay:
            return 0.0
        summary = info.get("summary")
        if not summary:
            return 0.0
        completed = float(summary.get("completed", 0.0))
        if completed <= 0 or self.step_duration_s <= 0:
            return 0.0

        lambda_rate = completed / float(self.step_duration_s)
        if self.use_retries_in_lambda:
            lambda_rate *= 1.0 + float(summary.get("retries_per_root", 0.0))

        moments_fn = getattr(model, "moments", None)
        if callable(moments_fn):
            mean_s, mean_s2 = moments_fn()
            mean_s = float(mean_s)
            mean_s2 = float(mean_s2)
        else:
            sample_count = max(10, int(self.queue_moment_samples))
            samples = model.sample(self._rng, sample_count)
            mean_s = float(np.mean(samples))
            mean_s2 = float(np.dot(samples, samples) / samples.size)
        workers = max(1, int(self.queue_workers))
        rho = (lambda_rate * mean_s) / workers if mean_s > 0 else 0.0
        rho = min(max(rho, 0.0), self.queue_max_utilization)
        if rho >= 1.0:
            rho = self.queue_max_utilization

        mean_s2_eff = mean_s2 / (workers**2)
        denom = max(1e-6, 2.0 * (1.0 - rho))
        wq = lambda_rate * mean_s2_eff / denom
        return max(0.0, float(wq))
