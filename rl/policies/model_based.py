from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rl.models.base import LatencyModel
from rl.models.lognormal import LogNormalModel
from rl.policies.action_space import ActionGrid, ActionValues
from rl.spec import RewardWeights


@dataclass
class LatencyBuffer:
    """Sliding window of attempt latencies with right-censoring flags.

    Censored entries are timeout-truncated observations: the recorded value is
    the timeout, a lower bound on the true latency. Keeping them (with flags)
    lets models do right-censored MLE instead of silently dropping the slow
    tail, which would bias the fitted distribution optimistic.
    """

    max_samples: int
    samples: list[float] = field(default_factory=list)
    censored: list[bool] = field(default_factory=list)
    _cache: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False)

    def add(self, values: list[float], censored: list[bool] | None = None) -> None:
        if not values:
            return
        flags = censored if censored is not None else [False] * len(values)
        if len(flags) != len(values):
            raise ValueError("censored flags must match values length")
        self.samples.extend(values)
        self.censored.extend(flags)
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples :]
            self.censored = self.censored[-self.max_samples :]
        self._cache = None

    def ingest_info(self, info: dict) -> None:
        latencies = info.get("attempt_latencies_ms", [])
        if not latencies:
            return
        flags = [bool(f) for f in info.get("attempt_censored", [])]
        if len(flags) != len(latencies):
            flags = [False] * len(latencies)
        self.add([float(v) for v in latencies], flags)

    def censored_data(self) -> tuple[np.ndarray, np.ndarray]:
        """All samples plus their right-censoring flags."""
        if self._cache is None:
            self._cache = (
                np.asarray(self.samples, dtype=np.float64),
                np.asarray(self.censored, dtype=bool),
            )
        return self._cache

    def data(self) -> np.ndarray:
        """Uncensored samples only."""
        values, flags = self.censored_data()
        return values[~flags]


def _simulate_action(
    model: LatencyModel,
    action: ActionValues,
    weights: RewardWeights,
    rng: np.random.Generator,
    n_roots: int,
    queue_delay_ms: float,
    queue_delay_mode: str,
) -> float:
    """Monte-Carlo reward estimate for one knob setting under the latency model.

    The surrogate only models attempt latency vs. timeout: queue-full drops and
    server failures in the real simulator are outside its vocabulary, so its
    failure mode is always "deadline".
    """
    successes = 0
    retries = 0
    latencies = []

    timeout = float(action.attempt_timeout_ms)
    backoff = float(action.backoff_ms)
    max_attempts = int(action.max_attempts)
    global_timeout = None if action.global_timeout_ms is None else float(action.global_timeout_ms)

    for _ in range(n_roots):
        elapsed = 0.0
        success = False
        attempts = 0
        for attempt in range(max_attempts):
            remaining = None if global_timeout is None else global_timeout - elapsed
            if remaining is not None and remaining <= 0:
                break
            # Mirror the real caller: per-attempt deadline is
            # min(attempt timeout, remaining global budget), and success needs
            # strictly-before completion (half-open deadline, caller.py).
            effective_timeout = timeout if remaining is None else min(timeout, remaining)
            attempts += 1
            sample = float(model.sample(rng, 1)[0])
            q_delay = 0.0
            if queue_delay_ms > 0:
                if queue_delay_mode == "exponential":
                    q_delay = float(rng.exponential(queue_delay_ms))
                else:
                    q_delay = queue_delay_ms
            total_sample = sample + q_delay
            attempt_latency = min(total_sample, effective_timeout)
            elapsed += attempt_latency

            if total_sample < effective_timeout:
                success = True
                latencies.append(elapsed)
                break

            if attempt < max_attempts - 1:
                elapsed += backoff

        retries += max(0, attempts - 1)
        if success:
            successes += 1

    success_rate = successes / n_roots if n_roots else 0.0
    deadline_rate = 1.0 - success_rate
    retries_per_root = retries / n_roots if n_roots else 0.0
    p95 = float(np.quantile(latencies, 0.95)) if latencies else 0.0

    return (
        success_rate * weights.success
        - deadline_rate * weights.deadline_drop
        - p95 * weights.p95_latency_ms
        - retries_per_root * weights.retries
    )


@dataclass
class ModelBasedPolicy:
    grid: ActionGrid
    model_factory: Callable[[], LatencyModel] = LogNormalModel
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    sample_size: int = 500
    buffer_size: int = 2000
    include_queue_delay: bool = False
    queue_delay_mode: str = "deterministic"
    queue_workers: int = 16
    queue_max_utilization: float = 0.95
    step_duration_s: float = 1.0
    # Minimum uncensored samples before the censored MLE is trusted.
    min_uncensored: int = 30
    name: str = "model_based"

    def __post_init__(self):
        self._buffer = LatencyBuffer(self.buffer_size)
        self._service_buffer = LatencyBuffer(self.buffer_size)
        self._rng = np.random.default_rng(123)
        self._min_uncensored = int(self.min_uncensored)
        self._last_model: LatencyModel | None = None

    def reset(self, seed: int | None = None) -> None:
        self._buffer = LatencyBuffer(self.buffer_size)
        self._service_buffer = LatencyBuffer(self.buffer_size)
        self._last_model = None
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _fit_buffer(self) -> LatencyBuffer:
        """Which stream the latency model is fitted on.

        The queue-aware variant decomposes attempt latency into service time
        plus Erlang-C queueing delay, so it must fit the queue-exclusive
        SERVICE stream — fitting queue-inclusive totals and then adding the
        queueing estimate again would double-count queueing. The plain variant
        models end-to-end totals directly.
        """
        return self._service_buffer if self.include_queue_delay else self._buffer

    def act(self, obs: np.ndarray, info: dict) -> tuple[int, ...]:
        self._buffer.ingest_info(info)
        service = [float(v) for v in info.get("attempt_service_ms", [])]
        if service:
            flags = [bool(f) for f in info.get("attempt_service_censored", [])]
            if len(flags) != len(service):
                flags = [False] * len(service)
            self._service_buffer.add(service, flags)
        values, censored = self._fit_buffer().censored_data()
        uncensored = values[~censored]
        uncensored = uncensored[uncensored > 0]
        if uncensored.size == 0:
            return self.grid.indices[0]

        # Identifiability guard: with almost every sample censored, the censored
        # MLE is unidentified and swings wildly (observed median 877ms -> 93ms ->
        # 5308ms across three consecutive 1s windows, with sigma collapsing to
        # 0.07). Under such a fit every action scores success-rate 0, the argmax
        # falls to the grid corner - shortest timeout, one attempt - and that
        # corner guarantees the next window is censored too, so the policy locks
        # itself there mid-fault. Hold the previous fit instead.
        censored_fraction = float(np.mean(censored)) if censored.size else 0.0
        identifiable = uncensored.size >= self._min_uncensored and censored_fraction < 0.9
        if not identifiable and self._last_model is not None:
            base_model = self._last_model
            queue_delay = self._estimate_queue_delay(info)
            return self._argmax_action(base_model, queue_delay)

        base_model = self.model_factory()
        fit_censored = getattr(base_model, "fit_censored", None)
        if callable(fit_censored) and bool(np.any(censored)):
            base_model = fit_censored(values, censored)
        else:
            # Model without censored-MLE support: fall back to uncensored
            # samples only (biased optimistic when timeouts are frequent).
            base_model = base_model.fit(uncensored)
        self._last_model = base_model
        queue_delay = self._estimate_queue_delay(info)
        return self._argmax_action(base_model, queue_delay)

    def _argmax_action(self, model: LatencyModel, queue_delay: float) -> tuple[int, ...]:
        """Score every candidate against the SAME random stream.

        Without common random numbers each action carries independent sampling
        error (sd ~0.004 at the default sample_size) that swamps the true spread
        among the top candidates (~0.002), so the argmax - and the knob traces
        the experiments export - is resampled noise rather than a decision. It
        also removes the winner's-curse bias in taking a max over noisy estimates.
        """
        state = self._rng.bit_generator.state
        best_idx = self.grid.indices[0]
        best_value = -float("inf")

        for action_idx, action_vals in zip(self.grid.indices, self.grid.values, strict=False):
            self._rng.bit_generator.state = state
            value = _simulate_action(
                model,
                action_vals,
                self.reward_weights,
                self._rng,
                self.sample_size,
                queue_delay,
                self.queue_delay_mode,
            )
            if value > best_value:
                best_value = value
                best_idx = action_idx

        # Advance the stream once so consecutive steps do not reuse the draws.
        self._rng.bit_generator.state = state
        self._rng.random()
        return best_idx

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        return

    def _service_moments(self) -> tuple[float, float, int]:
        """Mean / mean-square / count of buffered service-occupancy times.

        Censored (deadline-cut) samples are included at their observed value:
        occupancy is occupancy — utilization is driven by the time workers were
        actually held, not by the latent demand.
        """
        service, _ = self._service_buffer.censored_data()
        if service.size == 0:
            return 0.0, 0.0, 0
        return (
            float(service.mean()),
            float(np.dot(service, service) / service.size),
            int(service.size),
        )

    def _estimate_queue_delay(self, info: dict) -> float:
        """Expected queueing delay in ms via the Allen-Cunneen M/G/c approximation:

        Wq(M/G/c) ~= Wq(M/M/c) * (1 + CV_s^2) / 2,  Wq(M/M/c) from Erlang C.

        The moments are in milliseconds, so the arrival rate is per millisecond.
        """
        if not self.include_queue_delay:
            return 0.0
        # Lambda = arrivals at the MODELED queue. Prefer the env's own count of
        # attempts that entered service there: it covers EVERY client feeding
        # that queue. The sampled client's attempt count is a fallback and
        # undercounts whenever the service is shared (in multichain, svc-C is fed
        # by both the agent's edge and the bystander's, so it halves rho).
        attempt_count = info.get("modeled_admitted")
        if attempt_count is None:
            attempt_count = len(info.get("attempt_latencies_ms") or [])
        attempt_count = int(attempt_count)
        if attempt_count <= 0 or self.step_duration_s <= 0:
            return 0.0
        lambda_per_ms = attempt_count / (float(self.step_duration_s) * 1000.0)

        mean_s, mean_s2, count = self._service_moments()
        if count < 10 or mean_s <= 0:
            return 0.0  # cold start: no reliable occupancy stream yet

        # `c` from the modelled service itself; the CLI default is scenario-
        # independent and was simply wrong on the chain scenarios (svc-C has 24
        # workers in build_chain_env and 14 in build_multichain_env, against a
        # default of 16).
        workers = max(1, int(info.get("modeled_workers") or self.queue_workers))
        rho = min((lambda_per_ms * mean_s) / workers, self.queue_max_utilization)
        offered = rho * workers  # erlangs, consistent with the clamped utilization

        p_wait = _erlang_c(workers, offered)
        wq_mmc = p_wait * mean_s / (workers * (1.0 - rho))
        var_s = max(0.0, mean_s2 - mean_s * mean_s)
        cv2 = var_s / (mean_s * mean_s)
        return max(0.0, float(wq_mmc * (1.0 + cv2) / 2.0))


def _erlang_c(c: int, offered: float) -> float:
    """Erlang C waiting probability for an M/M/c queue with offered load in erlangs."""
    if offered <= 0:
        return 0.0
    rho = offered / c
    if rho >= 1.0:
        return 1.0
    # Iterative Erlang B recurrence, then the standard B -> C conversion.
    b = 1.0
    for k in range(1, c + 1):
        b = offered * b / (k + offered * b)
    return b / (1.0 - rho + rho * b)
