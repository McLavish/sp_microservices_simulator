from __future__ import annotations

import functools
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from caller import ClientConfig, ClientRuntime
from chain import DownstreamEdge, ServiceNodeRuntime
from core import (
    DropReason,
    RootOutcome,
    Summary,
    TimeDuration,
    TimeInterval,
    TimePoint,
    classify_root,
    ms_to_ns,
    ns_to_ms,
    ns_to_s,
    root_latency,
    s_to_ns,
)
from faults import LatencyInjection, LoadSpike, PartialFailure
from metrics import Metrics
from retry_policies import FixedBackoffRetryPolicy, RetryBudgetPolicy, RetryPolicy
from rl.spec import ActionSpec, ActionValues, RewardWeights, coerce_action
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload

if TYPE_CHECKING:
    import gymnasium as gym
    from gymnasium import spaces

    _GymEnvBase = gym.Env
    _GYM_IMPORT_ERROR: Exception | None = None
else:
    try:
        import gymnasium as gym
        from gymnasium import spaces

        _GymEnvBase = gym.Env
        _GYM_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - exercised in runtime environments
        gym = None
        spaces = None
        _GymEnvBase = object
        _GYM_IMPORT_ERROR = exc


# Domain separation for fault-parameter draws, same rationale as
# workload._ARRIVAL_SEED_OFFSET: the builders seed their fault RNG with the same
# integer they pass to `Simulator(seed=...)`, so without an offset both would
# draw from the identical Mersenne-Twister stream and fault difficulty would
# covary with the service times and failure rolls it is supposed to be
# independent of. Distinct from the arrival offset so faults, arrivals and
# service draws are three separate streams.
_FAULT_SEED_OFFSET = 700_001


@dataclass
class EnvComponents:
    sim: Simulator
    workload: Workload
    client: ClientRuntime
    # Maps decoded knob values onto this scenario's policy/timeout objects.
    # The builder owns the knowledge of which concrete types it constructed,
    # so the env stays agnostic of retry-policy and controller classes.
    apply_knobs: Callable[[ActionValues], None]
    # Where the constraint cost (retry amplification) is measured. In the chain
    # scenario this is the tunable B->C edge client, whose amplification is
    # invisible in the top-level client's summary. None -> cost is 0.
    cost_client: ClientRuntime | None = None
    # Top-level client of a fixed-policy bystander chain (multichain scenario).
    # Its success rate gauges the externality of the agent's edge retries; it
    # feeds reporting only, never reward or observation.
    bystander_client: ClientRuntime | None = None
    # The service whose queue a queueing-model policy should reason about: the
    # one the knobs actuate against (svc-C in the chain scenarios). Carries both
    # the worker count and the true admitted-attempt stream, so a policy never
    # has to guess `c` from a CLI default or infer lambda from one of several
    # clients feeding the same queue.
    modeled_service: ServiceRuntime | None = None
    # Any additional arrival processes the builder drives beyond `workload`.
    # The env halts EVERY registered workload at the episode horizon, so the
    # terminal drain completes in-flight work only. A builder that drives a
    # second Workload without registering it here would keep admitting brand-new
    # post-horizon roots into the drained population.
    extra_workloads: list[Workload] = field(default_factory=list)


@dataclass
class EnvConfig:
    build: Callable[[int | None], EnvComponents]
    action_spec: ActionSpec
    step_duration_s: float = 1.0
    episode_duration_s: float | None = None
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    drain_on_done: bool = True
    initial_action: tuple[int, ...] | None = None
    emit_latency_samples: bool = True
    max_latency_samples: int | None = None
    # Fold drained-root outcomes into the final step's reward. Default off:
    # the windowed objective is the documented design and keeps trained
    # behavior comparable with the pinned paper data; turning it on removes
    # the bulk of the defer-failures-past-the-horizon incentive for future
    # studies. A small boundary residual (~ one drained-tail window weight)
    # remains because in-horizon windows are rate-scored while the drain is
    # count-weighted — see docs/rl.md.
    terminal_drain_penalty: bool = False


class MicroservicesEnv(_GymEnvBase):
    """Gymnasium environment wrapper for the microservices simulator."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for MicroservicesEnv") from _GYM_IMPORT_ERROR
        assert spaces is not None  # narrow for type checker after the check above
        if config.step_duration_s <= 0:
            raise ValueError("step_duration_s must be > 0")
        if s_to_ns(config.step_duration_s) <= 0:
            # A sub-nanosecond step rounds to 0 ns, so step() never advances the
            # clock, the episode never terminates and the caller hangs forever.
            raise ValueError(
                f"step_duration_s={config.step_duration_s!r} rounds to 0 ns; the "
                "minimum representable step is 1e-9 s"
            )
        if config.episode_duration_s is not None and config.episode_duration_s <= 0:
            raise ValueError("episode_duration_s must be > 0")
        self._cfg = config

        self._action_sizes = [
            len(config.action_spec.attempt_timeout_ms),
            len(config.action_spec.max_attempts),
            len(config.action_spec.backoff_ms),
        ]
        if config.action_spec.global_timeout_ms is not None:
            self._action_sizes.append(len(config.action_spec.global_timeout_ms))

        self.action_space = spaces.MultiDiscrete(self._action_sizes)

        # See _build_observation for the feature list; the optional global-timeout
        # dimension adds one action-echo feature.
        self._obs_size = 16 + (1 if config.action_spec.global_timeout_ms is not None else 0)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(self._obs_size,), dtype=np.float32
        )
        # Normalization constants so an MLP sees features of comparable magnitude.
        self._timeout_norm_ms = float(max(config.action_spec.attempt_timeout_ms))
        self._retries_norm = float(max(1, max(config.action_spec.max_attempts) - 1))

        self._sim: Simulator | None = None
        self._workload: Workload | None = None
        self._client: ClientRuntime | None = None
        self._components: EnvComponents | None = None
        self._last_root_idx = 0
        self._last_cost_root_idx = 0
        self._last_sample_root_idx = 0
        self._horizon_root_count = 0
        self._bystander_horizon_count = 0
        self._last_admitted = 0
        self._cost_roots_seen = 0.0
        self._cost_windows = 0
        self._episode_steps = 0
        self._current_time_ns: TimePoint = 0
        self._end_time_ns: TimePoint = 0
        self._step_ns: TimeDuration = 0
        self._last_action_values: dict[str, int | None] = {}
        self._last_action_indices: tuple[int, ...] = tuple(0 for _ in self._action_sizes)

    def _require_sim(self) -> Simulator:
        if self._sim is None:
            raise RuntimeError("Simulator not initialized; call reset() first.")
        return self._sim

    def _require_client(self) -> ClientRuntime:
        if self._client is None:
            raise RuntimeError("Client not initialized; call reset() first.")
        return self._client

    def _require_workload(self) -> Workload:
        if self._workload is None:
            raise RuntimeError("Workload not initialized; call reset() first.")
        return self._workload

    def _require_components(self) -> EnvComponents:
        if self._components is None:
            raise RuntimeError("Components not initialized; call reset() first.")
        return self._components

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Unseeded resets must still vary across episodes. The builders fall back
        # to fixed seeds on None, and SB3 vector envs only seed the *first* reset
        # (auto-resets are unseeded) — without this draw, training would collapse
        # to one identical deterministic episode.
        build_seed = seed if seed is not None else int(self.np_random.integers(2**31 - 1))
        components = self._build(build_seed)
        self._components = components
        self._sim = components.sim
        self._workload = components.workload
        self._client = components.client

        self._current_time_ns = 0
        self._step_ns = s_to_ns(self._cfg.step_duration_s)
        episode_s = (
            self._cfg.episode_duration_s
            if self._cfg.episode_duration_s is not None
            else self._require_workload().duration_s
        )
        self._end_time_ns = s_to_ns(episode_s)

        self._last_root_idx = 0
        self._last_cost_root_idx = 0
        self._last_sample_root_idx = 0
        self._horizon_root_count = 0
        self._bystander_horizon_count = 0
        self._last_admitted = 0
        self._cost_roots_seen = 0.0
        self._cost_windows = 0
        self._episode_steps = 0
        self._require_workload().drive(
            self._require_sim(), lambda s: self._require_client().start_request(s)
        )

        # Window 0 runs before any feedback exists, so its action is pure
        # warmup cost paid by every policy. Index 0 is the grid corner - the
        # shortest timeout with no retries - which made that first window
        # pathological (measured ~11% success) and poisoned any controller that
        # seeds its state from it. Start mid-grid instead.
        initial_action = self._cfg.initial_action or tuple(n // 2 for n in self._action_sizes)
        self._apply_action(initial_action)

        obs = self._build_observation(self._summarize_roots([]))
        info: dict[str, object] = {"action": self._last_action_values.copy()}
        if self._cfg.emit_latency_samples:
            info.update(
                {
                    "attempt_latencies_ms": [],
                    "attempt_censored": [],
                    "attempt_service_ms": [],
                    "attempt_service_censored": [],
                    "root_latencies_ms": [],
                }
            )
        return obs, info

    def step(self, action):
        action = coerce_action(action)
        self._apply_action(action)

        next_time = min(self._current_time_ns + self._step_ns, self._end_time_ns)
        self._require_sim().run(until=next_time)
        self._current_time_ns = next_time

        self._episode_steps += 1
        new_roots = self._require_client().roots[self._last_root_idx :]
        self._last_root_idx = len(self._require_client().roots)

        summary = self._summarize_roots(new_roots)
        cost_ratio, cost_retries, cost_roots = self._window_cost()
        summary["cost"] = cost_ratio
        # Raw counts so a consumer can pool sum(retries)/sum(roots) over the
        # episode rather than averaging per-window ratios.
        summary["cost_retries"] = float(cost_retries)
        summary["cost_roots"] = float(cost_roots)
        obs = self._build_observation(summary)
        reward = self._reward_from_summary(summary)
        terminated = self._current_time_ns >= self._end_time_ns
        info: dict[str, object] = {"summary": summary, "action": self._last_action_values.copy()}
        if self._cfg.emit_latency_samples:
            # Latency-based policies must see the client the knobs actuate: in
            # chain scenarios that is the B->C edge (cost_client), not the
            # top-level client whose end-to-end latencies the knobs only
            # influence indirectly. Reward/observation stay top-level.
            sample_client = self._require_components().cost_client
            if sample_client is not None:
                sample_roots = sample_client.roots[self._last_sample_root_idx :]
                self._last_sample_root_idx = len(sample_client.roots)
            else:
                sample_roots = new_roots
            info.update(self._collect_latency_samples(sample_roots))
            # Ground truth for a queueing model: arrivals at the MODELLED queue
            # (all clients, not just the sampled one) and its real worker count.
            modeled = self._require_components().modeled_service
            if modeled is not None:
                admitted = modeled.admitted_total
                info["modeled_admitted"] = admitted - self._last_admitted
                info["modeled_workers"] = modeled.cfg.workers
                self._last_admitted = admitted

        if terminated:
            self._horizon_root_count = self._last_root_idx
            # The drain below completes IN-FLIGHT work only: halt EVERY arrival
            # process so a workload outliving the episode horizon cannot admit
            # new post-horizon roots into the drained population. Builders that
            # drive more than one Workload must register the rest in
            # EnvComponents.extra_workloads or their traffic escapes this halt.
            self._require_workload().stop()
            for extra in self._require_components().extra_workloads:
                extra.stop()
            bystander = self._require_components().bystander_client
            self._bystander_horizon_count = 0 if bystander is None else len(bystander.roots)
        # Drain in-flight work after the reward is computed: drained roots complete
        # past the episode horizon and must not be credited to the final action
        # (unless terminal_drain_penalty explicitly folds them in).
        if terminated and self._cfg.drain_on_done:
            self._require_sim().run()
            drained = self._require_client().roots[self._last_root_idx :]
            self._last_root_idx = len(self._require_client().roots)
            drained_summary = self._summarize_roots(drained)
            drained_ratio, drained_retries, drained_roots = self._window_cost()
            drained_summary["cost"] = drained_ratio
            drained_summary["cost_retries"] = float(drained_retries)
            drained_summary["cost_roots"] = float(drained_roots)
            info["drained_summary"] = drained_summary
            if self._cfg.terminal_drain_penalty:
                # Count-consistent weighting: per-step rewards are rates over a
                # window's completions, so a flat drained-window rate would
                # penalize 2 drained roots as hard as 100. Scale by how many
                # average-sized step windows the drained population represents.
                total_completed = self._last_root_idx
                per_step = total_completed / max(1, self._episode_steps)
                weight = drained_summary["completed"] / max(1.0, per_step)
                reward += weight * self._reward_from_summary(drained_summary)
        if terminated:
            components = self._components
            if components is not None and components.bystander_client is not None:
                roots = components.bystander_client.roots
                info["bystander_summary"] = self._summarize_roots(roots)
                # Score the externality on the same population as the agent's
                # objective J, which only sees in-horizon roots. The full-
                # population summary stays for continuity with existing columns.
                info["bystander_summary_in_horizon"] = self._summarize_roots(
                    roots[: self._bystander_horizon_count]
                )
        return obs, reward, terminated, False, info

    def _build(self, seed: int | None) -> EnvComponents:
        return self._cfg.build(seed)

    def _window_cost(self) -> tuple[float, int, int]:
        """Constraint cost at the cost client over the roots it completed since
        the last call.

        Returns ``(ratio, retries, roots)``. The raw counts are reported too so
        consumers can pool them across windows: averaging the per-window RATIO
        is a length-biased (Simpson) estimator, because high-amplification
        windows also complete more edge roots. Measured 13-33% understatement of
        the true episode amplification, and by different amounts per arm, so it
        does not cancel in comparisons.
        """
        components = self._components
        if components is None or components.cost_client is None:
            return 0.0, 0, 0
        roots = components.cost_client.roots[self._last_cost_root_idx :]
        self._last_cost_root_idx = len(components.cost_client.roots)
        if not roots:
            return 0.0, 0, 0
        attempts = sum(len(root.attempts) for root in roots)
        retries = attempts - len(roots)
        return retries / len(roots), retries, len(roots)

    def _apply_action(self, action: tuple[int, ...]) -> None:
        spec = self._cfg.action_spec
        if len(action) != len(self._action_sizes):
            raise ValueError(
                f"Action has {len(action)} entries but the action space expects "
                f"{len(self._action_sizes)} (did a policy emit the wrong dimensionality?)"
            )
        for position, (index, size) in enumerate(zip(action, self._action_sizes, strict=True)):
            if not 0 <= index < size:
                # Python would wrap a negative index to the end of the list, so
                # an out-of-range action silently selected a valid-looking knob
                # and returned an observation outside the declared Box.
                raise ValueError(
                    f"action[{position}]={index} is outside [0, {size}) for this ActionSpec"
                )
        attempt_timeout_ms = spec.attempt_timeout_ms[action[0]]
        max_attempts = spec.max_attempts[action[1]]
        backoff_ms = spec.backoff_ms[action[2]]
        global_timeout_ms = None
        if spec.global_timeout_ms is not None:
            global_timeout_ms = spec.global_timeout_ms[action[3]]

        self._require_components().apply_knobs(
            ActionValues(
                attempt_timeout_ms=attempt_timeout_ms,
                max_attempts=max_attempts,
                backoff_ms=backoff_ms,
                global_timeout_ms=global_timeout_ms,
            )
        )

        self._last_action_values = {
            "attempt_timeout_ms": attempt_timeout_ms,
            "global_timeout_ms": global_timeout_ms,
            "max_attempts": max_attempts,
            "backoff_ms": backoff_ms,
        }
        self._last_action_indices = tuple(int(a) for a in action)

    def _reward_from_summary(self, summary: dict[str, float]) -> float:
        w = self._cfg.reward_weights
        reward = (
            summary["success_rate"] * w.success
            - summary["deadline_drop_rate"] * w.deadline_drop
            - summary["queue_drop_rate"] * w.queue_drop
            - summary["server_drop_rate"] * w.server_drop
            - summary["circuit_open_drop_rate"] * w.circuit_open_drop
            - summary["p95_latency_ms"] * w.p95_latency_ms
            - summary["retries_per_root"] * w.retries
            # Count-weighted: the dual ascends on the POOLED retries-per-root,
            # so the primal must penalise a quantity whose episode sum is
            # proportional to it. Penalising the raw per-window RATIO instead
            # makes a Lagrangian whose primal and dual optimise different
            # constraints (measured 1.104 window-mean vs 1.253 pooled), and dual
            # ascent then converges to neither.
            - self._cost_penalty(summary) * w.cost
        )
        return float(reward)

    def _cost_penalty(self, summary: dict[str, float]) -> float:
        """Per-window cost term whose episode SUM tracks the pooled ratio.

        `retries_t / mean_roots_per_window` sums to `total_retries / mean_roots`
        = `T * pooled_ratio`, so minimising it minimises the pooled amplification
        the constraint is stated in. Falls back to the raw ratio before any
        cost-client root has completed.
        """
        roots = summary.get("cost_roots", 0.0)
        retries = summary.get("cost_retries", 0.0)
        self._cost_roots_seen += roots
        self._cost_windows += 1 if roots else 0
        if not self._cost_windows:
            return float(summary.get("cost", 0.0))
        mean_roots = self._cost_roots_seen / self._cost_windows
        return float(retries / mean_roots) if mean_roots > 0 else 0.0

    def _summarize_roots(self, roots) -> dict[str, float]:
        if not roots:
            return {
                "completed": 0,
                "success_rate": 0.0,
                "deadline_drop_rate": 0.0,
                "queue_drop_rate": 0.0,
                "server_drop_rate": 0.0,
                "cancelled_rate": 0.0,
                "cost_retries": 0.0,
                "cost_roots": 0.0,
                "circuit_open_drop_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "retries_per_root": 0.0,
                "queue_avg": 0.0,
                "cost": 0.0,
            }

        # Attempt-less roots are skipped by both Metrics paths, so exclude them
        # from the denominator too — counting them in `completed` while no
        # outcome bucket claims them leaves the rates not summing to 1.
        roots = [root for root in roots if root.attempts]
        completed = len(roots)
        if not completed:
            return self._summarize_roots([])
        successes = 0
        deadline_drops = 0
        queue_drops = 0
        server_drops = 0
        circuit_open_drops = 0
        cancelled_drops = 0
        latencies: list[int] = []
        attempts_total = 0
        queue_sizes: list[int] = []

        for root in roots:
            attempts_total += len(root.attempts)
            # Same classifier as Metrics.summary() and export_csv(): these three
            # populations must agree, and they drifted when each owned its rules.
            outcome = classify_root(root)
            if outcome is RootOutcome.SUCCESS:
                successes += 1
                latencies.append(root_latency(root))
            elif outcome is RootOutcome.CANCELLED:
                cancelled_drops += 1
            elif outcome is RootOutcome.DEADLINE:
                deadline_drops += 1
            elif outcome is RootOutcome.QUEUE_FULL:
                queue_drops += 1
            elif outcome is RootOutcome.CIRCUIT_OPEN:
                circuit_open_drops += 1
            else:
                server_drops += 1

            for attempt in root.attempts:
                if attempt.queue_size_at_end is not None:
                    queue_sizes.append(int(attempt.queue_size_at_end))

        avg_latency_ms = ns_to_ms(int(np.mean(latencies))) if latencies else 0.0
        p95_latency_ms = (
            ns_to_ms(int(np.quantile(np.asarray(latencies), 0.95))) if latencies else 0.0
        )
        queue_avg = float(np.mean(queue_sizes)) if queue_sizes else 0.0

        return {
            "completed": completed,
            "success_rate": successes / completed,
            "deadline_drop_rate": deadline_drops / completed,
            "queue_drop_rate": queue_drops / completed,
            "server_drop_rate": server_drops / completed,
            # Reported for visibility; deliberately carries no reward weight
            # and no observation feature, since cancelled work is abandoned
            # rather than a failure of this service.
            "cancelled_rate": cancelled_drops / completed,
            "circuit_open_drop_rate": circuit_open_drops / completed,
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "retries_per_root": (attempts_total - completed) / completed,
            "queue_avg": queue_avg,
            # Overridden by step() with the cost-client window when one exists.
            # Kept here so the empty-window and populated dicts carry identical
            # keys — a consumer reading info["summary"] must never KeyError
            # depending on whether the window had traffic.
            "cost": 0.0,
            "cost_retries": 0.0,
            "cost_roots": 0.0,
        }

    def _build_observation(self, summary: dict[str, float]) -> np.ndarray:
        """Normalized observation vector.

        Features: has_data flag (disambiguates "no traffic" from "all failed"),
        log-scaled throughput, the five outcome rates, latencies scaled by the
        largest attempt timeout in the spec, retries scaled by the max retry
        budget, constraint cost (cost-client retry amplification, same scaling),
        log-scaled queue depth, the last action echoed as index fractions, and
        the episode-time fraction.
        """
        completed = float(summary["completed"])

        def idx_frac(position: int) -> float:
            size = self._action_sizes[position]
            if size <= 1:
                return 0.0
            return float(self._last_action_indices[position]) / float(size - 1)

        features = [
            1.0 if completed > 0 else 0.0,
            float(np.log1p(completed)) / 10.0,
            summary["success_rate"],
            summary["deadline_drop_rate"],
            summary["queue_drop_rate"],
            summary["server_drop_rate"],
            summary["circuit_open_drop_rate"],
            summary["avg_latency_ms"] / self._timeout_norm_ms,
            summary["p95_latency_ms"] / self._timeout_norm_ms,
            summary["retries_per_root"] / self._retries_norm,
            summary["cost"] / self._retries_norm,
            float(np.log1p(summary["queue_avg"])) / 10.0,
            idx_frac(0),
            idx_frac(1),
            idx_frac(2),
        ]
        if len(self._action_sizes) > 3:
            features.append(idx_frac(3))
        time_s = ns_to_s(self._current_time_ns)
        episode_s = (
            self._cfg.episode_duration_s
            if self._cfg.episode_duration_s is not None
            else self._require_workload().duration_s
        )
        # Durations are validated positive; the epsilon only guards a
        # zero-duration workload fallback. No 1.0 floor: sub-second episodes
        # must still normalize to [0, 1].
        features.append(time_s / max(1e-9, episode_s))
        return np.array(features, dtype=np.float32)

    def _collect_latency_samples(self, roots) -> dict[str, list]:
        attempt_latencies = []
        attempt_censored = []
        attempt_service = []
        attempt_service_censored = []
        root_latencies = []

        for root in roots:
            if not root.attempts:
                continue
            begin = min(a.interval.begin for a in root.attempts)
            end = max(a.interval.end for a in root.attempts)
            if any(a.success for a in root.attempts):
                root_latencies.append(ns_to_ms(end - begin))

            for attempt in root.attempts:
                if (
                    attempt.drop_reason in (DropReason.QUEUE_FULL, DropReason.CIRCUIT_OPEN)
                    and attempt.interval.end == attempt.interval.begin
                ):
                    # Instant rejections carry no latency information; policies
                    # see them through the drop-rate features instead. A zero
                    # would poison lognormal fits (log 0) as an uncensored value.
                    continue
                # Deadline cuts and cancellations TRUNCATE the attempt, so its
                # latency is a lower bound on what it would have been.
                truncated = attempt.drop_reason in (DropReason.DEADLINE, DropReason.CANCELLED)
                attempt_latencies.append(ns_to_ms(attempt.interval.end - attempt.interval.begin))
                attempt_censored.append(truncated)
                if attempt.service_time_ns is not None:
                    attempt_service.append(ns_to_ms(attempt.service_time_ns))
                    # The OCCUPANCY is censored only when the worker was still
                    # held at the cut — the service records that directly. A
                    # tandem node's local stage finishes before the worker is
                    # handed back, so a later deadline censors the attempt but
                    # not this sample; inferring the flag from the attempt's
                    # drop reason fed complete observations to the censored MLE
                    # as lower bounds and inflated E[S] several-fold.
                    attempt_service_censored.append(attempt.service_time_censored)

        if self._cfg.max_latency_samples is not None:
            attempt_latencies = attempt_latencies[: self._cfg.max_latency_samples]
            attempt_censored = attempt_censored[: self._cfg.max_latency_samples]
            attempt_service = attempt_service[: self._cfg.max_latency_samples]
            attempt_service_censored = attempt_service_censored[: self._cfg.max_latency_samples]
            root_latencies = root_latencies[: self._cfg.max_latency_samples]

        return {
            "attempt_latencies_ms": attempt_latencies,
            "attempt_censored": attempt_censored,
            "attempt_service_ms": attempt_service,
            "attempt_service_censored": attempt_service_censored,
            "root_latencies_ms": root_latencies,
        }

    def metrics_summary(self) -> Summary | None:
        if self._client is None:
            return None
        return self._client.metrics().summary()

    def in_horizon_summary(self) -> Summary | None:
        """Summary over roots completed before the episode horizon — the
        population the reward actually saw. ``metrics_summary()`` is the full
        population including drained roots."""
        if self._client is None:
            return None
        roots = self._client.roots[: self._horizon_root_count]
        return Metrics(roots=roots, attempts_total=sum(len(r.attempts) for r in roots)).summary()

    @property
    def step_duration_s(self) -> float:
        return self._cfg.step_duration_s

    def bystander_metrics_summary(self) -> Summary | None:
        if self._components is None or self._components.bystander_client is None:
            return None
        return self._components.bystander_client.metrics().summary()

    def bystander_in_horizon_summary(self) -> Summary | None:
        """Bystander summary over roots completed before the episode horizon.

        Mirrors ``in_horizon_summary()`` so the retry externality is measured on
        the same population as the agent's objective J. ``bystander_metrics_summary()``
        is the full population including roots drained past the horizon.
        """
        if self._components is None or self._components.bystander_client is None:
            return None
        roots = self._components.bystander_client.roots[: self._bystander_horizon_count]
        return Metrics(roots=roots, attempts_total=sum(len(r.attempts) for r in roots)).summary()

    def set_cost_weight(self, value: float) -> None:
        """Set the constraint-cost reward weight (the Lagrange multiplier).

        Exists as a method so constrained training can broadcast lambda into
        SubprocVecEnv workers via ``vec_env.env_method`` — mutating a shared
        RewardWeights object does not cross process boundaries.
        """
        self._cfg.reward_weights.cost = float(value)


def _static_knob_applier(
    timeout_ctrl: StaticTimeoutController, retry: FixedBackoffRetryPolicy
) -> Callable[[ActionValues], None]:
    def apply_knobs(values: ActionValues) -> None:
        timeout_ctrl.attempt_timeout = ms_to_ns(values.attempt_timeout_ms)
        timeout_ctrl.global_timeout = (
            None if values.global_timeout_ms is None else ms_to_ns(values.global_timeout_ms)
        )
        retry.max_attempts = values.max_attempts
        retry.delay = ms_to_ns(values.backoff_ms)

    return apply_knobs


def _single_service_components(
    seed: int | None,
    duration: float,
    *,
    latency_injections: list[LatencyInjection] | None = None,
    partial_failures: list[PartialFailure] | None = None,
    load_spikes: list[LoadSpike] | None = None,
) -> EnvComponents:
    sim = Simulator(seed=1 if seed is None else seed)
    svc_cfg = ServiceConfig(
        name="svc-A",
        latency_median=ms_to_ns(20),
        latency_lognorm_sigma=0.5,
        workers=16,
        queue_capacity=200,
        latency_injections=latency_injections or [],
        partial_failures=partial_failures or [],
    )
    svc = ServiceRuntime(cfg=svc_cfg).bind()

    timeout_ctrl = StaticTimeoutController(global_timeout=None, attempt_timeout=ms_to_ns(50))
    retry = FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(50))
    client = ClientRuntime(
        cfg=ClientConfig(retry=retry, timeout_controller=timeout_ctrl), service=svc
    )

    workload = Workload(
        base_rps=200.0,
        duration_s=duration,
        load_spikes=load_spikes or [],
        rng_seed=seed,
    )

    return EnvComponents(
        sim=sim,
        workload=workload,
        client=client,
        modeled_service=svc,
        apply_knobs=_static_knob_applier(timeout_ctrl, retry),
    )


def build_single_service_env(
    seed: int | None = None, duration_s: float | None = None
) -> EnvComponents:
    duration = 30.0 if duration_s is None else float(duration_s)
    return _single_service_components(seed, duration)


def build_faulty_single_service_env(
    seed: int | None = None, duration_s: float | None = None
) -> EnvComponents:
    """Faulty single service with fault windows at *fixed* episode fractions.

    Because the observation includes the episode-time fraction, policies can
    memorize these windows; use the randomized variant for training.
    """
    duration = 30.0 if duration_s is None else float(duration_s)
    return _single_service_components(
        seed,
        duration,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(0.2 * duration), end=s_to_ns(0.35 * duration)),
                add_latency=ms_to_ns(30),
            )
        ],
        partial_failures=[
            PartialFailure(
                duration=TimeInterval(begin=s_to_ns(0.5 * duration), end=s_to_ns(0.65 * duration)),
                p_fail=0.3,
            )
        ],
        load_spikes=[
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(0.3 * duration), end=s_to_ns(0.45 * duration)),
                rps_multiplier=2.0,
            )
        ],
    )


def build_randomized_faulty_single_service_env(
    seed: int | None = None, duration_s: float | None = None
) -> EnvComponents:
    """Faulty single service with fault windows drawn from the seed.

    Onset, duration, and intensity vary per episode, so a policy cannot
    memorize the episode clock via the time-fraction feature — it has to react
    to observed degradation. Same seed reproduces the same fault draw.
    """
    duration = 30.0 if duration_s is None else float(duration_s)
    rng = random.Random((1 if seed is None else seed) + _FAULT_SEED_OFFSET)

    def window() -> TimeInterval:
        onset = rng.uniform(0.05, 0.6) * duration
        length = rng.uniform(0.1, 0.3) * duration
        return TimeInterval(begin=s_to_ns(onset), end=s_to_ns(min(onset + length, duration)))

    return _single_service_components(
        seed,
        duration,
        latency_injections=[
            LatencyInjection(duration=window(), add_latency=ms_to_ns(rng.uniform(10, 60)))
        ],
        partial_failures=[PartialFailure(duration=window(), p_fail=rng.uniform(0.1, 0.6))],
        load_spikes=[LoadSpike(duration=window(), rps_multiplier=rng.uniform(1.5, 3.0))],
    )


def build_chain_env(
    seed: int | None = None,
    duration_s: float | None = None,
    *,
    retry_budget_ratio: float | None = None,
) -> EnvComponents:
    """Chain A -> B -> C with bottleneck C degrading in a randomized fault window.

    The agent tunes the B->C edge client's retry/timeout knobs — the edge where
    retry storms amplify — while reward and observation come from the
    *top-level* client, so the agent is judged on end-to-end goodput. Fault
    onset, duration, and intensity are drawn from the seed (see the randomized
    single-service builder for why). The edge client is exposed as the env's
    ``cost_client`` so its retry amplification can serve as a constraint cost.

    ``retry_budget_ratio`` wraps the edge policy in a ``RetryBudgetPolicy``
    (the hand-coded baseline for constrained-RL comparisons). Note the wrapper
    limits retries against *successes* (token award), which approximates
    retries-per-root only at high success rates.
    """
    build_seed = 1 if seed is None else int(seed)
    sim = Simulator(seed=build_seed)
    duration = 30.0 if duration_s is None else float(duration_s)
    rng = random.Random(build_seed + _FAULT_SEED_OFFSET)

    onset = rng.uniform(0.1, 0.5) * duration
    length = rng.uniform(0.15, 0.35) * duration
    fault_window = TimeInterval(begin=s_to_ns(onset), end=s_to_ns(min(onset + length, duration)))

    c_cfg = ServiceConfig(
        name="svc-C",
        latency_median=ms_to_ns(20),
        latency_lognorm_sigma=0.7,
        workers=24,
        queue_capacity=800,
        latency_injections=[
            LatencyInjection(duration=fault_window, add_latency=ms_to_ns(rng.uniform(15, 45)))
        ],
        partial_failures=[PartialFailure(duration=fault_window, p_fail=rng.uniform(0.2, 0.5))],
    )
    a_cfg = ServiceConfig(
        name="svc-A",
        latency_median=ms_to_ns(12),
        latency_lognorm_sigma=0.4,
        workers=48,
        queue_capacity=2000,
    )
    b_cfg = ServiceConfig(
        name="svc-B",
        latency_median=ms_to_ns(12),
        latency_lognorm_sigma=0.4,
        workers=48,
        queue_capacity=2000,
    )

    a_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=a_cfg).bind())
    b_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=b_cfg).bind())
    c_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=c_cfg).bind())

    # Tunable edge: B -> C, where aggressive retries hammer the bottleneck.
    edge_timeout = StaticTimeoutController(global_timeout=None, attempt_timeout=ms_to_ns(60))
    edge_retry = FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(20))
    edge_policy: RetryPolicy = edge_retry
    if retry_budget_ratio is not None:
        # Knobs still route to the inner FixedBackoffRetryPolicy via
        # apply_knobs; the wrapper sees results through underlying_strategy.
        edge_policy = RetryBudgetPolicy(
            budget_ratio=retry_budget_ratio,
            max_retries=5,
            underlying_strategy=edge_retry,
        )
    b_to_c = ClientRuntime(
        cfg=ClientConfig(retry=edge_policy, timeout_controller=edge_timeout), service=c_node
    )

    a_to_b = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(20)),
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(150)
            ),
        ),
        service=b_node,
    )

    b_node.add_downstream(DownstreamEdge(client=b_to_c, target=c_node))
    a_node.add_downstream(DownstreamEdge(client=a_to_b, target=b_node))

    top_client = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(30)),
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(300)
            ),
        ),
        service=a_node,
    )

    workload = Workload(
        base_rps=180.0,
        duration_s=duration,
        load_spikes=[],
        rng_seed=seed,
    )

    return EnvComponents(
        sim=sim,
        workload=workload,
        client=top_client,
        apply_knobs=_static_knob_applier(edge_timeout, edge_retry),
        cost_client=b_to_c,
        modeled_service=c_node.service,
    )


def build_multichain_env(
    seed: int | None = None,
    duration_s: float | None = None,
    *,
    retry_budget_ratio: float | None = None,
) -> EnvComponents:
    """Two chains A -> B -> C and X -> Y -> C sharing bottleneck C.

    The agent tunes its own chain's B->C edge; reward comes from chain A's
    top-level client (a *selfish* objective). The X->Y->C bystander chain runs
    fixed, retry-free policies and is exposed as ``bystander_client`` — its
    success rate gauges the externality of the agent's retries, since both
    chains drain the same C worker pool and queue.

    Unlike the single-chain scenario, C degrades with pure partial-failure
    blips and *no* latency injection: p_fail is rolled fresh per attempt at
    attempt end, so an edge retry genuinely recovers a failed call and easily
    fits the 60 ms edge attempt timeout — but every failed attempt still burns
    a full service time on a shared worker. That makes some retrying optimal
    (cost > 0 at the unconstrained optimum) while amplification during blips
    steals capacity from the bystander: the regime where a retry budget forces
    a real trade-off. This mirrors production retry budgets (Finagle's
    ``retryBudget`` defaults to 20% of requests, gRPC retry throttling to a
    ~10% token ratio), which exist to bound exactly this externality.

    Blip count, onset, length, and p_fail are drawn from the seed (see the
    randomized single-service builder for why). ``retry_budget_ratio`` wraps
    the edge policy in ``RetryBudgetPolicy`` as in ``build_chain_env``.
    """
    build_seed = 1 if seed is None else int(seed)
    sim = Simulator(seed=build_seed)
    duration = 30.0 if duration_s is None else float(duration_s)
    rng = random.Random(build_seed + _FAULT_SEED_OFFSET)

    # p_fail must be high enough that upstream retry layers (A->B x2, top x2:
    # up to 4 end-to-end C rolls) cannot recover blips on their own — at ~0.65
    # a no-edge-retry root succeeds ~1-0.65^4 ~= 0.82 during a blip, so cheap
    # edge retries carry genuine value. Calibrated via `constrained_retry
    # --calibrate`; at U(0.3, 0.6) the edge-retry arm loses to no-retry.
    blips: list[PartialFailure] = []
    for _ in range(rng.randint(2, 4)):
        onset = rng.uniform(0.05, 0.75) * duration
        length = rng.uniform(0.08, 0.20) * duration
        window = TimeInterval(begin=s_to_ns(onset), end=s_to_ns(min(onset + length, duration)))
        blips.append(PartialFailure(duration=window, p_fail=rng.uniform(0.5, 0.8)))

    # Calibration knobs: median/sigma/workers size C against the two chains'
    # combined load (2 x 180 rps x ~22.7 ms mean ~= 8.2 erlangs -> ~0.58
    # utilization), leaving headroom for blip retries to actually complete
    # (at 12 workers C saturates during blips and retries time out, eroding
    # their value) while amplification still queues C enough to hurt the
    # bystander.
    c_cfg = ServiceConfig(
        name="svc-C",
        latency_median=ms_to_ns(20),
        latency_lognorm_sigma=0.5,
        workers=14,
        queue_capacity=800,
        partial_failures=blips,
    )

    def _relay_cfg(name: str) -> ServiceConfig:
        return ServiceConfig(
            name=name,
            latency_median=ms_to_ns(12),
            latency_lognorm_sigma=0.4,
            workers=48,
            queue_capacity=2000,
        )

    a_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=_relay_cfg("svc-A")).bind())
    b_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=_relay_cfg("svc-B")).bind())
    x_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=_relay_cfg("svc-X")).bind())
    y_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=_relay_cfg("svc-Y")).bind())
    c_node = ServiceNodeRuntime(service=ServiceRuntime(cfg=c_cfg).bind())

    # Tunable edge: B -> C on the agent's chain.
    edge_timeout = StaticTimeoutController(global_timeout=None, attempt_timeout=ms_to_ns(60))
    edge_retry = FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(20))
    edge_policy: RetryPolicy = edge_retry
    if retry_budget_ratio is not None:
        edge_policy = RetryBudgetPolicy(
            budget_ratio=retry_budget_ratio,
            max_retries=5,
            underlying_strategy=edge_retry,
        )
    b_to_c = ClientRuntime(
        cfg=ClientConfig(retry=edge_policy, timeout_controller=edge_timeout), service=c_node
    )

    # Bystander edge Y -> C: retry-free, so its success rate is an
    # uncontaminated gauge of contention at C.
    y_to_c = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=1, delay=0),
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(60)
            ),
        ),
        service=c_node,
    )

    # Relay edges (A->B, X->Y) do not retry: hop-by-hop retries at every layer
    # are the retry-storm anti-pattern, and they double as a hidden recovery
    # channel that erases the value of the agent's edge knob (calibration
    # showed the no-edge-retry arm winning when relays retried). Recovery
    # lives at the client of the flaky dependency (the tunable B->C edge) and
    # at the top-level clients.
    def _relay_client(node: ServiceNodeRuntime) -> ClientRuntime:
        return ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=1, delay=0),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(150)
                ),
            ),
            service=node,
        )

    b_node.add_downstream(DownstreamEdge(client=b_to_c, target=c_node))
    a_node.add_downstream(DownstreamEdge(client=_relay_client(b_node), target=b_node))
    y_node.add_downstream(DownstreamEdge(client=y_to_c, target=c_node))
    x_node.add_downstream(DownstreamEdge(client=_relay_client(y_node), target=y_node))

    def _top_client(node: ServiceNodeRuntime) -> ClientRuntime:
        return ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(30)),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(300)
                ),
            ),
            service=node,
        )

    top_client = _top_client(a_node)
    bystander_top = _top_client(x_node)

    workload = Workload(
        base_rps=180.0,
        duration_s=duration,
        load_spikes=[],
        rng_seed=seed,
    )
    # The env only drives EnvComponents.workload, so the bystander arrival
    # process is scheduled here; drive() just pushes tick events onto the sim
    # heap. It MUST also be registered in extra_workloads: the episode horizon
    # can be shorter than `duration`, and an unregistered workload keeps
    # admitting brand-new roots throughout the terminal drain.
    bystander_workload = Workload(
        base_rps=180.0,
        duration_s=duration,
        load_spikes=[],
        rng_seed=build_seed + 100_003,
    )
    bystander_workload.drive(sim, lambda s: bystander_top.start_request(s))

    return EnvComponents(
        sim=sim,
        workload=workload,
        client=top_client,
        apply_knobs=_static_knob_applier(edge_timeout, edge_retry),
        cost_client=b_to_c,
        modeled_service=c_node.service,
        bystander_client=bystander_top,
        extra_workloads=[bystander_workload],
    )


# Scenario name -> builder, shared by the CLI entry points (--fault-scenario).
SCENARIO_BUILDERS: dict[str, Callable[[int | None, float | None], EnvComponents]] = {
    "none": build_single_service_env,
    "basic": build_faulty_single_service_env,
    "randomized": build_randomized_faulty_single_service_env,
    "chain": build_chain_env,
    "chain_budget": functools.partial(build_chain_env, retry_budget_ratio=0.1),
    "multichain": build_multichain_env,
    "multichain_budget": functools.partial(build_multichain_env, retry_budget_ratio=0.1),
}
