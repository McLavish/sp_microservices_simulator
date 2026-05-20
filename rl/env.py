from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from caller import ClientConfig, ClientRuntime
from core import (
    DropReason,
    TimeDuration,
    TimeInterval,
    TimePoint,
    ms_to_ns,
    ns_to_ms,
    ns_to_s,
    s_to_ns,
)
from faults import LatencyInjection, LoadSpike, PartialFailure
from retry_policies import FixedBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - exercised in runtime environments
    gym = None
    spaces = None
    _GYM_IMPORT_ERROR = exc
else:
    _GYM_IMPORT_ERROR = None


@dataclass
class EnvComponents:
    sim: Simulator
    workload: Workload
    client: ClientRuntime


@dataclass
class RewardWeights:
    success: float = 1.0
    deadline_drop: float = 1.0
    queue_drop: float = 1.0
    server_drop: float = 1.0
    p95_latency_ms: float = 0.001
    retries: float = 0.1


@dataclass
class ActionSpec:
    attempt_timeout_ms: list[int]
    max_attempts: list[int]
    backoff_ms: list[int]
    global_timeout_ms: list[int | None] | None = None


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


_GymEnvBase = object if gym is None else gym.Env


class MicroservicesEnv(_GymEnvBase):
    """Gymnasium environment wrapper for the microservices simulator."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig):
        if gym is None:
            raise ImportError("gymnasium is required for MicroservicesEnv") from _GYM_IMPORT_ERROR
        self._cfg = config

        self._action_sizes = [
            len(config.action_spec.attempt_timeout_ms),
            len(config.action_spec.max_attempts),
            len(config.action_spec.backoff_ms),
        ]
        if config.action_spec.global_timeout_ms is not None:
            self._action_sizes.append(len(config.action_spec.global_timeout_ms))

        self.action_space = spaces.MultiDiscrete(self._action_sizes)

        self._obs_size = 12
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(self._obs_size,), dtype=np.float32
        )

        self._sim: Simulator | None = None
        self._workload: Workload | None = None
        self._client: ClientRuntime | None = None
        self._last_root_idx = 0
        self._current_time_ns: TimePoint = 0
        self._end_time_ns: TimePoint = 0
        self._step_ns: TimeDuration = 0
        self._last_action_values: dict[str, int | None] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        components = self._build(seed)
        self._sim = components.sim
        self._workload = components.workload
        self._client = components.client

        self._current_time_ns = 0
        self._step_ns = s_to_ns(self._cfg.step_duration_s)
        episode_s = (
            self._cfg.episode_duration_s
            if self._cfg.episode_duration_s is not None
            else self._workload.duration_s
        )
        self._end_time_ns = s_to_ns(episode_s)

        self._last_root_idx = 0
        self._workload.drive(self._sim, lambda s: self._client.start_request(s))

        initial_action = self._cfg.initial_action or tuple(
            0 for _ in range(len(self._action_sizes))
        )
        self._apply_action(initial_action)

        obs = self._build_observation(self._summarize_roots([]))
        info = {"action": self._last_action_values.copy()}
        if self._cfg.emit_latency_samples:
            info.update(
                {
                    "attempt_latencies_ms": [],
                    "attempt_censored": [],
                    "root_latencies_ms": [],
                }
            )
        return obs, info

    def step(self, action):
        from rl.policies.action_space import coerce_action

        action = coerce_action(action)
        self._apply_action(action)

        next_time = min(self._current_time_ns + self._step_ns, self._end_time_ns)
        self._sim.run(until=next_time)
        self._current_time_ns = next_time

        if self._current_time_ns >= self._end_time_ns and self._cfg.drain_on_done:
            self._sim.run()

        new_roots = self._client.roots[self._last_root_idx :]
        self._last_root_idx = len(self._client.roots)

        summary = self._summarize_roots(new_roots)
        obs = self._build_observation(summary)
        reward = self._reward_from_summary(summary)
        terminated = self._current_time_ns >= self._end_time_ns
        info = {"summary": summary, "action": self._last_action_values.copy()}
        if self._cfg.emit_latency_samples:
            info.update(self._collect_latency_samples(new_roots))
        return obs, reward, terminated, False, info

    def _build(self, seed: int | None) -> EnvComponents:
        return self._cfg.build(seed)

    def _apply_action(self, action: tuple[int, ...]) -> None:
        spec = self._cfg.action_spec
        idx = 0
        attempt_timeout_ms = spec.attempt_timeout_ms[action[idx]]
        idx += 1
        max_attempts = spec.max_attempts[action[idx]]
        idx += 1
        backoff_ms = spec.backoff_ms[action[idx]]
        idx += 1
        global_timeout_ms = None
        if spec.global_timeout_ms is not None:
            global_timeout_ms = spec.global_timeout_ms[action[idx]]

        timeout_controller = self._client.cfg.timeout_controller
        retry_policy = self._client.cfg.retry

        if not isinstance(timeout_controller, StaticTimeoutController):
            raise TypeError("MicroservicesEnv requires StaticTimeoutController for now")
        if not isinstance(retry_policy, FixedBackoffRetryPolicy):
            raise TypeError("MicroservicesEnv requires FixedBackoffRetryPolicy for now")

        timeout_controller.attempt_timeout = ms_to_ns(attempt_timeout_ms)
        timeout_controller.global_timeout = (
            None if global_timeout_ms is None else ms_to_ns(global_timeout_ms)
        )
        retry_policy.max_attempts = max_attempts
        retry_policy.delay = ms_to_ns(backoff_ms)

        self._last_action_values = {
            "attempt_timeout_ms": attempt_timeout_ms,
            "global_timeout_ms": global_timeout_ms,
            "max_attempts": max_attempts,
            "backoff_ms": backoff_ms,
        }

    def _reward_from_summary(self, summary: dict[str, float]) -> float:
        w = self._cfg.reward_weights
        reward = (
            summary["success_rate"] * w.success
            - summary["deadline_drop_rate"] * w.deadline_drop
            - summary["queue_drop_rate"] * w.queue_drop
            - summary["server_drop_rate"] * w.server_drop
            - summary["p95_latency_ms"] * w.p95_latency_ms
            - summary["retries_per_root"] * w.retries
        )
        return float(reward)

    def _summarize_roots(self, roots) -> dict[str, float]:
        if not roots:
            return {
                "completed": 0,
                "success_rate": 0.0,
                "deadline_drop_rate": 0.0,
                "queue_drop_rate": 0.0,
                "server_drop_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "retries_per_root": 0.0,
                "queue_avg": 0.0,
            }

        completed = len(roots)
        successes = 0
        deadline_drops = 0
        queue_drops = 0
        server_drops = 0
        latencies: list[int] = []
        attempts_total = 0
        queue_sizes: list[int] = []

        for root in roots:
            attempts_total += len(root.attempts)
            last_attempt = max(root.attempts, key=lambda a: a.interval.end)
            if any(a.success for a in root.attempts):
                successes += 1
                begin = min(a.interval.begin for a in root.attempts)
                end = max(a.interval.end for a in root.attempts)
                latencies.append(end - begin)
            else:
                if last_attempt.drop_reason == DropReason.DEADLINE:
                    deadline_drops += 1
                elif last_attempt.drop_reason == DropReason.QUEUE_FULL:
                    queue_drops += 1
                else:
                    server_drops += 1

            for attempt in root.attempts:
                if attempt.queue_size_at_end is not None:
                    queue_sizes.append(int(attempt.queue_size_at_end))

        success_rate = successes / completed if completed else 0.0
        deadline_rate = deadline_drops / completed if completed else 0.0
        queue_rate = queue_drops / completed if completed else 0.0
        server_rate = server_drops / completed if completed else 0.0
        avg_latency_ms = ns_to_ms(float(np.mean(latencies))) if latencies else 0.0
        p95_latency_ms = (
            ns_to_ms(float(np.quantile(np.asarray(latencies), 0.95))) if latencies else 0.0
        )
        retries_per_root = (attempts_total - completed) / completed if completed else 0.0
        queue_avg = float(np.mean(queue_sizes)) if queue_sizes else 0.0

        return {
            "completed": completed,
            "success_rate": success_rate,
            "deadline_drop_rate": deadline_rate,
            "queue_drop_rate": queue_rate,
            "server_drop_rate": server_rate,
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "retries_per_root": retries_per_root,
            "queue_avg": queue_avg,
        }

    def _build_observation(self, summary: dict[str, float]) -> np.ndarray:
        time_s = ns_to_s(self._current_time_ns)
        return np.array(
            [
                summary["success_rate"],
                summary["deadline_drop_rate"],
                summary["queue_drop_rate"],
                summary["server_drop_rate"],
                summary["avg_latency_ms"],
                summary["p95_latency_ms"],
                summary["retries_per_root"],
                summary["queue_avg"],
                float(self._last_action_values.get("attempt_timeout_ms") or 0),
                float(self._last_action_values.get("max_attempts") or 0),
                float(self._last_action_values.get("backoff_ms") or 0),
                float(time_s / max(1.0, self._cfg.episode_duration_s or self._workload.duration_s)),
            ],
            dtype=np.float32,
        )

    def _collect_latency_samples(self, roots) -> dict[str, list]:
        attempt_latencies = []
        attempt_censored = []
        root_latencies = []

        for root in roots:
            if not root.attempts:
                continue
            begin = min(a.interval.begin for a in root.attempts)
            end = max(a.interval.end for a in root.attempts)
            if any(a.success for a in root.attempts):
                root_latencies.append(ns_to_ms(end - begin))

            for attempt in root.attempts:
                attempt_latencies.append(ns_to_ms(attempt.interval.end - attempt.interval.begin))
                attempt_censored.append(attempt.drop_reason == DropReason.DEADLINE)

        if self._cfg.max_latency_samples is not None:
            attempt_latencies = attempt_latencies[: self._cfg.max_latency_samples]
            attempt_censored = attempt_censored[: self._cfg.max_latency_samples]
            root_latencies = root_latencies[: self._cfg.max_latency_samples]

        return {
            "attempt_latencies_ms": attempt_latencies,
            "attempt_censored": attempt_censored,
            "root_latencies_ms": root_latencies,
        }

    def metrics_summary(self):
        if self._client is None:
            return None
        return self._client.metrics().summary()


def build_single_service_env(
    seed: int | None = None, duration_s: int | None = None
) -> EnvComponents:
    sim = Simulator(seed=1 if seed is None else seed)
    duration = 30 if duration_s is None else int(duration_s)

    svc_cfg = ServiceConfig(
        name="svc-A",
        latency_median=ms_to_ns(20),
        latency_lognorm_sigma=0.5,
        workers=16,
        queue_capacity=200,
    )
    svc = ServiceRuntime(cfg=svc_cfg).bind()

    timeout_ctrl = StaticTimeoutController(global_timeout=None, attempt_timeout=ms_to_ns(50))
    retry = FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(50))
    client_cfg = ClientConfig(retry=retry, timeout_controller=timeout_ctrl)
    client = ClientRuntime(cfg=client_cfg, service=svc)

    workload = Workload(
        base_rps=200.0,
        duration_s=duration,
        load_spikes=[],
        rng_seed=seed,
    )

    return EnvComponents(sim=sim, workload=workload, client=client)


def build_faulty_single_service_env(
    seed: int | None = None, duration_s: int | None = None
) -> EnvComponents:
    sim = Simulator(seed=1 if seed is None else seed)
    duration = 30 if duration_s is None else int(duration_s)

    latency_injection = LatencyInjection(
        duration=TimeInterval(begin=s_to_ns(0.2 * duration), end=s_to_ns(0.35 * duration)),
        add_latency=ms_to_ns(30),
    )
    partial_failure = PartialFailure(
        duration=TimeInterval(begin=s_to_ns(0.5 * duration), end=s_to_ns(0.65 * duration)),
        p_fail=0.3,
    )

    svc_cfg = ServiceConfig(
        name="svc-A",
        latency_median=ms_to_ns(20),
        latency_lognorm_sigma=0.5,
        workers=16,
        queue_capacity=200,
        latency_injections=[latency_injection],
        partial_failures=[partial_failure],
    )
    svc = ServiceRuntime(cfg=svc_cfg).bind()

    timeout_ctrl = StaticTimeoutController(global_timeout=None, attempt_timeout=ms_to_ns(50))
    retry = FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(50))
    client_cfg = ClientConfig(retry=retry, timeout_controller=timeout_ctrl)
    client = ClientRuntime(cfg=client_cfg, service=svc)

    workload = Workload(
        base_rps=200.0,
        duration_s=duration,
        load_spikes=[
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(0.3 * duration), end=s_to_ns(0.45 * duration)),
                rps_multiplier=2.0,
            )
        ],
        rng_seed=seed,
    )

    return EnvComponents(sim=sim, workload=workload, client=client)
