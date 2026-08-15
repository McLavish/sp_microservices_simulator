"""One episode loop, shared by every evaluation path.

Four loops used to exist — `compare_policies._episode`,
`constrained.evaluate_objective_and_cost`, `constrained_retry._eval_arm` and the
eval loop in `train_sb3` — and they drifted in exactly the ways that matter:
one averaged per-window cost ratios while another pooled raw counts, and one
scored the bystander on the full population while the objective saw only
in-horizon roots. Both discrepancies produced numbers that looked comparable and
were not.

Everything specific to a caller (trace rows, violation flags, printing) stays
with the caller; only the loop and the aggregations that must agree live here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from core import Summary
from rl.spec import RewardWeights

ActFn = Callable[[Any, dict], Sequence[int]]


def mean_ignoring_nan(values: Sequence[float]) -> float:
    """Mean over the seeds that produced data; 0.0 for an empty sequence.

    `Metrics.summary()` reports latency percentiles as nan when no root
    succeeded — deliberately, so a missing value cannot be averaged toward
    zero. Plain `np.mean` then propagates that nan across every other eval
    seed, so one degenerate seed voids the whole cell and discards the rest.
    Keep the seeds that did report, and return nan only when NONE of them did,
    so a wholly missing column stays visibly missing.
    """
    if not values:
        return 0.0
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


class _SupportsAct(Protocol):
    def act(self, obs: Any, info: dict) -> Sequence[int]: ...


def policy_actor(policy: _SupportsAct) -> ActFn:
    """Adapt a `Policy` to the act-callable this module drives."""
    return lambda obs, info: policy.act(obs, info)


def model_actor(model, *, deterministic: bool = True) -> ActFn:
    """Adapt an SB3 model, whose `predict` returns `(action, state)`."""

    def act(obs, _info):
        action, _state = model.predict(obs, deterministic=deterministic)
        return action

    return act


@dataclass
class WindowRecord:
    """One `env.step()` — the reward-visible unit."""

    step: int
    reward: float
    summary: dict[str, float]
    action_values: dict[str, int | None]


@dataclass
class EpisodeResult:
    seed: int | None
    total_reward: float
    windows: list[WindowRecord] = field(default_factory=list)
    metrics: Summary | None = None
    in_horizon: Summary | None = None
    bystander_in_horizon: Summary | None = None
    drained_summary: dict[str, float] | None = None

    @property
    def pooled_cost(self) -> float:
        """Retry amplification POOLED over the episode: sum(retries)/sum(roots).

        Not a mean of per-window ratios. Windows with high amplification also
        complete more edge roots, so an unweighted mean is length-biased and
        understates the episode's true amplification — measured 13-33%, and by
        different amounts per arm, so it does not cancel between arms.
        """
        retries = sum(w.summary.get("cost_retries", 0.0) for w in self.windows)
        roots = sum(w.summary.get("cost_roots", 0.0) for w in self.windows)
        return retries / roots if roots else 0.0

    def objective(self, weights: RewardWeights) -> float:
        """Base objective J: the reward with retry and cost penalties zeroed.

        Every arm is scored on this regardless of its training weights, so the
        comparison scale does not depend on how an arm was trained.
        """
        return sum(
            w.summary["success_rate"] * weights.success
            - w.summary["deadline_drop_rate"] * weights.deadline_drop
            - w.summary["queue_drop_rate"] * weights.queue_drop
            - w.summary["server_drop_rate"] * weights.server_drop
            - w.summary["circuit_open_drop_rate"] * weights.circuit_open_drop
            - w.summary["p95_latency_ms"] * weights.p95_latency_ms
            for w in self.windows
        )

    def success_rate(self, *, in_horizon: bool) -> float:
        summary = self.in_horizon if in_horizon else self.metrics
        if summary is None or not summary.total:
            return 0.0
        return summary.succeeded / summary.total

    def bystander_success_rate(self) -> float | None:
        """In-horizon by construction: the externality must be scored on the
        same population as J, not on the drained tail that completes after the
        agent's chain has gone quiet."""
        summary = self.bystander_in_horizon
        if summary is None:
            return None
        return summary.succeeded / summary.total if summary.total else 0.0


def run_episode(
    env,
    act: ActFn,
    *,
    seed: int | None = None,
    on_step: Callable[[Any, float, bool, dict], None] | None = None,
) -> EpisodeResult:
    """Drive one episode to termination and collect everything a caller needs.

    `on_step(obs, reward, done, info)` fires after each step, for callers that
    need per-step side effects — learning policies call `update()` through it.
    """
    obs, info = env.reset(seed=seed)
    result = EpisodeResult(seed=seed, total_reward=0.0)
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        result.total_reward += float(reward)
        result.windows.append(
            WindowRecord(
                step=step,
                reward=float(reward),
                summary=info["summary"],
                action_values=info.get("action", {}),
            )
        )
        if on_step is not None:
            on_step(obs, float(reward), terminated or truncated, info)
        step += 1
    result.metrics = env.metrics_summary()
    result.in_horizon = env.in_horizon_summary()
    result.bystander_in_horizon = env.bystander_in_horizon_summary()
    result.drained_summary = info.get("drained_summary")
    return result
