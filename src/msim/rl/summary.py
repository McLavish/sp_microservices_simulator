"""How an episode is scored: one definition for every way of running it.

`RetryEnv` builds its episode summary from these pieces as it steps, and
`of_records` builds the same summary from the records of a run that had no
environment around it (a scenario file left as it is, say). Two policies are
therefore compared on the same quantities whichever way each was run.

The quantities, on the clients the reward is counted on unless said otherwise:

* `success_rate`: requests that succeeded, of all that finished after the
  warm-up (those still in flight at the horizon included, once they finish);
* `horizon_success_rate`: the same, of what finished inside the horizon, which
  is the population the rewards are counted on, and `return`, their sum;
* `fault_success_rate`: of the requests that STARTED inside a fault window;
* `amplification`: attempts per request, and `retry_efficiency`: the share of
  the retries that succeeded, both of the agent's clients and over the requests
  they finished after the warm-up;
* `recovery_s`: from the end of the last fault to the first window that is
  healthy again (`RECOVERED` of its finished requests succeeded);
* `on_time_rate`: requests that succeeded within the latency bound the reward
  is counted with, of all that finished after the warm-up. Without a bound it
  is `success_rate`. With one (`RetryEnv(reward_within_s=...)`,
  `of_records(within_s=...)`) a request served after it is a success and is
  not on time, and `return` adds up the on-time ones;
* `retries_served`: the retries of those requests that reached their target,
  that is, that no limiter refused and no breaker of the caller held back.
  With a price on them (`RetryEnv(retry_cost=...)`, `of_records(retry_cost=...)`)
  `return` is paid less by that price for each.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from msim.caller import ClientRuntime
from msim.core import (
    FAILURE_OUTCOMES,
    DropReason,
    RootOutcome,
    RootRequest,
    TimeDuration,
    TimePoint,
    classify_root,
    ns_to_s,
    root_completed_at,
    s_to_ns,
)

#: A window counts as recovered when this share of its finished requests succeeded.
RECOVERED = 0.95

#: A retry with one of these outcomes never reached its target.
NOT_SENT = (DropReason.REJECTED, DropReason.CIRCUIT_OPEN)

#: The ways a request can fail, as they are named in a summary (`failed_<name>`).
FAILED = tuple(outcome.name.lower() for outcome in FAILURE_OUTCOMES)

#: One window of an episode: when it began, the requests that finished in it, how many succeeded.
WindowCount = tuple[TimePoint, int, int]


@dataclass
class Tally:
    """Requests that finished, on the clients the reward is counted on."""

    requests: int = 0
    succeeded: int = 0
    #: Among the successes, those inside the latency bound (all of them without one).
    on_time: int = 0
    attempts: int = 0
    #: Retries that reached their target (not refused, not held back by a breaker).
    retries_served: int = 0
    #: Among them, the requests that started inside a fault window.
    fault_requests: int = 0
    fault_succeeded: int = 0
    failed: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FAILED, 0))

    def add(self, other: "Tally") -> None:
        self.requests += other.requests
        self.succeeded += other.succeeded
        self.on_time += other.on_time
        self.attempts += other.attempts
        self.retries_served += other.retries_served
        self.fault_requests += other.fault_requests
        self.fault_succeeded += other.fault_succeeded
        for name, count in other.failed.items():
            self.failed[name] += count

    def count(
        self,
        roots: Iterable[RootRequest],
        faults: Sequence[tuple[TimePoint, TimePoint]],
        within: TimeDuration | None = None,
    ):
        """Add finished requests. `faults` are the fault windows, in ns.

        `within` is the latency bound of `on_time`, in ns from a request's
        first attempt to its end; `None` counts every success as on time.
        """
        for root in roots:
            if not root.attempts:
                continue  # nothing was tried: `Metrics` skips these too
            outcome = classify_root(root)
            succeeded = outcome is RootOutcome.SUCCESS
            started = root.attempts[0].interval.begin
            self.requests += 1
            self.attempts += len(root.attempts)
            self.retries_served += sum(
                1 for attempt in root.attempts[1:] if attempt.drop_reason not in NOT_SENT
            )
            if succeeded:
                self.succeeded += 1
                if within is None or root_completed_at(root) - started <= within:
                    self.on_time += 1
            else:
                self.failed[outcome.name.lower()] += 1
            if any(begin <= started < end for begin, end in faults):
                self.fault_requests += 1
                self.fault_succeeded += succeeded


@dataclass
class Effort:
    """What a group of clients spent: the requests they finished and the attempts those took."""

    requests: int = 0
    attempts: int = 0
    retries: int = 0
    retry_ok: int = 0

    def count(self, roots: Iterable[RootRequest]) -> None:
        """Add finished requests, each with every attempt it made."""
        for root in roots:
            if not root.attempts:
                continue
            self.requests += 1
            self.attempts += len(root.attempts)
            self.retries += len(root.attempts) - 1
            self.retry_ok += sum(1 for attempt in root.attempts[1:] if attempt.success)


def paid(tally: Tally, retry_cost: float = 0.0) -> float:
    """What a window's reward counts: the requests served on time, less the price of its retries."""
    if not retry_cost:
        return tally.on_time  # the same number, to the last bit, as before there was a price
    return tally.on_time - retry_cost * tally.retries_served


def ratio(numerator: float, denominator: float) -> float:
    """A share, or nan when there is nothing to take a share of."""
    return numerator / denominator if denominator > 0 else math.nan


def recovery_s(windows: Sequence[WindowCount], faults: Sequence[tuple[TimePoint, TimePoint]]):
    """Seconds from the end of the last fault to the first window that is healthy again.

    The first window that begins at or after the fault's end and in which at
    least `RECOVERED` of the finished requests succeeded; 0 when that is the
    very next window. `inf` when no window qualifies, `nan` without a fault.
    """
    if not faults:
        return math.nan
    fault_end = max(end for _, end in faults)
    for begin, requests, succeeded in windows:
        if begin >= fault_end and requests and succeeded >= RECOVERED * requests:
            return ns_to_s(begin - fault_end)
    return math.inf


def summary(
    *,
    episode_return: float,
    in_horizon: Tally,
    tail: Tally,
    windows: Sequence[WindowCount],
    faults: Sequence[tuple[TimePoint, TimePoint]],
    effort: Effort,
) -> dict[str, float]:
    """The episode summary, by name. Every value is a float; some can be nan or inf."""
    total = Tally()
    total.add(in_horizon)
    total.add(tail)
    result = {
        "return": episode_return,
        "requests": float(total.requests),
        "success_rate": ratio(total.succeeded, total.requests),
        "on_time_rate": ratio(total.on_time, total.requests),
        "retries_served": float(total.retries_served),
        "horizon_success_rate": ratio(in_horizon.succeeded, in_horizon.requests),
        "fault_success_rate": ratio(total.fault_succeeded, total.fault_requests),
        "amplification": ratio(effort.attempts, effort.requests),
        "retry_efficiency": ratio(effort.retry_ok, effort.retries),
        "recovery_s": recovery_s(windows, faults),
    }
    for name, count in total.failed.items():
        result[f"failed_{name}"] = float(count)
    return result


def of_records(
    reward_clients: Sequence[ClientRuntime],
    *,
    expected_rps: float,
    fault_windows: Sequence[tuple[float, float]],
    horizon_s: float,
    warmup_s: float = 5.0,
    step_s: float = 1.0,
    effort_clients: Sequence[ClientRuntime] | None = None,
    within_s: float | None = None,
    retry_cost: float = 0.0,
) -> dict[str, float]:
    """The summary of a run that is over, from the records its clients kept.

    For a run made WITHOUT the environment: every workload driven, the
    simulation run until nothing is left (`Scenario.run()` of an imported
    file, an example script). The same cuts as the environment's: what
    finished during the warm-up is left out, a window `(t, t + step_s]` holds
    what finished in it, and what finished past the horizon is in the totals
    and in no window. `effort_clients` are the clients whose attempts and
    retries are reported (the reward clients unless given). `within_s` is the
    latency bound of the environment's `reward_within_s`: with it `return`
    adds up the requests that succeeded inside the bound. `retry_cost` is the
    environment's too: what `return` is paid less for every retry that reached
    its target.
    """
    faults = [(s_to_ns(begin), s_to_ns(end)) for begin, end in fault_windows]
    within = None if within_s is None else s_to_ns(within_s)
    warmup, horizon, step = s_to_ns(warmup_s), s_to_ns(horizon_s), s_to_ns(step_s)
    steps = (horizon - warmup) // step
    per_window: list[list[RootRequest]] = [[] for _ in range(steps)]
    late: list[RootRequest] = []
    for client in reward_clients:
        for root in client.roots:
            at = root_completed_at(root)
            if at <= warmup:
                continue
            index = -((warmup - at) // step) - 1  # ceil((at - warmup) / step) - 1
            (per_window[index] if index < steps else late).append(root)

    in_horizon, tail = Tally(), Tally()
    windows: list[WindowCount] = []
    on_time: list[float] = []  # what each window is paid for
    for index, roots in enumerate(per_window):
        window = Tally()
        window.count(roots, faults, within)
        in_horizon.add(window)
        windows.append((warmup + index * step, window.requests, window.succeeded))
        on_time.append(paid(window, retry_cost))
    tail.count(late, faults, within)

    effort = Effort()
    for client in reward_clients if effort_clients is None else effort_clients:
        effort.count(root for root in client.roots if root_completed_at(root) > warmup)
    # Added up window by window, the way the environment adds up its rewards
    # (`sum()` compensates its rounding and would differ in the last digit).
    episode_return = 0.0
    for amount in on_time:
        episode_return += amount / (expected_rps * step_s)
    return summary(
        episode_return=episode_return,
        in_horizon=in_horizon,
        tail=tail,
        windows=windows,
        faults=faults,
        effort=effort,
    )
