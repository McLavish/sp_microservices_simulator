"""The simulator as a Gymnasium environment: an agent that tunes retries.

One step is one window of simulated time. The agent picks a level of every
setting the scenario declares; the settings are applied, the simulation runs
`step_s` further, and the agent is shown what its own clients measured in that
window and is paid the scenario's goodput: the requests that succeeded in it,
over the number expected to arrive.

    env = RetryEnv(SingleService())
    observation, info = env.reset(seed=1)
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

The environment drives the simulation from outside, with `Simulator.run(until=)`,
and changes nothing in it but the settings: a run paused at every window is, event
for event, the run left alone.

An episode never terminates; it is cut at the scenario's horizon (`truncated`),
so a learner bootstraps over the cut and gains nothing by pushing failures past
it. After the cut the arrivals are stopped and the requests still in flight are
left to finish; they count in the episode summary that the last `info` carries
(the `ep_*` keys), never in a reward.
"""

import math
from typing import Any

import numpy as np

from msim.caller import ClientRuntime
from msim.core import RootRequest, TimePoint, ns_to_s, s_to_ns
from msim.rl.scenario import Episode, Knob, RetryKnobs, Scenario
from msim.rl.summary import Effort, Tally, WindowCount, summary

try:
    import gymnasium
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError("msim.rl needs Gymnasium: install microservices-simulator[rl]") from error

#: What the agent is shown of its clients, in this order, each in [0, 1]. The
#: level of every knob follows (`knob:<name>`, 0 for the first level, 1 for the last).
FEATURES = (
    "has_data",  # 1 when an attempt ended in the window: tells "idle" from "all failed"
    "request_rate",  # requests started, over twice the rate expected when all is well
    "in_flight",  # requests not yet finished, over the worst the settings allow
    "attempt_success",  # share of the attempts that ended, by how they ended ...
    "attempt_deadline",
    "attempt_queue_full",
    "attempt_server_failure",
    "attempt_other",
    "retry_share",  # share of the attempts that were retries
    "retry_success",  # share of the retries that succeeded
    "latency_mean",  # of successful attempts, over the longest an attempt may take
    "latency_p95",
    "budget_balance",  # how full the retry budget is
)


class RetryEnv(gymnasium.Env):
    """An agent that sets the retry policy of some clients, window after window.

    `scenario` is a `msim.rl.scenario.Scenario`: it declares the knobs, gives the
    length of an episode and builds a fresh simulation for each one. The
    observation is described by `feature_names`; the action is one level index
    per knob that has more than one level (`knob_names`).

    `reset(seed=...)` seeds the environment; every reset then draws the seed of
    its episode, so a seeded environment replays the same sequence of episodes.
    `reset(options={"episode_seed": n})` asks for one episode in particular,
    which is how two policies are put on the same episodes.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario,
        *,
        step_s: float = 1.0,
        warmup_s: float = 5.0,
        drain: bool = True,
        keep_records: bool = False,
        slow_s: float | None = None,
        reward_within_s: float | None = None,
    ) -> None:
        if not math.isfinite(step_s) or step_s <= 0:
            raise ValueError(f"step_s must be > 0, got {step_s}")
        if not math.isfinite(warmup_s) or warmup_s < 0:
            raise ValueError(f"warmup_s must be >= 0, got {warmup_s}")
        if slow_s is not None and (not math.isfinite(slow_s) or slow_s <= 0):
            raise ValueError(f"slow_s must be > 0 or None, got {slow_s}")
        if reward_within_s is not None and (
            not math.isfinite(reward_within_s) or reward_within_s <= 0
        ):
            raise ValueError(f"reward_within_s must be > 0 or None, got {reward_within_s}")
        self.scenario = scenario
        self.step_s = step_s
        self.warmup_s = warmup_s
        #: `None`, or the time constant in seconds of a second copy of every
        #: measured entry, averaged exponentially over about that long
        #: (`<name>~<slow_s>s`, after the entries of the last window). One window
        #: is a noisy measurement: a second of 160 requests per second tells the
        #: rate to within 8%. An agent that has to know its load, or whether a
        #: failure is a blip, needs the longer look as well, and has no memory.
        self.slow_s = slow_s
        #: `None`, or a latency bound in seconds on what the reward counts: a
        #: request that succeeded later than this after its first attempt is a
        #: success in the summary and pays nothing. Without it a limiter is paid
        #: the same for a request served in 10 ms and for one that waited out a
        #: failure in a queue for three seconds, and learns to fill the queue.
        self.reward_within_s = reward_within_s
        self._within = None if reward_within_s is None else s_to_ns(reward_within_s)
        #: Whether the requests in flight at the horizon are run to their end for
        #: the summary. Off: the summary covers what finished inside the horizon.
        self.drain = drain
        #: Whether the clients keep the records of finished requests
        #: (`ClientRuntime.roots`), for `Metrics` or a CSV afterwards. Off, the
        #: environment drops what it has read, which makes a long chain episode
        #: about twice as fast and far smaller.
        self.keep_records = keep_records

        self._step_ns = s_to_ns(step_s)
        self._warmup_ns = s_to_ns(warmup_s)
        horizon_ns = s_to_ns(scenario.horizon_s)
        steps, rest = divmod(horizon_ns - self._warmup_ns, self._step_ns)
        if steps < 1 or rest:
            raise ValueError(
                f"the scenario's horizon ({scenario.horizon_s} s) minus the warm-up ({warmup_s} s) "
                f"must be a whole number of steps of {step_s} s, and at least one"
            )
        #: Steps in an episode.
        self.episode_steps = int(steps)

        knobs = scenario.knobs
        if len({knob.name for knob in knobs}) != len(knobs):
            raise ValueError(f"two knobs share a name: {[knob.name for knob in knobs]}")
        self._knobs: tuple[Knob, ...] = tuple(knob for knob in knobs if len(knob.values) > 1)
        if not self._knobs:
            raise ValueError(
                "the scenario leaves the agent nothing to choose: no knob has two levels"
            )
        #: The knobs of the action, in its order, and how many levels each has.
        self.knob_names = tuple(knob.name for knob in self._knobs)
        self.knob_levels = tuple(len(knob.values) for knob in self._knobs)
        #: What each entry of the observation is.
        # What is measured: `FEATURES`, at the agent's clients, unless the scenario
        # names entries of its own (`features`), which its agent then supplies
        # (`Agent.measure`): a limiter measures at the server, not at a client.
        own = getattr(scenario, "features", None)
        self._measured: tuple[str, ...] = FEATURES if own is None else tuple(own)
        if len(set(self._measured)) != len(self._measured) or not self._measured:
            raise ValueError(f"the scenario's entries need distinct names: {self._measured}")
        slow = () if slow_s is None else tuple(f"{name}~{slow_s:g}s" for name in self._measured)
        self.feature_names = (
            self._measured + slow + tuple(f"knob:{name}" for name in self.knob_names)
        )
        self._slow_weight = 0.0 if slow_s is None else 1.0 - math.exp(-step_s / slow_s)
        self._slow: np.ndarray | None = None  # the averaged copy, once there is a window
        self.action_space = spaces.MultiDiscrete(list(self.knob_levels))
        self.observation_space = spaces.Box(0.0, 1.0, (len(self.feature_names),), np.float32)

        self._episode: Episode | None = None
        self._over = True
        self._now: TimePoint = 0
        self._step_index = 0
        self._levels: list[int] = [knob.default for knob in self._knobs]
        self._cursors: list[int] = []
        self._faults: list[tuple[TimePoint, TimePoint]] = []
        self._started = 0  # requests the agent's clients started, since the build
        self._finished = 0  # ... and finished
        self._in_horizon = Tally()
        self._tail = Tally()
        self._effort = Effort()  # of the agent's clients over the episode, warm-up excluded
        self._windows: list[WindowCount] = []
        self._return = 0.0

    # ------------------------------------------------------------------ API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        episode_seed = (options or {}).get("episode_seed")
        if episode_seed is None:
            # Drawn at EVERY reset: vector environments seed only the first one,
            # and an episode seed that fell back to a constant would train on
            # one episode repeated.
            episode_seed = int(self.np_random.integers(2**31 - 1))
        episode_seed = int(episode_seed)

        episode = self._episode = self.scenario.build(episode_seed)
        self._over = False
        self._now = 0
        self._step_index = 0
        self._levels = [knob.default for knob in self._knobs]
        self._cursors = [0] * len(episode.clients)
        self._faults = [(s_to_ns(a), s_to_ns(b)) for a, b in episode.fault_windows]
        self._started = self._finished = 0
        self._in_horizon, self._tail, self._effort = Tally(), Tally(), Effort()
        self._windows = []
        self._return = 0.0
        self._slow = None

        episode.agent.apply(self._setting())
        # The first observation is of the warm-up, run under the default setting:
        # the system the agent takes over is one that is already serving.
        self._run(self._warmup_ns)
        self._collect(scored=False)
        observation = self._observe(self.warmup_s)
        return observation, {"episode_seed": episode_seed, **self._step_info(Tally())}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        episode = self._episode
        if episode is None or self._over:
            raise RuntimeError("the episode is over (or was never started): call reset() first")
        levels = np.asarray(action, dtype=np.int64).reshape(-1)
        if levels.shape != (len(self._knobs),) or not self.action_space.contains(levels):
            raise ValueError(
                f"an action is one level index per knob {self.knob_names}, "
                f"within {list(self.knob_levels)}; got {action!r}"
            )
        self._levels = [int(level) for level in levels]
        episode.agent.apply(self._setting())

        begin = self._now
        self._run(self._step_ns)
        tally = self._collect()
        self._in_horizon.add(tally)
        self._windows.append((begin, tally.requests, tally.succeeded))
        observation = self._observe(self.step_s)
        # `on_time` is `succeeded` unless a latency bound is set.
        reward = tally.on_time / (episode.expected_rps * self.step_s)
        self._return += reward
        info = self._step_info(tally)

        self._step_index += 1
        truncated = self._step_index >= self.episode_steps
        if truncated:
            self._over = True
            info.update(self._finish())
        return observation, float(reward), False, truncated, info

    def close(self) -> None:
        self._episode = None
        self._over = True

    @property
    def episode(self) -> Episode:
        """The simulation of the current episode, for a look inside (tests, analysis)."""
        if self._episode is None:
            raise RuntimeError("there is no episode yet: call reset() first")
        return self._episode

    def setting(self, action: Any) -> dict[str, float]:
        """What an action means: the value each knob is put to."""
        return {
            knob.name: knob.values[int(level)]
            for knob, level in zip(self._knobs, action, strict=True)
        }

    # ------------------------------------------------------------ internals

    def _setting(self) -> dict[str, float]:
        return {
            knob.name: knob.values[level]
            for knob, level in zip(self._knobs, self._levels, strict=True)
        }

    def _run(self, duration: int) -> None:
        assert self._episode is not None
        self._now += duration
        self._episode.sim.run(until=self._now)

    def _take(self, index: int, client: ClientRuntime) -> list[RootRequest]:
        """The requests `client` finished since they were last taken."""
        if self.keep_records:
            fresh = client.roots[self._cursors[index] :]
            self._cursors[index] = len(client.roots)
            return fresh
        # A new list rather than `clear()`: whoever holds the old one (a
        # `Metrics` made earlier) keeps what it had.
        fresh, client.roots = client.roots, []
        return fresh

    def _collect(self, *, scored: bool = True) -> Tally:
        """Read what finished since the last call: the reward's side of a window.

        `scored=False` for the warm-up: what the agent's clients spent on it is
        not part of the episode.
        """
        episode = self._episode
        assert episode is not None
        rewarded = {id(client) for client in episode.reward_clients}
        agents = {id(client) for client in episode.agent.clients}
        tally = Tally()
        for index, client in enumerate(episode.clients):
            fresh = self._take(index, client)
            if id(client) in agents:
                self._finished += len(fresh)
                if scored:
                    self._effort.count(fresh)
            if id(client) in rewarded:
                tally.count(fresh, self._faults, self._within)
        return tally

    def _observe(self, window_s: float) -> np.ndarray:
        """What the agent measured in the window just run; resets its counts."""
        episode = self._episode
        assert episode is not None
        if self._measured is FEATURES:
            values = self._measure_clients(episode, window_s)
        else:
            # The scenario's own entries, from wherever its agent sits.
            started, values = episode.agent.measure(window_s)  # ty: ignore[unresolved-attribute]
            self._started += started
            values = [float(value) for value in values]
            if len(values) != len(self._measured):
                raise ValueError(
                    f"the scenario names {len(self._measured)} entries, "
                    f"its agent measured {len(values)}"
                )
        if self.slow_s is not None:
            # Of the entries as the agent is shown them, clipped; the warm-up is
            # where the average starts.
            measured = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
            if self._slow is None:
                self._slow = measured
            else:
                self._slow = self._slow + self._slow_weight * (measured - self._slow)
            values += self._slow.tolist()
        values += [
            level / (len(knob.values) - 1)
            for knob, level in zip(self._knobs, self._levels, strict=True)
        ]
        # A new array every call, clipped: the declared bounds are the real ones.
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)

    def _measure_clients(self, episode: Episode, window_s: float) -> list[float]:
        """`FEATURES`: what the agent's clients measured in the window; resets the window."""
        agent = episode.agent
        assert isinstance(agent, RetryKnobs), "FEATURES are measured at an agent's clients"
        window = agent.window
        self._started += window.requests

        attempts = window.attempts
        latencies = window.latencies
        if latencies:
            taken = np.asarray(latencies, dtype=np.float64) / agent.latency_scale
            latency_mean, latency_p95 = float(taken.mean()), float(np.quantile(taken, 0.95))
        else:
            latency_mean = latency_p95 = 0.0
        share = (lambda n: n / attempts) if attempts else (lambda n: 0.0)
        expected = agent.agent_rps * window_s  # 0 for a warm-up of no length
        values = [
            1.0 if attempts else 0.0,
            window.requests / (2.0 * expected) if expected > 0 else 0.0,
            max(0, self._started - self._finished) / agent.in_flight_scale,
            share(window.ok),
            share(window.deadline),
            share(window.queue_full),
            share(window.server_failure),
            share(window.other),
            share(window.retries),
            window.retry_ok / window.retries if window.retries else 0.0,
            latency_mean,
            latency_p95,
            agent.budget_balance(),
        ]
        window.reset()
        return values

    def _step_info(self, tally: Tally) -> dict[str, Any]:
        episode = self._episode
        assert episode is not None
        info: dict[str, Any] = {
            "t": ns_to_s(self._now),
            "requests": tally.requests,
            "succeeded": tally.succeeded,
            "attempts": tally.attempts,
            "agent_in_flight": max(0, self._started - self._finished),
        }
        for name, count in tally.failed.items():
            info[f"failed_{name}"] = count
        for name, value in self._setting().items():
            info[f"knob_{name}"] = value
        # What the agent is NOT shown: the state of the stations.
        for name, service in episode.services.items():
            info[f"queue_{name}"] = service.queue_len()
            info[f"in_flight_{name}"] = service.in_flight
        return info

    def _finish(self) -> dict[str, float]:
        """The episode summary (`msim.rl.summary`), as `ep_*` keys.

        Past the horizon nothing is rewarded any more: what is still in flight
        is run to its end for the totals only.
        """
        episode = self._episode
        assert episode is not None
        if self.drain:
            for workload in episode.workloads:
                workload.stop()
            episode.sim.run()
            self._tail = self._collect()
        scored = summary(
            episode_return=self._return,
            in_horizon=self._in_horizon,
            tail=self._tail,
            windows=self._windows,
            faults=self._faults,
            effort=self._effort,
        )
        return {f"ep_{name}": value for name, value in scored.items()}
