"""The Gymnasium environment: its contract, and that it leaves the simulation alone."""

import hashlib
import math
import pickle
import unittest
import warnings
from dataclasses import replace

import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from msim.caller import ClientRuntime
from msim.core import RootOutcome, classify_root, ms_to_ns, root_completed_at, s_to_ns
from msim.metrics import Metrics
from msim.retry_policies import FixedBackoffRetryPolicy, RetryBudgetPolicy, policy_layers
from msim.rl import FEATURES, RetryEnv, RetryGrid
from msim.simulator import Simulator
from tests.rl_small import GRID, Small


def _digest(sim: Simulator, clients: list[ClientRuntime]) -> tuple[str, int, str]:
    """Every record of a finished run, the events it scheduled, and where its RNG stands."""
    digest = hashlib.sha256()
    for client in clients:
        for root in client.roots:
            attempts = [
                (
                    a.interval.begin,
                    a.interval.end,
                    a.deadline,
                    a.success,
                    a.drop_reason.name,
                    a.queue_size_at_end,
                    a.service_time_ns,
                )
                for a in root.attempts
            ]
            digest.update(repr((root.cancelled, root.completed_at, attempts)).encode())
        digest.update(repr(client.attempts_total).encode())
    rng = hashlib.sha256(repr(sim.rng().getstate()).encode()).hexdigest()
    return digest.hexdigest(), sim._seq, rng


def _play(env: RetryEnv, actions, **reset):
    """Run one episode; `actions` is one action per step, or one action held throughout."""
    observation, info = env.reset(**reset)
    steps = [(observation, None, None, None, info)]
    held = not isinstance(actions[0], list)
    for index in range(env.episode_steps):
        steps.append(env.step(actions if held else actions[index]))
    return steps


def _default_action(env: RetryEnv) -> list[int]:
    return [knob.default for knob in env.scenario.knobs if len(knob.values) > 1]


class TestTheGymnasiumContract(unittest.TestCase):
    def test_it_passes_gymnasiums_own_checker_without_a_warning(self):
        env = RetryEnv(Small())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env, skip_render_check=True)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_spaces_follow_the_scenarios_knobs(self):
        env = RetryEnv(Small())
        self.assertEqual(
            env.knob_names, ("max_attempts", "attempt_timeout_ms", "backoff_ms", "budget_ratio")
        )
        self.assertEqual(env.knob_levels, (4, 3, 3, 5))
        self.assertEqual(env.action_space, spaces.MultiDiscrete([4, 3, 3, 5]))
        self.assertEqual(env.feature_names[: len(FEATURES)], FEATURES)
        self.assertEqual(env.observation_space.shape, (len(FEATURES) + 4,))
        self.assertEqual(env.observation_space.dtype, np.float32)
        self.assertEqual(env.episode_steps, 15)

    def test_a_knob_with_one_level_is_fixed_and_not_in_the_action(self):
        grid = replace(GRID, backoff_ms=(10.0,), budget_ratio=None)
        env = RetryEnv(Small(grid=grid))
        self.assertEqual(env.knob_names, ("max_attempts", "attempt_timeout_ms"))
        self.assertEqual(env.setting([0, 2]), {"max_attempts": 1.0, "attempt_timeout_ms": 80.0})
        env.reset(seed=1)
        env.step([0, 2])
        layers = list(policy_layers(env.episode.agent.clients[0].cfg.retry))
        self.assertFalse(any(isinstance(layer, RetryBudgetPolicy) for layer in layers))
        backoff = next(layer for layer in layers if isinstance(layer, FixedBackoffRetryPolicy))
        self.assertEqual((backoff.max_attempts, backoff.delay), (1, ms_to_ns(10)))

    def test_the_time_limit_is_a_truncation_never_a_termination(self):
        env = RetryEnv(Small())
        steps = _play(env, _default_action(env), seed=1)[1:]
        self.assertEqual([terminated for _, _, terminated, _, _ in steps], [False] * 15)
        self.assertEqual([truncated for _, _, _, truncated, _ in steps], [False] * 14 + [True])
        for _, _, _, _, info in steps[:-1]:
            self.assertFalse([key for key in info if key.startswith("ep_")])
        self.assertIn("ep_success_rate", steps[-1][4])
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(_default_action(env))
        env.reset()
        env.step(_default_action(env))  # and a new episode steps again

    def test_stepping_before_the_first_reset_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "reset"):
            RetryEnv(Small()).step([0, 0, 0, 0])

    def test_an_action_outside_the_levels_is_refused(self):
        env = RetryEnv(Small())
        env.reset(seed=1)
        for bad in ([4, 0, 0, 0], [0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0, 0]):
            with self.subTest(action=bad), self.assertRaisesRegex(ValueError, "level index"):
                env.step(bad)

    def test_the_horizon_must_be_a_whole_number_of_steps(self):
        with self.assertRaisesRegex(ValueError, "whole number of steps"):
            RetryEnv(Small(), step_s=2.0)  # 15 s after the warm-up
        with self.assertRaisesRegex(ValueError, "whole number of steps"):
            RetryEnv(Small(horizon_s=5.0))  # nothing after the warm-up
        self.assertEqual(RetryEnv(Small(), step_s=0.5).episode_steps, 30)
        self.assertEqual(RetryEnv(Small(), warmup_s=0.0, step_s=2.0).episode_steps, 10)

    def test_observations_and_infos_are_new_objects_every_time(self):
        env = RetryEnv(Small())
        first, first_info = env.reset(seed=1)
        kept = first.copy()
        second, _, _, _, second_info = env.step(_default_action(env))
        self.assertIsNot(first, second)
        self.assertIsNot(first_info, second_info)
        second[:] = -1.0
        np.testing.assert_array_equal(first, kept)

    def test_the_scenario_pickles(self):
        scenario = Small(rps=77.0)
        self.assertEqual(pickle.loads(pickle.dumps(scenario)), scenario)


class TestSeeding(unittest.TestCase):
    def test_the_same_seed_and_actions_give_the_same_trajectory(self):
        actions = [list(a) for a in np.random.default_rng(5).integers(0, [4, 3, 3, 5], (15, 4))]
        one = _play(RetryEnv(Small()), actions, seed=7)
        two = _play(RetryEnv(Small()), actions, seed=7)
        for (obs_a, *rest_a), (obs_b, *rest_b) in zip(one, two, strict=True):
            np.testing.assert_array_equal(obs_a, obs_b)
            np.testing.assert_equal(rest_a, rest_b)  # nan == nan here

    def test_every_reset_is_a_new_episode_and_a_seed_replays_the_sequence(self):
        env = RetryEnv(Small())
        seeds = [env.reset(seed=3)[1]["episode_seed"]]
        seeds += [env.reset()[1]["episode_seed"] for _ in range(3)]
        self.assertEqual(len(set(seeds)), 4)
        again = RetryEnv(Small())
        replay = [again.reset(seed=3)[1]["episode_seed"]]
        replay += [again.reset()[1]["episode_seed"] for _ in range(3)]
        self.assertEqual(replay, seeds)

    def test_two_environments_seeded_apart_see_different_episodes(self):
        seeds = {RetryEnv(Small()).reset(seed=seed)[1]["episode_seed"] for seed in range(8)}
        self.assertEqual(len(seeds), 8)

    def test_an_episode_can_be_asked_for_by_its_seed(self):
        env = RetryEnv(Small())
        first, info = env.reset(seed=1, options={"episode_seed": 123})
        self.assertEqual(info["episode_seed"], 123)
        other, _ = env.reset()
        again, _ = env.reset(options={"episode_seed": 123})
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))


class TestItLeavesTheSimulationAlone(unittest.TestCase):
    """The windows, the bookkeeping and the layers that count change no event."""

    def _plain(self, scenario: Small, seed: int):
        """The scenario's episode run with no environment: to the horizon, then drained."""
        episode = scenario.build(seed)
        episode.sim.run(until=s_to_ns(scenario.horizon_s))
        for workload in episode.workloads:
            workload.stop()
        episode.sim.run()
        return episode

    def test_an_episode_under_the_default_setting_is_the_plain_run(self):
        scenario = Small()
        env = RetryEnv(scenario, keep_records=True)
        _play(env, _default_action(env), options={"episode_seed": 11})
        plain = self._plain(scenario, 11)
        self.assertEqual(
            _digest(env.episode.sim, env.episode.clients), _digest(plain.sim, plain.clients)
        )
        self.assertGreater(len(plain.clients[0].roots), 2000)

    def test_the_counting_layers_change_no_event(self):
        """The same wiring with the plain policies the settings act on: the same run."""
        scenario = Small()
        tapped = self._plain(scenario, 11)
        sim, _, client, workload = scenario.parts(11)
        defaults = scenario.grid.defaults()
        client.cfg = replace(
            client.cfg,
            retry=RetryBudgetPolicy(
                budget_ratio=defaults["budget_ratio"],
                max_retries=scenario.grid.budget_max_retries,
                underlying_strategy=FixedBackoffRetryPolicy(
                    max_attempts=int(defaults["max_attempts"]),
                    delay=ms_to_ns(defaults["backoff_ms"]),
                ),
            ),
        )
        workload.drive(sim, client.start_request)
        sim.run()
        self.assertEqual(_digest(tapped.sim, tapped.clients), _digest(sim, [client]))

    def test_dropping_the_records_it_has_read_changes_nothing(self):
        actions = [list(a) for a in np.random.default_rng(9).integers(0, [4, 3, 3, 5], (15, 4))]
        kept = _play(RetryEnv(Small(), keep_records=True), actions, seed=4)
        dropped_env = RetryEnv(Small())
        dropped = _play(dropped_env, actions, seed=4)
        for (obs_a, *rest_a), (obs_b, *rest_b) in zip(kept, dropped, strict=True):
            np.testing.assert_array_equal(obs_a, obs_b)
            np.testing.assert_equal(rest_a, rest_b)
        self.assertEqual(dropped_env.episode.clients[0].roots, [])


class TestRewardAndSummary(unittest.TestCase):
    def _episode(self, scenario: Small, action=None, **kwargs):
        env = RetryEnv(scenario, keep_records=True, **kwargs)
        steps = _play(env, action or _default_action(env), options={"episode_seed": 21})
        return env, steps

    def test_the_rewards_are_the_requests_that_succeeded_in_time_counted(self):
        scenario = Small()
        env, steps = self._episode(scenario)
        client = env.episode.clients[0]
        begin, end = s_to_ns(env.warmup_s), s_to_ns(scenario.horizon_s)
        in_time = [r for r in client.roots if begin < root_completed_at(r) <= end]
        succeeded = sum(1 for r in in_time if classify_root(r) is RootOutcome.SUCCESS)
        rewards = [reward for _, reward, _, _, _ in steps[1:]]
        self.assertAlmostEqual(sum(rewards) * scenario.rps * env.step_s, succeeded, places=6)
        self.assertEqual(sum(info["succeeded"] for *_, info in steps[1:]), succeeded)
        self.assertEqual(sum(info["requests"] for *_, info in steps[1:]), len(in_time))
        self.assertAlmostEqual(steps[-1][4]["ep_return"], sum(rewards))
        # A healthy window pays about 1: goodput over the arrivals expected.
        self.assertAlmostEqual(float(np.mean(rewards[:2])), 1.0, delta=0.15)

    def test_the_summary_is_the_metrics_of_everything_after_the_warm_up(self):
        scenario = Small()
        env, steps = self._episode(scenario)
        client = env.episode.clients[0]
        after = [r for r in client.roots if root_completed_at(r) > s_to_ns(env.warmup_s)]
        summary = Metrics(roots=after).summary()
        info = steps[-1][4]
        self.assertEqual(info["ep_requests"], summary.total)
        self.assertAlmostEqual(info["ep_success_rate"], summary.succeeded / summary.total)
        self.assertEqual(info["ep_failed_server_failure"], summary.dropped_server_failure)
        self.assertEqual(info["ep_failed_deadline"], summary.dropped_deadline)
        # Attempts per request and what retrying bought, read off the same records.
        retries = [a for r in after for a in r.attempts[1:]]
        self.assertAlmostEqual(
            info["ep_amplification"], summary.attempts_total / summary.total, places=2
        )
        self.assertAlmostEqual(
            info["ep_retry_efficiency"], sum(a.success for a in retries) / len(retries), places=2
        )
        # Requests that STARTED in the fault window, wherever they finished.
        fault = [r for r in after if s_to_ns(8) <= r.attempts[0].interval.begin < s_to_ns(14)]
        ok = sum(1 for r in fault if classify_root(r) is RootOutcome.SUCCESS)
        self.assertAlmostEqual(info["ep_fault_success_rate"], ok / len(fault))
        self.assertLess(info["ep_fault_success_rate"], info["ep_success_rate"])

    def test_requests_still_in_flight_at_the_horizon_are_in_the_summary_not_in_a_reward(self):
        # A fault to the very end and long retries: work is left over at the horizon.
        scenario = Small(fault=(8.0, 20.0), p_fail=0.9)
        slow = [3, 2, 2, 4]  # 4 attempts, 80 ms timeout, 50 ms backoff, budget 1.0
        env, steps = self._episode(scenario, slow)
        info = steps[-1][4]
        in_time = sum(i["requests"] for *_, i in steps[1:])
        self.assertGreater(info["ep_requests"], in_time)
        undrained, steps = self._episode(scenario, slow, drain=False)
        self.assertEqual(steps[-1][4]["ep_requests"], in_time)
        self.assertEqual(steps[-1][4]["ep_return"], info["ep_return"])

    def test_recovery_time(self):
        _, steps = self._episode(Small())
        self.assertEqual(steps[-1][4]["ep_recovery_s"], 0.0)  # failures stop with the fault
        _, steps = self._episode(Small(fault=None))
        self.assertTrue(math.isnan(steps[-1][4]["ep_recovery_s"]))
        self.assertTrue(math.isnan(steps[-1][4]["ep_fault_success_rate"]))
        _, steps = self._episode(Small(fault=(8.0, 20.0), p_fail=1.0))
        self.assertEqual(steps[-1][4]["ep_recovery_s"], math.inf)  # the fault outlasts the episode


class TestWhatTheAgentSees(unittest.TestCase):
    def _named(self, env: RetryEnv, observation: np.ndarray) -> dict[str, float]:
        return dict(zip(env.feature_names, observation.tolist(), strict=True))

    def test_a_healthy_window_and_a_failing_one(self):
        env = RetryEnv(Small())
        steps = _play(env, _default_action(env), options={"episode_seed": 21})
        healthy = self._named(env, steps[2][0])  # 6 s to 7 s
        self.assertEqual(healthy["has_data"], 1.0)
        self.assertAlmostEqual(healthy["request_rate"], 0.5, delta=0.1)  # the expected rate
        self.assertGreater(healthy["attempt_success"], 0.97)
        self.assertEqual(healthy["attempt_server_failure"], 0.0)
        self.assertEqual(healthy["retry_share"], 0.0)
        self.assertAlmostEqual(healthy["latency_mean"], 11.3 / 80, delta=0.05)  # of 80 ms
        self.assertEqual(healthy["budget_balance"], 1.0)
        self.assertAlmostEqual(healthy["knob:max_attempts"], 2 / 3)
        self.assertEqual(healthy["knob:attempt_timeout_ms"], 0.5)

        failing = self._named(env, steps[6][0])  # 10 s to 11 s, half the calls fail
        self.assertAlmostEqual(failing["attempt_server_failure"], 0.5, delta=0.12)
        self.assertGreater(failing["retry_share"], 0.1)
        self.assertAlmostEqual(failing["retry_success"], 0.5, delta=0.25)
        self.assertLess(failing["budget_balance"], 1.0)
        shares = ("success", "deadline", "queue_full", "server_failure", "other")
        self.assertAlmostEqual(sum(failing[f"attempt_{name}"] for name in shares), 1.0, places=5)

    def test_it_stays_inside_its_space_under_a_hopeless_overload(self):
        # Twelve times what the station can serve, every call failing, random settings.
        env = RetryEnv(Small(rps=4000.0, p_fail=1.0, fault=(6.0, 20.0)))
        rng = np.random.default_rng(1)
        observation, _ = env.reset(seed=2)
        seen = [observation]
        for _ in range(env.episode_steps):
            seen.append(env.step(rng.integers(0, env.knob_levels))[0])
        for observation in seen:
            self.assertTrue(env.observation_space.contains(observation), observation)
        self.assertEqual(max(float(o.max()) for o in seen), 1.0)  # something did hit the bound
        self.assertLess(self._named(env, seen[-1])["attempt_success"], 0.01)

    def test_a_slow_copy_of_what_it_measures_when_asked_for(self):
        """`slow_s`: every measured entry again, averaged exponentially over about that long."""
        plain, env = RetryEnv(Small()), RetryEnv(Small(), slow_s=4.0)
        n = len(FEATURES)
        self.assertEqual(plain.feature_names[:n], FEATURES)  # unchanged without it
        self.assertEqual(
            env.feature_names,
            FEATURES
            + tuple(f"{name}~4s" for name in FEATURES)
            + tuple(f"knob:{name}" for name in env.knob_names),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_env(env, skip_render_check=True)

        action = _default_action(env)
        first, _ = env.reset(options={"episode_seed": 21})
        np.testing.assert_array_equal(first[n : 2 * n], first[:n])  # it starts at the warm-up
        # The same episode without it: the last-window entries are the same numbers.
        (start, *_), *steps = _play(plain, action, options={"episode_seed": 21})
        np.testing.assert_array_equal(first[:n], start[:n])
        slow, weight = first[:n].astype(np.float64), 1.0 - math.exp(-1.0 / 4.0)
        for expected, *_ in steps:
            observation, *_ = env.step(action)
            np.testing.assert_array_equal(observation[:n], expected[:n])
            np.testing.assert_array_equal(observation[2 * n :], expected[n:])  # the knobs
            slow = slow + weight * (expected[:n].astype(np.float64) - slow)
            np.testing.assert_allclose(observation[n : 2 * n], slow, atol=1e-5)
            self.assertTrue(env.observation_space.contains(observation))
        named = self._named(env, observation)
        # The failure is over at the end; the slow share of failures has not forgotten it.
        self.assertEqual(named["attempt_server_failure"], 0.0)
        self.assertGreater(named["attempt_server_failure~4s"], 0.0)

    def test_a_slow_copy_needs_a_time_constant(self):
        for bad in (0.0, -1.0, math.inf):
            with self.subTest(slow_s=bad), self.assertRaisesRegex(ValueError, "slow_s"):
                RetryEnv(Small(), slow_s=bad)

    def test_an_idle_window_says_so(self):
        env = RetryEnv(Small(rps=0.001))
        observation, _ = env.reset(seed=1)
        named = self._named(env, observation)
        self.assertEqual((named["has_data"], named["attempt_success"]), (0.0, 0.0))

    def test_info_carries_what_the_agent_is_not_shown(self):
        env = RetryEnv(Small())
        env.reset(seed=1)
        _, _, _, _, info = env.step([1, 0, 2, 3])
        self.assertEqual(
            {key: info[key] for key in info if key.startswith("knob_")},
            {
                "knob_max_attempts": 2.0,
                "knob_attempt_timeout_ms": 20.0,
                "knob_backoff_ms": 50.0,
                "knob_budget_ratio": 0.5,
            },
        )
        self.assertIn("queue_svc", info)
        self.assertIn("in_flight_svc", info)
        self.assertEqual(info["t"], 6.0)
        self.assertTrue(all(isinstance(v, int | float) for v in info.values()), info)


class TestTheSettingsReachTheClients(unittest.TestCase):
    def test_every_knob_is_put_on_every_agent_client(self):
        env = RetryEnv(Small())
        env.reset(seed=1)
        env.step([3, 2, 0, 4])
        client = env.episode.agent.clients[0]
        layers = list(policy_layers(client.cfg.retry))
        backoff = next(layer for layer in layers if isinstance(layer, FixedBackoffRetryPolicy))
        budget = next(layer for layer in layers if isinstance(layer, RetryBudgetPolicy))
        self.assertEqual((backoff.max_attempts, backoff.delay), (4, 0))
        self.assertEqual(client.cfg.timeout_controller.get_attempt_timeout(), ms_to_ns(80))
        self.assertEqual(client.cfg.timeout_controller.get_global_timeout(), ms_to_ns(400))
        self.assertEqual((budget.budget_ratio, budget._deposit, budget._retry_cost), (1.0, 1, 1))

    def test_more_attempts_recover_more_of_a_failing_station(self):
        returns = {}
        for attempts in (0, 3):  # one attempt, four attempts; budget 1.0 either way
            env = RetryEnv(Small())
            steps = _play(env, [attempts, 1, 1, 4], options={"episode_seed": 5})
            returns[attempts] = steps[-1][4]["ep_fault_success_rate"]
        self.assertAlmostEqual(returns[0], 0.5, delta=0.05)
        self.assertGreater(returns[3], 0.85)

    def test_an_unknown_setting_is_refused(self):
        env = RetryEnv(Small())
        env.reset(seed=1)
        with self.assertRaisesRegex(KeyError, "no retry setting named"):
            env.episode.agent.apply({"hedging": 1.0})

    def test_a_grid_checks_its_levels_and_defaults(self):
        for bad, message in (
            (lambda: RetryGrid(default_max_attempts=7), "not one of its levels"),
            (
                lambda: RetryGrid(max_attempts=(0, 1), default_max_attempts=1),
                "whole numbers >= 1",
            ),
            (lambda: RetryGrid(budget_ratio=(0.0, 0.2)), r"in \(0, 1\]"),
            (lambda: RetryGrid(backoff_ms=(10.0, 10.0)), "lists a value twice"),
            (lambda: RetryGrid(attempt_timeout_ms=(0.0, 100.0)), "must be > 0"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                bad()


if __name__ == "__main__":
    unittest.main()
