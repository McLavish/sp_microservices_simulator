"""Comparing policies on the same episodes (`msim.rl.evaluate`), and how an episode is scored."""

import contextlib
import csv
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from msim.core import s_to_ns
from msim.rl import FaultRanges, RetryEnv, RetryStorm, SingleService
from msim.rl import evaluate as ev
from msim.rl.summary import of_records
from tests.rl_small import TINY_GRID, Small

TINY = Small(grid=TINY_GRID)  # knobs: max attempts (1, 3) and budget ratio (0.1, 1.0)


class TestRollout(unittest.TestCase):
    def test_one_summary_per_seed_on_the_episode_asked_for(self):
        env = RetryEnv(TINY)
        episodes = ev.rollout(env, ev.constant([1, 1]), [7, 8, 7])
        self.assertEqual([e["seed"] for e in episodes], [7.0, 8.0, 7.0])
        self.assertEqual(episodes[0], episodes[2])  # the same episode twice
        self.assertNotEqual(episodes[0]["success_rate"], episodes[1]["success_rate"])
        for name in ("return", "success_rate", "fault_success_rate", "amplification", "recovery_s"):
            self.assertIn(name, episodes[0])

    def test_a_constant_policy_is_the_setting_held(self):
        env = RetryEnv(TINY)
        by_rollout = ev.rollout(env, ev.constant([0, 1]), [5])[0]
        env.reset(options={"episode_seed": 5})
        info: dict = {}
        for _ in range(env.episode_steps):
            *_, info = env.step([0, 1])
        self.assertEqual(by_rollout["return"], info["ep_return"])
        self.assertEqual(by_rollout["amplification"], 1.0)  # one attempt per request

    def test_a_policy_sees_the_observation(self):
        """Retry only while calls are failing: the server-failure share is in what it is shown."""
        env = RetryEnv(TINY)
        failing = env.feature_names.index("attempt_server_failure")
        seen = []

        def reactive(observation: np.ndarray) -> list[int]:
            seen.append(float(observation[failing]))
            return [1 if observation[failing] > 0.2 else 0, 1]

        reactive_run = ev.rollout(env, reactive, [5])[0]
        never = ev.rollout(env, ev.constant([0, 1]), [5])[0]
        self.assertGreater(max(seen), 0.3)
        self.assertGreater(reactive_run["fault_success_rate"], never["fault_success_rate"] + 0.2)


class TestFixedSettings(unittest.TestCase):
    def test_the_named_settings(self):
        env = RetryEnv(TINY)
        self.assertEqual(ev.settings(env), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(ev.default_action(env), (1, 1))
        self.assertEqual(ev.no_retry_action(env), (0, 1))
        self.assertEqual(len(ev.settings(RetryEnv(SingleService()))), 4 * 3 * 3 * 5)

    def test_every_setting_is_tried_on_the_same_episodes_and_ranked(self):
        search = ev.static_search(TINY, [1, 2])
        self.assertEqual(sorted(s.action for s in search), [(0, 0), (0, 1), (1, 0), (1, 1)])
        returns = [ev.mean(s.episodes, "return") for s in search]
        self.assertEqual(returns, sorted(returns, reverse=True))
        self.assertEqual({tuple(e["seed"] for e in s.episodes) for s in search}, {(1.0, 2.0)})
        # Half the calls fail for six seconds: retrying freely is the setting to hold.
        self.assertEqual(search[0].action, (1, 1))
        self.assertEqual(search[0].setting, {"max_attempts": 3.0, "budget_ratio": 1.0})

    def test_several_processes_find_what_one_finds(self):
        alone = ev.static_search(TINY, [1, 2])
        together = ev.static_search(TINY, [1, 2], jobs=2)

        def plain(search):  # compared so that nan equals nan
            return [(s.action, s.setting, s.episodes) for s in search]

        np.testing.assert_equal(plain(together), plain(alone))

    def test_the_search_can_be_restricted_and_takes_the_environments_arguments(self):
        search = ev.static_search(TINY, [1], actions=[(0, 0), (1, 1)], step_s=0.5)
        self.assertEqual(sorted(s.action for s in search), [(0, 0), (1, 1)])

    def test_the_best_setting_is_chosen_on_other_episodes_than_it_is_reported_on(self):
        runs, search = ev.baselines(TINY, selection_seeds=[1, 2], test_seeds=[11, 12, 13])
        self.assertEqual(list(runs), ["best static", "default", "no retries"])
        for episodes in runs.values():
            self.assertEqual([e["seed"] for e in episodes], [11.0, 12.0, 13.0])
        self.assertEqual({e["seed"] for s in search for e in s.episodes}, {1.0, 2.0})
        # Here the best fixed setting happens to be the default, so the two runs coincide.
        self.assertEqual(search[0].action, (1, 1))
        self.assertEqual(runs["best static"], runs["default"])
        self.assertLess(
            ev.mean(runs["no retries"], "success_rate"), ev.mean(runs["default"], "success_rate")
        )


class TestCompare(unittest.TestCase):
    def _episode(self, seed, success, recovery=0.0):
        return {
            "seed": float(seed),
            "return": 10 * success,
            "success_rate": success,
            "fault_success_rate": success / 2,
            "amplification": 1.5,
            "recovery_s": recovery,
        }

    def test_means_errors_and_the_paired_difference(self):
        runs = {
            "ppo": [self._episode(1, 0.90), self._episode(2, 0.95, math.inf)],
            "best static": [self._episode(1, 0.80), self._episode(2, 0.90, 4.0)],
        }
        ppo, static = ev.compare(runs, reference="best static")
        self.assertEqual((ppo["policy"], ppo["episodes"]), ("ppo", 2))
        self.assertAlmostEqual(ppo["success_rate"], 0.925)
        self.assertAlmostEqual(ppo["success_rate_se"], 0.025)
        self.assertAlmostEqual(ppo["vs_reference"], 0.075)  # (0.10 + 0.05) / 2, episode by episode
        self.assertAlmostEqual(ppo["vs_reference_se"], 0.025)
        self.assertEqual((ppo["recovered"], ppo["recovery_s"]), (0.5, 0.0))  # one never recovered
        self.assertEqual((static["recovered"], static["recovery_s"]), (1.0, 2.0))
        self.assertEqual((static["vs_reference"], static["vs_reference_se"]), (0.0, 0.0))
        text = ev.table([ppo, static], reference="best static")
        self.assertIn(" 92.50 ± 2.50", text)
        self.assertIn("  7.50 ± 2.50", text)

    def test_episodes_without_a_fault_are_left_out_of_what_needs_one(self):
        healthy = self._episode(1, 1.0, math.nan) | {"fault_success_rate": math.nan}
        (row,) = ev.compare({"a": [healthy, self._episode(2, 0.9, 3.0)]}, reference="a")
        self.assertEqual(row["fault_success_rate"], 0.45)
        self.assertEqual((row["recovered"], row["recovery_s"]), (1.0, 3.0))


class TestTheRecordsGiveTheSameSummary(unittest.TestCase):
    """`summary.of_records` on a run made without the environment, against the environment's."""

    def test_on_the_small_scenario(self):
        for scenario in (Small(), Small(fault=(8.0, 20.0), p_fail=0.9)):
            with self.subTest(fault=scenario.fault):
                env = RetryEnv(scenario)
                (by_env,) = ev.rollout(env, ev.constant([2, 1, 1, 2]), [21])

                episode = scenario.build(21)  # the same episode, left alone
                episode.sim.run(until=s_to_ns(scenario.horizon_s))
                episode.sim.run()
                by_records = of_records(
                    episode.reward_clients,
                    expected_rps=episode.expected_rps,
                    fault_windows=episode.fault_windows,
                    horizon_s=scenario.horizon_s,
                )
                del by_env["seed"]
                np.testing.assert_equal(by_records, by_env)  # every entry, nan for nan

    def test_on_the_retry_storm(self):
        failures = FaultRanges(kinds=("failure",), onset_s=(8.0, 10.0), length_s=(6.0, 8.0))
        scenario = RetryStorm(horizon_s=25.0, faults=failures)
        env = RetryEnv(scenario, step_s=0.5)
        (by_env,) = ev.rollout(env, ev.constant([2, 1, 1, 2]), [3])
        episode = scenario.build(3)
        episode.sim.run()
        by_records = of_records(
            episode.reward_clients,
            expected_rps=episode.expected_rps,
            fault_windows=episode.fault_windows,
            horizon_s=scenario.horizon_s,
            step_s=0.5,
            effort_clients=episode.agent.clients,
        )
        del by_env["seed"]
        np.testing.assert_equal(by_records, by_env)
        self.assertGreater(by_env["amplification"], 1.05)  # of the two callers of C


class TestTheCommandLine(unittest.TestCase):
    def test_it_prints_the_comparison_and_writes_the_tables(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            mock.patch.object(ev, "scenario_from", return_value=TINY),
        ):
            out = io.StringIO()
            arguments = ["tiny", "--selection-seeds", "2", "--test-seeds", "3", "--out", folder]
            with contextlib.redirect_stdout(out):
                ev.main(arguments)
            text = out.getvalue()
            self.assertIn("4 fixed settings on 2 selection episodes", text)
            self.assertIn("on 3 test episodes (best static: max_attempts=3, budget_ratio=1)", text)
            for name in ("best static", "default", "no retries"):
                self.assertIn(name, text)
            with (Path(folder) / "comparison.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["policy"] for row in rows], ["best static", "default", "no retries"]
            )
            self.assertEqual(float(rows[0]["vs_reference"]), 0.0)
            self.assertLess(float(rows[2]["vs_reference"]), -0.05)
            with (Path(folder) / "static.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            with (Path(folder) / "episodes.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 9)

    def test_scenarios_by_name(self):
        self.assertEqual(ev.scenario_from("single"), SingleService())
        self.assertEqual(ev.scenario_from("storm"), RetryStorm())
        with self.assertRaisesRegex(ValueError, "a scenario is single, storm or arolla"):
            ev.scenario_from("hotel")


if __name__ == "__main__":
    unittest.main()
