"""Training with Stable-Baselines3 (`msim.rl.train`).

Needs the `train` extra, which the dev environment does not install: without
it this whole file is skipped, and `make test-train` is what runs it.
"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("stable_baselines3", reason="needs the train extra: make test-train")

from stable_baselines3.common.env_checker import check_env  # noqa: E402

from msim.rl import RetryEnv  # noqa: E402
from msim.rl import evaluate as ev  # noqa: E402
from msim.rl import train as tr  # noqa: E402
from tests.rl_small import TINY_GRID, Small  # noqa: E402

TINY = Small(grid=TINY_GRID)  # knobs: max attempts (1, 3) and budget ratio (0.1, 1.0)


class TestWithStableBaselines(unittest.TestCase):
    def test_their_checker_accepts_the_environment(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(RetryEnv(Small()))
        self.assertEqual([str(w.message) for w in caught], [])

    def test_environments_in_processes_of_their_own(self):
        envs = tr.make_envs(TINY, n_envs=2, seed=5)
        try:
            self.assertEqual(envs.num_envs, 2)
            observations = envs.reset()
            self.assertEqual(observations.shape, (2, len(RetryEnv(TINY).feature_names)))
            self.assertFalse(np.array_equal(observations[0], observations[1]))  # seeded apart
            ends = []
            for _ in range(2 * 15):  # two episodes of fifteen steps
                _, _, dones, infos = envs.step(np.array([[1, 1], [0, 0]]))
                ends += [info for done, info in zip(dones, infos, strict=True) if done]
            self.assertEqual(len(ends), 4)
            for info in ends:
                # The horizon is a time limit: the learner is told so and given the last
                # observation, which is what makes it bootstrap instead of stopping there.
                self.assertTrue(info["TimeLimit.truncated"])
                self.assertEqual(info["terminal_observation"].shape, observations[0].shape)
                self.assertEqual(info["episode"]["l"], 15)
                self.assertIn("ep_success_rate", info["episode"])
        finally:
            envs.close()


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.out = Path(cls.folder.name)
        cls.sidecar = tr.train(TINY, cls.out, timesteps=500, n_envs=2, seed=3, verbose=0)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_it_trains_at_least_the_steps_asked_for_and_says_how_many(self):
        self.assertEqual(self.sidecar["timesteps_asked"], 500)
        self.assertEqual(self.sidecar["timesteps_trained"], 540)  # whole rollouts of 2 x 30 steps
        self.assertEqual(self.sidecar["episodes_trained"], 36)
        self.assertEqual(self.sidecar["ppo"]["n_steps"], 30)  # two episodes
        self.assertTrue((self.out / "model.zip").exists())
        self.assertEqual(json.loads((self.out / "model.json").read_text()), self.sidecar)

    def test_the_sidecar_says_what_the_model_was_trained_on(self):
        self.assertEqual(
            self.sidecar["knobs"],
            [
                {"name": "max_attempts", "values": [1.0, 3.0], "default": 1},
                {"name": "budget_ratio", "values": [0.1, 1.0], "default": 1},
            ],
        )
        self.assertEqual(self.sidecar["features"], list(RetryEnv(TINY).feature_names))
        self.assertEqual((self.sidecar["step_s"], self.sidecar["warmup_s"]), (1.0, 5.0))
        self.assertEqual(self.sidecar["scenario_settings"]["rps"], 120.0)
        for package in ("gymnasium", "stable_baselines3", "torch", "numpy", "python"):
            self.assertIn(package, self.sidecar["versions"])

    def test_training_is_logged(self):
        with (self.out / "progress.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 9)  # one per rollout
        self.assertEqual(rows[-1]["time/total_timesteps"], "540")
        self.assertGreater(float(rows[-1]["rollout/ep_success_rate"]), 0.5)
        self.assertEqual(len(list((self.out / "monitor").glob("*.monitor.csv"))), 2)

    def test_the_saved_model_is_a_policy_for_its_environment(self):
        env = RetryEnv(TINY)
        policy = tr.load_policy(self.out / "model.zip", env)
        observation, _ = env.reset(seed=1)
        action = policy(observation)
        self.assertTrue(env.action_space.contains(np.asarray(action)))
        self.assertEqual(
            policy(observation), action
        )  # greedy: the same observation, the same action
        (episode,) = ev.rollout(env, policy, [2000])
        self.assertGreater(episode["success_rate"], 0.5)

    def test_a_model_is_not_applied_to_another_grid_or_step(self):
        model = self.out / "model.zip"
        reordered = replace(TINY_GRID, budget_ratio=(1.0, 0.1))  # the same shapes, another meaning
        for other, what in (
            (RetryEnv(Small(grid=reordered)), "knobs"),
            (RetryEnv(TINY, step_s=0.5), "step_s"),
        ):
            with (
                self.subTest(what=what),
                self.assertRaisesRegex(ValueError, f"trained with {what}"),
            ):
                tr.load_policy(model, other)

    def test_evaluating_it_on_given_episodes(self):
        model = self.out / "model.zip"
        with self.assertRaisesRegex(ValueError, "does not name its scenario"):
            tr.evaluate_model(model, [2000])
        alone = tr.evaluate_model(model, [2000, 2001, 2002], scenario=TINY)
        self.assertEqual([e["seed"] for e in alone], [2000.0, 2001.0, 2002.0])
        together = tr.evaluate_model(model, [2000, 2001, 2002], scenario=TINY, jobs=2)
        np.testing.assert_equal(together, alone)


class TestTheCommandLine(unittest.TestCase):
    def test_it_trains_compares_and_writes_everything(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            mock.patch.object(ev, "scenario_from", return_value=TINY),
        ):
            out = io.StringIO()
            arguments = ["tiny", "--out", folder, "--timesteps", "200", "--n-envs", "2"]
            arguments += ["--selection-seeds", "2", "--test-seeds", "3", "--jobs", "1"]
            with contextlib.redirect_stdout(out):
                tr.main(arguments)
            text = out.getvalue()
            self.assertIn("trained 240 steps (16 episodes)", text)
            self.assertIn("on 3 test episodes (best static: max_attempts=3, budget_ratio=1)", text)
            with (Path(folder) / "comparison.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["policy"] for row in rows], ["ppo", "best static", "default", "no retries"]
            )
            self.assertEqual(
                json.loads((Path(folder) / "model.json").read_text())["scenario"], "tiny"
            )
            for name in ("model.zip", "progress.csv", "episodes.csv", "static.csv"):
                self.assertTrue((Path(folder) / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
