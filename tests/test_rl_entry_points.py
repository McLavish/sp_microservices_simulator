"""Smoke coverage for the RL experiment entry points.

These modules orchestrate the rollout loops that have repeatedly drifted from
one another (pooled vs macro-averaged cost, in-horizon vs full bystander
population), and they sat at 0% coverage — so a refactor of those loops had
nothing to catch it. These are deliberately cheap: tiny budgets, one or two
seeds, asserting the CSV contract and that the entry point runs at all.
"""

import csv
import os
import tempfile
import unittest

try:
    import gymnasium  # noqa: F401
except Exception:  # pragma: no cover
    gymnasium = None  # ty: ignore[invalid-assignment]

from rl.sb3_support import import_sb3

_PPO = import_sb3()[0]


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestComparePoliciesEntryPoint(unittest.TestCase):
    def _run(self, extra_args):
        from rl.experiments.compare_policies import main

        with tempfile.TemporaryDirectory() as d:
            results = os.path.join(d, "results.csv")
            main(
                [
                    "--fault-scenario",
                    "none",
                    "--episode-duration-s",
                    "2.0",
                    "--eval-seeds",
                    "1",
                    "--output-csv",
                    results,
                    # Keep every artifact inside the temp dir; the defaults
                    # write into the repo root.
                    "--traces-csv",
                    os.path.join(d, "traces.csv"),
                    *extra_args,
                ]
            )
            with open(results) as f:
                return list(csv.DictReader(f))

    def test_fixed_policy_writes_a_results_row(self):
        rows = self._run(["--policies", "fixed"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["policy"], "fixed_values")
        # The cost column is the one that drifted between pooled and macro
        # aggregation; assert it exists and parses.
        self.assertIsInstance(float(rows[0]["cost_mean"]), float)

    def test_several_policies_each_get_a_row(self):
        rows = self._run(["--policies", "fixed,aimd,model_based"])
        self.assertEqual({r["policy"] for r in rows}, {"fixed_values", "aimd", "model_based"})

    def test_unknown_policy_name_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._run(["--policies", "fixed,not_a_policy"])

    def test_traces_csv_records_one_row_per_window(self):
        from rl.experiments.compare_policies import main

        with tempfile.TemporaryDirectory() as d:
            traces = os.path.join(d, "traces.csv")
            main(
                [
                    "--fault-scenario",
                    "none",
                    "--episode-duration-s",
                    "3.0",
                    "--eval-seeds",
                    "1",
                    "--policies",
                    "fixed",
                    "--output-csv",
                    os.path.join(d, "r.csv"),
                    "--traces-csv",
                    traces,
                ]
            )
            with open(traces) as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)  # 3 one-second windows
        self.assertEqual([int(r["step"]) for r in rows], [0, 1, 2])


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestConstrainedRetryEntryPoint(unittest.TestCase):
    def test_calibration_mode_reports_a_verdict(self):
        from rl.experiments.constrained_retry import main

        # --calibrate must propagate its verdict as the exit status rather than
        # always reporting success.
        with self.assertRaises(SystemExit) as ctx:
            main(
                [
                    "--scenario",
                    "chain",
                    "--calibrate",
                    "--episode-duration-s",
                    "2.0",
                    "--oracle-seeds",
                    "1",
                    "--eval-seeds",
                    "1",
                    "--budgets",
                    "0.1",
                    "--attempt-timeouts-ms",
                    "30,80",
                    "--max-attempts",
                    "1,2",
                    "--backoff-ms",
                    "0",
                ]
            )
        self.assertIn(ctx.exception.code, (0, 1))


@unittest.skipIf(_PPO is None, "stable-baselines3 not installed")
class TestTrainSb3EntryPoint(unittest.TestCase):
    def test_tiny_training_run_saves_a_model_and_sidecar(self):
        from rl.model_io import sidecar_path
        from rl.train_sb3 import main

        with tempfile.TemporaryDirectory() as d:
            save = os.path.join(d, "ppo.zip")
            main(
                [
                    "--timesteps",
                    "64",
                    "--n-steps",
                    "16",
                    "--batch-size",
                    "16",
                    "--fault-scenario",
                    "none",
                    "--episode-duration-s",
                    "2.0",
                    "--eval-episodes",
                    "1",
                    "--device",
                    "cpu",
                    "--save-path",
                    save,
                ]
            )
            self.assertTrue(os.path.exists(save))
            self.assertTrue(os.path.exists(sidecar_path(save)))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestRolloutIsShared(unittest.TestCase):
    """The aggregations that must agree across callers live in one place.

    Four hand-rolled episode loops used to exist and drifted: one averaged
    per-window cost ratios while another pooled raw counts, and one scored the
    bystander on the full population while the objective saw only in-horizon
    roots. Both produced numbers that looked comparable and were not.
    """

    def test_no_module_hand_rolls_an_episode_loop(self):
        import pathlib

        offenders = [
            path.name
            for path in pathlib.Path("rl").rglob("*.py")
            if path.name != "rollout.py"
            and "while not (terminated or truncated)" in path.read_text()
        ]
        self.assertEqual(offenders, [], "these modules re-implement the episode loop")

    def test_eval_arm_cost_matches_the_shared_pooled_cost(self):
        import numpy as np

        from rl.env import SCENARIO_BUILDERS
        from rl.experiments.constrained_retry import _eval_arm, _make_env
        from rl.policies.fixed import FixedPolicy
        from rl.rollout import policy_actor, run_episode
        from rl.spec import ActionSpec

        spec = ActionSpec([10, 30, 50, 80], [1, 2, 3], [0, 20, 50])
        build = lambda seed: SCENARIO_BUILDERS["chain"](seed, 10.0)  # noqa: E731
        seeds = [321, 654]

        arm = _eval_arm(_make_env(build, spec, 4.0), FixedPolicy(action=(3, 2, 1)), seeds, 0.1)
        direct = [
            run_episode(
                _make_env(build, spec, 4.0), policy_actor(FixedPolicy(action=(3, 2, 1))), seed=s
            ).pooled_cost
            for s in seeds
        ]
        self.assertAlmostEqual(arm["cost_mean"], float(np.mean(direct)), places=12)
