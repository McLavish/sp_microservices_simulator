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

_PPO = import_sb3().ppo


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

    def test_one_zero_success_seed_does_not_void_the_p95_column(self):
        """`Metrics.summary().p95` is nan when no root succeeded — deliberately.

        Averaging the eval seeds with plain np.mean propagated that nan into the
        reported cell and threw away every other seed's data.
        """
        import math

        from caller import ClientConfig, ClientRuntime
        from core import TimeInterval, ms_to_ns, s_to_ns
        from faults import PartialFailure
        from retry_policies import FixedBackoffRetryPolicy
        from rl.env import EnvComponents, EnvConfig, MicroservicesEnv
        from rl.experiments.compare_policies import _evaluate
        from rl.policies.fixed import FixedPolicy
        from rl.spec import ActionSpec
        from service import ServiceConfig, ServiceRuntime
        from simulator import Simulator
        from timeout_controllers import StaticTimeoutController
        from workload import Workload

        def build(seed):
            # Odd seeds fail every request, so that episode reports p95 = nan.
            p_fail = 1.0 if int(seed) % 2 else 0.0
            sim = Simulator(seed=int(seed))
            svc_cfg = ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(5),
                latency_lognorm_sigma=0.1,
                workers=32,
                partial_failures=[
                    PartialFailure(duration=TimeInterval(begin=0, end=s_to_ns(1000)), p_fail=p_fail)
                ],
            )
            timeout_ctrl = StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(50)
            )
            retry = FixedBackoffRetryPolicy(max_attempts=1, delay=0)
            client = ClientRuntime(
                cfg=ClientConfig(retry=retry, timeout_controller=timeout_ctrl),
                service=ServiceRuntime(cfg=svc_cfg).bind(),
            )

            def apply_knobs(values):
                timeout_ctrl.attempt_timeout = ms_to_ns(values.attempt_timeout_ms)
                retry.max_attempts = values.max_attempts
                retry.delay = ms_to_ns(values.backoff_ms)

            return EnvComponents(
                sim=sim,
                workload=Workload(
                    base_rps=50.0, duration_s=2.0, load_spikes=[], rng_seed=int(seed)
                ),
                client=client,
                apply_knobs=apply_knobs,
            )

        spec = ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0])
        env = MicroservicesEnv(
            EnvConfig(build=build, action_spec=spec, step_duration_s=1.0, episode_duration_s=2.0)
        )
        metrics = _evaluate(env, FixedPolicy(action=(0, 0, 0)), [1, 2])
        self.assertFalse(math.isnan(metrics["p95_ms"]), "one nan seed voided the whole column")
        self.assertGreater(metrics["p95_ms"], 0.0)

    def test_ppo_import_error_names_the_real_cause(self):
        # import_sb3 catches bare Exception, so a broken torch/sb3 install must
        # not be reported as a missing package with the cause thrown away.
        import rl.experiments.compare_policies as mod

        cause = RuntimeError("torch ABI mismatch")
        saved = (mod.PPO, mod.make_vec_env, mod._SB3_IMPORT_ERROR)
        mod.PPO = None
        mod.make_vec_env = None
        mod._SB3_IMPORT_ERROR = cause
        try:
            with tempfile.TemporaryDirectory() as d, self.assertRaises(ImportError) as ctx:
                mod.main(
                    [
                        "--fault-scenario",
                        "none",
                        "--episode-duration-s",
                        "2.0",
                        "--eval-seeds",
                        "1",
                        "--policies",
                        "ppo",
                        "--output-csv",
                        os.path.join(d, "r.csv"),
                        "--traces-csv",
                        os.path.join(d, "t.csv"),
                    ]
                )
        finally:
            mod.PPO, mod.make_vec_env, mod._SB3_IMPORT_ERROR = saved
        self.assertIs(ctx.exception.__cause__, cause)

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
                    "1,2",
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
        # `in (0, 1)` accepts both branches, so it cannot detect the very
        # regression this guards against - an always-succeed exit. Pin the
        # verdict: this configuration deterministically fails the gate.
        self.assertEqual(ctx.exception.code, 1)

    def test_calibration_rejects_a_grid_without_a_no_retry_baseline(self):
        """`--max-attempts 2,3` is a legal ActionSpec but has no no-retry arm.

        Criteria (a)/(d)/(e) are all stated against that baseline, and the gate
        used to run the whole (expensive) sweep and then die on `max()` of an
        empty sequence.
        """
        from rl.experiments.constrained_retry import _run_calibration
        from rl.policies.action_space import ActionGrid
        from rl.spec import ActionSpec

        spec = ActionSpec(attempt_timeout_ms=[30], max_attempts=[2, 3], backoff_ms=[0])
        env = self._chain_env(spec)
        with self.assertRaises(SystemExit) as ctx:
            _run_calibration(env, ActionGrid(spec), [1], [1], [0.1])
        self.assertIn("max_attempts", str(ctx.exception))

    def test_calibration_reports_an_infeasible_budget(self):
        from rl.experiments.constrained_retry import _run_calibration
        from rl.policies.action_space import ActionGrid
        from rl.spec import ActionSpec

        spec = ActionSpec(attempt_timeout_ms=[30], max_attempts=[1], backoff_ms=[0])
        env = self._chain_env(spec)
        with self.assertRaises(SystemExit) as ctx:
            _run_calibration(env, ActionGrid(spec), [1], [1], [-1.0])
        self.assertIn("-1.0", str(ctx.exception))

    @staticmethod
    def _chain_env(spec):
        from rl.env import SCENARIO_BUILDERS
        from rl.experiments.constrained_retry import _make_env

        return _make_env(lambda seed: SCENARIO_BUILDERS["chain"](seed, 2.0), spec, 2.0)

    def test_calibration_exits_zero_when_the_gate_passes(self):
        """The passing branch, which no run ever reached."""
        import rl.experiments.constrained_retry as mod

        def passing(*_args, **_kwargs) -> bool:
            return True

        original = mod._run_calibration
        mod._run_calibration = passing  # ty: ignore[invalid-assignment]
        try:
            with self.assertRaises(SystemExit) as ctx:
                mod.main(
                    [
                        "--scenario",
                        "chain",
                        "--calibrate",
                        "--episode-duration-s",
                        "2.0",
                        "--oracle-seeds",
                        "1",
                        "--eval-seeds",
                        "1,2",
                        "--budgets",
                        "0.1",
                        "--attempt-timeouts-ms",
                        "30",
                        "--max-attempts",
                        "1",
                        "--backoff-ms",
                        "0",
                    ]
                )
            self.assertEqual(ctx.exception.code, 0)
        finally:
            mod._run_calibration = original


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
        import re

        # Anchor to the repo, not the cwd: pathlib.Path("rl") silently yields
        # nothing when pytest runs from anywhere else, so the guard passed while
        # scanning zero files.
        root = pathlib.Path(__file__).resolve().parent.parent / "rl"
        self.assertTrue(root.is_dir(), f"expected to scan {root}")
        scanned = [p for p in root.rglob("*.py") if p.resolve() != (root / "rollout.py").resolve()]
        self.assertGreater(len(scanned), 5, "scan found suspiciously few modules")

        # Match on shape, not one literal spelling: `while not terminated and
        # not truncated:` is the same loop and would have slipped through.
        loop = re.compile(r"while\s+not\s*\(?\s*(terminated|truncated)")
        offenders = [p.name for p in scanned if loop.search(p.read_text())]
        self.assertEqual(offenders, [], "these modules re-implement the episode loop")

    def test_pooled_cost_is_pooled_not_a_window_average(self):
        """Pin the RULE, with windows sized so the two aggregations differ.

        The agreement test below cannot catch a wrong rule: both of its sides
        read the same property, so it holds for any definition. Verified by
        mutation - replacing pooled_cost with the macro-average it was written
        to eliminate left the whole suite green.
        """
        from rl.rollout import EpisodeResult, WindowRecord

        def window(step, retries, roots):
            return WindowRecord(
                step=step,
                reward=0.0,
                summary={
                    "cost_retries": float(retries),
                    "cost_roots": float(roots),
                    "cost": retries / roots,
                },
                action_values={},
            )

        # Deliberately lopsided: 1/100 and 9/10.
        result = EpisodeResult(
            seed=None, total_reward=0.0, windows=[window(0, 1, 100), window(1, 9, 10)]
        )
        # Pooled: 10 retries over 110 roots.
        self.assertAlmostEqual(result.pooled_cost, 10 / 110, places=12)
        # The macro-average of the per-window ratios would be 0.455 — nearly 5x.
        macro = (1 / 100 + 9 / 10) / 2
        self.assertNotAlmostEqual(result.pooled_cost, macro, places=3)

    def test_empty_episode_has_no_cost_and_no_objective(self):
        from rl.rollout import EpisodeResult
        from rl.spec import RewardWeights

        empty = EpisodeResult(seed=None, total_reward=0.0, windows=[])
        self.assertEqual(empty.pooled_cost, 0.0)
        self.assertEqual(empty.objective(RewardWeights()), 0.0)

    def test_objective_matches_a_hand_computed_value(self):
        """The live J formula. The only exact-value test used to pin
        objective_from_summary, which has had no production caller since the
        rollout refactor - so a wrong J shipped green."""
        from rl.rollout import EpisodeResult, WindowRecord
        from rl.spec import RewardWeights

        summary = {
            "success_rate": 0.8,
            "deadline_drop_rate": 0.1,
            "queue_drop_rate": 0.05,
            "server_drop_rate": 0.05,
            "circuit_open_drop_rate": 0.0,
            "p95_latency_ms": 100.0,
            # Neither of these may enter J: it is the objective with the retry
            # and cost penalties zeroed, so every arm is comparable regardless
            # of the weights it was trained with.
            "retries_per_root": 5.0,
            "cost": 3.0,
        }
        weights = RewardWeights()
        result = EpisodeResult(
            seed=None,
            total_reward=0.0,
            windows=[WindowRecord(step=0, reward=0.0, summary=summary, action_values={})],
        )
        expected = 0.8 - 0.1 - 0.05 - 0.05 - 0.0 - 100.0 * weights.p95_latency_ms
        self.assertAlmostEqual(result.objective(weights), expected, places=12)

    def test_compare_policies_episode_returns_the_shared_result(self):
        """`_episode` used to flatten EpisodeResult into a positional 4-tuple.

        That flattening is why `_evaluate` re-implemented
        EpisodeResult.success_rate inline while every other caller used the
        accessor, and why bystander_success_rate()/drained_summary were out of
        reach from this path.
        """
        from rl.env import SCENARIO_BUILDERS
        from rl.experiments.compare_policies import _episode
        from rl.experiments.constrained_retry import _make_env
        from rl.policies.fixed import FixedPolicy
        from rl.rollout import EpisodeResult
        from rl.spec import ActionSpec

        spec = ActionSpec([30, 80], [1, 2], [0, 20])
        env = _make_env(lambda seed: SCENARIO_BUILDERS["chain"](seed, 3.0), spec, 3.0)
        result = _episode(env, FixedPolicy(action=(1, 1, 1)), 321)
        self.assertIsInstance(result, EpisodeResult)
        horizon = result.in_horizon
        assert horizon is not None
        self.assertAlmostEqual(
            result.success_rate(in_horizon=True),
            horizon.succeeded / horizon.total,
            places=12,
        )

    def test_mean_ignoring_nan_keeps_the_seeds_that_reported(self):
        import math

        from rl.rollout import mean_ignoring_nan

        self.assertAlmostEqual(mean_ignoring_nan([float("nan"), 1.0, 3.0]), 2.0)
        self.assertEqual(mean_ignoring_nan([]), 0.0)
        # All missing stays visibly missing rather than collapsing to 0 ms.
        self.assertTrue(math.isnan(mean_ignoring_nan([float("nan"), float("nan")])))

    def test_sb3_shim_exposes_named_slots(self):
        # Four modules unpacked this shim; positional unpacking meant inserting
        # or reordering a slot silently rebound every caller.
        self.assertEqual(
            import_sb3()._fields, ("ppo", "make_vec_env", "subproc_vec_env", "import_error")
        )

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


if __name__ == "__main__":
    unittest.main()
