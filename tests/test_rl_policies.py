import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from rl.env import ActionSpec, RewardWeights
from rl.model_io import check_action_spec, save_action_spec
from rl.policies.action_space import ActionGrid
from rl.policies.aimd import AIMDPolicy
from rl.policies.bayesian import BayesianPolicy
from rl.policies.model_based import ModelBasedPolicy, _simulate_action
from rl.spec import ActionValues


class TestRLPolicies(unittest.TestCase):
    def setUp(self):
        self.action_spec = ActionSpec(
            attempt_timeout_ms=[10, 20],
            max_attempts=[1, 2],
            backoff_ms=[0],
        )
        self.grid = ActionGrid(self.action_spec)

    def test_action_grid(self):
        self.assertEqual(len(self.grid.indices), 4)
        values = self.grid.values_for((1, 0, 0))
        self.assertEqual(values.attempt_timeout_ms, 20)
        self.assertEqual(values.max_attempts, 1)

    def test_model_based_policy_act(self):
        policy = ModelBasedPolicy(
            grid=self.grid,
            reward_weights=RewardWeights(),
            sample_size=50,
            buffer_size=100,
        )
        info = {
            "attempt_latencies_ms": [12.0, 15.0, 18.0],
            "attempt_censored": [False, False, False],
            "summary": {"success_rate": 1.0, "deadline_drop_rate": 0.0},
        }
        action = policy.act(np.zeros(12, dtype=np.float32), info)
        self.assertIn(action, self.grid.indices)

    def test_model_based_policy_queue_delay(self):
        policy = ModelBasedPolicy(
            grid=self.grid,
            reward_weights=RewardWeights(),
            sample_size=50,
            buffer_size=100,
            include_queue_delay=True,
            queue_workers=2,
            step_duration_s=1.0,
        )
        info = {
            "attempt_latencies_ms": [12.0, 15.0, 18.0],
            "attempt_censored": [False, False, False],
            "summary": {
                "completed": 5,
                "retries_per_root": 0.2,
                "success_rate": 1.0,
                "deadline_drop_rate": 0.0,
            },
        }
        action = policy.act(np.zeros(12, dtype=np.float32), info)
        self.assertIn(action, self.grid.indices)

    def test_queue_delay_matches_mm1_closed_form(self):
        # Service moments with E[S] = 1 ms and E[S^2] = 2 ms^2 (CV^2 = 1, the
        # exponential moments), lambda = 0.5/ms from the attempt stream, c = 1:
        # the Allen-Cunneen approximation must reduce to the exact M/M/1 result
        # Wq = rho / (1 - rho) * E[S] = 1.0 ms.
        policy = ModelBasedPolicy(
            grid=self.grid,
            include_queue_delay=True,
            queue_workers=1,
            step_duration_s=1.0,
        )
        policy._service_buffer.add([0.0, 2.0] * 250)  # mean 1, mean-square 2
        wq = policy._estimate_queue_delay({"attempt_latencies_ms": [1.0] * 500})
        self.assertAlmostEqual(wq, 1.0, places=9)

    def test_queue_delay_lambda_comes_from_the_sampled_attempt_stream(self):
        # In chain scenarios the samples describe the B->C edge; using
        # top-level window completions for lambda underestimates the modeled
        # queue's arrival rate by the whole retry amplification.
        policy = ModelBasedPolicy(
            grid=self.grid, include_queue_delay=True, queue_workers=1, step_duration_s=1.0
        )
        policy._service_buffer.add([0.0, 2.0] * 250)
        light = policy._estimate_queue_delay(
            {"attempt_latencies_ms": [1.0] * 100, "summary": {"completed": 100.0}}
        )
        heavy = policy._estimate_queue_delay(
            {"attempt_latencies_ms": [1.0] * 800, "summary": {"completed": 100.0}}
        )
        self.assertGreater(heavy, light)  # summary-based lambda would tie these

    def test_queue_delay_zero_without_traffic(self):
        policy = ModelBasedPolicy(grid=self.grid, include_queue_delay=True, queue_workers=1)
        self.assertEqual(policy._estimate_queue_delay({"attempt_latencies_ms": []}), 0.0)
        self.assertEqual(policy._estimate_queue_delay({}), 0.0)

    def test_bayesian_policy_act(self):
        policy = BayesianPolicy(
            grid=self.grid,
            reward_weights=RewardWeights(),
            sample_size=50,
            buffer_size=100,
        )
        info = {
            "attempt_latencies_ms": [12.0, 15.0, 18.0],
            "attempt_censored": [False, False, False],
            "summary": {"success_rate": 1.0, "deadline_drop_rate": 0.0},
        }
        action = policy.act(np.zeros(12, dtype=np.float32), info)
        self.assertIn(action, self.grid.indices)

    def test_aimd_policy_act(self):
        policy = AIMDPolicy(
            grid=self.grid,
            initial_timeout_ms=10,
            initial_max_attempts=1,
            initial_backoff_ms=0,
            adjust_max_attempts=True,
            adjust_backoff=False,
            ema_alpha=0.5,
            timeout_increase_step=1,
        )
        info = {
            "summary": {
                "completed": 5,
                "success_rate": 0.8,
                "deadline_drop_rate": 0.2,
            }
        }
        action = policy.act(np.zeros(12, dtype=np.float32), info)
        self.assertIn(action, self.grid.indices)

    def test_aimd_multiplicative_decrease_snaps_to_grid(self):
        spec = ActionSpec(
            attempt_timeout_ms=[10, 30, 50, 80],
            max_attempts=[1],
            backoff_ms=[0],
        )
        grid = ActionGrid(spec)
        policy = AIMDPolicy(grid=grid, initial_timeout_ms=80)
        healthy = {"summary": {"completed": 10, "success_rate": 1.0, "deadline_drop_rate": 0.0}}

        action = policy.act(np.zeros(15, dtype=np.float32), healthy)
        self.assertEqual(spec.attempt_timeout_ms[action[0]], 30)  # 80 * 0.5 = 40 -> snap down
        action = policy.act(np.zeros(15, dtype=np.float32), healthy)
        self.assertEqual(spec.attempt_timeout_ms[action[0]], 10)  # 30 * 0.5 = 15 -> snap down
        action = policy.act(np.zeros(15, dtype=np.float32), healthy)
        self.assertEqual(spec.attempt_timeout_ms[action[0]], 10)  # stays at the floor

    def test_aimd_rejects_invalid_decrease_factor(self):
        with self.assertRaises(ValueError):
            AIMDPolicy(grid=self.grid, timeout_decrease_factor=1.0)

    def test_aimd_policy_emits_global_timeout_dimension(self):
        spec = ActionSpec(
            attempt_timeout_ms=[10, 20],
            max_attempts=[1, 2],
            backoff_ms=[0],
            global_timeout_ms=[100, 200],
        )
        grid = ActionGrid(spec)
        policy = AIMDPolicy(
            grid=grid,
            initial_timeout_ms=10,
            initial_max_attempts=1,
            initial_backoff_ms=0,
            initial_global_timeout_ms=100,
        )
        info = {
            "summary": {
                "completed": 5,
                "success_rate": 0.8,
                "deadline_drop_rate": 0.2,
            }
        }
        action = policy.act(np.zeros(12, dtype=np.float32), info)
        self.assertEqual(len(action), 4)
        self.assertIn(action, grid.indices)


class TestServiceMoments(unittest.TestCase):
    def test_queue_variant_derives_moments_from_service_stream(self):
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0]))
        policy = ModelBasedPolicy(grid=grid, include_queue_delay=True)
        policy.act(
            obs=np.zeros(1),
            info={
                "attempt_latencies_ms": [40.0] * 50,  # totals: include queue wait
                "attempt_censored": [False] * 50,
                "attempt_service_ms": [20.0] * 50,  # queue-exclusive
                "summary": {"completed": 50.0, "retries_per_root": 0.0},
            },
        )
        mean_s, mean_s2, count = policy._service_moments()
        self.assertEqual(count, 50)
        self.assertAlmostEqual(mean_s, 20.0)
        self.assertAlmostEqual(mean_s2, 400.0)

    def test_queue_variant_fits_service_stream_not_totals(self):
        # Double-count guard: with include_queue_delay the latency model must
        # be fitted on queue-exclusive service times (queue wait is added back
        # via Erlang-C); the plain variant keeps fitting end-to-end totals.
        info = {
            "attempt_latencies_ms": [40.0] * 50,
            "attempt_censored": [False] * 50,
            "attempt_service_ms": [20.0] * 50,
            "attempt_service_censored": [False] * 50,
            "summary": {"completed": 50.0, "retries_per_root": 0.0},
        }
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0]))
        queue_policy = ModelBasedPolicy(grid=grid, include_queue_delay=True)
        queue_policy.act(np.zeros(1), info)
        self.assertAlmostEqual(float(queue_policy._fit_buffer().data().mean()), 20.0)
        plain_policy = ModelBasedPolicy(grid=grid)
        plain_policy.act(np.zeros(1), info)
        self.assertAlmostEqual(float(plain_policy._fit_buffer().data().mean()), 40.0)


class TestTraceTimeAxis(unittest.TestCase):
    def test_uses_step_duration_column(self):
        import pandas as pd

        from plotting.rl_traces_over_time import _step_seconds

        df = pd.DataFrame({"step": [0, 1, 2], "step_duration_s": [2.0, 2.0, 2.0]})
        self.assertEqual(_step_seconds(df), 2.0)

    def test_legacy_csv_defaults_to_one_second(self):
        import pandas as pd

        from plotting.rl_traces_over_time import _step_seconds

        df = pd.DataFrame({"step": [0, 1, 2]})
        self.assertEqual(_step_seconds(df), 1.0)


class TestNearestIndicesGlobalTimeout(unittest.TestCase):
    def _grid(self, global_list):
        return ActionGrid(
            ActionSpec(
                attempt_timeout_ms=[10, 30],
                max_attempts=[1],
                backoff_ms=[0],
                global_timeout_ms=global_list,
            )
        )

    def test_none_request_maps_to_the_none_entry(self):
        grid = self._grid([500, 1000, None])
        self.assertEqual(grid.nearest_indices(10, 1, 0, global_timeout_ms=None), (0, 0, 0, 2))

    def test_numeric_request_ignores_none_entries(self):
        grid = self._grid([None, 500, 1000])
        self.assertEqual(grid.nearest_indices(10, 1, 0, global_timeout_ms=600), (0, 0, 0, 1))

    def test_none_request_without_none_entry_raises(self):
        grid = self._grid([500, 1000])
        with self.assertRaises(ValueError):
            grid.nearest_indices(10, 1, 0, global_timeout_ms=None)


class TestActionSpecSidecar(unittest.TestCase):
    def setUp(self):
        self.spec = ActionSpec(
            attempt_timeout_ms=[10, 30],
            max_attempts=[1, 2],
            backoff_ms=[0, 20],
            global_timeout_ms=[500, None],
        )

    def test_roundtrip_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            save_action_spec(path, self.spec, step_duration_s=1.0, fault_scenario="chain")
            check_action_spec(path, self.spec)  # no raise

    def test_reordered_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            save_action_spec(path, self.spec, step_duration_s=1.0)
            mutated = ActionSpec(
                attempt_timeout_ms=[30, 10],
                max_attempts=[1, 2],
                backoff_ms=[0, 20],
                global_timeout_ms=[500, None],
            )
            with self.assertRaises(ValueError):
                check_action_spec(path, mutated)

    def test_sidecar_survives_sb3_implicit_zip_extension(self):
        # SB3 save("model") writes model.zip; load("model.zip") must find the
        # sidecar written at save time regardless of which form was used.
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "model")
            save_action_spec(base, self.spec, step_duration_s=1.0)
            check_action_spec(base + ".zip", self.spec)  # no raise, no warn-skip
            mutated = ActionSpec(
                attempt_timeout_ms=[30, 10],
                max_attempts=[1, 2],
                backoff_ms=[0, 20],
                global_timeout_ms=[500, None],
            )
            with self.assertRaises(ValueError):
                check_action_spec(base + ".zip", mutated)

    def test_missing_sidecar_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                check_action_spec(path, self.spec)
            self.assertTrue(any("sidecar" in str(w.message) for w in caught))


class TestZeroSampleRobustness(unittest.TestCase):
    def _info(self, values):
        return {
            "attempt_latencies_ms": values,
            "attempt_censored": [False] * len(values),
            "summary": {"completed": float(len(values)), "retries_per_root": 0.0},
        }

    def test_bayesian_ignores_nonpositive_samples(self):
        # Zeros must contribute nothing: same decision as the clean stream.
        # Pre-fix, log(0) collapses the posterior to NaN and the argmax
        # degenerates to the first grid entry (timeout 10 ms, which the clean
        # fit rejects because every sample exceeds it).
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[10, 50], max_attempts=[1], backoff_ms=[0]))
        with_zeros = BayesianPolicy(grid=grid).act(
            np.zeros(1), self._info([0.0, 0.0, 20.0, 25.0, 22.0, 24.0])
        )
        clean = BayesianPolicy(grid=grid).act(np.zeros(1), self._info([20.0, 25.0, 22.0, 24.0]))
        self.assertEqual(with_zeros, clean)
        self.assertEqual(clean, (1, 0, 0))

    def test_model_based_survives_all_zero_buffer(self):
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0]))
        policy = ModelBasedPolicy(grid=grid)
        action = policy.act(np.zeros(1), self._info([0.0, 0.0, 0.0]))
        self.assertEqual(action, grid.indices[0])


class _ConstantModel:
    """Minimal LatencyModel-protocol stub returning a constant latency."""

    def __init__(self, value_ms: float):
        self._value = value_ms

    def fit(self, samples):
        return self

    def logpdf(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def cdf(self, x):
        return (np.asarray(x, dtype=float) >= self._value).astype(float)

    def quantile(self, q: float) -> float:
        return self._value

    def sample(self, rng, size):
        return np.full(size, self._value)

    def params(self) -> dict[str, float]:
        return {"value_ms": self._value}

    def log_likelihood(self, samples) -> float:
        return 0.0


class TestSurrogateGlobalDeadline(unittest.TestCase):
    def _score(self, sample_ms, attempt_timeout_ms, global_timeout_ms):
        weights = RewardWeights(
            success=1.0,
            deadline_drop=1.0,
            queue_drop=0.0,
            server_drop=0.0,
            circuit_open_drop=0.0,
            p95_latency_ms=0.0,
            retries=0.0,
        )
        action = ActionValues(
            attempt_timeout_ms=attempt_timeout_ms,
            max_attempts=1,
            backoff_ms=0,
            global_timeout_ms=global_timeout_ms,
        )
        return _simulate_action(
            _ConstantModel(sample_ms),
            action,
            weights,
            np.random.default_rng(0),
            n_roots=10,
            queue_delay_ms=0.0,
            queue_delay_mode="deterministic",
        )

    def test_sample_beyond_global_budget_is_a_deadline_drop(self):
        # 80 ms sample, 100 ms attempt timeout, 50 ms global: the real caller
        # caps the attempt at min(remaining global, attempt timeout) and
        # requires strictly-before completion -> deadline drop, reward -1.
        self.assertAlmostEqual(self._score(80.0, 100, 50), -1.0)

    def test_sample_within_both_budgets_succeeds(self):
        self.assertAlmostEqual(self._score(30.0, 100, 200), 1.0)


if __name__ == "__main__":
    unittest.main()
