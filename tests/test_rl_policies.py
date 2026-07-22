import math
import unittest

import numpy as np

from rl.env import ActionSpec, RewardWeights
from rl.models.lognormal import LogNormalModel
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
        # Lognormal with E[S] = 1 ms and E[S^2] = 2 ms^2 (CV^2 = 1, the
        # exponential moments), lambda = 0.5/ms, c = 1: the Allen-Cunneen
        # approximation must reduce to the exact M/M/1 result
        # Wq = rho / (1 - rho) * E[S] = 1.0 ms.
        policy = ModelBasedPolicy(
            grid=self.grid,
            include_queue_delay=True,
            queue_workers=1,
            step_duration_s=1.0,
            use_retries_in_lambda=False,
        )
        model = LogNormalModel(mu=-math.log(2) / 2, sigma=math.sqrt(math.log(2)))
        info = {"summary": {"completed": 500, "retries_per_root": 0.0}}
        wq = policy._estimate_queue_delay(info, model)
        self.assertAlmostEqual(wq, 1.0, places=9)

    def test_queue_delay_zero_without_traffic(self):
        policy = ModelBasedPolicy(
            grid=self.grid,
            include_queue_delay=True,
            queue_workers=1,
        )
        model = LogNormalModel(mu=0.0, sigma=0.5)
        self.assertEqual(policy._estimate_queue_delay({"summary": {"completed": 0}}, model), 0.0)
        self.assertEqual(policy._estimate_queue_delay({}, model), 0.0)

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
    def __init__(self, value_ms: float):
        self._value = value_ms

    def sample(self, rng, n):
        return np.full(n, self._value)


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
