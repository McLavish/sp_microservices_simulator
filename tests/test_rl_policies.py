import math
import unittest

import numpy as np

from rl.env import ActionSpec, RewardWeights
from rl.models.lognormal import LogNormalModel
from rl.policies.action_space import ActionGrid
from rl.policies.aimd import AIMDPolicy
from rl.policies.bayesian import BayesianPolicy
from rl.policies.model_based import ModelBasedPolicy


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
            timeout_decrease_step=1,
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


if __name__ == "__main__":
    unittest.main()
