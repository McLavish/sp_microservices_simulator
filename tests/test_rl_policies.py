import unittest

import numpy as np

from rl.env import ActionSpec, RewardWeights
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


if __name__ == "__main__":
    unittest.main()
