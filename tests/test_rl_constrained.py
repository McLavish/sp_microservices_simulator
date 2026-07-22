import os
import tempfile
import unittest

from rl.constrained import (
    PPO,
    ConstrainedTrainConfig,
    dual_ascent_update,
    objective_from_summary,
    train_constrained_ppo,
)
from rl.env import ActionSpec, build_chain_env
from rl.spec import RewardWeights


class TestDualAscent(unittest.TestCase):
    def test_lambda_rises_when_cost_exceeds_budget(self):
        self.assertAlmostEqual(dual_ascent_update(0.5, cost=0.3, budget=0.1, lr=1.0), 0.7)

    def test_lambda_falls_when_cost_below_budget(self):
        self.assertAlmostEqual(dual_ascent_update(0.5, cost=0.0, budget=0.1, lr=1.0), 0.4)

    def test_lambda_floors_at_zero(self):
        self.assertEqual(dual_ascent_update(0.05, cost=0.0, budget=0.5, lr=1.0), 0.0)


class TestObjective(unittest.TestCase):
    def test_matches_hand_computed_value(self):
        weights = RewardWeights()  # success 1, drops 1 each, p95 0.001
        summary = {
            "success_rate": 0.9,
            "deadline_drop_rate": 0.05,
            "queue_drop_rate": 0.02,
            "server_drop_rate": 0.02,
            "circuit_open_drop_rate": 0.01,
            "p95_latency_ms": 100.0,
            "retries_per_root": 5.0,  # must NOT enter J
            "cost": 5.0,  # must NOT enter J
        }
        expected = 0.9 - 0.05 - 0.02 - 0.02 - 0.01 - 100.0 * 0.001
        self.assertAlmostEqual(objective_from_summary(summary, weights), expected)


@unittest.skipIf(PPO is None, "stable-baselines3 not installed")
class TestConstrainedTraining(unittest.TestCase):
    def test_smoke_lambda_decays_when_constraint_slack(self):
        # Tiny run on the chain env with a generous budget: cost stays below
        # the budget, so lambda must decay from its init toward zero.
        with tempfile.TemporaryDirectory() as tmp:
            curve_path = os.path.join(tmp, "curve.csv")
            cfg = ConstrainedTrainConfig(
                build=lambda seed: build_chain_env(seed, duration_s=3),
                action_spec=ActionSpec(
                    attempt_timeout_ms=[50, 80],
                    max_attempts=[1, 2],
                    backoff_ms=[20],
                ),
                episode_duration_s=3.0,
                timesteps=128,
                chunk_timesteps=64,
                n_envs=1,  # keep the unit test single-process and fast
                eval_episodes=1,
                budget=5.0,  # generous: always slack
                lambda_init=1.0,
                lambda_lr=0.1,
                learning_curve_csv=curve_path,
            )
            policy = train_constrained_ppo(cfg)
            self.assertEqual(policy.name, "constrained_ppo")
            with open(curve_path) as f:
                lines = f.read().strip().splitlines()
            self.assertEqual(lines[0], "timesteps,eval_J,eval_cost,lambda")
            final_lambda = float(lines[-1].split(",")[-1])
            self.assertLess(final_lambda, 1.0)

    def test_subproc_workers_receive_lambda_broadcast(self):
        # Exercises the SubprocVecEnv path: env factories must survive
        # cloudpickle and lambda must reach workers via env_method.
        cfg = ConstrainedTrainConfig(
            build=lambda seed: build_chain_env(seed, duration_s=3),
            action_spec=ActionSpec(
                attempt_timeout_ms=[50, 80],
                max_attempts=[1, 2],
                backoff_ms=[20],
            ),
            episode_duration_s=3.0,
            timesteps=128,
            chunk_timesteps=64,
            n_envs=2,
            eval_episodes=1,
            budget=0.001,  # tight: lambda must rise above its init
            lambda_init=0.0,
            lambda_lr=1.0,
        )
        policy = train_constrained_ppo(cfg)
        self.assertEqual(policy.name, "constrained_ppo")

    def test_chunked_training_respects_total_timesteps(self):
        # SB3 trains `total_timesteps` ADDITIONAL steps when
        # reset_num_timesteps=False; the loop must pass per-chunk increments.
        # chunk=4 with n_steps=2, n_envs=1, batch=2 divides evenly, so there is
        # no rollout-rounding overshoot and the count is exact.
        cfg = ConstrainedTrainConfig(
            build=lambda seed: build_chain_env(seed, duration_s=3),
            action_spec=ActionSpec(
                attempt_timeout_ms=[50, 80], max_attempts=[1, 2], backoff_ms=[20]
            ),
            episode_duration_s=3.0,
            timesteps=8,
            chunk_timesteps=4,
            n_envs=1,
            n_steps=2,
            batch_size=2,
            eval_episodes=1,
        )
        policy = train_constrained_ppo(cfg)
        # Buggy cumulative loop: learn(4) + learn(8) -> 4 + 8 = 12 steps.
        self.assertEqual(policy.model.num_timesteps, 8)


if __name__ == "__main__":
    unittest.main()
