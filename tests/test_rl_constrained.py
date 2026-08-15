import functools
import os
import tempfile
import unittest
from dataclasses import replace

from rl.constrained import (
    PPO,
    ConstrainedTrainConfig,
    dual_ascent_update,
    train_constrained_ppo,
)
from rl.env import ActionSpec, build_chain_env
from rl.rollout import EpisodeResult, WindowRecord
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
        result = EpisodeResult(
            seed=None,
            total_reward=0.0,
            windows=[WindowRecord(step=0, reward=0.0, summary=summary, action_values={})],
        )
        # EpisodeResult.objective is the LIVE formula; objective_from_summary was
        # a second copy with no production callers, and pinning the dead one let
        # a wrong J ship green.
        self.assertAlmostEqual(result.objective(weights), expected)


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
        """The multiplier must land in the WORKER processes' reward weights.

        Exercises the SubprocVecEnv path (the env factory must survive
        cloudpickle) and reads the weight back out of every worker after each
        broadcast. Without the read-back a lost broadcast is invisible: the
        parent-side curve still shows dual ascent while every worker keeps
        training on `lambda_init`, i.e. unconstrained PPO.

        The budget is generous and max_attempts caps the cost at 1 retry/root,
        so `1.0 + 0.5 * (cost - 5.0)` is projected to exactly 0.0 on every
        chunk — a value distinct from `lambda_init`, which is what the workers
        were constructed with.
        """
        import rl.constrained as mod

        broadcasts: list[tuple[float, list[float]]] = []
        original = mod.make_vec_env

        def spy(*args, **kwargs):
            vec_env = original(*args, **kwargs)
            inner_env_method = vec_env.env_method

            def env_method(method_name, *m_args, **m_kwargs):
                result = inner_env_method(method_name, *m_args, **m_kwargs)
                if method_name == "set_cost_weight":
                    # Round-trips each worker's own config out of its process.
                    weights = [
                        float(worker.reward_weights.cost) for worker in vec_env.get_attr("_cfg")
                    ]
                    broadcasts.append((float(m_args[0]), weights))
                return result

            vec_env.env_method = env_method
            return vec_env

        mod.make_vec_env = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                curve_path = os.path.join(tmp, "curve.csv")
                cfg = ConstrainedTrainConfig(
                    # A picklable partial, not a lambda: the workers ship their
                    # EnvConfig back for inspection. The factory itself is still
                    # a closure, so cloudpickle is still on the path.
                    build=functools.partial(build_chain_env, duration_s=3),
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
                    budget=5.0,
                    lambda_init=1.0,
                    lambda_lr=0.5,
                    learning_curve_csv=curve_path,
                )
                policy = train_constrained_ppo(cfg)
                with open(curve_path) as f:
                    curve = f.read().strip().splitlines()
        finally:
            mod.make_vec_env = original

        self.assertEqual(policy.name, "constrained_ppo")
        self.assertTrue(broadcasts, "lambda was never broadcast to the workers")
        for lam, worker_weights in broadcasts:
            self.assertEqual(worker_weights, [lam] * cfg.n_envs)
            self.assertNotEqual(lam, cfg.lambda_init)  # or the read-back proves nothing
        # The workers saw the same multiplier the curve reports.
        self.assertEqual(broadcasts[-1][0], float(curve[-1].split(",")[-1]))

    def _tiny_cfg(self, **overrides) -> ConstrainedTrainConfig:
        base = ConstrainedTrainConfig(
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
        return replace(base, **overrides)

    def test_config_is_validated_before_any_env_is_spawned(self):
        # A bad chunk size used to raise only after n_envs worker interpreters
        # and a torch model had been built, leaving them orphaned.
        import rl.constrained as mod

        def exploding_make_vec_env(*_args, **_kwargs):
            raise AssertionError("vec env built before the config was validated")

        original = mod.make_vec_env
        mod.make_vec_env = exploding_make_vec_env
        try:
            with self.assertRaises(ValueError):
                train_constrained_ppo(self._tiny_cfg(chunk_timesteps=0))
            with self.assertRaises(ValueError):
                train_constrained_ppo(self._tiny_cfg(timesteps=0))
        finally:
            mod.make_vec_env = original

    def test_vec_env_is_closed_when_training_finishes(self):
        # Nothing else closes it (SB3 defines no __del__ on VecEnv), and the
        # returned policy keeps model -> model.env alive, so every arm of a
        # multi-arm sweep would hold its workers open until the process exits.
        import rl.constrained as mod

        closed: list[str] = []
        original = mod.make_vec_env

        def spy(*args, **kwargs):
            vec_env = original(*args, **kwargs)
            inner_close = vec_env.close

            def close():
                closed.append("closed")
                inner_close()

            vec_env.close = close
            return vec_env

        mod.make_vec_env = spy
        try:
            policy = train_constrained_ppo(self._tiny_cfg())
        finally:
            mod.make_vec_env = original
        self.assertEqual(closed, ["closed"])
        # The policy still works after the close: predict never touches the env.
        self.assertEqual(policy.name, "constrained_ppo")

    def test_vec_env_is_closed_when_training_raises(self):
        import rl.constrained as mod

        closed: list[str] = []
        original_make = mod.make_vec_env

        def spy(*args, **kwargs):
            vec_env = original_make(*args, **kwargs)
            inner_close = vec_env.close

            def close():
                closed.append("closed")
                inner_close()

            vec_env.close = close
            return vec_env

        def exploding_eval(*_args, **_kwargs):
            raise RuntimeError("eval blew up")

        original_eval = mod.evaluate_objective_and_cost
        mod.make_vec_env = spy
        mod.evaluate_objective_and_cost = exploding_eval  # ty: ignore[invalid-assignment]
        try:
            with self.assertRaises(RuntimeError):
                train_constrained_ppo(self._tiny_cfg())
        finally:
            mod.make_vec_env = original_make
            mod.evaluate_objective_and_cost = original_eval
        self.assertEqual(closed, ["closed"])

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
