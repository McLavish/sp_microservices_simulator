import unittest

import numpy as np

try:
    import gymnasium  # noqa: F401
except Exception:
    gymnasium = None  # ty: ignore[invalid-assignment]

from core import s_to_ns
from rl.env import (
    ActionSpec,
    EnvConfig,
    MicroservicesEnv,
    build_randomized_faulty_single_service_env,
    build_single_service_env,
)


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestMicroservicesEnv(unittest.TestCase):
    def test_reset_step(self):
        action_spec = ActionSpec(
            attempt_timeout_ms=[10, 50],
            max_attempts=[1, 2],
            backoff_ms=[0, 50],
        )
        cfg = EnvConfig(
            build=build_single_service_env,
            action_spec=action_spec,
            step_duration_s=1.0,
            episode_duration_s=2.0,
        )
        env = MicroservicesEnv(cfg)

        obs, info = env.reset(seed=123)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertIn("action", info)
        self.assertEqual(obs[0], 0.0)  # has_data flag: nothing completed yet

        action = np.array([1, 0, 1])
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertTrue(env.observation_space.contains(obs))
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        self.assertIn("summary", info)
        self.assertEqual(info["action"]["attempt_timeout_ms"], 50)
        self.assertEqual(info["action"]["max_attempts"], 1)
        self.assertEqual(info["action"]["backoff_ms"], 50)
        self.assertFalse(terminated)

        _, _, terminated, _, _ = env.step(action)
        self.assertTrue(terminated)

    def _make_env(self, build, episode_duration_s=2.0):
        action_spec = ActionSpec(
            attempt_timeout_ms=[10, 50],
            max_attempts=[1, 2],
            backoff_ms=[0, 50],
        )
        cfg = EnvConfig(
            build=build,
            action_spec=action_spec,
            step_duration_s=1.0,
            episode_duration_s=episode_duration_s,
        )
        return MicroservicesEnv(cfg)

    def test_unseeded_resets_use_distinct_build_seeds(self):
        # Regression: SB3 vector envs only seed the first reset; unseeded resets
        # must not collapse to the builders' fixed fallback seeds.
        build_seeds = []

        def build(seed):
            build_seeds.append(seed)
            return build_single_service_env(seed)

        env = self._make_env(build)
        env.reset(seed=123)
        self.assertEqual(build_seeds[-1], 123)

        env.reset()
        env.reset()
        first, second = build_seeds[-2], build_seeds[-1]
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    def test_apply_knobs_reaches_policy_and_controller(self):
        from core import ms_to_ns
        from timeout_controllers import StaticTimeoutController

        action_spec = ActionSpec(
            attempt_timeout_ms=[10, 50],
            max_attempts=[1, 2],
            backoff_ms=[0, 50],
            global_timeout_ms=[100, 200],
        )
        cfg = EnvConfig(
            build=build_single_service_env,
            action_spec=action_spec,
            step_duration_s=1.0,
            episode_duration_s=2.0,
        )
        env = MicroservicesEnv(cfg)
        env.reset(seed=3)
        _, _, _, _, info = env.step(np.array([1, 1, 0, 1]))

        self.assertEqual(info["action"]["attempt_timeout_ms"], 50)
        self.assertEqual(info["action"]["global_timeout_ms"], 200)
        client = env._client
        assert client is not None
        controller = client.cfg.timeout_controller
        assert isinstance(controller, StaticTimeoutController)
        self.assertEqual(controller.attempt_timeout, ms_to_ns(50))
        self.assertEqual(controller.global_timeout, ms_to_ns(200))

    def test_randomized_faults_deterministic_per_seed(self):
        def fault_params(components):
            cfg = components.client.service.cfg
            inj = cfg.latency_injections[0]
            pf = cfg.partial_failures[0]
            spike = components.workload.load_spikes[0]
            return (
                inj.duration.begin,
                inj.duration.end,
                inj.add_latency,
                pf.duration.begin,
                pf.p_fail,
                spike.duration.begin,
                spike.rps_multiplier,
            )

        a = fault_params(build_randomized_faulty_single_service_env(5, duration_s=30))
        b = fault_params(build_randomized_faulty_single_service_env(5, duration_s=30))
        c = fault_params(build_randomized_faulty_single_service_env(6, duration_s=30))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_randomized_fault_windows_stay_inside_episode(self):
        for seed in range(20):
            components = build_randomized_faulty_single_service_env(seed, duration_s=30)
            cfg = components.client.service.cfg
            for interval in (
                cfg.latency_injections[0].duration,
                cfg.partial_failures[0].duration,
                components.workload.load_spikes[0].duration,
            ):
                self.assertGreaterEqual(interval.begin, 0)
                self.assertLess(interval.begin, interval.end)
                self.assertLessEqual(interval.end, s_to_ns(30))

    def test_drained_roots_reported_separately_from_final_summary(self):
        env = self._make_env(build_single_service_env, episode_duration_s=1.0)
        env.reset(seed=7)
        _, _, terminated, _, info = env.step(np.array([0, 0, 0]))
        self.assertTrue(terminated)
        # Drained roots are reported separately, not folded into the step summary.
        self.assertIn("drained_summary", info)
        self.assertIsInstance(info["drained_summary"], dict)


if __name__ == "__main__":
    unittest.main()
