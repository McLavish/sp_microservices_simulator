import unittest

import numpy as np

try:
    import gymnasium  # noqa: F401
except Exception:
    gymnasium = None

from rl.env import ActionSpec, EnvConfig, MicroservicesEnv, build_single_service_env


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


if __name__ == "__main__":
    unittest.main()
