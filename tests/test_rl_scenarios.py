"""The scenarios that ship with the environment, and the property that makes them worth training on."""

import pickle
import random
import unittest
import warnings
from collections import Counter

import gymnasium
from gymnasium.utils.env_checker import check_env

import msim.rl  # noqa: F401  (registers the environments)
from msim.core import s_to_ns
from msim.retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy, policy_layers
from msim.rl import Fault, FaultRanges, RetryEnv, RetryStorm, SingleService

#: Fixed settings to put side by side: (max attempts, attempt timeout, backoff, budget) levels.
NO_RETRIES = [0, 1, 1, 4]
RETRY_FREELY = [3, 1, 1, 4]  # four attempts, a budget of one retry per request


def _success(scenario, action, seed: int = 3) -> tuple[float, float]:
    """Success over the episode and inside the fault window, under one fixed setting."""
    env = RetryEnv(scenario)
    env.reset(options={"episode_seed": seed})
    info: dict = {}
    for _ in range(env.episode_steps):
        *_, info = env.step(action)
    return info["ep_success_rate"], info["ep_fault_success_rate"]


class TestTheFaultOfAnEpisode(unittest.TestCase):
    def test_the_same_seed_draws_the_same_episode_and_seeds_differ(self):
        for scenario in (SingleService(), RetryStorm()):
            with self.subTest(scenario=type(scenario).__name__):
                one, two = scenario.build(5), scenario.build(5)
                self.assertEqual(one.fault_windows, two.fault_windows)
                self.assertEqual(one.expected_rps, two.expected_rps)
                draws = {
                    (tuple(e.fault_windows), e.expected_rps)
                    for e in (scenario.build(seed) for seed in range(12))
                }
                self.assertGreater(len(draws), 9)

    def test_every_kind_comes_up_and_stays_inside_the_episode(self):
        ranges = FaultRanges()
        rng = random.Random(1)
        drawn = [ranges.draw(rng) for _ in range(400)]
        kinds = Counter("none" if fault is None else fault.kind for fault in drawn)
        self.assertEqual(set(kinds), {"failure", "slowdown", "spike", "none"})
        self.assertGreater(min(kinds.values()), 60)
        for fault in filter(None, drawn):
            self.assertGreaterEqual(fault.begin_s, 10.0)  # after the warm-up, with a margin
            self.assertLessEqual(fault.end_s, 55.0)  # and time to recover before 65 s
            low, high = getattr(ranges, fault.kind)
            self.assertTrue(low <= fault.strength <= high)

    def test_a_fault_is_put_where_its_kind_says(self):
        for fault, on in (
            (Fault("failure", 10, 20, 0.4), "partial_failures"),
            (Fault("slowdown", 10, 20, 25.0), "latency_injections"),
            (Fault("spike", 10, 20, 2.0), "load_spikes"),
        ):
            with self.subTest(kind=fault.kind):
                episode = SingleService(fault=fault).build(1)
                station = episode.services["svc"].cfg
                found = {
                    "partial_failures": station.partial_failures,
                    "latency_injections": station.latency_injections,
                    "load_spikes": episode.workloads[0].load_spikes,
                }
                self.assertEqual([name for name, items in found.items() if items], [on])
                self.assertEqual(found[on][0].duration.begin, s_to_ns(10))
                self.assertEqual(episode.fault_windows, [(10, 20)])

    def test_a_pinned_fault_is_used_in_every_episode(self):
        scenario = RetryStorm(fault=Fault("slowdown", 12, 25, 15.0))
        self.assertEqual({tuple(scenario.build(s).fault_windows) for s in range(5)}, {((12, 25),)})
        self.assertEqual(
            scenario.build(1).services["C"].cfg.latency_injections[0].add_latency, 15_000_000
        )

    def test_what_cannot_be_built_is_refused(self):
        with self.assertRaisesRegex(ValueError, "past the horizon"):
            SingleService(horizon_s=45.0)  # a fault could last until 55 s
        with self.assertRaisesRegex(ValueError, "past the horizon"):
            RetryStorm(fault=Fault("failure", 10, 70, 0.5))
        with self.assertRaisesRegex(ValueError, "a fault is one of"):
            Fault("meteor", 1, 2, 1.0)
        with self.assertRaisesRegex(ValueError, "kinds are among"):
            FaultRanges(kinds=("failure", "meteor"))
        with self.assertRaisesRegex(ValueError, "upstream_attempts"):
            RetryStorm(upstream_attempts=0)

    def test_scenarios_pickle(self):
        for scenario in (SingleService(rps=250.0), RetryStorm(upstream_attempts=2)):
            self.assertEqual(pickle.loads(pickle.dumps(scenario)), scenario)


class TestTheStormsWiring(unittest.TestCase):
    def test_the_agent_drives_both_callers_of_the_bottleneck(self):
        episode = RetryStorm().build(1)
        self.assertEqual(len(episode.agent.clients), 2)
        self.assertEqual({c.service.cfg.name for c in episode.agent.clients}, {"C"})
        self.assertEqual({c.service.cfg.name for c in episode.reward_clients}, {"A", "X"})
        self.assertEqual(len(episode.clients), 6)
        self.assertEqual(sorted(episode.services), ["A", "B", "C", "X", "Y"])
        # One setting, on both.
        episode.agent.apply({"max_attempts": 4.0, "attempt_timeout_ms": 200.0})
        for client in episode.agent.clients:
            backoff = next(
                layer
                for layer in policy_layers(client.cfg.retry)
                if isinstance(layer, FixedBackoffRetryPolicy)
            )
            self.assertEqual(backoff.max_attempts, 4)
            self.assertEqual(client.cfg.timeout_controller.get_attempt_timeout(), 200_000_000)

    def test_only_the_callers_of_the_bottleneck_retry_unless_asked(self):
        def upstream(scenario):
            episode = scenario.build(1)
            agents = {id(client) for client in episode.agent.clients}
            return [c.cfg.retry for c in episode.clients if id(c) not in agents]

        self.assertTrue(all(isinstance(p, NoRetryPolicy) for p in upstream(RetryStorm())))
        layered = upstream(RetryStorm(upstream_attempts=3))
        self.assertEqual([p.max_attempts for p in layered], [3, 3, 3, 3])


class TestAdaptingCanPay(unittest.TestCase):
    """The reason to train on these: the best fixed setting depends on the fault.

    If one of these fails after a change to a scenario's numbers, the scenario
    has become one a constant policy solves, whatever a learning curve says.
    Each comparison is one episode per setting on the same seed.
    """

    def _in_the_fault(self, scenario_type, kind: str, strength: float, action) -> float:
        """Success of the requests that started during a ten-second fault of this kind."""
        scenario = scenario_type(fault=Fault(kind, 10.0, 20.0, strength), horizon_s=30.0)
        return _success(scenario, action)[1]

    def test_single_service(self):
        def run(kind: str, strength: float, action) -> float:
            return self._in_the_fault(SingleService, kind, strength, action)

        # Calls fail: every retry is a fresh chance.
        self.assertGreater(
            run("failure", 0.4, RETRY_FREELY), run("failure", 0.4, NO_RETRIES) + 0.25
        )
        # The station is slow, or the load doubles: retries finish it off.
        for kind, strength in (("slowdown", 25.0), ("spike", 2.0)):
            with self.subTest(kind=kind):
                self.assertGreater(
                    run(kind, strength, NO_RETRIES), run(kind, strength, RETRY_FREELY) + 0.25
                )

    def test_retry_storm(self):
        def run(kind: str, strength: float, action) -> float:
            return self._in_the_fault(RetryStorm, kind, strength, action)

        self.assertGreater(
            run("failure", 0.5, RETRY_FREELY), run("failure", 0.5, NO_RETRIES) + 0.10
        )
        self.assertGreater(
            run("slowdown", 15.0, NO_RETRIES), run("slowdown", 15.0, RETRY_FREELY) + 0.25
        )

    def test_a_healthy_episode_succeeds_whatever_the_setting(self):
        nothing = FaultRanges(kinds=("none",))
        for scenario in (
            SingleService(faults=nothing, horizon_s=12.0),
            RetryStorm(faults=nothing, horizon_s=12.0),
        ):
            for action in (NO_RETRIES, RETRY_FREELY):
                with self.subTest(scenario=type(scenario).__name__, action=action):
                    overall, in_fault = _success(scenario, action)
                    self.assertGreater(overall, 0.97)
                    self.assertNotEqual(in_fault, in_fault)  # nan: there was no fault


class TestRegistration(unittest.TestCase):
    def test_gymnasium_makes_them_by_id(self):
        for env_id, scenario in (
            ("msim/SingleService-v0", SingleService),
            ("msim/RetryStorm-v0", RetryStorm),
        ):
            with self.subTest(env_id=env_id):
                env = gymnasium.make(env_id)
                inner = env.unwrapped
                assert isinstance(inner, RetryEnv)
                self.assertIsInstance(inner.scenario, scenario)
                observation, _ = env.reset(seed=1)
                self.assertTrue(env.observation_space.contains(observation))
                self.assertEqual(inner.episode_steps, 60)
                assert env.spec is not None
                self.assertIsNone(env.spec.max_episode_steps)  # the environment truncates itself

    def test_arguments_go_to_the_scenario_or_to_the_environment(self):
        env = gymnasium.make("msim/RetryStorm-v0", upstream_attempts=2, step_s=0.5, rps=30.0)
        inner = env.unwrapped
        assert isinstance(inner, RetryEnv)
        self.assertEqual(inner.scenario, RetryStorm(upstream_attempts=2, rps=30.0))
        self.assertEqual(inner.episode_steps, 120)

    def test_the_checker_passes_on_what_gymnasium_makes(self):
        nothing_late = FaultRanges(onset_s=(6.0, 8.0), length_s=(2.0, 4.0))
        for env_id in ("msim/SingleService-v0", "msim/RetryStorm-v0"):
            with self.subTest(env_id=env_id), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # A quarter of the load: this is about the interface, not the dynamics.
                rps = 100.0 if "Single" in env_id else 15.0
                env = gymnasium.make(env_id, horizon_s=15.0, faults=nothing_late, rps=rps)
                check_env(env.unwrapped, skip_render_check=True)
                self.assertEqual([str(w.message) for w in caught], [])


if __name__ == "__main__":
    unittest.main()
