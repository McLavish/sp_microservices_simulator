"""Arolla scenario files as situations for the agent (`msim.rl.arolla`)."""

import contextlib
import hashlib
import io
import os
import pickle
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from gymnasium.utils.env_checker import check_env

from msim.admission import AdmissionGate
from msim.compat.arolla import UnsupportedScenario, build, parse
from msim.retry_policies import FixedBackoffRetryPolicy, RetryBudgetPolicy, policy_layers
from msim.rl import Episode, RetryEnv, RetryGrid, RetryKnobs, evaluate
from msim.rl.arolla import ArollaScenario, Variation, agent_part, as_written, file_grid

#: The shape of their post-mortem files: one arm per service, the same fault on each.
ARMS = """
name: arms
seed: 7
services:
  - name: svc-none
    latency: &latency {median_ms: 15, lognorm_sigma: 0.5}
    workers: &workers 4
    queue_capacity: &queue 200
    partial_failures: &failures [{start_s: 8, end_s: 11, p_fail: 1}]
  - name: svc-retry
    latency: *latency
    workers: *workers
    queue_capacity: *queue
    partial_failures: *failures
  - name: svc-jittered
    latency: *latency
    workers: *workers
    queue_capacity: *queue
    partial_failures: *failures
clients:
  - name: no_retries
    target_service: svc-none
    replicas: 3
    retry: {type: fixed, max_attempts: 1, delay_ms: 0}
    timeout: &timeout {attempt_ms: 200}
    workload: &load {base_rps: 30, duration_s: 20, rng_seed: 1}
  - name: three_retries
    target_service: svc-retry
    replicas: 3
    retry: {type: fixed, max_attempts: 4, delay_ms: 0}
    timeout: *timeout
    workload: *load
  - name: jittered
    target_service: svc-jittered
    replicas: 3
    retry: {type: jittered, max_attempts: 4, initial_delay_ms: 20, max_delay_ms: 200, jitter_mode: full}
    timeout: *timeout
    workload: *load
"""

#: The shape of their policy files: tenants on one service behind its retry limiter.
TENANTS = """
name: tenants
seed: 5
services:
  - name: svc
    latency: {median_ms: 20, lognorm_sigma: 0.5}
    workers: 6
    queue_capacity: 100
    arolla_retry_budget: {alpha: 0.1}
    partial_failures: [{start_s: 8, end_s: 14, p_fail: 0.5}]
clients:
  - name: plain
    target_service: svc
    retry: {type: fixed, max_attempts: 4, delay_ms: 0}
    timeout: {attempt_ms: 150}
    workload: &load {base_rps: 40, duration_s: 20, rng_seed: 3, load_spikes: [{start_s: 15, end_s: 17, rps_multiplier: 2}]}
  - name: untimed
    target_service: svc
    retry: {type: fixed, max_attempts: 3, delay_ms: 10}
    workload: *load
"""

#: The shape of s04: clients that do not retry, services that retry their own calls.
LAYERS = """
name: layers
seed: 3
services:
  - name: db
    latency: {median_ms: 20, lognorm_sigma: 0.5}
    workers: 6
    queue_capacity: 100
    partial_failures: [{start_s: 8, end_s: 12, p_fail: 0.6}]
  - name: app
    latency: {median_ms: 5, lognorm_sigma: 0.5}
    workers: 8
    queue_capacity: 100
    retry: {type: fixed, max_attempts: 5, delay_ms: 0}
    timeout: {attempt_ms: 300, global_ms: 900}
    dependencies: [{service: db}]
  - name: gw
    latency: {median_ms: 2, lognorm_sigma: 0.5}
    workers: 8
    queue_capacity: 100
    retry: {type: fixed, max_attempts: 5, delay_ms: 0}
    timeout: {attempt_ms: 500, global_ms: 1500}
    dependencies: [{service: app}]
  - name: other-db
    latency: {median_ms: 20, lognorm_sigma: 0.5}
    workers: 6
  - name: other-gw
    latency: {median_ms: 2, lognorm_sigma: 0.5}
    workers: 8
    timeout: {attempt_ms: 500, global_ms: 1500}
    dependencies: [{service: other-db}]
clients:
  - name: through
    target_service: gw
    retry: {type: none, max_attempts: 1}
    timeout: {attempt_ms: 1000, global_ms: 3000}
    workload: &load {base_rps: 50, duration_s: 20, rng_seed: 1}
  - name: elsewhere
    target_service: other-gw
    retry: {type: none, max_attempts: 1}
    timeout: {attempt_ms: 1000, global_ms: 3000}
    workload: *load
"""


def _knobs(episode: Episode) -> RetryKnobs:
    """The agent's side of an episode whose agent is on clients."""
    assert isinstance(episode.agent, RetryKnobs)
    return episode.agent


def _scenario(text: str, agent, **kwargs) -> ArollaScenario:
    """`ArollaScenario.from_file` on a scenario written here."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "scenario.yaml"
        path.write_text(text, encoding="utf-8")
        return ArollaScenario.from_file(path, agent, **kwargs)


def _records(sim, clients) -> tuple[str, int]:
    digest = hashlib.sha256()
    for client in clients:
        for root in client.roots:
            for a in root.attempts:
                digest.update(
                    repr((a.interval.begin, a.interval.end, a.success, a.drop_reason.name)).encode()
                )
    return digest.hexdigest(), sim._seq


def _play(env: RetryEnv, action, seed: int = 4):
    observation, info = env.reset(options={"episode_seed": seed})
    observations = [observation]
    for _ in range(env.episode_steps):
        observation, _, _, _, info = env.step(action)
        observations.append(observation)
    return observations, info


class TestOneArmOfAFile(unittest.TestCase):
    def test_only_the_agents_arm_is_built(self):
        scenario = _scenario(ARMS, "three_retries")
        experiment = scenario.experiment
        self.assertEqual([s.name for s in experiment.services], ["svc-retry"])
        self.assertEqual([c.name for c in experiment.clients], ["three_retries"])
        episode = scenario.build(1)
        self.assertEqual(len(episode.agent.clients), 3)  # every replica
        self.assertEqual(len(episode.clients), 3)
        self.assertEqual(episode.expected_rps, 90)
        self.assertEqual(_knobs(episode).agent_rps, 90)
        self.assertEqual(episode.fault_windows, [(8, 11)])
        self.assertEqual(scenario.horizon_s, 20)

    def test_the_levels_come_from_the_file(self):
        self.assertEqual(
            _scenario(ARMS, "three_retries").grid,
            RetryGrid(
                max_attempts=(1, 2, 3, 4),
                default_max_attempts=4,
                backoff_ms=(0.0, 10.0, 50.0),
                default_backoff_ms=0.0,
                attempt_timeout_ms=(100.0, 200.0, 400.0),
                default_attempt_timeout_ms=200.0,
                default_budget_ratio=1.0,
            ),
        )
        jittered = _scenario(ARMS, "jittered").grid
        self.assertEqual(
            (jittered.backoff_ms, jittered.default_backoff_ms), ((0.0, 10.0, 20.0, 50.0), 20.0)
        )

    def test_the_default_setting_is_the_arm_as_the_file_has_it(self):
        """With no budget in the grid, an episode left at its defaults is their arm, record for record."""
        scenario = _scenario(ARMS, "three_retries")
        scenario = replace(scenario, grid=replace(scenario.grid, budget_ratio=None))
        env = RetryEnv(scenario, keep_records=True)
        action = [knob.default for knob in scenario.knobs if len(knob.values) > 1]
        _play(env, action, seed=11)

        plain = build(scenario.experiment, seed=11, arrivals=11)  # the importer's own build
        plain.run()
        self.assertEqual(
            _records(env.episode.sim, env.episode.clients),
            _records(plain.sim, [client.runtime for client in plain.clients]),
        )
        self.assertGreater(sum(len(c.runtime.roots) for c in plain.clients), 1500)

    def test_the_agent_can_be_several_entries(self):
        scenario = _scenario(ARMS, ["no_retries", "three_retries"])
        self.assertEqual([s.name for s in scenario.experiment.services], ["svc-none", "svc-retry"])
        episode = scenario.build(1)
        self.assertEqual(len(episode.agent.clients), 6)
        self.assertEqual(scenario.grid.default_max_attempts, 4)

    def test_a_scenario_pickles(self):
        scenario = _scenario(ARMS, "three_retries")
        self.assertEqual(pickle.loads(pickle.dumps(scenario)), scenario)

    def test_an_unknown_agent_is_refused(self):
        with self.assertRaisesRegex(ValueError, "neither a client entry .* nor a service"):
            _scenario(ARMS, "four_retries")
        with self.assertRaisesRegex(UnsupportedScenario, "no retry loop of its own"):
            _scenario(ARMS, "svc-retry")  # a plain service: its callers hold the policy


class TestTenantsBehindALimiter(unittest.TestCase):
    def test_the_whole_file_is_kept_and_the_limiter_stays_in_front(self):
        scenario = _scenario(TENANTS, ["plain", "untimed"])
        self.assertEqual(len(scenario.experiment.clients), 2)
        episode = scenario.build(1)
        for client in episode.agent.clients:
            self.assertIsInstance(client.service, AdmissionGate)  # still theirs, still there
            self.assertIsNotNone(client.cfg.tenant)
        self.assertEqual(episode.fault_windows, [(8, 14), (15, 17)])

    def test_one_tenant_can_be_the_agent_among_the_others(self):
        scenario = _scenario(TENANTS, "plain")
        episode = scenario.build(1)
        self.assertEqual(len(scenario.experiment.clients), 2)  # the other shares the service
        self.assertEqual(len(episode.agent.clients), 1)
        self.assertEqual(len(episode.reward_clients), 2)
        self.assertEqual((_knobs(episode).agent_rps, episode.expected_rps), (40, 80))
        (other,) = [c for c in episode.clients if c is not episode.agent.clients[0]]
        layers = list(policy_layers(other.cfg.retry))
        self.assertFalse(any(isinstance(layer, RetryBudgetPolicy) for layer in layers))  # untouched

    def test_clients_with_different_timeouts_keep_their_own(self):
        scenario = _scenario(TENANTS, ["plain", "untimed"])
        self.assertIsNone(scenario.grid.attempt_timeout_ms)
        self.assertNotIn("attempt_timeout_ms", [knob.name for knob in scenario.knobs])
        plain, untimed = scenario.build(1).agent.clients
        self.assertEqual(plain.cfg.timeout_controller.get_attempt_timeout(), 150_000_000)
        self.assertIsNone(untimed.cfg.timeout_controller.get_attempt_timeout())
        self.assertEqual(plain.cfg.server_deadline, "global")  # as the importer built it

    def test_refused_retries_reach_the_agent_and_the_summary(self):
        scenario = _scenario(TENANTS, ["plain", "untimed"])
        env = RetryEnv(scenario)
        observations, info = _play(env, [3, 0, 4])  # four attempts, no backoff, budget 1.0
        other = env.feature_names.index("attempt_other")
        self.assertGreater(max(float(o[other]) for o in observations), 0.02)
        self.assertGreater(info["ep_failed_rejected"], 0)
        self.assertLess(info["ep_fault_success_rate"], info["ep_success_rate"])


class TestServicesThatRetryTheirOwnCalls(unittest.TestCase):
    def test_the_agent_takes_over_the_services_retry_loops(self):
        scenario = _scenario(LAYERS, ["gw", "app"])
        self.assertEqual([s.name for s in scenario.experiment.services], ["db", "app", "gw"])
        self.assertEqual([c.name for c in scenario.experiment.clients], ["through"])
        episode = scenario.build(1)
        self.assertEqual(len(episode.agent.clients), 2)
        self.assertEqual({c.service.cfg.name for c in episode.agent.clients}, {"gw", "app"})
        self.assertEqual(len(episode.reward_clients), 1)  # the client that enters
        self.assertEqual((_knobs(episode).agent_rps, episode.expected_rps), (100, 50))
        self.assertEqual(scenario.grid.max_attempts, (1, 2, 3, 4, 5))
        self.assertEqual(scenario.grid.default_max_attempts, 5)
        self.assertIsNone(scenario.grid.attempt_timeout_ms)  # 500 ms and 300 ms: each its own
        for client in episode.agent.clients:
            backoff = next(
                layer
                for layer in policy_layers(client.cfg.retry)
                if isinstance(layer, FixedBackoffRetryPolicy)
            )
            self.assertEqual(backoff.max_attempts, 5)

    def test_fewer_attempts_at_every_layer_mean_fewer_calls_on_the_database(self):
        calls = {}
        for attempts in (0, 4):  # one attempt, five attempts
            env = RetryEnv(_scenario(LAYERS, ["gw", "app"]))
            _, info = _play(env, [attempts, 0, 4])
            calls[attempts] = env.episode.services["db"].admitted_total
        self.assertGreater(calls[4], 1.2 * calls[0])

    def test_a_loop_that_carries_the_services_limiter_is_not_taken_over(self):
        limited = LAYERS.replace(
            "    retry: {type: fixed, max_attempts: 5, delay_ms: 0}\n    timeout: {attempt_ms: 300",
            "    retry: {type: fixed, max_attempts: 5, delay_ms: 0}\n"
            "    global_retry_budget: {target_rps: 10}\n    timeout: {attempt_ms: 300",
        )
        with self.assertRaisesRegex(UnsupportedScenario, "would drop it"):
            _scenario(limited, ["gw", "app"])
        _scenario(limited, "gw")  # the gateway's loop holds no limiter


class TestTheCut(unittest.TestCase):
    def test_what_is_cut_shares_no_station_with_what_is_kept(self):
        experiment = parse(yaml.safe_load(LAYERS))
        part = agent_part(experiment, ["app"])
        self.assertEqual([s.name for s in part.services], ["db", "app", "gw"])  # and who calls it
        self.assertEqual([c.name for c in part.clients], ["through"])
        whole = agent_part(experiment, ["through", "elsewhere"])
        self.assertEqual(len(whole.services), 5)

    def test_a_grid_can_be_given(self):
        grid = replace(
            file_grid(parse(yaml.safe_load(ARMS)), ["three_retries"]), max_attempts=(1, 4)
        )
        self.assertEqual(_scenario(ARMS, "three_retries", grid=grid).grid.max_attempts, (1, 4))


class TestTheFileAsItIsWritten(unittest.TestCase):
    """`as_written`: their clients left alone, scored the way an agent's episode is."""

    def _summaries(self, text: str, seeds, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scenario.yaml"
            path.write_text(text, encoding="utf-8")
            return as_written(path, seeds, **kwargs)

    def test_an_arm_scores_as_the_environment_scores_it_at_its_defaults(self):
        scenario = _scenario(ARMS, "three_retries")
        scenario = replace(scenario, grid=replace(scenario.grid, budget_ratio=None))
        env = RetryEnv(scenario)
        action = [knob.default for knob in scenario.knobs if len(knob.values) > 1]
        _, info = _play(env, action, seed=11)

        (written,) = self._summaries(ARMS, [11], part_of="three_retries")
        self.assertEqual(written.pop("seed"), 11.0)
        by_env = {name[3:]: value for name, value in info.items() if name.startswith("ep_")}
        np.testing.assert_equal(written, by_env)  # every entry of the summary, exactly
        self.assertLess(written["fault_success_rate"], 0.2)  # three seconds of total failure

    def test_the_whole_file_is_every_arm_together(self):
        (whole,) = self._summaries(ARMS, [3])
        (arm,) = self._summaries(ARMS, [3], part_of="no_retries")
        self.assertAlmostEqual(
            whole["requests"], 3 * arm["requests"], delta=0.1 * whole["requests"]
        )
        self.assertEqual(arm["amplification"], 1.0)
        self.assertGreater(whole["amplification"], 1.05)

    def test_the_clients_behind_a_limiter_and_whose_effort_is_reported(self):
        both = self._summaries(TENANTS, [1, 2])
        self.assertEqual([episode["seed"] for episode in both], [1.0, 2.0])
        self.assertGreater(both[0]["failed_rejected"], 0)  # their limiter turned retries away
        (loops,) = self._summaries(LAYERS, [1], part_of="gw", effort=["gw", "app"])
        (entry,) = self._summaries(LAYERS, [1], part_of="gw")
        self.assertEqual(entry["amplification"], 1.0)  # the client that enters does not retry
        self.assertGreater(loops["amplification"], 1.1)  # the services behind it do
        self.assertEqual(loops["success_rate"], entry["success_rate"])


class TestEpisodesThatVary(unittest.TestCase):
    """`Variation`: a file's failure and load drawn per episode, the same for whoever runs it."""

    WIDE = Variation(p_fail=(0.1, 0.9), onset_s=(6.0, 9.0), length_s=(3.0, 8.0), load=(0.7, 1.3))

    def test_each_episode_draws_its_own_and_a_seed_draws_the_same_again(self):
        scenario = _scenario(TENANTS, ["plain", "untimed"], variation=self.WIDE)
        episodes = [scenario.build(seed) for seed in range(12)]
        (start, end), spike = episodes[3].fault_windows  # the failure, then the file's own spike
        self.assertEqual(spike, (15, 17))
        self.assertTrue(6.0 <= start <= 9.0 and 3.0 <= end - start <= 8.0)
        self.assertEqual(len({tuple(e.fault_windows) for e in episodes}), 12)
        rates = [e.expected_rps for e in episodes]
        self.assertTrue(all(0.7 * 80 <= rps <= 1.3 * 80 for rps in rates))
        self.assertGreater(max(rates) - min(rates), 20)
        again = scenario.build(3)
        self.assertEqual(
            (again.fault_windows, again.expected_rps), (episodes[3].fault_windows, rates[3])
        )
        failure = again.services["svc"].cfg.partial_failures[0]
        self.assertEqual(failure.p_fail, self.WIDE.draw(3)["p_fail"])

    def test_the_agent_can_see_the_load_it_was_given(self):
        """The request rate is measured against the file's rate, not the episode's own.

        Against the episode's own rate every load reads 0.5 and an agent cannot
        tell a quiet service from a busy one. The reward keeps the episode's
        rate: it is a share of the requests that came.
        """
        seen = {}
        for load in (0.7, 1.3):
            scenario = _scenario(
                TENANTS, ["plain", "untimed"], variation=Variation(load=(load, load))
            )
            episode = scenario.build(1)
            self.assertEqual(_knobs(episode).agent_rps, 80)
            self.assertAlmostEqual(episode.expected_rps, load * 80)
            env = RetryEnv(scenario)
            env.reset(options={"episode_seed": 1})
            rates = []
            for _ in range(4):  # before the failure
                observation, *_ = env.step(env.action_space.sample())
                rates.append(float(observation[env.feature_names.index("request_rate")]))
            seen[load] = sum(rates) / len(rates)
        self.assertAlmostEqual(seen[0.7], 0.35, delta=0.05)
        self.assertAlmostEqual(seen[1.3], 0.65, delta=0.05)

    def test_the_draws_depend_on_the_seed_alone(self):
        """Not on which ranges are set: pinning one leaves the others where they were."""
        pinned = replace(self.WIDE, p_fail=(0.75, 0.75), load=None)
        for seed in range(5):
            wide, narrow = self.WIDE.draw(seed), pinned.draw(seed)
            self.assertEqual(narrow["p_fail"], 0.75)
            self.assertNotIn("load", narrow)
            self.assertEqual(
                (narrow["onset_s"], narrow["length_s"]), (wide["onset_s"], wide["length_s"])
            )

    def test_their_file_and_the_agents_get_the_same_failure_and_load(self):
        """Two files that differ in the limiter only: what `as_written` is compared on."""
        limited = parse(yaml.safe_load(TENANTS))
        unlimited = parse(
            yaml.safe_load(TENANTS.replace("    arolla_retry_budget: {alpha: 0.1}\n", ""))
        )
        for seed in (1, 2):
            a, b = self.WIDE.apply(limited, seed), self.WIDE.apply(unlimited, seed)
            self.assertEqual(a.services[0].partial_failures, b.services[0].partial_failures)
            self.assertEqual([c.workload for c in a.clients], [c.workload for c in b.clients])
            self.assertNotEqual(
                a.services[0].partial_failures, limited.services[0].partial_failures
            )

    def test_the_file_as_written_varies_the_same_way(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scenario.yaml"
            path.write_text(TENANTS, encoding="utf-8")
            plain = as_written(path, [1, 2])
            varied = as_written(path, [1, 2], variation=self.WIDE)
            harsh = as_written(path, [1, 2], variation=replace(self.WIDE, p_fail=(0.9, 0.9)))
            # A variation that says what the file says changes nothing.
            same = Variation(
                p_fail=(0.5, 0.5), onset_s=(8.0, 8.0), length_s=(6.0, 6.0), load=(1.0, 1.0)
            )
            np.testing.assert_equal(as_written(path, [1, 2], variation=same), plain)
        self.assertNotEqual(varied[0]["requests"], plain[0]["requests"])  # another load
        for episode, worse in zip(varied, harsh, strict=True):
            self.assertLess(worse["fault_success_rate"], episode["fault_success_rate"])

    def test_what_makes_no_sense_is_refused(self):
        for bad, message in (
            (lambda: Variation(p_fail=(0.5, 1.2)), "probability"),
            (lambda: Variation(p_fail=(0.6, 0.4)), "low <= high"),
            (lambda: Variation(length_s=(0.0, 5.0)), "length > 0"),
            (lambda: Variation(load=(0.0, 1.0)), "factor > 0"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                bad()
        with self.assertRaisesRegex(ValueError, "past the 20 s the file runs for"):
            _scenario(
                TENANTS, "plain", variation=Variation(onset_s=(18.0, 18.0), length_s=(5.0, 5.0))
            )


class TestOnTheCommandLine(unittest.TestCase):
    def test_files_as_written_next_to_the_fixed_settings_of_one(self):
        """Their clients behind their limiter, and an agent's on the same service without one."""
        with tempfile.TemporaryDirectory() as folder:
            limited = Path(folder) / "limited.yaml"
            limited.write_text(TENANTS, encoding="utf-8")
            unlimited = Path(folder) / "unlimited.yaml"
            without = TENANTS.replace("    arolla_retry_budget: {alpha: 0.1}\n", "")
            self.assertNotEqual(without, TENANTS)
            unlimited.write_text(without, encoding="utf-8")
            arguments = [f"arolla:{unlimited}:plain,untimed", "--as-written", str(limited)]
            arguments += [str(unlimited), "--selection-seeds", "1", "--test-seeds", "2"]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                evaluate.main(arguments)
        text = out.getvalue()
        self.assertIn("60 fixed settings on 1 selection episodes", text)
        rows = {line[:34].strip(): line for line in text.splitlines()}
        for name in ("best static", "default", "no retries"):
            self.assertIn(name, rows)
        # Their file with no limiter is not the agent's default: its clients keep their own
        # policies (three attempts with a backoff for one of them, where the default is four).
        self.assertIn("as written: unlimited", rows)
        self.assertIn("as written: limited", rows)
        self.assertNotEqual(rows["as written: limited"][34:], rows["as written: unlimited"][34:])


#: A checkout of https://github.com/eth-easl/Arolla, to run the tests below.
AROLLA_CHECKOUT = os.environ.get("AROLLA_CHECKOUT")
_ALL_TENANTS = ("three_retries", "exponential_backoff_jitter", "circuit_breaker", "retry_budget")
#: The files the environment is wired to first, and who the agent is in each.
SHIPPED = {
    "post_mortem_simulation/s01__buggy_release.yaml": "three_retries",
    "post_mortem_simulation/s02__shaky_foundations.yaml": "three_retries",
    "post_mortem_simulation/s03__thundering_herd.yaml": "three_retries",
    "post_mortem_simulation/s04__internal_retry_rogue.yaml": ("gw-full-retry", "app-full-retry"),
    "snowflake/policy-none.yaml": _ALL_TENANTS,
    "snowflake/policy-cb.yaml": _ALL_TENANTS,
    "snowflake/policy-rb.yaml": _ALL_TENANTS,
    "snowflake/policy-aimd.yaml": _ALL_TENANTS,
    "snowflake/policy-arolla.yaml": _ALL_TENANTS,
}


@unittest.skipUnless(AROLLA_CHECKOUT, "set AROLLA_CHECKOUT to a checkout of Arolla to run this")
class TestTheFilesArollaShips(unittest.TestCase):
    def test_each_one_is_an_environment(self):
        assert AROLLA_CHECKOUT is not None
        folder = Path(AROLLA_CHECKOUT) / "simulator" / "experiments" / "yaml"
        for file, agent in SHIPPED.items():
            with self.subTest(file=file), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scenario = ArollaScenario.from_file(folder / file, agent)
                env = RetryEnv(scenario)
                check_env(env, skip_render_check=True)
                self.assertEqual(
                    [str(w.message) for w in caught if "gymnasium" in str(w.filename)], []
                )
                rng = np.random.default_rng(1)
                observation, _ = env.reset(seed=1)
                rewards = []
                for _ in range(20):
                    observation, reward, _, _, _ = env.step(rng.integers(0, env.knob_levels))
                    self.assertTrue(env.observation_space.contains(observation))
                    rewards.append(reward)
                self.assertGreater(
                    float(np.mean(rewards)), 0.5
                )  # the first 25 s are mostly healthy
                self.assertTrue(scenario.fault_windows())


if __name__ == "__main__":
    unittest.main()
