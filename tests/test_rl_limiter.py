"""The agent as a service's limiter: `rl/limiter.py` and `rl.arolla.ArollaLimiterScenario`.

The scenario is the two tenants of `test_rl_arolla.TENANTS` behind Arolla's
limiter. The clients are never touched: what is replaced is the controller
inside the gate.
"""

import pickle
import tempfile
import unittest
import warnings
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from msim.admission import MixedRefillBudget
from msim.compat.arolla import UnsupportedScenario
from msim.rl import RetryEnv, evaluate
from msim.rl.arolla import ArollaLimiterScenario, Variation, as_written
from msim.rl.limiter import ATTEMPTS, GOODPUT, MEASURES, LimiterKnobs, QuotaGrid
from msim.rl.summary import of_records
from tests.test_rl_arolla import TENANTS

UNLIMITED = TENANTS.replace("    arolla_retry_budget: {alpha: 0.1}\n", "")
#: A grid whose default refuses nothing: the limiter is there and lets everything through.
OPEN = QuotaGrid(shares=(0.0, 1000.0), default_share=1000.0)


def _path(folder: str, text: str) -> Path:
    path = Path(folder) / "scenario.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _scenario(text: str = TENANTS, **kwargs) -> ArollaLimiterScenario:
    with tempfile.TemporaryDirectory() as folder:
        return ArollaLimiterScenario.from_file(_path(folder, text), **kwargs)


def _limiter(env: RetryEnv) -> LimiterKnobs:
    agent = env.episode.agent
    assert isinstance(agent, LimiterKnobs)
    return agent


class TestTheLimiterAsAnEnvironment(unittest.TestCase):
    def test_gymnasiums_checker_and_the_spaces(self):
        scenario = _scenario()
        self.assertEqual((scenario.service, scenario.tenants), ("svc", ("plain", "untimed")))
        self.assertEqual([k.name for k in scenario.knobs], ["quota:plain", "quota:untimed"])
        self.assertEqual(
            scenario.features, tuple(f"{t}:{m}" for t in ("plain", "untimed") for m in MEASURES)
        )
        env = RetryEnv(scenario, slow_s=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_env(env, skip_render_check=True)
        self.assertEqual(env.knob_levels, (10, 10))
        self.assertEqual(env.observation_space.shape, (2 * 5 + 2 * 5 + 2,))
        self.assertEqual(env.feature_names[:10], scenario.features)
        self.assertEqual(env.feature_names[10], "plain:ended~5s")
        self.assertEqual(env.feature_names[-2:], ("knob:quota:plain", "knob:quota:untimed"))
        self.assertEqual(pickle.loads(pickle.dumps(scenario)), scenario)

    def test_a_quota_that_refuses_nothing_is_the_file_without_a_limiter(self):
        """The gate and the counting change nothing: only a refusal does."""
        seeds = [3, 4]
        env = RetryEnv(_scenario(grid=OPEN))
        ours = evaluate.rollout(env, evaluate.constant((1, 1)), seeds)
        with tempfile.TemporaryDirectory() as folder:
            theirs = as_written(_path(folder, UNLIMITED), seeds)
        np.testing.assert_equal(ours, theirs)

    def test_refusing_every_retry_is_the_clients_never_retrying(self):
        env = RetryEnv(_scenario())  # the default share is 0
        (closed,) = evaluate.rollout(env, evaluate.constant((0, 0)), [3])
        (open_,) = evaluate.rollout(RetryEnv(_scenario(grid=OPEN)), evaluate.constant((1, 1)), [3])
        self.assertGreater(closed["failed_rejected"], 0)  # refused, and booked so
        self.assertEqual(open_["failed_rejected"], 0)
        self.assertLess(closed["fault_success_rate"], open_["fault_success_rate"])
        # One try each: half of the 480 requests of the failure succeed, and the 320
        # of the load spike that follows (a fault window too) nearly all do.
        self.assertAlmostEqual(closed["fault_success_rate"], 0.7, delta=0.05)
        # A retry that was refused never reached the service: it is asked for
        # (the clients still made the attempt) and it is not served, so a price
        # on served retries charges an open limiter and not a closed one.
        self.assertGreater(closed["amplification"], 1.0)
        self.assertEqual(closed["retries_served"], 0)
        self.assertGreater(open_["retries_served"], 0)
        (priced,) = evaluate.rollout(
            RetryEnv(_scenario(), retry_cost=0.5), evaluate.constant((0, 0)), [3]
        )
        self.assertEqual(priced["return"], closed["return"])

    def test_the_summary_is_the_one_of_the_records(self):
        scenario = _scenario()
        env = RetryEnv(scenario, keep_records=True)
        (episode,) = evaluate.rollout(env, evaluate.constant((5, 3)), [7])
        built = env.episode
        again = of_records(
            built.reward_clients,
            expected_rps=built.expected_rps,
            fault_windows=built.fault_windows,
            horizon_s=scenario.horizon_s,
        )
        np.testing.assert_equal({key: episode[key] for key in again}, again)

    def test_a_file_with_no_limiter_has_no_gate_to_take_over(self):
        with self.assertRaisesRegex(UnsupportedScenario, "no gate for the agent"):
            _scenario(UNLIMITED)


class TestAnotherRuleInTheirGate(unittest.TestCase):
    """`as_written(..., limiter=)`: their file, their clients, a rule their format cannot name."""

    def test_the_rule_given_decides_in_place_of_the_files(self):
        seeds = [3, 4]
        with tempfile.TemporaryDirectory() as folder:
            path = _path(folder, TENANTS)
            theirs = as_written(path, seeds)
            # A bucket nothing can empty is no limiter at all ...
            open_ = as_written(path, seeds, limiter=partial(MixedRefillBudget, capacity=1e6, t=1e6))
            # ... and one that is never refilled gives out twenty retries and no more.
            shut = as_written(path, seeds, limiter=partial(MixedRefillBudget, r=0.0, t=0.0))
            unlimited = as_written(_path(folder, UNLIMITED), seeds)
        np.testing.assert_equal(open_, unlimited)
        for theirs_, shut_, open__ in zip(theirs, shut, open_, strict=True):
            self.assertEqual(open__["failed_rejected"], 0)
            self.assertGreater(shut_["failed_rejected"], 0)
            self.assertNotEqual(shut_["success_rate"], theirs_["success_rate"])  # another rule
            self.assertLess(shut_["fault_success_rate"], open__["fault_success_rate"])

    def test_a_file_with_no_limiter_has_no_gate_for_it(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(UnsupportedScenario, "no gate to give another rule"),
        ):
            as_written(_path(folder, UNLIMITED), [3], limiter=MixedRefillBudget)


class TestWhatTheLimiterSees(unittest.TestCase):
    def _named(self, env: RetryEnv, observation: np.ndarray) -> dict[str, float]:
        return dict(zip(env.feature_names, observation.tolist(), strict=True))

    def test_its_counts_are_what_passed_the_gate(self):
        """The successes it counts in a window are the requests the reward counts there."""
        env = RetryEnv(_scenario())
        env.reset(options={"episode_seed": 9})
        rates = _limiter(env).rates
        self.assertEqual(rates, {"plain": 40.0, "untimed": 40.0})
        asked = admitted = 0.0
        compared = 0
        for _ in range(env.episode_steps):
            observation, _, _, _, info = env.step((4, 4))  # a fifth of the first attempts
            named = self._named(env, observation)
            self.assertTrue(env.observation_space.contains(observation))
            goodput = sum(named[f"{t}:goodput"] * GOODPUT * rates[t] for t in rates)
            if max(named[f"{t}:goodput"] for t in rates) < 1.0:  # not at its bound (the spike)
                self.assertAlmostEqual(goodput, info["succeeded"], places=3)
                compared += 1
            asked += sum(named[f"{t}:retries_asked"] * ATTEMPTS * rates[t] for t in rates)
            admitted += sum(named[f"{t}:retries_admitted"] * ATTEMPTS * rates[t] for t in rates)
        self.assertGreater(compared, 10)
        self.assertGreater(admitted, 20)
        self.assertGreater(asked, 2 * admitted)  # most retries were turned away

    def test_a_share_of_zero_lets_no_retry_in(self):
        env = RetryEnv(_scenario())
        env.reset(options={"episode_seed": 9})
        seen = [self._named(env, env.step((0, 0))[0]) for _ in range(env.episode_steps)]
        self.assertEqual(
            sum(o["plain:retries_admitted"] + o["untimed:retries_admitted"] for o in seen), 0
        )
        self.assertGreater(sum(o["plain:retries_asked"] for o in seen), 0)
        failing = seen[5]  # 10 s to 11 s: half the calls fail, nothing is retried
        self.assertAlmostEqual(
            failing["plain:goodput"] / failing["plain:ended"], 2 * 0.5, delta=0.5
        )
        self.assertEqual(failing["plain:retry_success"], 0.0)

    def test_the_load_it_was_given_shows(self):
        """Counts are measured against the file's rate, not the episode's own."""
        seen = {}
        for load in (0.7, 1.3):
            scenario = _scenario(variation=Variation(load=(load, load)))
            env = RetryEnv(scenario)
            env.reset(options={"episode_seed": 1})
            self.assertEqual(_limiter(env).rates["plain"], 40.0)
            rows = [self._named(env, env.step((0, 0))[0]) for _ in range(3)]  # before the failure
            seen[load] = (
                float(np.mean([r["plain:goodput"] + r["untimed:goodput"] for r in rows])) / 2
            )
        self.assertAlmostEqual(seen[0.7], 0.35, delta=0.06)
        self.assertAlmostEqual(seen[1.3], 0.65, delta=0.06)

    def test_it_stays_inside_its_space_under_a_storm(self):
        heavy = _scenario(grid=OPEN, variation=Variation(load=(6.0, 6.0), p_fail=(0.9, 0.9)))
        env = RetryEnv(heavy)
        observation, _ = env.reset(options={"episode_seed": 2})
        seen = [observation] + [env.step((1, 1))[0] for _ in range(env.episode_steps)]
        for observation in seen:
            self.assertTrue(env.observation_space.contains(observation), observation)
        self.assertEqual(max(float(o.max()) for o in seen), 1.0)  # something did hit the bound


class TestTheQuotaGrid(unittest.TestCase):
    def test_a_share_reaches_the_tenant_it_is_set_for(self):
        env = RetryEnv(_scenario())
        env.reset(options={"episode_seed": 1})
        _, _, _, _, info = env.step((9, 2))
        self.assertEqual((info["knob_quota:plain"], info["knob_quota:untimed"]), (3.0, 0.05))
        quota = _limiter(env).quota
        self.assertEqual((quota.share_of("plain"), quota.share_of("untimed")), (3.0, 0.05))
        with self.assertRaisesRegex(KeyError, "no quota named"):
            _limiter(env).apply({"max_attempts": 2})
        with self.assertRaisesRegex(KeyError, "no quota named"):
            _limiter(env).apply({"quota:nobody": 0.1})

    def test_a_floor_keeps_the_lowest_level_from_being_blind(self):
        # Every retry refused: no retry ends, and their success share says nothing.
        (blind,) = evaluate.rollout(RetryEnv(_scenario()), evaluate.constant((0, 0)), [3])
        self.assertEqual(blind["retries_served"], 0)
        # With a floor of one, a retry per tenant and second gets through at level 0.
        probing = RetryEnv(_scenario(grid=QuotaGrid(floor=1)))
        (seen,) = evaluate.rollout(probing, evaluate.constant((0, 0)), [3])
        self.assertGreater(seen["retries_served"], 0)
        self.assertLess(seen["retries_served"], 2 * 2 * _scenario().horizon_s)  # two tenants
        self.assertEqual(_limiter(probing).quota.floor, 1)

    def test_with_rest_the_quota_is_at_its_default_while_no_retry_asks(self):
        env = RetryEnv(_scenario(grid=QuotaGrid(rest=True)))
        env.reset(options={"episode_seed": 1})
        wide = (9, 9)  # a share of 3 for both tenants
        # Nothing failed during the warm-up, so nothing asked to retry: the
        # quota rests, whatever is chosen, and the agent is shown that.
        observation, _, _, _, info = env.step(wide)
        shown = dict(zip(env.feature_names, observation, strict=True))
        self.assertEqual((shown["knob:quota:plain"], shown["knob:quota:untimed"]), (0.0, 0.0))
        self.assertEqual((info["knob_quota:plain"], info["knob_quota:untimed"]), (0.0, 0.0))
        self.assertEqual(_limiter(env).quota.share_of("plain"), 0.0)
        # Once retries ask (the failure), the share chosen is the share in force.
        for _ in range(env.episode_steps - 1):
            observation, _, _, _, info = env.step(wide)
            if info["knob_quota:plain"] == 3.0:
                break
        self.assertEqual((info["knob_quota:plain"], info["knob_quota:untimed"]), (3.0, 3.0))
        self.assertEqual(_limiter(env).quota.share_of("untimed"), 3.0)
        shown = dict(zip(env.feature_names, observation, strict=True))
        self.assertEqual(shown["knob:quota:plain"], 1.0)
        # Without it the share chosen holds from the first step on.
        plain = RetryEnv(_scenario())
        plain.reset(options={"episode_seed": 1})
        _, _, _, _, info = plain.step(wide)
        self.assertEqual(info["knob_quota:plain"], 3.0)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "not one of its levels"):
            QuotaGrid(default_share=0.07)
        with self.assertRaises(ValueError):
            QuotaGrid(shares=(-0.1, 0.0))
        with self.assertRaises(ValueError):
            replace(QuotaGrid(), window_s=0.0)


if __name__ == "__main__":
    unittest.main()
