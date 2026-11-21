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

from msim.admission import AdmissionOutcome, AdmissionRequest, MixedRefillBudget
from msim.compat.arolla import UnsupportedScenario
from msim.core import DropReason
from msim.rl import RetryEnv, evaluate
from msim.rl.arolla import ArollaLimiterScenario, Variation, as_written
from msim.rl.limiter import (
    ATTEMPTS,
    GOODPUT,
    IN_FLIGHT,
    MEASURES,
    LimiterKnobs,
    QuotaGrid,
    _AttemptBound,
    _GateTap,
)
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


class TestAttemptsInFlight(unittest.TestCase):
    """`QuotaGrid(in_flight=True)`: the one entry that is not a count of the window."""

    #: A queue long enough to hold a burst, and a burst the service cannot serve.
    BURST = TENANTS.replace("queue_capacity: 100", "queue_capacity: 5000").replace(
        "rps_multiplier: 2", "rps_multiplier: 8"
    )

    def test_the_gate_sums_its_attempts_in_flight_over_time(self):
        tap = _GateTap()

        def let_in(now: int, outstanding: int) -> None:
            tap.commit(AdmissionRequest(now, "a", 1, False, outstanding, 0))

        let_in(0, 0)
        let_in(10, 1)
        tap.observe(AdmissionOutcome(30, "a", 1, False, True, DropReason.NONE))
        # One attempt for 10 ns, two for 20, one for the 20 that are left.
        self.assertEqual(tap.take_flight(50), 1 * 10 + 2 * 20 + 1 * 20)
        self.assertEqual(tap.take_flight(60), 10)  # read, it starts again; one is still in flight
        # An attempt that was cancelled leaves the gate unobserved: its count is the gate's.
        let_in(60, 0)
        self.assertEqual(tap.in_flight, 1)

    def test_it_is_one_entry_more_and_only_when_asked_for(self):
        self.assertNotIn(IN_FLIGHT, RetryEnv(_scenario()).feature_names)
        grid = QuotaGrid(in_flight=True)
        env = RetryEnv(_scenario(grid=grid), slow_s=10.0)
        self.assertEqual(grid.features(("a", "b"))[-1], IN_FLIGHT)
        self.assertEqual(env.feature_names.count(IN_FLIGHT), 1)
        self.assertIn(f"{IN_FLIGHT}~10s", env.feature_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_env(env, skip_render_check=True)

    def test_it_is_near_zero_until_attempts_pile_up(self):
        grid = replace(OPEN, in_flight=True)
        env = RetryEnv(_scenario(self.BURST, grid=grid))
        at = env.feature_names.index(IN_FLIGHT)
        observation, _ = env.reset(options={"episode_seed": 1})
        shown = [float(observation[at])]  # the warm-up, then a window a second
        shown += [float(env.step((1, 1))[0][at]) for _ in range(env.episode_steps)]
        # Seconds 5 to 15: all is well, then half the attempts fail and every
        # retry is let in, which the service has room for.
        self.assertLess(max(shown[:11]), 0.4, shown)
        # Seconds 15 to 17: eight times the load, more than the service can serve.
        self.assertGreater(max(shown[11:13]), 0.8, shown)

    def test_without_a_window_at_rest_there_is_nothing_to_compare_with(self):
        # Overloaded from the first second: retries ask in every window.
        heavy = _scenario(
            self.BURST,
            grid=replace(OPEN, in_flight=True),
            variation=Variation(load=(8.0, 8.0), p_fail=(0.5, 0.5)),
        )
        env = RetryEnv(heavy)
        at = env.feature_names.index(IN_FLIGHT)
        observation, _ = env.reset(options={"episode_seed": 1})
        shown = [float(observation[at])] + [float(env.step((1, 1))[0][at]) for _ in range(5)]
        self.assertEqual(shown, [0.0] * 6)


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

    def test_with_step_up_the_quota_climbs_a_level_a_window(self):
        grid = QuotaGrid(shares=(0.0, 0.1, 0.2, 0.4), step_up=1)
        env = RetryEnv(_scenario(grid=grid))
        env.reset(options={"episode_seed": 1})
        top = (3, 3)
        climbed = []
        for _ in range(4):
            observation, _, _, _, info = env.step(top)
            climbed.append(info["knob_quota:plain"])
            # The agent is shown where it stands, not what it asked for.
            shown = dict(zip(env.feature_names, observation, strict=True))
            self.assertAlmostEqual(shown["knob:quota:plain"], grid.shares.index(climbed[-1]) / 3)
            self.assertEqual(_limiter(env).quota.share_of("untimed"), climbed[-1])
        self.assertEqual(climbed, [0.1, 0.2, 0.4, 0.4])
        # Down as far as it likes, at once; and one tenant's place is its own.
        _, _, _, _, info = env.step((0, 3))
        self.assertEqual((info["knob_quota:plain"], info["knob_quota:untimed"]), (0.0, 0.4))
        _, _, _, _, info = env.step((3, 1))
        self.assertEqual((info["knob_quota:plain"], info["knob_quota:untimed"]), (0.1, 0.1))
        with self.assertRaisesRegex(ValueError, "step_up"):
            QuotaGrid(step_up=-1)

    def test_with_hold_above_no_quota_rises_while_the_service_is_crowded(self):
        grid = QuotaGrid(shares=(0.0, 0.5, 1000.0), default_share=0.5, hold_above=0.5)
        env = RetryEnv(_scenario(grid=grid))
        env.reset(options={"episode_seed": 1})
        limiter = _limiter(env)
        # Crowded (the entry is measured whether or not the agent is shown it):
        # nothing rises, a share may still fall, and the agent is shown the level in force.
        limiter._entry = 0.9
        observation, _, _, _, info = env.step((2, 0))
        self.assertEqual((info["knob_quota:plain"], info["knob_quota:untimed"]), (0.5, 0.0))
        shown = dict(zip(env.feature_names, observation, strict=True))
        self.assertEqual((shown["knob:quota:plain"], shown["knob:quota:untimed"]), (0.5, 0.0))
        self.assertNotIn(IN_FLIGHT, env.feature_names)
        # Calm again (this scenario has room for every retry): it rises.
        self.assertLess(limiter._entry, 0.5)
        _, _, _, _, info = env.step((2, 1))
        self.assertEqual((info["knob_quota:plain"], info["knob_quota:untimed"]), (1000.0, 0.5))
        with self.assertRaisesRegex(ValueError, "hold_above"):
            QuotaGrid(hold_above=1.5)

    def test_with_an_in_flight_cap_retries_are_refused_while_attempts_pile_up(self):
        # Eight times the load in seconds 15 to 17, a queue that holds it, every retry let in.
        burst = TestAttemptsInFlight.BURST

        def episode(grid: QuotaGrid) -> tuple[dict, list[int]]:
            """The episode's summary, and the retries the cap refused by the end of each step."""
            env = RetryEnv(_scenario(burst, grid=grid))
            env.reset(options={"episode_seed": 1})
            cap = _limiter(env).cap
            if cap is not None:
                # The warm-up was at rest: there is a limit, and it is not below the floor.
                assert cap.limit is not None
                self.assertGreaterEqual(cap.limit, 8.0)
            refused, info = [], {}
            for _ in range(env.episode_steps):
                _, _, _, _, info = env.step((1, 1))
                refused.append(0 if cap is None else cap.refused)
            return info, refused

        without, _ = episode(OPEN)
        capped, refused = episode(replace(OPEN, in_flight_cap=2.5))
        # Seconds 5 to 15: half the attempts fail from 8 on, and the service
        # has room for every retry: (almost) none is refused. Then the burst.
        self.assertLess(refused[9], 20)
        self.assertGreater(refused[-1] - refused[9], 100)
        # Fewer retries reach a service that has no room for them.
        self.assertLess(capped["ep_retries_served"], without["ep_retries_served"])
        with self.assertRaisesRegex(ValueError, "in_flight_cap"):
            QuotaGrid(in_flight_cap=0.5)

    def test_a_caller_that_asks_more_often_is_not_shown_to_ask_more(self):
        # `plain` tries eleven times instead of four, every retry refused (a share of 0).
        eager = TENANTS.replace("max_attempts: 4, delay_ms: 0", "max_attempts: 11, delay_ms: 0")
        shown = {}
        for bound in (None, 3.0):
            grid = QuotaGrid(asked_per_failure=bound)
            env = RetryEnv(_scenario(eager, grid=grid))
            env.reset(options={"episode_seed": 1})
            names = env.feature_names
            windows = [dict(zip(names, env.step((0, 0))[0], strict=True)) for _ in range(8)]
            shown[bound] = max(window["plain:retries_asked"] for window in windows)
            failing = max(windows, key=lambda window: window["plain:retries_asked"])
            # What `plain` sends when all is well is 40 a second; the scale is 4 x 40.
            failed = failing["plain:ended"] * 160 - failing["plain:goodput"] * 80
            if bound is not None:
                self.assertLessEqual(failing["plain:retries_asked"] * 160, 3 * failed + 1e-3)
        # Ten asks per failed request as counted (off the scale), three as shown.
        self.assertEqual(shown[None], 1.0)
        self.assertLess(shown[3.0], 0.6)

    def test_with_confirm_a_quota_rises_once_a_whole_window_of_the_failure_was_seen(self):
        # The failure begins at second 8: the window of step 3 (the warm-up is 5 s).
        def first_open(grid: QuotaGrid) -> int:
            env = RetryEnv(_scenario(grid=grid))
            env.reset(options={"episode_seed": 1})
            for step in range(env.episode_steps):
                info = env.step((1, 1))[4]
                if info["knob_quota:plain"] > 0:
                    return step
            return -1

        ladder = QuotaGrid(shares=(0.0, 1000.0), rest=True)
        # At rest until retries ask (they do in the window of step 3), then open ...
        self.assertEqual(first_open(ladder), 4)
        # ... and a window later when a second window of asking is waited for.
        self.assertEqual(first_open(replace(ladder, confirm=True)), 5)

    def test_with_max_attempt_a_caller_gets_no_more_retries_per_request(self):
        bound = _AttemptBound(4)
        self.assertTrue(bound.check(AdmissionRequest(0, "a", 1, False, 0, 0)))
        self.assertTrue(bound.check(AdmissionRequest(0, "a", 4, True, 0, 0)))
        self.assertFalse(bound.check(AdmissionRequest(0, "a", 5, True, 0, 0)))
        # `plain` tries eleven times; every retry the quota is asked for is let in.
        eager = TENANTS.replace("max_attempts: 4, delay_ms: 0", "max_attempts: 11, delay_ms: 0")
        served = {}
        for limit in (None, 4):
            env = RetryEnv(_scenario(eager, grid=replace(OPEN, max_attempt=limit)))
            env.reset(options={"episode_seed": 1})
            info = {}
            for _ in range(env.episode_steps):
                info = env.step((1, 1))[4]
            served[limit] = info["ep_retries_served"]
        self.assertLess(served[4], served[None])
        with self.assertRaisesRegex(ValueError, "max_attempt"):
            QuotaGrid(max_attempt=1)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "not one of its levels"):
            QuotaGrid(default_share=0.07)
        with self.assertRaises(ValueError):
            QuotaGrid(shares=(-0.1, 0.0))
        with self.assertRaises(ValueError):
            replace(QuotaGrid(), window_s=0.0)


if __name__ == "__main__":
    unittest.main()
