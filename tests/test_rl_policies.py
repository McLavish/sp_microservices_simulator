import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from rl.env import ActionSpec, RewardWeights
from rl.model_io import check_action_spec, save_action_spec, sidecar_path
from rl.policies.action_space import ActionGrid
from rl.policies.aimd import AIMDPolicy
from rl.policies.bayesian import BayesianPolicy
from rl.policies.model_based import (
    LatencyBuffer,
    ModelBasedPolicy,
    _action_cost_key,
    _argmax_over_grid,
    _simulate_action,
)
from rl.spec import ActionValues


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
        # Pin the DECISION, not its type: every observed latency (12-18 ms)
        # exceeds the 10 ms grid entry, so the 20 ms timeout is the only one
        # that can succeed and retrying is free under it.
        self.assertEqual(action, (1, 1, 0))

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
        # The queue variant fits the SERVICE stream, and this info carries none,
        # so there is nothing to fit and nothing held: the decision must be the
        # mid-grid warm-up, never the grid corner (shortest timeout, no retry).
        self.assertEqual(action, self.grid.midpoint_indices())
        self.assertNotEqual(action, self.grid.indices[0])

    def test_queue_delay_matches_mm1_closed_form(self):
        # Service moments with E[S] = 1 ms and E[S^2] = 2 ms^2 (CV^2 = 1, the
        # exponential moments), lambda = 0.5/ms from the attempt stream, c = 1:
        # the Allen-Cunneen approximation must reduce to the exact M/M/1 result
        # Wq = rho / (1 - rho) * E[S] = 1.0 ms.
        policy = ModelBasedPolicy(
            grid=self.grid,
            include_queue_delay=True,
            queue_workers=1,
            step_duration_s=1.0,
        )
        policy._service_buffer.add([0.0, 2.0] * 250)  # mean 1, mean-square 2
        wq = policy._estimate_queue_delay({"attempt_latencies_ms": [1.0] * 500})
        self.assertAlmostEqual(wq, 1.0, places=9)

    def test_queue_delay_lambda_comes_from_the_sampled_attempt_stream(self):
        # In chain scenarios the samples describe the B->C edge; using
        # top-level window completions for lambda underestimates the modeled
        # queue's arrival rate by the whole retry amplification.
        policy = ModelBasedPolicy(
            grid=self.grid, include_queue_delay=True, queue_workers=1, step_duration_s=1.0
        )
        policy._service_buffer.add([0.0, 2.0] * 250)
        light = policy._estimate_queue_delay(
            {"attempt_latencies_ms": [1.0] * 100, "summary": {"completed": 100.0}}
        )
        heavy = policy._estimate_queue_delay(
            {"attempt_latencies_ms": [1.0] * 800, "summary": {"completed": 100.0}}
        )
        self.assertGreater(heavy, light)  # summary-based lambda would tie these

    def test_queue_delay_zero_without_traffic(self):
        policy = ModelBasedPolicy(grid=self.grid, include_queue_delay=True, queue_workers=1)
        self.assertEqual(policy._estimate_queue_delay({"attempt_latencies_ms": []}), 0.0)
        self.assertEqual(policy._estimate_queue_delay({}), 0.0)

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
        # Same crafted stream as the model-based case: the posterior puts no
        # mass below 10 ms, so only the 20 ms timeout can succeed.
        self.assertEqual(action, (1, 1, 0))

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
        # 20% deadline drops is unhealthy, so the additive-increase arm fires on
        # both adjusted knobs: one timeout step up and one more attempt.
        self.assertEqual(action, (1, 1, 0))

    def test_aimd_multiplicative_decrease_snaps_to_grid(self):
        spec = ActionSpec(
            attempt_timeout_ms=[10, 30, 50, 80],
            max_attempts=[1],
            backoff_ms=[0],
        )
        grid = ActionGrid(spec)
        policy = AIMDPolicy(grid=grid, initial_timeout_ms=80)
        healthy = {"summary": {"completed": 10, "success_rate": 1.0, "deadline_drop_rate": 0.0}}

        action = policy.act(np.zeros(15, dtype=np.float32), healthy)
        self.assertEqual(spec.attempt_timeout_ms[action[0]], 30)  # 80 * 0.5 = 40 -> snap down
        action = policy.act(np.zeros(15, dtype=np.float32), healthy)
        self.assertEqual(spec.attempt_timeout_ms[action[0]], 10)  # 30 * 0.5 = 15 -> snap down
        action = policy.act(np.zeros(15, dtype=np.float32), healthy)
        self.assertEqual(spec.attempt_timeout_ms[action[0]], 10)  # stays at the floor

    def test_aimd_rejects_invalid_decrease_factor(self):
        with self.assertRaises(ValueError):
            AIMDPolicy(grid=self.grid, timeout_decrease_factor=1.0)

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
        # The extra dimension must carry the requested budget (100 ms -> index
        # 0), not shift the knobs it was appended to.
        self.assertEqual(action, (1, 0, 0, 0))


class TestServiceMoments(unittest.TestCase):
    def test_queue_variant_derives_moments_from_service_stream(self):
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0]))
        policy = ModelBasedPolicy(grid=grid, include_queue_delay=True)
        policy.act(
            obs=np.zeros(1),
            info={
                "attempt_latencies_ms": [40.0] * 50,  # totals: include queue wait
                "attempt_censored": [False] * 50,
                "attempt_service_ms": [20.0] * 50,  # queue-exclusive
                "summary": {"completed": 50.0, "retries_per_root": 0.0},
            },
        )
        mean_s, mean_s2, count = policy._service_moments()
        self.assertEqual(count, 50)
        self.assertAlmostEqual(mean_s, 20.0)
        self.assertAlmostEqual(mean_s2, 400.0)

    def test_queue_variant_fits_service_stream_not_totals(self):
        # Double-count guard: with include_queue_delay the latency model must
        # be fitted on queue-exclusive service times (queue wait is added back
        # via Erlang-C); the plain variant keeps fitting end-to-end totals.
        info = {
            "attempt_latencies_ms": [40.0] * 50,
            "attempt_censored": [False] * 50,
            "attempt_service_ms": [20.0] * 50,
            "attempt_service_censored": [False] * 50,
            "summary": {"completed": 50.0, "retries_per_root": 0.0},
        }
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0]))
        queue_policy = ModelBasedPolicy(grid=grid, include_queue_delay=True)
        queue_policy.act(np.zeros(1), info)
        self.assertAlmostEqual(float(queue_policy._fit_buffer().data().mean()), 20.0)
        plain_policy = ModelBasedPolicy(grid=grid)
        plain_policy.act(np.zeros(1), info)
        self.assertAlmostEqual(float(plain_policy._fit_buffer().data().mean()), 40.0)


class TestQueueModelGroundTruth(unittest.TestCase):
    """The env's own queue measurements must beat the sampled-client fallback.

    `modeled_admitted` counts every attempt admitted at the modelled service and
    `modeled_workers` is its real concurrency; the fallbacks (the sampled edge's
    attempt count, the configured `queue_workers`) undercount a shared queue and
    describe a scenario-independent default service.
    """

    def setUp(self):
        self.grid = ActionGrid(
            ActionSpec(attempt_timeout_ms=[30, 80], max_attempts=[1], backoff_ms=[0])
        )

    def _policy(self):
        return ModelBasedPolicy(
            grid=self.grid,
            include_queue_delay=True,
            queue_workers=16,
            step_duration_s=1.0,
        )

    @staticmethod
    def _info(admitted=None, workers=None):
        # Constant 20 ms occupancy: E[S] = 20 ms, CV^2 = 0. Only 5 sampled edge
        # attempts, so the fallback lambda is 20x smaller than the real one.
        info = {
            "attempt_latencies_ms": [20.0] * 5,
            "attempt_censored": [False] * 5,
            "attempt_service_ms": [20.0] * 50,
            "attempt_service_censored": [False] * 50,
            "summary": {"completed": 5.0, "retries_per_root": 0.0},
        }
        if admitted is not None:
            info["modeled_admitted"] = admitted
        if workers is not None:
            info["modeled_workers"] = workers
        return info

    def test_modeled_admitted_is_preferred_over_the_sampled_attempt_count(self):
        policy = self._policy()
        policy._service_buffer.add([20.0] * 50)
        # c = 1, lambda = 30/1000 per ms, E[S] = 20 ms -> rho = 0.6 and the
        # M/M/1 wait is rho/(1-rho) * E[S] = 30 ms, halved by (1 + CV^2)/2.
        self.assertAlmostEqual(policy._estimate_queue_delay(self._info(30, 1)), 15.0, places=9)
        # The fallback would see 5 attempts (lambda = 0.005/ms, rho = 0.1), a
        # 13x smaller wait: 0.1/(1 - 0.1) * 20 ms / 2.
        fallback = policy._estimate_queue_delay(self._info(workers=1))
        self.assertAlmostEqual(fallback, 10.0 / 9.0, places=9)

    def test_modeled_workers_overrides_the_configured_default(self):
        policy = self._policy()  # queue_workers=16, the scenario-blind default
        policy._service_buffer.add([20.0] * 50)
        self.assertAlmostEqual(policy._estimate_queue_delay(self._info(30, 1)), 15.0, places=9)
        # Same offered load spread over 16 workers barely queues at all.
        self.assertLess(policy._estimate_queue_delay(self._info(30)), 1e-9)

    def test_queue_delay_moves_the_selected_timeout(self):
        """The delay must reach the argmax, not just the estimator.

        Identical service stream and identical sampled-attempt count: only the
        modelled queue's arrival count differs. Under the light load a 20 ms
        service fits inside the 30 ms timeout, so the cheaper knob wins; under
        the heavy one the 15 ms queueing delay pushes it past 30 ms and only the
        80 ms timeout can succeed.
        """
        light = self._policy().act(np.zeros(1), self._info(1, 1))
        heavy = self._policy().act(np.zeros(1), self._info(30, 1))
        self.assertEqual(light, (0, 0, 0))
        self.assertEqual(heavy, (1, 0, 0))


class TestTraceTimeAxis(unittest.TestCase):
    def test_uses_step_duration_column(self):
        import pandas as pd

        from plotting.rl_traces_over_time import _step_seconds

        df = pd.DataFrame({"step": [0, 1, 2], "step_duration_s": [2.0, 2.0, 2.0]})
        self.assertEqual(_step_seconds(df), 2.0)

    def test_legacy_csv_defaults_to_one_second(self):
        import pandas as pd

        from plotting.rl_traces_over_time import _step_seconds

        df = pd.DataFrame({"step": [0, 1, 2]})
        self.assertEqual(_step_seconds(df), 1.0)


class TestNearestIndicesGlobalTimeout(unittest.TestCase):
    def _grid(self, global_list):
        return ActionGrid(
            ActionSpec(
                attempt_timeout_ms=[10, 30],
                max_attempts=[1],
                backoff_ms=[0],
                global_timeout_ms=global_list,
            )
        )

    def test_none_request_maps_to_the_none_entry(self):
        grid = self._grid([500, 1000, None])
        self.assertEqual(grid.nearest_indices(10, 1, 0, global_timeout_ms=None), (0, 0, 0, 2))

    def test_numeric_request_ignores_none_entries(self):
        grid = self._grid([None, 500, 1000])
        self.assertEqual(grid.nearest_indices(10, 1, 0, global_timeout_ms=600), (0, 0, 0, 1))

    def test_none_request_without_none_entry_raises(self):
        grid = self._grid([500, 1000])
        with self.assertRaises(ValueError):
            grid.nearest_indices(10, 1, 0, global_timeout_ms=None)


class TestActionSpecSidecar(unittest.TestCase):
    def setUp(self):
        self.spec = ActionSpec(
            attempt_timeout_ms=[10, 30],
            max_attempts=[1, 2],
            backoff_ms=[0, 20],
            global_timeout_ms=[500, None],
        )

    def test_roundtrip_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            save_action_spec(path, self.spec, step_duration_s=1.0, fault_scenario="chain")
            check_action_spec(path, self.spec)  # no raise
            stored = json.loads(Path(sidecar_path(path)).read_text())
            self.assertEqual(stored["step_duration_s"], 1.0)
            self.assertEqual(stored["fault_scenario"], "chain")

    def test_step_duration_mismatch_is_rejected(self):
        # Both production callers pass the step duration; a model trained on 1 s
        # windows reads every rate feature and the episode-time fraction on a
        # different scale under 0.1 s windows.
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            save_action_spec(path, self.spec, step_duration_s=1.0)
            check_action_spec(path, self.spec, 1.0)  # matching: no raise
            with self.assertRaises(ValueError) as ctx:
                check_action_spec(path, self.spec, 0.1)
            self.assertIn("step_duration_s", str(ctx.exception))

    def test_legacy_sidecar_without_a_step_duration_is_accepted(self):
        # Sidecars written before the field existed carry no step duration:
        # they must still pass the grid check rather than raise on the guard.
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            save_action_spec(path, self.spec, step_duration_s=1.0)
            sidecar = Path(sidecar_path(path))
            payload = json.loads(sidecar.read_text())
            del payload["step_duration_s"]
            sidecar.write_text(json.dumps(payload))
            check_action_spec(path, self.spec, 0.1)  # no raise

    def test_different_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            save_action_spec(path, self.spec, step_duration_s=1.0)
            mutated = ActionSpec(
                attempt_timeout_ms=[10, 40],  # valid grid, different from the saved one
                max_attempts=[1, 2],
                backoff_ms=[0, 20],
                global_timeout_ms=[500, None],
            )
            with self.assertRaises(ValueError):
                check_action_spec(path, mutated)

    def test_sidecar_survives_sb3_implicit_zip_extension(self):
        # SB3 save("model") writes model.zip; load("model.zip") must find the
        # sidecar written at save time regardless of which form was used.
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "model")
            save_action_spec(base, self.spec, step_duration_s=1.0)
            check_action_spec(base + ".zip", self.spec)  # no raise, no warn-skip
            mutated = ActionSpec(
                attempt_timeout_ms=[10, 40],  # valid grid, different from the saved one
                max_attempts=[1, 2],
                backoff_ms=[0, 20],
                global_timeout_ms=[500, None],
            )
            with self.assertRaises(ValueError):
                check_action_spec(base + ".zip", mutated)

    def test_missing_sidecar_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                check_action_spec(path, self.spec)
            self.assertTrue(any("sidecar" in str(w.message) for w in caught))


class TestZeroSampleRobustness(unittest.TestCase):
    def _info(self, values):
        return {
            "attempt_latencies_ms": values,
            "attempt_censored": [False] * len(values),
            "summary": {"completed": float(len(values)), "retries_per_root": 0.0},
        }

    def test_bayesian_ignores_nonpositive_samples(self):
        # Zeros must contribute nothing: same decision as the clean stream.
        # Pre-fix, log(0) collapses the posterior to NaN and the argmax
        # degenerates to the first grid entry (timeout 10 ms, which the clean
        # fit rejects because every sample exceeds it).
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[10, 50], max_attempts=[1], backoff_ms=[0]))
        with_zeros = BayesianPolicy(grid=grid).act(
            np.zeros(1), self._info([0.0, 0.0, 20.0, 25.0, 22.0, 24.0])
        )
        clean = BayesianPolicy(grid=grid).act(np.zeros(1), self._info([20.0, 25.0, 22.0, 24.0]))
        self.assertEqual(with_zeros, clean)
        self.assertEqual(clean, (1, 0, 0))

    def test_model_based_survives_all_zero_buffer(self):
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0]))
        policy = ModelBasedPolicy(grid=grid)
        action = policy.act(np.zeros(1), self._info([0.0, 0.0, 0.0]))
        self.assertEqual(action, grid.indices[0])


class _ConstantModel:
    """Minimal LatencyModel-protocol stub returning a constant latency."""

    def __init__(self, value_ms: float):
        self._value = value_ms

    def fit(self, samples):
        return self

    def logpdf(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def cdf(self, x):
        return (np.asarray(x, dtype=float) >= self._value).astype(float)

    def quantile(self, q: float) -> float:
        return self._value

    def sample(self, rng, size):
        return np.full(size, self._value)

    def params(self) -> dict[str, float]:
        return {"value_ms": self._value}

    def log_likelihood(self, samples) -> float:
        return 0.0


class TestSurrogateGlobalDeadline(unittest.TestCase):
    def _score(self, sample_ms, attempt_timeout_ms, global_timeout_ms):
        weights = RewardWeights(
            success=1.0,
            deadline_drop=1.0,
            queue_drop=0.0,
            server_drop=0.0,
            circuit_open_drop=0.0,
            p95_latency_ms=0.0,
            retries=0.0,
        )
        action = ActionValues(
            attempt_timeout_ms=attempt_timeout_ms,
            max_attempts=1,
            backoff_ms=0,
            global_timeout_ms=global_timeout_ms,
        )
        return _simulate_action(
            _ConstantModel(sample_ms),
            action,
            weights,
            np.random.default_rng(0),
            n_roots=10,
            queue_delay_ms=0.0,
            queue_delay_mode="deterministic",
        )

    def test_sample_beyond_global_budget_is_a_deadline_drop(self):
        # 80 ms sample, 100 ms attempt timeout, 50 ms global: the real caller
        # caps the attempt at min(remaining global, attempt timeout) and
        # requires strictly-before completion -> deadline drop, reward -1.
        self.assertAlmostEqual(self._score(80.0, 100, 50), -1.0)

    def test_sample_within_both_budgets_succeeds(self):
        self.assertAlmostEqual(self._score(30.0, 100, 200), 1.0)


class TestActionGridUnspecifiedGlobalTimeout(unittest.TestCase):
    """`nearest_indices` must distinguish "no opinion" from "explicitly None".

    FixedValuePolicy and AIMDPolicy both default the argument, so conflating the
    two made them raise on any spec whose global-timeout grid is all numeric.
    """

    def test_omitted_global_timeout_on_numeric_only_grid(self):
        grid = ActionGrid(ActionSpec([10, 50], [1, 2], [0, 50], [100, 200]))
        self.assertEqual(len(grid.nearest_indices(30, 2, 20)), 4)

    def test_omitted_prefers_the_none_entry_when_present(self):
        spec = ActionSpec([10, 50], [1, 2], [0, 50], [None, 200])
        grid = ActionGrid(spec)
        idx = grid.nearest_indices(30, 2, 20)
        global_timeouts = spec.global_timeout_ms
        assert global_timeouts is not None  # narrow for the type checker
        self.assertIsNone(global_timeouts[idx[3]])

    def test_explicit_none_still_requires_a_none_entry(self):
        grid = ActionGrid(ActionSpec([10, 50], [1, 2], [0, 50], [100, 200]))
        with self.assertRaises(ValueError):
            grid.nearest_indices(30, 2, 20, None)


class TestEVTNumericStability(unittest.TestCase):
    def test_tight_dispersion_does_not_overflow(self):
        """u**alpha in probability space RAISES OverflowError once the Hill
        estimate is large; the density must be computed in log space."""
        import numpy as np

        from rl.models.evt_tail import EVTTailModel

        rng = np.random.default_rng(0)
        x = rng.lognormal(4.1, 0.012, 5000)
        model = EVTTailModel(threshold_quantile=0.95).fit(x)
        lp = model.logpdf(x[:100])
        self.assertTrue(np.all(np.isfinite(lp)))

    def test_body_is_fitted_without_truncation_bias(self):
        """The body sample is right-truncated at u, so a plain MLE biases both
        parameters toward zero and every body quantile comes out low."""
        import numpy as np

        from rl.models.evt_tail import EVTTailModel

        rng = np.random.default_rng(0)
        x = rng.lognormal(np.log(50), 0.8, 50_000)
        model = EVTTailModel(threshold_quantile=0.95).fit(x)
        # Within Monte-Carlo error of the true parameters; the truncated fit
        # gave mu=3.82 / sigma=0.72 against a true 3.912 / 0.8.
        self.assertAlmostEqual(model.body.mu, np.log(50), delta=0.05)
        self.assertAlmostEqual(model.body.sigma, 0.8, delta=0.06)


class TestAIMDBothArmsReachable(unittest.TestCase):
    """AIMD must actually be additive-increase MULTIPLICATIVE-DECREASE.

    With the previous defaults the decrease arm could not fire inside a
    default-length episode, so the baseline was a one-way ramp to the widest
    timeout -- effectively a delayed FixedValuePolicy, mislabelled as adaptive.
    """

    def test_both_arms_fire_at_shipped_defaults(self):
        from rl.env import SCENARIO_BUILDERS, EnvConfig, MicroservicesEnv
        from rl.policies.aimd import AIMDPolicy

        spec = ActionSpec([10, 30, 50, 80], [1, 2, 3], [0, 20, 50])
        for scenario in ("basic", "randomized", "multichain"):
            with self.subTest(scenario=scenario):
                increases = decreases = 0
                for seed in (321, 654, 987, 1320):
                    env = MicroservicesEnv(
                        EnvConfig(
                            build=lambda s, _sc=scenario: SCENARIO_BUILDERS[_sc](s, 30.0),
                            action_spec=spec,
                            step_duration_s=1.0,
                            episode_duration_s=10.0,
                        )
                    )
                    policy = AIMDPolicy(grid=ActionGrid(spec))
                    policy.reset(seed=seed)
                    obs, info = env.reset(seed=seed)
                    terminated = False
                    previous = None
                    while not terminated:
                        action = policy.act(obs, info)
                        if previous is not None:
                            if action[0] > previous[0]:
                                increases += 1
                            elif action[0] < previous[0]:
                                decreases += 1
                        previous = action
                        obs, _, terminated, _, info = env.step(action)
                self.assertGreater(increases, 0, "increase arm never fired")
                self.assertGreater(decreases, 0, "decrease arm never fired")


class TestColdStartAndCensoredWindows(unittest.TestCase):
    """Neither policy may fall back to the grid CORNER.

    Index 0 on every knob is the shortest timeout with the fewest attempts,
    measured pathological (~11% success) - which is why the env seeds mid-grid
    (rl/env.py). A policy that emits it burns window 0 of 10, and mid-episode
    it is self-reinforcing: the corner guarantees the next window is censored
    too, so the policy pins itself there for the rest of the fault.
    """

    def setUp(self):
        self.spec = ActionSpec(attempt_timeout_ms=[10, 80], max_attempts=[1], backoff_ms=[0])
        self.grid = ActionGrid(self.spec)

    def test_model_based_cold_start_is_mid_grid(self):
        policy = ModelBasedPolicy(grid=self.grid, sample_size=50)
        action = policy.act(np.zeros(1), {"attempt_latencies_ms": [], "attempt_censored": []})
        self.assertEqual(action, self.grid.midpoint_indices())
        self.assertNotEqual(action, self.grid.indices[0])

    def test_bayesian_cold_start_is_mid_grid(self):
        policy = BayesianPolicy(grid=self.grid, sample_size=50)
        action = policy.act(np.zeros(1), {"attempt_latencies_ms": [], "attempt_censored": []})
        self.assertEqual(action, self.grid.midpoint_indices())
        self.assertNotEqual(action, self.grid.indices[0])

    def test_fully_censored_window_holds_the_previous_fit(self):
        # buffer_size == window size, so the censored window fully evicts the
        # clean one - the shape the identifiability guard was written for.
        policy = ModelBasedPolicy(grid=self.grid, sample_size=50, buffer_size=50)
        clean = policy.act(
            np.zeros(1),
            {"attempt_latencies_ms": [40.0] * 50, "attempt_censored": [False] * 50},
        )
        self.assertEqual(clean, (1, 0, 0))  # only the 80 ms timeout covers 40 ms

        held = policy.act(
            np.zeros(1),
            {"attempt_latencies_ms": [10.0] * 50, "attempt_censored": [True] * 50},
        )
        self.assertEqual(policy._fit_buffer().data().size, 0)  # nothing usable left
        self.assertEqual(held, clean)
        self.assertNotEqual(held, self.grid.indices[0])


class TestConsecutiveWindowStreams(unittest.TestCase):
    """Common random numbers WITHIN a window, fresh draws ACROSS windows.

    Every candidate is scored from the same saved state, so a window consumes
    hundreds of words; nudging the generator by a single `random()` call left
    the next window re-scoring on the same samples, turning the Monte-Carlo
    error into a fixed per-episode bias instead of noise that averages out.
    """

    def setUp(self):
        self.grid = ActionGrid(
            ActionSpec(attempt_timeout_ms=[10, 80], max_attempts=[1], backoff_ms=[0])
        )
        self.info = {
            "attempt_latencies_ms": [20.0, 25.0, 22.0, 24.0] * 10,
            "attempt_censored": [False] * 40,
        }

    def _draws_per_window(self, act):
        recorded: list[list[float]] = []

        def recorder(model, action, weights, rng, n_roots, queue_delay_ms, queue_delay_mode):
            recorded.append(rng.random(n_roots).tolist())
            return 1.0

        with mock.patch("rl.policies.model_based._simulate_action", recorder):
            act()
            first = list(recorded)
            recorded.clear()
            act()
            second = list(recorded)
        return first, second

    def _assert_crn_within_and_fresh_across(self, first, second):
        self.assertEqual(len(first), len(self.grid.indices))
        self.assertEqual(first[0], first[1], "candidates must share one stream (CRN)")
        self.assertEqual(second[0], second[1], "candidates must share one stream (CRN)")
        self.assertFalse(
            set(first[0]) & set(second[0]),
            "consecutive windows re-used the previous window's draws",
        )

    def test_model_based_windows_draw_disjoint_samples(self):
        policy = ModelBasedPolicy(grid=self.grid, sample_size=16)
        first, second = self._draws_per_window(lambda: policy.act(np.zeros(1), self.info))
        self._assert_crn_within_and_fresh_across(first, second)

    def test_bayesian_windows_draw_disjoint_samples(self):
        policy = BayesianPolicy(grid=self.grid, sample_size=16)
        first, second = self._draws_per_window(lambda: policy.act(np.zeros(1), self.info))
        self._assert_crn_within_and_fresh_across(first, second)


class TestGlobalTimeoutTieBreak(unittest.TestCase):
    """`global_timeout_ms=None` is the most permissive setting of that knob, so
    the cheapest-of-equals tie-break must rank it last, not first."""

    def test_none_sorts_after_every_numeric_budget(self):
        numeric = _action_cost_key(ActionValues(50, 2, 0, 500))
        unbounded = _action_cost_key(ActionValues(50, 2, 0, None))
        self.assertLess(numeric, unbounded)

    def test_tied_scores_select_the_bounded_budget(self):
        grid = ActionGrid(ActionSpec([100], [1], [0], [None, 500]))
        idx = _argmax_over_grid(
            grid,
            _ConstantModel(10.0),
            RewardWeights(),
            np.random.default_rng(0),
            sample_size=20,
            queue_delay_ms=0.0,
            queue_delay_mode="deterministic",
        )
        self.assertEqual(grid.values_for(idx).global_timeout_ms, 500)


class TestLatencyBufferIngestion(unittest.TestCase):
    """Padding mismatched flags with `False` feeds deadline-cut attempts to the
    censored MLE as complete observations, which inflates E[S] several-fold -
    the exact failure `add` already raises on."""

    def test_mismatched_flags_raise_instead_of_padding(self):
        buffer = LatencyBuffer(max_samples=10)
        with self.assertRaises(ValueError):
            buffer.ingest_info(
                {"attempt_latencies_ms": [1.0, 2.0, 3.0], "attempt_censored": [True, False]}
            )

    def test_absent_flags_mean_uncensored(self):
        buffer = LatencyBuffer(max_samples=10)
        buffer.ingest_info({"attempt_latencies_ms": [1.0, 2.0]})
        self.assertEqual(buffer.censored, [False, False])

    def test_service_stream_uses_the_same_rule(self):
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0]))
        policy = ModelBasedPolicy(grid=grid, include_queue_delay=True)
        with self.assertRaises(ValueError):
            policy.act(
                np.zeros(1),
                {
                    "attempt_service_ms": [20.0, 21.0],
                    "attempt_service_censored": [True],
                },
            )


class TestAIMDSeedingHasOneSource(unittest.TestCase):
    """The EMA seeding rule is a calibration result; it must live in `reset`
    alone, since that is the path `compare_policies` runs per eval seed."""

    def test_construction_goes_through_reset(self):
        class _RecalibratedAIMD(AIMDPolicy):
            def reset(self, seed: int | None = None) -> None:
                super().reset(seed)
                self._ema_success = 0.123

        policy = _RecalibratedAIMD(grid=ActionGrid(ActionSpec([10, 50], [1, 2], [0])))
        self.assertEqual(policy._ema_success, 0.123)


if __name__ == "__main__":
    unittest.main()
