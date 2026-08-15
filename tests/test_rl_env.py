import functools
import unittest

import numpy as np

try:
    import gymnasium  # noqa: F401
except Exception:
    gymnasium = None  # ty: ignore[invalid-assignment]

from caller import ClientConfig, ClientRuntime
from core import DropReason, Request, RootRequest, TimeInterval, ms_to_ns, ns_to_ms, s_to_ns
from faults import PartialFailure
from retry_policies import FixedBackoffRetryPolicy
from rl.env import (
    ActionSpec,
    EnvComponents,
    EnvConfig,
    MicroservicesEnv,
    build_chain_env,
    build_multichain_env,
    build_randomized_faulty_single_service_env,
    build_single_service_env,
)
from rl.spec import ActionValues, RewardWeights, linear_reward
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload


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

    def _make_env(
        self,
        build,
        episode_duration_s=2.0,
        duration=None,
        cost_budget=None,
        reward_weights=None,
        terminal_drain_penalty=False,
    ):
        action_spec = ActionSpec(
            attempt_timeout_ms=[10, 50],
            max_attempts=[1, 2],
            backoff_ms=[0, 50],
        )
        # `duration` pins the BUILDER's workload duration independently of the
        # episode horizon, so a test can exercise the case where the arrival
        # process outlives the episode.
        if duration is not None:
            build = functools.partial(build, duration_s=duration)
        cfg = EnvConfig(
            build=build,
            action_spec=action_spec,
            step_duration_s=1.0,
            episode_duration_s=episode_duration_s,
            cost_budget=cost_budget,
            reward_weights=RewardWeights() if reward_weights is None else reward_weights,
            terminal_drain_penalty=terminal_drain_penalty,
        )
        return MicroservicesEnv(cfg)

    @staticmethod
    def _run_fixed_action(env, action, seed):
        """Drive a whole episode with one fixed action.

        Returns `(total_reward, window_summaries, final_info)` — the reward the
        agent actually accumulated plus the per-window summaries the env
        exposes, so a test can check the reward against the windows it came
        from rather than against a re-derivation of the same formula.
        """
        env.reset(seed=seed)
        total = 0.0
        windows = []
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(np.asarray(action))
            total += float(reward)
            windows.append(info["summary"])
        return total, windows, info

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

    def test_chain_env_knobs_reach_inner_edge(self):
        from chain import ServiceNodeRuntime
        from retry_policies import FixedBackoffRetryPolicy
        from timeout_controllers import StaticTimeoutController

        components = build_chain_env(3, duration_s=10)
        components.apply_knobs(
            ActionValues(
                attempt_timeout_ms=80, max_attempts=3, backoff_ms=50, global_timeout_ms=None
            )
        )

        a_node = components.client.service
        assert isinstance(a_node, ServiceNodeRuntime)
        b_node = a_node.downstreams[0].target
        edge_client = b_node.downstreams[0].client
        edge_retry = edge_client.cfg.retry
        edge_timeout = edge_client.cfg.timeout_controller
        assert isinstance(edge_retry, FixedBackoffRetryPolicy)
        assert isinstance(edge_timeout, StaticTimeoutController)
        self.assertEqual(edge_retry.max_attempts, 3)
        self.assertEqual(edge_retry.delay, ms_to_ns(50))
        self.assertEqual(edge_timeout.attempt_timeout, ms_to_ns(80))
        # The top-level client (reward source) keeps its own fixed knobs.
        top_timeout = components.client.cfg.timeout_controller
        assert isinstance(top_timeout, StaticTimeoutController)
        self.assertEqual(top_timeout.attempt_timeout, ms_to_ns(300))

    def test_cost_zero_without_cost_client(self):
        env = self._make_env(build_single_service_env, episode_duration_s=2.0)
        env.reset(seed=5)
        _, _, _, _, info = env.step(np.array([1, 0, 0]))
        self.assertEqual(info["summary"]["cost"], 0.0)

    def test_chain_env_cost_measures_edge_amplification(self):
        env = self._make_env(build_chain_env, episode_duration_s=5.0)
        env.reset(seed=11)
        # Aggressive knobs: 10 ms edge timeout vs C's 20 ms median forces edge
        # timeouts, so the max-attempts setting produces real retries.
        costs = []
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(np.array([0, 1, 0]))
            costs.append(info["summary"]["cost"])
        self.assertGreater(max(costs), 0.0)

    def _cost_window(self, env, retries, roots):
        summary = env._summarize_roots([])
        summary["cost_retries"] = float(retries)
        summary["cost_roots"] = float(roots)
        summary["cost"] = retries / roots if roots else 0.0
        return summary

    def test_cost_penalty_sum_is_order_independent(self):
        """The episode's cost term must price the constraint, not the order.

        Dividing each window by a RUNNING mean of roots-per-window made the sum
        depend on where the retry-heavy windows fell: windows (5 retries/5
        roots) then (0/100) summed to 1.0 while the reverse order summed to
        0.095 — identical pooled cost, 10.5x apart. The dual ascends on
        EpisodeResult.pooled_cost, which carries no such weighting, so primal
        and dual were priced on different quantities.
        """
        budget = 0.1
        env = self._make_env(build_chain_env, episode_duration_s=2.0, cost_budget=budget)
        env.reset(seed=11)
        heavy = self._cost_window(env, 5, 5)
        light = self._cost_window(env, 0, 100)

        forward = env._cost_penalty(heavy) + env._cost_penalty(light)
        backward = env._cost_penalty(light) + env._cost_penalty(heavy)
        self.assertAlmostEqual(forward, backward, places=12)

        # The sum is exactly sum(retries) - budget * sum(roots): the linearized
        # constraint, over the same population on both sides.
        self.assertAlmostEqual(forward, 5.0 - budget * 105.0, places=12)

    def test_cost_penalty_without_budget_prices_retry_counts(self):
        """No budget, no constraint: the term is a soft penalty on retries.

        Still order-independent (a constant divisor fixed at reset), but only
        an approximation of amplification — the divisor counts TOP-LEVEL roots
        while the retries are the edge client's. Documented as such; it must
        not silently become something else.
        """
        env = self._make_env(build_chain_env, episode_duration_s=2.0)
        env.reset(seed=11)
        heavy = self._cost_window(env, 5, 5)
        light = self._cost_window(env, 0, 100)

        forward = env._cost_penalty(heavy) + env._cost_penalty(light)
        backward = env._cost_penalty(light) + env._cost_penalty(heavy)
        self.assertAlmostEqual(forward, backward, places=12)
        expected_roots = env._require_components().workload.base_rps * env.step_duration_s
        self.assertAlmostEqual(forward, 5.0 / expected_roots, places=12)

    def test_episode_cost_terms_sum_to_the_linearized_constraint(self):
        """On a REAL episode: the cost terms sum to sum(r) - budget * sum(n).

        This is the property that pairs primal with dual — the sum equals
        sum(roots) * (pooled_cost - budget), the quantity `dual_ascent_update`
        moves lambda on. Normalising each window's retry count by the expected
        TOP-LEVEL roots per window broke it: edge roots are a different,
        action-dependent population (upstream retries create more of them), so
        the sum tracked raw retries and ranked actions differently from
        `EpisodeResult.pooled_cost`.
        """
        budget = 0.1
        action = (0, 1, 0)  # 10 ms edge timeout, 2 attempts: real amplification
        env = self._make_env(
            build_multichain_env, episode_duration_s=4.0, duration=4.0, cost_budget=budget
        )
        baseline_reward, windows, _ = self._run_fixed_action(env, action, seed=11)

        retries = sum(w["cost_retries"] for w in windows)
        roots = sum(w["cost_roots"] for w in windows)
        # An episode with no edge traffic would satisfy the identity vacuously.
        self.assertGreater(retries, 0.0)
        self.assertGreater(roots, 0.0)
        self.assertAlmostEqual(
            sum(env._cost_penalty(w) for w in windows), retries - budget * roots, places=6
        )

        # ...and the REWARD carries exactly that sum: same seeded episode, one
        # non-zero cost weight. The reward is what training sees; an identity
        # that held only for `_cost_penalty` would prove nothing about it.
        cost_weight = 0.01
        weighted = self._make_env(
            build_multichain_env,
            episode_duration_s=4.0,
            duration=4.0,
            cost_budget=budget,
            reward_weights=RewardWeights(cost=cost_weight),
        )
        weighted_reward, _, _ = self._run_fixed_action(weighted, action, seed=11)
        self.assertAlmostEqual(
            weighted_reward - baseline_reward, -cost_weight * (retries - budget * roots), places=6
        )

    def test_cost_term_sign_agrees_with_the_dual(self):
        """The episode's cost term is negative exactly when the arm is inside
        its budget — the sign the dual moves lambda on.

        The old term was a non-negative multiple of the retry count: an arm
        comfortably inside its budget still paid a positive penalty, and since
        edge-root counts vary with the action, two arms could rank one way on
        the penalty and the other way on the pooled cost.
        """
        budget = 0.1
        pooled_costs = []
        # A no-retry action (inside any budget) and an aggressive one (outside).
        for action in ((1, 0, 0), (0, 1, 0)):
            with self.subTest(action=action):
                env = self._make_env(
                    build_multichain_env,
                    episode_duration_s=3.0,
                    duration=3.0,
                    cost_budget=budget,
                )
                _, windows, _ = self._run_fixed_action(env, action, seed=11)
                retries = sum(w["cost_retries"] for w in windows)
                roots = sum(w["cost_roots"] for w in windows)
                self.assertGreater(roots, 0.0)
                pooled = retries / roots
                pooled_costs.append(pooled)
                term_sum = sum(env._cost_penalty(w) for w in windows)
                self.assertAlmostEqual(term_sum, roots * (pooled - budget), places=6)
                self.assertEqual(term_sum > 0.0, pooled > budget)
        # Non-vacuous only if the two arms straddle the budget.
        self.assertLess(min(pooled_costs), budget)
        self.assertGreater(max(pooled_costs), budget)

    def test_drained_cost_term_is_not_count_weighted(self):
        """`terminal_drain_penalty` count-adjusts the RATE terms only.

        The drain weight exists because per-window rewards are per-completion
        rates: a drained window of 2 roots must not be penalized as hard as a
        full window of 100. The cost term is already a count (retries minus
        budget x roots over the drained population), so scaling it by the same
        weight prices the drained tail's retries by a factor that depends on
        the window size — and breaks the sum(r) - budget*sum(n) identity the
        dual is paired with.
        """
        budget = 0.05
        cost_weight = 0.01
        action = (0, 1, 0)

        def run(penalty):
            env = self._make_env(
                build_multichain_env,
                episode_duration_s=3.0,
                duration=3.0,
                cost_budget=budget,
                reward_weights=RewardWeights(cost=cost_weight),
                terminal_drain_penalty=penalty,
            )
            total, _, info = self._run_fixed_action(env, action, seed=11)
            return env, total, info

        _, total_off, _ = run(False)
        env, total_on, info = run(True)
        drained = info["drained_summary"]
        self.assertGreater(drained["completed"], 0)
        self.assertGreater(drained["cost_roots"], 0.0)

        per_step = env._horizon_root_count / env._episode_steps
        weight = drained["completed"] / max(1.0, per_step)
        rate_part = linear_reward(drained, env._cfg.reward_weights, cost_term=0.0)
        cost_part = env._cost_penalty(drained) * cost_weight
        # The two forms only differ when the drain is not exactly one window.
        self.assertNotAlmostEqual(weight, 1.0, places=2)
        self.assertGreater(abs(cost_part), 1e-3)

        self.assertAlmostEqual(total_on - total_off, weight * rate_part - cost_part, places=6)
        # Weighting the cost term too (the old form) is measurably different.
        self.assertNotAlmostEqual(total_on - total_off, weight * (rate_part - cost_part), places=4)

    def test_cost_penalty_is_a_pure_function_of_the_window(self):
        """Scoring one window must not reprice the next.

        `_reward_from_summary` reads as a pure summary->float mapping and is
        called twice in one step when terminal_drain_penalty is on.
        """
        env = self._make_env(build_chain_env, episode_duration_s=2.0)
        env.reset(seed=11)
        other = self._cost_window(env, 1, 10)
        window = self._cost_window(env, 3, 200)

        alone = env._cost_penalty(window)
        env._cost_penalty(other)
        self.assertAlmostEqual(env._cost_penalty(window), alone, places=12)

    def test_summary_keys_are_identical_with_and_without_traffic(self):
        # A consumer reading info["summary"] must not KeyError only in the
        # windows where nothing completed.
        env = self._make_env(build_chain_env, episode_duration_s=2.0)
        env.reset(seed=11)
        _, _, _, _, info = env.step(np.array([1, 1, 0]))
        populated = info["summary"]
        self.assertGreater(populated["completed"], 0)
        self.assertEqual(set(env._summarize_roots([])), set(populated))

    def test_reward_penalizes_cost(self):
        env = self._make_env(build_single_service_env)
        summary = env._summarize_roots([])
        summary["success_rate"] = 1.0
        baseline = env._reward_from_summary(summary)
        env._cfg.reward_weights.cost = 2.0
        summary["cost"] = 0.5
        self.assertAlmostEqual(env._reward_from_summary(summary), baseline - 1.0)

    def test_chain_budget_knobs_route_through_wrapper(self):
        from retry_policies import FixedBackoffRetryPolicy, RetryBudgetPolicy

        components = build_chain_env(3, duration_s=10, retry_budget_ratio=0.1)
        components.apply_knobs(
            ActionValues(
                attempt_timeout_ms=80, max_attempts=3, backoff_ms=50, global_timeout_ms=None
            )
        )
        edge_client = components.cost_client
        assert edge_client is not None
        wrapper = edge_client.cfg.retry
        assert isinstance(wrapper, RetryBudgetPolicy)
        inner = wrapper.underlying_strategy
        assert isinstance(inner, FixedBackoffRetryPolicy)
        self.assertEqual(inner.max_attempts, 3)
        self.assertEqual(inner.delay, ms_to_ns(50))

    def test_retry_budget_wrapper_throttles_edge_retries(self):
        def mean_cost(retry_budget_ratio):
            env = self._make_env(
                lambda seed: build_chain_env(
                    seed, duration_s=5, retry_budget_ratio=retry_budget_ratio
                ),
                episode_duration_s=5.0,
            )
            env.reset(seed=11)
            costs = []
            terminated = False
            while not terminated:
                _, _, terminated, _, info = env.step(np.array([0, 1, 0]))
                costs.append(info["summary"]["cost"])
            return float(np.mean(costs))

        unbudgeted = mean_cost(None)
        budgeted = mean_cost(0.05)
        self.assertGreater(unbudgeted, 0.0)
        self.assertLess(budgeted, unbudgeted)

    def test_chain_env_episode_runs(self):
        env = self._make_env(build_chain_env, episode_duration_s=3.0)
        obs, _ = env.reset(seed=11)
        terminated = False
        steps = 0
        while not terminated and steps < 10:
            obs, reward, terminated, _, info = env.step(np.array([1, 1, 1]))
            steps += 1
        self.assertTrue(terminated)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertGreater(info["summary"]["completed"], 0)

    def test_drained_roots_reported_separately_from_final_summary(self):
        env = self._make_env(build_single_service_env, episode_duration_s=1.0)
        env.reset(seed=7)
        _, _, terminated, _, info = env.step(np.array([0, 0, 0]))
        self.assertTrue(terminated)
        # Drained roots are reported separately, not folded into the step summary.
        self.assertIn("drained_summary", info)
        self.assertIsInstance(info["drained_summary"], dict)

    def test_multichain_shared_bottleneck_single_runtime(self):
        from chain import ServiceNodeRuntime

        components = build_multichain_env(3, duration_s=10)
        assert components.bystander_client is not None
        a_node = components.client.service
        x_node = components.bystander_client.service
        assert isinstance(a_node, ServiceNodeRuntime)
        assert isinstance(x_node, ServiceNodeRuntime)
        agent_c = a_node.downstreams[0].target.downstreams[0].target
        bystander_c = x_node.downstreams[0].target.downstreams[0].target
        self.assertIs(agent_c, bystander_c)
        self.assertIs(agent_c.service, bystander_c.service)

    def test_multichain_knobs_reach_agent_edge_only(self):
        from chain import ServiceNodeRuntime
        from retry_policies import FixedBackoffRetryPolicy
        from timeout_controllers import StaticTimeoutController

        components = build_multichain_env(3, duration_s=10)
        components.apply_knobs(
            ActionValues(
                attempt_timeout_ms=80, max_attempts=3, backoff_ms=50, global_timeout_ms=None
            )
        )

        edge_client = components.cost_client
        assert edge_client is not None
        edge_retry = edge_client.cfg.retry
        edge_timeout = edge_client.cfg.timeout_controller
        assert isinstance(edge_retry, FixedBackoffRetryPolicy)
        assert isinstance(edge_timeout, StaticTimeoutController)
        self.assertEqual(edge_retry.max_attempts, 3)
        self.assertEqual(edge_retry.delay, ms_to_ns(50))
        self.assertEqual(edge_timeout.attempt_timeout, ms_to_ns(80))

        # The bystander's Y->C edge stays retry-free and both top-level
        # clients keep their fixed knobs.
        assert components.bystander_client is not None
        x_node = components.bystander_client.service
        assert isinstance(x_node, ServiceNodeRuntime)
        y_to_c = x_node.downstreams[0].target.downstreams[0].client
        bystander_retry = y_to_c.cfg.retry
        assert isinstance(bystander_retry, FixedBackoffRetryPolicy)
        self.assertEqual(bystander_retry.max_attempts, 1)
        for top in (components.client, components.bystander_client):
            controller = top.cfg.timeout_controller
            assert isinstance(controller, StaticTimeoutController)
            self.assertEqual(controller.attempt_timeout, ms_to_ns(300))

    def test_multichain_pure_partial_failure_blips(self):
        from chain import ServiceNodeRuntime

        for seed in range(20):
            components = build_multichain_env(seed, duration_s=30)
            a_node = components.client.service
            assert isinstance(a_node, ServiceNodeRuntime)
            c_cfg = a_node.downstreams[0].target.downstreams[0].target.service.cfg
            self.assertEqual(c_cfg.latency_injections, [])
            self.assertIn(len(c_cfg.partial_failures), (2, 3, 4))
            for blip in c_cfg.partial_failures:
                self.assertGreaterEqual(blip.duration.begin, 0)
                self.assertLess(blip.duration.begin, blip.duration.end)
                self.assertLessEqual(blip.duration.end, s_to_ns(30))
                self.assertGreaterEqual(blip.p_fail, 0.5)
                self.assertLessEqual(blip.p_fail, 0.8)

    def test_multichain_faults_deterministic_per_seed(self):
        from chain import ServiceNodeRuntime

        def blip_params(components):
            a_node = components.client.service
            assert isinstance(a_node, ServiceNodeRuntime)
            c_cfg = a_node.downstreams[0].target.downstreams[0].target.service.cfg
            return tuple(
                (pf.duration.begin, pf.duration.end, pf.p_fail) for pf in c_cfg.partial_failures
            )

        a = blip_params(build_multichain_env(5, duration_s=30))
        b = blip_params(build_multichain_env(5, duration_s=30))
        c = blip_params(build_multichain_env(6, duration_s=30))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_multichain_cost_measures_edge_amplification(self):
        env = self._make_env(build_multichain_env, episode_duration_s=10.0)
        env.reset(seed=11)
        # 50 ms timeout leaves headroom over C's 20 ms median, so cost comes
        # from the p_fail blips (recoverable failures), not forced timeouts.
        costs = []
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(np.array([1, 1, 0]))
            costs.append(info["summary"]["cost"])
        self.assertGreater(max(costs), 0.0)

    def test_multichain_budget_knobs_route_through_wrapper(self):
        from retry_policies import FixedBackoffRetryPolicy, RetryBudgetPolicy

        components = build_multichain_env(3, duration_s=10, retry_budget_ratio=0.1)
        components.apply_knobs(
            ActionValues(
                attempt_timeout_ms=80, max_attempts=3, backoff_ms=50, global_timeout_ms=None
            )
        )
        edge_client = components.cost_client
        assert edge_client is not None
        wrapper = edge_client.cfg.retry
        assert isinstance(wrapper, RetryBudgetPolicy)
        inner = wrapper.underlying_strategy
        assert isinstance(inner, FixedBackoffRetryPolicy)
        self.assertEqual(inner.max_attempts, 3)
        self.assertEqual(inner.delay, ms_to_ns(50))

    def test_multichain_budget_wrapper_throttles_edge_retries(self):
        def mean_cost(retry_budget_ratio):
            env = self._make_env(
                lambda seed: build_multichain_env(
                    seed, duration_s=10, retry_budget_ratio=retry_budget_ratio
                ),
                episode_duration_s=10.0,
            )
            env.reset(seed=11)
            costs = []
            terminated = False
            while not terminated:
                _, _, terminated, _, info = env.step(np.array([1, 1, 0]))
                costs.append(info["summary"]["cost"])
            return float(np.mean(costs))

        unbudgeted = mean_cost(None)
        budgeted = mean_cost(0.05)
        self.assertGreater(unbudgeted, 0.0)
        self.assertLess(budgeted, unbudgeted)

    def test_multichain_bystander_summary_reported(self):
        env = self._make_env(build_multichain_env, episode_duration_s=2.0)
        env.reset(seed=7)
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(np.array([1, 0, 0]))
        self.assertIn("bystander_summary", info)
        self.assertGreater(info["bystander_summary"]["completed"], 0)
        summary = env.bystander_metrics_summary()
        assert summary is not None
        self.assertGreater(summary.total, 0)

    def test_bystander_workload_halted_at_horizon(self):
        """The terminal drain must complete in-flight work only.

        build_multichain_env drives a SECOND workload for the bystander chain.
        It is registered in EnvComponents.extra_workloads so step() halts it at
        the horizon; without that registration the unbounded drain kept
        admitting brand-new bystander roots until the builder's own duration
        (measured: 70% of the population started after the horizon).
        """
        # Builder duration deliberately outlives the episode horizon.
        env = self._make_env(build_multichain_env, episode_duration_s=3.0, duration=10.0)
        env.reset(seed=5)
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(np.array([1, 0, 0]))

        components = env._require_components()
        assert components.bystander_client is not None
        roots = components.bystander_client.roots
        started_after_horizon = sum(
            1
            for root in roots
            if root.attempts and root.attempts[0].interval.begin > env._end_time_ns
        )
        self.assertEqual(
            started_after_horizon,
            0,
            f"{started_after_horizon} of {len(roots)} bystander roots started after the "
            f"episode horizon; the drain must complete in-flight work only",
        )

    def test_bystander_in_horizon_population_is_a_prefix(self):
        env = self._make_env(build_multichain_env, episode_duration_s=3.0, duration=10.0)
        env.reset(seed=5)
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(np.array([1, 0, 0]))

        self.assertIn("bystander_summary_in_horizon", info)
        in_horizon = env.bystander_in_horizon_summary()
        full = env.bystander_metrics_summary()
        assert in_horizon is not None and full is not None
        self.assertGreater(in_horizon.total, 0)

        # Pin the partition the way the agent-side twin does, not just
        # `in_horizon <= full`: the prefix slice makes that inequality true for
        # ANY horizon count, so booking the whole drained tail as reward-visible
        # would satisfy it. Recompute the reward-visible population from the
        # roots' own completion times.
        bystander = env._require_components().bystander_client
        assert bystander is not None
        roots = bystander.roots
        visible = [
            root
            for root in roots
            if root.completed_at is not None and root.completed_at <= env._end_time_ns
        ]
        drained = len(roots) - len(visible)
        self.assertGreater(drained, 0, "fixture must leave a drained tail to distinguish")
        self.assertEqual(in_horizon.total, len(visible))
        self.assertEqual(full.total, len(visible) + drained)
        self.assertLess(in_horizon.total, full.total)
        self.assertEqual(in_horizon.total, info["bystander_summary_in_horizon"]["completed"])

    def test_modeled_queue_ground_truth_describes_the_shared_service(self):
        """`modeled_admitted`/`modeled_workers` describe svc-C, not the sampled edge.

        A queueing policy left without them falls back to the sampled client's
        attempt count — which halves rho here, since C is fed by the agent's
        edge AND the bystander's — and to a scenario-independent worker default
        instead of C's real worker count.
        """
        env = self._make_env(build_multichain_env, episode_duration_s=3.0)
        env.reset(seed=11)
        _, _, _, _, first = env.step(np.array([1, 1, 0]))
        _, _, _, _, second = env.step(np.array([1, 1, 0]))

        modeled = env._require_components().modeled_service
        assert modeled is not None
        self.assertGreater(modeled.cfg.workers, 0)
        self.assertEqual(first["modeled_workers"], modeled.cfg.workers)
        self.assertEqual(second["modeled_workers"], modeled.cfg.workers)
        # Per-window deltas of the queue's OWN admitted-attempt counter.
        self.assertGreater(first["modeled_admitted"], 0)
        self.assertGreater(second["modeled_admitted"], 0)
        self.assertEqual(
            first["modeled_admitted"] + second["modeled_admitted"], modeled.admitted_total
        )
        # Strictly more than the sampled edge's own attempts in the same window.
        self.assertGreater(second["modeled_admitted"], len(second["attempt_latencies_ms"]))

    def test_bystander_summary_absent_without_bystander(self):
        env = self._make_env(build_chain_env, episode_duration_s=1.0)
        env.reset(seed=7)
        _, _, terminated, _, info = env.step(np.array([0, 0, 0]))
        self.assertTrue(terminated)
        self.assertNotIn("bystander_summary", info)
        self.assertIsNone(env.bystander_metrics_summary())

    def test_multichain_episode_runs(self):
        env = self._make_env(build_multichain_env, episode_duration_s=3.0)
        obs, _ = env.reset(seed=11)
        terminated = False
        steps = 0
        while not terminated and steps < 10:
            obs, reward, terminated, _, info = env.step(np.array([1, 1, 1]))
            steps += 1
        self.assertTrue(terminated)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertGreater(info["summary"]["completed"], 0)


def _build_slow_failing_env(seed=None, duration_s=None):
    """Every request fails after ~2.5 s of service: with a long attempt timeout
    all failures land in the drain; with a short one they land in-episode."""
    build_seed = 1 if seed is None else int(seed)
    sim = Simulator(seed=build_seed)
    duration = 2.0 if duration_s is None else float(duration_s)
    cfg = ServiceConfig(
        name="svc-slow",
        latency_median=ms_to_ns(2500),
        latency_lognorm_sigma=1e-6,
        workers=256,
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=0, end=s_to_ns(10_000)), p_fail=1.0)
        ],
    )
    timeout_ctrl = StaticTimeoutController(global_timeout=None, attempt_timeout=ms_to_ns(50))
    retry = FixedBackoffRetryPolicy(max_attempts=1, delay=0)
    client = ClientRuntime(
        cfg=ClientConfig(retry=retry, timeout_controller=timeout_ctrl),
        service=ServiceRuntime(cfg=cfg).bind(),
    )

    def apply_knobs(values):
        timeout_ctrl.attempt_timeout = ms_to_ns(values.attempt_timeout_ms)
        retry.max_attempts = values.max_attempts
        retry.delay = ms_to_ns(values.backoff_ms)

    workload = Workload(base_rps=50.0, duration_s=duration, load_spikes=[], rng_seed=seed)
    return EnvComponents(sim=sim, workload=workload, client=client, apply_knobs=apply_knobs)


SLOW_SPEC = ActionSpec(attempt_timeout_ms=[50, 100_000], max_attempts=[1], backoff_ms=[0])


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestPopulationAccounting(unittest.TestCase):
    def _run_episode(self, action, terminal_drain_penalty=False):
        env = MicroservicesEnv(
            EnvConfig(
                build=_build_slow_failing_env,
                action_spec=SLOW_SPEC,
                step_duration_s=1.0,
                episode_duration_s=2.0,
                terminal_drain_penalty=terminal_drain_penalty,
                initial_action=action,
            )
        )
        obs, info = env.reset(seed=42)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
        return env, total, info

    def test_summaries_partition_the_population(self):
        env, _, info = self._run_episode((1, 0, 0))  # 100 s timeout: all failures drain
        full = env.metrics_summary()
        horizon = env.in_horizon_summary()
        drained = info["drained_summary"]
        self.assertEqual(full.total, horizon.total + drained["completed"])
        self.assertGreater(drained["completed"], 0)

    def test_terminal_drain_penalty_removes_deferral_advantage(self):
        _, j_defer_off, _ = self._run_episode((1, 0, 0))
        _, j_fail_off, _ = self._run_episode((0, 0, 0))
        # The exploitable asymmetry: deferring failures past the horizon
        # scores better than failing in-episode, for identical offered work.
        self.assertGreater(j_defer_off, j_fail_off)
        _, j_defer_on, _ = self._run_episode((1, 0, 0), terminal_drain_penalty=True)
        _, j_fail_on, _ = self._run_episode((0, 0, 0), terminal_drain_penalty=True)
        self.assertLess(abs(j_defer_on - j_fail_on), abs(j_defer_off - j_fail_off))

    def test_drain_weight_uses_the_pre_drain_window_size(self):
        """The drained tail must not inflate its own denominator.

        The weight expresses the drained population as a number of
        average-sized step windows. Reading the window size off the root index
        *after* the drain has been folded into it understates the weight by
        exactly horizon/(horizon+drained) — worst precisely when the agent
        defers the most work past the horizon, which is what the flag punishes.
        """

        def run(penalty):
            env = MicroservicesEnv(
                EnvConfig(
                    build=lambda seed: _build_slow_failing_env(seed, duration_s=3.0),
                    action_spec=SLOW_SPEC,
                    step_duration_s=1.0,
                    episode_duration_s=3.0,
                    terminal_drain_penalty=penalty,
                    initial_action=(1, 0, 0),
                )
            )
            env.reset(seed=42)
            total = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                _, reward, terminated, truncated, info = env.step((1, 0, 0))
                total += float(reward)
            return env, total, info

        _, total_off, _ = run(False)
        env, total_on, info = run(True)
        drained = info["drained_summary"]
        # Both populations must be substantial or the two denominators agree.
        self.assertGreater(drained["completed"], 0)
        self.assertGreater(env._horizon_root_count, 0)

        per_step = env._horizon_root_count / env._episode_steps
        weight = drained["completed"] / max(1.0, per_step)
        self.assertAlmostEqual(
            total_on - total_off, weight * env._reward_from_summary(drained), places=9
        )
        # The post-drain population would have priced the same tail far cheaper.
        post_drain_weight = drained["completed"] / max(1.0, env._last_root_idx / env._episode_steps)
        self.assertGreater(weight, 2.0 * post_drain_weight)

    def test_drain_weight_falls_back_to_the_expected_window_size(self):
        # Nothing completes in-horizon here (every root outlives the episode),
        # so there is no measured window size; the offered rate supplies one
        # instead of the whole drain being priced as N one-root windows.
        _, total_off, _ = self._run_episode((1, 0, 0))
        env, total_on, info = self._run_episode((1, 0, 0), terminal_drain_penalty=True)
        drained = info["drained_summary"]
        self.assertEqual(env._horizon_root_count, 0)
        self.assertGreater(drained["completed"], 0)
        weight = drained["completed"] / env._roots_per_window
        self.assertAlmostEqual(
            total_on - total_off, weight * env._reward_from_summary(drained), places=9
        )


class TestSeedNormalization(unittest.TestCase):
    """`build_X(None)` must be `build_X(1)`, not a hybrid.

    The builders resolved None twice: the simulator and the fault RNG to 1, the
    Workload to whatever `Workload.__post_init__` falls back to (0). Both are
    deterministic, so nothing looked broken — but the two calls produced
    identical fault windows and service-time streams over *different* arrival
    processes, and nothing said which stream had moved.
    """

    @staticmethod
    def _arrival_times(workload):
        sim = Simulator(seed=0)
        times: list[int] = []
        workload.drive(sim, lambda s: times.append(s.timestep))
        sim.run()
        return times

    def test_none_seed_resolves_to_one_in_every_stream(self):
        for builder in (build_single_service_env, build_chain_env, build_multichain_env):
            with self.subTest(builder=builder.__name__):
                default = builder(None, 3.0)
                explicit = builder(1, 3.0)
                self.assertEqual(default.workload.rng_seed, 1)
                arrivals = self._arrival_times(default.workload)
                self.assertGreater(len(arrivals), 0)
                self.assertEqual(arrivals, self._arrival_times(explicit.workload))


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestSampleSource(unittest.TestCase):
    def test_chain_samples_come_from_the_actuated_edge(self):
        env = MicroservicesEnv(
            EnvConfig(
                build=lambda seed: build_chain_env(seed, duration_s=4),
                action_spec=ActionSpec(attempt_timeout_ms=[80], max_attempts=[2], backoff_ms=[20]),
                step_duration_s=1.0,
                episode_duration_s=4.0,
            )
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step((0, 0, 0))
        cost_client = env._require_components().cost_client
        assert cost_client is not None
        edge_latencies = sorted(
            ns_to_ms(a.interval.end - a.interval.begin)
            for root in cost_client.roots[: env._last_sample_root_idx]
            for a in root.attempts
            if not (
                a.drop_reason in (DropReason.QUEUE_FULL, DropReason.CIRCUIT_OPEN)
                and a.interval.end == a.interval.begin
            )
        )
        self.assertGreater(len(edge_latencies), 0)
        self.assertEqual(sorted(info["attempt_latencies_ms"]), edge_latencies)


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestInstantRejectionSamples(unittest.TestCase):
    def test_zero_latency_rejections_are_not_emitted(self):
        env = MicroservicesEnv(EnvConfig(build=_build_slow_failing_env, action_spec=SLOW_SPEC))
        env.reset(seed=1)
        rejected = RootRequest()
        rejected.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=1000, end=1000),
                drop_reason=DropReason.QUEUE_FULL,
            )
        )
        served = RootRequest()
        served.add_attempt(
            Request(service="svc", interval=TimeInterval(begin=0, end=ms_to_ns(20)), success=True)
        )
        samples = env._collect_latency_samples([rejected, served])
        self.assertEqual(len(samples["attempt_latencies_ms"]), 1)
        self.assertAlmostEqual(samples["attempt_latencies_ms"][0], 20.0)

    def test_service_stream_censors_deadline_cuts(self):
        env = MicroservicesEnv(EnvConfig(build=_build_slow_failing_env, action_spec=SLOW_SPEC))
        env.reset(seed=1)
        cut = RootRequest()
        cut.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=0, end=ms_to_ns(10)),
                drop_reason=DropReason.DEADLINE,
                service_time_ns=ms_to_ns(10),
                # The worker was still held at the cut, so this occupancy is a
                # lower bound. The service records that explicitly now — it is
                # NOT inferred from drop_reason, because a tandem chain node
                # hands its worker back before the downstream wait and its local
                # stage is a complete observation despite a later deadline.
                service_time_censored=True,
            )
        )
        ok = RootRequest()
        ok.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=0, end=ms_to_ns(20)),
                success=True,
                service_time_ns=ms_to_ns(20),
            )
        )
        samples = env._collect_latency_samples([cut, ok])
        self.assertEqual(samples["attempt_service_ms"], [10.0, 20.0])
        self.assertEqual(samples["attempt_service_censored"], [True, False])

    def test_summarize_tied_final_attempts_use_completion_order(self):
        env = MicroservicesEnv(EnvConfig(build=_build_slow_failing_env, action_spec=SLOW_SPEC))
        env.reset(seed=1)
        root = RootRequest()
        root.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=0, end=ms_to_ns(1)),
                drop_reason=DropReason.SERVER_FAILURE,
            )
        )
        root.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=ms_to_ns(1), end=ms_to_ns(1)),
                drop_reason=DropReason.QUEUE_FULL,
            )
        )
        summary = env._summarize_roots([root])
        self.assertEqual(summary["queue_drop_rate"], 1.0)
        self.assertEqual(summary["server_drop_rate"], 0.0)


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestDrainAdmitsNoNewTraffic(unittest.TestCase):
    def test_workload_outliving_the_horizon_is_halted_at_termination(self):
        env = MicroservicesEnv(
            EnvConfig(
                build=lambda seed: _build_slow_failing_env(seed, duration_s=30.0),
                action_spec=SLOW_SPEC,
                step_duration_s=1.0,
                episode_duration_s=1.0,
            )
        )
        env.reset(seed=3)
        obs, reward, terminated, truncated, info = env.step((1, 0, 0))
        self.assertTrue(terminated)
        client = env._require_client()
        starts = [min(a.interval.begin for a in root.attempts) for root in client.roots]
        self.assertGreater(len(starts), 10)
        self.assertLess(max(starts), s_to_ns(1.0))  # drained roots are in-flight only


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestEpisodeTimeFraction(unittest.TestCase):
    def test_sub_second_episode_time_fraction_reaches_one(self):
        env = MicroservicesEnv(
            EnvConfig(
                build=lambda seed: _build_slow_failing_env(seed, duration_s=0.4),
                action_spec=SLOW_SPEC,
                step_duration_s=0.2,
                episode_duration_s=0.4,
            )
        )
        obs, info = env.reset(seed=1)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step((0, 0, 0))
        self.assertAlmostEqual(float(obs[-1]), 1.0, places=5)


@unittest.skipIf(gymnasium is None, "gymnasium not installed")
class TestDurationValidation(unittest.TestCase):
    def test_zero_step_duration_rejected(self):
        with self.assertRaises(ValueError):
            MicroservicesEnv(
                EnvConfig(build=_build_slow_failing_env, action_spec=SLOW_SPEC, step_duration_s=0.0)
            )

    def test_negative_episode_duration_rejected(self):
        with self.assertRaises(ValueError):
            MicroservicesEnv(
                EnvConfig(
                    build=_build_slow_failing_env, action_spec=SLOW_SPEC, episode_duration_s=-1.0
                )
            )


if __name__ == "__main__":
    unittest.main()
