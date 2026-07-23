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
from rl.spec import ActionValues
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
