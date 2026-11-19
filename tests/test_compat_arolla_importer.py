"""Arolla scenarios as msim objects: the importer's mapping and its policy layers."""

import os
import unittest
from pathlib import Path
from typing import Any

import yaml

from msim.admission import (
    AdmissionGate,
    AimdBudget,
    ControlledRetryPolicy,
    FailureRateGate,
    GoodputCoupledBudget,
    SuccessFundedBudget,
    TokenBucketBudget,
)
from msim.caller import RetryLayer
from msim.compat.arolla import Scenario, UnsupportedScenario, arrival_seed, build, parse
from msim.compat.arolla.policies import (
    BreakerState,
    FirstAttemptFundedBudget,
    RetryBank,
    ThreeStateRetryBreaker,
    WindowRetryGate,
)
from msim.core import DropReason, ms_to_ns, s_to_ns
from msim.retry_policies import (
    ExponentialBackoffWithJitterRetryPolicy,
    FixedBackoffRetryPolicy,
    JitterMode,
    NoRetryPolicy,
    RetryContext,
    policy_layers,
)
from msim.service import ServiceRuntime

MS = ms_to_ns(1)
S = s_to_ns(1)


def _build(text: str, **kwargs) -> Scenario:
    return build(parse(yaml.safe_load(text)), **kwargs)


def _layers(scenario: Scenario, client: str) -> list[Any]:
    (found,) = [c for c in scenario.clients if c.name == client]
    return list(policy_layers(found.runtime.cfg.retry))


# ------------------------------------------------------------ policy layers


def _context(attempt: int, now: int) -> RetryContext:
    return RetryContext(attempt=attempt, now=now)


class TestFirstAttemptFundedBudget(unittest.TestCase):
    def _budget(self, **kwargs) -> FirstAttemptFundedBudget:
        defaults: dict[str, Any] = {"budget_ratio": 0.1, "max_retries": 2, "min_retries_per_sec": 0}
        bank = RetryBank(**{**defaults, **kwargs})
        return FirstAttemptFundedBudget(bank, FixedBackoffRetryPolicy(max_attempts=99, delay=0))

    def _drain(self, budget: FirstAttemptFundedBudget, now: int = 0) -> int:
        granted = 0
        while budget.next_delay(_context(1, now)).allow:
            granted += 1
        return granted

    def test_only_a_first_attempt_that_succeeds_makes_a_deposit(self):
        budget = self._budget()
        self.assertEqual(self._drain(budget), 2)  # starts full
        for _ in range(50):
            budget.record_attempt(_context(2, 0), True)  # a retry that succeeded
            budget.record_attempt(_context(1, 0), False)  # a first attempt that failed
        self.assertEqual(self._drain(budget), 0)
        for _ in range(9):
            budget.record_attempt(_context(1, 0), True)
        self.assertEqual(self._drain(budget), 0)  # 0.9 of a retry
        budget.record_attempt(_context(1, 0), True)
        self.assertEqual(self._drain(budget), 1)

    def test_a_ratio_with_many_decimals_is_kept(self):
        budget = self._budget(budget_ratio=0.08673377807052143, max_retries=18)
        self._drain(budget)
        for _ in range(12):  # 12 x 0.0867 = 1.04
            budget.record_attempt(_context(1, 0), True)
        self.assertEqual(self._drain(budget), 1)

    def test_retries_also_accrue_with_time(self):
        budget = self._budget(min_retries_per_sec=4)
        self._drain(budget)
        self.assertEqual(self._drain(budget, now=249 * MS), 0)
        self.assertEqual(self._drain(budget, now=250 * MS), 1)
        self.assertEqual(self._drain(budget, now=10 * S), 2)  # capped at max_retries

    def test_a_retry_the_inner_policy_refuses_costs_nothing(self):
        bank = RetryBank(budget_ratio=0.1, max_retries=2, min_retries_per_sec=0)
        budget = FirstAttemptFundedBudget(bank, FixedBackoffRetryPolicy(max_attempts=1, delay=0))
        self.assertFalse(budget.next_delay(_context(1, 0)).allow)
        self.assertEqual(bank.balance, 2.0)

    def test_a_refund_gives_the_retry_back(self):
        budget = self._budget()
        context = _context(1, 0)
        self.assertTrue(budget.next_delay(context).allow)
        self.assertEqual(budget.bank.balance, 1.0)
        budget.rollback(context)
        self.assertEqual(budget.bank.balance, 2.0)

    def test_two_clients_can_draw_on_one_bank(self):
        bank = RetryBank(budget_ratio=0.5, max_retries=2, min_retries_per_sec=0)
        one = FirstAttemptFundedBudget(bank, FixedBackoffRetryPolicy(max_attempts=9, delay=0))
        two = FirstAttemptFundedBudget(bank, FixedBackoffRetryPolicy(max_attempts=9, delay=0))
        self.assertTrue(one.next_delay(_context(1, 0)).allow)
        self.assertTrue(two.next_delay(_context(1, 0)).allow)
        self.assertFalse(one.next_delay(_context(1, 0)).allow)
        two.record_attempt(_context(1, 0), True)
        two.record_attempt(_context(1, 0), True)
        self.assertTrue(one.next_delay(_context(1, 0)).allow)  # funded by the other client

    def test_validation(self):
        for kwargs in ({"budget_ratio": 1.5}, {"max_retries": 0}, {"min_retries_per_sec": -1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self._budget(**kwargs)


class TestWindowRetryGate(unittest.TestCase):
    def _gate(self) -> WindowRetryGate:
        return WindowRetryGate(
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=99, delay=0),
            failure_threshold=0.5,
            window=S,
            min_window_size=4,
        )

    def _feed(self, gate, start: int, results: list[bool]) -> None:
        for index, ok in enumerate(results):
            gate.record_attempt(_context(1, start + index * MS), ok)

    def test_it_refuses_retries_at_the_threshold_and_not_below(self):
        gate = self._gate()
        self._feed(gate, 0, [False, True, True])
        self.assertTrue(gate.next_delay(_context(1, 10 * MS)).allow)  # too few outcomes
        self._feed(gate, 10 * MS, [False])  # 2 of 4: AT the threshold
        self.assertFalse(gate.next_delay(_context(1, 20 * MS)).allow)
        self._feed(gate, 20 * MS, [True])  # 2 of 5
        self.assertTrue(gate.next_delay(_context(1, 30 * MS)).allow)

    def test_it_reopens_as_failures_age_out(self):
        gate = self._gate()
        self._feed(gate, 0, [False] * 4)
        self.assertFalse(gate.next_delay(_context(1, 500 * MS)).allow)
        self.assertTrue(gate.next_delay(_context(1, 2 * S)).allow)


class TestThreeStateRetryBreaker(unittest.TestCase):
    def _breaker(self) -> ThreeStateRetryBreaker:
        return ThreeStateRetryBreaker(
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=99, delay=0),
            failure_threshold=0.5,
            window=10 * S,
            min_window_size=4,
            open_wait=500 * MS,
        )

    def _feed(self, breaker, start: int, results: list[bool]) -> None:
        for index, ok in enumerate(results):
            breaker.record_attempt(_context(1, start + index * MS), ok)

    def test_closed_open_half_open_closed(self):
        breaker = self._breaker()
        self._feed(breaker, 0, [False] * 4)
        self.assertIs(breaker.state, BreakerState.CLOSED)  # nothing moves until a retry asks
        self.assertFalse(breaker.next_delay(_context(1, 10 * MS)).allow)
        self.assertIs(breaker.state, BreakerState.OPEN)
        self.assertFalse(breaker.next_delay(_context(1, 400 * MS)).allow)  # still waiting
        # The first retry after the wait is let through and starts the trial.
        self.assertTrue(breaker.next_delay(_context(1, 510 * MS)).allow)
        self.assertIs(breaker.state, BreakerState.HALF_OPEN)
        # Retries pass until the trial has its quorum ...
        self._feed(breaker, 520 * MS, [True, True, False])
        self.assertTrue(breaker.next_delay(_context(1, 530 * MS)).allow)
        self.assertIs(breaker.state, BreakerState.HALF_OPEN)
        # ... then the next retry decides: 1 failure in 4 is below the threshold.
        self._feed(breaker, 540 * MS, [True])
        self.assertTrue(breaker.next_delay(_context(1, 550 * MS)).allow)
        self.assertIs(breaker.state, BreakerState.CLOSED)

    def test_a_failed_trial_opens_it_again(self):
        breaker = self._breaker()
        self._feed(breaker, 0, [False] * 4)
        breaker.next_delay(_context(1, 10 * MS))
        breaker.next_delay(_context(1, 600 * MS))
        self.assertIs(breaker.state, BreakerState.HALF_OPEN)
        self._feed(breaker, 610 * MS, [False, False, True, True])
        self.assertFalse(breaker.next_delay(_context(1, 620 * MS)).allow)
        self.assertIs(breaker.state, BreakerState.OPEN)
        # The wait starts over from the moment it re-opened.
        self.assertFalse(breaker.next_delay(_context(1, 1100 * MS)).allow)
        self.assertTrue(breaker.next_delay(_context(1, 1130 * MS)).allow)

    def test_the_window_starts_empty_after_every_change(self):
        breaker = self._breaker()
        self._feed(breaker, 0, [False] * 4)
        breaker.next_delay(_context(1, 10 * MS))  # opens, and forgets the four failures
        breaker.next_delay(_context(1, 600 * MS))  # half-open
        self._feed(breaker, 610 * MS, [True] * 4)
        self.assertTrue(breaker.next_delay(_context(1, 620 * MS)).allow)
        self.assertIs(breaker.state, BreakerState.CLOSED)


# ------------------------------------------------------------------ mapping

RELAY = """
name: relay
seed: 5
services:
  - name: svc
    latency: {median_ms: 20, lognorm_sigma: 0.4}
    workers: 4
    queue_capacity: 50
    retry: {type: fixed, max_attempts: 3, delay_ms: 5}
    timeout: {attempt_ms: 80, global_ms: 400}
    aimd_global_retry_budget: {min_rps: 5, max_rps: 50, initial_rps: 20}
    partial_failures: [{start_s: 1, end_s: 2, p_fail: 0.5}]
workload:
  base_rps: 100
  duration_s: 3
  rng_seed: 9
  load_spikes: [{start_s: 2, end_s: 3, rps_multiplier: 2}]
"""

TENANTS = """
name: tenants
seed: 5
services:
  - name: shared
    latency: {median_ms: 20, lognorm_sigma: 0.4}
    workers: 6
    queue_capacity: 50
    arolla_retry_budget: {alpha: 0.2, beta_up: 0.15}
    partial_failures: [{start_s: 1, end_s: 2, p_fail: 0.5}]
  - name: plain
    latency: {median_ms: 20}
    workers: 6
    timeout: {attempt_ms: 90, global_ms: 300}
clients:
  - name: timer
    target_service: shared
    retry: {type: fixed, max_attempts: 4, delay_ms: 0}
    timeout: {attempt_ms: 60}
    workload: &load {base_rps: 40, duration_s: 3, rng_seed: 11}
  - name: pooled
    target_service: shared
    replicas: 2
    retry: {type: jittered, max_attempts: 3, initial_delay_ms: 5, max_delay_ms: 40, jitter_mode: equal}
    retry_budget: {budget_ratio: 0.2, max_retries: 5, min_retries_per_sec: 3, shared_budget_id: pool}
    circuit_breaker: {type: retry_circuit_breaker, failure_threshold: 0.3, half_open_delay_ms: 0, window_duration_ms: 2000, min_requests: 10}
    workload: *load
  - name: breaker
    target_service: plain
    replicas: 2
    retry: {type: fixed, max_attempts: 2, delay_ms: 1}
    circuit_breaker: {type: time_based, failure_threshold: 0.1, half_open_delay_ms: 250, window_duration_ms: 1000, min_requests: 20}
    workload: *load
"""


class TestTheRelayClient(unittest.TestCase):
    """A top-level `workload`: one client that only relays the service's own retries."""

    def setUp(self):
        self.scenario = _build(RELAY)
        (self.client,) = self.scenario.clients

    def test_it_is_one_client_on_the_first_service(self):
        self.assertEqual(
            (self.client.name, self.client.group, self.client.target), ("client",) * 2 + ("svc",)
        )
        self.assertEqual(self.client.runtime.cfg.tenant, "client")

    def test_it_carries_the_services_retry_policy_inside_the_services_limiter(self):
        controlled, backoff = _layers(self.scenario, "client")
        self.assertIsInstance(controlled, ControlledRetryPolicy)
        self.assertIs(controlled.controllers[0], self.scenario.controllers["svc"])
        self.assertIsInstance(self.scenario.controllers["svc"], AimdBudget)
        self.assertEqual(backoff, FixedBackoffRetryPolicy(max_attempts=3, delay=5 * MS))

    def test_it_carries_the_services_timeouts_and_tells_the_server(self):
        cfg = self.client.runtime.cfg
        self.assertEqual(cfg.timeout_controller.get_attempt_timeout(), 80 * MS)
        self.assertEqual(cfg.timeout_controller.get_global_timeout(), 400 * MS)
        self.assertEqual(cfg.server_deadline, "attempt")

    def test_its_attempts_go_to_the_station_not_through_the_gate(self):
        self.assertIsInstance(self.client.runtime.service, ServiceRuntime)
        self.assertIs(self.client.runtime.service, self.scenario.stations["svc"])

    def test_the_station_and_the_workload_are_the_files(self):
        cfg = self.scenario.stations["svc"].cfg
        self.assertEqual((cfg.latency_median, cfg.workers, cfg.queue_capacity), (20 * MS, 4, 50))
        self.assertEqual(cfg.latency_lognorm_sigma, 0.4)
        self.assertEqual(cfg.partial_failures[0].p_fail, 0.5)
        workload = self.client.workload
        self.assertEqual((workload.base_rps, workload.duration_s, workload.rng_seed), (100, 3, 9))
        self.assertEqual(workload.load_spikes[0].rps_multiplier, 2)

    def test_it_runs(self):
        self.scenario.run()
        summary = self.client.runtime.metrics().summary()
        self.assertGreater(summary.total, 300)
        self.assertGreater(summary.succeeded, 0)
        self.assertGreater(summary.attempts_total, summary.total)  # the service's retries
        self.assertEqual(summary.dropped_rejected, 0)  # its retries stop; none is turned away
        self.assertEqual(self.scenario.stations["svc"].in_flight, 0)


class TestClientsWithTheirOwnPolicy(unittest.TestCase):
    def setUp(self):
        self.scenario = _build(TENANTS)

    def test_replicas_are_clients_of_their_own_each_a_tenant(self):
        self.assertEqual(
            [(c.name, c.group, c.target) for c in self.scenario.clients],
            [
                ("timer", "timer", "shared"),
                ("pooled.0", "pooled", "shared"),
                ("pooled.1", "pooled", "shared"),
                ("breaker.0", "breaker", "plain"),
                ("breaker.1", "breaker", "plain"),
            ],
        )
        for client in self.scenario.clients:
            self.assertEqual(client.runtime.cfg.tenant, client.name)
        self.assertEqual(len(self.scenario.group("pooled")), 2)

    def test_a_services_limiter_is_a_gate_in_front_of_it(self):
        gate = self.scenario.gates["shared"]
        self.assertIsInstance(gate, AdmissionGate)
        self.assertIs(gate.target, self.scenario.stations["shared"])
        self.assertEqual(gate.retry_scope, "hop")
        (controller,) = gate.controllers
        assert isinstance(controller, GoodputCoupledBudget)
        self.assertEqual(
            (controller.alpha, controller.beta_down, controller.beta_up), (0.2, 0.3, 0.15)
        )
        self.assertNotIn("plain", self.scenario.gates)
        for client in self.scenario.clients:
            expected = gate if client.target == "shared" else self.scenario.stations["plain"]
            self.assertIs(client.runtime.service, expected)

    def test_a_clients_own_attempt_timeout_is_kept_from_the_server(self):
        (timer,) = self.scenario.group("timer")
        cfg = timer.runtime.cfg
        self.assertEqual(cfg.server_deadline, "global")
        self.assertEqual(cfg.timeout_controller.get_attempt_timeout(), 60 * MS)
        self.assertIsNone(cfg.timeout_controller.get_global_timeout())

    def test_without_one_the_services_timeouts_apply_and_the_server_enforces_them(self):
        cfg = self.scenario.group("breaker")[0].runtime.cfg
        self.assertEqual(cfg.server_deadline, "attempt")
        self.assertEqual(cfg.timeout_controller.get_attempt_timeout(), 90 * MS)
        self.assertEqual(cfg.timeout_controller.get_global_timeout(), 300 * MS)
        # No timeout anywhere: none.
        cfg = self.scenario.group("pooled")[0].runtime.cfg
        self.assertIsNone(cfg.timeout_controller.get_attempt_timeout())
        self.assertEqual(cfg.server_deadline, "attempt")

    def test_the_policy_stack_is_budget_over_breaker_over_backoff(self):
        budget, gate, backoff = _layers(self.scenario, "pooled.0")
        self.assertIsInstance(budget, FirstAttemptFundedBudget)
        self.assertEqual(
            (budget.bank.budget_ratio, budget.bank.max_retries, budget.bank.min_retries_per_sec),
            (0.2, 5, 3),
        )
        self.assertIsInstance(gate, WindowRetryGate)
        self.assertEqual(
            (gate.failure_threshold, gate.window, gate.min_window_size), (0.3, 2 * S, 10)
        )
        self.assertIsInstance(backoff, ExponentialBackoffWithJitterRetryPolicy)
        self.assertIs(backoff.jitter_mode, JitterMode.EQUAL)
        self.assertIs(backoff.rng, self.scenario.sim.rng())

        breaker, backoff = _layers(self.scenario, "breaker.1")
        self.assertIsInstance(breaker, ThreeStateRetryBreaker)
        self.assertEqual(breaker.open_wait, 250 * MS)
        self.assertEqual(backoff, FixedBackoffRetryPolicy(max_attempts=2, delay=MS))

    def test_a_shared_budget_is_one_bank_and_a_backoff_each(self):
        one = _layers(self.scenario, "pooled.0")
        two = _layers(self.scenario, "pooled.1")
        self.assertIs(one[0].bank, two[0].bank)
        self.assertIsNot(one[0], two[0])
        self.assertIsNot(one[1], two[1])  # each its own window of outcomes
        self.assertIsNot(one[2], two[2])

    def test_arrival_streams(self):
        seeds = {c.name: c.workload.rng_seed for c in self.scenario.clients}
        # The first client of a service keeps the file's seed ...
        self.assertEqual(seeds["timer"], 11)
        self.assertEqual(seeds["breaker.0"], 11)  # ... on every service: the arms are paired
        # ... and everything else on the same service has a stream of its own.
        self.assertEqual(len({seeds["timer"], seeds["pooled.0"], seeds["pooled.1"]}), 3)
        self.assertEqual(seeds["pooled.1"], arrival_seed(11, position=1, replica=1))
        self.assertEqual(seeds["breaker.1"], arrival_seed(11, position=0, replica=1))

    def test_it_runs_and_the_gate_does_its_job(self):
        self.scenario.run()
        gate = self.scenario.gates["shared"]
        self.assertEqual(set(gate.counts), {"timer", "pooled.0", "pooled.1"})
        self.assertGreater(
            gate.counts["timer"].retries_asked, gate.counts["timer"].retries_admitted
        )
        self.assertGreater(gate.counts["timer"].retries_admitted, 0)
        self.assertEqual(gate.outstanding, 0)
        (timer,) = self.scenario.group("timer")
        reasons = {a.drop_reason for root in timer.runtime.roots for a in root.attempts}
        self.assertIn(DropReason.REJECTED, reasons)


class TestServiceLimiters(unittest.TestCase):
    def _controller(self, key: str, body: str):
        text = TENANTS.replace("arolla_retry_budget: {alpha: 0.2, beta_up: 0.15}", f"{key}: {body}")
        return _build(text).controllers["shared"]

    def test_each_limiter_is_its_controller(self):
        token = self._controller("global_retry_budget", "{target_rps: 30, max_burst: 7}")
        self.assertEqual(token, TokenBucketBudget(rate=30, burst=7))
        funded = self._controller("retry_budget", "{budget_ratio: 0.1, max_retries: 30}")
        self.assertEqual(funded, SuccessFundedBudget(ratio=0.1, max_retries=30))
        aimd = self._controller(
            "aimd_global_retry_budget",
            "{min_rps: 5, max_rps: 100, initial_rps: 40, additive_step: 5, window_ms: 500}",
        )
        assert isinstance(aimd, AimdBudget)
        self.assertEqual((aimd.min_rate, aimd.max_rate, aimd.initial_rate), (5, 100, 40))
        self.assertEqual((aimd.burst, aimd.additive_step, aimd.window), (10, 5, 500 * MS))
        gate = self._controller(
            "circuit_breaker",
            "{type: time_based, failure_threshold: 0.1, half_open_delay_ms: 500, "
            "window_duration_ms: 1000, min_requests: 20}",
        )
        assert isinstance(gate, FailureRateGate)
        self.assertEqual((gate.failure_threshold, gate.min_requests), (0.1, 20))
        self.assertEqual(
            (gate.window, gate.open_delay, gate.success_threshold), (S, 500 * MS, None)
        )


class TestRunControls(unittest.TestCase):
    def test_the_seed_and_the_arrival_seed_can_be_replaced(self):
        scenario = _build(TENANTS, seed=77, arrivals=500)
        default = _build(TENANTS)
        self.assertNotEqual(scenario.sim.rng().random(), default.sim.rng().random())
        seeds = {c.name: c.workload.rng_seed for c in scenario.clients}
        self.assertEqual(seeds["timer"], 500)
        self.assertEqual(seeds["pooled.0"], arrival_seed(500, position=1, replica=0))

    def test_a_run_is_reproducible_from_its_seeds(self):
        def totals(**kwargs) -> list[tuple[int, int]]:
            scenario = _build(TENANTS, **kwargs)
            scenario.run()
            return [
                (s.total, s.succeeded)
                for s in (c.runtime.metrics().summary() for c in scenario.clients)
            ]

        self.assertEqual(totals(seed=3, arrivals=4), totals(seed=3, arrivals=4))
        self.assertNotEqual(totals(seed=3, arrivals=4), totals(seed=3, arrivals=5))

    def test_the_horizon_can_be_cut(self):
        scenario = _build(TENANTS, duration_s=1)
        self.assertEqual({c.workload.duration_s for c in scenario.clients}, {1})
        scenario.run()
        self.assertLess(scenario.sim.timestep, 2 * S)


GRAPH = """
name: graph
seed: 3
services:
  - name: db
    latency: {median_ms: 10, lognorm_sigma: 0}
    workers: 4
    partial_failures: [{start_s: 0, end_s: 0.025, p_fail: 1}]
  - name: cache
    latency: {median_ms: 5, lognorm_sigma: 0}
    workers: 4
    partial_failures: [{start_s: 0, end_s: 10, p_fail: 1}]
  - name: app
    latency: {median_ms: 10, lognorm_sigma: 0}
    workers: 2
    timeout: {attempt_ms: 400}
    retry: {type: fixed, max_attempts: 3, delay_ms: 0}
    dependencies:
      - {service: cache, optional: true}
      - {service: db}
  - name: search
    latency: {median_ms: 10, lognorm_sigma: 0}
    workers: 2
    dependencies: [{service: db}]
  - name: gateway
    latency: {median_ms: 10, lognorm_sigma: 0}
    workers: 2
    dependencies: [{service: app}, {service: search}]
    dependency_call_pattern: parallel
    retry: {type: fixed, max_attempts: 2, delay_ms: 0}
clients:
  - name: user
    target_service: gateway
    retry: {type: fixed, max_attempts: 2, delay_ms: 0}
    timeout: {attempt_ms: 300, global_ms: 1000}
    workload: {base_rps: 1, duration_s: 1}
"""


class TestDependencies(unittest.TestCase):
    def setUp(self):
        self.scenario = _build(GRAPH)

    def test_a_service_in_a_call_graph_is_a_node_that_works_after_its_calls(self):
        nodes = self.scenario.nodes
        self.assertEqual(set(nodes), {"db", "cache", "app", "search", "gateway"})
        for name in ("app", "search", "gateway"):
            node = nodes[name]
            self.assertIs(node.service, self.scenario.stations[name])
            self.assertEqual((node.local_work, node.hold_worker_during_fanout), ("last", True))
        # A service that only gets called is a plain node.
        self.assertEqual(nodes["db"].downstreams, [])
        self.assertEqual(nodes["db"].local_work, "first")

    def test_sequential_dependencies_are_a_wave_each_and_parallel_ones_share_one(self):
        nodes = self.scenario.nodes
        app = nodes["app"]
        self.assertEqual(
            [(e.target.cfg.name, e.wave, e.optional) for e in app.downstreams],
            [("cache", 0, True), ("db", 1, False)],
        )
        self.assertEqual(app.wave_join, "short_circuit")
        gateway = nodes["gateway"]
        self.assertEqual(
            [(e.target.cfg.name, e.wave, e.optional) for e in gateway.downstreams],
            [("app", 0, False), ("search", 0, False)],
        )
        self.assertEqual(gateway.wave_join, "wait_all")  # it waits for every answer

    def test_what_a_service_calls_is_the_dependencys_own_retry_loop(self):
        loops = self.scenario.loops
        app_loop = loops["app"]
        self.assertIs(self.scenario.nodes["gateway"].downstreams[0].client, app_loop)
        self.assertEqual(app_loop.cfg.retry, FixedBackoffRetryPolicy(max_attempts=3, delay=0))
        self.assertEqual(app_loop.cfg.timeout_controller.get_attempt_timeout(), 400 * MS)
        self.assertIsNone(app_loop.cfg.timeout_controller.get_global_timeout())
        self.assertEqual(app_loop.cfg.server_deadline, "attempt")
        self.assertIs(app_loop.service, self.scenario.nodes["app"])
        self.assertIsInstance(loops["db"].cfg.retry, NoRetryPolicy)
        # One loop per service, whoever calls it.
        self.assertIs(self.scenario.nodes["app"].downstreams[1].client, loops["db"])
        self.assertIs(self.scenario.nodes["search"].downstreams[0].client, loops["db"])

    def test_a_caller_gets_one_answer_when_the_dependency_has_finished_retrying(self):
        """One request at t = 0. The db fails until 25 ms and the cache always.

        gateway calls app and search together. app: cache fails at 5 ms (optional,
        so on to the db), db fails at 15 ms, app retries: cache 15-20, db 20-30,
        own work 30-40. search: db fails at 10 ms, search fails, and gateway --
        which waits for app before it looks -- fails at 40 ms and retries:
        app 40-65 (cache 5, db 10, work 10), search 40-60, own work 65-75.
        """
        scenario = self.scenario
        (user,) = scenario.clients
        user.runtime.start_request(scenario.sim)
        scenario.sim.run()

        def spans(client) -> list[tuple[int, int, bool]]:
            return [
                (round(a.interval.begin / MS), round(a.interval.end / MS), a.success)
                for root in client.roots
                for a in root.attempts
            ]

        self.assertEqual(
            spans(scenario.loops["app"]), [(0, 15, False), (15, 40, True), (40, 65, True)]
        )
        self.assertEqual(spans(scenario.loops["search"]), [(0, 10, False), (40, 60, True)])
        self.assertEqual(spans(scenario.loops["gateway"]), [(0, 40, False), (40, 75, True)])
        # The user made one call and saw one answer: the gateway's own retry is its business.
        self.assertEqual(spans(user.runtime), [(0, 75, True)])
        self.assertEqual(user.runtime.roots[0].attempts[0].service, "gateway.retries")
        for station in scenario.stations.values():
            self.assertEqual((station.in_flight, station.queue_len()), (0, 0))


class TestTwoLoopsOnOneHop(unittest.TestCase):
    def _user(self, text: str):
        scenario = _build(text)
        (user,) = scenario.clients
        return scenario, user.runtime

    def test_a_retrying_client_calls_a_retrying_service_through_its_loop(self):
        scenario, user = self._user(GRAPH)
        layer = user.service
        self.assertIsInstance(layer, RetryLayer)
        self.assertIs(layer.client, scenario.loops["gateway"])
        self.assertEqual(layer.cfg.name, "gateway.retries")
        # The client keeps its own policy and its own timer.
        self.assertEqual(user.cfg.retry, FixedBackoffRetryPolicy(max_attempts=2, delay=0))
        self.assertEqual(user.cfg.timeout_controller.get_attempt_timeout(), 300 * MS)
        self.assertEqual(user.cfg.timeout_controller.get_global_timeout(), 1000 * MS)
        self.assertEqual(user.cfg.server_deadline, "global")
        self.assertEqual(scenario.loops["gateway"].cfg.retry, FixedBackoffRetryPolicy(2, 0))

    def test_two_attempt_timeouts_are_two_loops_too(self):
        text = TENANTS.replace(
            "    retry: {type: fixed, max_attempts: 2, delay_ms: 1}",
            "    retry: {type: fixed, max_attempts: 2, delay_ms: 1}\n    timeout: {attempt_ms: 50}",
        )
        scenario = _build(text)
        breaker = scenario.group("breaker")[0].runtime
        assert isinstance(breaker.service, RetryLayer)
        inner = breaker.service.client
        self.assertIsInstance(inner.cfg.retry, NoRetryPolicy)  # the service does not retry ...
        self.assertEqual(
            inner.cfg.timeout_controller.get_attempt_timeout(), 90 * MS
        )  # ... but times out
        self.assertEqual(breaker.cfg.timeout_controller.get_attempt_timeout(), 50 * MS)
        self.assertEqual(breaker.cfg.server_deadline, "global")
        # The client brings its own timeouts, so the service's global one is not read.
        self.assertIsNone(breaker.cfg.timeout_controller.get_global_timeout())

    def test_a_client_without_a_timer_tells_the_loop_its_global_deadline(self):
        text = GRAPH.replace("    timeout: {attempt_ms: 300, global_ms: 1000}\n", "")
        _, user = self._user(text)
        self.assertIsInstance(user.service, RetryLayer)
        self.assertIsNone(user.cfg.timeout_controller.get_attempt_timeout())
        self.assertEqual(user.cfg.server_deadline, "attempt")


class TestWhatIsNotMappedYet(unittest.TestCase):
    def _unsupported(self, text: str) -> str:
        with self.assertRaises(UnsupportedScenario) as raised:
            _build(text)
        return str(raised.exception)

    def test_each_is_named(self):
        budget = TENANTS.replace("    timeout: {attempt_ms: 60}", "    e2e_retry_budget: 2")
        self.assertIn("e2e_retry_budget", self._unsupported(budget))
        limiter = TENANTS.replace(
            "arolla_retry_budget: {alpha: 0.2, beta_up: 0.15}",
            "rate_limiter: {type: bursty, max_requests: 5, period_ms: 1000}",
        )
        self.assertIn("rate_limiter on a service", self._unsupported(limiter))
        between = GRAPH.replace(
            "    timeout: {attempt_ms: 400}", "    global_retry_budget: {target_rps: 10}"
        )
        self.assertIn("admission between services", self._unsupported(between))
        both = TENANTS.replace(
            "    arolla_retry_budget: {alpha: 0.2, beta_up: 0.15}",
            "    arolla_retry_budget: {alpha: 0.2, beta_up: 0.15}\n"
            "    retry: {type: fixed, max_attempts: 2, delay_ms: 0}",
        )
        self.assertIn("a limiter and a retry loop of its own", self._unsupported(both))
        relay_timer = RELAY.replace(
            "workload:\n  base_rps: 100",
            "clients:\n  - name: c\n    timeout: {attempt_ms: 50}\n    workload:\n      base_rps: 100",
        ).replace(
            "  duration_s: 3\n  rng_seed: 9\n  load_spikes: [{start_s: 2, end_s: 3, rps_multiplier: 2}]\n",
            "      duration_s: 3\n",
        )
        self.assertIn(
            "an attempt timeout on a client without a retry", self._unsupported(relay_timer)
        )


class TestSmallThings(unittest.TestCase):
    def test_no_retry_policy_at_all(self):
        text = RELAY.replace(
            "    retry: {type: fixed, max_attempts: 3, delay_ms: 5}\n", ""
        ).replace("    aimd_global_retry_budget: {min_rps: 5, max_rps: 50, initial_rps: 20}\n", "")
        (policy,) = _layers(_build(text), "client")
        self.assertIsInstance(policy, NoRetryPolicy)

    def test_arrival_seed_rule(self):
        self.assertEqual(arrival_seed(7), 7)
        self.assertIsNone(arrival_seed(None))
        self.assertEqual(arrival_seed(None, 1, 0), arrival_seed(0, 1, 0))
        seeds = {arrival_seed(7, p, r) for p in range(4) for r in range(40)}
        self.assertEqual(len(seeds), 160)


#: A checkout of https://github.com/eth-easl/Arolla, to run the test below.
AROLLA_CHECKOUT = os.environ.get("AROLLA_CHECKOUT")


@unittest.skipUnless(AROLLA_CHECKOUT, "set AROLLA_CHECKOUT to a checkout of Arolla to run this")
class TestTheScenariosArollaShips(unittest.TestCase):
    def test_every_file_is_read_and_built(self):
        from msim.compat.arolla import read

        assert AROLLA_CHECKOUT is not None
        root = Path(AROLLA_CHECKOUT).expanduser() / "simulator" / "experiments" / "yaml"
        files = sorted(root.rglob("*.yaml"))
        self.assertEqual(len(files), 44)
        points = 0
        for path in files:
            for point in read(path):
                points += 1
                build(parse(point.config))
        self.assertEqual(points, 183)


if __name__ == "__main__":
    unittest.main()
