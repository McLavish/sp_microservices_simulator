"""Retry by failure cause (`RetryByCausePolicy`).

A deployed client does not retry everything: gRPC retries the status codes it
is told are retryable, and a deadline or a full queue is usually not among them.
The policy chooses the retry policy by the cause of the failed attempt, and two
things about it must hold whatever sits in its branches:

* every distinct layer hears of each request and each result ONCE, however many
  causes lead to it -- a budget shared by two causes is one budget;
* a retry that was granted and then vetoed is refunded in the branch that
  granted it, and in no other.
"""

import unittest

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import DropReason, TimeInterval, ms_to_ns, s_to_ns
from msim.faults import LatencyInjection, PartialFailure
from msim.retry_policies import (
    CBState,
    CountBasedCircuitBreakerPolicy,
    DelayDecision,
    FixedBackoffRetryPolicy,
    NoRetryPolicy,
    RetryBudgetPolicy,
    RetryByCausePolicy,
    RetryContext,
    policy_layers,
)
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController

SERVER, DEADLINE, QUEUE, OPEN = (
    DropReason.SERVER_FAILURE,
    DropReason.DEADLINE,
    DropReason.QUEUE_FULL,
    DropReason.CIRCUIT_OPEN,
)


def _context(cause: DropReason | None, attempt: int = 1, now: int = 0) -> RetryContext:
    return RetryContext(attempt=attempt, now=now, drop_reason=cause)


def _backoff(max_attempts: int = 3, delay_ms: float = 1.0) -> FixedBackoffRetryPolicy:
    return FixedBackoffRetryPolicy(max_attempts=max_attempts, delay=ms_to_ns(delay_ms))


def _budget(underlying, ratio: float = 0.5, max_retries: int = 4) -> RetryBudgetPolicy:
    return RetryBudgetPolicy(
        budget_ratio=ratio, max_retries=max_retries, underlying_strategy=underlying
    )


def _service(
    sim: Simulator,
    *,
    name: str = "svc",
    latency_median: int = ms_to_ns(2),
    latency_lognorm_sigma: float = 0.05,
    workers: int = 4,
    queue_capacity: int | None = None,
    partial_failures: list[PartialFailure] | None = None,
    latency_injections: list[LatencyInjection] | None = None,
) -> ServiceRuntime:
    return ServiceRuntime(
        cfg=ServiceConfig(
            name=name,
            latency_median=latency_median,
            latency_lognorm_sigma=latency_lognorm_sigma,
            workers=workers,
            queue_capacity=queue_capacity,
            partial_failures=partial_failures or [],
            latency_injections=latency_injections or [],
        )
    ).bind()


def _client(service, policy, *, attempt_ms: float | None = None, global_ms: float | None = None):
    return ClientRuntime(
        cfg=ClientConfig(
            retry=policy,
            timeout_controller=StaticTimeoutController(
                global_timeout=None if global_ms is None else ms_to_ns(global_ms),
                attempt_timeout=None if attempt_ms is None else ms_to_ns(attempt_ms),
            ),
        ),
        service=service,
    )


def _always_failing():
    return [PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=1.0)]


def _causes(client: ClientRuntime) -> list[list[DropReason]]:
    return [[attempt.drop_reason for attempt in root.attempts] for root in client.roots]


class TestTheDecision(unittest.TestCase):
    def test_a_listed_cause_is_decided_by_its_policy_and_any_other_is_not_retried(self):
        policy = RetryByCausePolicy({SERVER: _backoff(3, 5), DEADLINE: _backoff(2, 7)})

        self.assertEqual(policy.next_delay(_context(SERVER)), DelayDecision(True, ms_to_ns(5)))
        self.assertEqual(policy.next_delay(_context(DEADLINE)), DelayDecision(True, ms_to_ns(7)))
        self.assertEqual(policy.next_delay(_context(QUEUE)), DelayDecision(False, 0))
        self.assertEqual(policy.next_delay(_context(OPEN)), DelayDecision(False, 0))
        # A caller that does not say why the attempt failed gets no retry.
        self.assertEqual(policy.next_delay(_context(None)), DelayDecision(False, 0))

    def test_the_allowance_counts_every_attempt_whatever_it_failed_with(self):
        """The policy is stateless: each branch sees the request's attempt number."""
        policy = RetryByCausePolicy({SERVER: _backoff(3), DEADLINE: _backoff(2)})

        self.assertTrue(policy.next_delay(_context(SERVER, attempt=2)).allow)
        self.assertFalse(policy.next_delay(_context(DEADLINE, attempt=2)).allow)
        self.assertFalse(policy.next_delay(_context(SERVER, attempt=3)).allow)

    def test_for_causes_is_the_fixed_list(self):
        backoff = _backoff()
        policy = RetryByCausePolicy.for_causes([SERVER, OPEN], backoff)

        self.assertEqual(set(policy.policies), {SERVER, OPEN})
        self.assertIs(policy.policies[SERVER], backoff)
        self.assertIs(policy.policies[OPEN], backoff)
        self.assertFalse(policy.next_delay(_context(DEADLINE)).allow)
        self.assertFalse(
            RetryByCausePolicy.for_causes([], backoff).next_delay(_context(SERVER)).allow
        )

    def test_the_mapping_is_copied(self):
        mapping = {SERVER: _backoff()}
        policy = RetryByCausePolicy(mapping)
        mapping[DEADLINE] = _backoff()

        self.assertFalse(policy.next_delay(_context(DEADLINE)).allow)

    def test_a_mapping_that_could_never_run_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a DropReason"):
            RetryByCausePolicy({"SERVER_FAILURE": _backoff()})  # ty: ignore[invalid-argument-type]
        for never in (DropReason.NONE, DropReason.CANCELLED):
            with self.assertRaisesRegex(ValueError, "never"):
                RetryByCausePolicy({never: _backoff()})
        with self.assertRaisesRegex(ValueError, "must be a RetryPolicy"):
            RetryByCausePolicy({SERVER: 3})  # ty: ignore[invalid-argument-type]


class TestTheWalk(unittest.TestCase):
    def test_without_a_context_every_branch_is_walked_and_each_layer_once(self):
        backoff = _backoff()
        budget = _budget(backoff)
        policy = RetryByCausePolicy({SERVER: backoff, DEADLINE: budget, QUEUE: budget})

        layers = list(policy_layers(policy))
        self.assertEqual(len(layers), 3)
        for layer, expected in zip(layers, (policy, backoff, budget), strict=True):
            self.assertIs(layer, expected)

    def test_with_the_context_of_a_decision_only_its_branch_is_walked(self):
        backoff = _backoff()
        budget = _budget(backoff)
        policy = RetryByCausePolicy({SERVER: backoff, DEADLINE: budget})

        self.assertEqual(
            [id(layer) for layer in policy_layers(policy, _context(DEADLINE))],
            [id(policy), id(budget), id(backoff)],
        )
        self.assertEqual(
            [id(layer) for layer in policy_layers(policy, _context(SERVER))],
            [id(policy), id(backoff)],
        )
        self.assertEqual(
            [id(layer) for layer in policy_layers(policy, _context(QUEUE))], [id(policy)]
        )

    def test_layers_above_the_branch_point_are_always_walked(self):
        by_cause = RetryByCausePolicy({SERVER: _backoff()})
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(5, 10),
            success_threshold_ratio=(3, 5),
            underlying_strategy=by_cause,
            half_open_delay=0,
        )

        self.assertEqual([type(layer).__name__ for layer in policy_layers(breaker)], [
            "CountBasedCircuitBreakerPolicy", "RetryByCausePolicy", "FixedBackoffRetryPolicy"
        ])  # fmt: skip
        self.assertEqual([type(layer).__name__ for layer in policy_layers(breaker, _context(QUEUE))], [
            "CountBasedCircuitBreakerPolicy", "RetryByCausePolicy"
        ])  # fmt: skip


class TestTheClient(unittest.TestCase):
    """The cause the client passes is the one it booked the attempt under."""

    def test_a_server_failure_is_retried_and_a_deadline_is_not(self):
        policy = RetryByCausePolicy.for_causes([SERVER, OPEN], _backoff(3))

        sim = Simulator(seed=1)
        failing = _client(_service(sim, partial_failures=_always_failing()), policy)
        failing.start_request(sim)
        sim.run()
        self.assertEqual(_causes(failing), [[SERVER, SERVER, SERVER]])

        sim = Simulator(seed=1)
        slow = _client(_service(sim, latency_median=ms_to_ns(50)), policy, attempt_ms=5)
        slow.start_request(sim)
        sim.run()
        self.assertEqual(_causes(slow), [[DEADLINE]])

    def test_the_same_failures_under_a_plain_backoff_are_all_retried(self):
        """The control: what the policy changes is the deadline's retries, nothing else."""
        sim = Simulator(seed=1)
        slow = _client(_service(sim, latency_median=ms_to_ns(50)), _backoff(3), attempt_ms=5)
        slow.start_request(sim)
        sim.run()
        self.assertEqual(_causes(slow), [[DEADLINE, DEADLINE, DEADLINE]])

    def test_a_full_queue_is_a_cause_of_its_own(self):
        sim = Simulator(seed=1)
        service = _service(sim, latency_median=ms_to_ns(20), workers=1, queue_capacity=0)
        only_queue = _client(service, RetryByCausePolicy.for_causes([QUEUE], _backoff(2, 30)))
        never = _client(service, RetryByCausePolicy.for_causes([SERVER], _backoff(2, 30)))

        only_queue.start_request(sim)  # takes the one worker
        only_queue.start_request(sim)  # rejected, retried after 30 ms, served
        never.start_request(sim)  # rejected, not retried
        sim.run()

        self.assertEqual(_causes(never), [[QUEUE]])
        self.assertEqual(
            sorted(_causes(only_queue), key=len), [[DropReason.NONE], [QUEUE, DropReason.NONE]]
        )

    def test_an_attempt_that_ran_out_of_time_is_a_deadline_whatever_the_service_said(self):
        """Half-open rule: a failure that lands AT the deadline is booked DEADLINE, and
        that is the cause the policy sees."""
        seen: list[DropReason | None] = []

        class Probe(NoRetryPolicy):
            def next_delay(self, context):
                seen.append(context.drop_reason)
                return DelayDecision(False, 0)

        sim = Simulator(seed=1)
        service = _service(
            sim,
            latency_median=ms_to_ns(2),
            latency_lognorm_sigma=0.0,
            partial_failures=_always_failing(),
            latency_injections=[
                LatencyInjection(TimeInterval(s_to_ns(1), s_to_ns(2)), add_latency=ms_to_ns(100))
            ],
        )
        client = _client(service, Probe(), attempt_ms=10)
        client.start_request(sim)  # fails in 2 ms: the server's failure
        sim.schedule(s_to_ns(1), lambda: client.start_request(sim))  # 102 ms: cut at 10 ms
        sim.run()

        self.assertEqual(seen, [SERVER, DEADLINE])
        self.assertEqual(_causes(client), [[SERVER], [DEADLINE]])

    def test_a_node_relays_the_cause_of_the_call_that_failed_it(self):
        """A caller two hops up retries by what happened at the leaf."""
        sim = Simulator(seed=1)
        leaf = ServiceNodeRuntime(
            service=_service(
                sim, name="leaf", latency_median=ms_to_ns(20), workers=1, queue_capacity=0
            )
        )
        middle = ServiceNodeRuntime(service=_service(sim, name="middle"))
        middle.add_downstream(DownstreamEdge(client=_client(leaf, NoRetryPolicy()), target=leaf))
        top = _client(middle, RetryByCausePolicy.for_causes([SERVER], _backoff(3)))

        top.start_request(sim)  # reaches the leaf first and is served
        top.start_request(sim)  # finds the leaf full: QUEUE_FULL, relayed, not retried
        sim.run()

        self.assertEqual(sorted(_causes(top), key=str), [[DropReason.NONE], [QUEUE]])


class TestOneBudgetForSomeCauses(unittest.TestCase):
    def _policy(self):
        backoff = _backoff(4, 1)
        budget = _budget(backoff, ratio=0.5, max_retries=4)
        policy = RetryByCausePolicy({SERVER: backoff, DEADLINE: budget, QUEUE: budget})
        return policy, budget

    def test_a_shared_budget_gets_one_deposit_per_request(self):
        policy, budget = self._policy()
        while budget.can_retry():  # drain, so a deposit shows below the cap
            budget.next_delay(_context(DEADLINE))
        sim = Simulator(seed=1)
        client = _client(_service(sim), policy)
        before = budget._tokens

        for _ in range(3):
            client.start_request(sim)
        sim.run()

        self.assertEqual(budget._tokens - before, 3 * budget._deposit)

    def test_only_the_budgeted_causes_spend_it(self):
        policy, budget = self._policy()
        full = budget._tokens
        sim = Simulator(seed=1)
        failing = _client(_service(sim, partial_failures=_always_failing()), policy)
        failing.start_request(sim)
        sim.run()
        self.assertEqual(len(failing.roots[0].attempts), 4)  # three free retries
        self.assertEqual(budget._tokens, full)

        sim = Simulator(seed=1)
        slow = _client(_service(sim, latency_median=ms_to_ns(50)), policy, attempt_ms=5)
        slow.start_request(sim)
        sim.run()
        self.assertEqual(len(slow.roots[0].attempts), 4)
        self.assertEqual(full - budget._tokens, 3 * budget._retry_cost)

    def test_a_dry_budget_stops_its_causes_and_not_the_others(self):
        policy, budget = self._policy()
        while budget.can_retry():
            budget.next_delay(_context(DEADLINE))

        self.assertFalse(policy.next_delay(_context(DEADLINE)).allow)
        self.assertFalse(policy.next_delay(_context(QUEUE)).allow)
        self.assertTrue(policy.next_delay(_context(SERVER)).allow)

    def test_a_vetoed_retry_is_refunded_in_the_branch_that_granted_it(self):
        policy, budget = self._policy()
        full = budget._tokens
        sim = Simulator(seed=1)
        # The retry would start 1 ms after the 5 ms attempt timeout; the 5.5 ms global
        # deadline vetoes it after the budget has been charged.
        slow = _client(
            _service(sim, latency_median=ms_to_ns(50)), policy, attempt_ms=5, global_ms=5.5
        )
        slow.start_request(sim)
        sim.run()

        self.assertEqual(_causes(slow), [[DEADLINE]])
        self.assertEqual(budget._tokens, full)  # charged 2, then refunded (at the cap)

    def test_a_retry_another_branch_granted_refunds_nothing_here(self):
        """Refunding every branch would MINT budget: this one charged nothing."""
        policy, budget = self._policy()
        while budget.can_retry():
            budget.next_delay(_context(DEADLINE))
        sim = Simulator(seed=1)
        # Fails in ~2 ms with SERVER_FAILURE; its free retry (1 ms later) is vetoed by the
        # 2.5 ms global deadline.
        failing = _client(
            _service(sim, partial_failures=_always_failing(), latency_lognorm_sigma=0.0),
            policy,
            global_ms=2.5,
        )
        before = budget._tokens

        failing.start_request(sim)
        sim.run()

        self.assertEqual(_causes(failing), [[SERVER]])
        # Exactly the request's own deposit, and no refund on top of it.
        self.assertEqual(budget._tokens - before, budget._deposit)


class TestUnderABreaker(unittest.TestCase):
    def test_the_breaker_hears_every_result_and_gates_admission(self):
        by_cause = RetryByCausePolicy.for_causes([SERVER], _backoff(2, 1))
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(4, 4),
            success_threshold_ratio=(1, 1),
            underlying_strategy=by_cause,
            half_open_delay=s_to_ns(1000),
        )
        sim = Simulator(seed=1)
        client = _client(_service(sim, partial_failures=_always_failing()), breaker)

        client.start_request(sim)
        sim.run()
        client.start_request(sim)
        sim.run()
        self.assertEqual(breaker.get_state(), CBState.OPEN)  # four failed attempts
        client.start_request(sim)
        self.assertEqual(_causes(client), [[SERVER, SERVER], [SERVER, SERVER], [OPEN]])


if __name__ == "__main__":
    unittest.main()
