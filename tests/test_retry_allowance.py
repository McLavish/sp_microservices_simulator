"""A retry allowance for the whole request (`core.RetryAllowance`).

A client with `ClientConfig.retry_allowance` gives every request it starts a
number of retries the request may cause, anywhere in the call tree. A client
whose policy is wrapped in `RetryAllowancePolicy` takes a unit for each retry it
issues and stops retrying when none is left.
"""

import unittest
from typing import Any

from msim.caller import ClientConfig, ClientRuntime, RetryLayer
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import (
    DropReason,
    RequestContext,
    RetryAllowance,
    TimeInterval,
    ms_to_ns,
)
from msim.faults import PartialFailure
from msim.retry_policies import (
    FixedBackoffRetryPolicy,
    RetryAllowancePolicy,
    RetryBudgetPolicy,
    RetryContext,
    RetryPolicy,
)
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController

MS = ms_to_ns(1)


def _station(name: str = "svc", *, median_ms: float = 10, fail_until_ms: float | None = 100_000):
    failures = []
    if fail_until_ms is not None:
        failures.append(PartialFailure(TimeInterval(0, ms_to_ns(fail_until_ms)), p_fail=1.0))
    return ServiceRuntime(
        cfg=ServiceConfig(
            name=name,
            latency_median=ms_to_ns(median_ms),
            latency_lognorm_sigma=0.0,
            workers=16,
            partial_failures=failures,
        )
    ).bind()


def _backoff(attempts: int, delay_ms: float = 0) -> FixedBackoffRetryPolicy:
    return FixedBackoffRetryPolicy(max_attempts=attempts, delay=ms_to_ns(delay_ms))


def _charged(attempts: int, delay_ms: float = 0) -> RetryAllowancePolicy:
    return RetryAllowancePolicy(_backoff(attempts, delay_ms))


def _client(
    target,
    retry: RetryPolicy,
    *,
    allowance: int | None = None,
    attempt_ms: float = 1000,
    global_ms: float | None = None,
) -> ClientRuntime:
    return ClientRuntime(
        cfg=ClientConfig(
            retry=retry,
            timeout_controller=StaticTimeoutController(
                global_timeout=None if global_ms is None else ms_to_ns(global_ms),
                attempt_timeout=ms_to_ns(attempt_ms),
            ),
            retry_allowance=allowance,
        ),
        service=target,
    )


def _run(client: ClientRuntime, requests: int = 1) -> Simulator:
    sim = Simulator(seed=1)
    for _ in range(requests):
        client.start_request(sim)
        sim.run()
    return sim


class TestOneClient(unittest.TestCase):
    def test_the_allowance_bounds_the_retries_of_a_request(self):
        station = _station()
        client = _client(station, _charged(attempts=10), allowance=2)
        _run(client)
        self.assertEqual(len(client.roots[0].attempts), 3)  # the first try and two retries
        self.assertEqual(station.admitted_total, 3)

    def test_every_request_gets_its_own(self):
        client = _client(_station(), _charged(attempts=10), allowance=2)
        _run(client, requests=3)
        self.assertEqual([len(root.attempts) for root in client.roots], [3, 3, 3])

    def test_the_policys_own_limit_still_applies(self):
        client = _client(_station(), _charged(attempts=2), allowance=5)
        _run(client)
        self.assertEqual(len(client.roots[0].attempts), 2)

    def test_an_allowance_of_zero_means_no_retries(self):
        client = _client(_station(), _charged(attempts=10), allowance=0)
        _run(client)
        self.assertEqual(len(client.roots[0].attempts), 1)

    def test_a_request_without_an_allowance_is_not_limited(self):
        client = _client(_station(), _charged(attempts=4))
        _run(client)
        self.assertEqual(len(client.roots[0].attempts), 4)

    def test_a_client_without_the_layer_does_not_spend_it(self):
        client = _client(_station(), _backoff(attempts=4), allowance=1)
        _run(client)
        self.assertEqual(len(client.roots[0].attempts), 4)

    def test_a_request_that_arrives_with_a_context_keeps_what_it_came_with(self):
        client = _client(_station(), _charged(attempts=10), allowance=1)
        sim = Simulator(seed=1)
        client.start_request(sim, ctx=RequestContext(tenant="t"))  # no allowance in it
        allowance = RetryAllowance(remaining=3)
        client.start_request(sim, ctx=RequestContext(retry_allowance=allowance))
        sim.run()
        self.assertEqual(sorted(len(root.attempts) for root in client.roots), [4, 10])
        self.assertEqual(allowance.remaining, 0)

    def test_validation(self):
        bad: list[Any] = [-1, True, 1.5]
        for value in bad:
            with self.subTest(retry_allowance=value), self.assertRaises(ValueError):
                _client(_station(), _backoff(2), allowance=value)


class TestAcrossHops(unittest.TestCase):
    def _tree(self, *, allowance: int, top: RetryPolicy, edge: RetryPolicy):
        """top client -> P -> edge client -> a leaf that always fails."""
        leaf = _station("leaf", median_ms=1)
        parent = ServiceNodeRuntime(service=_station("P", median_ms=1, fail_until_ms=None))
        edge_client = _client(leaf, edge)
        parent.add_downstream(DownstreamEdge(client=edge_client, target=parent))
        top_client = _client(parent, top, allowance=allowance)
        return top_client, edge_client, leaf

    def test_retries_below_draw_on_the_requests_allowance(self):
        # The originator retries freely (3 attempts, no layer); the edge is charged.
        top, edge, leaf = self._tree(allowance=4, top=_backoff(3), edge=_charged(5))
        _run(top)
        # Under the first attempt the edge spends all four retries; under the
        # second and third it has none left.
        self.assertEqual([len(root.attempts) for root in edge.roots], [5, 1, 1])
        self.assertEqual(leaf.admitted_total, 7)
        self.assertEqual(len(top.roots[0].attempts), 3)

    def test_without_it_the_layers_multiply(self):
        top, edge, leaf = self._tree(allowance=4, top=_backoff(3), edge=_backoff(5))
        _run(top)
        self.assertEqual(leaf.admitted_total, 15)

    def test_every_client_with_the_layer_is_charged(self):
        top, edge, leaf = self._tree(allowance=4, top=_charged(3), edge=_charged(5))
        _run(top)
        # The edge spends the four under the first attempt, and the originator
        # then has nothing to retry with.
        self.assertEqual([len(root.attempts) for root in edge.roots], [5])
        self.assertEqual(len(top.roots[0].attempts), 1)
        self.assertEqual(leaf.admitted_total, 5)

    def test_two_branches_share_one_allowance(self):
        parent = ServiceNodeRuntime(service=_station("P", median_ms=1, fail_until_ms=None))
        leaves = [_station(f"leaf-{i}", median_ms=1) for i in range(2)]
        edges = [_client(leaf, _charged(10)) for leaf in leaves]
        for edge in edges:
            parent.add_downstream(DownstreamEdge(client=edge, target=parent))
        top = _client(parent, _backoff(1), allowance=5)
        _run(top)
        retries = sum(len(root.attempts) - 1 for edge in edges for root in edge.roots)
        # The first branch to fail ends the parent's attempt; between them the
        # two never issue more than the five the request has.
        self.assertLessEqual(retries, 5)
        self.assertGreater(retries, 0)

    def test_it_crosses_a_retry_layer(self):
        station = _station()
        inner = _client(station, _charged(10))
        outer = _client(RetryLayer(inner), _backoff(2), allowance=3)
        _run(outer)
        self.assertEqual([len(root.attempts) for root in inner.roots], [4, 1])
        self.assertEqual(station.admitted_total, 5)


class TestRefunds(unittest.TestCase):
    def test_a_retry_the_deadline_vetoes_gives_its_unit_back(self):
        """The attempt is cut at the 10 ms global deadline; the retry would start there."""
        allowance = RetryAllowance(remaining=2)
        client = _client(_station(median_ms=50), _charged(5), global_ms=10)
        sim = Simulator(seed=1)
        client.start_request(sim, ctx=RequestContext(retry_allowance=allowance))
        sim.run()
        self.assertEqual(len(client.roots[0].attempts), 1)
        self.assertEqual(allowance.remaining, 2)

    def test_a_retry_cancelled_while_it_waits_gives_its_unit_back(self):
        """The refund arrives long after the charge: the layer says that is safe."""
        allowance = RetryAllowance(remaining=2)
        client = _client(_station(), _charged(5, delay_ms=100))
        sim = Simulator(seed=1)
        root = client.start_request(sim, ctx=RequestContext(retry_allowance=allowance))
        seen: list[int] = []

        def cancel() -> None:
            seen.append(allowance.remaining)  # one retry granted and pending
            assert root.cancel is not None
            root.cancel()

        sim.schedule(50 * MS, cancel)
        sim.run()
        self.assertEqual(seen, [1])
        self.assertEqual(allowance.remaining, 2)
        self.assertTrue(root.cancelled)
        self.assertTrue(_charged(2).supports_deferred_rollback)

    def test_a_refusal_below_costs_nothing(self):
        """The wrapped budget is empty: no retry, no charge."""
        allowance = RetryAllowance(remaining=2)
        budget = RetryBudgetPolicy(budget_ratio=0.1, max_retries=1, underlying_strategy=_backoff(5))
        while budget.next_delay(RetryContext(attempt=1, now=0)).allow:
            pass
        client = _client(_station(), RetryAllowancePolicy(budget))
        sim = Simulator(seed=1)
        client.start_request(sim, ctx=RequestContext(retry_allowance=allowance))
        sim.run()
        self.assertEqual(len(client.roots[0].attempts), 1)
        self.assertEqual(allowance.remaining, 2)

    def test_an_exhausted_allowance_does_not_ask_the_wrapped_policy(self):
        budget = RetryBudgetPolicy(budget_ratio=0.1, max_retries=5, underlying_strategy=_backoff(5))
        before = budget.get_token_balance()
        client = _client(_station(), RetryAllowancePolicy(budget), allowance=0)
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()
        self.assertEqual(len(client.roots[0].attempts), 1)
        # Only the request's own deposit moved the budget: nothing was withdrawn.
        self.assertGreaterEqual(budget.get_token_balance(), before)
        self.assertEqual(client.roots[0].attempts[0].drop_reason, DropReason.SERVER_FAILURE)


if __name__ == "__main__":
    unittest.main()
