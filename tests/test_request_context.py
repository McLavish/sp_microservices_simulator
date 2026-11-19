"""A request context travels with a request (`core.RequestContext`).

A client with a tenant gives every request it starts a context; from there it
reaches every attempt the request causes, down every edge of the call tree and
over every link. The client that issues an attempt sets the two hop-relative
fields (`attempt`, `is_retry`); the tenant and the retry allowance are the
request's and are handed on untouched.

A request without a context is handled exactly as before: no target is called
with a `ctx` argument, so one that only knows the old three arguments works.
"""

import unittest
from typing import Any

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import DropReason, RequestContext, RetryAllowance, ms_to_ns
from msim.retry_policies import (
    DelayDecision,
    FixedBackoffRetryPolicy,
    NoRetryPolicy,
    RetryContext,
    RetryPolicy,
)
from msim.service import AttemptControl, ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.transport import TransportDelay, TransportLink
from msim.workload import Workload


def _cfg(name: str) -> ServiceConfig:
    return ServiceConfig(
        name=name, latency_median=ms_to_ns(1), latency_lognorm_sigma=0.0, workers=4
    )


class _Recorder:
    """A target that answers at once and records HOW it was called.

    `calls` holds the keyword arguments beyond the three every target has always
    received: `{}` for a call made the old way, `{"ctx": ...}` for one that
    carries a context.
    """

    def __init__(self, name: str = "leaf", outcomes: tuple[bool, ...] = ()) -> None:
        self.cfg = _cfg(name)
        self.calls: list[dict] = []
        self._outcomes = list(outcomes)

    def submit_attempt(self, sim, on_done, attempt_deadline=None, **extra) -> AttemptControl:
        self.calls.append(extra)
        ok = self._outcomes.pop(0) if self._outcomes else True
        on_done(ok, 1, DropReason.NONE if ok else DropReason.SERVER_FAILURE, 0, False)
        return AttemptControl()

    @property
    def contexts(self) -> list[RequestContext]:
        return [call["ctx"] for call in self.calls]


class _ThreeArguments:
    """A target written before contexts existed: it accepts nothing else."""

    def __init__(self) -> None:
        self.cfg = _cfg("old")
        self.attempts = 0

    def submit_attempt(self, sim, on_done, attempt_deadline=None) -> AttemptControl:
        self.attempts += 1
        on_done(False, 1, DropReason.SERVER_FAILURE, 0, False)
        return AttemptControl()


class _RecordingPolicy(RetryPolicy):
    """Fixed number of attempts; keeps every `RetryContext` it was asked about."""

    def __init__(self, max_attempts: int) -> None:
        self.max_attempts = max_attempts
        self.asked: list[RetryContext] = []

    def next_delay(self, context: RetryContext) -> DelayDecision:
        self.asked.append(context)
        return DelayDecision(context.attempt < self.max_attempts, 0)


def _client(target, *, attempts: int = 1, tenant: str | None = None, retry=None) -> ClientRuntime:
    if retry is None:
        retry = (
            NoRetryPolicy()
            if attempts == 1
            else FixedBackoffRetryPolicy(max_attempts=attempts, delay=0)
        )
    return ClientRuntime(
        cfg=ClientConfig(
            retry=retry,
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(1000)
            ),
            tenant=tenant,
        ),
        service=target,
    )


def _run(client: ClientRuntime, requests: int = 1, **start) -> Simulator:
    sim = Simulator(seed=1)
    for _ in range(requests):
        client.start_request(sim, **start)
    sim.run()
    return sim


class TestWithoutAContext(unittest.TestCase):
    def test_a_client_without_a_tenant_passes_none(self):
        leaf = _Recorder(outcomes=(False, False, True))
        _run(_client(leaf, attempts=3))
        self.assertEqual(leaf.calls, [{}, {}, {}])

    def test_a_target_that_only_knows_three_arguments_still_works(self):
        old = _ThreeArguments()
        client = _client(old, attempts=3)
        _run(client)
        self.assertEqual(old.attempts, 3)
        self.assertEqual(len(client.roots), 1)

    def test_no_context_reaches_the_policy_either(self):
        policy = _RecordingPolicy(max_attempts=2)
        _run(_client(_Recorder(outcomes=(False, False)), retry=policy))
        self.assertEqual([context.request for context in policy.asked], [None, None])

    def test_nothing_is_handed_down_a_chain_or_over_a_link(self):
        leaf = _Recorder()
        parent = ServiceNodeRuntime(service=ServiceRuntime(cfg=_cfg("P")).bind())
        link = TransportLink(leaf, TransportDelay(median=ms_to_ns(1)))
        parent.add_downstream(DownstreamEdge(client=_client(link), target=parent))
        _run(_client(parent))
        self.assertEqual(leaf.calls, [{}])


class TestTheClientThatStartsTheRequest(unittest.TestCase):
    def test_every_attempt_carries_the_tenant_and_its_own_number(self):
        leaf = _Recorder(outcomes=(False, False, True))
        _run(_client(leaf, attempts=3, tenant="acme"))
        self.assertEqual(
            leaf.contexts,
            [
                RequestContext(tenant="acme", attempt=1, is_retry=False),
                RequestContext(tenant="acme", attempt=2, is_retry=True),
                RequestContext(tenant="acme", attempt=3, is_retry=True),
            ],
        )

    def test_each_request_starts_at_attempt_one(self):
        # Both first attempts fail, both retries succeed. The two requests
        # interleave (a retry goes through the event heap), and each has exactly
        # one first attempt and one retry.
        leaf = _Recorder(outcomes=(False, False, True, True))
        _run(_client(leaf, attempts=2, tenant="acme"), requests=2)
        self.assertEqual(sorted(context.attempt for context in leaf.contexts), [1, 1, 2, 2])
        self.assertEqual(
            [context.is_retry for context in leaf.contexts],
            [context.attempt > 1 for context in leaf.contexts],
        )

    def test_the_policy_is_asked_with_the_context_of_the_attempt_that_failed(self):
        policy = _RecordingPolicy(max_attempts=3)
        _run(_client(_Recorder(outcomes=(False, False, False)), retry=policy, tenant="acme"))
        self.assertEqual(
            [context.request for context in policy.asked],
            [
                RequestContext(tenant="acme", attempt=1, is_retry=False),
                RequestContext(tenant="acme", attempt=2, is_retry=True),
                RequestContext(tenant="acme", attempt=3, is_retry=True),
            ],
        )
        # Same number in both places.
        self.assertEqual([context.attempt for context in policy.asked], [1, 2, 3])

    def test_a_context_passed_in_is_used_as_given(self):
        leaf = _Recorder()
        _run(_client(leaf, tenant="mine"), ctx=RequestContext(tenant="theirs", is_retry=True))
        self.assertEqual(leaf.contexts, [RequestContext(tenant="theirs", attempt=1, is_retry=True)])


class TestDownTheCallTree(unittest.TestCase):
    def _tree(self, *, top_tenant, edge_tenant=None, leaf_outcomes=()):
        """top client (3 attempts) -> P -> edge client (2 attempts) -> leaf."""
        leaf = _Recorder(outcomes=leaf_outcomes)
        parent = ServiceNodeRuntime(service=ServiceRuntime(cfg=_cfg("P")).bind())
        edge = _client(leaf, attempts=2, tenant=edge_tenant)
        parent.add_downstream(DownstreamEdge(client=edge, target=parent))
        top = _client(parent, attempts=3, tenant=top_tenant)
        return top, leaf

    def test_the_hop_numbers_its_own_attempts_and_the_retry_flag_sticks(self):
        # The leaf always fails: 3 attempts at P, 2 leaf calls under each.
        top, leaf = self._tree(top_tenant="acme", leaf_outcomes=(False,) * 6)
        _run(top)
        self.assertEqual(
            [(c.tenant, c.attempt, c.is_retry) for c in leaf.contexts],
            [
                ("acme", 1, False),  # under P's first attempt: the edge's first try
                ("acme", 2, True),  # ... and its retry
                ("acme", 1, True),  # under P's SECOND attempt: a first try, of a retry
                ("acme", 2, True),
                ("acme", 1, True),
                ("acme", 2, True),
            ],
        )

    def test_the_tenant_is_the_one_of_the_client_that_started_the_request(self):
        top, leaf = self._tree(top_tenant="acme", edge_tenant="edge")
        _run(top)
        self.assertEqual([c.tenant for c in leaf.contexts], ["acme"])

    def test_an_edge_with_a_tenant_starts_a_context_when_none_arrives(self):
        top, leaf = self._tree(top_tenant=None, edge_tenant="edge")
        _run(top)
        self.assertEqual(leaf.contexts, [RequestContext(tenant="edge")])

    def test_a_link_carries_the_context(self):
        leaf = _Recorder()
        link = TransportLink(leaf, TransportDelay(median=ms_to_ns(1)))
        _run(_client(link, tenant="acme"))
        self.assertEqual(leaf.contexts, [RequestContext(tenant="acme")])

    def test_the_allowance_is_one_object_for_the_whole_request(self):
        top, leaf = self._tree(top_tenant=None, leaf_outcomes=(False,) * 6)
        allowance = RetryAllowance(remaining=5)
        _run(top, ctx=RequestContext(tenant="acme", retry_allowance=allowance))
        self.assertEqual(len(leaf.contexts), 6)
        for context in leaf.contexts:
            self.assertIs(context.retry_allowance, allowance)


class TestAContextChangesNothingElse(unittest.TestCase):
    def _run(self, tenant: str | None):
        """A -> B over a link, retries at both hops, a fault on B."""
        from msim.core import TimeInterval, s_to_ns
        from msim.faults import PartialFailure

        sim = Simulator(seed=11)
        b = ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="B",
                    latency_median=ms_to_ns(6),
                    latency_lognorm_sigma=0.5,
                    workers=2,
                    queue_capacity=4,
                    partial_failures=[
                        PartialFailure(TimeInterval(s_to_ns(1), s_to_ns(2)), p_fail=0.5)
                    ],
                )
            ).bind()
        )
        a = ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="A", latency_median=ms_to_ns(2), latency_lognorm_sigma=0.3, workers=4
                )
            ).bind()
        )
        edge = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(2)),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(30)
                ),
            ),
            service=TransportLink(b, TransportDelay(median=ms_to_ns(1), sigma=0.3)),
        )
        a.add_downstream(DownstreamEdge(client=edge, target=b))
        top = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=3, delay=ms_to_ns(5)),
                timeout_controller=StaticTimeoutController(
                    global_timeout=ms_to_ns(200), attempt_timeout=ms_to_ns(60)
                ),
                tenant=tenant,
            ),
            service=a,
        )
        Workload(base_rps=250.0, duration_s=3.0, load_spikes=[], rng_seed=11).drive(
            sim, top.start_request
        )
        sim.run()
        records = [
            (a.service, a.interval.begin, a.interval.end, a.success, a.drop_reason.name)
            for client in (top, edge)
            for root in client.roots
            for a in root.attempts
        ]
        return records, sim._seq, sim.rng().getstate()

    def test_same_attempts_events_and_draws_with_and_without_a_tenant(self):
        plain = self._run(None)
        tenanted = self._run("acme")
        self.assertGreater(len(plain[0]), 1000)
        self.assertIn("SERVER_FAILURE", {record[4] for record in plain[0]})
        self.assertEqual(tenanted, plain)


class TestValidation(unittest.TestCase):
    def test_attempt_numbers_start_at_one(self):
        bad_values: list[Any] = [0, -1, True, 1.0]
        for bad in bad_values:
            with self.subTest(attempt=bad), self.assertRaises(ValueError):
                RequestContext(attempt=bad)

    def test_an_allowance_is_a_non_negative_int(self):
        bad_values: list[Any] = [-1, True, 1.5]
        for bad in bad_values:
            with self.subTest(remaining=bad), self.assertRaises(ValueError):
                RetryAllowance(remaining=bad)
        self.assertEqual(RetryAllowance(remaining=0).remaining, 0)

    def test_a_context_is_frozen(self):
        context = RequestContext(tenant="acme")
        with self.assertRaises(AttributeError):
            context.tenant = "other"  # ty: ignore[invalid-assignment]


if __name__ == "__main__":
    unittest.main()
