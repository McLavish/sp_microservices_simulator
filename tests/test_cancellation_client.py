"""A root an upstream abandoned must stop issuing attempts.

`propagate_cancellation=True` exists to stop work nobody is waiting for. The
cancel can arrive while the abandoned node has already *committed* to an
outcome but has not finalized yet (`chain._FanOutAttempt._fail_from` runs
`_commit()` -> `_abandon_siblings()` -> `_finalize()`, and a re-entrant cancel
lands in that window). `AttemptControl.cancel()` then truthfully returns False,
which is what keeps a genuine failure from being rewritten as CANCELLED --
but the root is still abandoned, and must not start a fresh attempt afterwards.
"""

import unittest

from caller import ClientConfig, ClientRuntime
from chain import DownstreamEdge, ServiceNodeRuntime
from core import DropReason, TimeInterval, ms_to_ns, s_to_ns
from faults import PartialFailure
from retry_policies import (
    CBState,
    CountBasedCircuitBreakerPolicy,
    FixedBackoffRetryPolicy,
    NoRetryPolicy,
    RetryPolicy,
)
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController


def _node(name: str, median_ms: float, workers: int = 1, p_fail: float = 0.0):
    failures = [PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=p_fail)] if p_fail else []
    return ServiceNodeRuntime(
        service=ServiceRuntime(
            cfg=ServiceConfig(
                name=name,
                latency_median=ms_to_ns(median_ms),
                latency_lognorm_sigma=1e-6,
                workers=workers,
                partial_failures=failures,
            )
        ).bind()
    )


def _client(target: ServiceNodeRuntime, retry: RetryPolicy, attempt_timeout_ms: float):
    return ClientRuntime(
        cfg=ClientConfig(
            retry=retry,
            timeout_controller=StaticTimeoutController(None, ms_to_ns(attempt_timeout_ms)),
        ),
        service=target,
    )


class TestAbandonedRootStopsRetrying(unittest.TestCase):
    def _diamond(self, edge_retry: RetryPolicy):
        """A -> {B, E}; B -> {SHARED, C1}; E -> SHARED, with one SHARED worker.

        Timeline (service times are near-deterministic):

        * t=1ms  A fans out to B and E.
        * t=2ms  B fans out: its SHARED call takes the only worker (busy until
          22ms), its C1 call will fail at 7ms.
        * t=4ms  E fans out: its SHARED call queues behind B's, with a 2ms
          attempt timeout, so it is already expired from 6ms on.
        * t=7ms  C1 fails. B short-circuits: `_commit()` (its teardown is now
          revoked) then cancels its own SHARED call, which frees the worker and
          synchronously dequeues E's expired call -> DEADLINE. That failure
          unwinds through E up to A, which short-circuits too and cancels the
          A->B edge root -- re-entrantly, while B is still between `_commit()`
          and `_finalize()`.
        """
        sim = Simulator(seed=1)
        shared = _node("SHARED", 20)
        c1 = _node("C1", 5, p_fail=1.0)
        b = _node("B", 1)
        e = _node("E", 3)
        a = _node("A", 1)
        a.propagate_cancellation = True
        b.propagate_cancellation = True

        b.add_downstream(
            DownstreamEdge(client=_client(shared, NoRetryPolicy(), 100), target=shared)
        )
        b.add_downstream(DownstreamEdge(client=_client(c1, NoRetryPolicy(), 100), target=c1))
        e.add_downstream(DownstreamEdge(client=_client(shared, NoRetryPolicy(), 2), target=shared))

        edge_to_b = _client(b, edge_retry, 100)
        a.add_downstream(DownstreamEdge(client=edge_to_b, target=b))
        a.add_downstream(DownstreamEdge(client=_client(e, NoRetryPolicy(), 100), target=e))

        top = _client(a, NoRetryPolicy(), 200)
        top.start_request(sim)
        sim.run()
        return edge_to_b, b

    def test_abandoned_edge_root_issues_no_further_attempt(self):
        edge_to_b, b = self._diamond(FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(10)))
        summary = edge_to_b.metrics().summary()
        root = edge_to_b.roots[-1]

        # One attempt only: the retry budget was still there, but the parent had
        # already settled, so issuing it would be work nobody is waiting for.
        self.assertEqual(len(root.attempts), 1)
        self.assertEqual(edge_to_b.attempts_total, 1)
        self.assertEqual(b.service.admitted_total, 1)
        self.assertLess(root.completed_at, ms_to_ns(15))

        # The committed attempt keeps its OWN, genuine drop reason...
        self.assertEqual(root.attempts[-1].drop_reason, DropReason.SERVER_FAILURE)
        # ...while the ROOT is booked as abandoned, not as a server failure.
        self.assertTrue(root.cancelled)
        self.assertEqual(summary.dropped_cancelled, 1)
        self.assertEqual(summary.dropped_server_failure, 0)

    def test_committed_failure_still_reaches_the_circuit_breaker(self):
        """The attempt really did fail at the service, so it is evidence about
        it — unlike a truly cancelled attempt, whose work was cut short."""
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(1, 1),
            success_threshold_ratio=(1, 1),
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(10)),
            half_open_delay=s_to_ns(10),
        )
        edge_to_b, _ = self._diamond(breaker)
        self.assertEqual(breaker.get_state(), CBState.OPEN)
        self.assertEqual(edge_to_b.metrics().summary().dropped_cancelled, 1)


if __name__ == "__main__":
    unittest.main()
