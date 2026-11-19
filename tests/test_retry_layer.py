"""`RetryLayer`: a retry loop that another client calls.

Two retry loops on one hop, each with its own policy and timeouts: the caller's
attempt is one whole request of the inner client. The caller sees one answer
when the inner client has finished, the two keep separate books, and a
cancellation of the caller's attempt is truthful about what it could stop.

Service times use `sigma=0`, which a station floors to 1e-6, so instants are
compared in whole milliseconds.
"""

import unittest
from typing import Any

from msim.caller import ClientConfig, ClientRuntime, RetryLayer
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import DropReason, TimeInterval, ms_to_ns, s_to_ns
from msim.faults import PartialFailure
from msim.retry_policies import (
    CBState,
    CountBasedCircuitBreakerPolicy,
    DelayDecision,
    FixedBackoffRetryPolicy,
    NoRetryPolicy,
    RetryContext,
    RetryPolicy,
)
from msim.service import AttemptControl, ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController

MS = ms_to_ns(1)


def _station(
    name: str = "svc",
    *,
    median_ms: float = 10,
    workers: int = 4,
    capacity: int | None = None,
    fail: tuple[float, float] | None = None,
) -> ServiceRuntime:
    failures = (
        []
        if fail is None
        else [PartialFailure(TimeInterval(ms_to_ns(fail[0]), ms_to_ns(fail[1])), 1.0)]
    )
    return ServiceRuntime(
        cfg=ServiceConfig(
            name=name,
            latency_median=ms_to_ns(median_ms),
            latency_lognorm_sigma=0.0,
            workers=workers,
            queue_capacity=capacity,
            partial_failures=failures,
        )
    ).bind()


def _client(
    target,
    *,
    attempts: int = 1,
    delay_ms: float = 0,
    attempt_ms: float | None = 1000,
    global_ms: float | None = None,
    retry: RetryPolicy | None = None,
    tenant: str | None = None,
    mode: str = "attempt",
) -> ClientRuntime:
    if retry is None:
        retry = (
            NoRetryPolicy()
            if attempts == 1
            else FixedBackoffRetryPolicy(max_attempts=attempts, delay=ms_to_ns(delay_ms))
        )
    return ClientRuntime(
        cfg=ClientConfig(
            retry=retry,
            timeout_controller=StaticTimeoutController(
                global_timeout=None if global_ms is None else ms_to_ns(global_ms),
                attempt_timeout=None if attempt_ms is None else ms_to_ns(attempt_ms),
            ),
            tenant=tenant,
            server_deadline=mode,
        ),
        service=target,
    )


def _ms(instant: int) -> int:
    return round(instant / MS)


def _attempts(client: ClientRuntime) -> list[tuple[str, int, int, DropReason]]:
    return [
        (a.service, _ms(a.interval.begin), _ms(a.interval.end), a.drop_reason)
        for root in client.roots
        for a in root.attempts
    ]


def _run(client: ClientRuntime) -> Simulator:
    sim = Simulator(seed=1)
    client.start_request(sim)
    sim.run()
    return sim


class _Told(RetryPolicy):
    """`max_attempts` attempts without backoff; records what it is told."""

    def __init__(self, max_attempts: int) -> None:
        self.max_attempts = max_attempts
        self.told: list[tuple[int, bool, DropReason | None]] = []

    def record_attempt(self, context: RetryContext, success: bool) -> None:
        self.told.append((context.attempt, success, context.drop_reason))

    def next_delay(self, context: RetryContext) -> DelayDecision:
        return DelayDecision(context.attempt < self.max_attempts, 0)


class TestTwoLoopsOnOneHop(unittest.TestCase):
    def test_each_attempt_of_the_caller_is_a_whole_inner_request(self):
        station = _station(fail=(0, 1000))
        inner = _client(station, attempts=3)
        outer = _client(RetryLayer(inner), attempts=2)
        _run(outer)

        self.assertEqual(station.admitted_total, 6)  # 2 x 3
        self.assertEqual(
            _attempts(outer),
            [
                ("svc.retries", 0, 30, DropReason.SERVER_FAILURE),
                ("svc.retries", 30, 60, DropReason.SERVER_FAILURE),
            ],
        )
        self.assertEqual(len(inner.roots), 2)
        self.assertEqual({service for service, *_ in _attempts(inner)}, {"svc"})
        self.assertEqual([len(root.attempts) for root in inner.roots], [3, 3])

    def test_the_caller_sees_one_answer_when_the_inner_loop_has_finished(self):
        """The service fails until 25 ms: the inner client's third attempt works."""
        policy = _Told(max_attempts=2)
        inner = _client(_station(fail=(0, 25)), attempts=3)
        outer = _client(RetryLayer(inner), retry=policy)
        _run(outer)

        self.assertEqual(_attempts(outer), [("svc.retries", 0, 30, DropReason.NONE)])
        self.assertEqual(policy.told, [(1, True, DropReason.NONE)])  # told once, of a success
        self.assertEqual(outer.metrics().summary().succeeded, 1)

    def test_each_loop_has_its_own_timeout(self):
        """Inner: 10 ms per attempt, three attempts. Caller: 100 ms per call."""
        station = _station(median_ms=30)
        inner = _client(station, attempts=3, attempt_ms=10)
        outer = _client(RetryLayer(inner), attempts=2, attempt_ms=100)
        _run(outer)

        self.assertEqual(
            [(b, e, r) for _, b, e, r in _attempts(inner)][:3],
            [
                (0, 10, DropReason.DEADLINE),
                (10, 20, DropReason.DEADLINE),
                (20, 30, DropReason.DEADLINE),
            ],
        )
        # The caller's call fails at 30 ms, with the inner request's reason, well
        # inside its own 100 ms.
        self.assertEqual(_attempts(outer)[0], ("svc.retries", 0, 30, DropReason.DEADLINE))
        self.assertEqual(len(_attempts(outer)), 2)

    def test_the_callers_deadline_bounds_the_inner_loop(self):
        """A 25 ms call around a loop that would take 30 ms."""
        station = _station(median_ms=30)
        inner = _client(station, attempts=5, attempt_ms=10)
        outer = _client(RetryLayer(inner), attempt_ms=25)
        _run(outer)

        self.assertEqual([(b, e) for _, b, e, _ in _attempts(inner)], [(0, 10), (10, 20), (20, 25)])
        self.assertEqual(inner.roots[0].global_deadline, 25 * MS)
        self.assertEqual(_attempts(outer), [("svc.retries", 0, 25, DropReason.DEADLINE)])

    def test_what_the_last_inner_attempt_observed_is_what_is_reported(self):
        station = _station(median_ms=10, workers=1)
        inner = _client(station, attempts=2)
        outer = _client(RetryLayer(inner))
        sim = Simulator(seed=1)
        outer.start_request(sim)
        outer.start_request(sim)  # waits 10 ms behind the first
        sim.run()

        first, second = (root.attempts[0] for root in outer.roots)
        self.assertEqual(_ms(first.service_time_ns or 0), 10)
        self.assertEqual(first.queue_size_at_end, 1)  # the second was waiting when it ended
        self.assertEqual((_ms(second.interval.end), second.queue_size_at_end), (20, 0))
        self.assertFalse(first.service_time_censored)

    def test_an_answer_from_inside_the_call(self):
        """No worker, no queue, no inner retry: the layer answers synchronously."""
        station = _station(median_ms=50, workers=1, capacity=0)
        blocker = _client(station)
        layer = RetryLayer(_client(station))
        sim = Simulator(seed=1)
        blocker.start_request(sim)
        seen: list[tuple[bool, DropReason]] = []
        control = layer.submit_attempt(sim, lambda ok, _t, drop, _q, _c: seen.append((ok, drop)))
        self.assertEqual(seen, [(False, DropReason.QUEUE_FULL)])
        self.assertFalse(control.cancel())  # already over
        sim.run()
        self.assertEqual(len(seen), 1)

    def test_the_layer_has_a_name_of_its_own(self):
        inner = _client(_station("db"))
        self.assertEqual(RetryLayer(inner).cfg.name, "db.retries")
        self.assertEqual(RetryLayer(inner, name="db-sidecar").cfg.name, "db-sidecar")
        outer = _client(RetryLayer(inner, name="db-sidecar"))
        _run(outer)
        self.assertEqual([service for service, *_ in _attempts(outer)], ["db-sidecar"])
        self.assertEqual([service for service, *_ in _attempts(inner)], ["db"])

    def test_the_request_context_passes_through(self):
        seen: list[Any] = []

        class Recorder:
            cfg = _station().cfg

            def submit_attempt(self, sim, on_done, attempt_deadline=None, *, ctx=None):
                seen.append(ctx)
                on_done(False, 1, DropReason.SERVER_FAILURE, 0, False)
                return AttemptControl()

        inner = _client(Recorder(), attempts=2)
        outer = _client(RetryLayer(inner), attempts=2, tenant="acme")
        _run(outer)
        self.assertEqual(
            [(c.tenant, c.attempt, c.is_retry) for c in seen],
            [("acme", 1, False), ("acme", 2, True), ("acme", 1, True), ("acme", 2, True)],
        )

    def test_a_caller_that_leaves_does_not_stop_the_inner_loop(self):
        """The caller keeps its 15 ms timeout to itself: the inner client goes on retrying."""
        station = _station(fail=(0, 1000))
        inner = _client(station, attempts=3)
        outer = _client(RetryLayer(inner), attempt_ms=15, mode="global")
        sim = _run(outer)
        self.assertEqual(_attempts(outer), [("svc.retries", 0, 15, DropReason.DEADLINE)])
        self.assertEqual(station.admitted_total, 3)
        self.assertEqual(_ms(sim.timestep), 30)

    def test_in_a_chain(self):
        """A node calls a dependency that retries its own handler."""
        dependency = _station("D", fail=(0, 20))  # its first attempt, 5 to 15 ms, fails
        to_dependency = _client(RetryLayer(_client(dependency, attempts=3)))
        parent = ServiceNodeRuntime(service=_station("P", median_ms=5))
        parent.add_downstream(DownstreamEdge(client=to_dependency, target=parent))
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(to_dependency), [("D.retries", 5, 25, DropReason.NONE)])
        self.assertEqual(_attempts(top), [("P", 0, 25, DropReason.NONE)])


class TestCancellation(unittest.TestCase):
    def _layer_attempt(self, inner: ClientRuntime):
        sim = Simulator(seed=1)
        seen: list[tuple[int, bool, DropReason]] = []
        control = RetryLayer(inner).submit_attempt(
            sim, lambda ok, _t, drop, _q, _c: seen.append((_ms(sim.timestep), ok, drop))
        )
        return sim, control, seen

    def test_during_backoff(self):
        station = _station(fail=(0, 1000))
        inner = _client(station, attempts=3, delay_ms=50)
        sim, control, seen = self._layer_attempt(inner)
        answers: list[bool] = []
        sim.schedule(30 * MS, lambda: answers.append(control.cancel()))
        sim.run()

        self.assertEqual(answers, [True])
        self.assertEqual(seen, [(30, False, DropReason.CANCELLED)])  # one answer, at the cancel
        self.assertEqual(station.admitted_total, 1)  # the pending retry never ran
        self.assertTrue(inner.roots[0].cancelled)

    def test_in_service(self):
        station = _station(median_ms=20)
        inner = _client(station, attempts=3)
        sim, control, seen = self._layer_attempt(inner)
        answers: list[bool] = []
        sim.schedule(5 * MS, lambda: answers.append(control.cancel()))
        sim.run()

        self.assertEqual(answers, [True])
        self.assertEqual(seen, [(5, False, DropReason.CANCELLED)])
        self.assertEqual(station.in_flight, 0)  # the worker came back
        self.assertEqual(_ms(sim.timestep), 5)  # and nothing ran on

    def test_after_the_attempt_has_committed(self):
        """A target that cannot be cancelled: the cancel is refused, the inner
        client stops retrying, and the genuine outcome is what the caller gets."""

        class Committed:
            cfg = _station().cfg
            calls = 0

            def submit_attempt(self, sim, on_done, attempt_deadline=None, *, ctx=None):
                Committed.calls += 1
                sim.schedule(
                    sim.timestep + 10 * MS,
                    lambda: on_done(False, 10 * MS, DropReason.SERVER_FAILURE, 0, False),
                )
                return AttemptControl()  # nothing bound: not cancellable

        inner = _client(Committed(), attempts=3)
        sim, control, seen = self._layer_attempt(inner)
        answers: list[bool] = []
        sim.schedule(5 * MS, lambda: answers.append(control.cancel()))
        sim.run()

        self.assertEqual(answers, [False])
        self.assertEqual(seen, [(10, False, DropReason.SERVER_FAILURE)])  # itself, not CANCELLED
        self.assertEqual(Committed.calls, 1)  # no further attempt
        self.assertFalse(control.cancel())

    def test_once_it_is_over_there_is_nothing_to_cancel(self):
        sim, control, seen = self._layer_attempt(_client(_station()))
        sim.run()
        self.assertEqual(seen, [(10, True, DropReason.NONE)])
        self.assertFalse(control.cancel())
        self.assertEqual(len(seen), 1)


def _node(name: str, median_ms: float, workers: int = 1) -> ServiceNodeRuntime:
    return ServiceNodeRuntime(
        service=ServiceRuntime(
            cfg=ServiceConfig(
                name=name,
                latency_median=ms_to_ns(median_ms),
                latency_lognorm_sigma=1e-6,
                workers=workers,
            )
        ).bind()
    )


class TestACommittedOutcomeReachesTheCaller(unittest.TestCase):
    """The diamond of `test_cancellation_client`, with a layer on the A -> B edge.

    A -> {B, E}; B -> SHARED; E -> SHARED, one SHARED worker. At the deadline B
    commits to a DEADLINE failure and, while it is finalizing, the cancellation
    of the A -> B call comes back around through E. Through the layer it must
    still be refused, and B's genuine failure must reach the caller's policy.
    """

    def _diamond(self, outer_retry: RetryPolicy) -> tuple[ClientRuntime, ClientRuntime, Any]:
        sim = Simulator(seed=1)
        shared, b, e, a = _node("SHARED", 20), _node("B", 1), _node("E", 3), _node("A", 1)
        a.propagate_cancellation = True
        b.propagate_cancellation = True
        b.add_downstream(DownstreamEdge(client=_client(shared, attempt_ms=100), target=shared))
        e.add_downstream(DownstreamEdge(client=_client(shared, attempt_ms=100), target=shared))
        inner = _client(b, attempts=2, delay_ms=10, attempt_ms=None)
        outer = _client(RetryLayer(inner), retry=outer_retry, attempt_ms=10)
        a.add_downstream(DownstreamEdge(client=outer, target=b))
        a.add_downstream(DownstreamEdge(client=_client(e, attempt_ms=10), target=e))
        _client(a, attempt_ms=200).start_request(sim)
        sim.run()
        return outer, inner, b

    def test_the_genuine_reason_not_cancelled(self):
        outer, inner, b = self._diamond(FixedBackoffRetryPolicy(max_attempts=2, delay=10 * MS))
        (root,) = outer.roots
        # One call, reported once, with the failure B really had ...
        self.assertEqual([a.drop_reason for a in root.attempts], [DropReason.DEADLINE])
        # ... on a request that is booked as abandoned, and not retried by either loop.
        self.assertTrue(root.cancelled)
        self.assertEqual(outer.attempts_total, 1)
        self.assertEqual(inner.attempts_total, 1)
        self.assertEqual(b.service.admitted_total, 1)

    def test_it_reaches_the_callers_breaker(self):
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(1, 1),
            success_threshold_ratio=(1, 1),
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=2, delay=10 * MS),
            half_open_delay=s_to_ns(10),
        )
        self._diamond(breaker)
        self.assertEqual(breaker.get_state(), CBState.OPEN)


if __name__ == "__main__":
    unittest.main()
