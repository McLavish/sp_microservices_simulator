"""What the server is told about the client's deadline (`ClientConfig.server_deadline`).

By default the target is told each attempt's deadline and enforces it: it
reports a waiting attempt at that instant and cuts one in service. With
`"global"` it is told only the request's global deadline, with `"none"`
nothing, and the client enforces its own deadline with a timer: it walks away,
and the attempt stays at the server as an ORPHAN that keeps its worker or queue
slot until the server is done with it.

The race between the timer and the real answer is decided once per attempt:
whichever comes first is recorded and fed to the policies, the other is ignored.
A tie goes to the deadline (the half-open rule).
"""

import unittest

from msim.caller import SERVER_DEADLINE_MODES, ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import DropReason, TimeInterval, ms_to_ns
from msim.faults import PartialFailure
from msim.retry_policies import (
    DelayDecision,
    NoRetryPolicy,
    RetryContext,
    RetryPolicy,
)
from msim.service import AttemptControl, ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController

MS = ms_to_ns(1)


class _Exact:
    """A station whose attempts take EXACTLY `service_ns`, deadline or not.

    A plain `ServiceRuntime` floors its lognormal sigma, so a "10 ms" service
    lasts 10 ms give or take a few nanoseconds: too loose to place an answer one
    nanosecond before or after a deadline. This runs a custom behavior on a real
    station, so workers, the queue and `in_flight` are the real ones. It records
    the deadline each attempt was TOLD and when each entered service.
    """

    def __init__(self, service_ns: int, *, workers: int = 1, succeed: bool = True) -> None:
        self.service = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=1, latency_lognorm_sigma=0.0, workers=workers
            )
        ).bind()
        self.cfg = self.service.cfg
        self.service_ns = service_ns
        self.succeed = succeed
        self.told: list[int | None] = []
        self.started: list[int] = []
        self.finished: list[int] = []

    def submit_attempt(self, sim, on_done, attempt_deadline=None, *, ctx=None) -> AttemptControl:
        self.told.append(attempt_deadline)

        def behavior(service, sim_, deadline, finalize, begin_at, release_worker, register_cancel):
            self.started.append(begin_at)

            def finish() -> None:
                register_cancel(None)
                self.finished.append(sim_.timestep)
                reason = DropReason.NONE if self.succeed else DropReason.SERVER_FAILURE
                finalize(self.succeed, sim_.timestep - begin_at, reason)

            event = sim_.schedule(begin_at + self.service_ns, finish)

            def cancel() -> None:
                event.cancelled = True
                finalize(False, sim_.timestep - begin_at, DropReason.CANCELLED)

            register_cancel(cancel)

        return self.service.submit_attempt(
            sim, on_done, attempt_deadline=attempt_deadline, behavior=behavior
        )


class _Counting(RetryPolicy):
    """`max_attempts` attempts, `delay` apart; counts what it is told and asked."""

    def __init__(self, max_attempts: int, delay: int = 0) -> None:
        self.max_attempts = max_attempts
        self.delay = delay
        self.told: list[tuple[int, bool, DropReason | None]] = []

    def record_attempt(self, context: RetryContext, success: bool) -> None:
        self.told.append((context.attempt, success, context.drop_reason))

    def next_delay(self, context: RetryContext) -> DelayDecision:
        return DelayDecision(context.attempt < self.max_attempts, self.delay)


def _client(
    target,
    *,
    mode: str,
    attempt_ms: float | None = 10,
    global_ms: float | None = None,
    retry: RetryPolicy | None = None,
) -> ClientRuntime:
    return ClientRuntime(
        cfg=ClientConfig(
            retry=retry or NoRetryPolicy(),
            timeout_controller=StaticTimeoutController(
                global_timeout=None if global_ms is None else ms_to_ns(global_ms),
                attempt_timeout=None if attempt_ms is None else ms_to_ns(attempt_ms),
            ),
            server_deadline=mode,
        ),
        service=target,
    )


def _attempts(client: ClientRuntime) -> list[tuple[bool, DropReason, int]]:
    (root,) = client.roots
    return [(a.success, a.drop_reason, a.interval.end) for a in root.attempts]


class TestWhatTheServerIsTold(unittest.TestCase):
    def _told(self, mode: str, **timeouts) -> list[int | None]:
        target = _Exact(service_ns=MS)
        sim = Simulator(seed=1)
        _client(target, mode=mode, **timeouts).start_request(sim)
        sim.run()
        return target.told

    def test_attempt_mode_discloses_the_attempt_deadline(self):
        self.assertEqual(self._told("attempt"), [10 * MS])
        self.assertEqual(self._told("attempt", attempt_ms=10, global_ms=4), [4 * MS])

    def test_global_mode_discloses_only_the_global_deadline(self):
        self.assertEqual(self._told("global"), [None])
        self.assertEqual(self._told("global", attempt_ms=10, global_ms=30), [30 * MS])

    def test_none_mode_discloses_nothing(self):
        self.assertEqual(self._told("none"), [None])
        self.assertEqual(self._told("none", attempt_ms=10, global_ms=30), [None])

    def test_the_default_is_attempt(self):
        cfg = ClientConfig(retry=NoRetryPolicy(), timeout_controller=StaticTimeoutController())
        self.assertEqual(cfg.server_deadline, "attempt")
        self.assertEqual(SERVER_DEADLINE_MODES, ("attempt", "global", "none"))

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            _client(_Exact(MS), mode="request")
        self.assertIn("attempt, global, none", str(raised.exception))


class TestTheRaceAtTheDeadline(unittest.TestCase):
    """The answer lands 1 ns before, at, and 1 ns after a 10 ms client deadline."""

    def _run(self, offset_ns: int):
        target = _Exact(service_ns=10 * MS + offset_ns)
        policy = _Counting(max_attempts=2)
        client = _client(target, mode="global", retry=policy)
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()
        return target, policy, client, sim

    def test_an_answer_just_in_time_is_the_result(self):
        target, policy, client, sim = self._run(-1)
        self.assertEqual(_attempts(client), [(True, DropReason.NONE, 10 * MS - 1)])
        self.assertEqual(policy.told, [(1, True, DropReason.NONE)])
        # The disarmed timer left nothing behind: the run ends with the answer.
        self.assertEqual(sim.timestep, 10 * MS - 1)
        self.assertEqual(target.service.in_flight, 0)

    def test_a_tie_goes_to_the_deadline(self):
        target, policy, client, sim = self._run(0)
        # Both attempts take exactly as long as the client waits: two timeouts.
        self.assertEqual(
            _attempts(client),
            [(False, DropReason.DEADLINE, 10 * MS), (False, DropReason.DEADLINE, 20 * MS)],
        )
        # The orphan freed its worker at the same instant, so the retry started at once.
        self.assertEqual(target.started, [0, 10 * MS])
        self.assertEqual(target.finished, [10 * MS, 20 * MS])

    def test_a_late_answer_is_not_a_result(self):
        target, policy, client, sim = self._run(1)
        self.assertEqual(
            _attempts(client),
            [(False, DropReason.DEADLINE, 10 * MS), (False, DropReason.DEADLINE, 20 * MS)],
        )
        # One record and one report per attempt: the two late successes went nowhere.
        self.assertEqual(
            policy.told, [(1, False, DropReason.DEADLINE), (2, False, DropReason.DEADLINE)]
        )
        self.assertEqual(client.attempts_total, 2)
        # The single worker was held until the orphan finished, 1 ns after the
        # client left: the retry, submitted at 10 ms, waited for it.
        self.assertEqual(target.started, [0, 10 * MS + 1])
        self.assertEqual(target.finished, [10 * MS + 1, 20 * MS + 2])
        self.assertEqual(target.service.in_flight, 0)

    def test_the_timed_out_attempt_carries_no_queue_size_and_no_occupancy(self):
        _, _, client, _ = self._run(1)
        for attempt in client.roots[0].attempts:
            self.assertIsNone(attempt.queue_size_at_end)
            self.assertIsNone(attempt.service_time_ns)
            self.assertFalse(attempt.service_time_censored)
            self.assertEqual(attempt.deadline, attempt.interval.begin + 10 * MS)


class TestOrphans(unittest.TestCase):
    def _run(self, mode: str, *, global_ms: float | None = None, service_ms: float = 25):
        """One worker, a 10 ms client timeout, three attempts without backoff."""
        service = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(service_ms),
                latency_lognorm_sigma=0.0,
                workers=1,
            )
        ).bind()
        client = _client(service, mode=mode, global_ms=global_ms, retry=_Counting(max_attempts=3))
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()
        return service, client, sim

    def test_told_its_deadline_the_server_stops_working_at_it(self):
        service, client, sim = self._run("attempt")
        self.assertEqual([end for _, _, end in _attempts(client)], [10 * MS, 20 * MS, 30 * MS])
        self.assertEqual(sim.timestep, 30 * MS)  # nothing outlives the request
        self.assertEqual(service.admitted_total, 3)

    def test_not_told_it_works_on_for_a_client_that_has_left(self):
        service, client, sim = self._run("global")
        # The client sees the same three timeouts at the same instants ...
        self.assertEqual(
            _attempts(client),
            [
                (False, DropReason.DEADLINE, 10 * MS),
                (False, DropReason.DEADLINE, 20 * MS),
                (False, DropReason.DEADLINE, 30 * MS),
            ],
        )
        self.assertEqual(client.roots[0].completed_at, 30 * MS)
        # ... while the server serves all three attempts to the end, one after
        # the other: 75 ms of work for a request that failed at 30 ms.
        self.assertAlmostEqual(sim.timestep / MS, 75, delta=0.01)
        self.assertEqual(service.in_flight, 0)
        self.assertEqual(service.queue_len(), 0)

    def test_the_global_deadline_still_reaches_the_server_in_global_mode(self):
        """Global 30 ms, attempt 10 ms, a 100 ms service."""
        service, client, sim = self._run("global", global_ms=30, service_ms=100)
        self.assertEqual([end for _, _, end in _attempts(client)], [10 * MS, 20 * MS, 30 * MS])
        # The first attempt is cut at 30 ms by the server, which was told that
        # much; the two behind it expire in its queue at the same instant.
        self.assertEqual(sim.timestep, 30 * MS)
        self.assertEqual(service.in_flight, 0)

    def test_none_mode_leaves_even_the_global_deadline_to_the_client(self):
        service, client, sim = self._run("none", global_ms=30, service_ms=100)
        self.assertEqual([end for _, _, end in _attempts(client)], [10 * MS, 20 * MS, 30 * MS])
        self.assertEqual(client.roots[0].completed_at, 30 * MS)
        # Three attempts of 100 ms each, served to the end.
        self.assertAlmostEqual(sim.timestep / MS, 300, delta=0.01)

    def test_equal_deadlines_need_no_timer(self):
        """Global mode with no attempt timeout: the server is told the client's
        one deadline, so the run is event for event the default one."""

        def events(mode: str) -> tuple[int, list]:
            service, client, sim = self._run_global_only(mode)
            return sim._seq, _attempts(client)

        self.assertEqual(events("global"), events("attempt"))

    def _run_global_only(self, mode: str):
        service = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(25), latency_lognorm_sigma=0.0, workers=1
            )
        ).bind()
        client = _client(
            service, mode=mode, attempt_ms=None, global_ms=40, retry=_Counting(max_attempts=3)
        )
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()
        return service, client, sim

    def test_an_orphan_in_a_chain_keeps_calling_downstream(self):
        def run(mode: str):
            leaf = ServiceNodeRuntime(
                service=ServiceRuntime(
                    cfg=ServiceConfig(
                        name="L", latency_median=ms_to_ns(50), latency_lognorm_sigma=0.0, workers=4
                    )
                ).bind()
            )
            parent = ServiceNodeRuntime(
                service=ServiceRuntime(
                    cfg=ServiceConfig(
                        name="P", latency_median=ms_to_ns(1), latency_lognorm_sigma=0.0, workers=4
                    )
                ).bind()
            )
            edge = _client(leaf, mode="attempt", attempt_ms=1000)
            parent.add_downstream(DownstreamEdge(client=edge, target=leaf))
            top = _client(parent, mode=mode)
            sim = Simulator(seed=1)
            top.start_request(sim)
            sim.run()
            return top, edge

        top, edge = run("attempt")
        self.assertEqual(_attempts(top), [(False, DropReason.DEADLINE, 10 * MS)])
        # The parent passed its deadline down: the leaf call died with it.
        self.assertEqual(_attempts(edge), [(False, DropReason.DEADLINE, 10 * MS)])

        top, edge = run("global")
        self.assertEqual(_attempts(top), [(False, DropReason.DEADLINE, 10 * MS)])
        # The parent was told nothing, so its call to the leaf ran to the end.
        ((ok, reason, end),) = _attempts(edge)
        self.assertTrue(ok)
        self.assertAlmostEqual(end / MS, 51, delta=0.01)


class TestCancellation(unittest.TestCase):
    def _fanout(self, *, slow_retry: RetryPolicy, fast_fails_at_ms: float):
        """P calls a fast edge that fails and a slow edge with a 20 ms client timeout."""
        slow_target = _Exact(service_ns=50 * MS, workers=2)
        fast_service = ServiceRuntime(
            cfg=ServiceConfig(
                name="F",
                latency_median=ms_to_ns(fast_fails_at_ms - 1),
                latency_lognorm_sigma=0.0,
                workers=2,
                partial_failures=[PartialFailure(TimeInterval(0, ms_to_ns(1000)), p_fail=1.0)],
            )
        ).bind()
        parent = ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="P", latency_median=ms_to_ns(1), latency_lognorm_sigma=0.0, workers=2
                )
            ).bind(),
            propagate_cancellation=True,
        )
        fast = _client(fast_service, mode="attempt", attempt_ms=1000)
        slow = _client(slow_target, mode="global", attempt_ms=20, retry=slow_retry)
        parent.add_downstream(DownstreamEdge(client=fast, target=parent))
        parent.add_downstream(DownstreamEdge(client=slow, target=parent))
        top = _client(parent, mode="attempt", attempt_ms=1000)
        return top, slow, slow_target

    def test_a_cancelled_attempt_disarms_its_timer(self):
        """The sibling fails at 5 ms, long before the slow edge's 21 ms timer."""
        top, slow, slow_target = self._fanout(slow_retry=NoRetryPolicy(), fast_fails_at_ms=5)
        sim = Simulator(seed=1)
        top.start_request(sim)
        sim.run()

        (root,) = slow.roots
        self.assertTrue(root.cancelled)
        self.assertEqual([a.drop_reason for a in root.attempts], [DropReason.CANCELLED])
        self.assertAlmostEqual(root.completed_at / MS, 5, delta=0.01)
        self.assertEqual(slow_target.service.in_flight, 0)  # the worker came back
        self.assertEqual(slow_target.finished, [])

    def test_an_orphan_is_out_of_reach_of_a_later_cancellation(self):
        """The slow edge times out at 21 ms and backs off; the sibling fails at
        30 ms and the request is cancelled. The orphan works on to 51 ms."""
        top, slow, slow_target = self._fanout(
            slow_retry=_Counting(max_attempts=3, delay=ms_to_ns(100)), fast_fails_at_ms=30
        )
        sim = Simulator(seed=1)
        top.start_request(sim)
        seen: list[int] = []
        sim.schedule(40 * MS, lambda: seen.append(slow_target.service.in_flight))
        sim.run()

        (root,) = slow.roots
        self.assertTrue(root.cancelled)
        self.assertEqual([a.drop_reason for a in root.attempts], [DropReason.DEADLINE])
        self.assertAlmostEqual(root.completed_at / MS, 30, delta=0.01)
        self.assertEqual(seen, [1])  # still in service after the cancellation
        self.assertEqual(len(slow_target.finished), 1)
        self.assertAlmostEqual(slow_target.finished[0] / MS, 51, delta=0.01)
        self.assertEqual(slow_target.service.in_flight, 0)


if __name__ == "__main__":
    unittest.main()
