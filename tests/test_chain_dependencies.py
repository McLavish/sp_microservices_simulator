"""Three ways a node can treat its downstream calls differently.

By default a chain node does its local work, then makes its calls, and fails as
soon as one of them fails. Each of these is a per-node (or per-edge) choice:

* `ServiceNodeRuntime.local_work="last"`: the calls first, the local work after;
* `DownstreamEdge.optional=True`: a call the node waits for and can do without;
* `ServiceNodeRuntime.wave_join="wait_all"`: a wave runs to its end before a
  failure in it fails the node.

Service times use `sigma=0`, which a station floors to 1e-6: a "10 ms" service
lasts 10 ms give or take a few nanoseconds, so instants are compared in whole
milliseconds.
"""

import unittest
from typing import Any

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import LOCAL_WORK_MODES, WAVE_JOIN_MODES, DownstreamEdge, ServiceNodeRuntime
from msim.core import DropReason, TimeInterval, ms_to_ns
from msim.faults import LatencyInjection, PartialFailure
from msim.retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController

MS = ms_to_ns(1)


def _service(
    name: str,
    median_ms: float,
    *,
    workers: int = 4,
    capacity: int | None = None,
    fail: tuple[float, float] | None = None,
    slow: tuple[float, float, float] | None = None,
) -> ServiceRuntime:
    failures = (
        []
        if fail is None
        else [PartialFailure(TimeInterval(ms_to_ns(fail[0]), ms_to_ns(fail[1])), 1.0)]
    )
    injections = (
        []
        if slow is None
        else [
            LatencyInjection(
                TimeInterval(ms_to_ns(slow[0]), ms_to_ns(slow[1])), add_latency=ms_to_ns(slow[2])
            )
        ]
    )
    return ServiceRuntime(
        cfg=ServiceConfig(
            name=name,
            latency_median=ms_to_ns(median_ms),
            latency_lognorm_sigma=0.0,
            workers=workers,
            queue_capacity=capacity,
            partial_failures=failures,
            latency_injections=injections,
        )
    ).bind()


def _node(
    name: str, median_ms: float, *, node: dict[str, Any] | None = None, **service
) -> ServiceNodeRuntime:
    return ServiceNodeRuntime(service=_service(name, median_ms, **service), **(node or {}))


def _client(target, *, attempt_ms: float = 1000, attempts: int = 1) -> ClientRuntime:
    retry = (
        NoRetryPolicy()
        if attempts == 1
        else FixedBackoffRetryPolicy(max_attempts=attempts, delay=0)
    )
    return ClientRuntime(
        cfg=ClientConfig(
            retry=retry,
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(attempt_ms)
            ),
        ),
        service=target,
    )


def _call(parent: ServiceNodeRuntime, child, *, wave: int = 0, optional: bool = False, **client):
    edge = _client(child, **client)
    target = child if isinstance(child, ServiceNodeRuntime) else parent
    parent.add_downstream(DownstreamEdge(client=edge, target=target, wave=wave, optional=optional))
    return edge


def _run(top: ClientRuntime, requests: int = 1) -> Simulator:
    sim = Simulator(seed=1)
    for _ in range(requests):
        top.start_request(sim)
    sim.run()
    return sim


def _ms(instant: int) -> int:
    return round(instant / MS)


def _attempts(client: ClientRuntime) -> list[tuple[int, int, DropReason]]:
    """`(begin ms, end ms, reason)` of every attempt of the client, in completion order."""
    return [
        (_ms(a.interval.begin), _ms(a.interval.end), a.drop_reason)
        for root in client.roots
        for a in root.attempts
    ]


HOLD = {"hold_worker_during_fanout": True}
LAST = {**HOLD, "local_work": "last"}


class TestLocalWorkLast(unittest.TestCase):
    def test_the_calls_come_first_and_the_local_work_after(self):
        for mode, call_window in (("first", (10, 30)), ("last", (0, 20))):
            with self.subTest(local_work=mode):
                parent = _node("P", 10, node={**HOLD, "local_work": mode})
                edge = _call(parent, _node("D", 20))
                top = _client(parent)
                _run(top)
                self.assertEqual(_attempts(edge), [(*call_window, DropReason.NONE)])
                self.assertEqual(_attempts(top), [(0, 30, DropReason.NONE)])  # same total

    def test_the_worker_is_held_from_the_start_to_the_end(self):
        parent = _node("P", 10, workers=1, node=LAST)
        _call(parent, _node("D", 20))
        top = _client(parent)
        sim = Simulator(seed=1)
        top.start_request(sim)
        top.start_request(sim)
        seen: list[tuple[int, int]] = []
        for at in (5, 25, 35):
            sim.schedule(
                at * MS, lambda: seen.append((parent.service.in_flight, parent.service.queue_len()))
            )
        sim.run()
        # One worker: the second request waits for the whole first one, calls included.
        self.assertEqual(seen, [(1, 1), (1, 1), (1, 0)])
        self.assertEqual([(b, e) for b, e, _ in _attempts(top)], [(0, 30), (0, 60)])
        (first, second) = (root.attempts[0] for root in top.roots)
        # Occupancy is the whole subtree, as for any node that holds its worker.
        assert first.service_time_ns is not None and second.service_time_ns is not None
        self.assertEqual(_ms(first.service_time_ns), 30)
        self.assertEqual(_ms(second.service_time_ns), 30)
        self.assertEqual(parent.service.in_flight, 0)
        self.assertEqual(parent.service.queue_len(), 0)

    def test_the_service_time_is_drawn_when_the_calls_are_done(self):
        """A latency fault active from 15 to 25 ms adds 100 ms to work drawn in it."""
        for mode, end in (("first", 30), ("last", 130)):
            with self.subTest(local_work=mode):
                parent = _node("P", 10, slow=(15, 25, 100), node={**HOLD, "local_work": mode})
                _call(parent, _node("D", 20))
                top = _client(parent)
                _run(top)
                self.assertEqual(_attempts(top), [(0, end, DropReason.NONE)])

    def test_the_failure_roll_is_at_the_end(self):
        """The node fails between 25 and 35 ms: after its calls, in its own work."""
        parent = _node("P", 10, fail=(25, 35), node=LAST)
        edge = _call(parent, _node("D", 20))
        top = _client(parent)
        _run(top)
        self.assertEqual(
            _attempts(edge), [(0, 20, DropReason.NONE)]
        )  # the call was made, and worked
        self.assertEqual(_attempts(top), [(0, 30, DropReason.SERVER_FAILURE)])

    def test_a_failed_call_means_no_local_work(self):
        parent = _node("P", 10, node=LAST)
        _call(parent, _node("D", 20, fail=(0, 1000)))
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(top), [(0, 20, DropReason.SERVER_FAILURE)])
        self.assertEqual(parent.service.in_flight, 0)

    def test_the_deadline_covers_the_calls_and_the_local_work(self):
        for deadline_ms, why in ((15, "during the call"), (25, "during the local work")):
            with self.subTest(deadline=why):
                parent = _node("P", 10, node=LAST)
                _call(parent, _node("D", 20))
                top = _client(parent, attempt_ms=deadline_ms)
                _run(top)
                self.assertEqual(_attempts(top), [(0, deadline_ms, DropReason.DEADLINE)])
                self.assertEqual(parent.service.in_flight, 0)

    def test_every_wave_runs_before_the_local_work(self):
        parent = _node("P", 10, node=LAST)
        first = _call(parent, _node("D1", 20), wave=0)
        second = _call(parent, _node("D2", 5), wave=1)
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(first), [(0, 20, DropReason.NONE)])
        self.assertEqual(_attempts(second), [(20, 25, DropReason.NONE)])
        self.assertEqual(_attempts(top), [(0, 35, DropReason.NONE)])

    def test_a_node_without_calls_is_a_plain_service_either_way(self):
        top = _client(_node("P", 10, node=LAST))
        _run(top)
        self.assertEqual(_attempts(top), [(0, 10, DropReason.NONE)])

    def test_a_cancellation_reaches_it_in_either_stage(self):
        """A sibling fails at 5 ms (during the call) or at 25 ms (during the local work)."""
        for fails_at, why in ((5, "during the call"), (25, "during the local work")):
            with self.subTest(cancelled=why):
                root_node = _node("R", 1, node={"propagate_cancellation": True})
                _call(root_node, _node("F", fails_at - 1, fail=(0, 1000)))
                parent = _node("P", 10, node=LAST)
                _call(parent, _node("D", 20))
                to_parent = _call(root_node, parent)
                _run(_client(root_node))
                self.assertEqual(_attempts(to_parent), [(1, fails_at, DropReason.CANCELLED)])
                self.assertEqual(parent.service.in_flight, 0)

    def test_a_tandem_node_cannot_do_its_work_last(self):
        with self.assertRaises(ValueError) as raised:
            _node("P", 10, node={"local_work": "last"})
        self.assertIn("hold_worker_during_fanout=True", str(raised.exception))
        # ... nor when the switch is set after the node was built.
        parent = _node("P", 10)
        parent.local_work = "last"
        with self.assertRaises(ValueError):
            _client(parent).start_request(Simulator(seed=1))

    def test_validation(self):
        self.assertEqual(LOCAL_WORK_MODES, ("first", "last"))
        with self.assertRaises(ValueError):
            _node("P", 10, node={"local_work": "middle"})


class TestOptionalCalls(unittest.TestCase):
    def test_a_failed_optional_call_fails_nothing(self):
        parent = _node("P", 10)
        optional = _call(parent, _node("O", 5, fail=(0, 1000)), optional=True)
        _call(parent, _node("D", 20))
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(optional), [(10, 15, DropReason.SERVER_FAILURE)])
        self.assertEqual(_attempts(top), [(0, 30, DropReason.NONE)])

    def test_the_node_still_waits_for_it(self):
        """The optional call is the slow one: the node ends when it does."""
        parent = _node("P", 10)
        _call(parent, _node("O", 40, fail=(0, 1000)), optional=True)
        _call(parent, _node("D", 20))
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(top), [(0, 50, DropReason.NONE)])

    def test_a_required_call_still_fails_the_node(self):
        parent = _node("P", 10)
        _call(parent, _node("O", 40), optional=True)
        _call(parent, _node("D", 20, fail=(0, 1000)))
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(top), [(0, 30, DropReason.SERVER_FAILURE)])

    def test_after_a_failed_optional_wave_the_next_one_starts(self):
        parent = _node("P", 10)
        _call(parent, _node("O", 5, fail=(0, 1000)), wave=0, optional=True)
        later = _call(parent, _node("D", 20), wave=1)
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(later), [(15, 35, DropReason.NONE)])
        self.assertEqual(_attempts(top), [(0, 35, DropReason.NONE)])

    def test_an_optional_call_refused_on_the_spot_completes_its_wave(self):
        """The optional callee has no worker free and no queue: it answers from
        inside the call. The wave it is alone in must still be over."""
        busy = _service("O", 100, workers=1, capacity=0)
        blocker = _client(busy)
        parent = _node("P", 10)
        optional = _call(parent, busy, wave=0, optional=True)
        later = _call(parent, _node("D", 20), wave=1)
        top = _client(parent)
        sim = Simulator(seed=1)
        blocker.start_request(sim)  # takes the one worker for 100 ms
        top.start_request(sim)
        sim.run()
        self.assertEqual(_attempts(optional), [(10, 10, DropReason.QUEUE_FULL)])
        self.assertEqual(_attempts(later), [(10, 30, DropReason.NONE)])
        self.assertEqual(_attempts(top), [(0, 30, DropReason.NONE)])

    def test_an_optional_call_refused_on_the_spot_as_the_last_stage(self):
        busy = _service("O", 100, workers=1, capacity=0)
        blocker = _client(busy)
        parent = _node("P", 10)
        _call(parent, busy, optional=True)
        top = _client(parent)
        sim = Simulator(seed=1)
        blocker.start_request(sim)
        top.start_request(sim)
        sim.run()
        self.assertEqual(_attempts(top), [(0, 10, DropReason.NONE)])

    def test_with_the_local_work_last(self):
        parent = _node("P", 10, node=LAST)
        _call(parent, _node("O", 5, fail=(0, 1000)), wave=0, optional=True)
        _call(parent, _node("D", 20), wave=1)
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(top), [(0, 35, DropReason.NONE)])  # 5 + 20, then 10 of its own

    def test_validation(self):
        parent = _node("P", 10)
        bad: Any = "yes"
        with self.assertRaises(ValueError):
            DownstreamEdge(client=_client(parent), target=parent, optional=bad)


WAIT = {"wave_join": "wait_all"}


class TestWaitAll(unittest.TestCase):
    def _fanout(
        self, join: str
    ) -> tuple[ClientRuntime, ClientRuntime, ClientRuntime, ClientRuntime]:
        """P calls a fast call that fails (5 ms) and a slow one that works (20 ms); then a third."""
        parent = _node("P", 10, node={"wave_join": join})
        fast = _call(parent, _node("F", 5, fail=(0, 1000)), wave=0)
        slow = _call(parent, _node("S", 20), wave=0)
        later = _call(parent, _node("L", 5), wave=1)
        top = _client(parent)
        _run(top)
        return top, fast, slow, later

    def test_short_circuit_fails_at_the_first_failure(self):
        top, _, _, later = self._fanout("short_circuit")
        self.assertEqual(_attempts(top), [(0, 15, DropReason.SERVER_FAILURE)])
        self.assertEqual(_attempts(later), [])

    def test_wait_all_fails_when_the_wave_has_drained(self):
        top, fast, slow, later = self._fanout("wait_all")
        self.assertEqual(_attempts(fast), [(10, 15, DropReason.SERVER_FAILURE)])
        self.assertEqual(_attempts(slow), [(10, 30, DropReason.NONE)])
        # The node learns nothing until its slowest call is back.
        self.assertEqual(_attempts(top), [(0, 30, DropReason.SERVER_FAILURE)])
        # ... and starts no further wave.
        self.assertEqual(_attempts(later), [])

    def test_the_first_required_failure_is_the_reason(self):
        parent = _node("P", 10, node=WAIT)
        _call(parent, _node("F1", 5, fail=(0, 1000)))  # fails at 15 ms: a server failure
        _call(parent, _node("F2", 50), attempt_ms=10)  # times out at 20 ms: a deadline
        _call(parent, _node("S", 30))
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(top), [(0, 40, DropReason.SERVER_FAILURE)])

    def test_nothing_changes_when_every_call_succeeds(self):
        for join in WAVE_JOIN_MODES:
            with self.subTest(wave_join=join):
                parent = _node("P", 10, node={"wave_join": join})
                _call(parent, _node("A", 5), wave=0)
                _call(parent, _node("B", 20), wave=0)
                later = _call(parent, _node("C", 5), wave=1)
                top = _client(parent)
                _run(top)
                self.assertEqual(_attempts(later), [(30, 35, DropReason.NONE)])
                self.assertEqual(_attempts(top), [(0, 35, DropReason.NONE)])

    def test_the_deadline_does_not_wait(self):
        parent = _node("P", 10, node=WAIT)
        _call(parent, _node("F", 5, fail=(0, 1000)))
        _call(parent, _node("S", 100))
        top = _client(parent, attempt_ms=40)
        _run(top)
        self.assertEqual(_attempts(top), [(0, 40, DropReason.DEADLINE)])

    def test_an_optional_failure_is_not_remembered(self):
        parent = _node("P", 10, node=WAIT)
        _call(parent, _node("O", 5, fail=(0, 1000)), optional=True)
        _call(parent, _node("S", 20))
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(top), [(0, 30, DropReason.NONE)])

    def test_calls_refused_on_the_spot_are_counted_too(self):
        """Both calls of the wave answer from inside the launch: one refused, one served."""
        busy = _service("B", 100, workers=1, capacity=0)
        blocker = _client(busy)
        parent = _node("P", 10, node=WAIT)
        _call(parent, busy)
        slow = _call(parent, _node("S", 20))
        top = _client(parent)
        sim = Simulator(seed=1)
        blocker.start_request(sim)
        top.start_request(sim)
        sim.run()
        self.assertEqual(_attempts(slow), [(10, 30, DropReason.NONE)])
        self.assertEqual(_attempts(top), [(0, 30, DropReason.QUEUE_FULL)])

    def test_with_the_local_work_last_a_failed_wave_skips_it(self):
        parent = _node("P", 10, node={**LAST, **WAIT})
        _call(parent, _node("F", 5, fail=(0, 1000)))
        _call(parent, _node("S", 20))
        top = _client(parent)
        _run(top)
        self.assertEqual(_attempts(top), [(0, 20, DropReason.SERVER_FAILURE)])
        self.assertEqual(parent.service.in_flight, 0)

    def test_validation(self):
        self.assertEqual(WAVE_JOIN_MODES, ("short_circuit", "wait_all"))
        with self.assertRaises(ValueError):
            _node("P", 10, node={"wave_join": "any"})


if __name__ == "__main__":
    unittest.main()
