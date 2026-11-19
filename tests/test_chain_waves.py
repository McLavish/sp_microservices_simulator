"""Call waves: `DownstreamEdge.wave` orders a node's downstream calls.

Edges with the same wave are launched together; waves run in ascending order and
a wave starts only when every call of the previous one has succeeded. The tests
below pin the ordering rules, what each per-node switch means across waves, and
-- most importantly -- that a node whose edges share one wave value is exactly
the single concurrent fan-out it always was.

Service times use `sigma=0`, which `ServiceConfig` floors to 1e-6: a 10 ms
service lasts 10 ms give or take a few nanoseconds. Assertions therefore relate
recorded instants to each other, or allow a microsecond, rather than hard-coding
a total.
"""

import hashlib
import unittest

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import DropReason, TimeInterval, ms_to_ns
from msim.faults import PartialFailure
from msim.retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy, RetryPolicy
from msim.service import AttemptControl, ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.transport import TransportDelay, TransportLink
from msim.workload import Workload

US = 1_000  # one microsecond, the slack every "about equal" below allows


def _service(
    name: str,
    median_ms: float,
    *,
    workers: int = 8,
    capacity: int | None = None,
    sigma: float = 0.0,
    partial_failures: list[PartialFailure] | None = None,
) -> ServiceRuntime:
    return ServiceRuntime(
        cfg=ServiceConfig(
            name=name,
            latency_median=ms_to_ns(median_ms),
            latency_lognorm_sigma=sigma,
            workers=workers,
            queue_capacity=capacity,
            partial_failures=list(partial_failures or []),
        )
    ).bind()


def _node(name: str, median_ms: float, **kwargs) -> ServiceNodeRuntime:
    return ServiceNodeRuntime(service=_service(name, median_ms, **kwargs))


def _client(
    target,
    *,
    retry: RetryPolicy | None = None,
    attempt_ms: float | None = 1000.0,
    global_ms: float | None = None,
) -> ClientRuntime:
    return ClientRuntime(
        cfg=ClientConfig(
            retry=retry or NoRetryPolicy(),
            timeout_controller=StaticTimeoutController(
                global_timeout=None if global_ms is None else ms_to_ns(global_ms),
                attempt_timeout=None if attempt_ms is None else ms_to_ns(attempt_ms),
            ),
        ),
        service=target,
    )


def _call(parent: ServiceNodeRuntime, child, wave: int, **client_kwargs) -> ClientRuntime:
    """Add `parent -> child` on `wave`; return the edge's client."""
    client = _client(child, **client_kwargs)
    target = child.target if isinstance(child, TransportLink) else child
    parent.add_downstream(DownstreamEdge(client=client, target=target, wave=wave))
    return client


def _only_attempt(client: ClientRuntime):
    (root,) = client.roots
    (attempt,) = root.attempts
    return attempt


class TestWaveField(unittest.TestCase):
    def test_wave_default_is_zero_and_validated(self):
        node = _node("B", 1)
        client = _client(node)
        self.assertEqual(DownstreamEdge(client=client, target=node).wave, 0)
        self.assertEqual(DownstreamEdge(client=client, target=node, wave=3).wave, 3)
        for bad in (-1, True, False, 1.0, "1", None):
            with self.subTest(wave=bad), self.assertRaises(ValueError):
                DownstreamEdge(client=client, target=node, wave=bad)  # ty: ignore[invalid-argument-type]


class TestWaveOrdering(unittest.TestCase):
    def test_same_wave_edges_launch_together_in_list_order(self):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        first = _call(parent, _node("B", 10), wave=0)
        second = _call(parent, _node("C", 30), wave=0)
        order: list[str] = []
        for label, client in (("first", first), ("second", second)):
            original = client.start_request

            def spy(sim_, on_done=None, override_global_deadline=None, _l=label, _o=original):
                order.append(_l)
                return _o(sim_, on_done=on_done, override_global_deadline=override_global_deadline)

            client.start_request = spy  # ty: ignore[invalid-assignment]
        _client(parent).start_request(sim)
        sim.run()

        self.assertEqual(order, ["first", "second"])
        self.assertEqual(_only_attempt(first).interval.begin, _only_attempt(second).interval.begin)

    def test_next_wave_starts_when_the_previous_wave_has_fully_succeeded(self):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        fast = _call(parent, _node("B", 10), wave=0)
        slow = _call(parent, _node("C", 30), wave=0)
        last = _call(parent, _node("D", 5), wave=1)
        top = _client(parent)
        top.start_request(sim)
        sim.run()

        fast_a, slow_a, last_a = (_only_attempt(c) for c in (fast, slow, last))
        self.assertEqual(fast_a.interval.begin, slow_a.interval.begin)
        # The second wave starts at the instant the SLOWEST call of the first ends...
        self.assertEqual(last_a.interval.begin, max(fast_a.interval.end, slow_a.interval.end))
        self.assertGreater(slow_a.interval.end, fast_a.interval.end)
        # ... and the node completes with the last wave: local + 30 + 5 ms end to end.
        top_a = _only_attempt(top)
        self.assertTrue(top_a.success)
        self.assertEqual(top_a.interval.end, last_a.interval.end)
        self.assertAlmostEqual(top_a.interval.end, ms_to_ns(1 + 30 + 5), delta=US)

    def test_waves_run_in_ascending_order_whatever_the_edge_order(self):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        # Added last-wave first, with a gap in the numbering.
        third = _call(parent, _node("D", 5), wave=9)
        second = _call(parent, _node("C", 5), wave=4)
        first = _call(parent, _node("B", 5), wave=0)
        _client(parent).start_request(sim)
        sim.run()

        a, b, c = (_only_attempt(client) for client in (first, second, third))
        self.assertEqual(b.interval.begin, a.interval.end)
        self.assertEqual(c.interval.begin, b.interval.end)

    def test_a_retried_call_holds_its_wave_until_it_succeeds(self):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        # B's first attempt (1 -> 11 ms) rolls its failure inside the window, the
        # retry (16 -> 26 ms) outside it.
        flaky = _node(
            "B", 10, partial_failures=[PartialFailure(TimeInterval(0, ms_to_ns(12)), p_fail=1.0)]
        )
        first = _call(
            parent, flaky, wave=0, retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(5))
        )
        second = _call(parent, _node("D", 5), wave=1)
        top = _client(parent)
        top.start_request(sim)
        sim.run()

        (root,) = first.roots
        self.assertEqual([a.success for a in root.attempts], [False, True])
        self.assertEqual(_only_attempt(second).interval.begin, root.attempts[-1].interval.end)
        self.assertTrue(_only_attempt(top).success)


class TestWaveFailure(unittest.TestCase):
    def test_failure_in_a_wave_fails_the_node_and_skips_later_waves(self):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        broken = _node(
            "B", 10, partial_failures=[PartialFailure(TimeInterval(0, ms_to_ns(1000)), p_fail=1.0)]
        )
        later_node = _node("D", 5)
        _call(parent, broken, wave=0)
        later = _call(parent, later_node, wave=1)
        top = _client(parent)
        top.start_request(sim)
        sim.run()

        top_a = _only_attempt(top)
        self.assertFalse(top_a.success)
        self.assertEqual(top_a.drop_reason, DropReason.SERVER_FAILURE)
        # The second wave was never launched: no call, no arrival at D.
        self.assertEqual(later.roots, [])
        self.assertEqual(later.attempts_total, 0)
        self.assertEqual(later_node.service.admitted_total, 0)

    def test_sync_rejection_in_a_later_wave_still_launches_that_whole_wave(self):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        # E has one worker, no waiting room, and is busy for the whole test.
        full = _node("E", 500, workers=1, capacity=0)
        _client(full).start_request(sim)
        sibling_node = _node("F", 5)
        _call(parent, _node("B", 10), wave=0)
        rejected = _call(parent, full, wave=1)
        sibling = _call(parent, sibling_node, wave=1)
        top = _client(parent)
        top.start_request(sim)
        sim.run()

        self.assertEqual(_only_attempt(rejected).drop_reason, DropReason.QUEUE_FULL)
        # The rejection is synchronous, yet its sibling in the same wave was launched.
        self.assertEqual(len(sibling.roots), 1)
        self.assertEqual(sibling_node.service.admitted_total, 1)
        self.assertEqual(_only_attempt(top).drop_reason, DropReason.QUEUE_FULL)


class TestWaveGoesThroughTheHeap(unittest.TestCase):
    def test_next_wave_keeps_fcfs_at_a_shared_callee(self):
        """A second call to the callee that just answered must queue behind
        work that was already waiting there.

        The shared service has one worker. The parent's first call holds it
        from 1 to 11 ms; another client's request arrives at 5 ms and queues.
        When the first call completes, the callee has freed its worker but not
        yet resumed its queue, so a second call submitted from inside that
        completion callback would take the worker ahead of the queued request.
        Through the event heap it arrives after the queue has resumed.
        """
        sim = Simulator(seed=1)
        shared = _node("S", 10, workers=1)
        parent = _node("P", 1)
        first = _call(parent, shared, wave=0)
        second = _call(parent, shared, wave=1)
        other = _client(shared)
        sim.schedule(ms_to_ns(5), lambda: other.start_request(sim))
        _client(parent).start_request(sim)
        sim.run()

        first_a, second_a, other_a = (_only_attempt(c) for c in (first, second, other))
        self.assertTrue(first_a.success and second_a.success and other_a.success)
        self.assertEqual(second_a.interval.begin, first_a.interval.end)
        self.assertLess(other_a.interval.end, second_a.interval.end)
        # Served third: two 10 ms services after the first call ended.
        self.assertAlmostEqual(second_a.interval.end - first_a.interval.end, ms_to_ns(20), delta=US)

    def test_single_wave_schedules_no_wave_event(self):
        def scheduled_wave_events(waves: tuple[int, int]) -> int:
            sim = Simulator(seed=1)
            names: list[str] = []
            original = sim.schedule

            def spy(at, fn):
                names.append(getattr(fn, "__name__", ""))
                return original(at, fn)

            sim.schedule = spy  # ty: ignore[invalid-assignment]
            parent = _node("P", 1)
            _call(parent, _node("B", 10), wave=waves[0])
            _call(parent, _node("C", 10), wave=waves[1])
            top = _client(parent)
            for _ in range(3):
                top.start_request(sim)
            sim.run()
            return names.count("_on_next_wave")

        self.assertEqual(scheduled_wave_events((0, 0)), 0)
        self.assertEqual(scheduled_wave_events((5, 5)), 0)
        self.assertEqual(scheduled_wave_events((0, 1)), 3)  # one transition per request


class _Released:
    """A target whose attempts succeed when `release` is called, not before."""

    def __init__(self, name: str) -> None:
        self.cfg = ServiceConfig(
            name=name, latency_median=ms_to_ns(1), latency_lognorm_sigma=0.0, workers=1
        )
        self._waiting: list = []

    def submit_attempt(self, sim, on_done, attempt_deadline=None) -> AttemptControl:
        self._waiting.append(on_done)
        return AttemptControl()

    def release(self) -> None:
        waiting, self._waiting = self._waiting, []
        for on_done in waiting:
            on_done(True, 1, DropReason.NONE, 0, False)


class TestWaveDeadline(unittest.TestCase):
    def _run(self, release_at: int):
        """P's attempt deadline is 50 ms; its first-wave call succeeds at `release_at`."""
        sim = Simulator(seed=1)
        gate = _Released("G")
        # Scheduled BEFORE anything else, so at an equal instant it runs before
        # the deadline guard the parent arms later.
        sim.schedule(release_at, gate.release)
        parent = _node("P", 1)
        parent.propagate_deadline = False  # the child may succeed AT the parent's deadline
        first = _client(gate)
        parent.add_downstream(DownstreamEdge(client=first, target=parent, wave=0))
        later = _call(parent, _node("D", 5), wave=1)
        top = _client(parent, attempt_ms=50)
        top.start_request(sim)
        sim.run()
        return top, later

    def test_deadline_guard_beats_a_wave_transition_at_the_deadline_instant(self):
        deadline = ms_to_ns(50)
        top, later = self._run(release_at=deadline)
        # The first wave completed at the deadline instant; the guard, armed
        # earlier, fires before the pending wave event and cancels it.
        self.assertEqual(_only_attempt(top).drop_reason, DropReason.DEADLINE)
        self.assertEqual(_only_attempt(top).interval.end, deadline)
        self.assertEqual(later.roots, [])
        self.assertEqual(later.attempts_total, 0)

    def test_a_wave_completing_one_ns_earlier_starts_the_next(self):
        deadline = ms_to_ns(50)
        top, later = self._run(release_at=deadline - 1)
        self.assertEqual(_only_attempt(top).drop_reason, DropReason.DEADLINE)
        self.assertEqual(len(later.roots), 1)
        self.assertEqual(_only_attempt(later).interval.begin, deadline - 1)

    def test_every_wave_gets_the_same_propagated_deadline(self):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        first = _call(parent, _node("B", 10), wave=0)
        second = _call(parent, _node("D", 5), wave=1)
        top = _client(parent, attempt_ms=200)
        top.start_request(sim)
        sim.run()

        deadline = _only_attempt(top).deadline
        self.assertEqual(deadline, ms_to_ns(200))
        self.assertEqual(first.roots[0].global_deadline, deadline)
        self.assertEqual(second.roots[0].global_deadline, deadline)
        # The later wave has less of it left.
        self.assertGreater(
            _only_attempt(second).interval.begin, _only_attempt(first).interval.begin
        )


class TestWaveSwitches(unittest.TestCase):
    def _occupancy(self, hold: bool):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        parent.hold_worker_during_fanout = hold
        _call(parent, _node("B", 10), wave=0)
        _call(parent, _node("D", 10), wave=1)
        probes: list[int] = []
        for at_ms in (6, 16):  # mid first wave, mid second wave
            sim.schedule(ms_to_ns(at_ms), lambda: probes.append(parent.service.in_flight))
        top = _client(parent)
        top.start_request(sim)
        sim.run()
        return probes, _only_attempt(top), parent

    def test_tandem_releases_the_worker_once_before_the_first_wave(self):
        probes, attempt, parent = self._occupancy(hold=False)
        self.assertEqual(probes, [0, 0])
        self.assertTrue(attempt.success)
        self.assertAlmostEqual(attempt.service_time_ns, ms_to_ns(1), delta=US)  # local only
        self.assertEqual(parent.service.in_flight, 0)

    def test_hold_mode_keeps_the_worker_across_every_wave(self):
        probes, attempt, parent = self._occupancy(hold=True)
        self.assertEqual(probes, [1, 1])
        self.assertTrue(attempt.success)
        self.assertAlmostEqual(attempt.service_time_ns, ms_to_ns(1 + 10 + 10), delta=US)
        self.assertEqual(parent.service.in_flight, 0)

    def test_cancel_from_above_stops_unlaunched_waves(self):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        parent.propagate_cancellation = True
        first_node, later_node = _node("B", 10), _node("D", 5)
        first = _call(parent, first_node, wave=0)
        later = _call(parent, later_node, wave=1)
        top = _client(parent)
        root = top.start_request(sim)
        sim.schedule(ms_to_ns(5), lambda: root.cancel())  # mid first wave
        sim.run()

        self.assertTrue(root.cancelled)
        self.assertEqual(_only_attempt(first).drop_reason, DropReason.CANCELLED)
        self.assertEqual(later.roots, [])
        self.assertEqual(later_node.service.admitted_total, 0)
        for node in (parent, first_node, later_node):
            self.assertEqual(node.service.in_flight, 0)
            self.assertEqual(node.service.queue_len(), 0)

    def test_transport_link_edge_pays_both_legs_in_each_wave(self):
        sim = Simulator(seed=1)
        parent = _node("P", 1)
        delay = TransportDelay(median=ms_to_ns(1))  # sigma 0: exactly 1 ms per leg
        first = _call(parent, TransportLink(_node("B", 10), delay), wave=0)
        second = _call(parent, TransportLink(_node("D", 5), delay), wave=1)
        top = _client(parent)
        top.start_request(sim)
        sim.run()

        first_a, second_a = _only_attempt(first), _only_attempt(second)
        for attempt in (first_a, second_a):
            self.assertEqual(
                attempt.interval.end - attempt.interval.begin,
                2 * ms_to_ns(1) + attempt.service_time_ns,
            )
        self.assertEqual(second_a.interval.begin, first_a.interval.end)
        self.assertEqual(_only_attempt(top).interval.end, second_a.interval.end)


def _loaded_fanout(waves: tuple[int, int, int]):
    """A contended fan-out to three leaves, so ordering mistakes are observable."""
    sim = Simulator(seed=1234)
    parent = _node("A", 2, workers=4, capacity=3, sigma=0.3)
    clients = []
    for index, wave in enumerate(waves):
        leaf = _node(
            f"L{index}",
            10,
            workers=1,
            capacity=3,
            sigma=0.3,
            partial_failures=[PartialFailure(TimeInterval(ms_to_ns(500), ms_to_ns(1500)), 0.3)],
        )
        clients.append(
            _call(
                parent,
                leaf,
                wave=wave,
                retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(5)),
                attempt_ms=60,
            )
        )
    top = _client(
        parent, retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(10)), attempt_ms=300
    )
    Workload(base_rps=150.0, duration_s=3.0, load_spikes=[], rng_seed=1234).drive(
        sim, top.start_request
    )
    sim.run()
    digest = hashlib.sha256()
    for client in (top, *clients):
        for root in client.roots:
            for a in root.attempts:
                digest.update(
                    repr((a.interval.begin, a.interval.end, a.success, a.drop_reason.name)).encode()
                )
    return digest.hexdigest(), sim._seq, sim.rng().getstate()


class TestSingleWaveIsTheOldFanout(unittest.TestCase):
    def test_uniform_nonzero_wave_is_the_single_wave_path(self):
        # Same attempts, same number of events, same RNG state: only the ORDER
        # of the wave values matters, so one distinct value is one wave.
        self.assertEqual(_loaded_fanout((0, 0, 0)), _loaded_fanout((7, 7, 7)))

    def test_distinct_waves_change_the_run(self):
        self.assertNotEqual(_loaded_fanout((0, 0, 0))[0], _loaded_fanout((0, 1, 2))[0])
