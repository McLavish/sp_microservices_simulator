import os
import random
import unittest

from caller import ClientConfig, ClientRuntime
from chain import DownstreamEdge, ServiceNodeRuntime, _FanOutAttempt
from core import DropReason, RootRequest, TimeInterval, ms_to_ns, s_to_ns
from faults import PartialFailure
from retry_policies import (
    CBState,
    CountBasedCircuitBreakerPolicy,
    FixedBackoffRetryPolicy,
    LeakyRateLimiterPolicy,
    NoRetryPolicy,
    RetryBudgetPolicy,
)
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload


def _build_fanout(hold_worker: bool):
    """Parent node (1 worker, fast local) fanning out to one slow downstream.

    Local stage takes ~10ms; the downstream call takes ~200ms, so there is a wide
    window during which the parent is waiting on its fan-out. Returns
    (parent_node, downstream_node, downstream_client).
    """
    parent_cfg = ServiceConfig(
        name="parent",
        latency_median=ms_to_ns(10),
        latency_lognorm_sigma=1e-6,
        workers=1,
        queue_capacity=None,
    )
    downstream_cfg = ServiceConfig(
        name="downstream",
        latency_median=ms_to_ns(200),
        latency_lognorm_sigma=1e-6,
        workers=4,
        queue_capacity=None,
    )

    parent = ServiceNodeRuntime(
        service=ServiceRuntime(cfg=parent_cfg).bind(),
        hold_worker_during_fanout=hold_worker,
    )
    downstream = ServiceNodeRuntime(service=ServiceRuntime(cfg=downstream_cfg).bind())

    client = ClientRuntime(
        cfg=ClientConfig(
            retry=NoRetryPolicy(),
            timeout_controller=StaticTimeoutController(global_timeout=None, attempt_timeout=None),
        ),
        service=downstream,
    )
    parent.add_downstream(DownstreamEdge(client=client, target=downstream))
    return parent, downstream, client


class TestChainWorkerModel(unittest.TestCase):
    def test_default_is_release(self):
        """Tandem-queue (release) model is the default, per CLAUDE.md."""
        node = ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="n",
                    latency_median=ms_to_ns(1),
                    latency_lognorm_sigma=1e-6,
                    workers=1,
                )
            ).bind()
        )
        self.assertFalse(node.hold_worker_during_fanout)

    def test_release_mode_frees_worker_during_fanout(self):
        sim = Simulator(seed=1)
        parent, downstream, _client = _build_fanout(hold_worker=False)

        probe: dict[str, int] = {}

        def take_probe():
            probe["parent_in_flight"] = parent.service.in_flight
            probe["downstream_in_flight"] = downstream.service.in_flight

        ends: list[tuple[bool, int]] = []

        def on_done(success, _svc_time, _drop_reason, _q, _censored=False):
            ends.append((success, sim.timestep))

        # Probe mid fan-out: local stage done (~10ms), downstream still running (~210ms).
        sim.schedule(ms_to_ns(100), take_probe)
        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertEqual(probe["downstream_in_flight"], 1)
        # Tandem model: the parent worker is freed while it waits on the fan-out.
        self.assertEqual(probe["parent_in_flight"], 0)
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0][0])
        self.assertGreater(ends[0][1], ms_to_ns(100))

    def test_hold_mode_occupies_worker_during_fanout(self):
        sim = Simulator(seed=1)
        parent, downstream, _client = _build_fanout(hold_worker=True)

        probe: dict[str, int] = {}

        def take_probe():
            probe["parent_in_flight"] = parent.service.in_flight
            probe["downstream_in_flight"] = downstream.service.in_flight

        ends: list[tuple[bool, int]] = []

        def on_done(success, _svc_time, _drop_reason, _q, _censored=False):
            ends.append((success, sim.timestep))

        sim.schedule(ms_to_ns(100), take_probe)
        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertEqual(probe["downstream_in_flight"], 1)
        # Thread-per-request model: the parent worker stays occupied for the whole
        # downstream wait, so capacity is consumed all the way down the tree.
        self.assertEqual(probe["parent_in_flight"], 1)
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0][0])
        self.assertGreater(ends[0][1], ms_to_ns(100))

    def test_release_mode_allows_concurrent_requests(self):
        sim = Simulator(seed=1)
        parent, _downstream, _client = _build_fanout(hold_worker=False)

        ends: list[tuple[bool, int]] = []

        def on_done(success, _svc_time, _drop_reason, _q, _censored=False):
            ends.append((success, sim.timestep))

        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertEqual(len(ends), 2)
        self.assertTrue(all(success for success, _ in ends))
        # With the worker freed during fan-out, the second request's local stage
        # overlaps the first's downstream wait, so both finish around ~210-220ms.
        self.assertLess(max(t for _, t in ends), ms_to_ns(300))

    def test_hold_mode_serializes_requests(self):
        sim = Simulator(seed=1)
        parent, _downstream, _client = _build_fanout(hold_worker=True)

        ends: list[tuple[bool, int]] = []

        def on_done(success, _svc_time, _drop_reason, _q, _censored=False):
            ends.append((success, sim.timestep))

        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        parent.submit_attempt(sim, on_done, attempt_deadline=None)
        sim.run()

        self.assertEqual(len(ends), 2)
        self.assertTrue(all(success for success, _ in ends))
        # The single worker is held through the entire subtree, so the second
        # request cannot start until the first fully completes (~210ms), pushing
        # its completion out to ~420ms.
        self.assertGreater(max(t for _, t in ends), ms_to_ns(350))


class TestFinalizeExactlyOnce(unittest.TestCase):
    def test_double_finalize_raises_and_preserves_in_flight(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(10), latency_lognorm_sigma=1e-6, workers=1
            )
        ).bind()

        def behavior(service, sim2, deadline, finalize, begin_at, release_worker, register_cancel):
            def finish():
                finalize(True, ms_to_ns(5), DropReason.NONE)
                finalize(True, ms_to_ns(5), DropReason.NONE)  # bug: second call

            sim2.schedule(begin_at + ms_to_ns(5), finish)

        svc.submit_attempt(sim, lambda *a: None, behavior=behavior)
        with self.assertRaises(RuntimeError):
            sim.run()
        self.assertGreaterEqual(svc.in_flight, 0)

    def test_release_worker_after_finalize_raises(self):
        # finalize() on the unreleased path already returns the worker; a
        # subsequent release_worker() would decrement in_flight a second time.
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(10), latency_lognorm_sigma=1e-6, workers=1
            )
        ).bind()

        def behavior(service, sim2, deadline, finalize, begin_at, release_worker, register_cancel):
            def finish():
                finalize(True, ms_to_ns(5), DropReason.NONE)
                release_worker()  # bug: worker already released via finalize

            sim2.schedule(begin_at + ms_to_ns(5), finish)

        svc.submit_attempt(sim, lambda *a: None, behavior=behavior)
        with self.assertRaises(RuntimeError):
            sim.run()
        self.assertGreaterEqual(svc.in_flight, 0)


class TestCancellationPropagation(unittest.TestCase):
    """propagate_cancellation selects between two real architectures.

    False (default): a short-circuited fan-out leaves siblings running to
    completion, consuming downstream capacity — a stack with no cancellation
    plumbing. True: Go-style context propagation tears the subtree down.
    """

    def _fanout(self, propagate, slow_ms=200, depth1_workers=1):
        """A -> {FAST (always fails), SLOW (succeeds after slow_ms)}."""
        sim = Simulator(seed=1)

        def node(name, median_ms, p_fail=0.0):
            failures = (
                [PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=p_fail)] if p_fail else []
            )
            return ServiceNodeRuntime(
                service=ServiceRuntime(
                    cfg=ServiceConfig(
                        name=name,
                        latency_median=ms_to_ns(median_ms),
                        latency_lognorm_sigma=1e-6,
                        workers=depth1_workers,
                        partial_failures=failures,
                    )
                ).bind()
            )

        a = node("A", 1)
        fast = node("FAST", 5, p_fail=1.0)
        slow = node("SLOW", slow_ms)
        a.propagate_cancellation = propagate

        def edge(target):
            return DownstreamEdge(
                client=ClientRuntime(
                    cfg=ClientConfig(
                        retry=NoRetryPolicy(),
                        timeout_controller=StaticTimeoutController(
                            global_timeout=None, attempt_timeout=ms_to_ns(5000)
                        ),
                    ),
                    service=target,
                ),
                target=target,
            )

        e_fast, e_slow = edge(fast), edge(slow)
        a.add_downstream(e_fast)
        a.add_downstream(e_slow)
        top = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(5000)
                ),
            ),
            service=a,
        )
        top.start_request(sim)
        sim.run()
        return sim, top, e_slow.client, slow

    def test_default_leaves_orphaned_siblings_running(self):
        sim, top, slow_client, slow_node = self._fanout(propagate=False)
        summary = slow_client.metrics().summary()
        # The abandoned call runs to completion and really did succeed.
        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.dropped_cancelled, 0)
        # ...and held its worker for the full 200 ms.
        self.assertGreaterEqual(sim.timestep, ms_to_ns(200))

    def test_propagation_abandons_siblings_and_reclaims_capacity(self):
        sim, top, slow_client, slow_node = self._fanout(propagate=True)
        summary = slow_client.metrics().summary()
        self.assertEqual(summary.succeeded, 0)
        self.assertEqual(summary.dropped_cancelled, 1)
        # The worker is handed back when FAST fails (~6 ms), not at 200 ms.
        self.assertLess(sim.timestep, ms_to_ns(100))
        self.assertEqual(slow_node.service.in_flight, 0)

    def test_cancellation_does_not_change_the_parent_verdict(self):
        """Cancelling siblings must not alter why the parent failed."""
        for propagate in (False, True):
            with self.subTest(propagate=propagate):
                _, top, _, _ = self._fanout(propagate=propagate)
                self.assertEqual(top.roots[0].attempts[-1].drop_reason, DropReason.SERVER_FAILURE)

    def test_no_worker_is_leaked_by_cancellation(self):
        """finalize-exactly-once and release_worker-at-most-once must hold.

        Both raise on violation, so a leak or a double-release would surface as
        an exception during run(); this also pins that every worker is returned.
        """
        for propagate in (False, True):
            with self.subTest(propagate=propagate):
                _, top, _, slow_node = self._fanout(propagate=propagate)
                self.assertEqual(slow_node.service.in_flight, 0)
                self.assertEqual(slow_node.service.queue, [])

    def test_cancelled_roots_are_not_counted_as_failures(self):
        """Cancelled work is abandoned, not failed: its own bucket, so one root
        cause is not charged again at every sibling it cancelled."""
        _, _, slow_client, _ = self._fanout(propagate=True)
        summary = slow_client.metrics().summary()
        self.assertEqual(summary.dropped_cancelled, 1)
        self.assertEqual(summary.dropped_server_failure, 0)
        self.assertEqual(summary.dropped_deadline, 0)
        self.assertEqual(summary.dropped_queue, 0)


class TestSettleIsTheOnlyTeardownPath(unittest.TestCase):
    """Every settling transition runs the same teardown, in the same order.

    `_settle` used to be a partial teardown wearing the general name: commit,
    cancel the deadline guard, finalize — and no sibling teardown, which the
    other three settling paths each layered on top by hand. It was correct only
    because its callers happened to run with nothing launched or everything
    already done, a precondition recorded nowhere. A new completion path
    (quorum: succeed when k of n downstreams return) reaching for it would
    leave the outstanding siblings running, consuming shared downstream
    capacity and recording outcomes of their own — precisely the cross-chain
    retry externality `experiments/multichain_retry_storm.py` exhibits.
    """

    def _attempt(self):
        sim = Simulator(seed=1)
        node = ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="node",
                    latency_median=ms_to_ns(1),
                    latency_lognorm_sigma=1e-6,
                    workers=1,
                )
            ).bind(),
            propagate_cancellation=True,
        )
        finalized: list[tuple[bool, DropReason]] = []
        attempt = _FanOutAttempt(
            node=node,
            svc=node.service,
            sim=sim,
            deadline=None,
            finalize=lambda success, _elapsed, drop: finalized.append((success, drop)),
            begin_at=0,
            release_worker=lambda: None,
            register_cancel=lambda _fn: None,
        )
        return attempt, finalized

    def test_settle_abandons_downstream_calls_still_in_flight(self):
        attempt, finalized = self._attempt()
        cancelled: list[str] = []
        attempt._launched = [RootRequest(cancel=lambda: cancelled.append("sibling"))]

        attempt._settle(True, DropReason.NONE)

        self.assertEqual(cancelled, ["sibling"])
        self.assertEqual(finalized, [(True, DropReason.NONE)])

    def test_settle_keeps_the_root_that_reported_the_outcome(self):
        attempt, finalized = self._attempt()
        cancelled: list[str] = []
        reporter = RootRequest(cancel=lambda: cancelled.append("reporter"))
        sibling = RootRequest(cancel=lambda: cancelled.append("sibling"))
        attempt._launched = [reporter, sibling]

        attempt._settle(False, DropReason.SERVER_FAILURE, keep=reporter)

        self.assertEqual(cancelled, ["sibling"])
        self.assertEqual(finalized, [(False, DropReason.SERVER_FAILURE)])


class TestCancellationIntegration(unittest.TestCase):
    """Cancellation must not leak into the subsystems it crosses.

    Each case reproduces a regression from the first cancellation implementation,
    where the feature worked in isolation but corrupted queue capacity, breaker
    state, or budget accounting.
    """

    def _service(self, median_ms=100, workers=1, capacity=None, p_fail=0.0):
        failures = [PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=p_fail)] if p_fail else []
        return ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(median_ms),
                latency_lognorm_sigma=1e-6,
                workers=workers,
                queue_capacity=capacity,
                partial_failures=failures,
            )
        ).bind()

    def test_cancelled_queue_entry_frees_its_capacity_slot(self):
        sim = Simulator(seed=1)
        svc = self._service(workers=1, capacity=1)
        seen = []
        svc.submit_attempt(sim, lambda *a: seen.append(("A", a[2])))  # takes the worker
        handle = svc.submit_attempt(sim, lambda *a: seen.append(("B", a[2])))  # queued
        handle.cancel()
        self.assertEqual(svc.queue_len(), 0)
        svc.submit_attempt(sim, lambda *a: seen.append(("C", a[2])))
        # C must fit: B gave its slot back even though its tombstone remains.
        self.assertNotIn(("C", DropReason.QUEUE_FULL), seen)

    def test_dequeued_attempt_does_not_inherit_the_queued_cancel(self):
        """A behavior that registers no teardown must be uncancellable, not
        cancellable via the stale queued callback (which would report CANCELLED
        without releasing the worker, then finalize on top of it)."""
        sim = Simulator(seed=1)
        svc = self._service(workers=1)
        calls = []

        def behavior(service, sim2, deadline, finalize, begin_at, release_worker, register_cancel):
            sim2.schedule(
                begin_at + ms_to_ns(100),
                lambda: finalize(True, ms_to_ns(100), DropReason.NONE),
            )

        svc.submit_attempt(sim, lambda *a: calls.append(a[2]))
        handle = svc.submit_attempt(sim, lambda *a: calls.append(a[2]), behavior=behavior)
        sim.run(until=ms_to_ns(150))  # second attempt has been dequeued into `behavior`
        handle.cancel()
        sim.run()
        self.assertEqual(calls, [DropReason.NONE, DropReason.NONE])

    def test_cancellation_does_not_trip_the_circuit_breaker(self):
        """Cancelled work is abandoned, not evidence about the service."""
        sim = Simulator(seed=1)
        svc = self._service(median_ms=200, workers=1)
        breaker = CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(1, 1),
            success_threshold_ratio=(1, 1),
            underlying_strategy=NoRetryPolicy(),
            half_open_delay=s_to_ns(10),
        )
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=breaker,
                timeout_controller=StaticTimeoutController(None, ms_to_ns(5000)),
            ),
            service=svc,
        )
        root = client.start_request(sim)
        sim.run(until=ms_to_ns(10))
        root.cancel()
        sim.run()
        self.assertEqual(breaker.get_state(), CBState.CLOSED)

    def test_cancelling_during_backoff_does_not_mint_retry_quota(self):
        """A cancel-time refund is deliberately NOT issued for throttling
        policies.

        The charge stands. Refunding late is unsound: every policy here holds
        either single-slot undo state or a capped, refilling balance, so a
        refund arriving after an intervening deposit clamps at the cap and MINTS
        quota that was never spent. Over-counting one retry is strictly safer
        than handing out an extra one. The immediate deadline-veto refund, where
        nothing can intervene, is still exact — see
        TestVetoedRetriesAreRefunded.
        """
        sim = Simulator(seed=1)
        svc = self._service(median_ms=10, workers=1, p_fail=1.0)
        budget = RetryBudgetPolicy(
            budget_ratio=0.1,
            max_retries=5,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=5, delay=ms_to_ns(500)),
        )
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=budget,
                timeout_controller=StaticTimeoutController(None, ms_to_ns(5000)),
            ),
            service=svc,
        )
        before = budget._tokens
        root = client.start_request(sim)
        sim.run(until=ms_to_ns(50))  # first attempt failed; retry scheduled and charged
        root.cancel()
        sim.run()
        self.assertEqual(len(root.attempts) - 1, 0)  # no retry ever ran
        # Charged, not refunded: conservative rather than mintable.
        self.assertEqual(budget._tokens, before - budget._retry_cost)

    def test_backoff_cancelled_root_records_its_own_completion_time(self):
        sim = Simulator(seed=1)
        svc = self._service(median_ms=10, workers=1, p_fail=1.0)
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=5, delay=ms_to_ns(500)),
                timeout_controller=StaticTimeoutController(None, ms_to_ns(5000)),
            ),
            service=svc,
        )
        root = client.start_request(sim)
        sim.run(until=ms_to_ns(50))
        root.cancel()
        sim.run()
        # Completion is the cancel instant, not the previous attempt's end.
        self.assertIsNotNone(root.completed_at)
        self.assertGreater(root.completed_at, root.attempts[-1].interval.end)


class TestCancellationCrossSubsystem(unittest.TestCase):
    """Seams the cancellation feature crosses. Every case here is a regression
    that a feature-local test passed straight over."""

    def _backoff_cancelled_client(
        self, backoff_ms=5000, cancel_at_ms=3000, attempt_timeout_ms=5000
    ):
        """A root cancelled while BACKING OFF: it has no attempt of its own
        carrying CANCELLED, and completes later than any of its attempt ends."""
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(100),
                latency_lognorm_sigma=1e-6,
                workers=1,
                partial_failures=[PartialFailure(TimeInterval(0, s_to_ns(1000)), p_fail=1.0)],
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=5, delay=ms_to_ns(backoff_ms)),
                timeout_controller=StaticTimeoutController(None, ms_to_ns(attempt_timeout_ms)),
            ),
            service=svc,
        )
        root = client.start_request(sim)
        sim.run(until=ms_to_ns(cancel_at_ms))
        root.cancel()
        sim.run()
        return client, root

    def test_csv_bucket_range_covers_root_completion_times(self):
        """export_csv sized its bucket grid from attempt ends while bucketing
        roots at completed_at, so a backoff-cancelled root fell past the last
        bucket and reindex silently dropped it from every root-level column."""
        import tempfile

        import pandas as pd

        client, _ = self._backoff_cancelled_client()
        summary = client.metrics().summary()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.csv")
            client.metrics().export_csv(path, granularity_s=1.0)
            df = pd.read_csv(path)
        self.assertEqual(int(df["failure_cancelled"].sum()), summary.dropped_cancelled)
        self.assertEqual(int(df["failure_root"].sum()), summary.total - summary.succeeded)

    def test_committed_outcome_is_not_rewritten_by_a_reentrant_cancel(self):
        """A cancel arriving while a behavior is finalizing must report False,
        not be swallowed by an already-settled guard while the caller believes
        it succeeded — that rewrote genuine failures as CANCELLED."""
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(10), latency_lognorm_sigma=1e-6, workers=1
            )
        ).bind()
        seen = {}

        def behavior(service, sim2, deadline, finalize, begin_at, release_worker, register_cancel):
            register_cancel(lambda: seen.__setitem__("teardown_ran", True))

            def finish():
                register_cancel(None)  # commit
                seen["cancel_returned"] = handle.cancel()
                finalize(False, ms_to_ns(10), DropReason.SERVER_FAILURE)

            sim2.schedule(begin_at + ms_to_ns(10), finish)

        handle = svc.submit_attempt(
            sim, lambda *a: seen.setdefault("reason", a[2]), behavior=behavior
        )
        sim.run()
        self.assertFalse(seen["cancel_returned"])
        self.assertNotIn("teardown_ran", seen)
        self.assertEqual(seen["reason"], DropReason.SERVER_FAILURE)

    def test_deferred_refund_does_not_rewind_another_roots_reservation(self):
        """RetryPolicy.rollback undoes the IMMEDIATELY preceding next_delay.
        LeakyRateLimiterPolicy keeps one slot of undo state, so a cancel-time
        refund would release whichever root reserved most recently."""
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(10),
                latency_lognorm_sigma=1e-6,
                workers=4,
                partial_failures=[PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=1.0)],
            )
        ).bind()
        leaky = LeakyRateLimiterPolicy(max_requests=1, max_attempts=5, period=s_to_ns(1))
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=leaky,
                timeout_controller=StaticTimeoutController(None, ms_to_ns(60_000)),
            ),
            service=svc,
        )
        r1 = client.start_request(sim)
        sim.run(until=ms_to_ns(20))
        client.start_request(sim)  # r2 reserves the next slot
        sim.run(until=ms_to_ns(40))
        reserved = leaky._next_at
        r1.cancel()
        self.assertEqual(leaky._next_at, reserved)

    def test_no_invariant_breaks_under_randomized_cancellation(self):
        """Stress: random fan-outs with propagate_cancellation, asserting the
        worker/queue invariants and summary-vs-CSV agreement across seeds."""
        import tempfile

        import pandas as pd

        for seed in range(12):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                sim = Simulator(seed=seed)

                def node(name, median_ms, p_fail, workers, capacity):
                    failures = (
                        [PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=p_fail)]
                        if p_fail
                        else []
                    )
                    return ServiceNodeRuntime(
                        service=ServiceRuntime(
                            cfg=ServiceConfig(
                                name=name,
                                latency_median=ms_to_ns(median_ms),
                                latency_lognorm_sigma=0.3,
                                workers=workers,
                                queue_capacity=capacity,
                                partial_failures=failures,
                            )
                        ).bind()
                    )

                root_node = node("A", rng.randint(1, 10), 0.0, rng.randint(1, 3), None)
                root_node.propagate_cancellation = True
                root_node.hold_worker_during_fanout = bool(rng.getrandbits(1))
                children = []
                for i in range(rng.randint(2, 4)):
                    child = node(
                        f"C{i}",
                        rng.randint(5, 80),
                        rng.choice([0.0, 0.5, 1.0]),
                        rng.randint(1, 2),
                        rng.choice([None, 2]),
                    )
                    child.propagate_cancellation = True
                    children.append(child)
                    root_node.add_downstream(
                        DownstreamEdge(
                            client=ClientRuntime(
                                cfg=ClientConfig(
                                    retry=FixedBackoffRetryPolicy(
                                        max_attempts=rng.randint(1, 3),
                                        delay=ms_to_ns(rng.choice([0, 20, 200])),
                                    ),
                                    timeout_controller=StaticTimeoutController(
                                        None, ms_to_ns(rng.choice([20, 100, 5000]))
                                    ),
                                ),
                                service=child,
                            ),
                            target=child,
                        )
                    )

                top = ClientRuntime(
                    cfg=ClientConfig(
                        retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(10)),
                        timeout_controller=StaticTimeoutController(None, ms_to_ns(5000)),
                    ),
                    service=root_node,
                )
                wl = Workload(base_rps=60.0, duration_s=2.0, load_spikes=[], rng_seed=seed)
                wl.drive(sim, top.start_request)
                sim.run()

                for n in [root_node, *children]:
                    self.assertEqual(n.service.in_flight, 0, f"{n.cfg.name} leaked a worker")
                    self.assertEqual(n.service.queue_len(), 0, f"{n.cfg.name} leaked queue slots")

                # summary() and export_csv() must agree about every root.
                for edge in root_node.downstreams:
                    summary = edge.client.metrics().summary()
                    if summary.total == 0:
                        continue
                    with tempfile.TemporaryDirectory() as d:
                        path = os.path.join(d, "o.csv")
                        edge.client.metrics().export_csv(path, granularity_s=0.1)
                        df = pd.read_csv(path)
                    self.assertEqual(
                        int(df["failure_root"].sum()) + int(df["success_root"].sum()),
                        summary.total,
                        "roots lost between summary() and export_csv()",
                    )
                    self.assertEqual(int(df["failure_cancelled"].sum()), summary.dropped_cancelled)


class TestWorkerOccupancyReporting(unittest.TestCase):
    """`service_time_ns` means WORKER OCCUPANCY, per the documented contract.

    A tandem node (hold_worker_during_fanout=False) hands its worker back after
    the local stage and finalizes only once the whole subtree returns, so the
    duration its behavior reports spans a downstream wait during which no worker
    of this service was held. Reporting that as occupancy inflates any
    service-time statistic (E[S], utilization) derived from it.
    """

    def _fanout(self, hold: bool, local_ms=10, leaf_ms=200):
        sim = Simulator(seed=1)

        def node(name, median_ms):
            return ServiceNodeRuntime(
                service=ServiceRuntime(
                    cfg=ServiceConfig(
                        name=name,
                        latency_median=ms_to_ns(median_ms),
                        latency_lognorm_sigma=1e-6,
                        workers=2,
                    )
                ).bind()
            )

        a, leaf = node("A", local_ms), node("LEAF", leaf_ms)
        a.hold_worker_during_fanout = hold
        a.add_downstream(
            DownstreamEdge(
                client=ClientRuntime(
                    cfg=ClientConfig(
                        retry=NoRetryPolicy(),
                        timeout_controller=StaticTimeoutController(None, ms_to_ns(5000)),
                    ),
                    service=leaf,
                ),
                target=leaf,
            )
        )
        top = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(None, ms_to_ns(5000)),
            ),
            service=a,
        )
        top.start_request(sim)
        sim.run()
        return top.roots[0].attempts[-1]

    def test_tandem_node_reports_local_stage_only(self):
        attempt = self._fanout(hold=False)
        assert attempt.service_time_ns is not None
        self.assertAlmostEqual(attempt.service_time_ns / ms_to_ns(1), 10.0, delta=0.5)
        # The end-to-end node latency is still available on the interval.
        node_latency = attempt.interval.end - attempt.interval.begin
        self.assertAlmostEqual(node_latency / ms_to_ns(1), 210.0, delta=1.0)

    def test_hold_worker_node_reports_the_whole_subtree(self):
        attempt = self._fanout(hold=True)
        assert attempt.service_time_ns is not None
        self.assertAlmostEqual(attempt.service_time_ns / ms_to_ns(1), 210.0, delta=1.0)


class TestCancellationRecursion(unittest.TestCase):
    """Recursion through a sub-fan-out — claimed in the docs, only ever checked
    ad hoc until now. It is also the property most likely to break under a
    refactor of the fan-out state machine."""

    def _tree(self, mid_propagates: bool):
        """A -> {FAST (fails ~5ms), MID}; MID -> DEEP (200ms)."""
        sim = Simulator(seed=1)

        def node(name, median_ms, p_fail=0.0):
            failures = (
                [PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=p_fail)] if p_fail else []
            )
            return ServiceNodeRuntime(
                service=ServiceRuntime(
                    cfg=ServiceConfig(
                        name=name,
                        latency_median=ms_to_ns(median_ms),
                        latency_lognorm_sigma=1e-6,
                        workers=1,
                        partial_failures=failures,
                    )
                ).bind()
            )

        def edge(target):
            return DownstreamEdge(
                client=ClientRuntime(
                    cfg=ClientConfig(
                        retry=NoRetryPolicy(),
                        timeout_controller=StaticTimeoutController(None, ms_to_ns(5000)),
                    ),
                    service=target,
                ),
                target=target,
            )

        a, fast, mid, deep = node("A", 1), node("FAST", 5, 1.0), node("MID", 1), node("DEEP", 200)
        a.propagate_cancellation = True
        mid.propagate_cancellation = mid_propagates
        deep_edge = edge(deep)
        mid.add_downstream(deep_edge)
        a.add_downstream(edge(fast))
        a.add_downstream(edge(mid))
        top = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(None, ms_to_ns(5000)),
            ),
            service=a,
        )
        top.start_request(sim)
        sim.run()
        return sim, deep_edge.client, deep

    def test_cancel_unwinds_two_levels(self):
        sim, deep_client, deep = self._tree(mid_propagates=True)
        self.assertEqual(deep_client.metrics().summary().dropped_cancelled, 1)
        self.assertLess(sim.timestep, ms_to_ns(100))  # not DEEP's full 200 ms
        self.assertEqual(deep.service.in_flight, 0)

    def test_recursion_stops_at_a_node_that_does_not_propagate(self):
        """Documented semantics: each node models its own architecture, so the
        first descendant with the flag off keeps its own subtree running."""
        sim, deep_client, deep = self._tree(mid_propagates=False)
        self.assertEqual(deep_client.metrics().summary().dropped_cancelled, 0)
        self.assertGreaterEqual(sim.timestep, ms_to_ns(200))  # DEEP ran to completion
        self.assertEqual(deep.service.in_flight, 0)


class TestOccupancyCensoring(unittest.TestCase):
    """Whether an occupancy sample is right-censored is a property of the
    SAMPLE, not of the attempt's outcome.

    A tandem node hands its worker back before the downstream wait, so a later
    deadline censors the ATTEMPT while the local stage it measured ran to
    completion. Inferring the flag from drop_reason marks complete observations
    as lower bounds, which inflates any occupancy estimate built from the samples.
    """

    def _fanout_attempt(self, hold: bool):
        sim = Simulator(seed=1)

        def node(name, median_ms, workers=4):
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

        a, leaf = node("A", 10), node("LEAF", 200)
        a.hold_worker_during_fanout = hold
        a.add_downstream(
            DownstreamEdge(
                client=ClientRuntime(
                    cfg=ClientConfig(
                        retry=NoRetryPolicy(),
                        timeout_controller=StaticTimeoutController(None, ms_to_ns(5000)),
                    ),
                    service=leaf,
                ),
                target=leaf,
            )
        )
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                # Cuts the attempt long after A's local stage has finished.
                timeout_controller=StaticTimeoutController(None, ms_to_ns(150)),
            ),
            service=a,
        )
        client.start_request(sim)
        sim.run()
        return client.roots[0].attempts[-1]

    def test_released_worker_yields_a_complete_occupancy_sample(self):
        attempt = self._fanout_attempt(hold=False)
        self.assertEqual(attempt.drop_reason, DropReason.DEADLINE)  # attempt WAS cut
        assert attempt.service_time_ns is not None
        self.assertAlmostEqual(attempt.service_time_ns / ms_to_ns(1), 10.0, delta=0.5)
        self.assertFalse(attempt.service_time_censored)

    def test_held_worker_yields_a_censored_occupancy_sample(self):
        attempt = self._fanout_attempt(hold=True)
        self.assertEqual(attempt.drop_reason, DropReason.DEADLINE)
        self.assertTrue(attempt.service_time_censored)

    def test_plain_service_deadline_cut_is_censored(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(200), latency_lognorm_sigma=1e-6, workers=1
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(None, ms_to_ns(50)),
            ),
            service=svc,
        )
        client.start_request(sim)
        sim.run()
        attempt = client.roots[0].attempts[-1]
        self.assertTrue(attempt.service_time_censored)


if __name__ == "__main__":
    unittest.main()
