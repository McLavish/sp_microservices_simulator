import unittest
from dataclasses import fields

from msim.caller import ClientConfig, ClientRuntime
from msim.core import DropReason, ms_to_ns
from msim.retry_policies import FixedBackoffRetryPolicy
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController


class TestServiceDeadline(unittest.TestCase):
    def test_submit_attempt_expired_does_not_raise(self):
        sim = Simulator(seed=1)
        cfg = ServiceConfig(
            name="svc",
            latency_median=1,
            latency_lognorm_sigma=0.1,
            workers=1,
            queue_capacity=None,
        )
        svc = ServiceRuntime(cfg=cfg).bind()
        results = []

        def on_done(success, _svc_time, drop_reason, _queue_size, _censored=False):
            results.append((success, drop_reason))

        svc.submit_attempt(sim, on_done, attempt_deadline=0)

        self.assertEqual(results, [(False, DropReason.DEADLINE)])


class TestServiceRuntimeBind(unittest.TestCase):
    def test_bind_restores_a_freshly_constructed_runtime(self):
        """`bind()` must equal construction for EVERY state field.

        It used to restate the field defaults by hand, so a state field added
        to the declarations and forgotten in the reset would have gone
        unnoticed: every call site is `ServiceRuntime(cfg=...).bind()` on a
        fresh instance, which never exercises the reset at all. Compared
        against an independently constructed runtime rather than against the
        declarations, so a hand-written reset cannot satisfy it by accident.
        """
        cfg = ServiceConfig(
            name="svc", latency_median=ms_to_ns(10), latency_lognorm_sigma=1e-6, workers=1
        )
        sim = Simulator(seed=1)
        svc = ServiceRuntime(cfg=cfg).bind()
        for _ in range(3):
            svc.submit_attempt(sim, lambda *_a: None)  # 1 in service, 2 queued

        svc.bind()

        fresh = ServiceRuntime(cfg=cfg)
        state = [f.name for f in fields(ServiceRuntime) if f.name != "cfg"]
        self.assertEqual(
            [getattr(svc, name) for name in state],
            [getattr(fresh, name) for name in state],
        )
        self.assertIs(svc.cfg, cfg)  # inputs are left alone


class TestServiceTimeRecording(unittest.TestCase):
    def test_attempt_interval_includes_queue_wait_but_service_time_does_not(self):
        sim = Simulator(seed=1)
        cfg = ServiceConfig(
            name="svc", latency_median=ms_to_ns(20), latency_lognorm_sigma=1e-6, workers=1
        )
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=1, delay=0),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=None
                ),
            ),
            service=ServiceRuntime(cfg=cfg).bind(),
        )
        client.start_request(sim)
        client.start_request(sim)  # queues behind the first (workers=1)
        sim.run()
        attempts = sorted(
            (a for root in client.roots for a in root.attempts),
            key=lambda a: a.interval.end,
        )
        first, second = attempts
        assert first.service_time_ns is not None and second.service_time_ns is not None
        tol = ms_to_ns(2)
        self.assertLess(abs(first.service_time_ns - ms_to_ns(20)), tol)
        self.assertLess(abs(second.service_time_ns - ms_to_ns(20)), tol)
        # The second attempt's measured interval contains ~20 ms of queue wait.
        self.assertGreater(second.interval.end - second.interval.begin, ms_to_ns(35))

    def test_timeout_records_occupancy_not_latent_service_time(self):
        # A 100 ms latent draw cut by a 10 ms attempt deadline frees the worker
        # at 10 ms; the recorded service time must be the ~10 ms of actual
        # occupancy, not the latent 100 ms (which would corrupt the utilization
        # and service-time moments derived from it downstream).
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(100), latency_lognorm_sigma=1e-6, workers=1
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=1, delay=0),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(10)
                ),
            ),
            service=svc,
        )
        client.start_request(sim)
        sim.run()
        (attempt,) = [a for root in client.roots for a in root.attempts]
        assert attempt.service_time_ns is not None
        self.assertEqual(attempt.drop_reason, DropReason.DEADLINE)
        self.assertLess(attempt.service_time_ns, ms_to_ns(11))
        self.assertEqual(svc.in_flight, 0)  # worker freed at the deadline


class TestQueuedAttemptDeadline(unittest.TestCase):
    """The attempt deadline is the CALLER's clock, also while the attempt waits.

    Before this was pinned, a queued attempt whose deadline passed was reported
    only when a worker happened to free and dequeue it, so a client with a 50 ms
    timeout could observe a 1000 ms attempt and every retry it issued fired that
    late. In `examples.multichain_retry_storm` half of the deadline drops at the
    shared bottleneck were reported up to 50 ms late on a 60 ms timeout. The
    server side is unchanged: it discovers the expiry at dequeue and discards the
    entry then, so an expired entry keeps shedding load until the server reaches
    it (a server does not scan its backlog).
    """

    def _svc(self, capacity: int | None = None) -> ServiceRuntime:
        return ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(1000),
                latency_lognorm_sigma=1e-6,
                workers=1,
                queue_capacity=capacity,
            )
        ).bind()

    def _pin_worker(self, sim: Simulator, svc: ServiceRuntime) -> None:
        """Occupy the only worker for ~1000 ms with an attempt that has no deadline."""
        svc.submit_attempt(sim, lambda *_a: None)

    def test_queued_attempt_is_reported_at_its_deadline_not_at_dequeue(self):
        sim = Simulator(seed=1)
        svc = self._svc()
        self._pin_worker(sim, svc)
        calls = []
        svc.submit_attempt(
            sim, lambda *a: calls.append((sim.timestep, a)), attempt_deadline=ms_to_ns(50)
        )

        sim.run(until=ms_to_ns(49))
        self.assertEqual(calls, [])
        sim.run(until=ms_to_ns(50))
        # Queue depth EXCLUDING the reporting attempt, as on every other path.
        self.assertEqual(calls, [(ms_to_ns(50), (False, 0, DropReason.DEADLINE, 0, False))])
        sim.run()  # the server dequeues the expired entry at ~1000 ms
        self.assertEqual(len(calls), 1)  # ...and does not report it again
        self.assertEqual(svc.queue_len(), 0)
        self.assertEqual(svc.in_flight, 0)

    def test_expired_entry_holds_its_queue_slot_until_the_server_reaches_it(self):
        sim = Simulator(seed=1)
        svc = self._svc(capacity=1)
        self._pin_worker(sim, svc)
        svc.submit_attempt(sim, lambda *_a: None, attempt_deadline=ms_to_ns(50))  # queued

        sim.run(until=ms_to_ns(100))  # expired 50 ms ago; the server has not looked yet
        self.assertEqual(svc.queue_len(), 1)
        seen = []
        svc.submit_attempt(sim, lambda *a: seen.append(a[2]))
        self.assertEqual(seen, [DropReason.QUEUE_FULL])  # the zombie still sheds load

        sim.run()
        self.assertEqual(svc.queue_len(), 0)  # discarded at dequeue

    def test_worker_freeing_exactly_at_the_deadline_reports_once(self):
        """Same instant, the worker's completion has the earlier seq: the
        dequeue path (`_begin_service`) catches the expiry at the deadline and
        the queued attempt's own deadline event is disarmed, so exactly one
        report is delivered."""
        sim = Simulator(seed=1)
        svc = self._svc()
        svc.submit_attempt(sim, lambda *_a: None, attempt_deadline=ms_to_ns(50))  # cut at 50 ms
        calls = []
        svc.submit_attempt(
            sim, lambda *a: calls.append((sim.timestep, a[2], a[3])), attempt_deadline=ms_to_ns(50)
        )
        sim.run()
        self.assertEqual(calls, [(ms_to_ns(50), DropReason.DEADLINE, 0)])
        self.assertEqual(svc.in_flight, 0)
        self.assertEqual(svc.queue_len(), 0)

    def test_dequeued_before_the_deadline_is_served_and_cut_in_service(self):
        sim = Simulator(seed=1)
        svc = self._svc()
        svc.submit_attempt(sim, lambda *_a: None, attempt_deadline=ms_to_ns(30))  # frees at 30 ms
        calls = []
        svc.submit_attempt(
            sim, lambda *a: calls.append((sim.timestep, a)), attempt_deadline=ms_to_ns(50)
        )
        sim.run()
        ((at, (success, occupancy, drop, _qsize, censored)),) = calls
        self.assertEqual(at, ms_to_ns(50))
        self.assertFalse(success)
        self.assertEqual(drop, DropReason.DEADLINE)
        self.assertEqual(occupancy, ms_to_ns(20))  # served from 30 ms, cut at 50 ms
        self.assertTrue(censored)

    def test_cancel_before_the_deadline_disarms_it(self):
        sim = Simulator(seed=1)
        svc = self._svc()
        self._pin_worker(sim, svc)
        calls = []
        handle = svc.submit_attempt(
            sim, lambda *a: calls.append(a[2]), attempt_deadline=ms_to_ns(50)
        )
        sim.run(until=ms_to_ns(10))
        self.assertTrue(handle.cancel())
        sim.run()
        self.assertEqual(calls, [DropReason.CANCELLED])

    def test_cancel_after_the_deadline_is_a_truthful_no_op(self):
        sim = Simulator(seed=1)
        svc = self._svc()
        self._pin_worker(sim, svc)
        calls = []
        handle = svc.submit_attempt(
            sim, lambda *a: calls.append(a[2]), attempt_deadline=ms_to_ns(50)
        )
        sim.run(until=ms_to_ns(60))
        self.assertFalse(handle.cancel())
        sim.run()
        self.assertEqual(calls, [DropReason.DEADLINE])

    def test_client_retries_follow_the_deadline_not_the_dequeue(self):
        sim = Simulator(seed=1)
        svc = self._svc()
        self._pin_worker(sim, svc)
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=3, delay=ms_to_ns(10)),
                timeout_controller=StaticTimeoutController(None, ms_to_ns(50)),
            ),
            service=svc,
        )
        client.start_request(sim)
        sim.run()
        (root,) = client.roots
        self.assertEqual(
            [(a.interval.begin, a.interval.end) for a in root.attempts],
            [(0, ms_to_ns(50)), (ms_to_ns(60), ms_to_ns(110)), (ms_to_ns(120), ms_to_ns(170))],
        )
        self.assertEqual(root.completed_at, ms_to_ns(170))
        self.assertTrue(all(a.drop_reason is DropReason.DEADLINE for a in root.attempts))


if __name__ == "__main__":
    unittest.main()
