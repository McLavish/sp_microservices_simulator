"""Black-box tests for the per-link transport delay (`transport.py`).

Written against the public API only — `TransportDelay(median, sigma)` with
`sample(sim)`, and `TransportLink(target, delay)`, which satisfies
`caller.ServiceLike` by forwarding `cfg` and implementing `submit_attempt`.

The timings are pinned exactly wherever the spec pins them, which a
deterministic link makes possible: `sigma == 0` means "exactly `median`, no RNG
draw", so a link adds exactly `median` to each leg without perturbing any other
seeded draw in the run. That is what lets these tests say
`interval == 2 * median + service_time_ns` rather than "roughly".

The two populations a link must keep apart:

* `Request.service_time_ns` is WORKER OCCUPANCY and excludes transport (it is
  the service-time stream, not the round-trip one);
* `Request.interval` is end-to-end and spans both legs.
"""

import unittest
from collections.abc import Callable
from typing import cast

from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import (
    DropReason,
    RootOutcome,
    TimeDuration,
    TimeInterval,
    TimePoint,
    classify_root,
    ms_to_ns,
    root_latency,
    s_to_ns,
)
from msim.faults import PartialFailure
from msim.retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy, RetryPolicy
from msim.service import AttemptControl, AttemptDoneCallback, ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.transport import TransportDelay, TransportLink


def _svc(
    name: str,
    median_ms: float,
    workers: int = 1,
    queue_capacity: int | None = None,
    p_fail: float = 0.0,
) -> ServiceRuntime:
    """A near-deterministic service: sigma=1e-6 means the draw is ~= the median."""
    failures = [PartialFailure(TimeInterval(0, s_to_ns(100)), p_fail=p_fail)] if p_fail else []
    return ServiceRuntime(
        cfg=ServiceConfig(
            name=name,
            latency_median=ms_to_ns(median_ms),
            latency_lognorm_sigma=1e-6,
            workers=workers,
            queue_capacity=queue_capacity,
            partial_failures=failures,
        )
    ).bind()


def _node(name: str, median_ms: float, p_fail: float = 0.0, propagate: bool = False):
    return ServiceNodeRuntime(
        service=_svc(name, median_ms, p_fail=p_fail),
        propagate_cancellation=propagate,
    )


def _client(
    service,
    retry: RetryPolicy | None = None,
    attempt_timeout: TimeDuration | None = None,
    global_timeout: TimeDuration | None = None,
) -> ClientRuntime:
    return ClientRuntime(
        cfg=ClientConfig(
            retry=NoRetryPolicy() if retry is None else retry,
            timeout_controller=StaticTimeoutController(
                global_timeout=global_timeout, attempt_timeout=attempt_timeout
            ),
        ),
        service=service,
    )


class _Recorder:
    """Records every `on_done` a link delivers, with the instant it arrived.

    `calls` holds the 5-tuples the callback contract specifies; `times` holds
    `sim.timestep` at delivery, which is what distinguishes a synchronous
    delivery from one that took the return leg.
    """

    def __init__(self, sim: Simulator) -> None:
        self.sim = sim
        self.calls: list[tuple[bool, TimeDuration, DropReason, int | None, bool]] = []
        self.times: list[TimePoint] = []

    def __call__(
        self,
        success: bool,
        svc_time: TimeDuration,
        drop: DropReason,
        qsize: int | None,
        censored: bool,
    ) -> None:
        self.calls.append((success, svc_time, drop, qsize, censored))
        self.times.append(self.sim.timestep)


class _StubControl(AttemptControl):
    """An `AttemptControl` whose answer to `cancel()` the test chooses.

    `ServiceRuntime` always tears an attempt down while it is queued or in
    service, so the branch where the link's AT_SERVICE cancel has to relay a
    *refusal* is unreachable with a stock target. This makes it reachable.
    """

    __slots__ = ("_teardown",)

    def __init__(self, teardown: Callable[[], None] | None = None) -> None:
        super().__init__()
        self._teardown = teardown

    def cancel(self) -> bool:
        teardown = self._teardown
        if teardown is None:
            return False  # already committed to an outcome: say so truthfully
        self._teardown = None
        teardown()
        return True


class _StubTarget:
    """A minimal `ServiceLike` target with no queue, so each leg is observable.

    A real `ServiceRuntime` hides the instant an attempt reaches it: the only
    thing a caller ever sees is the sum `d_f + occupancy + d_r`, which an
    implementation that draws one delay and spends it twice reproduces exactly.
    This target records its own arrival and report instants, splitting that sum
    into the two legs the link is supposed to draw independently.

    It also reports whatever it is told to, which a real service cannot: a
    non-zero queue depth, a censored occupancy, a second report, or (via
    `cancel_report`) a genuine non-CANCELLED outcome delivered synchronously
    from inside its own teardown — "I already have the answer, take it now".
    And it deliberately ignores `attempt_deadline` — a real service would cut
    its own work short at the deadline, which would confound what is under test
    here: what the LINK does with a report that is already too late to deliver.
    """

    def __init__(
        self,
        name: str = "stub",
        service_ns: TimeDuration = 0,
        report: tuple[bool, TimeDuration, DropReason, int | None, bool] | None = None,
        cancellable: bool = True,
        rng_draws_before_report: int = 0,
        reports_twice: bool = False,
        cancel_report: tuple[bool, TimeDuration, DropReason, int | None, bool] | None = None,
    ) -> None:
        self.cfg: ServiceConfig = ServiceConfig(
            name=name, latency_median=1, latency_lognorm_sigma=0.1, workers=1
        )
        self.service_ns = service_ns
        self.report_tuple = (
            (True, service_ns, DropReason.NONE, 0, False) if report is None else report
        )
        self.cancellable = cancellable
        self.rng_draws_before_report = rng_draws_before_report
        self.reports_twice = reports_twice
        self.cancel_report = cancel_report
        self.arrivals: list[TimePoint] = []
        self.reports: list[TimePoint] = []

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: AttemptDoneCallback,
        attempt_deadline: TimePoint | None = None,
    ) -> AttemptControl:
        self.arrivals.append(sim.timestep)
        arrived_at = sim.timestep

        def report() -> None:
            for _ in range(self.rng_draws_before_report):
                sim.rng().random()
            self.reports.append(sim.timestep)
            on_done(*self.report_tuple)
            if self.reports_twice:
                on_done(*self.report_tuple)

        if self.service_ns == 0:
            # Reports from INSIDE submit_attempt, before the link has been
            # handed any control at all.
            report()
            return _StubControl()

        event = sim.schedule(arrived_at + self.service_ns, report)
        if not self.cancellable:
            return _StubControl()

        def teardown() -> None:
            event.cancelled = True
            if self.cancel_report is not None:
                # Tears down, then hands back a REAL outcome instead of a
                # CANCELLED one — still a truthful `True` from `cancel()`.
                on_done(*self.cancel_report)
                return
            on_done(False, sim.timestep - arrived_at, DropReason.CANCELLED, 0, True)

        return _StubControl(teardown)


class TestTransportDelaySampling(unittest.TestCase):
    def test_zero_sigma_returns_exactly_the_median_and_draws_no_rng(self):
        """A deterministic link must be expressible, so sigma is NOT floored.

        `service.MIN_LOGNORM_SIGMA` would turn sigma=0 into a 1e-6 lognormal:
        the sample would then be median +- a few ns AND would consume a draw
        from `sim.rng()`, shifting every later seeded draw in the run. Both
        halves are checked — the RNG state before and after must be identical.
        """
        delay = TransportDelay(median=ms_to_ns(2))
        self.assertEqual(delay.sigma, 0.0)  # survives __post_init__ unfloored

        sim = Simulator(seed=1)
        before = sim.rng().getstate()
        self.assertEqual(delay.sample(sim), ms_to_ns(2))
        self.assertEqual(delay.sample(sim), ms_to_ns(2))
        self.assertEqual(sim.rng().getstate(), before)

    def test_positive_sigma_draws_from_the_simulator_rng(self):
        delay = TransportDelay(median=ms_to_ns(2), sigma=0.5)
        sim = Simulator(seed=1)
        before = sim.rng().getstate()
        first = delay.sample(sim)
        self.assertNotEqual(sim.rng().getstate(), before)

        second = delay.sample(sim)
        self.assertNotEqual(first, second)  # a real draw, not a cached median
        self.assertGreaterEqual(first, 0)
        self.assertGreaterEqual(second, 0)

    def test_samples_are_reproducible_under_the_same_seed(self):
        """Every draw goes through `Simulator.rng()`, so the seed reproduces it."""
        delay = TransportDelay(median=ms_to_ns(2), sigma=0.5)

        def draws(seed: int) -> list[int]:
            sim = Simulator(seed=seed)
            return [delay.sample(sim) for _ in range(5)]

        self.assertEqual(draws(1), draws(1))
        self.assertNotEqual(draws(1), draws(2))


class TestTransportDelayValidation(unittest.TestCase):
    def test_median_below_one_nanosecond_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            TransportDelay(median=0)
        self.assertIn("0", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            TransportDelay(median=-7)
        self.assertIn("-7", str(cm.exception))

    def test_non_finite_median_is_rejected(self):
        """`nan < 1` is False, so a range check alone lets NaN straight through."""
        with self.assertRaises(ValueError) as cm:
            TransportDelay(median=cast(TimeDuration, float("nan")))
        self.assertIn("nan", str(cm.exception).lower())

        with self.assertRaises(ValueError) as cm:
            TransportDelay(median=cast(TimeDuration, float("inf")))
        self.assertIn("inf", str(cm.exception).lower())

    def test_negative_sigma_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            TransportDelay(median=ms_to_ns(1), sigma=-0.5)
        self.assertIn("-0.5", str(cm.exception))

    def test_non_finite_sigma_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            TransportDelay(median=ms_to_ns(1), sigma=float("nan"))
        self.assertIn("nan", str(cm.exception).lower())

        with self.assertRaises(ValueError) as cm:
            TransportDelay(median=ms_to_ns(1), sigma=float("inf"))
        self.assertIn("inf", str(cm.exception).lower())

    def test_a_tiny_positive_sigma_is_not_floored_to_deterministic(self):
        """`MIN_LOGNORM_SIGMA` is `ServiceConfig`'s floor, not this one.

        `ServiceConfig.__post_init__` rewrites sigma=1e-9 to 1e-6 because every
        pinned seeded result was produced that way. A link has no such history,
        and the floor would make `sigma == 0` (no draw) and `sigma == 1e-9`
        (a draw) collapse into one behaviour, which is exactly the distinction
        the zero-sigma rule rests on.
        """
        delay = TransportDelay(median=ms_to_ns(2), sigma=1e-9)
        self.assertEqual(delay.sigma, 1e-9)  # not raised to MIN_LOGNORM_SIGMA

        sim = Simulator(seed=1)
        before = sim.rng().getstate()
        sample = delay.sample(sim)
        self.assertNotEqual(sim.rng().getstate(), before)  # a positive sigma DRAWS
        self.assertLess(abs(sample - ms_to_ns(2)), 100)  # ...around the median

    def test_minimum_median_and_zero_sigma_are_accepted(self):
        delay = TransportDelay(median=1)
        self.assertEqual(delay.median, 1)
        self.assertEqual(delay.sigma, 0.0)
        self.assertEqual(delay.sample(Simulator(seed=1)), 1)


class TestRoundTripTiming(unittest.TestCase):
    def test_every_attempt_pays_the_delay_on_both_legs(self):
        """Retries cross the link too: they are issued by `ClientRuntime`, not
        by the fan-out loop, so a delay applied per root would miss them."""
        median = ms_to_ns(1)
        sim = Simulator(seed=1)
        svc = _svc("svc", 10, p_fail=1.0)
        client = _client(
            TransportLink(svc, TransportDelay(median=median)),
            retry=FixedBackoffRetryPolicy(max_attempts=3, delay=ms_to_ns(5)),
        )
        client.start_request(sim)
        sim.run()

        (root,) = client.roots
        self.assertEqual(len(root.attempts), 3)
        self.assertEqual(svc.admitted_total, 3)
        for attempt in root.attempts:
            occupancy = attempt.service_time_ns
            assert occupancy is not None  # every attempt reached the server
            self.assertEqual(attempt.drop_reason, DropReason.SERVER_FAILURE)
            self.assertEqual(attempt.interval.end - attempt.interval.begin, 2 * median + occupancy)

    def test_request_is_booked_under_the_target_service_name(self):
        sim = Simulator(seed=1)
        svc = _svc("svc-target", 5)
        link = TransportLink(svc, TransportDelay(median=ms_to_ns(1)))
        self.assertIs(link.cfg, svc.cfg)

        client = _client(link)
        client.start_request(sim)
        sim.run()

        (attempt,) = [a for root in client.roots for a in root.attempts]
        self.assertEqual(attempt.service, "svc-target")

    def test_service_time_excludes_transport_while_the_interval_spans_both_legs(self):
        """The same seeded run with and without the link, compared attempt to
        attempt.

        A sigma=0 link draws nothing, so the service-time sample is bit-identical
        in both runs: `service_time_ns` must be UNCHANGED (it is worker
        occupancy, which the wire never touches) while the interval grows by
        exactly one round trip.
        """
        median = ms_to_ns(4)

        sim_plain = Simulator(seed=3)
        plain_svc = _svc("svc", 10)
        plain = _client(plain_svc)
        plain.start_request(sim_plain)
        sim_plain.run()
        (bare,) = [a for root in plain.roots for a in root.attempts]

        sim_link = Simulator(seed=3)
        linked_svc = _svc("svc", 10)
        linked = _client(TransportLink(linked_svc, TransportDelay(median=median)))
        linked.start_request(sim_link)
        sim_link.run()
        (over_link,) = [a for root in linked.roots for a in root.attempts]

        self.assertTrue(bare.success)
        self.assertTrue(over_link.success)
        self.assertEqual(over_link.service_time_ns, bare.service_time_ns)
        self.assertEqual(over_link.interval.begin, bare.interval.begin)
        self.assertEqual(over_link.interval.end, bare.interval.end + 2 * median)
        self.assertEqual(over_link.queue_size_at_end, bare.queue_size_at_end)


class TestDeadlineSemantics(unittest.TestCase):
    def test_forward_transit_expiry_never_reaches_the_service(self):
        """Expired work is shed before it consumes anything, so the target is
        never called and no queue sample is fabricated."""
        sim = Simulator(seed=1)
        svc = _svc("svc", 1)
        client = _client(
            TransportLink(svc, TransportDelay(median=ms_to_ns(10))),
            attempt_timeout=ms_to_ns(5),
        )
        client.start_request(sim)
        sim.run()

        (root,) = client.roots
        (attempt,) = root.attempts
        self.assertEqual(attempt.drop_reason, DropReason.DEADLINE)
        self.assertFalse(attempt.success)
        self.assertEqual(attempt.deadline, ms_to_ns(5))
        self.assertEqual(attempt.interval.end, ms_to_ns(5))  # the caller's clock, not 10 ms
        self.assertIsNone(attempt.queue_size_at_end)  # never observed a queue
        self.assertIsNone(attempt.service_time_ns)
        self.assertFalse(attempt.service_time_censored)
        self.assertEqual(svc.admitted_total, 0)
        self.assertEqual(svc.in_flight, 0)
        self.assertEqual(client.metrics().summary().dropped_deadline, 1)

    def test_forward_leg_landing_exactly_on_the_deadline_expires_in_transit(self):
        """`arrive_at == deadline` is an expiry — the half-open rule on the wire.

        A packet that lands exactly at the deadline is already too late to
        produce a success, so the target must not be called at all. The tell is
        the queue size: had the link handed the attempt over, `ServiceRuntime`
        would have rejected it itself and reported a queue sample of `0` where
        the spec requires `None`.
        """
        median = ms_to_ns(4)
        sim = Simulator(seed=1)
        svc = _svc("svc", 1)
        link = TransportLink(svc, TransportDelay(median=median))
        rec = _Recorder(sim)
        control = link.submit_attempt(sim, rec, attempt_deadline=median)

        sim.run()
        self.assertEqual(rec.calls, [(False, 0, DropReason.DEADLINE, None, False)])
        self.assertEqual(rec.times, [median])
        self.assertEqual(svc.admitted_total, 0)
        self.assertEqual(svc.in_flight, 0)
        self.assertFalse(control.cancel())

    def test_the_link_itself_clamps_a_late_response_and_the_boundary_is_a_drop(self):
        """What the LINK delivers, not what the client makes of it.

        `ClientRuntime._on_attempt_done` re-applies the same half-open rule at
        the same instant, so an end-to-end assertion cannot tell a link that
        clamps from one that hands over a late success: both are recorded as a
        DEADLINE at the deadline. Driving `submit_attempt` directly is what pins
        the clamp — the tuple the link delivers must already say
        `(False, ..., DEADLINE, ...)`, with the occupancy the target reported
        kept intact because the work really was done.
        """
        median, service = ms_to_ns(3), ms_to_ns(5)
        round_trip = 2 * median + service  # 11 ms out-and-back

        cases = [
            ("response is late", round_trip - ms_to_ns(1), True),
            ("response lands exactly on the deadline", round_trip, True),
            ("response is in time", round_trip + ms_to_ns(1), False),
        ]
        for label, deadline, expect_drop in cases:
            with self.subTest(label):
                sim = Simulator(seed=1)
                stub = _StubTarget(service_ns=service)
                link = TransportLink(stub, TransportDelay(median=median))
                rec = _Recorder(sim)
                link.submit_attempt(sim, rec, attempt_deadline=deadline)
                sim.run()

                self.assertEqual(stub.arrivals, [median])  # the forward leg ran
                if expect_drop:
                    self.assertEqual(rec.calls, [(False, service, DropReason.DEADLINE, 0, False)])
                    self.assertEqual(rec.times, [deadline])
                else:
                    self.assertEqual(rec.calls, [(True, service, DropReason.NONE, 0, False)])
                    self.assertEqual(rec.times, [round_trip])

    def test_a_deadline_already_past_at_submit_expires_without_rewinding_time(self):
        """Submitting into an already-expired deadline, the other `max` case.

        `ServiceRuntime.submit_attempt` has the same guard (it sheds at once
        when `sim.timestep >= attempt_deadline`), so a link in front of it must
        not be the thing that blows up instead: firing "at the deadline" would
        mean scheduling into the past, which `Simulator.schedule` refuses.
        """
        sim = Simulator(seed=1)
        sim.run(until=ms_to_ns(10))  # nothing scheduled: this just moves the clock
        self.assertEqual(sim.timestep, ms_to_ns(10))

        svc = _svc("svc", 1)
        link = TransportLink(svc, TransportDelay(median=ms_to_ns(1)))
        rec = _Recorder(sim)
        link.submit_attempt(sim, rec, attempt_deadline=ms_to_ns(5))

        sim.run()
        self.assertEqual(rec.calls, [(False, 0, DropReason.DEADLINE, None, False)])
        self.assertEqual(rec.times, [ms_to_ns(10)])  # now, not the 5 ms behind us
        self.assertEqual(svc.admitted_total, 0)

    def test_a_report_that_arrives_after_the_deadline_is_delivered_at_once(self):
        """The clamp is `max(deadline, now)`, and the `max` is a live path.

        A `ServiceRuntime` deadline drop at dequeue fires whenever a worker
        frees *after* the deadline, so the link is genuinely handed reports from
        the past-the-deadline side. Clamping to the deadline itself would then
        mean scheduling into the past, which `Simulator.schedule` refuses with a
        `ValueError` — the run would abort rather than mis-report.

        Here one attempt holds the single worker until 20 ms while a second
        crosses a 1 ms link with a 5 ms deadline: it queues at 1 ms and is shed
        at dequeue at 20 ms, long past its deadline.
        """
        sim = Simulator(seed=1)
        svc = _svc("svc", 20, workers=1)
        hog = _Recorder(sim)
        svc.submit_attempt(sim, hog)  # holds the only worker until ~20 ms

        link = TransportLink(svc, TransportDelay(median=ms_to_ns(1)))
        rec = _Recorder(sim)
        link.submit_attempt(sim, rec, attempt_deadline=ms_to_ns(5))

        sim.run()
        self.assertEqual(len(rec.calls), 1)
        success, occupancy, drop, qsize, censored = rec.calls[0]
        self.assertFalse(success)
        self.assertEqual(drop, DropReason.DEADLINE)
        self.assertEqual(occupancy, 0)  # never entered service
        self.assertEqual(qsize, 0)
        self.assertFalse(censored)
        # Delivered when the drop was reported, NOT retroactively at 5 ms.
        self.assertGreater(rec.times[0], ms_to_ns(5))
        self.assertLess(abs(rec.times[0] - ms_to_ns(20)), ms_to_ns(0.01))
        self.assertEqual(svc.admitted_total, 2)  # it did reach the queue

    def test_return_leg_expiry_is_booked_at_the_deadline_with_full_occupancy(self):
        """The work was really done; only the response was late.

        Service starts at 3 ms and finishes at ~8 ms (inside the 10 ms
        deadline), so the response would land at ~11 ms. The attempt is a
        DEADLINE booked at exactly 10 ms, but the occupancy the server reported
        stands and is NOT censored — the server was never cut short.
        """
        median = ms_to_ns(3)
        deadline = ms_to_ns(10)
        sim = Simulator(seed=1)
        svc = _svc("svc", 5)
        client = _client(
            TransportLink(svc, TransportDelay(median=median)), attempt_timeout=deadline
        )
        client.start_request(sim)
        sim.run()

        (attempt,) = [a for root in client.roots for a in root.attempts]
        occupancy = attempt.service_time_ns
        assert occupancy is not None
        self.assertEqual(attempt.drop_reason, DropReason.DEADLINE)
        self.assertFalse(attempt.success)
        self.assertEqual(attempt.interval.end, deadline)
        self.assertFalse(attempt.service_time_censored)
        self.assertEqual(attempt.queue_size_at_end, 0)
        self.assertEqual(svc.admitted_total, 1)
        # The full service really ran (~5 ms), and the response really was late.
        self.assertLess(abs(occupancy - ms_to_ns(5)), ms_to_ns(0.01))
        self.assertGreater(median + occupancy + median, deadline)

    def test_response_landing_exactly_on_the_deadline_is_a_drop(self):
        """Half-open rule on the transport leg: `deliver_at == deadline` fails.

        The boundary is constructed exactly rather than guessed: a first run
        with a slack deadline measures the occupancy the seed produces, and the
        second run sets the attempt timeout to `2 * median + occupancy` so the
        response lands on the deadline to the nanosecond.
        """
        median = ms_to_ns(3)

        probe_sim = Simulator(seed=7)
        probe_client = _client(
            TransportLink(_svc("svc", 5), TransportDelay(median=median)),
            attempt_timeout=ms_to_ns(1000),
        )
        probe_client.start_request(probe_sim)
        probe_sim.run()
        (probe,) = [a for root in probe_client.roots for a in root.attempts]
        occupancy = probe.service_time_ns
        assert occupancy is not None
        self.assertTrue(probe.success)
        self.assertEqual(probe.interval.end, 2 * median + occupancy)

        deadline = 2 * median + occupancy
        sim = Simulator(seed=7)
        svc = _svc("svc", 5)
        client = _client(
            TransportLink(svc, TransportDelay(median=median)), attempt_timeout=deadline
        )
        client.start_request(sim)
        sim.run()

        (attempt,) = [a for root in client.roots for a in root.attempts]
        self.assertEqual(attempt.deadline, deadline)
        self.assertEqual(attempt.drop_reason, DropReason.DEADLINE)
        self.assertFalse(attempt.success)
        self.assertEqual(attempt.interval.end, deadline)
        self.assertEqual(attempt.service_time_ns, occupancy)  # identical seeded draw
        self.assertFalse(attempt.service_time_censored)
        self.assertEqual(svc.admitted_total, 1)


class TestReportFidelity(unittest.TestCase):
    """The wire relays the target's report; it does not author one."""

    def test_the_return_leg_carries_the_targets_report_unchanged(self):
        """Queue depth and the censoring flag are the target's observations.

        Both carry statistical semantics downstream — `queue_size_at_end` feeds
        `queue_avg_at_attempt_end`, and `service_time_censored` marks the
        occupancy as a LOWER BOUND rather than an exact observation —
        so a link that flattened either to its default would corrupt them
        silently. Every stock service in these tests happens to report an empty
        queue and an uncensored occupancy, which is precisely why this one is
        driven by a target that reports neither.
        """
        median, service = ms_to_ns(3), ms_to_ns(5)
        round_trip = 2 * median + service
        reported = (False, service, DropReason.SERVER_FAILURE, 7, True)

        with self.subTest("delivered in time, verbatim"):
            sim = Simulator(seed=1)
            stub = _StubTarget(service_ns=service, report=reported)
            link = TransportLink(stub, TransportDelay(median=median))
            rec = _Recorder(sim)
            link.submit_attempt(sim, rec, attempt_deadline=round_trip + ms_to_ns(1))
            sim.run()
            self.assertEqual(rec.calls, [reported])
            self.assertEqual(rec.times, [round_trip])

        with self.subTest("clamped to the deadline, only the verdict rewritten"):
            sim = Simulator(seed=1)
            stub = _StubTarget(service_ns=service, report=reported)
            link = TransportLink(stub, TransportDelay(median=median))
            rec = _Recorder(sim)
            deadline = round_trip - ms_to_ns(1)
            link.submit_attempt(sim, rec, attempt_deadline=deadline)
            sim.run()
            # Only `drop` moves to DEADLINE; the occupancy, the queue sample and
            # the censoring flag are the server's and stand.
            self.assertEqual(rec.calls, [(False, service, DropReason.DEADLINE, 7, True)])
            self.assertEqual(rec.times, [deadline])

    def test_a_target_that_reports_twice_is_refused_loudly(self):
        """`on_done` fires exactly once per attempt, and a violation says so.

        Swallowing the second report would be worse than crashing: the client
        would be handed two outcomes for one attempt and trip "Root request was
        already done!" somewhere far from the cause. This is the same stance
        `ServiceRuntime.finalize` takes.
        """
        sim = Simulator(seed=1)
        stub = _StubTarget(service_ns=ms_to_ns(5), reports_twice=True)
        link = TransportLink(stub, TransportDelay(median=ms_to_ns(3)))
        rec = _Recorder(sim)
        link.submit_attempt(sim, rec)

        with self.assertRaises(RuntimeError) as cm:
            sim.run()
        self.assertIn("more than once", str(cm.exception))
        self.assertEqual(rec.calls, [])  # the duplicate was refused, not delivered


class TestCancellationPhases(unittest.TestCase):
    """`cancel()` per phase, driving `submit_attempt` directly.

    Layout for every test here: a 1 ms link in front of a 10 ms service, so the
    attempt is in forward transit on (0, 1 ms), at the service on (1 ms, 11 ms)
    and in return transit on (11 ms, 12 ms). `sim.run(until=...)` parks the
    clock inside the phase under test.

    `cancel()` must return True ONLY when something was actually torn down:
    `ClientRuntime._cancel_root` reads a True as "on_done already fired
    synchronously and the root is complete".
    """

    def _link_and_service(self):
        svc = _svc("svc", 10)
        return svc, TransportLink(svc, TransportDelay(median=ms_to_ns(1)))

    def test_cancel_during_forward_transit(self):
        sim = Simulator(seed=1)
        svc, link = self._link_and_service()
        rec = _Recorder(sim)
        control = link.submit_attempt(sim, rec)

        sim.run(until=ms_to_ns(0.5))  # in flight towards the service
        self.assertEqual(rec.calls, [])
        self.assertEqual(svc.admitted_total, 0)

        self.assertTrue(control.cancel())
        # No occupancy, no queue observed: the packet never landed.
        self.assertEqual(rec.calls, [(False, 0, DropReason.CANCELLED, None, False)])
        self.assertEqual(rec.times, [ms_to_ns(0.5)])  # synchronous, no return leg
        self.assertEqual(svc.admitted_total, 0)
        self.assertEqual(svc.in_flight, 0)

        self.assertFalse(control.cancel())  # settled: nothing left to tear down
        sim.run()
        self.assertEqual(len(rec.calls), 1)  # the arrival event never fires

    def test_cancel_while_the_forward_expiry_is_pending(self):
        """Still forward transit: the target was never called, and the pending
        DEADLINE delivery must be dropped rather than arriving later."""
        sim = Simulator(seed=1)
        svc = _svc("svc", 10)
        link = TransportLink(svc, TransportDelay(median=ms_to_ns(10)))
        rec = _Recorder(sim)
        control = link.submit_attempt(sim, rec, attempt_deadline=ms_to_ns(5))

        sim.run(until=ms_to_ns(2))
        self.assertEqual(rec.calls, [])

        self.assertTrue(control.cancel())
        self.assertEqual(rec.calls, [(False, 0, DropReason.CANCELLED, None, False)])
        self.assertEqual(rec.times, [ms_to_ns(2)])
        self.assertEqual(svc.admitted_total, 0)

        self.assertFalse(control.cancel())
        sim.run()
        self.assertEqual(len(rec.calls), 1)  # the 5 ms expiry delivery was cancelled

    def test_cancel_at_the_service_releases_the_worker(self):
        """Delegates to the inner control and reports ITS result. The worker is
        handed back at once (capacity is reclaimed, not merely muted), and the
        CANCELLED outcome skips the return leg — cancellation is
        client-initiated and needs no network."""
        sim = Simulator(seed=1)
        svc, link = self._link_and_service()
        rec = _Recorder(sim)
        control = link.submit_attempt(sim, rec)

        sim.run(until=ms_to_ns(1.5))  # arrived at 1 ms, serving until ~11 ms
        self.assertEqual(svc.admitted_total, 1)
        self.assertEqual(svc.in_flight, 1)
        self.assertEqual(rec.calls, [])

        self.assertTrue(control.cancel())
        # Occupancy is the 0.5 ms actually served, censored because the worker
        # was still held when the cancel cut it short.
        self.assertEqual(rec.calls, [(False, ms_to_ns(0.5), DropReason.CANCELLED, 0, True)])
        self.assertEqual(rec.times, [ms_to_ns(1.5)])  # synchronous, no return leg
        self.assertEqual(svc.in_flight, 0)

        self.assertFalse(control.cancel())
        sim.run()
        self.assertEqual(len(rec.calls), 1)

    def test_cancel_at_the_service_relays_the_targets_refusal(self):
        """`cancel()` returns the INNER control's answer; a `False` changes nothing.

        No stock target can refuse — `ServiceRuntime` always tears down an
        attempt that is queued or in service — so the branch needs a target that
        says no. A `True` here would be a lie `ClientRuntime._cancel_root` acts
        on: it reads a `True` as "`on_done` already fired and the root is
        complete", and would trip "Root request was already done!" when the real
        outcome arrives afterwards. So the refusal must be relayed, the attempt
        left bound, and the target's genuine outcome must still come home over
        the return leg.
        """
        sim = Simulator(seed=1)
        stub = _StubTarget(service_ns=ms_to_ns(10), cancellable=False)
        link = TransportLink(stub, TransportDelay(median=ms_to_ns(1)))
        rec = _Recorder(sim)
        control = link.submit_attempt(sim, rec)

        sim.run(until=ms_to_ns(5))  # arrived at 1 ms, reports at 11 ms
        self.assertEqual(stub.arrivals, [ms_to_ns(1)])
        self.assertEqual(rec.calls, [])

        self.assertFalse(control.cancel())
        self.assertEqual(rec.calls, [])  # nothing torn down, so nothing delivered

        sim.run()
        self.assertEqual(rec.calls, [(True, ms_to_ns(10), DropReason.NONE, 0, False)])
        self.assertEqual(rec.times, [ms_to_ns(12)])  # the full return leg, once

    def test_a_refused_cancel_leaves_the_attempt_cancellable_in_its_next_phase(self):
        """ "Leave everything bound" is the other half of relaying a refusal.

        Settling the outer handle on a `False` would look harmless — the second
        `cancel()` returns `False` either way — right up to the moment the
        target's answer reaches the return leg: the response would then be
        un-abandonable and would land on a caller that gave up on it long ago.
        """
        sim = Simulator(seed=1)
        stub = _StubTarget(service_ns=ms_to_ns(10), cancellable=False)
        link = TransportLink(stub, TransportDelay(median=ms_to_ns(1)))
        rec = _Recorder(sim)
        control = link.submit_attempt(sim, rec)

        sim.run(until=ms_to_ns(5))
        self.assertFalse(control.cancel())  # refused at the service

        sim.run(until=ms_to_ns(11))  # the target reported; the response is on the wire
        self.assertEqual(stub.reports, [ms_to_ns(11)])
        self.assertEqual(rec.calls, [])

        self.assertTrue(control.cancel())
        self.assertEqual(rec.calls, [(False, ms_to_ns(10), DropReason.CANCELLED, 0, False)])
        self.assertEqual(rec.times, [ms_to_ns(11)])
        sim.run()
        self.assertEqual(len(rec.calls), 1)  # the 12 ms delivery was dropped

    def test_cancel_at_the_service_when_the_target_answers_with_a_real_outcome(self):
        """The target tears down and hands back a REAL outcome from inside its
        own `cancel()`, instead of a CANCELLED one.

        That puts the response on the return leg before `cancel()` has even
        returned, so the attempt is in return transit by the time the link gets
        the answer. It must still deliver exactly once: the pending delivery has
        to be killed, not merely forgotten — a live event fires a leg later and
        hands the client a second outcome for one attempt, which reaches a real
        `ClientRuntime` as "Root request was already done!".

        The delivery is the return-transit one: CANCELLED, carrying the
        occupancy, queue depth and censoring flag the service actually reported.
        """
        sim = Simulator(seed=1)
        stub = _StubTarget(
            service_ns=ms_to_ns(10),
            cancel_report=(True, ms_to_ns(3), DropReason.NONE, 4, False),
        )
        link = TransportLink(stub, TransportDelay(median=ms_to_ns(1)))
        rec = _Recorder(sim)
        control = link.submit_attempt(sim, rec)

        sim.run(until=ms_to_ns(2))  # arrived at 1 ms, would report at 11 ms
        self.assertEqual(stub.arrivals, [ms_to_ns(1)])
        self.assertEqual(rec.calls, [])

        self.assertTrue(control.cancel())  # something really was torn down
        # Synchronous, and exactly one outcome — not this one now and the
        # target's own answer at 3 ms.
        self.assertEqual(rec.calls, [(False, ms_to_ns(3), DropReason.CANCELLED, 4, False)])
        self.assertEqual(rec.times, [ms_to_ns(2)])

        self.assertFalse(control.cancel())
        sim.run()
        self.assertEqual(len(rec.calls), 1)  # no late delivery ever lands

    def test_a_client_survives_a_target_that_answers_from_inside_its_teardown(self):
        """The same target behind a real client: the root closes once, cancelled.

        A second delivery would arrive at `_on_attempt_done` after the root was
        completed and raise "Root request was already done!" — with the link the
        only thing between a legal target and the crash.
        """
        sim = Simulator(seed=1)
        stub = _StubTarget(
            service_ns=ms_to_ns(10),
            cancel_report=(True, ms_to_ns(3), DropReason.NONE, 4, False),
        )
        client = _client(TransportLink(stub, TransportDelay(median=ms_to_ns(1))))
        root = client.start_request(sim)

        sim.run(until=ms_to_ns(2))  # in service since 1 ms
        self.assertFalse(root.done)
        assert root.cancel is not None
        root.cancel()

        self.assertTrue(root.done)
        self.assertTrue(root.cancelled)
        self.assertEqual(root.attempt_count(), 1)
        self.assertEqual(root.attempts[0].drop_reason, DropReason.CANCELLED)
        self.assertEqual(root.attempts[0].interval.end, ms_to_ns(2))
        self.assertEqual(classify_root(root), RootOutcome.CANCELLED)

        sim.run()  # the killed delivery must not reopen the closed root
        self.assertEqual(root.attempt_count(), 1)
        self.assertEqual(len(client.roots), 1)

    def test_a_cancel_arriving_inside_on_done_is_a_truthful_no_op(self):
        """A cancel racing the completion must lose, and must say that it lost.

        This race is real: `chain.py`'s sibling teardown reaches a client
        re-entrantly from inside another root's completion callback, so
        `cancel()` genuinely runs while `on_done` is still on the stack. It must
        answer `False` — `ClientRuntime._cancel_root` reads a `True` as "the
        root is already complete" and returns without closing it — and it must
        not produce a second `on_done`.
        """
        sim = Simulator(seed=1)
        svc, link = self._link_and_service()
        rec = _Recorder(sim)
        holder: list[AttemptControl] = []
        answers: list[bool] = []

        def reentrant(
            success: bool,
            svc_time: TimeDuration,
            drop: DropReason,
            qsize: int | None,
            censored: bool,
        ) -> None:
            rec(success, svc_time, drop, qsize, censored)
            answers.append(holder[0].cancel())

        holder.append(link.submit_attempt(sim, reentrant))
        sim.run()

        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(answers, [False])  # already delivered: nothing to tear down
        self.assertFalse(holder[0].cancel())
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(svc.in_flight, 0)

    def test_cancel_during_return_transit_reports_the_target_outcome(self):
        """The target has already settled and the delivery is pending: the
        outcome is rewritten to CANCELLED but occupancy, queue size and the
        censoring flag stay exactly as the server reported them."""
        sim = Simulator(seed=1)
        svc, link = self._link_and_service()
        rec = _Recorder(sim)
        control = link.submit_attempt(sim, rec)

        sim.run(until=ms_to_ns(11.5))  # served at ~11 ms, delivery due at ~12 ms
        self.assertEqual(svc.in_flight, 0)
        self.assertEqual(rec.calls, [])

        self.assertTrue(control.cancel())
        self.assertEqual(len(rec.calls), 1)
        success, occupancy, drop, qsize, censored = rec.calls[0]
        self.assertFalse(success)
        self.assertEqual(drop, DropReason.CANCELLED)
        self.assertEqual(qsize, 0)
        self.assertFalse(censored)  # the server finished on its own
        # The reported occupancy is the ~10 ms the server spent, NOT the
        # 10.5 ms that had elapsed since the attempt reached it.
        self.assertLess(abs(occupancy - ms_to_ns(10)), ms_to_ns(0.01))
        self.assertEqual(rec.times, [ms_to_ns(11.5)])

        self.assertFalse(control.cancel())  # exactly once
        sim.run()
        self.assertEqual(len(rec.calls), 1)

    def test_cancel_after_delivery_returns_false(self):
        sim = Simulator(seed=1)
        svc, link = self._link_and_service()
        rec = _Recorder(sim)
        control = link.submit_attempt(sim, rec)

        sim.run()
        self.assertEqual(len(rec.calls), 1)
        success, _occupancy, drop, qsize, censored = rec.calls[0]
        self.assertTrue(success)
        self.assertEqual(drop, DropReason.NONE)
        self.assertEqual(qsize, 0)
        self.assertFalse(censored)
        self.assertLess(abs(rec.times[0] - ms_to_ns(12)), ms_to_ns(0.01))

        self.assertFalse(control.cancel())
        self.assertEqual(len(rec.calls), 1)


class TestSynchronousRejection(unittest.TestCase):
    def test_queue_full_rejection_still_pays_the_return_leg(self):
        """The target may fire the return hook INSIDE `submit_attempt`, before
        the inner control has been handed back. The rejection still travels
        home over the network, and exactly once."""
        sim = Simulator(seed=1)
        svc = _svc("svc", 50, workers=1, queue_capacity=0)
        busy = _Recorder(sim)
        svc.submit_attempt(sim, busy)  # takes the only worker until ~50 ms

        link = TransportLink(svc, TransportDelay(median=ms_to_ns(1)))
        rec = _Recorder(sim)
        link.submit_attempt(sim, rec)

        sim.run(until=ms_to_ns(1.5))  # rejected at 1 ms, delivery due at 2 ms
        self.assertEqual(rec.calls, [])

        sim.run()
        self.assertEqual(rec.calls, [(False, 0, DropReason.QUEUE_FULL, 0, False)])
        self.assertEqual(rec.times, [ms_to_ns(2)])
        self.assertEqual(svc.admitted_total, 1)  # shed before joining the queue
        self.assertEqual(len(busy.calls), 1)


class TestDeterminism(unittest.TestCase):
    def _records(self, seed: int, sigma: float):
        sim = Simulator(seed=seed)
        client = _client(
            TransportLink(_svc("svc", 10, p_fail=1.0), TransportDelay(ms_to_ns(1), sigma)),
            retry=FixedBackoffRetryPolicy(max_attempts=4, delay=ms_to_ns(5)),
        )
        client.start_request(sim)
        sim.run()
        return [
            (
                a.interval.begin,
                a.interval.end,
                a.success,
                a.drop_reason,
                a.queue_size_at_end,
                a.service_time_ns,
                a.service_time_censored,
            )
            for root in client.roots
            for a in root.attempts
        ]

    def test_same_seed_reproduces_every_attempt_record(self):
        first = self._records(11, 0.6)
        self.assertEqual(len(first), 4)
        self.assertEqual(first, self._records(11, 0.6))
        self.assertNotEqual(first, self._records(12, 0.6))

    def test_positive_sigma_varies_the_delay_across_attempts(self):
        sim = Simulator(seed=5)
        svc = _svc("svc", 10, p_fail=1.0)
        client = _client(
            TransportLink(svc, TransportDelay(median=ms_to_ns(1), sigma=0.6)),
            retry=FixedBackoffRetryPolicy(max_attempts=4, delay=ms_to_ns(5)),
        )
        client.start_request(sim)
        sim.run()

        round_trips = []
        for attempt in client.roots[0].attempts:
            occupancy = attempt.service_time_ns
            assert occupancy is not None
            round_trip = (attempt.interval.end - attempt.interval.begin) - occupancy
            self.assertGreaterEqual(round_trip, 0)
            round_trips.append(round_trip)

        self.assertEqual(len(round_trips), 4)
        self.assertGreater(len(set(round_trips)), 1)  # not a constant median


class TestPerLegDraws(unittest.TestCase):
    """Two independent draws, in a fixed order — the part a real service hides.

    Against a `ServiceRuntime` a caller only ever observes the sum
    `d_f + occupancy + d_r`, so "draw once and spend it twice", "reuse the
    median for the way back" and "draw both legs up front" all reproduce the
    end-to-end timings exactly. `_StubTarget` records its own arrival and report
    instants, which splits the round trip into the two legs.
    """

    def _legs(self, seed: int, sigma: float, draws_at_target: int = 0):
        """Return `(d_f, d_r)` for one attempt over a link, measured separately."""
        sim = Simulator(seed=seed)
        stub = _StubTarget(rng_draws_before_report=draws_at_target)
        link = TransportLink(stub, TransportDelay(median=ms_to_ns(2), sigma=sigma))
        rec = _Recorder(sim)
        link.submit_attempt(sim, rec)
        sim.run()

        (arrival,) = stub.arrivals
        (delivered,) = rec.times
        return arrival, delivered - arrival

    def test_each_leg_is_a_draw_of_its_own(self):
        forward, back = [], []
        for seed in range(1, 9):
            d_f, d_r = self._legs(seed, 0.6)
            self.assertGreaterEqual(d_f, 0)
            self.assertGreaterEqual(d_r, 0)
            forward.append(d_f)
            back.append(d_r)

        self.assertGreater(len(set(forward)), 1)  # the forward leg is sampled
        self.assertGreater(len(set(back)), 1)  # so is the return leg, not a constant
        self.assertNotEqual(forward, back)  # and not one draw spent twice

    def test_the_forward_leg_is_drawn_at_submit_and_the_return_leg_at_report(self):
        """The order of the two draws is load-bearing: it fixes where every
        later seeded draw in the run lands.

        The target consumes one draw of its own between the legs. The forward
        leg is drawn before the target has run at all, so it must be unmoved;
        the return leg is drawn after it, so it must move.
        """
        plain_f, plain_r = self._legs(5, 0.6, draws_at_target=0)
        noisy_f, noisy_r = self._legs(5, 0.6, draws_at_target=1)

        self.assertEqual(plain_f, noisy_f)
        self.assertNotEqual(plain_r, noisy_r)


class TestChainIntegration(unittest.TestCase):
    def test_edge_behind_a_slow_link_is_cancelled_when_the_parent_deadline_fires(self):
        """A -> B, where the A->B edge sits behind a 20 ms link and A's parent
        gives the whole call 10 ms.

        The edge client has no attempt timeout of its own, so the deadline that
        expires is the parent's, handed down through `override_global_deadline`.
        `chain._launch_fanout` schedules the parent's deadline guard BEFORE
        launching the edges, so at the 10 ms tie the guard fires first and
        cancels an attempt still in forward transit: the child root is booked
        CANCELLED (abandoned, not failed) and B never sees an arrival.
        """
        sim = Simulator(seed=1)
        a_node = _node("A", 1, propagate=True)
        b_node = _node("B", 5)
        edge_client = _client(TransportLink(b_node, TransportDelay(median=ms_to_ns(20))))
        a_node.add_downstream(DownstreamEdge(client=edge_client, target=b_node))

        top = _client(a_node, attempt_timeout=ms_to_ns(10))
        top.start_request(sim)
        sim.run()

        (top_root,) = top.roots
        self.assertIs(classify_root(top_root), RootOutcome.DEADLINE)
        self.assertEqual(top.metrics().summary().dropped_deadline, 1)

        (edge_root,) = edge_client.roots
        self.assertTrue(edge_root.cancelled)
        self.assertIs(classify_root(edge_root), RootOutcome.CANCELLED)
        self.assertEqual(edge_client.metrics().summary().dropped_cancelled, 1)
        self.assertEqual(edge_client.attempts_total, 1)

        (edge_attempt,) = edge_root.attempts
        self.assertEqual(edge_attempt.drop_reason, DropReason.CANCELLED)
        self.assertEqual(edge_attempt.interval.end, ms_to_ns(10))
        self.assertIsNone(edge_attempt.queue_size_at_end)
        self.assertIsNone(edge_attempt.service_time_ns)

        self.assertEqual(b_node.service.admitted_total, 0)  # nothing ever arrived
        self.assertEqual(a_node.service.admitted_total, 1)

    def _fanout_with_links(self, seed: int, c_fails: bool):
        """A -> {B, C}, a 2 ms link on the B edge and a 3 ms link on the C edge."""
        sim = Simulator(seed=seed)
        a_node = _node("A", 1)
        b_node = _node("B", 5)
        c_node = _node("C", 5, p_fail=1.0 if c_fails else 0.0)
        b_edge = _client(TransportLink(b_node, TransportDelay(median=ms_to_ns(2))))
        c_edge = _client(TransportLink(c_node, TransportDelay(median=ms_to_ns(3))))
        a_node.add_downstream(DownstreamEdge(client=b_edge, target=b_node))
        a_node.add_downstream(DownstreamEdge(client=c_edge, target=c_node))

        top = _client(a_node, attempt_timeout=ms_to_ns(500))
        top.start_request(sim)
        sim.run()
        return top, b_edge, c_edge, b_node, c_node

    def test_and_semantics_hold_with_links_on_every_edge(self):
        """One failing branch still fails the parent, and the successful run's
        latency really contains both round trips."""
        top, b_edge, c_edge, b_node, c_node = self._fanout_with_links(2, c_fails=True)
        (failed_root,) = top.roots
        self.assertIs(classify_root(failed_root), RootOutcome.SERVER_FAILURE)
        self.assertTrue(any(a.success for a in b_edge.roots[0].attempts))  # B did succeed
        self.assertIs(classify_root(c_edge.roots[0]), RootOutcome.SERVER_FAILURE)
        self.assertEqual(b_node.service.admitted_total, 1)
        self.assertEqual(c_node.service.admitted_total, 1)

        top, _b_edge, _c_edge, _b, _c = self._fanout_with_links(2, c_fails=False)
        (ok_root,) = top.roots
        self.assertIs(classify_root(ok_root), RootOutcome.SUCCESS)
        # 1 ms local at A, then the slower branch: 3 ms out + 5 ms at C + 3 ms
        # back = 12 ms, which is strictly more than the 10 ms of link delay the
        # four legs cost in total.
        latency = root_latency(ok_root)
        self.assertGreater(latency, 2 * ms_to_ns(2) + 2 * ms_to_ns(3))
        self.assertLess(abs(latency - ms_to_ns(12)), ms_to_ns(0.1))


class TestServiceLikeProtocol(unittest.TestCase):
    def test_link_is_accepted_as_a_client_service_and_as_an_edge_target(self):
        """`ty` is the real assertion here: a `TransportLink` must satisfy
        `caller.ServiceLike` structurally, with no cast at either call site, in
        front of a plain `ServiceRuntime` and in front of a chain node."""
        node = _node("N", 5)
        link = TransportLink(node, TransportDelay(median=ms_to_ns(1)))
        edge_client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(50)
                ),
            ),
            service=link,
        )
        edge = DownstreamEdge(client=edge_client, target=node)

        svc = _svc("S", 5)
        plain_client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(50)
                ),
            ),
            service=TransportLink(svc, TransportDelay(median=ms_to_ns(1))),
        )

        self.assertIs(edge.client, edge_client)
        self.assertIs(edge_client.service, link)
        self.assertIs(link.cfg, node.cfg)
        self.assertEqual(edge_client.service.cfg.name, "N")
        self.assertEqual(plain_client.service.cfg.name, "S")


if __name__ == "__main__":
    unittest.main()
