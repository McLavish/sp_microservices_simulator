"""Admission control (`msim.admission`).

An `AdmissionGate` puts controllers in front of a target: an attempt they refuse
is reported `REJECTED` at once and never reaches the target; one they admit is
forwarded, and the gate watches how it ends. The tests below pin the gate's
bookkeeping on every way an attempt can leave it, the two-phase rule, each
controller's arithmetic, and the same controllers used on the caller's side.

Service times use `sigma=0`, which a station floors to 1e-6: a "10 ms" service
lasts 10 ms give or take a few nanoseconds, so instants are compared with a
tolerance.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from msim.admission import (
    AdmissionController,
    AdmissionCounts,
    AdmissionGate,
    AdmissionOutcome,
    AdmissionRequest,
    AimdBudget,
    ConcurrencyRetryBudget,
    ControlledRetryPolicy,
    FailureRateGate,
    GoodputCoupledBudget,
    MixedRefillBudget,
    SuccessFundedBudget,
    TenantRetryQuota,
    TokenBucketBudget,
)
from msim.caller import ClientConfig, ClientRuntime
from msim.chain import DownstreamEdge, ServiceNodeRuntime
from msim.core import FAILURE_OUTCOMES, DropReason, TimeInterval, ms_to_ns, s_to_ns
from msim.faults import PartialFailure
from msim.retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy, RetryPolicy
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.transport import TransportDelay, TransportLink
from msim.workload import Workload

MS = ms_to_ns(1)
S = s_to_ns(1)


def _station(
    name: str = "svc",
    *,
    median_ms: float = 10,
    workers: int = 4,
    capacity: int | None = None,
    fail_until_ms: float | None = None,
) -> ServiceRuntime:
    failures = []
    if fail_until_ms is not None:
        failures.append(PartialFailure(TimeInterval(0, ms_to_ns(fail_until_ms)), p_fail=1.0))
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
    tenant: str | None = "a",
    attempts: int = 1,
    delay_ms: float = 0,
    attempt_ms: float = 1000,
    global_ms: float | None = None,
    mode: str = "attempt",
    retry: RetryPolicy | None = None,
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
                attempt_timeout=ms_to_ns(attempt_ms),
            ),
            tenant=tenant,
            server_deadline=mode,
        ),
        service=target,
    )


class _Spy(AdmissionController):
    """Admits what `admit` says (everything by default) and records every call."""

    def __init__(self, admit=lambda request: True) -> None:
        self.admit = admit
        self.checked: list[AdmissionRequest] = []
        self.committed: list[AdmissionRequest] = []
        self.refunded: list[AdmissionRequest] = []
        self.observed: list[AdmissionOutcome] = []

    def check(self, request: AdmissionRequest) -> bool:
        self.checked.append(request)
        return self.admit(request)

    def commit(self, request: AdmissionRequest) -> None:
        self.committed.append(request)

    def refund(self, request: AdmissionRequest) -> None:
        self.refunded.append(request)

    def observe(self, outcome: AdmissionOutcome) -> None:
        self.observed.append(outcome)

    @property
    def outcomes(self) -> list[tuple[bool, DropReason]]:
        return [(o.success, o.drop_reason) for o in self.observed]


def _reasons(client: ClientRuntime) -> list[list[DropReason]]:
    return [[a.drop_reason for a in root.attempts] for root in client.roots]


# ------------------------------------------------------------------ the gate


class TestEveryWayOutOfTheGate(unittest.TestCase):
    """Whatever happens to an admitted attempt, the gate's account returns to zero."""

    def test_success(self):
        spy = _Spy()
        gate = AdmissionGate(_station(), [spy])
        client = _client(gate)
        sim = Simulator(seed=1)
        client.start_request(sim)
        seen: list[int] = []
        sim.schedule(5 * MS, lambda: seen.append(gate.outstanding))
        sim.run()

        self.assertEqual(seen, [1])  # in flight while the station works on it
        self.assertEqual(gate.outstanding, 0)
        self.assertEqual(spy.outcomes, [(True, DropReason.NONE)])
        self.assertEqual(gate.counts, {"a": AdmissionCounts(attempts=1)})
        self.assertEqual(_reasons(client), [[DropReason.NONE]])

    def test_a_queue_that_is_full_reports_from_inside_the_submission(self):
        spy = _Spy()
        station = _station(workers=1, capacity=0)
        gate = AdmissionGate(station, [spy])
        client = _client(gate)
        sim = Simulator(seed=1)
        client.start_request(sim)
        client.start_request(sim)  # no worker, no queue: shed on arrival
        self.assertEqual(gate.outstanding, 1)  # the shed one is already off the books
        self.assertEqual(spy.outcomes, [(False, DropReason.QUEUE_FULL)])
        sim.run()

        self.assertEqual(gate.outstanding, 0)
        self.assertEqual(spy.outcomes, [(False, DropReason.QUEUE_FULL), (True, DropReason.NONE)])
        self.assertEqual(gate.counts["a"], AdmissionCounts(attempts=2))

    def test_an_attempt_that_expires_in_the_queue(self):
        spy = _Spy()
        gate = AdmissionGate(_station(median_ms=50, workers=1), [spy])
        patient = _client(gate, attempt_ms=1000)
        hasty = _client(gate, attempt_ms=10)
        sim = Simulator(seed=1)
        patient.start_request(sim)
        hasty.start_request(sim)  # waits behind the 50 ms one, gives up at 10 ms
        sim.run()

        self.assertEqual(gate.outstanding, 0)
        self.assertEqual(
            [(o.success, o.drop_reason, round(o.now / MS)) for o in spy.observed],
            [(False, DropReason.DEADLINE, 10), (True, DropReason.NONE, 50)],
        )

    def test_a_cancelled_attempt_leaves_the_account_and_is_not_observed(self):
        spy = _Spy()
        slow_station = _station("slow", median_ms=50)
        gate = AdmissionGate(slow_station, [spy])
        parent = ServiceNodeRuntime(service=_station("P", median_ms=1), propagate_cancellation=True)
        fast = _client(_station("fast", median_ms=5, fail_until_ms=1000))
        slow = _client(gate)
        parent.add_downstream(DownstreamEdge(client=fast, target=parent))
        parent.add_downstream(DownstreamEdge(client=slow, target=parent))
        sim = Simulator(seed=1)
        _client(parent).start_request(sim)
        sim.run()

        self.assertEqual(_reasons(slow), [[DropReason.CANCELLED]])
        self.assertEqual(spy.observed, [])  # abandoned work is not evidence
        self.assertEqual(gate.outstanding, 0)
        self.assertEqual(slow_station.in_flight, 0)
        self.assertEqual(gate.counts["a"], AdmissionCounts(attempts=1))

    def test_an_orphan_is_observed_when_the_server_finishes_it(self):
        """The client leaves at 10 ms; the gate sees the real answer at 30 ms."""
        spy = _Spy()
        gate = AdmissionGate(_station(median_ms=30), [spy])
        client = _client(gate, attempt_ms=10, mode="global")
        sim = Simulator(seed=1)
        client.start_request(sim)
        seen: list[int] = []
        sim.schedule(20 * MS, lambda: seen.append(gate.outstanding))
        sim.run()

        self.assertEqual(_reasons(client), [[DropReason.DEADLINE]])
        self.assertEqual(seen, [1])  # still the server's problem at 20 ms
        self.assertEqual(
            [(o.success, o.drop_reason, round(o.now / MS)) for o in spy.observed],
            [(True, DropReason.NONE, 30)],
        )
        self.assertEqual(gate.outstanding, 0)

    def test_a_refusal_behind_a_link_travels_back_over_the_wire(self):
        station = _station()
        gate = AdmissionGate(station, [_Spy(admit=lambda request: False)])
        link = TransportLink(gate, TransportDelay(median=2 * MS))
        client = _client(link)
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()

        (root,) = client.roots
        self.assertEqual([a.drop_reason for a in root.attempts], [DropReason.REJECTED])
        self.assertEqual(root.attempts[0].interval.end, 4 * MS)  # there and back
        self.assertIsNone(root.attempts[0].queue_size_at_end)
        self.assertEqual(station.admitted_total, 0)  # the station never heard of it
        self.assertEqual(gate.outstanding, 0)
        self.assertEqual(gate.counts["a"], AdmissionCounts(attempts=1, rejected=1))

    def test_a_refusal_inside_a_fan_out_does_not_skip_the_other_calls(self):
        refused_station = _station("refused")
        other_station = _station("other")
        gate = AdmissionGate(refused_station, [_Spy(admit=lambda request: False)])
        parent = ServiceNodeRuntime(service=_station("P", median_ms=1))
        first = _client(gate)
        second = _client(other_station)
        parent.add_downstream(DownstreamEdge(client=first, target=parent))
        parent.add_downstream(DownstreamEdge(client=second, target=parent))
        top = _client(parent)
        sim = Simulator(seed=1)
        top.start_request(sim)
        sim.run()

        self.assertEqual(_reasons(first), [[DropReason.REJECTED]])
        self.assertEqual(other_station.admitted_total, 1)  # launched all the same
        # The parent's attempt fails for the reason its call failed.
        self.assertEqual(_reasons(top), [[DropReason.REJECTED]])
        self.assertEqual(gate.outstanding, 0)

    def test_a_target_that_reports_twice_is_an_error(self):
        class Twice:
            cfg = _station().cfg

            def submit_attempt(self, sim, on_done, attempt_deadline=None, *, ctx=None):
                on_done(True, 1, DropReason.NONE, 0, False)
                on_done(True, 1, DropReason.NONE, 0, False)

        gate = AdmissionGate(Twice(), [_Spy()])
        sim = Simulator(seed=1)
        with self.assertRaises(RuntimeError):
            _client(gate).start_request(sim)


class TestWhatTheGateDecides(unittest.TestCase):
    def test_a_refused_attempt_is_rejected_and_the_request_is_booked_so(self):
        station = _station(fail_until_ms=1000)
        budget = TokenBucketBudget(rate=0, burst=1)
        gate = AdmissionGate(station, [budget])
        client = _client(gate, attempts=4)
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()

        # One retry is paid for; the next is refused, and a fixed-backoff client
        # retries a refusal like any failure until it runs out of attempts.
        self.assertEqual(
            _reasons(client),
            [
                [
                    DropReason.SERVER_FAILURE,
                    DropReason.SERVER_FAILURE,
                    DropReason.REJECTED,
                    DropReason.REJECTED,
                ]
            ],
        )
        self.assertEqual(station.admitted_total, 2)
        self.assertEqual(
            gate.counts["a"],
            AdmissionCounts(attempts=4, rejected=2, retries_asked=3, retries_admitted=1),
        )
        summary = client.metrics().summary()
        self.assertEqual(summary.dropped_rejected, 1)
        self.assertEqual(summary.dropped_server_failure, 0)

    def test_nobody_spends_until_everybody_agrees(self):
        for order in ("budget first", "budget last"):
            with self.subTest(order=order):
                budget = TokenBucketBudget(rate=0, burst=1)
                refuser = _Spy(admit=lambda request: not request.is_retry)
                controllers = [budget, refuser] if order == "budget first" else [refuser, budget]
                gate = AdmissionGate(_station(fail_until_ms=1000), controllers)
                sim = Simulator(seed=1)
                _client(gate, attempts=3).start_request(sim)
                sim.run()

                # Two retries asked, both refused by the other controller: the
                # token is still in the bucket, and nothing was committed.
                self.assertEqual(gate.counts["a"].retries_asked, 2)
                self.assertEqual(gate.counts["a"].retries_admitted, 0)
                self.assertEqual(budget.balance, 1.0)
                self.assertEqual([r.is_retry for r in refuser.committed], [False])

    def test_counts_are_kept_per_tenant(self):
        gate = AdmissionGate(_station(fail_until_ms=1000), [TokenBucketBudget(rate=0, burst=1)])
        sim = Simulator(seed=1)
        _client(gate, tenant="a", attempts=2).start_request(sim)
        sim.run()
        _client(gate, tenant="b", attempts=2).start_request(sim)
        sim.run()

        self.assertEqual(
            gate.counts,
            {
                "a": AdmissionCounts(attempts=2, retries_asked=1, retries_admitted=1),
                "b": AdmissionCounts(attempts=2, rejected=1, retries_asked=1),
            },
        )

    def test_the_gate_needs_a_request_context(self):
        gate = AdmissionGate(_station(), [_Spy()])
        with self.assertRaises(ValueError) as raised:
            _client(gate, tenant=None).start_request(Simulator(seed=1))
        self.assertIn("ClientConfig.tenant", str(raised.exception))

    def test_validation(self):
        with self.assertRaises(ValueError):
            AdmissionGate(_station(), [_Spy()], retry_scope="tree")
        not_a_controller: Any = object()
        with self.assertRaises(ValueError):
            AdmissionGate(_station(), [not_a_controller])


class TestRetryScope(unittest.TestCase):
    def _retries_seen(self, scope: str) -> list[bool]:
        """top client (2 attempts) -> P -> edge client (1 attempt) -> gate -> leaf."""
        spy = _Spy()
        gate = AdmissionGate(_station("leaf", fail_until_ms=15), [spy], retry_scope=scope)
        parent = ServiceNodeRuntime(service=_station("P", median_ms=1))
        parent.add_downstream(DownstreamEdge(client=_client(gate, tenant=None), target=parent))
        sim = Simulator(seed=1)
        _client(parent, attempts=2).start_request(sim)
        sim.run()
        self.assertEqual(len(spy.checked), 2)
        return [request.is_retry for request in spy.checked]

    def test_hop_counts_only_the_callers_own_retries(self):
        # The edge never retries: both of its calls are first tries.
        self.assertEqual(self._retries_seen("hop"), [False, False])

    def test_request_counts_what_a_retry_above_caused(self):
        self.assertEqual(self._retries_seen("request"), [False, True])


# ------------------------------------------------------------- the controllers


def _retry(now: int, tenant: str | None = None, **load) -> AdmissionRequest:
    return AdmissionRequest(now=now, tenant=tenant, attempt=2, is_retry=True, **load)


def _first(now: int, tenant: str | None = None) -> AdmissionRequest:
    return AdmissionRequest(now=now, tenant=tenant, attempt=1, is_retry=False)


def _outcome(now: int, success: bool, *, tenant: str | None = None, is_retry: bool = False):
    return AdmissionOutcome(
        now=now,
        tenant=tenant,
        attempt=2 if is_retry else 1,
        is_retry=is_retry,
        success=success,
        drop_reason=DropReason.NONE if success else DropReason.SERVER_FAILURE,
    )


def _take(controller: AdmissionController, now: int, tenant: str | None = None, **load) -> bool:
    """Ask for one retry and take it if granted, as a gate would."""
    request = _retry(now, tenant, **load)
    if not controller.check(request):
        return False
    controller.commit(request)
    return True


def _drain(controller: AdmissionController, now: int, tenant: str | None = None) -> int:
    granted = 0
    while _take(controller, now, tenant):
        granted += 1
    return granted


class TestFirstAttemptsAreNeverCharged(unittest.TestCase):
    def test_every_retry_controller_lets_a_first_attempt_through_for_free(self):
        bucket = TokenBucketBudget(rate=0, burst=1)
        funded = SuccessFundedBudget(ratio=0.1, max_retries=1)
        controllers = [
            bucket,
            AimdBudget(min_rate=1, max_rate=10, initial_rate=1, burst=1),
            funded,
            FailureRateGate(failure_threshold=0.5, min_requests=1, window=S, open_delay=0),
            GoodputCoupledBudget(),
            ConcurrencyRetryBudget(),
            TenantRetryQuota(share=0.0),
        ]
        for controller in controllers:
            with self.subTest(controller=type(controller).__name__):
                for i in range(50):
                    request = _first(i * MS)
                    self.assertTrue(controller.check(request))
                    controller.commit(request)
        # ... and the budgets still hold what they started with.
        self.assertEqual(bucket.balance, 1.0)
        self.assertEqual(funded.balance, 1.0)


class TestTokenBucketBudget(unittest.TestCase):
    def test_the_bucket_starts_full_and_refills_at_the_rate(self):
        budget = TokenBucketBudget(rate=10, burst=5)
        self.assertEqual(_drain(budget, 0), 5)
        self.assertFalse(_take(budget, 99 * MS))  # 0.99 of a token
        self.assertTrue(_take(budget, 100 * MS))
        self.assertFalse(_take(budget, 100 * MS))

    def test_the_refill_is_exact_however_often_it_is_asked(self):
        for step_ms in (1, 7, 50, 333):
            with self.subTest(step_ms=step_ms):
                budget = TokenBucketBudget(rate=10, burst=5)
                _drain(budget, 0)
                granted = sum(
                    _drain(budget, now) for now in range(step_ms * MS, 10 * S + 1, step_ms * MS)
                )
                # 10 per second for the part of the 10 s that was looked at.
                self.assertEqual(granted, 10 * ((10_000 // step_ms) * step_ms) // 1000)

    def test_an_idle_bucket_holds_no_more_than_its_size(self):
        budget = TokenBucketBudget(rate=10, burst=5)
        _drain(budget, 0)
        self.assertEqual(_drain(budget, 60 * S), 5)

    def test_a_zero_rate_is_a_one_off_allowance(self):
        budget = TokenBucketBudget(rate=0, burst=3)
        self.assertEqual(_drain(budget, 0), 3)
        self.assertEqual(_drain(budget, 3600 * S), 0)

    def test_a_refund_gives_one_token_back(self):
        budget = TokenBucketBudget(rate=0, burst=2)
        request = _retry(0)
        self.assertTrue(budget.check(request))
        budget.commit(request)
        self.assertEqual(budget.balance, 1.0)
        budget.refund(request)
        budget.refund(request)  # never above the bucket size
        self.assertEqual(budget.balance, 2.0)

    def test_checking_alone_spends_nothing(self):
        budget = TokenBucketBudget(rate=0, burst=1)
        for _ in range(5):
            self.assertTrue(budget.check(_retry(0)))
        self.assertEqual(budget.balance, 1.0)

    def test_validation(self):
        bad: list[dict[str, Any]] = [
            {"rate": -1, "burst": 1},
            {"rate": float("nan"), "burst": 1},
            {"rate": 1, "burst": 0},
            {"rate": 1, "burst": 1.5},
        ]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                TokenBucketBudget(**kwargs)


class TestAimdBudget(unittest.TestCase):
    def _budget(self, **kwargs) -> AimdBudget:
        defaults: dict[str, Any] = {
            "min_rate": 5,
            "max_rate": 100,
            "initial_rate": 40,
            "burst": 10,
            "additive_step": 5,
            "decrease_factor": 0.5,
            "window": S,
            "failure_threshold": 0.1,
        }
        return AimdBudget(**{**defaults, **kwargs})

    def _window(self, budget: AimdBudget, start: int, *, total: int, failed: int) -> None:
        """`total` outcomes spread over the second that starts at `start`."""
        for i in range(total):
            budget.observe(_outcome(start + i * (S // total), success=i >= failed))

    def test_a_failing_window_halves_the_rate_and_a_healthy_one_adds_a_step(self):
        budget = self._budget()
        self.assertEqual(budget.current_rate, 40)
        self._window(budget, 0, total=10, failed=2)  # 20 % > 10 %
        self._window(budget, S, total=10, failed=0)  # its first outcome closes window one
        self.assertEqual(budget.current_rate, 20)
        self._window(budget, 2 * S, total=10, failed=1)  # 10 % is not MORE than 10 %
        self.assertEqual(budget.current_rate, 25)
        budget.observe(_outcome(3 * S, True))
        self.assertEqual(budget.current_rate, 30)

    def test_the_rate_stays_within_its_bounds(self):
        budget = self._budget()
        for window in range(10):
            self._window(budget, window * S, total=10, failed=10)
        self.assertEqual(budget.current_rate, 5)
        for window in range(10, 40):
            self._window(budget, window * S, total=10, failed=0)
        self.assertEqual(budget.current_rate, 100)

    def test_a_window_without_attempts_changes_nothing(self):
        budget = self._budget()
        budget.check(_retry(0))
        budget.check(_retry(5 * S))
        self.assertEqual(budget.current_rate, 40)

    def test_the_bucket_refills_at_the_current_rate(self):
        budget = self._budget(burst=100)
        self._window(budget, 0, total=10, failed=10)
        self.assertEqual(_drain(budget, S), 100)  # the opening balance; rate now 20
        self.assertEqual(budget.current_rate, 20)
        self.assertEqual(_drain(budget, S + 500 * MS), 10)  # 20 per second for half a second

    def test_validation(self):
        bad: list[dict[str, Any]] = [
            {"min_rate": 10, "max_rate": 5},
            {"initial_rate": 200},
            {"decrease_factor": 1.0},
            {"decrease_factor": 0},
            {"failure_threshold": 1.5},
            {"window": 0},
            {"burst": 0},
        ]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self._budget(**kwargs)


class TestSuccessFundedBudget(unittest.TestCase):
    def test_successes_and_only_successes_pay_for_retries(self):
        budget = SuccessFundedBudget(ratio=0.1, max_retries=3)
        self.assertEqual(_drain(budget, 0), 3)  # starts full
        for _ in range(100):
            budget.observe(_outcome(0, success=False))
        self.assertEqual(_drain(budget, 0), 0)
        for _ in range(9):
            budget.observe(_outcome(0, success=True))
        self.assertFalse(_take(budget, 0))  # 0.9 of a retry
        budget.observe(_outcome(0, success=True))
        self.assertTrue(_take(budget, 0))

    def test_a_ratio_with_many_decimals_is_not_rounded_down(self):
        budget = SuccessFundedBudget(ratio=0.0867, max_retries=10)
        _drain(budget, 0)
        for _ in range(12):
            budget.observe(_outcome(0, success=True))
        self.assertEqual(_drain(budget, 0), 1)  # 12 x 0.0867 = 1.04

    def test_the_bank_is_capped(self):
        budget = SuccessFundedBudget(ratio=0.5, max_retries=3)
        for _ in range(1000):
            budget.observe(_outcome(0, success=True))
        self.assertEqual(_drain(budget, 0), 3)

    def test_validation(self):
        bad: list[dict[str, Any]] = [
            {"ratio": -0.1, "max_retries": 1},
            {"ratio": 1.5, "max_retries": 1},
            {"ratio": 0.1, "max_retries": 0},
        ]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                SuccessFundedBudget(**kwargs)


class TestFailureRateGate(unittest.TestCase):
    def _gate(self, **kwargs) -> FailureRateGate:
        defaults: dict[str, Any] = {
            "failure_threshold": 0.5,
            "min_requests": 4,
            "window": S,
            "open_delay": 500 * MS,
        }
        return FailureRateGate(**{**defaults, **kwargs})

    def _feed(self, gate: FailureRateGate, start: int, results: list[bool]) -> None:
        for i, ok in enumerate(results):
            gate.observe(_outcome(start + i * MS, ok))

    def test_it_opens_on_more_than_the_threshold(self):
        gate = self._gate()
        self._feed(gate, 0, [False, False, True, True])  # exactly half: not more
        self.assertFalse(gate.is_open)
        self.assertTrue(gate.check(_retry(10 * MS)))
        self._feed(gate, 10 * MS, [False])  # 3 of 5
        self.assertTrue(gate.is_open)
        self.assertFalse(gate.check(_retry(20 * MS)))
        self.assertTrue(gate.check(_first(20 * MS)))  # first attempts always pass

    def test_too_few_outcomes_decide_nothing(self):
        gate = self._gate()
        self._feed(gate, 0, [False, False, False])
        self.assertFalse(gate.is_open)

    def test_it_closes_when_the_success_rate_is_back_after_the_delay(self):
        gate = self._gate()
        self._feed(gate, 0, [False] * 4)
        self.assertTrue(gate.is_open)
        # Before the delay nothing closes it, whatever succeeds.
        self._feed(gate, 100 * MS, [True] * 20)
        self.assertTrue(gate.is_open)
        # At 1.2 s everything fed so far has left the 1 s window: these three
        # outcomes are all it holds, fewer than min_requests.
        self._feed(gate, 1200 * MS, [True, True, False])
        self.assertTrue(gate.is_open)
        # A fourth: 2 of 4 succeeded, which is the closing threshold (1 - 0.5).
        self._feed(gate, 1210 * MS, [False])
        self.assertFalse(gate.is_open)

    def test_the_default_closing_threshold_is_the_complement(self):
        for failure_threshold in (0.01, 0.05, 0.1, 0.2, 0.5):
            with self.subTest(failure_threshold=failure_threshold):
                gate = self._gate(
                    failure_threshold=failure_threshold, min_requests=20, open_delay=0
                )
                self._feed(gate, 0, [False] * 20)
                self.assertTrue(gate.is_open)
                failures = round(failure_threshold * 100)
                results = [False] * failures + [True] * (100 - failures)
                self._feed(gate, 5 * S, results[:-1])
                self.assertTrue(gate.is_open)  # one success short of the threshold
                gate.observe(_outcome(5 * S + 99 * MS, True))
                self.assertFalse(gate.is_open)

    def test_validation(self):
        bad: list[dict[str, Any]] = [
            {"failure_threshold": 1.2},
            {"success_threshold": -0.1},
            {"min_requests": 0},
            {"window": 0},
            {"open_delay": -1},
        ]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self._gate(**kwargs)


class TestGoodputCoupledBudget(unittest.TestCase):
    def _healthy_second(self, budget, start: int, tenant: str = "a", successes: int = 100):
        for i in range(successes):
            budget.observe(_outcome(start + i * (S // successes), True, tenant=tenant))

    def test_a_new_tenant_gets_one_retry_per_window(self):
        budget = GoodputCoupledBudget(alpha=0.1)
        self.assertTrue(_take(budget, 0, "a"))
        self.assertFalse(_take(budget, 10 * MS, "a"))

    def test_retries_are_a_fraction_of_the_goodput(self):
        budget = GoodputCoupledBudget(alpha=0.1)
        self._healthy_second(budget, 0)
        # 100 successes per second: 10 retries may be in any one window.
        granted = sum(_take(budget, S + i * MS, "a") for i in range(50))
        self.assertEqual(granted, 10)
        self.assertEqual(budget.state("a").goodput_rate, 100.0)
        # The tenant keeps succeeding, so its allowance stays at 10 ...
        for i in range(100):
            budget.observe(_outcome(S + 50 * MS + i * 9 * MS, True, tenant="a"))
        # ... and a retry leaves the window one second after it was admitted.
        self.assertFalse(_take(budget, 2 * S - MS, "a"))
        self.assertTrue(_take(budget, 2 * S + MS, "a"))
        self.assertFalse(_take(budget, 2 * S + MS, "a"))

    def test_goodput_falls_fast_and_recovers_slowly(self):
        budget = GoodputCoupledBudget(alpha=0.1, beta_down=0.3, beta_up=0.05)
        self._healthy_second(budget, 0)
        budget.check(_retry(2 * S, "a"))  # closes window one (100/s) and window two (0/s)
        self.assertAlmostEqual(budget.state("a").goodput_rate, 70.0)
        self._healthy_second(budget, 2 * S)
        budget.check(_retry(3 * S, "a"))
        self.assertAlmostEqual(budget.state("a").goodput_rate, 71.5)

    def test_one_tenant_cannot_spend_anothers_allowance(self):
        budget = GoodputCoupledBudget(alpha=0.1)
        self._healthy_second(budget, 0, "a")
        self._healthy_second(budget, 0, "b")
        self.assertEqual(sum(_take(budget, S + i * MS, "a") for i in range(50)), 10)
        self.assertEqual(sum(_take(budget, S + i * MS, "b") for i in range(50)), 10)

    def test_a_tenant_whose_requests_fail_is_left_with_one_retry_per_window(self):
        budget = GoodputCoupledBudget(alpha=0.1)
        self._healthy_second(budget, 0)
        self.assertEqual(budget.allowance("a"), 1.0)  # the opening allowance, until a window ends
        for second in range(1, 30):
            for i in range(100):
                budget.observe(_outcome(second * S + i * 10 * MS, False, tenant="a"))
        # The estimate decays towards zero and never reaches it ...
        self.assertGreater(budget.allowance("a"), 0)
        self.assertLess(budget.allowance("a"), 0.01)
        # ... and the count of retries in the window is compared with it as it
        # is, so the first retry of a window always fits and the second does not.
        self.assertTrue(_take(budget, 30 * S, "a"))
        self.assertFalse(_take(budget, 30 * S + MS, "a"))
        self.assertTrue(_take(budget, 31 * S + MS, "a"))

    def test_the_success_rate_gate_stops_retries_that_do_not_help(self):
        budget = GoodputCoupledBudget(alpha=0.5, success_rate_threshold=0.4)
        self._healthy_second(budget, 0)
        self.assertTrue(_take(budget, S, "a"))  # the estimate starts at 0.5
        for i in range(3):  # 0.5 -> 0.45 -> 0.405 -> 0.3645
            budget.observe(_outcome(S + i * MS, False, tenant="a", is_retry=True))
        self.assertFalse(_take(budget, S + 10 * MS, "a"))
        # First attempts do not move the estimate; a succeeding retry does.
        budget.observe(_outcome(S + 11 * MS, True, tenant="a"))
        self.assertFalse(_take(budget, S + 12 * MS, "a"))
        budget.observe(_outcome(S + 13 * MS, True, tenant="a", is_retry=True))
        self.assertTrue(_take(budget, S + 14 * MS, "a"))

    def test_the_burst_cap_bounds_retries_by_the_request_rate(self):
        budget = GoodputCoupledBudget(alpha=0.5, max_retry_ratio=0.05)
        self._healthy_second(budget, 0)
        # alpha would allow 50; 5 % of 100 attempts per second allows 5.
        self.assertEqual(sum(_take(budget, S + i * MS, "a") for i in range(50)), 5)

    def test_a_refund_takes_the_retry_out_of_the_window(self):
        budget = GoodputCoupledBudget(alpha=0.1)
        request = _retry(0, "a")
        self.assertTrue(budget.check(request))
        budget.commit(request)
        self.assertFalse(budget.check(_retry(MS, "a")))
        budget.refund(request)
        self.assertTrue(budget.check(_retry(MS, "a")))

    def test_validation(self):
        bad: list[dict[str, Any]] = [
            {"alpha": 1.5},
            {"beta_down": 0},
            {"beta_up": 1.1},
            {"window": 0},
            {"success_rate_threshold": 2},
            {"max_retry_ratio": 0},
        ]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                GoodputCoupledBudget(**kwargs)


class TestMixedRefillBudget(unittest.TestCase):
    """Arolla's prototype filter: one bucket, refilled by successes and by time."""

    def test_the_bucket_starts_full_and_a_retry_takes_one_token(self):
        budget = MixedRefillBudget(r=0.1, capacity=3, t=0.0)
        self.assertEqual(_drain(budget, 0, "a"), 3)
        self.assertFalse(_take(budget, S, "b"))  # no time refill, and one bucket for everybody

    def test_successes_deposit_r_whoever_they_belong_to(self):
        budget = MixedRefillBudget(r=0.1, capacity=3, t=0.0)
        _drain(budget, 0)
        for i in range(9):
            budget.observe(_outcome(i, success=True, tenant="a"))
        budget.observe(_outcome(9, success=False, tenant="a"))  # a failure pays nothing
        self.assertFalse(_take(budget, 10, "b"))  # 0.9 of a token
        budget.observe(_outcome(11, success=True, tenant="a"))
        self.assertTrue(_take(budget, 12, "b"))  # paid for by a's successes
        self.assertEqual(budget.balance, 0.0)

    def test_time_refills_it_too_and_exactly(self):
        budget = MixedRefillBudget(r=0.0, capacity=5, t=2.0)
        _drain(budget, 0)
        for i in range(1, 1001):  # looked at every millisecond for a second
            budget.check(_retry(i * MS))
        self.assertEqual(budget.balance, 2.0)
        self.assertEqual(_drain(budget, 10 * S), 5)  # and never above its capacity

    def test_both_refills_fill_one_bucket(self):
        budget = MixedRefillBudget(r=0.5, capacity=10, t=1.0)
        _drain(budget, 0)
        for _ in range(4):
            budget.observe(_outcome(S // 2, success=True))
        self.assertEqual(budget.balance, 0.5 + 4 * 0.5)  # half a second, and four successes
        self.assertEqual(_drain(budget, S // 2), 2)

    def test_a_deep_retry_costs_more_in_the_fairness_filter(self):
        def retry(attempt: int) -> AdmissionRequest:
            return AdmissionRequest(now=0, tenant="a", attempt=attempt, is_retry=True)

        for mode, cost in (("linear", 4), ("triangular", 6), ("quadratic", 16)):
            with self.subTest(deep_cost=mode):
                budget = MixedRefillBudget(r=0.0, capacity=20, t=0.0, deep_from=3, deep_cost=mode)
                for attempt in (2, 3):  # up to the threshold: one token each
                    budget.commit(retry(attempt))
                self.assertEqual(budget.balance, 18.0)
                self.assertTrue(budget.check(retry(4)))
                budget.commit(retry(4))
                self.assertEqual(budget.balance, 18.0 - cost)
                budget.refund(retry(4))
                self.assertEqual(budget.balance, 18.0)
        poor = MixedRefillBudget(r=0.0, capacity=5, t=0.0, deep_from=3, deep_cost="triangular")
        self.assertFalse(poor.check(retry(4)))  # six tokens, and the bucket holds five
        self.assertTrue(poor.check(retry(3)))

    def test_checking_alone_spends_nothing(self):
        budget = MixedRefillBudget(r=0.1, capacity=2, t=0.0)
        for _ in range(5):
            self.assertTrue(budget.check(_retry(0)))
        self.assertEqual(budget.balance, 2.0)

    def test_validation(self):
        bad: list[dict[str, Any]] = [
            {"r": -0.1},
            {"capacity": 0.5},
            {"t": -1.0},
            {"r": float("nan")},
            {"deep_from": 0},
            {"deep_cost": "cubic"},
        ]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MixedRefillBudget(**kwargs)


class TestTenantRetryQuota(unittest.TestCase):
    def _firsts(self, quota, n: int, start: int = 0, tenant: str | None = "a", step: int = MS):
        for i in range(n):
            quota.commit(_first(start + i * step, tenant))

    def test_retries_are_a_share_of_the_first_attempts_of_the_window(self):
        quota = TenantRetryQuota(share=0.1)
        self.assertFalse(_take(quota, 0, "a"))  # nothing has arrived: nothing is owed
        self._firsts(quota, 30)
        self.assertEqual(_drain(quota, 40 * MS, "a"), 3)
        self.assertEqual(quota.in_window("a", 40 * MS), (30, 3))
        # The count is compared with the product as it is, as Arolla's rule does:
        # 35 first attempts are 3.5 retries, and a fourth is let in while 3 < 3.5.
        self._firsts(quota, 5, start=40 * MS)
        self.assertEqual(_drain(quota, 50 * MS, "a"), 1)

    def test_a_first_attempt_and_a_retry_leave_the_window_a_second_later(self):
        quota = TenantRetryQuota(share=0.5)
        self._firsts(quota, 2)  # at 0 and 1 ms
        self.assertTrue(_take(quota, 2 * MS, "a"))
        self.assertFalse(_take(quota, 3 * MS, "a"))
        # At 1 s the first attempt of t = 0 is still inside; a moment later it is gone,
        # and so is half a retry of allowance, while the retry of 2 ms still counts.
        self.assertEqual(quota.in_window("a", S), (2, 1))
        self.assertEqual(quota.in_window("a", S + MS // 2), (1, 1))
        self._firsts(quota, 4, start=S + MS)
        self.assertTrue(_take(quota, S + 5 * MS, "a"))  # the old retry has left as well

    def test_a_share_of_zero_lets_no_retry_in_and_a_large_one_refuses_nobody(self):
        closed, wide = TenantRetryQuota(share=0.0), TenantRetryQuota(share=3.0)
        for quota in (closed, wide):
            self._firsts(quota, 10)
        self.assertEqual(_drain(closed, 20 * MS, "a"), 0)
        self.assertEqual(_drain(wide, 20 * MS, "a"), 30)  # three retries per first attempt

    def test_a_floor_lets_that_many_retries_in_whatever_the_share(self):
        # A share of 0 with a probe: one retry per tenant and window, no more.
        quota = TenantRetryQuota(share=0.0, floor=1)
        self.assertTrue(_take(quota, 0, "a"))  # even before any first attempt
        self._firsts(quota, 10)
        self.assertEqual(_drain(quota, 20 * MS, "a"), 0)
        self.assertEqual(_drain(quota, 20 * MS, "b"), 1)  # each tenant its own
        self.assertEqual(_drain(quota, S + MS, "a"), 1)  # and the next window its own
        # Above the floor the share counts as it does without one.
        wide = TenantRetryQuota(share=0.5, floor=1)
        self._firsts(wide, 10)
        self.assertEqual(_drain(wide, 20 * MS, "a"), 5)
        for bad in (-1, 1.5, True):
            with self.subTest(floor=bad), self.assertRaisesRegex(ValueError, "floor"):
                TenantRetryQuota(floor=bad)

    def test_each_tenant_has_its_own_share_and_its_own_counts(self):
        quota = TenantRetryQuota(share=0.1)
        quota.set_share("b", 0.5)
        self.assertEqual((quota.share_of("a"), quota.share_of("b")), (0.1, 0.5))
        self._firsts(quota, 10, tenant="a")
        self._firsts(quota, 10, tenant="b")
        self.assertEqual(_drain(quota, 20 * MS, "b"), 5)
        self.assertEqual(_drain(quota, 20 * MS, "a"), 1)  # b spent nothing of a's
        self.assertEqual(_drain(quota, 20 * MS, "c"), 0)  # and c has sent nothing

    def test_a_share_with_many_decimals_is_not_rounded_down(self):
        quota = TenantRetryQuota(share=0.0867)
        self._firsts(quota, 12)
        self.assertEqual(_drain(quota, 20 * MS, "a"), 2)  # 12 x 0.0867 = 1.04: a second one fits

    def test_a_refund_takes_the_retry_out_of_the_window(self):
        quota = TenantRetryQuota(share=0.1)
        self._firsts(quota, 10)
        request = _retry(20 * MS, "a")
        self.assertTrue(quota.check(request))
        quota.commit(request)
        self.assertFalse(quota.check(_retry(21 * MS, "a")))
        quota.refund(request)
        self.assertTrue(quota.check(_retry(21 * MS, "a")))

    def test_checking_alone_spends_nothing(self):
        quota = TenantRetryQuota(share=0.1)
        self._firsts(quota, 10)
        for _ in range(5):
            self.assertTrue(quota.check(_retry(20 * MS, "a")))
        self.assertEqual(quota.in_window("a", 20 * MS), (10, 0))

    def test_validation(self):
        for bad in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(share=bad), self.assertRaises(ValueError):
                TenantRetryQuota(share=bad)
            with self.subTest(set_share=bad), self.assertRaises(ValueError):
                TenantRetryQuota().set_share("a", bad)
        with self.assertRaises(ValueError):
            TenantRetryQuota(window=0)
        quota = TenantRetryQuota(share=0.2)
        quota.set_share("a", 0.4)
        with self.assertRaises(ValueError):
            quota.set_share("a", -1.0)
        self.assertEqual(quota.share_of("a"), 0.4)  # a refused share changes nothing


class TestConcurrencyRetryBudget(unittest.TestCase):
    def test_retries_are_a_share_of_what_is_in_flight(self):
        budget = ConcurrencyRetryBudget(budget_percent=20, min_retry_concurrency=3)
        self.assertTrue(budget.check(_retry(0, outstanding=100, outstanding_retries=19)))
        self.assertFalse(budget.check(_retry(0, outstanding=100, outstanding_retries=20)))

    def test_a_minimum_is_always_allowed(self):
        budget = ConcurrencyRetryBudget(budget_percent=20, min_retry_concurrency=3)
        self.assertTrue(budget.check(_retry(0, outstanding=2, outstanding_retries=2)))
        self.assertFalse(budget.check(_retry(0, outstanding=5, outstanding_retries=3)))

    def test_at_a_gate_it_follows_the_gates_own_account(self):
        """Ten 50 ms attempts in flight, each timing out at the client after
        10 ms and retried: with 20 % and no minimum, three retries get in."""
        station = _station(median_ms=50, workers=64)
        gate = AdmissionGate(
            station, [ConcurrencyRetryBudget(budget_percent=20, min_retry_concurrency=0)]
        )
        client = _client(gate, attempts=2, attempt_ms=10, mode="global")
        sim = Simulator(seed=1)
        for _ in range(10):
            client.start_request(sim)
        sim.run()

        # 10 orphans in flight when the retries arrive. A retry needs
        # retries / in flight < 20 %: 0/10, 1/11 and 2/12 pass, 3/13 does not,
        # and neither does any later one.
        self.assertEqual(gate.counts["a"].retries_asked, 10)
        self.assertEqual(gate.counts["a"].retries_admitted, 3)
        self.assertEqual(gate.outstanding, 0)
        self.assertEqual(gate.outstanding_retries, 0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ConcurrencyRetryBudget(budget_percent=120)
        with self.assertRaises(ValueError):
            ConcurrencyRetryBudget(min_retry_concurrency=-1)


# ------------------------------------------------- the caller's own retries


class TestControlledRetryPolicy(unittest.TestCase):
    def _policy(self, controller, attempts: int = 10) -> ControlledRetryPolicy:
        return ControlledRetryPolicy(
            [controller], FixedBackoffRetryPolicy(max_attempts=attempts, delay=0)
        )

    def test_a_client_holds_its_own_retries_to_a_budget(self):
        budget = TokenBucketBudget(rate=0, burst=2)
        client = _client(_station(fail_until_ms=10_000), tenant=None, retry=self._policy(budget))
        sim = Simulator(seed=1)
        for _ in range(3):
            client.start_request(sim)
            sim.run()
        # Two retries for the first request, none left for the others.
        self.assertEqual([len(root.attempts) for root in client.roots], [3, 1, 1])
        self.assertEqual(client.metrics().summary().dropped_server_failure, 3)

    def test_clients_share_a_budget_by_sharing_the_controller(self):
        budget = TokenBucketBudget(rate=0, burst=2)
        station = _station(fail_until_ms=10_000)
        one = _client(station, tenant=None, retry=self._policy(budget))
        two = _client(station, tenant=None, retry=self._policy(budget))
        sim = Simulator(seed=1)
        one.start_request(sim)
        two.start_request(sim)
        sim.run()
        self.assertEqual(len(one.roots[0].attempts) + len(two.roots[0].attempts), 2 + 2)

    def test_its_own_outcomes_fund_it(self):
        budget = SuccessFundedBudget(ratio=0.5, max_retries=1)
        # Healthy until 100 ms, failing after.
        station = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=MS,
                latency_lognorm_sigma=0.0,
                workers=8,
                partial_failures=[PartialFailure(TimeInterval(100 * MS, 10 * S), p_fail=1.0)],
            )
        ).bind()
        client = _client(station, tenant=None, retry=self._policy(budget, attempts=2))
        sim = Simulator(seed=1)
        sim.schedule(100 * MS, lambda: client.start_request(sim))  # spends the opening retry
        sim.schedule(120 * MS, lambda: client.start_request(sim))  # nothing in the bank
        sim.run()
        self.assertEqual([len(root.attempts) for root in client.roots], [2, 1])

        budget = SuccessFundedBudget(ratio=0.5, max_retries=1)
        client = _client(station.bind(), tenant=None, retry=self._policy(budget, attempts=2))
        sim = Simulator(seed=1)
        sim.schedule(100 * MS, lambda: client.start_request(sim))
        sim.run()
        self.assertEqual(budget.balance, 0.0)
        station.bind()
        sim = Simulator(seed=1)
        for i in range(2):  # two successes at 0.5 each: one retry
            sim.schedule(i * 10 * MS, lambda: client.start_request(sim))
        sim.run()
        self.assertEqual(budget.balance, 1.0)

    def test_a_retry_the_deadline_vetoes_is_refunded(self):
        budget = TokenBucketBudget(rate=0, burst=1)
        # The attempt is cut at the 10 ms global deadline, so the retry the
        # policy grants would start at the deadline: it is never issued.
        client = _client(
            _station(median_ms=50),
            tenant=None,
            retry=self._policy(budget),
            attempt_ms=1000,
            global_ms=10,
        )
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()
        self.assertEqual(len(client.roots[0].attempts), 1)
        self.assertEqual(budget.balance, 1.0)

    def test_the_wrapped_policy_must_agree_too(self):
        budget = TokenBucketBudget(rate=0, burst=5)
        client = _client(
            _station(fail_until_ms=10_000), tenant=None, retry=self._policy(budget, attempts=2)
        )
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()
        self.assertEqual(len(client.roots[0].attempts), 2)
        self.assertEqual(budget.balance, 4.0)  # charged for the one retry it granted

    def test_a_controller_that_needs_the_servers_load_refuses_to_work_here(self):
        client = _client(
            _station(fail_until_ms=10_000),
            tenant=None,
            retry=self._policy(ConcurrencyRetryBudget()),
        )
        sim = Simulator(seed=1)
        client.start_request(sim)
        with self.assertRaises(ValueError) as raised:
            sim.run()
        self.assertIn("AdmissionGate", str(raised.exception))

    def test_the_tenant_reaches_the_controller(self):
        spy = _Spy()
        client = _client(_station(fail_until_ms=10_000), tenant="acme", retry=self._policy(spy, 2))
        sim = Simulator(seed=1)
        client.start_request(sim)
        sim.run()
        self.assertEqual({o.tenant for o in spy.observed}, {"acme"})
        self.assertEqual([o.is_retry for o in spy.observed], [False, True])
        self.assertEqual(
            [(r.tenant, r.attempt, r.is_retry) for r in spy.committed], [("acme", 2, True)]
        )


# ------------------------------------------------------------- in a whole run


def _storm(gated: bool, seed: int = 31):
    """Two tenants on one station that fails for a second; retries without backoff."""
    sim = Simulator(seed=seed)
    station = ServiceRuntime(
        cfg=ServiceConfig(
            name="svc",
            latency_median=ms_to_ns(8),
            latency_lognorm_sigma=0.5,
            workers=4,
            queue_capacity=20,
            partial_failures=[PartialFailure(TimeInterval(S, 2 * S), p_fail=0.6)],
        )
    ).bind()
    gate = AdmissionGate(station, [TokenBucketBudget(rate=20, burst=5)]) if gated else None
    clients = []
    for index, tenant in enumerate(("a", "b")):
        client = _client(
            gate if gate is not None else station,
            tenant=tenant,
            attempts=4,
            attempt_ms=60,
        )
        Workload(base_rps=150.0, duration_s=3.0, load_spikes=[], rng_seed=seed + index).drive(
            sim, client.start_request
        )
        clients.append(client)
    sim.run()
    return sim, clients, station, gate


class TestAGateInAWholeRun(unittest.TestCase):
    def _csv(self, client: ClientRuntime) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "out.csv")
            client.metrics().export_csv(path)
            return pd.read_csv(path)

    def test_without_a_gate_nothing_is_rejected_and_no_column_appears(self):
        _, clients, _, _ = _storm(gated=False)
        for client in clients:
            self.assertEqual(client.metrics().summary().dropped_rejected, 0)
            self.assertNotIn("failure_rejected", self._csv(client).columns)

    def test_with_a_gate_the_column_appears_and_the_books_balance(self):
        sim, clients, station, gate = _storm(gated=True)
        assert gate is not None
        rejected = 0
        for client in clients:
            summary = client.metrics().summary()
            df = self._csv(client)
            self.assertGreater(summary.dropped_rejected, 0)
            self.assertEqual(df.columns[-1], "failure_rejected")
            reasons = [reporting.csv_column for reporting in FAILURE_OUTCOMES.values()]
            self.assertEqual(int(df[reasons].sum().sum()), int(df["failure_root"].sum()))
            for reporting in FAILURE_OUTCOMES.values():
                self.assertEqual(
                    int(df[reporting.csv_column].sum()), getattr(summary, reporting.summary_field)
                )
            rejected += sum(
                attempt.drop_reason is DropReason.REJECTED
                for root in client.roots
                for attempt in root.attempts
            )
        counts = list(gate.counts.values())
        self.assertEqual(sum(c.rejected for c in counts), rejected)
        self.assertEqual(
            sum(c.retries_asked for c in counts),
            sum(c.retries_admitted for c in counts) + rejected,
        )
        # What the gate let in is what arrived at the station (plus what its queue shed).
        self.assertGreaterEqual(
            sum(c.attempts - c.rejected for c in counts), station.admitted_total
        )
        self.assertEqual(gate.outstanding, 0)
        self.assertEqual(gate.outstanding_retries, 0)

    def test_the_gate_takes_load_off_the_station(self):
        _, _, open_station, _ = _storm(gated=False)
        _, _, gated_station, _ = _storm(gated=True)
        self.assertLess(gated_station.admitted_total, open_station.admitted_total)


if __name__ == "__main__":
    unittest.main()
