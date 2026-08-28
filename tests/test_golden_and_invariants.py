"""Characterization and invariant tests.

Two kinds of test live here, both aimed at the class of defect the 2026-07
review found by mutation: behaviour that no assertion pinned.

* GOLDEN tests pin exact results for a seeded end-to-end run. They are brittle
  BY DESIGN. A diff means the event loop, the RNG consumption order, or a
  modelling rule changed — which invalidates every previously published figure,
  so it must be a deliberate decision and the numbers regenerated, never
  rubber-stamped.
* INVARIANT tests pin documented rules that should survive any such change.
"""

import os
import tempfile
import unittest

import pandas as pd

from caller import ClientConfig, ClientRuntime
from chain import DownstreamEdge, ServiceNodeRuntime
from core import DropReason, TimeInterval, ms_to_ns, s_to_ns
from faults import LatencyInjection, PartialFailure
from metrics import Metrics
from retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy
from service import MIN_LOGNORM_SIGMA, ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload


def _single_service_run(seed=1234, duration_s=10.0):
    sim = Simulator(seed=seed)
    svc = ServiceRuntime(
        cfg=ServiceConfig(
            name="svc-A",
            latency_median=ms_to_ns(10),
            latency_lognorm_sigma=0.5,
            workers=2,
            queue_capacity=100,
        )
    ).bind()
    client = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(20)),
            timeout_controller=StaticTimeoutController(
                global_timeout=ms_to_ns(500), attempt_timeout=ms_to_ns(200)
            ),
        ),
        service=svc,
    )
    wl = Workload(base_rps=180.0, duration_s=duration_s, load_spikes=[], rng_seed=seed)
    wl.drive(sim, client.start_request)
    sim.run()
    return client


def _chain_run(seed=1234, duration_s=6.0):
    sim = Simulator(seed=seed)

    def node(name, median_ms, workers):
        return ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name=name,
                    latency_median=ms_to_ns(median_ms),
                    latency_lognorm_sigma=0.4,
                    workers=workers,
                    queue_capacity=200,
                )
            ).bind()
        )

    a, b, c = node("A", 5, 8), node("B", 8, 6), node("C", 12, 4)
    for parent, child in ((a, b), (b, c)):
        parent.add_downstream(
            DownstreamEdge(
                client=ClientRuntime(
                    cfg=ClientConfig(
                        retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(15)),
                        timeout_controller=StaticTimeoutController(None, ms_to_ns(150)),
                    ),
                    service=child,
                ),
                target=child,
            )
        )
    top = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(20)),
            timeout_controller=StaticTimeoutController(None, ms_to_ns(400)),
        ),
        service=a,
    )
    wl = Workload(base_rps=120.0, duration_s=duration_s, load_spikes=[], rng_seed=seed)
    wl.drive(sim, top.start_request)
    sim.run()
    return top


def _fanout_run(seed=1234, duration_s=6.0):
    """A fans out to three leaves with near-identical service times, so many
    downstream completions land on the SAME simulated instant. That is what
    makes the (at, seq) tie-break observable — a linear chain has almost no ties.
    """
    sim = Simulator(seed=seed)

    def node(name, median_ms, workers):
        return ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name=name,
                    latency_median=ms_to_ns(median_ms),
                    latency_lognorm_sigma=1e-6,  # near-deterministic => ties
                    workers=workers,
                    queue_capacity=3,
                )
            ).bind()
        )

    a = node("A", 2, 4)
    for i in range(3):
        # Tight capacity + a partial-failure window so that when several
        # completions share an instant, WHICH one is processed first decides who
        # gets the last queue slot and who is dropped. Without contention the
        # tie order is unobservable in the aggregate.
        leaf = node(f"L{i}", 10, 1)
        leaf.service.cfg.partial_failures.append(
            PartialFailure(TimeInterval(s_to_ns(2), s_to_ns(4)), p_fail=0.3)
        )
        a.add_downstream(
            DownstreamEdge(
                client=ClientRuntime(
                    cfg=ClientConfig(
                        retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(5)),
                        timeout_controller=StaticTimeoutController(None, ms_to_ns(60)),
                    ),
                    service=leaf,
                ),
                target=leaf,
            )
        )
    top = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(10)),
            timeout_controller=StaticTimeoutController(None, ms_to_ns(300)),
        ),
        service=a,
    )
    wl = Workload(base_rps=150.0, duration_s=duration_s, load_spikes=[], rng_seed=seed)
    wl.drive(sim, top.start_request)
    sim.run()
    return top


# Captured from the current implementation. A diff means the simulator's
# numbers moved; see the module docstring before touching them.
_FANOUT_GOLDEN = (866, 455, 379, 0, 1441)
_FANOUT_P50 = 45.20


class TestGoldenSeededRuns(unittest.TestCase):
    """If one of these fails, the simulator's numbers moved.

    Do not update the expected values without establishing WHY they moved. A
    change to the event loop, the RNG consumption order, or a modelling rule
    invalidates every published figure produced with the old behaviour.
    """

    def test_single_service_golden(self):
        summary = _single_service_run().metrics().summary()
        self.assertEqual(
            (
                summary.total,
                summary.succeeded,
                summary.dropped_queue,
                summary.dropped_deadline,
                summary.dropped_server_failure,
                summary.attempts_total,
            ),
            # Captured from the current implementation, not derived. The point
            # is that they cannot move silently. A 2-worker service at 180 rps
            # with a 10 ms median is deliberately overloaded, so the deadline
            # drops and the retries are expected.
            (1780, 1502, 0, 278, 0, 2251),
        )
        self.assertAlmostEqual(summary.p50, 130.38, places=2)
        self.assertAlmostEqual(summary.p99, 418.96, places=2)

    def test_chain_golden(self):
        summary = _chain_run().metrics().summary()
        self.assertEqual(
            (summary.total, summary.succeeded, summary.attempts_total),
            (680, 680, 680),
        )
        self.assertAlmostEqual(summary.p50, 26.29, places=2)
        self.assertAlmostEqual(summary.p99, 47.34, places=2)

    def test_fanout_golden(self):
        """Also guards the event-heap tie-break: with near-deterministic service
        times, many completions share a timestamp, so reversing the `seq`
        ordering changes which one is processed first."""
        summary = _fanout_run().metrics().summary()
        self.assertEqual(
            (
                summary.total,
                summary.succeeded,
                summary.dropped_queue,
                summary.dropped_deadline,
                summary.attempts_total,
            ),
            _FANOUT_GOLDEN,
        )
        self.assertAlmostEqual(summary.p50, _FANOUT_P50, places=2)


class TestServiceDisciplineInvariants(unittest.TestCase):
    def test_service_is_fcfs(self):
        """Replacing the heap pop with a LIFO pop left all 177 tests green while
        moving p99 from 425 ms to 6950 ms. Pin the ordering directly."""
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(10), latency_lognorm_sigma=1e-6, workers=1
            )
        ).bind()
        finished: list[str] = []
        for tag in ("A", "B", "C"):
            svc.submit_attempt(sim, lambda *_a, _t=tag: finished.append(_t))
        sim.run()
        self.assertEqual(finished, ["A", "B", "C"])

    def test_late_arrival_still_finishes_last(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(10), latency_lognorm_sigma=1e-6, workers=1
            )
        ).bind()
        finished: list[str] = []
        for tag in ("A", "B"):
            svc.submit_attempt(sim, lambda *_a, _t=tag: finished.append(_t))
        sim.run(until=ms_to_ns(5))  # A in service, B queued
        svc.submit_attempt(sim, lambda *_a: finished.append("D"))
        sim.run()
        self.assertEqual(finished, ["A", "B", "D"])


class TestLatencyInjectionInvariants(unittest.TestCase):
    """Deleting the additive term (`add += 0`) left the whole suite green."""

    def _sample(self, sim, svc, at_ns):
        return svc._sample_service_time(at_ns, sim)

    def test_additive_latency_applies_inside_its_window(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(10),
                latency_lognorm_sigma=1e-6,
                workers=1,
                latency_injections=[
                    LatencyInjection(
                        duration=TimeInterval(s_to_ns(2), s_to_ns(6)),
                        add_latency=ms_to_ns(40),
                    )
                ],
            )
        ).bind()
        inside = self._sample(sim, svc, s_to_ns(3))
        outside = self._sample(sim, svc, s_to_ns(8))
        self.assertAlmostEqual(inside / ms_to_ns(1), 50.0, delta=0.1)
        self.assertAlmostEqual(outside / ms_to_ns(1), 10.0, delta=0.1)

    def test_multiplier_applies_before_the_additive_term(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(10),
                latency_lognorm_sigma=1e-6,
                workers=1,
                latency_injections=[
                    LatencyInjection(
                        duration=TimeInterval(0, s_to_ns(10)),
                        add_latency=ms_to_ns(5),
                        multiplier=2,
                    )
                ],
            )
        ).bind()
        # base * multiplier + add == 10*2 + 5
        self.assertAlmostEqual(self._sample(sim, svc, s_to_ns(1)) / ms_to_ns(1), 25.0, delta=0.1)


class TestFaultFinitenessValidation(unittest.TestCase):
    """NaN and inf pass every ordering guard, then corrupt the physics silently.

    `LatencyInjection(iv, add_latency=nan)` made `_adjust_latency` compute
    `max(0, base * mult + nan)`, which returns 0 (two-arg `max` keeps the left
    operand when the comparison is False), floored to the 100 us minimum: a
    fault configured to ADD 40 ms made a 50 ms service ~500x FASTER for the
    whole window, exported and plotted as a latency injection all the while. An
    infinite value produced an infinite service time and crashed
    `Simulator.schedule`'s `int(at)` mid-run.

    Both fields are declared `int`, so `ty` rejects these calls statically —
    reaching the defect at all takes an untyped caller (a config loader, a
    runtime draw), which is exactly the population the guard is for.
    """

    def test_non_finite_latency_injection_fields_are_rejected(self):
        window = TimeInterval(0, s_to_ns(1))
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(add_latency=bad), self.assertRaises(ValueError):
                LatencyInjection(duration=window, add_latency=bad)  # ty: ignore[invalid-argument-type]
            with self.subTest(multiplier=bad), self.assertRaises(ValueError):
                LatencyInjection(duration=window, multiplier=bad)  # ty: ignore[invalid-argument-type]

    def test_finite_fields_still_construct(self):
        window = TimeInterval(0, s_to_ns(1))
        inj = LatencyInjection(duration=window, add_latency=ms_to_ns(40), multiplier=2)
        self.assertEqual((inj.add_latency, inj.multiplier), (ms_to_ns(40), 2))


class TestServiceConfigOwnsItsRanges(unittest.TestCase):
    """One authority per rule: the sampler reads the config, it does not re-clamp.

    The lognormal sigma floor is load-bearing, not defensive — every pinned
    seeded result was produced with it — so it lives on the config, where
    `latency_lognorm_sigma` reports the value that will actually be sampled.
    """

    def test_sub_floor_sigma_is_normalized_at_construction(self):
        cfg = ServiceConfig(
            name="svc", latency_median=ms_to_ns(50), latency_lognorm_sigma=1e-9, workers=1
        )
        self.assertEqual(cfg.latency_lognorm_sigma, MIN_LOGNORM_SIGMA)

    def test_sigma_above_the_floor_is_untouched(self):
        cfg = ServiceConfig(
            name="svc", latency_median=ms_to_ns(50), latency_lognorm_sigma=0.5, workers=1
        )
        self.assertEqual(cfg.latency_lognorm_sigma, 0.5)

    def test_sub_nanosecond_median_is_rejected_not_clamped(self):
        with self.assertRaises(ValueError):
            ServiceConfig(name="svc", latency_median=0, latency_lognorm_sigma=0.5, workers=1)


class _FinalizeAtOffsetService:
    """A `ServiceLike` whose attempts finalize SUCCESS at a fixed offset from the
    attempt deadline, through the custom-`behavior` path.

    `ServiceRuntime` deadline-enforces only its DEFAULT behavior
    (`_finish_service`), so on this path nothing but the client's half-open rule
    can reclassify a completion that lands exactly on the deadline.
    """

    def __init__(self, service: ServiceRuntime, offset_ns: int) -> None:
        self.service = service
        self.cfg = service.cfg
        self.offset_ns = offset_ns

    def submit_attempt(self, sim, on_done, attempt_deadline=None):
        def behavior(service, sim_, deadline, finalize, begin_at, release_worker, register_cancel):
            if deadline is None:
                raise AssertionError("this fixture needs an attempt deadline")

            def finish() -> None:
                finalize(True, sim_.timestep - begin_at, DropReason.NONE)

            sim_.schedule(deadline + self.offset_ns, finish)

        return self.service.submit_attempt(
            sim, on_done, attempt_deadline=attempt_deadline, behavior=behavior
        )


class TestStrictDeadlineInvariant(unittest.TestCase):
    """CLAUDE.md calls this load-bearing, and the review found it untested:
    a root finishing EXACTLY at its deadline is a drop, not a success."""

    def test_completion_exactly_at_the_deadline_is_a_drop(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(50),  # sigma ~0 => service takes exactly 50 ms
                latency_lognorm_sigma=1e-9,
                workers=1,
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                # Deadline lands exactly on the completion instant.
                timeout_controller=StaticTimeoutController(None, ms_to_ns(50)),
            ),
            service=svc,
        )
        client.start_request(sim)
        sim.run()
        root = client.roots[0]
        self.assertFalse(root.attempts[-1].success)
        self.assertEqual(root.attempts[-1].drop_reason, DropReason.DEADLINE)

    def test_completion_strictly_before_the_deadline_succeeds(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(50), latency_lognorm_sigma=1e-9, workers=1
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(None, ms_to_ns(51)),
            ),
            service=svc,
        )
        client.start_request(sim)
        sim.run()
        self.assertTrue(client.roots[0].attempts[-1].success)

    def _custom_behavior_attempt(self, offset_ns: int):
        """One attempt that reports SUCCESS `offset_ns` from its deadline."""
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(10), latency_lognorm_sigma=1e-9, workers=1
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(None, ms_to_ns(50)),
            ),
            service=_FinalizeAtOffsetService(svc, offset_ns),
        )
        client.start_request(sim)
        sim.run()
        return client.roots[0].attempts[-1]

    def test_custom_behavior_success_at_the_deadline_is_a_drop(self):
        """The CLIENT's rule, not the service's. The two cases above are decided
        by `ServiceRuntime._finish_service`, so they hold even with the client
        check inverted; a custom behavior is not deadline-enforced by the
        service, which is exactly the extension point CLAUDE.md warns about."""
        attempt = self._custom_behavior_attempt(offset_ns=0)
        self.assertFalse(attempt.success)
        self.assertEqual(attempt.drop_reason, DropReason.DEADLINE)

    def test_custom_behavior_success_one_ns_earlier_stands(self):
        attempt = self._custom_behavior_attempt(offset_ns=-1)
        self.assertTrue(attempt.success)
        self.assertEqual(attempt.drop_reason, DropReason.NONE)


class TestExportCsvBucketSemantics(unittest.TestCase):
    """The documented start-vs-end bucketing split had no per-bucket assertion,
    so it could be inverted without failing anything."""

    def _export(self, client, granularity_s=1.0):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.csv")
            client.metrics().export_csv(path, granularity_s=granularity_s)
            return pd.read_csv(path)

    def test_roots_are_bucketed_by_start_and_completions_by_end(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                # 1.5 s of service: the root starts in bucket 0 and ends in 1.
                latency_median=s_to_ns(1.5),
                latency_lognorm_sigma=1e-9,
                workers=1,
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(None, s_to_ns(10)),
            ),
            service=svc,
        )
        client.start_request(sim)
        sim.run()
        df = self._export(client)
        self.assertEqual(int(df.loc[0, "root_requests"]), 1)  # start bucket
        self.assertEqual(int(df.loc[0, "success_root"]), 0)
        self.assertEqual(int(df.loc[1, "root_requests"]), 0)
        self.assertEqual(int(df.loc[1, "success_root"]), 1)  # end bucket

    def test_retries_are_bucketed_by_start_and_failed_retries_by_end(self):
        """The other half of the documented split, and the half no assertion
        reached: `retries` is bucketed at the retry's START, `failure_retry`
        counts only attempts AFTER the first, at their END. This run separates
        all three instants — a first attempt that fails, a retry that starts in
        one bucket and fails in the next, and a retry that succeeds."""
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=s_to_ns(0.5),
                latency_lognorm_sigma=1e-9,
                workers=1,
                # Attempts 1 and 2 end inside the window and fail; attempt 3 ends
                # after it and succeeds.
                partial_failures=[PartialFailure(TimeInterval(0, s_to_ns(3)), p_fail=1.0)],
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=3, delay=s_to_ns(1.2)),
                timeout_controller=StaticTimeoutController(s_to_ns(10), s_to_ns(5)),
            ),
            service=svc,
        )
        client.start_request(sim)
        sim.run()
        # attempt 1 [0.0, 0.5) fails, retry 1 [1.7, 2.2) fails, retry 2 [3.4, 3.9) succeeds.
        self.assertEqual(
            [(a.interval.begin, a.interval.end, a.success) for a in client.roots[0].attempts],
            [
                (0, 500_000_303, False),
                (1_700_000_303, 2_200_000_178, False),
                (3_400_000_178, 3_900_000_685, True),
            ],
        )
        df = self._export(client)
        self.assertEqual(list(df["root_requests"]), [1, 0, 0, 0])  # start bucket
        self.assertEqual(list(df["retries"]), [0, 1, 0, 1])  # retry START buckets
        # Only the failed RETRY counts, and at its end: the first attempt failed
        # in bucket 0 and the retry that failed started in bucket 1.
        self.assertEqual(list(df["failure_retry"]), [0, 0, 1, 0])
        self.assertEqual(list(df["completed"]), [1, 0, 1, 1])  # attempt END buckets
        self.assertEqual(list(df["success_root"]), [0, 0, 0, 1])  # root END bucket

    def test_latency_columns_cover_successful_roots_only(self):
        client = _single_service_run()
        df = self._export(client)
        summary = client.metrics().summary()
        # Every bucket with a successful completion must report a p50; buckets
        # with none must report nan rather than 0.
        for _, row in df.iterrows():
            if row["success_root"] > 0:
                self.assertFalse(pd.isna(row["p50"]))
            else:
                self.assertTrue(pd.isna(row["p50"]))
        self.assertEqual(int(df["success_root"].sum()), summary.succeeded)

    def test_summary_and_csv_agree_on_totals(self):
        client = _single_service_run()
        summary = client.metrics().summary()
        df = self._export(client)
        self.assertEqual(int(df["root_requests"].sum()), summary.total)
        self.assertEqual(
            int(df["success_root"].sum()) + int(df["failure_root"].sum()), summary.total
        )


class TestRootClassificationAgreement(unittest.TestCase):
    """Every consumer must classify the same population identically.

    `Metrics.summary` and `Metrics.export_csv` each used to own these rules, and
    a third consumer that owned them too tested `cancelled` after the drop-reason
    chain instead of before it — booking the same root as cancelled in the CSV
    and as a deadline drop in its own accounting. `classify_root` is now the one
    rule; this asserts the property directly rather than testing each site
    separately.
    """

    def _population(self):
        """One root of every shape the classifier must distinguish."""
        from core import Request, RootRequest

        def root(reason, *, success=False, cancelled=False, attempts=1, completed_at=None):
            r = RootRequest(cancelled=cancelled, completed_at=completed_at)
            for i in range(attempts):
                last = i == attempts - 1
                r.add_attempt(
                    Request(
                        service="svc",
                        interval=TimeInterval(begin=ms_to_ns(10 * i), end=ms_to_ns(10 * i + 5)),
                        success=success and last,
                        drop_reason=reason if last else DropReason.SERVER_FAILURE,
                    )
                )
            r.done = True
            if completed_at is None:
                r.completed_at = r.attempts[-1].interval.end
            return r

        return [
            root(DropReason.NONE, success=True),
            root(DropReason.NONE, success=True, attempts=3),  # succeeded after retries
            root(DropReason.QUEUE_FULL),
            root(DropReason.DEADLINE),
            root(DropReason.SERVER_FAILURE),
            root(DropReason.CIRCUIT_OPEN),
            root(DropReason.CANCELLED, cancelled=True),
            # The shape that caused the drift: cancelled during backoff, so the
            # final attempt still carries DEADLINE and completion is later.
            root(DropReason.DEADLINE, cancelled=True, completed_at=ms_to_ns(500)),
            root(DropReason.QUEUE_FULL, cancelled=True, completed_at=ms_to_ns(500)),
        ]

    def test_summary_and_csv_agree_on_every_outcome(self):
        """Both consumers dispatch on `classify_root`, so what this still catches
        is the reporting half — the `FAILURE_OUTCOMES` table and the bucket range
        (the `completed_at=500 ms` roots in `_population`) — not the rule itself,
        which `test_classifier_rules_are_correct_not_merely_consistent` pins."""
        roots = self._population()
        summary = Metrics(roots=roots, attempts_total=sum(len(r.attempts) for r in roots)).summary()

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.csv")
            Metrics(roots=roots, attempts_total=1).export_csv(path, granularity_s=1.0)
            df = pd.read_csv(path)

        for label, from_summary, csv_col in (
            ("success", summary.succeeded, "success_root"),
            ("deadline", summary.dropped_deadline, "failure_deadline"),
            ("queue", summary.dropped_queue, "failure_queue_full"),
            ("server", summary.dropped_server_failure, "failure_server"),
            ("circuit", summary.dropped_circuit_open, "failure_circuit_open"),
            ("cancelled", summary.dropped_cancelled, "failure_cancelled"),
        ):
            with self.subTest(category=label):
                self.assertEqual(
                    from_summary,
                    int(df[csv_col].sum()),
                    f"summary() and export_csv disagree on {label}",
                )

    def test_classifier_rules_are_correct_not_merely_consistent(self):
        """Agreement alone is not enough once the rule is shared.

        With one classifier behind every consumer, a WRONG rule is wrong
        everywhere and the agreement test still passes. Pin the rule itself —
        in particular that `cancelled` beats the final attempt's drop reason,
        which is the ordering that drifted before it was centralised.
        """
        from core import Request, RootOutcome, RootRequest, classify_root

        def root(reason, *, success=False, cancelled=False):
            r = RootRequest(cancelled=cancelled)
            r.add_attempt(
                Request(
                    service="svc",
                    interval=TimeInterval(begin=0, end=ms_to_ns(5)),
                    success=success,
                    drop_reason=reason,
                )
            )
            return r

        # The shape that caused the drift: cancelled during backoff, so the last
        # attempt still says DEADLINE. Cancelled must win.
        self.assertEqual(
            classify_root(root(DropReason.DEADLINE, cancelled=True)), RootOutcome.CANCELLED
        )
        self.assertEqual(
            classify_root(root(DropReason.QUEUE_FULL, cancelled=True)), RootOutcome.CANCELLED
        )
        # Success outranks everything, including the cancelled flag.
        self.assertEqual(
            classify_root(root(DropReason.NONE, success=True, cancelled=True)),
            RootOutcome.SUCCESS,
        )
        # Plain mappings.
        for reason, expected in (
            (DropReason.QUEUE_FULL, RootOutcome.QUEUE_FULL),
            (DropReason.DEADLINE, RootOutcome.DEADLINE),
            (DropReason.SERVER_FAILURE, RootOutcome.SERVER_FAILURE),
            (DropReason.CIRCUIT_OPEN, RootOutcome.CIRCUIT_OPEN),
            (DropReason.CANCELLED, RootOutcome.CANCELLED),
        ):
            with self.subTest(reason=reason):
                self.assertEqual(classify_root(root(reason)), expected)

    def test_every_root_lands_in_exactly_one_category(self):
        roots = self._population()
        summary = Metrics(roots=roots, attempts_total=1).summary()
        self.assertEqual(
            summary.succeeded
            + summary.dropped_queue
            + summary.dropped_deadline
            + summary.dropped_server_failure
            + summary.dropped_circuit_open
            + summary.dropped_cancelled,
            summary.total,
        )


if __name__ == "__main__":
    unittest.main()
