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
from retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy
from service import ServiceConfig, ServiceRuntime
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


if __name__ == "__main__":
    unittest.main()
