from dataclasses import dataclass

import numpy as np

from msim.core import (
    FAILURE_OUTCOMES,
    MS_TO_NS,
    S_TO_NS,
    DropReason,
    RootOutcome,
    RootRequest,
    Summary,
    classify_root,
    ns_to_ms,
    root_completed_at,
    root_latency,
    root_started_at,
    s_to_ns,
)


@dataclass
class Metrics:
    roots: list[RootRequest]
    # LIFETIME attempt counter as supplied by the constructing client(s). A
    # different population from `roots`: clients bump it at attempt START while
    # roots are appended at COMPLETION, so it over-reports whenever work is
    # still in flight (a mid-run export, a summary over a sub-window). Read
    # it through `client_attempts_total()`; every `Summary` field, including
    # `attempts_total`, is computed over `roots` instead.
    attempts_total: int = 0

    def attempts_on_roots(self) -> int:
        """Attempts recorded on the roots being summarized."""
        return sum(len(request.attempts) for request in self.roots)

    def client_attempts_total(self) -> int:
        """The constructing client's lifetime attempt counter (start-time population)."""
        return self.attempts_total

    def _percentile_ms(self, p: float, vals: list[int]) -> float:
        if not vals:
            return float("nan")

        arr = np.asarray(vals, dtype=np.int64)
        q_ns = np.quantile(arr, p)
        return float(q_ns) / MS_TO_NS

    def summary(self) -> Summary:
        total = len(self.roots)
        succ_lat: list[int] = []
        succeeded = 0
        # Derived from the one authoritative table, so a new RootOutcome is
        # counted here the moment it exists instead of raising KeyError below.
        counters = dict.fromkeys(FAILURE_OUTCOMES, 0)
        for request in self.roots:
            if not request.attempts:
                continue
            if sum(1 for attempt in request.attempts if attempt.success) > 1:
                raise ValueError("Multiple successful attempts in RootRequest")

            outcome = classify_root(request)
            if outcome is RootOutcome.SUCCESS:
                succeeded += 1
                succ_lat.append(root_latency(request))
                continue
            counters[outcome] += 1

        # Each outcome goes to the Summary field the table names for it; core
        # validates those field names against Summary at import time.
        dropped = {
            reporting.summary_field: counters[outcome]
            for outcome, reporting in FAILURE_OUTCOMES.items()
        }

        # nan, not 0.0: "no successful request completed" is missing data, and
        # the percentiles below already use nan. Reporting 0 ms made an empty
        # window look like a perfect one, and a consumer averaging the column
        # silently pulled its mean toward zero.
        mean_ms = ns_to_ms(int(sum(succ_lat) / len(succ_lat))) if succ_lat else float("nan")
        max_ms = ns_to_ms(max(succ_lat)) if succ_lat else float("nan")
        # Count attempts on the roots we are summarizing, not the client's
        # lifetime counter: those disagree whenever work is still in flight
        # (mid-run exports, summaries over a sub-window), and every other
        # field here is computed over completed roots.
        attempts_on_roots = self.attempts_on_roots()
        retries_per_root = (attempts_on_roots - total) / total if total > 0 else 0.0
        return Summary(
            total=total,
            succeeded=succeeded,
            **dropped,
            p50=self._percentile_ms(0.5, succ_lat),
            p90=self._percentile_ms(0.9, succ_lat),
            p95=self._percentile_ms(0.95, succ_lat),
            p99=self._percentile_ms(0.99, succ_lat),
            p99_9=self._percentile_ms(0.999, succ_lat),
            p99_99=self._percentile_ms(0.9999, succ_lat),
            mean=mean_ms,
            max=max_ms,
            retries_per_root=retries_per_root,
            attempts_total=attempts_on_roots,
        )

    def print_stats(self):
        summary = self.summary()
        print(f"Total Requests: {summary.total}")
        print(f"Succeeded: {summary.succeeded}")
        for reporting in FAILURE_OUTCOMES.values():
            print(f"{reporting.label}: {getattr(summary, reporting.summary_field)}")
        print(f"P50 Latency (ms): {summary.p50:.2f}")
        print(f"P95 Latency (ms): {summary.p95:.2f}")
        print(f"P99 Latency (ms): {summary.p99:.2f}")
        print(f"P99.9 Latency (ms): {summary.p99_9:.2f}")
        print(f"P99.99 Latency (ms): {summary.p99_99:.2f}")
        print(f"Mean Latency (ms): {summary.mean:.2f}")
        print(f"Max Latency (ms): {summary.max:.2f}")
        print(f"Retries per Root Request: {summary.retries_per_root:.2f}")
        print(f"Total Attempts: {summary.attempts_total}")

    def export_csv(self, path: str, granularity_s: float = 1.0) -> None:
        """
        Notes:
        - failures are root-level, attributed to the final attempt's drop_reason,
        bucketed by the root's completion time (last attempt end).
        - failure_retry counts failed RETRY attempts (after the first), bucketed by attempt end.
        - latencies are for successful roots completed in the bucket.
        """
        # Lazy import: pandas is a dev dependency; the simulation core stays
        # numpy+stdlib-only until export time.
        import pandas as pd

        if granularity_s <= 0:
            raise ValueError("granularity_s must be > 0")
        bucket_ns = s_to_ns(granularity_s)
        if bucket_ns <= 0:
            raise ValueError(
                f"granularity_s={granularity_s!r} rounds to {bucket_ns} ns; "
                "the minimum representable bucket is 1 ns (1e-9 s)"
            )

        columns = [
            "timepoint",
            "root_requests",  # bucketed at the start
            "retries",  # bucketed at the start
            "success_root",  # roots that succeeded, bucketed at the end
            "completed",  # all attempts completed, bucketed at the end
            "failure_root",
            "failure_retry",
            # Per-reason breakdown, in the table's order — which is what fixes
            # this header, so appending an outcome is safe and reordering the
            # table rewrites every exported CSV. A new RootOutcome gets its
            # column here automatically (plotting still needs its own list).
            *(reporting.csv_column for reporting in FAILURE_OUTCOMES.values()),
            "total_request",  # cumulative root_requests, bucketed at the start
            "total_failure",  # cumulative failure_root, bucketed at the end
            "p50",  # percentiles are all bucketed at the end
            "p90",
            "p95",
            "p99",
            "p99.9",
            "p99.99",
            "min",
            "max",
            "queue_avg_at_attempt_end",
        ]

        # One row per attempt (retry order per root) and one per root; failed
        # roots are attributed to the final attempt's drop_reason.
        attempt_rows: list[dict] = []
        root_rows: list[dict] = []
        for root in self.roots:
            if not root.attempts:
                continue
            attempts_by_begin = sorted(root.attempts, key=lambda a: a.interval.begin)
            outcome = classify_root(root)
            root_rows.append(
                {
                    "begin": root_started_at(root),
                    "end": root_completed_at(root),
                    "succeeded": outcome is RootOutcome.SUCCESS,
                    "outcome": outcome,
                }
            )
            for i, a in enumerate(attempts_by_begin):
                attempt_rows.append(
                    {
                        "begin": a.interval.begin,
                        "end": a.interval.end,
                        "is_retry": i > 0,
                        # Cancelled attempts were abandoned, not failed: they
                        # are not evidence of a retry that went wrong.
                        "retry_failed": i > 0
                        and not a.success
                        and a.drop_reason not in (DropReason.NONE, DropReason.CANCELLED),
                        "queue": float("nan")
                        if a.queue_size_at_end is None
                        else int(a.queue_size_at_end),
                    }
                )

        if not attempt_rows:
            pd.DataFrame(columns=columns).to_csv(path, index=False, lineterminator="\r\n")
            return

        attempts = pd.DataFrame(attempt_rows)
        roots = pd.DataFrame(root_rows)

        # Align to bucket boundaries. The upper bound must cover BOTH frames:
        # roots are bucketed at `completed_at`, which for a root cancelled while
        # backing off is strictly later than any of its attempt ends. Sizing the
        # range from attempts alone put such a root past the last bucket, where
        # `reindex(buckets, fill_value=0)` silently discarded it — it stayed in
        # root_requests (bucketed at its first attempt's begin) but vanished from
        # success_root/failure_root/failure_cancelled, so summary() and
        # export_csv() disagreed about the same root.
        # start_ns needs no such union: roots["begin"] IS attempts_by_begin[0].
        start_ns = int(attempts["begin"].min()) // bucket_ns * bucket_ns
        last_end = max(int(attempts["end"].max()), int(roots["end"].max()))
        end_ns = (last_end // bucket_ns + 1) * bucket_ns
        buckets = pd.RangeIndex((end_ns - start_ns) // bucket_ns)

        def bucket_of(at: "pd.Series") -> "pd.Series":
            return (at - start_ns) // bucket_ns

        def per_bucket_count(at: "pd.Series") -> "pd.Series":
            return bucket_of(at).value_counts().reindex(buckets, fill_value=0)

        succeeded = roots[roots["succeeded"]]
        failed = roots[~roots["succeeded"]]

        # Latency stats (ms) over successful ROOT latencies, by completion bucket
        def percentile(p: float):
            return lambda lat_ns: self._percentile_ms(p, lat_ns.tolist())

        lat_stats = (
            (succeeded["end"] - succeeded["begin"])
            .groupby(bucket_of(succeeded["end"]))
            .agg(
                **{
                    "p50": percentile(0.5),
                    "p90": percentile(0.9),
                    "p95": percentile(0.95),
                    "p99": percentile(0.99),
                    "p99.9": percentile(0.999),
                    "p99.99": percentile(0.9999),
                    "min": lambda lat_ns: ns_to_ms(lat_ns.min()),
                    "max": lambda lat_ns: ns_to_ms(lat_ns.max()),
                }
            )
            .reindex(buckets)
        )

        with_queue = attempts.dropna(subset=["queue"])
        out = pd.DataFrame(
            {
                "timepoint": (start_ns + np.arange(len(buckets)) * bucket_ns) / S_TO_NS,
                "root_requests": per_bucket_count(roots["begin"]),
                "retries": per_bucket_count(attempts.loc[attempts["is_retry"], "begin"]),
                "success_root": per_bucket_count(succeeded["end"]),
                "completed": per_bucket_count(attempts["end"]),
                "failure_root": per_bucket_count(failed["end"]),
                "failure_retry": per_bucket_count(attempts.loc[attempts["retry_failed"], "end"]),
                **{
                    reporting.csv_column: per_bucket_count(
                        failed.loc[failed["outcome"] == outcome, "end"]
                    )
                    for outcome, reporting in FAILURE_OUTCOMES.items()
                },
                # queue size at attempt END, by completion bucket
                "queue_avg_at_attempt_end": (
                    with_queue.groupby(bucket_of(with_queue["end"]))["queue"]
                    .mean()
                    .reindex(buckets)
                ),
            }
        )
        out["total_request"] = out["root_requests"].cumsum()
        out["total_failure"] = out["failure_root"].cumsum()
        out = pd.concat([out, lat_stats], axis=1)[columns]
        # %.3f is fine for millisecond latency columns but collapses sub-ms
        # bucket timestamps; format the timepoint with enough decimals to
        # represent the bucket width exactly (bucket_ns is an integer, so the
        # required precision is 9 minus its trailing decimal zeros — e.g.
        # 1.5 ms = 1_500_000 ns needs 4 decimals, not ceil(-log10) = 3).
        decimals = 3
        # Any bucket that is not a whole number of milliseconds needs more than
        # 3 decimals, including buckets >= 1 s (e.g. 1.5 ms and 1500.5 ms are
        # both mis-serialized at 3 dp). The old `< S_TO_NS` gate only covered
        # sub-second widths.
        if bucket_ns % (S_TO_NS // 1000) != 0 or bucket_ns < S_TO_NS:
            digits = 9
            b = bucket_ns
            while b % 10 == 0:
                b //= 10
                digits -= 1
            decimals = max(3, digits)
        out["timepoint"] = [f"{v:.{decimals}f}" for v in out["timepoint"]]
        out.to_csv(path, index=False, float_format="%.3f", na_rep="nan", lineterminator="\r\n")
