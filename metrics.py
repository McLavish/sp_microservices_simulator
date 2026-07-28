from dataclasses import dataclass

import numpy as np

from core import (
    MS_TO_NS,
    S_TO_NS,
    DropReason,
    RootRequest,
    Summary,
    TimeInterval,
    ns_to_ms,
    s_to_ns,
)


@dataclass
class Metrics:
    roots: list[RootRequest]
    attempts_total: int

    def _percentile_ms(self, p: float, vals: list[int]) -> float:
        if not vals:
            return float("nan")

        arr = np.asarray(vals, dtype=np.int64)
        q_ns = np.quantile(arr, p)
        return float(q_ns) / MS_TO_NS

    def _compute_time_interval_from_attempts(self, root_request: RootRequest) -> TimeInterval:
        if not root_request.attempts:
            raise ValueError("No attempts found in RootRequest")

        begin = min(attempt.interval.begin for attempt in root_request.attempts)
        end = max(attempt.interval.end for attempt in root_request.attempts)

        return TimeInterval(begin=begin, end=end)

    def summary(self) -> Summary:
        total = len(self.roots)
        succ_lat: list[int] = []
        succeeded = 0
        dropped_q = 0
        dropped_deadline = 0
        dropped_server_failure = 0
        dropped_circuit_open = 0
        dropped_cancelled = 0
        for request in self.roots:
            if not request.attempts:
                continue

            success = False
            for attempt in request.attempts:
                if attempt.success:  # there should be only 1 success per root request
                    if success:
                        raise ValueError("Multiple successful attempts in RootRequest")
                    success = True
                    succeeded += 1
                    succ_lat.append(self._compute_time_interval_from_attempts(request).duration())

            if success:
                continue

            # Abandoned by an upstream cancellation. Checked before the
            # per-attempt reason because a root cancelled while backing off has
            # no attempt of its own carrying CANCELLED.
            if request.cancelled:
                dropped_cancelled += 1
                continue

            # Append order is completion order; ties on end time (zero-backoff
            # synchronous rejections) make max()/sort keys ambiguous.
            last_attempt = request.attempts[-1]
            if last_attempt.drop_reason == DropReason.QUEUE_FULL:
                dropped_q += 1
            elif last_attempt.drop_reason == DropReason.DEADLINE:
                dropped_deadline += 1
            elif last_attempt.drop_reason == DropReason.CIRCUIT_OPEN:
                dropped_circuit_open += 1
            else:
                dropped_server_failure += 1

        mean_ms = ns_to_ms(int(sum(succ_lat) / len(succ_lat))) if succ_lat else 0.0
        max_ms = ns_to_ms(max(succ_lat)) if succ_lat else 0.0
        retries_per_root = (self.attempts_total - total) / total if total > 0 else 0.0
        return Summary(
            total=total,
            succeeded=succeeded,
            dropped_queue=dropped_q,
            dropped_deadline=dropped_deadline,
            dropped_server_failure=dropped_server_failure,
            dropped_circuit_open=dropped_circuit_open,
            p50=self._percentile_ms(0.5, succ_lat),
            p90=self._percentile_ms(0.9, succ_lat),
            p95=self._percentile_ms(0.95, succ_lat),
            p99=self._percentile_ms(0.99, succ_lat),
            p99_9=self._percentile_ms(0.999, succ_lat),
            p99_99=self._percentile_ms(0.9999, succ_lat),
            mean=mean_ms,
            max=max_ms,
            retries_per_root=retries_per_root,
            attempts_total=self.attempts_total,
            dropped_cancelled=dropped_cancelled,
        )

    def print_stats(self):
        summary = self.summary()
        print(f"Total Requests: {summary.total}")
        print(f"Succeeded: {summary.succeeded}")
        print(f"Dropped (Queue Full): {summary.dropped_queue}")
        print(f"Dropped (Deadline): {summary.dropped_deadline}")
        print(f"Dropped (Server Failure): {summary.dropped_server_failure}")
        print(f"Dropped (Circuit Open): {summary.dropped_circuit_open}")
        print(f"Cancelled (Abandoned Upstream): {summary.dropped_cancelled}")
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
        - failure_retry_count counts failed RETRY attempts (after the first), bucketed by attempt end.
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
            "success_root",  # # bucketed at the end. Actually should be the same as success_retry
            "completed",  # all attempts completed, bucketed at the end
            "failure_root",
            "failure_retry",
            "failure_queue_full",
            "failure_deadline",
            "failure_server",
            "failure_circuit_open",
            "failure_cancelled",
            "total_request",  # bucketed at the start. rename total_started_roots?
            "total_failure",  # bucketed at the end. rename total_failed_roots_completed?
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
            # Append order is completion order (ties on end time are ambiguous).
            last_attempt = root.attempts[-1]
            root_rows.append(
                {
                    "begin": attempts_by_begin[0].interval.begin,
                    # Prefer the recorded completion instant: a root cancelled
                    # while backing off finished after its last attempt ended.
                    "end": (
                        root.completed_at
                        if root.completed_at is not None
                        else last_attempt.interval.end
                    ),
                    "succeeded": any(a.success for a in root.attempts),
                    # A root cancelled while backing off has no attempt marked
                    # CANCELLED, so the root-level flag wins (mirrors summary()).
                    "drop_reason": (
                        DropReason.CANCELLED if root.cancelled else last_attempt.drop_reason
                    ),
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
                "failure_queue_full": per_bucket_count(
                    failed.loc[failed["drop_reason"] == DropReason.QUEUE_FULL, "end"]
                ),
                "failure_deadline": per_bucket_count(
                    failed.loc[failed["drop_reason"] == DropReason.DEADLINE, "end"]
                ),
                "failure_server": per_bucket_count(
                    failed.loc[failed["drop_reason"] == DropReason.SERVER_FAILURE, "end"]
                ),
                "failure_circuit_open": per_bucket_count(
                    failed.loc[failed["drop_reason"] == DropReason.CIRCUIT_OPEN, "end"]
                ),
                "failure_cancelled": per_bucket_count(
                    failed.loc[failed["drop_reason"] == DropReason.CANCELLED, "end"]
                ),
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
        if bucket_ns < S_TO_NS:
            digits = 9
            b = bucket_ns
            while b % 10 == 0:
                b //= 10
                digits -= 1
            decimals = max(3, digits)
        out["timepoint"] = [f"{v:.{decimals}f}" for v in out["timepoint"]]
        out.to_csv(path, index=False, float_format="%.3f", na_rep="nan", lineterminator="\r\n")
