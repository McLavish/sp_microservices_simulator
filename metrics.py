from dataclasses import dataclass
from typing import List
import numpy as np

from simulator.core import (
    RootRequest,
    Summary,
    TimeInterval,
    ns_to_ms,
    DropReason,
    s_to_ns,
    MS_TO_NS,
)


@dataclass
class Metrics:
    roots: List[RootRequest]
    attempts_total: int

    def _percentile_ms(self, p: float, vals: List[int]) -> float:
        if not vals:
            return float("nan")

        arr = np.asarray(vals, dtype=np.int64)
        q_ns = np.quantile(arr, p)
        return float(q_ns) / MS_TO_NS

    def _compute_time_interval_from_attempts(
        self, root_request: RootRequest
    ) -> TimeInterval:
        if not root_request.attempts:
            raise ValueError("No attempts found in RootRequest")

        begin = min(attempt.interval.begin for attempt in root_request.attempts)
        end = max(attempt.interval.end for attempt in root_request.attempts)

        return TimeInterval(begin=begin, end=end)

    def summary(self) -> Summary:
        total = len(self.roots)
        succ_lat: List[int] = []
        succeeded = 0
        dropped_q = 0
        dropped_deadline = 0
        dropped_server_failure = 0
        for request in self.roots:
            success = False
            for attempt in request.attempts:
                if attempt.drop_reason == DropReason.QUEUE_FULL:
                    dropped_q += 1
                if attempt.drop_reason == DropReason.DEADLINE:
                    dropped_deadline += 1
                if attempt.drop_reason == DropReason.SERVER_FAILURE:
                    dropped_server_failure += 1

                if attempt.success:  # there should be only 1 success per root request
                    if success:
                        raise ValueError("Multiple successful attempts in RootRequest")
                    success = True
                    succeeded += 1
                    succ_lat.append(
                        self._compute_time_interval_from_attempts(request).duration()
                    )

        mean_ms = ns_to_ms(int(sum(succ_lat) / len(succ_lat))) if succ_lat else 0.0
        max_ms = ns_to_ms(max(succ_lat)) if succ_lat else 0.0
        retries_per_root = (self.attempts_total - total) / total if total > 0 else 0.0
        return Summary(
            total=total,
            succeeded=succeeded,
            dropped_queue=dropped_q,
            dropped_deadline=dropped_deadline,
            dropped_server_failure=dropped_server_failure,
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
        )

    def print_stats(self):
        summary = self.summary()
        print(f"Total Requests: {summary.total}")
        print(f"Succeeded: {summary.succeeded}")
        print(f"Dropped (Queue Full): {summary.dropped_queue}")
        print(f"Dropped (Deadline): {summary.dropped_deadline}")
        print(f"Dropped (Server Failure): {summary.dropped_server_failure}")
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
        import csv

        if granularity_s <= 0:
            raise ValueError("granularity_s must be > 0")
        bucket_ns = s_to_ns(granularity_s)

        all_attempts = [a for r in self.roots for a in r.attempts]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
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
                    "total_request",  # bucketed at the start. rename total_started_roots?
                    "total_failure",  # bucketed at the end. rename total_failed_roots_completed?
                    "p50",  # percentiles are all bucketed at the end
                    "p90",
                    "p95",
                    "p99",
                    "p99.9",
                    "p99.99",
                    "Min",
                    "Max",
                    "queue_avg_at_attempt_end",
                ]
            )

            min_begin = min(a.interval.begin for a in all_attempts)
            max_end = max(a.interval.end for a in all_attempts)

            # check if no attempt begins and ends at the exact same time
            # for a in all_attempts:
            #     assert a.interval.begin != a.interval.end

            # Align to bucket boundaries
            start_ns = (min_begin // bucket_ns) * bucket_ns
            # end_ns = ((max_end + bucket_ns - 1) // bucket_ns) * bucket_ns
            end_ns = ((max_end // bucket_ns) + 1) * bucket_ns
            nbuckets = int((end_ns - start_ns) // bucket_ns)

            # Per-bucket trackers (counts only)
            root_counts = [0] * nbuckets  # roots that START in bucket
            retry_counts = [0] * nbuckets  # retry attempts that START in bucket
            failure_counts = [0] * nbuckets  # failed ROOTS that COMPLETE in bucket
            failure_q_counts = [0] * nbuckets  # failed ROOTS by reason
            failure_dead_counts = [0] * nbuckets
            failure_srv_counts = [0] * nbuckets
            retry_fail_counts = [
                0
            ] * nbuckets  # failed RETRY attempts that END in bucket
            completed_counts = [0] * nbuckets  # all attempts that END in bucket
            latencies_by_bucket: List[List[int]] = [
                [] for _ in range(nbuckets)
            ]  # successful ROOT latencies (ns), by completion bucket
            queue_sizes_by_bucket: List[List[int]] = [
                [] for _ in range(nbuckets)
            ]  # queue size at attempt END, by completion bucket

            def bin_of(tns: int) -> int:
                return int((tns - start_ns) // bucket_ns)

            # Populate attempt-end buckets (completed_counts, queue sizes)
            for a in all_attempts:
                b_end = bin_of(a.interval.end)
                if 0 <= b_end < nbuckets:
                    completed_counts[b_end] += 1
                    if getattr(a, "queue_size_at_end", None) is not None:
                        queue_sizes_by_bucket[b_end].append(int(a.queue_size_at_end))

            # Populate buckets
            for root in self.roots:
                if not root.attempts:
                    continue

                attempts_by_begin = sorted(
                    root.attempts, key=lambda a: a.interval.begin
                )
                attempts_by_end = sorted(root.attempts, key=lambda a: a.interval.end)

                root_begin = attempts_by_begin[0].interval.begin
                root_end = attempts_by_end[-1].interval.end
                last_attempt = attempts_by_end[-1]

                # Root start -> root_requests_count
                b_start = bin_of(root_begin)
                if 0 <= b_start < nbuckets:
                    root_counts[b_start] += 1

                # Retries: attempts beyond first, bucketed by BEGIN
                for a in attempts_by_begin[1:]:
                    b_retry = bin_of(a.interval.begin)
                    if 0 <= b_retry < nbuckets:
                        retry_counts[b_retry] += 1

                # Failed retry attempts: bucket by END
                for a in attempts_by_begin[1:]:
                    if (not a.success) and (a.drop_reason != DropReason.NONE):
                        b_rfail = bin_of(a.interval.end)
                        if 0 <= b_rfail < nbuckets:
                            retry_fail_counts[b_rfail] += 1

                # Root completion: success/failure and latency allocation
                succeeded = any(a.success for a in root.attempts)
                b_done = bin_of(root_end)
                if 0 <= b_done < nbuckets:
                    if succeeded:
                        latencies_by_bucket[b_done].append(root_end - root_begin)
                    else:
                        failure_counts[b_done] += 1
                        # Attribute failed root to the final attempt's drop_reason
                        if last_attempt.drop_reason == DropReason.QUEUE_FULL:
                            failure_q_counts[b_done] += 1
                        elif last_attempt.drop_reason == DropReason.DEADLINE:
                            failure_dead_counts[b_done] += 1
                        elif last_attempt.drop_reason == DropReason.SERVER_FAILURE:
                            failure_srv_counts[b_done] += 1

            seconds_per_bucket = granularity_s
            total_requests_so_far = 0
            total_failures_so_far = 0

            for i in range(nbuckets):
                tp_sec = i * seconds_per_bucket

                n_roots = root_counts[i]
                n_retries = retry_counts[i]
                n_fail_roots = failure_counts[i]
                n_retry_fails = retry_fail_counts[i]
                lats = latencies_by_bucket[i]
                n_success_roots = len(lats)

                # Update cumulatives
                total_requests_so_far += n_roots
                total_failures_so_far += n_fail_roots

                # Latency stats (ms) for successful roots in this bucket
                if lats:
                    p50 = self._percentile_ms(0.5, lats)
                    p90 = self._percentile_ms(0.9, lats)
                    p95 = self._percentile_ms(0.95, lats)
                    p99 = self._percentile_ms(0.99, lats)
                    p999 = self._percentile_ms(0.999, lats)
                    p9999 = self._percentile_ms(0.9999, lats)
                    min_ms = ns_to_ms(min(lats))
                    max_ms = ns_to_ms(max(lats))
                else:
                    p50 = p90 = p95 = p99 = p999 = p9999 = min_ms = max_ms = float(
                        "nan"
                    )

                # Queue size stats for attempts ending in this bucket
                if queue_sizes_by_bucket[i]:
                    q_avg = float(np.mean(queue_sizes_by_bucket[i]))
                else:
                    q_avg = float("nan")

                writer.writerow(
                    [
                        f"{tp_sec:.3f}",
                        str(n_roots),
                        str(n_retries),
                        str(n_success_roots),
                        str(completed_counts[i]),
                        str(n_fail_roots),
                        str(n_retry_fails),
                        str(failure_q_counts[i]),
                        str(failure_dead_counts[i]),
                        str(failure_srv_counts[i]),
                        str(total_requests_so_far),
                        str(total_failures_so_far),
                        f"{p50:.3f}",
                        f"{p90:.3f}",
                        f"{p95:.3f}",
                        f"{p99:.3f}",
                        f"{p999:.3f}",
                        f"{p9999:.3f}",
                        f"{min_ms:.3f}",
                        f"{max_ms:.3f}",
                        f"{q_avg:.3f}",
                    ]
                )
