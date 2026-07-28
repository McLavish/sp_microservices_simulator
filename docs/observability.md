# Observability: metrics, CSV export & plotting

How a run becomes numbers and pictures. `Metrics` aggregates completed
`RootRequest`s into a summary and a per-bucket CSV; the `plotting/` scripts render
that CSV (optionally overlaid with the fault-event JSON).

```
ClientRuntime.roots --> Metrics.summary()      --> console stats
                    \-> Metrics.export_csv()    --> <name>_output.csv  --\
FaultEventsTracker  ---> export_json()          --> <name>_fault.json  --+--> plotting/*.py --> PNG
```

The CSV and the fault JSON share a **seconds** time axis, so they align when plotted
together. Always export both from the same run.

## `Metrics` (`metrics.py`)

Construct via `client.metrics()` (it wraps `roots` + `attempts_total`).

### `summary()` (`metrics.py`)

Returns a `Summary` (`core.py`): `total`, `succeeded`, `dropped_*` by reason,
latency percentiles `p50..p99_99`, `mean`, `max`, `retries_per_root`,
`attempts_total`. Notes:

- A root's **latency is wall-clock**: `max(attempt.end) - min(attempt.begin)`, i.e.
  including queueing and every retry, counted only for **successful** roots.
- A failed root is attributed to its **last attempt's** `drop_reason`.
- `retries_per_root = (attempts_total - total) / total`.

`print_stats()` (`metrics.py`) dumps the summary to stdout.

### `export_csv(path, granularity_s=1.0)` (`metrics.py`)

Writes one row per time bucket (default 1 s). `granularity_s` must round to at
least 1 ns (values below 5e-10 s raise `ValueError`); the `timepoint` column is
serialized with granularity-aware precision so sub-millisecond buckets never
collapse under the fixed `%.3f` used for latency columns. Count columns are raw
per-bucket counts — `plotting.qps_over_time` divides them by the bucket width
(inferred from consecutive timepoints) before labeling the axis "QPS".

The subtlety is **which event time a column is bucketed by** — start vs. end:

| Column | Bucketed by | Meaning |
| --- | --- | --- |
| `timepoint` | — | bucket start, seconds |
| `root_requests` | start | roots that *started* in the bucket |
| `retries` | start | retry attempts (after the first) that started |
| `success_root` | end | successful roots completing in the bucket |
| `completed` | end | all attempts ending in the bucket |
| `failure_root` | end | failed roots completing in the bucket |
| `failure_retry` | end | failed *retry* attempts ending in the bucket |
| `failure_queue_full` / `failure_deadline` / `failure_server` / `failure_circuit_open` | end | failed roots by final drop reason |
| `total_request` | start (cumulative) | running sum of started roots |
| `total_failure` | end (cumulative) | running sum of failed roots |
| `p50 p90 p95 p99 p99.9 p99.99 min max` | end | latency stats over **successful roots** completing in the bucket (ms) |
| `queue_avg_at_attempt_end` | end | mean `queue_size_at_end` over attempts ending in the bucket |

Latency/queue cells are `nan` for buckets with no qualifying data. Because roots
bucket by start but completions/latency by end, a spike of starts and the resulting
completions can land in different rows — expected, not a bug.

## Plotting (`plotting/`)

Each script is `python -m plotting.<name> <csv> [--fault-events JSON] [--output PNG]`.
With `--fault-events`, the active fault windows are shaded/annotated over the series
(via `plotting/utils/fault_events.py`).

| Script | Reads (CSV columns) | Renders |
| --- | --- | --- |
| `failures_over_time` | `failure_*`, request counts | stacked failure reasons + request amplification over time |
| `latencies_over_time` | `p50/p90/p95/p99/max` | latency percentile bands over time |
| `qps_over_time` | `root_requests`, `retries`, `failure_root` | throughput, retries, failures over time |
| `queue_size_over_time` | `queue_avg_at_attempt_end` | average queue depth over time |
| `latency_distribution` | *(none — synthetic)* | lognormal PDF/CDF for `--median-ms` / `--sigma` |

Each script writes a PNG via `--output`/`-o` (defaults: `latencies_over_time.png`,
`qps_over_time.png`, `queue_size_over_time.png`, `latency_distribution.png`, and
`reasons_and_request_amplification.png` for `failures_over_time`).
`latency_distribution` takes `--median-ms` and `--sigma` instead of a CSV.
`plot.sh` batch-renders the standard chain figures.

### Example

```bash
python -m experiments.default        # writes output.csv + fault_events.json
python -m plotting.failures_over_time output.csv --fault-events fault_events.json
python -m plotting.latencies_over_time output.csv --fault-events fault_events.json -o lat.png
```

## See also

- [services-and-workload](services-and-workload.md) — registering faults into the
  tracker that produces the overlay JSON.
- [experiments](experiments.md) — what each scenario writes (filenames differ per
  experiment).

## Mutation record

Behaviours below were, at some point, changed in the source without any test
failing. Each now has a guard; re-run these after a significant change to the
event loop, the queue, or the metrics path. `tests/test_golden_and_invariants.py`
is where they are pinned.

| Mutation | Caught by |
| --- | --- |
| `service.py` `heapq.heappop(self.queue)` → LIFO pop | `test_service_is_fcfs`, both golden runs |
| `service.py` `add += inj.add_latency` → `add += 0` | `test_additive_latency_applies_inside_its_window` |
| `simulator.py` `seq=self._seq` → `seq=-self._seq` | `test_fanout_golden` (needs contention: with every request succeeding, tie order is unobservable) |

The golden tests are brittle on purpose. A diff means the simulator's numbers
moved, which invalidates figures produced with the old behaviour — establish
why before updating the expected values.
