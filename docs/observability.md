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

Construct via `client.metrics()` (it wraps `roots` plus the client's lifetime
attempt counter, readable as `Metrics.client_attempts_total()`). That counter is
a *different population* from `roots` — clients bump it at attempt **start**
while roots are appended at **completion** — so no `Summary` field is derived
from it.

### One outcome table (`core.py`)

`classify_root(root)` (`core.py`) is the single classifier: it returns a
`RootOutcome`, and both `Metrics.summary` and `Metrics.export_csv` book roots by
its answer rather than re-deriving the rules.

`core.FAILURE_OUTCOMES` then maps each **non-success** outcome to where it is
reported: the `Summary` field it increments, the `export_csv` column that books
it, and the label `print_stats` prints. Every one of those lists is generated
from that mapping, in its declaration order — which is what fixes the CSV
header, so appending an outcome is safe and reordering rewrites it.

Adding a `RootOutcome` member is therefore a forced edit: `core.py` raises at
import if the new member has no `FAILURE_OUTCOMES` entry (or if an entry names a
`Summary` field that does not exist). Any consumer outside `core` that keeps its
own per-outcome table gets the same check by calling
`core.require_outcome_coverage(table, owner=...)` next to it.
`plotting/failures_over_time.py` derives its `REASON_COLUMNS` from the
same table (it still tolerates a column missing from an older CSV), so a new
outcome flows into the failure-mix figure without a manual edit.

### `summary()` (`metrics.py`)

Returns a `Summary` (`core.py`): `total`, `succeeded`, `dropped_*` by reason,
latency percentiles `p50..p99_99`, `mean`, `max`, `retries_per_root`,
`attempts_total`. Every field, `attempts_total` included, is computed over the
roots being summarized (`Metrics.attempts_on_roots()`). Notes:

- A root's **latency is wall-clock**: `max(attempt.end) - min(attempt.begin)`, i.e.
  including queueing and every retry, counted only for **successful** roots.
- A failed root is attributed to its **last attempt's** `drop_reason`.
- `retries_per_root = (attempts on the summarized roots - total) / total`.
  Counted over the roots being summarized, not the client's lifetime attempt
  counter — those disagree whenever work is still in flight (a mid-run export,
  or any summary taken over a sub-window of the run).
- `dropped_cancelled` counts roots abandoned by an upstream cancellation. It is
  deliberately its own bucket, NOT folded into the other `dropped_*` counts:
  cancelled work did not fail, and charging it to a failure category would count
  one root cause once where it failed and again at every sibling it cancelled.
- `mean` / `max` are `nan` when no successful root completed, matching the
  percentiles. They are not 0, which would make an empty window read as a
  perfect one.

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
| `failure_queue_full` / `failure_deadline` / `failure_server` / `failure_circuit_open` | end | failed roots by final drop reason (generated from `core.FAILURE_OUTCOMES`, in its order) |
| `failure_cancelled` | end | roots abandoned by an upstream cancellation (see `propagate_cancellation`); abandoned, not failed |
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
| `latency_distribution` | *(none — synthetic)* | lognormal PDF/CDF for `--median` (ms) / `--sigma` |

Each script writes a PNG via `--output`/`-o` (defaults: `latencies_over_time.png`,
`qps_over_time.png`, `queue_size_over_time.png`, `latency_distribution.png`, and
`reasons_and_request_amplification.png` for `failures_over_time`).
`latency_distribution` takes `--median` (in milliseconds) and `--sigma` instead of a
CSV; both must be finite and positive — degenerate values used to render a blank
figure and still report it as saved.
`plot.sh` batch-renders the standard chain figures.

Overlays fail loudly rather than rendering an empty-but-plausible figure: a
missing `--fault-events` file and an unknown fault `event_type` in that JSON both
exit with a message naming what *is* available. Buckets in which no root
completed are drawn as a gap, never as a 0% failure rate.

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
