# AI Agent Instructions for sp_microservices_simulator

**Overview**
- **Purpose:** Discrete-event simulation of microservice request flows, queues, retries, and faults.
- **Core units:** All time values are nanoseconds; conversions in [core.py](../core.py).
- **Outputs:** Per-client CSV metrics and fault events JSON; plotting scripts consume these.

**Architecture**
- **Simulator:** [simulator.py](../simulator.py) schedules `Event`s and advances `timestep` deterministically with a seeded RNG.
- **Service Queue:** [service.py](../service.py) implements FCFS with `workers` concurrency and optional `queue_capacity`. Latency is lognormal (median + sigma) adjusted by active `LatencyInjection`s; partial failures via `PartialFailure`.
- **Client + Retry:** [caller.py](../caller.py) `ClientRuntime` orchestrates attempts using a `RetryPolicy` and a `TimeoutController`. Success requires `drop_reason == NONE` and completion strictly before the attempt/global deadline.
- **Chain Graph:** [chain.py](../chain.py) composes services as nodes with downstream fan-out edges. Local stage runs, then downstreams start concurrently. AND semantics: any downstream failure fails the root; deadlines propagate. Workers can be released early via `release_worker()` to model tandem queues.
- **Workload:** [workload.py](../workload.py) generates arrivals using its own RNG (stable across experiments) with optional `LoadSpike`s.
- **Fault Events:** [fault_events.py](../fault_events.py) tracks latency injections, partial failures, and load spikes; exported to JSON for plotting.
- **Metrics:** [metrics.py](../metrics.py) computes summaries and writes bucketed CSV with per-second aggregations (roots, retries, failures, latency percentiles, queue sizes).

**Key Conventions**
- **Time:** `TimePoint`/`TimeDuration` are nanoseconds; convert with `s_to_ns`/`ms_to_ns`. Plotting expects seconds for X-axis, handled by metrics exporter.
- **Deadlines:** Half-open semantics: attempt success only if `end_time < deadline`. Global deadline is min(controller + override).
- **Drops:** `DropReason` covers `QUEUE_FULL`, `DEADLINE`, `SERVER_FAILURE`. Queue-full is decided before enqueue; server failure is post-service.
- **Latencies:** Lognormal sampling from median (as exp(mu)) and sigma; enforced minimum latency.

**Developer Workflows**
- **Run single-service experiment:**
  ```bash
  python -m experiments.default
  ```
  Produces [output.csv](../output.csv) and [fault_events.json](../fault_events.json).
- **Run chain/fanout experiment (A → B → {C, E}):**
  ```bash
  python -m experiments.chain_fanout
  ```
  Produces [chain_output.csv](../chain_output.csv), per-service CSVs (e.g., [svc-A_output.csv](../svc-A_output.csv)), and [chain_fault_events.json](../chain_fault_events.json).
- **Plotting examples:**
  ```bash
  python -m plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json
  ```
  Additional scripts in [plotting/](../plotting) visualize qps, latencies, distributions, and queue sizes.

**Patterns & Examples**
- **Registering faults:** Service/workload configs call `register_fault_events(tracker)` to emit aligned event windows used by plots; see [experiments/default.py](../experiments/default.py).
- **Retry composition:** Circuit breakers, budgets, and limiters in [retry_policies.py](../retry_policies.py) wrap an underlying `RetryPolicy` and may require `context.now` (rate limiters). Update result windows via `ClientRuntime._update_policy_results()`.
- **Tandem queues:** In [chain.py](../chain.py), the local stage schedules downstream calls and may `release_worker()` to free capacity while waiting on fan-out.
- **Bucketed metrics:** CSV rows are per-second buckets (configurable); roots are bucketed at start, completions and failures at end; retry-failures are counted per attempt end; queue sizes averaged per completion bucket.

**Integration Points**
- **Fault events → plots:** JSON from [fault_events.py](../fault_events.py) aligns with CSV timeline for overlay in plots.
- **Per-edge clients:** Each downstream edge uses its own `ClientConfig` to model differing retry/timeout behavior; see [experiments/chain_fanout.py](../experiments/chain_fanout.py).
- **Seeds & reproducibility:** Simulator RNG seed controls service time/failure sampling; workload has a separate RNG for stable arrival processes.

**Coding Tips for Agents**
- **Respect time units:** Keep all internal timings in ns; convert only for I/O/plots.
- **Preserve semantics:** When adding new policies or controllers, ensure `ClientRuntime` updates result windows and deadline checks remain half-open.
- **Extend graphs safely:** Add downstreams via `ServiceNodeRuntime.add_downstream(DownstreamEdge(...))`; ensure finalize logic maintains AND semantics.
- **Export artifacts:** After experiments, call `Metrics.export_csv()` and `FaultEventsTracker.export_json()` to keep plots consistent.
