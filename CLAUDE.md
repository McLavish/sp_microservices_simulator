# CLAUDE.md

This file provides guidance to the Agent when working with code in this repository.

## What this is

Discrete-event simulator of microservice request flows: queues, retries, timeouts, partial failures, latency injections, load spikes. Simulates single services, chains, and fan-outs, exports per-bucket CSV metrics + fault-event JSON for plotting. (A reinforcement-learning layer — Gymnasium env, learned/model-based retry policies, PPO training — used to live in `rl/`; it was split out to the `rl-layer` branch on 2026-08-28 and is not part of this branch.)

The project is uv-managed but **not an installable package** (`pyproject.toml` sets `package = false`). Everything runs as `python -m <module>` from the repo root against the in-tree `.venv`. Imports are flat (`from core import ...`, `from service import ...`, `from transport import ...`), so the cwd must be the repo root. The simulator core depends only on `numpy` + stdlib; `scipy`, `matplotlib`, `pandas` are dev deps pulled in only by plotting and `Metrics.export_csv` (which imports `pandas` lazily at export time).

Deeper reference docs live in `docs/` — start with `docs/architecture.md`, then the subsystem you're touching (`services-and-workload`, `request-handling`, `observability`, `experiments`). Keep them in sync when changing the behavior they describe. `AGENTS.md` is a symlink to this file — edit `CLAUDE.md`, never create a separate agents file.

## Architecture

The simulator is built around a single-threaded discrete-event loop. Events are pushed onto a heap with `(at, seq)` ordering, popped in time order, and `Simulator.timestep` advances to the event's `at` before its action runs (`simulator.py`). Every random draw goes through `Simulator.rng()` so a single seed makes service times, failures, and jitter reproducible — `Workload` deliberately keeps a *separate* RNG (`workload.py`) so arrival processes stay identical across experiments even when the simulator seed changes. That RNG is seeded at `rng_seed + _ARRIVAL_SEED_OFFSET`: callers routinely pass the same integer to `Simulator(seed=...)` and `Workload(rng_seed=...)`, and without the offset both would drive the identical Mersenne-Twister stream, correlating arrival times with service times and failure rolls.

**Time** is integer nanoseconds end-to-end (`TimePoint`, `TimeDuration` in `core.py`). Convert at the boundary with `s_to_ns` / `ms_to_ns` / `ns_to_ms`. Plots and CSVs emit seconds — that conversion happens in `metrics.export_csv` and `fault_events.export_json`, not in the core loop.

**Service queue** (`service.py`) is M/G/c/K: FCFS, `workers` concurrency, optional `queue_capacity`. Service time is lognormal with `latency_median = exp(mu)` and `latency_lognorm_sigma`, then mutated by any active `LatencyInjection` (additive `add_latency` and integer `multiplier`, with a 100 µs floor). `PartialFailure`s set a per-time `p_fail` rolled at attempt end. Drops are decided in three places: queue-full at submit, deadline at dequeue *and* at finish, server-failure post-service.

**Client + retry** (`caller.py`). `ClientRuntime.start_request` builds a `RootRequest`, computes `global_deadline = min(controller_global_timeout, override)`, and runs attempts until success, deadline, or the policy stops retrying. Two non-obvious invariants:

- **Half-open deadline semantics**: success requires `end_time < deadline` (strict). If a request *finishes* exactly at the deadline it counts as a deadline drop, not a success. Preserve this when adding new completion paths.
- **Policy result fan-out**: `_update_policy_results` walks `policy.underlying_strategy` recursively and calls `add_result(success, now)` on every layer that has it. Circuit breakers and retry budgets *only* update through this path — if you add a new wrapping policy, it must expose `underlying_strategy` (or `add_result`) or it will silently never see results.

**Chain / fan-out** (`chain.py`). `ServiceNodeRuntime` wraps a `ServiceRuntime` and a list of `DownstreamEdge`s. The node submits a custom `behavior` to its service that does (1) local service time, (2) launches **all** downstream calls concurrently (a synchronous rejection on an early edge no longer skips later edges' launches), (3) finalizes only when **all** downstreams succeed (AND semantics — first failure short-circuits the parent, but only after every edge is launched). Worker occupancy during fan-out is controlled by `ServiceNodeRuntime.hold_worker_during_fanout`:

- `False` (**default, tandem-queue**): `release_worker()` is called after the local stage, so the upstream worker is freed while waiting on downstreams. This is load-bearing for the multichain experiments — capacity is *not* consumed down the tree.
- `True` (**synchronous thread-per-request**): the worker is held for the whole subtree (local work + downstream wait), so capacity is consumed all the way down.

When adding completion paths in `behavior`, preserve AND semantics and route deadlines through `min(...)` rather than overwriting; the deadline guard schedules an `on_deadline` event that must be cancelled on success.

**Transport delay** (`transport.py`). `TransportLink` is a `ServiceLike` *decorator* — it forwards `cfg` and implements `submit_attempt` — that puts an independent lognormal delay on each leg of an attempt's round trip: `ClientRuntime(cfg=..., service=TransportLink(node, delay))` with `DownstreamEdge(client, node)` unchanged (the link wraps the target *as the edge's client sees it*; `ServiceNodeRuntime.service` stays a `ServiceRuntime`). It has to be a decorator and not a `chain.py` change: retries are issued from `ClientRuntime._start_attempt` and never pass through `chain.py`, so a delay in the fan-out loop would be charged once per root instead of once per attempt. Deadlines keep the strict half-open rule — a forward leg arriving at or after the attempt deadline expires in transit (the target is never called, `admitted_total` untouched, `queue_size_at_end` `None` as for `CIRCUIT_OPEN`), and a response that would land at or after it is clamped to the deadline (or to now, when the target itself reported later still — the past cannot be scheduled) and booked `DEADLINE`, keeping the occupancy and censoring flag the service reported. `cancel()` stays truthful (`True` only when something was really torn down and `on_done` was delivered synchronously) and `on_done` fires exactly once per attempt in every phase. `Request.service_time_ns` remains worker occupancy and **excludes** transport (it is the service-time sample any utilization or `E[S]` estimate is built from); `Request.interval` spans both legs. `TransportDelay.sigma == 0.0` means exactly `median` with no RNG draw (deliberately not floored by `MIN_LOGNORM_SIGMA`), and a scenario with no link draws nothing from `sim.rng()` and schedules no extra event, so the goldens are unaffected.

**Workload** (`workload.py`) is an exact nonhomogeneous Poisson process for the piecewise-constant rate (base RPS × active `LoadSpike.rps_multiplier`s): one Exp(1) budget per arrival is consumed across rate segments, so short spikes are never skipped and the first arrival is exponentially distributed (never deterministically at t=0). Zero-rate segments are skipped exactly without consuming RNG; an all-zero workload schedules no events.

**Metrics** (`metrics.py`). `Metrics.summary()` is per-root aggregate. `Metrics.export_csv()` produces per-second buckets where:
- `root_requests` and `retries` are bucketed by **start** time.
- `failure_*`, `success_root`, `completed`, percentiles are bucketed by **end** time.
- `failure_retry` counts failed *retry attempts* (not first attempts), bucketed at attempt end.
- Latency stats only include **successful root** latencies completed in that bucket.
- `queue_avg_at_attempt_end` averages `Request.queue_size_at_end` for attempts ending in the bucket.

**Fault events** (`fault_events.py`) is a separate tracker that mirrors faults already configured on services/workloads, but emits them in seconds-based JSON for plotting overlays. Both `ServiceConfig.register_fault_events(tracker)` and `Workload.register_fault_events(tracker)` must be called explicitly — registration is *not* automatic.

## Common commands

Setup and quality gates (all run through `uv` so the `.venv` stays consistent):
```bash
make install     # uv sync --group dev + install pre-commit hook
make check       # lint + format check + ty type check + tests (the full gate)
make lint        # ruff check
make fmt         # ruff format
make type        # ty check
make test        # uv run pytest
make cov         # pytest --cov
make help        # list all targets
```
Pre-commit runs `ruff check --fix`, `ruff format`, and `ty check` on every commit.

`tests/test_golden_and_invariants.py` holds characterization tests that pin
exact results for seeded runs, plus the invariants a change must not break
(FCFS order, additive latency injection, the strict `end_time < deadline` rule,
start-vs-end CSV bucketing). They are brittle by design: a diff means the
simulator's numbers moved and every figure produced with the old behaviour is
invalid. `docs/observability.md` records the mutations that once passed silently.

Tests (pytest; tests are `unittest`-style but run under pytest):
```bash
uv run pytest                              # full suite
uv run pytest tests/test_transport.py     # single module
uv run pytest -k chain_worker             # by keyword
```

Single-service / chain experiments (write CSVs + JSON to repo root):
```bash
python -m experiments.default
python -m experiments.chain_fanout
python -m experiments.multichain_retry_storm
python -m experiments.hotel_matched   # DeathStarBench hotelReservation approximation -> hotel_matched_results/
```
`experiments/hotel_matched.py` maps the hotel read workload onto two chains sharing `profile`; `--calibration-json` takes the calibration JSON emitted by the external `valharness` project (`orchestrator.record_real`) and falls back to built-in defaults without it. Calibration fault semantics: an *absent* `fault_scenario` key falls back to the default profile fault, but an explicit empty (or profile-free) scenario means genuinely no profile faults. CLI bounds and JSON finiteness are validated (NaN/Infinity rejected).
The tandem vs hold-worker fan-out behaviour is covered by
`tests/test_chain_worker_model.py`; broader chain scenarios (fan-out
monotonicity, deadline propagation, queue pressure) live in
`tests/test_chain_scenarios.py`.
`experiments/single/` holds focused single-service demos (circuit breakers, backoff variants, rate limiters, retry budget).

Plotting (consumes a CSV + matching fault-events JSON; the time axis is shared):
```bash
python -m plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json
python -m plotting.latencies_over_time chain_output.csv --fault-events chain_fault_events.json
python -m plotting.queue_size_over_time chain_output.csv --fault-events chain_fault_events.json
python -m plotting.qps_over_time chain_output.csv --fault-events chain_fault_events.json
./plot.sh   # batch-render the standard chain figures
```

## Conventions when extending

- New retry policies that wrap another should expose `underlying_strategy` so `_update_policy_results`, `_rollback_policy` *and* `_notify_request_start` reach them. New stateful policies should implement `add_result(success, now)` (once per attempt) and/or `on_request_start(now)` (once per admitted root request — that's the hook `RetryBudgetPolicy` deposits on, since its ratio is retries per *request*).
- If a policy's `next_delay` has **side effects** (withdrawing budget, spending a rate-limiter token, reserving a leaky-bucket slot), it must also implement `rollback(context)` to undo exactly one such charge. The client asks for a delay before it knows the global deadline will veto the retry; without a refund, retries that are never issued are still charged. `RetryPolicy.rollback` defaults to a no-op, so side-effect-free policies need nothing.
- Custom service behaviors must call `finalize` exactly once and `release_worker` at most once (both raise on repeats). Completion callbacks must not submit work synchronously if FCFS matters to your experiment — schedule through the event heap instead.
- A new `ServiceLike` decorator (the `transport.py` pattern) must forward `cfg` so attempts stay booked under the target's name, return an `AttemptControl` *immediately* (never after the target reports), keep `cancel()` truthful — `True` only when something was actually torn down and `on_done` was already delivered synchronously — and call `on_done` exactly once per attempt on every path.
- New downstream-graph features must preserve AND semantics in `chain.py`'s `behavior` and route deadlines through `min(...)` rather than overwriting.
- After running an experiment, call `Metrics.export_csv()` *and* `FaultEventsTracker.export_json()` so plotting overlays line up.
- Keep internal time in nanoseconds; convert only at I/O / plotting boundaries.
- `ty` is lenient on missing annotations but real type errors (None narrowing, protocol mismatches, numpy misuse) are kept clean — don't reintroduce them.
- Generated artifacts (`*.csv`, `*.json`, `*.png`, `results/`) are gitignored. Anything of that kind at the repo root (`*_output.csv`, `*_fault_events.json`, …) is untracked local output from prior runs; treat it as disposable.

## Git Commits

- Commit after every meaningful, self-contained change — don't batch unrelated work.
- Each commit should leave the codebase in a working state.
- Write short, clear commit messages that describe what changed.
- If a task involves multiple logical steps, make a separate commit for each step.
- Stage only the files relevant to the current change.
