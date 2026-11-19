# CLAUDE.md

This file provides guidance to the Agent when working with code in this repository.

## What this is

Discrete-event simulator of microservice request flows: queues, retries, timeouts, partial failures, latency injections, load spikes. Simulates single services, chains, and fan-outs, exports per-bucket CSV metrics + fault-event JSON for plotting. (A reinforcement-learning layer — Gymnasium env, learned/model-based retry policies, PPO training — used to live in `rl/`; it was split out to the `rl-layer` branch on 2026-08-28 and is not part of this branch.)

The project is uv-managed. The simulator library is the `msim` package in `src/msim/`, built by the `uv_build` backend: `uv sync` installs it editable into the in-tree `.venv`, so `from msim.core import ...`, `from msim.service import ...`, `from msim.transport import ...` work from any cwd. `msim.plotting` ships in the package; its heavy deps (`matplotlib`, `pandas`, `scipy`) are the `plot` extra — `pandas` is also what `Metrics.export_csv` needs (imported lazily at export time), so a consumer that exports CSVs installs `microservices-simulator[plot]`. The core depends only on `numpy` + stdlib. `examples/` (runnable demo scenarios) is *not* part of the package: run them as `python -m examples.<name>` from the repo root (pytest's `pythonpath = ["."]` exists for them).

Deeper reference docs live in `docs/` — start with `docs/architecture.md`, then the subsystem you're touching (`services-and-workload`, `request-handling`, `observability`, `examples`). Keep them in sync when changing the behavior they describe. `AGENTS.md` is a symlink to this file — edit `CLAUDE.md`, never create a separate agents file.

## Architecture

The simulator is built around a single-threaded discrete-event loop. Events are pushed onto a heap with `(at, seq)` ordering, popped in time order, and `Simulator.timestep` advances to the event's `at` before its action runs (`simulator.py`). Every random draw goes through `Simulator.rng()` so a single seed makes service times, failures, and jitter reproducible — `Workload` deliberately keeps a *separate* RNG (`workload.py`) so arrival processes stay identical across experiments even when the simulator seed changes. That RNG is seeded at `rng_seed + _ARRIVAL_SEED_OFFSET`: callers routinely pass the same integer to `Simulator(seed=...)` and `Workload(rng_seed=...)`, and without the offset both would drive the identical Mersenne-Twister stream, correlating arrival times with service times and failure rolls.

**Time** is integer nanoseconds end-to-end (`TimePoint`, `TimeDuration` in `core.py`). Convert at the boundary with `s_to_ns` / `ms_to_ns` / `ns_to_ms`. Plots and CSVs emit seconds — that conversion happens in `metrics.export_csv` and `fault_events.export_json`, not in the core loop.

**Service queue** (`service.py`) is M/G/c/K: FCFS, `workers` concurrency, optional `queue_capacity`. `ServiceConfig.queue_discipline` (default `QueueDiscipline.FCFS`, the unchanged heap path) can make a freed worker take the newest waiter (`LIFO`, a stack) or a uniformly random one (`RANDOM`, picked at dequeue from `sim.rng()`; no draw for a single waiter) — only WHICH entry is taken changes, never capacity, deadlines or what holds a slot. `ServiceConfig.evict_expired` (default `False`) is the separate switch for what holds a slot: `True` frees a waiting attempt's slot at its caller's deadline instead of when a worker reaches it (same report to the caller, at the same instant, with the same depth). Cancelled and evicted entries stay in `queue` as tombstones and are skipped by `_start_next`; a LIFO stack compacts them (`_note_dead`), which changes memory and nothing observable. Service time is lognormal with `latency_median = exp(mu)` and `latency_lognorm_sigma`, then mutated by any active `LatencyInjection` (additive `add_latency` and integer `multiplier`, with a 100 µs floor). `PartialFailure`s set a per-time `p_fail` rolled at attempt end. Drops are decided in four places: queue-full at submit; deadline while queued — the attempt deadline is the *caller's* clock, so a queued attempt is reported `DEADLINE` at the deadline itself, not when a worker next frees (the entry keeps its queue slot until the server dequeues and silently discards it, because a server does not scan its backlog; the dequeue path reports a deadline only on the exact tie where a worker frees at the deadline instant); deadline at finish (the worker is freed at the deadline); server-failure post-service.

**Client + retry** (`caller.py`). `ClientRuntime.start_request` builds a `RootRequest`, computes `global_deadline = min(controller_global_timeout, override)`, and runs attempts until success, deadline, or the policy stops retrying. Two non-obvious invariants:

- **Half-open deadline semantics**: success requires `end_time < deadline` (strict). If a request *finishes* exactly at the deadline it counts as a deadline drop, not a success. Preserve this when adding new completion paths.
- **Policy result fan-out**: `_update_policy_results` walks the policy's layers (`retry_policies.policy_layers`, the one walk all four client traversals share: outermost first through each wrapper's `underlying_strategy`) and calls `record_attempt(context, success)` on every layer, whose default is `add_result(success, now)` on a layer that has one. `context` is the `RetryContext` of the attempt that just ended (number, cause — `NONE` for a success — and request context), built once and reused for `next_delay` and a refund. Circuit breakers and retry budgets *only* update through this path — if you add a new wrapping policy, it must expose `underlying_strategy` (or `add_result`) or it will silently never see results.

**What the server is told** (`ClientConfig.server_deadline`). `"attempt"` (default) hands the target each attempt's deadline, and the server enforces it. `"global"` hands it only the request's global deadline and `"none"` nothing: the client then enforces its own deadline with a timer (`_race_own_deadline`), armed *before* `submit_attempt` so it wins a tie at the deadline instant. First of timer or report is delivered once and the other is ignored; a timeout is booked `DEADLINE` with `queue_size_at_end` and `service_time_ns` `None`. The attempt is not withdrawn — it stays at the target as an **orphan**, holding its worker or queue slot, unreachable by cancellation, and its late answer reaches neither the records nor the policies. Equal deadlines schedule no timer, so the default path is untouched.

**Request context** (`core.RequestContext`, off by default). A client with `ClientConfig.tenant` set gives every request it *starts* a context; a request that arrives with one (`start_request(..., ctx=...)`) keeps it. The client issuing an attempt sets the two hop-relative fields — `attempt` (its own numbering) and the sticky `is_retry` (`incoming.is_retry or attempt > 1`) — and passes the context as `submit_attempt(..., ctx=...)` **only when there is one**, so a target written for three arguments keeps working; `tenant` and the shared, mutable `RetryAllowance` are handed on untouched. `ServiceNodeRuntime` passes its attempt's context to every downstream call, `TransportLink` forwards it, `ServiceRuntime` accepts and ignores it, and the retry policy sees it as `RetryContext.request`. Nothing reads it unless a scenario adds something that does, so a tenant alone changes no event and no draw.

**Chain / fan-out** (`chain.py`). `ServiceNodeRuntime` wraps a `ServiceRuntime` and a list of `DownstreamEdge`s. The node submits a custom `behavior` to its service that does (1) local service time, (2) launches its downstream calls **wave by wave** — every edge of a wave concurrently (a synchronous rejection on an early edge no longer skips the wave's later launches), the next wave once the previous one has fully succeeded — and (3) finalizes only when **all** downstreams succeed (AND semantics — first failure short-circuits the parent, but only after every edge of its wave is launched; waves that had not started are never launched). `DownstreamEdge.wave` (default `0`) names the wave: a handler that calls B, then C, then D is three edges on waves 0, 1, 2. Only the order of the values matters, and a node whose edges all carry the same value is the single concurrent fan-out, event for event. The next wave is started through the event heap at the same instant (as retries are), never from inside the completion callback, so FCFS holds at a callee two waves share; the deadline guard is armed once, before the first wave, and so wins a tie against a wave that completes at the deadline instant. There is one local service-time draw and one failure roll, both before the first wave by default: no local work between waves, no conditional calls. Three off-by-default switches bend this: `ServiceNodeRuntime.local_work="last"` moves the draw and the roll AFTER the last wave (the service time is drawn then; the first wave goes through the heap; the guard is armed at the start; it requires `hold_worker_during_fanout=True` and raises on a tandem node); `DownstreamEdge.optional=True` is a call the node waits for whose failure fails nothing; `ServiceNodeRuntime.wave_join="wait_all"` lets a wave drain before the FIRST required failure in it settles the node (no later wave starts; the deadline guard still does not wait). Worker occupancy during fan-out is controlled by `ServiceNodeRuntime.hold_worker_during_fanout`:

- `False` (**default, tandem-queue**): `release_worker()` is called after the local stage (once, when the first wave has been launched), so the upstream worker is freed while waiting on downstreams. This is load-bearing for the multichain experiments — capacity is *not* consumed down the tree.
- `True` (**synchronous thread-per-request**): the worker is held for the whole subtree (local work + every wave's downstream wait), so capacity is consumed all the way down.

`ServiceNodeRuntime.service_time` (default `None`) gives ONE node its own lognormal (`service.ServiceTime(median, sigma)`) for its local stage. Several nodes can share one `ServiceRuntime` — one per handler of a service — so that methods with different local work share the station's workers, queue, faults and 100 µs floor without sharing a distribution. It is the same single draw (`ServiceTime` floors a zero sigma, it never skips the draw), so `None` is byte-identical to before; `node.cfg` stays the station's config, and `node.local_service_time` is the pair a node actually draws from.

`ServiceNodeRuntime.propagate_deadline` (default `True`) is the third per-node switch: `True` passes the node's own attempt deadline down each edge as `override_global_deadline` (gRPC/Go-context behaviour), `False` lets downstream calls run under their own controller timeouts only — the "no propagated deadline" ablation. The node enforces its own deadline on itself either way.

When adding completion paths in `behavior`, preserve AND semantics and route deadlines through `min(...)` rather than overwriting; the deadline guard schedules an `on_deadline` event that must be cancelled on success, and so must a pending wave event (`_settle` does both).

**Transport delay** (`transport.py`). `TransportLink` is a `ServiceLike` *decorator* — it forwards `cfg` and implements `submit_attempt` — that puts an independent lognormal delay on each leg of an attempt's round trip: `ClientRuntime(cfg=..., service=TransportLink(node, delay))` with `DownstreamEdge(client, node)` unchanged (the link wraps the target *as the edge's client sees it*; `ServiceNodeRuntime.service` stays a `ServiceRuntime`). It has to be a decorator and not a `chain.py` change: retries are issued from `ClientRuntime._start_attempt` and never pass through `chain.py`, so a delay in the fan-out loop would be charged once per root instead of once per attempt. Deadlines keep the strict half-open rule — a forward leg arriving at or after the attempt deadline expires in transit (the target is never called, `admitted_total` untouched, `queue_size_at_end` `None` as for `CIRCUIT_OPEN`), and a response that would land at or after it is clamped to the deadline (or to now, when the target itself reported later still — the past cannot be scheduled) and booked `DEADLINE`, keeping the occupancy and censoring flag the service reported. `cancel()` stays truthful (`True` only when something was really torn down and `on_done` was delivered synchronously) and `on_done` fires exactly once per attempt in every phase. `Request.service_time_ns` remains worker occupancy and **excludes** transport (it is the service-time sample any utilization or `E[S]` estimate is built from); `Request.interval` spans both legs. `TransportDelay.sigma == 0.0` means exactly `median` with no RNG draw (deliberately not floored by `MIN_LOGNORM_SIGMA`), and a scenario with no link draws nothing from `sim.rng()` and schedules no extra event, so the goldens are unaffected.

**Admission control** (`admission.py`). `AdmissionGate(target, controllers, retry_scope)` is a `ServiceLike` decorator like `TransportLink`: every attempt is put to each `AdmissionController` in two phases (all `check`, then all `commit`, so a later refusal spends nobody's quota); a refused attempt is reported **synchronously** as `DropReason.REJECTED` (queue size `None`, an already-settled handle) and never reaches the target; an admitted one is forwarded and every outcome the target reports is `observe`d — orphans included, cancellations not (they only leave the gate's account). The gate needs the attempt's `RequestContext` (it raises without one) and keeps its own `outstanding` / `outstanding_retries` and per-tenant `counts`, updated *before* forwarding because a target may report from inside `submit_attempt`. `retry_scope` is `"hop"` (`attempt > 1`) or `"request"` (the sticky flag). The stock controllers (`TokenBucketBudget`, `AimdBudget`, `SuccessFundedBudget`, `FailureRateGate`, `GoodputCoupledBudget` — Arolla's rule, per tenant — and `ConcurrencyRetryBudget`) refuse retries only and keep exact balances. `ControlledRetryPolicy` applies the same controllers as a layer of a client's own retry policy (charged only for a granted retry, `refund`ed through `rollback`).

**Workload** (`workload.py`) is an exact nonhomogeneous Poisson process for the piecewise-constant rate (base RPS × active `LoadSpike.rps_multiplier`s): one Exp(1) budget per arrival is consumed across rate segments, so short spikes are never skipped and the first arrival is exponentially distributed (never deterministically at t=0). Zero-rate segments are skipped exactly without consuming RNG; an all-zero workload schedules no events.

**Metrics** (`metrics.py`). `Metrics.summary()` is per-root aggregate. `Metrics.export_csv()` produces per-second buckets where:
- `root_requests` and `retries` are bucketed by **start** time.
- `failure_*`, `success_root`, `completed`, percentiles are bucketed by **end** time.
- `failure_retry` counts failed *retry attempts* (not first attempts), bucketed at attempt end.
- Latency stats only include **successful root** latencies completed in that bucket.
- `queue_avg_at_attempt_end` averages `Request.queue_size_at_end` for attempts ending in the bucket.
- An **optional** outcome (`OutcomeReporting.optional`; today only `RootOutcome.REJECTED`, a final attempt refused by admission control) is outside the fixed header: its column is written after every fixed column and only when it occurred, or with `export_csv(..., include_optional=True)`, and `print_stats` skips it at zero. A scenario that cannot produce it exports byte-identical files.

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
`tests/test_trace_goldens.py` pins the same kind of run more tightly — a digest
of every attempt record, the number of events scheduled and the final RNG state
— because everything downstream compares simulator output byte for byte: a new
mechanism must leave those values untouched for as long as it is switched off.

Tests (pytest; tests are `unittest`-style but run under pytest):
```bash
uv run pytest                              # full suite
uv run pytest tests/test_transport.py     # single module
uv run pytest -k chain_worker             # by keyword
```

Example scenarios (write CSVs + JSON to the cwd):
```bash
python -m examples.default
python -m examples.chain_fanout
python -m examples.multichain_retry_storm
```
The tandem vs hold-worker fan-out behaviour is covered by
`tests/test_chain_worker_model.py`; broader chain scenarios (fan-out
monotonicity, deadline propagation, queue pressure) live in
`tests/test_chain_scenarios.py`, and call waves in `tests/test_chain_waves.py`.
`examples/single/` holds focused single-service demos (circuit breakers, backoff variants, rate limiters, retry budget).
The DeathStarBench hotelReservation model (`hotel_matched`) is no longer here: it is the `hotel-sim` project in `microservices-system/benchmarks/hotelreservation/sim`, which depends on this package from git.

Plotting (consumes a CSV + matching fault-events JSON; the time axis is shared):
```bash
python -m msim.plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json
python -m msim.plotting.latencies_over_time chain_output.csv --fault-events chain_fault_events.json
python -m msim.plotting.queue_size_over_time chain_output.csv --fault-events chain_fault_events.json
python -m msim.plotting.qps_over_time chain_output.csv --fault-events chain_fault_events.json
./plot.sh   # batch-render the standard chain figures
```

## Conventions when extending

- `src/msim/compat/<name>/` holds readers for other simulators' scenario formats (today `arolla`: `docs/compat-arolla.md`). A reader maps a foreign file onto msim's own parts; a mechanism the mapping needs goes into the core package, generic and off by default, never into the reader. Each reader's dependency is an extra of the same name, imported lazily, so the core stays `numpy` + stdlib. A valid foreign scenario the reader cannot build must raise (`UnsupportedScenario`), not approximate.
- New library modules go in `src/msim/` and are imported absolutely (`from msim.<module> import ...`, never relative). `known-first-party` and the coverage `source` list cover the whole package, so nothing in `pyproject.toml` needs editing per module. Plotting scripts live in `src/msim/plotting/` (heavy deps go in the `plot` extra); runnable demos stay in `examples/`. Benchmark-specific scenarios belong with the benchmark (e.g. `hotel-sim` in `microservices-system`), not here.
- New retry policies that wrap another should expose `underlying_strategy` so `_update_policy_results`, `_rollback_policy` *and* `_notify_request_start` reach them. A policy that chooses between several (`RetryByCausePolicy`, which picks the branch by the failed attempt's `RetryContext.drop_reason`; a cause with no entry is not retried) implements `sub_policies(context=None)` instead: every branch without a context, only the deciding branch with one. `policy_layers(policy, context)` yields each distinct layer once by identity, so a budget shared by two causes gets one deposit per request, and a refund follows only the branch that granted the retry. New stateful policies should implement `add_result(success, now)` (once per attempt; override `record_attempt(context, success)` instead when the attempt's number, cause or request context matters) and/or `on_request_start(now)` (once per admitted root request — that's the hook `RetryBudgetPolicy` deposits on, since its ratio is retries per *request*).
- If a policy's `next_delay` has **side effects** (withdrawing budget, spending a rate-limiter token, reserving a leaky-bucket slot), it must also implement `rollback(context)` to undo exactly one such charge. The client asks for a delay before it knows the global deadline will veto the retry; without a refund, retries that are never issued are still charged. `RetryPolicy.rollback` defaults to a no-op, so side-effect-free policies need nothing.
- Custom service behaviors must call `finalize` exactly once and `release_worker` at most once (both raise on repeats). Completion callbacks must not submit work synchronously if FCFS matters to your experiment — schedule through the event heap instead.
- A new `ServiceLike` decorator (the `transport.py` pattern) must forward `cfg` so attempts stay booked under the target's name, accept the keyword-only `ctx` and pass it on to its target only when it is not `None`, return an `AttemptControl` *immediately* (never after the target reports), keep `cancel()` truthful — `True` only when something was actually torn down and `on_done` was already delivered synchronously — and call `on_done` exactly once per attempt on every path.
- A new `AdmissionController` must not spend anything in `check` (bringing a time-based balance up to date is fine), must spend only in `commit`, and should implement `refund` if it can be used in a `ControlledRetryPolicy`. Derive from `RetryController` unless it really means to refuse first attempts.
- New downstream-graph features must preserve AND semantics in `chain.py`'s `behavior` and route deadlines through `min(...)` rather than overwriting. Every completed call of a wave goes through `_call_ended` and a drained wave through `_wave_done`, the one place that decides what follows it (fail with a remembered failure, the next wave, the local stage, or success): put a new rule there rather than in a completion callback. They must also leave a node whose edges share one `wave` value on the exact event sequence it has today: no extra scheduled event, no extra RNG draw, and `release_worker` still called once after the first wave is launched (it resumes the queue, whose next attempt draws its service time).
- A new mechanism defaults to OFF and its default path must be byte-identical: append new dataclass fields last, with defaults, and add a seeded trace to `tests/test_trace_goldens.py` for the mechanism switched on.
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
