# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Discrete-event simulator of microservice request flows: queues, retries, timeouts, partial failures, latency injections, load spikes. Simulates single services, chains, and fan-outs, exports per-bucket CSV metrics + fault-event JSON for plotting, and exposes the simulator as a Gymnasium environment for RL experiments on retry/timeout policies.

There is no `pyproject.toml`, `requirements.txt`, or installable package — everything is run as `python -m <module>` from the repo root, against an in-tree `.venv`. Imports are flat (`from core import ...`, `from service import ...`), so cwd must be the repo root.

## Architecture

The simulator is built around a single-threaded discrete-event loop. Events are pushed onto a heap with `(at, seq)` ordering, popped in time order, and `Simulator.timestep` advances to the event's `at` before its action runs (`simulator.py`). Every random draw goes through `Simulator.rng()` so a single seed makes service times, failures, and jitter reproducible — `Workload` deliberately keeps a *separate* RNG (`workload.py`) so arrival processes stay identical across experiments even when the simulator seed changes.

**Time** is integer nanoseconds end-to-end (`TimePoint`, `TimeDuration` in `core.py`). Convert at the boundary with `s_to_ns` / `ms_to_ns` / `ns_to_ms`. Plots and CSVs emit seconds — that conversion happens in `metrics.export_csv` and `fault_events.export_json`, not in the core loop.

**Service queue** (`service.py`) is M/G/c/K: FCFS, `workers` concurrency, optional `queue_capacity`. Service time is lognormal with `latency_median = exp(mu)` and `latency_lognorm_sigma`, then mutated by any active `LatencyInjection` (additive `add_latency` and integer `multiplier`, with a 100 µs floor). `PartialFailure`s set a per-time `p_fail` rolled at attempt end. Drops are decided in three places: queue-full at submit, deadline at dequeue *and* at finish, server-failure post-service.

**Client + retry** (`caller.py`). `ClientRuntime.start_request` builds a `RootRequest`, computes `global_deadline = min(controller_global_timeout, override)`, and runs attempts until success, deadline, or the policy stops retrying. Two non-obvious invariants:

- **Half-open deadline semantics**: success requires `end_time < deadline` (strict). If a request *finishes* exactly at the deadline it counts as a deadline drop, not a success. Preserve this when adding new completion paths.
- **Policy result fan-out**: `_update_policy_results` walks `policy.underlying_strategy` recursively and calls `add_result(success, now)` on every layer that has it. Circuit breakers and retry budgets *only* update through this path — if you add a new wrapping policy, it must expose `underlying_strategy` (or `add_result`) or it will silently never see results.

**Chain / fan-out** (`chain.py`). `ServiceNodeRuntime` wraps a `ServiceRuntime` and a list of `DownstreamEdge`s. The node submits a custom `behavior` to its service that does (1) local service time, (2) launches downstream calls *concurrently*, (3) finalizes only when **all** downstreams succeed (AND semantics — first failure short-circuits the parent). The node calls `release_worker()` after the local stage so the upstream worker is freed while waiting on fan-out; this is the **tandem-queue** model and is the default. Don't change it to "hold the worker until downstreams finish" without understanding that capacity model — it's load-bearing for the multichain experiments.

**Workload** (`workload.py`) is a Poisson arrival process (`expovariate`) modulated by `LoadSpike.rps_multiplier`. RPS=0 means no arrivals but the tick keeps polling at 0.5s intervals.

**Metrics** (`metrics.py`). `Metrics.summary()` is per-root aggregate. `Metrics.export_csv()` produces per-second buckets where:
- `root_requests` and `retries` are bucketed by **start** time.
- `failure_*`, `success_root`, `completed`, percentiles are bucketed by **end** time.
- `failure_retry` counts failed *retry attempts* (not first attempts), bucketed at attempt end.
- Latency stats only include **successful root** latencies completed in that bucket.
- `queue_avg_at_attempt_end` averages `Request.queue_size_at_end` for attempts ending in the bucket.

**Fault events** (`fault_events.py`) is a separate tracker that mirrors faults already configured on services/workloads, but emits them in seconds-based JSON for plotting overlays. Both `ServiceConfig.register_fault_events(tracker)` and `Workload.register_fault_events(tracker)` must be called explicitly — registration is *not* automatic.

## RL layer (`rl/`)

`MicroservicesEnv` (`rl/env.py`) wraps a build function (`build_single_service_env` or `build_faulty_single_service_env`) into a Gymnasium env. Each `step()` advances the simulator by `step_duration_s` and returns reward + observation summarized over roots that completed in that window. The action is a `MultiDiscrete` index into `ActionSpec` lists for `(attempt_timeout_ms, max_attempts, backoff_ms[, global_timeout_ms])`. Hard constraint: `_apply_action` mutates `StaticTimeoutController.attempt_timeout` and `FixedBackoffRetryPolicy.delay/max_attempts` directly — it raises `TypeError` on any other policy/controller types. If you swap the underlying policy in a `build_*` function, update `_apply_action` accordingly.

`rl/policies/` ships baselines (Fixed, AIMD, ModelBased, Bayesian, SB3 PPO, oracle-fixed grid search). `ModelBasedPolicy` optionally adds an M/M/c queueing-delay term. `rl/models/` has the latency models used by Bayesian/ModelBased: lognormal, lognormal mixture, EVT tail.

`rl/policies/sb3.py` adapts a Stable-Baselines3 model to the `Policy` protocol. PPO training entry point is `rl.train_sb3`; multi-policy comparison (incl. early-stopped PPO and oracle search) is `rl.experiments.compare_policies`.

## Common commands

Tests:
```bash
python -m unittest discover -s tests -v
python -m unittest tests.test_rl_env -v   # single test module
```

Single-service / chain experiments (write CSVs + JSON to repo root):
```bash
python -m experiments.default
python -m experiments.chain_fanout
python -m experiments.multichain_retry_storm
```

Plotting (consumes the CSV + matching fault-events JSON):
```bash
python -m plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json
python -m plotting.latencies_over_time chain_output.csv --fault-events chain_fault_events.json
python -m plotting.queue_size_over_time chain_output.csv --fault-events chain_fault_events.json
python -m plotting.qps_over_time chain_output.csv --fault-events chain_fault_events.json
./plot.sh   # batch-render the standard chain figures
```

RL:
```bash
# Compare latency models on synthetic data
python -m rl.experiments.compare_models --dist {lognormal,mixture,pareto}

# Compare baseline policies (uses faulty env when --fault-scenario basic)
python -m rl.experiments.compare_policies \
    --policies fixed,aimd,model_based,model_based_queue,bayesian,ppo,oracle_fixed \
    --fault-scenario basic

# Train PPO standalone
python -m rl.train_sb3 --timesteps 5000 --save-path rl_models/ppo.zip
```

Optional dependencies (only required for the corresponding entry points): `gymnasium`, `stable-baselines3`. The simulator core has no external deps beyond `numpy` and the stdlib.

## Conventions when extending

- New retry policies that wrap another should expose `underlying_strategy` so `_update_policy_results` reaches them. New stateful policies should implement `add_result(success, now)`.
- New downstream-graph features must preserve AND semantics in `chain.py`'s `behavior` and route deadlines through `min(...)` rather than overwriting.
- After running an experiment, call `Metrics.export_csv()` *and* `FaultEventsTracker.export_json()` so plotting overlays line up — the time axis is shared.
- Generated artifacts (`*.csv`, `*.json`, `*.png`, `results/`) are gitignored. The committed `*_output.csv` / `*_fault_events.json` files at the repo root are sample outputs from prior runs; treat them as disposable.
