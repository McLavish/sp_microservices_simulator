# CLAUDE.md

This file provides guidance to the Agent when working with code in this repository.

## What this is

Discrete-event simulator of microservice request flows: queues, retries, timeouts, partial failures, latency injections, load spikes. Simulates single services, chains, and fan-outs, exports per-bucket CSV metrics + fault-event JSON for plotting, and exposes the simulator as a Gymnasium environment for RL experiments on retry/timeout policies.

The project is uv-managed but **not an installable package** (`pyproject.toml` sets `package = false`). Everything runs as `python -m <module>` from the repo root against the in-tree `.venv`. Imports are flat (`from core import ...`, `from service import ...`), so the cwd must be the repo root. The simulator core depends only on `numpy` + stdlib; `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `gymnasium`, `stable-baselines3` are dev/optional deps pulled in only by plotting, RL entry points, and `Metrics.export_csv` (which imports `pandas` lazily at export time).

Deeper reference docs live in `docs/` — start with `docs/architecture.md`, then the subsystem you're touching (`services-and-workload`, `request-handling`, `observability`, `experiments`, `rl`). Keep them in sync when changing the behavior they describe. `AGENTS.md` is a symlink to this file — edit `CLAUDE.md`, never create a separate agents file.

## Architecture

The simulator is built around a single-threaded discrete-event loop. Events are pushed onto a heap with `(at, seq)` ordering, popped in time order, and `Simulator.timestep` advances to the event's `at` before its action runs (`simulator.py`). Every random draw goes through `Simulator.rng()` so a single seed makes service times, failures, and jitter reproducible — `Workload` deliberately keeps a *separate* RNG (`workload.py`) so arrival processes stay identical across experiments even when the simulator seed changes.

**Time** is integer nanoseconds end-to-end (`TimePoint`, `TimeDuration` in `core.py`). Convert at the boundary with `s_to_ns` / `ms_to_ns` / `ns_to_ms`. Plots and CSVs emit seconds — that conversion happens in `metrics.export_csv` and `fault_events.export_json`, not in the core loop.

**Service queue** (`service.py`) is M/G/c/K: FCFS, `workers` concurrency, optional `queue_capacity`. Service time is lognormal with `latency_median = exp(mu)` and `latency_lognorm_sigma`, then mutated by any active `LatencyInjection` (additive `add_latency` and integer `multiplier`, with a 100 µs floor). `PartialFailure`s set a per-time `p_fail` rolled at attempt end. Drops are decided in three places: queue-full at submit, deadline at dequeue *and* at finish, server-failure post-service.

**Client + retry** (`caller.py`). `ClientRuntime.start_request` builds a `RootRequest`, computes `global_deadline = min(controller_global_timeout, override)`, and runs attempts until success, deadline, or the policy stops retrying. Two non-obvious invariants:

- **Half-open deadline semantics**: success requires `end_time < deadline` (strict). If a request *finishes* exactly at the deadline it counts as a deadline drop, not a success. Preserve this when adding new completion paths.
- **Policy result fan-out**: `_update_policy_results` walks `policy.underlying_strategy` recursively and calls `add_result(success, now)` on every layer that has it. Circuit breakers and retry budgets *only* update through this path — if you add a new wrapping policy, it must expose `underlying_strategy` (or `add_result`) or it will silently never see results.

**Chain / fan-out** (`chain.py`). `ServiceNodeRuntime` wraps a `ServiceRuntime` and a list of `DownstreamEdge`s. The node submits a custom `behavior` to its service that does (1) local service time, (2) launches downstream calls *concurrently*, (3) finalizes only when **all** downstreams succeed (AND semantics — first failure short-circuits the parent). Worker occupancy during fan-out is controlled by `ServiceNodeRuntime.hold_worker_during_fanout`:

- `False` (**default, tandem-queue**): `release_worker()` is called after the local stage, so the upstream worker is freed while waiting on downstreams. This is load-bearing for the multichain experiments — capacity is *not* consumed down the tree.
- `True` (**synchronous thread-per-request**): the worker is held for the whole subtree (local work + downstream wait), so capacity is consumed all the way down.

When adding completion paths in `behavior`, preserve AND semantics and route deadlines through `min(...)` rather than overwriting; the deadline guard schedules an `on_deadline` event that must be cancelled on success.

**Workload** (`workload.py`) is an exact nonhomogeneous Poisson process for the piecewise-constant rate (base RPS × active `LoadSpike.rps_multiplier`s): one Exp(1) budget per arrival is consumed across rate segments, so short spikes are never skipped and the first arrival is exponentially distributed (never deterministically at t=0). Zero-rate segments are skipped exactly without consuming RNG; an all-zero workload schedules no events.

**Metrics** (`metrics.py`). `Metrics.summary()` is per-root aggregate. `Metrics.export_csv()` produces per-second buckets where:
- `root_requests` and `retries` are bucketed by **start** time.
- `failure_*`, `success_root`, `completed`, percentiles are bucketed by **end** time.
- `failure_retry` counts failed *retry attempts* (not first attempts), bucketed at attempt end.
- Latency stats only include **successful root** latencies completed in that bucket.
- `queue_avg_at_attempt_end` averages `Request.queue_size_at_end` for attempts ending in the bucket.

**Fault events** (`fault_events.py`) is a separate tracker that mirrors faults already configured on services/workloads, but emits them in seconds-based JSON for plotting overlays. Both `ServiceConfig.register_fault_events(tracker)` and `Workload.register_fault_events(tracker)` must be called explicitly — registration is *not* automatic.

## RL layer (`rl/`)

`MicroservicesEnv` (`rl/env.py`) wraps a build function into a Gymnasium env; `SCENARIO_BUILDERS` maps the `--fault-scenario` names: `none`, `basic` (fixed fault windows — memorizable, prefer `randomized` for training), `randomized` (fault onset/duration/intensity drawn from the seed), `chain` (A→B→C retry-storm topology: the agent tunes the B→C edge knobs, reward comes from the top-level client), `chain_budget` (edge wrapped in `RetryBudgetPolicy`), `multichain` (two chains A→B→C and X→Y→C share bottleneck C, which degrades with pure partial-failure blips so edge retries genuinely recover successes; the fixed-policy bystander chain is exposed as `EnvComponents.bystander_client` and its success rate gauges the retry externality — the scenario where the retry budget actually binds), and `multichain_budget`. The chain builders expose the edge as `EnvComponents.cost_client`; its retry amplification is the constraint cost (`summary["cost"]`, weighted by `RewardWeights.cost`) used by the Lagrangian-PPO trainer in `rl/constrained.py` (dual ascent on the multiplier; three-arm study in `rl.experiments.constrained_retry`, which takes `--scenario {chain,multichain}` and a `--calibrate` mode that grid-checks the budget can bind before training). Each `step()` advances the simulator by `step_duration_s` and returns reward + observation summarized over roots that completed in that window; on episode end, in-flight work is drained *after* the final reward and reported under `info["drained_summary"]`. The action is a `MultiDiscrete` index into `ActionSpec` lists for `(attempt_timeout_ms, max_attempts, backoff_ms[, global_timeout_ms])`. Knob application is owned by the builder: each `build_*` function returns `EnvComponents` with an `apply_knobs` callback that maps decoded `ActionValues` onto its own policy/controller objects — the env itself is policy-agnostic, so new scenarios (chains, other retry policies) only need a builder that wires its own `apply_knobs`. Unseeded `reset()` draws a fresh build seed from the gym RNG; never make builders fall back to fixed seeds on the env path or SB3 training collapses to one episode.

`rl/policies/` ships baselines (Fixed, AIMD, ModelBased, Bayesian, SB3 PPO, oracle-fixed grid search). `ModelBasedPolicy` optionally adds an M/G/c queueing-delay term (Erlang-C / Allen-Cunneen, arrival rate per millisecond to match the ms latency models); timed-out attempts are kept as right-censored samples and fitted via censored MLE (`LogNormalModel.fit_censored`). `rl/models/` has the latency models used by Bayesian/ModelBased: lognormal (`lognormal.py`), lognormal mixture (`lognormal_mixture.py`), EVT tail (`evt_tail.py`).

`rl/policies/sb3.py` adapts a Stable-Baselines3 model to the `Policy` protocol. PPO training entry point is `rl.train_sb3`; multi-policy comparison (incl. early-stopped PPO and oracle search) is `rl.experiments.compare_policies`, which writes results/trace/learning-curve CSVs and supports `--ppo-save-path`/`--ppo-load-path` for cross-scenario evaluation. stable-baselines3 lives in the opt-in `rl` dependency group (`uv sync --group dev --group rl`; pulls in torch).

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

Tests (pytest; tests are `unittest`-style but run under pytest):
```bash
uv run pytest                              # full suite
uv run pytest tests/test_rl_env.py        # single module
uv run pytest -k chain_worker             # by keyword
```

Single-service / chain experiments (write CSVs + JSON to repo root):
```bash
python -m experiments.default
python -m experiments.chain_fanout
python -m experiments.multichain_retry_storm
python -m experiments.hotel_matched   # DeathStarBench hotelReservation approximation -> hotel_matched_results/
```
`experiments/hotel_matched.py` maps the hotel read workload onto two chains sharing `profile`; `--calibration-json` takes the calibration JSON emitted by the external `valharness` project (`orchestrator.record_real`) and falls back to built-in defaults without it.
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

RL:
```bash
python -m rl.experiments.compare_models --dist {lognormal,mixture,pareto}
python -m rl.experiments.compare_policies \
    --policies fixed,aimd,model_based,model_based_queue,bayesian,ppo,oracle_fixed \
    --fault-scenario randomized   # any SCENARIO_BUILDERS name: none, basic, randomized, chain, chain_budget, multichain, multichain_budget
python -m rl.train_sb3 --timesteps 100000 --fault-scenario randomized --save-path rl_models/ppo.zip
python -m rl.experiments.constrained_retry --budgets 0.05,0.1,0.2 --device cpu  # Lagrangian PPO vs RetryBudgetPolicy
python -m rl.experiments.constrained_retry --scenario multichain --calibrate --device cpu  # gate: budget must bind
python -m rl.experiments.constrained_retry --scenario multichain --budgets 0.05,0.1,0.2 --device cpu
```

RL result plots (consume the CSVs the experiments above write):
```bash
python -m plotting.rl_pareto rl_constrained_comparison_multichain.csv   # goodput-vs-amplification Pareto + bystander panel
python -m plotting.rl_traces_over_time rl_traces_basic.csv              # per-step knob/success traces; fault overlay only lines up for `basic`
```

## Conventions when extending

- New retry policies that wrap another should expose `underlying_strategy` so `_update_policy_results` reaches them. New stateful policies should implement `add_result(success, now)`.
- New downstream-graph features must preserve AND semantics in `chain.py`'s `behavior` and route deadlines through `min(...)` rather than overwriting.
- After running an experiment, call `Metrics.export_csv()` *and* `FaultEventsTracker.export_json()` so plotting overlays line up.
- Keep internal time in nanoseconds; convert only at I/O / plotting boundaries.
- `ty` is lenient on missing annotations but real type errors (None narrowing, protocol mismatches, numpy misuse) are kept clean — don't reintroduce them.
- Generated artifacts (`*.csv`, `*.json`, `*.png`, `results/`) are gitignored. The committed `*_output.csv` / `*_fault_events.json` files at the repo root are sample outputs from prior runs; treat them as disposable.

## Git Commits

- Commit after every meaningful, self-contained change — don't batch unrelated work.
- Each commit should leave the codebase in a working state.
- Write short, clear commit messages that describe what changed.
- If a task involves multiple logical steps, make a separate commit for each step.
- Stage only the files relevant to the current change.
