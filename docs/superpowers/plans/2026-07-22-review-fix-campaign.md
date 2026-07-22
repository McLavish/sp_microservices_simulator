# Review Fix Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all confirmed defects from the external review (spec: `docs/superpowers/specs/2026-07-22-review-fix-campaign-design.md`) in five severity waves, one commit per finding, TDD throughout.

**Architecture:** Discrete-event simulator (integer-ns time, single-threaded heap loop) with an RL layer. Fixes touch the trainer loop, workload sampling, env accounting, service/chain callbacks, retry policies, metrics/plotting, and CLIs. No new subsystems; every change slots into an existing module.

**Tech Stack:** Python 3.13, uv-managed venv (NOT installable — run from repo root, flat imports), pytest (unittest-style tests), ruff, ty, stable-baselines3 2.8.0 (optional dep), pandas/matplotlib (dev deps).

## Global Constraints

- Work on branch `dev` (already checked out). Wave 5 merges into `paper-branch`.
- Run everything from the repo root: `uv run --no-sync pytest ...`, `uv run --no-sync python -m ...`.
- One commit per task, message style: short imperative, no prefix tags (match `git log`).
- Before each commit: `uv run --no-sync ruff format <changed files> && uv run --no-sync ruff check <changed files>` and the task's tests green. Run `make check` at each wave end (it includes ty + full pytest).
- Internal time stays integer nanoseconds; convert only at I/O boundaries (`s_to_ns`/`ms_to_ns`/`ns_to_ms`/`ns_to_s` from `core.py`).
- Tests are `unittest`-style classes under `tests/`, run via pytest. SB3-dependent tests guard with `@unittest.skipIf(PPO is None, "stable-baselines3 not installed")`.
- NEVER modify or regenerate the pinned paper CSVs (`rl_constrained_*.csv` — they exist only on `paper-branch`).
- Docs updated in the same commit as the behavior they describe (CLAUDE.md rule).
- MPLBACKEND=Agg for any test importing matplotlib (set `os.environ.setdefault("MPLBACKEND", "Agg")` before the plotting import).

---

## Wave 1 — results integrity

### Task 1: Restore the format gate

**Files:**
- Modify: `experiments/hotel_matched.py:262` (formatting only)

- [ ] **Step 1: Format and verify**

```bash
uv run --no-sync ruff format experiments/hotel_matched.py
uv run --no-sync ruff format --check .
```
Expected: `73 files already formatted` (zero would-reformat).

- [ ] **Step 2: Run the existing hotel tests**

Run: `uv run --no-sync pytest tests/test_hotel_matched_experiment.py -q`
Expected: PASS (formatting-only change).

- [ ] **Step 3: Commit**

```bash
git add experiments/hotel_matched.py
git commit -m "Format hotel_matched leaf-config mapping (restore make check gate)"
```

### Task 2: Fix constrained-trainer overtraining (finding #1)

**Files:**
- Modify: `rl/constrained.py:156-170`
- Test: `tests/test_rl_constrained.py` (append)

**Interfaces:**
- Produces: `train_constrained_ppo` now trains exactly `cfg.timesteps` env steps (± one rollout of `n_steps * n_envs`); curve rows' `timesteps` column records `model.num_timesteps` (true cumulative).

**Background (why):** SB3 ≥1.x `_setup_learn` does `total_timesteps += self.num_timesteps` when `reset_num_timesteps=False`, so each `learn(T, reset_num_timesteps=False)` trains **T additional** steps. The current loop passes the cumulative `trained` → a 30k/5k config trains 105k steps.

- [ ] **Step 1: Write the failing test** (append to `tests/test_rl_constrained.py`, inside `TestConstrainedTraining`)

```python
    def test_chunked_training_respects_total_timesteps(self):
        # SB3 trains `total_timesteps` ADDITIONAL steps when
        # reset_num_timesteps=False; the loop must pass per-chunk increments.
        # chunk=4 with n_steps=2, n_envs=1, batch=2 divides evenly, so there is
        # no rollout-rounding overshoot and the count is exact.
        cfg = ConstrainedTrainConfig(
            build=lambda seed: build_chain_env(seed, duration_s=3),
            action_spec=ActionSpec(
                attempt_timeout_ms=[50, 80], max_attempts=[1, 2], backoff_ms=[20]
            ),
            episode_duration_s=3.0,
            timesteps=8,
            chunk_timesteps=4,
            n_envs=1,
            n_steps=2,
            batch_size=2,
            eval_episodes=1,
        )
        policy = train_constrained_ppo(cfg)
        # Buggy cumulative loop: learn(4) + learn(8) -> 4 + 8 = 12 steps.
        self.assertEqual(policy.model.num_timesteps, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_rl_constrained.py -k respects_total -q`
Expected: FAIL with `AssertionError: 12 != 8`

- [ ] **Step 3: Fix the loop** in `rl/constrained.py` (replace lines 156-170)

```python
    lam = cfg.lambda_init
    curve: list[tuple[int, float, float, float]] = []
    trained = 0
    while trained < cfg.timesteps:
        chunk = min(cfg.chunk_timesteps, cfg.timesteps - trained)
        # SB3 semantics: with reset_num_timesteps=False, learn() trains
        # `total_timesteps` ADDITIONAL steps (_setup_learn adds the current
        # num_timesteps). Pass the per-chunk increment; the counter, schedules,
        # and in-flight episodes still continue across chunks.
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        trained = int(model.num_timesteps)  # true cumulative; curve rows stay honest
        eval_objective, eval_cost = evaluate_objective_and_cost(
            model, eval_env, cfg.eval_episodes, cfg.base_weights
        )
        lam = dual_ascent_update(lam, eval_cost, cfg.budget, cfg.lambda_lr)
        vec_env.env_method("set_cost_weight", lam)
        curve.append((trained, eval_objective, eval_cost, lam))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_rl_constrained.py -q`
Expected: PASS (all, including the two existing smoke tests).

- [ ] **Step 5: Commit**

```bash
git add rl/constrained.py tests/test_rl_constrained.py
git commit -m "Fix constrained trainer overtraining: pass per-chunk increments to learn()"
```

### Task 3: Hotel calibration — explicit-empty fault scenario must mean no faults (finding #9)

**Files:**
- Modify: `experiments/hotel_matched.py:76-82` and `:355-370`
- Test: `tests/test_hotel_matched_experiment.py` (append)

**Interfaces:**
- Produces: `_profile_faults_from_calibration(calibration) -> list[tuple[float, float, float, float]] | None` (None = key absent → defaults; list, possibly empty = respect verbatim). New module function `_effective_profile_faults(config: HotelMatchedConfig, calibration: dict) -> list[tuple[float, float, float, float]]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_hotel_matched_experiment.py`)

```python
from experiments.hotel_matched import (
    DEFAULT_PROFILE_FAULT,
    HotelMatchedConfig,
    _effective_profile_faults,
)


class TestProfileFaultSelection(unittest.TestCase):
    def test_absent_fault_scenario_falls_back_to_default(self):
        self.assertEqual(
            _effective_profile_faults(HotelMatchedConfig(), {}), DEFAULT_PROFILE_FAULT
        )

    def test_explicit_empty_fault_scenario_means_no_faults(self):
        calibration = {"fault_scenario": []}
        self.assertEqual(_effective_profile_faults(HotelMatchedConfig(), calibration), [])

    def test_non_profile_only_scenario_means_no_profile_faults(self):
        calibration = {
            "fault_scenario": [
                {
                    "method": "/search.Search/Nearby",
                    "start_s": 5,
                    "end_s": 10,
                    "add_latency_ms": 20,
                    "p_fail": 0.1,
                }
            ]
        }
        self.assertEqual(_effective_profile_faults(HotelMatchedConfig(), calibration), [])

    def test_config_override_wins(self):
        cfg = HotelMatchedConfig(profile_faults=[(1.0, 2.0, 3.0, 0.4)])
        self.assertEqual(
            _effective_profile_faults(cfg, {"fault_scenario": []}), [(1.0, 2.0, 3.0, 0.4)]
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --no-sync pytest tests/test_hotel_matched_experiment.py -k ProfileFaultSelection -q`
Expected: FAIL with `ImportError: cannot import name '_effective_profile_faults'`

- [ ] **Step 3: Implement.** In `experiments/hotel_matched.py`, change `_profile_faults_from_calibration` to distinguish absent from present-but-empty:

```python
def _profile_faults_from_calibration(
    calibration: dict[str, Any],
) -> list[tuple[float, float, float, float]] | None:
    """None when the calibration has no fault_scenario key (fall back to the
    built-in default); a list — possibly empty — when it does. An explicit
    empty scenario, or one naming only non-profile methods, means "no profile
    faults" and must not be replaced by the default."""
    scenario = calibration.get("fault_scenario")
    if scenario is None:
        return None
    windows = []
    for window in scenario:
        if window.get("method") != "/profile.Profile/GetProfiles":
            continue
        windows.append(
            (
                float(window["start_s"]),
                float(window["end_s"]),
                float(window["add_latency_ms"]),
                float(window["p_fail"]),
            )
        )
    return windows


def _effective_profile_faults(
    config: HotelMatchedConfig, calibration: dict[str, Any]
) -> list[tuple[float, float, float, float]]:
    if config.profile_faults is not None:
        return config.profile_faults
    calibrated = _profile_faults_from_calibration(calibration)
    return calibrated if calibrated is not None else DEFAULT_PROFILE_FAULT
```

Then in `run_hotel_matched` replace the selection expression (lines 78-82) with:

```python
    profile_faults = _effective_profile_faults(config, calibration)
```

- [ ] **Step 4: Run tests**

Run: `uv run --no-sync pytest tests/test_hotel_matched_experiment.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/hotel_matched.py tests/test_hotel_matched_experiment.py
git commit -m "hotel_matched: explicit-empty calibration fault scenario means no faults"
```

### Task 4: Env population split + optional terminal drain penalty (finding #3)

**Files:**
- Modify: `rl/env.py` (EnvConfig, `step()`, new `in_horizon_summary()`), `rl/experiments/compare_policies.py:141-172`, `rl/experiments/constrained_retry.py:73-134`
- Test: `tests/test_rl_env.py` (append)

**Interfaces:**
- Produces: `EnvConfig.terminal_drain_penalty: bool = False`; `MicroservicesEnv.in_horizon_summary() -> Summary | None` (roots completed before the horizon); existing `metrics_summary()` unchanged (full population). Results dicts/CSVs gain `success_rate_in_horizon` and `p95_ms_in_horizon` **columns added, old columns keep their full-population meaning** (back-compat with pinned paper CSVs — deliberate deviation from the spec's "rename" wording, noted in the spec's ledger by Task 23).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rl_env.py`)

```python
from caller import ClientConfig, ClientRuntime
from retry_policies import FixedBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload
from core import ms_to_ns, s_to_ns
from faults import PartialFailure
from core import TimeInterval
from rl.env import EnvComponents, EnvConfig, MicroservicesEnv
from rl.spec import ActionSpec


def _build_slow_failing_env(seed=None, duration_s=None):
    """Every request fails after ~2.5 s of service: with a long attempt timeout
    all failures land in the drain; with a short one they land in-episode."""
    build_seed = 1 if seed is None else int(seed)
    sim = Simulator(seed=build_seed)
    duration = 2.0 if duration_s is None else float(duration_s)
    cfg = ServiceConfig(
        name="svc-slow",
        latency_median=ms_to_ns(2500),
        latency_lognorm_sigma=1e-6,
        workers=256,
        partial_failures=[
            PartialFailure(
                duration=TimeInterval(begin=0, end=s_to_ns(10_000)), p_fail=1.0
            )
        ],
    )
    timeout_ctrl = StaticTimeoutController(global_timeout=None, attempt_timeout=ms_to_ns(50))
    retry = FixedBackoffRetryPolicy(max_attempts=1, delay=0)
    client = ClientRuntime(
        cfg=ClientConfig(retry=retry, timeout_controller=timeout_ctrl),
        service=ServiceRuntime(cfg=cfg).bind(),
    )

    def apply_knobs(values):
        timeout_ctrl.attempt_timeout = ms_to_ns(values.attempt_timeout_ms)
        retry.max_attempts = values.max_attempts
        retry.delay = ms_to_ns(values.backoff_ms)

    workload = Workload(base_rps=50.0, duration_s=int(duration), load_spikes=[], rng_seed=seed)
    return EnvComponents(sim=sim, workload=workload, client=client, apply_knobs=apply_knobs)


SLOW_SPEC = ActionSpec(attempt_timeout_ms=[50, 100_000], max_attempts=[1], backoff_ms=[0])


class TestPopulationAccounting(unittest.TestCase):
    def _run_episode(self, action, terminal_drain_penalty=False):
        env = MicroservicesEnv(
            EnvConfig(
                build=_build_slow_failing_env,
                action_spec=SLOW_SPEC,
                step_duration_s=1.0,
                episode_duration_s=2.0,
                terminal_drain_penalty=terminal_drain_penalty,
                initial_action=action,
            )
        )
        obs, info = env.reset(seed=42)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
        return env, total, info

    def test_summaries_partition_the_population(self):
        env, _, info = self._run_episode((1, 0, 0))  # 100 s timeout: all failures drain
        full = env.metrics_summary()
        horizon = env.in_horizon_summary()
        drained = info["drained_summary"]
        self.assertEqual(full.total, horizon.total + drained["completed"])
        self.assertGreater(drained["completed"], 0)

    def test_terminal_drain_penalty_removes_deferral_advantage(self):
        _, j_defer_off, _ = self._run_episode((1, 0, 0))
        _, j_fail_off, _ = self._run_episode((0, 0, 0))
        # Bug being locked out: deferring failures past the horizon scores better.
        self.assertGreater(j_defer_off, j_fail_off)
        _, j_defer_on, _ = self._run_episode((1, 0, 0), terminal_drain_penalty=True)
        _, j_fail_on, _ = self._run_episode((0, 0, 0), terminal_drain_penalty=True)
        self.assertLess(abs(j_defer_on - j_fail_on), abs(j_defer_off - j_fail_off))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --no-sync pytest tests/test_rl_env.py -k PopulationAccounting -q`
Expected: FAIL with `TypeError: EnvConfig.__init__() got an unexpected keyword argument 'terminal_drain_penalty'`

- [ ] **Step 3: Implement in `rl/env.py`.**

(a) Add to `EnvConfig` (after `max_latency_samples`):

```python
    # Fold drained-root outcomes into the final step's reward. Default off:
    # the windowed objective is the documented design and keeps trained
    # behavior comparable with the pinned paper data; turning it on removes
    # the defer-failures-past-the-horizon incentive for future studies.
    terminal_drain_penalty: bool = False
```

(b) In `__init__`, add `self._horizon_root_count = 0`; in `reset()`, reset it to 0 next to `self._last_root_idx = 0`.

(c) In `step()`, replace the terminated/drain block with:

```python
        if terminated:
            self._horizon_root_count = self._last_root_idx
        # Drain in-flight work after the reward is computed: drained roots complete
        # past the episode horizon and must not be credited to the final action
        # (unless terminal_drain_penalty explicitly folds them in).
        if terminated and self._cfg.drain_on_done:
            self._require_sim().run()
            drained = self._require_client().roots[self._last_root_idx :]
            self._last_root_idx = len(self._require_client().roots)
            drained_summary = self._summarize_roots(drained)
            drained_summary["cost"] = self._window_cost()
            info["drained_summary"] = drained_summary
            if self._cfg.terminal_drain_penalty:
                reward += self._reward_from_summary(drained_summary)
```

(d) Add next to `metrics_summary()` (import `Metrics` from `metrics` at the top of the file):

```python
    def in_horizon_summary(self) -> Summary | None:
        """Summary over roots completed before the episode horizon — the
        population the reward actually saw. metrics_summary() is the full
        population including drained roots."""
        if self._client is None:
            return None
        roots = self._client.roots[: self._horizon_root_count]
        return Metrics(
            roots=roots, attempts_total=sum(len(r.attempts) for r in roots)
        ).summary()
```

(e) In `rl/experiments/compare_policies.py` `_episode`, return the horizon summary too: change the return to `return total_reward, env.metrics_summary(), env.in_horizon_summary(), cost_mean` and in `_evaluate` collect both:

```python
    successes_horizon = []
    p95s_horizon = []
    ...
        total_reward, summary, horizon, cost_mean = _episode(env, policy, seed, trace_rows=trace_rows)
        ...
        if horizon:
            successes_horizon.append(horizon.succeeded / horizon.total if horizon.total else 0.0)
            p95s_horizon.append(horizon.p95)
    ...
    return {
        "reward_mean": avg(rewards),
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "success_rate": avg(successes),
        "p95_ms": avg(p95s),
        "success_rate_in_horizon": avg(successes_horizon),
        "p95_ms_in_horizon": avg(p95s_horizon),
        "retries_per_root": avg(retries),
        "cost_mean": avg(costs),
    }
```

(f) In `rl/experiments/constrained_retry.py` `_eval_arm`, mirror the same: after `metrics = env.metrics_summary()` add

```python
        horizon = env.in_horizon_summary()
        if horizon:
            successes_horizon.append(horizon.succeeded / horizon.total if horizon.total else 0.0)
            p95s_horizon.append(horizon.p95)
```

(initialize `successes_horizon: list[float] = []` / `p95s_horizon: list[float] = []` with the other accumulators) and add to the result dict after `"p95_ms"`:

```python
        "success_rate_in_horizon": float(np.mean(successes_horizon)) if successes_horizon else 0.0,
        "p95_ms_in_horizon": float(np.mean(p95s_horizon)) if p95s_horizon else 0.0,
```

- [ ] **Step 4: Run tests**

Run: `uv run --no-sync pytest tests/test_rl_env.py tests/test_rl_constrained.py tests/test_rl_policies.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rl/env.py rl/experiments/compare_policies.py rl/experiments/constrained_retry.py tests/test_rl_env.py
git commit -m "RL env: split in-horizon vs full populations; optional terminal drain penalty"
```

### Task 5: De-synchronize multichain arrival streams (finding #10a)

**Files:**
- Modify: `experiments/multichain_retry_storm.py:145-148`
- Test: `tests/test_workload.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_workload.py`)

```python
class TestWorkloadSeeding(unittest.TestCase):
    def _arrivals(self, rng_seed):
        sim = Simulator(seed=1)
        times = []
        wl = Workload(base_rps=180.0, duration_s=1, load_spikes=[], rng_seed=rng_seed)
        wl.drive(sim, lambda s: times.append(s.timestep))
        sim.run()
        return times

    def test_distinct_seeds_give_distinct_streams(self):
        self.assertNotEqual(self._arrivals(0), self._arrivals(1))

    def test_default_seed_is_zero_and_deterministic(self):
        # Documented: rng_seed=None falls back to 0 — two unseeded workloads
        # emit identical streams, which is why concurrent workloads must pass
        # explicit distinct seeds (see experiments/multichain_retry_storm.py).
        self.assertEqual(self._arrivals(None), self._arrivals(0))
```

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_workload.py -k Seeding -q`. Expected: PASS already (this locks the mechanism; the experiment change is next). If `Simulator` is not imported in the test file, add `from simulator import Simulator`.

- [ ] **Step 3: Fix the experiment.** In `experiments/multichain_retry_storm.py` replace lines 145-148 with:

```python
    # Distinct explicit seeds: Workload defaults rng_seed=None -> seed 0, so
    # two unseeded workloads would emit nanosecond-identical arrival streams
    # into the shared bottleneck. The shared spike is registered once.
    wl_a = Workload(base_rps=180.0, duration_s=duration_s, load_spikes=[spike], rng_seed=0)
    wl_x = Workload(base_rps=180.0, duration_s=duration_s, load_spikes=[spike], rng_seed=1)
    wl_a.register_fault_events(fault_tracker)
```

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_workload.py -q` (PASS) and a 5 s smoke of the experiment is deferred to Task 23's integration pass (the module runs 100+ s).

- [ ] **Step 5: Commit**

```bash
git add experiments/multichain_retry_storm.py tests/test_workload.py
git commit -m "multichain experiment: distinct workload seeds, register shared spike once"
```

### Task 6: Fix plot.sh pairing and output dirs (P3-d)

**Files:**
- Modify: `plot.sh` (full rewrite)

- [ ] **Step 1: Rewrite `plot.sh`** with the correct fault-events pairing (`chain_*` CSVs come from `experiments.chain_fanout`, which writes `chain_fault_events.json`; `fault_events.json` belongs to `experiments.default`) and `mkdir -p`:

```bash
#!/bin/bash
set -e

# CSVs below are produced by `python -m experiments.chain_fanout`, whose fault
# overlay is chain_fault_events.json (fault_events.json is experiments.default's).
for pair in chain:chain svc-A:svA svc-B:svB svc-C:svC svc-E:svE; do
    csv="${pair%%:*}_output.csv"
    out="chain_results/${pair##*:}"
    mkdir -p "$out"
    python -m plotting.failures_over_time "$csv" --fault-events chain_fault_events.json --output "$out/failures_over_time.png"
    python -m plotting.latencies_over_time "$csv" --fault-events chain_fault_events.json --output "$out/latencies_over_time.png"
    python -m plotting.queue_size_over_time "$csv" --fault-events chain_fault_events.json --output "$out/queue_size_over_time.png"
    python -m plotting.qps_over_time "$csv" --fault-events chain_fault_events.json --output "$out/qps_over_time.png"
done
```

(Bare `python` is the documented project convention — keep it.)

- [ ] **Step 2: Syntax check** — `bash -n plot.sh`. Expected: no output. Full execution happens in Task 23's integration pass (needs `chain_fanout` outputs).

- [ ] **Step 3: Commit**

```bash
git add plot.sh
git commit -m "plot.sh: pair chain CSVs with chain_fault_events.json, create output dirs"
```

**Wave 1 gate:** `make check` → all green. Then proceed.

---

## Wave 2 — RL harness correctness

### Task 7: Surrogate honors the global deadline (finding #7)

**Files:**
- Modify: `rl/policies/model_based.py:90-120` (`_simulate_action` inner loop)
- Test: `tests/test_rl_policies.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_rl_policies.py`)

```python
import numpy as np

from rl.policies.model_based import _simulate_action
from rl.spec import ActionValues, RewardWeights


class _ConstantModel:
    def __init__(self, value_ms: float):
        self._value = value_ms

    def sample(self, rng, n):
        return np.full(n, self._value)


class TestSurrogateGlobalDeadline(unittest.TestCase):
    def _score(self, sample_ms, attempt_timeout_ms, global_timeout_ms):
        weights = RewardWeights(
            success=1.0, deadline_drop=1.0, queue_drop=0.0, server_drop=0.0,
            circuit_open_drop=0.0, p95_latency_ms=0.0, retries=0.0,
        )
        action = ActionValues(
            attempt_timeout_ms=attempt_timeout_ms, max_attempts=1,
            backoff_ms=0, global_timeout_ms=global_timeout_ms,
        )
        return _simulate_action(
            _ConstantModel(sample_ms), action, weights,
            np.random.default_rng(0), n_roots=10, queue_delay_ms=0.0,
            queue_delay_mode="deterministic",
        )

    def test_sample_beyond_global_budget_is_a_deadline_drop(self):
        # 80 ms sample, 100 ms attempt timeout, 50 ms global: the real caller
        # caps the attempt at min(remaining global, attempt timeout) and
        # requires strictly-before completion -> deadline drop, reward -1.
        self.assertAlmostEqual(self._score(80.0, 100, 50), -1.0)

    def test_sample_within_both_budgets_succeeds(self):
        self.assertAlmostEqual(self._score(30.0, 100, 200), 1.0)
```

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_rl_policies.py -k SurrogateGlobal -q`. Expected: FAIL (first case returns 1.0).

- [ ] **Step 3: Implement.** Replace the per-root attempt loop body in `_simulate_action` (lines 90-120) with:

```python
    for _ in range(n_roots):
        elapsed = 0.0
        success = False
        attempts = 0
        for attempt in range(max_attempts):
            remaining = None if global_timeout is None else global_timeout - elapsed
            if remaining is not None and remaining <= 0:
                break
            # Mirror the real caller: per-attempt deadline is
            # min(attempt timeout, remaining global budget), and success needs
            # strictly-before completion (half-open deadline, caller.py).
            effective_timeout = timeout if remaining is None else min(timeout, remaining)
            attempts += 1
            sample = float(model.sample(rng, 1)[0])
            q_delay = 0.0
            if queue_delay_ms > 0:
                if queue_delay_mode == "exponential":
                    q_delay = float(rng.exponential(queue_delay_ms))
                else:
                    q_delay = queue_delay_ms
            total_sample = sample + q_delay
            attempt_latency = min(total_sample, effective_timeout)
            elapsed += attempt_latency

            if total_sample < effective_timeout:
                success = True
                latencies.append(elapsed)
                break

            if attempt < max_attempts - 1:
                elapsed += backoff

        retries += max(0, attempts - 1)
        if success:
            successes += 1
```

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_rl_policies.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl/policies/model_based.py tests/test_rl_policies.py
git commit -m "Surrogate simulation: cap attempts by remaining global budget, strict deadline"
```

### Task 8: Latency samples come from the actuated client (finding #4a)

**Files:**
- Modify: `rl/env.py` (`__init__`, `reset`, `step`)
- Test: `tests/test_rl_env.py` (append)

**Interfaces:**
- Produces: `info["attempt_latencies_ms"]` / `attempt_censored` / `root_latencies_ms` now describe `EnvComponents.cost_client` when the builder sets one (chain/multichain), else the top-level client. Consumed by ModelBased/Bayesian via `LatencyBuffer.ingest_info` (unchanged).

- [ ] **Step 1: Write the failing test** (append to `tests/test_rl_env.py`)

```python
from core import ns_to_ms, DropReason
from rl.env import build_chain_env


class TestSampleSource(unittest.TestCase):
    def test_chain_samples_come_from_the_actuated_edge(self):
        env = MicroservicesEnv(
            EnvConfig(
                build=lambda seed: build_chain_env(seed, duration_s=4),
                action_spec=ActionSpec(
                    attempt_timeout_ms=[80], max_attempts=[2], backoff_ms=[20]
                ),
                step_duration_s=1.0,
                episode_duration_s=4.0,
            )
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step((0, 0, 0))
        cost_client = env._require_components().cost_client
        edge_latencies = sorted(
            ns_to_ms(a.interval.end - a.interval.begin)
            for root in cost_client.roots[: env._last_sample_root_idx]
            for a in root.attempts
            if not (
                a.drop_reason in (DropReason.QUEUE_FULL, DropReason.CIRCUIT_OPEN)
                and a.interval.end == a.interval.begin
            )
        )
        self.assertGreater(len(edge_latencies), 0)
        self.assertEqual(sorted(info["attempt_latencies_ms"]), edge_latencies)
```

(The instant-rejection filter in the comprehension anticipates Task 10; before Task 10 lands no such attempts occur in this scenario, so the test is stable either way.)

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_rl_env.py -k SampleSource -q`. Expected: FAIL (`AttributeError: _last_sample_root_idx` or multiset mismatch — top-level samples differ from edge samples).

- [ ] **Step 3: Implement in `rl/env.py`.**

(a) `__init__`: add `self._last_sample_root_idx = 0`; `reset()`: reset it to 0.

(b) In `step()`, replace `info.update(self._collect_latency_samples(new_roots))` with:

```python
        if self._cfg.emit_latency_samples:
            # Latency-based policies must see the client the knobs actuate:
            # in chain scenarios that is the B->C edge (cost_client), not the
            # top-level client whose end-to-end latencies the knobs only
            # influence indirectly. Reward/observation stay top-level.
            sample_client = self._require_components().cost_client
            if sample_client is not None:
                sample_roots = sample_client.roots[self._last_sample_root_idx :]
                self._last_sample_root_idx = len(sample_client.roots)
            else:
                sample_roots = new_roots
            info.update(self._collect_latency_samples(sample_roots))
```

(and delete the old `if self._cfg.emit_latency_samples: info.update(...)` line).

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_rl_env.py tests/test_rl_policies.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl/env.py tests/test_rl_env.py
git commit -m "RL env: emit latency samples from the knob-actuated client in chain scenarios"
```

### Task 9: Queue-exclusive service-time stream for the queue-aware policy (finding #4b)

**Files:**
- Modify: `core.py:49-58` (Request), `caller.py:162-196` (`_on_attempt_done`), `rl/env.py` (`_collect_latency_samples`), `rl/policies/model_based.py`
- Test: `tests/test_service_deadline.py` (append caller-level test), `tests/test_rl_policies.py` (append moments test)

**Interfaces:**
- Produces: `Request.service_time_ns: TimeDuration | None` (queue-exclusive; None when service never began; for chain nodes it is node-processing time — local stage + downstream wait — still excluding the local queue). `info["attempt_service_ms"]: list[float]`. `ModelBasedPolicy._service_moments() -> tuple[float, float, int]` (mean, mean-square, count of buffered service samples).

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_service_deadline.py`:

```python
class TestServiceTimeRecording(unittest.TestCase):
    def test_attempt_interval_includes_queue_wait_but_service_time_does_not(self):
        sim = Simulator(seed=1)
        cfg = ServiceConfig(
            name="svc", latency_median=ms_to_ns(20), latency_lognorm_sigma=1e-6, workers=1
        )
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=1, delay=0),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=None
                ),
            ),
            service=ServiceRuntime(cfg=cfg).bind(),
        )
        client.start_request(sim)
        client.start_request(sim)  # queues behind the first (workers=1)
        sim.run()
        attempts = sorted(
            (a for root in client.roots for a in root.attempts),
            key=lambda a: a.interval.end,
        )
        first, second = attempts
        tol = ms_to_ns(2)
        self.assertLess(abs(first.service_time_ns - ms_to_ns(20)), tol)
        self.assertLess(abs(second.service_time_ns - ms_to_ns(20)), tol)
        # Second attempt's measured interval contains ~20 ms of queue wait.
        self.assertGreater(second.interval.end - second.interval.begin, ms_to_ns(35))
```

Append to `tests/test_rl_policies.py`:

```python
from rl.policies.model_based import ModelBasedPolicy
from rl.policies.action_space import ActionGrid
from rl.spec import ActionSpec


class TestServiceMoments(unittest.TestCase):
    def test_queue_variant_derives_moments_from_service_stream(self):
        grid = ActionGrid(
            ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0])
        )
        policy = ModelBasedPolicy(grid=grid, include_queue_delay=True)
        policy.act(
            obs=np.zeros(1),
            info={
                "attempt_latencies_ms": [40.0] * 50,  # totals: include queue wait
                "attempt_censored": [False] * 50,
                "attempt_service_ms": [20.0] * 50,  # queue-exclusive
                "summary": {"completed": 50.0, "retries_per_root": 0.0},
            },
        )
        mean_s, mean_s2, count = policy._service_moments()
        self.assertEqual(count, 50)
        self.assertAlmostEqual(mean_s, 20.0)
        self.assertAlmostEqual(mean_s2, 400.0)
```

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_service_deadline.py -k ServiceTimeRecording tests/test_rl_policies.py -k ServiceMoments -q`. Expected: FAIL (`AttributeError: service_time_ns` / `_service_moments`). Add any missing imports the test file needs (`Simulator`, `ServiceConfig`, `ServiceRuntime`, `ClientConfig`, `ClientRuntime`, `FixedBackoffRetryPolicy`, `StaticTimeoutController`, `ms_to_ns`).

- [ ] **Step 3: Implement.**

(a) `core.py` — add to `Request` after `queue_size_at_end`:

```python
    # Queue-exclusive processing duration reported by the service (for chain
    # nodes: local stage + downstream wait, still excluding the local queue).
    # None when service never began (queue-full / deadline-at-dequeue drops).
    service_time_ns: TimeDuration | None = None
```

(b) `caller.py` `_on_attempt_done` — the `svc_time` parameter is currently unused; record it. After `ctx.req.queue_size_at_end = queue_size` add:

```python
        # svc_time == 0 marks paths where service never began (queue-full,
        # deadline-at-dequeue); real service times are floored at 100 us.
        ctx.req.service_time_ns = svc_time if svc_time > 0 else None
```

and delete the `# unused for now` comment on the parameter.

(c) `rl/env.py` `_collect_latency_samples` — add a service list:

```python
        attempt_latencies = []
        attempt_censored = []
        attempt_service = []
        root_latencies = []
        ...
            for attempt in root.attempts:
                attempt_latencies.append(ns_to_ms(attempt.interval.end - attempt.interval.begin))
                attempt_censored.append(attempt.drop_reason == DropReason.DEADLINE)
                if attempt.service_time_ns is not None:
                    attempt_service.append(ns_to_ms(attempt.service_time_ns))
```

apply the same `max_latency_samples` truncation (`attempt_service = attempt_service[: self._cfg.max_latency_samples]`), and add `"attempt_service_ms": attempt_service` to the returned dict (also add `"attempt_service_ms": []` to the reset-time info block).

(d) `rl/policies/model_based.py` — buffer the service stream and use it for queue moments. In `__post_init__` add `self._service_samples: list[float] = []`; in `reset()` clear it. At the top of `act()` after `ingest_info` add:

```python
        service = [float(v) for v in info.get("attempt_service_ms", [])]
        if service:
            self._service_samples.extend(service)
            self._service_samples = self._service_samples[-self.buffer_size :]
```

Add the helper:

```python
    def _service_moments(self) -> tuple[float, float, int]:
        """Mean / mean-square / count of buffered queue-exclusive service times."""
        service = np.asarray(self._service_samples, dtype=np.float64)
        if service.size == 0:
            return 0.0, 0.0, 0
        return (
            float(service.mean()),
            float(np.dot(service, service) / service.size),
            int(service.size),
        )
```

In `_estimate_queue_delay`, replace the moments block with:

```python
        # Erlang-C needs SERVICE-time moments; attempt intervals include queue
        # wait, so deriving moments from the fitted total-latency model would
        # double-count queueing. Fall back to the model only while the service
        # stream is still empty (legacy info dicts / cold start).
        mean_s, mean_s2, count = self._service_moments()
        if count < 10:
            moments_fn = getattr(model, "moments", None)
            if callable(moments_fn):
                mean_s, mean_s2 = (float(m) for m in moments_fn())
            else:
                sample_count = max(10, int(self.queue_moment_samples))
                samples = model.sample(self._rng, sample_count)
                mean_s = float(np.mean(samples))
                mean_s2 = float(np.dot(samples, samples) / samples.size)
```

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_service_deadline.py tests/test_rl_policies.py tests/test_rl_env.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core.py caller.py rl/env.py rl/policies/model_based.py tests/test_service_deadline.py tests/test_rl_policies.py
git commit -m "Record queue-exclusive service time per attempt; Erlang-C moments use it"
```

### Task 10: Exclude instant rejections from latency streams; guard the Bayesian log (finding #4c)

**Files:**
- Modify: `rl/env.py` (`_collect_latency_samples`), `rl/policies/bayesian.py:70-77`, `rl/policies/model_based.py` (`act` positive filter)
- Test: `tests/test_rl_env.py`, `tests/test_rl_policies.py` (append)

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_rl_env.py`:

```python
from core import Request, RootRequest, TimeInterval


class TestInstantRejectionSamples(unittest.TestCase):
    def test_zero_latency_rejections_are_not_emitted(self):
        env = MicroservicesEnv(
            EnvConfig(build=_build_slow_failing_env, action_spec=SLOW_SPEC)
        )
        env.reset(seed=1)
        rejected = RootRequest()
        rejected.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=1000, end=1000),
                drop_reason=DropReason.QUEUE_FULL,
            )
        )
        served = RootRequest()
        served.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=0, end=ms_to_ns(20)),
                success=True,
            )
        )
        samples = env._collect_latency_samples([rejected, served])
        self.assertEqual(len(samples["attempt_latencies_ms"]), 1)
        self.assertAlmostEqual(samples["attempt_latencies_ms"][0], 20.0)
```

Append to `tests/test_rl_policies.py`:

```python
class TestZeroSampleRobustness(unittest.TestCase):
    def _info(self, values):
        return {
            "attempt_latencies_ms": values,
            "attempt_censored": [False] * len(values),
            "summary": {"completed": float(len(values)), "retries_per_root": 0.0},
        }

    def test_bayesian_ignores_nonpositive_samples(self):
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0]))
        policy = BayesianPolicy(grid=grid)
        action = policy.act(np.zeros(1), self._info([0.0, 0.0, 20.0, 25.0]))
        self.assertIn(action, grid.indices)

    def test_model_based_survives_all_zero_buffer(self):
        grid = ActionGrid(ActionSpec(attempt_timeout_ms=[50], max_attempts=[1], backoff_ms=[0]))
        policy = ModelBasedPolicy(grid=grid)
        action = policy.act(np.zeros(1), self._info([0.0, 0.0, 0.0]))
        self.assertEqual(action, grid.indices[0])
```

(`BayesianPolicy` import: `from rl.policies.bayesian import BayesianPolicy`.)

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_rl_env.py -k InstantRejection tests/test_rl_policies.py -k ZeroSample -q`. Expected: FAIL (2 emitted samples; Bayesian produces NaN posterior / ModelBased raises `ValueError: No positive samples provided`).

- [ ] **Step 3: Implement.**

(a) `rl/env.py` `_collect_latency_samples`, at the top of the per-attempt loop:

```python
            for attempt in root.attempts:
                if (
                    attempt.drop_reason in (DropReason.QUEUE_FULL, DropReason.CIRCUIT_OPEN)
                    and attempt.interval.end == attempt.interval.begin
                ):
                    # Instant rejections carry no latency information; policies
                    # see them through the drop-rate features instead. A zero
                    # would poison lognormal fits (log 0) as an uncensored value.
                    continue
```

(b) `rl/policies/bayesian.py` `act()`: after `data = self._buffer.data()` add:

```python
        data = data[data > 0]  # defense in depth: log(0) would collapse the posterior
```

(c) `rl/policies/model_based.py` `act()`: change `uncensored = values[~censored]` handling to:

```python
        uncensored = values[~censored]
        uncensored = uncensored[uncensored > 0]
        if uncensored.size == 0:
            return self.grid.indices[0]
```

and guard the censored-MLE branch the same way (it already filters positives internally via `fit_censored`; keep as is).

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_rl_env.py tests/test_rl_policies.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl/env.py rl/policies/bayesian.py rl/policies/model_based.py tests/test_rl_env.py tests/test_rl_policies.py
git commit -m "Exclude instant rejections from latency streams; filter nonpositive samples"
```

### Task 11: Float workload durations end-to-end (finding #6)

**Files:**
- Modify: `workload.py:14`, `rl/env.py` (builder `int(duration_s)` sites at lines 506, 518, 552, 593, 716 and `__init__` validation), `rl/train_sb3.py:37`, `rl/experiments/compare_policies.py:70,483`, `rl/experiments/constrained_retry.py:324`
- Test: `tests/test_workload.py`, `tests/test_rl_env.py` (append)

**Interfaces:**
- Produces: `Workload.duration_s: float`; builder signatures accept `duration_s: float | None`; `MicroservicesEnv.__init__` raises `ValueError` on `step_duration_s <= 0` or `episode_duration_s <= 0`.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_workload.py`:

```python
class TestFractionalDuration(unittest.TestCase):
    def test_no_arrivals_start_after_fractional_horizon(self):
        sim = Simulator(seed=1)
        times = []
        wl = Workload(base_rps=200.0, duration_s=1.6, load_spikes=[], rng_seed=3)
        wl.drive(sim, lambda s: times.append(s.timestep))
        sim.run()
        self.assertGreater(len(times), 0)
        self.assertLess(max(times), s_to_ns(1.6))

    def test_sub_second_duration_still_generates_traffic(self):
        sim = Simulator(seed=1)
        times = []
        wl = Workload(base_rps=200.0, duration_s=0.4, load_spikes=[], rng_seed=3)
        wl.drive(sim, lambda s: times.append(s.timestep))
        sim.run()
        self.assertGreater(len(times), 0)
```

Append to `tests/test_rl_env.py`:

```python
class TestDurationValidation(unittest.TestCase):
    def test_zero_step_duration_rejected(self):
        with self.assertRaises(ValueError):
            MicroservicesEnv(
                EnvConfig(build=_build_slow_failing_env, action_spec=SLOW_SPEC, step_duration_s=0.0)
            )

    def test_negative_episode_duration_rejected(self):
        with self.assertRaises(ValueError):
            MicroservicesEnv(
                EnvConfig(
                    build=_build_slow_failing_env, action_spec=SLOW_SPEC, episode_duration_s=-1.0
                )
            )
```

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_workload.py -k Fractional tests/test_rl_env.py -k DurationValidation -q`. Expected: FAIL (0.4 → `duration_s` typed int truncates; no ValueError raised).

- [ ] **Step 3: Implement.**

(a) `workload.py`: `duration_s: int` → `duration_s: float` (the body already compares in ns via `s_to_ns(self.duration_s)`).

(b) `rl/env.py` `MicroservicesEnv.__init__`, first lines after the gym import check:

```python
        if config.step_duration_s <= 0:
            raise ValueError("step_duration_s must be > 0")
        if config.episode_duration_s is not None and config.episode_duration_s <= 0:
            raise ValueError("episode_duration_s must be > 0")
```

(c) `rl/env.py` builders: replace every `duration = 30 if duration_s is None else int(duration_s)` (lines 506, 518, 552, 593, 716) with `duration = 30.0 if duration_s is None else float(duration_s)`, and the builder signatures' `duration_s: int | None` annotations with `duration_s: float | None`. Also update `_build_slow_failing_env`-style int casts if `ty` flags any (`Workload(duration_s=int(duration))` in Task 4's test builder can drop its `int(...)`).

(d) Entry points — delete the roundings:
- `rl/train_sb3.py:37`: `duration_s = int(round(cfg.episode_duration_s)) if cfg.episode_duration_s else None` → `duration_s = cfg.episode_duration_s if cfg.episode_duration_s else None`
- `rl/experiments/compare_policies.py:70` and `:483`: same one-line change.
- `rl/experiments/constrained_retry.py:324`: `duration = int(round(args.episode_duration_s))` → `duration = float(args.episode_duration_s)`

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/ -q` and `uv run --no-sync ty check`. Expected: PASS / clean (this task touches typed signatures).

- [ ] **Step 5: Commit**

```bash
git add workload.py rl/env.py rl/train_sb3.py rl/experiments/compare_policies.py rl/experiments/constrained_retry.py tests/test_workload.py tests/test_rl_env.py
git commit -m "Support fractional episode durations: float workload horizon, env validation"
```

### Task 12: ActionSpec sidecar on PPO save/load (finding #12)

**Files:**
- Create: `rl/model_io.py`
- Modify: `rl/train_sb3.py` (save + load), `rl/experiments/compare_policies.py:469-480`, `rl/experiments/constrained_retry.py` (model-save sites — grep `save_model`)
- Test: `tests/test_rl_policies.py` (append; no SB3 needed — pure JSON)

**Interfaces:**
- Produces: `rl/model_io.py` with `sidecar_path(model_path) -> str`, `save_action_spec(model_path, spec, step_duration_s, fault_scenario=None) -> None`, `check_action_spec(model_path, spec) -> None` (raises `ValueError` on mismatch, `warnings.warn` when the sidecar is absent).

- [ ] **Step 1: Write the failing test** (append to `tests/test_rl_policies.py`)

```python
import tempfile
import warnings
from pathlib import Path

from rl.model_io import check_action_spec, save_action_spec, sidecar_path


class TestActionSpecSidecar(unittest.TestCase):
    def setUp(self):
        self.spec = ActionSpec(
            attempt_timeout_ms=[10, 30], max_attempts=[1, 2], backoff_ms=[0, 20],
            global_timeout_ms=[500, None],
        )

    def test_roundtrip_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            save_action_spec(path, self.spec, step_duration_s=1.0, fault_scenario="chain")
            check_action_spec(path, self.spec)  # no raise

    def test_reordered_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            save_action_spec(path, self.spec, step_duration_s=1.0)
            mutated = ActionSpec(
                attempt_timeout_ms=[30, 10], max_attempts=[1, 2], backoff_ms=[0, 20],
                global_timeout_ms=[500, None],
            )
            with self.assertRaises(ValueError):
                check_action_spec(path, mutated)

    def test_missing_sidecar_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.zip")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                check_action_spec(path, self.spec)
            self.assertTrue(any("sidecar" in str(w.message) for w in caught))
```

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_rl_policies.py -k Sidecar -q`. Expected: FAIL with `ModuleNotFoundError: rl.model_io`.

- [ ] **Step 3: Create `rl/model_io.py`:**

```python
"""Persist the action semantics next to saved SB3 models.

SB3 zips store only space *shapes* (MultiDiscrete nvec), so grids like
[10, 30] vs [30, 10] are indistinguishable after a bare load and every action
silently reinterprets. The sidecar records the ordered ActionSpec value lists
(the meaning of each index) so loads can verify them.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

from rl.spec import ActionSpec

SIDECAR_SUFFIX = ".spec.json"


def sidecar_path(model_path: str) -> str:
    return str(model_path) + SIDECAR_SUFFIX


def save_action_spec(
    model_path: str,
    spec: ActionSpec,
    step_duration_s: float,
    fault_scenario: str | None = None,
) -> None:
    payload = {
        "action_spec": dataclasses.asdict(spec),
        "step_duration_s": step_duration_s,
        "fault_scenario": fault_scenario,
    }
    Path(sidecar_path(model_path)).write_text(json.dumps(payload, indent=2))


def check_action_spec(model_path: str, spec: ActionSpec) -> None:
    """Raise ValueError when the stored spec disagrees; warn when absent."""
    path = Path(sidecar_path(model_path))
    if not path.exists():
        warnings.warn(
            f"no ActionSpec sidecar next to {model_path}; cannot verify that the "
            "eval-time action grids match the grids the model was trained with",
            stacklevel=2,
        )
        return
    stored = json.loads(path.read_text())["action_spec"]
    current = dataclasses.asdict(spec)
    if stored != current:
        raise ValueError(
            f"ActionSpec mismatch for {model_path}: trained with {stored}, "
            f"evaluating with {current}"
        )
```

Wire-up:
- `rl/train_sb3.py`: extract the spec construction from `_make_env` into `def _action_spec(cfg: TrainConfig) -> ActionSpec` and use it in both `_make_env` and `train()`. In `train()`: before `PPO.load(...)` add `check_action_spec(cfg.load_path, _action_spec(cfg))`; after `model.save(cfg.save_path)` add `save_action_spec(cfg.save_path, _action_spec(cfg), cfg.step_duration_s, cfg.fault_scenario)`.
- `rl/experiments/compare_policies.py`: before `PPO.load(args.ppo_load_path)` add `check_action_spec(args.ppo_load_path, action_spec)`; after `ppo_policy.model.save(args.ppo_save_path)` add `save_action_spec(args.ppo_save_path, action_spec, cfg.step_duration_s, cfg.fault_scenario)`.
- `rl/experiments/constrained_retry.py`: `grep -n "save" rl/experiments/constrained_retry.py` — after each `*.model.save(<path>)` site (the `--save-model-prefix` feature) add `save_action_spec(<path>, action_spec, step_duration_s=1.0, fault_scenario=args.scenario)`.

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_rl_policies.py -q` (PASS) and `uv run --no-sync ruff check rl/`.

- [ ] **Step 5: Commit**

```bash
git add rl/model_io.py rl/train_sb3.py rl/experiments/compare_policies.py rl/experiments/constrained_retry.py tests/test_rl_policies.py
git commit -m "Save ActionSpec sidecar with PPO models; verify on load"
```

### Task 13: nearest_indices maps None to the None entry (P3-a)

**Files:**
- Modify: `rl/policies/action_space.py:75-83`
- Test: `tests/test_rl_policies.py` (append)

- [ ] **Step 1: Write the failing test**

```python
class TestNearestIndicesGlobalTimeout(unittest.TestCase):
    def _grid(self, global_list):
        return ActionGrid(
            ActionSpec(
                attempt_timeout_ms=[10, 30], max_attempts=[1], backoff_ms=[0],
                global_timeout_ms=global_list,
            )
        )

    def test_none_request_maps_to_the_none_entry(self):
        grid = self._grid([500, 1000, None])
        self.assertEqual(grid.nearest_indices(10, 1, 0, global_timeout_ms=None), (0, 0, 0, 2))

    def test_numeric_request_ignores_none_entries(self):
        grid = self._grid([None, 500, 1000])
        self.assertEqual(grid.nearest_indices(10, 1, 0, global_timeout_ms=600), (0, 0, 0, 1))

    def test_none_request_without_none_entry_raises(self):
        grid = self._grid([500, 1000])
        with self.assertRaises(ValueError):
            grid.nearest_indices(10, 1, 0, global_timeout_ms=None)
```

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_rl_policies.py -k NearestIndices -q`. Expected: FAIL (`(0, 0, 0, 0)` returned for the None request).

- [ ] **Step 3: Implement** — replace lines 78-83 of `rl/policies/action_space.py`:

```python
        if global_timeout_ms is None:
            if None not in self._spec.global_timeout_ms:
                raise ValueError(
                    "requested global_timeout_ms=None but the ActionSpec has no None entry"
                )
            g = self._spec.global_timeout_ms.index(None)
        else:
            numeric = [
                (idx, v) for idx, v in enumerate(self._spec.global_timeout_ms) if v is not None
            ]
            if not numeric:
                raise ValueError(
                    f"requested global_timeout_ms={global_timeout_ms} but the "
                    "ActionSpec contains only None"
                )
            g = min(numeric, key=lambda iv: abs(iv[1] - global_timeout_ms))[0]
        return (i, j, k, g)
```

(Behavior note for the commit message: a `fixed` baseline that asks for no global timeout against a grid without a `None` entry now fails fast instead of silently picking index 0.)

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_rl_policies.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl/policies/action_space.py tests/test_rl_policies.py
git commit -m "nearest_indices: map None global timeout to the None entry, fail fast otherwise"
```

### Task 14: Trace plots in real seconds (P3-b)

**Files:**
- Modify: `rl/env.py` (public `step_duration_s` property), `rl/experiments/compare_policies.py:112-133` (trace row), `plotting/rl_traces_over_time.py`
- Test: `tests/test_rl_policies.py` (append — tests the axis helper, no matplotlib rendering)

**Interfaces:**
- Produces: `MicroservicesEnv.step_duration_s` property; trace CSVs gain a `step_duration_s` column; `plotting/rl_traces_over_time.py` exposes `_step_seconds(df) -> float` used to scale the x axis.

- [ ] **Step 1: Write the failing test** (put in `tests/test_rl_policies.py`; set `os.environ.setdefault("MPLBACKEND", "Agg")` at the very top of the file, before any plotting import)

```python
import pandas as pd

from plotting.rl_traces_over_time import _step_seconds


class TestTraceTimeAxis(unittest.TestCase):
    def test_uses_step_duration_column(self):
        df = pd.DataFrame({"step": [0, 1, 2], "step_duration_s": [2.0, 2.0, 2.0]})
        self.assertEqual(_step_seconds(df), 2.0)

    def test_legacy_csv_defaults_to_one_second(self):
        df = pd.DataFrame({"step": [0, 1, 2]})
        self.assertEqual(_step_seconds(df), 1.0)
```

- [ ] **Step 2: Run** — expected FAIL: `ImportError: cannot import name '_step_seconds'`.

- [ ] **Step 3: Implement.**

(a) `rl/env.py` — add next to `metrics_summary`:

```python
    @property
    def step_duration_s(self) -> float:
        return self._cfg.step_duration_s
```

(b) `rl/experiments/compare_policies.py` — in the trace row dict add `"step_duration_s": env.step_duration_s,` right after `"step": step_idx,`.

(c) `plotting/rl_traces_over_time.py` — add the helper and use it:

```python
def _step_seconds(df: pd.DataFrame) -> float:
    if "step_duration_s" in df.columns:
        return float(df["step_duration_s"].iloc[0])
    print("warning: trace CSV has no step_duration_s column (legacy); assuming 1.0 s/step")
    return 1.0
```

In `main()`, after filtering by seed: `step_s = _step_seconds(df)`, then plot `sub["step"] * step_s` on both axes (`ax_top.step(sub["step"] * step_s, ...)`, `ax_bottom.plot(sub["step"] * step_s, ...)`). The x-label already says "Time (s)" and the fault overlay is seconds-based, so both now align for any step duration.

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_rl_policies.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rl/env.py rl/experiments/compare_policies.py plotting/rl_traces_over_time.py tests/test_rl_policies.py
git commit -m "Trace CSV/plot: record step duration, plot knob traces in real seconds"
```

**Wave 2 gate:** `make check` → all green.

---

## Wave 3 — core simulator

### Task 15: Exact piecewise-constant NHPP workload (finding #2 — largest behavior change)

**Files:**
- Modify: `workload.py` (replace `_sample_interarrival` + `drive`)
- Test: `tests/test_workload.py` (append), update any existing assertions that encode the t=0 first arrival or the 0.5 s zero-rate polling (identify via pytest failures; align them with the new semantics)
- Docs (same commit): CLAUDE.md Workload sentence; `docs/services-and-workload.md` arrival-process section

- [ ] **Step 1: Write the failing tests** (append to `tests/test_workload.py`)

```python
class TestExactNHPP(unittest.TestCase):
    def _arrival_times(self, seed, base_rps, spikes, duration_s=1):
        sim = Simulator(seed=1)
        times = []
        wl = Workload(
            base_rps=base_rps, duration_s=duration_s, load_spikes=spikes, rng_seed=seed
        )
        wl.drive(sim, lambda s: times.append(s.timestep))
        sim.run()
        return times

    def test_first_arrival_is_never_at_time_zero(self):
        for seed in range(50):
            times = self._arrival_times(seed, base_rps=100.0, spikes=[])
            self.assertGreater(times[0], 0)

    def test_short_spike_is_not_skipped(self):
        # Review scenario: 1 rps base, x100 on [0.1, 0.2) s. Correct NHPP mean
        # over [0,1): 0.9*1 + 0.1*100 = 10.9 arrivals, ~10 inside the spike.
        spike = LoadSpike(
            duration=TimeInterval(begin=s_to_ns(0.1), end=s_to_ns(0.2)), rps_multiplier=100.0
        )
        counts, in_spike = [], []
        for seed in range(300):
            times = self._arrival_times(seed, base_rps=1.0, spikes=[spike])
            counts.append(len(times))
            in_spike.append(sum(1 for t in times if s_to_ns(0.1) <= t < s_to_ns(0.2)))
        mean = sum(counts) / len(counts)
        self.assertGreater(mean, 9.5)
        self.assertLess(mean, 12.5)
        # P(zero spike arrivals) = exp(-10) under the correct process.
        self.assertGreaterEqual(sum(1 for n in in_spike if n > 0) / len(in_spike), 0.99)

    def test_constant_rate_total_count_is_poisson_scale(self):
        times = self._arrival_times(seed=5, base_rps=100.0, spikes=[], duration_s=10)
        self.assertGreater(len(times), 850)
        self.assertLess(len(times), 1150)

    def test_zero_rate_workload_emits_nothing(self):
        times = self._arrival_times(seed=5, base_rps=0.0, spikes=[])
        self.assertEqual(times, [])
```

(Imports needed at top of file: `from faults import LoadSpike`, `from core import TimeInterval, s_to_ns`.)

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_workload.py -k ExactNHPP -q`. Expected: FAIL (first arrival at 0; spike skipped in ~90% of seeds).

- [ ] **Step 3: Implement** — in `workload.py`, replace `_sample_interarrival` and `drive` with:

```python
    def _next_boundary(self, t: TimePoint, end_at: TimePoint) -> TimePoint:
        """Next spike start/end after t (the rate is constant in between)."""
        nxt = end_at
        for spike in self.load_spikes:
            for edge in (spike.duration.begin, spike.duration.end):
                if t < edge < nxt:
                    nxt = edge
        return nxt

    def _next_arrival_after(self, t: TimePoint, end_at: TimePoint) -> TimePoint | None:
        """Exact next-arrival draw for the piecewise-constant rate.

        One Exp(1) budget is consumed across rate segments (spike boundaries):
        a segment of rate r and length dt supplies r*dt of exposure, and
        memorylessness makes segment-wise consumption exact — equivalent to
        re-drawing at every boundary, without needing a max-rate bound the way
        thinning would. Zero-rate segments supply no exposure and are skipped.
        Returns None when no arrival lands before end_at.
        """
        e = self._rng.expovariate(1.0)
        while t < end_at:
            rate = self.rps_at(t)
            boundary = self._next_boundary(t, end_at)
            if rate <= 0.0:
                t = boundary
                continue
            exposure = rate * ns_to_s(boundary - t)
            if e <= exposure:
                return t + s_to_ns(e / rate)
            e -= exposure
            t = boundary
        return None

    def drive(self, sim: Simulator, on_arrival: Callable[[Simulator], None]):
        end_at = s_to_ns(self.duration_s)

        def schedule_next(from_t: TimePoint):
            nxt = self._next_arrival_after(from_t, end_at)
            if nxt is not None:
                # Keep the historical 1 ns minimum interarrival clamp.
                sim.schedule(max(nxt, from_t + 1), arrive)

        def arrive():
            if sim.timestep >= end_at:
                return
            on_arrival(sim)
            schedule_next(sim.timestep)

        schedule_next(0)
```

Add `ns_to_s` to the `core` import. Delete `_sample_interarrival`. The 0.5 s zero-rate polling loop is gone: zero-rate segments are skipped exactly (an all-zero workload schedules no events). Update the comment on line 39 accordingly.

- [ ] **Step 4: Run the full suite** — `uv run --no-sync pytest -q`. Any failure encoding the old semantics (t=0 arrival, 0.5 s polling tick counts, exact seeded arrival sequences) is expected: update those assertions to the new exact-NHPP semantics rather than weakening the new tests. Then update the docs in the same commit:
  - CLAUDE.md: change "**Workload** (`workload.py`) is a Poisson arrival process (`expovariate`) modulated by `LoadSpike.rps_multiplier`. RPS=0 means no arrivals but the tick keeps polling at 0.5s intervals." to "**Workload** (`workload.py`) is an exact nonhomogeneous Poisson process for the piecewise-constant rate (base RPS × active `LoadSpike.rps_multiplier`s): one Exp(1) budget is consumed across rate segments, so short spikes are never skipped and the first arrival is exponentially distributed (never deterministically at t=0). Zero-rate segments are skipped exactly; an all-zero workload schedules no events."
  - `docs/services-and-workload.md`: find the "Poisson arrivals" paragraph and describe the same segment-consumption algorithm in 2-3 sentences.

- [ ] **Step 5: Commit**

```bash
git add workload.py tests/test_workload.py CLAUDE.md docs/services-and-workload.md
git commit -m "Workload: exact piecewise-constant NHPP sampling (no t=0 arrival, no spike skipping)"
```

**Ledger note:** every seeded arrival stream changes from this commit on. Committed `*_output.csv` samples at the repo root become stale (documented as disposable); pinned paper CSVs live on paper-branch and are not regenerated.

### Task 16: Fail fast on backwards time (finding #13)

**Files:**
- Modify: `simulator.py:30-45`, `retry_policies.py` (`FixedBackoffRetryPolicy`, `ExponentialBackoffRetryPolicy`, `ExponentialBackoffWithJitterRetryPolicy` — add `__post_init__` validation)
- Test: `tests/test_retry_policies.py`, new `tests/test_simulator_validation.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_simulator_validation.py`:

```python
import unittest

from core import s_to_ns
from simulator import Simulator


class TestSimulatorTimeValidation(unittest.TestCase):
    def test_scheduling_in_the_past_raises(self):
        sim = Simulator(seed=1)
        sim.schedule(s_to_ns(5), lambda: sim.schedule(s_to_ns(1), lambda: None))
        with self.assertRaises(ValueError):
            sim.run()

    def test_run_until_the_past_raises(self):
        sim = Simulator(seed=1)
        sim.run(until=s_to_ns(10))
        with self.assertRaises(ValueError):
            sim.run(until=s_to_ns(3))

    def test_scheduling_at_now_is_allowed(self):
        sim = Simulator(seed=1)
        fired = []
        sim.schedule(s_to_ns(5), lambda: sim.schedule(sim.timestep, lambda: fired.append(1)))
        sim.run()
        self.assertEqual(fired, [1])


if __name__ == "__main__":
    unittest.main()
```

Append to `tests/test_retry_policies.py`:

```python
class TestBackoffValidation(unittest.TestCase):
    def test_negative_fixed_backoff_rejected(self):
        with self.assertRaises(ValueError):
            FixedBackoffRetryPolicy(max_attempts=2, delay=-1)

    def test_nonpositive_attempts_rejected(self):
        with self.assertRaises(ValueError):
            FixedBackoffRetryPolicy(max_attempts=0, delay=0)

    def test_negative_exponential_delays_rejected(self):
        with self.assertRaises(ValueError):
            ExponentialBackoffRetryPolicy(max_attempts=2, initial_delay=-5, max_delay=10)
```

- [ ] **Step 2: Run** — expected FAIL (no exceptions raised; the past event silently rewinds the clock).

- [ ] **Step 3: Implement.**

(a) `simulator.py`:

```python
    def schedule(self, at: TimePoint, fn: Callable[[], None]):
        at = int(at)
        if at < self.timestep:
            raise ValueError(
                f"cannot schedule an event at {at} ns: simulator time is already "
                f"{self.timestep} ns (negative delay or misconfigured deadline?)"
            )
        self._seq += 1
        ev = Event(at=at, seq=self._seq, action=fn)
        heapq.heappush(self._eventheap, ev)
        return ev

    def run(self, until: TimePoint | None = None):
        if until is not None and int(until) < self.timestep:
            raise ValueError(
                f"cannot run until {int(until)} ns: simulator time is already {self.timestep} ns"
            )
        ...  # rest unchanged
```

(b) `retry_policies.py` — add validation (`BurstyRateLimiterPolicy.__post_init__` is the in-repo precedent):

```python
@dataclass
class FixedBackoffRetryPolicy(RetryPolicy):
    max_attempts: int  # total attempts including the initial try
    delay: TimeDuration

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.delay < 0:
            raise ValueError("delay must be >= 0 ns")
```

and equivalent `__post_init__` on `ExponentialBackoffRetryPolicy` / `ExponentialBackoffWithJitterRetryPolicy` (`max_attempts >= 1`, `initial_delay >= 0`, `max_delay >= 0`).

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_simulator_validation.py tests/test_retry_policies.py -q`, then the full `uv run --no-sync pytest -q` (env `step()` calls `run(until=next_time)` with `next_time >= now` — must stay green).

- [ ] **Step 5: Commit**

```bash
git add simulator.py retry_policies.py tests/test_simulator_validation.py tests/test_retry_policies.py
git commit -m "Fail fast on past events, backwards run(until), negative backoff"
```

### Task 17: Fan-out launches every edge (finding #8)

**Files:**
- Modify: `chain.py:130-161` (`on_downstream_done` + launch loop)
- Test: `tests/test_chain_scenarios.py` (append)
- Docs (same commit): CLAUDE.md fan-out sentence, `docs/request-handling.md` launch paragraph

- [ ] **Step 1: Write the failing test** (append to `tests/test_chain_scenarios.py`)

```python
class TestFanoutLaunchesAllEdges(unittest.TestCase):
    def _build(self, reject_first: bool):
        sim = Simulator(seed=1)

        def leaf(name, queue_capacity=None):
            return ServiceNodeRuntime(
                service=ServiceRuntime(
                    cfg=ServiceConfig(
                        name=name,
                        latency_median=ms_to_ns(10),
                        latency_lognorm_sigma=1e-6,
                        workers=1,
                        queue_capacity=queue_capacity,
                    )
                ).bind()
            )

        reject = leaf("svc-reject", queue_capacity=0)
        ok = leaf("svc-ok")
        # Occupy svc-reject's single worker so a fan-out submit hits the full
        # (capacity-0) queue and fails synchronously.
        reject.service.submit_attempt(sim, lambda *a: None)

        parent = ServiceNodeRuntime(
            service=ServiceRuntime(
                cfg=ServiceConfig(
                    name="svc-parent",
                    latency_median=ms_to_ns(5),
                    latency_lognorm_sigma=1e-6,
                    workers=4,
                )
            ).bind()
        )

        def edge(node):
            return DownstreamEdge(
                client=ClientRuntime(
                    cfg=ClientConfig(
                        retry=NoRetryPolicy(),
                        timeout_controller=StaticTimeoutController(
                            global_timeout=None, attempt_timeout=ms_to_ns(200)
                        ),
                    ),
                    service=node,
                ),
                target=node,
            )

        edges = [edge(reject), edge(ok)] if reject_first else [edge(ok), edge(reject)]
        for e in edges:
            parent.add_downstream(e)
        results = []
        parent.submit_attempt(
            sim, lambda success, t, drop, q: results.append((success, drop))
        )
        sim.run()
        ok_edge = edges[1] if reject_first else edges[0]
        return results, len(ok_edge.client.roots)

    def test_second_edge_still_receives_load_when_first_rejects_synchronously(self):
        results, ok_calls = self._build(reject_first=True)
        self.assertEqual(ok_calls, 1)  # was 0: the loop broke before launching it
        self.assertEqual(results[0][0], False)  # AND semantics: parent still fails
        self.assertEqual(results[0][1], DropReason.QUEUE_FULL)

    def test_edge_order_does_not_change_the_load_pattern(self):
        _, ok_calls = self._build(reject_first=False)
        self.assertEqual(ok_calls, 1)
```

(Reuse the file's existing imports; add `NoRetryPolicy` / `DropReason` if missing.)

- [ ] **Step 2: Run** — `uv run --no-sync pytest tests/test_chain_scenarios.py -k LaunchesAllEdges -q`. Expected: first case FAILS with `ok_calls == 0`.

- [ ] **Step 3: Implement** — in `chain.py`, replace `on_downstream_done` and the launch loop (lines 132-161) with:

```python
                launching = False
                sync_failure_root = None

                def fail_with(_root):
                    nonlocal done
                    done = True
                    cancel_deadline()
                    drop = downstream_drop_reason(_root)
                    if drop == DropReason.NONE:
                        drop = DropReason.SERVER_FAILURE
                    finalize(False, sim2.timestep - begin_at, drop)

                def on_downstream_done(_root):
                    nonlocal remaining, sync_failure_root
                    if done:
                        return
                    succeeded = any(a.success for a in _root.attempts)
                    if not succeeded:
                        if launching:
                            # A synchronous rejection (queue-full / circuit-open)
                            # during the launch loop: remember it, but launch the
                            # remaining edges first so downstream load does not
                            # depend on edge order.
                            if sync_failure_root is None:
                                sync_failure_root = _root
                            return
                        fail_with(_root)
                        return
                    remaining -= 1
                    maybe_finish()

                # Launch ALL downstreams concurrently; AND semantics still
                # short-circuit the parent, but only after every edge is launched.
                launching = True
                for edge in downstream_edges:
                    edge.client.start_request(
                        sim2,
                        on_done=on_downstream_done,
                        override_global_deadline=deadline,
                    )
                launching = False
                if sync_failure_root is not None and not done:
                    fail_with(sync_failure_root)

                if not done and not self.hold_worker_during_fanout:
                    release_worker()
```

(`done` is already nonlocal-visible; `fail_with` declares it. Successes are never synchronous — service times are floored at 100 µs — so only failures can arrive during the launch loop.)

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_chain_scenarios.py tests/test_chain_worker_model.py tests/test_chain_failure_reason.py -q`, then the full suite. Docs in the same commit:
  - CLAUDE.md: in the chain paragraph, change "(2) launches downstream calls *concurrently*" to "(2) launches **all** downstream calls concurrently (a synchronous rejection on an early edge no longer skips later edges' launches)".
  - `docs/request-handling.md`: same clarification in the fan-out launch paragraph.

- [ ] **Step 5: Commit**

```bash
git add chain.py tests/test_chain_scenarios.py CLAUDE.md docs/request-handling.md
git commit -m "Fan-out: launch every edge before short-circuiting on synchronous failure"
```

### Task 18: Exactly-once finalize guard (finding #5, guarded scope)

**Files:**
- Modify: `service.py:186-192` (finalize wrapper), `service.py:103-115` (submit_attempt docstring)
- Test: `tests/test_chain_worker_model.py` (append)
- Docs (same commit): CLAUDE.md conventions bullet

- [ ] **Step 1: Write the failing test** (append to `tests/test_chain_worker_model.py`)

```python
class TestFinalizeExactlyOnce(unittest.TestCase):
    def test_double_finalize_raises_and_preserves_in_flight(self):
        sim = Simulator(seed=1)
        svc = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc", latency_median=ms_to_ns(10), latency_lognorm_sigma=1e-6, workers=1
            )
        ).bind()

        def behavior(service, sim2, deadline, finalize, begin_at, release_worker):
            def finish():
                finalize(True, ms_to_ns(5), DropReason.NONE)
                finalize(True, ms_to_ns(5), DropReason.NONE)  # bug: second call

            sim2.schedule(begin_at + ms_to_ns(5), finish)

        svc.submit_attempt(sim, lambda *a: None, behavior=behavior)
        with self.assertRaises(RuntimeError):
            sim.run()
        self.assertGreaterEqual(svc.in_flight, 0)
```

- [ ] **Step 2: Run** — expected FAIL: no RuntimeError; `in_flight` ends at -1.

- [ ] **Step 3: Implement** — in `service.py` `_begin_service`, wrap finalize:

```python
            finalized = False

            def finalize(success: bool, service_time: TimeDuration, drop: DropReason):
                nonlocal finalized
                if finalized:
                    raise RuntimeError(
                        "finalize called more than once for the same attempt"
                    )
                finalized = True
                if released:
                    on_done(success, service_time, drop, len(self.queue))
                else:
                    self._complete_attempt(sim, on_done, success, service_time, drop)
```

Extend the `submit_attempt` docstring with the callback contract:

```python
        Callback contract: `on_done` runs after the worker is freed but BEFORE
        queued work resumes — a callback that synchronously submits new work can
        therefore start ahead of already-queued items (stock retry/chain paths
        never do this; they always go through the event heap). Callbacks must
        not raise: an exception aborts the run and strands queued items. Custom
        behaviors must call `finalize` exactly once (a second call raises) and
        `release_worker` at most once.
```

CLAUDE.md, "Conventions when extending", add a bullet:

```markdown
- Custom service behaviors must call `finalize` exactly once and `release_worker` at most once (both raise on repeats). Completion callbacks must not submit work synchronously if FCFS matters to your experiment — schedule through the event heap instead.
```

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_chain_worker_model.py -q`, then full suite (chain behavior path uses finalize once — must stay green).

- [ ] **Step 5: Commit**

```bash
git add service.py tests/test_chain_worker_model.py CLAUDE.md
git commit -m "Guard behavior finalize exactly-once; document the callback contract"
```

### Task 19: O(1) time-window circuit breaker (finding #14)

**Files:**
- Modify: `retry_policies.py:174-234` (`TimeBasedCircuitBreakerPolicy`)
- Test: `tests/test_retry_policies.py` (append)

- [ ] **Step 1: Write the failing-or-locking test** (append to `tests/test_retry_policies.py`)

```python
class TestTimeBasedBreakerCounters(unittest.TestCase):
    def test_incremental_counters_match_full_rescan(self):
        rng = random.Random(7)
        policy = TimeBasedCircuitBreakerPolicy(
            failure_threshold_rate=0.5,
            success_threshold_rate=0.6,
            min_requests=5,
            window_duration=s_to_ns(2),
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=3, delay=0),
            half_open_delay=s_to_ns(1),
        )
        now = 0
        for _ in range(3000):
            now += rng.randint(0, s_to_ns(0.01))
            policy.allow_request(now)
            policy.add_result(rng.random() < 0.6, now)
            self.assertEqual(
                policy._failures_in_window,
                sum(1 for _, ok in policy._window if not ok),
            )
```

- [ ] **Step 2: Run** — expected FAIL: `AttributeError: _failures_in_window`.

- [ ] **Step 3: Implement** — in `TimeBasedCircuitBreakerPolicy`:

```python
    _window: deque[tuple[TimePoint, bool]] = field(default_factory=deque)
    _failures_in_window: int = field(default=0, init=False)
```

```python
    def _evict_old(self, now: TimePoint) -> None:
        while self._window and (now - self._window[0][0]) > self.window_duration:
            _, ok = self._window.popleft()
            if not ok:
                self._failures_in_window -= 1

    def _clear_window(self) -> None:
        self._window.clear()
        self._failures_in_window = 0
```

In `add_result`, replace the append and the rescan:

```python
        if self._state != CBState.OPEN:
            self._window.append((now, success))
            if not success:
                self._failures_in_window += 1

        # Incremental counters instead of a full window rescan per result
        # (resilience4j's time-based sliding window keeps aggregates the same
        # way); the old sum() made every result O(window) -> O(n^2) overall.
        total = len(self._window)
        failures = self._failures_in_window
        rate = failures / total if total > 0 else 0.0
```

Replace the three `self._window.clear()` sites (CLOSED transition, half-open→OPEN transition, `allow_request` half-open transition) with `self._clear_window()`.

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_retry_policies.py -q` (all existing breaker tests must stay green — semantics are unchanged).

- [ ] **Step 5: Commit**

```bash
git add retry_policies.py tests/test_retry_policies.py
git commit -m "Time-based circuit breaker: incremental window counters (O(1) per result)"
```

### Task 20: Exact rational retry-budget tokens (P3-c)

**Files:**
- Modify: `retry_policies.py:266-307` (`RetryBudgetPolicy`)
- Test: `tests/test_retry_policies.py` (append)

- [ ] **Step 1: Write the failing test**

```python
class TestRetryBudgetExactArithmetic(unittest.TestCase):
    def _drained(self, ratio):
        policy = RetryBudgetPolicy(
            budget_ratio=ratio,
            max_retries=1,
            underlying_strategy=FixedBackoffRetryPolicy(max_attempts=10, delay=0),
        )
        while policy.can_retry():  # drain the initial burst
            policy.next_delay(RetryContext(attempt=1, now=0))
        return policy

    def test_one_third_ratio_earns_a_retry_after_three_successes(self):
        policy = self._drained(1 / 3)
        for _ in range(3):
            policy.add_result(True, now=0)
        self.assertTrue(policy.can_retry())  # float accumulation gave 99.99999999999999

    def test_one_seventh_ratio_earns_a_retry_after_seven_successes(self):
        policy = self._drained(1 / 7)
        for _ in range(6):
            policy.add_result(True, now=0)
        self.assertFalse(policy.can_retry())
        policy.add_result(True, now=0)
        self.assertTrue(policy.can_retry())

    def test_shipped_ratios_unchanged(self):
        for ratio, needed in ((0.05, 20), (0.1, 10), (0.2, 5)):
            policy = self._drained(ratio)
            for _ in range(needed - 1):
                policy.add_result(True, now=0)
            self.assertFalse(policy.can_retry())
            policy.add_result(True, now=0)
            self.assertTrue(policy.can_retry())
```

- [ ] **Step 2: Run** — expected FAIL on the 1/3 and 1/7 cases.

- [ ] **Step 3: Implement** — replace the token bookkeeping with exact integers (rational version of Finagle's deposit/withdraw RetryBudget):

```python
@dataclass
class RetryBudgetPolicy(RetryPolicy):
    budget_ratio: float  # retries can be at most this ratio of successes (e.g., 0.1 for 10%)
    max_retries: int  # maximum consecutive retries before throttling
    underlying_strategy: RetryPolicy

    # Integer token bucket in units of 1/denominator of the (rationalized)
    # ratio: one retry costs `denominator`, each success awards `numerator`.
    # Exact arithmetic — no float drift (cf. Finagle's integer RetryBudget);
    # limit_denominator(10_000) recovers intended ratios like 1/3 from floats.
    _retry_cost: int = field(init=False)
    _success_award: int = field(init=False)
    _max_tokens: int = field(init=False)
    _tokens: int = field(init=False)

    def __post_init__(self):
        if not 0 < self.budget_ratio <= 1:
            raise ValueError("budget_ratio must be in (0, 1]")
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        ratio = Fraction(self.budget_ratio).limit_denominator(10_000)
        self._retry_cost = ratio.denominator
        self._success_award = ratio.numerator
        self._max_tokens = self._retry_cost * self.max_retries
        self._tokens = self._max_tokens  # start with full budget

    def add_result(self, success: bool, now: TimePoint | None = None) -> None:
        if success:
            self._tokens = min(self._max_tokens, self._tokens + self._success_award)

    def can_retry(self) -> bool:
        return self._tokens >= self._retry_cost

    def get_token_balance(self) -> float:
        # Backward-compatible scale: 100 tokens == one retry, as before.
        return self._tokens / self._retry_cost * 100.0

    def next_delay(self, context: RetryContext) -> DelayDecision:
        if self.can_retry():
            can_retry, delay = self.underlying_strategy.next_delay(context)
            if can_retry:
                self._tokens -= self._retry_cost
                return DelayDecision(True, delay)
            return DelayDecision(False, 0)
        return DelayDecision(False, 0)
```

Add `from fractions import Fraction` to the imports.

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_retry_policies.py tests/test_rl_env.py -q` and the full suite (chain_budget/multichain_budget builders construct this policy; any test asserting `get_token_balance` values must still pass on the 100-per-retry scale).

- [ ] **Step 5: Commit**

```bash
git add retry_policies.py tests/test_retry_policies.py
git commit -m "Retry budget: exact rational token arithmetic (Finagle-style integers)"
```

**Wave 3 gate:** `make check` → all green.

---

## Wave 4 — metrics, plotting, CLI polish

### Task 21: Granularity validation, timepoint precision, honest QPS (finding #11)

**Files:**
- Modify: `metrics.py:125-127` and `:230-262`, `plotting/qps_over_time.py`
- Test: `tests/test_metrics_summary.py` (append), `tests/test_rl_policies.py` is NOT involved — qps helper test goes in a new `tests/test_plotting_io.py`
- Docs (same commit): `docs/observability.md` granularity note

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_metrics_summary.py` (reuse its existing Metrics-building helpers; if none exist, build roots inline):

```python
class TestExportGranularity(unittest.TestCase):
    def _metrics(self):
        root = RootRequest()
        root.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=ms_to_ns(0.2), end=ms_to_ns(0.9)),
                success=True,
            )
        )
        root2 = RootRequest()
        root2.add_attempt(
            Request(
                service="svc",
                interval=TimeInterval(begin=ms_to_ns(1.4), end=ms_to_ns(2.1)),
                success=True,
            )
        )
        return Metrics(roots=[root, root2], attempts_total=2)

    def test_sub_nanosecond_granularity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._metrics().export_csv(os.path.join(tmp, "out.csv"), granularity_s=1e-10)

    def test_sub_millisecond_buckets_keep_distinct_timepoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            self._metrics().export_csv(path, granularity_s=0.0005)
            import pandas as pd

            df = pd.read_csv(path)
            self.assertEqual(df["timepoint"].nunique(), len(df))
```

Create `tests/test_plotting_io.py`:

```python
import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from plotting.qps_over_time import _bucket_width_s


class TestQpsBucketWidth(unittest.TestCase):
    def test_width_inferred_from_timepoints(self):
        df = pd.DataFrame({"timepoint": [0.0, 0.5, 1.0, 1.5]})
        self.assertAlmostEqual(_bucket_width_s(df), 0.5)

    def test_single_row_defaults_to_one_second(self):
        df = pd.DataFrame({"timepoint": [0.0]})
        self.assertEqual(_bucket_width_s(df), 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** — expected FAIL: `ZeroDivisionError` (not ValueError) for 1e-10; `nunique() < len(df)` at 0.0005; `ImportError: _bucket_width_s`.

- [ ] **Step 3: Implement.**

(a) `metrics.py` validation (after the existing `granularity_s <= 0` check):

```python
        bucket_ns = s_to_ns(granularity_s)
        if bucket_ns <= 0:
            raise ValueError(
                f"granularity_s={granularity_s!r} rounds to {bucket_ns} ns; "
                "the minimum representable bucket is 1 ns (1e-9 s)"
            )
```

(b) `metrics.py` timepoint precision — before `out.to_csv(...)`:

```python
        # %.3f is fine for millisecond latency columns but collapses sub-ms
        # bucket timestamps; format the timepoint with granularity-aware
        # precision (as strings, so float_format leaves it alone).
        decimals = max(3, math.ceil(-math.log10(bucket_ns / S_TO_NS))) if bucket_ns < S_TO_NS else 3
        out["timepoint"] = [f"{v:.{decimals}f}" for v in out["timepoint"]]
        out.to_csv(path, index=False, float_format="%.3f", na_rep="nan", lineterminator="\r\n")
```

(`import math` at the top if absent; `S_TO_NS` is already imported.)

(c) `plotting/qps_over_time.py` — add the helper and divide:

```python
def _bucket_width_s(df) -> float:
    """Bucket width inferred from consecutive timepoints (default 1.0 s)."""
    if len(df) > 1:
        width = float(df["timepoint"].diff().median())
        if width > 0:
            return width
    return 1.0
```

and in `main()`:

```python
    width = _bucket_width_s(df)
    plt.plot(df["timepoint"], df["root_requests"] / width, label="Root Requests", linewidth=2)
    plt.plot(df["timepoint"], df["retries"] / width, label="Retries", linewidth=2)
    plt.plot(df["timepoint"], df["failure_root"] / width, label="Failures", linewidth=2)
```

(The "QPS" y-label is now honest for any granularity; at the default 1 s nothing changes.)

(d) `docs/observability.md`: note the granularity floor (≥ 1e-9 s), the timepoint precision rule, and that qps_over_time divides by bucket width.

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_metrics_summary.py tests/test_plotting_io.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add metrics.py plotting/qps_over_time.py tests/test_metrics_summary.py tests/test_plotting_io.py docs/observability.md
git commit -m "Metrics export: validate granularity, precise timepoints; qps divides by bucket width"
```

### Task 22: Header-only CSVs exit cleanly (P3-e)

**Files:**
- Modify: `plotting/utils/io.py` (add `require_rows`), `plotting/failures_over_time.py`, `plotting/latencies_over_time.py`, `plotting/queue_size_over_time.py`, `plotting/qps_over_time.py`, `plotting/rl_traces_over_time.py` (guard after load)
- Test: `tests/test_plotting_io.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_plotting_io.py`)

```python
import tempfile

from metrics import Metrics
from plotting.utils.io import load_metrics, require_rows
import plotting.failures_over_time as failures_over_time


class TestHeaderOnlyCsv(unittest.TestCase):
    def _header_only_csv(self, tmp):
        path = os.path.join(tmp, "empty.csv")
        Metrics(roots=[], attempts_total=0).export_csv(path)  # writes header only
        return path

    def test_require_rows_raises_system_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._header_only_csv(tmp)
            df = load_metrics(path)
            with self.assertRaises(SystemExit):
                require_rows(df, path)

    def test_failures_over_time_exits_cleanly_not_typeerror(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._header_only_csv(tmp)
            with self.assertRaises(SystemExit):
                failures_over_time.main(path, os.path.join(tmp, "out.png"))
```

- [ ] **Step 2: Run** — expected FAIL: `ImportError: require_rows`, and the second test dies with the numpy/matplotlib `TypeError` instead of SystemExit.

- [ ] **Step 3: Implement.**

(a) `plotting/utils/io.py`:

```python
def require_rows(df: pd.DataFrame, csv_file: str) -> None:
    """Exit with a clear message on header-only CSVs (a legitimate
    Metrics.export_csv product when an experiment completed zero attempts)."""
    if df.empty:
        raise SystemExit(f"no data rows in {csv_file} (the experiment completed zero attempts)")
```

(b) In each plotting entry point's `main()`, right after `df = load_metrics(csv_file)` (or `pd.read_csv` for rl_traces): add `require_rows(df, csv_file)` (import it from `plotting.utils.io`).

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_plotting_io.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plotting/utils/io.py plotting/failures_over_time.py plotting/latencies_over_time.py plotting/queue_size_over_time.py plotting/qps_over_time.py plotting/rl_traces_over_time.py tests/test_plotting_io.py
git commit -m "Plotting: clean exit on header-only metrics CSVs"
```

### Task 23: Hotel CLI/JSON validation (P3-f) + integration pass

**Files:**
- Modify: `experiments/hotel_matched.py` (parsers, `_load_calibration`, `_service_params`, argparse types)
- Test: `tests/test_hotel_matched_experiment.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_hotel_matched_experiment.py`)

```python
from experiments.hotel_matched import (
    _fraction,
    _load_calibration,
    _nonneg_float,
    _parse_load_spike,
    _parse_profile_fault,
    _positive_float,
)


class TestCliValidation(unittest.TestCase):
    def test_fraction_bounds(self):
        self.assertEqual(_fraction("0.25"), 0.25)
        for bad in ("1.5", "-0.5", "nan"):
            with self.assertRaises(argparse.ArgumentTypeError):
                _fraction(bad)

    def test_rates_and_durations(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _nonneg_float("-100")
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_float("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_float("inf")

    def test_fault_and_spike_windows(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_profile_fault("10,20,-500,0.5")  # negative added latency
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_load_spike("10,20,-3")  # negative multiplier
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_load_spike("-5,20,2")  # negative start

    def test_calibration_json_rejects_non_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            path.write_text('{"services": {"profile": {"median_ms": NaN}}}')
            with self.assertRaises(ValueError):
                _load_calibration(path)
```

(Add `import argparse`, `import tempfile`, `from pathlib import Path` to the test file if missing.)

- [ ] **Step 2: Run** — expected FAIL: `ImportError` on the new type helpers.

- [ ] **Step 3: Implement in `experiments/hotel_matched.py`.**

```python
def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _nonneg_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _fraction(value: str) -> float:
    parsed = _finite_float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed
```

Wire into `build_parser()`: `--base-rps` → `type=_nonneg_float`, `--duration-s` → `type=_positive_float`, `--recommendation-fraction` → `type=_fraction`.

Extend `_parse_profile_fault` (after the existing checks): reject non-finite parts and `add_latency_ms < 0`:

```python
    if not all(math.isfinite(part) for part in (start_s, end_s, add_latency_ms, p_fail)):
        raise argparse.ArgumentTypeError("profile fault values must be finite")
    if start_s < 0:
        raise argparse.ArgumentTypeError("profile fault start must be >= 0")
    if add_latency_ms < 0:
        raise argparse.ArgumentTypeError("profile fault add_latency_ms must be >= 0")
```

Extend `_parse_load_spike` similarly (`finite`, `start_s >= 0`, `multiplier >= 0`).

`_load_calibration` — reject non-finite JSON constants:

```python
def _load_calibration(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    def _reject_constant(name: str):
        raise ValueError(f"calibration JSON contains non-finite value {name}")

    return json.loads(Path(path).read_text(), parse_constant=_reject_constant)
```

`_service_params` — treat an invalid sigma like an invalid median (drop the observed block):

```python
    median_ms = observed.get("median_ms")
    sigma = observed.get("sigma", defaults["sigma"])
    if (
        not isinstance(median_ms, int | float)
        or not math.isfinite(float(median_ms))
        or not isinstance(sigma, int | float)
        or not math.isfinite(float(sigma))
        or float(sigma) <= 0
    ):
        observed = {}
```

- [ ] **Step 4: Run tests, then the wave-4 integration pass:**

```bash
uv run --no-sync pytest tests/test_hotel_matched_experiment.py -q
uv run --no-sync python -m experiments.default
uv run --no-sync python -m experiments.chain_fanout
uv run --no-sync python -m experiments.multichain_retry_storm
uv run --no-sync bash plot.sh
uv run --no-sync python -m rl.experiments.compare_policies --policies fixed --eval-seeds 321 --episode-duration-s 5
make check
```
Expected: all succeed; plot.sh writes PNGs under `chain_results/` (generated artifacts are gitignored — leave them). `make check` fully green.

- [ ] **Step 5: Commit**

```bash
git add experiments/hotel_matched.py tests/test_hotel_matched_experiment.py
git commit -m "hotel_matched: validate CLI bounds and reject non-finite calibration JSON"
```

### Task 24: Docs sync + spec ledger check

**Files:**
- Modify: `CLAUDE.md`, `docs/rl.md`, `docs/experiments.md`, `docs/superpowers/specs/2026-07-22-review-fix-campaign-design.md` (ledger note for Task 4's column decision)

- [ ] **Step 1: Update CLAUDE.md RL paragraph** — add, where `step()`/drain is described: samples for latency-based policies come from the knob-actuated client (`cost_client`) in chain scenarios; results CSVs report both `success_rate`/`p95_ms` (full population incl. drained) and `*_in_horizon` variants; `terminal_drain_penalty` optionally folds drained outcomes into the final reward; saved PPO models carry a `.spec.json` ActionSpec sidecar verified on load.

- [ ] **Step 2: Update `docs/rl.md`** — same four points, in its accounting/persistence sections; and `docs/experiments.md` — multichain workload seeding note (distinct explicit seeds) and hotel calibration semantics (absent key → default fault; explicit empty → no faults).

- [ ] **Step 3: Update the spec's Task-4 deviation** — in `docs/superpowers/specs/2026-07-22-review-fix-campaign-design.md` §1.3, replace "existing ambiguous columns are renamed, not silently redefined" with "existing columns keep their (full-population) meaning for figure-script back-compat with pinned CSVs; explicit `*_in_horizon` columns are added alongside".

- [ ] **Step 4: Verify docs claims** — `uv run --no-sync ruff format --check . && make check`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/rl.md docs/experiments.md docs/superpowers/specs/2026-07-22-review-fix-campaign-design.md
git commit -m "Docs: sync CLAUDE.md and docs/ with fix-campaign behavior changes"
```

---

## Wave 5 — paper-branch integration

### Task 25: Merge into paper-branch + legacy curve relabel

**Files:**
- Branch op: merge `dev` → `paper-branch`
- Modify (on paper-branch): `plotting/paper_rl_figs.py` (exists only there)

- [ ] **Step 1: Merge**

```bash
git switch paper-branch
git merge dev -m "Merge review fix campaign from dev"
uv run --no-sync pytest -q
```
Expected: clean merge (paper-branch was strictly ahead of dev before the campaign; the campaign's commits are new files/lines — resolve any conflict by taking both sides). Full suite green.

- [ ] **Step 2: Read `plotting/paper_rl_figs.py`** and locate where the curve CSVs (`rl_constrained_curve_*.csv`) are loaded for plotting.

- [ ] **Step 3: Add the legacy relabel** — insert this helper and apply it to the x column of every pinned curve CSV before plotting:

```python
# Curve CSVs produced before the trainer chunking fix (commit "Fix constrained
# trainer overtraining", 2026-07-22) carry per-call learn() targets in their
# `timesteps` column, and each call trained that many ADDITIONAL steps
# (SB3 reset_num_timesteps=False semantics). The actual cumulative position of
# row i is therefore the cumulative sum of the label column: a "30k" study
# actually spans 105k env steps, the "100k" arm-A curve 1.05M. Post-fix CSVs
# record true cumulative steps and must NOT be remapped.
LEGACY_CURVE_FILES = {
    "rl_constrained_curve_multichain_100k_unconstrained.csv",
    "rl_constrained_curve_multichain_30k_b0.05.csv",
    "rl_constrained_curve_multichain_30k_b0.1.csv",
    "rl_constrained_curve_multichain_30k_b0.2.csv",
    "rl_constrained_curve_multichain_30k_s555_b0.05.csv",
    "rl_constrained_curve_multichain_30k_s555_b0.1.csv",
    "rl_constrained_curve_multichain_30k_s555_b0.2.csv",
    "rl_constrained_curve_multichain_30k_s555_unconstrained.csv",
    "rl_constrained_curve_multichain_30k_s7_b0.05.csv",
    "rl_constrained_curve_multichain_30k_s7_b0.1.csv",
    "rl_constrained_curve_multichain_30k_s7_b0.2.csv",
    "rl_constrained_curve_multichain_30k_s7_unconstrained.csv",
    "rl_constrained_curve_multichain_30k_unconstrained.csv",
}


def actual_timesteps(df: "pd.DataFrame", csv_name: str) -> "pd.Series":
    """x positions in ACTUAL cumulative env steps (see LEGACY_CURVE_FILES)."""
    if csv_name in LEGACY_CURVE_FILES:
        return df["timesteps"].cumsum()
    return df["timesteps"]
```

Apply `actual_timesteps(df, os.path.basename(path))` wherever the script currently uses `df["timesteps"]` as the x axis, and update the x-axis label to "env steps (actual)".

- [ ] **Step 4: Regenerate the paper figures and eyeball them**

```bash
uv run --no-sync python -m plotting.paper_rl_figs
```
Expected: figures written to their configured output (`~/testlatex/plots/`); the 30k-study curves now span 5k→105k on x, the 100k arm-A curve 5k→1.05M. Do NOT touch the CSVs themselves.

- [ ] **Step 5: Commit (on paper-branch)**

```bash
git add plotting/paper_rl_figs.py
git commit -m "Paper figures: relabel legacy curve x-axes to actual env steps"
git switch dev
```

---

## Completion checklist

- [ ] All 25 tasks committed; `make check` green on both `dev` and `paper-branch`.
- [ ] Behavior-change ledger in the spec matches what landed (Task 24 updated it for the column decision).
- [ ] Remind the user of the out-of-repo follow-ups: paper text step counts (30k→105k actual, 100k→1.05M), dual-update cadence wording, §4 calibration-JSON fault-scenario check.
