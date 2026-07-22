# Review fix campaign — design

Date: 2026-07-22. Status: approved (user waived interactive gates; async review welcome).

## Background

An external review of the repo produced 14 numbered findings (P1/P2) plus 7 P3 items. Every
claim was independently verified in-session with code citations and live reproductions
(repro scripts in the session scratchpad; the essential ones get ported into `tests/` as
regression tests by this campaign). Verification highlights that shaped this design:

- **#1 (trainer overtraining) is real and paper-critical**: `rl/constrained.py:158-164`
  passes the *cumulative* `trained` count to `model.learn(..., reset_num_timesteps=False)`;
  SB3 ≥1.x treats `total_timesteps` in that mode as *additional* steps
  (`_setup_learn` does `total_timesteps += num_timesteps`). Nominal 30k/5k-chunks = 105k
  actual env steps; the "100k" arm-A curve = 1.05M actual. Introduced in `bbfd83d`.
- Several confirmed mechanisms have **no reachable trigger on shipped paths** (#4c crash,
  #5 FCFS, #13 rewind) — those get guards/validation, not behavioral redesign.
- Several fixes **change seeded outputs** (see Behavior-change ledger) — acceptable because
  paper data is pinned on `paper-branch` (commit `2d65d2b`) and committed sample outputs
  are documented as disposable.

## Decisions (locked with user)

1. **Paper data**: relabel, no re-run. Pinned CSVs stay untouched; the figure pipeline maps
   legacy labels → actual steps; text edits happen in the paper repo (out of scope here).
2. **Scope**: all confirmed defects (~19 items) in severity waves; #5 gets an
   exactly-once guard + documented ordering, **not** the `on_done`/`_start_next` reorder.
3. **Branch**: fixes land on `dev`, one commit per finding; `dev` is merged into
   `paper-branch` at the end; figure-relabel commits are `paper-branch`-only.
4. **Structure**: severity waves (A) with the behavior-change discipline of (C): within a
   wave, behavior-preserving fixes land before behavior-changing ones.

## Non-goals

- No re-run of the §5 study; no edits to `~/testlatex` (listed as follow-ups).
- No `on_done`/`_start_next` reorder in `service.py` (reshuffles same-timestep event and
  RNG ordering for zero stock-path benefit).
- `plot.sh` keeps bare `python` (documented project convention).
- No coverage-target work; only regression tests for the findings.

## Cross-cutting discipline

- **TDD per finding**: port the verification repro to a failing test first, then fix.
  Tests are `unittest`-style under `tests/`, run via pytest (repo convention).
- **One commit per finding**, `make check` green at every commit.
- **Docs sync in the same commit** as the behavior it describes (CLAUDE.md rule); a final
  docs pass catches anything missed.
- **Grounding**: where a real system defines the semantics, follow it and name it in the
  commit message (Finagle RetryBudget, resilience4j sliding-window breaker, exact
  piecewise-exponential NHPP sampling, SB3 documented `reset_num_timesteps` semantics).
  Flag honest gaps rather than inventing semantics.

---

## Wave 1 — results integrity

### 1.1 Trainer overtraining (#1) — `rl/constrained.py`

**Fix**: train per-chunk increments, keep the counter continuous:

```python
while trained < cfg.timesteps:
    chunk = min(cfg.chunk_timesteps, cfg.timesteps - trained)
    model.learn(total_timesteps=chunk, reset_num_timesteps=False)
    trained = model.num_timesteps  # true cumulative; curve rows now honest
```

Curve rows keep the `timesteps` column but it now records *actual* cumulative steps
(`model.num_timesteps`), so future CSVs are self-describing. Comment states the SB3
semantics explicitly.
**Test**: counting env through `train_constrained_ppo` with `timesteps=8, chunk=4,
n_steps=2, n_envs=1, batch=2` → assert `model.num_timesteps == 8` and env step count == 8
(under the bug: 12). Guard the test with the existing SB3 import skip.
**Ledger**: trainer behavior changes (that is the point); pinned curves unaffected.

### 1.2 Hotel calibration fault fallback (#9) — `experiments/hotel_matched.py`

**Fix**: distinguish "absent" from "empty": `_profile_faults_from_calibration` returns
`None` when the calibration has no `fault_scenario` key (or no calibration), and a list
(possibly empty) when the key is present — including when it lists only non-profile
methods. Selection becomes `parsed if parsed is not None else DEFAULT_PROFILE_FAULT`.
**Tests**: absent key → default fault; `"fault_scenario": []` → no faults;
non-profile-only scenario → no profile faults.
**Ledger**: calibration runs with explicit-empty scenarios lose the spurious default fault
(the fix). Default no-JSON path unchanged.

### 1.3 Reward/report population mismatch (#3) — `rl/env.py`, both experiment CSVs

**Fix** (reporting-side; J unchanged by default — windowed J is documented design and
matches Gymnasium truncation semantics):
- `MicroservicesEnv` exposes both populations explicitly: `metrics_summary()` stays the
  full population; add `in_horizon_summary()` (roots completed before the horizon).
- `compare_policies` and `constrained_retry` results CSVs report both. *(As landed:
  existing `success_rate`/`p95_ms` columns keep their historical full-population
  meaning — figure scripts must stay compatible with the pinned paper CSVs — and
  explicit `success_rate_in_horizon`/`p95_ms_in_horizon` columns are added
  alongside.)*
- Optional `terminal_drain_penalty: bool = False` on the env config: when on,
  drained-root outcomes are folded into the final step's reward, removing the
  defer-failures-past-horizon incentive for future studies. Default off (keeps trained
  behavior comparable with pinned data). *(As landed: the drained-window rate-reward is
  weighted by drained completions relative to average per-step completions — a flat
  rate would penalize 2 drained roots as hard as 100 and fail to remove the incentive.)*
**Tests**: long-vs-short-timeout repro asserts (a) both summaries present and populations
sum correctly, (b) with the flag on, the deferred-failure episode no longer scores higher
than the in-horizon-failure episode.
**Ledger**: CSV column names change; no training change unless flag enabled.

### 1.4 Multichain arrival synchronization (#10a) — `experiments/multichain_retry_storm.py`

**Fix**: explicit distinct seeds (`rng_seed=0` / `rng_seed=1`, mirroring
`hotel_matched.py`'s `seed` / `seed + 1` pattern) with a comment referencing the
default-seed pitfall. Register the shared spike's fault event once (single tracker entry).
**Test**: instantiating the experiment's two workloads yields different first-k arrival
sequences; tracker holds exactly one `load_spike` event.
**Ledger**: `multichain_retry_storm` outputs change (committed samples disposable).

### 1.5 plot.sh pairing and output dirs (P3-d)

**Fix**: pair `chain_output.csv` / `svc-*_output.csv` with `chain_fault_events.json`
(matching `experiments/chain_fanout.py` and CLAUDE.md's own examples); `mkdir -p` each
output dir at the top. Keep bare `python`.
**Test**: none (shell script); verified by running it after `chain_fanout` in the final
integration pass.

### 1.6 Formatting failure — `experiments/hotel_matched.py:262`

**Fix**: `ruff format` the file so `make check` passes. Lands first (repo gate).

---

## Wave 2 — RL harness correctness

### 2.1 Surrogate global-deadline ordering (#7) — `rl/policies/model_based.py`

**Fix**: `_simulate_action` mirrors the real caller: per-attempt effective deadline is
`min(remaining_global, attempt_timeout)`; an attempt only scores success if its sampled
latency fits the effective deadline *strictly* (`caller.py`'s half-open semantics);
global-budget exhaustion ends the simulated request as a deadline drop.
**Test**: the review's exact case (80 ms sample, 100 ms attempt timeout, 50 ms global) →
scored failure; sanity case (30 ms sample) → success.

### 2.2 Sample source for chain scenarios (#4a) — `rl/env.py`

**Fix**: `info["attempt_latencies_ms"]` (and the censoring flags) come from the
knob-actuated client: `EnvComponents.cost_client` when the builder sets it, else the
top-level client. Reward/observation stay top-level (unchanged — that is the objective).
Builders document which client feeds samples.
**Test**: chain env step → emitted samples match the B→C edge attempts, not top-level.
**Ledger**: model_based/bayesian behavior changes on chain/multichain scenarios (fix).

### 2.3 Queueing double-count (#4b) — `service.py` attempt records, `rl/env.py`,
`rl/policies/model_based.py`

**Fix**: record per-attempt *service* duration (begin-service → end; the sim never aborts
mid-service, so every begun attempt has a true service duration; queue-stage drops have
none). Env emits both streams: total attempt latency (existing, existing censoring rules)
and service duration. `model_based_queue` fits its latency model and derives Erlang-C
moments from the *service* stream; plain `model_based` keeps end-to-end totals (its
timeout calculus is about totals).
**Test**: deterministic 1-worker/2-request repro → totals 20/40 ms, service samples
20/20 ms; queue variant's moments computed from 20/20.
**Ledger**: `model_based_queue` decisions change (fix); others unchanged.

### 2.4 Zero-latency uncensored samples (#4c) — `rl/env.py`, `rl/policies/bayesian.py`

**Fix** (hardening; unreachable on shipped scenarios): instant rejections (queue-full,
circuit-open — `begin == end`, no service) are excluded from the latency sample stream
(they carry no latency information; they already show up in drop-rate features). Bayesian
update filters non-positive samples through the same `ensure_positive` guard the lognormal
fit uses.
**Test**: synthetic info with zero-latency samples → policies produce finite posteriors /
no `ValueError` even in the all-zeros corner.

### 2.5 Fractional durations (#6) — `workload.py`, builders, three entry points

**Fix**: make `Workload.duration_s` accept float seconds (converted once via `s_to_ns`;
internal comparisons already in ns), delete every `int(round(...))` at the call sites
(`rl/train_sb3.py`, `compare_policies`, `constrained_retry`, `rl/env.py` builders) so the
workload horizon equals the env horizon exactly. Validate `episode_duration_s > 0` and
`step_duration_s > 0` at `EnvConfig` construction (pairs with 3.2).
**Test**: 1.6 s episode → no arrivals start after 1.6 s; 0.4 s episode → arrivals exist
(previously zero-duration workload).
**Ledger**: only fractional-duration configs change; integer defaults byte-identical.

### 2.6 ActionSpec persistence (#12) — `rl/train_sb3.py`, `compare_policies`,
`constrained_retry`, `rl/policies/sb3.py`

**Fix**: every model save writes a sidecar `<path>.spec.json`: ordered knob value lists
(incl. `null` global-timeout entries), obs-normalization constants, `step_duration_s`,
scenario name, SB3 version. Every load: sidecar present → hard error on any mismatch with
the eval-time spec (diff printed); absent → loud warning (legacy model).
**Test**: save/load roundtrip passes; mutated grid (`[10,30]` → `[30,10]`) → raises.

### 2.7 `nearest_indices` None mapping (P3-a) — `rl/action_space.py`

**Fix**: `None` request maps to the index of the actual `None` entry (error if the spec
has no `None`); numeric requests exclude `None` entries from nearest-distance search.
**Test**: spec `[500, 1000, None]` → `None` request → index 2; numeric 50 → index 0.

### 2.8 Trace-plot time axis (P3-b) — `compare_policies`, `plotting/rl_traces_over_time.py`

**Fix**: trace CSV gains a `step_duration_s` column; the plot converts step index →
seconds and keeps the fault overlay aligned. Missing column (legacy CSV) → assume 1.0
with a warning.
**Test**: CSV with `step_duration_s=2.0` → x values doubled.

---

## Wave 3 — core simulator

### 3.1 Exact piecewise NHPP workload (#2) — `workload.py`  *(largest behavior change)*

**Fix**: replace sample-and-hold with exact sampling for piecewise-constant rates:
- First arrival is drawn, not emitted at t=0.
- Draw `e ~ Exp(1)` once per arrival; consume it across rate segments (spike starts/ends
  are the boundaries): while `e > rate_seg * seg_remaining`, subtract and advance to the
  next boundary; the arrival lands inside the segment that exhausts `e`. Memorylessness
  makes this exact — equivalent to re-drawing at each boundary, cheaper than thinning, no
  max-rate bound needed.
- Zero-rate segments consume no `e` and skip to the segment end; the existing 0.5 s poll
  remains only for open-ended zero rate (RPS=0 workloads), preserving that documented
  behavior. Keep the 1 ns minimum interarrival clamp and the separate workload RNG.
**Tests**: (a) first-arrival times over seeds are Exp-distributed (no zeros); (b) review
config (1 RPS, ×100 spike on [0.1, 0.2)) → mean arrivals in [0,1) ≈ 10.9 over seeds, spike
window non-empty in ~all seeds; (c) constant-rate stream matches Poisson counts
(chi-square-loose bounds, seeded); (d) RPS=0 still polls.
**Ledger**: **every seeded arrival stream changes.** All regenerated experiment outputs
shift. Pinned paper data unaffected (not regenerated).

### 3.2 Fail-fast time validation (#13) — `simulator.py`, `retry_policies.py`, `rl/env.py`

**Fix**: `Simulator.schedule` raises `ValueError` on `at < timestep`; `run` raises on
`until < timestep`; `FixedBackoffRetryPolicy` (and the other backoff policies) validate
non-negative delays in `__post_init__` (the codebase norm — `BurstyRateLimiterPolicy`
already does this); `EnvConfig` validates positive durations (done in 2.5).
**Tests**: each invalid construction/call raises; negative-backoff config now fails at
construction, not by rewinding the clock.
**Ledger**: invalid configs now error (previously silently corrupted time). No valid
config changes.

### 3.3 Fan-out launches all edges (#8) — `chain.py`

**Fix**: launch every downstream edge before honoring the short-circuit: iterate all edges
and call `start_request` on each; a synchronous failure recorded during the loop finalizes
the parent *after* the loop (AND semantics preserved; deadline still routed through
`min(...)`; the `on_deadline` cancel path unchanged). Late completions of already-launched
siblings after parent failure are already handled (same situation as an async first-edge
failure today).
**Test**: two-edge fan-out with a synchronously-rejecting first edge → second edge
receives exactly 1 attempt (was 0); parent outcome unchanged (failure).
**Ledger**: downstream load under saturation changes (notably `hotel_matched`,
`chain_fanout`); parent-level results unchanged by construction.

### 3.4 Finalize exactly-once guard (#5, guarded scope) — `service.py`

**Fix**: the custom-behavior `finalize` wrapper raises on a second invocation (mirroring
`release_worker`'s existing guard), so `in_flight` can never go negative and `on_done`
never double-fires. Document the callback-ordering contract (`on_done` runs before queued
work resumes; callbacks must not raise) in the `submit` docstring and CLAUDE.md
conventions. **No reorder** of `on_done`/`_start_next`.
**Test**: behavior calling `finalize` twice → second call raises; `in_flight` stays ≥ 0.

### 3.5 O(1) time-window circuit breaker (#14) — `retry_policies.py`

**Fix**: incremental counters: keep the `(t, ok)` deque plus running `total`/`failures`
maintained on append and evict — O(1) amortized per result (resilience4j's time-based
sliding window keeps per-bucket aggregates the same way). Semantics byte-identical.
**Tests**: existing CB tests stay green; property test — counter state equals full rescan
after random result sequences; micro-bench in the commit message (not a test).

### 3.6 Integer retry-budget tokens (P3-c) — `retry_policies.py`

**Fix**: Finagle-`RetryBudget`-style integer arithmetic: costs/awards in integer
milli-tokens (`cost = 100_000`, `award = round(ratio * 100_000)`), eliminating float
accumulation. `1/3` ratio × 3 awards → exactly one retry.
**Tests**: the `1/3` and `1/7` cases grant after the earned count; shipped ratios
(0.05/0.1/0.2) behave identically to before.
**Ledger**: only pathological ratios change (by design); shipped configs identical.

---

## Wave 4 — metrics, plotting, CLI polish

### 4.1 Granularity consistency (#11) — `metrics.py`, `plotting/qps_over_time.py`

**Fix**: (i) validate `granularity_s` yields ≥ 1 ns (`ValueError` otherwise, message names
the bound); (ii) per-column CSV formatting — latency columns keep `%.3f`, `timepoint` gets
granularity-aware precision so buckets never collapse; (iii) `qps_over_time` divides
counts by bucket width (inferred from median timepoint delta, fallback 1.0 with warning)
and keeps the "QPS" label honest.
**Tests**: 1e-10 granularity raises; 0.5 ms buckets stay distinct in CSV; qps values equal
counts/width at 0.5 s granularity.

### 4.2 Empty-CSV plotting (P3-e) — `plotting/utils/io.py` + entry points

**Fix**: `load_metrics` detects zero-row frames; entry points exit with a clear
"no data rows in <file>" message (exit code 1), no traceback. (Header-only CSVs are a
legitimate `Metrics.export_csv` product for zero-attempt runs.)
**Test**: header-only CSV → clean message, nonzero exit, no traceback.

### 4.3 Hotel CLI/JSON validation (P3-f) — `experiments/hotel_matched.py`

**Fix**: bounded argparse types (fractions ∈ [0,1], rates/durations/multipliers ≥ 0 and
finite); calibration JSON parsed with `parse_constant` rejecting NaN/Infinity; JSON fault
windows validated like CLI ones (same helper); `sigma` guard rewritten to an explicit
finite-and-positive check (replacing the argument-order-dependent `max(1e-6, x)`).
**Tests**: each rejected input from the verification list now errors with a clear message;
valid defaults unchanged.

### 4.4 Docs sync + behavior-change ledger commit

CLAUDE.md + `docs/{rl,services-and-workload,request-handling,observability,experiments}.md`
updated where behavior changed (workload sampling, fan-out launch, env sample sources,
population split, sidecar). The ledger below ships in this spec; the final commit checks
it against reality.

---

## Wave 5 — paper-branch integration (after dev waves)

1. Merge `dev` → `paper-branch` (expect clean; paper-branch is strictly ahead today).
2. `plotting/paper_rl_figs.py`: legacy-curve relabel — for CSVs produced by the buggy
   loop, actual cumulative steps = cumulative sum of the label column (exact for any chunk
   pattern, since each call trained `label_i` additional steps). A `legacy_curve=True`
   flag selects the mapping; new CSVs (post-fix) are already actual. Figures get honest
   x-axes (30k study → 105k; 100k arm-A → 1.05M).
3. Pinned CSVs are **not** modified or regenerated.

**Paper follow-ups (out of repo scope, for the user in `~/testlatex`)**: §5 step-count
wording (30k→105k actual, 100k→1.05M), dual-update cadence description (chunks grew
5k,10k,…), axis labels; §4 check — confirm the calibration JSON used for §4 had a
non-empty `fault_scenario` (else the default-fault bug affected those runs).

---

## Behavior-change ledger

| Change | Seeded outputs affected | Why acceptable |
|---|---|---|
| 1.1 trainer chunking | future constrained runs | the fix; pinned data untouched, relabeled downstream |
| 1.3 CSV columns | column names/meaning | ambiguity removed; both populations reported |
| 1.4 multichain seeds | `multichain_retry_storm` outputs | committed samples documented disposable |
| 2.2/2.3 sample streams | model_based/bayesian on chain scenarios | measurements now match actuated knob / queueing theory |
| 2.5 float durations | fractional-duration configs only | integer defaults byte-identical |
| 3.1 NHPP workload | **all** regenerated arrival streams | correctness; paper data pinned; docs updated |
| 3.3 launch-all fan-out | downstream load under saturation | parent outcomes unchanged; matches documented concurrency |
| 3.6 integer tokens | pathological ratios only | shipped ratios exact before and after |

Everything else is validation/guard/reporting only.

## Landing order & verification

Wave order 1→5; within a wave, the ledger-free items land first. Every commit: failing
test → fix → `make check` green. After Wave 4: run `experiments.default`,
`experiments.chain_fanout`, `experiments.multichain_retry_storm`, `./plot.sh`, and a
short `compare_policies` smoke to confirm end-to-end health; then Wave 5.
