# Reinforcement-learning layer (`rl/`)

The simulator doubles as a control problem: pick retry/timeout settings to maximize
goodput while minimizing drops, latency, and retries. `rl/` exposes the simulator as
a Gymnasium environment, ships baseline policies, fits latency models used by some of
those policies, and provides comparison/training entry points.

**Optional dependencies** (the simulator core needs only `numpy`):

| Dependency | Needed for |
| --- | --- |
| `gymnasium` | `MicroservicesEnv` (anything that builds the env) — in the `dev` group |
| `stable-baselines3` | PPO training/policy (`rl.train_sb3`, the `ppo`/`sb3` policy) — opt-in `rl` group |
| `scipy` | latency-model distributions, censored MLE (`scipy.stats`) |
| `scikit-learn` | lognormal-mixture fitting (`GaussianMixture`) |

```bash
uv sync --group dev --group rl   # rl pulls in torch (~GBs; CUDA build on Linux x86_64)
```

SB3 recommends CPU for small MLP policies; `--device` defaults to `auto`.

## The environment (`rl/env.py`)

`MicroservicesEnv` wraps a **build function** (which constructs a fresh
`Simulator` + `Workload` + `ClientRuntime` and a knob applicator) into a Gym env.
Each `step()` advances the sim by `step_duration_s` and summarizes the roots that
completed in that window. On episode end, **every** arrival process is halted
(`Workload.stop()` on `EnvComponents.workload` and on each entry of
`EnvComponents.extra_workloads`) and in-flight work is drained *after* the final
reward, reported under `info["drained_summary"]` so the last action is not
credited with the tail — drained roots are strictly in-flight work, even when the
builder's workload duration outlives `episode_duration_s`. Accounting notes:

- **Multi-workload builders.** A builder that drives more than one `Workload`
  *must* register the extras in `EnvComponents.extra_workloads`; the env halts
  only what it can reach. `build_multichain_env` drives a second workload for the
  bystander chain and registers it — before it did, the unbounded drain kept
  admitting brand-new bystander roots until the builder's own `duration_s` (70%
  of the population on a 3 s episode against a 10 s builder duration).
- **Populations.** `metrics_summary()` covers *all* roots including drained ones;
  `in_horizon_summary()` covers only roots completed before the horizon — the
  population the reward actually saw. The experiment CSVs report both
  (`success_rate`/`p95_ms` = full population, `*_in_horizon` = reward-visible).
  The bystander mirrors this split: `bystander_metrics_summary()` /
  `info["bystander_summary"]` are the full population, while
  `bystander_in_horizon_summary()` / `info["bystander_summary_in_horizon"]`
  score the externality on the same population as J.
- **`terminal_drain_penalty`** (default off) folds drained-root outcomes into the
  final step's reward, weighted by how many average-sized step windows the
  drained population represents. That window size is measured on the
  **in-horizon** population: the post-drain root count already includes the very
  tail being priced, so dividing by it understates the weight by exactly
  `horizon / (horizon + drained)` — worst precisely when the agent defers the
  most. When nothing completed in-horizon there is no measured window size and
  the expected roots per window (below) stands in. The flag removes the bulk of
  the incentive to defer
  failures past the horizon; a small boundary residual remains (roughly one
  drained-tail window weight, ~2% of J in the slow-failure repro) because
  in-horizon windows are rate-scored while the drain is count-weighted — fully
  removing it would require count-weighting every window, i.e. a different
  objective. Leave the flag off to reproduce results trained without it.
- **Latency samples** (`info["attempt_latencies_ms"]`, `attempt_censored`,
  `attempt_service_ms`, `attempt_service_censored`) come from the knob-actuated
  client — the B->C edge (`cost_client`) in chain scenarios, the top-level
  client otherwise — so latency-based policies fit the distribution their knobs
  actually control. `attempt_service_ms` records actual worker occupancy
  (queue-excluded; deadline cuts are flagged censored). `model_based_queue`
  fits its latency model on that stream, derives Erlang-C moments from it, and
  takes its arrival rate from the sampled attempt count — fitting
  queue-inclusive totals and adding the Erlang-C wait would double-count
  queueing. Instant rejections (queue-full/circuit-open) emit no samples.

Unseeded `reset()` draws a fresh build seed from the gym RNG. This matters: SB3
vector envs only seed the *first* reset, so builders must never fall back to fixed
seeds on the env path or training collapses to one repeated episode.

### Configuration

- **`EnvConfig`** — `build`, `action_spec`, `step_duration_s` (default 1.0),
  `episode_duration_s`, `reward_weights`, `cost_budget`, `drain_on_done`,
  `initial_action`, `emit_latency_samples`, `max_latency_samples`,
  `terminal_drain_penalty`. Durations may be fractional seconds; non-positive
  values (and a negative `cost_budget`) raise at construction. `cost_budget` is
  the retry-amplification budget the constraint is stated against and changes
  what the cost term measures — see [Constraint cost](#constraint-cost).
- **`ActionSpec`** (`rl/spec.py`) — the discrete choices: `attempt_timeout_ms`,
  `max_attempts`, `backoff_ms`, and optional `global_timeout_ms` (each a list). The
  action space is a `MultiDiscrete` over the *indices* into these lists. The
  study's default grid and seed lists live next to it as
  `DEFAULT_ATTEMPT_TIMEOUTS_MS` `(10, 30, 50, 80)`, `DEFAULT_MAX_ATTEMPTS`
  `(1, 2, 3)`, `DEFAULT_BACKOFF_MS` `(0, 20, 50)`, `DEFAULT_TRAIN_SEEDS` and
  `DEFAULT_EVAL_SEEDS`; all three entry points (`rl.train_sb3`,
  `compare_policies`, `constrained_retry`) take their CLI defaults from there, so
  a saved model still loads into another entry point's grid (`check_action_spec`
  rejects any mismatch) and separately reported experiments stay seed-comparable.
- **`RewardWeights`** (`rl/spec.py`) — defaults `success=1.0`, `deadline_drop=1.0`,
  `queue_drop=1.0`, `server_drop=1.0`, `circuit_open_drop=1.0`,
  `p95_latency_ms=0.001`, `retries=0.1`.

### Observation (normalized `Box`, 16 dims + 1 with a global-timeout dimension)

In order: a has-data flag (disambiguates "no traffic" from "all failed"),
log-scaled completed count, the five outcome rates (success, deadline, queue,
server, circuit-open), avg and p95 latency scaled by the largest attempt timeout in
the spec, retries scaled by the max retry budget, the constraint cost — feature 11,
the cost client's per-window retry amplification `summary["cost"]` (retries per
edge root), scaled like retries — log-scaled queue depth, the last action echoed as
index fractions (one per action dimension), and the episode-time fraction.

Feature 11 is the *ratio*, deliberately: the agent observes the amplification it
is causing. The reward's cost term is a different quantity (a count, see
[Constraint cost](#constraint-cost)), and the budget itself is **not** in the
observation — a constrained agent cannot see which budget it is being trained
against, which is why it retreats to the zero-retry corner at tight budgets.

### Reward

```
reward = success_rate
       - deadline_drop_rate - queue_drop_rate - server_drop_rate - circuit_open_drop_rate
       - p95_latency_ms * 0.001
       - retries_per_root * 0.1            (each term scaled by its RewardWeights)
       - cost_term * RewardWeights.cost    (see "Constraint cost" below)
```

That linear form lives once, in `rl.spec.linear_reward`, and has exactly two
callers: `MicroservicesEnv._reward_from_summary` (the training signal) and
`EpisodeResult.objective` (the base objective **J**), which calls it with
`retries` and `cost` weights replaced by 0. So "J is the reward with the retry
and cost penalties zeroed" is enforced by construction rather than by two copies
staying in step — a new weighted term is in the reward *and* in J, or in
neither. `cost_term` is a parameter rather than a summary key because it is not
a plain rate: the env penalises the linearized retry-amplification constraint
(below).

`cancelled_rate` is reported in the summary but deliberately carries **no**
weight and no observation feature: cancelled work was abandoned by an upstream,
not failed by this service.

### Constraint cost

`EnvComponents.cost_client` (set by the chain builders to the B→C edge client)
defines where the constraint cost is measured: `summary["cost"]` is the edge's
retry amplification (retries per edge root) over each window — distinct from
`retries_per_root`, which measures the top-level client. `summary["cost_retries"]`
and `summary["cost_roots"]` carry that window's raw counts so consumers can pool
them across the episode instead of averaging ratios.

The observation keeps that per-window **ratio** (feature 11, scaled like
`retries_per_root`): the agent sees the amplification it is causing. The reward
does not — what it subtracts depends on whether a budget is configured.

**With `EnvConfig.cost_budget = b` (constrained mode).** The constraint the dual
ascends on is the *pooled* amplification over the episode,
`Σr / Σn ≤ b` (`r` = a window's `cost_retries`, `n` its `cost_roots`), which
linearizes to `Σr − b·Σn ≤ 0`. The reward therefore subtracts that constraint's
per-window contribution, `cost_term = (r − b·n) / R` — `R` the constant expected
roots-per-window fixed at reset — times `RewardWeights.cost`. The episode sum is
then **exactly** `(Σr − b·Σn) / R = Σn · (pooled_cost − b) / R`, which is:
independent of the order of the windows and of how the episode is cut into them;
measured on one population on both sides (`R` is a pure, action-independent
rescale — no count from a different population weights any window); zero exactly
when the pooled constraint binds; and *signed* like `pooled_cost − b`, the
quantity `dual_ascent_update` moves λ on — so at a KKT point
`λ·(Σr − b·Σn) = 0` iff `λ·(pooled − b) = 0`. The `R` rescale keeps the term
O(1): λ and the default `--lambda-lr 0.5` keep the scale they were tuned for,
where the unscaled count would need both to be ~`R`× smaller and the dual's few
chunk updates would never reach a binding multiplier.

One honest consequence: because the constraint is a *ratio*, the `− b·n` half
gives the agent a second way to satisfy it — inflate the denominator by
generating more edge roots (upstream failures create fresh ones). That is a
property of the pooled constraint itself, which this term represents faithfully,
and the hand-coded `RetryBudgetPolicy` baseline has the same shape (retries per
*request*). Read `cost_roots` next to `cost_retries` before believing a
suspiciously low pooled cost.

**Without a budget (`cost_budget=None`, the default: plain soft-penalty mode).**
There is no constraint to linearize, and `RewardWeights.cost` is simply a
discouragement of retries: the term is the window's retry *count* divided by a
constant fixed at `reset()` — the roots a window is expected to carry at the
offered arrival rate, `base_rps × step_duration_s`. That keeps the episode sum
order-independent (a *running* mean of observed roots-per-window did not: windows
of (5 retries, 5 roots) then (0, 100) summed to 1.0 while the reverse order
summed to 0.095, same pooled cost), but it is an **approximation** whose scale
drifts with realised traffic and is *not* proportional to `pooled_cost`: the
divisor counts expected **top-level** roots while the retries are the **edge**
client's, and edge roots are action-dependent (upstream retries create more of
them). It prices retries, not amplification — measured rank inversions against
`pooled_cost` across the action grid. Use it to discourage retries; use a budget
to enforce one. It falls back to the raw per-window ratio only when the workload
declares no arrival rate.

`terminal_drain_penalty` splits these two kinds of term: the drained window's
**rate** terms are count-weighted (see above), while its cost term is added
**once, unweighted**, because it is already a count — weighting it would price
the drained tail's retries by a factor that depends on the window size and break
the episode identity.

### Applying an action

Knob application is owned by the **builder**: each `build_*` function returns
`EnvComponents` with an `apply_knobs(values: ActionValues)` callback that maps
decoded knob values onto its own policy/controller objects (see
`_static_knob_applier` for the `StaticTimeoutController` + `FixedBackoffRetryPolicy`
case). The env itself is policy-agnostic — a new scenario only needs a builder that
wires its own `apply_knobs`.

### Build functions (`SCENARIO_BUILDERS`, the `--fault-scenario` choices)

| Scenario | Builder | What it is |
| --- | --- | --- |
| `none` | `build_single_service_env` | one `svc-A` (20 ms median, 16 workers, queue 200), 200 rps, no faults |
| `basic` | `build_faulty_single_service_env` | same plus a latency injection, partial failure, and 2x load spike at **fixed** episode fractions — memorizable via the time feature; prefer `randomized` for training |
| `randomized` | `build_randomized_faulty_single_service_env` | fault onset/duration/intensity drawn from the seed, so policies must react to observed degradation |
| `chain` | `build_chain_env` | A → B → C with bottleneck C degrading in a randomized window; the agent tunes the **B→C edge** retry/timeout knobs while reward comes from the **top-level** client (end-to-end goodput) — the retry-storm setting |
| `chain_budget` | `build_chain_env(retry_budget_ratio=0.1)` | same, with the edge policy wrapped in the hand-coded `RetryBudgetPolicy` (knobs still reach the inner fixed-backoff policy) |
| `multichain` | `build_multichain_env` | two chains A → B → C and X → Y → C share bottleneck C (one worker pool + queue); C degrades with randomized **pure partial-failure blips** (no latency injection), so an edge retry genuinely recovers a failed call while burning shared capacity. The agent tunes its own B→C edge; the fixed, retry-free bystander chain is exposed as `EnvComponents.bystander_client` and its success rate gauges the externality — the scenario where a retry budget actually binds |
| `multichain_budget` | `build_multichain_env(retry_budget_ratio=0.1)` | same, with the agent's edge wrapped in `RetryBudgetPolicy` |

## Policies (`rl/policies/`)

All implement the `Policy` protocol (`rl/policies/base.py`): a `name`, plus
`reset(seed)`, `act(obs, info) -> action`, and `update(obs, reward, done, info)`.
Actions are index tuples matching the env's `MultiDiscrete`; `action_space.py`
provides `ActionGrid` (enumerate/snap-to-nearest) and `coerce_action`.

| Policy (class) | Idea |
| --- | --- |
| `FixedPolicy` (`fixed.py`) | emit a constant action; also used for the **oracle** (best constant action found by grid search) |
| `FixedValuePolicy` (`fixed.py`) | specify raw ms values; snaps to the nearest grid indices |
| `AIMDPolicy` (`aimd.py`) | additive increase / **multiplicative decrease** of the attempt timeout (TCP / adaptive-concurrency shape), driven by smoothed success/deadline rates vs. targets |
| `ModelBasedPolicy` (`model_based.py`) | fit a [latency model](#latency-models-rlmodels) to buffered samples (right-**censored MLE** for timed-out attempts), then pick the action grid point with the best Monte-Carlo expected reward; optional M/G/c queueing-delay term (Erlang-C / Allen–Cunneen) |
| `BayesianPolicy` (`bayesian.py`) | Thompson sampling with a Normal-Inverse-Gamma posterior over log-latency; drops censored samples (conjugacy needs exact data — biased optimistic under heavy timeouts) |
| `SB3Policy` (`sb3.py`) | adapt a trained Stable-Baselines3 (PPO) model to the `Policy` protocol |

The model-based surrogate only models attempt latency vs. timeout: queue-full drops
and server failures are outside its vocabulary.

Both search policies share `_argmax_over_grid` (`model_based.py`): candidates are
scored against **common random numbers** inside a window (so the argmax is a
decision, not resampling noise), exact ties break toward the cheapest action
(`_action_cost_key`; `global_timeout_ms=None` counts as the *most* permissive
setting of that knob), and the stream is then **jumped** so the next window draws
fresh samples instead of re-scoring the previous window's. With nothing usable to
fit — an empty buffer, or a fully censored one with no earlier fit to hold — they
fall back to the **mid-grid** action, matching the env's warm-up action; index 0 is
the pathological corner. A fully censored window with an earlier fit holds that fit
(identifiability guard).

## Latency models (`rl/models/`)

All implement `LatencyModel` (`rl/models/base.py`): `fit`, `logpdf`, `cdf`,
`quantile`, `sample`, `params`, `log_likelihood`.

| Model | Captures |
| --- | --- |
| `LogNormalModel` (`lognormal.py`) | a single lognormal backed by `scipy.stats.lognorm`; `fit_censored` does right-censored MLE via `scipy.stats.CensoredData` (the queueing math takes its moments from the raw service buffer, not from a fitted model — see `ModelBasedPolicy._service_moments`) |
| `LogNormalMixtureModel` (`lognormal_mixture.py`) | a K-component lognormal mixture fit in log-space, for multi-modal latency |
| `EVTTailModel` (`evt_tail.py`) | a lognormal **body** plus a Pareto tail above a threshold quantile (Hill estimator, not GPD peaks-over-threshold) |

## Entry points

### `python -m rl.train_sb3`

Train (and optionally evaluate) a PPO agent. Key args: `--timesteps 5000`,
`--seed 123`, `--n-envs 1`, `--step-duration-s 1.0`, `--episode-duration-s 10.0`,
`--fault-scenario {none,basic,randomized,chain}`, `--device auto`, the action-grid
lists, `--n-steps 64`, `--batch-size 64`, `--gamma 0.95`, `--save-path`,
`--load-path`, `--eval-episodes 1`. Requires the `rl` dependency group.

```bash
python -m rl.train_sb3 --timesteps 100000 --fault-scenario randomized \
    --save-path rl_models/ppo.zip
```

### Constrained training (`rl/constrained.py`)

`train_constrained_ppo(ConstrainedTrainConfig(...))` is Lagrangian PPO: the
env reward acts as the Lagrangian, with `RewardWeights.cost` as the multiplier
λ on the edge-amplification cost, updated by projected dual ascent between
training chunks (`λ ← max(0, λ + lr·(cost − budget))`). The trainer hands the
budget to the env as well (`EnvConfig.cost_budget = cfg.budget`), so the primal
penalises `Σr − b·Σn` — the linearization of the *same* pooled constraint the
dual step reads (see [Constraint cost](#constraint-cost)). Without that pairing
the primal prices raw retries while the dual prices amplification, and dual
ascent converges to neither. The hand `retries` penalty is zeroed during
constrained training; evaluation always reports the base objective **J** (reward
with retry/cost penalties zeroed) so arms with different training weights stay
comparable. Honest caveats: the constraint is enforced **in expectation** (not
per episode/window — report violation rates); the cost term's constant-`R`
rescale keeps λ and `--lambda-lr` at their pre-pairing O(1) scale; and
`RetryBudgetPolicy.budget_ratio` limits retries against *requests* (Finagle
semantics), matching the retries-per-root the constraint cost measures.

`compare_policies` accepts `constrained_ppo` as a policy name
(`--retry-budget`, `--lambda-lr`, `--lambda-init`); the three-arm study is:

```bash
python -m rl.experiments.constrained_retry --scenario multichain --calibrate --device cpu
python -m rl.experiments.constrained_retry --scenario multichain --budgets 0.05,0.1,0.2 --device cpu
```

which trains **unconstrained PPO** (default weights), **constrained PPO** per
budget, and grid-searches fixed knobs on the budget-wrapped env
(**retry_budget**, the hand-coded baseline), evaluating all arms on the same
seeds. Outputs `rl_constrained_comparison_<scenario>.csv` and per-budget
dual-ascent curves (`rl_constrained_curve_<scenario>_b*.csv`); plot with
`python -m plotting.rl_pareto <csv>`.

`--scenario` picks the environment. On `chain`, C's fault couples a latency
injection with `p_fail`, so retries rarely beat the deadline and every PPO
arm converges to zero edge retries — the constraint is vacuous there (a
finding, not a bug). `multichain` is built so the budget binds: C fails with
pure blips (retries genuinely recover successes), and two chains share C, so
amplification steals bottleneck capacity from the bystander chain. The
selfish reward (own chain only) plus a budget on edge amplification mirrors
production retry budgets, which bound exactly this externality — Finagle's
`retryBudget` defaults to 20% of requests, gRPC retry throttling to a ~10%
token ratio. `_eval_arm` reports `bystander_success_rate` alongside J and
cost, and the Pareto plot adds an externality panel when the column is
present.

`--calibrate` grid-sweeps all fixed actions (no training) and verdicts
whether the scenario supports the study at all: the unconstrained optimum
must retry (max_attempts ≥ 2) at a cost above the smallest budget, tighter
budgets must trade J for cost beyond eval noise, and the retries must
measurably hurt the bystander. Run it after touching the builder's fault or
capacity constants; the FAIL messages name the knob to turn. Three of those
criteria are stated against a no-retry baseline, so the grid must contain
`max_attempts = 1` (the default `--max-attempts 1,2,3` does); a grid without it
is rejected up front rather than after the sweep, and a budget below the
cheapest action in the grid is reported as infeasible.

#### Findings (multichain sweep, 100k timesteps, 10 eval seeds)

Reference run of the commands above (`--timesteps 100000`, otherwise defaults;
output committed as `rl_constrained_comparison_multichain.csv`). Produced
2026-08-16 on the review-fixed code: the reward is the paired Lagrangian
(`(r − b·n)/R`, default `--lambda-lr 0.5`), and the chunk loop now trains the
timesteps it reports — the earlier reference's "30k" actually trained ~105k
via the chunk-loop bug, so 100k is the honest match for its effective compute.
(At an honest 30k the study is basin-dominated: arm B's outcome is insensitive
to λ entirely, and the b=0.2 dual only starts binding after the last chunk.)

| arm | budget | J | edge cost | violations | bystander | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| unconstrained PPO | — | **6.35** | 0.300 | — | 0.767 | 258 ms |
| constrained PPO | 0.05 | 5.95 | 0.057 | 60% | 0.827 | 154 ms |
| retry_budget | 0.05 | 5.88 | 0.027 | 0% | 0.836 | 144 ms |
| constrained PPO | 0.1 | 5.90 | 0.005 | 0% | 0.840 | 142 ms |
| retry_budget | 0.1 | 5.95 | 0.050 | 0% | 0.834 | 156 ms |
| constrained PPO | 0.2 | 6.11 | 0.138 | 20% | 0.808 | 180 ms |
| retry_budget | 0.2 | 6.07 | 0.094 | 0% | 0.818 | 178 ms |

Three results worth keeping:

1. **The trade-off is real and two-sided.** J rises with allowed amplification
   (≈5.9 at the tight budgets → 6.11 → 6.35) and the bystander chain pays for
   it symmetrically (0.83–0.84 → 0.808 → 0.767, ~7 p.p. at the unconstrained
   optimum) — the externality production retry budgets exist to bound.
   Contrast with `chain`, where every arm sits at cost 0 and the same study
   shows nothing.
2. **Dual ascent enforces the mean, not the episode.** λ climbs while eval
   cost exceeds the budget, decays under slack, and holds interior points:
   b=0.2 is satisfied in expectation with real slack used (0.138 vs 0.2,
   2/10 seed violations) and b=0.05 sits at its budget (0.057, over by small
   margins on 6/10 seeds). Per-episode guarantees need more than a Lagrangian
   — report violation rates alongside the means.
3. **The hand-coded token bucket is competitive where it matters.** It never
   violates at any budget, and constrained PPO buys only near-equal J at the
   cost of in-expectation enforcement. RL earns its keep in adaptivity, not
   in replacing the 30-line mechanism.

Calibration history (why the builder's constants look the way they do): at
p_fail U(0.3, 0.6) and retrying relay edges, upstream layers recovered blips
for free and the no-retry arm won — relay edges are now retry-free and blips
are U(0.5, 0.8); at 12 workers C saturated during blips and retries timed
out, eroding their value — C has 14. Re-run `--calibrate` before trusting
results after touching any of these.

### `python -m rl.experiments.compare_models`

Fit all three latency models to synthetic samples and report NLL + p95/p99 quantile
error. Args: `--dist {lognormal,mixture,pareto}`, `--n 5000`, `--seed 123`, and
distribution parameters.

### `python -m rl.experiments.compare_policies`

Evaluate policies across train/eval seeds (10 eval seeds by default). `--policies`
is a comma list drawn from: `fixed`, `aimd`, `model_based`, `model_based_queue`,
`bayesian`, `ppo`, `oracle_fixed`. Outputs (repo convention: CSVs at repo root,
gitignored): per-policy results (`--output-csv`), per-step action/reward traces
(`--traces-csv`), and the PPO early-stopping learning curve
(`--ppo-learning-curve-csv`). `--ppo-save-path` / `--ppo-load-path` let you train
PPO once (e.g. on `randomized`) and evaluate the same model on another scenario.
Saves write an ActionSpec sidecar (`<model>.zip.spec.json`) recording the ordered
knob-value lists; loads verify it against the eval-time grids and **fail on any
mismatch** (SB3 zips alone store only space shapes, so reordered grids like
`[10,30]` vs `[30,10]` would otherwise silently reinterpret every action). A
missing sidecar (legacy model) only warns.

```bash
python -m rl.experiments.compare_policies \
    --policies fixed,aimd,model_based,model_based_queue,bayesian,ppo,oracle_fixed \
    --fault-scenario randomized --ppo-timesteps 100000 \
    --ppo-save-path rl_models/ppo_randomized.zip
# then cross-scenario eval of the same model:
python -m rl.experiments.compare_policies --fault-scenario basic \
    --policies fixed,aimd,ppo,oracle_fixed --ppo-load-path rl_models/ppo_randomized.zip
```

## See also

- [request-handling](request-handling.md) — the `StaticTimeoutController` /
  `FixedBackoffRetryPolicy` the env tunes.
- [observability](observability.md) — the per-window statistics the reward is
  built from.
