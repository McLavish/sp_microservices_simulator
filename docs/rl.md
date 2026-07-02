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
completed in that window. On episode end, in-flight work is drained *after* the
final reward and reported under `info["drained_summary"]` so the last action is not
credited with the tail.

Unseeded `reset()` draws a fresh build seed from the gym RNG. This matters: SB3
vector envs only seed the *first* reset, so builders must never fall back to fixed
seeds on the env path or training collapses to one repeated episode.

### Configuration

- **`EnvConfig`** — `build`, `action_spec`, `step_duration_s` (default 1.0),
  `episode_duration_s`, `reward_weights`, `drain_on_done`, `initial_action`,
  `emit_latency_samples`, `max_latency_samples`.
- **`ActionSpec`** (`rl/spec.py`) — the discrete choices: `attempt_timeout_ms`,
  `max_attempts`, `backoff_ms`, and optional `global_timeout_ms` (each a list). The
  action space is a `MultiDiscrete` over the *indices* into these lists.
- **`RewardWeights`** (`rl/spec.py`) — defaults `success=1.0`, `deadline_drop=1.0`,
  `queue_drop=1.0`, `server_drop=1.0`, `circuit_open_drop=1.0`,
  `p95_latency_ms=0.001`, `retries=0.1`.

### Observation (normalized `Box`, 15 dims + 1 with a global-timeout dimension)

In order: a has-data flag (disambiguates "no traffic" from "all failed"),
log-scaled completed count, the five outcome rates (success, deadline, queue,
server, circuit-open), avg and p95 latency scaled by the largest attempt timeout in
the spec, retries scaled by the max retry budget, log-scaled queue depth, the last
action echoed as index fractions (one per action dimension), and the episode-time
fraction.

### Reward

```
reward = success_rate
       - deadline_drop_rate - queue_drop_rate - server_drop_rate - circuit_open_drop_rate
       - p95_latency_ms * 0.001
       - retries_per_root * 0.1            (each term scaled by its RewardWeights)
```

### Constraint cost

`EnvComponents.cost_client` (set by the chain builders to the B→C edge client)
defines where the constraint cost is measured: `summary["cost"]` is the edge's
retry amplification (retries per edge root) over each window — distinct from
`retries_per_root`, which measures the top-level client. The reward subtracts
`cost * RewardWeights.cost`; the weight defaults to 0 and doubles as the
Lagrange multiplier in constrained training.

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

## Latency models (`rl/models/`)

All implement `LatencyModel` (`rl/models/base.py`): `fit`, `logpdf`, `cdf`,
`quantile`, `sample`, `params`, `log_likelihood`.

| Model | Captures |
| --- | --- |
| `LogNormalModel` (`lognormal.py`) | a single lognormal backed by `scipy.stats.lognorm`; `fit_censored` does right-censored MLE via `scipy.stats.CensoredData`; `moments()` feeds the queueing math |
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
training chunks (`λ ← max(0, λ + lr·(cost − budget))`). The hand `retries`
penalty is zeroed during constrained training; evaluation always reports the
base objective **J** (reward with retry/cost penalties zeroed) so arms with
different training weights stay comparable. Honest caveats: the constraint is
enforced **in expectation** (not per episode/window — report violation rates),
and `RetryBudgetPolicy.budget_ratio` limits retries against *successes*, which
approximates retries-per-root only at high success rates.

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
capacity constants; the FAIL messages name the knob to turn.

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
