# Reinforcement-learning layer (`rl/`)

The simulator doubles as a control problem: pick retry/timeout settings to maximize
goodput while minimizing drops, latency, and retries. `rl/` exposes the simulator as
a Gymnasium environment, ships baseline policies, fits latency models used by some of
those policies, and provides comparison/training entry points.

**Optional dependencies** (the simulator core needs only `numpy`):

| Dependency | Needed for |
| --- | --- |
| `gymnasium` | `MicroservicesEnv` (anything that builds the env) |
| `stable-baselines3` | PPO training/policy (`rl.train_sb3`, the `ppo`/`sb3` policy) |
| `scipy` | latency-model fitting / `compare_models` |

## The environment (`rl/env.py`)

`MicroservicesEnv` (`rl/env.py:86`) wraps a **build function** (which constructs a
fresh `Simulator` + `Workload` + `ClientRuntime`) into a Gym env. Each `step()`
advances the sim by `step_duration_s` and summarizes the roots that completed in that
window.

### Configuration

- **`EnvConfig`** (`rl/env.py:73`) — `build`, `action_spec`, `step_duration_s`
  (default 1.0), `episode_duration_s`, `reward_weights`, `drain_on_done`,
  `initial_action`, `emit_latency_samples`.
- **`ActionSpec`** (`rl/env.py:65`) — the discrete choices:
  `attempt_timeout_ms`, `max_attempts`, `backoff_ms`, and optional
  `global_timeout_ms` (each a list). The action space is a `MultiDiscrete` over the
  *indices* into these lists.
- **`RewardWeights`** (`rl/env.py:55`) — defaults `success=1.0`,
  `deadline_drop=1.0`, `queue_drop=1.0`, `server_drop=1.0`, `p95_latency_ms=0.001`,
  `retries=0.1`.

### Observation (12-dim `Box`, `rl/env.py:315`)

`[success_rate, deadline_drop_rate, queue_drop_rate, server_drop_rate,
avg_latency_ms, p95_latency_ms, retries_per_root, queue_avg, attempt_timeout_ms,
max_attempts, backoff_ms, time_fraction]` — eight per-window statistics, the three
current action values, and the fraction of the episode elapsed.

### Reward (`rl/env.py:237`)

```
reward = success_rate
       - deadline_drop_rate - queue_drop_rate - server_drop_rate
       - p95_latency_ms * 0.001
       - retries_per_root * 0.1            (each term scaled by its RewardWeights)
```

### Applying an action — the hard constraint

`_apply_action` (`rl/env.py:202`) mutates the client's controller and policy
**in place**: `StaticTimeoutController.attempt_timeout` / `.global_timeout` and
`FixedBackoffRetryPolicy.max_attempts` / `.delay`. It raises `TypeError` for any
other controller/policy type. If you change the policy in a build function, update
`_apply_action` to match.

### Build functions

- `build_single_service_env` (`rl/env.py:372`) — one `svc-A` (20 ms median, 16
  workers, queue 200), 200 rps, `StaticTimeoutController` + `FixedBackoffRetryPolicy`.
- `build_faulty_single_service_env` (`rl/env.py:402`) — same, plus a latency
  injection, a partial failure (`p=0.3`), and a 2x load spike scheduled relative to
  the episode duration.

### Example

```python
from rl.env import ActionSpec, EnvConfig, MicroservicesEnv, build_single_service_env

cfg = EnvConfig(
    build=build_single_service_env,
    action_spec=ActionSpec(attempt_timeout_ms=[10, 50], max_attempts=[1, 2], backoff_ms=[0, 50]),
)
env = MicroservicesEnv(cfg)
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(reward, info["summary"])
```

## Policies (`rl/policies/`)

All implement the `Policy` protocol (`rl/policies/base.py`): a `name`, plus
`reset(seed)`, `act(obs, info) -> action`, and `update(obs, reward, done, info)`.
Actions are index tuples matching the env's `MultiDiscrete`; `action_space.py`
provides `ActionGrid` (enumerate/snap-to-nearest) and `coerce_action`.

| Policy (class) | Idea |
| --- | --- |
| `FixedPolicy` (`fixed.py`) | emit a constant action; also used for the **oracle** (best constant action found by grid search) |
| `FixedValuePolicy` (`fixed.py`) | specify raw ms values; snaps to the nearest grid indices |
| `AIMDPolicy` (`aimd.py`) | additive-increase / multiplicative-decrease on the timeout index, driven by smoothed success/deadline rates vs. targets |
| `ModelBasedPolicy` (`model_based.py`) | fit a [latency model](#latency-models-rlmodels) to buffered samples, then pick the action grid point with the best Monte-Carlo expected reward; optional M/M/c queueing-delay term and multi-step rollouts |
| `BayesianPolicy` (`bayesian.py`) | maintain a Normal-Inverse-Gamma posterior over log-latency, sample parameters, fit a lognormal, then grid-search like `ModelBasedPolicy` |
| `SB3Policy` (`sb3.py`) | adapt a trained Stable-Baselines3 (PPO) model to the `Policy` protocol |

## Latency models (`rl/models/`)

Used by the model-based and Bayesian policies (and benchmarked by `compare_models`).
All implement `LatencyModel` (`rl/models/base.py`): `fit`, `logpdf`, `cdf`,
`quantile`, `sample`, `params`, `log_likelihood`.

| Model | Captures |
| --- | --- |
| `LogNormalModel` (`lognormal.py`) | a single lognormal (`mu`, `sigma`); also exposes `moments()` for queueing math |
| `LogNormalMixtureModel` (`lognormal_mixture.py`) | a K-component lognormal mixture (EM-fit) for multi-modal latency |
| `EVTTailModel` (`evt_tail.py`) | a lognormal **body** plus a Pareto (power-law) **tail** above a threshold quantile, for heavy tails |

## Entry points

### `python -m rl.train_sb3`

Train (and optionally evaluate) a PPO agent on the env. Key args (`rl/train_sb3.py`):
`--timesteps 5000`, `--seed 123`, `--n-envs 1`, `--step-duration-s 1.0`,
`--episode-duration-s 10.0`, `--attempt-timeouts-ms 10,30,50,80`,
`--max-attempts 1,2,3`, `--backoff-ms 0,20,50`, `--global-timeouts-ms`,
`--n-steps 64`, `--batch-size 64`, `--gamma 0.95`, `--save-path`, `--load-path`,
`--eval-episodes 1`. Requires `stable-baselines3`.

```bash
python -m rl.train_sb3 --timesteps 5000 --save-path rl_models/ppo.zip
```

### `python -m rl.experiments.compare_models`

Fit all three latency models to synthetic samples and report NLL + p95/p99 quantile
error. Args: `--dist {lognormal,mixture,pareto}`, `--n 5000`, `--seed 123`, and
distribution parameters (`--mu`, `--sigma`, `--w`, `--mu2`, `--sigma2`, `--alpha`,
`--x-min`). Requires `scipy`.

```bash
python -m rl.experiments.compare_models --dist mixture
```

### `python -m rl.experiments.compare_policies`

Evaluate policies across train/eval seeds and print reward / success rate / p95 /
retries per policy. `--policies` is a comma list drawn from: `fixed`, `aimd`,
`model_based`, `model_based_rl`, `model_based_queue`, `model_based_rl_queue`,
`bayesian`, `ppo`, `oracle_fixed`. `--fault-scenario {none,basic}` selects the clean
or faulty build function; additional flags tune the action grid and individual
policies. `ppo` requires `stable-baselines3`.

```bash
python -m rl.experiments.compare_policies \
    --policies fixed,aimd,model_based,model_based_queue,bayesian,oracle_fixed \
    --fault-scenario basic
```

## See also

- [request-handling](request-handling.md) — the `StaticTimeoutController` /
  `FixedBackoffRetryPolicy` the env tunes.
- [observability](observability.md) — the per-window statistics the reward is
  built from.
