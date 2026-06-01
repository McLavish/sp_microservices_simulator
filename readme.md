# microservices_simulator

A single-threaded **discrete-event simulator** of microservice request flows. It
models queues, retries, timeouts, partial failures, latency injection, and load
spikes across single services, chains, and fan-outs; exports per-second CSV metrics
and fault-event JSON for plotting; and exposes the whole thing as a Gymnasium
environment for reinforcement-learning experiments on retry/timeout policies.

- **Services** — M/G/c/K queues with lognormal service times, bounded queues, and
  configurable workers.
- **Clients** — retry policies (fixed/exponential/jittered backoff, circuit
  breakers, rate limiters, retry budgets) and static or adaptive timeout control.
- **Chains** — multi-hop fan-out with AND semantics and tandem vs. hold-worker
  capacity models.
- **Faults** — latency injection, partial failures, and load spikes scoped to time
  windows.
- **Outputs** — per-bucket CSV metrics + seconds-based fault JSON, plus plotting
  scripts that overlay the two.
- **RL** — a Gymnasium env, baseline policies, latency models, and PPO training.

## Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

This is **not** an installable package (`pyproject.toml` sets `package = false`).
Run everything as `python -m <module>` from the repo root; imports are flat
(`from core import ...`).

## Install

```bash
make install        # uv sync --group dev + install the git pre-commit hook
```

The simulator core depends only on `numpy`. The dev group adds `ruff`, `ty`,
`pytest`, and the optional `scipy` / `matplotlib` / `gymnasium` used by plotting and
the RL layer.

## Quickstart

Run a scenario, then plot it:

```bash
python -m experiments.default        # writes output.csv + fault_events.json
python -m plotting.failures_over_time output.csv --fault-events fault_events.json
```

Drive the RL environment directly:

```python
from rl.env import ActionSpec, EnvConfig, MicroservicesEnv, build_single_service_env

cfg = EnvConfig(build=build_single_service_env,
                action_spec=ActionSpec(attempt_timeout_ms=[10, 50], max_attempts=[1, 2], backoff_ms=[0, 50]))
env = MicroservicesEnv(cfg)
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Development

All quality gates run through `uv` so the `.venv` stays consistent:

```bash
make check     # lint + format check + ty type check + tests (the full gate)
make test      # pytest        (single module: uv run pytest tests/test_rl_env.py)
make lint      # ruff check
make fmt       # ruff format
make type      # ty check
make help      # list all targets
```

Pre-commit runs `ruff check --fix`, `ruff format`, and `ty check` on every commit.

## Documentation

Full technical docs live in [`docs/`](docs/README.md):

| Doc | What it covers |
| --- | --- |
| [architecture](docs/architecture.md) | Event loop, time model, core types, request lifecycle. |
| [services-and-workload](docs/services-and-workload.md) | Service queue, service times, arrivals, fault injection. |
| [request-handling](docs/request-handling.md) | Clients, retry policies, timeouts, chains/fan-out. |
| [observability](docs/observability.md) | Metrics, CSV schema, plotting. |
| [experiments](docs/experiments.md) | Runnable scenarios and their outputs. |
| [rl](docs/rl.md) | Gymnasium env, policies, latency models, training. |
