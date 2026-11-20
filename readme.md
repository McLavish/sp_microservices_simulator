# microservices_simulator

A single-threaded **discrete-event simulator** of microservice request flows. It
models queues, retries, timeouts, partial failures, latency injection, and load
spikes across single services, chains, and fan-outs, and exports per-second CSV
metrics and fault-event JSON for plotting.

- **Services** — M/G/c/K queues with lognormal service times (one per service, or
  one per handler over a shared station), bounded queues served first-come, newest-first
  or at random, and configurable workers.
- **Clients** — retry policies (fixed/exponential/jittered backoff, circuit
  breakers, rate limiters, retry budgets, retries chosen by the cause of the failure)
  and static or adaptive timeout control.
- **Chains** — multi-hop fan-out with AND semantics, concurrent or sequential
  downstream calls (call waves), and tandem vs. hold-worker capacity models.
- **Admission control** — a gate in front of a service that refuses retries by a
  server-side rule (token bucket, AIMD, success-funded and goodput-coupled
  budgets, a failure-rate gate, a share of what is in flight), per tenant where
  the rule is.
- **Faults** — latency injection, partial failures, and load spikes scoped to time
  windows.
- **Outputs** — per-bucket CSV metrics + seconds-based fault JSON, plus plotting
  scripts that overlay the two.
- **Other formats** — `msim.compat.arolla` reads the YAML scenarios of
  [Arolla](https://github.com/eth-easl/Arolla)'s simulator and builds them here
  (`docs/compat-arolla.md`).
- **Reinforcement learning** — `msim.rl` is a [Gymnasium](https://gymnasium.farama.org)
  environment in which an agent re-tunes the retry policy of some clients, window
  by window, while a simulation runs; with scenarios whose best fixed policy
  depends on the fault, a search over the fixed policies to compare against, and
  PPO training on simulations run side by side (`docs/rl.md`).

## Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

The simulator library is the `msim` package under `src/` (including `msim.plotting`);
`make install` (`uv sync`) installs it editable, so `from msim.core import ...` works
from any directory. Consumers depend on `microservices-simulator[plot]` — the `plot`
extra carries `matplotlib` / `pandas` / `scipy`, and `pandas` is also needed for CSV
export. The `examples/` demo scenarios are not part of the package — run them as
`python -m examples.<name>` from the repo root.

## Install

```bash
make install        # uv sync --group dev + install the git pre-commit hook
```

The simulator core depends only on `numpy`; the `plot` extra adds the `scipy` /
`matplotlib` / `pandas` used by plotting and CSV export, the `arolla` extra the
PyYAML that `msim.compat.arolla` reads scenario files with, and the `rl` extra the
Gymnasium that `msim.rl` is written against. The dev group adds `ruff`, `ty`, `pytest`
and those same libraries. Training (`msim.rl.train`) needs the `train` extra on top:
Stable-Baselines3 and a CPU build of PyTorch, about 1 GB
(`uv sync --group dev --extra train`).

## Quickstart

Run a scenario, then plot it:

```bash
python -m examples.default           # writes output.csv + fault_events.json
python -m msim.plotting.failures_over_time output.csv --fault-events fault_events.json
```

## Development

All quality gates run through `uv` so the `.venv` stays consistent:

```bash
make check     # lint + format check + ty type check + tests (the full gate)
make test      # pytest        (single module: uv run pytest tests/test_transport.py)
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
| [examples](docs/examples.md) | Runnable example scenarios and their outputs. |
