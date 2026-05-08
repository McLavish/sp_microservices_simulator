# Repository Guidelines

## Project Structure & Module Organization

This is a Python discrete-event simulator for microservice request flows, queues, retries, faults, and RL policy experiments. Core simulator modules live at the repository root: `simulator.py`, `service.py`, `caller.py`, `chain.py`, `retry_policies.py`, `timeout_controllers.py`, `workload.py`, `faults.py`, `fault_events.py`, and `metrics.py`. Experiment entry points are in `experiments/`, with single-service examples under `experiments/single/`. Plotting scripts live in `plotting/` and consume generated CSV/JSON artifacts. RL environments, models, policies, and training scripts live under `rl/`. Unit tests are in `tests/`.

## Build, Test, and Development Commands

- `python -m unittest discover -s tests`: run the full test suite.
- `python -m experiments.default`: run the default single-service simulation and generate `output.csv` plus `fault_events.json`.
- `python -m experiments.chain_fanout`: run the A -> B -> {C, E} chain/fanout scenario and generate chain/per-service CSVs.
- `python -m experiments.multichain_retry_storm`: run the multi-chain retry storm scenario.
- `python -m plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json`: render a failure timeline from simulation output.
- `python -m rl.experiments.compare_models --dist lognormal`: run an RL model comparison. Install optional dependencies such as `gymnasium` or `stable-baselines3` only when needed.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation and descriptive `snake_case` names for modules, functions, variables, and tests. Prefer dataclasses for structured simulator state, following existing patterns in `core.py` and `rl/`. Keep internal time values in nanoseconds using `TimePoint`, `TimeDuration`, `s_to_ns()`, and `ms_to_ns()`; convert only at I/O or plotting boundaries. Avoid broad refactors when adding policies, controllers, or experiments.

## Testing Guidelines

Tests use `unittest` and are discovered from `tests/test_*.py`. Add focused tests beside related behavior, for example `tests/test_retry_policies.py` for retry changes or `tests/test_chain_failure_reason.py` for chain semantics. Cover deadline, queue, retry, and fault edge cases explicitly. Optional Gymnasium tests skip when dependencies are absent.

## Commit & Pull Request Guidelines

Recent history uses short lowercase commit subjects such as `rl test` and `minor fixes`; keep subjects concise but more descriptive when possible, for example `fix retry deadline accounting`. Pull requests should state the scenario changed, commands run, and generated artifacts affected. Include plots or screenshots when changing plotting output. Do not commit transient `*.csv`, `*.json`, `*.png`, `results/`, `.venv/`, or cache files.

## Agent-Specific Notes

Preserve simulator semantics: deadlines are half-open, queue-full drops happen before enqueue, and chain fan-out uses AND semantics. When experiments export metrics, keep CSV output and fault-event JSON aligned so plotting overlays remain meaningful.
