## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management, [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [Ty](https://github.com/astral-sh/ty) for type checking. Pre-commit runs Ruff and Ty automatically on every commit.

```bash
# install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
make install
```

That installs dev dependencies and the git pre-commit hook, which runs `ruff check --fix`, `ruff format`, and `ty check` on every commit. Run `make help` to see all available commands (`lint`, `fmt`, `type`, `test`, `cov`, `check`, etc.).

Examples:

- Single service experiments (legacy examples):
  - `python -m experiments.default`

- Chain/fanout example (A -> B -> {C, E}):
  - `python -m experiments.chain_fanout`

Downstream wiring uses tandem semantics by default:

- Each `DownstreamEdge` immediately releases the upstream worker after the local
  stage completes, while the downstream requests continue asynchronously.
- Final success or failure still waits for all downstream fan-out to finish, so
  retries/failures propagate back through the chain even though capacity is
  freed.

Example:

```python
from chain import DownstreamEdge

# B fans out to C and E using tandem semantics on both links
b_node.add_downstream(DownstreamEdge(client=b_to_c_client, target=c_node))
b_node.add_downstream(DownstreamEdge(client=b_to_e_client, target=e_node))
```

Plotting:

- Failures over time with fault events:
  - `python -m plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json`
  - `python -m plotting.failures_over_time multichain_A_output.csv --fault-events multichain_fault_events.json`

Metastable retry storm example (multi-chain):

- `python -m experiments.multichain_retry_storm`

Gymnasium integration (RL):

- Install Gymnasium if needed: `python -m pip install gymnasium`
- Example usage:
  - `python - <<'PY'`
  - `from rl.env import ActionSpec, EnvConfig, MicroservicesEnv, build_single_service_env`
  - `cfg = EnvConfig(build=build_single_service_env, action_spec=ActionSpec([10, 50], [1, 2], [0, 50]))`
  - `env = MicroservicesEnv(cfg)`
  - `obs, info = env.reset(seed=123)`
  - `obs, reward, terminated, truncated, info = env.step(env.action_space.sample())`
  - `print(reward, info["summary"])`
  - `PY`
- Train with Stable-Baselines3 (PPO):
  - `python -m pip install stable-baselines3`
  - `python -m rl.train_sb3 --timesteps 5000 --save-path rl_models/ppo.zip`
- Compare latency models (synthetic):
  - `python -m rl.experiments.compare_models --dist lognormal`
  - `python -m rl.experiments.compare_models --dist mixture`
  - `python -m rl.experiments.compare_models --dist pareto`
- Compare policy baselines:
  - `python -m rl.experiments.compare_policies --policies fixed,aimd,model_based,model_based_queue,bayesian,ppo,oracle_fixed`
  - `python -m rl.experiments.compare_policies --fault-scenario basic --policies fixed,aimd,model_based,bayesian,ppo,oracle_fixed`
