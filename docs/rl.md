# Reinforcement learning (`msim.rl`)

The simulator as a control problem: a [Gymnasium](https://gymnasium.farama.org)
environment in which an agent re-tunes the retry policy of some clients while a
simulation runs, plus what is needed to train on it and to say honestly whether
training helped.

```python
import gymnasium
import msim.rl  # registers the environments

env = gymnasium.make("msim/RetryStorm-v0")
observation, info = env.reset(seed=1)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

| Extra | Brings | For |
| --- | --- | --- |
| `rl` | `gymnasium` | the environment, the scenarios, evaluation (`msim.rl`) |
| `train` | + `stable-baselines3`, PyTorch (CPU build) | `msim.rl.train` |
| `arolla` | `pyyaml` | scenarios from Arolla files (`msim.rl.arolla`) |

`make install` installs what the first and the third bring;
`uv sync --group dev --extra train` adds the second (about 1 GB), and
`make test-train` runs its tests.

## The environment (`rl/env.py`)

`RetryEnv(scenario, step_s=1.0, warmup_s=5.0)` drives a simulation from outside:

- **A step is a window of simulated time.** The agent's action is applied, the
  simulation runs `step_s` further (`Simulator.run(until=...)`), and the agent
  gets an observation and a reward for that window. A run paused at every
  window is, event for event, the run left alone, and nothing the environment
  does draws from the simulator's RNG or schedules an event: an episode under
  the default setting is the scenario run with no environment, record for
  record (`tests/test_rl_env.py`).
- **The action** is one level index per knob (`MultiDiscrete`). The knobs are
  the scenario's; the environment knows none by name. A level is an absolute
  value, so every fixed configuration is a constant action.
- **The observation** is what the agent's own clients measured in the window,
  each entry in `[0, 1]` (a `Box`, and every entry is clipped to it). Nothing
  of a service's state is in it; queue lengths and workers in use are in `info`.
- **The reward** is the scenario's goodput: requests that succeeded in the
  window, over the number expected to arrive in a window. A healthy window pays
  about 1. It is counted on requests with a scale that is fixed for the episode,
  so the rewards of an episode add up to its successes whatever the windows.
- **An episode never terminates.** It is cut at the scenario's horizon
  (`truncated`), so a learner bootstraps over the cut and gains nothing by
  pushing failures past it. After the cut the arrivals stop and what is in
  flight finishes; that tail is in the episode summary and in no reward.
- **Seeding.** `reset(seed=n)` seeds the environment, and every reset then
  draws the seed of its episode, so a seeded environment replays the same
  sequence of episodes and 48 of them seeded apart see 48 different ones.
  `reset(options={"episode_seed": n})` asks for one episode in particular.

The first observation is of a warm-up, run under the scenario's default
setting: the agent takes over a system that is already serving.

### What the agent sees

Pooled over the agent's clients and counted per attempt, as each one ends (a
pass-through policy layer and a timeout controller that notes latencies: an
attempt is known when it ends, not when its request has finished retrying).

| Entry | Meaning |
| --- | --- |
| `has_data` | 1 when an attempt ended in the window: tells "idle" from "everything failed" |
| `request_rate` | requests started, over twice the rate expected when all is well |
| `in_flight` | requests not yet finished, over the most the settings allow |
| `attempt_success`, `attempt_deadline`, `attempt_queue_full`, `attempt_server_failure`, `attempt_other` | shares of the attempts that ended, by how (`other`: refused by admission control) |
| `retry_share` | share of those attempts that were retries |
| `retry_success` | share of the retries that succeeded |
| `latency_mean`, `latency_p95` | of successful attempts, over the longest an attempt may take |
| `budget_balance` | how full the retry budget is |
| `knob:<name>` | the level each knob is at, 0 for the first and 1 for the last |

`env.feature_names` lists them in order. There is no clock and no fault flag:
the agent has to read the fault off its clients, as a deployed one would.

### The episode summary

The last step's `info` carries it as `ep_*` keys; `rl/summary.py` is the one
place that defines them, for an episode run through the environment and for a
run that had none (`summary.of_records`).

| Key | Meaning |
| --- | --- |
| `ep_return` | the rewards, added up |
| `ep_success_rate` | requests that succeeded, of all that finished after the warm-up, the tail included |
| `ep_horizon_success_rate` | the same, of what finished inside the horizon (what the rewards count) |
| `ep_fault_success_rate` | of the requests that started inside a fault window |
| `ep_amplification` | attempts per request of the agent's clients |
| `ep_retry_efficiency` | share of their retries that succeeded |
| `ep_recovery_s` | from the end of the last fault to the first window in which 95% of the finished requests succeeded (0: the next one; `inf`: none) |
| `ep_failed_<outcome>` | failed requests by outcome |

Every step's `info` also has the window's counts (`requests`, `succeeded`,
`failed_<outcome>`), the knob values (`knob_<name>`) and, for each station,
`queue_<name>` and `in_flight_<name>`. All of it is numeric.

## Scenarios (`rl/scenario.py`, `rl/scenarios.py`)

A scenario is a picklable object with three things: `knobs` (known before
anything is built), `horizon_s`, and `build(seed)`, which returns a fresh
simulation as an `Episode`: the simulator, the workloads, the agent's clients
with their settings, the clients the reward is counted on, the expected rate.

The stock settings are the retry settings of a group of clients, a `RetryGrid`:

| Knob | Default levels | Acts on |
| --- | --- | --- |
| `max_attempts` | 1, 2, 3, 4 | `FixedBackoffRetryPolicy.max_attempts` |
| `attempt_timeout_ms` | 50, 100, 200 | `StaticTimeoutController.attempt_timeout` |
| `backoff_ms` | 0, 10, 50 | `FixedBackoffRetryPolicy.delay` |
| `budget_ratio` | 0.05, 0.1, 0.2, 0.5, 1 | `RetryBudgetPolicy.set_budget_ratio` |

`RetryGrid.install(clients)` gives each client that stack and returns the
episode's `RetryKnobs`, which applies a value to all of them alike: one policy
rolled out to every agent client. The four are the settings
[request-handling](request-handling.md#changing-a-policy-while-a-run-is-in-progress)
documents as changeable while a run is in progress. A level list of one value
fixes a knob; `None` leaves out the timeout knob or the budget.

### The two that ship

Both put ONE fault in each episode, of a kind and at a time the agent is not
told: the station fails a share of its calls (`failure`), every service time
grows (`slowdown`), or the load rises (`spike`); sometimes nothing happens. Kind,
onset, length, strength and the episode's load are drawn from the episode seed
by a generator of the scenario's own, so what the agent does changes neither
the fault nor the arrivals.

- **`SingleService`** (`msim/SingleService-v0`): one client on one station, 400
  requests per second on 16 workers of 20 ms.
- **`RetryStorm`** (`msim/RetryStorm-v0`): two chains, A → B → C and X → Y → C,
  share the bottleneck C. The agent drives both callers of C; the reward is the
  goodput of both chains where the requests enter.

The kinds pull the policy in opposite directions, which is what makes these a
control problem. Fixed settings through the environment, five seeds, a fault
from 20 s to 40 s of the 65 s episode; success of all requests in %, and in
brackets of the requests that started during the fault:

| | no retries | 4 attempts | 3 attempts, 10% budget |
| --- | --- | --- | --- |
| **SingleService**, 40% failures | 86.6 (60.0) | **99.1** (97.3) | 88.6 (65.8) |
| +25 ms | **84.7** (54.3) | 68.1 (4.1) | 79.6 (38.7) |
| load x2 | **91.1** (82.2) | 64.2 (28.4) | 87.0 (73.9) |
| **RetryStorm**, 50% failures | 82.4 (49.6) | **93.1** (79.2) | 84.8 (54.2) |
| +15 ms | **95.6** (89.4) | 86.3 (58.9) | 95.5 (86.6) |
| load x1.6 | 96.8 (94.3) | 95.2 (89.2) | **98.0** (95.5) |

When calls fail, every retry is a fresh chance. When the station is slow or the
load spikes, attempts time out and a retry is one more call on a station that is
already behind. No fixed setting is best in both, and a budget is the compromise
that is best in neither. `tests/test_rl_scenarios.py` pins this property, so a
change of constants cannot quietly turn a scenario into one a constant solves.

`RetryStorm` has the layout of `examples/multichain_retry_storm.py` and not its
numbers. With the example's, nothing the callers of C do changes the outcome:
success in the fault window stays between 3 and 5% under its slowdown whether
they make one attempt or five, and at 100% under failures alone, because the
layers above them retry too and the fault leaves C beyond help. Here only the
callers of C retry and C runs at about 60% of what it can serve.
`RetryStorm(upstream_attempts=3)` brings the layered storm back.

### Arolla scenario files (`rl/arolla.py`)

```python
from msim.rl.arolla import ArollaScenario

scenario = ArollaScenario.from_file(".../s03__thundering_herd.yaml", agent="three_retries")
```

[Arolla](https://github.com/eth-easl/Arolla) writes a scenario as arms side by
side: one client entry per retry policy, often each on a service of its own.
The agent takes one arm over. The situation comes from the file through
[`msim.compat.arolla`](compat-arolla.md) (services, faults, server-side
limiters, load, timeouts, replicas); the agent's clients get the settings'
stack in place of the file's policy; and only the part of the file those clients
can interact with is built. `agent` names client entries (every replica of
each) or services that retry their own calls, whose retry loop is then taken
over. The default levels come from the file (`file_grid`).

Checked on the four post-mortem scenarios and on
`snowflake/policy-{none,cb,rb,aimd,arolla}.yaml`
(`AROLLA_CHECKOUT=<their repo> uv run pytest tests/test_rl_arolla.py`). A
70 s episode of one of them takes between 0.1 and 1.8 s.

`as_written(path, seeds)` runs a file with nobody taken over and scores it by
the same definitions: their clients, behind their limiter. That is the row a
learned policy on that file is put next to.

### Writing one

```python
@dataclass(frozen=True)
class Mine:
    grid: RetryGrid = RetryGrid()
    horizon_s: float = 65.0

    @property
    def knobs(self):
        return self.grid.knobs()

    def build(self, seed: int) -> Episode:
        sim = Simulator(seed=seed)
        ...  # stations, clients, workloads, as in any example; drive the workloads
        agent = self.grid.install([client], agent_rps=rps)  # before any request starts
        return Episode(sim=sim, workloads=[workload], agent=agent,
                       reward_clients=[client], clients=[client], expected_rps=rps)
```

`tests/rl_small.py` is a complete one in sixty lines. Two rules: draw everything
that varies from `seed`, with a generator that is not the simulator's; and list
every client in `clients`, because the environment drops the finished-request
records it has read from all of them. `keep_records=True` keeps them, for
`Metrics` or a CSV afterwards; on a chain as large as the example's an episode
then takes 22.7 s instead of 13 s, and 477 MB instead of 76 MB.

Other things to control (a server-side retry budget, a budget per failure
cause) need their own counterpart of `RetryGrid` and `RetryKnobs`; the
environment needs no change as long as the agent is still a group of clients.

## Comparing policies (`rl/evaluate.py`)

```bash
python -m msim.rl.evaluate storm --jobs 48 --out results/storm
python -m msim.rl.evaluate single --model runs/single/model.zip --jobs 48
```

A policy is anything that turns an observation into an action. `rollout` runs
one on given episodes, so two policies meet the same faults and the same
arrivals, and `compare` reports each metric with its standard error and the
difference in success rate against a reference, episode by episode.

The baselines are the fixed settings: every combination of knob levels is a
constant policy (180 with the default grid). `static_search` runs them all on
the same episodes and ranks them by return. `baselines` picks the best on the
selection episodes (seeds 1000 to 1015) and reports it, the scenario's default
and "no retries" on the test episodes (2000 to 2031). Chosen and scored on the
same episodes, the best of 180 would look better than it is.

`--as-written FILE ...` adds Arolla files run as they are written.

## Training (`rl/train.py`)

```bash
python -m msim.rl.train storm --timesteps 1000000 --n-envs 48 --out runs/storm
```

Stable-Baselines3's PPO (`MlpPolicy`) on `--n-envs` environments, each in a
process of its own. Between two updates every environment runs two episodes.
The run directory gets `model.zip`, its sidecar `model.json`, the
training log `progress.csv`, one monitor file per environment, and the
comparison with the fixed settings on the test episodes (`comparison.csv`,
`episodes.csv`, `static.csv`). The model compared is the one training ended
with: no checkpoint is picked on the test episodes.

The sidecar says what the model was trained on: the scenario and its settings,
the knobs with their levels in order, the observation's entries, the step, the
steps actually trained, the commit, the library versions. `load_policy` refuses
an environment that does not match it. A model's file knows only the shapes of
its spaces, and would otherwise run on a grid whose levels mean something else.

The learner is paid the reward minus 1 (`--reward-offset`). The environment
pays about 1 for every healthy window and an episode is cut, not ended, so the
learner bootstraps over the cut: left as it is, the value of a healthy state is
about `1 / (1 - gamma)`, a hundred at a discount of 0.99, and the value function
spends the first part of training climbing there while the advantages it yields
are mostly its own error. With 1 taken off, a healthy window pays 0 and what is
left to learn is what a fault costs. Every episode of a scenario has the same
number of steps, so the constant changes no comparison between two policies;
the logs and every evaluation use the environment's own reward.

`--device` says where the learner runs (`auto` by default: a GPU when the
installed PyTorch can use one). The `train` extra installs PyTorch's CPU build,
so on a GPU node a CUDA build has to be installed over it first. The learner is
a small part of a run either way: the policy is two layers of 64 units, and
with 48 simulations a training run made 500 steps per second where the same
simulations stepped with random settings and no learner made 597.

### How fast

One environment step is 15 ms on the single service and 21 ms on the storm,
almost all of it the simulation (random settings; a host with two Xeon Gold
6140, 36 cores and 72 threads). Side by side on that host through
Stable-Baselines3's vector environment:

| Simulations | SingleService, steps/s | RetryStorm, steps/s |
| --- | --- | --- |
| 1 | 58 | 35 |
| 8 | 205 | 114 |
| 16 | 368 | 159 |
| 48 | 597 | 278 |

Forty-eight simulations give ten times one, not forty-eight: every step of the
vector waits for its slowest simulation, and a window in which a client retries
into an overloaded station takes several times the median (p99 is 46 ms against
a median of 13 on the single service). Collecting whole episodes per process,
with no waiting between steps, is what would recover the rest.
