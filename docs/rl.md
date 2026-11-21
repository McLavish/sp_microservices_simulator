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
  `RetryEnv(reward_within_s=...)` (default `None`) pays only the requests that
  succeeded within that many seconds of their first attempt. Without it a
  request served in 10 ms and one that waited three seconds in a queue pay the
  same, and where a queue can carry requests past a failure, a limiter learns
  to fill it (see "On their Kubernetes prototype"). The bound changes what is
  paid and one line of the summary (`ep_on_time_rate`), never the simulation.
  `RetryEnv(retry_cost=...)` (default 0) takes that many requests off the
  reward for every retry that reached its target (`ep_retries_served`: not
  refused by a limiter, not held back by the caller's breaker). A retry that
  cannot succeed pays nothing either way, so without a price an agent that
  decides what is let in has no reason to refuse it, and keeps letting some in.
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

One window is a noisy measurement: a second of 160 requests tells the request
rate to within 8%, and the policy has no memory. `RetryEnv(scenario,
slow_s=10)` shows every measured entry a second time, averaged exponentially
over about that many seconds (`<name>~10s`, between the entries of the last
window and the knob levels; the average starts at the warm-up). It is off by
default and a model is tied to it by its sidecar. An agent whose best setting
depends on something that holds for a whole episode (the load it is under)
needs it: without it, it cannot tell 1.15 times a load from 1.3 times.

### The episode summary

The last step's `info` carries it as `ep_*` keys; `rl/summary.py` is the one
place that defines them, for an episode run through the environment and for a
run that had none (`summary.of_records`).

| Key | Meaning |
| --- | --- |
| `ep_return` | the rewards, added up |
| `ep_success_rate` | requests that succeeded, of all that finished after the warm-up, the tail included |
| `ep_on_time_rate` | those that succeeded within `reward_within_s` of their first attempt (`ep_success_rate` when there is no bound) |
| `ep_retries_served` | the retries of those requests that reached their target (what `retry_cost` is charged on) |
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

### An agent that is the limiter (`rl/limiter.py`)

The agent need not sit on clients. `ArollaLimiterScenario.from_file(path)`
builds a file as it is written, every client with its own retry policy and
timeout, and replaces one thing: the controller inside the `AdmissionGate` in
front of a service. It becomes an `admission.TenantRetryQuota`, and the knobs
are one share per tenant (`quota:<tenant>`): how many of that tenant's retries
are let in, per first attempt of the tenant, over the last second. `QuotaGrid`
holds the ladder (0, 2%, 5%, 10%, 20%, 30%, 50%, 75%, 100%, and 300%, which
refuses nothing a caller of four attempts sends) and the default, 0, which
holds through the warm-up. `QuotaGrid(floor=1)` lets one retry per tenant and
window in whatever the level (`TenantRetryQuota.floor`): much of what the
agent is shown is what its retries do, so at a level of 0 it is blind to a
failure ending, and an agent without the floor learns to hold a few percent
through a total failure just to go on seeing.

Such an agent is shown what a limiter can count where it stands and nothing
else, per tenant and window (`limiter.MEASURES`, as `<tenant>:<measure>`):

| Entry | Meaning |
| --- | --- |
| `ended` | the tenant's attempts that ended at the service, over four times its expected request rate |
| `goodput` | those that succeeded, over twice its expected request rate |
| `retries_asked` | its retries that asked to be let in, over four times its expected request rate |
| `retries_admitted` | those let in |
| `retry_success` | share of its retries that ended which succeeded |

These are the quantities Arolla's rule runs on (a tenant's request rate, its
goodput, its retries in the window, the success of its retries). There is no
queue length, no latency, nothing from inside a client. The rates are the
FILE's, so a load that an episode draws shows in the counts. A scenario with
an agent of this kind names its entries (`features`) and its agent object
measures them (`Agent.measure`); `env.FEATURES` is what an agent on clients
gets. `slow_s` and the knob levels are added as for any agent.

So the scenario and `as_written` on the same file differ in the limiter and in
nothing else: `tests/test_rl_limiter.py` checks that a quota which refuses
nothing gives the file with no limiter, summary for summary.

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
`--snapshots n` also keeps `n` copies of the model on the way, evenly spaced
(`snapshots/model-<steps>.zip`, each with a sidecar of its own, so each is a
model like the final one). Choosing among them is a choice like any other:
make it on the selection episodes.
`--lr-decay` lets the learning rate fall linearly to zero over the run, so
that the policy settles at the end.

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

## A learned limiter against Arolla's (`examples/rl_vs_arolla.py`)

[Arolla](https://github.com/eth-easl/Arolla) protects a service from its
callers' retries at the server: it admits each tenant's retries up to a share
of that tenant's goodput. Its simulator ships one situation under each
server-side policy, `snowflake/policy-{none,cb,rb,aimd,arolla}.yaml`: a
16-worker service, four tenants of 40 requests per second with four different
client retry policies, and half the calls failing from 20 s to 40 s. Can a
learned limiter do better than theirs, in front of the same clients?

```bash
python -m examples.rl_vs_arolla static   --arolla ~/Arolla --out study --jobs 48   # the fixed quotas
python -m examples.rl_vs_arolla informed --arolla ~/Arolla --out study --jobs 48   # the best one per situation
python -m examples.rl_vs_arolla train    --arolla ~/Arolla --out study/runs/a-100 --seed 100 --slow-s 10
python -m examples.rl_vs_arolla compare  --arolla ~/Arolla --out study --jobs 48 --select lead
```

**What is compared.** Both sides are `policy-arolla.yaml` as it is written:
the same four tenants, each with its own retry policy (four attempts for all;
one backs off, one has a circuit breaker, one a client-side budget and no
attempt timeout), and one admission gate in front of the service. The only
thing that differs is the controller inside that gate: their rule (the one of
their simulator, which that file names; the rule of their Kubernetes prototype
is another one and is compared further down), or a quota per tenant that an
agent sets every second (`ArollaLimiterScenario`, above).
The agent sees what their rule runs on, per tenant: attempts and successes at
the service, retries asked and let in. Situations: their file as it is; a
varied one, where every episode draws the failure rate (0.1 to 0.9), its start
and length (5 to 35 s) and the load (0.7 to 1.3 times the file's); and each of
those three axes walked alone. For a seed every row faces the same failure and
the same arrivals, so the differences are paired. Whatever is chosen is chosen
on other episodes than the ones reported: the fixed quota, the trained model,
and the settings of their limiter when it is tuned (`alpha`, `beta_down`,
`beta_up`, 45 combinations around their file's values).

**Result.** Success of all requests in %, on 64 episodes that nothing had been
chosen on (seeds 4000 to 4063), and the paired difference to their limiter
with its standard error:

| Situation | Learned limiter | Arolla | Learned - Arolla | Arolla, tuned | Best fixed quota for the situation | One fixed quota | Every retry refused | No limiter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Their file | 88.57 | 85.54 | +3.04 ± 0.12 | 85.68 | 88.51 | 86.04 | 84.00 | 36.71 |
| Varied | 90.09 | 87.73 | +2.36 ± 0.32 | 87.85 | | 88.01 | 86.10 | 52.47 |
| Failure rate 0.1 | 99.60 | 98.70 | +0.90 ± 0.02 | 98.84 | 99.64 | 98.89 | 96.17 | 99.66 |
| Failure rate 0.25 | 97.86 | 94.03 | +3.83 ± 0.12 | 94.18 | 98.00 | 94.41 | 91.58 | 85.14 |
| Failure rate 0.75 | 78.96 | 77.21 | +1.75 ± 0.07 | 77.39 | 78.98 | 77.70 | 76.36 | 34.15 |
| Failure rate 0.9 | 73.12 | 72.44 | +0.68 ± 0.04 | 72.47 | 73.15 | 72.60 | 71.73 | 33.01 |
| Failing for 5 s | 97.15 | 96.45 | +0.70 ± 0.05 | 96.51 | 97.17 | 96.54 | 95.44 | 38.91 |
| Failing for 10 s | 94.32 | 92.84 | +1.48 ± 0.08 | 92.95 | 94.27 | 93.02 | 91.60 | 37.68 |
| Failing for 35 s | 80.08 | 74.57 | +5.51 ± 0.14 | 74.83 | 79.91 | 75.62 | 72.59 | 35.10 |
| 0.7 times the load | 94.25 | 85.65 | +8.60 ± 0.07 | 85.78 | 94.38 | 86.16 | 84.01 | 94.37 |
| 0.85 times the load | 91.98 | 85.56 | +6.42 ± 0.17 | 85.82 | 92.09 | 86.15 | 83.98 | 53.66 |
| 1.15 times the load | 86.37 | 85.41 | +0.95 ± 0.07 | 85.58 | 86.63 | 85.93 | 83.87 | 34.30 |
| 1.3 times the load | 83.75 | 83.53 | +0.21 ± 0.20 | 83.11 | 83.88 | 81.45 | 83.41 | 27.07 |

The learned limiter is ahead of their limiter in every situation, by more
than two standard errors in twelve of the thirteen; at 1.3 times the load,
where the service runs at 88% before any retry, the two are level. Against
their limiter at its best it is ahead in all thirteen (+0.63 ± 0.25 at 1.3
times the load): tuning theirs gains 0.15 points on the selection episodes,
with `alpha` staying at the file's 0.1, and `alpha` = 0.5 loses 14.

Their rule lets through a tenth of the goodput as retries whatever the load,
and holds 85% to 86% from 0.7 to 1.15 times the file's load. The learned
limiter follows the load, which it has to read off its own counts. Shares it
gives while calls fail, as retries per first attempt of the tenant, in the
tenants' order (four immediate attempts; backoff; circuit breaker; client
budget):

| Load | The shares | What that is |
| --- | --- | --- |
| 0.7 | 2.9, 2.9, 3.0, 2.9 | nothing refused: the service has room, and the result is the one with no limiter (94.37%), which their limiter costs 8.7 points |
| 1.0 (their file) | 0.27, 0.52, 1.89, 2.77 | the two tenants that retry without restraint are held to a quarter and a half of their first attempts; the two that restrain themselves get what they ask |
| 1.15 | 0.09, 0.03, 0.31, 1.98 | the same, tighter |
| 1.3 | 0, 0.01, 0, 0.35 | next to nothing, and outside the failure nothing |

(Means over the seconds of the failure, six selection episodes.)

Nobody is let down for it. By tenant, on their file: 86.7%, 90.8%, 87.2% and
89.5% behind the learned limiter against 85.5%, 85.6%, 85.3% and 85.7% behind
theirs; on the varied situation 89.6%, 90.3%, 88.4% and 91.0% against 87.6%,
87.8%, 87.4% and 88.1%.

No fixed quota does this: the best single one (10% for everybody) is 0.3
points ahead of their limiter on the varied situation and 2.1 behind it at 1.3
times the load. The learned limiter is within 0.3 points of the best fixed
quota for each situation, which was chosen knowing the situation.

The row does not hang on the one model. Each of the 40 models shortlisted (of
60 scored, from 20 runs) is ahead of their limiter on their file (86.94% to
88.79%) and on the varied situation (89.24% to 90.11%), and 13 of the 20
runs' final models were ahead in every situation scored on the selection
episodes. The one reported was chosen with `--select lead` on 64 selection
episodes (largest smallest lead over the thirteen situations). It is seed 100
of

```bash
python -m examples.rl_vs_arolla train --arolla ~/Arolla --out study/runs/lc-100 --seed 100 \
    --timesteps 1500000 --gamma 0.5 --gae-lambda 0.9 --ent-coef 0.003 --slow-s 10 \
    --train-load 0.6 1.4 --lr-decay --snapshots 5
```

an hour on twelve simulations and one learner.

**The rule their prototype deploys.** "Arolla" above is the limiter of their
simulator files (`arolla_retry_budget`: per tenant, the retries of a window
against a share of a smoothed goodput). The limiter their paper evaluates on
Kubernetes is another rule (`prototype/arolla-filter`, their Algorithm 1): ONE
bucket for everybody, full at the start, a tenth of a token per success and
one token per second, a retry takes one. `admission.MixedRefillBudget` is that
filter, and `as_written(..., limiter=)` runs it in the same gate in front of
the same clients: as their manifests deploy it (`r` 0.1, capacity 20, `t` 1),
with their fairness filter (a fourth attempt costs six tokens), and at the
best of the 125 settings their own grid walks on their testbed, chosen on the
selection episodes. Same test episodes, same learned limiter:

| Situation | Learned limiter | Simulator's rule | Prototype's rule, as deployed | with fairness filter | Prototype's rule, tuned | Learned - as deployed | Learned - tuned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Their file | 88.57 | 85.54 | 85.47 | 85.46 | 85.89 | +3.10 ± 0.12 | +2.68 ± 0.12 |
| Varied | 90.09 | 87.73 | 87.77 | 87.74 | 88.14 | +2.32 ± 0.31 | +1.95 ± 0.29 |
| Failure rate 0.1 | 99.60 | 98.70 | 99.59 | 99.53 | 99.39 | +0.01 ± 0.01 | +0.22 ± 0.01 |
| Failure rate 0.25 | 97.86 | 94.03 | 94.28 | 94.28 | 94.54 | +3.58 ± 0.12 | +3.32 ± 0.12 |
| Failure rate 0.75 | 78.96 | 77.21 | 77.19 | 77.19 | 77.53 | +1.77 ± 0.07 | +1.44 ± 0.07 |
| Failure rate 0.9 | 73.12 | 72.44 | 72.41 | 72.38 | 72.57 | +0.71 ± 0.04 | +0.55 ± 0.04 |
| Failing for 5 s | 97.15 | 96.45 | 96.48 | 96.43 | 96.50 | +0.67 ± 0.05 | +0.64 ± 0.05 |
| Failing for 10 s | 94.32 | 92.84 | 92.81 | 92.76 | 92.96 | +1.51 ± 0.08 | +1.36 ± 0.08 |
| Failing for 35 s | 80.08 | 74.57 | 74.51 | 74.55 | 75.28 | +5.57 ± 0.14 | +4.80 ± 0.14 |
| 0.7 times the load | 94.25 | 85.65 | 85.71 | 85.61 | 86.38 | +8.54 ± 0.06 | +7.86 ± 0.06 |
| 0.85 times the load | 91.98 | 85.56 | 85.55 | 85.54 | 86.19 | +6.43 ± 0.17 | +5.79 ± 0.17 |
| 1.15 times the load | 86.37 | 85.41 | 85.41 | 85.42 | 85.74 | +0.95 ± 0.07 | +0.63 ± 0.06 |
| 1.3 times the load | 83.75 | 83.53 | 75.14 | 75.66 | 80.69 | +8.61 ± 1.20 | +3.06 ± 0.65 |

Up to 1.15 times the load the two rules of theirs are the same limiter to
within 0.3 points (0.9 at a failure rate of 0.1, which the bucket's refill
just about covers), and the learned one leads both, except at that failure
rate, where it and the bucket are level (+0.01 ± 0.01). At 1.3 times
the load they part: the bucket keeps paying out a tenth of the successes and
a retry a second into a service that has no room for either, and ends 8.3
points BELOW refusing every retry (83.41%), where their simulator's rule holds
83.53%. A plain success-funded budget does the same there (`policy-rb`:
74.67%), which is what the bucket is without its time refill. The fairness
filter changes little here: these tenants stop at four attempts.

As to whether its settings are fitted to their testbed: on this situation,
which they were not chosen on, the deployed settings are 0.35 points from the
best of their own grid (86.19% against 86.54% on the selection episodes,
at `r` 0.05, capacity 5, `t` 10), and no setting of the grid saves the rule at
1.3 times the load (80.69% at best). So what does not carry over is not the
choice of `r`, `t` and the capacity. It is that the rule admits in proportion
to what succeeds and has no notion of how full the service is.

**What it does not show.** Anything about their prototype itself (the next
section is about that): this is their simulator's situation in this
simulator, and both of their rules are `msim.admission`'s ports (the
simulator's rule reproduces their simulator within noise on all their
scenarios, [compat-arolla](compat-arolla.md); the prototype's rule is ported
from the filter's source and has not been run against the filter). That
the learned limiter carries over: it was trained on this service, 16 workers
and 60 ms, and on this kind of failure, and what it has learned about load is
this service's capacity. Their rule knows nothing about the service and needs
no training; on another service the learned limiter has to be trained again,
and on a fault it has not seen (a slowdown, a load spike) it has not been
tried. And the reward was the pooled goodput: a limiter judged on something
else (the latency of what succeeds, one tenant's share) is another agent.

### On their Kubernetes prototype (`deploy/envoy-limiter`)

Everything above is one simulator against a port of their rules. This is their
prototype itself: Online Boutique behind Istio, their manifests, their load
generator, their fault manifests, their harness (`run-experiment.sh`) and their
analysis, with their Wasm filter as they build it, on another cluster than
theirs. The learned limiter is a fifth policy of their harness: the same two
`WasmPlugin`s as `arolla.yaml` and `arolla-gateway.yaml` (every sidecar of the
namespace, and the gateway; same phase, same fail strategy, same headers, a 429
for a refused retry), with another module and another configuration. Nothing
else differs between the two arms of a cell: same pods restarted the same way,
same clients, same fault.

**The deployment.** Kubernetes 1.36 with Cilium, three nodes for the application
and one for the load generators, Istio 1.27.5, Online Boutique v0.10.4 with
their `2-replica` profile. One thing of theirs was changed for every policy
alike: the frontend's CPU limit, 1 core instead of 4. On these nodes their
profile serves 7000 `POST /cart` per second (5000 with a Wasm filter in every
proxy), and their headline load of 1200 is no stress at all: with a ten-second
outage of the cart service, no control, circuit breaker, retry budget and
Arolla all recover within three seconds. With the frontend at one core the
application serves 2000 per second cleanly and saturates near 3000, so their
loads stress it.

**What the benchmarks are.** Their artifact reproduces three figures: the
effectiveness of the four policies under a sustained failure at 1200 requests
per second (Figure 4: success rate, latency, retries received), the recovery
time across load, failure rate and fault duration (Figure 5), and the fairness
of retry admission across clients of different aggressiveness (Figure 6).
Each cell below is one run of their harness with their phases (30 s warm-up,
60 s before the fault, the fault, 60 s after, 10 s cool-down; their sweeps
watch 100 to 200 s after the fault), one run per cell where their paper
aggregates ten.

**The limiter that was deployed.** `deploy/envoy-limiter`: the quota of
`TenantRetryQuota` and the counts of `rl.limiter` as a Proxy-Wasm filter, the
trained policy as weights in its configuration, evaluated once a second in the
proxy. It was trained on a model of this deployment and of nothing else: one
station whose capacity is 1.5 times a nominal rate (3000 against 2000 requests
per second, as measured), in front of it their load generator's client (four
attempts, no backoff, a 3 s timeout), the fault a share of failing attempts.
An episode draws the load (0.2 to 1.1 of the nominal rate), the share (0.1 to
1), the length (5 to 35 s) and the capacity (0.7 to 1.4 of the measured one:
a ramp measures it to within some tens of percent, and a policy trained on one
capacity knows it by heart). PPO as above with a discount of 0.97, the slow
averages of 10 s, and a reward that counts only the requests served within
half a second (`reward_within_s`); 700000 steps, twelve simulations. The model
was chosen on the simulator alone, among the 25 models of that wave (five
runs, final and copies on the way): the largest SMALLEST lead over the better
of "no retry let in" and "every retry let in" across 36 cells (three
capacities, failures of 10 and 20 s, three loads, half and all attempts
failing), the mean deciding between leads closer than the noise. Its
configuration gives every proxy the nominal rate it serves (1000 requests per
second: two gateway pods, two frontend pods). Every run made on the cluster is
reported here, the discarded ones too. The harness side (their repository as
it was pointed at this cluster, the profiles at other rates, the fault
manifests at 25% and 75%) is not in this repository; what it needs from here
is `deploy/envoy-limiter` and the two reward options.

Four things had to be learned on the way, each of which a simulator-only study
would not have shown:

- **A short discount learns to flood.** The discount that won on their
  simulator file (0.5) learns here to let every retry in: on a service with a
  3 s timeout the queue a storm fills pays its bill seconds later. 0.95 to
  0.97 learns to hold back.
- **Goodput alone learns to queue.** With no limiter at all this deployment
  serves MORE requests than with Arolla (97.7% against 91.8% over the run of
  their headline experiment): the generator's requests wait up to 3 s in the
  frontend and are served when the failure ends, at a latency of seconds. A
  limiter paid for goodput learns exactly that. The first model deployed did
  (with its counts still per thread, the next point), and lost to Arolla on
  what their figures measure: 91.9% of requests but a p99 of 3.7 s and 34 s to
  recover, where Arolla has 9 ms and 3 s. Hence
  `reward_within_s`: the model reported was paid only for requests served
  within half a second.
- **An Envoy thread is not a proxy.** A filter instance lives in one worker
  thread, and Envoy does not spread connections evenly over its threads: one
  thread of a gateway pod was seen with 95% of the traffic. A policy that reads
  the load off its own counts then sees twice the load in one thread and none
  in the other. The threads of a proxy now pool their counts (shared data) and
  decide on the sum.
- **A refusal must not cost the caller its connection.** Both filters answer a
  refused retry at the request headers, and Envoy closes an HTTP/1 connection
  it answers before the body has arrived. Under a total failure that is 3500
  closed connections a second. After 30 s of it part of the generator's new
  connections hang for a while; Arolla, whose bucket refills slowly, rides that
  out with a few dips, and the learned limiter, which reopens its quota within
  two seconds of the first successes, fed it for half a minute (70% of the
  requests after the failure, 46 s to recover). The filter now holds such an
  answer for 2 to 4 ms, until the request is complete, and the connection
  survives: 99.5% after the same failure. Both are reported below, because the
  second is a gain of the filter and not of the policy.

**What a cell reports.** From the load generator's own records: the share of
the requests of the measured run that succeeded (a request has succeeded when
one of its attempts did), the same for the requests that started while the
fault was on, and the p99 latency of the attempts that succeeded. From their
analysis: the recovery time, in whole seconds from the end of the fault to the
first second at 95% of the goodput before it (it includes the second or two
the fault's removal takes to reach the proxies).

Their headline cell (Figure 4): 1200 requests per second, the cart service
failing every call for 10 s.

| Policy | Succeeded % | Started in the fault % | p99 ms | Recovery s |
| --- | --- | --- | --- | --- |
| No control | 97.69 | 56.2 | 2259 | 4 |
| Circuit breaker | 98.08 | 59.9 | 2157 | 3 |
| Envoy retry budget | 98.33 | 65.4 | 2453 | 4 |
| Arolla (four runs) | 92.18 | 7.4 | 9 | 2 to 3 |
| Learned limiter | 92.56 | 3.5 | 9 | 2 |

The three policies without a limiter in the proxies serve MORE requests than
either limiter, late: the frontend queues what the generator sends for up to
the generator's 3 s timeout and serves it when the cart service is back.
Nothing here is metastable (the generator's connection pool bounds what is in
flight), so the failure their headline figure is about, a storm that outlasts
the fault, is not this deployment's. Between the two limiters the cell is a tie, in the run
that had them back to back (92.57% and 92.56%, p99 9.4 ms both, frontend p50
4.0 ms both in their analysis, 2 s to recover both). On their third panel
Arolla is ahead: of some 18000 retries that arrived at each gateway pod during
the outage, 103 reached the frontend pod their analysis samples under Arolla
and 568 under the learned limiter. It holds its quota at 5% through a total
failure, where nothing in its reward made it hold it at zero (`retry_cost` is
there for that; the models trained with it did not pass the selection). And a
retry it refuses is answered in 4 ms where Arolla's is answered in 1 ms, which
is the price of the caller's connection (below).

Recovery across their three axes (Figure 5), Arolla and the learned limiter.
"Their filter's refusal" is the learned policy in a filter that answers a
refused retry at once, as Arolla's does; "learned limiter" is the same policy
in the filter that holds the answer until the request is complete.

| Cell | | Succeeded % | Started in the fault % | After the fault % | p99 ms | Recovery s |
| --- | --- | --- | --- | --- | --- | --- |
| **Load**, total failure of 10 s: 600 req/s | Arolla | 92.55 | 5.5 | 99.6 | 9 | 2 |
| | Learned limiter | 92.48 | 2.2 | 100.0 | 9 | 2 |
| 1200 req/s | Arolla | 92.57 | 4.7 | 99.8 | 9 | 2 |
| | Learned limiter | 92.56 | 3.5 | 100.0 | 9 | 2 |
| 1800 req/s | Arolla | 92.68 | 6.8 | 99.7 | 74 | 3 |
| | Learned limiter | 92.66 | 5.3 | 99.9 | 131 | 3 |
| **Failure rate** at 1200 req/s, 10 s: 25% | Arolla | 98.59 | 82.0 | 99.9 | 8 | 1 |
| | Learned limiter | 99.79 | 97.3 | 100.0 | 9 | 0 |
| 50% | Arolla | 96.52 | 56.3 | 99.7 | 9 | 2 |
| | Learned limiter | 97.69 | 75.0 | 99.6 | 349 | 3 |
| | with their filter's refusal | 97.20 | 66.0 | 99.6 | 9 | 2 |
| 75% | Arolla | 93.90 | 30.7 | 99.5 | 9 | 3 |
| | Learned limiter | 94.14 | 33.3 | 99.5 | 9 | 3 |
| **Duration** of a total failure at 1200 req/s: 30 s | Arolla | 77.80 | 1.8 | 93.5 | 12 | 3 |
| | Learned limiter | 80.15 | 1.8 | 99.5 | 9 | 3 |
| | with their filter's refusal | 68.51 | 1.4 | 70.1 | 2556 | 46 |
| 60 s | Arolla | 67.38 | 1.1 | 82.6 | 2112 | 19 |
| | Learned limiter | 66.75 | 1.0 | 99.3 | 8 | 3 |
| | with their filter's refusal | 67.78 | 3.8 | 84.3 | 2568 | 26 |

At a total failure of 10 s nothing separates the two at any load: no retry can
succeed, both refuse nearly all of them, and both are back within the 2 to 3 s
the removal of the fault takes. Where retries can succeed, the learned limiter
lets more of them in and more requests succeed, by 1 point of the run at a
quarter of the calls failing (97% against 82% of the requests that met the
fault, and no second below 95%) down to a quarter of a point at three quarters.
At 50% the run with the final filter let in more than the service had room
for on one of the two gateway pods (a quota of 100% there, 30% on the other):
three quarters of the requests that met the fault succeeded, the tail latency
rose to 350 ms, where Arolla's stays at 9 ms, and by their count it took a
second longer to recover.

The long failures are the filter's doing, in both directions. After 60 s of
refusing 3500 retries a second with a closed connection each, Arolla's
deployment needs 19 s to recover and serves 83% of the requests of the minute
after (the generator's connections hang; its 67.4% of the run is higher than
the learned limiter's 66.8% only because requests the stalled generator sent
late succeeded late). The learned policy behind the same kind of refusal does
no better at 60 s and worse at 30 s: it reopens its quota within two seconds
of the first successes and feeds the trouble. With the refusal held until the
request is complete there is no trouble to feed: 99.3% and 99.5% after the
failure, 3 s to recover.

A failure of half the calls for 20 s, across the load (the second number is
the requests that started during the fault):

| Load | No control | Arolla | Learned limiter |
| --- | --- | --- | --- |
| 600 req/s | 99.17 (94.2), p99 8 ms | 93.00 (54.7), p99 10 ms | 99.06 (93.6), p99 8 ms |
| 1200 req/s | 99.24 (94.7), p99 1099 ms | 93.13 (54.6), p99 8 to 70 ms | 94.98 (66.8), p99 8 ms |
| 2000 req/s | 99.49 (95.4), p99 2711 ms | 93.15 (52.4), p99 344 ms | 93.24 (53.3), p99 297 ms |

This is what the limiter was trained to do and Arolla's rule cannot: at 600
requests per second there is room for every retry, and it lets them all in
(the quota at 100% before the fault starts) and matches no control at no
control's latency; at 1200 it holds 30%; at 2000 it tries 30% for a few
seconds and falls back to 5%, which is about what Arolla admits.
Arolla admits a tenth of the successes plus one a second at every load. (The
600 and 2000 rows of the learned limiter are from the filter with their kind
of refusal; the 1200 row was run with both and gave 94.96 and 94.98.)

Fairness (Figure 6): their six clients, three that retry three times and three
that retry ten times ("polite" and "aggressive"), all at 200 requests per
second or at 50 and 350, half the calls failing for 20 s. The learned limiter
here is the SAME one-tenant model, asked once per client with that client's
counts; nothing was trained for six tenants. Their figure counts the retries
admitted to each group during the fault; share over demand is a group's share
of them over its share of the requests (1 is proportional).

| Run | Limiter | Retries admitted, polite / aggressive | Share over demand | Succeeded in the fault %, polite / aggressive | Succeeded % |
| --- | --- | --- | --- | --- | --- |
| Same rate | Arolla | 548 / 750 | 0.84 / 1.16 | 55.1 / 55.7 | 93.04 |
| | Arolla, fairness filter | 665 / 575 | 1.07 / 0.93 | 55.1 / 54.4 | 92.97 |
| | Learned limiter | 3467 / 1814 | 1.31 / 0.69 | 65.9 / 60.3 | 94.34 |
| Different rates | Arolla | 84 / 1190 | 0.54 / 1.06 | 54.4 / 54.8 | 92.97 |
| | Arolla, fairness filter | 164 / 1067 | 1.10 / 0.99 | 55.0 / 56.7 | 93.39 |
| | Learned limiter | 902 / 2990 | 1.91 / 0.87 | 68.6 / 59.5 | 94.11 |

Both groups get more retries and more of their requests through under the
learned limiter than under either of theirs, and the polite clients are not
crowded out, which is what their fairness filter is for. It is not the most
proportional of the three: each client's quota is set from that client's own
counts, a client that asks for ten retries per request reads as trouble, and
its quota is set lower, so the polite clients end with up to twice their share
where plain Arolla gives them half and the fairness filter very nearly their
own.

**A second model.** Two more training waves followed the one the deployed
model is from (one twice as long; one with a price on retries and the capacity
drawn within 15% of the measured one), and the 90 models of the three were
ranked by a stricter rule than the first: never behind their prototype's rule
in any simulated cell, no more retries than it at a total failure, then the
largest mean lead. None passes it, the deployed model included (below). The model it ranks first
lets nearly every retry in at low and middle load and none at high load. It
was run on five cells with the same filter (succeeded %, in brackets the
requests that started during the fault, and the p99):

| Cell | Arolla | Learned limiter | Second model |
| --- | --- | --- | --- |
| 600 req/s, half failing for 20 s | 93.00 (54.7), 10 ms | 99.06 (93.6), 8 ms | 99.03 (93.2), 9 ms |
| 1200 req/s, half failing for 20 s | 93.13 (54.6), 8 to 70 ms | 94.98 (66.8), 8 ms | 97.03 (79.2), 924 ms |
| 1200 req/s, half failing for 10 s | 96.52 (56.3), 9 ms | 97.69 (75.0), 349 ms | 98.89 (85.4), 1250 ms |
| 2000 req/s, half failing for 20 s | 93.15 (52.4), 344 ms | 93.24 (53.3), 297 ms | 91.87 (43.1), 403 ms |
| 1200 req/s, all failing for 10 s | 92.57 (4.7), 9 ms | 92.56 (3.5), 9 ms | 92.52 (3.1), 8 ms |

It buys its goodput at 1200 requests per second with a tail latency of a
second, and at 2000 it is behind Arolla, which still admits its tenth where
the model admits nothing. The bolder of the two, not the better.

**What it shows.** On their benchmarks, on a deployment that is not theirs:
the learned limiter is level with Arolla where no retry can succeed (their
headline cell, the load axis), ahead wherever some can (the failure-rate axis,
by more the fewer calls fail, and every cell with half the calls failing below
2000 requests per second), ahead on recovery after a long failure (3 s against
19 s at 60 s) because of how its filter refuses, and it serves both groups of
their fairness runs better while leaning towards the polite one. Arolla is
ahead on the retries that reach the service during a total failure (a fifth
as many), and the fairness variant of their filter is the more proportional.
As to whether Arolla was fitted to its testbed: nothing here says its settings
were. It does on this cluster what it is built to do, keep a failing
dependency from costing latency (a p99 of 9 ms through the outage, against
more than 2 s for the three policies with no limiter in the proxies), with the
settings of their manifests. What does not carry over is the premise of their
headline figure, that unlimited retries do not recover: with a load generator
whose connection pool is bounded they are not metastable here, and no control
ends with MORE requests served than Arolla, late. And two properties of their
filter cost it on this cluster where they presumably do not on theirs: it
admits the same tenth at every load, and it answers a refused retry by closing
the caller's connection.

**What it does not show.** One run per cell, where their paper aggregates ten:
Arolla's four runs of the headline cell spread over 0.8 points (91.79% to
92.57%), so a difference below a point of the run between two cells run at
different times is not a finding, and the differences above are the ones in
the requests that met the fault and in what follows it. One deployment, with
the frontend capped to make their loads stress it; uncapped, this cluster
saturates at the frontend's proxy, far above their loads, and that was only
ramped, not run. The model was trained for this deployment's capacity
and nominal rate, which Arolla's rule does not need, and in the simulator it
falls behind Arolla's rule in one corner it was not run in on the cluster: a
capacity a fifth below the measured one at the full nominal rate with part of
the calls failing (by 2 points at 10 s, 16 to 19 at 20 s; ahead or level in the
other 68 cells of that grid). The fixed 10%
quota the filter gives retries that name no tenant (calls inside the mesh) was
never exercised: their faults are injected in the caller's proxy, and the
gateway does not retry a 500. And the two changes to the filter that the long
failures and the uneven threads forced (holding a refusal, pooling the counts)
are the filter's, not the policy's: Arolla's rule in such a filter would gain
from the first as well.

### The side study: a learned policy on the clients, with no limiter

The first version of this study put the agent on the other side. It takes the
server-side policy away (`policy-none`) and gives all four tenants ONE learned
retry policy (`--agent clients`). That changes the clients, so it says what
coordinated clients can reach and not which limiter is better: the
comparison above is the one of limiters. What it found, and what it took to
find it, is kept here.

**What is compared.** The agent sets max attempts (1 to 4), backoff (0 to
100 ms) and the retry budget (5% to 100% of the requests) of all four
tenants alike, every second; each tenant keeps the attempt timeout its file
gives it. Their rows are their files as written, behind each server-side
policy. Situations: their file as it is; a varied one, where every episode
draws the failure rate (0.1 to 0.9), its start and length (5 to 35 s) and the
load (0.7 to 1.3 times the file's); and each of those three axes walked alone.
For a seed every row faces the same failure and the same arrivals, so the
differences are paired. Whatever is chosen is chosen on other episodes than
the ones reported: the fixed setting, the trained model, and the `alpha` of
their limiter (their file's own 0.1 turned out to be its best of 0.01 to 0.5
on the varied situation; 0.5 collapses above the file's load).

**Result.** Success of all requests in %, on 64 episodes that nothing had been
chosen on (seeds 3000 to 3063), and the paired difference to their clients
behind Arolla's limiter with its standard error:

| Situation | Learned policy | Arolla | Learned - Arolla | Best fixed setting for the situation | One fixed setting | No retries |
| --- | --- | --- | --- | --- | --- | --- |
| Their file | 89.00 | 85.54 | +3.46 ± 0.10 | 88.78 | 86.33 | 84.00 |
| Varied | 88.78 | 86.57 | +2.21 ± 0.41 | | 86.49 | 84.95 |
| Failure rate 0.1 | 99.79 | 98.69 | +1.11 ± 0.02 | 100.00 | 99.61 | 96.15 |
| Failure rate 0.25 | 98.47 | 94.02 | +4.45 ± 0.08 | 98.68 | 94.94 | 91.58 |
| Failure rate 0.75 | 78.97 | 77.28 | +1.69 ± 0.06 | 78.91 | 77.76 | 76.40 |
| Failure rate 0.9 | 73.19 | 72.45 | +0.74 ± 0.03 | 73.20 | 72.71 | 71.81 |
| Failing for 5 s | 97.28 | 96.46 | +0.82 ± 0.06 | 97.10 | 96.79 | 95.42 |
| Failing for 10 s | 94.53 | 92.84 | +1.69 ± 0.08 | 94.30 | 93.32 | 91.65 |
| Failing for 35 s | 80.51 | 74.54 | +5.97 ± 0.12 | 80.41 | 75.82 | 72.52 |
| 0.7 times the load | 97.35 | 85.67 | +11.69 ± 0.10 | 96.64 | 86.65 | 84.02 |
| 0.85 times the load | 92.36 | 85.51 | +6.84 ± 0.16 | 92.03 | 86.40 | 84.05 |
| 1.15 times the load | 85.90 | 85.44 | +0.46 ± 0.12 | 85.59 | 85.53 | 83.86 |
| 1.3 times the load | 83.34 | 82.96 | +0.38 ± 0.30 | 83.34 | 72.21 | 83.34 |

The learned policy is ahead in every situation, by more than two standard
errors in twelve of the thirteen. What it has learned is to follow the load:
at 0.7 times the file's load it retries freely (1.27 attempts per request),
at the file's load it allows two attempts under a budget of about a third
while calls fail, and at 1.3 times the load, where the service runs at 88%
without a single retry, it sends exactly one attempt per request. There its row IS
the row of never retrying, and any retry budget is a loss (0.5% of the
requests costs 0.45 ± 0.21 points). It does as well as the best fixed setting
for each situation, which was chosen knowing the situation, and with fewer
attempts than their clients issue: 1.11 per request on their file against
1.29, most of which their limiter refuses.

No fixed setting does this. The best single one (two attempts, 10 ms, a 10%
budget) is level with their limiter on the varied situation (-0.09 ± 0.33) and
10.8 points behind it at 1.3 times the load. Their limiter holds 85% to 86%
at every load up to 1.15: it protects the service and, at 0.7 times the load,
costs its own clients the 8.7 points they get with nothing in front of them
(94.39%).

The row does not hang on the one model: each of the 40 models shortlisted
(from 212 scored, of 60 runs) beats their limiter on their file (88.07% to
89.33%) and on the varied situation (88.56% to 89.05%). The one reported was
chosen with `--select lead` on 64 selection episodes: the largest smallest
lead over the thirteen situations. It is seed 400 of

```bash
python -m examples.rl_vs_arolla train --agent clients --arolla ~/Arolla --out study/runs/w-400 \
    --seed 400 --timesteps 2000000 --gamma 0.7 --gae-lambda 0.9 --slow-s 10 --train-load 0.6 1.4 \
    --snapshots 7
```

after 1.75 million steps (an hour on twelve simulations and one learner).

**What it does not show.** Which limiter is better: the two sides differ in
their clients, not only in what stands in front of the service. Nor that their
limiter is unnecessary: the agent sets the policy of EVERY caller, and a
server-side limiter is there for the callers nobody controls. One tenant that
keeps retrying freely is not in this comparison. Nor anything about their
prototype: this is their simulator's
situation in this simulator, with Arolla's rule as `msim.admission` implements
it from their files. And the agent has one setting for four tenants, although
the tenant without an attempt timeout gains from retries at a load where the
others lose.

**What had to be right first.** Three things about the environment, none of
them about the learner, each worth more than any learner setting:

- *The agent has to see its load.* With the request rate measured against
  the episode's own drawn load, every load read 0.5 (fixed in
  `ArollaScenario.build`), and the best retry setting depends on the load
  more than on anything else.
- *The warm-up runs on the default setting.* The grid taken from a file
  defaults to the file's own clients, four attempts with nothing in the way:
  from 1.15 times the load on, every episode began in a retry storm that no
  policy had chosen, and at that load a backlog takes most of an episode to
  drain. Clients that never retry scored 80.7% at 1.3 times the load through
  the environment and 83.3% once the warm-up did not retry either. A scenario
  for training should default to a setting that leaves the system standing.
  With this and the first point wrong, the first agents still beat their
  limiter on their file by 1.9 points, reading the load off the latency, and
  were 5.1 behind it at 1.3 times the load.
- *One second is a noisy look at the load.* `slow_s=10` (the measurements
  again, averaged over about ten seconds) is what lets a policy without
  memory retry freely at 0.7 times the load (97% instead of 96%) and learn
  in a sixth of the steps what it otherwise needs a full run for.

About the learner: a short horizon works best (discount 0.5 to 0.8 against
the usual 0.99: what a setting does is over within seconds), and the reward
offset of the training section is what makes the value function fit at all.

Training on loads drawn a little past both ends of the comparison's range
(`--train-load 0.6 1.4`), for two million steps, is what settled the top end:
at 1.3 times the load a policy trained on 0.7 to 1.3 stops retrying while
calls fail and still retries a little in between, which is free on a calm
second and costs a point over an episode. Taking the entropy bonus away does
not change that.
