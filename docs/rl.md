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
holds through the warm-up.

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
thing that differs is the controller inside that gate: their rule, or a quota
per tenant that an agent sets every second (`ArollaLimiterScenario`, above).
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

**What it does not show.** Anything about their prototype: this is their
simulator's situation in this simulator, with Arolla's rule as
`msim.admission` implements it from their files (their numbers are reproduced
within noise on all their scenarios: [compat-arolla](compat-arolla.md)). That
the learned limiter carries over: it was trained on this service, 16 workers
and 60 ms, and on this kind of failure, and what it has learned about load is
this service's capacity. Their rule knows nothing about the service and needs
no training; on another service the learned limiter has to be trained again,
and on a fault it has not seen (a slowdown, a load spike) it has not been
tried. And the reward was the pooled goodput: a limiter judged on something
else (the latency of what succeeds, one tenant's share) is another agent.

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
