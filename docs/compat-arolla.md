# Running Arolla's scenarios (`msim.compat.arolla`)

[Arolla](https://github.com/eth-easl/Arolla) (ETH EASL) ships a simulator that
began as a copy of this one, and a set of YAML scenarios for it. This package
reads those files and builds the msim objects that mean the same thing, so their
experiments can be run, checked and extended here.

It is a reader, not a second engine: everything a scenario needs at run time is
msim's own -- client-side timeouts ([request-handling](request-handling.md#what-the-server-learns-about-the-deadline)),
admission control ([request-handling](request-handling.md#admission-control-admissionpy)),
request contexts -- and lives in the core package. What is specific to Arolla's
file format and to its client-side retry controls lives here, behind the
`arolla` extra (PyYAML).

```python
from msim.compat.arolla import build, parse, read

for point in read("experiments/yaml/snowflake/policy-arolla.yaml"):
    scenario = build(parse(point.config))
    scenario.run()
    for client in scenario.clients:
        print(point.label, client.name, client.runtime.metrics().summary().succeeded)
```

| Module | What it does |
| --- | --- |
| `sweeps` | one file is one scenario per point of its inline sweeps (`expand`) |
| `schema` | their schema as dataclasses, read strictly (`parse`) |
| `importer` | the mapping to stations, clients, gates and policies (`build`, `Scenario`) |
| `policies` | their client-side retry controls as msim policy layers |
| `compare` | one scenario on both simulators over many seeds, side by side |

## Reading a file

**Sweeps.** A list where one value belongs (`p_fail: [0.0, 0.05, 0.1]`) is a
sweep dimension; a list written once and referenced through a YAML anchor is
one dimension that all its places follow together. `read(path)` returns one
`SweepPoint` per combination, in the order of their runner (dimensions as the
file meets them, the last varying fastest), so point `i` here is point `i`
there. Only lists under `services`, `clients` and `workload` are dimensions.

**Schema.** `parse(point.config)` validates one concrete scenario and returns an
`Experiment`. It accepts what their schema accepts and means, and is stricter
where theirs lets a mistake through:

- a key the schema does not know is an error naming its path
  (`services[0].querue_capacity: unknown key`); theirs ignores it;
- ranges are checked (`p_fail` in [0, 1], `end_s` after `start_s`, ...);
- a service with more than one limiter is an error; theirs keeps one by a fixed
  priority and silently drops the rest;
- what their runner reads and then ignores is accepted and listed in
  `Experiment.warnings`: a `sweeps:` block, a service's `timeout.global_ms` that
  no client reads, a circuit breaker or budget on a client without a `retry`
  policy, a `count_based` breaker on a client.

## What a scenario becomes

| In the file | In msim |
| --- | --- |
| a service | a station (`ServiceRuntime`): latency, workers, queue, faults |
| a service's limiter (`arolla_retry_budget`, `aimd_global_retry_budget`, `global_retry_budget`, `retry_budget`, a `time_based` `circuit_breaker`) | one admission controller (`GoodputCoupledBudget`, `AimdBudget`, `TokenBucketBudget`, `SuccessFundedBudget`, `FailureRateGate`). It refuses the retries the service is **sent** (a call whose caller is on its second or later attempt): an `AdmissionGate` in front of a plain service, an `AdmittingLoop` where the service runs a loop of its own. And it gates the retries the service **makes**: a `ControlledRetryPolicy` inside that loop, which also reports every attempt to it |
| a `count_based` or `retry_circuit_breaker` breaker, or a `rate_limiter`, on a service | a layer of the service's own retry policy and nothing else (`CountRetryBreaker`, `WindowRetryGate`, `LeakyRetryGate`, msim's bursty and fixed-window limiters): their runner never applies these on arrival |
| a client entry with `replicas: n` | `n` clients named `name.i`, each its own tenant |
| a client **with** a `retry` policy | a client that runs it: backoff, wrapped in its `circuit_breaker` (`WindowRetryGate` for `retry_circuit_breaker`, `ThreeStateRetryBreaker` for `time_based`), wrapped in its `retry_budget` (`FirstAttemptFundedBudget`; one `RetryBank` per `shared_budget_id`). It submits to the service's gate where there is one |
| ... its `timeout.attempt_ms` | a timer of the client's own: `server_deadline="global"`, so the attempts it gives up on live on at the server |
| ... no `timeout` at all | the service's `timeout` applies: its `global_ms` to the request, its `attempt_ms` to each call, enforced by the server |
| a client **without** a `retry` policy, and the single client of a top-level `workload` | a relay: a client that carries the SERVICE's retry policy and timeouts, tells the server its deadline, and submits straight to the station. The service's limiter gates those retries from inside the loop |
| a service with `dependencies` | a chain node that holds its worker, makes its calls first and works last (`local_work="last"`): `sequential` dependencies are a call wave each, `parallel` ones share a wave the service waits out (`wave_join="wait_all"`), an `optional` one is an optional edge |
| a call from one service to another | a call to the dependency's own retry **loop**: a client carrying the dependency's `retry` policy and `timeout.attempt_ms`, one per service and shared by its callers. The caller gets one answer per call, when the dependency has finished retrying |
| a client with a `retry` policy in front of a service that retries too, or that has an `attempt_ms` of its own | two loops on one hop: the client calls the service's loop through a `RetryLayer`, its policy and timer outside, the service's inside |
| a client's `e2e_retry_budget: n` | a retry allowance of `n` on every request the client starts (`ClientConfig.retry_allowance`), which every service's loop draws on (`RetryAllowancePolicy`). The client's own retries are not charged, as in their runner |
| a client with `timeout.attempt_ms` and no `retry` policy | a client that does not retry (`retry: {type: none}`), with a warning: one call, one result |
| `workload` | a `Workload` with its own arrival stream per client |

**Arrival streams.** Each client's stream is derived from the workload's
`rng_seed`, the rank of its entry among the entries that target the same
service, and its replica index (`arrival_seed`). Replicas and co-located tenants
therefore arrive independently, while arms laid out on separate services see
identical arrivals and stay paired.

**Run controls.** `build(experiment, seed=, arrivals=, duration_s=)` replaces
the simulator seed, the base of every arrival stream, and cuts every workload
short. `Scenario.run()` drives the workloads to the end and lets the system
drain. `Scenario.admission_counts(client)` is what the limiter of the service a
client calls counted of its calls -- the retries it sent and how many were let
in -- wherever the mapping put the door.

## Intended behaviour, not their bugs

The mapping reproduces what their simulator is meant to do. Where its code does
something else, the intent is kept here. The same defects are fixed on a branch
of their repository, which is what msim is compared against (below).

| Their code | Here |
| --- | --- |
| a dependency call completes on the dependency's first attempt, and again on each later one | once, when the dependency has finished retrying |
| a client with its own policy takes the service's first attempt as the call's result | the result of the call: the service's last attempt |
| the answer to an attempt the client timed out is processed as a new result | ignored (an orphan's answer goes nowhere) |
| all replicas of an entry, and all entries sharing a workload block, arrive at the same instants | one arrival stream per client |
| a retry-budget deposit is cut to whole hundredths of a retry | exact |
| a retry refused at the server is booked as a server failure | booked as `REJECTED` |
| a retry the client then does not issue stays charged to its budget | refunded |
| a shared client budget hands every client the first client's whole policy and ignores `min_retries_per_sec` | one balance, a backoff per client, the configured rate |
| a `count_based` breaker's thresholds are cut down to whole counts (0.29 of 100 is 28) | the ratio is compared as it is |
| a client with a timeout and no retry policy has the timer armed for the service's first attempt only | read as a client that does not retry, and reported |
| a bursty or fixed-window rate limiter that refuses a retry ends the run (a `TypeError`) | the retry is refused |
| the first request of every client arrives at t = 0 | an exact Poisson process |

One property of their admission rule is kept as it is, because it IS the rule:
`GoodputCoupledBudget` compares a count of retries with a fractional allowance,
so every tenant gets the first retry of each window while its allowance is
above zero.

## Coverage

All 44 scenario files Arolla ships are built, and so is the rest of what their
schema can say: the keys no shipped file uses (`e2e_retry_budget`, optional and
several sequential dependencies, the goodput budget's success-rate gate and
burst cap, latency injections, rate limiters, the count-based breaker) are
exercised by small scenarios in `tests/test_compat_arolla_importer.py`, and by
five scenario files in `tests/arolla_scenarios/` written so that both
simulators can be put to them (below). Nothing
valid raises `UnsupportedScenario` today; the class stays for whatever their
schema grows next, since a reader must refuse what it cannot build rather than
approximate it. Two things their schema accepts are refused outright, because
neither simulator implements them: decorrelated jitter, and a service with more
than one limiter.

## Comparing the two simulators

```bash
python -m msim.compat.arolla.compare SCENARIO.yaml --arolla ~/checkouts/Arolla --seeds 200 --jobs 32
```

runs every sweep point of the file on both simulators, `--seeds` times each with
a different seed for the simulator and for the arrival streams, and prints per
client group and per measure (success over the whole run and for the requests
that started in each fault window, attempts per request, median latency of
successes, retries asked and admitted at the server) the two means and their
difference in standard errors, marking a difference beyond `--threshold` (3).
`--point i` runs one sweep point, `--duration-s` cuts the workloads short for a
scenario too large to run whole, `--json` saves the rows. Each sweep point takes
its own block of seeds, so the points of a file share no randomness.

One difference is in the arrival process and is deliberate (the table above):
their workload sends every client's first request at exactly t = 0. With many
clients that is a burst at the start and one request more per client, enough to
show in cells whose standard error is tiny. `--reference-start poisson`
replaces, in memory, the reference's first arrival by a drawn one, so that
everything else can be compared.

The reference is imported from the checkout given with `--arolla` and needs what
that code needs (pydantic, numpy, pandas, PyYAML); it should be a checkout whose
known defects are fixed, or the comparison measures those. The tool is not part
of the test suite.

## How close the two simulators are

Every scenario file Arolla ships was put to both simulators: each sweep point
(183 scenarios), 200 seeds per engine and per point, msim at `958357c` against
their `main` with its defects fixed (the local branch `Arolla-PR`, at
`57f9ae8`) and its first arrival drawn (`--reference-start poisson`). A cell is
one measure of one client group at one sweep point.

| Scenario file | Points | Seeds | Cells | Worst cell (SE) | Beyond 3 SE → on fresh seeds | Their fixes that change it | Largest shift in success, shipped → fixed |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `dependency/diamond` | 1 | 200 | 4 | 1.51 |  | 1, 2, 3 | `user` [20-40s]: 47.5 → 96.8 % |
| `diversity/N10_M1` | 1 | 200 | 4 | 1.72 |  | 4, 5 | none above 0.3 points |
| `diversity/N100_M1` | 1 | 200 | 4 | 1.42 |  | 4, 5 | none above 0.3 points |
| `diversity/N200_M1` | 1 | 200 | 4 | 1.44 |  | 4, 5 | `group_0`: 30.4 → 27.0 % |
| `diversity/N1000_M1` (first 5 s) | 1 | 200 | 3 | 0.84 |  | 4 | none above 0.3 points |
| `diversity/N10000_M1` (first 1 s) | 1 | 200 | 3 | 1.62 |  | 4 | none above 0.3 points |
| `industry_retreat/motivation` | 1 | 200 | 20 | 1.61 |  | 3 | `exponential_backoff_jitter` [20-40s]: 5.0 → 1.8 % |
| `industry_retreat/motivation-detail` | 1 | 200 | 4 | 1.34 |  | 3 | none above 0.3 points |
| `industry_retreat/motivation-mixed` | 1 | 200 | 16 | 1.85 |  | 3, 4 | `retry_budget`: 52.7 → 74.9 % |
| `industry_retreat/motivation-varied` | 1 | 200 | 20 | 1.76 |  | 3 | `exponential_backoff_jitter` [20-40s]: 5.0 → 1.8 % |
| `industry_retreat/multi-client` | 27 | 200 | 216 | 3.20 | 1: 3.20 → 1.14 | 4 | `retry-budget` at p_fail=0.2,replicas=1000: 88.7 → 93.2 % |
| `industry_retreat/multi-client-test` | 1 | 200 | 8 | 0.99 |  | 4 | `circuit-breaker`: 66.3 → 67.6 % |
| `load_spike_metastable_failure/ep1` | 1 | 200 | 4 | 0.98 |  | none | identical runs |
| `load_spike_metastable_failure/ep2_1` | 1 | 200 | 4 | 0.41 |  | none | identical runs |
| `load_spike_metastable_failure/ep2_2` | 1 | 200 | 4 | 0.41 |  | none | identical runs |
| `load_spike_metastable_failure/ep2_3` | 1 | 200 | 4 | 0.78 |  | none | identical runs |
| `load_spike_metastable_failure/ep3` | 1 | 200 | 6 | 0.94 |  | none | identical runs |
| `load_spike_metastable_failure/ep4` | 1 | 200 | 6 | 1.03 |  | none | identical runs |
| `partial_failure_metastable_failure/ep3` | 1 | 200 | 4 | 1.44 |  | none | identical runs |
| `partial_failure_metastable_failure/ep4` | 1 | 200 | 4 | 1.16 |  | none | identical runs |
| `partial_failure_metastable_failure/ep5` | 1 | 200 | 6 | 1.04 |  | none | identical runs |
| `partial_failure_metastable_failure/ep5_aimd` | 1 | 200 | 6 | 0.53 |  | none | identical runs |
| `post_mortem_simulation/s01__buggy_release` | 1 | 200 | 20 | 2.36 |  | 3 | `jittered` [180-360s]: 3.5 → 1.7 % |
| `post_mortem_simulation/s02__shaky_foundations` | 1 | 200 | 20 | 2.06 |  | 3 | `three_retries` [20-40s]: 67.3 → 54.6 % |
| `post_mortem_simulation/s03__thundering_herd` | 1 | 200 | 20 | 2.12 |  | 3, 4 | `jittered`: 64.8 → 28.7 % |
| `post_mortem_simulation/s04__internal_retry_rogue` | 1 | 200 | 15 | 1.86 |  | 1, 2 | `full_retries`: 40.5 → 52.9 % |
| `sensitivity/sensitivity-aimd` | 25 | 200 | 600 | 3.26 | 4: 3.13 → 0.49, 3.26 → 0.54, 3.14 → 0.98, 3.10 → 0.84 | 3, 4 | `circuit_breaker` [20-40s] at p_fail=0.3,failure_threshold=0.5: 13.8 → 52.3 % |
| `sensitivity/sensitivity-arolla` | 25 | 200 | 600 | 3.13 | 1: 3.13 → 0.16 | 3, 4 | `three_retries` [20-40s] at p_fail=0.3,alpha=0.5: 75.1 → 91.8 % |
| `sensitivity/sensitivity-cb` | 25 | 200 | 600 | 3.10 | 1: 3.10 → 0.53 | 3, 4, 8 | `circuit_breaker` [20-40s] at p_fail=0.3,failure_threshold=0.5: 14.1 → 52.3 % |
| `sensitivity/sensitivity-rb` | 25 | 200 | 600 | 3.55 | 3: 3.55 → 0.62, 3.47 → 0.89, 3.19 → 1.51 | 3, 4 | `circuit_breaker` [20-40s] at p_fail=0.5,budget_ratio=0.5: 43.8 → 54.0 % |
| `snowflake/motivation` | 1 | 200 | 22 | 1.69 |  | 3 | `no_control` [20-40s]: 3.1 → 1.3 % |
| `snowflake/motivation_f10` | 1 | 200 | 22 | 1.48 |  | 3, 8 | none above 0.3 points |
| `snowflake/motivation_f25` | 1 | 200 | 22 | 2.00 |  | 3 | `no_control` [20-40s]: 8.9 → 5.6 % |
| `snowflake/motivation_f75` | 1 | 200 | 22 | 2.50 |  | 3 | `no_control` [20-40s]: 1.2 → 0.4 % |
| `snowflake/motivation_f90` | 1 | 200 | 22 | 2.16 |  | 3 | none above 0.3 points |
| `snowflake/multi-client` | 18 | 200 | 144 | 2.31 |  | 4 | `retry-budget` at p_fail=1.0,replicas=1000: 1.7 → 0.1 % |
| `snowflake/policy-aimd` | 1 | 200 | 24 | 1.82 |  | 3, 4 | none above 0.3 points |
| `snowflake/policy-arolla` | 1 | 200 | 24 | 1.95 |  | 3, 4 | `circuit_breaker` [20-40s]: 50.6 → 52.1 % |
| `snowflake/policy-cb` | 1 | 200 | 24 | 2.89 |  | 3, 4 | none above 0.3 points |
| `snowflake/policy-none` | 1 | 200 | 16 | 1.85 |  | 3, 4 | `retry_budget`: 52.7 → 74.9 % |
| `snowflake/policy-rb` | 1 | 200 | 24 | 2.23 |  | 3, 4 | `circuit_breaker` [20-40s]: 48.8 → 50.6 % |
| `snowflake/sensitivity_arolla` | 1 | 200 | 6 | 1.44 |  | 3 | none above 0.3 points |
| `snowflake/sensitivity_circuitbreaker` | 1 | 200 | 6 | 1.03 |  | 3 | none above 0.3 points |
| `snowflake/sensitivity_retrybudget` | 1 | 200 | 6 | 1.22 |  | 3 | none above 0.3 points |
| **44 files** | **183** | | **3213** | | **10** | | |

The 3213 cells differ as chance has them differ: 68.5 % are within one
standard error and 95.5 % within two (a normal law gives 68.3 and 95.4),
ten are beyond three where about nine are expected, and none of them
is beyond three when its sweep point is run again on fresh seeds. Across the
points of a sweep no measure leans one way either: of the 112 measures of the
six sweep files, none has a combined gap beyond 3 SE.

`diversity/N1000_M1` and `N10000_M1` are run on the first 5 s and 1 s of their
100 s. As written they send 188 000 and 1.88 million requests per second to a
service with 16 workers: nothing succeeds from the first second on (0.004 % and
0.002 %, in both simulators), and the fault at 30 s has nothing left to change.

**The last two columns** are about their simulator alone. The first names the
commits of their branch that change the file's numbers: every file was run on
every commit with the same seeds, and their simulator is deterministic for a
seed, so a difference is the commit's. The second is the largest change in a
success rate between the simulator they ship and the fixed one over 30 seeds
(10 per sweep point), among the changes above 0.3 points and three standard
errors of the paired difference.

| Fix | Their code before it | Files it changes |
| --- | --- | ---: |
| 1 `bd7fc72` | a dependency call completes on the dependency's first attempt, and again on each later one | 2 |
| 2 `83dea18` | a client with its own policy takes the service's first attempt for the call's result | 2 |
| 3 `5c95f08` | the answer to a call the client has timed out is processed as a new result | 25 |
| 4 `bb5f8d3` | replicas, and clients that share a workload block, arrive at the same instants | 19 |
| 5 `15a2f3f` | a retry-budget deposit is cut to whole hundredths of a retry | 3 |
| 8 `b85063c` | the time-based breaker does not close when its window sits exactly on the threshold | 2 |
| 6, 7, 9, 10, 11 | the queue average in their own output; the bursty and leaky rate limiters and the refusal that ends the run, which no shipped file configures; an event scheduled in the past, which no shipped file produces | 0 |

A success rate moves in 22 of the 44 files. The ten `load_spike_*` and
`partial_failure_*` files, one service and one relaying client each, are
untouched by every fix: their shipped numbers stand. Fix 8 changes single runs
of its two files, where a decision falls exactly on the breaker's threshold, and
no average that 40 to 120 paired seeds can tell apart.

**The first arrival.** With the reference left as it is -- every client's first
request at t = 0 -- the same comparison shows what that rule does where there
are many clients. In `snowflake/multi-client` 35 of the 144 cells were beyond
3 SE, all of them at 100 and 1000 clients: at 1000 clients and `p_fail` 0.2 the
median latency was 10.21 ms in their simulator against 10.07 in msim (117 SE),
and their retry-budget clients made 1.1642 attempts per request against 1.1650
(20 SE). With the first arrival drawn, no cell of that file is beyond 2.4 SE.

**What no shipped file uses.** The five scenarios of `tests/arolla_scenarios/`,
the same comparison, msim at `bc951ea` against their branch at `ed79bdd` (the
rate limiters need its last fix):

| Scenario | What it covers | Cells | Worst cell (SE) |
| --- | --- | ---: | ---: |
| `admission_between_services` | a limiter on a service that another service calls, and on one that retries under a client that retries too | 10 | 1.36 |
| `dependencies` | optional, several sequential, and parallel dependencies whose callees retry | 12 | 1.50 |
| `e2e_budget` | an end-to-end retry budget across three layers | 8 | 1.90 |
| `faults_and_backoff` | a latency injection, plain exponential backoff, equal jitter, the goodput budget's burst cap | 24 | 2.57 |
| `own_retry_gates` | the three rate limiters, the count-based breaker and the retry gate on a service | 20 | 1.45 |
