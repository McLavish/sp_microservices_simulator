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
drain.

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
exercised by small scenarios in `tests/test_compat_arolla_importer.py`. Nothing
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
scenario too large to run whole, `--json` saves the rows.

The reference is imported from the checkout given with `--arolla` and needs what
that code needs (pydantic, numpy, pandas, PyYAML); it should be a checkout whose
known defects are fixed, or the comparison measures those. The tool is not part
of the test suite.
