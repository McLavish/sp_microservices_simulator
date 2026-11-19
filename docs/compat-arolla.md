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
| a service's limiter (`arolla_retry_budget`, `aimd_global_retry_budget`, `global_retry_budget`, `retry_budget`, a `time_based` `circuit_breaker`) | one admission controller (`GoodputCoupledBudget`, `AimdBudget`, `TokenBucketBudget`, `SuccessFundedBudget`, `FailureRateGate`), used in two places: an `AdmissionGate` in front of the station for callers that retry themselves, and a `ControlledRetryPolicy` inside the service's own retry loop |
| a client entry with `replicas: n` | `n` clients named `name.i`, each its own tenant |
| a client **with** a `retry` policy | a client that runs it: backoff, wrapped in its `circuit_breaker` (`WindowRetryGate` for `retry_circuit_breaker`, `ThreeStateRetryBreaker` for `time_based`), wrapped in its `retry_budget` (`FirstAttemptFundedBudget`; one `RetryBank` per `shared_budget_id`). It submits to the service's gate where there is one |
| ... its `timeout.attempt_ms` | a timer of the client's own: `server_deadline="global"`, so the attempts it gives up on live on at the server |
| ... no `timeout` at all | the service's `timeout` applies: its `global_ms` to the request, its `attempt_ms` to each call, enforced by the server |
| a client **without** a `retry` policy, and the single client of a top-level `workload` | a relay: a client that carries the SERVICE's retry policy and timeouts, tells the server its deadline, and submits straight to the station. The service's limiter gates those retries from inside the loop |
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
something else, the intent is kept here. 
| Their code | Here |
| --- | --- |
| the answer to an attempt the client timed out is processed as a new result | ignored (an orphan's answer goes nowhere) |
| all replicas of an entry, and all entries sharing a workload block, arrive at the same instants | one arrival stream per client |
| a retry-budget deposit is cut to whole hundredths of a retry | exact |
| a retry refused at the server is booked as a server failure | booked as `REJECTED` |
| a retry the client then does not issue stays charged to its budget | refunded |
| a shared client budget hands every client the first client's whole policy and ignores `min_retries_per_sec` | one balance, a backoff per client, the configured rate |
| the first request of every client arrives at t = 0 | an exact Poisson process |

One property of their admission rule is kept as it is, because it IS the rule:
`GoodputCoupledBudget` compares a count of retries with a fractional allowance,
so every tenant gets the first retry of each window while its allowance is
above zero.

## Not mapped yet

`build` raises `UnsupportedScenario`, naming the feature, for a scenario that
uses service `dependencies`, a client `e2e_retry_budget`, a retry policy on both
a client and the service it calls, an attempt timeout on both, or a
`count_based` breaker / `rate_limiter` on a service. Of the 44 scenario files
Arolla ships, 42 are built today; the two with dependencies are not.
