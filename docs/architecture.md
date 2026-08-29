# Architecture

How the simulator is put together: the event loop, the time model, the core data
types, and the path a request takes from arrival to completion. Read this first;
the other docs drill into individual subsystems.

## The model in one paragraph

The simulator is a **single-threaded discrete-event** engine. Nothing runs in real
time: every future action (an arrival, the end of a service stage, a retry, a
deadline firing) is an `Event` pushed onto a min-heap keyed by `(at, seq)`. The
loop pops events in time order, advances a logical clock (`Simulator.timestep`) to
each event's timestamp, and runs its callback. Because all randomness flows through
a single seeded RNG, a run is fully reproducible from its seed.

## Request lifecycle

```
Workload (Poisson arrivals)
      | on_arrival
      v
ClientRuntime.start_request ----> RootRequest(global_deadline = min(global_timeout, override))
      |  retry.allow_request()? --no--> drop CIRCUIT_OPEN, complete root
      | yes
      v
 _start_attempt ----> Request(deadline = min(global_deadline, now + attempt_timeout))
      | service.submit_attempt(deadline)
      v
ServiceRuntime
      |  free worker? --no--> queue full? --yes--> drop QUEUE_FULL
      |                       (else enqueue, FCFS)
      |  on dequeue: already past deadline? --yes--> drop DEADLINE
      v
  sample service time  (lognormal x latency injections)
      v
  finish: past deadline? --> drop DEADLINE
          fails_now()?   --> drop SERVER_FAILURE
      | else success
      v
ClientRuntime._on_attempt_done
      |  success AND end_time < deadline? --yes--> complete root --> Metrics
      |  no: retry.next_delay() -> (retry: schedule _start_attempt) or (give up: complete root)
```

The same `submit_attempt` path is reused by **chains**: a `ServiceNodeRuntime`
wraps a `ServiceRuntime` and injects a custom *behavior* that fans out to
downstream nodes before finalizing. A `TransportLink` (`transport.py`) can sit
between a client and its service — a `ServiceLike` decorator that adds a
lognormal network delay to each leg of every attempt's round trip. See
[request-handling](request-handling.md).

## Time model

All internal time is **integer nanoseconds**. Two aliases in `core.py` document
intent (both are `int`): `TimePoint` (an absolute instant) and `TimeDuration` (a
span). Convert only at boundaries with the helpers in `core.py`:

| Helper | Direction |
| --- | --- |
| `s_to_ns(s)` / `ms_to_ns(ms)` | seconds / milliseconds -> ns |
| `ns_to_s(ns)` / `ns_to_ms(ns)` | ns -> seconds / milliseconds |

`TimeInterval(begin, end)` (`core.py`) is **half-open** `[begin, end)`:
`contains(t)` is `begin <= t < end`, and `duration()` is `end - begin`. Fault
windows, request lifespans, and bucket ranges all use this convention.

> Half-open time shows up again in deadlines: a request that *finishes exactly at*
> its deadline is a `DEADLINE` drop, not a success (`end_time < deadline` is
> strict). See [request-handling](request-handling.md).

## Core data types (`core.py`)

- **`DropReason`** (`core.py`) — `NONE`, `QUEUE_FULL`, `DEADLINE`,
  `SERVER_FAILURE`, `CIRCUIT_OPEN`, `CANCELLED`. Every non-success carries one of
  these. `CANCELLED` means *abandoned, not failed* (an upstream fan-out
  short-circuited); it is counted separately and kept out of the failure buckets.
- **`Request`** (`core.py`) — a single attempt: the handling `service` name, its
  `interval` (lifespan), the `deadline` it ran under, `success`, `drop_reason`,
  `queue_size_at_end` (queue length when the attempt finished, `None` if it never
  reached the service) and `service_time_ns` (queue-exclusive occupancy).
- **`RootRequest`** (`core.py`) — one client-visible request = a list of
  `attempts` (original + retries), a `global_deadline`, a `done` flag, an optional
  `on_done` callback fired once on completion, a `cancelled` flag, a `cancel()`
  handle, and `completed_at` (prefer it over the last attempt's end: a root
  cancelled while backing off finishes later than any attempt it made).
- **`Summary`** (`core.py`) — the aggregate returned by `Metrics.summary()`
  (counts by outcome, latency percentiles, retries per root). See
  [observability](observability.md).

## The event loop (`simulator.py`)

`Simulator` (`simulator.py`) is the whole engine:

- `schedule(at, fn) -> Event` — push a callback for time `at`; returns the `Event`,
  whose `cancelled` flag can be set to skip it (used by the chain deadline guard).
- `run(until=None)` — pop and execute events while `heap[0].at <= until` (or to
  exhaustion if `until is None`), advancing `timestep` to each event. With `until`
  set, `timestep` ends at `max(timestep, until)` — never *before* `until`, and
  never rewound if a nested `run()` inside an event already advanced past it.
- `rng()` — the single `random.Random` seeded in the constructor.

`Event` (`simulator.py`) is ordered by `(at, seq)`; `seq` is a monotonic counter
so events scheduled for the same instant fire in insertion order — this keeps runs
deterministic even under ties.

The usual driver pattern (see any file in `experiments/`) is:

```python
sim.run(until=s_to_ns(workload.duration_s))  # run the measured window
sim.run()                                     # then drain in-flight work
```

## Determinism: two RNGs on purpose

There are **two** independent random streams:

- `Simulator.rng()` — service times, server-failure rolls, retry jitter. Seeded per
  run via `Simulator(seed=...)`.
- `Workload._rng` — interarrival times only. Seeded separately from `rng_seed`
  (default `0`), offset by `_ARRIVAL_SEED_OFFSET` so that passing the *same*
  integer to `Simulator(seed=...)` and `Workload(rng_seed=...)` — which callers
  routinely do — still yields two genuinely independent streams rather than one
  stream drawn twice. See `Workload.__post_init__`.

Keeping arrivals on their own RNG means you can change the simulator seed (to
resample service/failure behavior) while the **arrival pattern stays fixed** across
experiments, so comparisons are apples-to-apples. See
[services-and-workload](services-and-workload.md).

## Module map

```
src/msim/       the installable library package (import as `msim.<module>`)
  core            time types, Request/RootRequest, DropReason, Summary   (no deps)
   |
  simulator       event heap + clock
  faults          LoadSpike / LatencyInjection / PartialFailure  --> fault_events (JSON overlay)
  service         M/G/c/K queue            <- faults, fault_events, simulator
  workload        Poisson arrivals         <- faults, fault_events, simulator
  retry_policies  retry / circuit-breaker / rate-limit / budget
  timeout_controllers  static / adaptive timeouts
  caller          client + retry loop      <- retry_policies, timeout_controllers, service, metrics
  transport       per-link delay decorator <- caller, service, simulator
  chain           fan-out nodes            <- caller, service, simulator
  metrics         aggregation + CSV export

experiments/    runnable scenarios (entry points, not in the package)   --> CSV + fault JSON
plotting/       render CSV + fault JSON to PNG (not in the package)
```

## Minimal example

The engine on its own, with no services:

```python
from msim.simulator import Simulator
from msim.core import s_to_ns

sim = Simulator(seed=1)
sim.schedule(s_to_ns(2), lambda: print("B at", sim.timestep))
sim.schedule(s_to_ns(1), lambda: print("A at", sim.timestep))
sim.run()
# A at 1000000000
# B at 2000000000
```

## Where to go next

- [services-and-workload](services-and-workload.md) — the queue, service times,
  arrivals, and fault injection.
- [request-handling](request-handling.md) — clients, retries, timeouts, and chains.
- [observability](observability.md) — metrics, CSV export, and plotting.
- [experiments](experiments.md) — runnable scenarios.
