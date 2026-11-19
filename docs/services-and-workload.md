# Services & workload

The supply and demand sides of the simulation: a **service** is an M/G/c/K queue
that takes time and sometimes fails; a **workload** is the arrival process that
feeds it. Both can have **faults** layered on top, and both can register those
faults into a tracker for plot overlays.

## Service: an M/G/c/K queue (`service.py`)

A service is a Poisson-fed queue with **c** workers, a general (lognormal) service
time distribution, and an optional **K**-bounded queue:

```
arrivals --> [ queue (FCFS*, cap = queue_capacity) ] --> [ c workers ] --> done
                       |                                      |
                   full -> drop QUEUE_FULL          service time ~ lognormal
```

(\* FCFS unless `queue_discipline` says otherwise.)

### `ServiceConfig` (`service.py`, frozen)

| Field | Meaning |
| --- | --- |
| `name` | service identifier (appears in `Request.service` and metrics) |
| `latency_median` | median service time in **ns** (`median = exp(mu)`) |
| `latency_lognorm_sigma` | sigma (shape) of the underlying lognormal |
| `workers` | concurrency, the **c** in M/G/c/K |
| `queue_capacity` | max queued items; `None` = unbounded (no `QUEUE_FULL` drops) |
| `latency_injections` | list of `LatencyInjection` windows (see below) |
| `partial_failures` | list of `PartialFailure` windows |
| `queue_discipline` | which waiting attempt a freed worker takes: `QueueDiscipline.FCFS` (default), `LIFO` or `RANDOM` (see below) |
| `evict_expired` | `False` (default): a waiting attempt whose caller gave up keeps its queue slot until a worker reaches it; `True`: it gives the slot back at its deadline (see below) |

`register_fault_events(tracker)` (`service.py`) pushes this service's latency
injections and partial failures into a `FaultEventsTracker` for plotting.

### Queue discipline

`queue_discipline` decides WHICH waiting attempt a freed worker takes, and nothing
else: the capacity, the deadline rules and what counts as holding a queue slot are
the same for all three orders.

| `QueueDiscipline` | A freed worker takes | Draws from `sim.rng()` |
| --- | --- | --- |
| `FCFS` (default) | the oldest waiter (a heap on `(enqueued_at, seq)`) | no |
| `LIFO` | the newest waiter (a stack) | no |
| `RANDOM` | a waiter uniformly at random, picked **at dequeue** | one draw per pick among two or more entries |

`FCFS` is what every pinned result was produced with, and its code path is
unchanged. The other two are a what-if about an overloaded server. Under FCFS the
oldest waiter goes first, and once the backlog is as long as the callers' patience
that is an attempt about to time out: with 300 rps offered to a station that
serves 190 rps and callers that give up after 100 ms, FCFS completes 38 % of the
requests in time, LIFO 66 % and RANDOM 64 % (`tests/test_queue_discipline.py`).
The price of LIFO is that an old entry is reached LAST: an attempt whose caller
already gave up keeps its queue slot until the server reaches it (a server does
not scan its backlog), so under LIFO a dead entry at the bottom of the stack goes
on shedding load for as long as newer work keeps arriving.

`RANDOM` is picked when a worker frees, not at arrival -- a pick at arrival would
be a fixed priority, a different discipline -- and from the simulator's own
generator, so the run is still reproducible from the seed. A single waiter is no
choice and takes no draw.

### Evicting expired entries

By default a server does not scan its backlog: an attempt whose caller's deadline
passes while it waits is reported `DEADLINE` to the caller at that instant, but
the entry keeps its queue slot -- and so keeps shedding new arrivals at a full
queue -- until a worker reaches it and discards it. `evict_expired=True` models a
server that drops a waiting request as soon as its caller has gone: the slot is
free from the deadline on. The caller is told at the same instant with the same
queue depth either way, and the entry never takes a worker either way; only the
occupancy (`queue_len()`, hence the capacity check and the reported depths that
follow) differs.

When it matters depends on the order. Under FCFS with callers of ONE patience
it hardly does: the oldest entry is also the first to expire, and a saturated
queue settles where entries are taken just before they would (and are then cut
in service), so nothing expires while waiting. It matters as soon as impatient
entries sit behind patient ones, and most under LIFO, where a dead entry at the
bottom of the stack would otherwise hold its slot indefinitely.

Entries that hold no slot any more (cancelled, or evicted) stay in `queue` as
tombstones until taken. A heap brings them to its head and `RANDOM` draws them
like any other entry; a LIFO stack never reaches its bottom under load, so it is
rebuilt without its tombstones once they outnumber the live entries. That is
bookkeeping only -- skipping a tombstone and never having held it are the same
to the dequeue loop.

### `ServiceRuntime` (`service.py`)

Holds the live queue state (`in_flight`, `queue`). Construction already initializes
it; `.bind()` resets that state back to empty and returns `self`, so the
`ServiceRuntime(cfg=...).bind()` spelling used throughout the repo is a
constructor-chaining idiom, not required setup. The one method clients call is:

```python
submit_attempt(sim, on_done, attempt_deadline=None, behavior=None) -> AttemptControl
```

`on_done(success, occupancy, drop_reason, queue_size, occupancy_censored)` is
invoked exactly once when the attempt resolves — all five arguments are passed
positionally, so a shorter callback raises `TypeError` at the first completion.
Its type is spelled `service.AttemptDoneCallback`, named where the original
producer lives and imported by everything that describes the same callback
(`caller.ServiceLike`, `chain.ServiceNodeRuntime.submit_attempt`, and
`transport.TransportLink`, which produces it too on the client's side of the
wire). The fourth argument is `None` for an attempt that never reached a queue
(circuit-open, or expired in transit in front of a `TransportLink`).
The second argument is **worker occupancy**, not end-to-end node latency: for a
tandem chain node it covers the local stage only, because the worker is handed
back before the downstream wait (end-to-end latency is `Request.interval`).
`occupancy_censored` is True when that occupancy is a lower bound — the worker
was still held when a deadline or a cancellation cut the attempt short. The
returned `AttemptControl.cancel()` abandons the attempt while it is still
outstanding and is a no-op afterwards. The optional `behavior` hook is what
chains use to run multi-stage work on a worker — see
[request-handling](request-handling.md).

**Service time** (`_sample_service_time`, `service.py`) draws
`lognormvariate(mu=log(median), sigma)` from the simulator RNG, applies any active
latency injections, and floors the result at **100 µs** (`max(100_000, ...)`).
`ServiceConfig.__post_init__` owns both input ranges: it rejects a median below
1 ns and floors sigma at `MIN_LOGNORM_SIGMA` (1e-6), so a config asking for a
near-deterministic service reports back the sigma that will actually be sampled.

The `(median, sigma)` pair of the config is the station's default. A chain node
can carry its own (`ServiceNodeRuntime.service_time`, a `ServiceTime(median,
sigma)` validated and floored the same way): several nodes over one
`ServiceRuntime` — one per handler of a service — then share its workers, queue
and faults while each draws its local work from its own lognormal. It is still
one draw per attempt, through the same injections and the same 100 µs floor —
see [request-handling](request-handling.md#service-time-per-handler-node).

**Latency injection math** (`_adjust_latency`, `service.py`): across all *active*
injections, multipliers multiply and additive terms sum, applied as
`base * mult + add` (then clamped at 0). So a `multiplier=2, add_latency=ms_to_ns(10)`
window doubles the draw and adds 10 ms.

**Server failures** (`_fails_now`, `service.py`): the effective failure
probability at time `t` is the **max** `p_fail` over active partial failures; the
attempt fails if `rng.random() < p_fail`.

### The drop points

A `DropReason` can be assigned at these distinct moments — worth knowing when
reasoning about where load is shed:

| When | Reason | Code |
| --- | --- | --- |
| At submit, queue already full | `QUEUE_FULL` | `service.py` |
| At submit, already past deadline | `DEADLINE` | `service.py` |
| While queued, the deadline passes | `DEADLINE`, reported **at the deadline** | `service.py` |
| At finish, now past deadline | `DEADLINE` | `service.py` |
| At finish, failure roll lost | `SERVER_FAILURE` | `service.py` |

**The attempt deadline is the caller's clock.** A client whose attempt is still
waiting for a worker times out at its deadline regardless of when the server
next looks at the queue, so a queued attempt is reported at exactly that instant
(`service.py`). The server side is deliberately not deadline-aware until it
dequeues: the expired entry keeps its queue slot — still shedding `QUEUE_FULL`
on later arrivals — until the server reaches it, and is then discarded without
a second report. The dequeue path itself reports a `DEADLINE` only on the exact
tie where a worker frees at the deadline instant. (Reporting only at dequeue,
the old behaviour, made the client learn of a timeout whenever a worker happened
to free: in `examples.multichain_retry_storm` half the deadline drops at the
bottleneck were reported up to 50 ms late on a 60 ms timeout, and every retry
fired that late.) An attempt already **in service** is cut at the deadline and
its worker freed there; the occupancy it reports is censored.

### Example: a faulty service

```python
from msim.core import TimeInterval, ms_to_ns, s_to_ns
from msim.faults import LatencyInjection, PartialFailure
from msim.service import ServiceConfig, ServiceRuntime

cfg = ServiceConfig(
    name="svc-A",
    latency_median=ms_to_ns(20),
    latency_lognorm_sigma=0.5,
    workers=16,
    queue_capacity=200,
    latency_injections=[
        LatencyInjection(TimeInterval(s_to_ns(10), s_to_ns(30)), add_latency=ms_to_ns(10)),
    ],
    partial_failures=[
        PartialFailure(TimeInterval(s_to_ns(50), s_to_ns(70)), p_fail=0.5),
    ],
)
svc = ServiceRuntime(cfg=cfg).bind()
```

## Workload: Poisson arrivals (`workload.py`)

`Workload` (`workload.py`) generates arrivals as an **exact nonhomogeneous
Poisson process** for the piecewise-constant rate (base RPS x active spike
multipliers): each arrival draws one Exp(1) budget and consumes it across rate
segments, whose boundaries are the spike starts/ends. A segment of rate `r` and
length `dt` supplies `r*dt` of exposure; the arrival lands in the segment that
exhausts the budget. Memorylessness makes this exact, so short spikes are never
skipped and the first arrival is exponentially distributed (never
deterministically at t=0).

| Field | Meaning |
| --- | --- |
| `base_rps` | baseline arrival rate (requests/second) |
| `duration_s` | length of the arrival process, in seconds |
| `load_spikes` | list of `LoadSpike` windows |
| `rng_seed` | seed for the workload's **own** RNG (`None` -> `0`) |

- `rps_at(t)` (`workload.py`) — effective rate = `base_rps` times every active
  spike's `rps_multiplier`.
- `drive(sim, on_arrival)` (`workload.py`) — schedules the arrival chain; calls
  `on_arrival(sim)` at each arrival (typically `client.start_request`).
- `stop()` (`workload.py`) — cancels the pending arrival so nothing further is
  generated; lets a caller close the arrival process early and drain only the
  in-flight work when the workload's own `duration_s` outlives the run.
- Zero-rate segments are skipped exactly and consume no extra RNG (one Exp(1)
  draw per arrival regardless of how many segments it crosses), so arrival
  timing stays identical across runs whenever the RPS-positive intervals match.
  An all-zero workload schedules no events at all. This is the determinism
  property described in
  [architecture](architecture.md#determinism-two-rngs-on-purpose).

### Example: workload with a spike

```python
from msim.core import TimeInterval, s_to_ns
from msim.faults import LoadSpike
from msim.workload import Workload

wl = Workload(
    base_rps=300.0,
    duration_s=100,
    load_spikes=[LoadSpike(TimeInterval(s_to_ns(20), s_to_ns(40)), rps_multiplier=2.0)],
)
wl.drive(sim, lambda s: client.start_request(s))  # sim/client built elsewhere
```

## Faults (`faults.py`)

Three small dataclasses, each scoped to a half-open `TimeInterval`:

| Fault | Fields | Effect |
| --- | --- | --- |
| `LoadSpike` | `duration`, `rps_multiplier` | multiplies workload RPS while active |
| `LatencyInjection` | `duration`, `add_latency=0`, `multiplier=1` | inflates service time while active |
| `PartialFailure` | `duration`, `p_fail` | per-attempt failure probability while active |

`LatencyInjection` and `PartialFailure` expose `active(t)`. Multiple overlapping
windows compose as described above (latency multiplies+adds; failure takes the max).

All three validate their fields at construction, **finiteness included**: NaN and
inf pass every ordering check (`nan < 0` is False), and a fault that survives
construction is exported and shaded on the plots even when it has silently
collapsed the physics — a NaN multiplier blacks the spike window out, a NaN
`add_latency` makes the service *faster* than baseline, and an infinite value
either saturates the 1 ns interarrival clamp or overflows the event clock
mid-run.

## Fault events: overlays for plots (`fault_events.py`)

Faults live in nanoseconds inside the sim, but plots want seconds.
`FaultEventsTracker` (`fault_events.py`) is a separate record that mirrors the
configured faults in **seconds-based JSON**, so plotting can shade the windows where
something was wrong.

- `add_latency_injection`, `add_partial_failure`, `add_load_spike` — append events
  (converting ns -> s, and `add_latency` -> ms in the parameters).
- `export_json(filepath)` — write the event list.

**Registration is explicit and not automatic.** You must call
`ServiceConfig.register_fault_events(tracker)` for each service and
`Workload.register_fault_events(tracker)` for the workload, or the overlay will be
empty:

```python
from msim.fault_events import FaultEventsTracker

tracker = FaultEventsTracker()
cfg.register_fault_events(tracker)   # service faults
wl.register_fault_events(tracker)    # load spikes
tracker.export_json("fault_events.json")
```

The CSV from `Metrics` and this JSON share a seconds time axis, so they line up when
plotted together — see [observability](observability.md).

## See also

- [request-handling](request-handling.md) — who calls `submit_attempt`, and the
  `behavior` hook used by chains.
- [observability](observability.md) — turning runs into CSV/JSON and plots.
