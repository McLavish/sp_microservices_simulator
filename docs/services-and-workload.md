# Services & workload

The supply and demand sides of the simulation: a **service** is an M/G/c/K queue
that takes time and sometimes fails; a **workload** is the arrival process that
feeds it. Both can have **faults** layered on top, and both can register those
faults into a tracker for plot overlays.

## Service: an M/G/c/K queue (`service.py`)

A service is a Poisson-fed queue with **c** workers, a general (lognormal) service
time distribution, and an optional **K**-bounded queue:

```
arrivals --> [ queue (FCFS, cap = queue_capacity) ] --> [ c workers ] --> done
                       |                                      |
                   full -> drop QUEUE_FULL          service time ~ lognormal
```

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

`register_fault_events(tracker)` (`service.py`) pushes this service's latency
injections and partial failures into a `FaultEventsTracker` for plotting.

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

**Latency injection math** (`_adjust_latency`, `service.py`): across all *active*
injections, multipliers multiply and additive terms sum, applied as
`base * mult + add` (then clamped at 0). So a `multiplier=2, add_latency=ms_to_ns(10)`
window doubles the draw and adds 10 ms.

**Server failures** (`_fails_now`, `service.py`): the effective failure
probability at time `t` is the **max** `p_fail` over active partial failures; the
attempt fails if `rng.random() < p_fail`.

### The four drop points

A `DropReason` can be assigned at four distinct moments — worth knowing when
reasoning about where load is shed:

| When | Reason | Code |
| --- | --- | --- |
| At submit, queue already full | `QUEUE_FULL` | `service.py` |
| At submit / at dequeue, already past deadline | `DEADLINE` | `service.py`, `service.py` |
| At finish, now past deadline | `DEADLINE` | `service.py` |
| At finish, failure roll lost | `SERVER_FAILURE` | `service.py` |

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
