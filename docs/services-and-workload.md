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

### `ServiceConfig` (`service.py:13`, frozen)

| Field | Meaning |
| --- | --- |
| `name` | service identifier (appears in `Request.service` and metrics) |
| `latency_median` | median service time in **ns** (`median = exp(mu)`) |
| `latency_lognorm_sigma` | sigma (shape) of the underlying lognormal |
| `workers` | concurrency, the **c** in M/G/c/K |
| `queue_capacity` | max queued items; `None` = unbounded (no `QUEUE_FULL` drops) |
| `latency_injections` | list of `LatencyInjection` windows (see below) |
| `partial_failures` | list of `PartialFailure` windows |

`register_fault_events(tracker)` (`service.py:25`) pushes this service's latency
injections and partial failures into a `FaultEventsTracker` for plotting.

### `ServiceRuntime` (`service.py:44`)

Holds the live queue state (`in_flight`, `queue`). Call `.bind()` to (re)initialize
before use. The one method clients call is:

```python
submit_attempt(sim, on_done, attempt_deadline=None, behavior=None)
```

`on_done(success, service_time, drop_reason, queue_size)` is invoked exactly once
when the attempt resolves. The optional `behavior` hook is what chains use to run
multi-stage work on a worker — see [request-handling](request-handling.md).

**Service time** (`_sample_service_time`, `service.py:66`) draws
`lognormvariate(mu=log(median), sigma)` from the simulator RNG, applies any active
latency injections, and floors the result at **100 µs** (`max(100_000, ...)`).

**Latency injection math** (`_adjust_latency`, `service.py:57`): across all *active*
injections, multipliers multiply and additive terms sum, applied as
`base * mult + add` (then clamped at 0). So a `multiplier=2, add_latency=ms_to_ns(10)`
window doubles the draw and adds 10 ms.

**Server failures** (`_fails_now`, `service.py:79`): the effective failure
probability at time `t` is the **max** `p_fail` over active partial failures; the
attempt fails if `rng.random() < p_fail`.

### The four drop points

A `DropReason` can be assigned at four distinct moments — worth knowing when
reasoning about where load is shed:

| When | Reason | Code |
| --- | --- | --- |
| At submit, queue already full | `QUEUE_FULL` | `service.py:132` |
| At submit / at dequeue, already past deadline | `DEADLINE` | `service.py:121`, `service.py:167` |
| At finish, now past deadline | `DEADLINE` | `service.py:214` |
| At finish, failure roll lost | `SERVER_FAILURE` | `service.py:218` |

### Example: a faulty service

```python
from core import TimeInterval, ms_to_ns, s_to_ns
from faults import LatencyInjection, PartialFailure
from service import ServiceConfig, ServiceRuntime

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

`Workload` (`workload.py:11`) generates arrivals via an exponential interarrival
process whose rate is modulated by load spikes.

| Field | Meaning |
| --- | --- |
| `base_rps` | baseline arrival rate (requests/second) |
| `duration_s` | length of the arrival process, in seconds |
| `load_spikes` | list of `LoadSpike` windows |
| `rng_seed` | seed for the workload's **own** RNG (`None` -> `0`) |

- `rps_at(t)` (`workload.py:28`) — effective rate = `base_rps` times every active
  spike's `rps_multiplier`.
- `drive(sim, on_arrival)` (`workload.py:44`) — schedules the arrival loop; calls
  `on_arrival(sim)` at each arrival (typically `client.start_request`).
- When the rate is `0`, the loop idles forward in 0.5 s ticks **without consuming
  the RNG**, so arrival timing stays identical across runs whenever the
  RPS-positive intervals match (`workload.py:35`). This is the determinism property
  described in [architecture](architecture.md#determinism-two-rngs-on-purpose).

### Example: workload with a spike

```python
from core import TimeInterval, s_to_ns
from faults import LoadSpike
from workload import Workload

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

## Fault events: overlays for plots (`fault_events.py`)

Faults live in nanoseconds inside the sim, but plots want seconds.
`FaultEventsTracker` (`fault_events.py:16`) is a separate record that mirrors the
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
from fault_events import FaultEventsTracker

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
