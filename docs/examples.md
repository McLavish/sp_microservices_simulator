# Examples

Runnable example scenarios that wire the pieces together, run the simulation, and
export CSV (+ fault JSON). They live under `examples/` at the repo root — outside the
`msim` package, so they are not shipped in the wheel — and run from the repo root:

```bash
python -m examples.<name>
```

Output files are written to the **current directory** (the repo root) and are
gitignored; any `*_output.csv` / `*_fault_events.json` lying around at the root
are disposable leftovers from earlier runs.

## The shape of an example

Every example follows the same skeleton (see `examples/default.py` for the
canonical version):

```python
sim = Simulator(seed=...)
tracker = FaultEventsTracker()

svc = ServiceRuntime(cfg=ServiceConfig(...)).bind()
svc.cfg.register_fault_events(tracker)               # so faults show on plots

client = ClientRuntime(cfg=ClientConfig(retry=..., timeout_controller=...), service=svc)

wl = Workload(base_rps=..., duration_s=..., load_spikes=[...])
wl.register_fault_events(tracker)
wl.drive(sim, lambda s: client.start_request(s))     # hook arrivals to the client

sim.run(until=s_to_ns(wl.duration_s))                # measured window
sim.run()                                            # drain in-flight work

client.metrics().export_csv("output.csv")
tracker.export_json("fault_events.json")
```

To author a new scenario, copy `default.py` and change the service config, faults,
policy, or workload. The building blocks are documented in
[services-and-workload](services-and-workload.md) and
[request-handling](request-handling.md).

## Top-level scenarios

| Module | Scenario | Outputs |
| --- | --- | --- |
| `examples.default` | Single `svc-A` (20 ms, 16 workers, queue 20) under a latency injection `[10,30)s +10ms`, partial failure `[50,70)s p=0.5`, and a `[20,40)s` 2x load spike; 300 rps for 100 s; 1 retry @ 100 ms, 50 ms attempt timeout | `output.csv`, `fault_events.json` |
| `examples.nofaults` | Same single service, **no faults** — a clean baseline; 300 rps for 100 s | `nofaults_output.csv` |
| `examples.chain_fanout` | Fan-out `A -> B -> {C, E}` (all 60 ms; A/B wide, `E` 32 workers, `C` 64 because the retries pile up there); `C` has a partial failure `[10,20)s p=0.8`. The leaves keep ~2.5x headroom or better over the 188 rps offered load and the top client has a 2 s global timeout, which propagates down as each node's deadline — sized tighter (16 workers, no global timeout), the three nested retry layers collapsed the run permanently the moment `C` degraded. The fault is severe on purpose: `3 x 3 x 3` nested attempts absorbed a `p=0.5` blip completely, leaving the top-level failure panel blank, whereas `p=0.8` fails ~5% of the roots completing during the window (mostly deadline drops) and still recovers one bucket after it closes | `chain_output.csv`, `chain_fault_events.json`, and `svc-{B,C,E}_output.csv` |
| `examples.multichain_retry_storm` | Two chains `A->B->C` and `X->Y->C` sharing a bottleneck `C`; `C` outage `[40,60)s` (+30 ms latency **and** `p=0.4`) drives a metastable retry storm; 120 s. The two workloads use explicit distinct seeds (0 and 1) — `Workload` defaults to seed 0, so unseeded concurrent workloads would emit identical arrival streams | `multichain_{A,X}_output.csv`, `multichain_{B_to_C,Y_to_C}.csv`, `multichain_fault_events.json` |

The **tandem vs hold-worker** fan-out occupancy model (see
[request-handling](request-handling.md#worker-occupancy-tandem-vs-hold-worker)) is
exercised by `tests/test_chain_worker_model.py`; broader chain scenarios live in
`tests/test_chain_scenarios.py`.

## Single-service policy demos (`examples/single/`)

Each isolates one retry/limiter/breaker policy (documented in
[request-handling](request-handling.md)) under load and writes a CSV plus a
matching fault-events JSON named after itself (`<name>_output.csv` and
`<name>_fault_events.json`), so the [plotting](observability.md#plotting-msimplotting)
overlays line up:

| Module | Policy demonstrated | Output (`<name>` =) |
| --- | --- | --- |
| `single.exponential_backoff` | `ExponentialBackoffRetryPolicy` | `exponential_backoff` |
| `single.jittered_backoff` | `ExponentialBackoffWithJitterRetryPolicy` | `jittered_backoff` |
| `single.circuit_breaker_count` | `CountBasedCircuitBreakerPolicy` | `circuit_breaker_count` |
| `single.circuit_breaker_time` | `TimeBasedCircuitBreakerPolicy` | `circuit_breaker_time` |
| `single.rate_limiter_leaky` | `LeakyRateLimiterPolicy` | `rate_limiter_leaky` |
| `single.bursty_limiter` | `BurstyRateLimiterPolicy` | `bursty_limiter` |
| `single.fixed_window_limiter` | `FixedWindowBurstyLimiterPolicy` | `fixed_window_limiter` |
| `single.retry_budget` | `RetryBudgetPolicy` | `retry_budget` |

Two of these have configuration worth copying deliberately:

- `single.rate_limiter_leaky` sets a **finite** `global_timeout` (500 ms). The
  leaky bucket is a queue: a granted retry reserves the next free slot however
  far ahead it is, so without a deadline to veto against, sustained retry demand
  above the 100/s leak rate pushed the reservation clock ~24 s past the 80 s
  workload and the exported percentiles measured limiter backlog (p50 44 s on a
  15 ms service) instead of the limiter. The veto refunds the slot via
  `RetryPolicy.rollback`.
- `single.retry_budget` sizes `budget_ratio` against **requests**, not
  successes — the Finagle denominator `RetryBudgetPolicy` implements (see
  [request-handling](request-handling.md)). `max_retries` is the token-bucket
  burst ceiling, not the per-request attempt cap; that one is the underlying
  backoff's `max_attempts`.

## Plotting the results

Feed any `*_output.csv` (plus its matching `*_fault_events.json`) to the
[plotting](observability.md#plotting-msimplotting) scripts, e.g.:

```bash
python -m examples.chain_fanout
python -m msim.plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json
```

`plot.sh` batch-renders the standard chain figures.

## Where the hotelReservation model went

The DeathStarBench hotelReservation approximation that used to live here as
`experiments/hotel_matched.py` is a benchmark model, not simulator functionality. It
is now the `hotel-sim` project at `microservices-system/benchmarks/hotelreservation/sim`,
which depends on this package (`microservices-simulator[plot]`) from git and exposes
the scenario as the `hotel-sim` console script.
