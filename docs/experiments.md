# Experiments

Runnable scenarios that wire the pieces together, run the simulation, and export
CSV (+ fault JSON). Each is a module under `experiments/`, run from the repo root:

```bash
python -m experiments.<name>
```

Output files are written to the **current directory** (the repo root) and are
gitignored; the `*_output.csv` / `*_fault_events.json` committed at the root are
disposable samples from earlier runs.

## The shape of an experiment

Every experiment follows the same skeleton (see `experiments/default.py` for the
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
| `experiments.default` | Single `svc-A` (20 ms, 16 workers, queue 20) under a latency injection `[10,30)s +10ms`, partial failure `[50,70)s p=0.5`, and a `[20,40)s` 2x load spike; 300 rps for 100 s; 1 retry @ 100 ms, 50 ms attempt timeout | `output.csv`, `fault_events.json` |
| `experiments.nofaults` | Same single service, **no faults** — a clean baseline; 300 rps for 100 s | `nofaults_output.csv` |
| `experiments.chain_fanout` | Fan-out `A -> B -> {C, E}` (all 60 ms; A/B wide, C/E 16 workers); `C` has a partial failure `[10,20)s p=0.5` | `chain_output.csv`, `chain_fault_events.json`, and `svc-{A,B,C,E}_output.csv` |
| `experiments.multichain_retry_storm` | Two chains `A->B->C` and `X->Y->C` sharing a bottleneck `C`; `C` outage `[40,60)s` (+30 ms latency **and** `p=0.4`) drives a metastable retry storm; 120 s | `multichain_{A,X}_output.csv`, `multichain_{B_to_C,Y_to_C}.csv`, `multichain_fault_events.json` |
| `experiments.test_chain_modes` | Demo/comparison of **tandem vs hold-worker** fan-out (prints results; see [request-handling](request-handling.md#worker-occupancy-tandem-vs-hold-worker)) | console output |

## Single-service policy demos (`experiments/single/`)

Each isolates one retry/limiter/breaker policy (documented in
[request-handling](request-handling.md)) under load and writes a CSV plus a
matching fault-events JSON named after itself (`<name>_output.csv` and
`<name>_fault_events.json`), so the [plotting](observability.md#plotting-plotting)
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

## Plotting the results

Feed any `*_output.csv` (plus its matching `*_fault_events.json`) to the
[plotting](observability.md#plotting-plotting) scripts, e.g.:

```bash
python -m experiments.chain_fanout
python -m plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json
```

`plot.sh` batch-renders the standard chain figures.
