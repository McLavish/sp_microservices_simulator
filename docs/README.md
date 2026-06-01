# Documentation

Technical reference for the microservices simulator. For install and a quickstart,
see the [top-level readme](../readme.md). Start with **architecture**, then read the
subsystem you care about.

| Doc | What it covers |
| --- | --- |
| [architecture](architecture.md) | The discrete-event loop, time model, core data types, request lifecycle, and module map. Read first. |
| [services-and-workload](services-and-workload.md) | The M/G/c/K service queue, service-time sampling, Poisson arrivals/load spikes, and fault injection. |
| [request-handling](request-handling.md) | Clients and the attempt loop, retry-policy catalog, timeout controllers, and chain/fan-out. |
| [observability](observability.md) | `Metrics` summary, the per-bucket CSV schema, and the plotting scripts. |
| [experiments](experiments.md) | Runnable scenarios (`experiments/`), their outputs, and how to write your own. |
| [rl](rl.md) | The Gymnasium environment, baseline policies, latency models, and training/comparison entry points. |
