Examples:

- Single service experiments (legacy examples):
  - `python -m experiments.default`

- Chain/fanout example (A -> B -> {C, E}):
  - `python -m experiments.chain_fanout`

Plotting:

- Failures over time with fault events:
  - `python -m plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json`
