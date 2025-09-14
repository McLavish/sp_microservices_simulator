Examples:

- Single service experiments (legacy examples):
  - `python -m experiments.default`

- Chain/fanout example (A -> B -> {C, E}):
  - `python -m experiments.chain_fanout`

Tandem vs blocking downstream calls:

- Per-edge mode can be selected when wiring the graph using `DownstreamEdge`:
  - `EdgeMode.BLOCKING` (default): holds the upstream worker and waits for downstream completion (AND semantics).
  - `EdgeMode.TANDEM`: releases the upstream worker after local work; downstream requests are fired and not awaited by the upstream node.

Example:

```python
from chain import DownstreamEdge, EdgeMode

# B waits for C but fires E in tandem (does not wait)
b_node.add_downstream(DownstreamEdge(client=b_to_c_client, target=c_node, mode=EdgeMode.BLOCKING))
b_node.add_downstream(DownstreamEdge(client=b_to_e_client, target=e_node, mode=EdgeMode.TANDEM))
```

Plotting:

- Failures over time with fault events:
  - `python -m plotting.failures_over_time chain_output.csv --fault-events chain_fault_events.json`
