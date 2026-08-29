#!/bin/bash
set -e

# CSVs below are produced by `python -m examples.chain_fanout`, whose fault
# overlay is chain_fault_events.json (fault_events.json is examples.default's).
# chain_output.csv IS the svc-A view (the top client is A's only caller), so
# there is no separate svc-A pair to render.
for pair in chain:chain svc-B:svB svc-C:svC svc-E:svE; do
    csv="${pair%%:*}_output.csv"
    out="chain_results/${pair##*:}"
    mkdir -p "$out"
    python -m msim.plotting.failures_over_time "$csv" --fault-events chain_fault_events.json --output "$out/failures_over_time.png"
    python -m msim.plotting.latencies_over_time "$csv" --fault-events chain_fault_events.json --output "$out/latencies_over_time.png"
    python -m msim.plotting.queue_size_over_time "$csv" --fault-events chain_fault_events.json --output "$out/queue_size_over_time.png"
    python -m msim.plotting.qps_over_time "$csv" --fault-events chain_fault_events.json --output "$out/qps_over_time.png"
done
