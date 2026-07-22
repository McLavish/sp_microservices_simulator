#!/bin/bash
set -e

# CSVs below are produced by `python -m experiments.chain_fanout`, whose fault
# overlay is chain_fault_events.json (fault_events.json is experiments.default's).
for pair in chain:chain svc-A:svA svc-B:svB svc-C:svC svc-E:svE; do
    csv="${pair%%:*}_output.csv"
    out="chain_results/${pair##*:}"
    mkdir -p "$out"
    python -m plotting.failures_over_time "$csv" --fault-events chain_fault_events.json --output "$out/failures_over_time.png"
    python -m plotting.latencies_over_time "$csv" --fault-events chain_fault_events.json --output "$out/latencies_over_time.png"
    python -m plotting.queue_size_over_time "$csv" --fault-events chain_fault_events.json --output "$out/queue_size_over_time.png"
    python -m plotting.qps_over_time "$csv" --fault-events chain_fault_events.json --output "$out/qps_over_time.png"
done
