#!/bin/bash

python -m plotting.failures_over_time chain_output.csv --fault-events fault_events.json --output chain_results/chain/failures_over_time.png
python -m plotting.latencies_over_time chain_output.csv --fault-events fault_events.json --output chain_results/chain/latencies_over_time.png
python -m plotting.queue_size_over_time chain_output.csv --fault-events fault_events.json --output chain_results/chain/queue_size_over_time.png
python -m plotting.qps_over_time chain_output.csv --fault-events fault_events.json --output chain_results/chain/qps_over_time.png


python -m plotting.failures_over_time svc-A_output.csv --fault-events fault_events.json --output chain_results/svA/failures_over_time.png
python -m plotting.latencies_over_time svc-A_output.csv --fault-events fault_events.json --output chain_results/svA/latencies_over_time.png
python -m plotting.queue_size_over_time svc-A_output.csv --fault-events fault_events.json --output chain_results/svA/queue_size_over_time.png
python -m plotting.qps_over_time svc-A_output.csv --fault-events fault_events.json --output chain_results/svA/qps_over_time.png

python -m plotting.failures_over_time svc-B_output.csv --fault-events fault_events.json --output chain_results/svB/failures_over_time.png
python -m plotting.latencies_over_time svc-B_output.csv --fault-events fault_events.json --output chain_results/svB/latencies_over_time.png
python -m plotting.queue_size_over_time svc-B_output.csv --fault-events fault_events.json --output chain_results/svB/queue_size_over_time.png
python -m plotting.qps_over_time svc-B_output.csv --fault-events fault_events.json --output chain_results/svB/qps_over_time.png

python -m plotting.failures_over_time svc-C_output.csv --fault-events fault_events.json --output chain_results/svC/failures_over_time.png
python -m plotting.latencies_over_time svc-C_output.csv --fault-events fault_events.json --output chain_results/svC/latencies_over_time.png
python -m plotting.queue_size_over_time svc-C_output.csv --fault-events fault_events.json --output chain_results/svC/queue_size_over_time.png
python -m plotting.qps_over_time svc-C_output.csv --fault-events fault_events.json --output chain_results/svC/qps_over_time.png

python -m plotting.failures_over_time svc-E_output.csv --fault-events fault_events.json --output chain_results/svE/failures_over_time.png
python -m plotting.latencies_over_time svc-E_output.csv --fault-events fault_events.json --output chain_results/svE/latencies_over_time.png
python -m plotting.queue_size_over_time svc-E_output.csv --fault-events fault_events.json --output chain_results/svE/queue_size_over_time.png
python -m plotting.qps_over_time svc-E_output.csv --fault-events fault_events.json --output chain_results/svE/qps_over_time.png
