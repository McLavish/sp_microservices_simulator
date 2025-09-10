import json
import os
import matplotlib.pyplot as plt


def load_fault_events(fault_events_file):
    """Load fault events from JSON file"""
    if not fault_events_file or not os.path.exists(fault_events_file):
        return []

    with open(fault_events_file, "r") as f:
        return json.load(f)


def plot_fault_events(events):
    """Add fault events as background spans to the current plot"""
    for event in events:
        start_time = event["start_time_s"]
        end_time = event["end_time_s"]
        event_type = event["event_type"]
        params = event["parameters"]

        # add vertical lines for injected failures
        if event_type == "latency_injection":
            add_latency_ms = params.get("add_latency_ms", 0)
            multiplier = params.get("multiplier", 1)
            label = f"Latency Injection ({add_latency_ms}ms, x{multiplier})"
            plt.axvspan(start_time, end_time, color="orange", alpha=0.3, label=label)

        elif event_type == "partial_failure":
            failure_rate = params.get("failure_rate", 0)
            label = f"Partial Failure ({failure_rate*100:.0f}%)"
            plt.axvspan(start_time, end_time, color="red", alpha=0.3, label=label)

        elif event_type == "load_spike":
            multiplier = params.get("rps_multiplier", 1)
            label = f"Load Spike (x{multiplier})"
            plt.axvspan(start_time, end_time, color="purple", alpha=0.3, label=label)
