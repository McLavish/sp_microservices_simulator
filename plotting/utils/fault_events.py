import json
import os

import matplotlib.pyplot as plt


def load_fault_events(fault_events_file):
    """Load fault events from JSON file.

    An explicitly-passed path that does not exist RAISES: silently returning []
    turned a typo or a stale filename into a fault-free-looking figure that
    exited 0. No path at all is still fine — a fault-free run has nothing to
    overlay.
    """
    if not fault_events_file:
        return []
    if not os.path.exists(fault_events_file):
        raise SystemExit(
            f"fault-events file not found: {fault_events_file}\n"
            "(a missing overlay would render as 'no faults' rather than an error)"
        )

    with open(fault_events_file) as f:
        return json.load(f)


def plot_fault_events(events):
    """Add fault events as background spans to the current plot"""
    for event in events:
        start_time = event["start_time_s"]
        end_time = event["end_time_s"]
        event_type = event["event_type"]
        params = event["parameters"]

        if event_type == "latency_injection":
            add_latency_ms = params.get("add_latency_ms", 0)
            multiplier = params.get("multiplier", 1)
            label = f"Latency Injection ({add_latency_ms}ms, x{multiplier})"
            plt.axvspan(start_time, end_time, color="orange", alpha=0.3, label=label)

        elif event_type == "partial_failure":
            failure_rate = params.get("failure_rate", 0)
            label = f"Partial Failure ({failure_rate * 100:.0f}%)"
            plt.axvspan(start_time, end_time, color="red", alpha=0.3, label=label)

        elif event_type == "load_spike":
            multiplier = params.get("rps_multiplier", 1)
            label = f"Load Spike (x{multiplier})"
            plt.axvspan(start_time, end_time, color="purple", alpha=0.3, label=label)
