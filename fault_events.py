import json
from dataclasses import asdict, dataclass
from typing import Any

from core import TimeInterval, ns_to_s


@dataclass
class FaultEvent:
    event_type: str  # "latency_injection", "partial_failure", "load_spike"
    start_time_s: float
    end_time_s: float
    parameters: dict[str, Any]  # fault-specific parameters


class FaultEventsTracker:
    def __init__(self):
        self.events: list[FaultEvent] = []

    def add_latency_injection(self, duration: TimeInterval, add_latency: int, multiplier: int = 1):
        """Add a latency injection event"""
        params = {
            "add_latency_ms": add_latency / 1_000_000,  # Convert ns to ms
            "multiplier": multiplier,
        }
        event = FaultEvent(
            event_type="latency_injection",
            start_time_s=ns_to_s(duration.begin),
            end_time_s=ns_to_s(duration.end),
            parameters=params,
        )
        self.events.append(event)

    def add_partial_failure(self, duration: TimeInterval, p_fail: float):
        """Add a partial failure event"""
        params = {"failure_rate": p_fail}
        event = FaultEvent(
            event_type="partial_failure",
            start_time_s=ns_to_s(duration.begin),
            end_time_s=ns_to_s(duration.end),
            parameters=params,
        )
        self.events.append(event)

    def add_load_spike(self, duration: TimeInterval, rps_multiplier: float):
        """Add a load spike event"""
        params = {"rps_multiplier": rps_multiplier}
        event = FaultEvent(
            event_type="load_spike",
            start_time_s=ns_to_s(duration.begin),
            end_time_s=ns_to_s(duration.end),
            parameters=params,
        )
        self.events.append(event)

    def export_json(self, filepath: str):
        """Export fault events to JSON file"""
        events_dict = [asdict(event) for event in self.events]
        with open(filepath, "w") as f:
            json.dump(events_dict, f, indent=2)
