import json
from dataclasses import asdict, dataclass
from typing import Any

from core import TimeInterval, ns_to_ms, ns_to_s


@dataclass
class FaultEvent:
    event_type: str  # "latency_injection", "partial_failure", "load_spike"
    start_time_s: float
    end_time_s: float
    parameters: dict[str, Any]  # fault-specific parameters


class FaultEventsTracker:
    def __init__(self):
        self.events: list[FaultEvent] = []

    def _add_event(self, event_type: str, duration: TimeInterval, parameters: dict[str, Any]):
        self.events.append(
            FaultEvent(
                event_type=event_type,
                start_time_s=ns_to_s(duration.begin),
                end_time_s=ns_to_s(duration.end),
                parameters=parameters,
            )
        )

    def add_latency_injection(self, duration: TimeInterval, add_latency: int, multiplier: int = 1):
        """Add a latency injection event"""
        self._add_event(
            "latency_injection",
            duration,
            {"add_latency_ms": ns_to_ms(add_latency), "multiplier": multiplier},
        )

    def add_partial_failure(self, duration: TimeInterval, p_fail: float):
        """Add a partial failure event"""
        self._add_event("partial_failure", duration, {"failure_rate": p_fail})

    def add_load_spike(self, duration: TimeInterval, rps_multiplier: float):
        """Add a load spike event"""
        self._add_event("load_spike", duration, {"rps_multiplier": rps_multiplier})

    def export_json(self, filepath: str):
        """Export fault events to JSON file"""
        events_dict = [asdict(event) for event in self.events]
        with open(filepath, "w") as f:
            json.dump(events_dict, f, indent=2)
