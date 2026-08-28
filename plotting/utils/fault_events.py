import json
import os

import matplotlib.pyplot as plt

# Event types mirror fault_events.FaultEventsTracker; the colours are the shared
# visual vocabulary across every figure.
SPAN_COLORS = {
    "latency_injection": "orange",
    "partial_failure": "red",
    "load_spike": "purple",
}


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


def span_style(event):
    """(label, colour) for one fault event — the single place the JSON schema is read.

    Both drift modes fail loudly: an unrecognized ``event_type`` used to be
    dropped by a missing else branch (a fault window vanishing from the figure
    while the run exited 0), and a renamed parameter used to fall back to a
    ``.get`` default (a band labelled "Partial Failure (0%)").
    """
    event_type = event["event_type"]
    params = event["parameters"]
    try:
        if event_type == "latency_injection":
            label = f"Latency Injection ({params['add_latency_ms']}ms, x{params['multiplier']})"
        elif event_type == "partial_failure":
            label = f"Partial Failure ({params['failure_rate'] * 100:.0f}%)"
        elif event_type == "load_spike":
            label = f"Load Spike (x{params['rps_multiplier']})"
        else:
            raise SystemExit(
                f"unknown fault event type: {event_type!r}\n"
                f"(known types: {', '.join(sorted(SPAN_COLORS))})"
            )
    except KeyError as missing:
        raise SystemExit(
            f"fault event {event_type!r} is missing parameter {missing}: {params}"
        ) from None
    return label, SPAN_COLORS[event_type]


def plot_fault_events(events, ax=None, label=True):
    """Shade the fault windows onto ``ax`` (default: the current axes).

    Callers must build their legend AFTER this call — matplotlib collects
    labels when ``legend()`` runs, so overlaying afterwards leaves the bands
    unexplained. Labels are deduped within a call because matplotlib does not:
    N identically-parameterized windows would otherwise add N identical rows.
    Pass ``label=False`` on a secondary panel whose legend lives on a sibling
    axes.
    """
    ax = plt.gca() if ax is None else ax
    labelled: set[str] = set()
    for event in events:
        span_label, color = span_style(event)
        if not label or span_label in labelled:
            span_label = "_nolegend_"
        else:
            labelled.add(span_label)
        ax.axvspan(
            event["start_time_s"],
            event["end_time_s"],
            color=color,
            alpha=0.3,
            label=span_label,
        )
