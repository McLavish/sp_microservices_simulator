"""Persist the action semantics next to saved SB3 models.

SB3 zips store only space *shapes* (MultiDiscrete nvec), so grids like
[10, 30] vs [30, 10] are indistinguishable after a bare load and every action
silently reinterprets. The sidecar records the ordered ActionSpec value lists
(the meaning of each index) so loads can verify them.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

from rl.spec import ActionSpec

SIDECAR_SUFFIX = ".spec.json"


def sidecar_path(model_path: str) -> str:
    return str(model_path) + SIDECAR_SUFFIX


def save_action_spec(
    model_path: str,
    spec: ActionSpec,
    step_duration_s: float,
    fault_scenario: str | None = None,
) -> None:
    payload = {
        "action_spec": dataclasses.asdict(spec),
        "step_duration_s": step_duration_s,
        "fault_scenario": fault_scenario,
    }
    Path(sidecar_path(model_path)).write_text(json.dumps(payload, indent=2))


def check_action_spec(model_path: str, spec: ActionSpec) -> None:
    """Raise ValueError when the stored spec disagrees; warn when absent."""
    path = Path(sidecar_path(model_path))
    if not path.exists():
        warnings.warn(
            f"no ActionSpec sidecar next to {model_path}; cannot verify that the "
            "eval-time action grids match the grids the model was trained with",
            stacklevel=2,
        )
        return
    stored = json.loads(path.read_text())["action_spec"]
    current = dataclasses.asdict(spec)
    if stored != current:
        raise ValueError(
            f"ActionSpec mismatch for {model_path}: trained with {stored}, "
            f"evaluating with {current}"
        )
