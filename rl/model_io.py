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


def _model_zip_path(model_path: str) -> str:
    """SB3 save/load append '.zip' when absent; key the sidecar on the real file
    so save("model") and load("model.zip") resolve the same sidecar."""
    path = str(model_path)
    return path if path.endswith(".zip") else path + ".zip"


def sidecar_path(model_path: str) -> str:
    return _model_zip_path(model_path) + SIDECAR_SUFFIX


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


def check_action_spec(
    model_path: str, spec: ActionSpec, step_duration_s: float | None = None
) -> None:
    """Raise ValueError when the stored spec disagrees; warn when absent.

    When `step_duration_s` is given it is checked too: the sidecar records it
    because a model trained on 1 s windows reads a different world from 0.1 s
    windows (every rate and the episode-time feature rescale), so loading under
    a different step duration silently reinterprets every observation.
    """
    path = Path(sidecar_path(model_path))
    if not path.exists():
        warnings.warn(
            f"no ActionSpec sidecar next to {model_path}; cannot verify that the "
            "eval-time action grids match the grids the model was trained with",
            stacklevel=2,
        )
        return
    payload = json.loads(path.read_text())
    if step_duration_s is not None:
        stored_step = payload.get("step_duration_s")
        if stored_step is not None and float(stored_step) != float(step_duration_s):
            raise ValueError(
                f"{model_path} was trained with step_duration_s={stored_step} but is "
                f"being evaluated with {step_duration_s}; every observation rate and "
                "the episode-time feature are scaled by it, so the model would be "
                "reading a different world than it was trained on"
            )
    stored = payload["action_spec"]
    current = dataclasses.asdict(spec)
    if stored != current:
        raise ValueError(
            f"ActionSpec mismatch for {model_path}: trained with {stored}, "
            f"evaluating with {current}"
        )
