"""The simulator as a control problem: a Gymnasium environment for retry policies.

    import gymnasium
    import msim.rl  # registers the environments

    env = gymnasium.make("msim/SingleService-v0")
    # or, with the scenario in hand:
    env = msim.rl.RetryEnv(msim.rl.RetryStorm(upstream_attempts=2), step_s=0.5)

`RetryEnv` (`env.py`) steps a simulation one window of simulated time at a time
and lets an agent re-tune the retry settings of some of its clients in between.
A `Scenario` (`scenario.py`) says which settings and builds the simulation of
each episode; `scenarios.py` has the two that ship. `docs/rl.md` has the whole
picture.

Needs Gymnasium: install `microservices-simulator[rl]`. Training on it
(Stable-Baselines3) is the `train` extra.
"""

from typing import Any

import gymnasium

from msim.rl.env import FEATURES, RetryEnv
from msim.rl.scenario import Episode, Knob, RetryGrid, RetryKnobs, Scenario, Window
from msim.rl.scenarios import Fault, FaultRanges, RetryStorm, SingleService

__all__ = [
    "FEATURES",
    "Episode",
    "Fault",
    "FaultRanges",
    "Knob",
    "RetryEnv",
    "RetryGrid",
    "RetryKnobs",
    "RetryStorm",
    "Scenario",
    "SingleService",
    "Window",
]

#: Arguments of `gymnasium.make` that are the environment's; the rest are the scenario's.
_ENV_ARGUMENTS = ("step_s", "warmup_s", "drain", "keep_records")


def _split(arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    env = {name: arguments.pop(name) for name in _ENV_ARGUMENTS if name in arguments}
    return env, arguments


def single_service_env(**arguments: Any) -> RetryEnv:
    """`msim/SingleService-v0`: `SingleService(**arguments)`, minus the environment's own."""
    env, scenario = _split(arguments)
    return RetryEnv(SingleService(**scenario), **env)


def retry_storm_env(**arguments: Any) -> RetryEnv:
    """`msim/RetryStorm-v0`: `RetryStorm(**arguments)`, minus the environment's own."""
    env, scenario = _split(arguments)
    return RetryEnv(RetryStorm(**scenario), **env)


for _name, _factory in (
    ("SingleService", "single_service_env"),
    ("RetryStorm", "retry_storm_env"),
):
    if f"msim/{_name}-v0" not in gymnasium.registry:  # importing twice is not an error
        gymnasium.register(id=f"msim/{_name}-v0", entry_point=f"msim.rl:{_factory}")
