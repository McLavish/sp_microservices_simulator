"""The simulator as a control problem: a Gymnasium environment for retry policies.

    from msim.rl import RetryEnv

`RetryEnv` (`env.py`) steps a simulation one window of simulated time at a time
and lets an agent re-tune the retry settings of some of its clients in between.
A `Scenario` (`scenario.py`) says which settings and builds the simulation of
each episode. `docs/rl.md` has the whole picture.

Needs Gymnasium: install `microservices-simulator[rl]`. Training on it
(Stable-Baselines3) is the `train` extra.
"""

from msim.rl.env import FEATURES, RetryEnv
from msim.rl.scenario import Episode, Knob, RetryGrid, RetryKnobs, Scenario, Window

__all__ = [
    "FEATURES",
    "Episode",
    "Knob",
    "RetryEnv",
    "RetryGrid",
    "RetryKnobs",
    "Scenario",
    "Window",
]
