"""Run Arolla's simulator scenarios on msim.

Arolla (ETH EASL, https://github.com/eth-easl/Arolla) ships a simulator that
began as a copy of this one and a set of YAML scenarios for it. This package
reads those files and builds the msim objects that mean the same thing:

    from msim.compat.arolla import read, parse, build

    for point in read("experiments/yaml/snowflake/policy-arolla.yaml"):
        scenario = build(parse(point.config))
        scenario.run()
        for client in scenario.clients:
            print(client.name, client.runtime.metrics().summary().succeeded)

* `sweeps`: one file is one scenario per point of its inline sweeps;
* `schema`: their schema, read strictly (unknown keys and bad ranges are errors,
  keys their runner ignores are warnings);
* `importer`: the mapping to stations, clients, admission gates and policies;
* `policies`: their client-side retry controls as msim policy layers;
* `compare`: one scenario on both simulators over many seeds, side by side.

Reading a file needs PyYAML: install `microservices-simulator[arolla]`. The
mechanisms the mapping relies on (client-side timeouts, admission control,
request contexts) are msim's own and live in the core package.
"""

from pathlib import Path
from typing import Any

from msim.compat.arolla.importer import (
    ImportedClient,
    Scenario,
    UnsupportedScenario,
    arrival_seed,
    build,
)
from msim.compat.arolla.schema import Experiment, SchemaError, parse
from msim.compat.arolla.sweeps import SweepPoint, expand

__all__ = [
    "Experiment",
    "ImportedClient",
    "Scenario",
    "SchemaError",
    "SweepPoint",
    "UnsupportedScenario",
    "arrival_seed",
    "build",
    "expand",
    "load",
    "parse",
    "read",
]


def load(path: str | Path) -> dict[str, Any]:
    """The content of an Arolla scenario file, as YAML gives it (sweeps not expanded)."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading an Arolla scenario file needs PyYAML: install microservices-simulator[arolla]"
        ) from error
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: expected a mapping at the top of the file")
    return data


def read(path: str | Path) -> list[SweepPoint]:
    """Every concrete scenario of a file: one `SweepPoint` per point of its sweeps."""
    return expand(load(path))
