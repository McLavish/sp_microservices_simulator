"""Sweeps in an Arolla scenario file: one file, many concrete scenarios.

Arolla writes a parameter sweep inline: where a single value belongs, a list of
values, as in `p_fail: [0.0, 0.05, 0.1]`. Every such list is one dimension, and
the scenarios the file stands for are the Cartesian product of the dimensions.
A list that is written once and referenced from several places through a YAML
anchor is ONE dimension: all the places take the same value together.

`expand` reproduces the order of their runner (`bin/run_sweep.py`): dimensions
in the order the lists are first met reading the file top to bottom, the last
dimension varying fastest. So point `i` of a file here is point `i` there.

Only lists under `services`, `clients` and `workload` are dimensions. Their
runner treats a list of scalars anywhere in the file as one; nothing else in
the schema takes a value that could be swept, and a stray list elsewhere is
left for the schema to reject by name.
"""

import copy
import itertools
from dataclasses import dataclass
from typing import Any

#: The top-level keys under which an inline list is a sweep dimension.
SWEPT_ROOTS = ("services", "clients", "workload")


@dataclass(frozen=True)
class Dimension:
    """One swept list and every place it is used (several, through an anchor)."""

    paths: tuple[str, ...]
    values: tuple[Any, ...]

    @property
    def name(self) -> str:
        """The key the list sits under, e.g. `p_fail`."""
        return self.paths[0].rsplit(".", 1)[-1]


@dataclass(frozen=True)
class SweepPoint:
    """One concrete scenario of a file."""

    index: int  # 0-based, in their runner's order
    total: int
    #: `(dimension, value)` for this point, in dimension order. Empty when the
    #: file is not a sweep.
    values: tuple[tuple[Dimension, Any], ...]
    #: The file's content with every swept list replaced by its value here.
    config: dict[str, Any]

    @property
    def label(self) -> str:
        return ",".join(f"{dimension.name}={value}" for dimension, value in self.values)


def find_dimensions(raw: dict[str, Any]) -> list[Dimension]:
    """The sweep dimensions of a loaded scenario file, in their runner's order."""
    found: dict[int, tuple[list[str], list[Any]]] = {}

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list) and node:
            if isinstance(node[0], dict):
                for index, item in enumerate(node):
                    walk(item, f"{path}.{index}")
            elif not isinstance(node[0], list):
                # A list of scalars. An aliased list is the same object wherever
                # it appears, which is what makes it one dimension.
                paths, _ = found.setdefault(id(node), ([], node))
                paths.append(path)

    # The roots in the order the FILE has them, as their runner walks it: the
    # first place a list is met decides where its dimension stands.
    for root in raw:
        if root in SWEPT_ROOTS:
            walk(raw[root], str(root))
    return [Dimension(tuple(paths), tuple(values)) for paths, values in found.values()]


def _set(config: dict[str, Any], path: str, value: Any) -> None:
    node: Any = config
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def expand(raw: dict[str, Any]) -> list[SweepPoint]:
    """Every concrete scenario of a loaded file (one, when it sweeps nothing)."""
    dimensions = find_dimensions(raw)
    combinations = list(itertools.product(*(dimension.values for dimension in dimensions)))
    points = []
    for index, combination in enumerate(combinations):
        config = copy.deepcopy(raw)
        for dimension, value in zip(dimensions, combination, strict=True):
            for path in dimension.paths:
                _set(config, path, value)
        points.append(
            SweepPoint(
                index=index,
                total=len(combinations),
                values=tuple(zip(dimensions, combination, strict=True)),
                config=config,
            )
        )
    return points
