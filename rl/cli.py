"""Shared argparse helpers for the RL entry points."""

from __future__ import annotations


def parse_int_list(value: str) -> list[int]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in items]


def parse_float_list(value: str) -> list[float]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [float(v) for v in items]


def parse_optional_int_list(value: str) -> list[int | None]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    parsed: list[int | None] = []
    for item in items:
        if item.lower() in ("none", "null"):
            parsed.append(None)
        else:
            parsed.append(int(item))
    return parsed
