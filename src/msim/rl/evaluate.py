"""Putting policies side by side on the same episodes.

    python -m msim.rl.evaluate storm --jobs 48 --out results/storm
    python -m msim.rl.evaluate single --model runs/single/model.zip

A policy is anything that turns an observation into an action. Every policy is
run on the SAME episodes (`reset(options={"episode_seed": ...})`), so what
separates two of them is the policy and not the luck of the draw, and the
difference is reported per episode (paired) as well as on average.

The baselines a learned policy has to face are the fixed settings: every
combination of knob levels is a constant policy. `static_search` runs them all
and ranks them. The best one is chosen on one set of seeds and reported on
another: picked and scored on the same episodes, the best of a few hundred
candidates would look better than it is.
"""

import argparse
import csv
import itertools
import math
import multiprocessing
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from msim.rl.env import RetryEnv
from msim.rl.scenario import Scenario
from msim.rl.scenarios import RetryStorm, SingleService

#: An observation in, an action out (one level index per knob).
Policy = Callable[[np.ndarray], Sequence[int]]

#: Episodes a fixed setting is chosen on, and episodes everything is reported on.
#: Training draws its own, from the environment's generator.
SELECTION_SEEDS = tuple(range(1_000, 1_016))
TEST_SEEDS = tuple(range(2_000, 2_032))

#: What a comparison reports, of the episode summary (`msim.rl.summary`).
METRICS = ("return", "success_rate", "fault_success_rate", "amplification")


def constant(action: Sequence[int]) -> Policy:
    """The policy that holds one setting."""
    held = [int(level) for level in action]
    return lambda observation: held


def rollout(env: RetryEnv, policy: Policy, seeds: Iterable[int]) -> list[dict[str, float]]:
    """One episode per seed under `policy`: the episode summaries, each with its `seed`."""
    episodes = []
    for seed in seeds:
        observation, info = env.reset(options={"episode_seed": seed})
        done = False
        while not done:
            observation, _, terminated, truncated, info = env.step(policy(observation))
            done = terminated or truncated
        summary = {name[3:]: value for name, value in info.items() if name.startswith("ep_")}
        episodes.append({"seed": float(seed), **summary})
    return episodes


def settings(env: RetryEnv) -> list[tuple[int, ...]]:
    """Every fixed setting: one level of each knob."""
    return list(itertools.product(*(range(levels) for levels in env.knob_levels)))


def default_action(env: RetryEnv) -> tuple[int, ...]:
    """The setting the scenario starts from."""
    return tuple(knob.default for knob in env.scenario.knobs if len(knob.values) > 1)


def no_retry_action(env: RetryEnv) -> tuple[int, ...] | None:
    """The default setting with max attempts at its lowest level, if that is a knob."""
    if "max_attempts" not in env.knob_names:
        return None
    action = list(default_action(env))
    index = env.knob_names.index("max_attempts")
    knob = next(knob for knob in env.scenario.knobs if knob.name == "max_attempts")
    action[index] = knob.values.index(min(knob.values))
    return tuple(action)


def mean(episodes: Sequence[Mapping[str, float]], metric: str) -> float:
    """The mean of a metric over the episodes that have it (nan and inf left out)."""
    values = [e[metric] for e in episodes if math.isfinite(e[metric])]
    return statistics.fmean(values) if values else math.nan


def _standard_error(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan


@dataclass(frozen=True)
class Static:
    """A fixed setting and how it did."""

    action: tuple[int, ...]
    setting: dict[str, float]
    episodes: list[dict[str, float]]


def _run_static(task: tuple[Scenario, dict[str, Any], tuple[int, ...], tuple[int, ...]]) -> Static:
    scenario, env_arguments, action, seeds = task
    env = RetryEnv(scenario, **env_arguments)
    return Static(action, env.setting(action), rollout(env, constant(action), seeds))


def in_parallel(function: Callable[[Any], Any], tasks: Sequence[Any], jobs: int) -> list[Any]:
    """`[function(task) for task in tasks]`, over `jobs` processes when that is more than one."""
    if jobs <= 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    # forkserver: workers start clean, whatever the parent has loaded or started.
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), mp_context=context) as pool:
        return list(pool.map(function, tasks))


def static_search(
    scenario: Scenario,
    seeds: Sequence[int] = SELECTION_SEEDS,
    *,
    jobs: int = 1,
    actions: Sequence[Sequence[int]] | None = None,
    **env_arguments: Any,
) -> list[Static]:
    """Every fixed setting on the same episodes, best first (by mean return).

    `actions` restricts the search. The first entry is the best fixed setting
    ON THESE SEEDS: report it on others.
    """
    env = RetryEnv(scenario, **env_arguments)
    candidates = [tuple(int(a) for a in action) for action in (actions or settings(env))]
    tasks = [(scenario, env_arguments, action, tuple(seeds)) for action in candidates]
    found = in_parallel(_run_static, tasks, jobs)
    return sorted(found, key=lambda static: mean(static.episodes, "return"), reverse=True)


def compare(
    runs: Mapping[str, Sequence[Mapping[str, float]]], *, reference: str
) -> list[dict[str, Any]]:
    """One row per policy: each metric's mean and standard error, and the paired
    difference in success rate against `reference`, episode by episode."""
    base = {e["seed"]: e["success_rate"] for e in runs[reference]}
    rows = []
    for name, episodes in runs.items():
        row: dict[str, Any] = {"policy": name, "episodes": len(episodes)}
        for metric in METRICS:
            values = [e[metric] for e in episodes if math.isfinite(e[metric])]
            row[metric] = statistics.fmean(values) if values else math.nan
            row[f"{metric}_se"] = _standard_error(values)
        faulted = [e["recovery_s"] for e in episodes if not math.isnan(e["recovery_s"])]
        recovered = [r for r in faulted if math.isfinite(r)]
        row["recovered"] = len(recovered) / len(faulted) if faulted else math.nan
        row["recovery_s"] = statistics.fmean(recovered) if recovered else math.nan
        paired = [e["success_rate"] - base[e["seed"]] for e in episodes if e["seed"] in base]
        row["vs_reference"] = statistics.fmean(paired) if paired else math.nan
        row["vs_reference_se"] = _standard_error(paired)
        rows.append(row)
    return rows


def table(rows: Sequence[Mapping[str, Any]], *, reference: str) -> str:
    """The comparison as text: rates in percent, the standard error after each."""

    def percent(row: Mapping[str, Any], key: str) -> str:
        return f"{100 * row[key]:6.2f} ± {100 * row[f'{key}_se']:4.2f}"

    lines = [
        f"{'policy':34s} {'return':>14s} {'success %':>13s} {'in fault %':>13s} "
        f"{'attempts/req':>12s} {'recovered':>9s} {'in s':>6s} {'vs ' + reference + ', points':>26s}"
    ]
    for row in rows:
        lines.append(
            f"{row['policy']:34s} {row['return']:7.2f} ± {row['return_se']:4.2f} "
            f"{percent(row, 'success_rate')} {percent(row, 'fault_success_rate')} "
            f"{row['amplification']:12.2f} {100 * row['recovered']:8.0f}% {row['recovery_s']:6.1f} "
            f"{percent(row, 'vs_reference'):>26s}"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Rows as a CSV file; the columns are the keys of the first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scenario_from(spec: str) -> Scenario:
    """A scenario by name: `single`, `storm`, or `arolla:<file>:<agent>[,<agent>...]`."""
    if spec == "single":
        return SingleService()
    if spec == "storm":
        return RetryStorm()
    if spec.startswith("arolla:"):
        from msim.rl.arolla import ArollaScenario  # needs the `arolla` extra

        _, path, agent = spec.split(":", 2)
        return ArollaScenario.from_file(path, agent.split(","))
    raise ValueError(f"a scenario is single, storm or arolla:<file>:<agent>; got {spec!r}")


def baselines(
    scenario: Scenario,
    *,
    selection_seeds: Sequence[int] = SELECTION_SEEDS,
    test_seeds: Sequence[int] = TEST_SEEDS,
    jobs: int = 1,
    **env_arguments: Any,
) -> tuple[dict[str, list[dict[str, float]]], list[Static]]:
    """The fixed settings a learned policy is put next to, on the test episodes.

    Returns the runs by name (the best fixed setting, chosen on the selection
    episodes; the scenario's default; no retries) and the whole search.
    """
    env = RetryEnv(scenario, **env_arguments)
    search = static_search(scenario, selection_seeds, jobs=jobs, **env_arguments)
    named = {"best static": search[0].action, "default": default_action(env)}
    without = no_retry_action(env)
    if without is not None:
        named["no retries"] = without
    tasks = [(scenario, env_arguments, action, tuple(test_seeds)) for action in named.values()]
    runs = in_parallel(_run_static, tasks, jobs)
    return {name: run.episodes for name, run in zip(named, runs, strict=True)}, search


def _describe(setting: Mapping[str, float]) -> str:
    return ", ".join(f"{name}={value:g}" for name, value in setting.items())


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Every fixed retry setting of a scenario, and the best ones side by side."
    )
    parser.add_argument("scenario", help="single, storm, or arolla:<file>:<agent>[,<agent>...]")
    parser.add_argument("--jobs", type=int, default=1, help="processes to run episodes in")
    parser.add_argument("--selection-seeds", type=int, default=len(SELECTION_SEEDS))
    parser.add_argument("--test-seeds", type=int, default=len(TEST_SEEDS))
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--out", type=Path, help="a directory for static.csv and comparison.csv")
    args = parser.parse_args(arguments)

    scenario = scenario_from(args.scenario)
    selection = range(SELECTION_SEEDS[0], SELECTION_SEEDS[0] + args.selection_seeds)
    test = range(TEST_SEEDS[0], TEST_SEEDS[0] + args.test_seeds)
    runs, search = baselines(
        scenario, selection_seeds=selection, test_seeds=test, jobs=args.jobs, step_s=args.step_s
    )

    print(f"{args.scenario}: {len(search)} fixed settings on {len(selection)} selection episodes")
    for static in search[:5]:
        print(
            f"  return {mean(static.episodes, 'return'):6.2f}  "
            f"success {100 * mean(static.episodes, 'success_rate'):5.1f}%  "
            f"{_describe(static.setting)}"
        )
    rows = compare(runs, reference="best static")
    print(f"\non {len(test)} test episodes (best static: {_describe(search[0].setting)})")
    print(table(rows, reference="best static"))
    if args.out is not None:
        write_csv(args.out / "comparison.csv", rows)
        write_csv(
            args.out / "static.csv",
            [
                {
                    **static.setting,
                    **{metric: mean(static.episodes, metric) for metric in METRICS},
                }
                for static in search
            ],
        )
        write_csv(
            args.out / "episodes.csv",
            [
                {"policy": name, **episode}
                for name, episodes in runs.items()
                for episode in episodes
            ],
        )


if __name__ == "__main__":
    main()
