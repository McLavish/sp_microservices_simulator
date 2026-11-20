"""One Arolla scenario on both simulators, many seeds, side by side.

    python -m msim.compat.arolla.compare SCENARIO.yaml --arolla ~/checkouts/Arolla \\
        --seeds 200 --jobs 32

Runs every sweep point of the file (or `--point i`) on Arolla's simulator and on
msim through the importer, `--seeds` times each with different seeds, and
prints, per client group and per measure, the two means and their difference in
standard errors. A difference beyond `--threshold` (3 by default) is marked.

Measures, per group (the replicas of one client entry together):

* `success %`: requests that succeeded, over the whole run, and for the
  requests that STARTED inside each fault window of the scenario
  (`success % [20-40s]`);
* `attempts/request`;
* `p50 ms`: median latency of the requests that succeeded;
* `retries asked` / `retries admitted`: at the admission control of the service
  the group calls, when it has one.

Each run uses its seed for the simulator AND for the arrival streams, on both
sides, so the samples are independent of each other and of the other engine's,
and each sweep point takes its own block of seeds, so the points of a file are
independent of each other too.

One known difference is in the arrival process itself: Arolla's workload sends
every client's first request at exactly t = 0, where msim's Poisson process
draws the first arrival like every other. With many clients that is a burst at
the start and one request more per client, which shows in cells whose standard
error is tiny. `--reference-start poisson` replaces, in memory, the reference's
first arrival by a drawn one, to compare everything else.

The reference simulator is imported from the checkout given with `--arolla`
(its `simulator/src`), and needs what that code needs (pydantic, numpy, pandas,
PyYAML). It should be a checkout whose known defects are fixed, or the
comparison measures those. This tool is not part of the test suite.
"""

import argparse
import copy
import json
import math
import os
import statistics
import sys
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from msim.compat.arolla import build, parse, read
from msim.compat.arolla.importer import IMPLICIT_CLIENT
from msim.compat.arolla.schema import Experiment
from msim.core import RootOutcome, classify_root, root_latency, root_started_at, s_to_ns

Measures = dict[str, dict[str, float]]  # group -> measure -> value


def fault_windows(experiment: Experiment) -> list[tuple[float, float]]:
    """Every distinct `(start_s, end_s)` a fault of the scenario is active in."""
    windows: set[tuple[float, float]] = set()
    for service in experiment.services:
        for fault in (*service.partial_failures, *service.latency_injections):
            windows.add((fault.start_s, fault.end_s))
    workloads = [client.workload for client in experiment.clients]
    if experiment.workload is not None:
        workloads.append(experiment.workload)
    for workload in workloads:
        for spike in workload.load_spikes:
            windows.add((spike.start_s, spike.end_s))
    return sorted(windows)


def _window_name(window: tuple[float, float]) -> str:
    return f"success % [{window[0]:g}-{window[1]:g}s]"


class _Group:
    """What one client group did in one run, as it is collected."""

    def __init__(self, windows: list[tuple[float, float]]) -> None:
        self.windows = [(s_to_ns(a), s_to_ns(b)) for a, b in windows]
        self.names = [_window_name(w) for w in windows]
        self.requests = 0
        self.succeeded = 0
        self.attempts = 0
        self.latencies: list[int] = []
        self.in_window = [0] * len(windows)
        self.ok_in_window = [0] * len(windows)
        self.asked: int | None = None
        self.admitted: int | None = None

    def add(self, started: int, ok: bool, attempts: int, latency: int) -> None:
        self.requests += 1
        self.attempts += attempts
        if ok:
            self.succeeded += 1
            self.latencies.append(latency)
        for index, (begin, end) in enumerate(self.windows):
            if begin <= started < end:
                self.in_window[index] += 1
                self.ok_in_window[index] += ok

    def admission(self, asked: int, admitted: int) -> None:
        self.asked = (self.asked or 0) + asked
        self.admitted = (self.admitted or 0) + admitted

    def measures(self) -> dict[str, float]:
        nan = float("nan")
        out = {
            "requests": float(self.requests),
            "success %": 100 * self.succeeded / self.requests if self.requests else nan,
            "attempts/request": self.attempts / self.requests if self.requests else nan,
            "p50 ms": statistics.median(self.latencies) / 1e6 if self.latencies else nan,
        }
        for name, total, ok in zip(self.names, self.in_window, self.ok_in_window, strict=True):
            out[name] = 100 * ok / total if total else nan
        if self.asked is not None and self.admitted is not None:
            out["retries asked"] = float(self.asked)
            out["retries admitted"] = float(self.admitted)
        return out


def _group_of(client_name: str, groups: dict[str, _Group]) -> str:
    """The client entry a (possibly replicated) client name belongs to."""
    if client_name in groups:
        return client_name
    return client_name.rsplit(".", 1)[0]


def run_msim(config: dict[str, Any], seed: int, duration_s: float | None) -> Measures:
    experiment = parse(config)
    scenario = build(experiment, seed=seed, arrivals=seed, duration_s=duration_s)
    scenario.run()
    windows = fault_windows(experiment)
    groups: dict[str, _Group] = {}
    for client in scenario.clients:
        group = groups.setdefault(client.group, _Group(windows))
        for root in client.runtime.roots:
            if not root.attempts:
                continue
            ok = classify_root(root) is RootOutcome.SUCCESS
            group.add(root_started_at(root), ok, len(root.attempts), root_latency(root))
        counts = scenario.admission_counts(client)
        if counts is not None:
            group.admission(counts.retries_asked, counts.retries_admitted)
    return {name: group.measures() for name, group in groups.items()}


def _draw_first_arrival(workload_class: Any, s_to_ns: Any) -> None:
    """Make the reference's workload draw its first arrival instead of forcing it to t = 0."""

    def drive(self: Any, sim: Any, on_arrival: Any) -> None:
        end_at = s_to_ns(self.duration_s)

        def tick() -> None:
            if sim.timestep >= end_at:
                return
            on_arrival(sim)
            sim.schedule(sim.timestep + max(1, self._sample_interarrival(sim.timestep)), tick)

        sim.schedule(max(1, self._sample_interarrival(0)), tick)

    workload_class.drive = drive


def run_arolla(
    config: dict[str, Any],
    seed: int,
    duration_s: float | None,
    checkout: str,
    start: str = "as-is",
) -> Measures:
    """The same scenario on the reference simulator.

    Their engine completes requests through nested callbacks, so a long run
    needs a deep stack: it runs in a thread that has one.
    """
    result: dict[str, Any] = {}

    def body() -> None:
        source = os.path.join(os.path.expanduser(checkout), "simulator", "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from simulator.config.loader import ConfigLoader  # ty: ignore[unresolved-import]
        from simulator.config.schema import ExperimentConfig  # ty: ignore[unresolved-import]

        if start == "poisson":
            from simulator.runtime.workload import Workload  # ty: ignore[unresolved-import]
            from simulator.utils import time as their_time  # ty: ignore[unresolved-import]

            _draw_first_arrival(Workload, their_time.s_to_ns)

        cfg = copy.deepcopy(config)
        cfg["seed"] = seed
        workloads = [client["workload"] for client in cfg.get("clients") or []]
        if cfg.get("workload"):
            workloads.append(cfg["workload"])
        for workload in workloads:
            workload["rng_seed"] = seed
            if duration_s is not None:
                workload["duration_s"] = max(1, min(int(duration_s), workload["duration_s"]))
        sim, clients, loads, _tracker, services = ConfigLoader.build_simulation(
            ExperimentConfig(**cfg)
        )
        for client, load in zip(clients, loads, strict=True):
            load.drive(sim, lambda s, c=client: c.start_request(s))
        sim.run()

        windows = fault_windows(parse(config))
        entries = [c["name"] for c in cfg.get("clients") or []] or [IMPLICIT_CLIENT]
        groups = {name: _Group(windows) for name in entries}
        for client in clients:
            group = groups[_group_of(client.cfg.name, groups)]
            for root in client.roots:
                if not root.attempts:
                    continue
                ok = any(attempt.success for attempt in root.attempts)
                started = min(attempt.interval.begin for attempt in root.attempts)
                ended = max(attempt.interval.end for attempt in root.attempts)
                group.add(started, ok, len(root.attempts), ended - started)
            stats = client.service.get_admission_stats()
            if client.service.cfg.load_limiter is not None and hasattr(
                client.service.cfg.load_limiter, "applies_pre_queue_admission"
            ):
                mine = stats.get(client.cfg.name, {"requested": 0, "admitted": 0})
                if client.service.cfg.load_limiter.applies_pre_queue_admission(True):
                    group.admission(mine["requested"], mine["admitted"])
        result["measures"] = {name: group.measures() for name, group in groups.items()}

    sys.setrecursionlimit(1_000_000)
    threading.stack_size(512 * 1024 * 1024)
    thread = threading.Thread(target=body)
    thread.start()
    thread.join()
    if "measures" not in result:
        raise RuntimeError(f"the reference simulator failed on seed {seed} (see its traceback)")
    return result["measures"]


def _job(
    job: tuple[str, dict[str, Any], int, float | None, str, str],
) -> tuple[str, int, Measures]:
    engine, config, seed, duration_s, checkout, start = job
    if engine == "msim":
        return engine, seed, run_msim(config, seed, duration_s)
    return engine, seed, run_arolla(config, seed, duration_s, checkout, start)


def _mean_se(values: list[float]) -> tuple[float, float, int]:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se, len(values)


def summarize(runs: dict[str, list[Measures]]) -> list[dict[str, Any]]:
    """One row per (group, measure): both means, their standard errors, the gap in SE."""
    rows = []
    groups = list(runs["msim"][0])
    for group in groups:
        measures = list(runs["msim"][0][group])
        for measure in measures:
            if measure not in runs["arolla"][0].get(group, {}):
                continue
            a_mean, a_se, a_n = _mean_se([run[group][measure] for run in runs["arolla"]])
            m_mean, m_se, m_n = _mean_se([run[group][measure] for run in runs["msim"]])
            spread = math.hypot(a_se, m_se)
            if a_n == 0 or m_n == 0:
                gap = float("nan")
            elif spread == 0:
                gap = 0.0 if a_mean == m_mean else math.copysign(math.inf, m_mean - a_mean)
            else:
                gap = (m_mean - a_mean) / spread
            rows.append(
                {
                    "group": group,
                    "measure": measure,
                    "arolla": a_mean,
                    "arolla_se": a_se,
                    "msim": m_mean,
                    "msim_se": m_se,
                    "gap_se": gap,
                }
            )
    return rows


def compare_point(
    config: dict[str, Any],
    *,
    checkout: str,
    seeds: int,
    first_seed: int,
    jobs: int,
    duration_s: float | None = None,
    reference_start: str = "as-is",
) -> list[dict[str, Any]]:
    todo = [
        (engine, config, first_seed + index, duration_s, checkout, reference_start)
        for index in range(seeds)
        for engine in ("arolla", "msim")
    ]
    runs: dict[str, list[Measures]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for engine, _seed, measures in pool.map(_job, todo, chunksize=1):
            runs[engine].append(measures)
    return summarize(runs)


#: Measures that describe the workload, not the outcome: reported, not judged.
NOT_JUDGED = ("requests",)


def print_rows(rows: list[dict[str, Any]], threshold: float) -> float:
    """Print the table; return the largest gap (in SE) among the judged cells."""
    worst = 0.0
    print(
        f"  {'group':28s} {'measure':22s} {'arolla':>12s} {'±se':>8s} {'msim':>12s} {'±se':>8s} {'gap/se':>7s}"
    )
    for row in rows:
        judged = row["measure"] not in NOT_JUDGED
        gap = row["gap_se"]
        mark = ""
        if judged and not math.isnan(gap):
            worst = max(worst, abs(gap))
            mark = "  <<<" if abs(gap) > threshold else ""
        print(
            f"  {row['group'][:28]:28s} {row['measure']:22s} {row['arolla']:12.4f} "
            f"{row['arolla_se']:8.4f} {row['msim']:12.4f} {row['msim_se']:8.4f} "
            f"{gap:7.2f}{mark}"
        )
    return worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("scenario", help="an Arolla scenario file (YAML)")
    parser.add_argument("--arolla", required=True, help="checkout of the reference simulator")
    parser.add_argument("--seeds", type=int, default=200, help="runs per engine (default 200)")
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--point", type=int, default=None, help="one sweep point (0-based)")
    parser.add_argument("--duration-s", type=float, default=None, help="cut every workload short")
    parser.add_argument("--threshold", type=float, default=3.0, help="gap in SE that is marked")
    parser.add_argument(
        "--reference-start",
        choices=("as-is", "poisson"),
        default="as-is",
        help="the reference's first arrival: at t = 0 as it ships, or drawn like the others",
    )
    parser.add_argument("--json", default=None, help="also write the rows to this file")
    args = parser.parse_args(argv)

    points = read(args.scenario)
    if args.point is not None:
        points = [points[args.point]]
    if args.reference_start == "poisson":
        print("reference: first arrival of every client drawn, not forced to t = 0")
    saved = []
    worst_overall = 0.0
    for point in points:
        experiment = parse(point.config)
        label = f" [{point.label}]" if point.label else ""
        print(
            f"{experiment.name}{label}: point {point.index + 1}/{point.total}, {args.seeds} seeds per engine"
        )
        for warning in experiment.warnings:
            print(f"  note: {warning}")
        rows = compare_point(
            point.config,
            checkout=args.arolla,
            seeds=args.seeds,
            # A block of seeds per sweep point: the points share no randomness.
            first_seed=args.first_seed + point.index * args.seeds,
            jobs=args.jobs,
            duration_s=args.duration_s,
            reference_start=args.reference_start,
        )
        worst = print_rows(rows, args.threshold)
        worst_overall = max(worst_overall, worst)
        print(f"  worst cell: {worst:.2f} SE")
        saved.append(
            {"scenario": experiment.name, "point": point.index, "label": point.label, "rows": rows}
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "seeds": args.seeds,
                    "first_seed": args.first_seed,
                    "reference_start": args.reference_start,
                    "points": saved,
                },
                handle,
                indent=1,
            )
    print(f"worst cell over {len(points)} point(s): {worst_overall:.2f} SE")
    return 0 if worst_overall <= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
