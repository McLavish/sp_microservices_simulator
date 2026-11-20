"""A learned client retry policy against Arolla's server-side retry admission.

Arolla (https://github.com/eth-easl/Arolla) protects a service from its
callers' retries at the server: it admits retries up to a budget coupled to
the goodput. Its simulator ships one situation under each server-side policy,
`snowflake/policy-{none,cb,rb,aimd,arolla}.yaml`: a 16-worker service, four
tenants with four different client retry policies, and a failure of half the
calls for twenty seconds.

This study asks the question from the other side. Take the server-side
policy away (`policy-none`), give all four tenants ONE client retry policy,
and let an agent tune it (max attempts, backoff, retry budget) from what the
clients themselves measure. Is the service better off than with the tenants
as their files have them behind Arolla's limiter? And is that the agent's
doing, or would any sensible fixed client setting do as well?

    python -m examples.rl_vs_arolla static   --arolla ~/Arolla --out study --jobs 48
    python -m examples.rl_vs_arolla informed --arolla ~/Arolla --out study --jobs 48
    python -m examples.rl_vs_arolla train    --arolla ~/Arolla --out study/runs/a-100 --seed 100
    python -m examples.rl_vs_arolla compare  --arolla ~/Arolla --out study --jobs 48

Training episodes vary the failure the way Arolla's paper varies it in its
evaluation (failure rate, how long it lasts, offered load); the comparison is
made on their file as it is, on that distribution, and along each axis.
Everything is paired: for a seed, every row faces the same failure and the
same arrivals. Whatever is chosen is chosen on other episodes than it is
reported on: the fixed setting, the trained model among the runs and their
snapshots, and the `alpha` of the row that gives their limiter its best value.
`informed` adds a yardstick that is no contender: per condition, the fixed
setting that is best for that condition alone.

Needs the `train` and `arolla` extras and a checkout of their repository.
"""

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from msim.rl import evaluate
from msim.rl.arolla import ArollaScenario, Variation, as_written, file_grid
from msim.rl.env import RetryEnv

#: Their four tenants: the agent drives them all, one policy for every caller.
TENANTS = ("three_retries", "exponential_backoff_jitter", "circuit_breaker", "retry_budget")
#: Their server-side policies, by file.
SERVER_POLICIES = ("none", "cb", "rb", "aimd", "arolla")
#: The row everything is measured against: their limiter as their file sets it.
REFERENCE = "their clients, arolla"
#: And their limiter at its best: `alpha`, the share of the goodput it admits as
#: retries, over the values their own `sensitivity/sensitivity-arolla.yaml` tries.
TUNED = "their clients, arolla tuned"
ALPHAS = (0.01, 0.05, 0.1, 0.2, 0.5)
#: The yardstick: per condition, the fixed client setting best for it alone.
INFORMED = "fixed, told the fault"

#: What a training episode draws. Their file is the middle of it: half the calls
#: failing from 20 s for 20 s, at the file's load.
TRAINING = Variation(p_fail=(0.1, 0.9), onset_s=(10.0, 20.0), length_s=(5.0, 35.0), load=(0.7, 1.3))
#: The axes of their evaluation, each walked with the others at the file's values.
AXES: dict[str, tuple[float, ...]] = {
    "p_fail": (0.1, 0.25, 0.5, 0.75, 0.9),
    "length_s": (5.0, 10.0, 20.0, 35.0),
    "load": (0.7, 0.85, 1.0, 1.15, 1.3),
}
FILE = {"p_fail": 0.5, "onset_s": 20.0, "length_s": 20.0, "load": 1.0}


def point(**values: float) -> Variation:
    """The file's situation with some of its values replaced: no draw at all."""
    pinned = {**FILE, **values}
    return Variation(**{name: (value, value) for name, value in pinned.items()})


def conditions() -> dict[str, Variation]:
    """Where policies are compared: their file, the training distribution, each axis."""
    found = {"their file": point(), "training distribution": TRAINING}
    for axis, values in AXES.items():
        for value in values:
            if value != FILE[axis]:  # the file's own value is "their file"
                found[f"{axis}={value:g}"] = point(**{axis: value})
    return found


def folder(arolla: Path) -> Path:
    return arolla / "simulator" / "experiments" / "yaml" / "snowflake"


def situation(arolla: Path, variation: Variation | None) -> ArollaScenario:
    """Their situation with no server-side policy and the agent on every tenant."""
    path = folder(arolla) / "policy-none.yaml"
    base = ArollaScenario.from_file(path, TENANTS)
    # Finer budget levels than the stock grid: the best budget here is set by the
    # service's spare capacity, which lies between the stock 0.2 and 0.5.
    budgets = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
    # Until an agent's first action its clients run on the grid's defaults,
    # and that is the whole warm-up. The file's own (four attempts, nothing in
    # the way) is a retry storm in the making from 1.15 times the load on: an
    # episode would begin in one that no policy chose, and at that load a
    # backlog takes most of the episode to drain. No retries until the agent
    # says otherwise.
    grid = replace(
        file_grid(base.experiment, TENANTS), budget_ratio=budgets, default_max_attempts=1
    )
    return replace(base, grid=grid, variation=variation)


def _episodes(task: tuple[str, Any, Path, Variation, tuple[int, ...]]) -> list[dict[str, float]]:
    """One row of one condition: a trained model, a fixed setting, or their file as written."""
    kind, what, arolla, variation, seeds = task
    if kind == "written":  # one of their files by its policy's name, or a file of ours
        path = what if isinstance(what, Path) else folder(arolla) / f"policy-{what}.yaml"
        return as_written(path, seeds, variation=variation)
    scenario = situation(arolla, variation)
    if kind == "model":
        from msim.rl.train import evaluate_model

        return evaluate_model(what, seeds, scenario=scenario)
    return evaluate.rollout(RetryEnv(scenario), evaluate.constant(what), seeds)


def _success(episodes: Sequence[dict[str, float]]) -> float:
    """What choices are made on: the share of all requests that succeeded, in percent."""
    return 100.0 * evaluate.mean(episodes, "success_rate")


def _name(out: Path, model: Path) -> str:
    """A model by its run, and by its steps when it is a copy taken on the way."""
    run, *rest = model.relative_to(out / "runs").parts
    return run if len(rest) == 1 else f"{run}@{model.stem.removeprefix('model-')}"


def with_alpha(arolla: Path, out: Path) -> dict[float, Path]:
    """Their `policy-arolla.yaml` once per value of `alpha`, written under `out`."""
    text = (folder(arolla) / "policy-arolla.yaml").read_text("utf-8")
    if text.count("alpha: 0.1\n") != 1:
        raise ValueError("policy-arolla.yaml no longer sets `alpha: 0.1` exactly once")
    (out / "alpha").mkdir(parents=True, exist_ok=True)
    files = {}
    for alpha in ALPHAS:
        files[alpha] = out / "alpha" / f"policy-arolla-alpha-{alpha:g}.yaml"
        files[alpha].write_text(text.replace("alpha: 0.1\n", f"alpha: {alpha:g}\n"), "utf-8")
    return files


def static(args: argparse.Namespace) -> None:
    """Every fixed client setting on the training distribution, ranked."""
    search = evaluate.static_search(
        situation(args.arolla, TRAINING), args.selection, jobs=args.jobs
    )
    ranking = [
        {
            "action": list(s.action),
            **s.setting,
            **{m: evaluate.mean(s.episodes, m) for m in evaluate.METRICS},
        }
        for s in search
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "static.json").write_text(json.dumps(ranking, indent=1) + "\n", "utf-8")
    for row in ranking[:8] + ranking[-2:]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def informed(args: argparse.Namespace) -> None:
    """The best fixed client setting for each condition by itself.

    What a client could do with ONE setting had it been told the failure and
    the load in advance. Nobody is told, so this is a yardstick and not a
    contender: an agent that adapts well should come close to it everywhere,
    and the one setting chosen for the whole distribution cannot.
    """
    best = {}
    for name, variation in conditions().items():
        scenario = situation(args.arolla, variation)
        first = evaluate.static_search(scenario, args.selection, jobs=args.jobs)[0]
        best[name] = {
            "action": list(first.action),
            **first.setting,
            **{m: evaluate.mean(first.episodes, m) for m in evaluate.METRICS},
        }
        print(
            name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best[name].items()}
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "informed.json").write_text(json.dumps(best, indent=1) + "\n", "utf-8")


def train(args: argparse.Namespace) -> None:
    """One PPO run on the training distribution."""
    from msim.rl.train import train as train_ppo

    sidecar = train_ppo(
        situation(args.arolla, TRAINING),
        args.out,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        torch_threads=2,
        snapshots=args.snapshots,
        slow_s=args.slow_s,
    )
    print(
        f"trained {sidecar['timesteps_trained']} steps in {sidecar['wall_s']} s "
        f"({sidecar['steps_per_s']} steps/s)"
    )


def compare(args: argparse.Namespace) -> None:
    """Every row on every condition, paired, against their clients behind Arolla."""
    arolla, out = args.arolla, args.out
    best_static = tuple(json.loads((out / "static.json").read_text("utf-8"))[0]["action"])
    env = RetryEnv(situation(arolla, None))
    models = sorted(path for path in (out / "runs").glob("*/model.zip"))
    # Copies taken on the way are candidates like the final models.
    candidates = models + sorted((out / "runs").glob("*/snapshots/model-*.zip"))

    # Which trained model to report: the best on the SELECTION episodes of the
    # training distribution. The test episodes play no part in the choice.
    tasks = [("model", model, arolla, TRAINING, args.selection) for model in candidates]
    scores = [_success(e) for e in evaluate.in_parallel(_episodes, tasks, args.jobs)]
    ranked = sorted(zip(scores, candidates, strict=True), reverse=True)
    for score, model in ranked:
        print(f"selection success {score:6.2f}%  {_name(out, model)}")
    chosen = ranked[0][1]

    rows: dict[str, tuple[str, Any]] = {"ppo": ("model", chosen)}
    rows["best fixed setting"] = ("constant", best_static)
    without = evaluate.no_retry_action(env)
    assert without is not None
    rows["no retries"] = ("constant", without)
    for policy in SERVER_POLICIES:
        rows[f"their clients, {policy}"] = ("written", policy)

    # Their limiter at its best, chosen the way our rows are: on the selection
    # episodes of the training distribution.
    files = with_alpha(arolla, out)
    tasks = [("written", path, arolla, TRAINING, args.selection) for path in files.values()]
    scores = [_success(e) for e in evaluate.in_parallel(_episodes, tasks, args.jobs)]
    by_alpha = dict(zip(files, scores, strict=True))
    for alpha, score in by_alpha.items():
        print(f"selection success {score:6.2f}%  their clients, arolla with alpha = {alpha:g}")
    best_alpha = max(by_alpha, key=lambda alpha: by_alpha[alpha])
    rows[TUNED] = ("written", files[best_alpha])

    # The yardstick, when `informed` has been run: per condition, the fixed
    # setting that is best for that condition alone.
    told = out / "informed.json"
    per_condition = json.loads(told.read_text("utf-8")) if told.exists() else {}

    def rows_of(name: str) -> dict[str, tuple[str, Any]]:
        if name not in per_condition:
            return rows
        return rows | {INFORMED: ("constant", tuple(per_condition[name]["action"]))}

    where = conditions()
    tasks = [
        (k, w, arolla, v, args.test) for name, v in where.items() for k, w in rows_of(name).values()
    ]
    done = iter(evaluate.in_parallel(_episodes, tasks, args.jobs))
    results: dict[str, list[dict[str, Any]]] = {}
    against_tuned: dict[str, dict[str, float]] = {}
    for name in where:
        runs = {row: next(done) for row in rows_of(name)}
        results[name] = evaluate.compare(runs, reference=REFERENCE)
        tuned = {row["policy"]: row for row in evaluate.compare(runs, reference=TUNED)}
        against_tuned[name] = {
            row: tuned[row]["vs_reference"] for row in ("ppo", "best fixed setting")
        } | {"ppo_se": tuned["ppo"]["vs_reference_se"]}
        print(f"\n== {name}")
        print(evaluate.table(results[name], reference=REFERENCE))

    print(f"\nsuccess of all requests, %; alpha = {best_alpha:g} for the tuned limiter")
    header = ("ppo", "best fixed setting", INFORMED, "no retries", REFERENCE, TUNED)
    header += ("their clients, none",)
    print(
        f"{'':24s}"
        + "".join(f"{h.replace('their clients, ', ''):>20s}" for h in header)
        + f"{'ppo - arolla':>18s}{'ppo - tuned':>18s}"
    )
    for name, table in results.items():
        by_row = {row["policy"]: row for row in table}
        gap, tuned_gap = by_row["ppo"], against_tuned[name]
        cells = [by_row[h]["success_rate"] if h in by_row else math.nan for h in header]
        print(
            f"{name:24s}"
            + "".join(f"{100 * cell:20.2f}" for cell in cells)
            + f"{100 * gap['vs_reference']:11.2f} ± {100 * gap['vs_reference_se']:4.2f}"
            + f"{100 * tuned_gap['ppo']:11.2f} ± {100 * tuned_gap['ppo_se']:4.2f}"
        )

    # How much rides on the run that was chosen: every run, on their file and
    # on the training distribution.
    every: dict[str, dict[str, float]] = {}
    for name in ("their file", "training distribution"):
        tasks = [("model", model, arolla, where[name], args.test) for model in models]
        runs = evaluate.in_parallel(_episodes, tasks, args.jobs)
        every[name] = {
            model.parent.name: evaluate.mean(episodes, "success_rate")
            for model, episodes in zip(models, runs, strict=True)
        }
        spread = sorted(every[name].values())
        print(
            f"\n{name}: success of each of the {len(spread)} runs, lowest to highest: "
            + " ".join(f"{100 * s:.2f}" for s in spread)
        )

    def plain(value: Any) -> Any:  # json has no nan or inf
        return None if isinstance(value, float) and not math.isfinite(value) else value

    summary = {
        "chosen": _name(out, chosen),
        "selection": {_name(out, model): score for score, model in ranked},
        "best_fixed_setting": env.setting(best_static),
        "arolla_alpha": {
            "selection": {f"{a:g}": s for a, s in by_alpha.items()},
            "best": best_alpha,
        },
        "against_tuned": against_tuned,
        "conditions": {
            name: [{k: plain(v) for k, v in row.items()} for row in table]
            for name, table in results.items()
        },
        "every_run": every,
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=1) + "\n", "utf-8")


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A learned client retry policy against Arolla.")
    parser.add_argument("command", choices=("static", "informed", "train", "compare"))
    parser.add_argument("--arolla", type=Path, required=True, help="a checkout of their repository")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--n-steps", type=int, default=260, help="four episodes")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--snapshots", type=int, default=0, help="copies on the way")
    parser.add_argument("--slow-s", type=float, help="the agent also sees slow averages")
    parser.add_argument(
        "--selection-seeds", type=int, default=32, help="episodes choices are made on"
    )
    parser.add_argument(
        "--test-seeds", type=int, default=64, help="episodes results are reported on"
    )
    args = parser.parse_args(arguments)
    args.selection = tuple(range(1_000, 1_000 + args.selection_seeds))
    args.test = tuple(range(2_000, 2_000 + args.test_seeds))
    commands = {"static": static, "informed": informed, "train": train, "compare": compare}
    commands[args.command](args)


if __name__ == "__main__":
    main()
