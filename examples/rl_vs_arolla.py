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

    python -m examples.rl_vs_arolla static  --arolla ~/Arolla --out study --jobs 48
    python -m examples.rl_vs_arolla train   --arolla ~/Arolla --out study/runs/a-100 --seed 100
    python -m examples.rl_vs_arolla compare --arolla ~/Arolla --out study --jobs 48

Training episodes vary the failure the way Arolla's paper varies it in its
evaluation (failure rate, how long it lasts, offered load); the comparison is
made on their file as it is, on that distribution, and along each axis.
Everything is paired: for a seed, every row faces the same failure and the
same arrivals. The fixed setting is chosen on other episodes than it is
reported on, and so is the trained run when there are several.

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
#: The row everything is measured against.
REFERENCE = "their clients, arolla"

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
    grid = replace(file_grid(base.experiment, TENANTS), budget_ratio=budgets)
    return replace(base, grid=grid, variation=variation)


def _episodes(task: tuple[str, Any, Path, Variation, tuple[int, ...]]) -> list[dict[str, float]]:
    """One row of one condition: a trained model, a fixed setting, or their file as written."""
    kind, what, arolla, variation, seeds = task
    if kind == "written":
        path = folder(arolla) / f"policy-{what}.yaml"
        return as_written(path, seeds, variation=variation)
    scenario = situation(arolla, variation)
    if kind == "model":
        from msim.rl.train import evaluate_model

        return evaluate_model(what, seeds, scenario=scenario)
    return evaluate.rollout(RetryEnv(scenario), evaluate.constant(what), seeds)


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

    # Which trained run to report: the best on the SELECTION episodes of the
    # training distribution. The test episodes play no part in the choice.
    tasks = [("model", model, arolla, TRAINING, args.selection) for model in models]
    scores = [evaluate.mean(e, "return") for e in evaluate.in_parallel(_episodes, tasks, args.jobs)]
    ranked = sorted(zip(scores, models, strict=True), reverse=True)
    for score, model in ranked:
        print(f"selection return {score:7.3f}  {model.parent.name}")
    chosen = ranked[0][1]

    rows: dict[str, tuple[str, Any]] = {"ppo": ("model", chosen)}
    rows["best fixed setting"] = ("constant", best_static)
    without = evaluate.no_retry_action(env)
    assert without is not None
    rows["no retries"] = ("constant", without)
    for policy in SERVER_POLICIES:
        rows[f"their clients, {policy}"] = ("written", policy)

    where = conditions()
    tasks = [(k, w, arolla, v, args.test) for v in where.values() for k, w in rows.values()]
    done = iter(evaluate.in_parallel(_episodes, tasks, args.jobs))
    results: dict[str, list[dict[str, Any]]] = {}
    for name in where:
        runs = {row: next(done) for row in rows}
        results[name] = evaluate.compare(runs, reference=REFERENCE)
        print(f"\n== {name}")
        print(evaluate.table(results[name], reference=REFERENCE))

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
        "chosen": chosen.parent.name,
        "selection": {model.parent.name: score for score, model in ranked},
        "best_fixed_setting": env.setting(best_static),
        "conditions": {
            name: [{k: plain(v) for k, v in row.items()} for row in table]
            for name, table in results.items()
        },
        "every_run": every,
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=1) + "\n", "utf-8")


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A learned client retry policy against Arolla.")
    parser.add_argument("command", choices=("static", "train", "compare"))
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
    parser.add_argument(
        "--selection-seeds", type=int, default=32, help="episodes choices are made on"
    )
    parser.add_argument(
        "--test-seeds", type=int, default=64, help="episodes results are reported on"
    )
    args = parser.parse_args(arguments)
    args.selection = tuple(range(1_000, 1_000 + args.selection_seeds))
    args.test = tuple(range(2_000, 2_000 + args.test_seeds))
    {"static": static, "train": train, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
