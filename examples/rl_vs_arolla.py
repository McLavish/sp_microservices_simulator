"""A learned retry limiter against Arolla's, in front of the same clients.

Arolla (https://github.com/eth-easl/Arolla) protects a service from its
callers' retries at the server: it admits each tenant's retries up to a share
of that tenant's goodput. Its simulator ships one situation under each
server-side policy, `snowflake/policy-{none,cb,rb,aimd,arolla}.yaml`: a
16-worker service, four tenants with four different client retry policies, and
a failure of half the calls for twenty seconds.

`--agent limiter` (the default) is the comparison of limiters. Both sides are
`policy-arolla.yaml` as it is written: the same four tenants, their own retry
policies, their own timeouts, one admission gate in front of the service. The
only difference is what decides inside the gate: their rule, or a quota per
tenant that an agent sets every second (`msim.rl.limiter`), from the counts
their rule is driven by and nothing else.

`--agent clients` is another question, kept as a side study: no server-side
policy at all (`policy-none`) and ONE learned retry policy on all four tenants.
It changes the clients, so it says what coordinated clients can reach, not
which limiter is better.

    python -m examples.rl_vs_arolla static   --arolla ~/Arolla --out study --jobs 48
    python -m examples.rl_vs_arolla informed --arolla ~/Arolla --out study --jobs 48
    python -m examples.rl_vs_arolla train    --arolla ~/Arolla --out study/runs/a-100 --seed 100
    python -m examples.rl_vs_arolla compare  --arolla ~/Arolla --out study --jobs 48

Training episodes vary the failure the way Arolla's paper varies it in its
evaluation (failure rate, how long it lasts, offered load); the comparison is
made on their file as it is, on that varied situation, and along each axis.
Everything is paired: for a seed, every row faces the same failure and the
same arrivals. Whatever is chosen is chosen on other episodes than it is
reported on: the fixed setting, the trained model among the runs and their
snapshots (`--select`), and the settings of the row that gives their limiter
its best values (`alpha`, `beta_down`, `beta_up`).
`informed` adds a yardstick that is no contender: per condition, the fixed
setting that is best for that condition alone.

Needs the `train` and `arolla` extras and a checkout of their repository.
"""

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from msim.compat.arolla import build, parse, read
from msim.core import RootOutcome, classify_root, root_completed_at, s_to_ns
from msim.rl import evaluate
from msim.rl.arolla import (
    ArollaLimiterScenario,
    ArollaScenario,
    Variation,
    as_written,
    file_grid,
)
from msim.rl.env import RetryEnv
from msim.rl.scenario import Scenario

#: Their four tenants.
TENANTS = ("three_retries", "exponential_backoff_jitter", "circuit_breaker", "retry_budget")
#: Their server-side policies, by file.
SERVER_POLICIES = ("none", "cb", "rb", "aimd", "arolla")
#: The row everything is measured against: their limiter as their file sets it.
REFERENCE = "their clients, arolla"
#: And their limiter at its best: `alpha`, the share of the goodput it admits as
#: retries, over the values their own `sensitivity/sensitivity-arolla.yaml`
#: tries, with how fast its goodput estimate falls and recovers around the
#: file's values.
TUNED = "their clients, arolla tuned"
ALPHAS = (0.01, 0.05, 0.1, 0.2, 0.5)
BETAS_DOWN = (0.1, 0.3, 0.6)
BETAS_UP = (0.05, 0.15, 0.4)
#: The yardstick: per condition, the fixed setting best for it alone.
INFORMED = "fixed, told the fault"

#: The varied situation: what an episode draws, in training and in the "varied"
#: row of the comparison. Their file is the middle of it: half the calls failing
#: from 20 s for 20 s, at the file's load.
TRAINING = Variation(p_fail=(0.1, 0.9), onset_s=(10.0, 20.0), length_s=(5.0, 35.0), load=(0.7, 1.3))
#: The axes of their evaluation, each walked with the others at the file's values.
AXES: dict[str, tuple[float, ...]] = {
    "p_fail": (0.1, 0.25, 0.5, 0.75, 0.9),
    "length_s": (5.0, 10.0, 20.0, 35.0),
    "load": (0.7, 0.85, 1.0, 1.15, 1.3),
}
FILE = {"p_fail": 0.5, "onset_s": 20.0, "length_s": 20.0, "load": 1.0}
#: What the summaries leave out: the first seconds of a run.
WARMUP_S = 5.0


@dataclass(frozen=True)
class Study:
    """Which study: where their repository is, and where the agent sits."""

    arolla: Path
    #: "limiter": the agent is the limiter in front of their clients.
    #: "clients": the agent is their clients' retry policy, with no limiter.
    agent: str = "limiter"


def point(**values: float) -> Variation:
    """The file's situation with some of its values replaced: no draw at all."""
    pinned = {**FILE, **values}
    return Variation(**{name: (value, value) for name, value in pinned.items()})


def conditions() -> dict[str, Variation]:
    """Where policies are compared: their file, the varied situation, each axis."""
    found = {"their file": point(), "varied": TRAINING}
    for axis, values in AXES.items():
        for value in values:
            if value != FILE[axis]:  # the file's own value is "their file"
                found[f"{axis}={value:g}"] = point(**{axis: value})
    return found


def folder(arolla: Path) -> Path:
    return arolla / "simulator" / "experiments" / "yaml" / "snowflake"


def situation(study: Study, variation: Variation | None) -> Scenario:
    """Their situation with the agent in it, as the limiter or as the clients' policy."""
    if study.agent == "limiter":
        # Their file with their limiter in it; the agent takes the limiter's
        # place and nothing else moves. The quota's default, which holds through
        # the warm-up, lets no retry in: the service starts from its own state.
        path = folder(study.arolla) / "policy-arolla.yaml"
        return ArollaLimiterScenario.from_file(path, variation=variation)
    path = folder(study.arolla) / "policy-none.yaml"
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


def fixed_actions(study: Study, env: RetryEnv) -> list[tuple[int, ...]] | None:
    """The fixed settings that are searched: all of them, or what can be enumerated.

    A quota per tenant has ten thousand fixed settings; the search takes the
    ten that give every tenant the same share.
    """
    if study.agent != "limiter":
        return None
    return [(level,) * len(env.knob_levels) for level in range(min(env.knob_levels))]


def no_retries(study: Study, env: RetryEnv) -> tuple[int, ...]:
    """The setting under which no retry reaches the service."""
    if study.agent == "limiter":
        return (0,) * len(env.knob_levels)  # a share of nothing, for every tenant
    without = evaluate.no_retry_action(env)
    assert without is not None
    return without


def _episodes(task: tuple[str, Any, Study, Variation, tuple[int, ...]]) -> list[dict[str, float]]:
    """One row of one condition: a trained model, a fixed setting, or their file as written."""
    kind, what, study, variation, seeds = task
    if kind == "written":  # one of their files by its policy's name, or a file of ours
        path = what if isinstance(what, Path) else folder(study.arolla) / f"policy-{what}.yaml"
        return as_written(path, seeds, variation=variation)
    scenario = situation(study, variation)
    if kind == "model":
        from msim.rl.train import evaluate_model

        return evaluate_model(what, seeds, scenario=scenario)
    return evaluate.rollout(RetryEnv(scenario), evaluate.constant(what), seeds)


def _by_tenant(task: tuple[str, Any, Study, Variation, tuple[int, ...]]) -> dict[str, list[int]]:
    """The same row, tenant by tenant: requests that finished after the warm-up, and successes."""
    kind, what, study, variation, seeds = task
    counts = {tenant: [0, 0] for tenant in TENANTS}

    def count(clients: Sequence[Any]) -> None:
        for client in clients:
            tally = counts[client.cfg.tenant]
            for root in client.roots:
                if root.attempts and root_completed_at(root) >= s_to_ns(WARMUP_S):
                    tally[0] += 1
                    tally[1] += classify_root(root) is RootOutcome.SUCCESS

    if kind == "written":
        path = what if isinstance(what, Path) else folder(study.arolla) / f"policy-{what}.yaml"
        written = parse(read(path)[0].config)
        for seed in seeds:
            built = build(variation.apply(written, seed), seed=seed, arrivals=seed)
            built.run()
            count([client.runtime for client in built.clients])
        return counts
    scenario = situation(study, variation)
    if kind == "model":
        from msim.rl.train import load_policy

        sidecar = json.loads(Path(what).with_suffix(".json").read_text("utf-8"))
        env = RetryEnv(scenario, keep_records=True, slow_s=sidecar.get("slow_s"))
        policy = load_policy(what, env)
    else:
        env = RetryEnv(scenario, keep_records=True)
        policy = evaluate.constant(what)
    for seed in seeds:
        observation, _ = env.reset(options={"episode_seed": seed})
        truncated = False
        while not truncated:
            observation, _, _, truncated, _ = env.step(policy(observation))
        count(env.episode.reward_clients)
    return counts


def _success(episodes: Sequence[dict[str, float]]) -> float:
    """What choices are made on: the share of all requests that succeeded, in percent."""
    return 100.0 * evaluate.mean(episodes, "success_rate")


def _name(out: Path, model: Path) -> str:
    """A model by its run, and by its steps when it is a copy taken on the way."""
    run, *rest = model.relative_to(out / "runs").parts
    return run if len(rest) == 1 else f"{run}@{model.stem.removeprefix('model-')}"


def their_limiter(arolla: Path, out: Path) -> dict[tuple[float, float, float], Path]:
    """Their `policy-arolla.yaml` once per `(alpha, beta_down, beta_up)`, written under `out`."""
    text = (folder(arolla) / "policy-arolla.yaml").read_text("utf-8")
    written = {
        "alpha": "alpha: 0.1\n",
        "beta_down": "beta_down: 0.3\n",
        "beta_up": "beta_up: 0.15\n",
    }
    for line in written.values():
        if text.count(line) != 1:
            raise ValueError(f"policy-arolla.yaml no longer sets `{line.strip()}` exactly once")
    (out / "arolla").mkdir(parents=True, exist_ok=True)
    files = {}
    for alpha, down, up in itertools.product(ALPHAS, BETAS_DOWN, BETAS_UP):
        file = text
        for name, value in (("alpha", alpha), ("beta_down", down), ("beta_up", up)):
            file = file.replace(written[name], f"{name}: {value:g}\n")
        files[alpha, down, up] = out / "arolla" / f"policy-arolla-{alpha:g}-{down:g}-{up:g}.yaml"
        files[alpha, down, up].write_text(file, "utf-8")
    return files


def static(args: argparse.Namespace) -> None:
    """The fixed settings on the varied situation, ranked."""
    scenario = situation(args.study, TRAINING)
    search = evaluate.static_search(
        scenario,
        args.selection,
        jobs=args.jobs,
        actions=fixed_actions(args.study, RetryEnv(scenario)),
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
    """The best fixed setting for each condition by itself.

    What ONE setting could do had it been chosen knowing the failure and the
    load in advance. Nobody is told, so this is a yardstick and not a
    contender: an agent that adapts well should come close to it everywhere,
    and the one setting chosen for the whole distribution cannot.
    """
    best = {}
    for name, variation in conditions().items():
        scenario = situation(args.study, variation)
        actions = fixed_actions(args.study, RetryEnv(scenario))
        first = evaluate.static_search(scenario, args.selection, jobs=args.jobs, actions=actions)[0]
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
    """One PPO run on the varied situation (or on a wider range of loads, `--train-load`)."""
    from msim.rl.train import train as train_ppo

    # What the comparison calls the varied situation is fixed; what a run trains
    # on may reach further, so that the ends of the range are not the ends of
    # what the policy has seen.
    training = TRAINING if args.train_load is None else replace(TRAINING, load=args.train_load)
    sidecar = train_ppo(
        situation(args.study, training),
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
        lr_decay=args.lr_decay,
    )
    print(
        f"trained {sidecar['timesteps_trained']} steps in {sidecar['wall_s']} s "
        f"({sidecar['steps_per_s']} steps/s)"
    )


def compare(args: argparse.Namespace) -> None:
    """Every row on every condition, paired, against their clients behind Arolla."""
    study, out = args.study, args.out
    best_static = tuple(json.loads((out / "static.json").read_text("utf-8"))[0]["action"])
    env = RetryEnv(situation(study, None))
    models = sorted(path for path in (out / "runs").glob("*/model.zip"))
    # Copies taken on the way are candidates like the final models.
    candidates = models + sorted((out / "runs").glob("*/snapshots/model-*.zip"))

    # Which trained model to report: the best on the SELECTION episodes of the
    # varied situation. The test episodes play no part in the choice.
    tasks = [("model", model, study, TRAINING, args.selection) for model in candidates]
    scores = [_success(e) for e in evaluate.in_parallel(_episodes, tasks, args.jobs)]
    ranked = sorted(zip(scores, candidates, strict=True), reverse=True)
    for score, model in ranked:
        print(f"selection success {score:6.2f}%  {_name(out, model)}")
    chosen = ranked[0][1]
    leads: dict[str, dict[str, Any]] = {}
    if args.select == "lead":
        # Among the best of those, the one that is nowhere far behind: the
        # largest SMALLEST lead over their limiter, over every condition, still
        # on the selection episodes. A mean over the varied situation hardly
        # sees its ends, and the ends are where there is least room.
        shortlist = [model for _, model in ranked[: args.shortlist]]
        where = conditions()
        tasks = [("written", "arolla", study, v, args.selection) for v in where.values()]
        tasks += [
            ("model", model, study, v, args.selection)
            for model in shortlist
            for v in where.values()
        ]
        done = iter(evaluate.in_parallel(_episodes, tasks, args.jobs))
        theirs = {name: _success(next(done)) for name in where}
        for model in shortlist:
            lead = {name: _success(next(done)) - theirs[name] for name in where}
            worst = min(lead, key=lambda name: lead[name])
            leads[_name(out, model)] = {"smallest": lead[worst], "at": worst, "model": model}
        for name, found in sorted(leads.items(), key=lambda item: -item[1]["smallest"]):
            print(f"smallest lead {found['smallest']:6.2f} points at {found['at']:12s}  {name}")
        chosen = max(leads.values(), key=lambda found: found["smallest"])["model"]

    rows: dict[str, tuple[str, Any]] = {"ppo": ("model", chosen)}
    rows["best fixed setting"] = ("constant", best_static)
    rows["no retries"] = ("constant", no_retries(study, env))
    for policy in SERVER_POLICIES:
        rows[f"their clients, {policy}"] = ("written", policy)

    # Their limiter at its best, chosen the way our rows are: on the selection
    # episodes of the varied situation.
    files = their_limiter(study.arolla, out)
    tasks = [("written", path, study, TRAINING, args.selection) for path in files.values()]
    scores = [_success(e) for e in evaluate.in_parallel(_episodes, tasks, args.jobs)]
    by_setting = dict(zip(files, scores, strict=True))
    best_theirs = max(by_setting, key=lambda setting: by_setting[setting])
    for setting, score in sorted(by_setting.items(), key=lambda item: -item[1])[:6]:
        print(
            f"selection success {score:6.2f}%  their clients, arolla with "
            f"alpha, beta_down, beta_up = {setting}"
        )
    as_file = by_setting[0.1, 0.3, 0.15]
    print(f"selection success {as_file:6.2f}%  their clients, arolla as their file sets it")
    rows[TUNED] = ("written", files[best_theirs])

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
        (k, w, study, v, args.test) for name, v in where.items() for k, w in rows_of(name).values()
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

    print(f"\nsuccess of all requests, %; alpha, beta_down, beta_up = {best_theirs} when tuned")
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

    # Tenant by tenant: a limiter is also judged by whom it lets down.
    shown = ("ppo", REFERENCE, TUNED, "their clients, none")
    tenants: dict[str, dict[str, dict[str, float]]] = {}
    for name in ("their file", "varied"):
        tasks = [(*rows[row], study, where[name], args.test) for row in shown]
        found = evaluate.in_parallel(_by_tenant, tasks, args.jobs)
        tenants[name] = {
            row: {tenant: ok / max(1, n) for tenant, (n, ok) in counts.items()}
            for row, counts in zip(shown, found, strict=True)
        }
        print(f"\n{name}, success by tenant, %")
        print(f"{'':32s}" + "".join(f"{tenant:>28s}" for tenant in TENANTS))
        for row in shown:
            print(f"{row:32s}" + "".join(f"{100 * tenants[name][row][t]:28.2f}" for t in TENANTS))

    # How much rides on the run that was chosen: every run, on their file and
    # on the varied situation.
    every: dict[str, dict[str, float]] = {}
    for name in ("their file", "varied"):
        tasks = [("model", model, study, where[name], args.test) for model in models]
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
        "agent": study.agent,
        "chosen": _name(out, chosen),
        "chosen_by": args.select,
        "selection": {_name(out, model): score for score, model in ranked},
        "smallest_lead": {
            name: {"points": found["smallest"], "at": found["at"]} for name, found in leads.items()
        },
        "test_seeds": [args.test[0], args.test[-1]],
        "best_fixed_setting": env.setting(best_static),
        "their_limiter": {
            "selection": {repr(setting): score for setting, score in by_setting.items()},
            "best": list(best_theirs),
        },
        "against_tuned": against_tuned,
        "conditions": {
            name: [{k: plain(v) for k, v in row.items()} for row in table]
            for name, table in results.items()
        },
        "tenants": tenants,
        "every_run": every,
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=1) + "\n", "utf-8")


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A learned retry limiter against Arolla's.")
    parser.add_argument("command", choices=("static", "informed", "train", "compare"))
    parser.add_argument("--arolla", type=Path, required=True, help="a checkout of their repository")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--agent",
        choices=("limiter", "clients"),
        default="limiter",
        help="the agent as the limiter in front of their clients, or (the side study) as the "
        "clients' own retry policy with no limiter",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--n-steps", type=int, default=260, help="four episodes")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-decay", action="store_true", help="falling to zero over the run")
    parser.add_argument("--snapshots", type=int, default=0, help="copies on the way")
    parser.add_argument("--slow-s", type=float, help="the agent also sees slow averages")
    parser.add_argument(
        "--selection-seeds", type=int, default=32, help="episodes choices are made on"
    )
    parser.add_argument(
        "--test-seeds", type=int, default=64, help="episodes results are reported on"
    )
    parser.add_argument(
        "--test-from", type=int, default=2_000, help="the first of the episodes reported on"
    )
    parser.add_argument(
        "--select",
        choices=("varied", "lead"),
        default="varied",
        help="choose the trained model by its success on the varied situation, or among the "
        "best of those by its smallest lead over their limiter across the conditions",
    )
    parser.add_argument("--shortlist", type=int, default=24, help="how many `--select lead` weighs")
    parser.add_argument(
        "--train-load",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="train on loads drawn from this range instead of the comparison's 0.7 to 1.3",
    )
    args = parser.parse_args(arguments)
    if args.train_load is not None:
        args.train_load = tuple(args.train_load)
    args.study = Study(args.arolla, args.agent)
    args.selection = tuple(range(1_000, 1_000 + args.selection_seeds))
    args.test = tuple(range(args.test_from, args.test_from + args.test_seeds))
    commands = {"static": static, "informed": informed, "train": train, "compare": compare}
    commands[args.command](args)


if __name__ == "__main__":
    main()
