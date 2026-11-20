"""Training a retry policy with PPO, and putting it next to the fixed settings.

    python -m msim.rl.train storm --timesteps 1000000 --n-envs 48 --out runs/storm

Stable-Baselines3's PPO on `--n-envs` simulations, each in a process of its own
(`SubprocVecEnv`): the simulator is the bottleneck, so the simulations run side
by side and the learner waits for them. The model is saved with a sidecar
(`model.json`) that says what it was trained on: the scenario, the knobs and
their levels, the observation's entries, the step length. A model is only ever
applied to an environment that matches its sidecar.

After training, the policy is run on the test episodes next to the best fixed
setting (chosen on other episodes), the scenario's default and no retries, and
the comparison is printed and written beside the model. The model that is
compared is the one training ended with: no checkpoint is picked on the test
episodes.

Needs the `train` extra (Stable-Baselines3, PyTorch). Everything of those two
is imported inside the functions, so that importing this module does not.
"""

import argparse
import json
import math
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from msim.rl import evaluate
from msim.rl.env import RetryEnv
from msim.rl.evaluate import Policy
from msim.rl.scenario import Scenario

#: What of the episode summary goes into the training log, episode by episode.
LOGGED = ("ep_success_rate", "ep_fault_success_rate", "ep_amplification")


def make_envs(
    scenario: Scenario, n_envs: int, seed: int, monitor_dir: Path | None = None, **env: Any
):
    """`n_envs` environments on the scenario, each in its own process when there are several.

    Environment `i` is seeded with `seed + i` at its first reset and draws
    every later episode from its own generator, so no two of them see the same
    episodes.
    """
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    return make_vec_env(
        RetryEnv,  # the class, not an id: nothing to look up in a worker
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"scenario": scenario, **env},
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        monitor_dir=None if monitor_dir is None else str(monitor_dir),
        monitor_kwargs={"info_keywords": LOGGED},
    )


def _summary_logger():
    """A callback that adds the episode summaries to what PPO logs after each rollout."""
    from stable_baselines3.common.callbacks import BaseCallback

    class LogSummaries(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            episodes = list(self.model.ep_info_buffer or [])
            for name in LOGGED:
                values = [e[name] for e in episodes if name in e and math.isfinite(e[name])]
                if values:
                    self.logger.record(f"rollout/{name}", float(np.mean(values)))

    return LogSummaries()


def _commit() -> str | None:
    """The commit of the checkout this package runs from, when it is one."""
    try:
        done = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _settings(scenario: object) -> Any:
    """The scenario's own settings, for the record: its fields when it is a dataclass."""
    if is_dataclass(scenario) and not isinstance(scenario, type):
        return asdict(scenario)
    return repr(scenario)


def _describe(env: RetryEnv) -> dict[str, Any]:
    """What a model is tied to: the knobs, the observation, the step."""
    return {
        "knobs": [
            {"name": k.name, "values": list(k.values), "default": k.default}
            for k in env.scenario.knobs
            if len(k.values) > 1
        ],
        "features": list(env.feature_names),
        "step_s": env.step_s,
        "warmup_s": env.warmup_s,
    }


def train(
    scenario: Scenario,
    out: Path,
    *,
    timesteps: int,
    n_envs: int = 8,
    seed: int = 1,
    scenario_name: str | None = None,
    step_s: float = 1.0,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    learning_rate: float = 3e-4,
    torch_threads: int = 4,
    verbose: int = 1,
) -> dict[str, Any]:
    """Train PPO on the scenario and save `model.zip` and `model.json` in `out`.

    `n_steps` (per environment, between two updates) defaults to two episodes
    and `batch_size` to a twelfth of a rollout. Returns the sidecar.
    """
    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure

    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(torch_threads)  # the cores are for the simulations
    probe = RetryEnv(scenario, step_s=step_s)
    n_steps = n_steps or 2 * probe.episode_steps
    batch_size = batch_size or max(32, n_steps * n_envs // 12)

    envs = make_envs(scenario, n_envs, seed, monitor_dir=out / "monitor", step_s=step_s)
    model = PPO(
        "MlpPolicy",
        envs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        seed=seed,
        device="cpu",
        verbose=verbose,
    )
    model.set_logger(configure(str(out), ["stdout", "csv"] if verbose else ["csv"]))
    started = time.monotonic()
    try:
        model.learn(total_timesteps=timesteps, callback=_summary_logger())
    finally:
        envs.close()
    wall_s = time.monotonic() - started
    if model.num_timesteps < timesteps:
        raise RuntimeError(f"asked for {timesteps} steps, trained {model.num_timesteps}")
    model.save(out / "model.zip")

    sidecar = {
        "scenario": scenario_name,
        "scenario_settings": _settings(scenario),
        **_describe(probe),
        "timesteps_asked": timesteps,
        "timesteps_trained": int(model.num_timesteps),
        "episodes_trained": int(model.num_timesteps) // probe.episode_steps,
        "wall_s": round(wall_s, 1),
        "steps_per_s": round(model.num_timesteps / wall_s, 1),
        "ppo": {
            "n_envs": n_envs,
            "seed": seed,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
        },
        "msim_commit": _commit(),
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    text = json.dumps(sidecar, indent=2, default=str)
    (out / "model.json").write_text(text + "\n", "utf-8")
    return json.loads(text)  # as it reads back: tuples are lists


def load_policy(model: Path, env: RetryEnv) -> Policy:
    """The saved model as a policy for `env`, which must be what the model was trained on.

    Its sidecar is checked against the environment: the same knobs with the
    same levels in the same order, the same observation, the same step. A
    model's file alone only knows the SHAPES of its spaces, so without this a
    model would run happily on an action it reads as another setting.
    """
    from stable_baselines3 import PPO

    sidecar = json.loads(model.with_suffix(".json").read_text("utf-8"))
    here = _describe(env)
    for name, value in here.items():
        if sidecar[name] != value:
            raise ValueError(
                f"{model} was trained with {name} = {sidecar[name]!r}, "
                f"this environment has {value!r}"
            )
    trained = PPO.load(model, device="cpu")

    def policy(observation: np.ndarray) -> Sequence[int]:
        action, _ = trained.predict(observation, deterministic=True)
        return action.tolist()

    return policy


def _run_model(task: tuple[Path, Scenario, dict[str, Any], tuple[int, ...]]):
    model, scenario, env_arguments, seeds = task
    import torch

    torch.set_num_threads(1)  # one small forward pass per step: threads only get in the way
    env = RetryEnv(scenario, **env_arguments)
    return evaluate.rollout(env, load_policy(model, env), seeds)


def evaluate_model(
    model: Path,
    seeds: Sequence[int] = evaluate.TEST_SEEDS,
    *,
    scenario: Scenario | None = None,
    jobs: int = 1,
) -> list[dict[str, float]]:
    """The saved model on the given episodes, acting greedily: one summary per episode.

    The scenario is the one named in the model's sidecar unless given (a model
    trained on a scenario object that has no name needs it given).
    """
    sidecar = json.loads(model.with_suffix(".json").read_text("utf-8"))
    if scenario is None:
        if sidecar["scenario"] is None:
            raise ValueError(
                f"{model} does not name its scenario: pass the scenario it was trained on"
            )
        scenario = evaluate.scenario_from(sidecar["scenario"])
    env_arguments = {"step_s": sidecar["step_s"], "warmup_s": sidecar["warmup_s"]}
    seeds = tuple(seeds)
    chunks = [seeds[i::jobs] for i in range(max(1, min(jobs, len(seeds))))]
    tasks = [(model, scenario, env_arguments, chunk) for chunk in chunks]
    episodes = [e for part in evaluate.in_parallel(_run_model, tasks, jobs) for e in part]
    return sorted(episodes, key=lambda episode: episode["seed"])


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a retry policy with PPO and compare it with the fixed settings."
    )
    parser.add_argument("scenario", help="single, storm, or arolla:<file>:<agent>[,<agent>...]")
    parser.add_argument("--out", type=Path, required=True, help="where the run is written")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8, help="simulations run side by side")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, help="steps per environment between updates")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--jobs", type=int, help="processes for the comparison (default: n-envs)")
    parser.add_argument("--test-seeds", type=int, default=len(evaluate.TEST_SEEDS))
    parser.add_argument("--selection-seeds", type=int, default=len(evaluate.SELECTION_SEEDS))
    parser.add_argument("--no-comparison", action="store_true", help="train and save only")
    args = parser.parse_args(arguments)

    scenario = evaluate.scenario_from(args.scenario)
    sidecar = train(
        scenario,
        args.out,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        scenario_name=args.scenario,
        step_s=args.step_s,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
    )
    print(
        f"trained {sidecar['timesteps_trained']} steps ({sidecar['episodes_trained']} episodes) "
        f"in {sidecar['wall_s']} s: {sidecar['steps_per_s']} steps/s on {args.n_envs} simulations"
    )
    if args.no_comparison:
        return

    jobs = args.jobs or args.n_envs
    first_test = evaluate.TEST_SEEDS[0]
    test = range(first_test, first_test + args.test_seeds)
    first_selection = evaluate.SELECTION_SEEDS[0]
    selection = range(first_selection, first_selection + args.selection_seeds)
    runs, search = evaluate.baselines(
        scenario, selection_seeds=selection, test_seeds=test, jobs=jobs, step_s=args.step_s
    )
    model = args.out / "model.zip"
    runs = {"ppo": evaluate_model(model, test, scenario=scenario, jobs=jobs), **runs}
    rows = evaluate.compare(runs, reference="best static")
    best = ", ".join(f"{name}={value:g}" for name, value in search[0].setting.items())
    print(f"\non {len(test)} test episodes (best static: {best})")
    print(evaluate.table(rows, reference="best static"))
    evaluate.write_csv(args.out / "comparison.csv", rows)
    evaluate.write_csv(
        args.out / "episodes.csv",
        [{"policy": name, **episode} for name, episodes in runs.items() for episode in episodes],
    )
    evaluate.write_csv(
        args.out / "static.csv",
        [
            {**s.setting, **{m: evaluate.mean(s.episodes, m) for m in evaluate.METRICS}}
            for s in search
        ],
    )


if __name__ == "__main__":
    main()
