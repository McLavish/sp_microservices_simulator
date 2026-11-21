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
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium
import numpy as np

from msim.rl import evaluate
from msim.rl.env import RetryEnv
from msim.rl.evaluate import Policy
from msim.rl.scenario import Scenario

#: What of the episode summary goes into the training log, episode by episode.
LOGGED = ("ep_success_rate", "ep_fault_success_rate", "ep_amplification")


class _Offset(gymnasium.RewardWrapper):
    """What the learner is paid: the environment's reward minus a constant.

    The environment pays about 1 for every healthy window, and an episode is
    cut at its horizon, not ended, so the learner bootstraps over the cut: the
    value of a healthy state is then about `1 / (1 - gamma)`, a hundred at the
    usual discount, and the value function spends the first part of training
    climbing there while the advantages it yields are mostly its own error.
    With 1 taken off, a healthy window pays 0 and only what a fault costs is
    left to learn. All episodes of a scenario have the same number of steps,
    so the constant changes no comparison between two policies.
    """

    def __init__(self, env: gymnasium.Env, offset: float) -> None:
        super().__init__(env)
        self.offset = offset

    def reward(self, reward: SupportsFloat) -> float:
        return float(reward) - self.offset


def make_envs(
    scenario: Scenario,
    n_envs: int,
    seed: int,
    monitor_dir: Path | None = None,
    reward_offset: float = 0.0,
    **env: Any,
):
    """`n_envs` environments on the scenario, each in its own process when there are several.

    Environment `i` is seeded with `seed + i` at its first reset and draws
    every later episode from its own generator, so no two of them see the same
    episodes. `reward_offset` is taken off every reward the learner is given
    (`_Offset`); what is logged per episode stays the environment's own return.
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
        # Outside the monitor, which therefore records what the environment paid.
        wrapper_class=partial(_Offset, offset=reward_offset) if reward_offset else None,
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


def _decaying(progress_remaining: float, start: float) -> float:
    """A learning rate that falls linearly from `start` to zero over a run.

    A module-level function and not a lambda: the schedule is saved with the model.
    """
    return start * progress_remaining


def _snapshots(every: int, folder: Path, sidecar: Callable[[int], str]):
    """A callback that saves the model every `every` steps, each copy with a sidecar.

    A copy is taken where a rollout starts, so it is the model as the updates on
    every step counted so far left it; its name carries that count.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class Snapshots(BaseCallback):
        due = every

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            if self.num_timesteps < self.due:
                return
            while self.due <= self.num_timesteps:
                self.due += every
            path = folder / f"model-{self.num_timesteps}.zip"
            self.model.save(path)
            path.with_suffix(".json").write_text(sidecar(self.num_timesteps), "utf-8")

    return Snapshots()


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
        "slow_s": env.slow_s,
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
    reward_offset: float = 1.0,
    device: str = "auto",
    torch_threads: int = 4,
    verbose: int = 1,
    snapshots: int = 0,
    slow_s: float | None = None,
    lr_decay: bool = False,
    reward_within_s: float | None = None,
    retry_cost: float = 0.0,
) -> dict[str, Any]:
    """Train PPO on the scenario and save `model.zip` and `model.json` in `out`.

    `n_steps` (per environment, between two updates) defaults to two episodes
    and `batch_size` to a twelfth of a rollout. `reward_offset` is what the
    learner's reward is shifted down by (`_Offset`); the logs and every
    evaluation use the environment's own reward. `snapshots` asks for that
    many copies of the model on the way, evenly spaced
    (`snapshots/model-<steps>.zip`, each with a sidecar of its own). `slow_s`
    is the environment's (`RetryEnv.slow_s`). `lr_decay` lets the learning rate
    fall linearly to zero over the run, so that the policy settles at the end
    instead of moving as much as on the first day. `reward_within_s` is the
    environment's latency bound on what the reward counts
    (`RetryEnv.reward_within_s`) and `retry_cost` its price of a retry that
    reaches its target; both are written to the sidecar and are no part of
    what a model is matched against, since they change neither what the agent
    is shown nor what its actions mean. Returns the sidecar.
    """
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure

    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(torch_threads)  # the cores are for the simulations
    probe = RetryEnv(
        scenario,
        step_s=step_s,
        slow_s=slow_s,
        reward_within_s=reward_within_s,
        retry_cost=retry_cost,
    )
    n_steps = n_steps or 2 * probe.episode_steps
    batch_size = batch_size or max(32, n_steps * n_envs // 12)

    envs = make_envs(
        scenario,
        n_envs,
        seed,
        monitor_dir=out / "monitor",
        reward_offset=reward_offset,
        step_s=step_s,
        slow_s=slow_s,
        reward_within_s=reward_within_s,
        retry_cost=retry_cost,
    )
    model = PPO(
        "MlpPolicy",
        envs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        learning_rate=partial(_decaying, start=learning_rate) if lr_decay else learning_rate,
        seed=seed,
        device=device,
        verbose=verbose,
    )
    model.set_logger(configure(str(out), ["stdout", "csv"] if verbose else ["csv"]))
    commit = _commit()  # of the code that is running: read before a long run, not after
    started = time.monotonic()

    def sidecar(steps: int) -> str:
        """What a model saved after `steps` steps was trained on, as its file reads."""
        wall_s = time.monotonic() - started
        return (
            json.dumps(
                {
                    "scenario": scenario_name,
                    "scenario_settings": _settings(scenario),
                    **_describe(probe),
                    "reward_within_s": reward_within_s,
                    "retry_cost": retry_cost,
                    "timesteps_asked": timesteps,
                    "timesteps_trained": steps,
                    "episodes_trained": steps // probe.episode_steps,
                    "wall_s": round(wall_s, 1),
                    "steps_per_s": round(steps / wall_s, 1),
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
                        "lr_decay": lr_decay,
                        "reward_offset": reward_offset,
                        "device": str(model.device),
                    },
                    "msim_commit": commit,
                    "versions": {
                        "python": platform.python_version(),
                        "gymnasium": gymnasium.__version__,
                        "stable_baselines3": stable_baselines3.__version__,
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                    },
                },
                indent=2,
                default=str,
            )
            + "\n"
        )

    callbacks = [_summary_logger()]
    if snapshots > 0:
        (out / "snapshots").mkdir(exist_ok=True)
        every = max(1, timesteps // (snapshots + 1))
        callbacks.append(_snapshots(every, out / "snapshots", sidecar))
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks)
    finally:
        envs.close()
    if model.num_timesteps < timesteps:
        raise RuntimeError(f"asked for {timesteps} steps, trained {model.num_timesteps}")
    model.save(out / "model.zip")
    text = sidecar(int(model.num_timesteps))
    (out / "model.json").write_text(text, "utf-8")
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
        # `.get`: a sidecar written before a setting existed was trained without it.
        if sidecar.get(name) != value:
            raise ValueError(
                f"{model} was trained with {name} = {sidecar.get(name)!r}, "
                f"this environment has {value!r}"
            )
    trained = PPO.load(model, device="cpu")

    def policy(observation: np.ndarray) -> Sequence[int]:
        action, _ = trained.predict(observation, deterministic=True)
        return action.tolist()

    return policy


def policy_weights(model: Path) -> dict[str, Any]:
    """The saved model's policy network as plain numbers, for an evaluator that is not Python.

    The network `train` gives PPO: tanh layers, then one linear layer whose
    outputs are the logits of every knob's levels, one knob after the other.
    Acting greedily is the largest logit of each knob (the first of equals).
    Returned as `{"hidden": [{"w": rows, "b": [...]}, ...], "head": {...}}`,
    `w[out][in]`, each number with the nine significant digits that give a
    32-bit weight back exactly. `greedy_actions` is what such an evaluator has
    to reproduce; `deploy/envoy-limiter` is one (a limiter as an Envoy filter).
    """
    from stable_baselines3 import PPO

    trained = PPO.load(model, device="cpu")

    def layer(module: Any) -> dict[str, Any]:
        def short(values: Any) -> list[float]:
            return [float(f"{float(value):.9g}") for value in values]

        return {
            "w": [short(row) for row in module.weight.detach().numpy()],
            "b": short(module.bias.detach().numpy()),
        }

    network = trained.policy.mlp_extractor.policy_net
    return {
        "hidden": [layer(module) for module in network if hasattr(module, "weight")],
        "head": layer(trained.policy.action_net),
    }


def greedy_actions(model: Path, observations: np.ndarray) -> list[list[int]]:
    """What the saved model does on each row of `observations`, acting greedily.

    On raw observations, with no environment to match: for checking another
    evaluator of `policy_weights` against the model itself.
    """
    from stable_baselines3 import PPO

    actions, _ = PPO.load(model, device="cpu").predict(observations, deterministic=True)
    return np.asarray(actions).reshape(len(observations), -1).tolist()


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
    reward_within_s: float | None = None,
) -> list[dict[str, float]]:
    """The saved model on the given episodes, acting greedily: one summary per episode.

    The scenario is the one named in the model's sidecar unless given (a model
    trained on a scenario object that has no name needs it given).
    `reward_within_s` is the latency bound the episodes are SCORED with
    (`return`, `on_time_rate`), whatever the model was trained with.
    """
    sidecar = json.loads(model.with_suffix(".json").read_text("utf-8"))
    if scenario is None:
        if sidecar["scenario"] is None:
            raise ValueError(
                f"{model} does not name its scenario: pass the scenario it was trained on"
            )
        scenario = evaluate.scenario_from(sidecar["scenario"])
    env_arguments = {
        "step_s": sidecar["step_s"],
        "warmup_s": sidecar["warmup_s"],
        "slow_s": sidecar.get("slow_s"),
        "reward_within_s": reward_within_s,
    }
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
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--reward-offset",
        type=float,
        default=1.0,
        help="taken off the reward the learner is given (a healthy window pays about 1)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-decay", action="store_true", help="let it fall to zero over the run")
    parser.add_argument(
        "--device",
        default="auto",
        help="where the learner runs: cpu, cuda, or auto (cuda when this PyTorch build has it)",
    )
    parser.add_argument("--jobs", type=int, help="processes for the comparison (default: n-envs)")
    parser.add_argument("--test-seeds", type=int, default=len(evaluate.TEST_SEEDS))
    parser.add_argument("--selection-seeds", type=int, default=len(evaluate.SELECTION_SEEDS))
    parser.add_argument("--snapshots", type=int, default=0, help="copies of the model on the way")
    parser.add_argument(
        "--slow-s", type=float, help="also show the measurements averaged over about this long"
    )
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
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        reward_offset=args.reward_offset,
        device=args.device,
        snapshots=args.snapshots,
        slow_s=args.slow_s,
        lr_decay=args.lr_decay,
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
