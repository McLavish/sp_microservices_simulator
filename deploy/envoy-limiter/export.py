"""A trained limiter (`msim.rl.limiter`) as the configuration of the Envoy filter next to this file.

    python deploy/envoy-limiter/export.py MODEL.zip OUT.json --rates R [R ...]
        [--tenants NAME [NAME ...]] [--one-level] [--rest] [--in-flight-cap C] [--step-up N]
        [--asked-per-failure A] [--max-attempt N] [--confirm]
        [--hand-rule ALIVE DOWN_ABOVE [HOLD_ABOVE]] [--fixture FILE]

Writes the `limiter` block of the filter's plugin configuration: the tenants in the policy's
order, the requests per second each sends one PROXY when all is well (`--rates`, the scale of
everything the policy is shown), the ladder of shares, the probe floor and the resting quota
the model was trained with (`--rest`, `--in-flight-cap`, `--step-up`, `--asked-per-failure`,
`--max-attempt` and `--confirm` turn those on for a model trained without them), the
level held until the policy has spoken, the window, the time constant of the slow
averages, and the policy's weights (`msim.rl.train.policy_weights`).

`--tenants` names the tenants as the deployment's requests name them (the model knows them by
position). A model trained for ONE tenant can be given several: the filter then asks it once
per tenant, each with its own rate, or, with `--one-level`, once for all of them on their
counts added up, and gives every tenant that level (retries are then let in in proportion to
the requests each tenant sends). `--fixture` also writes random observations with the
actions the model takes on them, which `cargo test` checks the filter's forward pass against.

Needs the `train` extra (Stable-Baselines3), as training does.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from msim.rl import train


def trained_grid(sidecar: dict) -> dict:
    """The `QuotaGrid` of the scenario the model was trained on, as its sidecar has it."""
    settings = sidecar.get("scenario_settings") or {}
    first = (settings.get("scenarios") or [settings])[0]  # one of several, or the only one
    return first.get("grid") or {}


def limiter_block(
    model: Path,
    tenants: list[str] | None,
    rates: list[float],
    one_level: bool = False,
    rest: bool = False,
    in_flight_cap: float | None = None,
    step_up: int | None = None,
    asked_per_failure: float | None = None,
    max_attempt: int | None = None,
    confirm: bool = False,
    hand_rule: list[float] | None = None,
) -> dict:
    sidecar = json.loads(model.with_suffix(".json").read_text("utf-8"))
    knobs = sidecar["knobs"]
    trained = [knob["name"].removeprefix("quota:") for knob in knobs]
    if any(not knob["name"].startswith("quota:") for knob in knobs):
        raise SystemExit(f"{model} is no limiter: its knobs are {[k['name'] for k in knobs]}")
    shares = knobs[0]["values"]
    if any(knob["values"] != shares or knob["default"] != knobs[0]["default"] for knob in knobs):
        raise SystemExit("the filter keeps one ladder of shares and one default for every tenant")
    tenants = tenants or trained
    if len(trained) not in (1, len(tenants)):
        raise SystemExit(f"the model decides for {len(trained)} tenants, {len(tenants)} were named")
    if len(rates) != len(tenants):
        raise SystemExit(f"{len(tenants)} tenants need as many rates, got {len(rates)}")
    if one_level and len(trained) != 1:
        raise SystemExit("one level for every tenant needs a model that decides for one tenant")
    return {
        "tenants": tenants,
        "rates": rates,
        "one_level": one_level,
        "floor": int(trained_grid(sidecar).get("floor", 0)),
        "rest": bool(rest or trained_grid(sidecar).get("rest", False)),
        "in_flight": bool(trained_grid(sidecar).get("in_flight", False)),
        "step_up": int(step_up or trained_grid(sidecar).get("step_up", 0)),
        "hold_above": trained_grid(sidecar).get("hold_above"),
        "in_flight_cap": in_flight_cap or trained_grid(sidecar).get("in_flight_cap"),
        "asked_per_failure": asked_per_failure or trained_grid(sidecar).get("asked_per_failure"),
        "max_attempt": max_attempt or trained_grid(sidecar).get("max_attempt"),
        "confirm": bool(confirm or trained_grid(sidecar).get("confirm", False)),
        # `rl.limiter.hand_rule` in place of the policy: ALIVE DOWN_ABOVE [HOLD_ABOVE].
        "hand_rule": None
        if hand_rule is None
        else {
            "alive": hand_rule[0],
            "down_above": hand_rule[1],
            "hold_above": hand_rule[2] if len(hand_rule) > 2 else None,
        },
        "shares": shares,
        "default_level": knobs[0]["default"],
        "window_ms": round(1000 * sidecar["step_s"]),
        "slow_s": sidecar.get("slow_s") or 0.0,
        "policy": train.policy_weights(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A trained limiter as the configuration of the Envoy filter."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--rates", nargs="+", type=float, required=True)
    parser.add_argument("--tenants", nargs="+")
    parser.add_argument("--one-level", action="store_true")
    parser.add_argument("--rest", action="store_true")
    parser.add_argument("--in-flight-cap", type=float, default=None)
    parser.add_argument("--step-up", type=int, default=None)
    parser.add_argument("--asked-per-failure", type=float, default=None)
    parser.add_argument("--max-attempt", type=int, default=None)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument(
        "--hand-rule",
        type=float,
        nargs="+",
        metavar="X",
        help="decide by the hand-written rule instead of the policy: ALIVE DOWN_ABOVE [HOLD_ABOVE]",
    )
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()

    limiter = limiter_block(
        args.model,
        args.tenants,
        args.rates,
        args.one_level,
        args.rest,
        args.in_flight_cap,
        args.step_up,
        args.asked_per_failure,
        args.max_attempt,
        args.confirm,
        args.hand_rule,
    )
    args.out.write_text(json.dumps(limiter), "utf-8")
    entries = len(limiter["policy"]["hidden"][0]["w"][0])
    print(
        f"{args.model}: {len(limiter['tenants'])} tenants, {entries} entries per decision, "
        f"{len(limiter['shares'])} levels -> {args.out} ({args.out.stat().st_size} bytes)"
    )
    if args.fixture:
        observations = np.random.default_rng(1).random((64, entries), dtype=np.float32)
        fixture = {
            "limiter": limiter,
            "observations": observations.astype(float).tolist(),
            "actions": train.greedy_actions(args.model, observations),
        }
        args.fixture.write_text(json.dumps(fixture), "utf-8")
        print(f"fixture: {len(observations)} observations -> {args.fixture}")


if __name__ == "__main__":
    main()
