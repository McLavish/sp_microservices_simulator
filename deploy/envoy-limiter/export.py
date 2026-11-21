"""A trained limiter (`msim.rl.limiter`) as the configuration of the Envoy filter next to this file.

    python deploy/envoy-limiter/export.py MODEL.zip OUT.json --rates R [R ...]
        [--tenants NAME [NAME ...]] [--fixture FILE]

Writes the `limiter` block of the filter's plugin configuration: the tenants in the policy's
order, the requests per second each sends one PROXY when all is well (`--rates`, the scale of
everything the policy is shown), the ladder of shares, the level held until the policy has
spoken, the window, the time constant of the slow averages, and the policy's weights
(`msim.rl.train.policy_weights`).

`--tenants` names the tenants as the deployment's requests name them (the model knows them by
position). A model trained for ONE tenant can be given several: the filter then asks it once
per tenant, each with its own rate. `--fixture` also writes random observations with the
actions the model takes on them, which `cargo test` checks the filter's forward pass against.

Needs the `train` extra (Stable-Baselines3), as training does.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from msim.rl import train


def limiter_block(model: Path, tenants: list[str] | None, rates: list[float]) -> dict:
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
    return {
        "tenants": tenants,
        "rates": rates,
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
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()

    limiter = limiter_block(args.model, args.tenants, args.rates)
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
