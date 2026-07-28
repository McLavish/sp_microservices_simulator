"""Standalone matched hotelReservation scenario for qualitative validation.

This is intentionally approximate: the real benchmark remains the reference.
The scenario maps the hotel read workload onto two chains sharing profile:

* hotels: root -> search -> {geo, rate}, plus reservation and profile
* recommendations: root -> recommendation and profile

It consumes the one-way calibration JSON emitted by
`valharness.orchestrator.record_real` when available.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caller import ClientConfig, ClientRuntime
from chain import DownstreamEdge, ServiceNodeRuntime
from core import TimeInterval, ms_to_ns, s_to_ns
from fault_events import FaultEventsTracker
from faults import LatencyInjection, LoadSpike, PartialFailure
from metrics import Metrics
from retry_policies import ExponentialBackoffWithJitterRetryPolicy, JitterMode, NoRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload

DEFAULT_SERVICE_PARAMS = {
    "hotels-root": {"median_ms": 0.2, "sigma": 0.2, "workers": 128, "queue_capacity": 4096},
    "recommendations-root": {
        "median_ms": 0.2,
        "sigma": 0.2,
        "workers": 128,
        "queue_capacity": 4096,
    },
    "search": {"median_ms": 1.5, "sigma": 0.4, "workers": 64, "queue_capacity": 2048},
    "geo": {"median_ms": 1.0, "sigma": 0.4, "workers": 64, "queue_capacity": 2048},
    "rate": {"median_ms": 1.0, "sigma": 0.4, "workers": 64, "queue_capacity": 2048},
    "reservation": {"median_ms": 1.0, "sigma": 0.4, "workers": 64, "queue_capacity": 2048},
    "profile": {"median_ms": 1.5, "sigma": 0.5, "workers": 32, "queue_capacity": 1024},
    "recommendation": {"median_ms": 1.0, "sigma": 0.4, "workers": 64, "queue_capacity": 2048},
}

DEFAULT_PROFILE_FAULT = [(20.0, 40.0, 50.0, 0.2)]


@dataclass(frozen=True)
class HotelMatchedConfig:
    output_dir: Path = Path("hotel_matched_results")
    calibration_json: Path | None = None
    # None -> take the calibration JSON's recorded workload when one is given,
    # else the built-in default. Explicit CLI values always win.
    base_rps: float | None = None
    duration_s: float | None = None
    # None -> derive from the calibration's recorded request mix, else default.
    recommendation_fraction: float | None = None
    seed: int = 1234
    profile_faults: list[tuple[float, float, float, float]] | None = None
    load_spikes: list[tuple[float, float, float]] | None = None
    leaf_max_attempts: int = 3


@dataclass
class HotelMatchedRun:
    hotels_client: ClientRuntime
    recommendations_client: ClientRuntime
    per_service_clients: dict[str, list[ClientRuntime]]
    fault_tracker: FaultEventsTracker


_DEFAULT_BASE_RPS = 100.0
_DEFAULT_DURATION_S = 60.0


_RECOMMENDATION_PATHS = ("/recommendation", "recommendation")
_DEFAULT_RECOMMENDATION_FRACTION = 0.25


def _recommendation_fraction_from(recorded: dict) -> float:
    """Recommendation share of the recorded request mix.

    `workload.path` maps request path -> count (or share). Falls back to the
    built-in default when the recording says nothing about the mix.
    """
    paths = recorded.get("path")
    if not isinstance(paths, dict) or not paths:
        return _DEFAULT_RECOMMENDATION_FRACTION
    total = 0.0
    recommendation = 0.0
    for path, count in paths.items():
        value = float(count)
        if value < 0:
            raise ValueError(f"calibration workload.path[{path!r}] must be >= 0, got {value}")
        total += value
        if any(str(path).startswith(prefix) for prefix in _RECOMMENDATION_PATHS):
            recommendation += value
    if total <= 0:
        return _DEFAULT_RECOMMENDATION_FRACTION
    return recommendation / total


def _validated_base_rps(value: float | int | str | None) -> float:
    """Shared bound check for a calibration-supplied arrival rate."""
    if value is None:
        return _DEFAULT_BASE_RPS
    rate = float(value)
    if not math.isfinite(rate) or rate < 0:
        raise ValueError(f"calibration workload.base_rps must be finite and >= 0, got {value!r}")
    return rate


def _validated_duration(value: float | int | str | None) -> float:
    if value is None:
        return _DEFAULT_DURATION_S
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"calibration workload.duration_s must be finite and > 0, got {value!r}")
    return duration


def run_hotel_matched(config: HotelMatchedConfig) -> dict[str, str]:
    calibration = _load_calibration(config.calibration_json)
    profile_faults = _effective_profile_faults(config, calibration)
    load_spikes = (
        config.load_spikes if config.load_spikes is not None else _load_spikes(calibration)
    )

    sim = Simulator(seed=config.seed)
    run = _build_runtime(
        sim,
        calibration=calibration,
        profile_faults=profile_faults,
        seed=config.seed,
        leaf_max_attempts=config.leaf_max_attempts,
    )

    # Resolve the workload: an explicit CLI value wins, else the calibration's
    # own recording, else the built-in default. Ignoring the recorded workload
    # (the previous behaviour) ran the "matched" scenario at 100 rps against
    # per-service workers/queue_capacity taken from that same file - an
    # order-of-magnitude offered-load error that invalidates the comparison the
    # module exists to make.
    recorded = calibration.get("workload") or {}
    base_rps = config.base_rps
    if base_rps is None:
        base_rps = _validated_base_rps(recorded.get("base_rps"))
    duration_s = config.duration_s
    if duration_s is None:
        duration_s = _validated_duration(recorded.get("duration_s"))
    # The MIX matters as much as the rate: a recommendations-only recording that
    # still generates search traffic is not a matched comparison. Prefer the
    # recorded fraction (derived from workload.path counts when present) unless
    # the CLI set one explicitly.
    fraction = config.recommendation_fraction
    if fraction is None:
        fraction = _recommendation_fraction_from(recorded)
    if not 0 <= fraction <= 1:
        raise ValueError(
            f"recommendation_fraction must be in [0, 1], got {fraction}; outside "
            "that range the search rate goes negative and is silently clamped to 0"
        )
    recommendation_rps = base_rps * fraction
    hotels_rps = base_rps - recommendation_rps
    sim_load_spikes = [
        LoadSpike(
            duration=TimeInterval(begin=s_to_ns(start_s), end=s_to_ns(end_s)),
            rps_multiplier=multiplier,
        )
        for start_s, end_s, multiplier in load_spikes
    ]

    hotels_workload = Workload(
        base_rps=hotels_rps,
        duration_s=duration_s,
        load_spikes=sim_load_spikes,
        rng_seed=config.seed,
    )
    recommendation_workload = Workload(
        base_rps=recommendation_rps,
        duration_s=duration_s,
        load_spikes=sim_load_spikes,
        rng_seed=config.seed + 1,
    )
    hotels_workload.register_fault_events(run.fault_tracker)
    hotels_workload.drive(sim, run.hotels_client.start_request)
    recommendation_workload.drive(sim, run.recommendations_client.start_request)

    sim.run(until=s_to_ns(duration_s))
    sim.run()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "root_csv": output_dir / "hotel_matched_root.csv",
        "fault_events_json": output_dir / "hotel_matched_fault_events.json",
    }

    root_metrics = _combined_metrics([run.hotels_client, run.recommendations_client])
    root_metrics.export_csv(str(outputs["root_csv"]), granularity_s=1.0)
    for service_name, clients in run.per_service_clients.items():
        path = output_dir / f"svc-{service_name}_output.csv"
        _combined_metrics(clients).export_csv(str(path), granularity_s=1.0)
        outputs[f"svc_{service_name}_csv"] = path

    run.fault_tracker.export_json(str(outputs["fault_events_json"]))
    return {key: str(value) for key, value in outputs.items()}


def _build_runtime(
    sim: Simulator,
    *,
    calibration: dict[str, Any],
    profile_faults: list[tuple[float, float, float, float]],
    seed: int,
    leaf_max_attempts: int = 3,
) -> HotelMatchedRun:
    fault_tracker = FaultEventsTracker()

    profile_injections = [
        LatencyInjection(
            duration=TimeInterval(begin=s_to_ns(start_s), end=s_to_ns(end_s)),
            add_latency=ms_to_ns(add_latency_ms),
        )
        for start_s, end_s, add_latency_ms, _p_fail in profile_faults
        if add_latency_ms > 0
    ]
    profile_failures = [
        PartialFailure(
            duration=TimeInterval(begin=s_to_ns(start_s), end=s_to_ns(end_s)),
            p_fail=p_fail,
        )
        for start_s, end_s, _add_latency_ms, p_fail in profile_faults
        if p_fail > 0
    ]

    cfgs = {
        name: _service_cfg(
            f"svc-{name}",
            name,
            calibration,
            latency_injections=profile_injections if name == "profile" else None,
            partial_failures=profile_failures if name == "profile" else None,
        )
        for name in DEFAULT_SERVICE_PARAMS
    }
    for cfg in cfgs.values():
        cfg.register_fault_events(fault_tracker)

    nodes = {name: ServiceNodeRuntime(ServiceRuntime(cfg=cfg).bind()) for name, cfg in cfgs.items()}

    policies = _client_policies(seed, leaf_max_attempts=leaf_max_attempts)

    hotels_to_search = ClientRuntime(policies["aggregator_no_retry"], nodes["search"])
    search_to_geo = ClientRuntime(policies["search_downstream_geo"], nodes["geo"])
    search_to_rate = ClientRuntime(policies["search_downstream_rate"], nodes["rate"])
    hotels_to_reservation = ClientRuntime(policies["leaf_reservation"], nodes["reservation"])
    hotels_to_profile = ClientRuntime(policies["leaf_hotels_profile"], nodes["profile"])
    recommendations_to_recommendation = ClientRuntime(
        policies["leaf_recommendation"],
        nodes["recommendation"],
    )
    recommendations_to_profile = ClientRuntime(
        policies["leaf_recommendations_profile"],
        nodes["profile"],
    )

    nodes["search"].add_downstream(DownstreamEdge(client=search_to_geo, target=nodes["geo"]))
    nodes["search"].add_downstream(DownstreamEdge(client=search_to_rate, target=nodes["rate"]))
    nodes["hotels-root"].add_downstream(
        DownstreamEdge(client=hotels_to_search, target=nodes["search"])
    )
    nodes["hotels-root"].add_downstream(
        DownstreamEdge(client=hotels_to_reservation, target=nodes["reservation"])
    )
    nodes["hotels-root"].add_downstream(
        DownstreamEdge(client=hotels_to_profile, target=nodes["profile"])
    )
    nodes["recommendations-root"].add_downstream(
        DownstreamEdge(
            client=recommendations_to_recommendation,
            target=nodes["recommendation"],
        )
    )
    nodes["recommendations-root"].add_downstream(
        DownstreamEdge(client=recommendations_to_profile, target=nodes["profile"])
    )

    hotels_client = ClientRuntime(policies["top"], nodes["hotels-root"])
    recommendations_client = ClientRuntime(policies["top"], nodes["recommendations-root"])

    return HotelMatchedRun(
        hotels_client=hotels_client,
        recommendations_client=recommendations_client,
        per_service_clients={
            "search": [hotels_to_search],
            "geo": [search_to_geo],
            "rate": [search_to_rate],
            "reservation": [hotels_to_reservation],
            "profile": [hotels_to_profile, recommendations_to_profile],
            "recommendation": [recommendations_to_recommendation],
        },
        fault_tracker=fault_tracker,
    )


def _client_policies(seed: int, *, leaf_max_attempts: int = 3) -> dict[str, ClientConfig]:
    return {
        "top": _no_retry_cfg(attempt_timeout_ms=1000),
        "aggregator_no_retry": _no_retry_cfg(attempt_timeout_ms=900),
        "search_downstream_geo": _exp_jitter_cfg(
            seed=seed + 101,
            attempt_timeout_ms=500,
            max_attempts=3,
            initial_delay_ms=20,
            max_delay_ms=150,
        ),
        "search_downstream_rate": _exp_jitter_cfg(
            seed=seed + 102,
            attempt_timeout_ms=500,
            max_attempts=3,
            initial_delay_ms=20,
            max_delay_ms=150,
        ),
        "leaf_reservation": _leaf_read_cfg(seed=seed + 201, max_attempts=leaf_max_attempts),
        "leaf_hotels_profile": _leaf_read_cfg(seed=seed + 202, max_attempts=leaf_max_attempts),
        "leaf_recommendation": _leaf_read_cfg(seed=seed + 203, max_attempts=leaf_max_attempts),
        "leaf_recommendations_profile": _leaf_read_cfg(
            seed=seed + 204, max_attempts=leaf_max_attempts
        ),
    }


def _leaf_read_cfg(seed: int, max_attempts: int = 3) -> ClientConfig:
    return _exp_jitter_cfg(
        seed=seed,
        attempt_timeout_ms=750,
        max_attempts=max_attempts,
        initial_delay_ms=25,
        max_delay_ms=200,
    )


def _no_retry_cfg(attempt_timeout_ms: int) -> ClientConfig:
    return ClientConfig(
        retry=NoRetryPolicy(),
        timeout_controller=StaticTimeoutController(
            global_timeout=None,
            attempt_timeout=ms_to_ns(attempt_timeout_ms),
        ),
    )


def _exp_jitter_cfg(
    *,
    seed: int,
    attempt_timeout_ms: int,
    max_attempts: int,
    initial_delay_ms: int,
    max_delay_ms: int,
) -> ClientConfig:
    return ClientConfig(
        retry=ExponentialBackoffWithJitterRetryPolicy(
            max_attempts=max_attempts,
            initial_delay=ms_to_ns(initial_delay_ms),
            max_delay=ms_to_ns(max_delay_ms),
            rng=random.Random(seed),
            jitter_mode=JitterMode.EQUAL,
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None,
            attempt_timeout=ms_to_ns(attempt_timeout_ms),
        ),
    )


def _service_cfg(
    sim_name: str,
    real_name: str,
    calibration: dict[str, Any],
    *,
    latency_injections: list[LatencyInjection] | None = None,
    partial_failures: list[PartialFailure] | None = None,
) -> ServiceConfig:
    params = _service_params(real_name, calibration)
    return ServiceConfig(
        name=sim_name,
        latency_median=ms_to_ns(max(0.1, float(params["median_ms"]))),
        latency_lognorm_sigma=max(1e-6, float(params["sigma"])),
        workers=max(1, int(params["workers"])),
        queue_capacity=None
        if params.get("queue_capacity") is None
        else max(1, int(params["queue_capacity"])),
        latency_injections=latency_injections or [],
        partial_failures=partial_failures or [],
    )


def _service_params(real_name: str, calibration: dict[str, Any]) -> dict[str, Any]:
    defaults = DEFAULT_SERVICE_PARAMS[real_name]
    observed = (calibration.get("services") or {}).get(real_name) or {}
    median_ms = observed.get("median_ms")
    sigma = observed.get("sigma", defaults["sigma"])
    # A malformed observed block (non-finite median, non-positive/non-finite
    # sigma) is dropped wholesale in favor of the defaults.
    if (
        not isinstance(median_ms, int | float)
        or not math.isfinite(float(median_ms))
        or not isinstance(sigma, int | float)
        or not math.isfinite(float(sigma))
        or float(sigma) <= 0
    ):
        observed = {}
    return {**defaults, **observed}


def _combined_metrics(clients: Sequence[ClientRuntime]) -> Metrics:
    roots = []
    attempts_total = 0
    for client in clients:
        roots.extend(client.roots)
        attempts_total += client.attempts_total
    return Metrics(roots=roots, attempts_total=attempts_total)


def _load_calibration(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    def _reject_constant(name: str):
        raise ValueError(f"calibration JSON contains non-finite value {name}")

    data = json.loads(Path(path).read_text(), parse_constant=_reject_constant)

    # parse_constant only catches the literal NaN/Infinity tokens. A finite
    # literal that OVERFLOWS the float type (e.g. 1e309) parses to inf without
    # ever reaching it, and would then flow into rates and durations -- an
    # infinite rps multiplier schedules an arrival every nanosecond.
    def _check_finite(node: Any, where: str) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError(
                f"calibration JSON has a non-finite number at {where}: {node!r} "
                "(a literal like 1e309 overflows to infinity)"
            )
        if isinstance(node, dict):
            for k, v in node.items():
                _check_finite(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _check_finite(v, f"{where}[{i}]")

    _check_finite(data, "<root>")
    return data


def _validated_profile_fault(
    start_s: float, end_s: float, add_latency_ms: float, p_fail: float, origin: str
) -> tuple[float, float, float, float]:
    """Bound check shared by the CLI parser and the calibration-JSON path.

    These used to diverge: the CLI rejected inverted windows and out-of-range
    p_fail while the JSON path applied a bare float(). An inverted window makes
    TimeInterval.contains() false everywhere, so the fault is INERT while
    fault_events.json still exports it and the plot shades it — a figure showing
    a fault the simulation never ran.
    """
    if start_s < 0:
        raise ValueError(f"{origin}: start must be >= 0, got {start_s}")
    if end_s < start_s:
        raise ValueError(
            f"{origin}: end ({end_s}) must be >= start ({start_s}); an inverted "
            "window is silently inert but is still exported and plotted"
        )
    if add_latency_ms < 0:
        raise ValueError(f"{origin}: add_latency_ms must be >= 0, got {add_latency_ms}")
    if not 0 <= p_fail <= 1:
        raise ValueError(
            f"{origin}: p_fail must be a fraction in [0, 1], got {p_fail}"
            + (" — did you pass a percentage?" if p_fail > 1 else "")
        )
    return (start_s, end_s, add_latency_ms, p_fail)


def _validated_load_spike(
    start_s: float, end_s: float, multiplier: float, origin: str
) -> tuple[float, float, float]:
    if start_s < 0:
        raise ValueError(f"{origin}: start must be >= 0, got {start_s}")
    if end_s < start_s:
        raise ValueError(f"{origin}: end ({end_s}) must be >= start ({start_s})")
    if multiplier < 0:
        raise ValueError(
            f"{origin}: rps_multiplier must be >= 0, got {multiplier}; a negative "
            "multiplier clamps the rate to zero, i.e. a traffic blackout exported "
            "and plotted as a load spike"
        )
    return (start_s, end_s, multiplier)


def _profile_faults_from_calibration(
    calibration: dict[str, Any],
) -> list[tuple[float, float, float, float]] | None:
    """None when the calibration has no fault_scenario key (fall back to the
    built-in default); a list — possibly empty — when it does. An explicit
    empty scenario, or one naming only non-profile methods, means "no profile
    faults" and must not be replaced by the default."""
    scenario = calibration.get("fault_scenario")
    if scenario is None:
        return None
    windows = []
    for window in scenario:
        if window.get("method") != "/profile.Profile/GetProfiles":
            continue
        windows.append(
            _validated_profile_fault(
                float(window["start_s"]),
                float(window["end_s"]),
                float(window["add_latency_ms"]),
                float(window["p_fail"]),
                origin="calibration fault_scenario",
            )
        )
    return windows


def _effective_profile_faults(
    config: HotelMatchedConfig, calibration: dict[str, Any]
) -> list[tuple[float, float, float, float]]:
    if config.profile_faults is not None:
        return config.profile_faults
    calibrated = _profile_faults_from_calibration(calibration)
    return calibrated if calibrated is not None else DEFAULT_PROFILE_FAULT


def _load_spikes(calibration: dict[str, Any]) -> list[tuple[float, float, float]]:
    workload = calibration.get("workload") or {}
    return [
        _validated_load_spike(
            float(spike[0]),
            float(spike[1]),
            float(spike[2]),
            origin="calibration workload.load_spikes",
        )
        for spike in workload.get("load_spikes") or []
    ]


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _nonneg_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _fraction(value: str) -> float:
    parsed = _finite_float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def _parse_profile_fault(value: str) -> tuple[float, float, float, float]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("profile fault must be start,end,add_latency_ms,p_fail")
    start_s, end_s, add_latency_ms, p_fail = (float(part) for part in parts)
    if not all(math.isfinite(part) for part in (start_s, end_s, add_latency_ms, p_fail)):
        raise argparse.ArgumentTypeError("profile fault values must be finite")
    if start_s < 0:
        raise argparse.ArgumentTypeError("profile fault start must be >= 0")
    if end_s < start_s:
        raise argparse.ArgumentTypeError("profile fault end must be >= start")
    if add_latency_ms < 0:
        raise argparse.ArgumentTypeError("profile fault add_latency_ms must be >= 0")
    if not 0 <= p_fail <= 1:
        raise argparse.ArgumentTypeError("profile fault p_fail must be in [0,1]")
    return start_s, end_s, add_latency_ms, p_fail


def _parse_load_spike(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("load spike must be start,end,multiplier")
    start_s, end_s, multiplier = (float(part) for part in parts)
    if not all(math.isfinite(part) for part in (start_s, end_s, multiplier)):
        raise argparse.ArgumentTypeError("load spike values must be finite")
    if start_s < 0:
        raise argparse.ArgumentTypeError("load spike start must be >= 0")
    if end_s < start_s:
        raise argparse.ArgumentTypeError("load spike end must be >= start")
    if multiplier < 0:
        raise argparse.ArgumentTypeError("load spike multiplier must be >= 0")
    return start_s, end_s, multiplier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run matched hotelReservation simulator scenario")
    parser.add_argument("--output-dir", type=Path, default=Path("hotel_matched_results"))
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--base-rps", type=_nonneg_float, default=None)
    parser.add_argument("--duration-s", type=_positive_float, default=None)
    parser.add_argument("--recommendation-fraction", type=_fraction, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--profile-fault",
        action="append",
        type=_parse_profile_fault,
        default=None,
        help="Profile fault as start,end,add_latency_ms,p_fail; overrides calibration/default.",
    )
    parser.add_argument(
        "--load-spike",
        action="append",
        type=_parse_load_spike,
        default=None,
        help="Load spike as start,end,multiplier; overrides calibration.",
    )
    parser.add_argument(
        "--leaf-max-attempts",
        type=_parse_leaf_max_attempts,
        default=3,
        help="Total attempts (initial + retries) for leaf-read clients; mirrors leaf-read max_retries + 1.",
    )
    return parser


def _parse_leaf_max_attempts(value: str) -> int:
    attempts = int(value)
    if attempts < 1:
        raise argparse.ArgumentTypeError("leaf max attempts must be >= 1")
    return attempts


def config_from_args(argv: Sequence[str] | None = None) -> HotelMatchedConfig:
    args = build_parser().parse_args(argv)
    return HotelMatchedConfig(
        output_dir=args.output_dir,
        calibration_json=args.calibration_json,
        base_rps=args.base_rps,
        duration_s=args.duration_s,
        recommendation_fraction=args.recommendation_fraction,
        seed=args.seed,
        profile_faults=list(args.profile_fault) if args.profile_fault is not None else None,
        load_spikes=list(args.load_spike) if args.load_spike is not None else None,
        leaf_max_attempts=args.leaf_max_attempts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    outputs = run_hotel_matched(config_from_args(argv))
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
