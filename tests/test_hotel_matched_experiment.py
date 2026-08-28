import argparse
import csv
import json

import pytest

from core import ms_to_ns
from experiments.hotel_matched import (
    _DEFAULT_BASE_RPS,
    _DEFAULT_DURATION_S,
    _DEFAULT_RECOMMENDATION_FRACTION,
    DEFAULT_PROFILE_FAULT,
    HotelMatchedConfig,
    _build_runtime,
    _effective_profile_faults,
    _fraction,
    _load_calibration,
    _nonneg_float,
    _parse_load_spike,
    _parse_profile_fault,
    _positive_float,
    _recommendation_fraction_from,
    _validated_base_rps,
    _validated_duration,
    _validated_load_spike,
    _validated_profile_fault,
    config_from_args,
    run_hotel_matched,
)
from retry_policies import ExponentialBackoffWithJitterRetryPolicy, JitterMode, NoRetryPolicy


def test_config_from_args_accepts_calibration_fault_and_load_spike(tmp_path):
    config = config_from_args(
        [
            "--output-dir",
            str(tmp_path),
            "--calibration-json",
            str(tmp_path / "calibration.json"),
            "--base-rps",
            "12",
            "--duration-s",
            "3",
            "--recommendation-fraction",
            "0.4",
            "--profile-fault",
            "1,2,20,0.3",
            "--load-spike",
            "1,2,1.5",
        ]
    )

    assert config.output_dir == tmp_path
    assert config.calibration_json == tmp_path / "calibration.json"
    assert config.base_rps == 12
    assert config.duration_s == 3
    assert config.recommendation_fraction == 0.4
    assert config.profile_faults == [(1.0, 2.0, 20.0, 0.3)]
    assert config.load_spikes == [(1.0, 2.0, 1.5)]


def test_run_hotel_matched_writes_root_service_and_fault_outputs(tmp_path):
    calibration_json = tmp_path / "calibration.json"
    calibration_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "services": {
                    "search": _service(2.0, workers=16),
                    "geo": _service(1.0, workers=16),
                    "rate": _service(1.0, workers=16),
                    "reservation": _service(1.0, workers=16),
                    "profile": _service(2.0, workers=8),
                    "recommendation": _service(1.0, workers=16),
                },
                "workload": {
                    "base_rps": 8.0,
                    "duration_s": 3.0,
                    "path": "hotel-mixed",
                    "load_spikes": [],
                },
                "fault_scenario": [
                    {
                        "method": "/profile.Profile/GetProfiles",
                        "start_s": 1.0,
                        "end_s": 2.0,
                        "add_latency_ms": 10.0,
                        "p_fail": 0.2,
                    }
                ],
            }
        )
    )

    outputs = run_hotel_matched(
        HotelMatchedConfig(
            output_dir=tmp_path / "out",
            calibration_json=calibration_json,
            base_rps=8.0,
            duration_s=3.0,
            recommendation_fraction=0.25,
            seed=123,
        )
    )

    root_rows = _read_rows(outputs["root_csv"])
    profile_rows = _read_rows(outputs["svc_profile_csv"])
    fault_events = json.loads((tmp_path / "out" / "hotel_matched_fault_events.json").read_text())

    assert root_rows
    assert profile_rows
    assert list(root_rows[0]) == [
        "timepoint",
        "root_requests",
        "retries",
        "success_root",
        "completed",
        "failure_root",
        "failure_retry",
        "failure_queue_full",
        "failure_deadline",
        "failure_server",
        "failure_circuit_open",
        "failure_cancelled",
        "total_request",
        "total_failure",
        "p50",
        "p90",
        "p95",
        "p99",
        "p99.9",
        "p99.99",
        "min",
        "max",
        "queue_avg_at_attempt_end",
    ]
    assert sum(int(row["root_requests"]) for row in root_rows) > 0
    assert sum(int(row["root_requests"]) for row in profile_rows) > 0
    assert {event["event_type"] for event in fault_events} == {
        "latency_injection",
        "partial_failure",
    }


def test_run_hotel_matched_follows_the_recorded_workload(tmp_path):
    """With no CLI overrides the run must take rate, duration AND mix from the
    recording — the branch every other calibration test skips by pinning all
    three in the config. Ignoring the recording ran the "matched" scenario at
    the built-in 100 rps for 60 s against workers/queue_capacity read from that
    same file, and generated 75% search traffic against a
    recommendations-heavy recording.
    """
    calibration_json = tmp_path / "calibration.json"
    calibration_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "services": {
                    name: _service(1.0, workers=16)
                    for name in (
                        "search",
                        "geo",
                        "rate",
                        "reservation",
                        "profile",
                        "recommendation",
                    )
                },
                "workload": {
                    # Deliberately unlike the built-in defaults (100 rps, 60 s,
                    # 25% recommendations): 3 of every 4 requests are reads of
                    # the recommendation path.
                    "base_rps": 20.0,
                    "duration_s": 2.0,
                    "path": {
                        "/recommendation.Recommendation/GetRecommendations": 3,
                        "/search.Search/Nearby": 1,
                    },
                    "load_spikes": [],
                },
                "fault_scenario": [],
            }
        )
    )

    outputs = run_hotel_matched(
        HotelMatchedConfig(
            output_dir=tmp_path / "out",
            calibration_json=calibration_json,
            base_rps=None,
            duration_s=None,
            recommendation_fraction=None,
            seed=99,
        )
    )

    root_rows = _read_rows(outputs["root_csv"])
    total_roots = sum(int(row["root_requests"]) for row in root_rows)
    # The recorded 2 s, not the 60 s default: two one-second buckets.
    assert [row["timepoint"] for row in root_rows] == ["0.000", "1.000"]
    # 20 rps x 2 s ~= 40 roots; the default rate would offer ~6000.
    assert 15 <= total_roots <= 100

    recommendation_roots = sum(
        int(row["root_requests"]) for row in _read_rows(outputs["svc_recommendation_csv"])
    )
    search_roots = sum(int(row["root_requests"]) for row in _read_rows(outputs["svc_search_csv"]))
    # 75/25 split from workload.path; the default fraction reverses this.
    assert recommendation_roots > 2 * search_roots
    assert search_roots > 0


def test_recommendation_fraction_comes_from_the_recorded_mix():
    recorded = {
        "path": {
            "/recommendation.Recommendation/GetRecommendations": 3,
            "/search.Search/Nearby": 1,
        }
    }
    assert _recommendation_fraction_from(recorded) == 0.75
    # Nothing recorded about the mix -> the built-in default.
    assert _recommendation_fraction_from({}) == _DEFAULT_RECOMMENDATION_FRACTION
    assert _recommendation_fraction_from({"path": {}}) == _DEFAULT_RECOMMENDATION_FRACTION
    with pytest.raises(ValueError):
        _recommendation_fraction_from({"path": {"/search.Search/Nearby": -1}})


def test_recorded_rate_and_duration_are_bound_checked():
    assert _validated_base_rps(None) == _DEFAULT_BASE_RPS
    assert _validated_base_rps("12.5") == 12.5
    assert _validated_base_rps(0) == 0.0  # a silent recording is legal
    with pytest.raises(ValueError):
        _validated_base_rps(-1.0)

    assert _validated_duration(None) == _DEFAULT_DURATION_S
    assert _validated_duration("2.5") == 2.5
    with pytest.raises(ValueError):
        _validated_duration(0.0)  # a zero-length run exports nothing
    with pytest.raises(ValueError):
        _validated_duration(-3.0)


def test_run_hotel_matched_uses_default_profile_fault_when_no_calibration(tmp_path):
    outputs = run_hotel_matched(
        HotelMatchedConfig(
            output_dir=tmp_path,
            base_rps=5.0,
            duration_s=2.0,
            seed=456,
        )
    )

    fault_events = json.loads((tmp_path / "hotel_matched_fault_events.json").read_text())

    assert outputs["root_csv"] == str(tmp_path / "hotel_matched_root.csv")
    assert fault_events[0]["start_time_s"] == DEFAULT_PROFILE_FAULT[0][0]
    assert fault_events[0]["end_time_s"] == DEFAULT_PROFILE_FAULT[0][1]


def test_hotel_matched_retry_policies_follow_real_resilience_profiles():
    run = _build_runtime(
        calibration={},
        profile_faults=[],
        seed=123,
    )

    _assert_no_retry(run.per_service_clients["search"][0], attempt_timeout_ms=900)
    _assert_exp_jitter(
        run.per_service_clients["geo"][0],
        attempt_timeout_ms=500,
        initial_delay_ms=20,
        max_delay_ms=150,
    )
    _assert_exp_jitter(
        run.per_service_clients["rate"][0],
        attempt_timeout_ms=500,
        initial_delay_ms=20,
        max_delay_ms=150,
    )
    _assert_exp_jitter(
        run.per_service_clients["reservation"][0],
        attempt_timeout_ms=750,
        initial_delay_ms=25,
        max_delay_ms=200,
    )
    _assert_exp_jitter(
        run.per_service_clients["recommendation"][0],
        attempt_timeout_ms=750,
        initial_delay_ms=25,
        max_delay_ms=200,
    )
    for profile_client in run.per_service_clients["profile"]:
        _assert_exp_jitter(
            profile_client,
            attempt_timeout_ms=750,
            initial_delay_ms=25,
            max_delay_ms=200,
        )


def _service(median_ms, *, workers):
    return {
        "median_ms": median_ms,
        "sigma": 0.2,
        "samples": 10,
        "observed_max_in_flight": max(1, workers // 2),
        "workers": workers,
        "queue_capacity": 100,
    }


def _read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _assert_no_retry(client, *, attempt_timeout_ms):
    assert isinstance(client.cfg.retry, NoRetryPolicy)
    assert client.cfg.timeout_controller.get_attempt_timeout() == ms_to_ns(attempt_timeout_ms)


def _assert_exp_jitter(client, *, attempt_timeout_ms, initial_delay_ms, max_delay_ms):
    retry = client.cfg.retry
    assert isinstance(retry, ExponentialBackoffWithJitterRetryPolicy)
    assert retry.max_attempts == 3
    assert retry.initial_delay == ms_to_ns(initial_delay_ms)
    assert retry.max_delay == ms_to_ns(max_delay_ms)
    assert retry.jitter_mode == JitterMode.EQUAL
    assert client.cfg.timeout_controller.get_attempt_timeout() == ms_to_ns(attempt_timeout_ms)


def test_absent_fault_scenario_falls_back_to_default():
    assert _effective_profile_faults(HotelMatchedConfig(), {}) == DEFAULT_PROFILE_FAULT


def test_explicit_empty_fault_scenario_means_no_faults():
    assert _effective_profile_faults(HotelMatchedConfig(), {"fault_scenario": []}) == []


def test_non_profile_only_scenario_means_no_profile_faults():
    calibration = {
        "fault_scenario": [
            {
                "method": "/search.Search/Nearby",
                "start_s": 5,
                "end_s": 10,
                "add_latency_ms": 20,
                "p_fail": 0.1,
            }
        ]
    }
    assert _effective_profile_faults(HotelMatchedConfig(), calibration) == []


def test_config_profile_faults_override_wins():
    cfg = HotelMatchedConfig(profile_faults=[(1.0, 2.0, 3.0, 0.4)])
    assert _effective_profile_faults(cfg, {"fault_scenario": []}) == [(1.0, 2.0, 3.0, 0.4)]


def test_fractional_duration_changes_traffic(tmp_path):
    def total_roots(duration_s, out):
        outputs = run_hotel_matched(
            HotelMatchedConfig(
                output_dir=tmp_path / out, base_rps=50.0, duration_s=duration_s, seed=7
            )
        )
        rows = _read_rows(outputs["root_csv"])
        return sum(int(row["root_requests"]) for row in rows)

    # Pre-fix, 0.4 s was ceil-rounded to 1 s and produced identical traffic.
    assert total_roots(0.4, "short") < total_roots(1.0, "full")


def test_fraction_bounds():
    assert _fraction("0.25") == 0.25
    for bad in ("1.5", "-0.5", "nan"):
        with pytest.raises(argparse.ArgumentTypeError):
            _fraction(bad)


def test_rates_and_durations_validated():
    with pytest.raises(argparse.ArgumentTypeError):
        _nonneg_float("-100")
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float("0")
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float("inf")


def test_fault_and_spike_windows_validated():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_profile_fault("10,20,-500,0.5")  # negative added latency
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_load_spike("10,20,-3")  # negative multiplier
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_load_spike("-5,20,2")  # negative start


def test_cli_parsers_delegate_to_the_shared_validators():
    """The CLI must enforce the JSON path's bounds, not a private copy of them.

    Pinned through the diagnostics only `_validated_*` produce: a second copy of
    the checks passes the bound but not the message, so this fails if the CLI
    goes back to re-implementing them.
    """
    assert _parse_profile_fault("1,2,20,0.3") == _validated_profile_fault(
        1.0, 2.0, 20.0, 0.3, origin="--profile-fault"
    )
    assert _parse_load_spike("1,2,1.5") == _validated_load_spike(
        1.0, 2.0, 1.5, origin="--load-spike"
    )

    with pytest.raises(argparse.ArgumentTypeError, match="--profile-fault.*silently inert"):
        _parse_profile_fault("20,10,50,0.2")
    with pytest.raises(argparse.ArgumentTypeError, match="did you pass a percentage"):
        _parse_profile_fault("10,20,50,20")
    with pytest.raises(argparse.ArgumentTypeError, match="--load-spike.*traffic blackout"):
        _parse_load_spike("10,20,-3")
    with pytest.raises(argparse.ArgumentTypeError, match="--load-spike.*must be >= start"):
        _parse_load_spike("20,10,1.5")


def test_calibration_json_rejects_non_finite(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text('{"services": {"profile": {"median_ms": NaN}}}')
    with pytest.raises(ValueError):
        _load_calibration(path)


def test_calibration_rejects_numbers_that_overflow_to_infinity(tmp_path):
    """parse_constant only catches the literal NaN/Infinity tokens. A finite
    literal that overflows the float type reaches inf without ever hitting it,
    and an infinite rps multiplier schedules an arrival every nanosecond."""
    cal = tmp_path / "cal.json"
    cal.write_text('{"workload": {"load_spikes": [[1.0, 2.0, 1e309]]}}')
    with pytest.raises(ValueError, match="non-finite"):
        _load_calibration(cal)


def test_calibration_still_accepts_ordinary_numbers(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text('{"workload": {"base_rps": 8.0, "duration_s": 3.0, "load_spikes": []}}')
    assert _load_calibration(cal)["workload"]["base_rps"] == 8.0
