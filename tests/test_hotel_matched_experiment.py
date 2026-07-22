import csv
import json

from core import ms_to_ns
from experiments.hotel_matched import (
    DEFAULT_PROFILE_FAULT,
    HotelMatchedConfig,
    _build_runtime,
    _effective_profile_faults,
    config_from_args,
    run_hotel_matched,
)
from retry_policies import ExponentialBackoffWithJitterRetryPolicy, JitterMode, NoRetryPolicy
from simulator import Simulator


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
        Simulator(seed=123),
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
