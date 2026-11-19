"""Arolla's scenario schema, read strictly, and its inline sweeps.

The reader accepts what their own schema accepts and means, and refuses what
theirs lets through by accident: unknown keys, values out of range, two
limiters on one service. What their runner reads and then ignores is a warning.
"""

import unittest
from typing import Any

import yaml

from msim.compat.arolla import SchemaError, expand, parse
from msim.compat.arolla.sweeps import find_dimensions

MINIMAL = """
name: minimal
services:
  - name: svc
    latency: {median_ms: 10}
    workers: 4
workload: {base_rps: 100, duration_s: 5}
"""

FULL = """
name: full
seed: 7
services:
  - name: front
    latency: {median_ms: 20, lognorm_sigma: 0.3}
    workers: 8
    queue_capacity: 100
    timeout: {attempt_ms: 150}
    arolla_retry_budget: {alpha: 0.2, beta_up: 0.15}
    partial_failures:
      - {type: partial_failure, start_s: 2, end_s: 4, p_fail: 0.5}
    latency_injections:
      - {start_s: 5, end_s: 6, add_latency_ms: 30, multiplier: 2}
clients:
  - name: steady
    target_service: front
    replicas: 3
    retry: {type: jittered, max_attempts: 4, initial_delay_ms: 10, max_delay_ms: 80, jitter_mode: equal}
    circuit_breaker: {type: time_based, failure_threshold: 0.1, half_open_delay_ms: 500, window_duration_ms: 1000, min_requests: 20}
    retry_budget: {budget_ratio: 0.1, max_retries: 30, shared_budget_id: pool}
    workload:
      base_rps: 30
      duration_s: 8
      rng_seed: 3
      load_spikes: [{type: load_spike, start_s: 1, end_s: 2, rps_multiplier: 2.5}]
output_csv: results/out.csv
fault_events_json: results/faults.json
granularity_s: 1.0
plotting_script: plotting/x.py
plotting_script_args: {xticks: "0,1"}
"""


def _doc(text: str) -> dict[str, Any]:
    return yaml.safe_load(text)


def _edit(text: str, path: str, value: Any) -> dict[str, Any]:
    """`text` as a dict with the value at dotted `path` set (or deleted for `...`)."""
    doc = _doc(text)
    node: Any = doc
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    last: Any = int(keys[-1]) if isinstance(node, list) else keys[-1]
    if value is ...:
        del node[last]
    else:
        node[last] = value
    return doc


class TestWhatIsAccepted(unittest.TestCase):
    def test_the_smallest_scenario_and_its_defaults(self):
        experiment = parse(_doc(MINIMAL))
        self.assertEqual(experiment.name, "minimal")
        self.assertEqual(experiment.seed, 42)
        (service,) = experiment.services
        self.assertEqual(service.latency.lognorm_sigma, 0.5)
        self.assertIsNone(service.queue_capacity)
        self.assertIsNone(service.limiter)
        self.assertEqual(service.dependency_call_pattern, "sequential")
        assert experiment.workload is not None
        self.assertEqual(experiment.workload.base_rps, 100)
        self.assertIsNone(experiment.workload.rng_seed)
        self.assertEqual(experiment.clients, ())
        self.assertEqual(experiment.warnings, ())

    def test_every_part_of_a_full_scenario(self):
        experiment = parse(_doc(FULL))
        (service,) = experiment.services
        budget = service.arolla_retry_budget
        assert budget is not None
        self.assertEqual(service.limiter, ("arolla_retry_budget", budget))
        self.assertEqual(budget.alpha, 0.2)
        self.assertEqual(budget.beta_down, 0.3)  # their default
        self.assertEqual(service.partial_failures[0].p_fail, 0.5)
        self.assertEqual(service.latency_injections[0].multiplier, 2)
        (client,) = experiment.clients
        self.assertEqual(client.replicas, 3)
        assert client.retry is not None
        assert client.circuit_breaker is not None
        assert client.retry_budget is not None
        self.assertEqual(client.retry.jitter_mode, "equal")
        self.assertEqual(client.circuit_breaker.type, "time_based")
        self.assertEqual(client.retry_budget.min_retries_per_sec, 10)  # their default
        self.assertEqual(client.retry_budget.shared_budget_id, "pool")
        self.assertEqual(client.workload.load_spikes[0].rps_multiplier, 2.5)
        # Their runner's output settings are read and mean nothing here.
        self.assertEqual(experiment.warnings, ())

    def test_a_whole_number_may_be_written_as_a_float(self):
        experiment = parse(_edit(MINIMAL, "workload.duration_s", 5.0))
        assert experiment.workload is not None
        self.assertEqual(experiment.workload.duration_s, 5)
        self.assertIsInstance(experiment.workload.duration_s, int)


class TestWhatIsRefused(unittest.TestCase):
    def _error(self, doc: dict[str, Any]) -> str:
        with self.assertRaises(SchemaError) as raised:
            parse(doc)
        return str(raised.exception)

    def test_an_unknown_key_is_named_with_its_path(self):
        cases = {
            "services.0.querue_capacity": "services[0].querue_capacity",
            "services.0.latency.median": "services[0].latency.median",
            "workload.rps": "workload.rps",
            "servcies": "servcies",
        }
        for path, shown in cases.items():
            with self.subTest(path=path):
                message = self._error(_edit(MINIMAL, path, 1))
                self.assertIn(f"{shown}: unknown key", message)
        message = self._error(_edit(FULL, "clients.0.retry.backoff_ms", 5))
        self.assertIn("clients[0].retry.backoff_ms: unknown key", message)
        self.assertIn("max_attempts", message)  # the keys that do exist are listed

    def test_values_out_of_range(self):
        cases = [
            (MINIMAL, "services.0.latency.median_ms", 0, "must be > 0"),
            (MINIMAL, "services.0.workers", 0, "must be >= 1"),
            (MINIMAL, "services.0.queue_capacity", -1, "must be >= 0"),
            (MINIMAL, "workload.base_rps", 0, "must be > 0"),
            (MINIMAL, "workload.duration_s", 2.5, "whole number"),
            (FULL, "services.0.partial_failures.0.p_fail", 1.5, "must be <= 1"),
            (FULL, "services.0.partial_failures.0.end_s", 2, "after start_s"),
            (FULL, "services.0.arolla_retry_budget.alpha", 2, "must be <= 1"),
            (FULL, "clients.0.replicas", 0, "must be >= 1"),
            (FULL, "clients.0.retry.max_attempts", 0, "must be >= 1"),
            (FULL, "clients.0.circuit_breaker.failure_threshold", 1.5, "must be <= 1"),
        ]
        for text, path, value, expected in cases:
            with self.subTest(path=path):
                self.assertIn(expected, self._error(_edit(text, path, value)))

    def test_a_value_of_the_wrong_kind(self):
        self.assertIn("expected a number", self._error(_edit(MINIMAL, "services.0.workers", "4")))
        self.assertIn("expected a number", self._error(_edit(MINIMAL, "workload.base_rps", True)))
        self.assertIn("expected a mapping", self._error(_edit(MINIMAL, "services.0.latency", 10)))
        self.assertIn("expected a list", self._error(_edit(MINIMAL, "services", {"name": "x"})))
        self.assertIn(
            "must be one of none, fixed, exponential, jittered",
            self._error(_edit(FULL, "clients.0.retry.type", "linear")),
        )

    def test_a_sweep_list_that_was_not_expanded(self):
        message = self._error(_edit(FULL, "services.0.partial_failures.0.p_fail", [0.1, 0.2]))
        self.assertIn("services[0].partial_failures[0].p_fail: expected a number", message)

    def test_what_a_policy_needs(self):
        doc = _edit(FULL, "clients.0.retry", {"type": "fixed"})
        self.assertIn("delay_ms: required for a fixed backoff", self._error(doc))
        doc = _edit(FULL, "clients.0.retry", {"type": "exponential", "initial_delay_ms": 1})
        self.assertIn("max_delay_ms: required for exponential backoff", self._error(doc))
        doc = _edit(FULL, "clients.0.retry.jitter_mode", "decorrelated")
        self.assertIn("implemented by neither simulator", self._error(doc))

    def test_a_breaker_on_a_service_needs_its_window(self):
        doc = _edit(FULL, "services.0.arolla_retry_budget", ...)
        doc["services"][0]["circuit_breaker"] = {
            "type": "time_based",
            "failure_threshold": 0.1,
            "half_open_delay_ms": 500,
        }
        self.assertIn("window_duration_ms: required for a time_based breaker", self._error(doc))

    def test_more_than_one_limiter_on_a_service(self):
        doc = _doc(FULL)
        doc["services"][0]["global_retry_budget"] = {"target_rps": 30}
        message = self._error(doc)
        self.assertIn("more than one limiter", message)
        self.assertIn("global_retry_budget, arolla_retry_budget", message)

    def test_workload_and_clients_are_alternatives(self):
        doc = _doc(FULL)
        doc["workload"] = {"base_rps": 10, "duration_s": 1}
        self.assertIn("both set", self._error(doc))
        self.assertIn("is required", self._error(_edit(MINIMAL, "workload", ...)))

    def test_names_must_resolve(self):
        self.assertIn(
            "targets 'frnt', which is not a service",
            self._error(_edit(FULL, "clients.0.target_service", "frnt")),
        )
        doc = _doc(MINIMAL)
        doc["services"][0]["dependencies"] = [{"service": "db"}]
        self.assertIn("depends on 'db', which is not a service", self._error(doc))
        doc = _doc(MINIMAL)
        doc["services"].append(dict(doc["services"][0]))
        self.assertIn("'svc' is used more than once", self._error(doc))

    def test_dependencies_must_not_form_a_cycle(self):
        doc = _doc(MINIMAL)
        first = doc["services"][0]
        second = {**first, "name": "other", "dependencies": [{"service": "svc"}]}
        first["dependencies"] = [{"service": "other"}]
        doc["services"].append(second)
        self.assertIn("cycle", self._error(doc))


class TestWhatIsReadAndIgnored(unittest.TestCase):
    """Accepted, as by their schema, and reported: their runner does nothing with it."""

    def test_a_sweeps_block(self):
        doc = _doc(MINIMAL)
        doc["sweeps"] = [{"parameter": "services.0.workers", "values": [1, 2]}]
        (warning,) = parse(doc).warnings
        self.assertIn("sweeps:", warning)

    def test_a_services_global_timeout_that_nothing_reads(self):
        # Read for the implicit client (it has no timeout of its own) ...
        doc = _edit(MINIMAL, "services.0.timeout", {"global_ms": 500})
        self.assertEqual(parse(doc).warnings, ())
        # ... and not for a client that brings its own.
        doc = _doc(FULL)
        doc["services"][0]["timeout"]["global_ms"] = 500
        doc["clients"][0]["timeout"] = {"attempt_ms": 100}
        (warning,) = parse(doc).warnings
        self.assertIn("service 'front': timeout.global_ms", warning)

    def test_retry_gates_on_a_client_that_does_not_retry(self):
        doc = _edit(FULL, "clients.0.retry", ...)
        warnings = parse(doc).warnings
        self.assertEqual(len(warnings), 2)
        self.assertIn("clients[0].circuit_breaker", warnings[0])
        self.assertIn("clients[0].retry_budget", warnings[1])

    def test_a_count_based_breaker_on_a_client(self):
        doc = _edit(FULL, "clients.0.circuit_breaker.type", "count_based")
        (warning,) = parse(doc).warnings
        self.assertIn("count_based", warning)

    def test_client_only_keys_in_a_services_retry_budget(self):
        doc = _edit(FULL, "services.0.arolla_retry_budget", ...)
        doc["services"][0]["retry_budget"] = {
            "budget_ratio": 0.1,
            "max_retries": 3,
            "shared_budget_id": "x",
        }
        (warning,) = parse(doc).warnings
        self.assertIn("services[0].retry_budget.shared_budget_id", warning)

    def test_the_legacy_dependency_key(self):
        doc = _doc(MINIMAL)
        doc["services"].append({**doc["services"][0], "name": "db"})
        doc["services"][0]["dependency"] = "db"
        experiment = parse(doc)
        self.assertEqual([d.service for d in experiment.services[0].dependencies], ["db"])
        self.assertEqual(experiment.warnings, ())
        doc["services"][0]["dependencies"] = [{"service": "db", "optional": True}]
        experiment = parse(doc)
        self.assertTrue(experiment.services[0].dependencies[0].optional)
        self.assertIn("dependency", experiment.warnings[0])


SWEEP = """
name: sweep
services:
  - name: a
    latency: {median_ms: 10}
    workers: 4
    partial_failures:
      - {start_s: 0, end_s: 10, p_fail: &fails [0.0, 0.1, 0.2]}
  - name: b
    latency: {median_ms: 10}
    workers: 4
    partial_failures:
      - {start_s: 0, end_s: 10, p_fail: *fails}
clients:
  - name: one
    target_service: a
    replicas: &n [1, 10]
    workload: {base_rps: 10, duration_s: 10}
  - name: two
    target_service: b
    replicas: *n
    workload: {base_rps: 10, duration_s: 10}
"""


class TestSweeps(unittest.TestCase):
    def test_a_file_without_lists_is_one_scenario(self):
        (point,) = expand(_doc(MINIMAL))
        self.assertEqual((point.index, point.total, point.label), (0, 1, ""))
        self.assertEqual(point.config, _doc(MINIMAL))

    def test_an_anchored_list_is_one_dimension(self):
        dimensions = find_dimensions(_doc(SWEEP))
        self.assertEqual([d.name for d in dimensions], ["p_fail", "replicas"])
        self.assertEqual(
            dimensions[0].paths,
            ("services.0.partial_failures.0.p_fail", "services.1.partial_failures.0.p_fail"),
        )
        self.assertEqual(dimensions[1].values, (1, 10))

    def test_the_points_are_the_product_last_dimension_fastest(self):
        points = expand(_doc(SWEEP))
        self.assertEqual(len(points), 6)
        self.assertEqual(
            [point.label for point in points],
            [
                "p_fail=0.0,replicas=1",
                "p_fail=0.0,replicas=10",
                "p_fail=0.1,replicas=1",
                "p_fail=0.1,replicas=10",
                "p_fail=0.2,replicas=1",
                "p_fail=0.2,replicas=10",
            ],
        )
        self.assertEqual([point.index for point in points], list(range(6)))

    def test_every_place_of_a_dimension_takes_the_value_together(self):
        point = expand(_doc(SWEEP))[3]
        experiment = parse(point.config)
        self.assertEqual([s.partial_failures[0].p_fail for s in experiment.services], [0.1, 0.1])
        self.assertEqual([c.replicas for c in experiment.clients], [10, 10])

    def test_dimensions_follow_the_order_of_the_file(self):
        doc = _doc(SWEEP)
        reordered = {"name": doc["name"], "clients": doc["clients"], "services": doc["services"]}
        self.assertEqual([d.name for d in find_dimensions(reordered)], ["replicas", "p_fail"])

    def test_the_file_is_left_as_it_was(self):
        doc = _doc(SWEEP)
        expand(doc)
        self.assertEqual(doc["clients"][0]["replicas"], [1, 10])

    def test_a_list_outside_the_swept_parts_is_not_a_dimension(self):
        doc = _doc(MINIMAL)
        doc["plotting_script_args"] = {"xticks": [0, 1, 2]}
        self.assertEqual(find_dimensions(doc), [])
        self.assertEqual(len(expand(doc)), 1)


if __name__ == "__main__":
    unittest.main()
