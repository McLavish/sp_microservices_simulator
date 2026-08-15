"""Coverage for the runnable demos under `experiments/`.

The demos are `python -m` scripts whose whole point is the CSV they export, so
they are exercised the way a reader runs them: execute the module in a scratch
working directory and read back what it wrote. `runpy` hands back the module
globals, so the assertions read the demo's own configuration instead of
duplicating its constants.
"""

import csv
import os
import runpy
from pathlib import Path

import pytest

from core import ns_to_ms


def _run_demo(module: str, cwd: Path) -> dict:
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        return runpy.run_module(module, run_name="__main__")
    finally:
        os.chdir(previous)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _finite(values):
    return [float(v) for v in values if v not in ("", "nan")]


@pytest.fixture(scope="module")
def chain_fanout(tmp_path_factory):
    cwd = tmp_path_factory.mktemp("chain_fanout")
    return cwd, _run_demo("experiments.chain_fanout", cwd)


@pytest.fixture(scope="module")
def rate_limiter_leaky(tmp_path_factory):
    cwd = tmp_path_factory.mktemp("rate_limiter_leaky")
    return cwd, _run_demo("experiments.single.rate_limiter_leaky", cwd)


def test_chain_fanout_demo_reports_traffic_for_its_whole_run(chain_fanout):
    """Every exported bucket must carry successes and a latency percentile.

    Sized at 16 workers with no top-level global timeout, the leaves ran at ~80%
    utilization against three nested retry layers: the [10,20)s partial failure
    at svc-C tipped the run into a metastable collapse that never recovered, so
    `success_root` was 0 and every latency column `nan` from t=12 s to the end —
    ~80% of the CSV, and of the figures plot.sh renders from it.
    """
    cwd, globs = chain_fanout
    duration_s = globs["wl"].duration_s
    rows = _rows(cwd / "chain_output.csv")

    assert rows
    assert [row["timepoint"] for row in rows if int(row["success_root"]) == 0] == []
    assert [row["timepoint"] for row in rows if row["p50"] in ("", "nan")] == []
    # In-flight work drains within one bucket of the workload horizon.
    assert float(rows[-1]["timepoint"]) <= duration_s + 1

    global_timeout = globs["top_client_cfg"].timeout_controller.get_global_timeout()
    assert global_timeout is not None
    assert max(_finite(row["max"] for row in rows)) <= ns_to_ms(global_timeout)


def test_chain_fanout_demo_still_shows_the_fault_window(chain_fanout):
    """Calibrating the demo must not calibrate the phenomenon away."""
    cwd, _ = chain_fanout
    rows = {float(row["timepoint"]): row for row in _rows(cwd / "svc-C_output.csv")}
    during = [rows[t] for t in range(10, 20)]
    after = [rows[t] for t in range(21, 30)]

    assert all(int(row["failure_server"]) > 0 for row in during)
    assert all(int(row["retries"]) > 0 for row in during)
    # ... and recovers once the window closes.
    assert all(int(row["failure_server"]) == 0 for row in after)
    quiet_p50 = max(float(row["p50"]) for row in after)
    assert min(float(row["p50"]) for row in during) > quiet_p50


def test_chain_fanout_exports_each_population_once(chain_fanout):
    """svc-A_output.csv was a byte-identical copy of chain_output.csv.

    The top client is svc-A's only caller, so re-exporting its metrics under the
    node's name produced a duplicate file and a duplicate set of figures.
    """
    cwd, _ = chain_fanout
    exported = sorted(path.name for path in cwd.glob("svc-*_output.csv"))

    assert exported == ["svc-B_output.csv", "svc-C_output.csv", "svc-E_output.csv"]
    chain_csv = (cwd / "chain_output.csv").read_bytes()
    assert all((cwd / name).read_bytes() != chain_csv for name in exported)


def test_leaky_limiter_demo_keeps_its_reservation_queue_bounded(rate_limiter_leaky):
    """The leaky bucket is a queue: a granted retry reserves the next free slot
    however far ahead it is. With `global_timeout=None` nothing vetoed those
    reservations, so the run spilled ~24 s past its 80 s workload and the tail
    buckets reported p50 44 s / max 75 s for a 15 ms service — limiter backlog,
    not limiter behaviour.
    """
    cwd, globs = rate_limiter_leaky
    duration_s = globs["wl"].duration_s
    global_timeout = globs["cli_cfg"].timeout_controller.get_global_timeout()
    rows = _rows(cwd / "rate_limiter_leaky_output.csv")

    assert rows
    assert global_timeout is not None
    assert float(rows[-1]["timepoint"]) <= duration_s + 1
    assert max(_finite(row["max"] for row in rows)) <= ns_to_ms(global_timeout)
