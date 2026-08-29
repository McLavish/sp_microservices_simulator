from msim.caller import ClientConfig, ClientRuntime
from msim.core import TimeInterval, ms_to_ns, s_to_ns
from msim.fault_events import FaultEventsTracker
from msim.faults import LatencyInjection, LoadSpike, PartialFailure
from msim.retry_policies import ExponentialBackoffRetryPolicy, RetryBudgetPolicy
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.workload import Workload

if __name__ == "__main__":
    sim = Simulator(seed=33445)
    fault_tracker = FaultEventsTracker()

    # Service with mixed success/failure patterns to test budget behavior
    svc_cfg = ServiceConfig(
        name="svc-retry-budget",
        latency_median=ms_to_ns(28),
        latency_lognorm_sigma=0.6,
        workers=14,
        queue_capacity=30,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(15), end=s_to_ns(25)),
                add_latency=ms_to_ns(35),
            ),
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(65), end=s_to_ns(75)),
                add_latency=ms_to_ns(50),
            ),
        ],
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(30), end=s_to_ns(45)), p_fail=0.3),
            PartialFailure(duration=TimeInterval(begin=s_to_ns(80), end=s_to_ns(95)), p_fail=0.5),
        ],
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()
    svc_cfg.register_fault_events(fault_tracker)

    # Retry budget (Finagle-style): retries capped at 10% of admitted *requests*
    # — not of successes — with a token bucket that bursts up to 50 retries; the
    # underlying backoff allows 4 attempts (3 retries) per request.
    cli_cfg = ClientConfig(
        retry=RetryBudgetPolicy(
            budget_ratio=0.1,
            max_retries=50,
            underlying_strategy=ExponentialBackoffRetryPolicy(
                max_attempts=4,
                initial_delay=ms_to_ns(15),
                max_delay=ms_to_ns(300),
            ),
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None,
            attempt_timeout=ms_to_ns(80),
        ),
    )
    client = ClientRuntime(cfg=cli_cfg, service=svc)

    wl = Workload(
        base_rps=180.0,
        duration_s=120,
        load_spikes=[
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(35), end=s_to_ns(50)),
                rps_multiplier=2.0,
            ),
        ],
    )
    wl.register_fault_events(fault_tracker)

    wl.drive(sim, client.start_request)

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()
    m.export_csv(path="retry_budget_output.csv", granularity_s=1.0)
    fault_tracker.export_json("retry_budget_fault_events.json")
