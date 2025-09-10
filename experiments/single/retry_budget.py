from core import TimeInterval, s_to_ns, ms_to_ns, ns_to_ms
from faults import LoadSpike, LatencyInjection, PartialFailure
from retry_policies import RetryBudgetPolicy, ExponentialBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload
from caller import ClientConfig, ClientRuntime

if __name__ == "__main__":
    sim = Simulator(seed=33445)

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
            PartialFailure(
                duration=TimeInterval(begin=s_to_ns(30), end=s_to_ns(45)), p_fail=0.3
            ),
            PartialFailure(
                duration=TimeInterval(begin=s_to_ns(80), end=s_to_ns(95)), p_fail=0.5
            ),
        ],
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()

    # Retry budget: retries can be at most 10% of successes, max 4 consecutive retries
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

    wl.drive(sim, lambda s: client.start_request(s))

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()
    m.export_csv(path="retry_budget_output.csv", granularity_s=1.0)
