from caller import ClientConfig, ClientRuntime
from core import TimeInterval, ms_to_ns, s_to_ns
from fault_events import FaultEventsTracker
from faults import LatencyInjection, PartialFailure
from retry_policies import (
    CountBasedCircuitBreakerPolicy,
    FixedBackoffRetryPolicy,
)
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload

if __name__ == "__main__":
    sim = Simulator(seed=99887)
    fault_tracker = FaultEventsTracker()

    svc_cfg = ServiceConfig(
        name="svc-cb-count",
        latency_median=ms_to_ns(35),
        latency_lognorm_sigma=0.8,
        workers=6,
        queue_capacity=10,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(25), end=s_to_ns(45)),
                add_latency=ms_to_ns(100),
            ),
        ],
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(30), end=s_to_ns(50)), p_fail=0.8),
            PartialFailure(duration=TimeInterval(begin=s_to_ns(70), end=s_to_ns(85)), p_fail=0.6),
        ],
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()
    svc_cfg.register_fault_events(fault_tracker)

    # Circuit breaker: open after 4/6 failures, close after 3/4 successes
    cli_cfg = ClientConfig(
        retry=CountBasedCircuitBreakerPolicy(
            failure_threshold_ratio=(4, 6),
            success_threshold_ratio=(3, 4),
            underlying_strategy=FixedBackoffRetryPolicy(
                max_attempts=3,
                delay=ms_to_ns(50),
            ),
            half_open_delay=ms_to_ns(2000),
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None,
            attempt_timeout=ms_to_ns(120),
        ),
    )
    client = ClientRuntime(cfg=cli_cfg, service=svc)

    wl = Workload(
        base_rps=150.0,
        duration_s=120,
        load_spikes=[],
    )
    wl.register_fault_events(fault_tracker)

    wl.drive(sim, client.start_request)

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()
    m.export_csv(path="circuit_breaker_count_output.csv", granularity_s=1.0)
    fault_tracker.export_json("circuit_breaker_count_fault_events.json")
