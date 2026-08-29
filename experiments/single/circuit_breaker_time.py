from msim.caller import ClientConfig, ClientRuntime
from msim.core import TimeInterval, ms_to_ns, s_to_ns
from msim.fault_events import FaultEventsTracker
from msim.faults import LatencyInjection, LoadSpike, PartialFailure
from msim.retry_policies import (
    ExponentialBackoffRetryPolicy,
    TimeBasedCircuitBreakerPolicy,
)
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.workload import Workload

if __name__ == "__main__":
    sim = Simulator(seed=77432)
    fault_tracker = FaultEventsTracker()

    # Service with prolonged failure periods
    svc_cfg = ServiceConfig(
        name="svc-cb-time",
        latency_median=ms_to_ns(40),
        latency_lognorm_sigma=0.9,
        workers=12,
        queue_capacity=20,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(20), end=s_to_ns(60)),
                add_latency=ms_to_ns(80),
            ),
        ],
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(35), end=s_to_ns(75)), p_fail=0.7),
        ],
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()
    svc_cfg.register_fault_events(fault_tracker)

    cli_cfg = ClientConfig(
        retry=TimeBasedCircuitBreakerPolicy(
            failure_threshold_rate=0.8,
            success_threshold_rate=0.8,
            min_requests=10,
            window_duration=s_to_ns(5),
            underlying_strategy=ExponentialBackoffRetryPolicy(
                max_attempts=4,
                initial_delay=ms_to_ns(25),
                max_delay=ms_to_ns(400),
            ),
            half_open_delay=s_to_ns(3),
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None,
            attempt_timeout=ms_to_ns(150),
        ),
    )
    client = ClientRuntime(cfg=cli_cfg, service=svc)

    wl = Workload(
        base_rps=250.0,
        duration_s=100,
        load_spikes=[
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(40), end=s_to_ns(50)),
                rps_multiplier=1.8,
            ),
        ],
    )
    wl.register_fault_events(fault_tracker)

    wl.drive(sim, client.start_request)

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()
    m.export_csv(path="circuit_breaker_time_output.csv", granularity_s=1.0)
    fault_tracker.export_json("circuit_breaker_time_fault_events.json")
