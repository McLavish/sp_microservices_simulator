from caller import ClientConfig, ClientRuntime
from core import TimeInterval, ms_to_ns, s_to_ns
from fault_events import FaultEventsTracker
from faults import LatencyInjection, LoadSpike, PartialFailure
from retry_policies import (
    ExponentialBackoffRetryPolicy,
    FixedWindowBurstyLimiterPolicy,
)
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload

if __name__ == "__main__":
    sim = Simulator(seed=88990)
    fault_tracker = FaultEventsTracker()

    svc_cfg = ServiceConfig(
        name="svc-fixed-window",
        latency_median=ms_to_ns(22),
        latency_lognorm_sigma=0.5,
        workers=16,
        queue_capacity=35,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(25), end=s_to_ns(35)),
                add_latency=ms_to_ns(30),
            ),
        ],
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(55), end=s_to_ns(65)), p_fail=0.3),
        ],
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()
    svc_cfg.register_fault_events(fault_tracker)

    # Fixed window limiter: 150 requests per 5-second window
    cli_cfg = ClientConfig(
        retry=FixedWindowBurstyLimiterPolicy(
            max_requests=150,
            period=s_to_ns(5),
            underlying_strategy=ExponentialBackoffRetryPolicy(
                max_attempts=3,
                initial_delay=ms_to_ns(30),
                max_delay=ms_to_ns(200),
            ),
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None,
            attempt_timeout=ms_to_ns(90),
        ),
    )
    client = ClientRuntime(cfg=cli_cfg, service=svc)

    wl = Workload(
        base_rps=200.0,
        duration_s=90,
        load_spikes=[
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(20), end=s_to_ns(40)),
                rps_multiplier=2.2,
            ),
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(70), end=s_to_ns(80)),
                rps_multiplier=1.8,
            ),
        ],
    )
    wl.register_fault_events(fault_tracker)

    wl.drive(sim, lambda s: client.start_request(s))

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()
    m.export_csv(path="fixed_window_limiter_output.csv", granularity_s=1.0)
    fault_tracker.export_json("fixed_window_limiter_fault_events.json")
