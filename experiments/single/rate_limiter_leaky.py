from caller import ClientConfig, ClientRuntime
from core import TimeInterval, ms_to_ns, s_to_ns
from fault_events import FaultEventsTracker
from faults import LoadSpike, PartialFailure
from retry_policies import LeakyRateLimiterPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload

if __name__ == "__main__":
    sim = Simulator(seed=11223)
    fault_tracker = FaultEventsTracker()

    svc_cfg = ServiceConfig(
        name="svc-leaky-limiter",
        latency_median=ms_to_ns(15),
        latency_lognorm_sigma=0.3,
        workers=5,
        queue_capacity=50,
        latency_injections=[],
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(40), end=s_to_ns(60)), p_fail=0.2),
        ],
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()
    svc_cfg.register_fault_events(fault_tracker)

    # Leaky bucket rate limiter: 100 requests per second max
    cli_cfg = ClientConfig(
        retry=LeakyRateLimiterPolicy(
            max_requests=100,
            period=s_to_ns(1),
            max_attempts=3,
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None,
            attempt_timeout=ms_to_ns(50),
        ),
    )
    client = ClientRuntime(cfg=cli_cfg, service=svc)

    wl = Workload(
        base_rps=150.0,
        duration_s=80,
        load_spikes=[
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(20), end=s_to_ns(30)),
                rps_multiplier=3.0,
            ),
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(50), end=s_to_ns(60)),
                rps_multiplier=2.5,
            ),
        ],
    )
    wl.register_fault_events(fault_tracker)

    wl.drive(sim, client.start_request)

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()
    m.export_csv(path="rate_limiter_leaky_output.csv", granularity_s=1.0)
    fault_tracker.export_json("rate_limiter_leaky_fault_events.json")
