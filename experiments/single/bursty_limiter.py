from caller import ClientConfig, ClientRuntime
from core import TimeInterval, ms_to_ns, s_to_ns
from fault_events import FaultEventsTracker
from faults import LatencyInjection, LoadSpike, PartialFailure
from retry_policies import BurstyRateLimiterPolicy, FixedBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload

if __name__ == "__main__":
    sim = Simulator(seed=43232)
    fault_tracker = FaultEventsTracker()

    svc_cfg = ServiceConfig(
        name="svc-bursty-limiter",
        latency_median=ms_to_ns(18),
        latency_lognorm_sigma=0.4,
        workers=16,
        queue_capacity=20,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(25), end=s_to_ns(45)),
                add_latency=ms_to_ns(40),
            ),
        ],
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(60), end=s_to_ns(70)), p_fail=0.5),
        ],
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()
    svc_cfg.register_fault_events(fault_tracker)

    # Bursty rate limiter: 200 max requests, refills at 50/sec
    cli_cfg = ClientConfig(
        retry=BurstyRateLimiterPolicy(
            max_requests=200,
            refill_rate=50,
            period=s_to_ns(1),
            underlying_strategy=FixedBackoffRetryPolicy(
                max_attempts=3,
                delay=ms_to_ns(75),
            ),
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None,
            attempt_timeout=ms_to_ns(60),
        ),
    )
    client = ClientRuntime(cfg=cli_cfg, service=svc)

    # Bursty workload pattern to test token bucket behavior
    wl = Workload(
        base_rps=100.0,
        duration_s=100,
        load_spikes=[
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(10), end=s_to_ns(15)),
                rps_multiplier=5.0,
            ),
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(50), end=s_to_ns(55)),
                rps_multiplier=4.0,
            ),
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(75), end=s_to_ns(95)),
                rps_multiplier=8.5,
            ),
        ],
    )
    wl.register_fault_events(fault_tracker)

    wl.drive(sim, lambda s: client.start_request(s))

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()
    m.export_csv(path="bursty_limiter_output.csv", granularity_s=1.0)
    fault_tracker.export_json("bursty_limiter_fault_events.json")
