from simulator.core import TimeInterval, s_to_ns, ms_to_ns, ns_to_ms
from simulator.faults import LoadSpike, LatencyInjection, PartialFailure
from simulator.retry_policies import LeakyRateLimiterPolicy
from simulator.service import ServiceConfig, ServiceRuntime
from simulator.simulator import Simulator
from simulator.timeout_controllers import StaticTimeoutController
from simulator.workload import Workload
from simulator.caller import ClientConfig, ClientRuntime

if __name__ == "__main__":
    sim = Simulator(seed=11223)

    svc_cfg = ServiceConfig(
        name="svc-leaky-limiter",
        latency_median=ms_to_ns(15),
        latency_lognorm_sigma=0.3,
        workers=5,
        queue_capacity=50,
        latency_injections=[],
        partial_failures=[
            PartialFailure(
                duration=TimeInterval(begin=s_to_ns(40), end=s_to_ns(60)), p_fail=0.2
            ),
        ],
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()

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

    wl.drive(sim, lambda s: client.start_request(s))

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()
    m.export_csv(path="rate_limiter_leaky_output.csv", granularity_s=1.0)
