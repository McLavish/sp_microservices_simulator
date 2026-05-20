from caller import ClientConfig, ClientRuntime
from core import TimeInterval, ms_to_ns, s_to_ns
from faults import LatencyInjection, PartialFailure
from retry_policies import ExponentialBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload

if __name__ == "__main__":
    sim = Simulator(seed=12345)

    svc_cfg = ServiceConfig(
        name="svc-exponential",
        latency_median=ms_to_ns(25),
        latency_lognorm_sigma=0.6,
        workers=8,
        queue_capacity=15,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(20), end=s_to_ns(40)),
                add_latency=ms_to_ns(50),
            ),
        ],
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(60), end=s_to_ns(80)), p_fail=0.3),
        ],
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()

    # Exponential backoff: 5 attempts, starting at 10ms, max 500ms
    cli_cfg = ClientConfig(
        retry=ExponentialBackoffRetryPolicy(
            max_attempts=5,
            initial_delay=ms_to_ns(10),
            max_delay=ms_to_ns(500),
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None,
            attempt_timeout=ms_to_ns(75),
        ),
    )
    client = ClientRuntime(cfg=cli_cfg, service=svc)

    wl = Workload(
        base_rps=200.0,
        duration_s=120,
        load_spikes=[],
    )

    wl.drive(sim, lambda s: client.start_request(s))

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()
    m.export_csv(path="exponential_backoff_output.csv", granularity_s=1.0)
