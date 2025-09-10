from simulator.core import TimeInterval, s_to_ns, ms_to_ns, ns_to_ms
from simulator.faults import LoadSpike, LatencyInjection, PartialFailure
from simulator.retry_policies import FixedBackoffRetryPolicy, NoRetryPolicy
from simulator.service import ServiceConfig, ServiceRuntime
from simulator.simulator import Simulator
from simulator.timeout_controllers import StaticTimeoutController
from simulator.workload import Workload
from simulator.caller import ClientConfig, ClientRuntime
from simulator.fault_events import FaultEventsTracker

if __name__ == "__main__":
    sim = Simulator(seed=43232)

    # Create fault events tracker (will be empty for no-faults experiment)
    fault_tracker = FaultEventsTracker()

    svc_cfg = ServiceConfig(
        name="svc-A",
        latency_median=ms_to_ns(20),
        latency_lognorm_sigma=0.5,
        workers=16,
        queue_capacity=20,
    )

    svc = ServiceRuntime(cfg=svc_cfg).bind()

    cli_cfg = ClientConfig(
        retry=FixedBackoffRetryPolicy(
            max_attempts=2,
            delay=ms_to_ns(300),
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None, attempt_timeout=ms_to_ns(100)
        ),
    )
    client = ClientRuntime(cfg=cli_cfg, service=svc)

    wl = Workload(
        base_rps=300.0,
        duration_s=100,
        load_spikes=[],
    )

    wl.drive(sim, lambda s: client.start_request(s))

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()

    m.export_csv(path="output.csv", granularity_s=1.0)
