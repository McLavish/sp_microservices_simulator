from caller import ClientConfig, ClientRuntime
from core import ms_to_ns, s_to_ns
from fault_events import FaultEventsTracker
from retry_policies import FixedBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload

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

    wl.drive(sim, client.start_request)

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    m = client.metrics()

    m.export_csv(path="nofaults_output.csv", granularity_s=1.0)
