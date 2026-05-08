from core import TimeInterval, s_to_ns, ms_to_ns
from faults import LatencyInjection, PartialFailure, LoadSpike
from retry_policies import FixedBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload
from caller import ClientConfig, ClientRuntime
from fault_events import FaultEventsTracker
from chain import ServiceNodeRuntime, DownstreamEdge


def make_client_cfg(
    attempt_timeout_ms: int,
    max_attempts: int,
    delay_ms: int,
    global_timeout_ms: int | None = None,
) -> ClientConfig:
    return ClientConfig(
        retry=FixedBackoffRetryPolicy(
            max_attempts=max_attempts,
            delay=ms_to_ns(delay_ms),
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None if global_timeout_ms is None else ms_to_ns(global_timeout_ms),
            attempt_timeout=ms_to_ns(attempt_timeout_ms),
        ),
    )


if __name__ == "__main__":
    sim = Simulator(seed=4242)
    fault_tracker = FaultEventsTracker()

    duration_s = 120
    outage_begin = s_to_ns(40)
    outage_end = s_to_ns(60)

    # Shared downstream bottleneck (C) with an outage window.
    c_cfg = ServiceConfig(
        name="svc-C",
        latency_median=ms_to_ns(20),
        latency_lognorm_sigma=0.7,
        workers=24,
        queue_capacity=800,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=outage_begin, end=outage_end),
                add_latency=ms_to_ns(30),
            )
        ],
        partial_failures=[
            PartialFailure(
                duration=TimeInterval(begin=outage_begin, end=outage_end),
                p_fail=0.4,
            )
        ],
    )

    # Two chains: A -> B -> C and X -> Y -> C
    a_cfg = ServiceConfig(
        name="svc-A",
        latency_median=ms_to_ns(12),
        latency_lognorm_sigma=0.4,
        workers=48,
        queue_capacity=2000,
    )
    b_cfg = ServiceConfig(
        name="svc-B",
        latency_median=ms_to_ns(12),
        latency_lognorm_sigma=0.4,
        workers=48,
        queue_capacity=2000,
    )
    x_cfg = ServiceConfig(
        name="svc-X",
        latency_median=ms_to_ns(12),
        latency_lognorm_sigma=0.4,
        workers=48,
        queue_capacity=2000,
    )
    y_cfg = ServiceConfig(
        name="svc-Y",
        latency_median=ms_to_ns(12),
        latency_lognorm_sigma=0.4,
        workers=48,
        queue_capacity=2000,
    )

    a_rt = ServiceRuntime(cfg=a_cfg).bind()
    b_rt = ServiceRuntime(cfg=b_cfg).bind()
    c_rt = ServiceRuntime(cfg=c_cfg).bind()
    x_rt = ServiceRuntime(cfg=x_cfg).bind()
    y_rt = ServiceRuntime(cfg=y_cfg).bind()

    a_node = ServiceNodeRuntime(service=a_rt)
    b_node = ServiceNodeRuntime(service=b_rt)
    c_node = ServiceNodeRuntime(service=c_rt)
    x_node = ServiceNodeRuntime(service=x_rt)
    y_node = ServiceNodeRuntime(service=y_rt)

    # Aggressive retries to C to trigger retry storm during outage.
    b_to_c_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=60, max_attempts=5, delay_ms=5),
        service=c_node,  # type: ignore[arg-type]
    )
    y_to_c_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=60, max_attempts=5, delay_ms=5),
        service=c_node,  # type: ignore[arg-type]
    )

    a_to_b_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=120, max_attempts=3, delay_ms=20),
        service=b_node,  # type: ignore[arg-type]
    )
    x_to_y_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=120, max_attempts=3, delay_ms=20),
        service=y_node,  # type: ignore[arg-type]
    )

    b_node.add_downstream(DownstreamEdge(client=b_to_c_client, target=c_node))
    y_node.add_downstream(DownstreamEdge(client=y_to_c_client, target=c_node))
    a_node.add_downstream(DownstreamEdge(client=a_to_b_client, target=b_node))
    x_node.add_downstream(DownstreamEdge(client=x_to_y_client, target=y_node))

    # Register fault events for plotting.
    for cfg in (a_cfg, b_cfg, c_cfg, x_cfg, y_cfg):
        cfg.register_fault_events(fault_tracker)

    # Two top-level clients (two chains).
    top_client_a = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=200, max_attempts=3, delay_ms=30),
        service=a_node,  # type: ignore[arg-type]
    )
    top_client_x = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=200, max_attempts=3, delay_ms=30),
        service=x_node,  # type: ignore[arg-type]
    )

    # Load spikes overlapping outage window to amplify retries.
    spike = LoadSpike(
        duration=TimeInterval(begin=s_to_ns(45), end=s_to_ns(65)),
        rps_multiplier=1.5,
    )
    wl_a = Workload(base_rps=180.0, duration_s=duration_s, load_spikes=[spike])
    wl_x = Workload(base_rps=180.0, duration_s=duration_s, load_spikes=[spike])
    wl_a.register_fault_events(fault_tracker)
    wl_x.register_fault_events(fault_tracker)

    wl_a.drive(sim, lambda s: top_client_a.start_request(s))
    wl_x.drive(sim, lambda s: top_client_x.start_request(s))

    sim.run(until=s_to_ns(duration_s))
    sim.run()

    # Export metrics and fault events.
    top_client_a.metrics().export_csv(path="multichain_A_output.csv", granularity_s=1.0)
    top_client_x.metrics().export_csv(path="multichain_X_output.csv", granularity_s=1.0)
    b_to_c_client.metrics().export_csv(path="multichain_B_to_C.csv", granularity_s=1.0)
    y_to_c_client.metrics().export_csv(path="multichain_Y_to_C.csv", granularity_s=1.0)
    fault_tracker.export_json("multichain_fault_events.json")

    print("=== Top-level A chain ===")
    top_client_a.metrics().print_stats()
    print("=== Top-level X chain ===")
    top_client_x.metrics().print_stats()
    print("=== Shared C (from B) ===")
    b_to_c_client.metrics().print_stats()
    print("=== Shared C (from Y) ===")
    y_to_c_client.metrics().print_stats()
