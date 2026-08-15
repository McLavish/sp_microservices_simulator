from caller import ClientConfig, ClientRuntime
from chain import DownstreamEdge, ServiceNodeRuntime
from core import TimeInterval, ms_to_ns, s_to_ns
from fault_events import FaultEventsTracker
from faults import PartialFailure
from retry_policies import FixedBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload


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
    sim = Simulator(seed=1234)
    fault_tracker = FaultEventsTracker()

    # Define services A -> B -> (C & E). The leaves are sized so retry demand
    # during the fault window stays inside their capacity: 32 workers x ~68 ms
    # mean is ~470 rps each against 188 rps of offered load. Sized at 16 workers
    # (~235 rps, ~80% utilization) the nested retry layers tipped the run into a
    # metastable collapse the moment svc-C degraded — root success went to zero
    # at t=12 s and never recovered, so ~80% of the exported buckets, and of the
    # figures plot.sh renders from them, were empty.
    a_cfg = ServiceConfig(
        name="svc-A",
        latency_median=ms_to_ns(60),
        latency_lognorm_sigma=0.5,
        workers=64,
        queue_capacity=10000000,
    )
    b_cfg = ServiceConfig(
        name="svc-B",
        latency_median=ms_to_ns(60),
        latency_lognorm_sigma=0.5,
        workers=256,
        queue_capacity=10000000,
    )
    c_cfg = ServiceConfig(
        name="svc-C",
        latency_median=ms_to_ns(60),
        latency_lognorm_sigma=0.5,
        workers=32,
        queue_capacity=10000000,
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(10), end=s_to_ns(20)), p_fail=0.5)
        ],
    )
    e_cfg = ServiceConfig(
        name="svc-E",
        latency_median=ms_to_ns(60),
        latency_lognorm_sigma=0.5,
        workers=32,
        queue_capacity=10000000,
    )

    a_rt = ServiceRuntime(cfg=a_cfg).bind()
    b_rt = ServiceRuntime(cfg=b_cfg).bind()
    c_rt = ServiceRuntime(cfg=c_cfg).bind()
    e_rt = ServiceRuntime(cfg=e_cfg).bind()

    # Wrap as nodes
    a_node = ServiceNodeRuntime(service=a_rt)
    b_node = ServiceNodeRuntime(service=b_rt)
    c_node = ServiceNodeRuntime(service=c_rt)
    e_node = ServiceNodeRuntime(service=e_rt)

    # Per-edge client configurations
    # B's client to C and E (fanout)
    b_to_c_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=250, max_attempts=3, delay_ms=40),
        service=c_node,
    )
    b_to_e_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=250, max_attempts=3, delay_ms=40),
        service=e_node,
    )

    # A's client to B
    a_to_b_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=800, max_attempts=3, delay_ms=60),
        service=b_node,
    )

    # Wire graph: B -> {C, E}; A -> B
    b_node.add_downstream(DownstreamEdge(client=b_to_c_client, target=c_node))
    b_node.add_downstream(DownstreamEdge(client=b_to_e_client, target=e_node))

    a_node.add_downstream(DownstreamEdge(client=a_to_b_client, target=b_node))

    # Register fault events for plotting
    for cfg in (a_cfg, b_cfg, c_cfg, e_cfg):
        cfg.register_fault_events(fault_tracker)

    # Top-level client calling A. The global timeout is what bounds the work one
    # root can put on the leaves: it propagates down the graph as each node's
    # deadline (chain.py hands it to every downstream call), so a root cannot
    # keep re-driving the whole subtree once its budget is spent.
    top_client_cfg = make_client_cfg(
        attempt_timeout_ms=1000, max_attempts=3, delay_ms=100, global_timeout_ms=2000
    )
    top_client = ClientRuntime(cfg=top_client_cfg, service=a_node)

    wl = Workload(
        base_rps=188.0,
        duration_s=60,
        load_spikes=[],
    )

    wl.drive(sim, top_client.start_request)

    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    # Export metrics and faults
    m = top_client.metrics()
    m.export_csv(path="chain_output.csv", granularity_s=1.0)
    fault_tracker.export_json("chain_fault_events.json")

    # Per-service CSVs via the client that targets each node. svc-A has no entry
    # of its own: the top client is the only caller of A, so its view IS
    # chain_output.csv above — exporting it again as svc-A_output.csv produced a
    # byte-identical file and a duplicate set of figures.
    per_service_clients = [
        (a_to_b_client, b_node.cfg.name),
        (b_to_c_client, c_node.cfg.name),
        (b_to_e_client, e_node.cfg.name),
    ]

    for cli, name in per_service_clients:
        cli.metrics().export_csv(path=f"{name}_output.csv", granularity_s=1.0)

    print("=== Top-level (A) ===")
    m.print_stats()
    print("=== Per-service summaries ===")
    for cli, name in per_service_clients:
        print(f"-- {name} --")
        cli.metrics().print_stats()

    # Simple AND-fanout sanity: successes must be monotonic
    top_summary = m.summary()
    sA = top_summary.succeeded
    sB = a_to_b_client.metrics().summary().succeeded
    sC = b_to_c_client.metrics().summary().succeeded
    sE = b_to_e_client.metrics().summary().succeeded
    print("Sanity check (AND-fanout): succ(A) <= succ(B) <= min(succ(C), succ(E))")
    print(f"succ(A)={sA}, succ(B)={sB}, succ(C)={sC}, succ(E)={sE}")
    # A collapsed run satisfies the monotonicity above just as well as a healthy
    # one (0 <= 0 <= 0), so report the rate the figures are drawn from too.
    print(f"top-level success rate: {sA / max(1, top_summary.total):.3f}")
