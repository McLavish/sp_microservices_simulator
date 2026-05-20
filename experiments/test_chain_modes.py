from caller import ClientConfig, ClientRuntime
from chain import DownstreamEdge, ServiceNodeRuntime
from core import TimeInterval, ms_to_ns, s_to_ns
from faults import LatencyInjection, LoadSpike, PartialFailure
from retry_policies import FixedBackoffRetryPolicy
from service import ServiceConfig, ServiceRuntime
from simulator import Simulator
from timeout_controllers import StaticTimeoutController
from workload import Workload


def make_client_cfg(attempt_timeout_ms: int, max_attempts: int, delay_ms: int) -> ClientConfig:
    return ClientConfig(
        retry=FixedBackoffRetryPolicy(
            max_attempts=max_attempts,
            delay=ms_to_ns(delay_ms),
        ),
        timeout_controller=StaticTimeoutController(
            global_timeout=None, attempt_timeout=ms_to_ns(attempt_timeout_ms)
        ),
    )


def run_scenario(seed: int, top_attempt_timeout_ms: int = 200):
    """Run A -> B -> {C, E} with unified tandem semantics.

    Returns per-client metrics summaries for (top A, B, C, E).
    """

    sim = Simulator(seed=seed)

    # Define services
    a_cfg = ServiceConfig(
        name="svc-A",
        latency_median=ms_to_ns(5),
        latency_lognorm_sigma=0.4,
        workers=16,
        queue_capacity=100,
    )
    b_cfg = ServiceConfig(
        name="svc-B",
        latency_median=ms_to_ns(10),
        latency_lognorm_sigma=0.5,
        workers=16,
        queue_capacity=100,
    )
    c_cfg = ServiceConfig(
        name="svc-C",
        latency_median=ms_to_ns(15),
        latency_lognorm_sigma=0.6,
        workers=16,
        queue_capacity=100,
        partial_failures=[
            PartialFailure(duration=TimeInterval(begin=s_to_ns(20), end=s_to_ns(40)), p_fail=0.2)
        ],
    )
    e_cfg = ServiceConfig(
        name="svc-E",
        latency_median=ms_to_ns(8),
        latency_lognorm_sigma=0.5,
        workers=12,
        queue_capacity=80,
        latency_injections=[
            LatencyInjection(
                duration=TimeInterval(begin=s_to_ns(30), end=s_to_ns(45)),
                add_latency=ms_to_ns(10),
            )
        ],
    )

    # Bind runtimes
    a_rt = ServiceRuntime(cfg=a_cfg).bind()
    b_rt = ServiceRuntime(cfg=b_cfg).bind()
    c_rt = ServiceRuntime(cfg=c_cfg).bind()
    e_rt = ServiceRuntime(cfg=e_cfg).bind()

    # Wrap as nodes
    a_node = ServiceNodeRuntime(service=a_rt)
    b_node = ServiceNodeRuntime(service=b_rt)
    c_node = ServiceNodeRuntime(service=c_rt)
    e_node = ServiceNodeRuntime(service=e_rt)

    # Clients for edges
    b_to_c_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=80, max_attempts=2, delay_ms=40),
        service=c_node,  # type: ignore[arg-type]
    )
    b_to_e_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=80, max_attempts=2, delay_ms=40),
        service=e_node,  # type: ignore[arg-type]
    )
    a_to_b_client = ClientRuntime(
        cfg=make_client_cfg(attempt_timeout_ms=120, max_attempts=2, delay_ms=60),
        service=b_node,  # type: ignore[arg-type]
    )

    # Wire graph using tandem semantics on every edge
    b_node.add_downstream(DownstreamEdge(client=b_to_c_client, target=c_node))
    b_node.add_downstream(DownstreamEdge(client=b_to_e_client, target=e_node))
    a_node.add_downstream(DownstreamEdge(client=a_to_b_client, target=b_node))

    # Top-level client calling A
    top_client_cfg = ClientConfig(
        retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(100)),
        timeout_controller=StaticTimeoutController(
            global_timeout=None, attempt_timeout=ms_to_ns(top_attempt_timeout_ms)
        ),
    )
    top_client = ClientRuntime(cfg=top_client_cfg, service=a_node)  # type: ignore[arg-type]

    # Workload
    wl = Workload(
        base_rps=300.0,
        duration_s=60,
        load_spikes=[
            LoadSpike(
                duration=TimeInterval(begin=s_to_ns(25), end=s_to_ns(35)),
                rps_multiplier=1.8,
            )
        ],
    )

    wl.drive(sim, lambda s: top_client.start_request(s))
    sim.run(until=s_to_ns(wl.duration_s))
    sim.run()

    # Return metrics per client
    return {
        "A": top_client.metrics().summary(),
        "B": a_to_b_client.metrics().summary(),
        "C": b_to_c_client.metrics().summary(),
        "E": b_to_e_client.metrics().summary(),
    }


def test_chain_tandem_behavior():
    print("\n=== Running default chain with tandem semantics ===")
    results = run_scenario(seed=1234)
    print({k: (v.succeeded, v.p95, v.dropped_server_failure) for k, v in results.items()})

    # Success counts should be monotonic across the chain
    assert results["A"].succeeded == results["B"].succeeded, (
        "Upstream client should match immediate downstream"
    )
    assert results["B"].succeeded <= min(results["C"].succeeded, results["E"].succeeded), (
        "Fanout should be limited by the least successful downstream"
    )

    # Partial failures on C must be observable downstream and at the root
    assert results["C"].dropped_server_failure > 0, (
        "C should register server failures during outage window"
    )
    assert results["A"].dropped_server_failure > 0, (
        "Failures from C must propagate back to the caller"
    )

    # Tandem semantics still allow retries at the root when downstream fails
    assert results["A"].retries_per_root > 0, "Caller should exercise retries under failure"

    print("Tandem behavior test passed.")


def test_deadline_propagation():
    print("\n=== Testing deadline propagation (tight top-level timeout) ===")
    # Tight top-level attempt timeout forces upstream to cut by deadline even if downstreams run long.
    res = run_scenario(seed=4321, top_attempt_timeout_ms=30)
    print({k: (v.succeeded, v.dropped_deadline) for k, v in res.items()})

    # Expect noticeable deadline drops at A and B under tight budget
    assert res["A"].dropped_deadline > 0, "Top-level should see deadline drops"
    assert res["B"].dropped_deadline > 0, "B should see deadline drops due to propagated deadline"

    print("Deadline propagation test passed.")


def test_queue_pressure_and_depth():
    print("\n=== Testing queue pressure with small capacities ===")

    # Reuse run_scenario by temporarily building a custom scenario with tiny queues
    def run_small_queues(seed: int):
        sim = Simulator(seed=seed)

        a_cfg = ServiceConfig(
            name="svc-A",
            latency_median=ms_to_ns(3),
            latency_lognorm_sigma=0.4,
            workers=2,
            queue_capacity=5,
        )
        b_cfg = ServiceConfig(
            name="svc-B",
            latency_median=ms_to_ns(5),
            latency_lognorm_sigma=0.5,
            workers=2,
            queue_capacity=5,
        )
        c_cfg = ServiceConfig(
            name="svc-C",
            latency_median=ms_to_ns(7),
            latency_lognorm_sigma=0.3,
            workers=2,
            queue_capacity=5,
        )
        # Bind runtimes and nodes
        a_node = ServiceNodeRuntime(service=ServiceRuntime(a_cfg).bind())
        b_node = ServiceNodeRuntime(service=ServiceRuntime(b_cfg).bind())
        c_node = ServiceNodeRuntime(service=ServiceRuntime(c_cfg).bind())

        # Clients with short timeouts to force drops under pressure
        a_to_b_client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=1, delay=ms_to_ns(10)),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(20)
                ),
            ),
            service=b_node,  # type: ignore[arg-type]
        )
        b_to_c_client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=1, delay=ms_to_ns(10)),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(25)
                ),
            ),
            service=c_node,  # type: ignore[arg-type]
        )

        # Wire downstream edges
        a_node.add_downstream(DownstreamEdge(client=a_to_b_client, target=b_node))
        b_node.add_downstream(DownstreamEdge(client=b_to_c_client, target=c_node))

        top_client = ClientRuntime(
            cfg=ClientConfig(
                retry=FixedBackoffRetryPolicy(max_attempts=1, delay=ms_to_ns(10)),
                timeout_controller=StaticTimeoutController(
                    global_timeout=None, attempt_timeout=ms_to_ns(30)
                ),
            ),
            service=a_node,  # type: ignore[arg-type]
        )

        wl = Workload(base_rps=200.0, duration_s=10, load_spikes=[])
        wl.drive(sim, lambda s: top_client.start_request(s))
        sim.run(until=s_to_ns(10))
        sim.run()

        return {
            "A": top_client.metrics().summary(),
            "B": a_to_b_client.metrics().summary(),
            "C": b_to_c_client.metrics().summary(),
        }

    small = run_small_queues(seed=7)
    print({k: (v.succeeded, v.dropped_queue, v.dropped_deadline) for k, v in small.items()})
    assert small["A"].dropped_queue > 0 or small["B"].dropped_queue > 0, (
        "Expect queue drops under pressure"
    )
    assert small["A"].succeeded > 0, "Even under pressure some requests should succeed"

    print("Queue pressure tests passed.")

    print("\n=== Testing deeper chain A->B->C->D ===")
    # Depth test: build explicit deeper chain inline
    sim = Simulator(seed=99)

    a_rt = ServiceRuntime(
        ServiceConfig("svc-A", ms_to_ns(3), 0.3, workers=8, queue_capacity=50)
    ).bind()
    b_rt = ServiceRuntime(
        ServiceConfig("svc-B", ms_to_ns(4), 0.3, workers=8, queue_capacity=50)
    ).bind()
    c_rt = ServiceRuntime(
        ServiceConfig("svc-C", ms_to_ns(5), 0.3, workers=8, queue_capacity=50)
    ).bind()
    d_rt = ServiceRuntime(
        ServiceConfig("svc-D", ms_to_ns(6), 0.3, workers=8, queue_capacity=50)
    ).bind()

    a_node = ServiceNodeRuntime(a_rt)
    b_node = ServiceNodeRuntime(b_rt)
    c_node = ServiceNodeRuntime(c_rt)
    d_node = ServiceNodeRuntime(d_rt)

    c_to_d = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=1, delay=ms_to_ns(10)),
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(80)
            ),
        ),
        service=d_node,  # type: ignore[arg-type]
    )
    b_to_c = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=1, delay=ms_to_ns(10)),
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(80)
            ),
        ),
        service=c_node,  # type: ignore[arg-type]
    )
    a_to_b = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=1, delay=ms_to_ns(10)),
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(100)
            ),
        ),
        service=b_node,  # type: ignore[arg-type]
    )

    # Wire: D is leaf
    c_node.add_downstream(DownstreamEdge(client=c_to_d, target=d_node))
    b_node.add_downstream(DownstreamEdge(client=b_to_c, target=c_node))
    a_node.add_downstream(DownstreamEdge(client=a_to_b, target=b_node))

    top_client = ClientRuntime(
        cfg=ClientConfig(
            retry=FixedBackoffRetryPolicy(max_attempts=1, delay=ms_to_ns(10)),
            timeout_controller=StaticTimeoutController(
                global_timeout=None, attempt_timeout=ms_to_ns(120)
            ),
        ),
        service=a_node,  # type: ignore[arg-type]
    )

    wl = Workload(base_rps=100.0, duration_s=10, load_spikes=[])
    wl.drive(sim, lambda s: top_client.start_request(s))
    sim.run(until=s_to_ns(10))
    sim.run()

    A_sum = top_client.metrics().summary()
    B_sum = a_to_b.metrics().summary()
    C_sum = b_to_c.metrics().summary()
    D_sum = c_to_d.metrics().summary()
    print(
        {
            "A": (A_sum.succeeded, A_sum.p95),
            "B": (B_sum.succeeded, B_sum.p95),
            "C": (C_sum.succeeded, C_sum.p95),
            "D": (D_sum.succeeded, D_sum.p95),
        }
    )
    assert A_sum.succeeded == B_sum.succeeded == C_sum.succeeded, (
        "Depth chain: upstream should match all until leaf"
    )
    assert C_sum.succeeded <= D_sum.succeeded, "Depth chain: limited by leaf"

    print("Deeper chain tests passed.")


if __name__ == "__main__":
    test_chain_tandem_behavior()
    test_deadline_propagation()
    test_queue_pressure_and_depth()
    print("\nALL TESTS PASSED")
