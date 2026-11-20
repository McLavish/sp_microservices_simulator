"""A scenario for the `msim.rl` tests: small, cheap, and with nothing drawn but the run itself."""

from dataclasses import dataclass

from msim.caller import ClientConfig, ClientRuntime
from msim.core import TimeInterval, ms_to_ns, s_to_ns
from msim.faults import PartialFailure
from msim.retry_policies import NoRetryPolicy
from msim.rl import Episode, Knob, RetryGrid
from msim.service import ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.workload import Workload

GRID = RetryGrid(attempt_timeout_ms=(20.0, 40.0, 80.0), default_attempt_timeout_ms=40.0)

#: Two knobs of two levels: four fixed settings, for tests that try them all.
TINY_GRID = RetryGrid(
    max_attempts=(1, 3),
    default_max_attempts=3,
    attempt_timeout_ms=None,
    backoff_ms=(10.0,),
    budget_ratio=(0.1, 1.0),
    default_budget_ratio=1.0,
)


@dataclass(frozen=True)
class Small:
    """One client on one small station with one fault at a known time: cheap, and fully known."""

    rps: float = 120.0
    p_fail: float = 0.5
    fault: tuple[float, float] | None = (8.0, 14.0)
    horizon_s: float = 20.0
    grid: RetryGrid = GRID

    @property
    def knobs(self) -> tuple[Knob, ...]:
        return self.grid.knobs()

    def parts(self, seed: int):
        sim = Simulator(seed=seed)
        failures = []
        if self.fault is not None:
            window = TimeInterval(s_to_ns(self.fault[0]), s_to_ns(self.fault[1]))
            failures.append(PartialFailure(window, p_fail=self.p_fail))
        station = ServiceRuntime(
            cfg=ServiceConfig(
                name="svc",
                latency_median=ms_to_ns(10),
                latency_lognorm_sigma=0.5,
                workers=4,
                queue_capacity=50,
                partial_failures=failures,
            )
        ).bind()
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(
                    global_timeout=ms_to_ns(400), attempt_timeout=ms_to_ns(40)
                ),
            ),
            service=station,
        )
        workload = Workload(
            base_rps=self.rps, duration_s=self.horizon_s, load_spikes=[], rng_seed=seed
        )
        return sim, station, client, workload

    def build(self, seed: int) -> Episode:
        sim, station, client, workload = self.parts(seed)
        agent = self.grid.install([client], agent_rps=self.rps)
        workload.drive(sim, client.start_request)
        return Episode(
            sim=sim,
            workloads=[workload],
            agent=agent,
            reward_clients=[client],
            clients=[client],
            expected_rps=self.rps,
            fault_windows=[] if self.fault is None else [self.fault],
            services={"svc": station},
        )
