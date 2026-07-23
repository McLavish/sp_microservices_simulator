import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from core import DropReason, TimeDuration, TimePoint
from fault_events import FaultEventsTracker
from faults import LatencyInjection, PartialFailure
from simulator import Simulator


# https://en.wikipedia.org/wiki/Log-normal_distribution
@dataclass(frozen=True)
class ServiceConfig:
    name: str

    # Lognormal latency (median + sigma) in NANOSECONDS
    latency_median: TimeDuration
    latency_lognorm_sigma: float  # sigma of the underlying lognormal (shape)
    workers: int
    queue_capacity: int | None = None  # None means unbounded
    latency_injections: list[LatencyInjection] = field(default_factory=list)
    partial_failures: list[PartialFailure] = field(default_factory=list)

    def register_fault_events(self, tracker: FaultEventsTracker):
        """Register all fault events with the tracker"""
        for latency_inj in self.latency_injections:
            tracker.add_latency_injection(
                latency_inj.duration, latency_inj.add_latency, latency_inj.multiplier
            )

        for partial_failure in self.partial_failures:
            tracker.add_partial_failure(partial_failure.duration, partial_failure.p_fail)


# Custom attempt behavior: (service, sim, deadline, finalize, begin_at, release_worker)
Behavior = Callable[
    [
        "ServiceRuntime",
        Simulator,
        TimePoint | None,
        Callable[[bool, TimeDuration, DropReason], None],
        TimePoint,
        Callable[[], None],
    ],
    None,
]


@dataclass(order=True)
class QItem:
    # heap order keys (in order): enqueued_at, seq
    enqueued_at: TimePoint
    seq: int
    start: Callable[[], None] = field(compare=False)


@dataclass
class ServiceRuntime:
    cfg: ServiceConfig
    in_flight: int = 0
    queue: list[QItem] = field(default_factory=list)
    _seq: int = 0

    def bind(self):
        self.in_flight = 0
        self.queue.clear()
        self._seq = 0
        return self

    def _adjust_latency(self, t: TimePoint, base: TimeDuration) -> TimeDuration:
        mult: int = 1
        add: int = 0
        for inj in self.cfg.latency_injections:
            if inj.active(t):
                mult *= inj.multiplier
                add += inj.add_latency
        return max(0, base * mult + add)

    def _sample_service_time(self, t: TimePoint, sim: Simulator) -> TimeDuration:
        rng = sim.rng()
        median = max(1, self.cfg.latency_median)
        sigma = max(1e-6, self.cfg.latency_lognorm_sigma)
        mu = math.log(median)  # since median = exp(mu)
        raw = rng.lognormvariate(mu, sigma)
        adj = self._adjust_latency(t, int(raw))
        return max(100_000, adj)  # put 1_000_000 to fix 1ms

    def _fails_now(self, t: TimePoint, sim: Simulator) -> bool:
        rng = sim.rng()
        p_fail = 0.0
        for pf in self.cfg.partial_failures:
            if pf.active(t):
                p_fail = max(p_fail, pf.p_fail)
        return rng.random() < p_fail

    def _start_next(self, sim: Simulator):
        while self.in_flight < self.cfg.workers and self.queue:
            item = heapq.heappop(self.queue)
            item.start()

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: Callable[
            [bool, TimeDuration, DropReason, int], None
        ],  # success, service_time, whydropped, queue_size
        attempt_deadline: TimePoint | None = None,
        behavior: Behavior | None = None,
    ):
        """
        Enqueue an attempt for this service.
        Execution is FCFS with 'workers' concurrency.

        Callback contract: `on_done` runs after the worker is freed but BEFORE
        queued work resumes — a callback that synchronously submits new work can
        therefore start ahead of already-queued items (stock retry/chain paths
        never do this; they always go through the event heap). Callbacks must
        not raise: an exception aborts the run and strands queued items. Custom
        behaviors must call `finalize` exactly once (a second call raises) and
        `release_worker` at most once.
        """

        self._seq += 1
        seq = self._seq

        # Immediately drop the request if somehow it has already expired?
        if attempt_deadline is not None and sim.timestep >= attempt_deadline:
            on_done(False, 0, DropReason.DEADLINE, len(self.queue))
            return

        start_cb = partial(self._begin_service, sim, on_done, attempt_deadline, behavior)

        if self.in_flight < self.cfg.workers:
            start_cb()  # Immediately start handling if we have free worker
            return

        # Drop request if queue is full (load shedding)
        if self.cfg.queue_capacity is not None and len(self.queue) >= self.cfg.queue_capacity:
            on_done(False, 0, DropReason.QUEUE_FULL, len(self.queue))  # dropped from queue
            return

        # Enqueue the request
        heapq.heappush(
            self.queue,
            QItem(
                enqueued_at=sim.timestep,
                seq=seq,
                start=start_cb,
            ),
        )

    # begin_service can be called either immediately or after the request was dequeued
    def _begin_service(
        self,
        sim: Simulator,
        on_done: Callable[[bool, TimeDuration, DropReason, int], None],
        attempt_deadline: TimePoint | None,
        behavior: Behavior | None = None,
    ):
        # If the attempt expired while waiting in the queue, drop immediately without
        # consuming a worker. We make sure we are not scheduling an event in the past.
        if attempt_deadline is not None and sim.timestep >= attempt_deadline:
            on_done(False, 0, DropReason.DEADLINE, len(self.queue))
            return

        self.in_flight += 1
        begin_at = sim.timestep

        if behavior is None:
            # Default behavior: original single-stage service with optional deadline cut
            service_time = self._sample_service_time(begin_at, sim)
            expiry = (
                min(begin_at + service_time, attempt_deadline)
                if attempt_deadline is not None
                else begin_at + service_time
            )

            finish_cb = partial(self._finish_service, sim, on_done, begin_at, attempt_deadline)
            sim.schedule(expiry, finish_cb)
        else:
            # Custom behavior can optionally release the worker early (for tandem queues)
            released = False

            def release_worker():
                nonlocal released
                if released:
                    raise RuntimeError("release_worker called more than once")
                released = True
                self.in_flight -= 1
                self._start_next(sim)

            finalized = False

            def finalize(success: bool, service_time: TimeDuration, drop: DropReason):
                nonlocal finalized
                if finalized:
                    raise RuntimeError("finalize called more than once for the same attempt")
                finalized = True
                if released:
                    on_done(success, service_time, drop, len(self.queue))
                else:
                    self._complete_attempt(sim, on_done, success, service_time, drop)

            behavior(self, sim, attempt_deadline, finalize, begin_at, release_worker)

    def _finish_service(
        self,
        sim: Simulator,
        on_done: Callable[[bool, TimeDuration, DropReason, int], None],
        begin_at: TimePoint,
        attempt_deadline: TimePoint | None,
    ):
        success = True
        drop_reason = DropReason.NONE
        if attempt_deadline is not None and sim.timestep >= attempt_deadline:
            success = False
            drop_reason = DropReason.DEADLINE
        else:
            if self._fails_now(sim.timestep, sim):
                success = False
                drop_reason = DropReason.SERVER_FAILURE
        # Report actual worker occupancy (queue-excluded): a deadline cut frees
        # the worker at the deadline, so the latent sampled duration would
        # overstate time spent serving and corrupt utilization-style consumers.
        occupancy = sim.timestep - begin_at
        self._complete_attempt(sim, on_done, success, occupancy, drop_reason)

    def _complete_attempt(
        self,
        sim: Simulator,
        on_done: Callable[[bool, TimeDuration, DropReason, int], None],
        success: bool,
        service_time: TimeDuration,
        drop_reason: DropReason,
    ):
        self.in_flight -= 1
        on_done(success, service_time, drop_reason, len(self.queue))
        self._start_next(sim)
