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

    def __post_init__(self):
        if self.workers < 1:
            raise ValueError(
                f"workers must be >= 1, got {self.workers}; with no workers "
                "_start_next can never dequeue, so every request is accepted and "
                "then never completes"
            )
        if self.latency_median < 1:
            raise ValueError(f"latency_median must be >= 1 ns, got {self.latency_median}")
        if self.latency_lognorm_sigma < 0:
            raise ValueError(
                f"latency_lognorm_sigma must be >= 0, got {self.latency_lognorm_sigma}"
            )
        if self.queue_capacity is not None and self.queue_capacity < 0:
            raise ValueError(f"queue_capacity must be >= 0 or None, got {self.queue_capacity}")

    def register_fault_events(self, tracker: FaultEventsTracker):
        """Register all fault events with the tracker"""
        for latency_inj in self.latency_injections:
            tracker.add_latency_injection(
                latency_inj.duration, latency_inj.add_latency, latency_inj.multiplier
            )

        for partial_failure in self.partial_failures:
            tracker.add_partial_failure(partial_failure.duration, partial_failure.p_fail)


# Custom attempt behavior:
# (service, sim, deadline, finalize, begin_at, release_worker, register_cancel)
#
# `register_cancel` lets a behavior publish its own teardown so an upstream can
# abandon this attempt mid-flight. The callback must leave the usual invariants
# intact: finalize exactly once, release_worker at most once. It is invoked at
# most once, and never after finalize. Call `register_cancel(None)` to REVOKE
# the teardown at the instant the behavior commits to an outcome, so a
# re-entrant cancel is reported as failed rather than silently swallowed.
Behavior = Callable[
    [
        "ServiceRuntime",
        Simulator,
        TimePoint | None,
        Callable[[bool, TimeDuration, DropReason], None],
        TimePoint,
        Callable[[], None],
        Callable[[Callable[[], None] | None], None],
    ],
    None,
]


class AttemptControl:
    """Cancellation handle for one submitted attempt.

    `cancel()` is a no-op once the attempt has completed (or already been
    cancelled), so callers may fire it defensively without tracking state.
    """

    __slots__ = ("_cancel_fn", "_settled")

    def __init__(self) -> None:
        self._cancel_fn: Callable[[], None] | None = None
        self._settled = False

    def _bind(self, fn: Callable[[], None]) -> None:
        if not self._settled:
            self._cancel_fn = fn

    def _clear(self) -> None:
        """Drop the current teardown without settling the handle.

        Used when an attempt changes phase (queued -> in service): the queued
        teardown is no longer valid, and the new phase installs its own. Leaving
        the stale one bound would cancel an attempt that is already running.
        """
        if not self._settled:
            self._cancel_fn = None

    def _settle(self) -> None:
        self._settled = True
        self._cancel_fn = None

    def cancel(self) -> bool:
        """Abandon the attempt. Returns True if it was still cancellable."""
        fn = self._cancel_fn
        if self._settled or fn is None:
            return False
        self._settle()
        fn()
        return True


@dataclass(order=True)
class QItem:
    # heap order keys (in order): enqueued_at, seq
    enqueued_at: TimePoint
    seq: int
    start: Callable[[], None] = field(compare=False)
    # Cancelled while still queued: skipped (and discarded) by _start_next.
    # Flagged rather than removed, because removing from the middle of a heap
    # would cost O(n) and break the FCFS invariant the heap order encodes.
    cancelled: bool = field(default=False, compare=False)


@dataclass
class ServiceRuntime:
    cfg: ServiceConfig
    in_flight: int = 0
    queue: list[QItem] = field(default_factory=list)
    _seq: int = 0
    # Live (non-tombstoned) queue occupancy. `queue` still holds cancelled
    # entries until they surface at the heap head, so its raw length overstates
    # occupancy: using it for the capacity check would let a cancelled entry go
    # on shedding load, and using it for telemetry would inflate queue depth.
    _queued: int = 0
    # Attempts ARRIVING at this queue, from every client feeding it. This is
    # lambda for a queueing model, so it must be counted at submission, not at
    # worker acquisition: counting service starts makes lambda equal to the
    # throughput the server can sustain, which understates and time-shifts
    # arrivals exactly under the congestion the model exists to predict.
    # Excludes work rejected before it ever joined the queue (queue-full,
    # already-expired), which genuinely never arrived at the server.
    admitted_total: int = 0

    def bind(self):
        self.in_flight = 0
        self.queue.clear()
        self._seq = 0
        self._queued = 0
        self.admitted_total = 0
        return self

    def queue_len(self) -> int:
        """Queue occupancy excluding cancelled entries still in the heap."""
        return self._queued

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
            if item.cancelled:
                continue  # already discounted at cancel time; never took a worker
            self._queued -= 1
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

        Returns an `AttemptControl` whose `cancel()` abandons the attempt if it
        is still outstanding (queued or in service) and is a no-op afterwards.
        """

        self._seq += 1
        seq = self._seq

        control = AttemptControl()

        def settled_on_done(
            success: bool, svc_time: TimeDuration, drop: DropReason, qsize: int
        ) -> None:
            # However the attempt ends, the handle stops being cancellable
            # first — so a cancel racing a completion is a no-op, not a
            # double-completion.
            control._settle()
            on_done(success, svc_time, drop, qsize)

        # Immediately drop the request if somehow it has already expired?
        if attempt_deadline is not None and sim.timestep >= attempt_deadline:
            settled_on_done(False, 0, DropReason.DEADLINE, self.queue_len())
            return control

        start_cb = partial(
            self._begin_service, sim, settled_on_done, attempt_deadline, behavior, control
        )

        self.admitted_total += 1  # arrival at the queue, not a service start

        if self.in_flight < self.cfg.workers:
            start_cb()  # Immediately start handling if we have free worker
            return control

        # Drop request if queue is full (load shedding). Counts live entries
        # only: a cancelled entry has given its slot back even though its
        # tombstone is still sitting in the heap.
        if self.cfg.queue_capacity is not None and self.queue_len() >= self.cfg.queue_capacity:
            self.admitted_total -= 1  # shed before joining the queue
            settled_on_done(False, 0, DropReason.QUEUE_FULL, self.queue_len())
            return control

        # Enqueue the request
        item = QItem(
            enqueued_at=sim.timestep,
            seq=seq,
            start=start_cb,
        )
        heapq.heappush(self.queue, item)
        self._queued += 1

        def cancel_queued() -> None:
            # Never entered service, so no worker to release and no _start_next.
            # Free the slot now, not when the tombstone reaches the heap head.
            item.cancelled = True
            self._queued -= 1
            on_done(False, 0, DropReason.CANCELLED, self.queue_len())

        control._bind(cancel_queued)
        return control

    # begin_service can be called either immediately or after the request was dequeued
    def _begin_service(
        self,
        sim: Simulator,
        on_done: Callable[[bool, TimeDuration, DropReason, int], None],
        attempt_deadline: TimePoint | None,
        behavior: Behavior | None = None,
        control: AttemptControl | None = None,
    ):
        # Entering service: the queued teardown no longer applies. Drop it before
        # anything else, so a behavior that registers no teardown of its own is
        # simply not cancellable rather than running the stale queued one — which
        # would report CANCELLED without releasing the worker and then let the
        # behavior finalize on top of it.
        if control is not None:
            control._clear()

        # If the attempt expired while waiting in the queue, drop immediately without
        # consuming a worker. We make sure we are not scheduling an event in the past.
        if attempt_deadline is not None and sim.timestep >= attempt_deadline:
            on_done(False, 0, DropReason.DEADLINE, self.queue_len())
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
            finish_ev = sim.schedule(expiry, finish_cb)

            def cancel_in_service() -> None:
                # Abandon the work: drop the pending completion and hand the
                # worker straight back, so cancellation actually reclaims
                # capacity rather than merely suppressing the result.
                finish_ev.cancelled = True
                self._complete_attempt(
                    sim, on_done, False, sim.timestep - begin_at, DropReason.CANCELLED
                )

            if control is not None:
                control._bind(cancel_in_service)
        else:
            # Custom behavior can optionally release the worker early (for tandem queues)
            released = False
            released_at: TimePoint | None = None
            finalized = False

            def release_worker():
                nonlocal released, released_at
                if finalized:
                    raise RuntimeError(
                        "release_worker called after finalize (the worker was already released)"
                    )
                if released:
                    raise RuntimeError("release_worker called more than once")
                released = True
                released_at = sim.timestep  # occupancy ends HERE, not at finalize
                self.in_flight -= 1
                self._start_next(sim)

            def finalize(success: bool, service_time: TimeDuration, drop: DropReason):
                nonlocal finalized
                if finalized:
                    raise RuntimeError("finalize called more than once for the same attempt")
                finalized = True
                if released:
                    # Report WORKER OCCUPANCY, which ended when the worker was
                    # handed back. A tandem node releases after its local stage
                    # and finalizes only once the whole subtree returns, so the
                    # duration the behavior reports spans the downstream wait —
                    # time no worker of THIS service was held. Utilization-style
                    # consumers (the Erlang-C E[S]) need the occupancy; the
                    # end-to-end node latency is still in Request.interval.
                    occupancy = released_at - begin_at if released_at is not None else 0
                    on_done(success, occupancy, drop, self.queue_len())
                else:
                    self._complete_attempt(sim, on_done, success, service_time, drop)

            def register_cancel(fn: Callable[[], None] | None) -> None:
                # The behavior owns its own teardown (it knows what it launched
                # downstream); it must still route through `finalize`, so the
                # exactly-once guards above cover the cancel path too.
                #
                # Passing None REVOKES the teardown. A behavior must do this the
                # moment it commits to an outcome, so that a cancel arriving
                # re-entrantly during its own finalize gets a truthful False
                # from AttemptControl.cancel() instead of being swallowed by an
                # already-settled guard while the caller believes it succeeded.
                if control is None:
                    return
                if fn is None:
                    control._clear()
                else:
                    control._bind(fn)

            behavior(
                self, sim, attempt_deadline, finalize, begin_at, release_worker, register_cancel
            )

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
        on_done(success, service_time, drop_reason, self.queue_len())
        self._start_next(sim)
