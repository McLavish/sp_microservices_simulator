import heapq
import math
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from functools import partial

from msim.core import DropReason, TimeDuration, TimePoint
from msim.fault_events import FaultEventsTracker
from msim.faults import LatencyInjection, PartialFailure
from msim.simulator import Event, Simulator

# Floor applied to the lognormal shape parameter. Load-bearing, not defensive:
# the sampler has floored sigma here since the first commit, so every pinned
# seeded result was produced with it, and configs that ask for a
# near-deterministic service (sigma=1e-9) really are sampled at 1e-6. It lives
# on the config so `cfg.latency_lognorm_sigma` reports the sigma actually used
# instead of the one that was asked for.
MIN_LOGNORM_SIGMA = 1e-6


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
        if self.latency_lognorm_sigma < MIN_LOGNORM_SIGMA:
            object.__setattr__(self, "latency_lognorm_sigma", MIN_LOGNORM_SIGMA)
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


#: The service -> client completion callback: (success, occupancy, drop reason,
#: queue size, occupancy_censored). Spelled out rather than `Callable[..., None]`
#: because the last argument carries statistical semantics — it marks the
#: occupancy sample as a LOWER BOUND for a censored-data estimator — so a
#: producer that omits it silently reports a truncated sample as a complete
#: observation.
#: Lives here, with the original producer; `caller.ServiceLike`, `chain` and
#: `transport.TransportLink` (which produces it too, on the client's side of the
#: wire) describe the same callback and import it.
#:
#: The queue size is `None` for an attempt that never reached a service queue at
#: all — shed by a circuit breaker, or expired on the wire in front of a
#: `transport.TransportLink`. Reporting 0 there would inject a fabricated "the
#: queue is empty" sample into `queue_avg_at_attempt_end` at exactly the moments
#: work is being shed hardest; `metrics` already reads `None` as NaN.
AttemptDoneCallback = Callable[[bool, TimeDuration, DropReason, int | None, bool], None]


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
    # The caller's deadline passed while this entry was still waiting for a
    # worker. The caller was told at the deadline (its own clock fired); the
    # entry keeps its queue slot until the server reaches it, and is then
    # discarded without a second report.
    expired: bool = field(default=False, compare=False)


@dataclass
class ServiceRuntime:
    cfg: ServiceConfig
    in_flight: int = 0
    queue: list[QItem] = field(default_factory=list)
    _seq: int = 0
    # Queue occupancy: every entry holding a slot, i.e. the heap minus the
    # cancelled tombstones. `queue` still holds those until they surface at the
    # heap head, so its raw length overstates occupancy: using it for the
    # capacity check would let a cancelled entry go on shedding load, and using
    # it for telemetry would inflate queue depth. Expired entries (the caller
    # timed out while waiting) DO count: the server has not looked at them yet,
    # so they still occupy the backlog and still shed load.
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
        """Reset the live queue state back to a freshly-constructed runtime.

        Derived from the field declarations rather than restating them: fields
        without a default are inputs (`cfg`) and are left alone, everything else
        is state and goes back to its declared default. A second, hand-written
        copy of the defaults could silently drift from this list — nothing
        exercises the reset path today, since every call site is
        `ServiceRuntime(cfg=...).bind()` on a fresh instance.
        """
        for f in fields(self):
            if f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
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
        # Both ranges are owned by ServiceConfig.__post_init__ on a frozen
        # dataclass: median >= 1 ns is validated there, sigma is floored there.
        mu = math.log(self.cfg.latency_median)  # since median = exp(mu)
        raw = rng.lognormvariate(mu, self.cfg.latency_lognorm_sigma)
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
            if item.expired:
                # The caller timed out while this waited and was told at its
                # deadline; the server discovers that now and drops the entry
                # without taking a worker or reporting again.
                continue
            item.start()

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: AttemptDoneCallback,
        attempt_deadline: TimePoint | None = None,
        behavior: Behavior | None = None,
    ):
        """
        Enqueue an attempt for this service.
        Execution is FCFS with 'workers' concurrency.

        Deadline contract: `attempt_deadline` is the CALLER's clock. An attempt
        still waiting for a worker when it passes is reported as a DEADLINE drop
        at exactly that instant, not when the server next dequeues; the entry
        keeps its queue slot until then (a server does not scan its backlog) and
        is discarded silently when it surfaces. An attempt in service is cut at
        the deadline and its worker freed there.

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
            success: bool,
            svc_time: TimeDuration,
            drop: DropReason,
            qsize: int,
            occupancy_censored: bool = False,
        ) -> None:
            # However the attempt ends, the handle stops being cancellable
            # first — so a cancel racing a completion is a no-op, not a
            # double-completion.
            control._settle()
            on_done(success, svc_time, drop, qsize, occupancy_censored)

        # Immediately drop the request if somehow it has already expired?
        if attempt_deadline is not None and sim.timestep >= attempt_deadline:
            settled_on_done(False, 0, DropReason.DEADLINE, self.queue_len(), False)
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
            settled_on_done(False, 0, DropReason.QUEUE_FULL, self.queue_len(), False)
            return control

        # Enqueue the request. The caller's deadline is armed as its own event:
        # a client whose attempt is still waiting for a worker times out at its
        # deadline regardless of when the server next looks at the queue.
        # Reporting only at dequeue (the old behaviour) made the client learn of
        # the timeout whenever a worker happened to free — in the retry-storm
        # example half the deadline drops at the bottleneck were reported up to
        # 50 ms late on a 60 ms timeout, and every retry fired that much late.
        deadline_ev: Event | None = None

        def start() -> None:
            # Dequeued in time: the caller's clock no longer applies here. The
            # in-service path enforces the deadline from now on (it cuts the
            # service at the deadline, and _begin_service still catches a
            # worker that frees at exactly the deadline instant).
            if deadline_ev is not None:
                deadline_ev.cancelled = True
            start_cb()

        item = QItem(enqueued_at=sim.timestep, seq=seq, start=start)
        heapq.heappush(self.queue, item)
        self._queued += 1

        if attempt_deadline is not None:

            def expire_in_queue() -> None:
                # Still queued (start() and cancel_queued() both disarm this
                # event). Tell the caller now; the entry stays in the heap,
                # holding its slot, until the server reaches and discards it.
                item.expired = True
                # Depth EXCLUDING this attempt, as every other completion path
                # reports it (an entry in service or one that never joined is
                # not part of the queue it observes).
                settled_on_done(False, 0, DropReason.DEADLINE, self.queue_len() - 1, False)

            deadline_ev = sim.schedule(attempt_deadline, expire_in_queue)

        def cancel_queued() -> None:
            # Never entered service, so no worker to release and no _start_next.
            # Free the slot now, not when the tombstone reaches the heap head.
            if deadline_ev is not None:
                deadline_ev.cancelled = True
            item.cancelled = True
            self._queued -= 1
            on_done(False, 0, DropReason.CANCELLED, self.queue_len(), False)

        control._bind(cancel_queued)
        return control

    # begin_service can be called either immediately or after the request was dequeued
    def _begin_service(
        self,
        sim: Simulator,
        on_done: Callable[..., None],
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

        # A worker that frees at EXACTLY the deadline instant can dequeue this
        # attempt before its own deadline event fires (same `at`, earlier seq):
        # drop it here, at the deadline, without consuming a worker. Strictly
        # later dequeues never reach this point — the expiry event has already
        # reported and _start_next discards the entry.
        if attempt_deadline is not None and sim.timestep >= attempt_deadline:
            on_done(False, 0, DropReason.DEADLINE, self.queue_len(), False)
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
                    # consumers (anything deriving a queueing model's E[S] from
                    # this stream) need the occupancy; the end-to-end node
                    # latency is still in Request.interval.
                    occupancy = released_at - begin_at if released_at is not None else 0
                    # Released early: the worker was handed back before whatever
                    # cut the attempt short, so this occupancy is a COMPLETE
                    # observation even when the attempt itself was censored.
                    on_done(success, occupancy, drop, self.queue_len(), False)
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
        on_done: Callable[..., None],
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
        on_done: Callable[..., None],
        success: bool,
        service_time: TimeDuration,
        drop_reason: DropReason,
    ):
        self.in_flight -= 1
        # The worker was still held when the attempt ended, so a deadline or
        # cancellation cut the SERVICE itself: the occupancy is right-censored,
        # a lower bound on the work the server would have done.
        occupancy_censored = drop_reason in (DropReason.DEADLINE, DropReason.CANCELLED)
        on_done(success, service_time, drop_reason, self.queue_len(), occupancy_censored)
        self._start_next(sim)
