import heapq
import math
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
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


class QueueDiscipline(Enum):
    """Which waiting attempt a freed worker takes next.

    `FCFS` is the default and what every pinned result was produced with. The
    other two answer a what-if about an overloaded server: under FCFS the oldest
    waiter is served first, and beyond a point it is a waiter whose caller has
    already given up, so the server works on dead requests while live ones age
    behind them.

    * `LIFO` takes the NEWEST waiter (a stack). Fresh requests are served within
      their deadline; old ones sink and expire in the queue.
    * `RANDOM` takes a waiter uniformly at random, drawn at the moment a worker
      frees, from the simulator's own generator -- so the run stays reproducible
      from the seed, and a station that is not `RANDOM` draws nothing.

    The order only decides WHICH entry is taken. Capacity, the deadline rules
    and what counts as holding a queue slot are the same for all three.
    """

    FCFS = "fcfs"
    LIFO = "lifo"
    RANDOM = "random"


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
    # Which waiting attempt a freed worker takes (`QueueDiscipline`).
    queue_discipline: QueueDiscipline = QueueDiscipline.FCFS
    # Whether a waiting attempt gives its queue slot back the moment its caller's
    # deadline passes. False (the default, and every pinned result): the server
    # does not scan its backlog, so a dead entry holds its slot -- and sheds load
    # -- until a worker reaches it. True: a server that drops a waiting request as
    # soon as its caller has gone. Either way the caller is told at its deadline
    # and the entry never takes a worker.
    evict_expired: bool = False

    def __post_init__(self):
        if not isinstance(self.queue_discipline, QueueDiscipline):
            raise ValueError(
                f"queue_discipline must be a QueueDiscipline, got {self.queue_discipline!r}"
            )
        if not isinstance(self.evict_expired, bool):
            raise ValueError(f"evict_expired must be a bool, got {self.evict_expired!r}")
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


@dataclass(frozen=True)
class ServiceTime:
    """A lognormal service-time distribution on its own: `median` ns, shape `sigma`.

    What `ServiceConfig.latency_median` / `latency_lognorm_sigma` are for a whole
    station, as a value that can be attached to ONE handler node of it
    (`chain.ServiceNodeRuntime.service_time`), so that two methods of a service
    with very different local work share the station's workers and queue without
    sharing one distribution.

    Validated like `ServiceConfig`, and floored the same way: a sigma below
    `MIN_LOGNORM_SIGMA` becomes the floor, and a draw is ALWAYS taken. That is
    the opposite of `transport.TransportDelay`, where `sigma == 0` means "no
    draw" -- a node with an override must consume exactly the RNG draws the
    station's own distribution would have, or switching one on would shift every
    later draw of the run.
    """

    median: TimeDuration  # nanoseconds
    sigma: float  # sigma of the underlying lognormal (shape)

    def __post_init__(self) -> None:
        # Finiteness first: NaN passes every ordering guard below (`nan < 1` is
        # False) and would only surface inside a scheduled event.
        if not math.isfinite(self.median):
            raise ValueError(f"service-time median must be finite, got {self.median}")
        if not math.isfinite(self.sigma):
            raise ValueError(f"service-time sigma must be finite, got {self.sigma}")
        if self.median < 1:
            raise ValueError(f"service-time median must be >= 1 ns, got {self.median}")
        if self.sigma < 0:
            raise ValueError(f"service-time sigma must be >= 0, got {self.sigma}")
        if self.sigma < MIN_LOGNORM_SIGMA:
            object.__setattr__(self, "sigma", MIN_LOGNORM_SIGMA)


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
    # heap order keys (in order): enqueued_at, seq. Only FCFS orders by them;
    # LIFO and RANDOM keep `ServiceRuntime.queue` as a plain list in arrival order.
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
    # Expired AND already discounted (`ServiceConfig.evict_expired`): the slot was
    # given back at the deadline, so like a cancelled entry this is a tombstone.
    evicted: bool = field(default=False, compare=False)


#: A LIFO stack is rebuilt without its tombstones once they outnumber the live
#: entries, and never for fewer than this many (a rebuild is O(n)).
_COMPACT_MIN_DEAD = 64


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
    # Tombstones still sitting in `queue`: cancelled or evicted entries, which
    # hold no slot. Only a LIFO stack needs the count (`_note_dead`).
    _dead: int = 0

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

    def _sample_service_time(
        self, t: TimePoint, sim: Simulator, service_time: ServiceTime | None = None
    ) -> TimeDuration:
        """One service-time draw at `t`: lognormal, then injections, then the floor.

        `service_time` replaces the DISTRIBUTION only (a chain node's own
        handler, `ServiceNodeRuntime.service_time`). Everything else stays the
        station's: its latency injections and the 100 us floor still apply, and
        it is still exactly one `lognormvariate` draw, so a node with an
        override consumes the RNG stream the way it would without one.
        """
        rng = sim.rng()
        # Both ranges are owned by ServiceConfig.__post_init__ on a frozen
        # dataclass (and by ServiceTime.__post_init__ for an override): median
        # >= 1 ns is validated there, sigma is floored there.
        if service_time is None:
            median, sigma = self.cfg.latency_median, self.cfg.latency_lognorm_sigma
        else:
            median, sigma = service_time.median, service_time.sigma
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

    def _enqueue(self, item: QItem) -> None:
        if self.cfg.queue_discipline is QueueDiscipline.FCFS:
            heapq.heappush(self.queue, item)
        else:
            self.queue.append(item)  # arrival order; `_dequeue` decides who is taken

    def _dequeue(self, sim: Simulator) -> QItem:
        """Take the next entry in the station's order. `self.queue` is not empty.

        The entry may be a tombstone (cancelled) or expired; `_start_next` looks
        at it exactly as it looks at a heap head, whatever order produced it.
        """
        discipline = self.cfg.queue_discipline
        if discipline is QueueDiscipline.FCFS:
            return heapq.heappop(self.queue)
        if discipline is QueueDiscipline.LIFO:
            return self.queue.pop()
        # RANDOM: uniform over what is waiting, drawn NOW -- a pick made at arrival
        # would be a fixed priority, which is a different discipline. One entry is
        # no choice and takes no draw.
        last = len(self.queue) - 1
        index = sim.rng().randrange(last + 1) if last > 0 else 0
        item = self.queue[index]
        self.queue[index] = self.queue[last]
        self.queue.pop()
        return item

    def _note_dead(self) -> None:
        """An entry still in `queue` stopped holding a slot (cancelled or evicted).

        A heap brings its tombstones to the head as time passes, and RANDOM draws
        them like any other entry. A stack does neither: under sustained load its
        bottom never surfaces, so tombstones there would pile up for the whole
        run. Rebuild the stack without them once they are most of it. Skipping a
        tombstone and never having held it are the same thing to `_start_next`,
        so this changes memory, not behaviour.
        """
        self._dead += 1
        if (
            self.cfg.queue_discipline is QueueDiscipline.LIFO
            and self._dead >= _COMPACT_MIN_DEAD
            and 2 * self._dead > len(self.queue)
        ):
            self.queue = [item for item in self.queue if not (item.cancelled or item.evicted)]
            self._dead = 0

    def _start_next(self, sim: Simulator):
        while self.in_flight < self.cfg.workers and self.queue:
            item = self._dequeue(sim)
            if item.cancelled or item.evicted:
                self._dead -= 1
                continue  # already discounted (at cancel or eviction); never took a worker
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
        Execution is FCFS with 'workers' concurrency (or the station's
        `queue_discipline`, which changes only which waiter a freed worker takes).

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
        self._enqueue(item)
        self._queued += 1

        if attempt_deadline is not None:

            def expire_in_queue() -> None:
                # Still queued (start() and cancel_queued() both disarm this
                # event). Tell the caller now; the entry stays in the heap,
                # holding its slot, until the server reaches and discards it --
                # unless the station evicts, in which case the slot is free from
                # this instant and only a tombstone stays behind.
                item.expired = True
                # Depth EXCLUDING this attempt, as every other completion path
                # reports it (an entry in service or one that never joined is
                # not part of the queue it observes).
                depth = self.queue_len() - 1
                if self.cfg.evict_expired:
                    item.evicted = True
                    self._queued -= 1
                    self._note_dead()
                settled_on_done(False, 0, DropReason.DEADLINE, depth, False)

            deadline_ev = sim.schedule(attempt_deadline, expire_in_queue)

        def cancel_queued() -> None:
            # Never entered service, so no worker to release and no _start_next.
            # Free the slot now, not when the tombstone reaches the heap head.
            if deadline_ev is not None:
                deadline_ev.cancelled = True
            item.cancelled = True
            self._queued -= 1
            self._note_dead()
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
