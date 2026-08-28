"""Per-link network delay between a client and the service it calls.

`TransportLink` is a `caller.ServiceLike` decorator: it wraps a `ServiceRuntime`
or a `ServiceNodeRuntime` and models the WIRE between a caller and that target —
one independent lognormal draw on the forward leg (request travelling to the
service) and one on the return leg (response travelling back). It is a
per-ATTEMPT cost, so retries pay for it again: the client issues every attempt
through `ClientRuntime._start_attempt`, which submits to whatever `service` it
holds, and the link sits exactly there.

    link = TransportLink(node, TransportDelay(median=ms_to_ns(2), sigma=0.3))
    client = ClientRuntime(cfg=client_cfg, service=link)
    parent.add_downstream(DownstreamEdge(client=client, target=node))

The link wraps the target *as that edge's client sees it*; a node's own
`ServiceNodeRuntime.service` stays a plain `ServiceRuntime` and `DownstreamEdge`
is unchanged, so the same node can be reached over different links by different
callers.

Attempts stay booked under the target's name (`cfg` forwards to the target), and
the delay is deliberately NOT folded into service time:

* `Request.service_time_ns` stays WORKER OCCUPANCY and excludes transport. It is
  the service-time sample any utilization or E[S] estimate is built from, so
  widening it to "occupancy + wire time" would silently corrupt those estimates.
* End-to-end attempt latency, both legs included, is `Request.interval`.

Deadlines are enforced against the caller's clock, not the packet's: a request
that would land at or after `attempt_deadline` never reaches the target at all,
and a response that would land at or after it is booked as a DEADLINE drop at
the deadline. That is the strict half-open rule (`end_time < deadline` is
required for success) applied to the transport leg.
"""

import math
from dataclasses import dataclass
from enum import Enum, auto

from caller import ServiceLike
from core import DropReason, TimeDuration, TimePoint
from service import AttemptControl, AttemptDoneCallback, ServiceConfig
from simulator import Event, Simulator


@dataclass(frozen=True)
class TransportDelay:
    """One-way network delay: lognormal with `median`, shape `sigma`.

    Deliberately does NOT apply `service.MIN_LOGNORM_SIGMA`. That floor exists
    on `ServiceConfig` because every pinned seeded result was produced with it;
    here `sigma == 0.0` is a documented mode of its own — a perfectly
    deterministic link, which is what makes the round trip exactly
    `2 * median` and therefore assertable in a test. A floored "almost zero"
    sigma would draw from the RNG and shift every later draw in the run.
    """

    median: TimeDuration  # nanoseconds
    sigma: float = 0.0  # shape of the underlying lognormal; 0 means deterministic

    def __post_init__(self) -> None:
        # Finiteness first: NaN passes every ordering guard below (`nan < 1` is
        # False) and only surfaces as `int(nan)` blowing up inside a scheduled
        # event, long after the misconfiguration.
        if not math.isfinite(self.median):
            raise ValueError(f"transport median must be finite, got {self.median}")
        if self.median < 1:
            raise ValueError(
                f"transport median must be >= 1 ns, got {self.median}; "
                "math.log(median) is the mu of the lognormal draw"
            )
        if not math.isfinite(self.sigma):
            raise ValueError(f"transport sigma must be finite, got {self.sigma}")
        if self.sigma < 0:
            raise ValueError(f"transport sigma must be >= 0, got {self.sigma}")

    def sample(self, sim: Simulator) -> TimeDuration:
        """Draw one leg's delay in ns.

        `sigma == 0` returns `median` WITHOUT touching the RNG: a deterministic
        link must not consume draws, or adding one to a scenario would move
        every seeded service time and failure roll after it.
        """
        if self.sigma == 0.0:
            return self.median
        raw = sim.rng().lognormvariate(math.log(self.median), self.sigma)
        return max(0, int(raw))


class _Phase(Enum):
    """Where one attempt currently is on the wire."""

    FORWARD = auto()  # travelling to the target (or waiting to expire in transit)
    AT_SERVICE = auto()  # handed to the target, waiting for its report
    RETURN = auto()  # the target reported; the response is travelling back
    DONE = auto()  # delivered to the client; `on_done` has fired


class _LinkControl(AttemptControl):
    """Cancellation handle for an attempt in flight over a link.

    The base class cancels by running a single bound teardown; a link attempt
    has a different teardown in each phase (and in one of them the answer comes
    from the target's own control), so `cancel()` delegates to the state machine
    instead of binding a closure.
    """

    __slots__ = ("_attempt",)

    def __init__(self, attempt: "_TransitAttempt") -> None:
        super().__init__()
        self._attempt = attempt

    def cancel(self) -> bool:
        if self._settled:
            return False
        return self._attempt.cancel()


class _TransitAttempt:
    """One attempt crossing a `TransportLink`, as an explicit state machine.

    The lifecycle is FORWARD -> AT_SERVICE -> RETURN -> DONE, with shortcuts:
    an attempt that expires in transit goes FORWARD -> DONE without ever
    touching the target, and a cancellation ends the attempt from whichever
    phase it is in. `on_done` fires EXACTLY ONCE, on every path, and always
    after the handle has been settled — mirroring `ServiceRuntime`'s
    `settled_on_done`, so a cancel racing a completion is a truthful no-op
    rather than a second completion.
    """

    def __init__(
        self,
        target: ServiceLike,
        delay: TransportDelay,
        sim: Simulator,
        on_done: AttemptDoneCallback,
        deadline: TimePoint | None,
    ) -> None:
        self.target = target
        self.delay = delay
        self.sim = sim
        self.deadline = deadline
        self._on_done = on_done

        self.phase = _Phase.FORWARD
        self.control = _LinkControl(self)
        #: The pending arrival / expiry / delivery event of the current phase.
        self._event: Event | None = None
        #: The target's own handle, bound only while we are in AT_SERVICE.
        self._inner: AttemptControl | None = None

        # What the target reported, kept for the return leg: a cancel during
        # return transit must hand the client the occupancy/queue/censoring the
        # service really observed, not zeros.
        self._res_occupancy: TimeDuration = 0
        self._res_qsize: int | None = None
        self._res_censored = False
        self._res_success = False
        self._res_drop = DropReason.NONE

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> AttemptControl:
        """Draw the forward leg and either expire in transit or schedule arrival."""
        arrive_at = self.sim.timestep + self.delay.sample(self.sim)

        if self.deadline is not None and arrive_at >= self.deadline:
            # Expired in transit: never call the target. Expired work is shed
            # before it consumes anything anywhere else in the model
            # (`ServiceRuntime` drops at submit and at dequeue without taking a
            # worker), and `admitted_total` counts arrivals AT THE QUEUE — a
            # packet that died on the wire never arrived.
            #
            # The client's clock fires at its deadline, not when the packet
            # would have landed. `max` only guards `Simulator.schedule`'s
            # past-instant ValueError for a deadline that is already behind us.
            fire_at = max(self.deadline, self.sim.timestep)
            self._event = self.sim.schedule(fire_at, self._on_forward_expired)
        else:
            self._event = self.sim.schedule(arrive_at, self._on_arrival)

        return self.control

    def _deliver(
        self,
        success: bool,
        occupancy: TimeDuration,
        drop: DropReason,
        qsize: int | None,
        censored: bool,
    ) -> None:
        """The one exit: settle the handle, then report to the client."""
        self.phase = _Phase.DONE
        if self._event is not None:
            # Nothing this attempt scheduled may still fire after the exit, so
            # kill it rather than merely dropping the reference. A delivery can
            # be pending here even on a cancellation path: a target that
            # reports a real outcome from INSIDE its own `cancel()` puts the
            # response on the return leg before `cancel()` has returned, and a
            # dropped-but-live event would then call `on_done` a second time —
            # `caller.py`'s "Root request was already done!" a leg later.
            self._event.cancelled = True
        self._event = None
        self._inner = None
        self.control._settle()
        self._on_done(success, occupancy, drop, qsize, censored)

    # -- phase FORWARD -------------------------------------------------------

    def _on_forward_expired(self) -> None:
        # Never reached a queue, so there is no queue sample to report: `None`,
        # the convention `ClientRuntime` already uses for a circuit-open attempt
        # (reporting 0 would fabricate an "empty queue" observation).
        self._event = None
        self._deliver(False, 0, DropReason.DEADLINE, None, False)

    def _on_arrival(self) -> None:
        self._event = None
        self.phase = _Phase.AT_SERVICE
        inner = self.target.submit_attempt(
            self.sim, self._on_target_done, attempt_deadline=self.deadline
        )
        # The target may report from INSIDE submit_attempt (queue-full, already
        # expired), i.e. before this assignment: by then the attempt has already
        # moved on to RETURN or DONE and this control is a stale handle on
        # nothing. Binding it anyway would let a later cancel tear down an
        # attempt that is no longer at the service.
        if self.phase is _Phase.AT_SERVICE:
            self._inner = inner

    # -- phase AT_SERVICE ----------------------------------------------------

    def _on_target_done(
        self,
        success: bool,
        occupancy: TimeDuration,
        drop: DropReason,
        qsize: int | None,
        censored: bool,
    ) -> None:
        if self.phase is not _Phase.AT_SERVICE:
            # Exactly-once, loudly — same stance as `finalize`/`release_worker`
            # in `ServiceRuntime`. A target reporting twice would otherwise
            # deliver twice and trip "Root request was already done!" somewhere
            # far from the cause.
            raise RuntimeError(
                "transport return hook fired for an attempt in phase "
                f"{self.phase.name}; the target reported more than once"
            )
        self._inner = None

        if drop == DropReason.CANCELLED:
            # Cancellation is client-initiated: the caller already knows, and
            # nothing has to travel back over the wire for it to learn.
            self._deliver(success, occupancy, drop, qsize, censored)
            return

        deliver_at = self.sim.timestep + self.delay.sample(self.sim)
        if self.deadline is not None and deliver_at >= self.deadline:
            # The response is late. The caller's clock fires at the deadline and
            # it never sees the answer, so the attempt is a DEADLINE drop booked
            # at the deadline — whatever the service said. Occupancy, queue size
            # and censoring are kept EXACTLY as reported: the work really was
            # done, only the response was late.
            #
            # `max` matters when the target itself reported at or after the
            # deadline (a deadline drop at dequeue happens strictly after it):
            # the past cannot be scheduled, so delivery is now.
            deliver_at = max(self.deadline, self.sim.timestep)
            success = False
            drop = DropReason.DEADLINE

        self._res_success = success
        self._res_occupancy = occupancy
        self._res_drop = drop
        self._res_qsize = qsize
        self._res_censored = censored
        self.phase = _Phase.RETURN
        self._event = self.sim.schedule(deliver_at, self._on_delivery)

    # -- phase RETURN --------------------------------------------------------

    def _on_delivery(self) -> None:
        self._event = None
        self._deliver(
            self._res_success,
            self._res_occupancy,
            self._res_drop,
            self._res_qsize,
            self._res_censored,
        )

    # -- cancellation --------------------------------------------------------

    def cancel(self) -> bool:
        """Abandon the attempt. True ONLY if something was really torn down.

        The truthful-`False` contract is load-bearing: `ClientRuntime._cancel_root`
        reads a `True` as "the teardown already delivered `on_done` and the root
        is complete", and `_FanOutAttempt` relies on the same to keep a genuine
        outcome instead of rewriting it as CANCELLED.
        """
        if self.phase is _Phase.DONE:
            return False

        if self.phase is _Phase.AT_SERVICE:
            inner = self._inner
            if inner is None:
                return False  # nothing bound: not cancellable, and say so
            if not inner.cancel():
                # `False` leaves everything bound on purpose: the target has
                # committed to a real outcome, which will arrive through the
                # hook and take the return leg like any other response.
                return False
            if self.phase is _Phase.DONE:
                # The stock targets report CANCELLED synchronously from inside
                # cancel(): the return hook has already delivered and settled
                # us, which is the whole answer.
                return True
            # Two ways to land here, both of which leave the attempt ours to
            # finish: the target tore down WITHOUT reporting (still AT_SERVICE),
            # or it reported a real, non-CANCELLED outcome from inside cancel()
            # and that response is now on the return leg (RETURN). The client
            # asked to abandon and nobody else will report, so fall through to
            # the common teardown — which drops the pending delivery, so the
            # attempt cannot come home twice.

        # FORWARD (a pending arrival or forward-expiry event), RETURN (the
        # response is travelling back) and the two AT_SERVICE cases above all
        # end the same way: report the cancellation with whatever the service
        # actually observed. `_res_*` are still their initial zeros/`None` while
        # nothing has been observed yet, which is exactly what a packet that
        # never landed has to report.
        self._deliver(
            False,
            self._res_occupancy,
            DropReason.CANCELLED,
            self._res_qsize,
            self._res_censored,
        )
        return True


@dataclass
class TransportLink:
    """A `ServiceLike` decorator that puts `delay` on each leg of every attempt.

    Drop-in for a `ServiceRuntime` or a `ServiceNodeRuntime`: `ClientRuntime`
    only ever reaches through `service` for `submit_attempt` and `cfg.name`.
    """

    target: ServiceLike
    delay: TransportDelay

    @property
    def cfg(self) -> ServiceConfig:
        # The wire is not a service: attempts stay booked under the target's
        # name, so metrics keyed on `Request.service` are unaffected by wrapping.
        return self.target.cfg

    def submit_attempt(
        self,
        sim: Simulator,
        on_done: AttemptDoneCallback,
        attempt_deadline: TimePoint | None = None,
    ) -> AttemptControl:
        """Send one attempt over the link; see the module docstring for semantics."""
        return _TransitAttempt(
            target=self.target,
            delay=self.delay,
            sim=sim,
            on_done=on_done,
            deadline=attempt_deadline,
        ).start()
