"""Queue discipline: which waiting attempt a freed worker takes.

`FCFS` is the default and is pinned by the goldens. Here: what `LIFO` and
`RANDOM` change (the order, and nothing else), that `RANDOM` is drawn at
dequeue from the simulator's own generator, and that everything a queue entry
can be -- waiting, cancelled, expired -- is handled the same way in all three.
"""

import unittest
from collections import Counter

from msim.caller import ClientConfig, ClientRuntime
from msim.core import DropReason, ms_to_ns
from msim.retry_policies import NoRetryPolicy
from msim.service import QueueDiscipline, ServiceConfig, ServiceRuntime
from msim.simulator import Simulator
from msim.timeout_controllers import StaticTimeoutController
from msim.workload import Workload

FCFS, LIFO, RANDOM = QueueDiscipline.FCFS, QueueDiscipline.LIFO, QueueDiscipline.RANDOM


def _service(
    discipline: QueueDiscipline = FCFS,
    *,
    workers: int = 1,
    queue_capacity: int | None = None,
    median_ms: float = 10.0,
    sigma: float = 0.0,
) -> ServiceRuntime:
    return ServiceRuntime(
        cfg=ServiceConfig(
            name="svc",
            latency_median=ms_to_ns(median_ms),
            latency_lognorm_sigma=sigma,
            workers=workers,
            queue_capacity=queue_capacity,
            queue_discipline=discipline,
        )
    ).bind()


def _submit(service: ServiceRuntime, sim: Simulator, label: str, order: list, deadline=None):
    """Submit one attempt; `order` receives `(label, drop reason)` when it completes."""

    def done(success, svc_time, drop, qsize, censored):
        order.append((label, drop.name))

    return service.submit_attempt(sim, done, attempt_deadline=deadline)


def _served(order: list) -> list[str]:
    return [label for label, drop in order if drop == "NONE"]


def _order_of(discipline: QueueDiscipline, seed: int = 1, waiting: str = "abcdefgh") -> list[str]:
    """One worker, held by `head`; `waiting` arrive in order while it is busy."""
    sim = Simulator(seed=seed)
    service = _service(discipline)
    order: list = []
    _submit(service, sim, "head", order)
    for label in waiting:
        _submit(service, sim, label, order)
    sim.run()
    return _served(order)


class TestTheOrder(unittest.TestCase):
    def test_fcfs_is_the_default_and_serves_in_arrival_order(self):
        config = ServiceConfig(name="svc", latency_median=1, latency_lognorm_sigma=0.1, workers=1)
        self.assertIs(config.queue_discipline, FCFS)
        self.assertEqual(_order_of(FCFS), ["head", *"abcdefgh"])

    def test_lifo_serves_the_newest_waiter_first(self):
        self.assertEqual(_order_of(LIFO), ["head", *"hgfedcba"])

    def test_lifo_takes_an_arrival_that_comes_while_older_ones_wait(self):
        """A stack, not a reversed batch: `late` arrives mid-way and jumps the rest."""
        sim = Simulator(seed=1)
        service = _service(LIFO)
        order: list = []
        _submit(service, sim, "head", order)
        for label in "abc":
            _submit(service, sim, label, order)
        # head ends at 10 ms and c starts; `late` arrives at 15 ms, during c.
        sim.schedule(ms_to_ns(15), lambda: _submit(service, sim, "late", order))
        sim.run()
        self.assertEqual(_served(order), ["head", "c", "late", "b", "a"])

    def test_random_serves_everyone_once_in_an_order_the_seed_fixes(self):
        first, again, other = (
            _order_of(RANDOM, seed=1),
            _order_of(RANDOM, seed=1),
            _order_of(RANDOM, seed=2),
        )

        self.assertEqual(first, again)
        self.assertEqual(sorted(first), sorted(["head", *"abcdefgh"]))
        self.assertEqual(first[0], "head")
        self.assertNotEqual(first, ["head", *"abcdefgh"])
        self.assertNotEqual(first, other)

    def test_random_is_uniform_over_the_waiters(self):
        """Who goes first out of four, over many seeds: each about a quarter."""
        firsts = Counter(_order_of(RANDOM, seed=seed, waiting="abcd")[1] for seed in range(800))

        self.assertEqual(sorted(firsts), list("abcd"))
        for label in "abcd":
            self.assertTrue(150 < firsts[label] < 250, firsts)

    def test_random_is_picked_at_dequeue_not_at_arrival(self):
        """An attempt that arrives later competes with the ones already waiting: over
        many seeds it is taken first about as often as either of them."""
        taken_first = Counter()
        for seed in range(600):
            sim = Simulator(seed=seed)
            service = _service(RANDOM)
            order: list = []
            _submit(service, sim, "head", order)
            _submit(service, sim, "a", order)
            _submit(service, sim, "b", order)
            sim.schedule(ms_to_ns(5), lambda s=service, m=sim, o=order: _submit(s, m, "late", o))
            sim.run()
            taken_first[_served(order)[1]] += 1

        self.assertEqual(sorted(taken_first), ["a", "b", "late"])
        for label in ("a", "b", "late"):
            self.assertTrue(150 < taken_first[label] < 250, taken_first)


class TestTheGenerator(unittest.TestCase):
    """`RANDOM` draws from `sim.rng()`; the other two leave it alone."""

    def _state_after(self, discipline: QueueDiscipline, waiting: str) -> tuple:
        sim = Simulator(seed=7)
        service = _service(discipline)
        order: list = []
        _submit(service, sim, "head", order)
        for label in waiting:
            _submit(service, sim, label, order)
        sim.run()
        return sim.rng().getstate()

    def test_lifo_consumes_the_stream_exactly_as_fcfs_does(self):
        self.assertEqual(self._state_after(LIFO, "abcd"), self._state_after(FCFS, "abcd"))

    def test_random_draws_for_a_choice_and_only_for_a_choice(self):
        # Four waiters: three picks among more than one entry.
        self.assertNotEqual(self._state_after(RANDOM, "abcd"), self._state_after(FCFS, "abcd"))
        # One waiter: nothing to choose, nothing drawn.
        self.assertEqual(self._state_after(RANDOM, "a"), self._state_after(FCFS, "a"))


class TestEveryKindOfEntry(unittest.TestCase):
    """Cancelled and expired entries, and the capacity, behave alike in every order."""

    def _run(self, discipline: QueueDiscipline):
        sim = Simulator(seed=3)
        service = _service(discipline, queue_capacity=3)
        order: list = []
        _submit(service, sim, "head", order)
        controls = {label: _submit(service, sim, label, order) for label in "ab"}
        _submit(service, sim, "expires", order, deadline=ms_to_ns(5))
        shed = _submit(service, sim, "shed", order)  # the queue is full: a, b, expires
        self.assertEqual(order, [("shed", "QUEUE_FULL")])
        self.assertFalse(shed.cancel())
        self.assertTrue(controls["a"].cancel())  # frees its slot at once
        self.assertEqual(service.queue_len(), 2)
        _submit(service, sim, "c", order)  # ... which `c` takes
        self.assertEqual(service.queue_len(), 3)
        sim.run()
        return order, service

    def test_the_same_attempts_are_served_shed_cancelled_and_expired(self):
        outcomes = {}
        for discipline in QueueDiscipline:
            order, service = self._run(discipline)
            outcomes[discipline] = sorted(order)
            self.assertEqual((service.queue_len(), service.in_flight, service.queue), (0, 0, []))
        self.assertEqual(outcomes[LIFO], outcomes[FCFS])
        self.assertEqual(outcomes[RANDOM], outcomes[FCFS])
        self.assertEqual(
            outcomes[FCFS],
            sorted([("shed", "QUEUE_FULL"), ("a", "CANCELLED"), ("expires", "DEADLINE"),
                    ("head", "NONE"), ("b", "NONE"), ("c", "NONE")]),
        )  # fmt: skip

    def test_only_the_order_of_service_differs(self):
        self.assertEqual(_served(self._run(FCFS)[0]), ["head", "b", "c"])
        self.assertEqual(_served(self._run(LIFO)[0]), ["head", "c", "b"])

    def test_an_expired_entry_holds_its_slot_until_the_server_reaches_it(self):
        """Under LIFO the server reaches an old entry LAST, so a dead entry at the bottom
        of the stack sheds load for as long as newer work keeps arriving."""
        for discipline, sheds_d in ((FCFS, False), (LIFO, True)):
            sim = Simulator(seed=3)
            service = _service(discipline, queue_capacity=2)
            order: list = []
            _submit(service, sim, "head", order)
            _submit(service, sim, "old", order, deadline=ms_to_ns(5))  # expires waiting
            _submit(service, sim, "b", order)
            # head ends at 10 ms. FCFS reaches `old` there, discards it and starts b; LIFO
            # starts b and leaves `old` at the bottom.
            sim.schedule(ms_to_ns(12), lambda s=service, m=sim, o=order: _submit(s, m, "c", o))
            sim.schedule(ms_to_ns(13), lambda s=service, m=sim, o=order: _submit(s, m, "d", o))
            sim.run()
            self.assertEqual(("d", "QUEUE_FULL") in order, sheds_d, (discipline, order))


class TestUnderOverload(unittest.TestCase):
    """What the order is FOR: with callers that give up, LIFO works on live requests."""

    def _successes(self, discipline: QueueDiscipline) -> tuple[int, int]:
        sim = Simulator(seed=11)
        service = _service(discipline, workers=2, queue_capacity=200, median_ms=10, sigma=0.3)
        client = ClientRuntime(
            cfg=ClientConfig(
                retry=NoRetryPolicy(),
                timeout_controller=StaticTimeoutController(attempt_timeout=ms_to_ns(100)),
            ),
            service=service,
        )
        # 300 rps offered to 2 workers x 10 ms = 200 rps of capacity.
        Workload(base_rps=300.0, duration_s=5.0, load_spikes=[], rng_seed=11).drive(
            sim, client.start_request
        )
        sim.run()
        reasons = Counter(root.attempts[-1].drop_reason for root in client.roots)
        return reasons[DropReason.NONE], len(client.roots)

    def test_lifo_completes_more_requests_within_their_deadline_than_fcfs(self):
        fcfs, offered = self._successes(FCFS)
        lifo, _ = self._successes(LIFO)
        random_order, _ = self._successes(RANDOM)

        # FCFS spends its workers on entries that are about to expire or already have.
        self.assertLess(fcfs, 0.5 * offered)
        self.assertGreater(lifo, 1.25 * fcfs)
        self.assertGreater(random_order, fcfs)
        self.assertLessEqual(lifo, offered)


class TestConfiguration(unittest.TestCase):
    def test_the_discipline_must_be_one_of_the_three(self):
        with self.assertRaisesRegex(ValueError, "queue_discipline must be a QueueDiscipline"):
            ServiceConfig(
                name="svc",
                latency_median=1,
                latency_lognorm_sigma=0.1,
                workers=1,
                queue_discipline="lifo",  # ty: ignore[invalid-argument-type]
            )
        self.assertEqual([d.value for d in QueueDiscipline], ["fcfs", "lifo", "random"])
        self.assertIs(QueueDiscipline("lifo"), LIFO)


if __name__ == "__main__":
    unittest.main()
