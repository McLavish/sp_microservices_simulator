# Request handling: clients, retries, timeouts & chains

The client side of the simulation. A `ClientRuntime` turns one logical request into
a sequence of attempts against a service, governed by a **retry policy** and a
**timeout controller**. A `ServiceNodeRuntime` extends this to **chains** that fan
out to downstream services.

## The client & the attempt loop (`caller.py`)

`ClientRuntime` (`caller.py`) is configured with a `ClientConfig(retry,
timeout_controller)` and a `service` (anything matching `ServiceLike` — a plain
`ServiceRuntime` or a chain `ServiceNodeRuntime`). It accumulates completed
`RootRequest`s in `.roots` and exposes `.metrics()`.

### `start_request` (`caller.py`)

1. Build a `RootRequest` and compute its **global deadline** as
   `min(now + controller.global_timeout, override_global_deadline)` — whichever
   bounds are present (`caller.py`).
2. Ask the policy stack whether the request is allowed (`_allow_policy_request`). If
   a circuit breaker is open, record a single `CIRCUIT_OPEN` attempt and complete
   the root immediately.
3. Otherwise announce the *admitted* request to every policy layer
   (`_notify_request_start` -> `on_request_start`) — this is the retry-budget
   deposit, and it fires **below** the gate, so a request an outer breaker sheds
   never funds anybody's retries (Finagle's `RetryFilter` likewise deposits per
   request that enters the filter).
4. Launch the first attempt.

### Each attempt

`_start_attempt` (`caller.py`) builds a `Request` with
`deadline = min(global_deadline, now + attempt_timeout)` and submits it to the
service. When the service calls back, `_on_attempt_done` (`caller.py`) decides
what happens next:

- **Half-open deadline (strict).** `within_deadline = end_time < req.deadline`
  (`caller.py`). An attempt that *finishes exactly at* its deadline is converted
  to a `DEADLINE` drop, not a success. Effective success requires
  `success AND within_deadline AND drop_reason == NONE`.
- **Policy result fan-out.** `_update_policy_results` (`caller.py`) walks the
  policy via its `underlying_strategy` chain and calls `add_result(success, now)` on
  every layer that supports it. This is the *only* way circuit breakers and retry
  budgets learn outcomes.
- **Timeout feedback.** The attempt latency is reported to
  `timeout_controller.record_result(...)` (so an adaptive controller can adjust).
- **Retry or stop.** On failure it asks `retry.next_delay(RetryContext(attempt,
  now))`. It retries only if the policy says so **and** the next start would land
  strictly before the global deadline (`caller.py`); otherwise the root
  completes as a failure.

> **Extending policies:** any wrapping policy must expose `underlying_strategy` (so
> results propagate) and any stateful policy should implement `add_result(success,
> now)`. Miss this and the inner policy silently never sees outcomes.

### Example: client with fixed-backoff retries

```python
from msim.caller import ClientConfig, ClientRuntime
from msim.core import ms_to_ns
from msim.retry_policies import FixedBackoffRetryPolicy
from msim.timeout_controllers import StaticTimeoutController

cfg = ClientConfig(
    retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(100)),  # 1 retry
    timeout_controller=StaticTimeoutController(global_timeout=None, attempt_timeout=ms_to_ns(50)),
)
client = ClientRuntime(cfg=cfg, service=svc)  # svc built in services doc
```

## Timeout controllers (`timeout_controllers.py`)

A controller supplies two timeouts and receives feedback:
`get_global_timeout()`, `get_attempt_timeout()`, `record_result(success, latency)`.

- **`StaticTimeoutController`** (`timeout_controllers.py`) — fixed
  `global_timeout` and `attempt_timeout`; `record_result` is a no-op. This is the
  default for nearly every experiment.
- **`AdaptiveTimeoutController`** (`timeout_controllers.py`) — adapts the
  **attempt** timeout to `target_percentile` (default p95) of observed latency times
  a `safety_margin` (default 2.0), clamped to `[min_timeout, max_timeout]`. Two
  deliberate design points:
  - Only **successful** latencies feed the percentile window — drops carry
    near-zero latency and would otherwise collapse the timeout downward and amplify
    drops (`timeout_controllers.py`).
  - The **global** timeout is intentionally *not* adapted: it's the caller's
    deadline budget. If the success rate falls below `min_success_rate`, the attempt
    timeout is inflated to cut failures.
  - `initial_attempt_timeout` is served verbatim until `min_samples` successful
    latencies accumulate, so it must itself lie within `[min_timeout,
    max_timeout]`. A config that violates that (or `max_timeout < min_timeout`)
    raises at construction rather than being silently clamped — clamping would
    override an explicitly configured value.

## Retry policies (`retry_policies.py`)

Every policy implements `RetryPolicy`:

- `allow_request(now) -> bool` — gate a *new* root request (default `True`; circuit
  breakers override it).
- `next_delay(context) -> (should_retry, delay_ns)` — decide on a retry after a
  failed attempt. `RetryContext` carries `attempt` (1-indexed; the original try is
  attempt 1) and `now`.

### Backoff strategies

| Policy | Params | Behavior |
| --- | --- | --- |
| `NoRetryPolicy` | — | never retries |
| `FixedBackoffRetryPolicy` | `max_attempts`, `delay` | constant `delay` until `attempt == max_attempts` |
| `ExponentialBackoffRetryPolicy` | `max_attempts`, `initial_delay`, `max_delay` | `min(initial_delay * 2^(attempt-1), max_delay)` |
| `ExponentialBackoffWithJitterRetryPolicy` | + `rng`, `jitter_mode` | exponential then jitter: `FULL` = `U(0, d)`, `EQUAL` = `d/2 + U(0, d/2)` (`DECORRELATED` raises) |

`max_attempts` counts the **original try plus retries** (so `2` = one retry). Pass
the simulator's RNG (`sim.rng()`) into the jittered policy so jitter is reproducible.

### Circuit breakers (wrapping)

Both wrap an `underlying_strategy`, track outcomes via `add_result`, and reject new
requests via `allow_request` while `OPEN`. State is `CLOSED -> OPEN -> HALF_OPEN`;
after `half_open_delay` the breaker admits a probe. (The half-open probe stage is
only partially implemented.)

| Policy | Opens when | Closes when |
| --- | --- | --- |
| `CountBasedCircuitBreakerPolicy` (`retry_policies.py`) | `failure_threshold_ratio` e.g. `(5,10)` = 5 failures in last 10 | `success_threshold_ratio` successes among half-open probes |
| `TimeBasedCircuitBreakerPolicy` (`retry_policies.py`) | failure **rate** > `failure_threshold_rate` over `window_duration` (needs `min_requests`) | failure rate falls below `1 - success_threshold_rate` |

Both validate their configuration in `__post_init__` and raise `ValueError`:
thresholds within range, `min_requests >= 1`, `window_duration > 0`,
`half_open_delay >= 0`. A window that can never fill (a non-positive
`window_duration` evicts every prior result) would otherwise leave the breaker
permanently closed with no signal.

### Rate limiters & budget (wrapping, except leaky)

These throttle retries. The limiters require `context.now` to be set (the client
sets it).

| Policy | Params | Mechanism |
| --- | --- | --- |
| `LeakyRateLimiterPolicy` (`retry_policies.py`) | `max_requests`, `max_attempts`, `period` | leaky-bucket-as-queue: spaces retries exactly `period/max_requests` apart (the reserved slot is kept as a rational instant, so an uneven division does not accumulate a rounding gain), returns the whole-ns delay. Does **not** wrap an underlying policy. |
| `BurstyRateLimiterPolicy` (`retry_policies.py`) | `max_requests`, `refill_rate`, `period`, `underlying_strategy` | token bucket with continuous refill; one token per retry |
| `FixedWindowBurstyLimiterPolicy` (`retry_policies.py`) | `max_requests`, `period`, `underlying_strategy` | fixed-window quota, reset on each window boundary |
| `RetryBudgetPolicy` (`retry_policies.py`) | `budget_ratio`, `max_retries`, `underlying_strategy` | token economy: each **admitted request** deposits `100 * budget_ratio` tokens (cap `100 * max_retries`), each retry costs 100 and is refunded if the retry is never issued. Finagle semantics — the ratio is retries per *request*. No deposit TTL (documented divergence). `budget_ratio` is rationalized exactly; a ratio too small to represent (below roughly `1e-9`) is rejected rather than silently becoming a zero deposit |

The three wrapping throttles share one `next_delay`/`rollback` implementation
(`_ThrottledRetryPolicy`): quota is charged **only** on the path that returns
`allow=True`, and `rollback` refunds exactly one unit. That matters because the
client asks for a delay before it knows whether the global deadline will veto
the retry (`caller.py`); a policy whose `next_delay` has side effects must
therefore implement `rollback`. `RetryPolicy.supports_deferred_rollback` says
whether that refund is still correct when it arrives *late* (a root cancelled
mid-backoff). It defaults to **False** — capped balances and single-slot
reservations both mis-refund out of order, so a new side-effecting policy is
safe without opting out; skipping the refund over-counts, which is the
conservative direction.

Both breakers likewise share their admission half (`_CircuitBreakerBase`:
`allow_request`, the OPEN -> HALF_OPEN transition and retry delegation), so the
uncapped half-open probe admission noted above is one code path, not two.

### Example: a circuit breaker around no-retry

```python
from msim.core import s_to_ns
from msim.retry_policies import CountBasedCircuitBreakerPolicy, NoRetryPolicy

breaker = CountBasedCircuitBreakerPolicy(
    failure_threshold_ratio=(5, 10),   # open at 5 failures / 10 calls
    success_threshold_ratio=(3, 5),    # close at 3 successes / 5 probes
    underlying_strategy=NoRetryPolicy(),
    half_open_delay=s_to_ns(1),
)
# use as ClientConfig(retry=breaker, timeout_controller=...)
```

## Chains & fan-out (`chain.py`)

A `ServiceNodeRuntime` (`chain.py`) is a service node that may call downstream
services. It wraps a `ServiceRuntime` and a list of `DownstreamEdge(client, target)`
(`chain.py`), and it satisfies `ServiceLike`, so a `ClientRuntime` drives it
exactly like a plain service.

```
            ClientRuntime
                 |
                 v
            A (ServiceNodeRuntime)
          local service time
                 |  fan out (concurrent)
        +--------+--------+
        v                 v
        B                 C        AND semantics:
   (downstream)     (downstream)   A succeeds only if B AND C succeed;
        ...               ...      first failure short-circuits A.
```

Internally `submit_attempt` (`chain.py`) hands the service a custom `behavior`
that:

1. runs the node's **local** service time, then
2. checks local deadline/failure (drops as `DEADLINE` / `SERVER_FAILURE`), then
3. launches **all** downstream `client.start_request(...)` calls concurrently,
   propagating the parent deadline via `override_global_deadline` (unless
   `propagate_deadline` is off, see below) — a
   *synchronous* rejection (queue-full / circuit-open) on an early edge is
   remembered but does not skip later edges' launches, so downstream load never
   depends on edge order, and
4. finalizes with success only when *all* downstreams succeed; the **first**
   downstream failure short-circuits the parent (AND semantics), applied after
   the launch loop for synchronous rejections. A deadline guard event finalizes
   as `DEADLINE` if time runs out first, and is cancelled on success
   (`chain.py`).

### Worker occupancy: tandem vs hold-worker

`hold_worker_during_fanout` (`chain.py`) controls whether the node's worker is
held while waiting on downstreams:

```
tandem (default, =False)          hold-worker (=True)
worker: [local]                    worker: [local .... waiting on B,C ....]
        ^ released here                            ^ released only after fan-out
        downstream calls run               downstream calls run
        while worker is free              while worker stays occupied
```

- **`False` (tandem, default):** `release_worker()` fires after the local stage, so
  the node's capacity is freed during the downstream wait. Capacity is **not**
  consumed down the tree — this is the model the multichain experiments rely on.
- **`True` (synchronous thread-per-request):** the worker is held for the whole
  subtree, so capacity is consumed all the way down.

`tests/test_chain_worker_model.py` asserts the in-flight counts for both modes.

### Deadline propagation

`propagate_deadline` (`chain.py`) is the third per-node switch, next to
`hold_worker_during_fanout` and `propagate_cancellation`:

| `propagate_deadline` | Behaviour | Models |
| --- | --- | --- |
| `True` (default) | The node passes its own attempt deadline down each edge as `override_global_deadline`, so a downstream call runs under `min(its own controller, that deadline)` and books a `DEADLINE` drop of its own when it expires. | A stack that plumbs the caller's deadline through the wire: Go contexts, gRPC's `grpc-timeout` header. |
| `False` | The deadline stops here. Downstream calls run under their own controller timeouts only, so they keep running (and consuming downstream capacity) after this node's attempt was abandoned upstream. | A stack with no deadline propagation — the modelling-choice ablation "no propagated deadline". |

The node still enforces its own deadline on itself either way, so the *caller's*
outcome is unchanged by the switch; only the downstream's view of the deadline
disappears. `tests/test_chain_scenarios.py::TestChainScenarios::test_propagate_deadline_switch`
pins both branches.

### Example: a node fanning out to two services

```python
from msim.chain import DownstreamEdge, ServiceNodeRuntime

a = ServiceNodeRuntime(service=svc_a)            # ServiceRuntimes built earlier
a.add_downstream(DownstreamEdge(client=a_to_b, target=ServiceNodeRuntime(service=svc_b)))
a.add_downstream(DownstreamEdge(client=a_to_c, target=ServiceNodeRuntime(service=svc_c)))
# a is ServiceLike: drive it with a top-level ClientRuntime(service=a)
```

## Transport delay (`transport.py`)

`TransportLink` (`transport.py`) puts a network delay on **each leg** of the
round trip between a client and the service it calls. It is a `ServiceLike`
*decorator*: it forwards `cfg` and implements `submit_attempt`, so a client
drives it exactly like the thing it wraps.

```python
from msim.caller import ClientRuntime
from msim.chain import DownstreamEdge
from msim.core import ms_to_ns
from msim.transport import TransportDelay, TransportLink

delay = TransportDelay(median=ms_to_ns(2), sigma=0.4)
a_to_b = ClientRuntime(cfg=edge_cfg, service=TransportLink(node_b, delay))
a.add_downstream(DownstreamEdge(client=a_to_b, target=node_b))
```

The link wraps the target *as the edge's client sees it*: `node_b`'s own
`ServiceNodeRuntime.service` stays a `ServiceRuntime`, and `DownstreamEdge` is
unchanged. The delay lives on the caller's side of the edge, not inside the node.

> **Why a decorator and not a change to `chain.py`.** `_launch_fanout`
> (`chain.py`) runs once per downstream *root request*, but retries are issued
> from `ClientRuntime._start_attempt` (`caller.py`) and never pass through
> `chain.py` at all. A delay added to the fan-out loop would therefore be charged
> once per root instead of once per attempt. Wrapping the service is the one
> place every attempt — first try and retry alike — crosses.

### Per-leg, per-attempt draws

Every attempt draws **two independent delays**: `d_f` at submit (forward leg) and
`d_r` when the target reports (return leg). Both come from `TransportDelay.sample`
(`transport.py`), a lognormal around `median` with shape `sigma`, drawn through
`sim.rng()` so a seed reproduces the run.

`sigma == 0.0` means *exactly* `median` and takes **no** RNG draw at all. Unlike
`ServiceConfig.latency_lognorm_sigma`, `TransportDelay.sigma` is deliberately
**not** floored at `MIN_LOGNORM_SIGMA` (`service.py`), so a genuinely
deterministic link is expressible and testable. A scenario with no `TransportLink`
in it draws nothing and schedules nothing extra, so existing seeded results are
untouched.

### Deadlines on the wire

The attempt deadline the client computed (`min(global_deadline, now +
attempt_timeout)`) is passed through the link unchanged and applied to both legs.
Both rules are the strict half-open rule — `end_time < deadline` is required for
success — carried onto the transport leg:

| Leg | Rule |
| --- | --- |
| forward | `arrive_at = now + d_f`. If `arrive_at >= deadline` the request expires in transit: the target is **never called**, and the client is delivered a `DEADLINE` drop at *exactly* the deadline (the caller's clock fires there, not where the packet would have landed). |
| return | `deliver_at = now + d_r`. If `deliver_at >= deadline` the response is late: delivery is clamped to the deadline and rewritten to `DEADLINE`, while `service_time_ns` and `service_time_censored` stay **exactly** as the service reported them — the work really was done, only the answer was late. `deliver_at == deadline` is a `DEADLINE`, not a success. |

Both rows really clamp to `max(deadline, now)`, because the past cannot be
scheduled (`Simulator.schedule` raises on it). The `max` only bites when the
clock is already past the deadline: at submit on the forward row (a deadline
handed down that has already elapsed — the analogue of `ServiceRuntime`'s
already-expired drop), or on the return row when the target itself reports
*after* the deadline — a `ServiceRuntime` deadline drop at dequeue fires whenever
a worker frees late (`service.py`). Delivery is then immediate instead of
retroactive; everywhere else `now` is still short of the deadline and the clamp
lands on the deadline itself.

An attempt that expires in forward transit never reaches a queue, so the target's
`admitted_total` is untouched (it counts arrivals *at the queue*) and
`queue_size_at_end` is `None` — the same convention `CIRCUIT_OPEN` uses in
`caller.py`, because reporting `0` would inject a fabricated "queue is empty"
sample into `queue_avg_at_attempt_end`.

### Cancellation, phase by phase

`submit_attempt` returns an `AttemptControl` immediately, and `cancel()` keeps the
truthful-`False` contract that `chain.py`'s teardown and
`ClientRuntime._cancel_root` rely on: `True` means something was actually torn
down *and* `on_done` has already been delivered synchronously.

| Phase | `cancel()` does | Returns |
| --- | --- | --- |
| forward transit | cancels the pending arrival (or the pending expiry event) and reports `CANCELLED` synchronously, with queue size `None` | `True` |
| at the service | delegates to the inner `AttemptControl` and returns **its** answer; the outer handle settles only when the inner said `True`, so a target that has already committed keeps its own outcome | the inner's |
| return transit | cancels the pending delivery and reports `CANCELLED` synchronously, keeping the occupancy, queue size and censoring flag the service reported | `True` |
| delivered | nothing — the attempt is settled | `False` |

`on_done` fires **exactly once** per attempt on every one of those paths. A
`CANCELLED` outcome coming back from the target is delivered to the client
synchronously with *no* return leg: cancellation is client-initiated and needs no
network.

The stock targets (`ServiceRuntime`, `ServiceNodeRuntime`, a nested
`TransportLink`) all report `CANCELLED` from inside their own `cancel()`, which
settles the attempt before that `cancel()` even returns. A target is not obliged
to: `AttemptControl.cancel()` only promises `True` when something was torn down,
so a target may answer with a *real* outcome instead ("I already have the answer,
take it now"). The link then treats the attempt as being in return transit — it
kills the delivery it had just put on the wire and reports `CANCELLED` carrying
the occupancy, queue size and censoring flag the target reported. Letting that
delivery survive would hand the client a second outcome for one attempt a leg
later, which a `ClientRuntime` meets as `Root request was already done!`. `_deliver`
therefore cancels *any* event still pending, rather than merely dropping the
reference to it.

### `service_time_ns` excludes transport

`Request.service_time_ns` keeps its meaning unchanged — worker occupancy at the
handling service, transport **excluded**. End-to-end attempt latency including
both legs is `Request.interval` (`d_f` + queue + service + `d_r`). The split is
load-bearing: `service_time_ns` is the service-time sample any utilization or
`E[S]` estimate is built from, so folding wire time into it would inflate `E[S]`
exactly the way reporting a tandem node's downstream wait as occupancy would.

`Request.service` stays the *target's* `cfg.name`: the link forwards `cfg`, so an
attempt is booked under the service that handled it, never under the link.

## See also

- [services-and-workload](services-and-workload.md) — the `submit_attempt` / `behavior`
  contract these build on.
- [examples](examples.md) — `single/` demos for each policy, plus the chain
  scenarios.
- [observability](observability.md) — how attempts become metrics.
- [architecture](architecture.md) — the module map, the half-open time convention
  the transport deadline rules follow, and the event heap a `TransportLink`
  schedules its two legs on.

## Cancellation propagation

When AND semantics short-circuit a fan-out (a downstream fails, or the node's
deadline guard fires), the sibling calls already in flight are no longer wanted.
What happens to them is an explicit modelling choice on `ServiceNodeRuntime`,
alongside `hold_worker_during_fanout`:

| `propagate_cancellation` | Behaviour | Models |
| --- | --- | --- |
| `False` (default) | Siblings keep running to completion. They hold their workers for their full service time and record their own outcomes — an abandoned call that succeeds is still counted a success at that service. | A stack with no cancellation plumbing: plain HTTP/1.1 clients, older Java stacks. Preserves every pre-existing result. |
| `True` | Siblings are torn down: pending retries dropped, in-flight attempts abandon their worker immediately, queued attempts are skipped without ever consuming one. Those roots are recorded with `DropReason.CANCELLED`. | Go `context` cancellation, gRPC deadline propagation. |

**The flag is per node, and so is the recursion.** Cancelling a downstream that
is itself a fan-out runs *that node's* teardown, which unwinds its children only
if it too sets `propagate_cancellation`. The first descendant with the flag off
keeps its own subtree running — setting the flag on the fan-out root alone buys
no propagation below level 1. That is deliberate (each node models its own
architecture, so a Go service calling a legacy one is expressible), but it means
whole-subtree teardown requires the flag on every node along the path.

Cancelled roots land in their own bucket (`Summary.dropped_cancelled`,
`failure_cancelled` in the CSV) and are deliberately **not** folded into the
deadline/queue/server drop counts. Cancelled work did not fail; charging it to a
failure category would count one root cause twice — once where it actually
failed, and again at every sibling it cancelled.

Mechanically, `ServiceRuntime.submit_attempt` returns an `AttemptControl` whose
`cancel()` is a no-op once the attempt has settled, so a cancel racing a
completion cannot double-complete. Custom behaviours publish their own teardown
via the `register_cancel` callback; that teardown must still route through
`finalize`, so the exactly-once guards cover the cancel path too.

### Cancelling an attempt that has already committed

A cancel can arrive after the abandoned node committed to an outcome but before
it finalized — `_fail_from` runs `_commit()` -> `_abandon_siblings()` ->
`_finalize()`, and in a diamond topology the teardown of one sibling can unwind
back into an ancestor's fan-out. `AttemptControl.cancel()` then returns False,
and `ClientRuntime._cancel_root` splits the outcome in two:

- **The attempt keeps its own drop reason.** It really did fail (or time out) at
  the service, so it is reported as itself and still feeds the circuit breaker
  and the timeout controller. Rewriting it as `CANCELLED` would hide a genuine
  failure from the breaker.
- **The root is still abandoned** (`AttemptCtx.abandoned`). No further attempt
  is issued — the retry would be load nobody is waiting for, which is exactly
  what `propagate_cancellation` exists to eliminate — and the root is classified
  `CANCELLED`, not by the reason of an attempt whose caller had already given up.

`tests/test_cancellation_client.py` pins both halves.


## `service_time_ns` is worker occupancy, not node latency

`Request.service_time_ns` records how long a worker of the handling service was
actually **held**. It is queue-exclusive, and for a fan-out node it depends on
the occupancy model:

| `hold_worker_during_fanout` | `service_time_ns` covers |
| --- | --- |
| `False` (default, tandem) | the local stage only — the worker is handed back before the downstream wait |
| `True` (thread-per-request) | the whole subtree, because the worker is genuinely held throughout |

End-to-end node latency is always `Request.interval`, so nothing is lost by the
narrower definition. The distinction matters for any queueing analysis that
takes `E[S]` and utilization from this field: reporting a tandem node's
downstream wait as occupancy inflates `E[S]` by the entire subtree latency and
makes the estimate meaningless.
