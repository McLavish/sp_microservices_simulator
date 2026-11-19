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
   bounds are present (`caller.py`). The request keeps the context it arrives
   with (`ctx=`), or gets a fresh one when the client has a `tenant`
   ([below](#request-context-what-travels-with-a-request)).
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
  `success AND within_deadline AND drop_reason == NONE`. The deadline is the
  caller's clock: an attempt still queued at the service when it passes is
  reported at the deadline itself, not when a worker next frees (see
  [services-and-workload](services-and-workload.md#the-drop-points)).
- **Policy result fan-out.** `_update_policy_results` (`caller.py`) walks the
  policy's layers and tells every one how the attempt ended:
  `record_attempt(context, success)`, whose default calls `add_result(success,
  now)` on a layer that has one. This is the *only* way circuit breakers and
  retry budgets learn outcomes. `context` is the `RetryContext` of the attempt
  that just ended -- its number, what it ended with (`NONE` for a success), its
  request context -- so a policy can tell a first attempt from a retry; a
  cancelled attempt is not reported at all.
  The walk is `retry_policies.policy_layers(policy)` -- outermost first, through
  each wrapper's `underlying_strategy` -- and it is the same one admission
  (`allow_request`), the per-request notification (`on_request_start`) and refunds
  (`rollback`) use, so a wrapper is reached by all four or by none.
- **Timeout feedback.** The attempt latency is reported to
  `timeout_controller.record_result(...)` (so an adaptive controller can adjust).
- **Retry or stop.** On failure it asks `retry.next_delay(context)` with that
  same context (`RetryContext(attempt, now, drop_reason, request)`), and refunds
  with it if the retry is then not issued. It retries only if the policy says so **and** the next
  start would land strictly before the global deadline (`caller.py`); otherwise
  the root completes as a failure. `drop_reason` is the cause the attempt was
  just booked under -- `DEADLINE` for one that ran out of time, whatever the
  service reported -- for a policy that decides by cause
  ([below](#retry-by-failure-cause)); every other policy ignores it.

> **Extending policies:** any wrapping policy must expose `underlying_strategy` (so
> results propagate) and any stateful policy should implement `add_result(success,
> now)` -- or override `record_attempt(context, success)` when it needs to know
> WHICH attempt ended (a budget only first attempts fund, a gate that counts
> retries apart). Miss this and the inner policy silently never sees outcomes. A policy
> that chooses between SEVERAL wrapped policies implements
> `sub_policies(context=None)` instead: every branch without a context, and only
> the branch that made the decision with one (`policy_layers` then refunds a
> vetoed retry in that branch alone).

### What the server learns about the deadline

`ClientConfig.server_deadline` decides which of the client's deadlines the target
is told, and so who enforces the attempt timeout:

| Value | The target is told | Who enforces the attempt timeout |
| --- | --- | --- |
| `"attempt"` (default) | `min(global deadline, now + attempt timeout)` | the **server**: it reports a waiting attempt at that instant and cuts one in service, freeing the worker |
| `"global"` | the request's global deadline (`None` if it has none) | the **client**, with a timer of its own |
| `"none"` | nothing | the client, for every deadline |

The default is a stack that sends its deadline with the call (gRPC's
`grpc-timeout`). The other two are a client-side timeout the server never hears
about: when the timer fires the client books a `DEADLINE` drop -- with no queue
size and no occupancy, since it left without learning either -- gives feedback,
and decides on a retry, while the attempt stays at the server as an **orphan**.
An orphan keeps its worker or its queue slot until the server is done with it
(in `"global"` mode the global deadline still cuts it), and its answer goes
nowhere: no record, no policy feedback. Under overload this is the difference
between a server that sheds dead work and one that serves it while live callers
time out behind it.

The timer and the real answer race once per attempt: whichever comes first is
delivered, the other is ignored. The timer is armed before the attempt is
submitted, so at the deadline instant it runs first and a tie goes to the
deadline, as the half-open rule requires. A report that arrives first disarms
the timer, including the `CANCELLED` report of an attempt an upstream cancels.
An orphan is out of reach of a later cancellation: nothing holds a handle on it.
When the deadline the client works to is the one the target is told (no attempt
timeout in `"global"` mode, for instance), there is no timer and the run is
event for event the default one.

### Request context: what travels with a request

By default a request carries nothing from hop to hop but its deadline. A
**request context** (`core.RequestContext`) adds what a component may need to
know about the request an attempt belongs to:

| Field | Meaning | Set by |
| --- | --- | --- |
| `tenant` | who the request belongs to | the client that starts the request (`ClientConfig.tenant`); unchanged down the tree |
| `attempt` | the number of this attempt **at the hop that issued it** (1 = that client's first try) | each client, for its own attempts |
| `is_retry` | **sticky**: this attempt is a retry, or was caused by one anywhere above it | each client: `incoming.is_retry or attempt > 1` |
| `retry_allowance` | a `core.RetryAllowance`, one mutable object shared by the whole request: retries it may still issue over every hop | whoever creates the context; handed on by reference |

A context exists only for a request that was given one. A client whose
`ClientConfig.tenant` is set creates one for every request it **starts**; a
caller can pass its own as `start_request(sim, ctx=...)`; and a request that
arrives with a context keeps it, so in a call tree the root client's tenant is
the tenant at every hop (an edge client's own `tenant` only matters when nothing
arrives). From there:

- the client passes the attempt's context to its target as
  `submit_attempt(..., ctx=...)` -- **only when there is one**, so a target that
  never serves such requests is called with the three arguments it always was;
- a `ServiceNodeRuntime` hands the context of the attempt it is serving to every
  downstream call that attempt makes, a `TransportLink` carries it over the wire,
  and a plain `ServiceRuntime` accepts and ignores it;
- the retry policy is asked with it: `RetryContext.request` is the context of
  the attempt that just ended (`None` without one).

`attempt` and `is_retry` answer two different questions. Under an upstream
retry, a downstream client's first call is `attempt=1, is_retry=True`: not a
retry *of that hop*, but load that exists only because something above retried.

Nothing in a scenario without a tenant creates a context, and a context that
nothing reads changes no event and no random draw (a seeded run with a tenant on
its root client reproduces its pinned trace).

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
  attempt 1), `now`, and `drop_reason`, the cause of the failure (`None` when the
  caller does not say).

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
| `TimeBasedCircuitBreakerPolicy` (`retry_policies.py`) | failure **rate** > `failure_threshold_rate` over `window_duration` (needs `min_requests`) | success rate among the half-open probes (at least `half_open_min_requests`) reaches `success_threshold_rate` (`>=`, so `1.0` means "all probes succeed") |

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
| `RetryBudgetPolicy` (`retry_policies.py`) | `budget_ratio`, `max_retries`, `underlying_strategy`, `min_retries_per_sec=None` | token economy: each **admitted request** deposits `100 * budget_ratio` tokens (cap `100 * max_retries`), each retry costs 100 and is refunded if the retry is never issued. Finagle semantics — the ratio is retries per *request*. No deposit TTL (documented divergence). `budget_ratio` is rationalized exactly; a ratio too small to represent (below roughly `1e-9`) is rejected rather than silently becoming a zero deposit. `min_retries_per_sec` (off by default) is a floor under the ratio: that many retries per second accrue with time, exactly, under the same cap -- for a client that sends too few requests for a ratio to fund a retry |

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

### Retry by failure cause

A deployed client does not retry everything: gRPC retries the status codes it is
told are retryable, and a deadline or a full queue is usually not among them.
`RetryByCausePolicy(policies)` maps a failure cause (`DropReason`) to the policy
that governs the retries of attempts that failed that way; **a cause with no
entry is not retried**.

```python
from msim.core import DropReason, ms_to_ns
from msim.retry_policies import FixedBackoffRetryPolicy, RetryBudgetPolicy, RetryByCausePolicy

backoff = FixedBackoffRetryPolicy(max_attempts=3, delay=ms_to_ns(5))

# The fixed list: retry what the server failed, nothing a deadline or a full queue ended.
listed = RetryByCausePolicy.for_causes(
    [DropReason.SERVER_FAILURE, DropReason.CIRCUIT_OPEN], backoff
)

# A budget per cause: server failures retry freely, the two overload signals
# share ONE budget of 10 % of the requests.
budget = RetryBudgetPolicy(budget_ratio=0.1, max_retries=10, underlying_strategy=backoff)
budgeted = RetryByCausePolicy({
    DropReason.SERVER_FAILURE: backoff,
    DropReason.DEADLINE: budget,
    DropReason.QUEUE_FULL: budget,
})
```

- The policy is stateless and draws nothing: `next_delay` is the chosen branch's,
  so an allowance (`max_attempts`) counts every attempt of the request, whatever
  each one failed with.
- The cause is the one the **client** books: `DEADLINE` for an attempt that
  finishes at or after its deadline, and for a chain node's caller the cause of
  the downstream call that failed the node -- so `QUEUE_FULL` and `CIRCUIT_OPEN`
  can come from deeper in the tree (`CIRCUIT_OPEN` only that way: a client whose
  own breaker is open never asks for a retry). `NONE` and `CANCELLED` are
  rejected as keys; no retry decision is ever asked about them.
- The same policy object may sit under several causes, and one budget may be
  shared by several. `policy_layers` yields each distinct layer **once**, so a
  shared budget gets one deposit per request and a breaker one result per
  attempt; and a retry that was granted and then vetoed by the global deadline is
  refunded **in the branch that granted it**, never in another (which charged
  nothing, and would be handed quota it never spent).
- Admission is not by cause -- a new request has none yet -- so a breaker placed
  *inside* a branch gates every request. Put breakers above the policy.

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
services. It wraps a `ServiceRuntime` and a list of
`DownstreamEdge(client, target, wave=0)` (`chain.py`), and it satisfies
`ServiceLike`, so a `ClientRuntime` drives it exactly like a plain service.

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
3. launches the downstream `client.start_request(...)` calls **wave by wave**
   (one wave, i.e. all of them concurrently, unless the edges say otherwise —
   see [Call waves](#call-waves-sequential-and-staged-calls)), propagating the
   parent deadline via `override_global_deadline` (unless `propagate_deadline`
   is off, see below) — a *synchronous* rejection (queue-full / circuit-open)
   on an early edge is remembered but does not skip the later launches of its
   wave, so downstream load never depends on edge order, and
4. finalizes with success only when *all* downstreams succeed; the **first**
   downstream failure short-circuits the parent (AND semantics), applied after
   the launch loop for synchronous rejections. A deadline guard event finalizes
   as `DEADLINE` if time runs out first, and is cancelled on success
   (`chain.py`).

### Call waves: sequential and staged calls

`DownstreamEdge.wave` (an `int >= 0`, default `0`) orders a node's calls. Edges
with the **same** wave are launched together; waves run in **ascending** order,
and a wave starts only when every call of the previous one has succeeded.

| Handler in the real system | Edges |
| --- | --- |
| calls B and C concurrently (the default) | B and C on wave 0 |
| calls B, then C, then D | B on wave 0, C on wave 1, D on wave 2 |
| four concurrent calls, then three more | four edges on wave 0, three on wave 1 |

```python
a.add_downstream(DownstreamEdge(client=a_to_b, target=b))            # wave 0
a.add_downstream(DownstreamEdge(client=a_to_c, target=c, wave=1))    # after B succeeded
```

Rules, all pinned by `tests/test_chain_waves.py`:

- **Only the order of the values matters** (gaps are fine), and a node whose
  edges all carry one value is the single concurrent fan-out it always was —
  the same events and the same RNG draws, so every earlier seeded result stands.
- **A "call" is the edge client's whole root request**, retries included: a wave
  is complete when each of its calls has succeeded, and end-to-end latency over
  sequential waves is the **sum** of the waves, each wave costing its slowest
  call.
- **A failure ends the node.** The first failed call fails the node (after the
  rest of its wave was launched); waves that had not started are never
  launched, so a failing early call sheds the load of the later ones.
- **The next wave goes through the event heap**, at the same instant, never
  from inside the completion callback. The callee that just answered has freed
  its worker but not yet resumed its queue, so a second call to the same callee
  made synchronously would overtake work that was queued first. Retries take
  the same hop.
- **One deadline for all the waves.** The deadline guard is armed once, before
  the first wave, and every wave is given the same absolute deadline, so a
  later wave has less of it left. A wave that completes at the deadline instant
  loses the tie: the node is a `DEADLINE` drop and the next wave never starts.
- **Worker occupancy** follows `hold_worker_during_fanout` as before: released
  once the first wave has been launched (tandem), or held across every wave.

What a wave does **not** model: there is one local service-time draw and one
failure roll, both *before* the first wave — no local work between or after
waves, and in tandem mode the continuation between two waves needs no worker.
There are no conditional or optional calls and no quorum: every edge is called
on every attempt, and N calls are N edges. Once the node has settled (deadline,
failure, cancellation) its remaining waves are skipped whatever the
`propagate_*` flags say — those describe calls already in flight.

### Local work last, optional calls, waiting for a wave

Three choices change how a node treats its calls. Each is off by default, and a
node that sets none of them is the node described above, event for event.

**`ServiceNodeRuntime.local_work = "last"`** puts the node's own work AFTER its
calls instead of before them: a handler that gathers what it needs and then
works on it.

| | `"first"` (default) | `"last"` |
| --- | --- | --- |
| order | local service time, failure roll, then the waves | the waves, then local service time and failure roll |
| service time drawn | when the worker is taken | when the last wave has ended (a latency injection active *then* applies) |
| a call fails | the local work was already done | no local work is done |
| deadline guard armed | before the first wave | at the start of the attempt |
| worker | released after the first wave, unless held | held throughout: needs `hold_worker_during_fanout=True` |

A tandem node hands its worker back before it waits on its calls and so has
none left to do work on afterwards: `local_work="last"` on a node that does not
hold its worker raises. The first wave is started through the event heap (the
attempt may begin inside another attempt's completion, where a call submitted
synchronously could overtake queued work), and the local stage that follows is
cut by the same deadline guard. A node with no downstreams is a plain service
either way.

**`DownstreamEdge.optional = True`** marks a call the node can do without. The
node still makes it and still waits for it -- its wave is not complete until the
call has ended -- but a failed optional call fails nothing: the wave counts it as
done and the node carries on to the next wave, or to its own work, or settles.
A required call in the same wave fails the node as usual.

**`ServiceNodeRuntime.wave_join = "wait_all"`** makes a wave a barrier. By
default (`"short_circuit"`) the first required call that fails settles the node
at once and the calls still in flight are abandoned. With `"wait_all"` the node
waits until every call of the wave has ended and then fails with the **first**
required failure; no later wave starts. A caller that collects all its answers
before it looks at any. The deadline guard does not wait: it settles the node
at its deadline in both modes.

```python
# A gateway that fetches three things, the third only if it can, then renders.
gateway = ServiceNodeRuntime(
    service=...,
    hold_worker_during_fanout=True,
    local_work="last",
    wave_join="wait_all",
)
gateway.add_downstream(DownstreamEdge(client=to_users, target=users))
gateway.add_downstream(DownstreamEdge(client=to_cart, target=cart))
gateway.add_downstream(DownstreamEdge(client=to_ads, target=ads, optional=True))
```

### Service time per handler node

A real service runs different code for different methods: a `Read` can take a
third of a millisecond where a `Write` on the same service takes six. One
`ServiceConfig` has one lognormal, so modelling the service as one node gives
both methods the same — wrong for each, and with an inflated spread.

Build one `ServiceNodeRuntime` per handler **over the same `ServiceRuntime`**
and give each its own `service_time`:

```python
from msim.service import ServiceTime

store = ServiceRuntime(cfg=store_cfg).bind()          # ONE station: workers, queue, faults
read = ServiceNodeRuntime(service=store, service_time=ServiceTime(ms_to_ns(0.3), 0.3))
write = ServiceNodeRuntime(service=store, service_time=ServiceTime(ms_to_ns(6), 0.4))
# a caller of Read targets `read`, a caller of Write targets `write`
```

| | Comes from |
| --- | --- |
| the lognormal the local stage draws from | the node's `service_time`, else the station's config |
| workers, queue, `QUEUE_FULL` shedding | the shared station |
| latency injections, partial failures, the 100 µs floor | the shared station |
| the name attempts are booked under (`Request.service`) | the shared station (`node.cfg.name`) |

`service_time=None` (the default) is the station's own distribution, exactly as
before, and an override takes the same single draw — `ServiceTime` floors a zero
sigma like `ServiceConfig` does rather than skipping the draw, so switching one
on never shifts the RNG stream of the rest of the run. Because `node.cfg` is the
*station's* config, read `node.local_service_time` for the distribution a given
handler actually draws from. `tests/test_node_service_time.py` pins the split.

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
  subtree — every call wave included — so capacity is consumed all the way down.

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

> **Why a decorator and not a change to `chain.py`.** `_launch_wave`
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
*after* the deadline. A stock `ServiceRuntime` never does — a queued attempt's
caller-clock event fires *at* the deadline and an attempt in service is cut
there — so that branch is reached only by a third-party `ServiceLike` that
ignores the deadline it is given. Delivery is then immediate instead of
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

## Two retry loops on one hop (`RetryLayer`)

A caller that retries is often in front of something that retries the same call
itself: a service whose handler has its own retry loop, a sidecar, a gateway.
`caller.RetryLayer(client, name=None)` makes a `ClientRuntime` usable as a
target, so the two loops stack, each with its own policy and its own timeouts:

```python
inner = ClientRuntime(cfg=service_side_cfg, service=service)   # 3 attempts, 100 ms each
outer = ClientRuntime(cfg=caller_cfg, service=RetryLayer(inner))  # 2 attempts, 1 s each
```

- One attempt of the caller is one whole **request** of the inner client, which
  makes as many attempts at the real target as its policy allows.
- The caller sees **one** answer, when the inner client has finished: the outcome
  of the inner request's last attempt, with that attempt's occupancy, queue size
  and censoring. Its policy is told once per call, not once per inner attempt.
- The caller's attempt deadline is the inner request's global deadline
  (tightened by the inner client's own, if it has one).
- The two keep separate books: the inner client's records are attempts at the
  real target, under its name; the caller's are booked under the layer's own
  name (`name`, by default `<target>.retries`), so counting attempts per service
  does not count a call twice.
- A request context passes through; the inner client numbers its own attempts.
- A caller that keeps its timeout to itself (`server_deadline="global"`) and
  leaves does not stop the inner loop: it goes on retrying for nobody.

Cancelling the caller's attempt cancels the inner request. A pending retry is
dropped, an attempt in flight is abandoned, `cancel()` returns `True`, and
`CANCELLED` has been reported. If the attempt in flight has already committed
to an outcome, `cancel()` returns `False`, the inner client issues no further
attempt, and that outcome is reported as itself -- so a genuine failure still
reaches the caller's breaker ([below](#cancelling-an-attempt-that-has-already-committed)).

## Admission control (`admission.py`)

Everything above that sheds load does it for a reason of its own: a station
refuses an attempt when its queue is full, a client's breaker stops sending.
**Admission control** is a policy decision taken on the server's side about one
attempt as it arrives: let it in, or turn it away before it costs a queue slot.
The attempts it is about are retries -- a server cannot stop its callers from
retrying, but it can decline the retries it is sent, and it sees all its callers
at once.

`AdmissionGate(target, controllers, retry_scope="hop")` is a `ServiceLike`
decorator, like `TransportLink`: a client (or a link, or an edge's client)
submits to the gate, and attempts stay booked under the target's name.

```python
from msim.admission import AdmissionGate, TokenBucketBudget

gated = AdmissionGate(service, [TokenBucketBudget(rate=30, burst=10)])
client = ClientRuntime(cfg=ClientConfig(..., tenant="a"), service=gated)
```

For each attempt:

1. Every controller is asked `check(request)`. Only if **all** agree is each told
   to `commit(request)` -- two phases, so a refusal by a later controller costs an
   earlier one none of its quota.
2. A refused attempt is reported **at once** as `DropReason.REJECTED`, with no
   queue size, on a handle that is already settled. The target never hears of it:
   no queue slot, no worker, no `admitted_total`. The client treats it as any
   failed attempt (feedback, retry decision), and a request whose final attempt
   was refused is booked rejected ([observability](observability.md)).
3. An admitted attempt is forwarded with the gate watching: when the target
   reports, every controller is told (`observe(outcome)`), then the caller. That
   includes the answer to an attempt whose caller has stopped waiting
   ([orphans](#what-the-server-learns-about-the-deadline)). A cancelled attempt is
   not observed -- abandoned work says nothing about the server.

The gate decides by the attempt's [request context](#request-context-what-travels-with-a-request),
so the requests that reach it must carry one (it raises otherwise: give the
client that starts them a `ClientConfig.tenant`). `retry_scope` says which
attempts are retries here: `"hop"`, a retry of the caller in front of the gate
(`attempt > 1`), or `"request"`, anything a retry anywhere above caused
(`is_retry`).

The gate keeps its own account of what it let in and has not seen end
(`outstanding`, `outstanding_retries`) and per-tenant `counts` (`attempts`,
`rejected`, `retries_asked`, `retries_admitted`). It does not read the target's
`in_flight`: that excludes what waits in the queue, includes work for callers
that have left, and a tandem chain node holds no worker while it waits.

| Controller | A retry is admitted ... | State |
| --- | --- | --- |
| `TokenBucketBudget(rate, burst)` | while a bucket refilled at `rate` per second has a token | global |
| `AimdBudget(min_rate, max_rate, initial_rate, ...)` | like the bucket, with a rate that is halved (`decrease_factor`) after a window in which more than `failure_threshold` of the attempts failed, and raised by `additive_step` otherwise | global |
| `SuccessFundedBudget(ratio, max_retries)` | while the bank holds a retry; every **success** deposits `ratio` of one | global |
| `FailureRateGate(failure_threshold, min_requests, window, open_delay)` | unless more than the threshold of the last window's outcomes failed; closes again when the success rate is back and `open_delay` has passed | global |
| `GoodputCoupledBudget(alpha, beta_down, beta_up, window, ...)` | while the tenant has fewer than `alpha x goodput` retries in the window, its goodput being its smoothed rate of successes here (quick to fall, slow to recover) | **per tenant** |
| `ConcurrencyRetryBudget(budget_percent, min_retry_concurrency)` | while the retries in flight are less than a share of all attempts in flight (Envoy's retry budget) | the gate's account |

All of them only ever refuse retries (`RetryController`): a first attempt passes
untouched and uncharged. Balances are exact (integer token units, rational
thresholds), so a budget does not depend on how often it is consulted.
`GoodputCoupledBudget` is the admission rule of [Arolla](https://github.com/eth-easl/Arolla)
(ETH EASL), ported; its count of retries is compared with a fractional allowance,
so a tenant always gets the first retry of a window while its allowance is above
zero -- one retry per tenant per window as a floor.

**On the caller's side.** `ControlledRetryPolicy(controllers, underlying_strategy)`
is the same rules as a layer of a client's own retry policy: it is told every
attempt's outcome (`record_attempt`), a retry needs every controller's agreement
and the wrapped policy's, the controllers are charged only for a retry that is
granted and refunded (`refund`) if the deadline then vetoes it. It sees one
client's attempts only: clients share a budget by sharing the controller object,
and `ConcurrencyRetryBudget`, which needs the server's load, refuses to work
there.

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
