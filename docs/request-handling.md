# Request handling: clients, retries, timeouts & chains

The client side of the simulation. A `ClientRuntime` turns one logical request into
a sequence of attempts against a service, governed by a **retry policy** and a
**timeout controller**. A `ServiceNodeRuntime` extends this to **chains** that fan
out to downstream services.

## The client & the attempt loop (`caller.py`)

`ClientRuntime` (`caller.py:62`) is configured with a `ClientConfig(retry,
timeout_controller)` and a `service` (anything matching `ServiceLike` — a plain
`ServiceRuntime` or a chain `ServiceNodeRuntime`). It accumulates completed
`RootRequest`s in `.roots` and exposes `.metrics()`.

### `start_request` (`caller.py:99`)

1. Build a `RootRequest` and compute its **global deadline** as
   `min(now + controller.global_timeout, override_global_deadline)` — whichever
   bounds are present (`caller.py:107`).
2. Ask the policy stack whether the request is allowed (`_allow_policy_request`). If
   a circuit breaker is open, record a single `CIRCUIT_OPEN` attempt and complete
   the root immediately.
3. Otherwise launch the first attempt.

### Each attempt

`_start_attempt` (`caller.py:145`) builds a `Request` with
`deadline = min(global_deadline, now + attempt_timeout)` and submits it to the
service. When the service calls back, `_on_attempt_done` (`caller.py:167`) decides
what happens next:

- **Half-open deadline (strict).** `within_deadline = end_time < req.deadline`
  (`caller.py:182`). An attempt that *finishes exactly at* its deadline is converted
  to a `DEADLINE` drop, not a success. Effective success requires
  `success AND within_deadline AND drop_reason == NONE`.
- **Policy result fan-out.** `_update_policy_results` (`caller.py:77`) walks the
  policy via its `underlying_strategy` chain and calls `add_result(success, now)` on
  every layer that supports it. This is the *only* way circuit breakers and retry
  budgets learn outcomes.
- **Timeout feedback.** The attempt latency is reported to
  `timeout_controller.record_result(...)` (so an adaptive controller can adjust).
- **Retry or stop.** On failure it asks `retry.next_delay(RetryContext(attempt,
  now))`. It retries only if the policy says so **and** the next start would land
  strictly before the global deadline (`caller.py:219`); otherwise the root
  completes as a failure.

> **Extending policies:** any wrapping policy must expose `underlying_strategy` (so
> results propagate) and any stateful policy should implement `add_result(success,
> now)`. Miss this and the inner policy silently never sees outcomes.

### Example: client with fixed-backoff retries

```python
from caller import ClientConfig, ClientRuntime
from core import ms_to_ns
from retry_policies import FixedBackoffRetryPolicy
from timeout_controllers import StaticTimeoutController

cfg = ClientConfig(
    retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(100)),  # 1 retry
    timeout_controller=StaticTimeoutController(global_timeout=None, attempt_timeout=ms_to_ns(50)),
)
client = ClientRuntime(cfg=cfg, service=svc)  # svc built in services doc
```

## Timeout controllers (`timeout_controllers.py`)

A controller supplies two timeouts and receives feedback:
`get_global_timeout()`, `get_attempt_timeout()`, `record_result(success, latency)`.

- **`StaticTimeoutController`** (`timeout_controllers.py:24`) — fixed
  `global_timeout` and `attempt_timeout`; `record_result` is a no-op. This is the
  default for nearly every experiment and the only one the RL env supports.
- **`AdaptiveTimeoutController`** (`timeout_controllers.py:41`) — adapts the
  **attempt** timeout to `target_percentile` (default p95) of observed latency times
  a `safety_margin` (default 2.0), clamped to `[min_timeout, max_timeout]`. Two
  deliberate design points:
  - Only **successful** latencies feed the percentile window — drops carry
    near-zero latency and would otherwise collapse the timeout downward and amplify
    drops (`timeout_controllers.py:122`).
  - The **global** timeout is intentionally *not* adapted: it's the caller's
    deadline budget. If the success rate falls below `min_success_rate`, the attempt
    timeout is inflated to cut failures.

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
| `CountBasedCircuitBreakerPolicy` (`retry_policies.py:101`) | `failure_threshold_ratio` e.g. `(5,10)` = 5 failures in last 10 | `success_threshold_ratio` successes among half-open probes |
| `TimeBasedCircuitBreakerPolicy` (`retry_policies.py:171`) | failure **rate** > `failure_threshold_rate` over `window_duration` (needs `min_requests`) | failure rate falls below `1 - success_threshold_rate` |

### Rate limiters & budget (wrapping, except leaky)

These throttle retries. The limiters require `context.now` to be set (the client
sets it).

| Policy | Params | Mechanism |
| --- | --- | --- |
| `LeakyRateLimiterPolicy` (`retry_policies.py:236`) | `max_requests`, `max_attempts`, `period` | leaky-bucket-as-queue: spaces retries `period/max_requests` apart, returns the delay. Does **not** wrap an underlying policy. |
| `BurstyRateLimiterPolicy` (`retry_policies.py:305`) | `max_requests`, `refill_rate`, `period`, `underlying_strategy` | token bucket with continuous refill; one token per retry |
| `FixedWindowBurstyLimiterPolicy` (`retry_policies.py:351`) | `max_requests`, `period`, `underlying_strategy` | fixed-window quota, reset on each window boundary |
| `RetryBudgetPolicy` (`retry_policies.py:260`) | `budget_ratio`, `max_retries`, `underlying_strategy` | token economy: each success awards `100 * budget_ratio` tokens (cap `100 * max_retries`), each retry costs 100 |

### Example: a circuit breaker around no-retry

```python
from core import s_to_ns
from retry_policies import CountBasedCircuitBreakerPolicy, NoRetryPolicy

breaker = CountBasedCircuitBreakerPolicy(
    failure_threshold_ratio=(5, 10),   # open at 5 failures / 10 calls
    success_threshold_ratio=(3, 5),    # close at 3 successes / 5 probes
    underlying_strategy=NoRetryPolicy(),
    half_open_delay=s_to_ns(1),
)
# use as ClientConfig(retry=breaker, timeout_controller=...)
```

## Chains & fan-out (`chain.py`)

A `ServiceNodeRuntime` (`chain.py:17`) is a service node that may call downstream
services. It wraps a `ServiceRuntime` and a list of `DownstreamEdge(client, target)`
(`chain.py:11`), and it satisfies `ServiceLike`, so a `ClientRuntime` drives it
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

Internally `submit_attempt` (`chain.py:41`) hands the service a custom `behavior`
that:

1. runs the node's **local** service time, then
2. checks local deadline/failure (drops as `DEADLINE` / `SERVER_FAILURE`), then
3. launches **all** downstream `client.start_request(...)` calls concurrently,
   propagating the parent deadline via `override_global_deadline` — a
   *synchronous* rejection (queue-full / circuit-open) on an early edge is
   remembered but does not skip later edges' launches, so downstream load never
   depends on edge order, and
4. finalizes with success only when *all* downstreams succeed; the **first**
   downstream failure short-circuits the parent (AND semantics), applied after
   the launch loop for synchronous rejections. A deadline guard event finalizes
   as `DEADLINE` if time runs out first, and is cancelled on success
   (`chain.py:106`).

### Worker occupancy: tandem vs hold-worker

`hold_worker_during_fanout` (`chain.py:32`) controls whether the node's worker is
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

### Example: a node fanning out to two services

```python
from chain import DownstreamEdge, ServiceNodeRuntime

a = ServiceNodeRuntime(service=svc_a)            # ServiceRuntimes built earlier
a.add_downstream(DownstreamEdge(client=a_to_b, target=ServiceNodeRuntime(service=svc_b)))
a.add_downstream(DownstreamEdge(client=a_to_c, target=ServiceNodeRuntime(service=svc_c)))
# a is ServiceLike: drive it with a top-level ClientRuntime(service=a)
```

## See also

- [services-and-workload](services-and-workload.md) — the `submit_attempt` / `behavior`
  contract these build on.
- [experiments](experiments.md) — `single/` demos for each policy, plus the chain
  scenarios.
- [observability](observability.md) — how attempts become metrics.

## Cancellation propagation

When AND semantics short-circuit a fan-out (a downstream fails, or the node's
deadline guard fires), the sibling calls already in flight are no longer wanted.
What happens to them is an explicit modelling choice on `ServiceNodeRuntime`,
alongside `hold_worker_during_fanout`:

| `propagate_cancellation` | Behaviour | Models |
| --- | --- | --- |
| `False` (default) | Siblings keep running to completion. They hold their workers for their full service time and record their own outcomes — an abandoned call that succeeds is still counted a success at that service. | A stack with no cancellation plumbing: plain HTTP/1.1 clients, older Java stacks. Preserves every pre-existing result. |
| `True` | Siblings are torn down: pending retries dropped, in-flight attempts abandon their worker immediately, queued attempts are skipped without ever consuming one. Recursive — cancelling a downstream that is itself a fan-out unwinds its whole subtree. Those roots are recorded with `DropReason.CANCELLED`. | Go `context` cancellation, gRPC deadline propagation. |

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
