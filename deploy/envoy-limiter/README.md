# A learned retry limiter as an Envoy filter

`msim.rl.limiter` trains a limiter in simulation: a retry quota per tenant, as a
share of that tenant's first attempts over the last second, set once a second
by a small policy from what the limiter itself counted. This directory is that
limiter outside the simulator: a Proxy-Wasm HTTP filter in Rust (about 240 KB)
that an Envoy proxy loads, and the script that turns a trained model into its
configuration. `docs/rl.md` ("On their Kubernetes prototype") has the
deployment it was written for and what it measured there.

## What it does

| Piece | Where | In the simulator |
| --- | --- | --- |
| The quota: a retry is let in while the tenant's retries of the last window are fewer than `share x` its first attempts of that window, or fewer than `floor` | `src/limiter.rs`, `Limiter::admit` | `admission.TenantRetryQuota` |
| The counts per tenant and window: attempts that ended, those that succeeded, retries that asked, retries let in, success share of the retries; their slow averages; the tenant's level | `Limiter::take`, `Limiter::decide` | `rl.limiter.LimiterKnobs.measure`, `RetryEnv(slow_s=...)` |
| Attempts in flight (with `in_flight`): let in and not yet answered, summed over the window's time per attempt let in, against the lowest that was in a window with no retry asking (an attempt is in flight until its response headers, or until its stream ends without any) | `Limiter::admit`, `ended`, `decide` | `rl.limiter.IN_FLIGHT` (`_GateTap`, `LimiterKnobs._crowding`) |
| The rule around the policy: the level at its default while no retry asks (`rest`), up at most `step_up` levels a window, no retry let in while more attempts are in flight than `in_flight_cap` times their number at rest (per Envoy thread, never below 8), the retries shown as asked at most `asked_per_failure` per failed first attempt | `Limiter::admit`, `Limiter::decide` | `QuotaGrid(rest, step_up, in_flight_cap, asked_per_failure)`, `rl.limiter._FlightCap` |
| The policy: tanh layers and a linear head, the largest logit per tenant | `Policy::act` | the PPO model, acting greedily (`rl.train.policy_weights`) |
| Envoy: attempt number and tenant from request headers, outcome from the response status, a 429 for a refused retry, a tick per window | `src/lib.rs` | `rl.limiter._GateTap` in an `AdmissionGate` |

A first attempt is never refused. A request is a retry when either attempt
header says more than 1 (`x-envoy-attempt-count`, set by the mesh, or
`X-Attempt-Number`, set by a client). A request that names no configured tenant
(a call inside the mesh) is kept apart: it is in no tenant's counts, the policy
never sees it, and its retries get the fixed `other_share` (10%).

Four things the simulator has no notion of:

- **Threads.** Every Envoy worker thread has a filter instance of its own, and
  Envoy does not spread connections evenly over them (one thread of a gateway
  was measured with 95% of the traffic). The policy reads the load off the
  counts, so the threads of a proxy pool them (`pool_threads`, on by default):
  each leaves its last window in the proxy's shared data and decides on the
  sum. `rates` are therefore requests per second per PROXY. The quota needs no
  pooling: it is a share of a thread's own first attempts.
- **No traffic.** A simulated service always has some; a proxy starts before
  its traffic does. A window in which nothing arrived and nothing ended is not
  something the policy was trained on, and what it answers there sticks (its
  own level is part of what it is shown next): a filter was seen to open every
  quota at start-up and stay open. Such a window is taken as the start of an
  episode: the default level, the averages starting afresh.
- **Callers that give up.** An attempt whose caller went away before any
  response is counted as ended and failed when its stream closes, as the
  simulator's gate is told of a deadline.
- **What a refusal costs the caller.** Envoy closes an HTTP/1 connection it
  answers before the request is complete, and a filter decides at the request
  headers, before a body has arrived. A 429 sent there costs the caller its
  connection (Arolla's filter does this), and in a storm of refused retries
  that is a storm of new connections: 3500 a second were measured, and after
  a 30 s outage the callers' new connections hung for another half minute. So
  the answer to a refused retry with a body is held for one to two ticks of
  `hold_refusal_ms` (2 ms), by when a small body has arrived, and the
  connection survives. `hold_refusal_ms: 0` answers at once.

## Build and test

```bash
cargo test                                     # the limiter natively (no Envoy needed)
cargo build --target wasm32-wasip1 --release   # target/wasm32-wasip1/release/learned_filter.wasm
```

`rust-toolchain.toml` pins the toolchain and the target. `tests/fixture.rs`
checks the filter's forward pass against the trained model on 64 observations;
it needs `tests/fixture.json`, which `export.py --fixture` writes (it is not in
the repository: it holds a model's weights) and is skipped without it.

## Configure

```bash
uv run --extra train python deploy/envoy-limiter/export.py MODEL.zip limiter.json \
    --rates 1000 --tenants checkout --fixture deploy/envoy-limiter/tests/fixture.json
```

`limiter.json` is the `limiter` block of the plugin configuration:

```yaml
pluginConfig:
  attempt_header: x-envoy-attempt-count      # the defaults, as in Arolla's filter
  attempt_header_secondary: X-Attempt-Number
  tenant_header: X-Retry-Client-Type
  reject_status: 429
  pool_threads: true
  hold_refusal_ms: 2
  log_windows: false      # one warning line per instance and window, for a run being looked at
  limiter:                # export.py
    tenants: [checkout]
    rates: [1000.0]       # requests per second per proxy when all is well
    shares: [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 3.0]
    default_level: 0
    window_ms: 1000
    slow_s: 10.0
    other_share: 0.1
    one_level: false      # one level for every tenant, from their counts together
    floor: 0              # retries per tenant, window and thread let in whatever the level (a probe)
    rest: false           # the default level while no retry asks (`QuotaGrid.rest`)
    in_flight: false      # show the policy the attempts in flight (`QuotaGrid.in_flight`)
    step_up: 0            # levels a quota may rise per window, 0: any (`QuotaGrid.step_up`)
    hold_above: null      # no quota rises while the in-flight entry is above this (`QuotaGrid.hold_above`)
    in_flight_cap: null   # a retry is refused above this multiple of the attempts in flight at rest (`QuotaGrid.in_flight_cap`)
    asked_per_failure: null  # the policy is shown at most this many asking retries per failed first attempt (`QuotaGrid.asked_per_failure`)
    max_attempt: null     # a retry beyond this attempt of its request is refused (`QuotaGrid.max_attempt`)
    confirm: false        # no level rises until retries asked in two windows running (`QuotaGrid.confirm`)
    hand_rule: null       # {alive, down_above, hold_above}: `rl.limiter.hand_rule` decides instead of the policy
    policy: {hidden: [...], head: {...}}
```

`rates` is the scale of everything the policy is shown: a model trained around
a nominal rate reads twice that rate as twice the load. Give each proxy the
tenant's nominal rate divided by the number of proxies that share it. A model
trained for one tenant can be configured with several. The filter then asks it
once per tenant, each against its own rate; or, with `one_level: true`
(`export.py --one-level`), once for all of them, on their counts and rates
added up, and gives every tenant that level. A tenant's quota is a share of
its own first attempts either way, so with one level for everybody the retries
let in are in proportion to the requests each tenant sends, which a level per
tenant does not give (a tenant that asks for many retries reads as trouble and
is given a lower one).

With Istio the module is a `WasmPlugin` (`url`, `sha256`, the configuration
above); it exports the counters `learned_first_attempts_total`,
`learned_retries_admitted_total`, `learned_retries_rejected_total` and the
gauge `learned_level_sum`.
