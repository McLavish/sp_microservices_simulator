//! A learned retry limiter as an Envoy HTTP filter (Proxy-Wasm).
//!
//! Stands where Arolla's filter stands (`prototype/arolla-filter`) and reads the
//! same things off a request: the attempt number from a header (1: a first
//! attempt, always let in; above 1: a retry, which is what is limited), and the
//! outcome from the response status. What differs is the rule. Arolla's is one
//! bucket for everybody, refilled by successes and by time. This one keeps a
//! quota per tenant, as a share of that tenant's first attempts over the last
//! second, and once a second a small policy sets every tenant's share from what
//! the filter counted in that second (`limiter.rs`). The policy is trained in
//! simulation (`msim.rl.limiter`) and arrives as weights in the plugin's
//! configuration; with none, the shares stay at the default level.
//!
//! Like their filter, the state belongs to one Wasm VM: every Envoy worker
//! thread has a limiter of its own, with its own quota. Envoy does not spread
//! connections evenly over its threads (one thread of a gateway was seen with
//! 95% of the traffic), and the policy reads the load off the counts, so the
//! threads of a proxy POOL their counts (`pool_threads`, on by default): each
//! leaves the counts of its last window in the proxy's shared data and decides
//! on the sum. The configured rates are then the requests per second the whole
//! proxy sees when all is well. The quota needs no such care: it is a share of
//! a thread's own first attempts.

pub mod limiter;

#[cfg(target_arch = "wasm32")]
mod filter {
    use crate::limiter::{decode, encode, Limiter, Seen, Settings};
    use proxy_wasm::traits::*;
    use proxy_wasm::types::*;
    use serde::Deserialize;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    proxy_wasm::main! {{
        proxy_wasm::set_log_level(LogLevel::Info);
        proxy_wasm::set_root_context(|_| -> Box<dyn RootContext> {
            Box::new(Root::default())
        });
    }}

    #[derive(Debug, Deserialize, Clone)]
    #[serde(default)]
    struct Config {
        /// Headers carrying the attempt number, as in their filter: the mesh's
        /// (`x-envoy-attempt-count`) and the client's (`X-Attempt-Number`).
        attempt_header: String,
        attempt_header_secondary: String,
        /// Header naming the tenant; their load generator sends the client
        /// profile's name in `X-Retry-Client-Type`.
        tenant_header: String,
        /// HTTP status returned when a retry is refused.
        reject_status: u32,
        /// Log every tenant's counts and level once a window (for a run that
        /// checks what one Envoy thread really sees).
        log_windows: bool,
        /// Decide on the counts of every thread of this proxy together.
        pool_threads: bool,
        limiter: Settings,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                attempt_header: "x-envoy-attempt-count".to_string(),
                attempt_header_secondary: "X-Attempt-Number".to_string(),
                tenant_header: "X-Retry-Client-Type".to_string(),
                reject_status: 429,
                log_windows: false,
                pool_threads: true,
                limiter: Settings::default(),
            }
        }
    }

    fn millis(now: SystemTime) -> u64 {
        now.duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
    }

    #[derive(Default, Clone, Copy)]
    struct Metrics {
        admitted: u32,
        rejected: u32,
        first_attempts: u32,
        level_sum: u32,
    }

    fn counter(name: &str) -> u32 {
        proxy_wasm::hostcalls::define_metric(MetricType::Counter, name).unwrap_or(0)
    }

    #[derive(Default)]
    struct Root {
        /// Shared with every request of this thread, never copied: the limiter's
        /// settings (the policy's weights among them) are moved out of it into
        /// the limiter when the plugin is configured.
        config: Rc<Config>,
        limiter: Option<Rc<RefCell<Limiter>>>,
        metrics: Metrics,
        /// Where this thread leaves its counts: the prefix of the plugin's keys
        /// in the proxy's shared data, and this thread's number among them.
        keys: String,
        slot: Option<u32>,
    }

    impl Context for Root {}

    /// A thread whose counts are older than this many windows is not counted:
    /// it is gone (a configuration that was replaced), or has not ticked yet.
    const FRESH_WINDOWS: u64 = 3;
    /// At most this many of the latest slots are read (replaced configurations
    /// leave theirs behind).
    const SLOTS_READ: u32 = 64;

    impl Root {
        /// A number no other thread of this plugin in this proxy has: the value
        /// of a shared counter, taken with compare-and-swap.
        fn claim_slot(&self) -> Option<u32> {
            let key = format!("{}n", self.keys);
            for _ in 0..64 {
                let (bytes, cas) = self.get_shared_data(&key);
                let taken = bytes
                    .and_then(|b| b.as_slice().try_into().ok().map(u32::from_le_bytes))
                    .unwrap_or(0);
                match self.set_shared_data(&key, Some(&(taken + 1).to_le_bytes()), cas) {
                    Ok(()) => return Some(taken),
                    Err(Status::CasMismatch) => continue,
                    Err(_) => return None,
                }
            }
            None
        }

        /// This thread's counts, left for the others, and the sum of every
        /// thread's latest. `None` when the shared data cannot be used.
        fn pooled(&self, own: &[Seen], now: u64, window_ms: u64) -> Option<Vec<Seen>> {
            let slot = self.slot?;
            self.set_shared_data(&format!("{}{}", self.keys, slot), Some(&encode(now, own)), None)
                .ok()?;
            let (bytes, _) = self.get_shared_data(&format!("{}n", self.keys));
            let slots = bytes
                .and_then(|b| b.as_slice().try_into().ok().map(u32::from_le_bytes))
                .unwrap_or(slot + 1);
            let mut total: Vec<Seen> = own.to_vec();
            for other in slots.saturating_sub(SLOTS_READ)..slots {
                if other == slot {
                    continue;
                }
                let (bytes, _) = self.get_shared_data(&format!("{}{}", self.keys, other));
                let Some((at, counts)) = bytes.and_then(|b| decode(&b, own.len())) else {
                    continue;
                };
                if now.saturating_sub(at) <= FRESH_WINDOWS * window_ms {
                    for (sum, seen) in total.iter_mut().zip(&counts) {
                        sum.add(seen);
                    }
                }
            }
            Some(total)
        }
    }

    impl RootContext for Root {
        fn on_configure(&mut self, _size: usize) -> bool {
            let mut config = Config::default();
            if let Some(bytes) = self.get_plugin_configuration() {
                match serde_json::from_slice::<Config>(&bytes) {
                    Ok(given) => config = given,
                    Err(error) => {
                        // Fail the configuration: a limiter running on defaults
                        // it was not given would be worse than none.
                        proxy_wasm::hostcalls::log(
                            LogLevel::Error,
                            &format!("learned-limiter: bad configuration: {}", error),
                        )
                        .ok();
                        return false;
                    }
                }
            }
            let settings = std::mem::take(&mut config.limiter);
            let window = settings.window_ms;
            let described = format!(
                "learned-limiter: {} tenant(s) {:?}, rates {:?}, policy {}",
                settings.tenants.len(),
                settings.tenants,
                settings.rates,
                if settings.policy.head.is_some() { "loaded" } else { "none" },
            );
            let pool = config.pool_threads;
            self.config = Rc::new(config);
            if pool {
                // The plugin's name keeps two plugins of one proxy apart (a
                // gateway has the namespace's and its own).
                let name = self
                    .get_property(vec!["plugin_name"])
                    .and_then(|b| String::from_utf8(b).ok())
                    .unwrap_or_default();
                self.keys = format!("learned-limiter/{}/", name);
                self.slot = self.claim_slot();
                if self.slot.is_none() {
                    proxy_wasm::hostcalls::log(
                        LogLevel::Warn,
                        "learned-limiter: no shared data, this thread decides on its own counts",
                    )
                    .ok();
                }
            }
            match Limiter::new(settings) {
                Ok(limiter) => self.limiter = Some(Rc::new(RefCell::new(limiter))),
                Err(error) => {
                    proxy_wasm::hostcalls::log(
                        LogLevel::Error,
                        &format!("learned-limiter: {}", error),
                    )
                    .ok();
                    return false;
                }
            }
            self.metrics = Metrics {
                admitted: counter("learned_retries_admitted_total"),
                rejected: counter("learned_retries_rejected_total"),
                first_attempts: counter("learned_first_attempts_total"),
                level_sum: proxy_wasm::hostcalls::define_metric(
                    MetricType::Gauge,
                    "learned_level_sum",
                )
                .unwrap_or(0),
            };
            self.set_tick_period(Duration::from_millis(window));
            proxy_wasm::hostcalls::log(LogLevel::Info, &described).ok();
            true
        }

        fn on_tick(&mut self) {
            if let Some(limiter) = &self.limiter {
                let now = millis(self.get_current_time());
                let mut limiter = limiter.borrow_mut();
                let own = limiter.take();
                let own_firsts: u64 = own.iter().map(|seen| seen.firsts).sum();
                let window_ms = limiter.settings.window_ms;
                let counts = self.pooled(&own, now, window_ms).unwrap_or(own);
                let levels = limiter.decide(&counts, now);
                let sum: usize = levels.iter().sum();
                let _ = proxy_wasm::hostcalls::record_metric(self.metrics.level_sum, sum as u64);
                if self.config.log_windows && limiter.last_counts.iter().any(|c| c.firsts > 0) {
                    let counts: Vec<String> = limiter
                        .last_counts
                        .iter()
                        .map(|c| {
                            format!(
                                "{}/{}/{}/{}",
                                c.firsts, c.goodput, c.retries_asked, c.retries_admitted
                            )
                        })
                        .collect();
                    // At the level a proxy shows by default: this line is asked
                    // for by configuration, for a run that is being looked at.
                    proxy_wasm::hostcalls::log(
                        LogLevel::Warn,
                        &format!(
                            "learned-limiter: t={} plugin={} own={} firsts/ok/asked/admitted {:?} levels {:?}",
                            now,
                            self.keys.trim_end_matches('/').rsplit(['/', '.']).next().unwrap_or(""),
                            own_firsts,
                            counts,
                            levels
                        ),
                    )
                    .ok();
                }
            }
        }

        fn create_http_context(&self, _id: u32) -> Option<Box<dyn HttpContext>> {
            Some(Box::new(Http {
                config: self.config.clone(),
                limiter: self.limiter.clone(),
                metrics: self.metrics,
                tenant: 0,
                is_retry: false,
                admitted: false,
            }))
        }

        fn get_type(&self) -> Option<ContextType> {
            Some(ContextType::HttpContext)
        }
    }

    struct Http {
        config: Rc<Config>,
        limiter: Option<Rc<RefCell<Limiter>>>,
        metrics: Metrics,
        tenant: usize,
        is_retry: bool,
        /// Whether this attempt was let in: only those are observed on the way back.
        admitted: bool,
    }

    impl Context for Http {}

    impl HttpContext for Http {
        fn on_http_request_headers(&mut self, _n: usize, _eos: bool) -> Action {
            let Some(limiter) = self.limiter.clone() else {
                return Action::Continue;
            };
            let number = |name: &str| -> u32 {
                self.get_http_request_header(name).and_then(|v| v.parse().ok()).unwrap_or(1)
            };
            let attempt = number(&self.config.attempt_header)
                .max(number(&self.config.attempt_header_secondary));
            let name = self.get_http_request_header(&self.config.tenant_header);
            let now = millis(self.get_current_time());
            let mut limiter = limiter.borrow_mut();
            self.tenant = limiter.tenant(name.as_deref());
            self.is_retry = attempt > 1;
            self.admitted = limiter.admit(self.tenant, self.is_retry, now);
            drop(limiter);
            if !self.is_retry {
                let _ = proxy_wasm::hostcalls::increment_metric(self.metrics.first_attempts, 1);
                return Action::Continue;
            }
            if self.admitted {
                let _ = proxy_wasm::hostcalls::increment_metric(self.metrics.admitted, 1);
                Action::Continue
            } else {
                let _ = proxy_wasm::hostcalls::increment_metric(self.metrics.rejected, 1);
                self.send_http_response(
                    self.config.reject_status,
                    vec![("x-learned-rejected", "1")],
                    Some(b"retry refused by the learned limiter\n"),
                );
                Action::Pause
            }
        }

        fn on_http_response_headers(&mut self, _n: usize, _eos: bool) -> Action {
            if !self.admitted {
                return Action::Continue; // our own refusal on its way out
            }
            self.admitted = false; // once per attempt
            let status: u32 =
                self.get_http_response_header(":status").and_then(|v| v.parse().ok()).unwrap_or(0);
            // As their filter counts it: answered, and not with a server error
            // or a refusal from a limiter further in.
            let success = status > 0 && status < 500 && status != self.config.reject_status;
            if let Some(limiter) = &self.limiter {
                limiter.borrow_mut().ended(self.tenant, self.is_retry, success);
            }
            Action::Continue
        }

        /// The stream is over. An attempt that was let in and never got response
        /// headers (its caller gave up, or the connection went away) has ended
        /// too, as a failure: in the simulator the gate is told of every outcome,
        /// a deadline included.
        fn on_log(&mut self) {
            if self.admitted {
                self.admitted = false;
                if let Some(limiter) = &self.limiter {
                    limiter.borrow_mut().ended(self.tenant, self.is_retry, false);
                }
            }
        }
    }
}
