//! The limiter itself, with no Envoy in it: a retry quota per tenant, and the
//! policy that sets the quotas from what the limiter counted.
//!
//! This is `msim.admission.TenantRetryQuota` and `msim.rl.limiter` in Rust:
//!
//! * a retry of a tenant is let in while its retries let in during the last
//!   window are fewer than `share x n`, `n` being ITS first attempts let in
//!   during that window;
//! * once per window the counts of that window are turned into the
//!   observation the policy was trained on (per tenant: attempts that ended,
//!   those that succeeded, retries that asked, retries let in, success share of
//!   the retries; then the same averaged over `slow_s`; then each tenant's
//!   current level), the policy picks a level per tenant, and the levels hold
//!   for the next window.
//!
//! Times are milliseconds on whatever clock the caller has.

use serde::Deserialize;
use std::collections::VecDeque;

/// What a count is measured against, in requests of the tenant when all is
/// well (`msim.rl.limiter.ATTEMPTS`, `GOODPUT`).
const ATTEMPTS: f64 = 4.0;
const GOODPUT: f64 = 2.0;

/// One dense layer: `w[out][in]`, `b[out]`.
#[derive(Debug, Deserialize, Clone)]
pub struct Layer {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
}

impl Layer {
    fn forward(&self, input: &[f64], tanh: bool) -> Vec<f64> {
        self.w
            .iter()
            .zip(&self.b)
            .map(|(row, bias)| {
                let sum: f64 = row.iter().zip(input).map(|(w, x)| w * x).sum::<f64>() + bias;
                if tanh {
                    sum.tanh()
                } else {
                    sum
                }
            })
            .collect()
    }
}

/// The policy: tanh layers, then one linear head whose outputs are the logits
/// of every tenant's levels, one tenant after the other. Greedy.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct Policy {
    pub hidden: Vec<Layer>,
    pub head: Option<Layer>,
}

impl Policy {
    /// How many tenants the policy decides for at once, each among `levels`.
    pub fn tenants(&self, levels: usize) -> Option<usize> {
        let outputs = self.head.as_ref()?.b.len();
        (levels > 0 && outputs % levels == 0).then_some(outputs / levels)
    }

    /// The level of each of `tenants` tenants, `levels` to choose from.
    pub fn act(&self, observation: &[f64], tenants: usize, levels: usize) -> Option<Vec<usize>> {
        let head = self.head.as_ref()?;
        let mut x = observation.to_vec();
        for layer in &self.hidden {
            x = layer.forward(&x, true);
        }
        let logits = head.forward(&x, false);
        if logits.len() != tenants * levels {
            return None;
        }
        Some(
            logits
                .chunks(levels)
                .map(|chunk| {
                    let mut best = 0;
                    for (i, value) in chunk.iter().enumerate() {
                        if *value > chunk[best] {
                            best = i; // the first of equals, as argmax does
                        }
                    }
                    best
                })
                .collect(),
        )
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct Settings {
    /// The tenants, in the order the policy was trained with. A request whose
    /// tenant is none of them (a call inside the mesh carries no tenant) is
    /// kept apart: it is in no tenant's counts, the policy never sees it, and
    /// its retries get the fixed `other_share`.
    pub tenants: Vec<String>,
    /// Requests per second each tenant sends when all is well, to whatever the
    /// counts given to `decide` are the counts of (one limiter, or the limiters
    /// of every thread of a proxy together): the scale of what is measured.
    pub rates: Vec<f64>,
    /// The ladder of shares a level stands for.
    pub shares: Vec<f64>,
    /// The level every tenant has until the policy has spoken.
    pub default_level: usize,
    /// The window of the quota and the step of the policy.
    pub window_ms: u64,
    /// Time constant of the averaged copy of the observation; 0: none.
    pub slow_s: f64,
    /// The share of the retries of requests that are no tenant's.
    pub other_share: f64,
    /// One level for every tenant, decided from their counts added up, instead
    /// of a level per tenant (`decide`).
    pub one_level: bool,
    pub policy: Policy,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            tenants: vec!["default".to_string()],
            rates: vec![100.0],
            shares: vec![0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 3.0],
            default_level: 0,
            window_ms: 1000,
            slow_s: 10.0,
            other_share: 0.1,
            one_level: false,
            policy: Policy::default(),
        }
    }
}

/// What a limiter counted for one tenant in one window.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Seen {
    pub firsts: u64,
    pub ended: u64,
    pub goodput: u64,
    pub retries_asked: u64,
    pub retries_admitted: u64,
    pub retries_ended: u64,
    pub retries_ok: u64,
}

impl Seen {
    const FIELDS: usize = 7;

    fn fields(&self) -> [u64; Self::FIELDS] {
        [
            self.firsts,
            self.ended,
            self.goodput,
            self.retries_asked,
            self.retries_admitted,
            self.retries_ended,
            self.retries_ok,
        ]
    }

    pub fn add(&mut self, other: &Seen) {
        self.firsts += other.firsts;
        self.ended += other.ended;
        self.goodput += other.goodput;
        self.retries_asked += other.retries_asked;
        self.retries_admitted += other.retries_admitted;
        self.retries_ended += other.retries_ended;
        self.retries_ok += other.retries_ok;
    }
}

/// A window's counts as bytes, with the time they were taken: what one Envoy
/// thread leaves for the others of its proxy to add to their own.
pub fn encode(now: u64, counts: &[Seen]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 * (1 + Seen::FIELDS * counts.len()));
    bytes.extend_from_slice(&now.to_le_bytes());
    for seen in counts {
        for value in seen.fields() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// The time and the counts of `encode`, or `None` when the bytes are not the
/// counts of `tenants` tenants.
pub fn decode(bytes: &[u8], tenants: usize) -> Option<(u64, Vec<Seen>)> {
    if bytes.len() != 8 * (1 + Seen::FIELDS * tenants) {
        return None;
    }
    let mut words = bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap()));
    let at = words.next()?;
    let mut counts = Vec::with_capacity(tenants);
    for _ in 0..tenants {
        let mut next = || words.next().unwrap_or(0);
        counts.push(Seen {
            firsts: next(),
            ended: next(),
            goodput: next(),
            retries_asked: next(),
            retries_admitted: next(),
            retries_ended: next(),
            retries_ok: next(),
        });
    }
    Some((at, counts))
}

#[derive(Debug, Default)]
struct Tenant {
    /// When each first attempt, and each retry, of the last window was let in.
    firsts: VecDeque<u64>,
    retries: VecDeque<u64>,
    seen: Seen,
    level: usize,
}

/// What one window held for one tenant, for a log line.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Counts {
    pub firsts: u64,
    pub goodput: u64,
    pub retries_asked: u64,
    pub retries_admitted: u64,
}

pub struct Limiter {
    pub settings: Settings,
    /// One per tenant, then one for the requests that are nobody's.
    tenants: Vec<Tenant>,
    slow: Option<Vec<f64>>,
    last_step: Option<u64>,
    /// The observation the policy was last shown, for inspection.
    pub last_observation: Vec<f64>,
    /// The counts of the window that just ended, per tenant.
    pub last_counts: Vec<Counts>,
}

impl Limiter {
    pub fn new(settings: Settings) -> Result<Self, String> {
        let n = settings.tenants.len();
        if n == 0 || settings.rates.len() != n {
            return Err(format!("{} tenants need as many rates, got {}", n, settings.rates.len()));
        }
        if settings.shares.is_empty() || settings.default_level >= settings.shares.len() {
            return Err("the default level is not on the ladder of shares".to_string());
        }
        if settings.rates.iter().any(|rate| !(*rate > 0.0)) || settings.window_ms == 0 {
            return Err("rates and the window must be > 0".to_string());
        }
        if !(settings.other_share >= 0.0) {
            return Err("the share of the others must be >= 0".to_string());
        }
        let level = settings.default_level;
        let tenants = (0..=n).map(|_| Tenant { level, ..Tenant::default() }).collect();
        Ok(Self {
            settings,
            tenants,
            slow: None,
            last_step: None,
            last_observation: Vec::new(),
            last_counts: Vec::new(),
        })
    }

    fn known(&self) -> usize {
        self.settings.tenants.len()
    }

    /// Which tenant a name is: its place in the list, or the place after the
    /// last, which is where the requests that are nobody's are kept.
    pub fn tenant(&self, name: Option<&str>) -> usize {
        name.and_then(|name| self.settings.tenants.iter().position(|t| t == name))
            .unwrap_or(self.known())
    }

    /// The level of each tenant of the list.
    pub fn levels(&self) -> Vec<usize> {
        self.tenants[..self.known()].iter().map(|t| t.level).collect()
    }

    pub fn share(&self, tenant: usize) -> f64 {
        if tenant >= self.known() {
            self.settings.other_share
        } else {
            self.settings.shares[self.tenants[tenant].level]
        }
    }

    fn trim(&mut self, tenant: usize, now: u64) {
        let window = self.settings.window_ms;
        let cutoff = now.saturating_sub(window);
        let state = &mut self.tenants[tenant];
        for times in [&mut state.firsts, &mut state.retries] {
            while times.front().is_some_and(|t| *t < cutoff) {
                times.pop_front();
            }
        }
    }

    /// An attempt arrives. `true`: let it in.
    pub fn admit(&mut self, tenant: usize, is_retry: bool, now: u64) -> bool {
        self.trim(tenant, now);
        let share = self.share(tenant);
        let state = &mut self.tenants[tenant];
        if !is_retry {
            state.firsts.push_back(now);
            state.seen.firsts += 1;
            return true;
        }
        state.seen.retries_asked += 1;
        // retries < share x firsts, the count against the product as it is.
        if (state.retries.len() as f64) < share * state.firsts.len() as f64 {
            state.retries.push_back(now);
            state.seen.retries_admitted += 1;
            true
        } else {
            false
        }
    }

    /// An attempt that was let in has ended.
    pub fn ended(&mut self, tenant: usize, is_retry: bool, success: bool) {
        let seen = &mut self.tenants[tenant].seen;
        seen.ended += 1;
        seen.goodput += success as u64;
        if is_retry {
            seen.retries_ended += 1;
            seen.retries_ok += success as u64;
        }
    }

    /// The end of a window, alone: this limiter's counts become the
    /// observation, the policy sets the levels of the next window.
    pub fn step(&mut self, now: u64) -> Vec<usize> {
        let own = self.take();
        self.decide(&own, now)
    }

    /// What this limiter counted for each tenant since the last call; the
    /// counts start again from nothing.
    pub fn take(&mut self) -> Vec<Seen> {
        let known = self.known();
        self.tenants[known].seen = Seen::default(); // nobody's: counted, never shown
        self.tenants[..known].iter_mut().map(|t| std::mem::take(&mut t.seen)).collect()
    }

    /// The end of a window: `counts` (this limiter's, or the sum over every
    /// limiter the configured rates are the rates of) become the observation,
    /// the policy sets the levels of the next window. Returns the levels.
    pub fn decide(&mut self, counts: &[Seen], now: u64) -> Vec<usize> {
        let window_s = match self.last_step {
            Some(last) if now > last => (now - last) as f64 / 1000.0,
            _ => self.settings.window_ms as f64 / 1000.0,
        };
        self.last_step = Some(now);
        let known = self.known();
        self.last_counts = counts
            .iter()
            .map(|seen| Counts {
                firsts: seen.firsts,
                goodput: seen.goodput,
                retries_asked: seen.retries_asked,
                retries_admitted: seen.retries_admitted,
            })
            .collect();
        // One level for everybody: the tenants' counts and rates added up, and
        // the policy shown them as one tenant's. Each tenant's quota is still
        // a share of ITS first attempts, so with one share for all of them the
        // retries let in are in proportion to the requests each one sends.
        let together = self.settings.one_level && known > 1;
        let (counts, rates): (Vec<Seen>, Vec<f64>) = if together {
            let mut sum = Seen::default();
            for seen in counts {
                sum.add(seen);
            }
            (vec![sum], vec![self.settings.rates.iter().sum()])
        } else {
            (counts.to_vec(), self.settings.rates.clone())
        };
        let mut measured = Vec::with_capacity(5 * counts.len());
        for (seen, rate) in counts.iter().zip(&rates) {
            let per_request = 1.0 / (rate * window_s);
            measured.push(seen.ended as f64 * per_request / ATTEMPTS);
            measured.push(seen.goodput as f64 * per_request / GOODPUT);
            measured.push(seen.retries_asked as f64 * per_request / ATTEMPTS);
            measured.push(seen.retries_admitted as f64 * per_request / ATTEMPTS);
            measured.push(if seen.retries_ended > 0 {
                seen.retries_ok as f64 / seen.retries_ended as f64
            } else {
                0.0
            });
        }
        for value in measured.iter_mut() {
            *value = value.clamp(0.0, 1.0);
        }
        let mut observation = measured.clone();
        if self.settings.slow_s > 0.0 {
            let weight = 1.0 - (-window_s / self.settings.slow_s).exp();
            let slow = match self.slow.take() {
                None => measured.clone(),
                Some(old) => old.iter().zip(&measured).map(|(s, m)| s + weight * (m - s)).collect(),
            };
            observation.extend(slow.iter().map(|v| (*v as f32) as f64));
            self.slow = Some(slow);
        }
        let top = (self.settings.shares.len() - 1).max(1) as f64;
        let shown = if together { 1 } else { known };
        observation.extend(self.tenants[..shown].iter().map(|t| t.level as f64 / top));
        // The policy was trained on 32-bit observations.
        for value in observation.iter_mut() {
            *value = ((*value as f32) as f64).clamp(0.0, 1.0);
        }
        let levels = self.settings.shares.len();
        if together {
            if let Some(chosen) = self.settings.policy.act(&observation, 1, levels) {
                for tenant in self.tenants[..known].iter_mut() {
                    tenant.level = chosen[0];
                }
            }
        } else if self.settings.policy.tenants(levels) == Some(1) && known > 1 {
            // A policy trained for ONE tenant, in front of several: it is asked
            // once per tenant, each time shown that tenant's entries alone
            // (the window's, the averaged ones, its level), against that
            // tenant's own rate.
            let slow = if self.settings.slow_s > 0.0 { 5 * known } else { 0 };
            for index in 0..known {
                let mut one = observation[5 * index..5 * index + 5].to_vec();
                if slow > 0 {
                    one.extend_from_slice(&observation[5 * known + 5 * index..5 * known + 5 * index + 5]);
                }
                one.push(observation[5 * known + slow + index]);
                if let Some(chosen) = self.settings.policy.act(&one, 1, levels) {
                    self.tenants[index].level = chosen[0];
                }
            }
        } else if let Some(chosen) = self.settings.policy.act(&observation, known, levels) {
            for (tenant, level) in self.tenants[..known].iter_mut().zip(chosen) {
                tenant.level = level;
            }
        }
        self.last_observation = observation;
        self.levels()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limiter(shares: Vec<f64>, default_level: usize) -> Limiter {
        Limiter::new(Settings {
            tenants: vec!["a".into(), "b".into()],
            rates: vec![40.0, 40.0],
            shares,
            default_level,
            ..Settings::default()
        })
        .unwrap()
    }

    #[test]
    fn retries_are_a_share_of_the_first_attempts_of_the_window() {
        let mut l = limiter(vec![0.1], 0);
        assert!(!l.admit(0, true, 0)); // nothing has arrived: nothing is owed
        for i in 0..30 {
            assert!(l.admit(0, false, i));
        }
        let granted = (0..10).filter(|_| l.admit(0, true, 40)).count();
        assert_eq!(granted, 3);
        for i in 0..5 {
            l.admit(0, false, 40 + i);
        }
        // 35 first attempts are 3.5 retries: a fourth is let in while 3 < 3.5.
        assert_eq!((0..10).filter(|_| l.admit(0, true, 50)).count(), 1);
    }

    #[test]
    fn what_was_let_in_leaves_the_window_a_second_later() {
        let mut l = limiter(vec![0.5], 0);
        l.admit(0, false, 0);
        l.admit(0, false, 1);
        assert!(l.admit(0, true, 2));
        assert!(!l.admit(0, true, 3));
        for i in 0..4 {
            l.admit(0, false, 1001 + i);
        }
        assert!(l.admit(0, true, 1005)); // the first attempts of 0 and 1 ms and the retry of 2 ms are gone
    }

    #[test]
    fn each_tenant_has_its_own_counts() {
        let mut l = limiter(vec![0.1], 0);
        assert_eq!(l.tenant(Some("b")), 1);
        for i in 0..10 {
            l.admit(0, false, i);
        }
        assert!(!l.admit(1, true, 20)); // b has sent nothing
        assert!(l.admit(0, true, 20));
    }

    #[test]
    fn a_request_that_is_nobodys_is_kept_apart_on_a_fixed_share() {
        let mut l = limiter(vec![0.0], 0); // the tenants get no retry at all
        let other = l.tenant(None);
        assert_eq!((other, l.tenant(Some("nobody"))), (2, 2));
        for i in 0..20 {
            l.admit(other, false, i);
            l.ended(other, false, false);
        }
        // 10% of 20 first attempts, whatever the tenants' level is.
        assert_eq!((0..10).filter(|_| l.admit(other, true, 30)).count(), 2);
        // None of it is in what the policy is shown.
        l.step(1000);
        assert_eq!(l.levels().len(), 2);
        assert!(l.last_observation[..20].iter().all(|v| *v == 0.0));
        assert_eq!(l.last_counts, vec![Counts::default(), Counts::default()]);
        assert!(!l.admit(0, true, 1030)); // and a tenant's retry is still refused
    }

    #[test]
    fn a_window_becomes_the_observation_of_the_environment() {
        let mut l = limiter(vec![0.0, 0.5, 3.0], 1);
        for i in 0..40 {
            l.admit(0, false, i * 10);
            l.ended(0, false, i % 2 == 0); // half of them succeed
        }
        for i in 0..10 {
            if l.admit(0, true, 400 + i) {
                l.ended(0, true, true);
            }
        }
        let levels = l.step(1000);
        assert_eq!(levels, vec![1, 1]); // no policy: the levels stay
        let o = &l.last_observation;
        // tenant a: 50 attempts ended (40 + 10 retries) of 40 expected, over 4;
        // 30 succeeded, over 2; 10 retries asked and let in, over 4; all succeeded.
        let near = |a: f64, b: f64| (a - b).abs() < 1e-6;
        assert!(near(o[0], 50.0 / 40.0 / 4.0) && near(o[1], 30.0 / 40.0 / 2.0));
        assert!(near(o[2], 10.0 / 40.0 / 4.0) && near(o[3], 10.0 / 40.0 / 4.0) && near(o[4], 1.0));
        assert!(o[5..10].iter().all(|v| *v == 0.0)); // tenant b
        assert_eq!(o.len(), 10 + 10 + 2);
        assert!(near(o[10], o[0])); // the average starts at the first window
        assert!(near(o[20], 0.5) && near(o[21], 0.5)); // level 1 of 0..2
        // A second, empty window: the slow copy decays towards it.
        l.step(2000);
        let weight = 1.0 - (-1.0f64 / 10.0).exp();
        assert!(near(l.last_observation[0], 0.0));
        assert!(near(l.last_observation[10], (50.0 / 160.0) * (1.0 - weight)));
    }

    #[test]
    fn the_policy_picks_a_level_per_tenant() {
        // One hidden unit that copies tenant a's goodput; a head that prefers
        // level 2 for tenant a when the goodput is high and level 0 otherwise,
        // and always level 1 for tenant b.
        let mut hidden_w = vec![0.0; 22];
        hidden_w[1] = 4.0;
        let policy = Policy {
            hidden: vec![Layer { w: vec![hidden_w], b: vec![-1.0] }],
            head: Some(Layer {
                w: vec![vec![-1.0], vec![0.0], vec![1.0], vec![0.0], vec![0.0], vec![0.0]],
                b: vec![0.0, -5.0, 0.0, 0.0, 1.0, 0.0],
            }),
        };
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into(), "b".into()],
            rates: vec![40.0, 40.0],
            shares: vec![0.0, 0.5, 3.0],
            default_level: 0,
            policy,
            ..Settings::default()
        })
        .unwrap();
        for i in 0..40 {
            l.admit(0, false, i);
            l.ended(0, false, true);
        }
        assert_eq!(l.step(1000), vec![2, 1]); // goodput 0.5: tanh(1) > 0
        assert_eq!(l.step(2000), vec![0, 1]); // nothing arrived: tanh(-1) < 0
        assert_eq!(l.share(0), 0.0);
    }

    #[test]
    fn the_threads_of_a_proxy_decide_on_their_counts_together() {
        // Two limiters, as two Envoy threads have them: one gets nearly all of
        // the traffic. Each decides on the SUM, and the rates are the proxy's.
        let settings = Settings {
            tenants: vec!["a".into()],
            rates: vec![100.0],
            shares: vec![0.0, 0.5, 3.0],
            default_level: 1,
            ..Settings::default()
        };
        let (mut busy, mut idle) =
            (Limiter::new(settings.clone()).unwrap(), Limiter::new(settings).unwrap());
        for i in 0..95 {
            busy.admit(0, false, i);
            busy.ended(0, false, true);
        }
        for i in 0..5 {
            idle.admit(0, false, i);
            idle.ended(0, false, true);
        }
        let (left, right) = (busy.take(), idle.take());
        // As bytes and back: what one thread leaves for the other.
        let (at, read) = decode(&encode(1000, &left), 1).unwrap();
        assert_eq!((at, &read), (1000, &left));
        assert!(decode(&encode(1000, &left), 2).is_none());
        let mut total = read;
        total[0].add(&right[0]);
        busy.decide(&total, 1000);
        idle.decide(&total, 1000);
        // 100 attempts ended of 100 expected, over 4: both see the proxy's load.
        assert_eq!(busy.last_observation, idle.last_observation);
        assert!((busy.last_observation[0] - 0.25).abs() < 1e-6);
        // Their own counts are gone, and the quota is still each one's own.
        assert_eq!(busy.take(), vec![Seen::default()]);
        // (a retry is let in while the count is below the share of the first attempts)
        assert_eq!((0..100).filter(|_| busy.admit(0, true, 500)).count(), 48); // 47.5 of 95
        assert_eq!((0..100).filter(|_| idle.admit(0, true, 500)).count(), 3); // 2.5 of 5
    }

    #[test]
    fn a_policy_for_one_tenant_is_asked_once_per_tenant() {
        // Entries of one tenant: 5 of the window, 5 averaged, its level. The
        // policy looks at the window's goodput (entry 1): level 2 when it is
        // above a quarter of the scale, level 0 below.
        let mut hidden_w = vec![0.0; 11];
        hidden_w[1] = 8.0;
        let policy = Policy {
            hidden: vec![Layer { w: vec![hidden_w], b: vec![-2.0] }],
            head: Some(Layer { w: vec![vec![-1.0], vec![0.0], vec![1.0]], b: vec![0.0, -5.0, 0.0] }),
        };
        assert_eq!(policy.tenants(3), Some(1));
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into(), "b".into(), "c".into()],
            rates: vec![40.0, 40.0, 400.0],
            shares: vec![0.0, 0.5, 3.0],
            default_level: 1,
            policy,
            ..Settings::default()
        })
        .unwrap();
        // a: 40 of its 40 succeed; b: nothing; c: 40 of its 400 expected.
        for i in 0..40 {
            l.admit(0, false, i);
            l.ended(0, false, true);
            l.admit(2, false, i);
            l.ended(2, false, true);
        }
        assert_eq!(l.step(1000), vec![2, 0, 0]); // goodput 0.5, 0 and 0.05 of the scale
        assert_eq!(l.last_observation.len(), 3 * 11);
    }

    #[test]
    fn one_level_for_every_tenant_from_their_counts_together() {
        // The policy of the test above: level 2 when the goodput it is shown is
        // above a quarter of the scale, level 0 below.
        let mut hidden_w = vec![0.0; 11];
        hidden_w[1] = 8.0;
        let policy = Policy {
            hidden: vec![Layer { w: vec![hidden_w], b: vec![-2.0] }],
            head: Some(Layer { w: vec![vec![-1.0], vec![0.0], vec![1.0]], b: vec![0.0, -5.0, 0.0] }),
        };
        let mut l = Limiter::new(Settings {
            tenants: vec!["small".into(), "large".into()],
            rates: vec![10.0, 90.0],
            shares: vec![0.0, 0.5, 3.0],
            default_level: 0,
            one_level: true,
            policy,
            ..Settings::default()
        })
        .unwrap();
        // 10 and 90 requests succeed: 100 of the 100 expected together, half the scale.
        for i in 0..90 {
            if i < 10 {
                l.admit(0, false, i);
                l.ended(0, false, true);
            }
            l.admit(1, false, i);
            l.ended(1, false, true);
        }
        assert_eq!(l.step(1000), vec![2, 2]);
        assert_eq!(l.last_observation.len(), 11); // one tenant's worth of entries
        assert!((l.last_observation[1] - 0.5).abs() < 1e-6);
        // The same share for both, of each one's own first attempts: three
        // retries per first attempt, 30 for the small tenant and 270 for the large.
        for i in 0..90 {
            if i < 10 {
                l.admit(0, false, 1000 + i);
            }
            l.admit(1, false, 1000 + i);
        }
        assert_eq!((0..400).filter(|_| l.admit(0, true, 1100)).count(), 30);
        assert_eq!((0..400).filter(|_| l.admit(1, true, 1100)).count(), 270);
    }

    #[test]
    fn settings_that_make_no_sense_are_refused() {
        let bad = |s: Settings| Limiter::new(s).is_err();
        assert!(bad(Settings { rates: vec![1.0, 2.0], ..Settings::default() }));
        assert!(bad(Settings { default_level: 99, ..Settings::default() }));
        assert!(bad(Settings { rates: vec![0.0], ..Settings::default() }));
        assert!(bad(Settings { other_share: -1.0, ..Settings::default() }));
    }
}
