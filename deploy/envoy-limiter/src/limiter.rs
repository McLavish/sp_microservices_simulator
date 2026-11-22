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
/// `rl.limiter.DRIFT` and `AT_REST`: how fast the time in flight at rest may
/// rise from one window to the next, and the attempts a window needs to move it.
const DRIFT: f64 = 0.02;
const AT_REST: u64 = 10;
/// `rl.limiter.CAP_FLOOR`: the cap never refuses while fewer attempts are in flight.
const CAP_FLOOR: f64 = 8.0;

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

/// The policy written by hand (`rl.limiter.hand_rule`), in place of the
/// trained one: level 0 while fewer than `alive` of a tenant's attempts that
/// ended succeeded; a level down while the in-flight entry is above
/// `down_above`; held above `hold_above`, if given; otherwise the top level.
#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct HandRule {
    pub alive: f64,
    pub down_above: f64,
    pub hold_above: Option<f64>,
}

impl Default for HandRule {
    fn default() -> Self {
        Self { alive: 0.05, down_above: 0.5, hold_above: None }
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
    /// Retries per tenant and window let in whatever the level: a probe, so
    /// that a level of 0 still shows whether a retry succeeds again.
    pub floor: usize,
    /// While no retry asks, every tenant's level is the default: after a
    /// window in which none did, whatever the policy answers. The level held
    /// while nothing fails earns nothing either way, and a failure should not
    /// start from wherever it happened to be.
    pub rest: bool,
    /// Show the policy one entry more, after the tenants' own: how much longer
    /// an attempt is in flight than while nothing fails (`rl.limiter.IN_FLIGHT`).
    pub in_flight: bool,
    /// The levels a tenant's quota may rise from one window to the next
    /// (`QuotaGrid.step_up`); it may fall as far as the policy says. 0: any.
    pub step_up: usize,
    /// A value of the in-flight entry above which no quota rises, whatever
    /// the policy says (`QuotaGrid.hold_above`). Not set: no such rule.
    pub hold_above: Option<f64>,
    /// A retry is refused while this limiter has more attempts in flight than
    /// this many times what the attempts it let in during the last window
    /// have in flight at rest (`QuotaGrid.in_flight_cap`): the one rule that
    /// acts within a window. Not set: no such rule.
    pub in_flight_cap: Option<f64>,
    /// The retries a tenant is SHOWN to have asked are at most this many per
    /// first attempt of its that failed in the window
    /// (`QuotaGrid.asked_per_failure`): a caller that retries ten times does
    /// not read as a worse failure than one that retries three times.
    pub asked_per_failure: Option<f64>,
    /// A retry beyond this attempt of its request is refused whatever the
    /// level (`QuotaGrid.max_attempt`): with 4, a caller gets three retries
    /// per request however often it asks. Not set: any.
    pub max_attempt: Option<u32>,
    /// No level rises until retries have asked in two windows running
    /// (`QuotaGrid.confirm`): a failure begins inside a window, and that
    /// window looks like a partial failure whatever follows.
    pub confirm: bool,
    /// Every level stays at its default until there has been a window at rest
    /// (`QuotaGrid.await_rest`): before that there is no time in flight to
    /// compare with, the entry reads 0 and the cap refuses nothing.
    pub await_rest: bool,
    /// Show the policy ratios of the window's own counts (`rl.limiter.SCALE_FREE`)
    /// instead of counts against `rates`, which are then not needed: nothing
    /// about the deployment is configured.
    pub scale_free: bool,
    /// Decide by the hand-written rule instead of the policy. Not set: the policy.
    pub hand_rule: Option<HandRule>,
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
            floor: 0,
            rest: false,
            in_flight: false,
            step_up: 0,
            hold_above: None,
            in_flight_cap: None,
            asked_per_failure: None,
            max_attempt: None,
            confirm: false,
            await_rest: false,
            scale_free: false,
            hand_rule: None,
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
    /// Attempts in flight (let in, not yet ended) summed over the window's
    /// time: attempts x ms.
    pub flight: u64,
}

impl Seen {
    const FIELDS: usize = 8;

    fn fields(&self) -> [u64; Self::FIELDS] {
        [
            self.firsts,
            self.ended,
            self.goodput,
            self.retries_asked,
            self.retries_admitted,
            self.retries_ended,
            self.retries_ok,
            self.flight,
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
        self.flight += other.flight;
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
            flight: next(),
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
    /// Its attempts in flight, and the instant `seen.flight` is up to.
    in_flight: u64,
    flight_at: u64,
}

impl Tenant {
    fn advance(&mut self, now: u64) {
        if now > self.flight_at {
            self.seen.flight += self.in_flight * (now - self.flight_at);
            self.flight_at = now;
        }
    }
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
    /// Time in flight per attempt let in while nothing fails (ms): the lowest
    /// it was in a window in which no retry asked, drifting up by `DRIFT`.
    flight_at_rest: Option<f64>,
    /// The attempts this limiter let in during the window last taken, and the
    /// attempts in flight above which it refuses a retry (`in_flight_cap`).
    own_admitted: u64,
    /// Whether retries asked in the window before the one last counted (`confirm`).
    asked_before: bool,
    pub cap_limit: Option<f64>,
    /// Retries the cap refused since the filter started.
    pub cap_refused: u64,
    /// The observation the policy was last shown, for inspection.
    pub last_observation: Vec<f64>,
    /// The counts of the window that just ended, per tenant.
    pub last_counts: Vec<Counts>,
    /// The in-flight entry of the window that just ended, and the time in
    /// flight it was made from (ms per attempt let in).
    pub last_in_flight: Option<f64>,
    pub last_flight_ms: f64,
}

impl Limiter {
    pub fn new(mut settings: Settings) -> Result<Self, String> {
        let n = settings.tenants.len();
        if settings.scale_free && settings.rates.len() != n {
            // Not used: the entries have no unit.
            settings.rates = vec![1.0; n];
        }
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
            flight_at_rest: None,
            own_admitted: 0,
            asked_before: false,
            cap_limit: None,
            cap_refused: 0,
            last_observation: Vec::new(),
            last_counts: Vec::new(),
            last_in_flight: None,
            last_flight_ms: 0.0,
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

    /// An attempt arrives, with its number in its request. `true`: let it in.
    pub fn admit_attempt(&mut self, tenant: usize, attempt: u32, now: u64) -> bool {
        if let Some(most) = self.settings.max_attempt {
            if attempt > most.max(1) && tenant < self.known() {
                // It has asked all the same.
                self.tenants[tenant].seen.retries_asked += 1;
                return false;
            }
        }
        self.admit(tenant, attempt > 1, now)
    }

    /// An attempt arrives. `true`: let it in.
    pub fn admit(&mut self, tenant: usize, is_retry: bool, now: u64) -> bool {
        self.trim(tenant, now);
        let share = self.share(tenant);
        let known = self.known();
        let in_flight: u64 = self.tenants[..known].iter().map(|t| t.in_flight).sum();
        let state = &mut self.tenants[tenant];
        state.advance(now);
        if !is_retry {
            state.firsts.push_back(now);
            state.seen.firsts += 1;
            state.in_flight += 1;
            return true;
        }
        state.seen.retries_asked += 1;
        // Crowded: no retry gets in, whatever its tenant's share.
        if tenant < known {
            if let Some(limit) = self.cap_limit {
                if in_flight as f64 > limit {
                    self.cap_refused += 1;
                    return false;
                }
            }
        }
        let state = &mut self.tenants[tenant];
        // The probe, then: retries < share x firsts, the count against the
        // product as it is.
        let floor = if tenant < self.settings.tenants.len() { self.settings.floor } else { 0 };
        if state.retries.len() < floor
            || (state.retries.len() as f64) < share * state.firsts.len() as f64
        {
            state.retries.push_back(now);
            state.seen.retries_admitted += 1;
            state.in_flight += 1;
            true
        } else {
            false
        }
    }

    /// An attempt that was let in has ended.
    pub fn ended(&mut self, tenant: usize, is_retry: bool, success: bool, now: u64) {
        let state = &mut self.tenants[tenant];
        state.advance(now);
        state.in_flight = state.in_flight.saturating_sub(1);
        let seen = &mut state.seen;
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
        let own = self.take(now);
        self.decide(&own, now)
    }

    /// What this limiter counted for each tenant since the last call, up to
    /// `now`; the counts start again from nothing.
    pub fn take(&mut self, now: u64) -> Vec<Seen> {
        let known = self.known();
        for tenant in self.tenants.iter_mut() {
            tenant.advance(now);
        }
        self.own_admitted =
            self.tenants[..known].iter().map(|t| t.seen.firsts + t.seen.retries_admitted).sum();
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
        // A window in which nothing arrived and nothing ended is not a
        // situation the policy was trained on (a simulated service always has
        // traffic), and what it answers there sticks, because its own level is
        // part of what it is shown next: a proxy that starts before its
        // traffic was seen to open every quota and stay open. Such a window is
        // the start of an episode instead: the default level, no averages yet.
        if counts.iter().all(|seen| seen.firsts == 0 && seen.ended == 0 && seen.retries_asked == 0) {
            self.asked_before = false;
            let default = self.settings.default_level;
            for tenant in self.tenants[..known].iter_mut() {
                tenant.level = default;
            }
            self.slow = None;
            self.last_observation.clear();
            return self.levels();
        }
        let asked: u64 = counts.iter().map(|seen| seen.retries_asked).sum();
        let resting = self.settings.rest && asked == 0;
        // Attempts in flight, for the gate as a whole (`rl.limiter.IN_FLIGHT`):
        // their sum over the window's time per attempt let in during it is the
        // time an attempt is in flight (Little's law), and what is shown is how
        // much longer that is than while nothing fails.
        // Measured in every case (a log line shows it); shown to the policy
        // only when it was trained with it.
        let crowding = {
            let admitted: u64 = counts.iter().map(|seen| seen.firsts + seen.retries_admitted).sum();
            let flight: u64 = counts.iter().map(|seen| seen.flight).sum();
            let in_flight = flight as f64 / admitted.max(1) as f64;
            self.last_flight_ms = in_flight;
            if in_flight > 0.0 && asked == 0 && admitted >= AT_REST {
                self.flight_at_rest = Some(match self.flight_at_rest {
                    None => in_flight,
                    Some(at_rest) => in_flight.min(at_rest * (1.0 + DRIFT)),
                });
            }
            match self.flight_at_rest {
                Some(at_rest) if in_flight > 0.0 => (1.0 - at_rest / in_flight).max(0.0),
                _ => 0.0,
            }
        };
        self.last_in_flight = Some(crowding);
        if let (Some(times), Some(at_rest)) = (self.settings.in_flight_cap, self.flight_at_rest) {
            // What the attempts THIS limiter let in have in flight when each
            // takes the time it takes at rest (rate x time), times the cap.
            let in_flight_at_rest = self.own_admitted as f64 / (window_s * 1000.0) * at_rest;
            self.cap_limit = Some((times * in_flight_at_rest).max(CAP_FLOOR));
        }
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
            let firsts_ended = seen.ended.saturating_sub(seen.retries_ended);
            let firsts_ok = seen.goodput.saturating_sub(seen.retries_ok);
            let mut asked = seen.retries_asked as f64;
            if let Some(bound) = self.settings.asked_per_failure {
                asked = asked.min(bound * firsts_ended.saturating_sub(firsts_ok) as f64);
            }
            if self.settings.scale_free {
                // `rl.limiter.SCALE_FREE`, in the same five places.
                let share = |part: f64, whole: f64| if whole > 0.0 { part / whole } else { 0.0 };
                measured.push(share(seen.goodput as f64, seen.ended as f64));
                measured.push(share(firsts_ok as f64, firsts_ended as f64));
                measured.push(if seen.firsts > 0 {
                    asked / ((ATTEMPTS - 1.0) * seen.firsts as f64)
                } else if asked > 0.0 {
                    1.0
                } else {
                    0.0
                });
                measured.push(share(seen.retries_admitted as f64, seen.retries_asked as f64));
                measured.push(share(seen.retries_ok as f64, seen.retries_ended as f64));
                continue;
            }
            measured.push(seen.ended as f64 * per_request / ATTEMPTS);
            measured.push(seen.goodput as f64 * per_request / GOODPUT);
            measured.push(asked * per_request / ATTEMPTS);
            measured.push(seen.retries_admitted as f64 * per_request / ATTEMPTS);
            measured.push(if seen.retries_ended > 0 {
                seen.retries_ok as f64 / seen.retries_ended as f64
            } else {
                0.0
            });
        }
        if self.settings.in_flight {
            measured.push(crowding);
        }
        for value in measured.iter_mut() {
            *value = value.clamp(0.0, 1.0);
        }
        // Per tenant shown, then the entry of the gate as a whole if there is one.
        let extra = measured.len() - 5 * counts.len();
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
        let before = self.levels();
        if let Some(rule) = self.settings.hand_rule.clone() {
            // Per tenant shown (the pooled one with one level for all): its
            // entries `ended` and `goodput` of the window, and the entry of the
            // gate as a whole.
            let shown = if together { 1 } else { known };
            let crowded = crowding.clamp(0.0, 1.0);
            for index in 0..shown {
                let attempts = observation[5 * index] * ATTEMPTS;
                let succeeded = if self.settings.scale_free {
                    observation[5 * index] // the share that succeeded is an entry of its own
                } else if attempts > 0.0 {
                    observation[5 * index + 1] * GOODPUT / attempts
                } else {
                    0.0
                };
                let level = before[index];
                let chosen = if succeeded < rule.alive {
                    0
                } else if crowded > rule.down_above {
                    level.saturating_sub(1)
                } else if rule.hold_above.is_some_and(|above| crowded > above) {
                    level
                } else {
                    levels - 1
                };
                if together {
                    for tenant in self.tenants[..known].iter_mut() {
                        tenant.level = chosen;
                    }
                } else {
                    self.tenants[index].level = chosen;
                }
            }
        } else if together {
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
            let width = 5 * known + extra;
            let slow = if self.settings.slow_s > 0.0 { width } else { 0 };
            for index in 0..known {
                let mut one = observation[5 * index..5 * index + 5].to_vec();
                one.extend_from_slice(&observation[5 * known..width]);
                if slow > 0 {
                    one.extend_from_slice(&observation[width + 5 * index..width + 5 * index + 5]);
                    one.extend_from_slice(&observation[width + 5 * known..width + width]);
                }
                one.push(observation[width + slow + index]);
                if let Some(chosen) = self.settings.policy.act(&one, 1, levels) {
                    self.tenants[index].level = chosen[0];
                }
            }
        } else if let Some(chosen) = self.settings.policy.act(&observation, known, levels) {
            for (tenant, level) in self.tenants[..known].iter_mut().zip(chosen) {
                tenant.level = level;
            }
        }
        // No further up than `step_up` levels from where the tenant stood, and
        // not up at all while the service is crowded.
        let held = self.settings.hold_above.is_some_and(|above| crowding > above)
            || (self.settings.confirm && !self.asked_before);
        self.asked_before = asked > 0;
        if held || self.settings.step_up > 0 {
            let rise = if held { 0 } else { self.settings.step_up };
            for (tenant, was) in self.tenants[..known].iter_mut().zip(&before) {
                tenant.level = tenant.level.min(was + rise);
            }
        }
        let waiting = self.settings.await_rest && self.flight_at_rest.is_none();
        if resting || waiting {
            let default = self.settings.default_level;
            for tenant in self.tenants[..known].iter_mut() {
                tenant.level = default;
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
            l.ended(other, false, false, 0);
        }
        // 10% of 20 first attempts, whatever the tenants' level is.
        assert_eq!((0..10).filter(|_| l.admit(other, true, 30)).count(), 2);
        // None of it is in what the policy is shown: for the tenants the
        // window is an empty one.
        l.step(1000);
        assert_eq!(l.levels().len(), 2);
        assert!(l.last_observation.is_empty());
        assert_eq!(l.last_counts, vec![Counts::default(), Counts::default()]);
        assert!(!l.admit(0, true, 1030)); // and a tenant's retry is still refused
    }

    #[test]
    fn a_window_becomes_the_observation_of_the_environment() {
        let mut l = limiter(vec![0.0, 0.5, 3.0], 1);
        for i in 0..40 {
            l.admit(0, false, i * 10);
            l.ended(0, false, i % 2 == 0, 0); // half of them succeed
        }
        for i in 0..10 {
            if l.admit(0, true, 400 + i) {
                l.ended(0, true, true, 0);
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
        // A second, nearly empty window: the slow copy moves towards it.
        for i in 0..8 {
            l.admit(0, false, 1000 + i);
            l.ended(0, false, true, 0);
        }
        l.step(2000);
        let weight = 1.0 - (-1.0f64 / 10.0).exp();
        let (was, is) = (50.0 / 160.0, 8.0 / 160.0);
        assert!(near(l.last_observation[0], is));
        assert!(near(l.last_observation[10], was + weight * (is - was)));
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
            l.ended(0, false, true, 0);
        }
        assert_eq!(l.step(1000), vec![2, 1]); // goodput 0.5: tanh(1) > 0
        for i in 0..4 {
            l.admit(0, false, 1000 + i);
            l.ended(0, false, false, 0);
        }
        assert_eq!(l.step(2000), vec![0, 1]); // nothing succeeded: tanh(-1) < 0
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
            busy.ended(0, false, true, 0);
        }
        for i in 0..5 {
            idle.admit(0, false, i);
            idle.ended(0, false, true, 0);
        }
        let (left, right) = (busy.take(0), idle.take(0));
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
        assert_eq!(busy.take(0), vec![Seen::default()]);
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
            l.ended(0, false, true, 0);
            l.admit(2, false, i);
            l.ended(2, false, true, 0);
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
                l.ended(0, false, true, 0);
            }
            l.admit(1, false, i);
            l.ended(1, false, true, 0);
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
    fn a_floor_lets_one_retry_a_window_in_at_a_share_of_zero() {
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into(), "b".into()],
            rates: vec![40.0, 40.0],
            shares: vec![0.0],
            floor: 1,
            ..Settings::default()
        })
        .unwrap();
        for i in 0..20 {
            l.admit(0, false, i);
        }
        assert_eq!((0..10).filter(|_| l.admit(0, true, 30)).count(), 1);
        assert_eq!((0..10).filter(|_| l.admit(1, true, 30)).count(), 1); // each tenant its own
        assert_eq!((0..10).filter(|_| l.admit(0, true, 1100)).count(), 1); // and each window
        // Not for requests that are nobody's: theirs is the fixed share.
        let other = l.tenant(None);
        assert_eq!((0..10).filter(|_| l.admit(other, true, 30)).count(), 0);
    }

    #[test]
    fn a_window_with_no_traffic_is_the_start_of_an_episode() {
        // The policy of the tests above, turned round: level 2 when the goodput
        // it is shown is LOW. Shown nothing at all it would open the quota.
        let mut hidden_w = vec![0.0; 22];
        hidden_w[1] = -8.0;
        let policy = Policy {
            hidden: vec![Layer { w: vec![hidden_w], b: vec![2.0] }],
            head: Some(Layer {
                w: vec![vec![-1.0], vec![0.0], vec![1.0], vec![0.0], vec![0.0], vec![0.0]],
                b: vec![0.0, -5.0, 0.0, 1.0, 0.0, 0.0],
            }),
        };
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into(), "b".into()],
            rates: vec![40.0, 40.0],
            shares: vec![0.0, 0.5, 3.0],
            default_level: 1,
            policy,
            ..Settings::default()
        })
        .unwrap();
        assert_eq!(l.step(1000), vec![1, 1]); // nothing arrived: nothing is decided
        assert!(l.last_observation.is_empty());
        for i in 0..40 {
            l.admit(0, false, 1000 + i);
            l.ended(0, false, false, 0); // every one fails: low goodput
        }
        assert_eq!(l.step(2000), vec![2, 0]);
        let first = l.last_observation[10]; // the average starts at this window
        assert!((first - l.last_observation[0]).abs() < 1e-9);
        assert_eq!(l.step(3000), vec![1, 1]); // quiet again: back to the default
    }

    #[test]
    fn with_rest_the_levels_are_the_default_while_no_retry_asks() {
        // A policy that always answers level 2.
        let policy = Policy {
            hidden: vec![Layer { w: vec![vec![0.0; 11]], b: vec![0.0] }],
            head: Some(Layer { w: vec![vec![0.0], vec![0.0], vec![0.0]], b: vec![0.0, 0.0, 1.0] }),
        };
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![40.0],
            shares: vec![0.0, 0.5, 3.0],
            default_level: 0,
            rest: true,
            policy,
            ..Settings::default()
        })
        .unwrap();
        for i in 0..40 {
            l.admit(0, false, i);
            l.ended(0, false, true, 0);
        }
        assert_eq!(l.step(1000), vec![0]); // traffic, and no retry asked: at rest
        assert_eq!(l.last_observation.len(), 11); // the window is still what the averages see
        for i in 0..40 {
            l.admit(0, false, 1000 + i);
            l.ended(0, false, false, 0);
        }
        assert!(!l.admit(0, true, 1050)); // a retry asks, and is refused at level 0
        assert_eq!(l.step(2000), vec![2]); // now the policy's answer is in force
        assert!((l.last_observation[10] - 0.0).abs() < 1e-9); // and it was shown level 0
    }

    #[test]
    fn attempts_in_flight_are_shown_against_their_time_at_rest() {
        // A policy that always answers level 1.
        let policy = Policy {
            hidden: vec![Layer { w: vec![vec![0.0; 13]], b: vec![0.0] }],
            head: Some(Layer { w: vec![vec![0.0], vec![0.0]], b: vec![0.0, 1.0] }),
        };
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![40.0],
            shares: vec![0.0, 3.0],
            default_level: 0,
            in_flight: true,
            policy,
            ..Settings::default()
        })
        .unwrap();
        // At rest: forty attempts a second, each in flight for 10 ms.
        for second in 0..3u64 {
            for i in 0..40 {
                let at = second * 1000 + i * 25;
                l.admit(0, false, at);
                l.ended(0, false, true, at + 10);
            }
            l.step((second + 1) * 1000);
            assert_eq!(l.last_observation.len(), 13); // 5 + 1, the same averaged, the level
            assert!(l.last_observation[5].abs() < 1e-9, "{:?}", l.last_observation);
        }
        // Half of them fail and retry; every attempt is now in flight for 40 ms.
        // (time, a retry?, an end? and if so a success?), in the order of their times.
        let mut events: Vec<(u64, bool, Option<bool>)> = Vec::new();
        for i in 0..40u64 {
            let at = 3000 + i * 25;
            events.push((at, false, None));
            events.push((at + 40, false, Some(i % 2 == 0)));
            if i % 2 == 1 {
                events.push((at + 40, true, None));
                events.push((at + 80, true, Some(true)));
            }
        }
        events.sort_by_key(|event| event.0);
        for (at, is_retry, end) in events.into_iter().filter(|event| event.0 < 4000) {
            match end {
                None => assert!(l.admit(0, is_retry, at)),
                Some(success) => l.ended(0, is_retry, success, at),
            }
        }
        l.step(4000);
        // 40 ms against 10 at rest: 1 - 10/40. (The last attempts end after the
        // window does; what they add is in the next one.)
        assert!((l.last_observation[5] - 0.75).abs() < 0.02, "{:?}", l.last_observation);
        // A window with retries asking does not move the time at rest.
        assert!((l.flight_at_rest.unwrap() - 10.0).abs() < 1e-9);
        // The counts of the threads of a proxy are added, the time in flight with them.
        let mut sum = Seen { firsts: 30, flight: 300, ..Seen::default() };
        sum.add(&Seen { firsts: 10, flight: 500, ..Seen::default() });
        assert_eq!((sum.firsts, sum.flight), (40, 800));
        let bytes = encode(7, &[sum.clone()]);
        assert_eq!(decode(&bytes, 1), Some((7, vec![sum])));
    }

    #[test]
    fn with_step_up_a_quota_climbs_a_level_a_window() {
        // A policy that answers the top level while it is shown a level below
        // the top, and level 0 once it is shown the top (the level is the last
        // of the 11 entries: 5 counts, the same averaged, the level).
        let mut bottom = vec![0.0; 11];
        bottom[10] = 30.0;
        let policy = Policy {
            hidden: Vec::new(),
            head: Some(Layer {
                w: vec![bottom, vec![0.0; 11], vec![0.0; 11], vec![0.0; 11]],
                b: vec![-25.0, -100.0, -100.0, 0.0],
            }),
        };
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![40.0],
            shares: vec![0.0, 0.1, 0.2, 0.4],
            default_level: 0,
            step_up: 1,
            policy,
            ..Settings::default()
        })
        .unwrap();
        let mut climbed = Vec::new();
        for second in 0..5u64 {
            for i in 0..40 {
                l.admit(0, false, second * 1000 + i);
            }
            climbed.push(l.step((second + 1) * 1000)[0]);
        }
        // Up a level a window whatever it asks for; down at once.
        assert_eq!(climbed, vec![1, 2, 3, 0, 1]);
    }

    #[test]
    fn with_hold_above_no_quota_rises_while_attempts_pile_up() {
        // A policy that always answers the top level.
        let policy = Policy {
            hidden: Vec::new(),
            head: Some(Layer { w: vec![vec![0.0; 11], vec![0.0; 11]], b: vec![0.0, 1.0] }),
        };
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![40.0],
            shares: vec![0.0, 3.0],
            default_level: 0,
            rest: true,
            hold_above: Some(0.5),
            policy,
            ..Settings::default()
        })
        .unwrap();
        // At rest, 10 ms in flight. Then every attempt fails, retries ask, and
        // attempts take 100 ms: crowded, so the quota does not rise.
        for i in 0..40 {
            l.admit(0, false, i * 25);
            l.ended(0, false, true, i * 25 + 10);
        }
        assert_eq!(l.step(1000), vec![0]);
        for i in 0..9 {
            l.admit(0, false, 1000 + i * 100);
            l.ended(0, false, false, 1000 + i * 100 + 99);
            assert!(!l.admit(0, true, 1000 + i * 100 + 99));
        }
        l.admit(0, false, 1900);
        assert_eq!(l.step(2000), vec![0]);
        assert!(l.last_in_flight.unwrap() > 0.5);
        // Calm again while retries still ask: now it rises.
        l.ended(0, false, false, 2000);
        for i in 0..40 {
            l.admit(0, false, 2001 + i * 20);
            l.ended(0, false, false, 2001 + i * 20 + 10);
            l.admit(0, true, 2001 + i * 20 + 10);
        }
        assert_eq!(l.step(3000), vec![1]);
    }

    #[test]
    fn with_an_in_flight_cap_retries_are_refused_while_attempts_pile_up() {
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![1000.0],
            shares: vec![3.0],
            default_level: 0,
            in_flight_cap: Some(2.5),
            ..Settings::default()
        })
        .unwrap();
        // At rest: a thousand attempts a second, each in flight for 10 ms, so
        // ten in flight; the cap is 2.5 times that.
        for i in 0..1000 {
            l.admit(0, false, i);
            if i >= 10 {
                l.ended(0, false, true, i);
            }
        }
        assert!(l.cap_limit.is_none()); // nothing to compare with yet
        l.step(1000);
        assert!((l.cap_limit.unwrap() - 25.0).abs() < 0.5, "{:?}", l.cap_limit);
        // Ten in flight: a retry is let in. Thirty: it is not, at a share that
        // refuses nothing; and it has asked all the same.
        assert!(l.admit(0, true, 1000));
        for i in 0..20 {
            l.admit(0, false, 1001 + i);
        }
        assert!(!l.admit(0, true, 1030));
        assert_eq!(l.cap_refused, 1);
        // They end: there is room again.
        for _ in 0..20 {
            l.ended(0, false, false, 1040);
        }
        assert!(l.admit(0, true, 1041));
        let seen = &l.take(1050)[0];
        assert_eq!((seen.retries_asked, seen.retries_admitted), (3, 2));
        // A limiter with little traffic never refuses below the floor.
        let mut quiet = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![10.0],
            shares: vec![3.0],
            default_level: 0,
            in_flight_cap: Some(2.5),
            ..Settings::default()
        })
        .unwrap();
        for i in 0..10 {
            quiet.admit(0, false, i * 100);
            quiet.ended(0, false, true, i * 100 + 10);
        }
        quiet.step(1000);
        assert_eq!(quiet.cap_limit, Some(8.0));
    }

    #[test]
    fn a_caller_that_asks_more_often_is_not_shown_to_ask_more() {
        let run = |asked_per_failure: Option<f64>| {
            let mut l = Limiter::new(Settings {
                tenants: vec!["a".into()],
                rates: vec![40.0],
                shares: vec![0.0, 3.0],
                default_level: 0,
                asked_per_failure,
                ..Settings::default()
            })
            .unwrap();
            // Forty requests, half of them failing, each failure asking ten times.
            for i in 0..40 {
                l.admit(0, false, i * 25);
                l.ended(0, false, i % 2 == 0, i * 25 + 5);
                if i % 2 == 1 {
                    for k in 0..10 {
                        assert!(!l.admit(0, true, i * 25 + 5 + k));
                    }
                }
            }
            l.step(1000);
            l.last_observation[2]
        };
        // 200 asks on a scale of 4 x 40 as counted (off the scale); 3 x 20 as shown.
        assert!((run(None) - 1.0).abs() < 1e-9);
        assert!((run(Some(3.0)) - 60.0 / 160.0).abs() < 1e-6);
    }

    #[test]
    fn with_max_attempt_a_caller_gets_no_more_retries_per_request() {
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![40.0],
            shares: vec![3.0],
            default_level: 0,
            max_attempt: Some(4),
            ..Settings::default()
        })
        .unwrap();
        for i in 0..10 {
            assert!(l.admit_attempt(0, 1, i));
        }
        // The fourth attempt is the last a caller is let in with, at a share
        // that refuses nothing; the fifth has asked and is refused.
        assert!(l.admit_attempt(0, 2, 20));
        assert!(l.admit_attempt(0, 4, 21));
        assert!(!l.admit_attempt(0, 5, 22));
        assert!(!l.admit_attempt(0, 11, 23));
        let seen = &l.take(100)[0];
        assert_eq!((seen.retries_asked, seen.retries_admitted), (4, 2));
    }

    #[test]
    fn with_confirm_a_level_rises_once_retries_asked_in_two_windows_running() {
        // A policy that always answers the top level.
        let policy = Policy {
            hidden: Vec::new(),
            head: Some(Layer { w: vec![vec![0.0; 11], vec![0.0; 11]], b: vec![0.0, 1.0] }),
        };
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![40.0],
            shares: vec![0.0, 3.0],
            default_level: 0,
            rest: true,
            confirm: true,
            policy,
            ..Settings::default()
        })
        .unwrap();
        let mut levels = Vec::new();
        for second in 0..5u64 {
            for i in 0..40 {
                l.admit(0, false, second * 1000 + i);
                l.ended(0, false, second < 2, second * 1000 + i + 5);
                if second >= 2 {
                    l.admit(0, true, second * 1000 + i + 5);
                }
            }
            levels.push(l.step((second + 1) * 1000)[0]);
        }
        // Retries first ask in the third window: at rest before, held after it
        // (a first window of asking), open after the fourth.
        assert_eq!(levels, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn the_hand_written_rule_decides_in_place_of_the_policy() {
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![40.0],
            shares: vec![0.0, 0.1, 0.3, 0.6, 1.0, 3.0],
            default_level: 0,
            hand_rule: Some(HandRule::default()),
            ..Settings::default()
        })
        .unwrap();
        // All is well, 10 ms in flight: the top level (nothing bounds the rise here).
        for i in 0..40 {
            l.admit(0, false, i * 25);
            l.ended(0, false, true, i * 25 + 10);
        }
        assert_eq!(l.step(1000), vec![5]);
        // Half the attempts fail and attempts take four times as long: a level down.
        for i in 0..24 {
            l.admit(0, false, 1000 + i * 40);
            l.ended(0, false, i % 2 == 0, 1000 + i * 40 + 39);
        }
        assert_eq!(l.step(2000), vec![4]);
        assert!(l.last_in_flight.unwrap() > 0.5);
        // Nothing succeeds: closed, whatever the level was.
        for i in 0..40 {
            l.admit(0, false, 2000 + i * 25);
            l.ended(0, false, false, 2000 + i * 25 + 10);
        }
        assert_eq!(l.step(3000), vec![0]);
        // With a band, it holds where the plain rule would climb.
        let mut held = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![40.0],
            shares: vec![0.0, 0.1, 0.3, 0.6, 1.0, 3.0],
            default_level: 2,
            hand_rule: Some(HandRule { hold_above: Some(0.3), ..HandRule::default() }),
            ..Settings::default()
        })
        .unwrap();
        for i in 0..40 {
            held.admit(0, false, i * 25);
            held.ended(0, false, true, i * 25 + 10);
        }
        assert_eq!(held.step(1000), vec![5]);
        for i in 0..40 {
            held.admit(0, false, 1000 + i * 25);
            held.ended(0, false, true, 1000 + i * 25 + 16); // 1 - 10/16 = 0.375: in the band
        }
        assert_eq!(held.step(2000), vec![5]);
        assert!((held.last_in_flight.unwrap() - 0.375).abs() < 0.02);
    }

    #[test]
    fn with_await_rest_nothing_opens_before_a_window_at_rest() {
        // A policy that always answers the top level.
        let policy = Policy {
            hidden: Vec::new(),
            head: Some(Layer { w: vec![vec![0.0; 11], vec![0.0; 11]], b: vec![0.0, 1.0] }),
        };
        let mut l = Limiter::new(Settings {
            tenants: vec!["a".into()],
            rates: vec![40.0],
            shares: vec![0.0, 3.0],
            default_level: 0,
            await_rest: true,
            policy,
            ..Settings::default()
        })
        .unwrap();
        // Trouble from the first window on: retries ask in every one.
        for second in 0..3u64 {
            for i in 0..40 {
                l.admit(0, false, second * 1000 + i * 25);
                l.ended(0, false, false, second * 1000 + i * 25 + 10);
                l.admit(0, true, second * 1000 + i * 25 + 10);
            }
            assert_eq!(l.step((second + 1) * 1000), vec![0]);
        }
        // One window in which nothing asks: there is a time at rest now ...
        for i in 0..40 {
            l.admit(0, false, 3000 + i * 25);
            l.ended(0, false, true, 3000 + i * 25 + 10);
        }
        assert_eq!(l.step(4000), vec![1]);
        // ... and the policy's answer holds from then on.
        for i in 0..40 {
            l.admit(0, false, 4000 + i * 25);
            l.ended(0, false, false, 4000 + i * 25 + 10);
            l.admit(0, true, 4000 + i * 25 + 10);
        }
        assert_eq!(l.step(5000), vec![1]);
    }

    #[test]
    fn scale_free_entries_are_ratios_and_need_no_rate() {
        let run = |rates: Vec<f64>| {
            let mut l = Limiter::new(Settings {
                tenants: vec!["a".into()],
                rates,
                shares: vec![3.0],
                default_level: 0,
                scale_free: true,
                ..Settings::default()
            })
            .unwrap();
            // Forty first attempts, half failing; thirty retries ask, all are
            // let in (a share of 3), twenty have ended, five of them well.
            for i in 0..40 {
                l.admit(0, false, i);
                l.ended(0, false, i % 2 == 0, i + 1);
            }
            for i in 0..30 {
                assert!(l.admit(0, true, 100 + i));
                if i < 20 {
                    l.ended(0, true, i < 5, 101 + i);
                }
            }
            l.step(1000);
            l.last_observation[..5].to_vec()
        };
        let shown = run(Vec::new()); // no rate given at all
        let expect = [25.0 / 60.0, 20.0 / 40.0, 30.0 / 120.0, 1.0, 5.0 / 20.0];
        for (value, wanted) in shown.iter().zip(expect) {
            assert!((value - wanted).abs() < 1e-6, "{:?}", shown);
        }
        assert_eq!(shown, run(vec![4000.0])); // and a rate changes nothing
        // The hand-written rule reads the share that succeeded off its entry.
        let mut ruled = Limiter::new(Settings {
            tenants: vec!["a".into()],
            shares: vec![0.0, 3.0],
            default_level: 1,
            scale_free: true,
            hand_rule: Some(HandRule::default()),
            ..Settings::default()
        })
        .unwrap();
        for i in 0..40 {
            ruled.admit(0, false, i);
            ruled.ended(0, false, false, i + 1);
        }
        assert_eq!(ruled.step(1000), vec![0]); // nothing succeeds: closed
        for i in 0..40 {
            ruled.admit(0, false, 1000 + i);
            ruled.ended(0, false, i % 2 == 0, 1001 + i);
        }
        assert_eq!(ruled.step(2000), vec![1]); // half succeeds: up
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
