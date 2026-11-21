//! The filter's forward pass picks what the trained model picks.
//!
//! `tests/fixture.json` is written by `export_policy.py --fixture`: the exported
//! weights, observations, and the actions Stable-Baselines3 takes on them.

use learned_filter::limiter::Settings;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    limiter: Settings,
    observations: Vec<Vec<f64>>,
    actions: Vec<Vec<usize>>,
}

#[test]
fn the_same_action_as_the_trained_model_on_every_observation() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixture.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("no {}: run export_policy.py --fixture to make it", path);
        return;
    };
    let fixture: Fixture = serde_json::from_str(&text).unwrap();
    let levels = fixture.limiter.shares.len();
    // What the POLICY decides for at once: a policy for one tenant may be
    // configured in front of several, and is then asked once per tenant.
    let tenants = fixture.limiter.policy.tenants(levels).unwrap();
    assert!(!fixture.observations.is_empty());
    for (observation, expected) in fixture.observations.iter().zip(&fixture.actions) {
        let action = fixture.limiter.policy.act(observation, tenants, levels).unwrap();
        assert_eq!(&action, expected);
    }
}
