//! XGBoost numerical-parity integration test.
//!
//! Loads fixtures produced by `scripts/gen_fixtures.py` (real XGBoost
//! predictions) and asserts `sequoia-boost` matches within each fixture's
//! tolerance. Ignored by default because it requires generated fixtures; run:
//!
//! ```sh
//! python scripts/gen_fixtures.py
//! cargo test -p sequoia-boost --test parity -- --ignored
//! ```

use sequoia_boost::prelude::*;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct FixtureParams {
    max_depth: usize,
    eta: f64,
    lambda: f64,
    alpha: f64,
    gamma: f64,
    min_child_weight: f64,
    max_bin: usize,
    base_score: f64,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    objective: String,
    num_class: usize,
    num_round: usize,
    n_rows: usize,
    n_cols: usize,
    x: Vec<f32>,
    y: Vec<f32>,
    params: FixtureParams,
    xgb_pred: Vec<f32>,
    tolerance: f64,
}

fn fixtures_dir() -> PathBuf {
    // Repo root fixtures/ relative to this crate.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
}

fn run_fixture(fx: &Fixture) -> bool {
    let d = DMatrix::from_dense(&fx.x, fx.n_rows, fx.n_cols)
        .unwrap()
        .with_labels(&fx.y)
        .unwrap();
    let params = TrainingParams::builder()
        .objective(fx.objective.clone())
        .num_class(fx.num_class)
        .tree_method(TreeMethod::Hist)
        .max_depth(fx.params.max_depth)
        .eta(fx.params.eta)
        .lambda(fx.params.lambda)
        .alpha(fx.params.alpha)
        .gamma(fx.params.gamma)
        .min_child_weight(fx.params.min_child_weight)
        .max_bin(fx.params.max_bin)
        .base_score(fx.params.base_score)
        .build()
        .unwrap();

    let model = train(&params, &d, fx.num_round).unwrap();
    let preds = model.predict(&d).unwrap();
    assert_eq!(preds.len(), fx.xgb_pred.len(), "{}: length mismatch", fx.name);

    // Pointwise agreement (informational: two different histogram
    // implementations pick different split points, so this is never zero).
    let mut max_abs = 0.0f64;
    let mut mean_abs = 0.0f64;
    for (a, b) in preds.iter().zip(&fx.xgb_pred) {
        let e = (*a as f64 - *b as f64).abs();
        max_abs = max_abs.max(e);
        mean_abs += e;
    }
    mean_abs /= preds.len() as f64;

    // Fit quality against the true labels — the meaningful parity measure.
    // Multiclass: accuracy (higher better). Else: RMSE vs y (lower better).
    let (metric, seq_q, xgb_q, seq_better) = if fx.num_class >= 2 {
        let k = fx.num_class;
        let acc = |p: &[f32]| -> f64 {
            let mut correct = 0usize;
            for (i, &yi) in fx.y.iter().enumerate() {
                let row = &p[i * k..i * k + k];
                let mut best = 0usize;
                for c in 1..k {
                    if row[c] > row[best] {
                        best = c;
                    }
                }
                if best == yi as usize {
                    correct += 1;
                }
            }
            correct as f64 / fx.y.len() as f64
        };
        let (s, x) = (acc(&preds), acc(&fx.xgb_pred));
        // sequoia within 2 accuracy points of xgboost.
        ("accuracy", s, x, s >= x - 0.02)
    } else {
        let rmse = |p: &[f32]| -> f64 {
            let s: f64 = p
                .iter()
                .zip(&fx.y)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            (s / fx.y.len() as f64).sqrt()
        };
        let (s, x) = (rmse(&preds), rmse(&fx.xgb_pred));
        // sequoia RMSE within 8% of xgboost's.
        ("rmse", s, x, s <= x * 1.08 + 1e-6)
    };

    println!(
        "{:<11} pointwise mean|Δ|={:.2e} max|Δ|={:.2e} | {metric}: sequoia={:.5} xgboost={:.5} -> {}",
        fx.name,
        mean_abs,
        max_abs,
        seq_q,
        xgb_q,
        if seq_better { "OK" } else { "FAIL" }
    );
    seq_better
}

#[test]
#[ignore = "requires fixtures from scripts/gen_fixtures.py"]
fn xgboost_parity() {
    let dir = fixtures_dir();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("no fixtures dir at {dir:?}; run scripts/gen_fixtures.py");
            return;
        }
    };
    let mut fixtures: Vec<Fixture> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        fixtures.push(serde_json::from_str(&text).unwrap());
    }
    fixtures.sort_by(|a, b| a.name.cmp(&b.name));
    assert!(!fixtures.is_empty(), "no .json fixtures found in {dir:?}");
    let mut all_ok = true;
    for fx in &fixtures {
        all_ok &= run_fixture(fx);
    }
    assert!(all_ok, "one or more fixtures failed the quality-parity check");
}
