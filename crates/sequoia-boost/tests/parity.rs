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

fn run_fixture(fx: &Fixture) {
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

    let mut max_abs = 0.0f64;
    let mut mean_abs = 0.0f64;
    for (a, b) in preds.iter().zip(&fx.xgb_pred) {
        let e = (*a as f64 - *b as f64).abs();
        max_abs = max_abs.max(e);
        mean_abs += e;
    }
    mean_abs /= preds.len() as f64;
    println!(
        "{}: mean|Δ|={:.2e} max|Δ|={:.2e} (tol {:.1e})",
        fx.name, mean_abs, max_abs, fx.tolerance
    );
    assert!(
        mean_abs <= fx.tolerance,
        "{}: mean abs error {:.3e} exceeds tolerance {:.3e}",
        fx.name,
        mean_abs,
        fx.tolerance
    );
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
    let mut ran = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let fx: Fixture = serde_json::from_str(&text).unwrap();
        run_fixture(&fx);
        ran += 1;
    }
    assert!(ran > 0, "no .json fixtures found in {dir:?}");
}
