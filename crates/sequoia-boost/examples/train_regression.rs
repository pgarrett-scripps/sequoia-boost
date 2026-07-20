//! End-to-end regression example.
//!
//! Fits a noisy nonlinear function with squared-error boosting and reports the
//! training RMSE and the top feature importances.
//!
//! Run with: `cargo run --release --example train_regression`

use sequoia_boost::metric::Rmse;
use sequoia_boost::prelude::*;
use sequoia_boost::Metric;

fn main() -> Result<()> {
    // Synthetic dataset: y = 2*x0 - 3*x1^2 + 0.5*x2, with x3 an unused feature.
    let n_rows = 2000;
    let n_cols = 4;
    let mut x = Vec::with_capacity(n_rows * n_cols);
    let mut y = Vec::with_capacity(n_rows);
    // A tiny deterministic LCG so the example needs no rng dependency.
    let mut state: u64 = 0x1234_5678;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32) / (1u32 << 31) as f32
    };
    for _ in 0..n_rows {
        let x0 = next();
        let x1 = next();
        let x2 = next();
        let x3 = next();
        x.extend_from_slice(&[x0, x1, x2, x3]);
        y.push(2.0 * x0 - 3.0 * x1 * x1 + 0.5 * x2);
    }

    let dtrain = DMatrix::from_dense(&x, n_rows, n_cols)?.with_labels(&y)?;

    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(4)
        .eta(0.1)
        .subsample(0.9)
        .colsample_bytree(1.0)
        .lambda(1.0)
        .build()?;

    let model = train(&params, &dtrain, 200)?;

    let preds = model.predict(&dtrain)?;
    let rmse = Rmse.eval(&preds, dtrain.labels().unwrap(), None);
    println!("trained {} trees", model.num_trees());
    println!("training RMSE: {rmse:.5}");

    let importance = model.feature_importance(ImportanceType::Weight);
    let mut imp: Vec<_> = importance.into_iter().collect();
    imp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("feature importance (split count):");
    for (feat, score) in imp {
        println!("  feature {feat}: {score}");
    }

    Ok(())
}
