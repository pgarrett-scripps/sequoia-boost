//! Saving and loading models: native binary, native JSON, and XGBoost-format
//! JSON (interoperable with real XGBoost). Run:
//! `cargo run --release --example model_io`.

use sequoia_boost::prelude::*;
use std::path::Path;

fn main() -> Result<()> {
    let (n, f) = (500usize, 4usize);
    let mut rng = lcg(99);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    for i in 0..n {
        for j in 0..f {
            x[i * f + j] = rng();
        }
        y[i] = x[i * f] - 2.0 * x[i * f + 1];
    }
    let d = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(3)
        .eta(0.2)
        .build()?;
    let model = train(&params, &d, 40)?;
    let before = model.predict(&d)?;

    let dir = std::env::temp_dir();
    let bin = dir.join("sequoia_model.sqb");
    let json = dir.join("sequoia_model.json");
    let xgb = dir.join("sequoia_xgb.json");

    // 1) Native binary (compact) round-trip.
    model.save_binary(&bin)?;
    let m_bin = BoostedModel::load_binary(&bin)?;

    // 2) Native JSON (human-readable) round-trip.
    model.save_json(&json)?;
    let m_json = BoostedModel::load_json(&json)?;

    // 3) XGBoost-format JSON — readable by real XGBoost's `Booster.load_model`.
    model.save_xgboost_json(&xgb)?;
    let m_xgb = BoostedModel::load_xgboost_json(&xgb)?;

    for (label, m) in [
        ("binary", &m_bin),
        ("json", &m_json),
        ("xgboost-json", &m_xgb),
    ] {
        let after = m.predict(&d)?;
        let max_diff = before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{label:<13} round-trip max |Δ| = {max_diff:.2e}");
    }

    for p in [&bin, &json, &xgb] {
        let _ = std::fs::remove_file(Path::new(p));
    }
    Ok(())
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u32 << 31) as f32
    }
}
