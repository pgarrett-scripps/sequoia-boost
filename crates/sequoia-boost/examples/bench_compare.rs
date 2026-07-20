//! Fair head-to-head timing harness vs real XGBoost.
//!
//! Reads a dataset written by `scripts/bench_xgb.py` (identical bytes for both
//! engines), trains sequoia-boost with matching `hist` parameters, and prints a
//! JSON line with fit time and train RMSE. Thread count is controlled by
//! `RAYON_NUM_THREADS`. Run via `scripts/run_bench.sh`.

use sequoia_boost::metric::Rmse;
use sequoia_boost::prelude::*;
use sequoia_boost::Metric;
use std::time::Instant;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read data file");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<()> {
    let dir = std::env::var("BENCH_DIR").expect("set BENCH_DIR");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap()).unwrap();
    let n_rows = meta["n_rows"].as_u64().unwrap() as usize;
    let n_cols = meta["n_cols"].as_u64().unwrap() as usize;
    let num_round = meta["num_round"].as_u64().unwrap() as usize;
    let max_depth = meta["max_depth"].as_u64().unwrap() as usize;
    let eta = meta["eta"].as_f64().unwrap();
    let lambda = meta["lambda"].as_f64().unwrap();
    let max_bin = meta["max_bin"].as_u64().unwrap() as usize;
    let base_score = meta["base_score"].as_f64().unwrap();

    let x = read_f32(&format!("{dir}/X.bin"));
    let y = read_f32(&format!("{dir}/y.bin"));
    assert_eq!(x.len(), n_rows * n_cols);

    let dtrain = DMatrix::from_dense(&x, n_rows, n_cols)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)
        .max_depth(max_depth)
        .eta(eta)
        .lambda(lambda)
        .max_bin(max_bin)
        .base_score(base_score)
        .build()?;

    // Best of N runs (warm allocator, reduce noise); N via BENCH_REPEATS.
    let repeats: usize = std::env::var("BENCH_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let mut best = f64::INFINITY;
    let mut rmse = 0.0;
    for _ in 0..repeats {
        let t = Instant::now();
        let model = train(&params, &dtrain, num_round)?;
        let secs = t.elapsed().as_secs_f64();
        best = best.min(secs);
        let preds = model.predict(&dtrain)?;
        rmse = Rmse.eval(&preds, dtrain.labels().unwrap(), None);
    }

    println!(
        "{{\"impl\":\"sequoia\",\"threads\":{},\"fit_s\":{:.3},\"rmse\":{:.5}}}",
        rayon::current_num_threads(),
        best,
        rmse
    );
    Ok(())
}
