//! Coarse phase profiler for the histogram training pipeline.
//!
//! perf/flamegraph are unavailable in some sandboxes (`perf_event_paranoid`),
//! so this reconstructs the boosting loop from public building blocks and times
//! each phase: binning, gradient computation, tree construction, margin update,
//! and prediction. Run: `BENCH_DIR=<dir> cargo run --release --example profile`.

use sequoia_boost::data::ghist::GHistIndex;
use sequoia_boost::data::quantile::HistCuts;
use sequoia_boost::objective::{create_objective, GradPair};
use sequoia_boost::prelude::*;
use sequoia_boost::tree::builder::HistTreeBuilder;
use sequoia_boost::tree::sampler::ColumnSampler;
use std::time::{Duration, Instant};

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<()> {
    let dir = std::env::var("BENCH_DIR").expect("set BENCH_DIR");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap())
            .unwrap();
    let n = meta["n_rows"].as_u64().unwrap() as usize;
    let f = meta["n_cols"].as_u64().unwrap() as usize;
    let rounds = meta["num_round"].as_u64().unwrap() as usize;
    let max_depth = meta["max_depth"].as_u64().unwrap() as usize;
    let max_bin = meta["max_bin"].as_u64().unwrap() as usize;

    let x = read_f32(&format!("{dir}/X.bin"));
    let y = read_f32(&format!("{dir}/y.bin"));
    let d = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)
        .max_depth(max_depth)
        .eta(0.1)
        .max_bin(max_bin)
        .base_score(0.5)
        .build()?;

    // Phase: binning (done once per train() call).
    let t = Instant::now();
    let cuts = HistCuts::from_dmatrix(&d, max_bin);
    let ghist = GHistIndex::from_dmatrix(&d, cuts);
    let t_bin = t.elapsed();

    let obj = create_objective(&params)?;
    let mut margin = vec![0.5f32; n];
    let mut gpair = vec![GradPair::default(); n];
    let rows: Vec<u32> = (0..n as u32).collect();

    let (mut t_grad, mut t_build, mut t_update) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);

    let total = Instant::now();
    for _ in 0..rounds {
        let t = Instant::now();
        obj.gradient(&margin, &y, None, &mut gpair);
        t_grad += t.elapsed();

        let t = Instant::now();
        let mut sampler = ColumnSampler::all(f);
        let mut tree = HistTreeBuilder::new(&params).build(&ghist, &gpair, &rows, &mut sampler);
        tree.scale_leaves(params.eta as f32);
        t_build += t.elapsed();

        let t = Instant::now();
        for (row, m) in margin.iter_mut().enumerate() {
            *m += tree.predict_row(&d, row);
        }
        t_update += t.elapsed();
    }
    let t_loop = total.elapsed();

    let pct = |d: Duration| 100.0 * d.as_secs_f64() / t_loop.as_secs_f64();
    println!("dataset {n} x {f}, {rounds} rounds, depth {max_depth}\n");
    println!("binning (once)      {:>8.3?}", t_bin);
    println!("--- per-round loop total {:>8.3?} ---", t_loop);
    println!("  gradient          {:>8.3?}  {:5.1}%", t_grad, pct(t_grad));
    println!(
        "  tree build        {:>8.3?}  {:5.1}%",
        t_build,
        pct(t_build)
    );
    println!(
        "  margin update     {:>8.3?}  {:5.1}%",
        t_update,
        pct(t_update)
    );

    Ok(())
}
