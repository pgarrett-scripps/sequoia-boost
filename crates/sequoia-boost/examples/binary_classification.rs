//! Binary classification with `binary:logistic`, a watched eval set, and early
//! stopping. Run: `cargo run --release --example binary_classification`.

use sequoia_boost::metric::Auc;
use sequoia_boost::prelude::*;
use sequoia_boost::Metric;

fn main() -> Result<()> {
    // Synthetic separable-ish data: label depends on a logit of two features.
    let (n, f) = (2000usize, 4usize);
    let mut rng = lcg(42);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    for i in 0..n {
        for j in 0..f {
            x[i * f + j] = rng();
        }
        let logit = 3.0 * x[i * f] - 2.0 * x[i * f + 1] - 0.5;
        let p = 1.0 / (1.0 + (-logit).exp());
        y[i] = if p > rng() { 1.0 } else { 0.0 };
    }
    // Simple train/valid split.
    let split = 1600 * f;
    let dtrain = DMatrix::from_dense(&x[..split], 1600, f)?.with_labels(&y[..1600])?;
    let dvalid = DMatrix::from_dense(&x[split..], 400, f)?.with_labels(&y[1600..])?;

    let params = TrainingParams::builder()
        .objective("binary:logistic")
        .eval_metric("logloss")
        .eval_metric("auc")
        .max_depth(4)
        .eta(0.1)
        .subsample(0.9)
        .build()?;

    // Watch `dvalid`; stop after 20 rounds without improvement on the last metric.
    let out = train_with_eval(&params, &dtrain, 500, &[(&dvalid, "valid")], Some(20))?;
    let model = out.model;
    println!(
        "stopped at {} trees (best iteration {:?})",
        model.num_trees(),
        model.best_iteration()
    );

    let probs = model.predict(&dvalid)?; // probabilities in [0, 1]
    let classes = model.predict_class(&dvalid)?; // hard 0/1 labels
    let acc = classes
        .iter()
        .zip(dvalid.labels().unwrap())
        .filter(|(c, l)| **c as f32 == **l)
        .count() as f32
        / 400.0;
    let auc = Auc.eval(&probs, dvalid.labels().unwrap(), None);
    println!("valid accuracy {acc:.3}, AUC {auc:.3}");
    Ok(())
}

/// A tiny deterministic RNG so the example needs no dependency.
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u32 << 31) as f32
    }
}
