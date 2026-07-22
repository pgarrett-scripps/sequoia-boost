//! Multiclass classification with `multi:softprob` (per-class probabilities) and
//! `predict_class` (argmax). Run: `cargo run --release --example multiclass`.

use sequoia_boost::prelude::*;

fn main() -> Result<()> {
    // 3 classes determined by a region of two features.
    let (n, f, k) = (1500usize, 4usize, 3usize);
    let mut rng = lcg(7);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    for i in 0..n {
        for j in 0..f {
            x[i * f + j] = rng();
        }
        let s = x[i * f] + x[i * f + 1];
        y[i] = if s < 0.7 {
            0.0
        } else if s < 1.3 {
            1.0
        } else {
            2.0
        };
    }
    let dtrain = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;

    let params = TrainingParams::builder()
        .objective("multi:softprob")
        .num_class(k) // required for multiclass
        .max_depth(4)
        .eta(0.2)
        .build()?;

    let model = train(&params, &dtrain, 60)?;

    // Probabilities are laid out `[row][class]` (length n * num_class).
    let probs = model.predict(&dtrain)?;
    let row0 = &probs[0..k];
    println!(
        "row 0 class probabilities: {row0:?} (sums to {:.3})",
        row0.iter().sum::<f32>()
    );

    // Hard predictions via argmax.
    let classes = model.predict_class(&dtrain)?;
    let acc = classes
        .iter()
        .zip(&y)
        .filter(|(c, l)| **c as f32 == **l)
        .count() as f32
        / n as f32;
    println!("training accuracy: {acc:.3}");
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
