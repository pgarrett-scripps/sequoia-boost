//! Model explainability: TreeSHAP feature contributions and interaction values.
//! Run: `cargo run --release --example shap`.

use sequoia_boost::prelude::*;

fn main() -> Result<()> {
    // y = 2*x0 - 3*x1 + a small x0*x2 interaction.
    let (n, f) = (1000usize, 3usize);
    let mut rng = lcg(3);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    for i in 0..n {
        for j in 0..f {
            x[i * f + j] = rng();
        }
        y[i] = 2.0 * x[i * f] - 3.0 * x[i * f + 1] + x[i * f] * x[i * f + 2];
    }
    let d = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(4)
        .eta(0.2)
        .build()?;
    let model = train(&params, &d, 80)?;

    // Contributions: shape [row][n_features + 1]; last column is the bias.
    // Their sum equals the raw margin prediction (SHAP additivity).
    let contribs = model.predict_contribs(&d)?;
    let width = f + 1;
    let margin0 = model.predict_margin(&d)[0];
    let sum0: f32 = contribs[0..width].iter().sum();
    println!("row 0 SHAP contributions {:?}", &contribs[0..width]);
    println!("  sum {sum0:.4} ≈ margin {margin0:.4}");

    // Interaction values: shape [row][(n_features+1) x (n_features+1)].
    // Off-diagonal (0,2) should be non-trivial thanks to the x0*x2 term.
    let inter = model.predict_interactions(&d)?;
    let w2 = width * width;
    let m = |row: usize, i: usize, j: usize| inter[row * w2 + i * width + j];
    println!("row 0 interaction[0][2] = {:.4}", m(0, 0, 2));
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
