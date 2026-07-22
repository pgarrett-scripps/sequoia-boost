//! Structural constraints: monotone constraints, interaction constraints, and
//! native categorical features. Run: `cargo run --release --example constraints`.

use sequoia_boost::prelude::*;

fn main() -> Result<()> {
    // ---- Monotone constraint: force the model non-decreasing in feature 0. ----
    let n = 400usize;
    let mut rng = lcg(1);
    let mut x = vec![0f32; n];
    let mut y = vec![0f32; n];
    for i in 0..n {
        x[i] = rng();
        // Noisy, non-monotone target the constraint will smooth to monotone.
        y[i] = x[i] + 0.3 * (rng() - 0.5);
    }
    let d = DMatrix::from_dense(&x, n, 1)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .monotone_constraints(vec![Monotone::Increasing]) // one entry per feature
        .max_depth(4)
        .eta(0.2)
        .build()?;
    let model = train(&params, &d, 60)?;
    // Predictions are non-decreasing in x when sorted by x.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let preds = model.predict(&d)?;
    let monotone = idx.windows(2).all(|w| preds[w[1]] >= preds[w[0]] - 1e-5);
    println!("monotone constraint respected: {monotone}");

    // ---- Interaction constraints: features may only interact within a group. ----
    let (n2, f2) = (600usize, 4usize);
    let mut x2 = vec![0f32; n2 * f2];
    let mut y2 = vec![0f32; n2];
    for i in 0..n2 {
        for j in 0..f2 {
            x2[i * f2 + j] = rng();
        }
        y2[i] = x2[i * f2] * x2[i * f2 + 1] + x2[i * f2 + 2];
    }
    let d2 = DMatrix::from_dense(&x2, n2, f2)?.with_labels(&y2)?;
    let params2 = TrainingParams::builder()
        .objective("reg:squarederror")
        // Feature 0 may only co-occur with 1; feature 2 only with 3.
        .interaction_constraints(vec![vec![0, 1], vec![2, 3]])
        .max_depth(4)
        .eta(0.2)
        .build()?;
    let m2 = train(&params2, &d2, 40)?;
    println!("interaction-constrained model: {} trees", m2.num_trees());

    // ---- Native categorical feature (unordered set-membership splits). ----
    let cats = [0.0f32, 1.0, 2.0, 3.0];
    let mut xc = Vec::new();
    let mut yc = Vec::new();
    for _ in 0..100 {
        for &c in &cats {
            xc.push(c);
            yc.push(if (c as u32) % 2 == 1 { 1.0 } else { 0.0 }); // non-ordinal
        }
    }
    let dc = DMatrix::from_dense(&xc, xc.len(), 1)?
        .with_labels(&yc)?
        .with_feature_types(&[FeatureType::Categorical])?;
    let mc = train(
        &TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(2)
            .eta(0.3)
            .build()?,
        &dc,
        30,
    )?;
    let pc = mc.predict(&dc)?;
    let rmse = (pc
        .iter()
        .zip(&yc)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / yc.len() as f32)
        .sqrt();
    println!("categorical fit RMSE on non-ordinal pattern: {rmse:.4}");
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
