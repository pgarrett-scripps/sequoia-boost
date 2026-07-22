//! Bring-your-own loss and metric: the custom-objective and custom-metric hooks.
//! Run: `cargo run --release --example custom_objective`.

use sequoia_boost::prelude::*;

fn main() -> Result<()> {
    let (n, f) = (800usize, 3usize);
    let mut rng = lcg(5);
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    for i in 0..n {
        for j in 0..f {
            x[i * f + j] = rng();
        }
        y[i] = 1.5 * x[i * f] - x[i * f + 1];
    }
    let d = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;
    let params = TrainingParams::builder().max_depth(3).eta(0.2).build()?;

    // --- Custom objective: squared error via first/second-order gradients. ---
    // Signature: (raw_margins, labels, optional_weights, out_gradients).
    let obj = CustomObjective::new(
        "my:squarederror",
        1,
        0.0,
        "rmse",
        |preds, labels, w, out| {
            for i in 0..preds.len() {
                let wi = w.map_or(1.0, |ws| ws[i]);
                out[i] = GradPair::new((preds[i] - labels[i]) * wi, wi); // grad, hess
            }
        },
    );
    let model = train_with_objective(&params, &d, 60, Box::new(obj))?;
    let preds = model.predict(&d)?;
    let rmse = (preds
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / n as f32)
        .sqrt();
    println!("custom-objective RMSE: {rmse:.4}");

    // --- Custom metric: mean absolute error, used for early stopping. ---
    // Signature: (predictions, labels, optional_weights) -> f64; `maximize=false`.
    let mae = CustomMetric::new("my:mae", false, |p, l, _w| {
        p.iter()
            .zip(l)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .sum::<f64>()
            / p.len() as f64
    });
    let builtin = TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(3)
        .eta(0.2)
        .build()?;
    let out =
        train_with_custom_metric(&builtin, &d, 100, &[(&d, "train")], Some(10), Box::new(mae))?;
    println!(
        "custom-metric run: {} trees, last MAE = {:.4}",
        out.model.num_trees(),
        out.history.last().unwrap().scores.last().unwrap().2
    );
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
