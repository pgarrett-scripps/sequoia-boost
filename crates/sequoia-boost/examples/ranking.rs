//! Learning-to-rank with LambdaMART (`rank:ndcg`) over query groups. Run:
//! `cargo run --release --example ranking`.

use sequoia_boost::prelude::*;

fn main() -> Result<()> {
    // `n_groups` queries, each with `per` documents. A document's single feature
    // is correlated with its relevance grade (0..per-1), plus noise.
    let (n_groups, per) = (200usize, 6usize);
    let n = n_groups * per;
    let mut rng = lcg(11);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n_groups {
        for d in 0..per {
            let rel = d as f32; // graded relevance
            x.push(rel + (rng() - 0.5) * 0.9);
            y.push(rel);
        }
    }
    // `with_group_sizes` marks the query boundaries (each group has `per` rows).
    let sizes = vec![per; n_groups];
    let dtrain = DMatrix::from_dense(&x, n, 1)?
        .with_labels(&y)?
        .with_group_sizes(&sizes)?;

    let params = TrainingParams::builder()
        .objective("rank:ndcg")
        .max_depth(3)
        .eta(0.2)
        .build()?;

    // Watch NDCG on the training set to see it improve.
    let out = train_with_eval(&params, &dtrain, 40, &[(&dtrain, "train")], None)?;
    let ndcg = |r: &sequoia_boost::learner::RoundEval| {
        r.scores.iter().find(|(_, m, _)| m == "ndcg").unwrap().2
    };
    println!(
        "NDCG: round 0 = {:.3}  →  final = {:.3}",
        ndcg(&out.history[0]),
        ndcg(out.history.last().unwrap())
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
