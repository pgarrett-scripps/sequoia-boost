//! Rough wall-clock timing (not a benchmark): hist training, 50k x 20,
//! depth 6, 50 rounds, median of 3; prediction with 100 trees, mean of 3.
//! Prints `TIMING <objective> <threads> <train_ms> <predict_ms>` lines.
//! Usage: timing_rough <threads>...
use sequoia_boost::prelude::*;
use sequoia_verify::lcg;
use std::time::Instant;

fn main() -> Result<()> {
    let threads: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("threads"))
        .collect();
    let (n, p) = (50_000usize, 20usize);
    let mut s = 42u64;
    let x: Vec<f32> = (0..n * p).map(|_| lcg(&mut s) * 4.0 - 2.0).collect();
    let yr: Vec<f32> = (0..n)
        .map(|i| x[i * p] * 1.5 - x[i * p + 1] * x[i * p + 2] + (x[i * p + 3] * 2.0).sin())
        .collect();
    let yb: Vec<f32> = yr.iter().map(|&v| (v > 0.2) as u8 as f32).collect();
    for (obj, y) in [("reg:squarederror", &yr), ("binary:logistic", &yb)] {
        for &nt in &threads {
            let d = DMatrix::from_dense(&x, n, p)?.with_labels(y)?;
            let params = TrainingParams::builder()
                .objective(obj)
                .max_depth(6)
                .eta(0.1)
                .max_bin(256)
                .tree_method(TreeMethod::Hist)
                .nthread(nt)
                .build()?;
            let mut ts = vec![];
            for _ in 0..3 {
                let t = Instant::now();
                std::hint::black_box(train(&params, &d, 50)?);
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = train(&params, &d, 100)?;
            let t = Instant::now();
            for _ in 0..3 {
                std::hint::black_box(m.predict(&d)?);
            }
            let tp = t.elapsed().as_secs_f64() * 1e3 / 3.0;
            println!("TIMING {obj} {nt} {:.1} {:.1}", ts[1], tp);
        }
    }
    Ok(())
}
