//! Criterion benchmarks for the hot paths: histogram construction and
//! end-to-end training with the histogram and exact tree methods.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sequoia_boost::data::ghist::GHistIndex;
use sequoia_boost::data::quantile::HistCuts;
use sequoia_boost::objective::GradPair;
use sequoia_boost::prelude::*;
use sequoia_boost::tree::hist::{zeroed, CpuBackend, HistogramBackend};

/// Deterministic synthetic regression dataset.
fn make_data(n: usize, f: usize) -> DMatrix {
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    let mut s: u64 = 0x2545F4914F6CDD1D;
    let mut rng = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u32 << 31) as f32
    };
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..f {
            let v = rng();
            x[i * f + j] = v;
            if j < 5 {
                acc += v * (j as f32 + 1.0);
            }
        }
        y[i] = acc + rng() * 0.1;
    }
    DMatrix::from_dense(&x, n, f).unwrap().with_labels(&y).unwrap()
}

fn bench_histogram_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram_build");
    for &n in &[10_000usize, 100_000] {
        let data = make_data(n, 30);
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = (0..n).map(|i| GradPair::new((i % 7) as f32 - 3.0, 1.0)).collect();
        let rows: Vec<u32> = (0..n as u32).collect();
        let mut out = zeroed(ghist.total_bins());

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| CpuBackend.build(&ghist, &rows, &gpair, &mut out));
        });
    }
    group.finish();
}

fn bench_train(c: &mut Criterion) {
    let data = make_data(50_000, 20);
    let mut group = c.benchmark_group("train_50k_x20_50rounds");
    group.sample_size(10);

    for method in [TreeMethod::Hist, TreeMethod::Exact] {
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(method)
            .max_depth(6)
            .eta(0.1)
            .build()
            .unwrap();
        group.bench_function(format!("{method:?}"), |b| {
            b.iter(|| train(&params, &data, 50).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_histogram_build, bench_train);
criterion_main!(benches);
