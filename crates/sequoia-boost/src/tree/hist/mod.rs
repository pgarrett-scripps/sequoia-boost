//! Gradient-histogram construction backends.
//!
//! The [`HistogramBackend`] trait is the single seam a future GPU implementation
//! plugs into: everything above it (the histogram tree builder) is
//! backend-agnostic. The CPU backend uses `rayon` to accumulate per-thread
//! partial histograms and reduce them, and provides the *subtraction trick*
//! (`sibling = parent − child`) that halves histogram construction cost.

use crate::data::ghist::GHistIndex;
use crate::objective::GradPair;
use crate::tree::gain::GradStats;
use rayon::prelude::*;

/// A gradient histogram: one [`GradStats`] bucket per global bin.
pub type Histogram = Vec<GradStats>;

/// Construct a fresh zeroed histogram of the given length.
pub fn zeroed(total_bins: usize) -> Histogram {
    vec![GradStats::default(); total_bins]
}

/// Backend that builds and combines gradient histograms.
pub trait HistogramBackend: Send + Sync {
    /// Accumulate the gradients of `rows` into `out` (length = total bins).
    /// `out` is overwritten (not added to).
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]);

    /// Compute `out[i] = parent[i] − child[i]` for every bin (the sibling
    /// histogram via subtraction).
    fn subtract(&self, parent: &[GradStats], child: &[GradStats], out: &mut [GradStats]) {
        debug_assert_eq!(parent.len(), child.len());
        debug_assert_eq!(parent.len(), out.len());
        for i in 0..out.len() {
            out[i] = parent[i].sub(child[i]);
        }
    }
}

/// Multi-core CPU histogram backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

/// Below this row count the sequential path avoids rayon overhead.
const PARALLEL_THRESHOLD: usize = 4096;

impl HistogramBackend for CpuBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        let total = out.len();
        out.iter_mut().for_each(|s| *s = GradStats::default());

        if rows.len() < PARALLEL_THRESHOLD {
            accumulate(ghist, rows, gpair, out);
            return;
        }

        // Parallel: each chunk builds a private histogram; reduce by summation.
        let grain = (rows.len() / rayon::current_num_threads().max(1)).max(1);
        let partial = rows
            .par_chunks(grain)
            .map(|chunk| {
                let mut local = zeroed(total);
                accumulate(ghist, chunk, gpair, &mut local);
                local
            })
            .reduce(
                || zeroed(total),
                |mut a, b| {
                    for i in 0..total {
                        a[i].add(b[i]);
                    }
                    a
                },
            );
        out.copy_from_slice(&partial);
    }
}

/// Sequential accumulation of `rows` into `out` (added, not reset).
#[inline]
fn accumulate(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
    for &r in rows {
        let ri = r as usize;
        let gp = gpair[ri];
        let g = GradStats::new(gp.grad as f64, gp.hess as f64);
        for &bin in ghist.row_bins(ri) {
            out[bin as usize].add(g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::quantile::HistCuts;
    use crate::data::DMatrix;

    fn brute_force(
        ghist: &GHistIndex,
        rows: &[u32],
        gpair: &[GradPair],
        total: usize,
    ) -> Histogram {
        let mut h = zeroed(total);
        accumulate(ghist, rows, gpair, &mut h);
        h
    }

    #[test]
    fn build_matches_brute_force() {
        let n = 200;
        let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32).collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 32);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair::new((i as f32) * 0.1 - 5.0, 1.0))
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();

        let mut out = zeroed(ghist.total_bins());
        CpuBackend.build(&ghist, &rows, &gpair, &mut out);
        let expect = brute_force(&ghist, &rows, &gpair, ghist.total_bins());
        for (a, b) in out.iter().zip(&expect) {
            assert!((a.grad - b.grad).abs() < 1e-4);
            assert!((a.hess - b.hess).abs() < 1e-4);
        }
    }

    #[test]
    fn subtraction_identity() {
        // parent = left + right, so parent - left = right.
        let total = 8;
        let mut parent = zeroed(total);
        let mut left = zeroed(total);
        let mut right = zeroed(total);
        for i in 0..total {
            left[i] = GradStats::new(i as f64, 1.0);
            right[i] = GradStats::new(-(i as f64) * 0.5, 2.0);
            parent[i] = GradStats::new(left[i].grad + right[i].grad, left[i].hess + right[i].hess);
        }
        let mut out = zeroed(total);
        CpuBackend.subtract(&parent, &left, &mut out);
        for i in 0..total {
            assert!((out[i].grad - right[i].grad).abs() < 1e-12);
            assert!((out[i].hess - right[i].hess).abs() < 1e-12);
        }
    }

    #[test]
    fn parallel_matches_sequential_large() {
        let n = 20_000; // exceeds PARALLEL_THRESHOLD
        let x: Vec<f32> = (0..n).map(|i| (i % 251) as f32).collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 64);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair::new(((i * 7) % 13) as f32 - 6.0, 1.0))
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();

        let mut out = zeroed(ghist.total_bins());
        CpuBackend.build(&ghist, &rows, &gpair, &mut out);
        let expect = brute_force(&ghist, &rows, &gpair, ghist.total_bins());
        for (a, b) in out.iter().zip(&expect) {
            assert!(
                (a.grad - b.grad).abs() < 1e-2,
                "grad {} vs {}",
                a.grad,
                b.grad
            );
            assert!((a.hess - b.hess).abs() < 1e-2);
        }
    }
}
