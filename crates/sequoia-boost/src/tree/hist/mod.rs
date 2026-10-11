//! Gradient-histogram construction backends.
//!
//! The [`HistogramBackend`] trait is the single seam a future GPU implementation
//! plugs into: everything above it (the histogram tree builder) is
//! backend-agnostic. The CPU backend uses `rayon` to accumulate per-thread
//! partial histograms and reduce them, and provides the *subtraction trick*
//! (`sibling = parent − child`) that halves histogram construction cost.

use crate::data::ghist::{Bins, GHistIndex};
use crate::objective::GradPair;
use crate::tree::gain::GradStats;
use rayon::prelude::*;

/// A gradient histogram: one [`GradStats`] bucket per global bin.
pub type Histogram = Vec<GradStats>;

/// Construct a fresh zeroed histogram of the given length.
pub fn zeroed(total_bins: usize) -> Histogram {
    vec![GradStats::default(); total_bins]
}

/// Turn `parent` into the sibling histogram `parent − child` in place. Reusing
/// the parent's buffer avoids allocating and writing a third histogram.
pub fn subtract_in_place(parent: &mut [GradStats], child: &[GradStats]) {
    debug_assert_eq!(parent.len(), child.len());
    for (p, c) in parent.iter_mut().zip(child) {
        *p = p.sub(*c);
    }
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

/// Each task needs enough rows to amortize its histogram allocation and merge.
/// Capping the task count avoids creating a full histogram for every worker
/// when a shallow node has only a few thousand rows.
const ROWS_PER_TASK: usize = 4096;
const PARALLEL_THRESHOLD: usize = 2 * ROWS_PER_TASK;
/// Bins per task when the partial histograms are summed.
const REDUCE_BINS: usize = 2048;

impl HistogramBackend for CpuBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        let total = out.len();
        let threads = rayon::current_num_threads();
        if threads <= 1 || rows.len() < PARALLEL_THRESHOLD {
            out.iter_mut().for_each(|s| *s = GradStats::default());
            accumulate(ghist, rows, gpair, out);
            return;
        }

        // A contiguous row range with a column-major copy is split by feature
        // instead of by rows: each task streams its features' columns over
        // every row straight into the feature's slice of `out`. Every bin has
        // one writer that adds its rows in ascending order, so there are no
        // partial histograms to allocate or reduce and the result is identical
        // to the sequential sweep.
        if let (Some(columns), Some(range)) = (ghist.column_bins(), contiguous_range(rows)) {
            let n_rows = ghist.n_rows();
            let cuts = ghist.cuts();
            let mut slices = Vec::with_capacity(ghist.n_cols());
            let mut rest = out;
            let mut next = 0;
            for f in 0..ghist.n_cols() {
                let (fs, fe) = cuts.feature_bins(f);
                assert_eq!(
                    fs, next,
                    "feature bin ranges must be contiguous and ordered"
                );
                let (head, tail) = rest.split_at_mut(fe - fs);
                slices.push((fs, head));
                rest = tail;
                next = fe;
            }
            assert!(
                rest.is_empty(),
                "feature bin ranges must cover the histogram"
            );
            slices
                .into_par_iter()
                .enumerate()
                .for_each(|(f, (fs, slice))| {
                    slice.iter_mut().for_each(|s| *s = GradStats::default());
                    match columns {
                        Bins::U16(c) => {
                            accumulate_column(&c[f * n_rows..][..n_rows], fs, &range, gpair, slice)
                        }
                        Bins::U32(c) => {
                            accumulate_column(&c[f * n_rows..][..n_rows], fs, &range, gpair, slice)
                        }
                    }
                });
            return;
        }

        // Each task builds a private histogram over a contiguous run of rows.
        // The partials are then summed into `out` in task order, so the result
        // is deterministic for a given worker count; the reduction is split by
        // bin range, which keeps that per-bin order while using every worker.
        let tasks = threads.min(rows.len() / ROWS_PER_TASK);
        let grain = rows.len().div_ceil(tasks);
        let partials: Vec<Histogram> = rows
            .par_chunks(grain)
            .map(|chunk| {
                let mut local = zeroed(total);
                accumulate(ghist, chunk, gpair, &mut local);
                local
            })
            .collect();
        out.par_chunks_mut(REDUCE_BINS)
            .enumerate()
            .for_each(|(i, out)| {
                let start = i * REDUCE_BINS;
                let end = start + out.len();
                out.copy_from_slice(&partials[0][start..end]);
                for partial in &partials[1..] {
                    for (o, p) in out.iter_mut().zip(&partial[start..end]) {
                        o.add(*p);
                    }
                }
            });
    }
}

/// Rows to run ahead of the accumulation loop when prefetching. Each row's bins
/// and gradient are fetched into L1 before the loop needs them. Subsets deep in
/// the tree are too sparse for hardware stride prediction.
const PREFETCH_ROWS: usize = 8;
const CACHE_LINE: usize = 64;

/// Bin-index storage widths the accumulation and partition loops specialize on.
pub(crate) trait BinIndex: Copy + Send + Sync {
    fn index(self) -> usize;
}

impl BinIndex for u16 {
    #[inline(always)]
    fn index(self) -> usize {
        self as usize
    }
}

impl BinIndex for u32 {
    #[inline(always)]
    fn index(self) -> usize {
        self as usize
    }
}

/// Sequential accumulation of `rows` into `out` (added, not reset). Specialized
/// on the bin-index width so the inner loop reads the narrowest integers.
#[inline]
fn accumulate(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
    match ghist.bins() {
        Bins::U16(bins) => accumulate_bins(ghist, bins, rows, gpair, out),
        Bins::U32(bins) => accumulate_bins(ghist, bins, rows, gpair, out),
    }
}

#[inline(always)]
fn accumulate_bins<B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    rows: &[u32],
    gpair: &[GradPair],
    out: &mut [GradStats],
) {
    // Establishes the bound used by `add_row`: with `out` covering every bin,
    // the `GHistIndex` invariant (all stored bins < total_bins) makes every
    // histogram index in range.
    assert_eq!(
        out.len(),
        ghist.total_bins(),
        "histogram length must equal the binned index's bin count"
    );
    let add_row = |row_bins: &[B], gp: GradPair, out: &mut [GradStats]| {
        let g = GradStats::new(gp.grad as f64, gp.hess as f64);
        for &bin in row_bins {
            // SAFETY: `bin < ghist.total_bins() == out.len()` by the index
            // invariant and the assertion above.
            unsafe { out.get_unchecked_mut(bin.index()) }.add(g);
        }
    };
    let prefetch_row = |start: usize, len: usize| {
        for offset in (0..len).step_by(CACHE_LINE / std::mem::size_of::<B>()) {
            if let Some(bin) = bins.get(start + offset) {
                crate::simd::prefetch_read(bin);
            }
        }
    };

    // A contiguous row range (the root, or a root chunk, without row
    // sampling) sweeps the column-major copy one feature at a time: the bins
    // stream sequentially and each feature's histogram slice stays in L1.
    // Every bin still receives its rows in ascending order, so the sums are
    // identical to the row sweep.
    if let Some(columns) = ghist.column_bins() {
        if let Some(range) = contiguous_range(rows) {
            let n_rows = ghist.n_rows();
            match columns {
                Bins::U16(columns) => accumulate_columns(columns, n_rows, range, gpair, out),
                Bins::U32(columns) => accumulate_columns(columns, n_rows, range, gpair, out),
            }
            return;
        }
    }
    if let Some(stride) = ghist.dense_stride() {
        accumulate_dense(ghist, bins, stride, rows, gpair, out, add_row, prefetch_row);
    } else {
        let rp = ghist.row_ptr();
        for (i, &r) in rows.iter().enumerate() {
            if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
                let ahead = ahead as usize;
                if let (Some(&start), Some(&end)) = (rp.get(ahead), rp.get(ahead + 1)) {
                    prefetch_row(start, end - start);
                }
                if let Some(gp) = gpair.get(ahead) {
                    crate::simd::prefetch_read(gp);
                }
            }
            let ri = r as usize;
            add_row(&bins[rp[ri]..rp[ri + 1]], gpair[ri], out);
        }
    }
}

/// `Some(first..end)` when `rows` is exactly the ascending run
/// `first, first + 1, ..., end - 1` (checked element by element, so unsorted
/// or repeated indices never take the column path).
#[inline]
fn contiguous_range(rows: &[u32]) -> Option<std::ops::Range<usize>> {
    let first = *rows.first()? as usize;
    let end = first.checked_add(rows.len())?;
    let contiguous = rows
        .iter()
        .enumerate()
        .all(|(i, &row)| row as usize == first + i);
    contiguous.then_some(first..end)
}

/// Column-wise accumulation of the rows in `range` over every feature.
/// `columns` is the column-major bin copy (`n_rows` entries per feature).
#[inline(always)]
fn accumulate_columns<B: BinIndex>(
    columns: &[B],
    n_rows: usize,
    range: std::ops::Range<usize>,
    gpair: &[GradPair],
    out: &mut [GradStats],
) {
    let gpair = &gpair[range.clone()];
    for column in columns.chunks_exact(n_rows) {
        for (&bin, gp) in column[range.clone()].iter().zip(gpair) {
            let g = GradStats::new(gp.grad as f64, gp.hess as f64);
            // SAFETY: `bin < ghist.total_bins() == out.len()` by the index
            // invariant and the caller's assertion.
            unsafe { out.get_unchecked_mut(bin.index()) }.add(g);
        }
    }
}

/// Column-wise accumulation of the rows in `range` for one feature whose
/// global bins start at `first_bin`, into that feature's histogram `slice`.
/// Bins in a column are global indices, so the slice is indexed relative to
/// `first_bin`. The subtraction is unchecked because the binned-index
/// invariant guarantees every bin of this feature's column is at least `first_bin`, and
/// the slice index bounds check catches any violation.
#[inline(always)]
fn accumulate_column<B: BinIndex>(
    column: &[B],
    first_bin: usize,
    range: &std::ops::Range<usize>,
    gpair: &[GradPair],
    slice: &mut [GradStats],
) {
    for (&bin, gp) in column[range.clone()].iter().zip(&gpair[range.clone()]) {
        let g = GradStats::new(gp.grad as f64, gp.hess as f64);
        slice[bin.index() - first_bin].add(g);
    }
}

/// Rows per tile of the dense accumulation. A tile's row lines stay in L2 while
/// its feature blocks are swept, so re-reading them per block is cheap.
const TILE_ROWS: usize = 4096;
/// Histogram bins a feature block may span, sized so the block's histogram
/// slice (16 bytes per bin) stays resident in a 64 KiB L1 while a tile is
/// accumulated into it.
const BLOCK_BINS: usize = 4096;

/// Dense accumulation tiled by rows and feature blocks. Every bin still
/// receives its rows in ascending order, so the result is identical to a
/// straight row sweep. The tiling only changes which histogram bins are hot.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn accumulate_dense<B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    stride: usize,
    rows: &[u32],
    gpair: &[GradPair],
    out: &mut [GradStats],
    add_row: impl Fn(&[B], GradPair, &mut [GradStats]),
    prefetch_row: impl Fn(usize, usize),
) {
    // Feature blocks `[f0, f1)` whose bin ranges each span at most BLOCK_BINS.
    let cuts = ghist.cuts();
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut block_start = 0;
    for f in 1..=stride {
        let span = cuts.feature_bins(f - 1).1 - cuts.feature_bins(block_start).0;
        if span > BLOCK_BINS && f - 1 > block_start {
            blocks.push((block_start, f - 1));
            block_start = f - 1;
        }
    }
    blocks.push((block_start, stride));

    let prefetch = |rows: &[u32], i: usize| {
        if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
            let ahead = ahead as usize;
            prefetch_row(ahead * stride, stride);
            if let Some(gp) = gpair.get(ahead) {
                crate::simd::prefetch_read(gp);
            }
        }
    };
    if blocks.len() == 1 {
        // Dense rows sit at `r * stride`: no row-pointer load, and the address
        // of a future row is known without touching memory.
        for (i, &r) in rows.iter().enumerate() {
            prefetch(rows, i);
            let start = r as usize * stride;
            add_row(&bins[start..start + stride], gpair[r as usize], out);
        }
        return;
    }
    for tile in rows.chunks(TILE_ROWS) {
        for (block, &(f0, f1)) in blocks.iter().enumerate() {
            for (i, &r) in tile.iter().enumerate() {
                if block == 0 {
                    prefetch(tile, i);
                }
                let start = r as usize * stride;
                add_row(&bins[start + f0..start + f1], gpair[r as usize], out);
            }
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
        let n = PARALLEL_THRESHOLD + 37;
        let x: Vec<f32> = (0..n).map(|i| (i % 251) as f32).collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 64);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair::new(((i * 7) % 13) as f32 - 6.0, 1.0))
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();

        let mut out = zeroed(ghist.total_bins());
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| CpuBackend.build(&ghist, &rows, &gpair, &mut out));
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

    /// Row-major reference independent of `accumulate`: every bin receives
    /// its rows in ascending order, which is the order both the row sweep and
    /// the column sweep must reproduce bit for bit.
    fn row_order_reference(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair]) -> Histogram {
        let stride = ghist.dense_stride().expect("dense index");
        let mut h = zeroed(ghist.total_bins());
        for &r in rows {
            let r = r as usize;
            let g = GradStats::new(gpair[r].grad as f64, gpair[r].hess as f64);
            for f in 0..stride {
                let bin = match ghist.bins() {
                    Bins::U16(b) => b[r * stride + f] as usize,
                    Bins::U32(b) => b[r * stride + f] as usize,
                };
                h[bin].add(g);
            }
        }
        h
    }

    #[test]
    fn column_and_row_sweeps_match_reference_bit_for_bit() {
        let (n, f) = (3 * PARALLEL_THRESHOLD + 129, 7);
        let x: Vec<f32> = (0..n * f)
            .map(|i| ((i * 2654435761usize) % 1009) as f32 / 7.0)
            .collect();
        let data = DMatrix::from_dense(&x, n, f).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 64);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        assert!(
            ghist.column_bins().is_some(),
            "dense index keeps a column copy"
        );
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| {
                GradPair::new(
                    ((i * 7919) % 1237) as f32 / 331.0 - 1.9,
                    0.25 + (i % 5) as f32,
                )
            })
            .collect();
        let bits = |h: &Histogram| -> Vec<(u64, u64)> {
            h.iter()
                .map(|s| (s.grad.to_bits(), s.hess.to_bits()))
                .collect()
        };
        let all: Vec<u32> = (0..n as u32).collect();
        let subset: Vec<u32> = (0..n as u32).filter(|r| r % 3 != 1).collect();
        let offset: Vec<u32> = (1000..(1000 + PARALLEL_THRESHOLD) as u32).collect();
        assert!(contiguous_range(&all).is_some() && contiguous_range(&offset).is_some());
        assert!(contiguous_range(&subset).is_none());
        assert!(contiguous_range(&[2, 0, 1]).is_none() && contiguous_range(&[5, 5]).is_none());
        for rows in [&all, &subset, &offset] {
            let expect = bits(&row_order_reference(&ghist, rows, &gpair));
            for threads in [1, 4] {
                let mut out = zeroed(ghist.total_bins());
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| CpuBackend.build(&ghist, rows, &gpair, &mut out));
                assert_eq!(bits(&out), expect, "rows={} threads={threads}", rows.len());
            }
        }
    }
}
