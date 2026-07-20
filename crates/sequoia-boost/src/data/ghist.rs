//! Pre-binned feature storage (`GHistIndexMatrix` in XGBoost).
//!
//! Every non-missing entry is replaced by its global bin index (see
//! [`HistCuts`]). This compact, CSR-shaped layout is what the histogram builder
//! scans: accumulating gradients into per-bin buckets is a single indexed add.

use crate::data::quantile::HistCuts;
use crate::data::{DMatrix, Entry};

/// Binned dataset: for each row, the global bin indices of its non-missing
/// features, stored CSR-style.
#[derive(Debug, Clone)]
pub struct GHistIndex {
    n_rows: usize,
    row_ptr: Vec<usize>,
    /// Global bin index per present entry.
    bins: Vec<u32>,
    cuts: HistCuts,
}

impl GHistIndex {
    /// Bin a dataset against precomputed cuts.
    pub fn from_dmatrix(data: &DMatrix, cuts: HistCuts) -> Self {
        let n_rows = data.n_rows();
        let mut row_ptr = Vec::with_capacity(n_rows + 1);
        row_ptr.push(0usize);
        let mut bins: Vec<u32> = Vec::new();
        let mut row: Vec<Entry> = Vec::new();
        for r in 0..n_rows {
            data.row_into(r, &mut row);
            for e in &row {
                bins.push(cuts.bin_of(e.index as usize, e.value));
            }
            row_ptr.push(bins.len());
        }
        GHistIndex {
            n_rows,
            row_ptr,
            bins,
            cuts,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// The cut table this index was built against.
    #[inline]
    pub fn cuts(&self) -> &HistCuts {
        &self.cuts
    }

    /// Total number of histogram bins.
    #[inline]
    pub fn total_bins(&self) -> usize {
        self.cuts.total_bins()
    }

    /// The global bin indices present in row `r`.
    #[inline]
    pub fn row_bins(&self, r: usize) -> &[u32] {
        &self.bins[self.row_ptr[r]..self.row_ptr[r + 1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_roundtrip_dense() {
        let data = DMatrix::from_dense(&[0.0, 10.0, 1.0, 20.0], 2, 2).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        // Two present features per row.
        assert_eq!(ghist.row_bins(0).len(), 2);
        assert_eq!(ghist.row_bins(1).len(), 2);
        // Row 1's feature-0 value (1.0) bins higher than row 0's (0.0).
        let b0 = ghist.cuts().bin_of(0, 0.0);
        let b1 = ghist.cuts().bin_of(0, 1.0);
        assert!(b1 > b0);
    }

    #[test]
    fn missing_entries_absent() {
        let data = DMatrix::from_dense(&[0.0, f32::NAN, 1.0, 2.0], 2, 2).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        // Row 0 has a missing feature 1 -> only one present entry.
        assert_eq!(ghist.row_bins(0).len(), 1);
        assert_eq!(ghist.row_bins(1).len(), 2);
    }
}
