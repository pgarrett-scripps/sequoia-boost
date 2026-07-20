//! Pre-binned feature storage (`GHistIndexMatrix` in XGBoost).
//!
//! Every non-missing entry is replaced by its global bin index (see
//! [`HistCuts`]). This compact, CSR-shaped layout is what the histogram builder
//! scans: accumulating gradients into per-bin buckets is a single indexed add.
//!
//! Bin indices are stored in the **narrowest** integer type that fits the total
//! bin count — `u16` when there are ≤ 65 536 bins (the common case, e.g. 256
//! features × 256 bins), else `u32`. The build loop is memory-bandwidth bound,
//! so halving the index width is a direct throughput win.

use crate::data::quantile::HistCuts;
use crate::data::{DMatrix, Entry};

/// Backing storage for bin indices, in the narrowest width that fits.
#[derive(Debug, Clone)]
enum BinStore {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

/// A view over one row's (or all rows') bin indices, tagged by width so hot
/// loops can specialize with a single outer branch.
pub enum Bins<'a> {
    /// 16-bit bin indices.
    U16(&'a [u16]),
    /// 32-bit bin indices.
    U32(&'a [u32]),
}

/// Binned dataset: for each row, the global bin indices of its non-missing
/// features, stored CSR-style.
#[derive(Debug, Clone)]
pub struct GHistIndex {
    n_rows: usize,
    n_cols: usize,
    row_ptr: Vec<usize>,
    store: BinStore,
    cuts: HistCuts,
    /// True when every row is complete and in ascending feature order (a dense
    /// matrix with no missing values). Then feature `f` of row `r` is at offset
    /// `row_ptr[r] + f`, so routing needs no per-row scan.
    dense: bool,
}

impl GHistIndex {
    /// Bin a dataset against precomputed cuts.
    pub fn from_dmatrix(data: &DMatrix, cuts: HistCuts) -> Self {
        let n_rows = data.n_rows();
        let n_cols = cuts.n_features();
        let mut row_ptr = Vec::with_capacity(n_rows + 1);
        row_ptr.push(0usize);
        let mut bins: Vec<u32> = Vec::new();
        let mut row: Vec<Entry> = Vec::new();
        let mut dense = true;
        for r in 0..n_rows {
            data.row_into(r, &mut row);
            // Dense iff every row lists all features in ascending index order.
            dense &=
                row.len() == n_cols && row.iter().enumerate().all(|(c, e)| e.index as usize == c);
            for e in &row {
                bins.push(cuts.bin_of(e.index as usize, e.value));
            }
            row_ptr.push(bins.len());
        }

        // Downcast to u16 when every global bin index fits.
        let store = if cuts.total_bins() <= u16::MAX as usize + 1 {
            BinStore::U16(bins.iter().map(|&b| b as u16).collect())
        } else {
            BinStore::U32(bins)
        };

        GHistIndex {
            n_rows,
            n_cols,
            row_ptr,
            store,
            cuts,
            dense,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of feature columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
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

    /// CSR row-offset table (length `n_rows + 1`).
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// All bin indices, tagged by width. Combine with [`GHistIndex::row_ptr`] to
    /// slice a row's entries in a width-specialized loop.
    #[inline]
    pub fn bins(&self) -> Bins<'_> {
        match &self.store {
            BinStore::U16(v) => Bins::U16(v),
            BinStore::U32(v) => Bins::U32(v),
        }
    }

    /// Number of present (non-missing) entries in row `r`.
    #[inline]
    pub fn row_len(&self, r: usize) -> usize {
        self.row_ptr[r + 1] - self.row_ptr[r]
    }

    /// The global bin of `feature` (with global bin range `[fs, fe)`) in row `r`,
    /// or `None` if missing. Uses an O(1) direct index for dense datasets and
    /// falls back to a per-row scan otherwise.
    #[inline]
    pub fn feature_bin_at(&self, r: usize, feature: usize, fs: usize, fe: usize) -> Option<u32> {
        if self.dense {
            let idx = self.row_ptr[r] + feature;
            let b = match &self.store {
                BinStore::U16(v) => v[idx] as u32,
                BinStore::U32(v) => v[idx],
            };
            return Some(b); // dense entry at offset `feature` is that feature's bin
        }
        self.feature_bin(r, fs, fe)
    }

    /// The global bin of `feature` (whose global bin range is `[fs, fe)`) in row
    /// `r`, or `None` when that feature is missing for the row.
    #[inline]
    pub fn feature_bin(&self, r: usize, fs: usize, fe: usize) -> Option<u32> {
        let (s, e) = (self.row_ptr[r], self.row_ptr[r + 1]);
        match &self.store {
            BinStore::U16(v) => {
                for &b in &v[s..e] {
                    let b = b as usize;
                    if b >= fs && b < fe {
                        return Some(b as u32);
                    }
                }
            }
            BinStore::U32(v) => {
                for &b in &v[s..e] {
                    let b = b as usize;
                    if b >= fs && b < fe {
                        return Some(b as u32);
                    }
                }
            }
        }
        None
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
        assert_eq!(ghist.row_len(0), 2);
        assert_eq!(ghist.row_len(1), 2);
        // Row 1's feature-0 value (1.0) bins higher than row 0's (0.0).
        let b0 = ghist.cuts().bin_of(0, 0.0);
        let b1 = ghist.cuts().bin_of(0, 1.0);
        assert!(b1 > b0);
        // Small bin count -> u16 storage.
        assert!(matches!(ghist.bins(), Bins::U16(_)));
    }

    #[test]
    fn missing_entries_absent() {
        let data = DMatrix::from_dense(&[0.0, f32::NAN, 1.0, 2.0], 2, 2).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        // Row 0 has a missing feature 1 -> only one present entry.
        assert_eq!(ghist.row_len(0), 1);
        assert_eq!(ghist.row_len(1), 2);
    }

    #[test]
    fn feature_bin_lookup() {
        let data = DMatrix::from_dense(&[0.0, 10.0, 1.0, 20.0], 2, 2).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let (f0s, f0e) = cuts.feature_bins(0);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let b = ghist.feature_bin(0, f0s, f0e).unwrap();
        assert!((b as usize) >= f0s && (b as usize) < f0e);
        // A feature range with no entry for a fully-present row still resolves.
        assert!(ghist.feature_bin(1, f0s, f0e).is_some());
    }
}
