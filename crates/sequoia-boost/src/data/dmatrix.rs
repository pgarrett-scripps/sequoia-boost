//! The core dataset container: a feature matrix plus training metadata.

use crate::data::meta::{FeatureType, GroupInfo};
use crate::error::{Result, SequoiaError};

/// Returns `true` if `v` should be treated as missing given the sentinel
/// `missing`. NaN sentinels match any NaN; otherwise an exact bit-compatible
/// equality is used (mirroring XGBoost's semantics).
#[inline]
pub(crate) fn is_missing(v: f32, missing: f32) -> bool {
    if missing.is_nan() {
        v.is_nan()
    } else {
        v == missing
    }
}

/// Backing storage for the feature matrix.
#[derive(Debug, Clone)]
enum Storage {
    /// Row-major dense matrix of length `n_rows * n_cols`.
    Dense(Vec<f32>),
    /// Compressed sparse row: `indptr` has `n_rows + 1` entries.
    Csr {
        indptr: Vec<usize>,
        indices: Vec<u32>,
        values: Vec<f32>,
    },
}

/// A dataset: features in dense or sparse form, plus labels, weights, base
/// margins, ranking groups, and per-feature metadata.
///
/// Missing values are first-class: in dense storage any entry equal to the
/// [`DMatrix::missing`] sentinel (NaN by default) is treated as absent, and in
/// sparse storage implicit zeros are *present* zeros unless the sentinel is
/// `0.0`. Split finding learns a default direction for absent values, matching
/// XGBoost's sparsity-aware algorithm.
#[derive(Debug, Clone)]
pub struct DMatrix {
    n_rows: usize,
    n_cols: usize,
    storage: Storage,
    missing: f32,
    labels: Option<Vec<f32>>,
    weights: Option<Vec<f32>>,
    base_margin: Option<Vec<f32>>,
    group: Option<GroupInfo>,
    feature_types: Vec<FeatureType>,
    feature_names: Option<Vec<String>>,
}

/// A single materialized `(feature_index, value)` entry from a row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry {
    /// Column index of the feature.
    pub index: u32,
    /// Feature value (guaranteed non-missing when yielded by row iterators).
    pub value: f32,
}

impl DMatrix {
    /// Build a dense matrix from a row-major slice of length `n_rows * n_cols`.
    /// The missing sentinel defaults to NaN.
    pub fn from_dense(data: &[f32], n_rows: usize, n_cols: usize) -> Result<Self> {
        Self::from_dense_with_missing(data, n_rows, n_cols, f32::NAN)
    }

    /// Build a dense matrix with an explicit missing-value sentinel.
    pub fn from_dense_with_missing(
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
        missing: f32,
    ) -> Result<Self> {
        if n_rows == 0 || n_cols == 0 {
            return Err(SequoiaError::EmptyDataset("from_dense: zero rows or columns"));
        }
        if data.len() != n_rows * n_cols {
            return Err(SequoiaError::DimensionMismatch {
                what: "dense data length",
                expected: n_rows * n_cols,
                got: data.len(),
            });
        }
        Ok(DMatrix {
            n_rows,
            n_cols,
            storage: Storage::Dense(data.to_vec()),
            missing,
            labels: None,
            weights: None,
            base_margin: None,
            group: None,
            feature_types: vec![FeatureType::Numerical; n_cols],
            feature_names: None,
        })
    }

    /// Build a matrix from compressed-sparse-row arrays.
    ///
    /// `indptr` must have `n_rows + 1` entries; row `i` spans
    /// `indices[indptr[i]..indptr[i + 1]]`. Absent columns are treated as
    /// missing (sparsity-aware), so the sentinel is set to NaN.
    pub fn from_csr(
        indptr: Vec<usize>,
        indices: Vec<u32>,
        values: Vec<f32>,
        n_cols: usize,
    ) -> Result<Self> {
        if indptr.is_empty() {
            return Err(SequoiaError::EmptyDataset("from_csr: empty indptr"));
        }
        let n_rows = indptr.len() - 1;
        if n_rows == 0 || n_cols == 0 {
            return Err(SequoiaError::EmptyDataset("from_csr: zero rows or columns"));
        }
        if indices.len() != values.len() {
            return Err(SequoiaError::DimensionMismatch {
                what: "csr indices/values length",
                expected: indices.len(),
                got: values.len(),
            });
        }
        if *indptr.last().unwrap() != values.len() {
            return Err(SequoiaError::DimensionMismatch {
                what: "csr indptr terminal",
                expected: values.len(),
                got: *indptr.last().unwrap(),
            });
        }
        if let Some(&m) = indices.iter().max() {
            if (m as usize) >= n_cols {
                return Err(SequoiaError::FeatureOutOfBounds {
                    index: m as usize,
                    num_features: n_cols,
                });
            }
        }
        Ok(DMatrix {
            n_rows,
            n_cols,
            storage: Storage::Csr {
                indptr,
                indices,
                values,
            },
            missing: f32::NAN,
            labels: None,
            weights: None,
            base_margin: None,
            group: None,
            feature_types: vec![FeatureType::Numerical; n_cols],
            feature_names: None,
        })
    }

    /// Attach regression/classification labels (`len == n_rows`).
    pub fn with_labels(mut self, labels: &[f32]) -> Result<Self> {
        self.check_row_len("labels", labels.len())?;
        self.labels = Some(labels.to_vec());
        Ok(self)
    }

    /// Attach per-instance weights (`len == n_rows`).
    pub fn with_weights(mut self, weights: &[f32]) -> Result<Self> {
        self.check_row_len("weights", weights.len())?;
        self.weights = Some(weights.to_vec());
        Ok(self)
    }

    /// Attach a per-instance base margin (raw prediction offset, `len == n_rows`
    /// for single-output objectives).
    pub fn with_base_margin(mut self, base_margin: &[f32]) -> Result<Self> {
        self.base_margin = Some(base_margin.to_vec());
        Ok(self)
    }

    /// Attach ranking group information (sizes sum to `n_rows`).
    pub fn with_group_sizes(mut self, sizes: &[usize]) -> Result<Self> {
        let g = GroupInfo::from_sizes(sizes);
        if g.num_rows() != self.n_rows {
            return Err(SequoiaError::DimensionMismatch {
                what: "group sizes sum",
                expected: self.n_rows,
                got: g.num_rows(),
            });
        }
        self.group = Some(g);
        Ok(self)
    }

    /// Set the feature types (`len == n_cols`).
    pub fn with_feature_types(mut self, types: &[FeatureType]) -> Result<Self> {
        if types.len() != self.n_cols {
            return Err(SequoiaError::DimensionMismatch {
                what: "feature_types length",
                expected: self.n_cols,
                got: types.len(),
            });
        }
        self.feature_types = types.to_vec();
        Ok(self)
    }

    /// Set human-readable feature names (`len == n_cols`).
    pub fn with_feature_names(mut self, names: &[String]) -> Result<Self> {
        if names.len() != self.n_cols {
            return Err(SequoiaError::DimensionMismatch {
                what: "feature_names length",
                expected: self.n_cols,
                got: names.len(),
            });
        }
        self.feature_names = Some(names.to_vec());
        Ok(self)
    }

    fn check_row_len(&self, what: &'static str, got: usize) -> Result<()> {
        if got != self.n_rows {
            return Err(SequoiaError::DimensionMismatch {
                what,
                expected: self.n_rows,
                got,
            });
        }
        Ok(())
    }

    /// Number of rows (instances).
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of feature columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The missing-value sentinel.
    #[inline]
    pub fn missing(&self) -> f32 {
        self.missing
    }

    /// Labels, if attached.
    #[inline]
    pub fn labels(&self) -> Option<&[f32]> {
        self.labels.as_deref()
    }

    /// Weights, if attached.
    #[inline]
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Base margin, if attached.
    #[inline]
    pub fn base_margin(&self) -> Option<&[f32]> {
        self.base_margin.as_deref()
    }

    /// Ranking group info, if attached.
    #[inline]
    pub fn group(&self) -> Option<&GroupInfo> {
        self.group.as_ref()
    }

    /// Feature types (always populated; defaults to all numerical).
    #[inline]
    pub fn feature_types(&self) -> &[FeatureType] {
        &self.feature_types
    }

    /// Feature names, if attached.
    #[inline]
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    /// Fetch a single value, returning `None` when the entry is missing.
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        debug_assert!(row < self.n_rows && col < self.n_cols);
        match &self.storage {
            Storage::Dense(data) => {
                let v = data[row * self.n_cols + col];
                if is_missing(v, self.missing) {
                    None
                } else {
                    Some(v)
                }
            }
            Storage::Csr {
                indptr,
                indices,
                values,
            } => {
                let (s, e) = (indptr[row], indptr[row + 1]);
                // Rows are not assumed sorted by column; linear scan of the row.
                for k in s..e {
                    if indices[k] as usize == col {
                        let v = values[k];
                        return if is_missing(v, self.missing) {
                            None
                        } else {
                            Some(v)
                        };
                    }
                }
                None
            }
        }
    }

    /// Materialize a single row's non-missing `(index, value)` entries into
    /// `out`. Reuses the buffer to avoid per-row allocation in hot loops.
    pub fn row_into(&self, row: usize, out: &mut Vec<Entry>) {
        out.clear();
        match &self.storage {
            Storage::Dense(data) => {
                let base = row * self.n_cols;
                for c in 0..self.n_cols {
                    let v = data[base + c];
                    if !is_missing(v, self.missing) {
                        out.push(Entry {
                            index: c as u32,
                            value: v,
                        });
                    }
                }
            }
            Storage::Csr {
                indptr,
                indices,
                values,
            } => {
                let (s, e) = (indptr[row], indptr[row + 1]);
                for k in s..e {
                    let v = values[k];
                    if !is_missing(v, self.missing) {
                        out.push(Entry {
                            index: indices[k],
                            value: v,
                        });
                    }
                }
            }
        }
    }

    /// Build a compressed-sparse-**column** view for column-oriented split
    /// finding (used by the exact tree method). Each column lists its
    /// non-missing `(row, value)` pairs.
    pub fn to_csc(&self) -> CscView {
        // Count non-missing entries per column.
        let mut col_counts = vec![0usize; self.n_cols];
        self.for_each_entry(|_row, col, _v| col_counts[col as usize] += 1);

        let mut col_ptr = vec![0usize; self.n_cols + 1];
        #[allow(clippy::needless_range_loop)]
        for c in 0..self.n_cols {
            col_ptr[c + 1] = col_ptr[c] + col_counts[c];
        }
        let nnz = col_ptr[self.n_cols];
        let mut rows = vec![0u32; nnz];
        let mut vals = vec![0f32; nnz];
        let mut cursor = col_ptr.clone();
        self.for_each_entry(|row, col, v| {
            let c = col as usize;
            let pos = cursor[c];
            rows[pos] = row as u32;
            vals[pos] = v;
            cursor[c] = pos + 1;
        });
        CscView {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            col_ptr,
            rows,
            vals,
        }
    }

    /// Build a new matrix containing only `rows` (in the given order), carrying
    /// over labels, weights, base margin, and feature metadata. Used for
    /// cross-validation folds. Ranking group info is not carried over.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Self> {
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        indptr.push(0usize);
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        let mut buf: Vec<Entry> = Vec::new();
        for &r in rows {
            self.row_into(r, &mut buf);
            for e in &buf {
                indices.push(e.index);
                values.push(e.value);
            }
            indptr.push(values.len());
        }
        let mut out = DMatrix::from_csr(indptr, indices, values, self.n_cols)?;
        out.feature_types = self.feature_types.clone();
        out.feature_names = self.feature_names.clone();
        if let Some(l) = &self.labels {
            out.labels = Some(rows.iter().map(|&r| l[r]).collect());
        }
        if let Some(w) = &self.weights {
            out.weights = Some(rows.iter().map(|&r| w[r]).collect());
        }
        if let Some(bm) = &self.base_margin {
            out.base_margin = Some(rows.iter().map(|&r| bm[r]).collect());
        }
        Ok(out)
    }

    /// Visit every non-missing entry as `(row, col, value)`.
    fn for_each_entry(&self, mut f: impl FnMut(usize, u32, f32)) {
        match &self.storage {
            Storage::Dense(data) => {
                for r in 0..self.n_rows {
                    let base = r * self.n_cols;
                    for c in 0..self.n_cols {
                        let v = data[base + c];
                        if !is_missing(v, self.missing) {
                            f(r, c as u32, v);
                        }
                    }
                }
            }
            Storage::Csr {
                indptr,
                indices,
                values,
            } => {
                for r in 0..self.n_rows {
                    for k in indptr[r]..indptr[r + 1] {
                        let v = values[k];
                        if !is_missing(v, self.missing) {
                            f(r, indices[k], v);
                        }
                    }
                }
            }
        }
    }
}

/// A compressed-sparse-column view of a [`DMatrix`], built by
/// [`DMatrix::to_csc`]. Within each column the `(row, value)` pairs are stored
/// in row order; callers that need value-sorted order (e.g. the exact split
/// finder) sort per-column slices themselves.
#[derive(Debug, Clone)]
pub struct CscView {
    n_rows: usize,
    n_cols: usize,
    col_ptr: Vec<usize>,
    rows: Vec<u32>,
    vals: Vec<f32>,
}

impl CscView {
    /// Number of rows in the originating matrix.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The `(rows, values)` slices of non-missing entries for a column.
    #[inline]
    pub fn column(&self, col: usize) -> (&[u32], &[f32]) {
        let (s, e) = (self.col_ptr[col], self.col_ptr[col + 1]);
        (&self.rows[s..e], &self.vals[s..e])
    }

    /// Number of non-missing entries in a column.
    #[inline]
    pub fn col_len(&self, col: usize) -> usize {
        self.col_ptr[col + 1] - self.col_ptr[col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dense() -> DMatrix {
        // 3 rows x 2 cols, with a NaN missing in row 1 col 0.
        let data = vec![1.0, 2.0, f32::NAN, 5.0, 3.0, 6.0];
        DMatrix::from_dense(&data, 3, 2).unwrap()
    }

    #[test]
    fn dense_get_and_missing() {
        let d = sample_dense();
        assert_eq!(d.get(0, 0), Some(1.0));
        assert_eq!(d.get(1, 0), None); // missing
        assert_eq!(d.get(1, 1), Some(5.0));
    }

    #[test]
    fn row_into_skips_missing() {
        let d = sample_dense();
        let mut buf = Vec::new();
        d.row_into(1, &mut buf);
        assert_eq!(buf, vec![Entry { index: 1, value: 5.0 }]);
    }

    #[test]
    fn csc_matches_dense() {
        let d = sample_dense();
        let csc = d.to_csc();
        // Column 0 has rows {0, 2} (row 1 is missing).
        let (rows, vals) = csc.column(0);
        assert_eq!(rows, &[0, 2]);
        assert_eq!(vals, &[1.0, 3.0]);
        // Column 1 has all three rows.
        assert_eq!(csc.col_len(1), 3);
    }

    #[test]
    fn csr_roundtrip() {
        // Same logical matrix as sample_dense but sparse (row 1 col 0 absent).
        let indptr = vec![0, 2, 3, 5];
        let indices = vec![0, 1, 1, 0, 1];
        let values = vec![1.0, 2.0, 5.0, 3.0, 6.0];
        let d = DMatrix::from_csr(indptr, indices, values, 2).unwrap();
        assert_eq!(d.get(0, 0), Some(1.0));
        assert_eq!(d.get(1, 0), None);
        assert_eq!(d.get(2, 1), Some(6.0));
        let csc = d.to_csc();
        let (rows, vals) = csc.column(0);
        assert_eq!(rows, &[0, 2]);
        assert_eq!(vals, &[1.0, 3.0]);
    }

    #[test]
    fn label_length_checked() {
        let d = sample_dense();
        assert!(d.clone().with_labels(&[1.0, 2.0]).is_err());
        assert!(d.with_labels(&[1.0, 2.0, 3.0]).is_ok());
    }
}
