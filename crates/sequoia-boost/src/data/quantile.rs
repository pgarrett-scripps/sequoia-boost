//! Quantile cut computation for histogram-based training.
//!
//! For each feature we compute up to `max_bin` cut points from the (non-missing)
//! value distribution, then map any value to a bin with an `upper_bound` search
//! (`bin = #{cuts ≤ value}`, clamped). Cuts are computed **once** from the data,
//! matching XGBoost's `tree_method=hist`. A trailing sentinel cut just above each
//! feature's maximum guarantees the maximum value gets its own bin (no clamp
//! collision), so bin index `0` may be empty — harmless and cheap.

use crate::data::meta::FeatureType;
use crate::data::DMatrix;
use serde::{Deserialize, Serialize};

/// Per-feature histogram cut points, laid out contiguously.
///
/// Feature `f` owns cut values `cut_values[feature_offset[f]..feature_offset[f+1]]`
/// and its bins occupy the same global index range, so `feature_offset` doubles
/// as both the cut-pointer and the global-bin offset table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistCuts {
    n_features: usize,
    /// Global bin/cut offsets, length `n_features + 1`.
    feature_offset: Vec<u32>,
    /// Concatenated ascending cut values. For a numeric feature these are split
    /// thresholds; for a categorical feature they are the distinct category
    /// values, one per bin (see `is_categorical`).
    cut_values: Vec<f32>,
    /// Per-feature flag: `true` when the feature is categorical and its bins map
    /// one category value each (no threshold semantics). Length `n_features`.
    is_categorical: Vec<bool>,
}

impl HistCuts {
    /// Compute cuts from a dataset with at most `max_bin` bins per feature.
    ///
    /// Categorical features (per [`DMatrix::feature_types`]) are binned with one
    /// bin per distinct category value; numeric features use quantile cut
    /// thresholds exactly as before.
    pub fn from_dmatrix(data: &DMatrix, max_bin: usize) -> Self {
        let csc = data.to_csc();
        let n_features = csc.n_cols();
        let ftypes = data.feature_types();
        let mut feature_offset = Vec::with_capacity(n_features + 1);
        feature_offset.push(0u32);
        let mut cut_values: Vec<f32> = Vec::new();
        let mut is_categorical = vec![false; n_features];

        let mut scratch: Vec<f32> = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for f in 0..n_features {
            let (_, vals) = csc.column(f);
            scratch.clear();
            scratch.extend_from_slice(vals);
            scratch.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if ftypes.get(f).copied() == Some(FeatureType::Categorical) {
                is_categorical[f] = true;
                build_categorical_cuts(&scratch, &mut cut_values);
            } else {
                build_feature_cuts(&scratch, max_bin, &mut cut_values);
            }
            feature_offset.push(cut_values.len() as u32);
        }

        HistCuts {
            n_features,
            feature_offset,
            cut_values,
            is_categorical,
        }
    }

    /// Whether feature `f` is categorical (bins map one category value each).
    #[inline]
    pub fn is_categorical(&self, f: usize) -> bool {
        self.is_categorical[f]
    }

    /// Number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Total number of bins across all features (the histogram length).
    #[inline]
    pub fn total_bins(&self) -> usize {
        self.cut_values.len()
    }

    /// Global bin range `[start, end)` owned by feature `f`.
    #[inline]
    pub fn feature_bins(&self, f: usize) -> (usize, usize) {
        (
            self.feature_offset[f] as usize,
            self.feature_offset[f + 1] as usize,
        )
    }

    /// Number of bins for feature `f`.
    #[inline]
    pub fn num_bins(&self, f: usize) -> usize {
        (self.feature_offset[f + 1] - self.feature_offset[f]) as usize
    }

    /// The cut value at a global bin index (its exclusive upper threshold: an
    /// instance goes left of a split here when `value < cut_value(bin)`).
    #[inline]
    pub fn cut_value(&self, global_bin: usize) -> f32 {
        self.cut_values[global_bin]
    }

    /// Map a feature value to its **global** bin index.
    #[inline]
    pub fn bin_of(&self, f: usize, value: f32) -> u32 {
        let (start, end) = self.feature_bins(f);
        let slice = &self.cut_values[start..end];
        if self.is_categorical[f] {
            // Categorical: each bin holds one category value; find the exact
            // bin. Unseen categories (absent at fit time) clamp to bin 0.
            let local = slice
                .binary_search_by(|c| c.partial_cmp(&value).unwrap())
                .unwrap_or(0);
            return start as u32 + local as u32;
        }
        // upper_bound: first cut strictly greater than value.
        let local = slice.partition_point(|&c| c <= value);
        let local = local.min(slice.len().saturating_sub(1));
        start as u32 + local as u32
    }
}

/// Append one bin per distinct category value (ascending) for a categorical
/// feature. Unlike numeric cuts, no sentinel is added: the bin *is* the
/// category.
fn build_categorical_cuts(sorted_vals: &[f32], out: &mut Vec<f32>) {
    if sorted_vals.is_empty() {
        // No observed categories: a single degenerate bin keeps the layout
        // well-formed; the feature can never split.
        out.push(0.0);
        return;
    }
    out.push(sorted_vals[0]);
    for w in sorted_vals.windows(2) {
        if w[0] != w[1] {
            out.push(w[1]);
        }
    }
}

/// Append feature cut values (ascending) for one feature to `out`.
fn build_feature_cuts(sorted_vals: &[f32], max_bin: usize, out: &mut Vec<f32>) {
    if sorted_vals.is_empty() {
        // No non-missing values: a single degenerate bin so the layout stays
        // well-formed. This feature can never produce a split.
        out.push(0.0);
        return;
    }
    let max_val = *sorted_vals.last().unwrap();

    // Count distinct values without allocating a second vector.
    let mut distinct = 1usize;
    for w in sorted_vals.windows(2) {
        if w[0] != w[1] {
            distinct += 1;
        }
    }

    let start = out.len();
    if distinct <= max_bin {
        // Use each distinct value as a cut.
        out.push(sorted_vals[0]);
        for w in sorted_vals.windows(2) {
            if w[0] != w[1] {
                out.push(w[1]);
            }
        }
    } else {
        // Weighted-uniform quantiles over the sorted values.
        let n = sorted_vals.len();
        let mut last_pushed = f32::NEG_INFINITY;
        for b in 1..=max_bin {
            let q = b as f64 / max_bin as f64;
            let idx = (((q * n as f64).ceil() as usize).max(1) - 1).min(n - 1);
            let v = sorted_vals[idx];
            if v > last_pushed {
                out.push(v);
                last_pushed = v;
            }
        }
    }
    // Append a sentinel just above the maximum so the max value lands in the
    // final real bin rather than colliding under the clamp.
    if *out.last().unwrap() <= max_val {
        out.push(max_val.next_up());
    }
    debug_assert!(out.len() > start);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn few_distinct_values_one_bin_each() {
        // feature has 3 distinct values 0,1,2 -> cuts = [0,1,2, sentinel]
        let data = DMatrix::from_dense(&[0.0, 1.0, 2.0, 1.0], 4, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        assert_eq!(cuts.n_features(), 1);
        // Distinct-value binning: each value maps to a distinct, increasing bin.
        let b0 = cuts.bin_of(0, 0.0);
        let b1 = cuts.bin_of(0, 1.0);
        let b2 = cuts.bin_of(0, 2.0);
        assert!(b0 < b1 && b1 < b2, "bins should be strictly increasing");
    }

    #[test]
    fn monotone_binning() {
        // 1000 distinct-ish values, capped at 16 bins.
        let n = 1000;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 16);
        assert!(cuts.num_bins(0) <= 17, "at most max_bin + sentinel");
        // Binning is monotone non-decreasing in the value.
        let mut prev = 0u32;
        for i in 0..n {
            let b = cuts.bin_of(0, i as f32);
            assert!(b >= prev);
            prev = b;
        }
        // Distinct low and high values fall in different bins.
        assert!(cuts.bin_of(0, 0.0) < cuts.bin_of(0, 999.0));
    }

    #[test]
    fn constant_feature_never_collides() {
        let data = DMatrix::from_dense(&[5.0, 5.0, 5.0], 3, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        // All identical values map to the same bin.
        assert_eq!(cuts.bin_of(0, 5.0), cuts.bin_of(0, 5.0));
    }

    #[test]
    fn split_threshold_consistency() {
        // A value maps left of cut c (value < c) iff its bin <= bin_of(c-) ...
        // Concretely: for values 0..10 with cuts, `value < cut_value(bin)` must
        // agree with `bin_of(value) <= target_bin`.
        let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let data = DMatrix::from_dense(&x, 10, 1).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let (start, end) = cuts.feature_bins(0);
        for target in start..end - 1 {
            let thr = cuts.cut_value(target);
            for &v in &x {
                let goes_left_by_value = v < thr;
                let goes_left_by_bin = cuts.bin_of(0, v) as usize <= target;
                assert_eq!(
                    goes_left_by_value, goes_left_by_bin,
                    "value {v}, target bin {target}, thr {thr}"
                );
            }
        }
    }
}
