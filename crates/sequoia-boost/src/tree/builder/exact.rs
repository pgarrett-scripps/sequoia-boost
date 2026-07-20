//! Exact greedy tree construction (XGBoost's `tree_method=exact`, ColMaker).
//!
//! For each node we scan every feature's value-sorted entries and evaluate every
//! candidate threshold, trying both missing-value directions (sparsity-aware).
//! This is the reference builder: not the fastest (the histogram builder in a
//! later phase is), but the easiest to verify against XGBoost. Growth is
//! level-wise (depth-wise): a whole level is scanned per feature pass.

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::objective::GradPair;
use crate::tree::gain::{calc_weight, split_gain, GradStats, RegParams};
use crate::tree::regtree::RegTree;

/// Tiny epsilon guarding against accepting numerically-zero-gain splits, mirror
/// of XGBoost's `kRtEps`.
const K_RT_EPS: f64 = 1e-6;

/// Value-sorted column index over a [`DMatrix`], built once and reused across
/// boosting rounds. Within each column, `(row, value)` pairs are sorted by
/// ascending value; missing entries are omitted (sparsity-aware).
#[derive(Debug, Clone)]
pub struct SortedColumns {
    n_rows: usize,
    n_cols: usize,
    col_ptr: Vec<usize>,
    rows: Vec<u32>,
    vals: Vec<f32>,
}

impl SortedColumns {
    /// Build the value-sorted column index from a dataset.
    pub fn from_dmatrix(data: &DMatrix) -> Self {
        let csc = data.to_csc();
        let n_cols = csc.n_cols();
        let mut col_ptr = vec![0usize; n_cols + 1];
        #[allow(clippy::needless_range_loop)]
        for c in 0..n_cols {
            col_ptr[c + 1] = col_ptr[c] + csc.col_len(c);
        }
        let nnz = col_ptr[n_cols];
        let mut rows = vec![0u32; nnz];
        let mut vals = vec![0f32; nnz];
        #[allow(clippy::needless_range_loop)]
        for c in 0..n_cols {
            let (crows, cvals) = csc.column(c);
            // Sort this column's entries by ascending value (NaN cannot appear:
            // missing entries were excluded when building the CSC).
            let mut order: Vec<usize> = (0..crows.len()).collect();
            order.sort_by(|&a, &b| cvals[a].partial_cmp(&cvals[b]).unwrap());
            let base = col_ptr[c];
            for (k, &o) in order.iter().enumerate() {
                rows[base + k] = crows[o];
                vals[base + k] = cvals[o];
            }
        }
        SortedColumns {
            n_rows: csc.n_rows(),
            n_cols,
            col_ptr,
            rows,
            vals,
        }
    }

    /// Number of rows in the source matrix.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    #[inline]
    fn column(&self, f: usize) -> (&[u32], &[f32]) {
        let (s, e) = (self.col_ptr[f], self.col_ptr[f + 1]);
        (&self.rows[s..e], &self.vals[s..e])
    }
}

/// The best split found so far for one node.
#[derive(Debug, Clone, Copy)]
struct BestSplit {
    loss_chg: f64,
    feature: u32,
    threshold: f32,
    default_left: bool,
    left: GradStats,
    right: GradStats,
}

impl BestSplit {
    fn none() -> Self {
        BestSplit {
            loss_chg: 0.0,
            feature: 0,
            threshold: 0.0,
            default_left: true,
            left: GradStats::default(),
            right: GradStats::default(),
        }
    }

    /// Update if `loss_chg` improves on the current best.
    #[inline]
    fn consider(
        &mut self,
        loss_chg: f64,
        feature: u32,
        threshold: f32,
        default_left: bool,
        left: GradStats,
        right: GradStats,
    ) {
        if loss_chg > self.loss_chg + K_RT_EPS {
            self.loss_chg = loss_chg;
            self.feature = feature;
            self.threshold = threshold;
            self.default_left = default_left;
            self.left = left;
            self.right = right;
        }
    }

    #[inline]
    fn found(&self) -> bool {
        self.loss_chg > K_RT_EPS
    }
}

/// Exact greedy tree builder.
pub struct ExactTreeBuilder<'a> {
    params: &'a TrainingParams,
    reg: RegParams,
}

impl<'a> ExactTreeBuilder<'a> {
    /// Create a builder bound to a training configuration.
    pub fn new(params: &'a TrainingParams) -> Self {
        ExactTreeBuilder {
            params,
            reg: RegParams::from_params(params),
        }
    }

    /// Grow a single tree.
    ///
    /// * `cols` — value-sorted column index over the *full* dataset.
    /// * `data` — the dataset (for routing rows after a split).
    /// * `gpair` — per-row gradient/Hessian (length = dataset rows).
    /// * `row_subset` — the sampled rows to train this tree on.
    /// * `feature_subset` — the sampled features available to this tree.
    pub fn build(
        &self,
        cols: &SortedColumns,
        data: &DMatrix,
        gpair: &[GradPair],
        row_subset: &[u32],
        feature_subset: &[u32],
    ) -> RegTree {
        let n_rows = cols.n_rows();

        // node_of_row[r] = current node id for row r, or -1 if r is not sampled.
        let mut node_of_row = vec![-1i32; n_rows];
        let mut root = GradStats::default();
        for &r in row_subset {
            node_of_row[r as usize] = 0;
            let gp = gpair[r as usize];
            root.add(GradStats::new(gp.grad as f64, gp.hess as f64));
        }

        let mut tree = RegTree::with_root(root.hess as f32);
        let mut node_stats: Vec<GradStats> = vec![root];

        let depth_limit = if self.params.max_depth == 0 {
            usize::MAX
        } else {
            self.params.max_depth
        };

        let mut active: Vec<usize> = vec![0];
        let mut depth = 0;

        while depth < depth_limit && !active.is_empty() {
            let k = active.len();
            // slot_of_node maps an active node id to its dense slot index.
            let mut slot_of_node = vec![usize::MAX; tree.num_nodes()];
            for (slot, &nid) in active.iter().enumerate() {
                slot_of_node[nid] = slot;
            }

            let mut best = vec![BestSplit::none(); k];

            // Scratch buffers, reused per feature.
            let mut present_total = vec![GradStats::default(); k];
            let mut acc = vec![GradStats::default(); k];
            let mut last_val = vec![f32::NAN; k];
            let mut has = vec![false; k];

            for &f in feature_subset {
                let (crows, cvals) = cols.column(f as usize);

                // Pass 1: total present statistics per active node for feature f.
                present_total.iter_mut().for_each(|s| *s = GradStats::default());
                for &rr in crows {
                    let r = rr as usize;
                    let nid = node_of_row[r];
                    if nid < 0 {
                        continue;
                    }
                    let slot = slot_of_node[nid as usize];
                    if slot == usize::MAX {
                        continue;
                    }
                    let gp = gpair[r];
                    present_total[slot].add(GradStats::new(gp.grad as f64, gp.hess as f64));
                }

                // Pass 2: enumerate thresholds, trying both missing directions.
                acc.iter_mut().for_each(|s| *s = GradStats::default());
                has.iter_mut().for_each(|h| *h = false);

                for (&rr, &val) in crows.iter().zip(cvals) {
                    let r = rr as usize;
                    let nid = node_of_row[r];
                    if nid < 0 {
                        continue;
                    }
                    let slot = slot_of_node[nid as usize];
                    if slot == usize::MAX {
                        continue;
                    }
                    if has[slot] && val != last_val[slot] {
                        self.eval_boundary(
                            &mut best[slot],
                            node_stats[nid as usize],
                            present_total[slot],
                            acc[slot],
                            last_val[slot],
                            val,
                            f,
                        );
                    }
                    let gp = gpair[r];
                    acc[slot].add(GradStats::new(gp.grad as f64, gp.hess as f64));
                    last_val[slot] = val;
                    has[slot] = true;
                }
            }

            // Apply splits and prepare the next level.
            let gamma = self.params.gamma;
            let mut next_active = Vec::new();
            // Record splits to route rows afterwards: (nid, feature, threshold,
            // default_left, left_id, right_id).
            let mut splits: Vec<(usize, u32, f32, bool, usize, usize)> = Vec::new();

            for &nid in &active {
                let slot = slot_of_node[nid];
                let b = best[slot];
                let valid = b.found()
                    && b.loss_chg > gamma
                    && b.left.hess >= self.reg.min_child_weight
                    && b.right.hess >= self.reg.min_child_weight;
                if !valid {
                    continue; // stays a leaf; value finalized below
                }
                let lw = calc_weight(b.left, &self.reg) as f32;
                let rw = calc_weight(b.right, &self.reg) as f32;
                let (left_id, right_id) = tree.expand(
                    nid,
                    b.feature,
                    b.threshold,
                    b.default_left,
                    lw,
                    b.left.hess as f32,
                    rw,
                    b.right.hess as f32,
                );
                tree.set_split_gain(nid, b.loss_chg as f32);
                debug_assert_eq!(left_id, node_stats.len());
                node_stats.push(b.left);
                node_stats.push(b.right);
                next_active.push(left_id);
                next_active.push(right_id);
                splits.push((nid, b.feature, b.threshold, b.default_left, left_id, right_id));
            }

            // Route each sampled row into its child for the nodes that split.
            if !splits.is_empty() {
                // Map nid -> split info for O(1) routing.
                let mut split_of_node = vec![usize::MAX; tree.num_nodes()];
                for (idx, s) in splits.iter().enumerate() {
                    split_of_node[s.0] = idx;
                }
                #[allow(clippy::needless_range_loop)]
                for r in 0..n_rows {
                    let nid = node_of_row[r];
                    if nid < 0 {
                        continue;
                    }
                    let si = split_of_node[nid as usize];
                    if si == usize::MAX {
                        continue;
                    }
                    let (_, feature, threshold, default_left, left_id, right_id) = splits[si];
                    let go_left = match data.get(r, feature as usize) {
                        Some(v) => v < threshold,
                        None => default_left,
                    };
                    node_of_row[r] = if go_left { left_id as i32 } else { right_id as i32 };
                }
            }

            active = next_active;
            depth += 1;
        }

        // Finalize every leaf's weight from its accumulated statistics.
        #[allow(clippy::needless_range_loop)]
        for id in 0..tree.num_nodes() {
            if tree.node(id).is_leaf() {
                let w = calc_weight(node_stats[id], &self.reg) as f32;
                tree.set_leaf_value(id, w);
            }
        }
        tree
    }

    /// Evaluate the split boundary between two consecutive distinct values,
    /// considering both missing-value directions, and update `best`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn eval_boundary(
        &self,
        best: &mut BestSplit,
        total: GradStats,
        present: GradStats,
        left_present: GradStats,
        prev_val: f32,
        cur_val: f32,
        feature: u32,
    ) {
        let threshold = 0.5 * (prev_val + cur_val);
        let missing = total.sub(present);
        let mcw = self.reg.min_child_weight;

        // Direction A: missing values go right. Left = present-so-far.
        let la = left_present;
        let ra = total.sub(left_present);
        if la.hess >= mcw && ra.hess >= mcw {
            let g = split_gain(la, ra, total, &self.reg);
            best.consider(g, feature, threshold, false, la, ra);
        }

        // Direction B: missing values go left. Left = present-so-far + missing.
        let mut lb = left_present;
        lb.add(missing);
        let rb = present.sub(left_present);
        if lb.hess >= mcw && rb.hess >= mcw {
            let g = split_gain(lb, rb, total, &self.reg);
            best.consider(g, feature, threshold, true, lb, rb);
        }
    }
}

/// Utility: the full row index `0..n_rows` as `u32` (no subsampling).
pub fn all_rows(n_rows: usize) -> Vec<u32> {
    (0..n_rows as u32).collect()
}

/// Utility: the full feature index `0..n_cols` as `u32` (no column sampling).
pub fn all_features(n_cols: usize) -> Vec<u32> {
    (0..n_cols as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingParams;

    fn gp(grad: f32, hess: f32) -> GradPair {
        GradPair::new(grad, hess)
    }

    /// A clean separable problem: feature 0 perfectly separates the sign of the
    /// gradient at threshold 0.5, so the root should split there.
    #[test]
    fn splits_on_separating_feature() {
        // 4 rows, 1 feature. values 0,0,1,1. gradients push low->+, high->-.
        let x = vec![0.0f32, 0.0, 1.0, 1.0];
        let data = DMatrix::from_dense(&x, 4, 1).unwrap();
        let cols = SortedColumns::from_dmatrix(&data);
        // squared-error-like gradients: left group wants negative weight, right positive
        let gpair = vec![gp(1.0, 1.0), gp(1.0, 1.0), gp(-1.0, 1.0), gp(-1.0, 1.0)];

        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .build()
            .unwrap();
        let b = ExactTreeBuilder::new(&params);
        let tree = b.build(&cols, &data, &gpair, &all_rows(4), &all_features(1));

        assert_eq!(tree.num_nodes(), 3, "root should have split into two leaves");
        let root = tree.node(0);
        assert_eq!(root.split_feature, 0);
        assert!((root.split_cond - 0.5).abs() < 1e-6);
        // left leaf: G=2,H=2 -> w=-1 ; right leaf: G=-2,H=2 -> w=+1
        assert!((tree.predict_row(&data, 0) - (-1.0)).abs() < 1e-6);
        assert!((tree.predict_row(&data, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn no_split_when_gain_below_gamma() {
        let x = vec![0.0f32, 1.0];
        let data = DMatrix::from_dense(&x, 2, 1).unwrap();
        let cols = SortedColumns::from_dmatrix(&data);
        let gpair = vec![gp(1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(3)
            .gamma(1e9) // impossibly high min split loss
            .build()
            .unwrap();
        let b = ExactTreeBuilder::new(&params);
        let tree = b.build(&cols, &data, &gpair, &all_rows(2), &all_features(1));
        assert_eq!(tree.num_nodes(), 1, "no split should be taken");
    }

    #[test]
    fn missing_values_pick_a_direction() {
        // 3 rows, feature 0 missing for row 2. Non-missing rows separate cleanly.
        let x = vec![0.0f32, 1.0, f32::NAN];
        let data = DMatrix::from_dense(&x, 3, 1).unwrap();
        let cols = SortedColumns::from_dmatrix(&data);
        // row2 (missing) shares the sign of the high group.
        let gpair = vec![gp(1.0, 1.0), gp(-1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .build()
            .unwrap();
        let b = ExactTreeBuilder::new(&params);
        let tree = b.build(&cols, &data, &gpair, &all_rows(3), &all_features(1));
        assert_eq!(tree.num_nodes(), 3);
        // The missing row should be routed with the negative-gradient group
        // (right, positive weight). default_left should therefore be false.
        assert!(!tree.node(0).default_left);
        assert!(tree.predict_row(&data, 2) > 0.0);
    }
}
