//! Exact greedy tree construction (XGBoost's `tree_method=exact`, ColMaker).
//!
//! For each node we scan every feature's value-sorted entries and evaluate every
//! candidate threshold, trying both missing-value directions (sparsity-aware).
//! This is the reference builder: not the fastest (the histogram builder in a
//! later phase is), but the easiest to verify against XGBoost. Growth is
//! level-wise (depth-wise): a whole level is scanned per feature pass.

use crate::config::TrainingParams;
use crate::data::{DMatrix, FeatureType};
use crate::objective::GradPair;
use crate::tree::constraints::{
    calc_weight_bounded, child_bounds, gain_at_weight, satisfies, Bounds, MonotoneConstraints,
};
use crate::tree::gain::{calc_gain, calc_weight, GradStats, RegParams};
use crate::tree::regtree::RegTree;
use crate::tree::sampler::ColumnSampler;
use std::collections::HashMap;

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
#[derive(Debug, Clone)]
struct BestSplit {
    loss_chg: f64,
    feature: u32,
    threshold: f32,
    default_left: bool,
    left: GradStats,
    right: GradStats,
    /// Bounded child weights (used to derive monotone child bounds).
    w_left: f64,
    w_right: f64,
    /// Whether this is a categorical (set-membership) split.
    is_categorical: bool,
    /// For a categorical split, the category values routed left.
    cat_left: Vec<u32>,
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
            w_left: 0.0,
            w_right: 0.0,
            is_categorical: false,
            cat_left: Vec::new(),
        }
    }

    #[inline]
    fn found(&self) -> bool {
        self.loss_chg > K_RT_EPS
    }
}

/// A split that was applied at a level, used to route rows into their children.
struct Split {
    nid: usize,
    feature: u32,
    threshold: f32,
    default_left: bool,
    is_categorical: bool,
    cat_left: Vec<u32>,
    left_id: usize,
    right_id: usize,
}

/// Exact greedy tree builder.
pub struct ExactTreeBuilder<'a> {
    params: &'a TrainingParams,
    reg: RegParams,
    cons: MonotoneConstraints,
}

impl<'a> ExactTreeBuilder<'a> {
    /// Create a builder bound to a training configuration.
    pub fn new(params: &'a TrainingParams) -> Self {
        ExactTreeBuilder {
            params,
            reg: RegParams::from_params(params),
            cons: MonotoneConstraints::from_params(&params.monotone_constraints),
        }
    }

    /// Grow a single tree.
    ///
    /// * `cols` — value-sorted column index over the *full* dataset.
    /// * `data` — the dataset (for routing rows after a split).
    /// * `gpair` — per-row gradient/Hessian (length = dataset rows).
    /// * `row_subset` — the sampled rows to train this tree on.
    /// * `sampler` — per-tree column sampler; the exact builder draws one subset
    ///   per level (shared across that level's nodes).
    pub fn build(
        &self,
        cols: &SortedColumns,
        data: &DMatrix,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
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
        // Per-node monotone weight bounds (default `±∞` when unconstrained).
        let mut node_bounds: Vec<Bounds> = vec![Bounds::default()];

        // With no monotone constraints, the cheap closed-form gain path is exact
        // and behaves exactly as before; the bounded path is used otherwise.
        let constrained = self.cons.is_active();
        let ftypes = data.feature_types();

        let depth_limit = if self.params.max_depth == 0 {
            usize::MAX
        } else {
            self.params.max_depth
        };

        let mut active: Vec<usize> = vec![0];
        let mut depth = 0;

        while depth < depth_limit && !active.is_empty() {
            // One column subset for the whole level (bylevel ∘ bynode).
            let feature_subset = sampler.sample();
            let k = active.len();
            // slot_of_node maps an active node id to its dense slot index.
            let mut slot_of_node = vec![usize::MAX; tree.num_nodes()];
            for (slot, &nid) in active.iter().enumerate() {
                slot_of_node[nid] = slot;
            }

            let mut best = vec![BestSplit::none(); k];

            // Parent structure score per active node (subtracted from the split
            // gain). Bounded when constraints are active, closed-form otherwise.
            let mut parent_gain = vec![0.0f64; k];
            for (slot, &nid) in active.iter().enumerate() {
                parent_gain[slot] = if constrained {
                    let w = calc_weight_bounded(node_stats[nid], &self.reg, node_bounds[nid]);
                    gain_at_weight(node_stats[nid], &self.reg, w)
                } else {
                    calc_gain(node_stats[nid], &self.reg)
                };
            }

            // Scratch buffers, reused per feature.
            let mut present_total = vec![GradStats::default(); k];
            let mut acc = vec![GradStats::default(); k];
            let mut last_val = vec![f32::NAN; k];
            let mut has = vec![false; k];

            for &f in &feature_subset {
                let (crows, cvals) = cols.column(f as usize);
                let dir = self.cons.dir(f as usize);

                // Categorical features use a set-membership split instead of a
                // numeric threshold.
                if ftypes[f as usize] == FeatureType::Categorical {
                    // Gather per-node, per-category statistics for this feature.
                    let mut cat_stats: Vec<HashMap<u32, GradStats>> = vec![HashMap::new(); k];
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
                        let gp = gpair[r];
                        cat_stats[slot]
                            .entry(val as u32)
                            .or_default()
                            .add(GradStats::new(gp.grad as f64, gp.hess as f64));
                    }
                    for (slot, &nid) in active.iter().enumerate() {
                        self.eval_categorical(
                            &mut best[slot],
                            node_stats[nid],
                            parent_gain[slot],
                            node_bounds[nid],
                            dir,
                            constrained,
                            f,
                            &cat_stats[slot],
                        );
                    }
                    continue;
                }

                // Pass 1: total present statistics per active node for feature f.
                present_total
                    .iter_mut()
                    .for_each(|s| *s = GradStats::default());
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
                            parent_gain[slot],
                            node_bounds[nid as usize],
                            dir,
                            constrained,
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
            let mut splits: Vec<Split> = Vec::new();

            for &nid in &active {
                let slot = slot_of_node[nid];
                let b = &best[slot];
                let valid = b.found()
                    && b.loss_chg > gamma
                    && b.left.hess >= self.reg.min_child_weight
                    && b.right.hess >= self.reg.min_child_weight;
                if !valid {
                    continue; // stays a leaf; value finalized below
                }

                // Monotone child bounds derived from the (bounded) child weights.
                let dir = self.cons.dir(b.feature as usize);
                let (lb_bounds, rb_bounds) =
                    child_bounds(node_bounds[nid], dir, b.w_left, b.w_right);

                let (lw, rw) = if constrained {
                    (b.w_left as f32, b.w_right as f32)
                } else {
                    (
                        calc_weight(b.left, &self.reg) as f32,
                        calc_weight(b.right, &self.reg) as f32,
                    )
                };
                let (left_id, right_id) = if b.is_categorical {
                    tree.expand_categorical(
                        nid,
                        b.feature,
                        &b.cat_left,
                        b.default_left,
                        lw,
                        b.left.hess as f32,
                        rw,
                        b.right.hess as f32,
                    )
                } else {
                    tree.expand(
                        nid,
                        b.feature,
                        b.threshold,
                        b.default_left,
                        lw,
                        b.left.hess as f32,
                        rw,
                        b.right.hess as f32,
                    )
                };
                tree.set_split_gain(nid, b.loss_chg as f32);
                debug_assert_eq!(left_id, node_stats.len());
                node_stats.push(b.left);
                node_stats.push(b.right);
                node_bounds.push(lb_bounds);
                node_bounds.push(rb_bounds);
                next_active.push(left_id);
                next_active.push(right_id);
                splits.push(Split {
                    nid,
                    feature: b.feature,
                    threshold: b.threshold,
                    default_left: b.default_left,
                    is_categorical: b.is_categorical,
                    cat_left: b.cat_left.clone(),
                    left_id,
                    right_id,
                });
            }

            // Route each sampled row into its child for the nodes that split.
            if !splits.is_empty() {
                // Map nid -> split info for O(1) routing.
                let mut split_of_node = vec![usize::MAX; tree.num_nodes()];
                for (idx, s) in splits.iter().enumerate() {
                    split_of_node[s.nid] = idx;
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
                    let s = &splits[si];
                    let go_left = match data.get(r, s.feature as usize) {
                        Some(v) => {
                            if s.is_categorical {
                                // Present categories in the left set go left;
                                // every other present category goes right.
                                s.cat_left.contains(&(v as u32))
                            } else {
                                v < s.threshold
                            }
                        }
                        None => s.default_left,
                    };
                    node_of_row[r] = if go_left {
                        s.left_id as i32
                    } else {
                        s.right_id as i32
                    };
                }
            }

            active = next_active;
            depth += 1;
        }

        // Finalize every leaf's weight (respecting each leaf's monotone bounds).
        #[allow(clippy::needless_range_loop)]
        for id in 0..tree.num_nodes() {
            if tree.node(id).is_leaf() {
                let w = calc_weight_bounded(node_stats[id], &self.reg, node_bounds[id]) as f32;
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
        parent_gain: f64,
        bounds: Bounds,
        dir: i8,
        constrained: bool,
    ) {
        let threshold = 0.5 * (prev_val + cur_val);
        let missing = total.sub(present);
        let mcw = self.reg.min_child_weight;

        // Direction A: missing values go right. Left = present-so-far.
        let la = left_present;
        let ra = total.sub(left_present);
        if la.hess >= mcw && ra.hess >= mcw {
            self.consider(
                best, la, ra, parent_gain, bounds, dir, constrained, feature, threshold, false,
            );
        }

        // Direction B: missing values go left. Left = present-so-far + missing.
        let mut lb = left_present;
        lb.add(missing);
        let rb = present.sub(left_present);
        if lb.hess >= mcw && rb.hess >= mcw {
            self.consider(
                best, lb, rb, parent_gain, bounds, dir, constrained, feature, threshold, true,
            );
        }
    }

    /// Structure-score gain for splitting `total` into `left`/`right` plus the
    /// bounded child weights, or `None` when a monotone constraint is violated.
    /// Unconstrained builds take the cheap closed-form path (weights unused).
    #[inline]
    fn eval_gain(
        &self,
        left: GradStats,
        right: GradStats,
        parent_gain: f64,
        bounds: Bounds,
        dir: i8,
        constrained: bool,
    ) -> Option<(f64, f64, f64)> {
        if constrained {
            let wl = calc_weight_bounded(left, &self.reg, bounds);
            let wr = calc_weight_bounded(right, &self.reg, bounds);
            if !satisfies(dir, wl, wr) {
                return None; // monotone constraint violated
            }
            let g = gain_at_weight(left, &self.reg, wl) + gain_at_weight(right, &self.reg, wr)
                - parent_gain;
            Some((g, wl, wr))
        } else {
            let g = calc_gain(left, &self.reg) + calc_gain(right, &self.reg) - parent_gain;
            Some((g, 0.0, 0.0))
        }
    }

    /// Evaluate one numeric candidate split and update `best` if it improves.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn consider(
        &self,
        best: &mut BestSplit,
        left: GradStats,
        right: GradStats,
        parent_gain: f64,
        bounds: Bounds,
        dir: i8,
        constrained: bool,
        feature: u32,
        threshold: f32,
        default_left: bool,
    ) {
        let Some((g, wl, wr)) = self.eval_gain(left, right, parent_gain, bounds, dir, constrained)
        else {
            return;
        };
        if g > best.loss_chg + K_RT_EPS {
            *best = BestSplit {
                loss_chg: g,
                feature,
                threshold,
                default_left,
                left,
                right,
                w_left: wl,
                w_right: wr,
                is_categorical: false,
                cat_left: Vec::new(),
            };
        }
    }

    /// Find the best categorical (set-membership) split for one feature at one
    /// node, given that node's per-category statistics.
    ///
    /// Follows XGBoost's sorted-partition strategy: rank categories by their
    /// gradient/Hessian ratio, then sweep prefix partitions of that order. The
    /// prefix categories form the "left" set; every other present category (and
    /// missing) goes right.
    #[allow(clippy::too_many_arguments)]
    fn eval_categorical(
        &self,
        best: &mut BestSplit,
        total: GradStats,
        parent_gain: f64,
        bounds: Bounds,
        dir: i8,
        constrained: bool,
        feature: u32,
        cat_stats: &HashMap<u32, GradStats>,
    ) {
        if cat_stats.len() < 2 {
            return; // no interior partition
        }
        let mcw = self.reg.min_child_weight;
        let ratio = |s: GradStats| s.grad / (s.hess + self.reg.lambda);
        let mut cats: Vec<(u32, GradStats)> = cat_stats.iter().map(|(&c, &s)| (c, s)).collect();
        cats.sort_by(|a, b| ratio(a.1).total_cmp(&ratio(b.1)));

        let mut left = GradStats::default();
        let mut cats_left: Vec<u32> = Vec::new();
        // Sweep prefixes, always leaving at least one category on the right.
        for &(cat, s) in &cats[..cats.len() - 1] {
            left.add(s);
            cats_left.push(cat);
            // `total` includes any missing mass, which stays on the right.
            let right = total.sub(left);
            if left.hess < mcw || right.hess < mcw {
                continue;
            }
            let Some((g, wl, wr)) =
                self.eval_gain(left, right, parent_gain, bounds, dir, constrained)
            else {
                continue;
            };
            if g > best.loss_chg + K_RT_EPS {
                *best = BestSplit {
                    loss_chg: g,
                    feature,
                    threshold: 0.0,
                    // Present categories not in the left set (and missing) go
                    // right, as XGBoost defaults for categorical features.
                    default_left: false,
                    left,
                    right,
                    w_left: wl,
                    w_right: wr,
                    is_categorical: true,
                    cat_left: cats_left.clone(),
                };
            }
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
        let tree = b.build(
            &cols,
            &data,
            &gpair,
            &all_rows(4),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );

        assert_eq!(
            tree.num_nodes(),
            3,
            "root should have split into two leaves"
        );
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
        let tree = b.build(
            &cols,
            &data,
            &gpair,
            &all_rows(2),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
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
        let tree = b.build(
            &cols,
            &data,
            &gpair,
            &all_rows(3),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
        assert_eq!(tree.num_nodes(), 3);
        // The missing row should be routed with the negative-gradient group
        // (right, positive weight). default_left should therefore be false.
        assert!(!tree.node(0).default_left);
        assert!(tree.predict_row(&data, 2) > 0.0);
    }

    #[test]
    fn monotone_increasing_is_enforced() {
        use crate::config::Monotone;
        // Data where the *unconstrained* fit would be non-monotone: a V shape.
        let n = 60;
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for i in 0..n {
            let xi = i as f32 / n as f32;
            x.push(xi);
            let target = (xi - 0.5).abs(); // V shape, non-monotone
            gpair.push(gp(-(target - 0.25), 1.0));
        }
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let cols = SortedColumns::from_dmatrix(&data);
        let params = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(1.0)
            .monotone_constraints(vec![Monotone::Increasing])
            .build()
            .unwrap();
        let tree = ExactTreeBuilder::new(&params).build(
            &cols,
            &data,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );

        // Predictions must be non-decreasing in x under the increasing constraint.
        let mut prev = f32::NEG_INFINITY;
        for (i, &xi) in x.iter().enumerate() {
            let p = tree.predict_row(&data, i);
            assert!(
                p >= prev - 1e-5,
                "monotonicity violated at x={xi}: {p} < {prev}"
            );
            prev = p;
        }

        // Sanity: the unconstrained fit on the same data is *not* monotone, so
        // the constraint is doing real work.
        let unconstrained = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(1.0)
            .build()
            .unwrap();
        let free = ExactTreeBuilder::new(&unconstrained).build(
            &cols,
            &data,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
        let mut any_decrease = false;
        let mut prev = f32::NEG_INFINITY;
        for i in 0..n {
            let p = free.predict_row(&data, i);
            if p < prev - 1e-5 {
                any_decrease = true;
            }
            prev = p;
        }
        assert!(any_decrease, "unconstrained fit should be non-monotone");
    }

    #[test]
    fn categorical_splits_on_non_ordinal_pattern() {
        use crate::data::FeatureType;
        // 4 categories with a NON-ordinal target: {0,2} vs {1,3}. A numeric
        // threshold cannot separate them; a set-membership split can.
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for _ in 0..10 {
            for c in 0u32..4 {
                x.push(c as f32);
                // Residual around 0.5: even cats want negative weight, odd positive.
                let g = if c % 2 == 0 { 0.5 } else { -0.5 };
                gpair.push(gp(g, 1.0));
            }
        }
        let n = x.len();
        let data = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();
        let cols = SortedColumns::from_dmatrix(&data);
        let params = TrainingParams::builder()
            .max_depth(1)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(1.0)
            .build()
            .unwrap();
        let tree = ExactTreeBuilder::new(&params).build(
            &cols,
            &data,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );

        assert_eq!(tree.num_nodes(), 3, "root should split");
        assert!(tree.node(0).is_categorical, "split should be categorical");

        // Prediction for a bare category value.
        let pred = |c: f32| tree.leaf_id_with(|_| Some(c));
        // Even categories share a leaf; odd categories share the other leaf.
        assert_eq!(pred(0.0), pred(2.0));
        assert_eq!(pred(1.0), pred(3.0));
        assert_ne!(pred(0.0), pred(1.0), "the two groups must be separated");

        // Even cats (positive grad) want negative weight; odd cats positive.
        let val = |c: f32| tree.node(pred(c)).leaf_value;
        assert!(val(0.0) < 0.0 && val(2.0) < 0.0);
        assert!(val(1.0) > 0.0 && val(3.0) > 0.0);
    }
}
