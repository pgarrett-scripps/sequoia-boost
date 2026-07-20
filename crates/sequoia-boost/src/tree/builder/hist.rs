//! Histogram-based tree construction (XGBoost's `tree_method=hist`).
//!
//! Features are pre-binned once ([`GHistIndex`]); growing a node reduces to
//! scanning its per-bin gradient histogram. Sibling histograms are obtained by
//! subtraction (`sibling = parent − smaller_child`), so only the smaller child
//! is ever built directly. Supports both `depthwise` and `lossguide` growth.

use crate::config::{GrowPolicy, TrainingParams};
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::objective::GradPair;
use crate::tree::constraints::{
    calc_weight_bounded, child_bounds, gain_at_weight, satisfies, Bounds, MonotoneConstraints,
};
use crate::tree::gain::{GradStats, RegParams};
use crate::tree::hist::{zeroed, CpuBackend, Histogram, HistogramBackend};
use crate::tree::regtree::RegTree;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const K_RT_EPS: f64 = 1e-6;

/// The best split found for a node, in bin space.
#[derive(Debug, Clone)]
struct BestSplit {
    loss_chg: f64,
    feature: u32,
    /// Global bin index `i`: instances with bin ≤ `i` go left (numeric splits).
    split_bin: usize,
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
            split_bin: 0,
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

/// A node awaiting or undergoing expansion.
struct NodeEntry {
    nid: usize,
    depth: usize,
    rows: Vec<u32>,
    hist: Histogram,
    best: BestSplit,
    bounds: Bounds,
}

// Ordering for the loss-guided priority queue (max-heap on loss change).
impl PartialEq for NodeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.best.loss_chg == other.best.loss_chg
    }
}
impl Eq for NodeEntry {}
impl PartialOrd for NodeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NodeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.best.loss_chg.total_cmp(&other.best.loss_chg)
    }
}

/// Histogram tree builder.
pub struct HistTreeBuilder<'a> {
    params: &'a TrainingParams,
    reg: RegParams,
    cons: MonotoneConstraints,
    backend: CpuBackend,
}

impl<'a> HistTreeBuilder<'a> {
    /// Create a builder bound to a training configuration.
    pub fn new(params: &'a TrainingParams) -> Self {
        HistTreeBuilder {
            params,
            reg: RegParams::from_params(params),
            cons: MonotoneConstraints::from_params(&params.monotone_constraints),
            backend: CpuBackend,
        }
    }

    /// Grow one tree from the binned dataset.
    ///
    /// * `ghist` — the binned dataset (built once, reused across rounds).
    /// * `gpair` — per-row gradient/Hessian (length = dataset rows).
    /// * `row_subset` — sampled rows for this tree.
    /// * `feature_subset` — sampled features for this tree.
    pub fn build(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        row_subset: &[u32],
        feature_subset: &[u32],
    ) -> RegTree {
        let total_bins = ghist.total_bins();

        // Root statistics and histogram.
        let mut root_stats = GradStats::default();
        for &r in row_subset {
            let gp = gpair[r as usize];
            root_stats.add(GradStats::new(gp.grad as f64, gp.hess as f64));
        }
        let mut root_hist = zeroed(total_bins);
        self.backend.build(ghist, row_subset, gpair, &mut root_hist);

        let mut tree = RegTree::with_root(root_stats.hess as f32);
        let mut store = NodeStore {
            stats: vec![root_stats],
            bounds: vec![Bounds::default()],
        };

        let best = self.evaluate(
            ghist.cuts(),
            &root_hist,
            root_stats,
            feature_subset,
            Bounds::default(),
        );
        let root = NodeEntry {
            nid: 0,
            depth: 0,
            rows: row_subset.to_vec(),
            hist: root_hist,
            best,
            bounds: Bounds::default(),
        };

        match self.params.grow_policy {
            GrowPolicy::DepthWise => {
                self.grow_depthwise(&mut tree, &mut store, ghist, gpair, feature_subset, root)
            }
            GrowPolicy::LossGuide => {
                self.grow_lossguide(&mut tree, &mut store, ghist, gpair, feature_subset, root)
            }
        }

        // Finalize leaf weights (respecting each leaf's monotone bounds).
        #[allow(clippy::needless_range_loop)]
        for id in 0..tree.num_nodes() {
            if tree.node(id).is_leaf() {
                let w = calc_weight_bounded(store.stats[id], &self.reg, store.bounds[id]);
                tree.set_leaf_value(id, w as f32);
            }
        }
        tree
    }

    fn depth_limit(&self) -> usize {
        if self.params.max_depth == 0 {
            usize::MAX
        } else {
            self.params.max_depth
        }
    }

    fn grow_depthwise(
        &self,
        tree: &mut RegTree,
        store: &mut NodeStore,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        feature_subset: &[u32],
        root: NodeEntry,
    ) {
        let limit = self.depth_limit();
        let mut frontier = vec![root];
        let mut depth = 0;
        while depth < limit && !frontier.is_empty() {
            let mut next = Vec::new();
            for entry in frontier.drain(..) {
                if self.valid(&entry.best) {
                    let (l, r) = self.split(tree, store, ghist, gpair, feature_subset, entry);
                    next.push(l);
                    next.push(r);
                }
            }
            frontier = next;
            depth += 1;
        }
    }

    fn grow_lossguide(
        &self,
        tree: &mut RegTree,
        store: &mut NodeStore,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        feature_subset: &[u32],
        root: NodeEntry,
    ) {
        let limit = self.depth_limit();
        let max_leaves = if self.params.max_leaves == 0 {
            usize::MAX
        } else {
            self.params.max_leaves
        };
        let mut heap = BinaryHeap::new();
        heap.push(root);
        let mut n_leaves = 1usize;
        while let Some(entry) = heap.pop() {
            if n_leaves >= max_leaves {
                break;
            }
            if entry.depth >= limit || !self.valid(&entry.best) {
                continue; // permanent leaf
            }
            let (l, r) = self.split(tree, store, ghist, gpair, feature_subset, entry);
            n_leaves += 1; // one leaf became two
            heap.push(l);
            heap.push(r);
        }
    }

    /// Whether a node's best split should be taken.
    fn valid(&self, best: &BestSplit) -> bool {
        best.found()
            && best.loss_chg > self.params.gamma
            && best.left.hess >= self.reg.min_child_weight
            && best.right.hess >= self.reg.min_child_weight
    }

    /// Split a node: expand the tree, partition rows, and build both child
    /// histograms (the larger via subtraction). Returns the two child entries.
    fn split(
        &self,
        tree: &mut RegTree,
        store: &mut NodeStore,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        feature_subset: &[u32],
        entry: NodeEntry,
    ) -> (NodeEntry, NodeEntry) {
        let cuts = ghist.cuts();
        let b = &entry.best;

        // Monotone child bounds derived from the (bounded) child weights.
        let dir = self.cons.dir(b.feature as usize);
        let (lb_bounds, rb_bounds) = child_bounds(entry.bounds, dir, b.w_left, b.w_right);

        let (fs, fe) = cuts.feature_bins(b.feature as usize);
        let (left_id, right_id) = if b.is_categorical {
            tree.expand_categorical(
                entry.nid,
                b.feature,
                &b.cat_left,
                b.default_left,
                b.w_left as f32,
                b.left.hess as f32,
                b.w_right as f32,
                b.right.hess as f32,
            )
        } else {
            let threshold = cuts.cut_value(b.split_bin);
            tree.expand(
                entry.nid,
                b.feature,
                threshold,
                b.default_left,
                b.w_left as f32,
                b.left.hess as f32,
                b.w_right as f32,
                b.right.hess as f32,
            )
        };
        tree.set_split_gain(entry.nid, b.loss_chg as f32);
        debug_assert_eq!(left_id, store.stats.len());
        store.push(b.left, lb_bounds);
        store.push(b.right, rb_bounds);

        // Partition rows using the binned index.
        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();
        for &r in &entry.rows {
            let go_left = match row_feature_bin(ghist, r as usize, fs, fe) {
                Some(bin) => {
                    if b.is_categorical {
                        let cv = cuts.cut_value(bin as usize) as u32;
                        b.cat_left.contains(&cv)
                    } else {
                        (bin as usize) <= b.split_bin
                    }
                }
                None => b.default_left,
            };
            if go_left {
                left_rows.push(r);
            } else {
                right_rows.push(r);
            }
        }

        let total_bins = entry.hist.len();
        // Build the smaller child directly; derive the sibling by subtraction.
        let (left_hist, right_hist) = if left_rows.len() <= right_rows.len() {
            let mut lh = zeroed(total_bins);
            self.backend.build(ghist, &left_rows, gpair, &mut lh);
            let mut rh = zeroed(total_bins);
            self.backend.subtract(&entry.hist, &lh, &mut rh);
            (lh, rh)
        } else {
            let mut rh = zeroed(total_bins);
            self.backend.build(ghist, &right_rows, gpair, &mut rh);
            let mut lh = zeroed(total_bins);
            self.backend.subtract(&entry.hist, &rh, &mut lh);
            (lh, rh)
        };

        let left_best = self.evaluate(cuts, &left_hist, b.left, feature_subset, lb_bounds);
        let right_best = self.evaluate(cuts, &right_hist, b.right, feature_subset, rb_bounds);

        let left = NodeEntry {
            nid: left_id,
            depth: entry.depth + 1,
            rows: left_rows,
            hist: left_hist,
            best: left_best,
            bounds: lb_bounds,
        };
        let right = NodeEntry {
            nid: right_id,
            depth: entry.depth + 1,
            rows: right_rows,
            hist: right_hist,
            best: right_best,
            bounds: rb_bounds,
        };
        (left, right)
    }

    /// Find the best split for a node from its histogram, scanning each sampled
    /// feature's bin range and trying both missing-value directions. Split gain
    /// is evaluated at bounded child weights so monotone constraints are honored
    /// (with no constraints the bounds are infinite and this is the standard
    /// closed-form gain).
    fn evaluate(
        &self,
        cuts: &HistCuts,
        hist: &[GradStats],
        total: GradStats,
        feature_subset: &[u32],
        bounds: Bounds,
    ) -> BestSplit {
        let mut best = BestSplit::none();
        let mcw = self.reg.min_child_weight;
        let parent_w = calc_weight_bounded(total, &self.reg, bounds);
        let parent_gain = gain_at_weight(total, &self.reg, parent_w);

        for &f in feature_subset {
            let (fs, fe) = cuts.feature_bins(f as usize);
            if fe <= fs + 1 {
                continue; // degenerate feature, no interior boundary
            }
            let dir = self.cons.dir(f as usize);

            if cuts.is_categorical(f as usize) {
                self.evaluate_categorical(
                    &mut best,
                    cuts,
                    hist,
                    total,
                    parent_gain,
                    bounds,
                    dir,
                    f,
                    fs,
                    fe,
                );
                continue;
            }

            // Present statistics for this feature (sum over its bins).
            let mut present = GradStats::default();
            for &h in &hist[fs..fe] {
                present.add(h);
            }
            let missing = total.sub(present);

            let mut acc = GradStats::default();
            #[allow(clippy::needless_range_loop)]
            for i in fs..fe {
                acc.add(hist[i]);
                if i + 1 >= fe {
                    break; // no right side beyond the last bin
                }

                // Direction A: missing -> right. Left = present-so-far.
                let la = acc;
                let ra = total.sub(acc);
                if la.hess >= mcw && ra.hess >= mcw {
                    self.consider(&mut best, la, ra, parent_gain, bounds, dir, f, i, false);
                }

                // Direction B: missing -> left. Left = present-so-far + missing.
                let mut lb = acc;
                lb.add(missing);
                let rb = present.sub(acc);
                if lb.hess >= mcw && rb.hess >= mcw {
                    self.consider(&mut best, lb, rb, parent_gain, bounds, dir, f, i, true);
                }
            }
        }
        best
    }

    /// Evaluate one candidate split (bounded weights, monotone check) and update
    /// `best` if it improves.
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
        feature: u32,
        split_bin: usize,
        default_left: bool,
    ) {
        let wl = calc_weight_bounded(left, &self.reg, bounds);
        let wr = calc_weight_bounded(right, &self.reg, bounds);
        if !satisfies(dir, wl, wr) {
            return; // monotone constraint violated
        }
        let g = gain_at_weight(left, &self.reg, wl) + gain_at_weight(right, &self.reg, wr)
            - parent_gain;
        if g > best.loss_chg + K_RT_EPS {
            *best = BestSplit {
                loss_chg: g,
                feature,
                split_bin,
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

    /// Find the best categorical (set-membership) split for one feature.
    ///
    /// Follows XGBoost's sorted-partition strategy: rank the feature's category
    /// bins by their gradient/Hessian ratio, then sweep prefix partitions of that
    /// order. This yields the optimal two-set partition under the standard result
    /// that sorting by the score ratio makes the best subset contiguous. The
    /// prefix categories form the "left" set; every other (and missing) category
    /// goes right.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_categorical(
        &self,
        best: &mut BestSplit,
        cuts: &HistCuts,
        hist: &[GradStats],
        total: GradStats,
        parent_gain: f64,
        bounds: Bounds,
        dir: i8,
        feature: u32,
        fs: usize,
        fe: usize,
    ) {
        let mcw = self.reg.min_child_weight;
        // Only non-empty category bins can move; sort them by grad/hess ratio.
        let mut order: Vec<usize> = (fs..fe).filter(|&i| hist[i].hess > 0.0).collect();
        if order.len() < 2 {
            return;
        }
        let ratio = |s: GradStats| s.grad / (s.hess + self.reg.lambda);
        order.sort_by(|&a, &b| ratio(hist[a]).total_cmp(&ratio(hist[b])));

        let mut left = GradStats::default();
        let mut cats_left: Vec<u32> = Vec::new();
        // Sweep prefixes, always leaving at least one category on the right.
        for &bin in &order[..order.len() - 1] {
            left.add(hist[bin]);
            cats_left.push(cuts.cut_value(bin) as u32);
            // `total` includes any missing mass, which stays on the right.
            let right = total.sub(left);
            if left.hess < mcw || right.hess < mcw {
                continue;
            }
            self.consider_cat(
                best,
                left,
                right,
                parent_gain,
                bounds,
                dir,
                feature,
                &cats_left,
            );
        }
    }

    /// Evaluate one categorical candidate (left = `cats_left`, rest = right) and
    /// update `best` if it improves. Mirrors [`Self::consider`] but records the
    /// category set instead of a bin threshold.
    #[allow(clippy::too_many_arguments)]
    fn consider_cat(
        &self,
        best: &mut BestSplit,
        left: GradStats,
        right: GradStats,
        parent_gain: f64,
        bounds: Bounds,
        dir: i8,
        feature: u32,
        cats_left: &[u32],
    ) {
        let wl = calc_weight_bounded(left, &self.reg, bounds);
        let wr = calc_weight_bounded(right, &self.reg, bounds);
        if !satisfies(dir, wl, wr) {
            return; // monotone constraint violated
        }
        let g = gain_at_weight(left, &self.reg, wl) + gain_at_weight(right, &self.reg, wr)
            - parent_gain;
        if g > best.loss_chg + K_RT_EPS {
            *best = BestSplit {
                loss_chg: g,
                feature,
                split_bin: 0,
                // Present categories not in the left set go right; route missing
                // right as well (XGBoost's default for categorical features).
                default_left: false,
                left,
                right,
                w_left: wl,
                w_right: wr,
                is_categorical: true,
                cat_left: cats_left.to_vec(),
            };
        }
    }
}

/// Per-node statistics and monotone bounds, indexed by node id.
struct NodeStore {
    stats: Vec<GradStats>,
    bounds: Vec<Bounds>,
}

impl NodeStore {
    #[inline]
    fn push(&mut self, stats: GradStats, bounds: Bounds) {
        self.stats.push(stats);
        self.bounds.push(bounds);
    }
}

/// Find the global bin of `feature` (range `[fs, fe)`) in row `r`, or `None`
/// when that feature is missing for the row.
#[inline]
fn row_feature_bin(ghist: &GHistIndex, r: usize, fs: usize, fe: usize) -> Option<u32> {
    for &bin in ghist.row_bins(r) {
        let b = bin as usize;
        if b >= fs && b < fe {
            return Some(bin);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::tree::builder::{all_features, all_rows};

    fn gp(g: f32, h: f32) -> GradPair {
        GradPair::new(g, h)
    }

    fn binned(data: &DMatrix, max_bin: usize) -> GHistIndex {
        let cuts = HistCuts::from_dmatrix(data, max_bin);
        GHistIndex::from_dmatrix(data, cuts)
    }

    #[test]
    fn splits_on_separating_feature() {
        let x = vec![0.0f32, 0.0, 1.0, 1.0];
        let data = DMatrix::from_dense(&x, 4, 1).unwrap();
        let ghist = binned(&data, 256);
        let gpair = vec![gp(1.0, 1.0), gp(1.0, 1.0), gp(-1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .build()
            .unwrap();
        let tree =
            HistTreeBuilder::new(&params).build(&ghist, &gpair, &all_rows(4), &all_features(1));
        assert_eq!(tree.num_nodes(), 3);
        assert!((tree.predict_row(&data, 0) - (-1.0)).abs() < 1e-6);
        assert!((tree.predict_row(&data, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn no_split_below_gamma() {
        let x = vec![0.0f32, 1.0];
        let data = DMatrix::from_dense(&x, 2, 1).unwrap();
        let ghist = binned(&data, 256);
        let gpair = vec![gp(1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(3)
            .gamma(1e9)
            .build()
            .unwrap();
        let tree =
            HistTreeBuilder::new(&params).build(&ghist, &gpair, &all_rows(2), &all_features(1));
        assert_eq!(tree.num_nodes(), 1);
    }

    #[test]
    fn lossguide_respects_max_leaves() {
        // Enough structure that greedy growth would exceed the leaf cap.
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y = Vec::new();
        for i in 0..n {
            y.push(gp(if i % 2 == 0 { 1.0 } else { -1.0 }, 1.0));
        }
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let ghist = binned(&data, 256);
        let params = TrainingParams::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .max_leaves(4)
            .max_depth(0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(0.0)
            .build()
            .unwrap();
        let tree = HistTreeBuilder::new(&params).build(&ghist, &y, &all_rows(n), &all_features(1));
        assert!(tree.num_leaves() <= 4, "got {} leaves", tree.num_leaves());
    }

    #[test]
    fn monotone_increasing_is_enforced() {
        use crate::config::Monotone;
        // Data where the *unconstrained* fit would be non-monotone: a V shape.
        // y dips in the middle, so an unconstrained tree would go down then up.
        let n = 60;
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for i in 0..n {
            let xi = i as f32 / n as f32;
            x.push(xi);
            // gradient sign: negative residual mid-range -> would push weights down
            let target = (xi - 0.5).abs(); // V shape, non-monotone
            gpair.push(gp(-(target - 0.25), 1.0)); // pseudo-residual around mean
        }
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let ghist = binned(&data, 256);
        let params = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(1.0)
            .monotone_constraints(vec![Monotone::Increasing])
            .build()
            .unwrap();
        let tree =
            HistTreeBuilder::new(&params).build(&ghist, &gpair, &all_rows(n), &all_features(1));

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
    }

    #[test]
    fn hist_matches_exact_on_small_problem() {
        use crate::tree::builder::{ExactTreeBuilder, SortedColumns};
        // Random-ish separable-ish data; hist with enough bins should match exact.
        let n = 60;
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for i in 0..n {
            let xi = (i as f32) * 0.1;
            x.push(xi);
            gpair.push(gp((xi - 3.0).sin(), 1.0));
        }
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let params = TrainingParams::builder()
            .max_depth(3)
            .lambda(1.0)
            .min_child_weight(1.0)
            .gamma(0.0)
            .build()
            .unwrap();

        let cols = SortedColumns::from_dmatrix(&data);
        let exact = ExactTreeBuilder::new(&params).build(
            &cols,
            &data,
            &gpair,
            &all_rows(n),
            &all_features(1),
        );

        // 256 bins over 60 distinct values -> each value its own bin -> exact match.
        let ghist = binned(&data, 256);
        let hist =
            HistTreeBuilder::new(&params).build(&ghist, &gpair, &all_rows(n), &all_features(1));

        for r in 0..n {
            let pe = exact.predict_row(&data, r);
            let ph = hist.predict_row(&data, r);
            assert!((pe - ph).abs() < 1e-5, "row {r}: exact {pe} vs hist {ph}");
        }
    }
}
