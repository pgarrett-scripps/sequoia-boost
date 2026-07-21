//! Exact per-prediction feature attributions via TreeSHAP.
//!
//! This implements the path-dependent, exact TreeSHAP algorithm of Lundberg et
//! al., *"Consistent Individualized Feature Attribution for Tree Ensembles"*,
//! matching XGBoost's `pred_contribs=True`. For a single tree the returned
//! per-feature values sum to `f_tree(x) - E[f_tree]`, where the expectation is
//! taken over the tree's cover (Hessian) distribution; the missing offset
//! `E[f_tree]` is folded into the bias term. Summed over the whole ensemble the
//! contributions therefore satisfy the exact-additivity property
//!
//! ```text
//! Σ_j contribs[j] + bias == margin(x)
//! ```
//!
//! where `bias == base_score + Σ_tree E[f_tree]` and `margin(x)` is the model's
//! raw margin ([`BoostedModel::predict_margin`]).
//!
//! The core recursion is the standard `O(T · L · D²)` algorithm (`T` trees, `L`
//! leaves, `D` maximum depth): each root-to-leaf traversal maintains the set of
//! unique features seen so far together with their `zero`/`one` fractions and a
//! permutation weight, using the EXTEND / UNWIND operations to add and remove
//! features as the path forks.

use crate::data::DMatrix;
use crate::error::Result;
use crate::learner::model::BoostedModel;
use crate::tree::RegTree;

/// One element of a decision path: a unique feature together with the fraction
/// of permutations in which it is "one" (present / in the coalition) and "zero"
/// (absent), plus the accumulated proportion of subset weights (`pweight`).
#[derive(Clone, Copy)]
struct PathElement {
    /// Feature index for this path element, or `-1` for the root placeholder.
    feature_index: i64,
    /// Fraction of paths that flow through the "zero" (absent) branch.
    zero_fraction: f64,
    /// Fraction of paths that flow through the "one" (present) branch.
    one_fraction: f64,
    /// Accumulated proportion of the subset weights for this element.
    pweight: f64,
}

/// Grow the decision path by adding a new feature split.
///
/// `path` holds the parent path (`unique_depth` elements before the call); the
/// new element is appended, and every existing element's `pweight` is updated to
/// account for one extra split in the coalition ordering.
fn extend_path(
    path: &mut Vec<PathElement>,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: i64,
) {
    let unique_depth = path.len(); // index the new element will occupy
    path.push(PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    });
    let denom = (unique_depth + 1) as f64;
    for i in (0..unique_depth).rev() {
        let pw_i = path[i].pweight;
        path[i + 1].pweight += one_fraction * pw_i * (i + 1) as f64 / denom;
        path[i].pweight = zero_fraction * pw_i * (unique_depth - i) as f64 / denom;
    }
}

/// Undo a previous [`extend_path`], removing the element at `path_index` and
/// restoring the `pweight`s of the remaining elements. Shrinks `path` by one.
fn unwind_path(path: &mut Vec<PathElement>, path_index: usize) {
    let unique_depth = path.len() - 1; // top index
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;
    let denom = (unique_depth + 1) as f64;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = path[i].pweight;
            path[i].pweight = next_one_portion * denom / ((i + 1) as f64 * one_fraction);
            next_one_portion =
                tmp - path[i].pweight * zero_fraction * (unique_depth - i) as f64 / denom;
        } else if zero_fraction != 0.0 {
            path[i].pweight = path[i].pweight * denom / (zero_fraction * (unique_depth - i) as f64);
        }
    }
    for i in path_index..unique_depth {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
    path.pop();
}

/// The total permutation weight that element `path_index` would contribute if it
/// were unwound, computed without mutating `path`.
fn unwound_path_sum(path: &[PathElement], path_index: usize) -> f64 {
    let unique_depth = path.len() - 1; // top index
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;
    let denom = (unique_depth + 1) as f64;
    let mut total = 0.0;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = next_one_portion * denom / ((i + 1) as f64 * one_fraction);
            total += tmp;
            next_one_portion =
                path[i].pweight - tmp * zero_fraction * (unique_depth - i) as f64 / denom;
        } else if zero_fraction != 0.0 {
            total += (path[i].pweight / zero_fraction) / ((unique_depth - i) as f64 / denom);
        }
    }
    total
}

/// Recursive TreeSHAP traversal for a single tree, accumulating per-feature
/// contributions into `phi`.
///
/// `path` is the parent decision path (owned, so it can be forked at each
/// internal node); `get` accesses the instance's feature values (`None` =
/// missing, routed by the node's default direction). This is the ordinary
/// (unconditioned) traversal used by [`BoostedModel::predict_contribs`]; it is a
/// thin wrapper over [`tree_shap_cond`] with `condition == 0`.
#[allow(clippy::too_many_arguments)]
fn tree_shap(
    tree: &RegTree,
    node_index: usize,
    path: Vec<PathElement>,
    parent_zero_fraction: f64,
    parent_one_fraction: f64,
    parent_feature_index: i64,
    get: &impl Fn(u32) -> Option<f32>,
    phi: &mut [f64],
) {
    tree_shap_cond(
        tree,
        node_index,
        path,
        parent_zero_fraction,
        parent_one_fraction,
        parent_feature_index,
        get,
        phi,
        0,
        -1,
        1.0,
    );
}

/// Recursive TreeSHAP traversal generalized to compute contributions
/// *conditioned* on a feature being present or absent — the core building block
/// for SHAP interaction values (Lundberg et al.).
///
/// `condition` selects the conditioning mode: `0` reproduces the ordinary
/// TreeSHAP contributions; `+1` fixes `condition_feature` to be **present** (in
/// the coalition); `-1` fixes it **absent**. `condition_fraction` is the running
/// weight carried down the tree by that conditioning (it starts at `1.0`). When
/// conditioning is active the `condition_feature` is never entered into the
/// decision path, so it receives no attribution of its own; the half-difference
/// of the `+1` and `-1` runs yields the interaction of `condition_feature` with
/// every other feature.
#[allow(clippy::too_many_arguments)]
fn tree_shap_cond(
    tree: &RegTree,
    node_index: usize,
    mut path: Vec<PathElement>,
    parent_zero_fraction: f64,
    parent_one_fraction: f64,
    parent_feature_index: i64,
    get: &impl Fn(u32) -> Option<f32>,
    phi: &mut [f64],
    condition: i32,
    condition_feature: i64,
    condition_fraction: f64,
) {
    // No weight flows down this branch under the conditioning: nothing to do.
    if condition_fraction == 0.0 {
        return;
    }

    // Extend the path with the parent split, unless we are conditioning on the
    // parent feature (in which case it is deliberately kept off the path).
    if condition == 0 || condition_feature != parent_feature_index {
        extend_path(
            &mut path,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        );
    }
    let node = tree.node(node_index);
    let unique_depth = path.len() - 1;

    if node.is_leaf() {
        let leaf = node.leaf_value as f64;
        for i in 1..=unique_depth {
            let w = unwound_path_sum(&path, i);
            let el = path[i];
            phi[el.feature_index as usize] +=
                w * (el.one_fraction - el.zero_fraction) * leaf * condition_fraction;
        }
        return;
    }

    // Route the instance to determine the "hot" (taken) and "cold" child.
    let split = node.split_feature;
    let go_left = match get(split) {
        Some(v) => v < node.split_cond,
        None => node.default_left,
    };
    let (hot, cold) = if go_left {
        (node.left as usize, node.right as usize)
    } else {
        (node.right as usize, node.left as usize)
    };

    // Cover-based child weights: hot/cold fraction = child_cover / node_cover.
    let node_cover = node.sum_hess as f64;
    let (hot_zero, cold_zero) = if node_cover > 0.0 {
        (
            tree.node(hot).sum_hess as f64 / node_cover,
            tree.node(cold).sum_hess as f64 / node_cover,
        )
    } else {
        (0.0, 0.0)
    };

    // If this feature is already on the path, unwind it first so it is not
    // double-counted, carrying its incoming fractions forward.
    let split_i = split as i64;
    let mut incoming_zero = 1.0;
    let mut incoming_one = 1.0;
    let found = path[1..=unique_depth]
        .iter()
        .position(|e| e.feature_index == split_i)
        .map(|p| p + 1);
    if let Some(pi) = found {
        incoming_zero = path[pi].zero_fraction;
        incoming_one = path[pi].one_fraction;
        unwind_path(&mut path, pi);
    }

    // Split the conditioning weight between the two children. When we condition
    // the split feature present, all weight follows the hot (taken) branch; when
    // we condition it absent, each branch keeps only its cover fraction.
    let mut hot_condition_fraction = condition_fraction;
    let mut cold_condition_fraction = condition_fraction;
    if condition > 0 && split_i == condition_feature {
        cold_condition_fraction = 0.0;
    } else if condition < 0 && split_i == condition_feature {
        hot_condition_fraction *= hot_zero;
        cold_condition_fraction *= cold_zero;
    }

    tree_shap_cond(
        tree,
        hot,
        path.clone(),
        hot_zero * incoming_zero,
        incoming_one,
        split_i,
        get,
        phi,
        condition,
        condition_feature,
        hot_condition_fraction,
    );
    tree_shap_cond(
        tree,
        cold,
        path,
        cold_zero * incoming_zero,
        0.0,
        split_i,
        get,
        phi,
        condition,
        condition_feature,
        cold_condition_fraction,
    );
}

/// Cover-weighted mean prediction of the subtree rooted at `node_index` — the
/// tree's expected output `E[f_tree]` when evaluated at the root. This is the
/// offset TreeSHAP folds into the bias term.
fn node_mean_value(tree: &RegTree, node_index: usize) -> f64 {
    let node = tree.node(node_index);
    if node.is_leaf() {
        return node.leaf_value as f64;
    }
    let cover = node.sum_hess as f64;
    if cover <= 0.0 {
        return 0.0;
    }
    let l = node.left as usize;
    let r = node.right as usize;
    let lc = tree.node(l).sum_hess as f64;
    let rc = tree.node(r).sum_hess as f64;
    (lc * node_mean_value(tree, l) + rc * node_mean_value(tree, r)) / cover
}

impl BoostedModel {
    /// Exact TreeSHAP feature contributions, matching XGBoost `pred_contribs=True`.
    ///
    /// For a single-output model the result is row-major with shape
    /// `n_rows × (n_features + 1)`: within each row, columns `0..n_features` are
    /// the per-feature contributions and the final column is the bias
    /// (`base_score` plus each tree's expected value).
    ///
    /// For a multiclass model (`n_outputs > 1`) the layout is
    /// `n_rows × n_outputs × (n_features + 1)`, row-major: the contributions for
    /// row `r`, output `c`, feature `j` live at
    /// `((r * n_outputs + c) * (n_features + 1)) + j`, with the bias at column
    /// `n_features`. Tree `t` contributes to output `t % n_outputs`.
    ///
    /// The key guarantee is exact additivity: for every row (and output) the sum
    /// of the `n_features + 1` values equals the raw margin from
    /// [`BoostedModel::predict_margin`].
    pub fn predict_contribs(&self, data: &DMatrix) -> Result<Vec<f32>> {
        let n = data.n_rows();
        let k = self.n_outputs();
        let nf = self.n_features();
        let width = nf + 1;
        let trees = self.trees();

        // Each tree's root mean value is instance-independent; compute once.
        let tree_means: Vec<f64> = trees.iter().map(|t| node_mean_value(t, 0)).collect();

        let base = self.base_score() as f64;
        let mut out = vec![0f32; n * k * width];
        let mut acc = vec![0f64; k * width]; // reused per row

        for row in 0..n {
            for a in acc.iter_mut() {
                *a = 0.0;
            }
            // Seed every output's bias with the base score.
            for c in 0..k {
                acc[c * width + nf] += base;
            }
            let get = |f: u32| data.get(row, f as usize);
            for (ti, tree) in trees.iter().enumerate() {
                let cls = ti % k;
                let off = cls * width;
                // Feature attributions.
                tree_shap(
                    tree,
                    0,
                    Vec::new(),
                    1.0,
                    1.0,
                    -1,
                    &get,
                    &mut acc[off..off + nf],
                );
                // Tree expected value folds into the bias column.
                acc[off + nf] += tree_means[ti];
            }
            let row_off = row * k * width;
            for (i, &v) in acc.iter().enumerate() {
                out[row_off + i] = v as f32;
            }
        }
        Ok(out)
    }

    /// Exact TreeSHAP interaction values, matching XGBoost `pred_interactions=True`.
    ///
    /// For a single-output model the result is row-major with per-row shape
    /// `(n_features + 1) × (n_features + 1)`. Within a row's matrix `M`:
    ///
    /// * the off-diagonal entry `M[i][j]` (`i, j < n_features`) is the SHAP
    ///   interaction between features `i` and `j` (symmetric: `M[i][j] ==
    ///   M[j][i]`), split evenly between the two cells;
    /// * the diagonal entry `M[i][i]` is feature `i`'s *main* effect, set so that
    ///   the row sums to feature `i`'s full SHAP value (its
    ///   [`BoostedModel::predict_contribs`] contribution);
    /// * the final row/column (index `n_features`) carry the bias: `M[nf][nf]`
    ///   holds each tree's expected value `Σ E[f_tree]`, and the remaining bias
    ///   cells are zero.
    ///
    /// Consequently the whole matrix sums to `margin(x) − base_score` — the raw
    /// margin from [`BoostedModel::predict_margin`] minus the global base score,
    /// which (unlike [`BoostedModel::predict_contribs`]) is *not* folded into the
    /// bias cell here.
    ///
    /// For a multiclass model (`n_outputs > 1`) the layout is
    /// `n_rows × n_outputs × (n_features + 1)^2`, row-major: the matrix for row
    /// `r`, output `c` occupies the `(n_features + 1)^2` values starting at
    /// `(r * n_outputs + c) * (n_features + 1)^2`. Tree `t` contributes to output
    /// `t % n_outputs`.
    #[allow(clippy::needless_range_loop)]
    pub fn predict_interactions(&self, data: &DMatrix) -> Result<Vec<f32>> {
        let n = data.n_rows();
        let k = self.n_outputs();
        let nf = self.n_features();
        let width = nf + 1;
        let mwidth = width * width;
        let trees = self.trees();

        // Each tree's root mean value is instance-independent; compute once.
        let tree_means: Vec<f64> = trees.iter().map(|t| node_mean_value(t, 0)).collect();

        let mut out = vec![0f32; n * k * mwidth];

        // Per-(row) scratch, reused across rows.
        let mut diag = vec![0f64; k * width]; // unconditioned contributions
        let mut on = vec![0f64; k * width]; // condition = +1 (feature present)
        let mut off = vec![0f64; k * width]; // condition = -1 (feature absent)
        let mut mat = vec![0f64; k * mwidth]; // full interaction matrices

        for row in 0..n {
            let get = |f: u32| data.get(row, f as usize);
            for v in diag.iter_mut() {
                *v = 0.0;
            }
            for v in mat.iter_mut() {
                *v = 0.0;
            }

            // 1. Unconditioned contributions (the diagonal / main effects). The
            //    bias cell carries each tree's expected value only (no base
            //    score), so the full matrix sums to `margin − base_score`.
            for (ti, tree) in trees.iter().enumerate() {
                let cls = ti % k;
                let base = cls * width;
                tree_shap(
                    tree,
                    0,
                    Vec::new(),
                    1.0,
                    1.0,
                    -1,
                    &get,
                    &mut diag[base..base + nf],
                );
                diag[base + nf] += tree_means[ti];
            }
            for c in 0..k {
                let mbase = c * mwidth;
                let dbase = c * width;
                for j in 0..width {
                    mat[mbase + j * width + j] = diag[dbase + j];
                }
            }

            // 2. Interaction terms: for each feature `j`, the half-difference of
            //    the present/absent conditioned contributions gives the
            //    interaction with every other feature; the diagonal is reduced so
            //    the row keeps summing to feature `j`'s SHAP value.
            for j in 0..nf {
                for v in on.iter_mut() {
                    *v = 0.0;
                }
                for v in off.iter_mut() {
                    *v = 0.0;
                }
                for (ti, tree) in trees.iter().enumerate() {
                    let cls = ti % k;
                    let base = cls * width;
                    tree_shap_cond(
                        tree,
                        0,
                        Vec::new(),
                        1.0,
                        1.0,
                        -1,
                        &get,
                        &mut on[base..base + nf],
                        1,
                        j as i64,
                        1.0,
                    );
                    tree_shap_cond(
                        tree,
                        0,
                        Vec::new(),
                        1.0,
                        1.0,
                        -1,
                        &get,
                        &mut off[base..base + nf],
                        -1,
                        j as i64,
                        1.0,
                    );
                }
                for c in 0..k {
                    let mbase = c * mwidth;
                    let dbase = c * width;
                    for kk in 0..width {
                        // The conditioned feature `j` never attributes to itself
                        // (on/off are zero there), so `kk == j` contributes 0.
                        let val = 0.5 * (on[dbase + kk] - off[dbase + kk]);
                        mat[mbase + j * width + kk] += val;
                        mat[mbase + j * width + j] -= val;
                    }
                }
            }

            let row_off = row * k * mwidth;
            for (i, &v) in mat.iter().enumerate() {
                out[row_off + i] = v as f32;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::learner::train;

    /// Build a small dense dataset with `nf` features; feature 0 and 1 carry
    /// signal, the rest are noise. Returns (data, n_rows).
    fn make_data(n: usize, nf: usize) -> DMatrix {
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            for j in 0..nf {
                // Deterministic pseudo-random values.
                let v = ((i * 31 + j * 17 + 7) % 97) as f32 / 97.0;
                x[i * nf + j] = v;
            }
            let f0 = x[i * nf];
            let f1 = x[i * nf + 1];
            y[i] = 2.0 * f0 - 1.5 * f1 + 0.3;
        }
        DMatrix::from_dense(&x, n, nf)
            .unwrap()
            .with_labels(&y)
            .unwrap()
    }

    #[test]
    fn additivity_single_output() {
        let n = 80;
        let nf = 5;
        let d = make_data(n, nf);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(4)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 30).unwrap();

        let contribs = model.predict_contribs(&d).unwrap();
        let margin = model.predict_margin(&d);
        let width = nf + 1;
        assert_eq!(contribs.len(), n * width);

        let mut max_err = 0f64;
        for row in 0..n {
            let s: f64 = contribs[row * width..row * width + width]
                .iter()
                .map(|&v| v as f64)
                .sum();
            let err = (s - margin[row] as f64).abs();
            max_err = max_err.max(err);
        }
        assert!(
            max_err < 1e-4,
            "max additivity error {max_err} exceeded 1e-4"
        );
    }

    #[test]
    fn additivity_multiclass() {
        let n = 90;
        let nf = 4;
        let k = 3;
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            for j in 0..nf {
                x[i * nf + j] = ((i * 13 + j * 29 + 3) % 101) as f32 / 101.0;
            }
            y[i] = (i % k) as f32;
        }
        let d = DMatrix::from_dense(&x, n, nf)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("multi:softprob")
            .num_class(k)
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 15).unwrap();
        assert_eq!(model.n_outputs(), k);

        let contribs = model.predict_contribs(&d).unwrap();
        let margin = model.predict_margin(&d);
        let width = nf + 1;
        assert_eq!(contribs.len(), n * k * width);

        let mut max_err = 0f64;
        for row in 0..n {
            for c in 0..k {
                let base = (row * k + c) * width;
                let s: f64 = contribs[base..base + width].iter().map(|&v| v as f64).sum();
                let err = (s - margin[row * k + c] as f64).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(
            max_err < 1e-4,
            "max multiclass additivity error {max_err} exceeded 1e-4"
        );
    }

    #[test]
    fn unused_feature_has_zero_contribution() {
        // Feature 3 is pure constant noise (never predictive) AND we verify no
        // split uses it; its contribution must be ~0 for every row.
        let n = 70;
        let nf = 4;
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            x[i * nf] = ((i * 7 + 1) % 53) as f32 / 53.0;
            x[i * nf + 1] = ((i * 11 + 2) % 53) as f32 / 53.0;
            x[i * nf + 2] = ((i * 5 + 3) % 53) as f32 / 53.0;
            x[i * nf + 3] = 0.5; // constant -> never a useful split
            y[i] = 3.0 * x[i * nf] - 2.0 * x[i * nf + 1];
        }
        let d = DMatrix::from_dense(&x, n, nf)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(4)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 25).unwrap();

        // Sanity: feature 3 is never used in any split.
        let used = model
            .trees()
            .iter()
            .flat_map(|t| t.nodes().iter())
            .any(|nd| !nd.is_leaf() && nd.split_feature == 3);
        assert!(!used, "feature 3 unexpectedly used in a split");

        let contribs = model.predict_contribs(&d).unwrap();
        let width = nf + 1;
        let mut max_abs = 0f32;
        for row in 0..n {
            max_abs = max_abs.max(contribs[row * width + 3].abs());
        }
        assert!(
            max_abs < 1e-6,
            "unused feature contribution {max_abs} not ~0"
        );
    }

    #[test]
    fn interactions_single_output() {
        let n = 80;
        let nf = 5;
        let d = make_data(n, nf);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(4)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 30).unwrap();

        let width = nf + 1;
        let mwidth = width * width;
        let inter = model.predict_interactions(&d).unwrap();
        assert_eq!(inter.len(), n * mwidth);

        let contribs = model.predict_contribs(&d).unwrap();
        let margin = model.predict_margin(&d);
        let base = model.base_score() as f64;

        let mut max_row_err = 0f64;
        let mut max_eff_err = 0f64;
        let mut max_sym_err = 0f64;
        for row in 0..n {
            let m = &inter[row * mwidth..row * mwidth + mwidth];
            // Row consistency: each feature row sums to its SHAP contribution.
            for i in 0..nf {
                let s: f64 = (0..width).map(|j| m[i * width + j] as f64).sum();
                let cval = contribs[row * width + i] as f64;
                max_row_err = max_row_err.max((s - cval).abs());
            }
            // Efficiency: the whole matrix sums to margin - base_score.
            let total: f64 = m.iter().map(|&v| v as f64).sum();
            let target = margin[row] as f64 - base;
            max_eff_err = max_eff_err.max((total - target).abs());
            // Symmetry.
            for i in 0..width {
                for j in 0..width {
                    let e = (m[i * width + j] as f64 - m[j * width + i] as f64).abs();
                    max_sym_err = max_sym_err.max(e);
                }
            }
        }
        assert!(max_row_err < 1e-4, "row-consistency error {max_row_err}");
        assert!(max_eff_err < 1e-4, "efficiency error {max_eff_err}");
        assert!(max_sym_err < 1e-5, "symmetry error {max_sym_err}");
    }

    #[test]
    fn interactions_multiclass() {
        let n = 90;
        let nf = 4;
        let k = 3;
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            for j in 0..nf {
                x[i * nf + j] = ((i * 13 + j * 29 + 3) % 101) as f32 / 101.0;
            }
            y[i] = (i % k) as f32;
        }
        let d = DMatrix::from_dense(&x, n, nf)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("multi:softprob")
            .num_class(k)
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 15).unwrap();
        assert_eq!(model.n_outputs(), k);

        let width = nf + 1;
        let mwidth = width * width;
        let inter = model.predict_interactions(&d).unwrap();
        assert_eq!(inter.len(), n * k * mwidth);

        let contribs = model.predict_contribs(&d).unwrap();
        let margin = model.predict_margin(&d);
        let base = model.base_score() as f64;

        let mut max_row_err = 0f64;
        let mut max_eff_err = 0f64;
        let mut max_sym_err = 0f64;
        for row in 0..n {
            for c in 0..k {
                let m = &inter[(row * k + c) * mwidth..(row * k + c) * mwidth + mwidth];
                let cbase = (row * k + c) * width;
                for i in 0..nf {
                    let s: f64 = (0..width).map(|j| m[i * width + j] as f64).sum();
                    let cval = contribs[cbase + i] as f64;
                    max_row_err = max_row_err.max((s - cval).abs());
                }
                let total: f64 = m.iter().map(|&v| v as f64).sum();
                let target = margin[row * k + c] as f64 - base;
                max_eff_err = max_eff_err.max((total - target).abs());
                for i in 0..width {
                    for j in 0..width {
                        let e = (m[i * width + j] as f64 - m[j * width + i] as f64).abs();
                        max_sym_err = max_sym_err.max(e);
                    }
                }
            }
        }
        assert!(max_row_err < 1e-4, "row-consistency error {max_row_err}");
        assert!(max_eff_err < 1e-4, "efficiency error {max_eff_err}");
        assert!(max_sym_err < 1e-5, "symmetry error {max_sym_err}");
    }
}
