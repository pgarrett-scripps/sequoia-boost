//! XGBoost-format (JSON) model import and export.
//!
//! XGBoost serializes a booster as a nested JSON document:
//!
//! ```text
//! {"version": [..],
//!  "learner": {
//!    "gradient_booster": {
//!      "name": "gbtree",
//!      "model": {"trees": [ {..per-tree arrays..} ], "tree_info": [..],
//!                "gbtree_model_param": {..}}},
//!    "learner_model_param": {"base_score", "num_class", "num_feature"},
//!    "objective": {"name": ..}}}
//! ```
//!
//! Each tree is stored as a set of parallel, node-indexed arrays rather than a
//! nested structure: `left_children`, `right_children`, `split_indices`,
//! `split_conditions`, `default_left`, `base_weights`, `sum_hessian` and
//! `loss_changes`. A node `i` is a **leaf** when `left_children[i] == -1`; its
//! weight is carried in `split_conditions[i]` (and, redundantly,
//! `base_weights[i]`). Internal nodes route `x[split_indices[i]] <
//! split_conditions[i]`, sending missing values in the `default_left[i]`
//! direction — the exact semantics of [`crate::RegTree`].
//!
//! # Scope and caveats
//!
//! Import is **best-effort** and targets the common case: a `gbtree` booster
//! with a scalar or multiclass objective. Unsupported boosters (e.g. `gblinear`,
//! `dart`) yield a clear [`SequoiaError::ModelFormat`]. Categorical splits,
//! vector leaves (`size_leaf_vector > 1`) and non-numeric split types are not
//! interpreted; only numeric splits round-trip.
//!
//! ## `base_score`
//!
//! `sequoia-boost` stores `base_score` in **margin** space. XGBoost stores it in
//! whatever space its objective reports: raw margin for `reg:squarederror`, but
//! **probability** space for objectives with a link function (newer XGBoost
//! writes `0.5` for `binary:logistic`, not its logit). We reconcile this using
//! the objective's own transforms: on **import** the stored value is mapped to
//! margin space via the inverse link ([`Objective::prob_to_margin`]); on
//! **export** the margin is mapped back with the forward transform
//! ([`Objective::pred_transform`]). For multiclass objectives (and any objective
//! we cannot reconstruct) the value is passed through unchanged.

use crate::config::TrainingParams;
use crate::error::{Result, SequoiaError};
use crate::learner::BoostedModel;
use crate::objective::{create_objective, Objective};
use crate::tree::{Node, RegTree};
use serde_json::{json, Value};

/// Sentinel XGBoost writes for the parent of the root node (`kInvalidNodeId`).
const INVALID_NODE: i32 = i32::MAX;

/// Serialize a [`BoostedModel`] into XGBoost's JSON model schema.
///
/// The result is a pretty-printed JSON string equivalent to what
/// `xgboost.Booster.save_model("m.json")` produces for a `gbtree` model, and is
/// accepted by [`import_xgboost_json`] as well as upstream XGBoost. See the
/// [module docs](self) for the `base_score` space convention.
pub fn export_xgboost_json(model: &BoostedModel) -> Result<String> {
    let num_feature = model.n_features();
    let num_class = model.num_class();
    let objective = model.objective().to_string();
    let n_outputs = model.n_outputs();
    let n_trees = model.num_trees();

    let trees: Vec<Value> = model
        .trees()
        .iter()
        .enumerate()
        .map(|(id, t)| tree_to_json(id, t, num_feature))
        .collect();

    // `tree_info[t]` is the output group tree `t` contributes to. For scalar
    // objectives that is always 0; multiclass trees are laid out round-robin,
    // matching `BoostedModel`'s `t % n_outputs` convention.
    let tree_info: Vec<Value> = (0..n_trees).map(|t| json!((t % n_outputs) as i32)).collect();

    let base_score = link_export(model.base_score(), &objective, num_class);

    let value = json!({
        "version": [2, 0, 0],
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "gbtree_model_param": {
                        "num_parallel_tree": "1",
                        "num_trees": n_trees.to_string(),
                    },
                    "tree_info": tree_info,
                    "trees": trees,
                }
            },
            "learner_model_param": {
                "base_score": base_score.to_string(),
                "num_class": num_class.to_string(),
                "num_feature": num_feature.to_string(),
                "num_target": "1",
            },
            "objective": objective_to_json(&objective, num_class),
        }
    });

    Ok(serde_json::to_string_pretty(&value)?)
}

/// Parse an XGBoost JSON model document into a [`BoostedModel`].
///
/// Best-effort for `gbtree` boosters; other booster kinds produce a
/// [`SequoiaError::ModelFormat`]. See the [module docs](self) for details and
/// the `base_score` space convention.
pub fn import_xgboost_json(json: &str) -> Result<BoostedModel> {
    let root: Value = serde_json::from_str(json)?;
    let learner = field(&root, "learner")?;
    let booster = field(learner, "gradient_booster")?;

    let booster_name = booster
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("gbtree");
    if booster_name != "gbtree" {
        return Err(fmt_err(format!(
            "unsupported gradient_booster `{booster_name}`: only `gbtree` is supported"
        )));
    }

    let model = field(booster, "model")?;
    let lmp = field(learner, "learner_model_param")?;

    let num_feature = lmp
        .get("num_feature")
        .and_then(scalar_f64)
        .map(|v| v as usize)
        .ok_or_else(|| fmt_err("missing/invalid `num_feature`"))?;
    let num_class = lmp
        .get("num_class")
        .and_then(scalar_f64)
        .map(|v| v as usize)
        .unwrap_or(0);

    let objective = learner
        .get("objective")
        .and_then(|o| o.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("reg:squarederror")
        .to_string();

    let trees_json = model
        .get("trees")
        .and_then(Value::as_array)
        .ok_or_else(|| fmt_err("missing `model.trees` array"))?;
    let mut trees = Vec::with_capacity(trees_json.len());
    for (i, tj) in trees_json.iter().enumerate() {
        trees.push(tree_from_json(tj).map_err(|e| fmt_err(format!("tree {i}: {e}")))?);
    }

    let stored_base = lmp
        .get("base_score")
        .and_then(scalar_f64)
        .map(|v| v as f32)
        .ok_or_else(|| fmt_err("missing/invalid `base_score`"))?;
    let base_margin = link_import(stored_base, &objective, num_class);

    Ok(BoostedModel::from_parts(
        trees,
        base_margin,
        objective,
        num_class,
        num_feature,
    ))
}

// ---------------------------------------------------------------------------
// Tree (de)serialization
// ---------------------------------------------------------------------------

/// Encode one [`RegTree`] as XGBoost's node-indexed array bundle.
fn tree_to_json(id: usize, tree: &RegTree, num_feature: usize) -> Value {
    let nodes = tree.nodes();
    let n = nodes.len();

    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    let mut split_indices = Vec::with_capacity(n);
    let mut split_conditions = Vec::with_capacity(n);
    let mut default_left = Vec::with_capacity(n);
    let mut base_weights = Vec::with_capacity(n);
    let mut loss_changes = Vec::with_capacity(n);
    let mut sum_hessian = Vec::with_capacity(n);
    let mut split_type = Vec::with_capacity(n);
    let mut parents = vec![INVALID_NODE; n];

    for (i, node) in nodes.iter().enumerate() {
        if !node.is_leaf() {
            parents[node.left as usize] = i as i32;
            parents[node.right as usize] = i as i32;
        }
    }

    for node in nodes {
        left.push(node.left);
        right.push(node.right);
        sum_hessian.push(node.sum_hess);
        split_type.push(0); // 0 = numerical split
        if node.is_leaf() {
            // XGBoost carries the leaf weight in both arrays for leaves.
            split_indices.push(0u32);
            split_conditions.push(node.leaf_value);
            base_weights.push(node.leaf_value);
            default_left.push(1i32);
            loss_changes.push(0.0f32);
        } else {
            split_indices.push(node.split_feature);
            split_conditions.push(node.split_cond);
            base_weights.push(0.0f32);
            default_left.push(i32::from(node.default_left));
            loss_changes.push(node.split_gain);
        }
    }

    json!({
        "id": id,
        "tree_param": {
            "num_deleted": "0",
            "num_feature": num_feature.to_string(),
            "num_nodes": n.to_string(),
            "size_leaf_vector": "0",
        },
        "left_children": left,
        "right_children": right,
        "parents": parents,
        "split_indices": split_indices,
        "split_conditions": split_conditions,
        "default_left": default_left,
        "base_weights": base_weights,
        "loss_changes": loss_changes,
        "sum_hessian": sum_hessian,
        "split_type": split_type,
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
    })
}

/// Decode one XGBoost tree object into a [`RegTree`].
fn tree_from_json(tj: &Value) -> Result<RegTree> {
    let left = arr(tj, "left_children", scalar_f64).ok_or_else(|| fmt_err("missing `left_children`"))?;
    let n = left.len();
    let left: Vec<i32> = left.iter().map(|&v| v as i32).collect();

    let right: Vec<i32> = arr(tj, "right_children", scalar_f64)
        .ok_or_else(|| fmt_err("missing `right_children`"))?
        .iter()
        .map(|&v| v as i32)
        .collect();

    let split_indices = arr(tj, "split_indices", scalar_f64).unwrap_or_default();
    let split_conditions =
        arr(tj, "split_conditions", scalar_f64).ok_or_else(|| fmt_err("missing `split_conditions`"))?;
    let default_left = arr(tj, "default_left", scalar_f64).unwrap_or_default();
    let base_weights = arr(tj, "base_weights", scalar_f64).unwrap_or_default();
    let sum_hessian = arr(tj, "sum_hessian", scalar_f64).unwrap_or_default();
    let loss_changes = arr(tj, "loss_changes", scalar_f64).unwrap_or_default();

    let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let sum_hess = at(&sum_hessian, i) as f32;
        if left[i] == -1 {
            // Leaf: prefer split_conditions, fall back to base_weights.
            let leaf_value = split_conditions
                .get(i)
                .copied()
                .or_else(|| base_weights.get(i).copied())
                .unwrap_or(0.0) as f32;
            nodes.push(Node {
                split_feature: 0,
                split_cond: 0.0,
                default_left: true,
                left: -1,
                right: -1,
                leaf_value,
                sum_hess,
                split_gain: 0.0,
                is_categorical: false,
                cat_begin: 0,
                cat_end: 0,
            });
        } else {
            nodes.push(Node {
                split_feature: at(&split_indices, i) as u32,
                split_cond: at(&split_conditions, i) as f32,
                default_left: at(&default_left, i) != 0.0,
                left: left[i],
                right: right[i],
                leaf_value: 0.0,
                sum_hess,
                split_gain: at(&loss_changes, i) as f32,
                is_categorical: false,
                cat_begin: 0,
                cat_end: 0,
            });
        }
    }

    // Reuse `RegTree`'s own serde representation to build it from our nodes
    // without exposing its private field layout.
    let tree: RegTree = serde_json::from_value(json!({ "nodes": nodes }))?;
    Ok(tree)
}

// ---------------------------------------------------------------------------
// base_score link handling
// ---------------------------------------------------------------------------

/// Reconstruct the objective from name + `num_class`, if it is one we support.
fn build_objective(name: &str, num_class: usize) -> Option<Box<dyn Objective>> {
    let params = TrainingParams::builder()
        .objective(name)
        .num_class(num_class)
        .build_unchecked();
    create_objective(&params).ok()
}

/// Map a margin-space `base_score` into the space XGBoost stores it in.
fn link_export(margin: f32, objective: &str, num_class: usize) -> f32 {
    if let Some(obj) = build_objective(objective, num_class) {
        // Only scalar objectives have a well-defined single-value transform;
        // multiclass (softmax) operates across classes, so pass it through.
        if obj.n_outputs() == 1 {
            let mut buf = [margin];
            obj.pred_transform(&mut buf);
            return buf[0];
        }
    }
    margin
}

/// Map XGBoost's stored `base_score` back into margin space.
fn link_import(stored: f32, objective: &str, num_class: usize) -> f32 {
    match build_objective(objective, num_class) {
        Some(obj) => obj.prob_to_margin(stored),
        None => stored,
    }
}

// ---------------------------------------------------------------------------
// Small JSON helpers
// ---------------------------------------------------------------------------

/// Build the `objective` sub-document, with a best-effort parameter block so
/// upstream XGBoost accepts the file. Only the `name` is required by our reader.
fn objective_to_json(objective: &str, num_class: usize) -> Value {
    match objective {
        "multi:softmax" | "multi:softprob" => json!({
            "name": objective,
            "softmax_multiclass_param": { "num_class": num_class.to_string() },
        }),
        "count:poisson" => json!({
            "name": objective,
            "poisson_regression_param": { "max_delta_step": "0.7" },
        }),
        _ => json!({
            "name": objective,
            "reg_loss_param": { "scale_pos_weight": "1", "scale_pos_weight_default": "1" },
        }),
    }
}

/// Fetch a required object field, erroring with its name if absent.
fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| fmt_err(format!("missing `{key}`")))
}

/// Coerce a scalar JSON value (number, numeric string, or bool) to `f64`.
/// XGBoost writes learner/tree parameters as strings but arrays as numbers.
fn scalar_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Read a JSON array field, mapping each element through `f`. Returns `None` if
/// the field is missing or is not an array.
fn arr(v: &Value, key: &str, f: fn(&Value) -> Option<f64>) -> Option<Vec<f64>> {
    v.get(key)?
        .as_array()
        .map(|a| a.iter().map(|e| f(e).unwrap_or(0.0)).collect())
}

/// Construct a [`SequoiaError::ModelFormat`] from any displayable message.
fn fmt_err(msg: impl Into<String>) -> SequoiaError {
    SequoiaError::ModelFormat(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::learner::train;

    /// Train a small squared-error model on a noisy nonlinear signal.
    fn reg_model() -> (BoostedModel, DMatrix) {
        let n = 120;
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let a = i as f32 / n as f32;
            let b = ((i * 7) % n) as f32 / n as f32;
            x.push(a);
            x.push(b);
            y.push(2.0 * a - 3.0 * b + if a > 0.5 { 1.0 } else { -1.0 });
        }
        let d = DMatrix::from_dense(&x, n, 2)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        (train(&params, &d, 15).unwrap(), d)
    }

    #[test]
    fn roundtrip_reg_preserves_predictions() {
        let (model, d) = reg_model();
        let before = model.predict(&d).unwrap();

        let json = export_xgboost_json(&model).unwrap();
        let restored = import_xgboost_json(&json).unwrap();
        let after = restored.predict(&d).unwrap();

        assert_eq!(restored.num_trees(), model.num_trees());
        assert_eq!(restored.n_features(), model.n_features());
        assert_eq!(restored.objective(), model.objective());
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-5, "pred drift: {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_binary_preserves_predictions() {
        // Binary logistic exercises the prob<->margin base_score link.
        let n = 80;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let y: Vec<f32> = x.iter().map(|&v| f32::from(v > 0.4)).collect();
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("binary:logistic")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 20).unwrap();
        let before = model.predict(&d).unwrap();

        let json = export_xgboost_json(&model).unwrap();
        let restored = import_xgboost_json(&json).unwrap();
        assert_eq!(restored.objective(), "binary:logistic");
        // base_score should round-trip through the logit/sigmoid link.
        assert!((restored.base_score() - model.base_score()).abs() < 1e-4);
        let after = restored.predict(&d).unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-5, "pred drift: {a} vs {b}");
        }
    }

    /// A minimal, hand-written XGBoost-format stump: feature 0 with threshold
    /// 1.5, left leaf +10, right leaf -10, base_score 0 (raw margin).
    fn hand_stump_json() -> &'static str {
        r#"{
          "version": [2, 0, 0],
          "learner": {
            "gradient_booster": {
              "name": "gbtree",
              "model": {
                "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": "1"},
                "tree_info": [0],
                "trees": [{
                  "id": 0,
                  "tree_param": {"num_nodes": "3", "num_feature": "1", "size_leaf_vector": "0"},
                  "left_children":  [1, -1, -1],
                  "right_children": [2, -1, -1],
                  "parents":        [2147483647, 0, 0],
                  "split_indices":  [0, 0, 0],
                  "split_conditions": [1.5, 10.0, -10.0],
                  "default_left":   [1, 0, 0],
                  "base_weights":   [0.0, 10.0, -10.0],
                  "loss_changes":   [42.0, 0.0, 0.0],
                  "sum_hessian":    [8.0, 5.0, 3.0],
                  "split_type":     [0, 0, 0]
                }]
              }
            },
            "learner_model_param": {"base_score": "0", "num_class": "0", "num_feature": "1"},
            "objective": {"name": "reg:squarederror"}
          }
        }"#
    }

    #[test]
    fn import_hand_written_stump_routes_correctly() {
        let model = import_xgboost_json(hand_stump_json()).unwrap();
        assert_eq!(model.num_trees(), 1);
        assert_eq!(model.n_features(), 1);
        assert_eq!(model.base_score(), 0.0);

        // x=1.0 (< 1.5) -> left leaf +10 ; x=2.0 (>= 1.5) -> right leaf -10.
        let d = DMatrix::from_dense(&[1.0, 2.0], 2, 1).unwrap();
        let margins = model.predict_margin(&d);
        assert!((margins[0] - 10.0).abs() < 1e-6, "got {}", margins[0]);
        assert!((margins[1] + 10.0).abs() < 1e-6, "got {}", margins[1]);

        // Missing value follows default_left = true -> left leaf.
        let dm = DMatrix::from_dense(&[f32::NAN], 1, 1).unwrap();
        let mm = model.predict_margin(&dm);
        assert!((mm[0] - 10.0).abs() < 1e-6, "missing routed wrong: {}", mm[0]);
    }

    #[test]
    fn unsupported_booster_is_rejected() {
        let js = r#"{"learner": {"gradient_booster": {"name": "gblinear"},
                     "learner_model_param": {"num_feature": "3", "base_score": "0"}}}"#;
        let err = import_xgboost_json(js).unwrap_err();
        assert!(matches!(err, SequoiaError::ModelFormat(_)));
    }
}
