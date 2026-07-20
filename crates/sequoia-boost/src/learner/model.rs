//! The trained model: an ensemble of trees plus the metadata needed to turn
//! their sum into calibrated predictions.

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::error::Result;
use crate::objective::create_objective;
use crate::tree::RegTree;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The kind of feature-importance score to compute, mirroring XGBoost's
/// `importance_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportanceType {
    /// Number of times a feature is used to split.
    Weight,
    /// Total loss reduction attributed to splits on a feature.
    TotalGain,
    /// Average loss reduction per split on a feature.
    Gain,
    /// Total Hessian (cover) of splits on a feature.
    TotalCover,
    /// Average Hessian (cover) per split on a feature.
    Cover,
}

/// A gradient-boosted tree ensemble.
///
/// Leaf weights already include the learning rate (shrinkage), so a raw margin
/// prediction is simply `base_score + Σ tree(x)`. The stored `objective` name
/// drives the prediction transform (e.g. the logistic sigmoid).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoostedModel {
    trees: Vec<RegTree>,
    base_score: f32,
    objective: String,
    num_class: usize,
    n_features: usize,
    /// The best iteration index selected by early stopping, if any.
    best_iteration: Option<usize>,
    /// Per-tree contribution weights. For a plain `gbtree` model every weight is
    /// `1.0`; the DART booster stores fractional weights here so dropped trees
    /// can be rescaled. Defaults to empty for models serialized before this
    /// field existed, in which case every tree is treated as weight `1.0`.
    #[serde(default)]
    tree_weights: Vec<f32>,
}

impl BoostedModel {
    pub(crate) fn new(
        base_score: f32,
        objective: String,
        num_class: usize,
        n_features: usize,
    ) -> Self {
        BoostedModel {
            trees: Vec::new(),
            base_score,
            objective,
            num_class,
            n_features,
            best_iteration: None,
            tree_weights: Vec::new(),
        }
    }

    pub(crate) fn push_tree(&mut self, tree: RegTree) {
        self.trees.push(tree);
        self.tree_weights.push(1.0);
    }

    /// Append a tree with an explicit contribution weight (used by DART).
    pub(crate) fn push_tree_weighted(&mut self, tree: RegTree, weight: f32) {
        self.trees.push(tree);
        self.tree_weights.push(weight);
    }

    /// Contribution weight of tree `i` (`1.0` when weights are absent, e.g. for
    /// legacy models or plain `gbtree`).
    #[inline]
    pub(crate) fn tree_weight(&self, i: usize) -> f32 {
        self.tree_weights.get(i).copied().unwrap_or(1.0)
    }

    /// Multiply tree `i`'s contribution weight by `factor` (DART rescaling).
    pub(crate) fn scale_tree_weight(&mut self, i: usize, factor: f32) {
        if i < self.tree_weights.len() {
            self.tree_weights[i] *= factor;
        }
    }

    /// Raw margin predictions that exclude the trees marked `true` in `dropped`
    /// (indexed by tree id). Used by the DART training loop to compute a round's
    /// gradients from the ensemble minus its dropout set. Output is laid out
    /// `[instance][output]`.
    pub(crate) fn predict_margin_dropout(&self, data: &DMatrix, dropped: &[bool]) -> Vec<f32> {
        let n = data.n_rows();
        let k = self.n_outputs();
        let mut out = vec![self.base_score; n * k];
        for (ti, tree) in self.trees.iter().enumerate() {
            if dropped.get(ti).copied().unwrap_or(false) {
                continue;
            }
            let w = self.tree_weight(ti);
            let cls = ti % k;
            for row in 0..n {
                out[row * k + cls] += w * tree.predict_row(data, row);
            }
        }
        out
    }

    pub(crate) fn set_best_iteration(&mut self, it: Option<usize>) {
        self.best_iteration = it;
    }

    /// Reassemble a model from its constituent parts. Used by the XGBoost-JSON
    /// importer, which builds trees and metadata externally.
    pub(crate) fn from_parts(
        trees: Vec<RegTree>,
        base_score: f32,
        objective: String,
        num_class: usize,
        n_features: usize,
    ) -> Self {
        let tree_weights = vec![1.0; trees.len()];
        BoostedModel {
            trees,
            base_score,
            objective,
            num_class,
            n_features,
            best_iteration: None,
            tree_weights,
        }
    }

    /// The configured `num_class` (`0` for regression / binary objectives).
    pub(crate) fn num_class(&self) -> usize {
        self.num_class
    }

    /// Number of trees (boosting rounds × outputs).
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of raw outputs per instance (`num_class` for multiclass, else 1).
    #[inline]
    pub fn n_outputs(&self) -> usize {
        if self.num_class >= 2 {
            self.num_class
        } else {
            1
        }
    }

    /// Number of boosting rounds (`num_trees / n_outputs`).
    pub fn num_boost_rounds(&self) -> usize {
        self.trees.len() / self.n_outputs()
    }

    /// The base score (global bias) in margin space.
    pub fn base_score(&self) -> f32 {
        self.base_score
    }

    /// The objective name this model was trained with.
    pub fn objective(&self) -> &str {
        &self.objective
    }

    /// The best iteration chosen by early stopping, if applicable.
    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
    }

    /// Number of trees to use at prediction time: `(best_iteration + 1) ×
    /// n_outputs` when early stopping selected one, else all trees.
    fn effective_ntrees(&self) -> usize {
        match self.best_iteration {
            Some(it) => (it + 1) * self.n_outputs(),
            None => self.trees.len(),
        }
    }

    /// Raw margin predictions using the effective tree count. The output is laid
    /// out `[instance][output]` (length `n_rows × n_outputs`).
    pub fn predict_margin(&self, data: &DMatrix) -> Vec<f32> {
        self.predict_margin_limited(data, self.effective_ntrees())
    }

    /// Raw margin predictions using only the first `ntree_limit` trees
    /// (`0` = all trees, ignoring early stopping). Tree `t` contributes to
    /// output `t % n_outputs`.
    pub fn predict_margin_limited(&self, data: &DMatrix, ntree_limit: usize) -> Vec<f32> {
        let n = data.n_rows();
        let k = self.n_outputs();
        let limit = if ntree_limit == 0 {
            self.trees.len()
        } else {
            ntree_limit.min(self.trees.len())
        };
        // Initialize from the dataset's per-instance base margin when present
        // (it overrides the scalar base score, matching XGBoost); otherwise use
        // the trained global bias.
        let mut out = vec![self.base_score; n * k];
        if let Some(bm) = data.base_margin() {
            if bm.len() == n * k {
                out.copy_from_slice(bm);
            } else if bm.len() == n {
                for row in 0..n {
                    for c in 0..k {
                        out[row * k + c] = bm[row];
                    }
                }
            }
        }
        for (ti, tree) in self.trees[..limit].iter().enumerate() {
            let w = self.tree_weight(ti);
            let cls = ti % k;
            for row in 0..n {
                out[row * k + cls] += w * tree.predict_row(data, row);
            }
        }
        out
    }

    /// Predictions in the objective's reported space (probabilities for
    /// logistic/softmax, rates for count objectives). For multiclass this is an
    /// `n_rows × num_class` probability matrix; see [`BoostedModel::predict_class`]
    /// for hard class labels.
    pub fn predict(&self, data: &DMatrix) -> Result<Vec<f32>> {
        let mut margin = self.predict_margin(data);
        // A model trained with a custom objective cannot reconstruct its
        // transform from the name; fall back to the identity (raw margins),
        // mirroring how XGBoost returns margins for custom objectives.
        if let Ok(obj) = self.rebuild_objective() {
            obj.pred_transform(&mut margin);
        }
        Ok(margin)
    }

    /// For multiclass, the predicted class index per row (argmax over classes).
    /// For single-output models this returns the transformed prediction rounded
    /// to the nearest class at 0.5.
    pub fn predict_class(&self, data: &DMatrix) -> Result<Vec<u32>> {
        let probs = self.predict(data)?;
        let k = self.n_outputs();
        let n = data.n_rows();
        let mut out = vec![0u32; n];
        if k == 1 {
            for (i, o) in out.iter_mut().enumerate() {
                *o = u32::from(probs[i] > 0.5);
            }
        } else {
            for i in 0..n {
                let row = &probs[i * k..i * k + k];
                let mut best = 0usize;
                for c in 1..k {
                    if row[c] > row[best] {
                        best = c;
                    }
                }
                out[i] = best as u32;
            }
        }
        Ok(out)
    }

    /// Per-row leaf indices for each tree (shape `n_rows × num_trees`, row-major).
    pub fn predict_leaf(&self, data: &DMatrix) -> Vec<u32> {
        let n = data.n_rows();
        let t = self.trees.len();
        let mut out = vec![0u32; n * t];
        for row in 0..n {
            for (ti, tree) in self.trees.iter().enumerate() {
                let leaf = tree.leaf_id_with(|f| data.get(row, f as usize));
                out[row * t + ti] = leaf as u32;
            }
        }
        out
    }

    /// Compute feature importance of the requested type, returned as a map from
    /// feature index to score (features that never split are absent).
    pub fn feature_importance(&self, kind: ImportanceType) -> HashMap<u32, f64> {
        let mut count: HashMap<u32, f64> = HashMap::new();
        let mut cover: HashMap<u32, f64> = HashMap::new();
        let mut gain: HashMap<u32, f64> = HashMap::new();
        for tree in &self.trees {
            for node in tree.nodes() {
                if node.is_leaf() {
                    continue;
                }
                *count.entry(node.split_feature).or_default() += 1.0;
                *cover.entry(node.split_feature).or_default() += node.sum_hess as f64;
                *gain.entry(node.split_feature).or_default() += node.split_gain as f64;
            }
        }
        // Divide a total by the split count to get the per-split average.
        let average = |totals: HashMap<u32, f64>| -> HashMap<u32, f64> {
            totals
                .into_iter()
                .map(|(f, t)| {
                    let n = count.get(&f).copied().unwrap_or(1.0);
                    (f, t / n)
                })
                .collect()
        };
        match kind {
            ImportanceType::Weight => count,
            ImportanceType::TotalCover => cover,
            ImportanceType::Cover => average(cover),
            ImportanceType::TotalGain => gain,
            ImportanceType::Gain => average(gain),
        }
    }

    /// Read-only access to the trees (e.g. for serialization or SHAP).
    pub fn trees(&self) -> &[RegTree] {
        &self.trees
    }

    /// Number of features the model expects.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Serialize the model to a compact binary blob (bincode).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| crate::error::SequoiaError::ModelFormat(e.to_string()))
    }

    /// Deserialize a model from a binary blob produced by [`BoostedModel::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| crate::error::SequoiaError::ModelFormat(e.to_string()))
    }

    /// Save the model to a file in the native binary format.
    pub fn save_binary(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_bytes()?)?;
        Ok(())
    }

    /// Load a model from a native binary file.
    pub fn load_binary(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Serialize the model to a (human-readable) JSON string.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a model from a JSON string.
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Save the model to a JSON file.
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_json()?)?;
        Ok(())
    }

    /// Load a model from a JSON file.
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Serialize the model to XGBoost's JSON model schema (the text form of
    /// `booster.save_model("m.json")`), so XGBoost-compatible tooling can read
    /// it. See [`crate::model::export_xgboost_json`] for details and caveats.
    pub fn to_xgboost_json(&self) -> Result<String> {
        crate::model::export_xgboost_json(self)
    }

    /// Parse a model saved in XGBoost's JSON model schema. Best-effort for
    /// `gbtree` boosters and common objectives; see
    /// [`crate::model::import_xgboost_json`] for the mapping and limitations.
    pub fn from_xgboost_json(json: &str) -> Result<Self> {
        crate::model::import_xgboost_json(json)
    }

    /// Save the model to a file in XGBoost's JSON model format.
    pub fn save_xgboost_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_xgboost_json()?)?;
        Ok(())
    }

    /// Load a model from a file written in XGBoost's JSON model format.
    pub fn load_xgboost_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_xgboost_json(&std::fs::read_to_string(path)?)
    }

    fn rebuild_objective(&self) -> Result<Box<dyn crate::objective::Objective>> {
        let params = TrainingParams::builder()
            .objective(self.objective.clone())
            .num_class(self.num_class)
            .build_unchecked();
        create_objective(&params)
    }
}

#[cfg(test)]
mod tests {
    use super::BoostedModel;
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::learner::train;

    fn small_model() -> (BoostedModel, DMatrix) {
        let n = 60;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let y: Vec<f32> = x.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
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
        (train(&params, &d, 20).unwrap(), d)
    }

    #[test]
    fn binary_roundtrip_preserves_predictions() {
        let (model, d) = small_model();
        let before = model.predict(&d).unwrap();
        let bytes = model.to_bytes().unwrap();
        let restored = BoostedModel::from_bytes(&bytes).unwrap();
        let after = restored.predict(&d).unwrap();
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(restored.num_trees(), model.num_trees());
        assert_eq!(restored.objective(), model.objective());
    }

    #[test]
    fn json_roundtrip_preserves_predictions() {
        let (model, d) = small_model();
        let before = model.predict(&d).unwrap();
        let json = model.to_json().unwrap();
        let restored = BoostedModel::from_json(&json).unwrap();
        let after = restored.predict(&d).unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
