//! Learning objectives: gradients, Hessians, prediction transforms, and base
//! score estimation.
//!
//! Every objective implements [`Objective`]. Boosting works in *margin* space
//! (raw additive scores); the [`Objective::pred_transform`] maps margins to the
//! reported prediction (e.g. the logistic sigmoid). This mirrors XGBoost's
//! separation of `GetGradient` / `PredTransform`.

mod classification;
mod count;
mod custom;
mod multiclass;
mod ranking;
mod regression;

pub use classification::LogisticObjective;
pub use count::{GammaObjective, PoissonObjective, TweedieObjective};
pub use custom::CustomObjective;
pub use multiclass::SoftmaxObjective;
pub use ranking::LambdaMartObjective;
pub use regression::{PseudoHuberObjective, SquaredErrorObjective};

use crate::config::TrainingParams;
use crate::error::{Result, SequoiaError};

/// A first- and second-order gradient for one instance/output.
///
/// Stored as `f32` to match XGBoost's memory layout and to keep histogram
/// accumulation cache-friendly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradPair {
    /// First-order gradient of the loss w.r.t. the margin.
    pub grad: f32,
    /// Second-order gradient (Hessian) of the loss w.r.t. the margin.
    pub hess: f32,
}

impl GradPair {
    /// Construct a gradient pair.
    #[inline]
    pub fn new(grad: f32, hess: f32) -> Self {
        GradPair { grad, hess }
    }
}

/// A differentiable learning objective.
///
/// Implementors are `Send + Sync` so gradient computation can be parallelized.
pub trait Objective: Send + Sync {
    /// The XGBoost-compatible objective name (e.g. `"reg:squarederror"`).
    fn name(&self) -> &str;

    /// Number of raw outputs produced per instance. `1` for regression and
    /// binary classification; `num_class` for multiclass objectives.
    fn n_outputs(&self) -> usize {
        1
    }

    /// Compute per-instance gradients and Hessians.
    ///
    /// `preds` holds raw margins laid out as `n_rows * n_outputs` (row-major by
    /// instance). `out` is written in the same layout. `weights`, if present,
    /// scales each instance's contribution.
    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    );

    /// Compute gradients with optional query-group structure.
    ///
    /// Learning-to-rank objectives (LambdaMART) override this to form document
    /// pairs *within* each group supplied by `group`. The default forwards to
    /// [`Objective::gradient`], ignoring the grouping — correct for all
    /// non-ranking objectives.
    fn gradient_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        _group: Option<&crate::data::GroupInfo>,
        out: &mut [GradPair],
    ) {
        self.gradient(preds, labels, weights, out)
    }

    /// Transform raw margins into reported predictions, in place. Default is the
    /// identity (used by squared-error regression).
    fn pred_transform(&self, _preds: &mut [f32]) {}

    /// Estimate the optimal constant prediction in *margin* space, used to
    /// initialize `base_score` when the user does not supply one.
    fn base_margin(&self, labels: &[f32], weights: Option<&[f32]>) -> f32;

    /// Convert a user-supplied `base_score` (given in prediction space) into
    /// margin space via the objective's inverse link. Default is the identity;
    /// objectives with a link function (e.g. logistic) override it.
    fn prob_to_margin(&self, base_score: f32) -> f32 {
        base_score
    }

    /// The default evaluation metric name for this objective.
    fn default_metric(&self) -> &str;
}

/// Weighted mean of `labels`, or the plain mean when `weights` is `None`.
/// Shared by objectives that initialize from the label mean.
pub(crate) fn weighted_label_mean(labels: &[f32], weights: Option<&[f32]>) -> f64 {
    match weights {
        Some(w) => {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (l, wi) in labels.iter().zip(w) {
                num += (*l as f64) * (*wi as f64);
                den += *wi as f64;
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        }
        None => {
            if labels.is_empty() {
                0.0
            } else {
                labels.iter().map(|l| *l as f64).sum::<f64>() / labels.len() as f64
            }
        }
    }
}

/// Resolve an objective by name, configured from `params`.
pub fn create_objective(params: &TrainingParams) -> Result<Box<dyn Objective>> {
    match params.objective.as_str() {
        "reg:squarederror" | "reg:linear" => Ok(Box::new(SquaredErrorObjective)),
        "reg:pseudohubererror" => Ok(Box::new(PseudoHuberObjective)),
        "binary:logistic" | "reg:logistic" => Ok(Box::new(LogisticObjective::new(
            params.scale_pos_weight as f32,
        ))),
        "multi:softmax" | "multi:softprob" => {
            if params.num_class < 2 {
                return Err(SequoiaError::invalid_param(
                    "num_class",
                    "multiclass objectives require num_class >= 2",
                ));
            }
            let prob = params.objective == "multi:softprob";
            Ok(Box::new(SoftmaxObjective::new(params.num_class, prob)))
        }
        "count:poisson" => Ok(Box::new(PoissonObjective::default())),
        "reg:gamma" => Ok(Box::new(GammaObjective)),
        "reg:tweedie" => Ok(Box::new(TweedieObjective::default())),
        "rank:pairwise" => Ok(Box::new(LambdaMartObjective::pairwise())),
        "rank:ndcg" => Ok(Box::new(LambdaMartObjective::ndcg())),
        "rank:map" => Ok(Box::new(LambdaMartObjective::map())),
        other => Err(SequoiaError::unknown("objective", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_mean_basic() {
        let labels = [1.0f32, 3.0];
        assert!((weighted_label_mean(&labels, None) - 2.0).abs() < 1e-9);
        let w = [3.0f32, 1.0];
        // (1*3 + 3*1) / 4 = 1.5
        assert!((weighted_label_mean(&labels, Some(&w)) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn factory_resolves_known_and_rejects_unknown() {
        let p = TrainingParams::builder()
            .objective("reg:squarederror")
            .build_unchecked();
        assert_eq!(create_objective(&p).unwrap().name(), "reg:squarederror");
        let p = TrainingParams::builder()
            .objective("nope:whatever")
            .build_unchecked();
        assert!(create_objective(&p).is_err());
    }
}
