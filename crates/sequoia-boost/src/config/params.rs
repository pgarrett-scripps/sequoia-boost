//! Training configuration.
//!
//! Parameter names and default values deliberately mirror XGBoost so that
//! existing knowledge and configurations transfer directly. Where XGBoost
//! exposes aliases (e.g. `eta`/`learning_rate`), we pick the canonical field
//! name and document the alias.

use crate::error::{Result, SequoiaError};
use serde::{Deserialize, Serialize};

/// Which booster to use in the ensemble.
///
/// Mirrors XGBoost's `booster` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BoosterKind {
    /// Gradient boosted trees (XGBoost `gbtree`).
    #[default]
    GbTree,
    /// Dropout Additive Regression Trees (XGBoost `dart`).
    Dart,
    /// Linear booster with coordinate descent (XGBoost `gblinear`).
    GbLinear,
}

/// Tree construction algorithm.
///
/// Mirrors XGBoost's `tree_method`. `Auto` resolves to [`TreeMethod::Hist`] for
/// all but the smallest datasets, matching modern XGBoost behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TreeMethod {
    /// Pick automatically based on dataset size.
    #[default]
    Auto,
    /// Exact greedy algorithm (enumerate every split candidate).
    Exact,
    /// Approximate algorithm using weighted quantile sketch per split.
    Approx,
    /// Fast histogram algorithm with pre-binned features.
    Hist,
}

/// Order in which the tree is grown.
///
/// Mirrors XGBoost's `grow_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GrowPolicy {
    /// Split nodes closest to the root first (level-wise). XGBoost default.
    #[default]
    DepthWise,
    /// Split nodes with the highest loss reduction first (leaf-wise).
    LossGuide,
}

/// Per-feature monotonicity direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Monotone {
    /// No constraint on this feature.
    #[default]
    None,
    /// Prediction must be non-decreasing in this feature.
    Increasing,
    /// Prediction must be non-increasing in this feature.
    Decreasing,
}

/// The complete training configuration.
///
/// Construct with [`TrainingParams::builder`] or start from
/// [`TrainingParams::default`] and mutate fields directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingParams {
    // ---- General ----
    /// Which booster to train. XGBoost `booster`.
    pub booster: BoosterKind,
    /// Number of worker threads. `0` uses the global Rayon pool. XGBoost `nthread`.
    pub nthread: usize,
    /// RNG seed for subsampling and column sampling. XGBoost `seed`.
    pub seed: u64,

    // ---- Learning task ----
    /// Objective function name, e.g. `"reg:squarederror"`, `"binary:logistic"`.
    /// XGBoost `objective`.
    pub objective: String,
    /// Number of classes for multiclass objectives. XGBoost `num_class`.
    pub num_class: usize,
    /// Global bias / initial prediction (in probability space where applicable).
    /// `None` means "estimate from the labels", matching modern XGBoost.
    /// XGBoost `base_score`.
    pub base_score: Option<f64>,
    /// Evaluation metric names. Empty means "use the objective's default".
    /// XGBoost `eval_metric`.
    pub eval_metric: Vec<String>,

    // ---- Tree booster ----
    /// Learning rate / step-size shrinkage. XGBoost `eta` / `learning_rate`.
    pub eta: f64,
    /// Minimum loss reduction to make a split. XGBoost `gamma` / `min_split_loss`.
    pub gamma: f64,
    /// Maximum tree depth (`0` = no limit). XGBoost `max_depth`.
    pub max_depth: usize,
    /// Maximum number of leaves for `LossGuide` growth (`0` = no limit).
    /// XGBoost `max_leaves`.
    pub max_leaves: usize,
    /// Minimum sum of instance hessian needed in a child. XGBoost `min_child_weight`.
    pub min_child_weight: f64,
    /// Maximum delta step allowed for each leaf weight (`0` = no constraint).
    /// XGBoost `max_delta_step`.
    pub max_delta_step: f64,
    /// Row subsample ratio per boosting round. XGBoost `subsample`.
    pub subsample: f64,
    /// Column subsample ratio per tree. XGBoost `colsample_bytree`.
    pub colsample_bytree: f64,
    /// Column subsample ratio per level. XGBoost `colsample_bylevel`.
    pub colsample_bylevel: f64,
    /// Column subsample ratio per node. XGBoost `colsample_bynode`.
    pub colsample_bynode: f64,
    /// L2 regularization on leaf weights. XGBoost `lambda` / `reg_lambda`.
    pub lambda: f64,
    /// L1 regularization on leaf weights. XGBoost `alpha` / `reg_alpha`.
    pub alpha: f64,
    /// Balancing of positive/negative weights for imbalanced binary problems.
    /// XGBoost `scale_pos_weight`.
    pub scale_pos_weight: f64,
    /// Tree construction algorithm. XGBoost `tree_method`.
    pub tree_method: TreeMethod,
    /// Tree growth order. XGBoost `grow_policy`.
    pub grow_policy: GrowPolicy,
    /// Maximum number of histogram bins per feature. XGBoost `max_bin`.
    pub max_bin: usize,
    /// Per-feature monotone constraints (empty = none). XGBoost `monotone_constraints`.
    pub monotone_constraints: Vec<Monotone>,

    // ---- DART-specific ----
    /// Fraction of trees to drop each round (DART). XGBoost `rate_drop`.
    pub rate_drop: f64,
    /// Probability of skipping dropout in a round (DART). XGBoost `skip_drop`.
    pub skip_drop: f64,

    // ---- Missing value ----
    /// Value treated as "missing" in dense inputs. Defaults to NaN, like XGBoost.
    pub missing: f64,
}

impl Default for TrainingParams {
    fn default() -> Self {
        TrainingParams {
            booster: BoosterKind::GbTree,
            nthread: 0,
            seed: 0,
            objective: "reg:squarederror".to_string(),
            num_class: 0,
            base_score: None,
            eval_metric: Vec::new(),
            eta: 0.3,
            gamma: 0.0,
            max_depth: 6,
            max_leaves: 0,
            min_child_weight: 1.0,
            max_delta_step: 0.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            lambda: 1.0,
            alpha: 0.0,
            scale_pos_weight: 1.0,
            tree_method: TreeMethod::Auto,
            grow_policy: GrowPolicy::DepthWise,
            max_bin: 256,
            monotone_constraints: Vec::new(),
            rate_drop: 0.0,
            skip_drop: 0.0,
            missing: f64::NAN,
        }
    }
}

impl TrainingParams {
    /// Start a builder for ergonomic, chained configuration.
    pub fn builder() -> TrainingParamsBuilder {
        TrainingParamsBuilder {
            params: TrainingParams::default(),
        }
    }

    /// Validate mutually-consistent ranges. Called automatically before training.
    pub fn validate(&self) -> Result<()> {
        let unit = |name: &'static str, v: f64| -> Result<()> {
            if !(0.0..=1.0).contains(&v) {
                Err(SequoiaError::invalid_param(
                    name,
                    format!("must be in [0, 1], got {v}"),
                ))
            } else {
                Ok(())
            }
        };
        let positive = |name: &'static str, v: f64| -> Result<()> {
            if v.is_nan() || v <= 0.0 {
                Err(SequoiaError::invalid_param(
                    name,
                    format!("must be > 0, got {v}"),
                ))
            } else {
                Ok(())
            }
        };
        let non_negative = |name: &'static str, v: f64| -> Result<()> {
            if v.is_nan() || v < 0.0 {
                Err(SequoiaError::invalid_param(
                    name,
                    format!("must be >= 0, got {v}"),
                ))
            } else {
                Ok(())
            }
        };

        positive("eta", self.eta)?;
        non_negative("gamma", self.gamma)?;
        non_negative("min_child_weight", self.min_child_weight)?;
        non_negative("max_delta_step", self.max_delta_step)?;
        non_negative("lambda", self.lambda)?;
        non_negative("alpha", self.alpha)?;
        positive("scale_pos_weight", self.scale_pos_weight)?;
        unit("subsample", self.subsample)?;
        // subsample of exactly 0 is meaningless.
        if self.subsample == 0.0 {
            return Err(SequoiaError::invalid_param("subsample", "must be > 0"));
        }
        unit("colsample_bytree", self.colsample_bytree)?;
        unit("colsample_bylevel", self.colsample_bylevel)?;
        unit("colsample_bynode", self.colsample_bynode)?;
        unit("rate_drop", self.rate_drop)?;
        unit("skip_drop", self.skip_drop)?;

        if self.max_bin < 2 {
            return Err(SequoiaError::invalid_param(
                "max_bin",
                format!("must be >= 2, got {}", self.max_bin),
            ));
        }
        if self.grow_policy == GrowPolicy::LossGuide
            && self.max_leaves == 0
            && self.max_depth == 0
        {
            return Err(SequoiaError::invalid_param(
                "max_leaves",
                "lossguide growth needs a bound: set max_leaves or max_depth > 0",
            ));
        }
        Ok(())
    }
}

/// Builder for [`TrainingParams`].
///
/// Every setter returns `self` for chaining. Terminal method is
/// [`TrainingParamsBuilder::build`], which validates the configuration.
#[derive(Debug, Clone)]
pub struct TrainingParamsBuilder {
    params: TrainingParams,
}

macro_rules! setter {
    ($(#[$m:meta])* $name:ident, $ty:ty) => {
        $(#[$m])*
        pub fn $name(mut self, v: $ty) -> Self {
            self.params.$name = v;
            self
        }
    };
}

impl TrainingParamsBuilder {
    setter!(/// Set the booster kind.
        booster, BoosterKind);
    setter!(/// Set the number of worker threads (`0` = global pool).
        nthread, usize);
    setter!(/// Set the RNG seed.
        seed, u64);
    setter!(/// Set the number of classes (multiclass objectives).
        num_class, usize);
    setter!(/// Set the learning rate (`eta`).
        eta, f64);
    setter!(/// Set the minimum split loss (`gamma`).
        gamma, f64);
    setter!(/// Set the maximum tree depth.
        max_depth, usize);
    setter!(/// Set the maximum number of leaves (lossguide).
        max_leaves, usize);
    setter!(/// Set the minimum child hessian weight.
        min_child_weight, f64);
    setter!(/// Set the maximum delta step.
        max_delta_step, f64);
    setter!(/// Set the row subsample ratio.
        subsample, f64);
    setter!(/// Set the per-tree column subsample ratio.
        colsample_bytree, f64);
    setter!(/// Set the per-level column subsample ratio.
        colsample_bylevel, f64);
    setter!(/// Set the per-node column subsample ratio.
        colsample_bynode, f64);
    setter!(/// Set the L2 regularization (`lambda`).
        lambda, f64);
    setter!(/// Set the L1 regularization (`alpha`).
        alpha, f64);
    setter!(/// Set the positive-class weight scaling.
        scale_pos_weight, f64);
    setter!(/// Set the tree construction method.
        tree_method, TreeMethod);
    setter!(/// Set the tree growth policy.
        grow_policy, GrowPolicy);
    setter!(/// Set the maximum histogram bins per feature.
        max_bin, usize);
    setter!(/// Set the DART per-round drop rate (`rate_drop`).
        rate_drop, f64);
    setter!(/// Set the DART dropout-skip probability (`skip_drop`).
        skip_drop, f64);

    /// Set the objective by name (e.g. `"binary:logistic"`).
    pub fn objective(mut self, name: impl Into<String>) -> Self {
        self.params.objective = name.into();
        self
    }

    /// Set the base score / global bias.
    pub fn base_score(mut self, v: f64) -> Self {
        self.params.base_score = Some(v);
        self
    }

    /// Add an evaluation metric by name.
    pub fn eval_metric(mut self, name: impl Into<String>) -> Self {
        self.params.eval_metric.push(name.into());
        self
    }

    /// Set the per-feature monotone constraints.
    pub fn monotone_constraints(mut self, c: Vec<Monotone>) -> Self {
        self.params.monotone_constraints = c;
        self
    }

    /// Validate and produce the [`TrainingParams`].
    pub fn build(self) -> Result<TrainingParams> {
        self.params.validate()?;
        Ok(self.params)
    }

    /// Produce the [`TrainingParams`] without validation (useful in tests).
    pub fn build_unchecked(self) -> TrainingParams {
        self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_xgboost() {
        let p = TrainingParams::default();
        assert_eq!(p.eta, 0.3);
        assert_eq!(p.max_depth, 6);
        assert_eq!(p.min_child_weight, 1.0);
        assert_eq!(p.lambda, 1.0);
        assert_eq!(p.alpha, 0.0);
        assert_eq!(p.max_bin, 256);
        assert_eq!(p.booster, BoosterKind::GbTree);
        assert_eq!(p.grow_policy, GrowPolicy::DepthWise);
        assert!(p.base_score.is_none());
        p.validate().unwrap();
    }

    #[test]
    fn builder_chains_and_validates() {
        let p = TrainingParams::builder()
            .objective("binary:logistic")
            .eta(0.1)
            .max_depth(4)
            .subsample(0.8)
            .lambda(2.0)
            .build()
            .unwrap();
        assert_eq!(p.objective, "binary:logistic");
        assert_eq!(p.eta, 0.1);
        assert_eq!(p.max_depth, 4);
        assert_eq!(p.subsample, 0.8);
    }

    #[test]
    fn rejects_bad_params() {
        assert!(TrainingParams::builder().eta(0.0).build().is_err());
        assert!(TrainingParams::builder().subsample(1.5).build().is_err());
        assert!(TrainingParams::builder().lambda(-1.0).build().is_err());
        assert!(TrainingParams::builder().max_bin(1).build().is_err());
    }

    #[test]
    fn lossguide_requires_bound() {
        let r = TrainingParams::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .max_depth(0)
            .max_leaves(0)
            .build();
        assert!(r.is_err());
        // With a leaf bound it is fine.
        TrainingParams::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .max_leaves(31)
            .build()
            .unwrap();
    }
}
