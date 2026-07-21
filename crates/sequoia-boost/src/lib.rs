//! # sequoia-boost
//!
//! A faithful, fast, pure-Rust reimplementation of
//! [XGBoost](https://github.com/dmlc/xgboost) gradient boosting.
//!
//! `sequoia-boost` re-implements XGBoost's algorithms — the regularized
//! second-order boosting objective, exact / approximate / histogram tree
//! construction, the full objective and metric catalog, and advanced modeling
//! features (constraints, categorical splits, DART, SHAP) — from scratch in
//! idiomatic Rust, with multi-core (`rayon`) and SIMD acceleration.
//!
//! ## Status
//!
//! Under active construction. The public API will stabilize as the phased
//! implementation lands; until then, expect breaking changes.
//!
//! ## Quick start (target API)
//!
//! ```ignore
//! use sequoia_boost::prelude::*;
//!
//! let dtrain = DMatrix::from_dense(&x, n_rows, n_cols)?.with_labels(&y)?;
//! let params = TrainingParams::builder()
//!     .objective("reg:squarederror")
//!     .max_depth(4)
//!     .eta(0.1)
//!     .build()?;
//! let model = train(&params, &dtrain, 100)?;
//! let preds = model.predict(&dtrain)?;
//! ```
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod data;
pub mod error;
pub mod learner;
pub mod metric;
pub mod model;
pub mod objective;
pub mod tree;

pub use config::{
    BoosterKind, GrowPolicy, Monotone, TrainingParams, TrainingParamsBuilder, TreeMethod,
};
pub use data::DMatrix;
pub use error::{Result, SequoiaError};
pub use learner::{
    cv, train, train_with_custom_metric, train_with_eval, train_with_objective, BoostedModel,
    CvResult, ImportanceType, TrainResult,
};
pub use metric::Metric;
pub use objective::{CustomObjective, GradPair, Objective};
pub use tree::{Node, RegTree};

/// Commonly used imports.
pub mod prelude {
    pub use crate::config::{BoosterKind, GrowPolicy, Monotone, TrainingParams, TreeMethod};
    pub use crate::data::{CsvOptions, DMatrix};
    pub use crate::error::{Result, SequoiaError};
    pub use crate::learner::{train, train_with_eval, BoostedModel, ImportanceType};
    pub use crate::metric::Metric;
}
