//! Top-level orchestration: the boosting loop, the trained model, prediction,
//! and feature importance.

mod cv;
mod model;
mod shap;
mod train;

pub use cv::{cv, CvResult};
pub use model::{BoostedModel, ImportanceType};
pub use train::{train, train_with_eval, train_with_objective, EvalSet, RoundEval, TrainResult};
