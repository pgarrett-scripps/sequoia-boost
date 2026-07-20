//! Cross-tool model interchange.
//!
//! This module implements import/export of gradient-boosted models in the
//! JSON schema used by upstream [XGBoost](https://github.com/dmlc/xgboost)
//! (`booster.save_model("m.json")`), complementing the crate's own native
//! [`BoostedModel::to_json`](crate::BoostedModel::to_json) format. It lets
//! `sequoia-boost` load models trained by real XGBoost and emit models that
//! XGBoost-compatible tooling can read.

pub mod xgboost_json;

pub use xgboost_json::{export_xgboost_json, import_xgboost_json};
