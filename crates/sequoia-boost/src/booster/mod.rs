//! Non-tree boosters.
//!
//! Currently this houses the linear (`gblinear`) booster, which fits a linear
//! model by coordinate descent instead of growing trees. The tree-based
//! boosters (`gbtree`, `dart`) live in the [`crate::learner`] training loop.

pub mod gblinear;
