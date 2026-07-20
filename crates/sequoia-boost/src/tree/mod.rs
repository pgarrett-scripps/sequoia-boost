//! Decision-tree representation, split-scoring math, and construction.

pub mod builder;
pub mod constraints;
pub mod gain;
pub mod hist;
mod regtree;

pub use gain::{calc_gain, calc_weight, split_gain, GradStats, RegParams};
pub use regtree::{Node, RegTree};
