//! Tree construction algorithms.
//!
//! Each builder grows a single [`crate::tree::RegTree`] from per-instance
//! gradients. The exact builder is the reference; approximate and histogram
//! builders (added in a later phase) share the same regularized gain math.

mod exact;
mod hist;

pub use exact::{all_features, all_rows, ExactTreeBuilder, SortedColumns};
pub use hist::HistTreeBuilder;
