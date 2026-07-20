//! Dataset containers, metadata, and loaders.

mod dmatrix;
pub mod ghist;
mod loaders;
mod meta;
pub mod quantile;

pub use dmatrix::{CscView, DMatrix, Entry};
pub use ghist::GHistIndex;
pub use loaders::{load_csv, load_libsvm, read_csv, read_libsvm, CsvOptions};
pub use meta::{FeatureType, GroupInfo};
pub use quantile::HistCuts;
