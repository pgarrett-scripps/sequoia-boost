//! Feature and instance metadata carried alongside the feature matrix.

use serde::{Deserialize, Serialize};

/// How a feature column should be treated during split finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FeatureType {
    /// Ordered numerical feature; splits are `x < threshold`.
    #[default]
    Numerical,
    /// Unordered categorical feature; splits partition category sets.
    Categorical,
}

/// Ranking group layout, stored as a prefix-sum (`group_ptr`) over rows.
///
/// `group_ptr` has `num_groups + 1` entries; group `g` spans rows
/// `group_ptr[g]..group_ptr[g + 1]`. This matches XGBoost's CSR-style group
/// encoding for learning-to-rank objectives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GroupInfo {
    /// Prefix-sum group boundaries over the row index.
    pub group_ptr: Vec<usize>,
}

impl GroupInfo {
    /// Build a [`GroupInfo`] from per-group sizes (e.g. `[3, 2, 4]`).
    pub fn from_sizes(sizes: &[usize]) -> Self {
        let mut group_ptr = Vec::with_capacity(sizes.len() + 1);
        group_ptr.push(0);
        let mut acc = 0;
        for &s in sizes {
            acc += s;
            group_ptr.push(acc);
        }
        GroupInfo { group_ptr }
    }

    /// Number of groups.
    pub fn num_groups(&self) -> usize {
        self.group_ptr.len().saturating_sub(1)
    }

    /// The total number of rows spanned by all groups.
    pub fn num_rows(&self) -> usize {
        self.group_ptr.last().copied().unwrap_or(0)
    }

    /// Iterate `(start, end)` row ranges, one per group.
    pub fn iter_ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.group_ptr
            .windows(2)
            .map(|w| (w[0], w[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_prefix_sum() {
        let g = GroupInfo::from_sizes(&[3, 2, 4]);
        assert_eq!(g.group_ptr, vec![0, 3, 5, 9]);
        assert_eq!(g.num_groups(), 3);
        assert_eq!(g.num_rows(), 9);
        let ranges: Vec<_> = g.iter_ranges().collect();
        assert_eq!(ranges, vec![(0, 3), (3, 5), (5, 9)]);
    }
}
