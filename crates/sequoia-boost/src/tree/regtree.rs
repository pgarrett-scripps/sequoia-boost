//! Regression tree representation and prediction.
//!
//! A tree is a flat array of [`Node`]s (node `0` is the root). Internal nodes
//! carry a numeric split `x[feature] < threshold`; instances whose feature is
//! *missing* follow the node's `default_left` direction, implementing XGBoost's
//! sparsity-aware routing. Leaf nodes carry the raw leaf weight (the learning
//! rate is applied by the boosting loop, not baked into the tree).

use crate::data::DMatrix;
use serde::{Deserialize, Serialize};

/// Sentinel used in child pointers to mark "no child" (i.e. a leaf).
const NO_CHILD: i32 = -1;

/// A single tree node.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Split feature index (meaningful only for internal nodes).
    pub split_feature: u32,
    /// Split threshold: an instance goes left when `value < split_cond`.
    pub split_cond: f32,
    /// Direction taken by instances with a missing value at this node.
    pub default_left: bool,
    /// Left child index, or [`NO_CHILD`] for a leaf.
    pub left: i32,
    /// Right child index, or [`NO_CHILD`] for a leaf.
    pub right: i32,
    /// Leaf weight (used only for leaves).
    pub leaf_value: f32,
    /// Sum of Hessians routed through this node (for cover-based importance/SHAP).
    pub sum_hess: f32,
    /// Loss reduction (gain) achieved by this node's split (0 for leaves).
    pub split_gain: f32,
    /// Whether this internal node splits on a categorical feature by set
    /// membership rather than a numeric threshold. `false` for numeric splits
    /// and leaves.
    #[serde(default)]
    pub is_categorical: bool,
    /// For a categorical node, the start index into the owning tree's category
    /// list of the categories routed left. Unused (`0`) otherwise.
    #[serde(default)]
    pub cat_begin: u32,
    /// For a categorical node, the end index (exclusive) into the owning tree's
    /// category list of the categories routed left. Unused (`0`) otherwise.
    #[serde(default)]
    pub cat_end: u32,
}

impl Node {
    /// A fresh leaf node with the given weight and cover.
    fn leaf(value: f32, sum_hess: f32) -> Self {
        Node {
            split_feature: 0,
            split_cond: 0.0,
            default_left: true,
            left: NO_CHILD,
            right: NO_CHILD,
            leaf_value: value,
            sum_hess,
            split_gain: 0.0,
            is_categorical: false,
            cat_begin: 0,
            cat_end: 0,
        }
    }

    /// Whether this node is a leaf.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left == NO_CHILD
    }
}

/// A regression tree: a flat node array with node `0` as the root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RegTree {
    nodes: Vec<Node>,
    /// Flat pool of category values routed left by categorical nodes. Node
    /// `n` (when `n.is_categorical`) owns `categories[n.cat_begin..n.cat_end]`.
    /// Empty for trees with no categorical splits (including all legacy trees).
    #[serde(default)]
    categories: Vec<u32>,
}

impl RegTree {
    /// Create a tree consisting of a single leaf.
    pub fn single_leaf(value: f32, sum_hess: f32) -> Self {
        RegTree {
            nodes: vec![Node::leaf(value, sum_hess)],
            categories: Vec::new(),
        }
    }

    /// Create an empty tree with a placeholder root leaf, ready to be grown by a
    /// builder. Returns the root node id (`0`).
    pub(crate) fn with_root(sum_hess: f32) -> Self {
        RegTree {
            nodes: vec![Node::leaf(0.0, sum_hess)],
            categories: Vec::new(),
        }
    }

    /// Number of nodes (internal + leaf).
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaf nodes.
    pub fn num_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Read-only access to the node array.
    #[inline]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Access a node by id.
    #[inline]
    pub fn node(&self, id: usize) -> &Node {
        &self.nodes[id]
    }

    /// Turn leaf `nid` into an internal node by attaching two child leaves.
    /// Returns `(left_id, right_id)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn expand(
        &mut self,
        nid: usize,
        split_feature: u32,
        split_cond: f32,
        default_left: bool,
        left_value: f32,
        left_hess: f32,
        right_value: f32,
        right_hess: f32,
    ) -> (usize, usize) {
        let left_id = self.nodes.len();
        let right_id = left_id + 1;
        self.nodes.push(Node::leaf(left_value, left_hess));
        self.nodes.push(Node::leaf(right_value, right_hess));
        let n = &mut self.nodes[nid];
        n.split_feature = split_feature;
        n.split_cond = split_cond;
        n.default_left = default_left;
        n.left = left_id as i32;
        n.right = right_id as i32;
        (left_id, right_id)
    }

    /// Turn leaf `nid` into a categorical (set-membership) internal node.
    /// Instances whose value of `split_feature` is one of `cats_left` go to the
    /// left child; all other (and missing) categories follow `default_left`
    /// only when missing — present categories not in the set go right.
    /// Returns `(left_id, right_id)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn expand_categorical(
        &mut self,
        nid: usize,
        split_feature: u32,
        cats_left: &[u32],
        default_left: bool,
        left_value: f32,
        left_hess: f32,
        right_value: f32,
        right_hess: f32,
    ) -> (usize, usize) {
        let left_id = self.nodes.len();
        let right_id = left_id + 1;
        self.nodes.push(Node::leaf(left_value, left_hess));
        self.nodes.push(Node::leaf(right_value, right_hess));
        let begin = self.categories.len() as u32;
        self.categories.extend_from_slice(cats_left);
        let end = self.categories.len() as u32;
        let n = &mut self.nodes[nid];
        n.split_feature = split_feature;
        n.is_categorical = true;
        n.cat_begin = begin;
        n.cat_end = end;
        n.default_left = default_left;
        n.left = left_id as i32;
        n.right = right_id as i32;
        (left_id, right_id)
    }

    /// Set a leaf's weight (used to finalize leaf values after growth).
    pub(crate) fn set_leaf_value(&mut self, nid: usize, value: f32) {
        self.nodes[nid].leaf_value = value;
    }

    /// Record the loss reduction achieved by an internal node's split.
    pub(crate) fn set_split_gain(&mut self, nid: usize, gain: f32) {
        self.nodes[nid].split_gain = gain;
    }

    /// Multiply every leaf weight by `factor`. Used to apply the learning rate
    /// (shrinkage) so that stored trees already carry their scaled contribution,
    /// matching XGBoost's saved-model semantics.
    pub fn scale_leaves(&mut self, factor: f32) {
        for n in &mut self.nodes {
            if n.is_leaf() {
                n.leaf_value *= factor;
            }
        }
    }

    /// Route a single feature vector (via an accessor) to its leaf id.
    ///
    /// `get` returns `None` for a missing feature. Generic over the accessor so
    /// the same code serves dense rows, sparse rows, and SHAP traversals.
    pub fn leaf_id_with(&self, get: impl Fn(u32) -> Option<f32>) -> usize {
        let mut nid = 0usize;
        loop {
            let node = &self.nodes[nid];
            if node.is_leaf() {
                return nid;
            }
            let go_left = if node.is_categorical {
                match get(node.split_feature) {
                    // Categories are integer-coded; membership in the left set
                    // routes left, everything else (present, not in set) right.
                    Some(v) => {
                        let c = v as u32;
                        self.categories[node.cat_begin as usize..node.cat_end as usize].contains(&c)
                    }
                    None => node.default_left,
                }
            } else {
                match get(node.split_feature) {
                    Some(v) => v < node.split_cond,
                    None => node.default_left,
                }
            };
            nid = if go_left {
                node.left as usize
            } else {
                node.right as usize
            };
        }
    }

    /// Predict the raw leaf weight for row `row` of `data`.
    pub fn predict_row(&self, data: &DMatrix, row: usize) -> f32 {
        let leaf = self.leaf_id_with(|f| data.get(row, f as usize));
        self.nodes[leaf].leaf_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small tree by hand:
    /// root: feature 0 < 0.5 ? left : right, missing -> left
    ///   left leaf = -1.0, right leaf = +2.0
    fn stump() -> RegTree {
        let mut t = RegTree::with_root(10.0);
        t.expand(0, 0, 0.5, true, -1.0, 5.0, 2.0, 5.0);
        t
    }

    #[test]
    fn routing_numeric() {
        let t = stump();
        // value 0.2 < 0.5 -> left leaf -1.0
        assert_eq!(t.leaf_id_with(|_| Some(0.2)), 1);
        // value 0.9 >= 0.5 -> right leaf +2.0
        assert_eq!(t.leaf_id_with(|_| Some(0.9)), 2);
    }

    #[test]
    fn routing_missing_follows_default() {
        let t = stump();
        // missing -> default_left = true -> left leaf
        assert_eq!(t.leaf_id_with(|_| None), 1);
        assert_eq!(t.node(1).leaf_value, -1.0);
    }

    #[test]
    fn routing_categorical_set_membership() {
        // Categorical split: categories {0, 2} go left, everything else right.
        let mut t = RegTree::with_root(10.0);
        t.expand_categorical(0, 0, &[0, 2], false, -1.0, 5.0, 2.0, 5.0);
        assert!(t.node(0).is_categorical);
        // In-set categories route left (leaf 1, value -1.0).
        assert_eq!(t.leaf_id_with(|_| Some(0.0)), 1);
        assert_eq!(t.leaf_id_with(|_| Some(2.0)), 1);
        // Out-of-set present categories route right (leaf 2, value 2.0).
        assert_eq!(t.leaf_id_with(|_| Some(1.0)), 2);
        assert_eq!(t.leaf_id_with(|_| Some(3.0)), 2);
        // Unseen category also routes right (not in the left set).
        assert_eq!(t.leaf_id_with(|_| Some(9.0)), 2);
        // Missing follows default_left = false -> right.
        assert_eq!(t.leaf_id_with(|_| None), 2);
    }

    #[test]
    fn predict_row_dense() {
        let t = stump();
        let d = DMatrix::from_dense(&[0.1, 0.9], 2, 1).unwrap();
        assert_eq!(t.predict_row(&d, 0), -1.0);
        assert_eq!(t.predict_row(&d, 1), 2.0);
        assert_eq!(t.num_leaves(), 2);
        assert_eq!(t.num_nodes(), 3);
    }
}
