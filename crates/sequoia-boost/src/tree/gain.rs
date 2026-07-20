//! Regularized structure-score (gain) and leaf-weight computation.
//!
//! These follow XGBoost's second-order formulation. For a set of instances with
//! gradient sum `G` and Hessian sum `H`:
//!
//! * optimal leaf weight `w* = −Tα(G) / (H + λ)`  (clamped by `max_delta_step`)
//! * structure score `Tα(G)² / (H + λ)`
//!
//! where `Tα` is the soft-threshold operator applied by the L1 term `α`, and `λ`
//! is the L2 term. The split loss change is `gain(L) + gain(R) − gain(parent)`,
//! and a split is accepted when that exceeds `γ` (`min_split_loss`). Computed in
//! `f64` for numerical stability, matching XGBoost's accumulation precision.

use crate::config::TrainingParams;

/// Accumulated first/second-order statistics for a set of instances.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradStats {
    /// Sum of gradients.
    pub grad: f64,
    /// Sum of Hessians.
    pub hess: f64,
}

impl GradStats {
    /// Construct from raw sums.
    #[inline]
    pub fn new(grad: f64, hess: f64) -> Self {
        GradStats { grad, hess }
    }

    /// Add another set of statistics.
    #[inline]
    pub fn add(&mut self, other: GradStats) {
        self.grad += other.grad;
        self.hess += other.hess;
    }

    /// Subtract another set of statistics (used for the "other side" of a split
    /// and for histogram subtraction later).
    #[inline]
    pub fn sub(&self, other: GradStats) -> GradStats {
        GradStats {
            grad: self.grad - other.grad,
            hess: self.hess - other.hess,
        }
    }
}

/// The regularization hyper-parameters relevant to gain/weight, extracted from
/// [`TrainingParams`] so this math is testable in isolation.
#[derive(Debug, Clone, Copy)]
pub struct RegParams {
    /// L2 regularization (`lambda`).
    pub lambda: f64,
    /// L1 regularization (`alpha`).
    pub alpha: f64,
    /// Maximum absolute leaf weight (`max_delta_step`, `0` = unconstrained).
    pub max_delta_step: f64,
    /// Minimum child Hessian (`min_child_weight`).
    pub min_child_weight: f64,
}

impl RegParams {
    /// Extract the regularization parameters from a full training config.
    pub fn from_params(p: &TrainingParams) -> Self {
        RegParams {
            lambda: p.lambda,
            alpha: p.alpha,
            max_delta_step: p.max_delta_step,
            min_child_weight: p.min_child_weight,
        }
    }
}

/// Soft-threshold operator `Tα(g)` applied by L1 regularization.
#[inline]
pub fn threshold_l1(g: f64, alpha: f64) -> f64 {
    if g > alpha {
        g - alpha
    } else if g < -alpha {
        g + alpha
    } else {
        0.0
    }
}

/// Optimal leaf weight `w*` for the given statistics, clamped by
/// `max_delta_step` when set. Returns `0` when the Hessian is below
/// `min_child_weight` (the node cannot form a valid leaf on its own).
pub fn calc_weight(stats: GradStats, reg: &RegParams) -> f64 {
    if stats.hess < reg.min_child_weight || stats.hess <= 0.0 {
        return 0.0;
    }
    let mut w = -threshold_l1(stats.grad, reg.alpha) / (stats.hess + reg.lambda);
    if reg.max_delta_step > 0.0 {
        w = w.clamp(-reg.max_delta_step, reg.max_delta_step);
    }
    w
}

/// Structure score (gain) for a node. When `max_delta_step` is unset this is the
/// closed-form `Tα(G)² / (H + λ)`; otherwise it is evaluated at the clamped
/// weight to stay consistent with [`calc_weight`].
pub fn calc_gain(stats: GradStats, reg: &RegParams) -> f64 {
    if stats.hess < reg.min_child_weight || stats.hess <= 0.0 {
        return 0.0;
    }
    if reg.max_delta_step == 0.0 {
        let t = threshold_l1(stats.grad, reg.alpha);
        (t * t) / (stats.hess + reg.lambda)
    } else {
        // Gain at the (possibly clamped) optimal weight:
        //   −(2·Tα(G)·w + (H + λ)·w²)
        let w = calc_weight(stats, reg);
        let t = threshold_l1(stats.grad, reg.alpha);
        -(2.0 * t * w + (stats.hess + reg.lambda) * w * w)
    }
}

/// Loss change from splitting `parent` into `left` and `right`:
/// `gain(L) + gain(R) − gain(parent)`. Larger is better; the caller compares it
/// against `γ` (`min_split_loss`) to decide whether to keep the split.
pub fn split_gain(left: GradStats, right: GradStats, parent: GradStats, reg: &RegParams) -> f64 {
    calc_gain(left, reg) + calc_gain(right, reg) - calc_gain(parent, reg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn reg(lambda: f64, alpha: f64) -> RegParams {
        RegParams {
            lambda,
            alpha,
            max_delta_step: 0.0,
            min_child_weight: 0.0,
        }
    }

    #[test]
    fn threshold_l1_shrinks_toward_zero() {
        assert_eq!(threshold_l1(5.0, 2.0), 3.0);
        assert_eq!(threshold_l1(-5.0, 2.0), -3.0);
        assert_eq!(threshold_l1(1.0, 2.0), 0.0);
    }

    #[test]
    fn weight_and_gain_closed_form() {
        // G = -4, H = 2, lambda = 1 -> w = 4/3, gain = 16/3
        let s = GradStats::new(-4.0, 2.0);
        let r = reg(1.0, 0.0);
        assert_relative_eq!(calc_weight(s, &r), 4.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(calc_gain(s, &r), 16.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn l1_reduces_weight_and_gain() {
        let s = GradStats::new(-4.0, 2.0);
        let plain = calc_gain(s, &reg(1.0, 0.0));
        let l1 = calc_gain(s, &reg(1.0, 1.0));
        assert!(l1 < plain);
    }

    #[test]
    fn max_delta_step_clamps_weight() {
        let s = GradStats::new(-100.0, 1.0);
        let r = RegParams {
            lambda: 0.0,
            alpha: 0.0,
            max_delta_step: 1.0,
            min_child_weight: 0.0,
        };
        // Unclamped weight would be 100; clamp to 1.
        assert_relative_eq!(calc_weight(s, &r), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn min_child_weight_zeros_out() {
        let s = GradStats::new(-4.0, 0.5);
        let r = RegParams {
            lambda: 1.0,
            alpha: 0.0,
            max_delta_step: 0.0,
            min_child_weight: 1.0,
        };
        assert_eq!(calc_weight(s, &r), 0.0);
        assert_eq!(calc_gain(s, &r), 0.0);
    }

    #[test]
    fn split_gain_positive_when_children_separate() {
        // Parent grad 0 (perfectly mixed), children strongly separated.
        let left = GradStats::new(-4.0, 2.0);
        let right = GradStats::new(4.0, 2.0);
        let parent = GradStats::new(0.0, 4.0);
        let g = split_gain(left, right, parent, &reg(1.0, 0.0));
        assert!(g > 0.0);
    }
}
