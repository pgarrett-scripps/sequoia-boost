//! Monotone-constraint support: bounded leaf weights and constrained gain.
//!
//! Each node carries a `[lower, upper]` interval on its leaf weight. A split on a
//! monotone feature is only allowed when the child weights respect the requested
//! direction, and the children inherit tightened bounds so the constraint holds
//! for the whole subtree — the scheme XGBoost uses.

use crate::config::Monotone;
use crate::tree::gain::{threshold_l1, GradStats, RegParams};

/// Per-feature monotone directions (`+1` increasing, `-1` decreasing, `0` none).
#[derive(Debug, Clone, Default)]
pub struct MonotoneConstraints {
    dirs: Vec<i8>,
}

impl MonotoneConstraints {
    /// Build from the parameter list (empty ⇒ no constraints).
    pub fn from_params(list: &[Monotone]) -> Self {
        let dirs = list
            .iter()
            .map(|m| match m {
                Monotone::Increasing => 1,
                Monotone::Decreasing => -1,
                Monotone::None => 0,
            })
            .collect();
        MonotoneConstraints { dirs }
    }

    /// Whether any feature is constrained.
    pub fn is_active(&self) -> bool {
        self.dirs.iter().any(|&d| d != 0)
    }

    /// Direction for a feature (`0` when unconstrained or out of range).
    #[inline]
    pub fn dir(&self, feature: usize) -> i8 {
        self.dirs.get(feature).copied().unwrap_or(0)
    }
}

/// A weight interval `[lower, upper]` on a node's leaf value.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    /// Lower bound (may be `-∞`).
    pub lower: f64,
    /// Upper bound (may be `+∞`).
    pub upper: f64,
}

impl Default for Bounds {
    fn default() -> Self {
        Bounds {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }
}

impl Bounds {
    /// Clamp a weight into the interval.
    #[inline]
    pub fn clamp(&self, w: f64) -> f64 {
        w.clamp(self.lower, self.upper)
    }
}

/// Optimal leaf weight subject to `max_delta_step` and the bound interval.
pub fn calc_weight_bounded(stats: GradStats, reg: &RegParams, bounds: Bounds) -> f64 {
    if stats.hess < reg.min_child_weight || stats.hess <= 0.0 {
        return bounds.clamp(0.0);
    }
    let mut w = -threshold_l1(stats.grad, reg.alpha) / (stats.hess + reg.lambda);
    if reg.max_delta_step > 0.0 {
        w = w.clamp(-reg.max_delta_step, reg.max_delta_step);
    }
    bounds.clamp(w)
}

/// Structure score evaluated at a specific weight `w`:
/// `−(2·Tα(G)·w + (H + λ)·w²)`. At the unconstrained optimum this equals the
/// closed-form `Tα(G)²/(H + λ)`.
pub fn gain_at_weight(stats: GradStats, reg: &RegParams, w: f64) -> f64 {
    if stats.hess < reg.min_child_weight || stats.hess <= 0.0 {
        return 0.0;
    }
    let t = threshold_l1(stats.grad, reg.alpha);
    -(2.0 * t * w + (stats.hess + reg.lambda) * w * w)
}

/// Child bounds produced by a monotone split, given the parent bounds, the split
/// feature direction, and the two child weights.
pub fn child_bounds(parent: Bounds, dir: i8, w_left: f64, w_right: f64) -> (Bounds, Bounds) {
    if dir == 0 {
        return (parent, parent);
    }
    let mid = 0.5 * (w_left + w_right);
    if dir > 0 {
        // Increasing: left ≤ mid ≤ right.
        (
            Bounds {
                lower: parent.lower,
                upper: mid,
            },
            Bounds {
                lower: mid,
                upper: parent.upper,
            },
        )
    } else {
        // Decreasing: left ≥ mid ≥ right.
        (
            Bounds {
                lower: mid,
                upper: parent.upper,
            },
            Bounds {
                lower: parent.lower,
                upper: mid,
            },
        )
    }
}

/// Whether a monotone split direction is satisfied by the child weights.
#[inline]
pub fn satisfies(dir: i8, w_left: f64, w_right: f64) -> bool {
    match dir {
        d if d > 0 => w_left <= w_right,
        d if d < 0 => w_left >= w_right,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg() -> RegParams {
        RegParams {
            lambda: 1.0,
            alpha: 0.0,
            max_delta_step: 0.0,
            min_child_weight: 0.0,
        }
    }

    #[test]
    fn unbounded_matches_closed_form() {
        let s = GradStats::new(-4.0, 2.0);
        let r = reg();
        let w = calc_weight_bounded(s, &r, Bounds::default());
        // gain at optimal weight equals Tα(G)^2/(H+λ) = 16/3.
        let g = gain_at_weight(s, &r, w);
        assert!((g - 16.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn increasing_child_bounds_order() {
        let (l, r) = child_bounds(Bounds::default(), 1, -1.0, 2.0);
        assert_eq!(l.upper, 0.5); // mid
        assert_eq!(r.lower, 0.5);
        assert!(satisfies(1, -1.0, 2.0));
        assert!(!satisfies(1, 2.0, -1.0));
    }

    #[test]
    fn bounds_clamp_weight() {
        let b = Bounds {
            lower: 0.0,
            upper: 1.0,
        };
        assert_eq!(b.clamp(-5.0), 0.0);
        assert_eq!(b.clamp(5.0), 1.0);
        assert_eq!(b.clamp(0.5), 0.5);
    }
}
