//! Regression objectives.

use super::{weighted_label_mean, GradPair, Objective};

/// Squared-error regression (`reg:squarederror`).
///
/// Loss `½ (pred − label)²` gives gradient `pred − label` and constant Hessian
/// `1`. The prediction transform is the identity and the optimal base margin is
/// the (weighted) label mean.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredErrorObjective;

impl Objective for SquaredErrorObjective {
    fn name(&self) -> &str {
        "reg:squarederror"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), out.len());
        match weights {
            Some(w) => {
                for i in 0..preds.len() {
                    let g = preds[i] - labels[i];
                    out[i] = GradPair::new(g * w[i], w[i]);
                }
            }
            None => {
                for i in 0..preds.len() {
                    out[i] = GradPair::new(preds[i] - labels[i], 1.0);
                }
            }
        }
    }

    fn base_margin(&self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        weighted_label_mean(labels, weights) as f32
    }

    fn default_metric(&self) -> &str {
        "rmse"
    }
}

/// Pseudo-Huber regression (`reg:pseudohubererror`): a smooth approximation of
/// the absolute error, robust to outliers. With `d = margin − y` and
/// `s = 1 + d²`, gradient is `d/√s` and Hessian `1/(s·√s)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PseudoHuberObjective;

impl Objective for PseudoHuberObjective {
    fn name(&self) -> &str {
        "reg:pseudohubererror"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        for i in 0..preds.len() {
            let w = weights.map_or(1.0, |ws| ws[i]);
            let d = preds[i] - labels[i];
            let s = 1.0 + d * d;
            let sqrt_s = s.sqrt();
            out[i] = GradPair::new((d / sqrt_s) * w, (1.0 / (s * sqrt_s)) * w);
        }
    }

    fn base_margin(&self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        weighted_label_mean(labels, weights) as f32
    }

    fn default_metric(&self) -> &str {
        "mae"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_closed_form() {
        let obj = SquaredErrorObjective;
        let preds = [2.0f32, 0.0, -1.0];
        let labels = [1.0f32, 0.5, -3.0];
        let mut out = vec![GradPair::default(); 3];
        obj.gradient(&preds, &labels, None, &mut out);
        assert_eq!(out[0], GradPair::new(1.0, 1.0)); // 2 - 1
        assert_eq!(out[1], GradPair::new(-0.5, 1.0)); // 0 - 0.5
        assert_eq!(out[2], GradPair::new(2.0, 1.0)); // -1 - (-3)
    }

    #[test]
    fn weighted_gradient_scales() {
        let obj = SquaredErrorObjective;
        let preds = [2.0f32];
        let labels = [1.0f32];
        let w = [4.0f32];
        let mut out = vec![GradPair::default(); 1];
        obj.gradient(&preds, &labels, Some(&w), &mut out);
        assert_eq!(out[0], GradPair::new(4.0, 4.0));
    }

    #[test]
    fn base_margin_is_label_mean() {
        let obj = SquaredErrorObjective;
        assert!((obj.base_margin(&[1.0, 2.0, 3.0], None) - 2.0).abs() < 1e-6);
    }
}
