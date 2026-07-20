//! Classification objectives.

use super::{weighted_label_mean, GradPair, Objective};

/// Numerically stable logistic sigmoid.
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Lower bound on the logistic Hessian, matching XGBoost's `kRtEps` guard so
/// that confidently-classified instances still contribute a positive Hessian.
const MIN_HESS: f32 = 1e-16;

/// Binary logistic regression (`binary:logistic`).
///
/// With `p = σ(margin)` the gradient is `p − label` and the Hessian is
/// `max(p (1 − p), ε)`. `scale_pos_weight` rescales the loss of positive
/// instances to combat class imbalance, exactly as in XGBoost.
#[derive(Debug, Clone, Copy)]
pub struct LogisticObjective {
    scale_pos_weight: f32,
}

impl LogisticObjective {
    /// Create a logistic objective with the given positive-class weight.
    pub fn new(scale_pos_weight: f32) -> Self {
        LogisticObjective { scale_pos_weight }
    }
}

impl Default for LogisticObjective {
    fn default() -> Self {
        LogisticObjective {
            scale_pos_weight: 1.0,
        }
    }
}

impl Objective for LogisticObjective {
    fn name(&self) -> &str {
        "binary:logistic"
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
        for i in 0..preds.len() {
            let y = labels[i];
            let p = sigmoid(preds[i]);
            let mut w = weights.map_or(1.0, |ws| ws[i]);
            // scale_pos_weight multiplies the weight of positive instances.
            if y == 1.0 {
                w *= self.scale_pos_weight;
            }
            let grad = (p - y) * w;
            let hess = (p * (1.0 - p)).max(MIN_HESS) * w;
            out[i] = GradPair::new(grad, hess);
        }
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        for p in preds.iter_mut() {
            *p = sigmoid(*p);
        }
    }

    fn base_margin(&self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        // Optimal constant probability is the (weighted) positive rate; the
        // margin is its logit, clamped away from the asymptotes.
        let mut p = weighted_label_mean(labels, weights);
        p = p.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln() as f32
    }

    fn prob_to_margin(&self, base_score: f32) -> f32 {
        let p = base_score.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    }

    fn default_metric(&self) -> &str {
        "logloss"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sigmoid_symmetry() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-6);
        assert_relative_eq!(sigmoid(2.0) + sigmoid(-2.0), 1.0, epsilon = 1e-6);
        // Extreme values do not overflow.
        assert!(sigmoid(80.0) <= 1.0 && sigmoid(80.0) > 0.999);
        assert!(sigmoid(-80.0) >= 0.0 && sigmoid(-80.0) < 0.001);
    }

    #[test]
    fn gradient_matches_closed_form() {
        let obj = LogisticObjective::default();
        // margin 0 -> p = 0.5
        let preds = [0.0f32];
        let labels = [1.0f32];
        let mut out = vec![GradPair::default(); 1];
        obj.gradient(&preds, &labels, None, &mut out);
        assert_relative_eq!(out[0].grad, -0.5, epsilon = 1e-6); // 0.5 - 1
        assert_relative_eq!(out[0].hess, 0.25, epsilon = 1e-6); // 0.5 * 0.5
    }

    #[test]
    fn scale_pos_weight_scales_positive() {
        let obj = LogisticObjective::new(3.0);
        let preds = [0.0f32, 0.0];
        let labels = [1.0f32, 0.0];
        let mut out = vec![GradPair::default(); 2];
        obj.gradient(&preds, &labels, None, &mut out);
        // positive instance gradient/hess scaled by 3
        assert_relative_eq!(out[0].grad, -1.5, epsilon = 1e-6);
        assert_relative_eq!(out[0].hess, 0.75, epsilon = 1e-6);
        // negative instance unaffected
        assert_relative_eq!(out[1].grad, 0.5, epsilon = 1e-6);
        assert_relative_eq!(out[1].hess, 0.25, epsilon = 1e-6);
    }

    #[test]
    fn base_margin_is_logit_of_rate() {
        let obj = LogisticObjective::default();
        // 50% positive -> logit(0.5) = 0
        assert_relative_eq!(obj.base_margin(&[1.0, 0.0], None), 0.0, epsilon = 1e-6);
    }
}
