//! User-defined objective hook.
//!
//! Wraps caller-supplied closures so any custom loss can drive training via
//! [`crate::learner::train_with_objective`]. The gradient closure receives raw
//! margins and writes gradient/Hessian pairs, matching the built-in objectives.

use super::{GradPair, Objective};

type GradFn = dyn Fn(&[f32], &[f32], Option<&[f32]>, &mut [GradPair]) + Send + Sync;
type TransformFn = dyn Fn(&mut [f32]) + Send + Sync;

/// An [`Objective`] backed by user closures.
pub struct CustomObjective {
    name: String,
    n_outputs: usize,
    base: f32,
    default_metric: String,
    grad_fn: Box<GradFn>,
    transform_fn: Option<Box<TransformFn>>,
}

impl CustomObjective {
    /// Build a custom objective.
    ///
    /// * `grad_fn` — `(margins, labels, weights, out)` fills `out` with gradients.
    /// * `base` — the initial margin (base score).
    /// * `transform_fn` — optional prediction transform (identity if `None`).
    pub fn new(
        name: impl Into<String>,
        n_outputs: usize,
        base: f32,
        default_metric: impl Into<String>,
        grad_fn: impl Fn(&[f32], &[f32], Option<&[f32]>, &mut [GradPair]) + Send + Sync + 'static,
    ) -> Self {
        CustomObjective {
            name: name.into(),
            n_outputs,
            base,
            default_metric: default_metric.into(),
            grad_fn: Box::new(grad_fn),
            transform_fn: None,
        }
    }

    /// Attach a prediction transform.
    pub fn with_transform(
        mut self,
        transform: impl Fn(&mut [f32]) + Send + Sync + 'static,
    ) -> Self {
        self.transform_fn = Some(Box::new(transform));
        self
    }
}

impl Objective for CustomObjective {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    fn gradient(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>, out: &mut [GradPair]) {
        (self.grad_fn)(preds, labels, weights, out);
    }

    fn pred_transform(&self, preds: &mut [f32]) {
        if let Some(t) = &self.transform_fn {
            t(preds);
        }
    }

    fn base_margin(&self, _labels: &[f32], _weights: Option<&[f32]>) -> f32 {
        self.base
    }

    fn default_metric(&self) -> &str {
        &self.default_metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_squared_error_behaves() {
        // Reimplement squared error as a custom objective.
        let obj = CustomObjective::new("custom:sqerr", 1, 0.0, "rmse", |preds, labels, w, out| {
            for i in 0..preds.len() {
                let wi = w.map_or(1.0, |ws| ws[i]);
                out[i] = GradPair::new((preds[i] - labels[i]) * wi, wi);
            }
        });
        let mut out = vec![GradPair::default(); 2];
        obj.gradient(&[2.0, 0.0], &[1.0, 0.5], None, &mut out);
        assert_eq!(out[0], GradPair::new(1.0, 1.0));
        assert_eq!(out[1], GradPair::new(-0.5, 1.0));
    }
}
