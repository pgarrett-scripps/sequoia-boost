//! Evaluation metrics used for reporting and early stopping.
//!
//! Metrics receive predictions that have already passed through the objective's
//! [`crate::objective::Objective::pred_transform`] (so classification metrics
//! see probabilities), matching XGBoost's evaluation pipeline.

use crate::error::{Result, SequoiaError};

/// An evaluation metric over predictions and labels.
pub trait Metric: Send + Sync {
    /// The XGBoost-compatible metric name (e.g. `"rmse"`, `"logloss"`).
    fn name(&self) -> &str;

    /// Whether a *larger* value is better (e.g. AUC). Drives early stopping.
    fn maximize(&self) -> bool {
        false
    }

    /// Evaluate the metric. `preds` are post-transform predictions.
    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64;
}

/// Root-mean-square error (`rmse`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Rmse;

impl Metric for Rmse {
    fn name(&self) -> &str {
        "rmse"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let mut sq = 0.0f64;
        let mut wsum = 0.0f64;
        for i in 0..preds.len() {
            let w = weights.map_or(1.0, |ws| ws[i] as f64);
            let d = preds[i] as f64 - labels[i] as f64;
            sq += w * d * d;
            wsum += w;
        }
        if wsum > 0.0 {
            (sq / wsum).sqrt()
        } else {
            0.0
        }
    }
}

/// Mean absolute error (`mae`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Mae;

impl Metric for Mae {
    fn name(&self) -> &str {
        "mae"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let mut abs = 0.0f64;
        let mut wsum = 0.0f64;
        for i in 0..preds.len() {
            let w = weights.map_or(1.0, |ws| ws[i] as f64);
            abs += w * (preds[i] as f64 - labels[i] as f64).abs();
            wsum += w;
        }
        if wsum > 0.0 {
            abs / wsum
        } else {
            0.0
        }
    }
}

/// Binary logistic loss (`logloss`). Predictions are probabilities.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogLoss;

impl Metric for LogLoss {
    fn name(&self) -> &str {
        "logloss"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        const EPS: f64 = 1e-15;
        let mut loss = 0.0f64;
        let mut wsum = 0.0f64;
        for i in 0..preds.len() {
            let w = weights.map_or(1.0, |ws| ws[i] as f64);
            let p = (preds[i] as f64).clamp(EPS, 1.0 - EPS);
            let y = labels[i] as f64;
            loss += w * -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            wsum += w;
        }
        if wsum > 0.0 {
            loss / wsum
        } else {
            0.0
        }
    }
}

/// Binary classification error rate at threshold 0.5 (`error`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ErrorRate;

impl Metric for ErrorRate {
    fn name(&self) -> &str {
        "error"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let mut wrong = 0.0f64;
        let mut wsum = 0.0f64;
        for i in 0..preds.len() {
            let w = weights.map_or(1.0, |ws| ws[i] as f64);
            let pred_pos = preds[i] > 0.5;
            let label_pos = labels[i] > 0.5;
            if pred_pos != label_pos {
                wrong += w;
            }
            wsum += w;
        }
        if wsum > 0.0 {
            wrong / wsum
        } else {
            0.0
        }
    }
}

/// Binary ROC AUC (`auc`), computed with the Mann–Whitney rank-sum and average
/// ranks for ties. Higher is better. Weights are ignored (unweighted AUC).
#[derive(Debug, Clone, Copy, Default)]
pub struct Auc;

impl Metric for Auc {
    fn name(&self) -> &str {
        "auc"
    }

    fn maximize(&self) -> bool {
        true
    }

    fn eval(&self, preds: &[f32], labels: &[f32], _weights: Option<&[f32]>) -> f64 {
        let n = preds.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| preds[a].partial_cmp(&preds[b]).unwrap());

        // Assign average ranks (1-based), resolving ties.
        let mut ranks = vec![0.0f64; n];
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && preds[order[j]] == preds[order[i]] {
                j += 1;
            }
            let avg = ((i + 1 + j) as f64) / 2.0; // average of ranks i+1..=j
            for &idx in &order[i..j] {
                ranks[idx] = avg;
            }
            i = j;
        }

        let mut sum_pos_rank = 0.0f64;
        let mut n_pos = 0.0f64;
        let mut n_neg = 0.0f64;
        for k in 0..n {
            if labels[k] > 0.5 {
                sum_pos_rank += ranks[k];
                n_pos += 1.0;
            } else {
                n_neg += 1.0;
            }
        }
        if n_pos == 0.0 || n_neg == 0.0 {
            return 0.5; // undefined; XGBoost-like neutral value
        }
        (sum_pos_rank - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    }
}

/// Multiclass log loss (`mlogloss`). Predictions are `n × num_class`
/// probabilities; labels are class indices.
#[derive(Debug, Clone, Copy)]
pub struct MLogLoss {
    num_class: usize,
}

impl Metric for MLogLoss {
    fn name(&self) -> &str {
        "mlogloss"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        const EPS: f64 = 1e-15;
        let k = self.num_class;
        let mut loss = 0.0;
        let mut wsum = 0.0;
        for (i, &lab) in labels.iter().enumerate() {
            let w = weights.map_or(1.0, |ws| ws[i] as f64);
            let p = (preds[i * k + lab as usize] as f64).clamp(EPS, 1.0);
            loss += -w * p.ln();
            wsum += w;
        }
        if wsum > 0.0 {
            loss / wsum
        } else {
            0.0
        }
    }
}

/// Multiclass error rate (`merror`): fraction whose argmax ≠ label.
#[derive(Debug, Clone, Copy)]
pub struct MError {
    num_class: usize,
}

impl Metric for MError {
    fn name(&self) -> &str {
        "merror"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let k = self.num_class;
        let mut wrong = 0.0;
        let mut wsum = 0.0;
        for (i, &lab) in labels.iter().enumerate() {
            let w = weights.map_or(1.0, |ws| ws[i] as f64);
            let row = &preds[i * k..i * k + k];
            let mut best = 0usize;
            for c in 1..k {
                if row[c] > row[best] {
                    best = c;
                }
            }
            if best != lab as usize {
                wrong += w;
            }
            wsum += w;
        }
        if wsum > 0.0 {
            wrong / wsum
        } else {
            0.0
        }
    }
}

/// Poisson negative log-likelihood (`poisson-nloglik`). Predictions are rates.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoissonNLogLik;

impl Metric for PoissonNLogLik {
    fn name(&self) -> &str {
        "poisson-nloglik"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let mut loss = 0.0;
        let mut wsum = 0.0;
        for i in 0..preds.len() {
            let w = weights.map_or(1.0, |ws| ws[i] as f64);
            let p = (preds[i] as f64).max(1e-8);
            let y = labels[i] as f64;
            loss += w * (p - y * p.ln());
            wsum += w;
        }
        if wsum > 0.0 {
            loss / wsum
        } else {
            0.0
        }
    }
}

/// Gamma negative log-likelihood (`gamma-nloglik`). Predictions are means.
#[derive(Debug, Clone, Copy, Default)]
pub struct GammaNLogLik;

impl Metric for GammaNLogLik {
    fn name(&self) -> &str {
        "gamma-nloglik"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let mut loss = 0.0;
        let mut wsum = 0.0;
        for i in 0..preds.len() {
            let w = weights.map_or(1.0, |ws| ws[i] as f64);
            let mu = (preds[i] as f64).max(1e-8);
            let y = labels[i] as f64;
            loss += w * (y / mu + mu.ln());
            wsum += w;
        }
        if wsum > 0.0 {
            loss / wsum
        } else {
            0.0
        }
    }
}

/// Tweedie negative log-likelihood (`tweedie-nloglik`) with variance power `rho`.
#[derive(Debug, Clone, Copy)]
pub struct TweedieNLogLik {
    rho: f64,
}

impl Metric for TweedieNLogLik {
    fn name(&self) -> &str {
        "tweedie-nloglik"
    }

    fn eval(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f64 {
        let mut loss = 0.0;
        let mut wsum = 0.0;
        for i in 0..preds.len() {
            let w = weights.map_or(1.0, |ws| ws[i] as f64);
            let mu = (preds[i] as f64).max(1e-8);
            let y = labels[i] as f64;
            let a = y * mu.powf(1.0 - self.rho) / (1.0 - self.rho);
            let b = mu.powf(2.0 - self.rho) / (2.0 - self.rho);
            loss += w * (-a + b);
            wsum += w;
        }
        if wsum > 0.0 {
            loss / wsum
        } else {
            0.0
        }
    }
}

/// Resolve a metric by name. `num_class` is used by multiclass metrics.
pub fn create_metric(name: &str, num_class: usize) -> Result<Box<dyn Metric>> {
    // Accept the XGBoost `tweedie-nloglik@1.5` suffix form.
    let (base, rho) = match name.split_once('@') {
        Some((b, r)) => (b, r.parse::<f64>().ok()),
        None => (name, None),
    };
    match base {
        "rmse" => Ok(Box::new(Rmse)),
        "mae" => Ok(Box::new(Mae)),
        "logloss" => Ok(Box::new(LogLoss)),
        "error" => Ok(Box::new(ErrorRate)),
        "auc" => Ok(Box::new(Auc)),
        "mlogloss" => Ok(Box::new(MLogLoss {
            num_class: num_class.max(2),
        })),
        "merror" => Ok(Box::new(MError {
            num_class: num_class.max(2),
        })),
        "poisson-nloglik" => Ok(Box::new(PoissonNLogLik)),
        "gamma-nloglik" => Ok(Box::new(GammaNLogLik)),
        "tweedie-nloglik" => Ok(Box::new(TweedieNLogLik {
            rho: rho.unwrap_or(1.5),
        })),
        other => Err(SequoiaError::unknown("metric", other)),
    }
}

/// Build the list of metrics to evaluate: the user's `eval_metric` list if any,
/// otherwise the single `default_name` supplied by the objective.
pub fn create_metrics(
    eval_metric: &[String],
    default_name: &str,
    num_class: usize,
) -> Result<Vec<Box<dyn Metric>>> {
    if eval_metric.is_empty() {
        Ok(vec![create_metric(default_name, num_class)?])
    } else {
        eval_metric
            .iter()
            .map(|n| create_metric(n, num_class))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rmse_basic() {
        let m = Rmse;
        // errors: 1, -1 -> mean sq 1 -> rmse 1
        assert_relative_eq!(m.eval(&[2.0, 0.0], &[1.0, 1.0], None), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn logloss_perfect_and_wrong() {
        let m = LogLoss;
        // near-perfect predictions -> ~0 loss
        let loss = m.eval(&[0.999999, 0.000001], &[1.0, 0.0], None);
        assert!(loss < 1e-4);
        // p=0.5 everywhere -> ln 2
        let loss = m.eval(&[0.5, 0.5], &[1.0, 0.0], None);
        assert_relative_eq!(loss, 2.0f64.ln(), epsilon = 1e-6);
    }

    #[test]
    fn error_rate_counts_misclassified() {
        let m = ErrorRate;
        // preds: 0.9->pos ok, 0.4->neg but label pos -> wrong, 0.2->neg ok
        assert_relative_eq!(
            m.eval(&[0.9, 0.4, 0.2], &[1.0, 1.0, 0.0], None),
            1.0 / 3.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn factory_defaults_to_objective_metric() {
        let ms = create_metrics(&[], "rmse", 0).unwrap();
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].name(), "rmse");
        assert!(create_metrics(&["nope".to_string()], "rmse", 0).is_err());
    }

    #[test]
    fn auc_ranks_perfectly_separable() {
        let m = Auc;
        // preds perfectly separate: positives above negatives -> AUC 1.
        let auc = m.eval(&[0.1, 0.2, 0.8, 0.9], &[0.0, 0.0, 1.0, 1.0], None);
        assert!((auc - 1.0).abs() < 1e-9);
        assert!(m.maximize());
    }

    #[test]
    fn mlogloss_and_merror() {
        // 2 rows, 3 classes. Confident correct predictions.
        let ml = MLogLoss { num_class: 3 };
        let me = MError { num_class: 3 };
        let preds = [0.8, 0.1, 0.1, 0.05, 0.9, 0.05];
        let labels = [0.0, 1.0];
        assert!(ml.eval(&preds, &labels, None) < 0.3);
        assert_eq!(me.eval(&preds, &labels, None), 0.0);
        // A wrong argmax raises merror.
        let labels_wrong = [1.0, 1.0];
        assert_eq!(me.eval(&preds, &labels_wrong, None), 0.5);
    }
}
