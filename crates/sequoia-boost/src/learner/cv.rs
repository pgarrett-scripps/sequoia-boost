//! K-fold cross-validation, mirroring `xgboost.cv`.

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::error::{Result, SequoiaError};
use crate::learner::train::train_with_eval;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::BTreeMap;

/// Per-metric cross-validation history, aggregated across folds.
#[derive(Debug, Clone)]
pub struct CvResult {
    /// Metric name.
    pub metric: String,
    /// Mean held-out metric value per boosting round.
    pub test_mean: Vec<f64>,
    /// Standard deviation of the held-out metric per round.
    pub test_std: Vec<f64>,
}

/// Run `nfold` cross-validation, returning one [`CvResult`] per evaluation
/// metric. Every fold trains for the full `num_boost_round` rounds (no early
/// stopping); the metric list comes from `params.eval_metric` or the objective's
/// default.
pub fn cv(
    params: &TrainingParams,
    data: &DMatrix,
    num_boost_round: usize,
    nfold: usize,
    seed: u64,
) -> Result<Vec<CvResult>> {
    if nfold < 2 {
        return Err(SequoiaError::invalid_param("nfold", "must be >= 2"));
    }
    let n = data.n_rows();
    if n < nfold {
        return Err(SequoiaError::invalid_param("nfold", "more folds than rows"));
    }

    // Shuffle then round-robin assign rows to folds.
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed));
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); nfold];
    for (i, &row) in order.iter().enumerate() {
        folds[i % nfold].push(row);
    }

    // metric name -> per-round vector of per-fold values.
    let mut collected: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();

    for f in 0..nfold {
        let test_rows = &folds[f];
        let train_rows: Vec<usize> = folds
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != f)
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        let dtrain = data.select_rows(&train_rows)?;
        let dtest = data.select_rows(test_rows)?;
        let res = train_with_eval(params, &dtrain, num_boost_round, &[(&dtest, "test")], None)?;
        for round in &res.history {
            for (_ds, metric, value) in &round.scores {
                let entry = collected.entry(metric.clone()).or_default();
                if entry.len() <= round.iteration {
                    entry.resize(round.iteration + 1, Vec::new());
                }
                entry[round.iteration].push(*value);
            }
        }
    }

    // Aggregate mean/std across folds per round.
    let mut out = Vec::new();
    for (metric, per_round) in collected {
        let mut mean = Vec::with_capacity(per_round.len());
        let mut std = Vec::with_capacity(per_round.len());
        for vals in &per_round {
            let m = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
            let var =
                vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / vals.len().max(1) as f64;
            mean.push(m);
            std.push(var.sqrt());
        }
        out.push(CvResult {
            metric,
            test_mean: mean,
            test_std: std,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cv_reports_decreasing_rmse() {
        // Simple learnable data.
        let n = 200;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let xi = i as f32 / n as f32;
            x.push(xi);
            y.push(if xi > 0.5 { 1.0 } else { 0.0 });
        }
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();

        let results = cv(&params, &d, 30, 5, 42).unwrap();
        assert_eq!(results.len(), 1);
        let rmse = &results[0];
        assert_eq!(rmse.metric, "rmse");
        assert_eq!(rmse.test_mean.len(), 30);
        // Held-out error should drop from first to last round.
        assert!(rmse.test_mean[29] < rmse.test_mean[0]);
        // Std is non-negative and finite.
        assert!(rmse.test_std.iter().all(|s| s.is_finite() && *s >= 0.0));
    }
}
