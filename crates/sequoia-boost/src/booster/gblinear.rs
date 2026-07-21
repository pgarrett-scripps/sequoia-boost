//! The linear (`gblinear`) booster: a generalized linear model fit by
//! coordinate descent.
//!
//! Instead of growing trees, `gblinear` keeps a per-output weight vector and
//! bias and improves them one coordinate at a time. Each round recomputes the
//! objective's gradients/Hessians from the current margins, then, for every
//! output and feature, applies the closed-form coordinate step
//!
//! ```text
//! G_f = Σ_i g_i x_if,   H_f = Σ_i h_i x_if²,   Δw = -(G_f + reg) / (H_f + λ)
//! ```
//!
//! soft-thresholded by the L1 penalty `alpha` and scaled by the learning rate
//! `eta`. The per-output bias is the intercept feature (`x ≡ 1`, unregularized).
//! Gradients and running margins are updated incrementally after each coordinate
//! change, matching XGBoost's `CoordinateUpdater`.
//!
//! Multiclass is supported directly: each output `k` gets its own weight column
//! and bias, fit from that output's gradient slice.

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::error::{Result, SequoiaError};
use crate::learner::LinearModel;
use crate::objective::{GradPair, Objective};

/// One feature's present entries, stored column-major as parallel `(row, value)`
/// vectors so a coordinate update touches only the rows where the feature is
/// present (missing entries contribute nothing).
struct Column {
    rows: Vec<u32>,
    vals: Vec<f32>,
}

/// Closed-form coordinate step for a single weight with L1 (`alpha`) and L2
/// (`lambda`) regularization. Mirrors XGBoost's `CoordinateDelta`: returns the
/// change in the weight (before the `eta` scaling), soft-thresholded so the step
/// never crosses past zero.
fn coordinate_delta(sum_grad: f64, sum_hess: f64, w: f64, alpha: f64, lambda: f64) -> f64 {
    if sum_hess < 1e-5 {
        return 0.0;
    }
    let sum_grad_l2 = sum_grad + lambda * w;
    let sum_hess_l2 = sum_hess + lambda;
    let tmp = w - sum_grad_l2 / sum_hess_l2;
    if tmp >= 0.0 {
        (-(sum_grad_l2 + alpha) / sum_hess_l2).max(-w)
    } else {
        (-(sum_grad_l2 - alpha) / sum_hess_l2).min(-w)
    }
}

/// Coordinate step for the (unregularized) bias / intercept term.
fn bias_delta(sum_grad: f64, sum_hess: f64) -> f64 {
    if sum_hess < 1e-5 {
        return 0.0;
    }
    -sum_grad / sum_hess
}

/// Fit a linear booster by coordinate descent.
///
/// `base_margin` is the scalar starting margin (the model's `base_score`), and
/// `n_out` is the number of outputs (`num_class` for multiclass, else 1). The
/// returned [`LinearModel`] holds `weights` laid out `[feature][output]` and a
/// per-output `bias`.
pub(crate) fn train_gblinear(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_round: usize,
    base_margin: f32,
    n_out: usize,
    objective: &dyn Objective,
) -> Result<LinearModel> {
    let n = dtrain.n_rows();
    let n_features = dtrain.n_cols();
    let labels = dtrain
        .labels()
        .ok_or(SequoiaError::EmptyDataset("gblinear: dtrain has no labels"))?;
    let weights = dtrain.weights();
    let group = dtrain.group();

    let eta = params.eta;
    let lambda = params.lambda;
    let alpha = params.alpha;

    // Column-major cache of present feature values.
    let mut cols: Vec<Column> = (0..n_features)
        .map(|_| Column {
            rows: Vec::new(),
            vals: Vec::new(),
        })
        .collect();
    for row in 0..n {
        for (f, col) in cols.iter_mut().enumerate() {
            if let Some(x) = dtrain.get(row, f) {
                if x != 0.0 {
                    col.rows.push(row as u32);
                    col.vals.push(x);
                }
            }
        }
    }

    let mut lin_weights = vec![0.0f32; n_features * n_out];
    let mut bias = vec![0.0f32; n_out];

    // Running margins [instance][output]; gradients recomputed each round, then
    // updated incrementally as each coordinate moves.
    let mut margin = vec![base_margin; n * n_out];
    let mut gpair = vec![GradPair::default(); n * n_out];

    for _round in 0..num_round {
        objective.gradient_grouped(&margin, labels, weights, group, &mut gpair);

        for k in 0..n_out {
            // 1. Bias (intercept) update: G = Σ g, H = Σ h.
            let mut g = 0.0f64;
            let mut h = 0.0f64;
            for i in 0..n {
                let gp = gpair[i * n_out + k];
                g += gp.grad as f64;
                h += gp.hess as f64;
            }
            let db = eta * bias_delta(g, h);
            if db != 0.0 {
                let db32 = db as f32;
                bias[k] += db32;
                for i in 0..n {
                    let gp = &mut gpair[i * n_out + k];
                    gp.grad += gp.hess * db32;
                    margin[i * n_out + k] += db32;
                }
            }

            // 2. Per-feature coordinate updates.
            for f in 0..n_features {
                let col = &cols[f];
                let mut g = 0.0f64;
                let mut h = 0.0f64;
                for (idx, &row) in col.rows.iter().enumerate() {
                    let x = col.vals[idx] as f64;
                    let gp = gpair[row as usize * n_out + k];
                    g += gp.grad as f64 * x;
                    h += gp.hess as f64 * x * x;
                }
                let w = lin_weights[f * n_out + k] as f64;
                let dw = eta * coordinate_delta(g, h, w, alpha, lambda);
                if dw == 0.0 {
                    continue;
                }
                let dw32 = dw as f32;
                lin_weights[f * n_out + k] += dw32;
                for (idx, &row) in col.rows.iter().enumerate() {
                    let x = col.vals[idx];
                    let gp = &mut gpair[row as usize * n_out + k];
                    gp.grad += gp.hess * x * dw32;
                    margin[row as usize * n_out + k] += x * dw32;
                }
            }
        }
    }

    Ok(LinearModel::new(lin_weights, bias))
}
