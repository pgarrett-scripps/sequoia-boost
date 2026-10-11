//! Learning objectives: gradients, Hessians, prediction transforms, and base
//! score estimation.
//!
//! Every objective implements [`Objective`]. Boosting works in *margin* space
//! (raw additive scores). The [`Objective::pred_transform`] maps margins to the
//! reported prediction (e.g. the logistic sigmoid). This mirrors XGBoost's
//! separation of `GetGradient` / `PredTransform`.

mod classification;
mod count;
mod custom;
mod multiclass;
mod ranking;
mod regression;

pub use classification::LogisticObjective;
pub use count::{GammaObjective, PoissonObjective, TweedieObjective};
pub use custom::CustomObjective;
pub use multiclass::SoftmaxObjective;
pub use ranking::LambdaMartObjective;
pub use regression::{PseudoHuberObjective, SquaredErrorObjective};

use crate::config::TrainingParams;
use crate::error::{Result, SequoiaError};

/// A first- and second-order gradient for one instance/output.
///
/// Stored as `f32` to match XGBoost's memory layout and to keep histogram
/// accumulation cache-friendly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct GradPair {
    /// First-order gradient of the loss w.r.t. the margin.
    pub grad: f32,
    /// Second-order gradient (Hessian) of the loss w.r.t. the margin.
    pub hess: f32,
}

impl GradPair {
    /// Construct a gradient pair.
    #[inline]
    pub fn new(grad: f32, hess: f32) -> Self {
        GradPair { grad, hess }
    }
}

/// Rows per parallel gradient chunk. A multiple of every vector kernel's block
/// (4 rows, and 4 values for any class count), so chunk boundaries fall where
/// the kernels' block boundaries already are and every element is computed
/// by the same path as in one whole-batch call.
const GRADIENT_CHUNK_ROWS: usize = 8192;

/// Run a row-independent gradient `kernel` over `n_rows` instances with
/// `n_outputs` values each, in parallel row chunks when the batch is large and
/// a thread pool is available. Every row's outputs depend only on that row,
/// and the chunking is fixed (not thread-count dependent): a short final
/// chunk is folded into the last full chunk so every row takes the same
/// vector/scalar path as in one whole-batch call, and the result is
/// identical.
pub(crate) fn rowwise_gradient<K>(
    n_rows: usize,
    n_outputs: usize,
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &mut [GradPair],
    kernel: K,
) where
    K: Fn(&[f32], &[f32], Option<&[f32]>, &mut [GradPair]) + Sync,
{
    let complete = n_rows
        .checked_mul(n_outputs)
        .is_some_and(|values| preds.len() == values && out.len() == values)
        && labels.len() == n_rows
        && weights.is_none_or(|w| w.len() == n_rows);
    if !complete
        || n_rows == 0
        || n_outputs == 0
        || n_rows < 2 * GRADIENT_CHUNK_ROWS
        || rayon::current_num_threads() <= 1
    {
        kernel(preds, labels, weights, out);
        return;
    }
    use rayon::prelude::*;
    let run_chunk = |first: usize, out: &mut [GradPair]| {
        let rows = out.len() / n_outputs;
        kernel(
            &preds[first * n_outputs..(first + rows) * n_outputs],
            &labels[first..first + rows],
            weights.map(|w| &w[first..first + rows]),
            out,
        );
    };
    let chunk_values = GRADIENT_CHUNK_ROWS * n_outputs;
    if out.len() % chunk_values == 0 {
        out.par_chunks_mut(chunk_values)
            .enumerate()
            .for_each(|(index, out)| run_chunk(index * GRADIENT_CHUNK_ROWS, out));
        return;
    }
    // A separate short tail chunk would compute its rows on the scalar path
    // where a whole-batch call vectorizes them (or vice versa); fold it into
    // the last full chunk so every row keeps the whole-batch vector/scalar
    // split — chunk starts stay multiples of every kernel block.
    let head_rows = (n_rows / GRADIENT_CHUNK_ROWS - 1) * GRADIENT_CHUNK_ROWS;
    let (head, tail) = out.split_at_mut(head_rows * n_outputs);
    rayon::join(
        || {
            head.par_chunks_mut(chunk_values)
                .enumerate()
                .for_each(|(index, out)| run_chunk(index * GRADIENT_CHUNK_ROWS, out));
        },
        || run_chunk(head_rows, tail),
    );
}

/// A differentiable learning objective.
///
/// Implementors are `Send + Sync` so gradient computation can be parallelized.
pub trait Objective: Send + Sync {
    /// The XGBoost-compatible objective name (e.g. `"reg:squarederror"`).
    fn name(&self) -> &str;

    /// Number of raw outputs produced per instance. `1` for regression and
    /// binary classification. It is `num_class` for multiclass objectives.
    fn n_outputs(&self) -> usize {
        1
    }

    /// Compute per-instance gradients and Hessians.
    ///
    /// `preds` holds raw margins laid out as `n_rows * n_outputs` (row-major by
    /// instance). `out` is written in the same layout. `weights`, if present,
    /// scales each instance's contribution.
    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    );

    /// Compute gradients with optional query-group structure.
    ///
    /// Learning-to-rank objectives (LambdaMART) override this to form document
    /// pairs *within* each group supplied by `group`. The default forwards to
    /// [`Objective::gradient`], ignoring the grouping. This is correct for all
    /// non-ranking objectives.
    fn gradient_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        _group: Option<&crate::data::GroupInfo>,
        out: &mut [GradPair],
    ) {
        self.gradient(preds, labels, weights, out)
    }

    /// Transform raw margins into reported predictions, in place. Default is the
    /// identity (used by squared-error regression).
    fn pred_transform(&self, _preds: &mut [f32]) {}

    /// Estimate the optimal constant prediction in *margin* space, used to
    /// initialize `base_score` when the user does not supply one.
    fn base_margin(&self, labels: &[f32], weights: Option<&[f32]>) -> f32;

    /// Convert a user-supplied `base_score` (given in prediction space) into
    /// margin space via the objective's inverse link. Default is the identity.
    /// objectives with a link function (e.g. logistic) override it.
    fn prob_to_margin(&self, base_score: f32) -> f32 {
        base_score
    }

    /// The default evaluation metric name for this objective.
    fn default_metric(&self) -> &str;
}

/// Weighted mean of `labels`, or the plain mean when `weights` is `None`.
/// Shared by objectives that initialize from the label mean.
pub(crate) fn weighted_label_mean(labels: &[f32], weights: Option<&[f32]>) -> f64 {
    match weights {
        Some(w) => {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (l, wi) in labels.iter().zip(w) {
                num += (*l as f64) * (*wi as f64);
                den += *wi as f64;
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        }
        None => {
            if labels.is_empty() {
                0.0
            } else {
                labels.iter().map(|l| *l as f64).sum::<f64>() / labels.len() as f64
            }
        }
    }
}

/// Resolve an objective by name, configured from `params`.
pub fn create_objective(params: &TrainingParams) -> Result<Box<dyn Objective>> {
    match params.objective.as_str() {
        "reg:squarederror" | "reg:linear" => Ok(Box::new(SquaredErrorObjective)),
        "reg:pseudohubererror" => Ok(Box::new(PseudoHuberObjective)),
        "binary:logistic" | "reg:logistic" => Ok(Box::new(LogisticObjective::new(
            params.scale_pos_weight as f32,
        ))),
        "multi:softmax" | "multi:softprob" => {
            if params.num_class < 2 {
                return Err(SequoiaError::invalid_param(
                    "num_class",
                    "multiclass objectives require num_class >= 2",
                ));
            }
            let prob = params.objective == "multi:softprob";
            Ok(Box::new(SoftmaxObjective::new(params.num_class, prob)))
        }
        "count:poisson" => Ok(Box::new(PoissonObjective::new(
            if params.max_delta_step > 0.0 {
                params.max_delta_step as f32
            } else {
                0.7
            },
        ))),
        "reg:gamma" => Ok(Box::new(GammaObjective)),
        "reg:tweedie" => Ok(Box::new(TweedieObjective::default())),
        "rank:pairwise" => Ok(Box::new(LambdaMartObjective::pairwise())),
        "rank:ndcg" => Ok(Box::new(LambdaMartObjective::ndcg())),
        "rank:map" => Ok(Box::new(LambdaMartObjective::map())),
        other => Err(SequoiaError::unknown("objective", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_mean_basic() {
        let labels = [1.0f32, 3.0];
        assert!((weighted_label_mean(&labels, None) - 2.0).abs() < 1e-9);
        let w = [3.0f32, 1.0];
        // (1*3 + 3*1) / 4 = 1.5
        assert!((weighted_label_mean(&labels, Some(&w)) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn factory_resolves_known_and_rejects_unknown() {
        let p = TrainingParams::builder()
            .objective("reg:squarederror")
            .build_unchecked();
        assert_eq!(create_objective(&p).unwrap().name(), "reg:squarederror");
        let p = TrainingParams::builder()
            .objective("nope:whatever")
            .build_unchecked();
        assert!(create_objective(&p).is_err());
    }

    /// Parallel row chunks must reproduce the whole-batch gradient bit for bit
    /// for every objective routed through the chunked helper. Lengths are not
    /// multiples of the chunk or of any vector block. The logistic case sweeps
    /// every short-tail residue r in 1..=15, where a separate final chunk
    /// would fall below the vector dispatch length and compute its rows on
    /// the scalar path. The sweep also pins the structural invariant the fold
    /// relies on: chunk boundaries stay multiples of every kernel block, so
    /// a uniform `chunk + tail` chunking would fail here.
    #[test]
    fn chunked_gradients_match_whole_batch() {
        let c = GRADIENT_CHUNK_ROWS;
        let objectives: Vec<(Box<dyn Objective>, usize, Vec<usize>)> = vec![
            (Box::new(SquaredErrorObjective), 1, vec![2 * c + 4097]),
            (
                Box::new(LogisticObjective::new(1.5)),
                1,
                (1..=15).map(|r| 2 * c + r).collect(),
            ),
            (Box::new(SoftmaxObjective::new(2, true)), 2, vec![2 * c + 4]),
            (Box::new(SoftmaxObjective::new(3, true)), 3, vec![2 * c + 4]),
            (
                Box::new(SoftmaxObjective::new(9, false)),
                9,
                vec![2 * c + 1],
            ),
        ];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (objective, k, ns) in objectives {
            for n in ns {
                let preds: Vec<f32> = (0..n * k)
                    .map(|i| ((i * 7919) % 2003) as f32 / 97.0 - 10.0)
                    .collect();
                let labels: Vec<f32> = (0..n)
                    .map(|i| {
                        if k == 1 {
                            (i % 2) as f32
                        } else {
                            (i % k) as f32
                        }
                    })
                    .collect();
                let weights: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
                for weights in [None, Some(weights.as_slice())] {
                    let mut whole = vec![GradPair::default(); n * k];
                    // A single-thread pool takes the whole-batch path.
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .unwrap()
                        .install(|| objective.gradient(&preds, &labels, weights, &mut whole));
                    let mut chunked = vec![GradPair::default(); n * k];
                    pool.install(|| objective.gradient(&preds, &labels, weights, &mut chunked));
                    for (i, (a, b)) in whole.iter().zip(&chunked).enumerate() {
                        assert_eq!(
                            a.grad.to_bits(),
                            b.grad.to_bits(),
                            "{} grad {i}",
                            objective.name()
                        );
                        assert_eq!(
                            a.hess.to_bits(),
                            b.hess.to_bits(),
                            "{} hess {i}",
                            objective.name()
                        );
                    }
                }
            }
        }
    }
}
