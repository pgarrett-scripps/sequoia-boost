//! Learning-to-rank objectives (LambdaMART).
//!
//! These objectives operate on *query groups*: contiguous blocks of rows that
//! belong to the same query (see [`crate::data::GroupInfo`]). Within each group
//! we form pairs of documents with different relevance labels and apply the
//! pairwise-logistic (RankNet) gradient. For the `rank:ndcg` and `rank:map`
//! variants each pair's gradient is additionally scaled by the magnitude of the
//! change in the ranking metric (NDCG or MAP) that would result from swapping
//! the two documents — the "lambda" weighting that turns RankNet into
//! LambdaMART.
//!
//! The objective is *stateless*: query-group boundaries are supplied at
//! gradient time through [`Objective::gradient_grouped`]. When no group
//! information is available the whole batch is treated as a single group.

use super::{GradPair, Objective};
use crate::data::GroupInfo;

/// Maximum number of document pairs formed per query group.
///
/// A group with `m` documents can yield up to `m*(m-1)/2` pairs, which is
/// quadratic and can explode for large result lists. When the number of
/// candidate pairs exceeds this cap we keep each candidate pair independently
/// with probability `CAP / total_pairs`, giving roughly `CAP` pairs while
/// remaining an unbiased estimate of the full pairwise gradient. Sampling is
/// deterministic (seeded per group) so training is reproducible.
const MAX_PAIRS_PER_GROUP: usize = 4096;

/// Which ranking loss the LambdaMART objective optimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RankMode {
    /// Plain pairwise logistic loss (`rank:pairwise`); every pair weighted 1.
    Pairwise,
    /// Pairs weighted by |ΔNDCG| (`rank:ndcg`).
    Ndcg,
    /// Pairs weighted by |ΔMAP| (`rank:map`).
    Map,
}

/// The LambdaMART pairwise ranking objective.
///
/// Supports three XGBoost-compatible modes: `rank:pairwise`, `rank:ndcg`, and
/// `rank:map`. See the [module documentation](self) for the algorithm.
#[derive(Debug, Clone, Copy)]
pub struct LambdaMartObjective {
    mode: RankMode,
}

impl LambdaMartObjective {
    /// Plain pairwise logistic ranking (`rank:pairwise`).
    pub fn pairwise() -> Self {
        LambdaMartObjective {
            mode: RankMode::Pairwise,
        }
    }

    /// NDCG-weighted LambdaMART (`rank:ndcg`).
    pub fn ndcg() -> Self {
        LambdaMartObjective {
            mode: RankMode::Ndcg,
        }
    }

    /// MAP-weighted LambdaMART (`rank:map`).
    pub fn map() -> Self {
        LambdaMartObjective {
            mode: RankMode::Map,
        }
    }

    /// Accumulate gradients for one contiguous query group spanning rows
    /// `start..end` of `preds`/`labels`, writing into the same slice of `out`.
    fn accumulate_group(
        &self,
        preds: &[f32],
        labels: &[f32],
        start: usize,
        end: usize,
        out: &mut [GradPair],
    ) {
        let m = end - start;
        if m < 2 {
            return;
        }

        // Local score/label views for this group.
        let scores: Vec<f64> = (start..end).map(|i| preds[i] as f64).collect();
        let labs: Vec<f64> = (start..end).map(|i| labels[i] as f64).collect();

        // Rank documents by descending score; `pos[local]` is the 0-based rank.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let mut pos = vec![0usize; m];
        for (rank, &local) in order.iter().enumerate() {
            pos[local] = rank;
        }

        // Precompute the metric context used to weight pairs.
        let ctx = MetricCtx::build(self.mode, &labs, &order, &pos);

        // Candidate-pair sampling probability (1.0 unless the group is huge).
        let total_pairs = m * (m - 1) / 2;
        let keep_prob = if total_pairs > MAX_PAIRS_PER_GROUP {
            MAX_PAIRS_PER_GROUP as f64 / total_pairs as f64
        } else {
            1.0
        };
        // Deterministic per-group RNG seeded from the group size and layout.
        let mut rng = SplitMix64::new(0x9E37_79B9_7F4A_7C15 ^ (start as u64).wrapping_mul(2654435761));

        for a in 0..m {
            for b in (a + 1)..m {
                // Only pairs with different relevance contribute.
                if labs[a] == labs[b] {
                    continue;
                }
                if keep_prob < 1.0 && rng.next_f64() >= keep_prob {
                    continue;
                }
                // `hi` is the more-relevant document, which we want ranked above.
                let (hi, lo) = if labs[a] > labs[b] { (a, b) } else { (b, a) };
                let s_hi = scores[hi];
                let s_lo = scores[lo];

                // Weight this pair by the metric delta of swapping hi and lo.
                let delta = ctx.delta(hi, lo);
                if delta == 0.0 {
                    continue;
                }

                // Pairwise-logistic gradient. rho = P(lo ranked above hi).
                let rho = 1.0 / (1.0 + (s_hi - s_lo).exp());
                let grad = (rho * delta) as f32;
                let hess = (rho * (1.0 - rho) * delta).max(1e-16) as f32;

                // Push the relevant doc up (negative gradient) and the other down.
                out[start + hi].grad -= grad;
                out[start + hi].hess += hess;
                out[start + lo].grad += grad;
                out[start + lo].hess += hess;
            }
        }
    }

    /// Shared gradient computation over an optional group layout.
    fn compute(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&GroupInfo>,
        out: &mut [GradPair],
    ) {
        for g in out.iter_mut() {
            *g = GradPair::default();
        }

        match group {
            Some(g) if g.num_rows() == preds.len() => {
                for (start, end) in g.iter_ranges() {
                    self.accumulate_group(preds, labels, start, end, out);
                }
            }
            // No usable group info: treat the whole batch as one query.
            _ => self.accumulate_group(preds, labels, 0, preds.len(), out),
        }

        // Optional per-document weighting of the aggregated gradient.
        if let Some(w) = weights {
            for (o, &wi) in out.iter_mut().zip(w) {
                o.grad *= wi;
                o.hess *= wi;
            }
        }
    }
}

impl Objective for LambdaMartObjective {
    fn name(&self) -> &str {
        match self.mode {
            RankMode::Pairwise => "rank:pairwise",
            RankMode::Ndcg => "rank:ndcg",
            RankMode::Map => "rank:map",
        }
    }

    fn gradient(&self, preds: &[f32], labels: &[f32], weights: Option<&[f32]>, out: &mut [GradPair]) {
        // Without group info the whole batch is one query group.
        self.compute(preds, labels, weights, None, out);
    }

    fn gradient_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&GroupInfo>,
        out: &mut [GradPair],
    ) {
        self.compute(preds, labels, weights, group, out);
    }

    fn base_margin(&self, _labels: &[f32], _weights: Option<&[f32]>) -> f32 {
        0.0
    }

    fn default_metric(&self) -> &str {
        match self.mode {
            RankMode::Ndcg => "ndcg",
            RankMode::Pairwise | RankMode::Map => "map",
        }
    }
}

/// Per-group precomputed data used to compute |ΔMetric| for a candidate pair.
enum MetricCtx {
    /// Plain pairwise loss: every pair weighted 1.
    Uniform,
    /// NDCG weighting: gains per local doc, positions, and the ideal DCG.
    Ndcg {
        gains: Vec<f64>,
        pos: Vec<usize>,
        idcg: f64,
    },
    /// MAP weighting: binary relevance in score order plus prefix sums.
    Map {
        pos: Vec<usize>,
        rel_sorted: Vec<f64>,
        cum: Vec<usize>,
        prefix: Vec<f64>,
        num_rel: usize,
    },
}

impl MetricCtx {
    fn build(mode: RankMode, labs: &[f64], order: &[usize], pos: &[usize]) -> MetricCtx {
        match mode {
            RankMode::Pairwise => MetricCtx::Uniform,
            RankMode::Ndcg => {
                let gains: Vec<f64> = labs.iter().map(|&l| gain(l)).collect();
                // Ideal DCG: gains sorted descending, standard log2 discount.
                let mut ideal = gains.clone();
                ideal.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let idcg: f64 = ideal
                    .iter()
                    .enumerate()
                    .map(|(p, &g)| g * discount(p))
                    .sum();
                MetricCtx::Ndcg {
                    gains,
                    pos: pos.to_vec(),
                    idcg,
                }
            }
            RankMode::Map => {
                let m = labs.len();
                // Relevance in score order (relevant iff label > 0).
                let rel_sorted: Vec<f64> = order
                    .iter()
                    .map(|&local| if labs[local] > 0.0 { 1.0 } else { 0.0 })
                    .collect();
                let mut cum = vec![0usize; m];
                let mut prefix = vec![0.0f64; m + 1];
                let mut acc = 0usize;
                for p in 0..m {
                    acc += rel_sorted[p] as usize;
                    cum[p] = acc;
                    prefix[p + 1] = prefix[p] + rel_sorted[p] / (p + 1) as f64;
                }
                MetricCtx::Map {
                    pos: pos.to_vec(),
                    rel_sorted,
                    cum,
                    prefix,
                    num_rel: acc,
                }
            }
        }
    }

    /// Magnitude of the metric change from swapping local docs `hi` and `lo`.
    fn delta(&self, hi: usize, lo: usize) -> f64 {
        match self {
            MetricCtx::Uniform => 1.0,
            MetricCtx::Ndcg { gains, pos, idcg } => {
                if *idcg <= 0.0 {
                    return 0.0;
                }
                let d = (gains[hi] - gains[lo]) * (discount(pos[hi]) - discount(pos[lo]));
                (d / idcg).abs()
            }
            MetricCtx::Map {
                pos,
                rel_sorted,
                cum,
                prefix,
                num_rel,
            } => {
                if *num_rel == 0 {
                    return 0.0;
                }
                let (mut p, mut q) = (pos[hi], pos[lo]);
                if p > q {
                    std::mem::swap(&mut p, &mut q);
                }
                delta_map(rel_sorted, cum, prefix, *num_rel, p, q)
            }
        }
    }
}

/// NDCG gain of a (possibly graded) relevance label: `2^rel - 1`.
#[inline]
fn gain(rel: f64) -> f64 {
    (2.0f64).powf(rel) - 1.0
}

/// NDCG position discount for 0-based rank `p`: `1 / log2(p + 2)`.
#[inline]
fn discount(p: usize) -> f64 {
    1.0 / ((p + 2) as f64).log2()
}

/// |ΔAP| from swapping the documents currently at score-ranks `p < q`.
///
/// `rel_sorted` is binary relevance in score order, `cum[k]` the number of
/// relevant documents in ranks `0..=k`, and `prefix[k] = Σ_{t<k} rel[t]/(t+1)`.
fn delta_map(
    rel_sorted: &[f64],
    cum: &[usize],
    prefix: &[f64],
    num_rel: usize,
    p: usize,
    q: usize,
) -> f64 {
    let rp = rel_sorted[p];
    let rq = rel_sorted[q];
    if rp == rq {
        return 0.0;
    }
    let cum_p = cum[p] as f64;
    let cum_q = cum[q] as f64;
    // Change of the average-precision numerator (AP * num_rel):
    //   term at rank p, the middle span (p, q), and term at rank q.
    let at_p = (rq * (cum_p - rp + rq) - rp * cum_p) / (p + 1) as f64;
    let middle = (rq - rp) * (prefix[q] - prefix[p + 1]);
    let at_q = (rp - rq) * cum_q / (q + 1) as f64;
    ((at_p + middle + at_q) / num_rel as f64).abs()
}

/// A tiny deterministic SplitMix64 PRNG for reproducible pair subsampling.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GroupInfo;

    #[test]
    fn pairwise_pushes_relevant_up() {
        // One group of 3 docs, labels 2 > 1 > 0, all scores equal at start.
        let obj = LambdaMartObjective::ndcg();
        let preds = [0.0f32, 0.0, 0.0];
        let labels = [2.0f32, 1.0, 0.0];
        let g = GroupInfo::from_sizes(&[3]);
        let mut out = vec![GradPair::default(); 3];
        obj.gradient_grouped(&preds, &labels, None, Some(&g), &mut out);
        // Negative gradient => leaf value positive => score goes up.
        // Most-relevant doc should get the most-negative gradient.
        assert!(out[0].grad < out[1].grad, "{:?}", out);
        assert!(out[1].grad < out[2].grad, "{:?}", out);
        assert!(out[0].grad < 0.0 && out[2].grad > 0.0);
        // Hessians are non-negative.
        assert!(out.iter().all(|g| g.hess >= 0.0));
    }

    #[test]
    fn no_pairs_when_all_labels_equal() {
        let obj = LambdaMartObjective::pairwise();
        let preds = [0.5f32, -0.2, 1.0];
        let labels = [1.0f32, 1.0, 1.0];
        let g = GroupInfo::from_sizes(&[3]);
        let mut out = vec![GradPair::default(); 3];
        obj.gradient_grouped(&preds, &labels, None, Some(&g), &mut out);
        assert!(out.iter().all(|g| g.grad == 0.0 && g.hess == 0.0));
    }

    #[test]
    fn groups_are_independent() {
        // Two groups; a cross-group pair must never be formed.
        let obj = LambdaMartObjective::pairwise();
        let preds = [0.0f32, 0.0, 0.0, 0.0];
        let labels = [1.0f32, 0.0, 0.0, 1.0];
        let g = GroupInfo::from_sizes(&[2, 2]);
        let mut out = vec![GradPair::default(); 4];
        obj.gradient_grouped(&preds, &labels, None, Some(&g), &mut out);
        // Within each group the relevant doc is pushed up, the other down.
        assert!(out[0].grad < 0.0 && out[1].grad > 0.0);
        assert!(out[3].grad < 0.0 && out[2].grad > 0.0);
    }

    #[test]
    fn map_delta_only_relevant_vs_nonrelevant() {
        // Labels 2 and 3 are both "relevant" -> MAP delta is 0 for that pair.
        let ctx = MetricCtx::build(
            RankMode::Map,
            &[3.0, 2.0],
            &[0, 1], // score order
            &[0, 1], // positions
        );
        assert_eq!(ctx.delta(0, 1), 0.0);
    }
}
