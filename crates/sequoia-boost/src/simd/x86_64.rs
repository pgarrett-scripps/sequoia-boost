//! AVX2/FMA kernels for x86-64. The split-gain scan is a pure prefilter whose
//! accepted result is bit-identical to the scalar path it replaces. The
//! transcendental kernels mirror the NEON formulas and stay within a few f32
//! ULPs of the scalar library functions.

use super::{scalar, sigmoid_scalar, SplitCandidate};
use crate::objective::GradPair;
use crate::tree::gain::{calc_gain, GradStats, RegParams};
use std::arch::x86_64::*;

const _: () = assert!(std::mem::size_of::<GradStats>() == 2 * std::mem::size_of::<f64>());
const _: () = assert!(std::mem::size_of::<GradPair>() == 2 * std::mem::size_of::<f32>());

/// f32 lanes per vector.
const WIDTH: usize = 8;
/// Inputs with a larger magnitude take the scalar path, as on NEON.
const MAX_FAST_EXP_INPUT: f32 = 80.0;

/// Exponential for finite f32 lanes in [-80, 80]: range reduction to
/// [-ln(2)/2, ln(2)/2] and a seventh-order polynomial (Estrin pairs for
/// latency), as the NEON kernel.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_f32(value: __m256) -> __m256 {
    // SAFETY: the caller guarantees AVX2/FMA support; all operations use
    // registers.
    unsafe {
        let multiply: unsafe fn(__m256, __m256) -> __m256 = _mm256_mul_ps;
        let exponent =
            _mm256_cvtps_epi32(multiply(value, _mm256_set1_ps(std::f32::consts::LOG2_E)));
        let exponent_f32 = _mm256_cvtepi32_ps(exponent);
        // Split ln(2) so the range reduction loses fewer low bits.
        let reduced = _mm256_fnmadd_ps(exponent_f32, _mm256_set1_ps(0.693_359_4), value);
        let reduced = _mm256_fmadd_ps(exponent_f32, _mm256_set1_ps(2.121_944_4e-4), reduced);

        let c = |x: f32| _mm256_set1_ps(x);
        let polynomial = {
            let squared = multiply(reduced, reduced);
            let fourth = multiply(squared, squared);
            let pair_0 = _mm256_add_ps(c(1.0), reduced);
            let pair_1 = _mm256_fmadd_ps(c(1.0 / 6.0), reduced, c(0.5));
            let pair_2 = _mm256_fmadd_ps(c(1.0 / 120.0), reduced, c(1.0 / 24.0));
            let pair_3 = _mm256_fmadd_ps(c(1.0 / 5_040.0), reduced, c(1.0 / 720.0));
            let low = _mm256_fmadd_ps(pair_1, squared, pair_0);
            let high = _mm256_fmadd_ps(pair_3, squared, pair_2);
            _mm256_fmadd_ps(high, fourth, low)
        };

        let exponent_bits =
            _mm256_slli_epi32::<23>(_mm256_add_epi32(exponent, _mm256_set1_epi32(127)));
        multiply(polynomial, _mm256_castsi256_ps(exponent_bits))
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn abs_f32(value: __m256) -> __m256 {
    // SAFETY: the caller guarantees AVX2 support; register-only.
    unsafe {
        let and_not: unsafe fn(__m256, __m256) -> __m256 = _mm256_andnot_ps;
        and_not(_mm256_set1_ps(-0.0), value)
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_f32(value: __m256) -> __m256 {
    // SAFETY: the caller guarantees AVX2/FMA support; register-only.
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let exp = exp_f32(_mm256_sub_ps(_mm256_setzero_ps(), abs_f32(value)));
        let denominator = _mm256_add_ps(one, exp);
        let positive = _mm256_div_ps(one, denominator);
        let negative = _mm256_div_ps(exp, denominator);
        let non_negative = _mm256_cmp_ps::<_CMP_GE_OQ>(value, _mm256_setzero_ps());
        _mm256_blendv_ps(negative, positive, non_negative)
    }
}

/// Whether every lane is finite with magnitude at most [`MAX_FAST_EXP_INPUT`].
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn regular_input(value: __m256) -> bool {
    // SAFETY: the caller guarantees AVX2 support; register-only.
    unsafe {
        let in_range =
            _mm256_cmp_ps::<_CMP_LE_OQ>(abs_f32(value), _mm256_set1_ps(MAX_FAST_EXP_INPUT));
        _mm256_movemask_ps(in_range) == 0xFF
    }
}

/// Store eight `(grad, hess)` pairs row-major as `GradPair`s.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn store_pairs(dest: *mut GradPair, grad: __m256, hess: __m256) {
    // SAFETY: the caller guarantees AVX2 support and eight writable pairs
    // at `dest`; GradPair is repr(C) with two adjacent f32 fields.
    unsafe {
        let low = _mm256_unpacklo_ps(grad, hess); // g0 h0 g1 h1 | g4 h4 g5 h5
        let high = _mm256_unpackhi_ps(grad, hess); // g2 h2 g3 h3 | g6 h6 g7 h7
        let dest = dest.cast::<f32>();
        _mm256_storeu_ps(dest, _mm256_permute2f128_ps::<0x20>(low, high));
        _mm256_storeu_ps(dest.add(WIDTH), _mm256_permute2f128_ps::<0x31>(low, high));
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn exp_inplace(values: &mut [f32]) {
    // SAFETY: the caller guarantees AVX2/FMA support; every vector access is
    // bounded by the loop condition.
    unsafe {
        let mut index = 0;
        while index + WIDTH <= values.len() {
            let input = _mm256_loadu_ps(values.as_ptr().add(index));
            if regular_input(input) {
                _mm256_storeu_ps(values.as_mut_ptr().add(index), exp_f32(input));
            } else {
                for value in &mut values[index..index + WIDTH] {
                    *value = value.exp();
                }
            }
            index += WIDTH;
        }
        for value in &mut values[index..] {
            *value = value.exp();
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn sigmoid_inplace(values: &mut [f32]) {
    // SAFETY: as `exp_inplace`.
    unsafe {
        let mut index = 0;
        while index + WIDTH <= values.len() {
            let input = _mm256_loadu_ps(values.as_ptr().add(index));
            if regular_input(input) {
                _mm256_storeu_ps(values.as_mut_ptr().add(index), sigmoid_f32(input));
            } else {
                for value in &mut values[index..index + WIDTH] {
                    *value = sigmoid_scalar(*value);
                }
            }
            index += WIDTH;
        }
        for value in &mut values[index..] {
            *value = sigmoid_scalar(*value);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn logistic_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees AVX2/FMA support and that `labels`,
    // `weights` and `out` cover `preds.len()` elements; the loop condition
    // bounds every vector access.
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let scale = _mm256_set1_ps(scale_pos_weight);
        let min_hess_vector = _mm256_set1_ps(min_hess);
        let mut index = 0;
        while index + WIDTH <= preds.len() {
            let pred = _mm256_loadu_ps(preds.as_ptr().add(index));
            if !regular_input(pred) {
                scalar::logistic_gradient(
                    preds,
                    labels,
                    weights,
                    (scale_pos_weight, min_hess),
                    out,
                    index..index + WIDTH,
                );
                index += WIDTH;
                continue;
            }
            let label = _mm256_loadu_ps(labels.as_ptr().add(index));
            let probability = sigmoid_f32(pred);
            let mut weight = match weights {
                Some(values) => _mm256_loadu_ps(values.as_ptr().add(index)),
                None => one,
            };
            let positive = _mm256_cmp_ps::<_CMP_EQ_OQ>(label, one);
            weight = _mm256_mul_ps(weight, _mm256_blendv_ps(one, scale, positive));
            let grad = _mm256_mul_ps(_mm256_sub_ps(probability, label), weight);
            let hess = _mm256_mul_ps(
                _mm256_max_ps(
                    _mm256_mul_ps(probability, _mm256_sub_ps(one, probability)),
                    min_hess_vector,
                ),
                weight,
            );
            store_pairs(out.as_mut_ptr().add(index), grad, hess);
            index += WIDTH;
        }
        scalar::logistic_gradient(
            preds,
            labels,
            weights,
            (scale_pos_weight, min_hess),
            out,
            index..preds.len(),
        );
    }
}

/// Row-local horizontal reduction of a vector holding `WIDTH / K` complete
/// rows of `K` classes (`K` is 2 or 4): every lane receives its row's result.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_reduce<const K: usize>(
    value: __m256,
    op: unsafe fn(__m256, __m256) -> __m256,
) -> __m256 {
    // SAFETY: the caller guarantees AVX2 support; register-only permutes.
    unsafe {
        // Swap adjacent lanes: pairs are complete rows for K = 2.
        let reduced = op(value, _mm256_permute_ps::<0b10_11_00_01>(value));
        if K == 2 {
            reduced
        } else {
            // Swap the halves of each 128-bit lane: quads are rows for K = 4.
            op(reduced, _mm256_permute_ps::<0b01_00_11_10>(reduced))
        }
    }
}

/// Softmax of the `WIDTH / K` rows held in one vector, or `None` when a row is
/// non-finite or spans more than [`MAX_FAST_EXP_INPUT`].
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn short_softmax_batch<const K: usize>(preds: *const f32) -> Option<__m256> {
    // SAFETY: the caller guarantees AVX2/FMA support and `WIDTH` readable
    // values at `preds`.
    unsafe {
        let values = _mm256_loadu_ps(preds);
        let maximum = row_reduce::<K>(values, _mm256_max_ps);
        let minimum = row_reduce::<K>(values, _mm256_min_ps);
        // NaNs propagate through min/max and infinities give a non-finite
        // range, so this ordered comparison fails for such rows.
        let regular = _mm256_cmp_ps::<_CMP_LE_OQ>(
            _mm256_sub_ps(maximum, minimum),
            _mm256_set1_ps(MAX_FAST_EXP_INPUT),
        );
        if _mm256_movemask_ps(regular) != 0xFF {
            return None;
        }
        let exp = exp_f32(_mm256_sub_ps(values, maximum));
        let sum = row_reduce::<K>(exp, _mm256_add_ps);
        Some(_mm256_mul_ps(exp, _mm256_div_ps(_mm256_set1_ps(1.0), sum)))
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn short_softmax_rows<const K: usize>(values: &mut [f32]) {
    // SAFETY: the caller guarantees AVX2/FMA support and K in {2, 4};
    // `chunks_exact_mut` bounds every vector access to complete rows.
    unsafe {
        let mut batches = values.chunks_exact_mut(WIDTH);
        for batch in &mut batches {
            match short_softmax_batch::<K>(batch.as_ptr()) {
                Some(probabilities) => _mm256_storeu_ps(batch.as_mut_ptr(), probabilities),
                None => {
                    for row in batch.chunks_mut(K) {
                        super::softmax_scalar(row);
                    }
                }
            }
        }
        for row in batches.into_remainder().chunks_mut(K) {
            super::softmax_scalar(row);
        }
    }
}

/// Broadcast the per-row `values` (labels or weights) of the `WIDTH / K` rows
/// in a batch across their `K` lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn broadcast_rows<const K: usize>(values: *const f32) -> __m256 {
    // SAFETY: the caller guarantees AVX2 support and `WIDTH / K` readable
    // values.
    unsafe {
        let (loaded, index) = if K == 2 {
            (
                _mm_loadu_ps(values),
                _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3),
            )
        } else {
            (
                _mm_castpd_ps(_mm_load_sd(values.cast::<f64>())),
                _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1),
            )
        };
        _mm256_permutevar8x32_ps(_mm256_castps128_ps256(loaded), index)
    }
}

/// `1.0` in the lanes whose class equals `label as usize`, following Rust's
/// saturating float-to-integer cast: `NaN` and negatives select class 0.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn class_indicator<const K: usize>(label: __m256) -> __m256 {
    // SAFETY: the caller guarantees AVX2 support; register-only.
    unsafe {
        let class = if K == 2 {
            _mm256_setr_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        } else {
            _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0)
        };
        let one = _mm256_set1_ps(1.0);
        let next = _mm256_add_ps(class, one);
        // class >= 1: class <= label < class + 1 (false for NaN).
        let in_range = _mm256_and_ps(
            _mm256_cmp_ps::<_CMP_GE_OQ>(label, class),
            _mm256_cmp_ps::<_CMP_LT_OQ>(label, next),
        );
        // class 0: everything not >= 1, including NaN and negatives.
        let is_zero = _mm256_cmp_ps::<_CMP_NGE_UQ>(label, one);
        let class_is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(class, _mm256_setzero_ps());
        let and: unsafe fn(__m256, __m256) -> __m256 = _mm256_and_ps;
        and(_mm256_blendv_ps(in_range, is_zero, class_is_zero), one)
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn short_softmax_gradient<const K: usize>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the dispatcher checked complete matrices (`labels.len() * K ==
    // preds.len() <= out.len()`, weights cover the labels) and AVX2/FMA
    // support; K is 2 or 4 so a batch is `WIDTH / K` whole rows.
    unsafe {
        const { assert!(K == 2 || K == 4) };
        let rows_per_batch = WIDTH / K;
        let one = _mm256_set1_ps(1.0);
        let minimum = _mm256_set1_ps(min_hess);
        let mut row = 0;
        while row + rows_per_batch <= labels.len() {
            let base = row * K;
            match short_softmax_batch::<K>(preds.as_ptr().add(base)) {
                Some(probability) => {
                    let label = broadcast_rows::<K>(labels.as_ptr().add(row));
                    let weight = match weights {
                        Some(values) => broadcast_rows::<K>(values.as_ptr().add(row)),
                        None => one,
                    };
                    let indicator = class_indicator::<K>(label);
                    let grad = _mm256_mul_ps(_mm256_sub_ps(probability, indicator), weight);
                    let hess = _mm256_max_ps(
                        _mm256_mul_ps(
                            _mm256_mul_ps(
                                _mm256_mul_ps(probability, _mm256_set1_ps(2.0)),
                                _mm256_sub_ps(one, probability),
                            ),
                            weight,
                        ),
                        minimum,
                    );
                    store_pairs(out.as_mut_ptr().add(base), grad, hess);
                }
                None => {
                    for current in row..row + rows_per_batch {
                        let base = current * K;
                        super::softmax_gradient_row_scalar(
                            &preds[base..base + K],
                            labels[current] as usize,
                            weights.map_or(1.0, |values| values[current]),
                            min_hess,
                            &mut out[base..base + K],
                        );
                    }
                }
            }
            row += rows_per_batch;
        }
        for current in row..labels.len() {
            let base = current * K;
            super::softmax_gradient_row_scalar(
                &preds[base..base + K],
                labels[current] as usize,
                weights.map_or(1.0, |values| values[current]),
                min_hess,
                &mut out[base..base + K],
            );
        }
    }
}

/// Candidates examined per vector iteration (one `f64` lane each).
const CANDIDATES: usize = 4;

/// Relative slack applied to the division-free prefilter threshold. The
/// cross-multiplied test and the exact quotient test each round to within a
/// few ULPs, so this margin guarantees the prefilter never rejects a candidate
/// the exact comparison would accept. False positives merely pay for a
/// division.
const PREFILTER_SLACK: f64 = 1e-9;

/// Scaled `gain(L) + gain(R)` a candidate must exceed to beat `best_loss`.
#[inline]
fn prefilter_target(best_loss: f64, comparison_epsilon: f64, parent_gain: f64) -> f64 {
    (best_loss + comparison_epsilon + parent_gain) * (1.0 - PREFILTER_SLACK)
}

/// Incumbent of one feature's scan plus the fixed inputs of the exact check.
struct Scan<'a> {
    total: GradStats,
    reg: &'a RegParams,
    parent_gain: f64,
    comparison_epsilon: f64,
    best: Option<SplitCandidate>,
    best_loss: f64,
}

impl Scan<'_> {
    /// Exact scalar check of the candidate with left statistics `left`: the
    /// same tests, in the same order, as the histogram builder's scalar scan.
    #[inline]
    fn accept(&mut self, left: GradStats, split_offset: usize) {
        let right = self.total.sub(left);
        let mcw = self.reg.min_child_weight;
        if left.hess < mcw || right.hess < mcw {
            return;
        }
        let loss_change = calc_gain(left, self.reg) + calc_gain(right, self.reg) - self.parent_gain;
        if loss_change > self.best_loss + self.comparison_epsilon {
            self.best_loss = loss_change;
            self.best = Some(SplitCandidate {
                loss_change,
                split_offset,
                left,
                right,
            });
        }
    }

    /// See [`prefilter_target`].
    #[inline]
    fn target(&self) -> f64 {
        prefilter_target(self.best_loss, self.comparison_epsilon, self.parent_gain)
    }
}

// Arithmetic-only helpers express the AVX2 precondition through an unsafe
// intrinsic function pointer. This keeps their unsafe blocks valid with the
// Rust 1.86 MSRV as well as the current stdarch API, without lint overrides.
// The compiler inlines these constant function pointers.

/// `Tα(G)²` for four candidates, `0` for a NaN gradient like `threshold_l1`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn numerator<const L1: bool>(gradient: __m256d, alpha: __m256d) -> __m256d {
    // SAFETY: the caller guarantees AVX2 support; all operations use registers.
    unsafe {
        let multiply: unsafe fn(__m256d, __m256d) -> __m256d = _mm256_mul_pd;
        let zero = _mm256_setzero_pd();
        if L1 {
            // |g| via sign-bit clear; `max(x, 0)` yields 0 for NaN because
            // `_mm256_max_pd` returns its second operand when either is NaN.
            let magnitude = _mm256_andnot_pd(_mm256_set1_pd(-0.0), gradient);
            let thresholded = _mm256_max_pd(_mm256_sub_pd(magnitude, alpha), zero);
            multiply(thresholded, thresholded)
        } else {
            _mm256_max_pd(multiply(gradient, gradient), zero)
        }
    }
}

/// Lanes holding a normal, finite, positive value: the range in which the
/// relative rounding-error bounds behind [`PREFILTER_SLACK`] hold.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn normal_positive(values: __m256d) -> __m256d {
    // SAFETY: the caller guarantees AVX2 support; all operations use registers.
    unsafe {
        let compare_lt: unsafe fn(__m256d, __m256d) -> __m256d = _mm256_cmp_pd::<_CMP_LT_OQ>;
        _mm256_and_pd(
            _mm256_cmp_pd::<_CMP_GE_OQ>(values, _mm256_set1_pd(f64::MIN_POSITIVE)),
            compare_lt(values, _mm256_set1_pd(f64::INFINITY)),
        )
    }
}

/// Best dense, unconstrained split of one feature's `histogram`, or `None`.
///
/// The prefix sums advance bin by bin in scalar-identical order. Four
/// candidates at a time are tested with the division-free bound
/// `a_L·b_R + a_R·b_L > target·b_L·b_R` (`a = Tα(G)²`, `b = H + λ ≥ 0`). Only
/// lanes that pass go through the exact scalar acceptance, so the returned
/// candidate matches the scalar scan exactly. Callers must ensure
/// `reg.max_delta_step == 0` (closed-form gain) and AVX2+FMA support.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dense_unconstrained_best_split(
    histogram: &[GradStats],
    total: GradStats,
    reg: &RegParams,
    parent_gain: f64,
    comparison_epsilon: f64,
) -> Option<SplitCandidate> {
    // SAFETY: the caller guarantees AVX2 and FMA support and `scan` has no
    // further preconditions.
    unsafe {
        if reg.alpha == 0.0 {
            scan::<false>(histogram, total, reg, parent_gain, comparison_epsilon)
        } else {
            scan::<true>(histogram, total, reg, parent_gain, comparison_epsilon)
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn scan<const L1: bool>(
    histogram: &[GradStats],
    total: GradStats,
    reg: &RegParams,
    parent_gain: f64,
    comparison_epsilon: f64,
) -> Option<SplitCandidate> {
    // SAFETY: the caller guarantees AVX2 and FMA support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let candidate_count = histogram.len().saturating_sub(1);
        let bins = histogram.as_ptr().cast::<f64>();
        let total_gradients = _mm256_set1_pd(total.grad);
        let total_hessians = _mm256_set1_pd(total.hess);
        let alpha = _mm256_set1_pd(reg.alpha);
        let lambda = _mm256_set1_pd(reg.lambda);
        let ones = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));

        let mut scan = Scan {
            total,
            reg,
            parent_gain,
            comparison_epsilon,
            best: None,
            best_loss: 0.0,
        };
        let mut target = scan.target();
        // `(grad, hess)` prefix over the bins consumed so far.
        let mut accumulated = _mm_setzero_pd();
        let mut index = 0;

        while index + CANDIDATES - 1 < candidate_count {
            // SAFETY: `index + 3 < candidate_count < histogram.len()`, and
            // GradStats is two adjacent f64 fields, so all four loads are in
            // range.
            let p0 = _mm_add_pd(accumulated, _mm_loadu_pd(bins.add(2 * index)));
            let p1 = _mm_add_pd(p0, _mm_loadu_pd(bins.add(2 * index + 2)));
            let p2 = _mm_add_pd(p1, _mm_loadu_pd(bins.add(2 * index + 4)));
            let p3 = _mm_add_pd(p2, _mm_loadu_pd(bins.add(2 * index + 6)));
            accumulated = p3;
            let left_gradients = _mm256_set_m128d(_mm_unpacklo_pd(p2, p3), _mm_unpacklo_pd(p0, p1));
            let left_hessians = _mm256_set_m128d(_mm_unpackhi_pd(p2, p3), _mm_unpackhi_pd(p0, p1));
            let right_gradients = _mm256_sub_pd(total_gradients, left_gradients);
            let right_hessians = _mm256_sub_pd(total_hessians, left_hessians);

            let left_numerator = numerator::<L1>(left_gradients, alpha);
            let right_numerator = numerator::<L1>(right_gradients, alpha);
            let left_denominator = _mm256_add_pd(left_hessians, lambda);
            let right_denominator = _mm256_add_pd(right_hessians, lambda);
            let cross = _mm256_fmadd_pd(
                right_numerator,
                left_denominator,
                _mm256_mul_pd(left_numerator, right_denominator),
            );
            let bound = _mm256_mul_pd(
                _mm256_mul_pd(_mm256_set1_pd(target), left_denominator),
                right_denominator,
            );
            // The cross-multiplied test is equivalent to the quotient test only
            // for positive denominators, and its rounding error is relative
            // (covered by `PREFILTER_SLACK`) only while every product stays a
            // normal finite number — a subnormal denominator would round the
            // intermediate `target * bL` with unbounded relative error. Every
            // other lane goes to the exact check unconditionally. `cross >= 0`
            // always holds, so "normal positive" also excludes a `cross` that
            // underflowed to zero or subnormal while `bound` is comparable,
            // where an absolute error could hide.
            let trusted = _mm256_and_pd(
                _mm256_and_pd(
                    normal_positive(left_denominator),
                    normal_positive(right_denominator),
                ),
                _mm256_and_pd(normal_positive(cross), normal_positive(bound)),
            );
            let improving = _mm256_cmp_pd::<_CMP_GT_OQ>(cross, bound);
            let mask = _mm256_movemask_pd(_mm256_or_pd(improving, _mm256_andnot_pd(trusted, ones)));
            if mask != 0 {
                let mut grads = [0.0f64; CANDIDATES];
                let mut hesses = [0.0f64; CANDIDATES];
                _mm256_storeu_pd(grads.as_mut_ptr(), left_gradients);
                _mm256_storeu_pd(hesses.as_mut_ptr(), left_hessians);
                for lane in 0..CANDIDATES {
                    if mask & (1 << lane) != 0 {
                        scan.accept(GradStats::new(grads[lane], hesses[lane]), index + lane);
                    }
                }
                target = scan.target();
            }
            index += CANDIDATES;
        }

        let mut lanes = [0.0f64; 2];
        _mm_storeu_pd(lanes.as_mut_ptr(), accumulated);
        let mut left = GradStats::new(lanes[0], lanes[1]);
        while index < candidate_count {
            left.add(histogram[index]);
            scan.accept(left, index);
            index += 1;
        }
        scan.best
    }
}

/// Entries of exactly 16 `cuts` that are `<= value` (ordered comparison).
///
/// # Safety
/// `cuts` must hold exactly 16 values.
pub(super) unsafe fn count_le_16(cuts: &[f32], value: f32) -> usize {
    debug_assert_eq!(cuts.len(), 16);
    // SAFETY: the caller guarantees exactly 16 readable values; each load
    // covers one quarter of them.
    unsafe {
        let value = _mm_set1_ps(value);
        let ptr = cuts.as_ptr();
        let mut count = 0usize;
        for quarter in 0..4 {
            let mask = _mm_cmple_ps(_mm_loadu_ps(ptr.add(quarter * 4)), value);
            count += _mm_movemask_ps(mask).count_ones() as usize;
        }
        count
    }
}
