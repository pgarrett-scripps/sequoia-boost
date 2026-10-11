use super::{scalar, sigmoid_scalar, LOG_LOSS_EPSILON, MIN_POSITIVE_PREDICTION};
use crate::objective::GradPair;
use crate::tree::gain::{calc_gain, GradStats, RegParams};
use std::arch::aarch64::*;

// Arithmetic-only helpers express the NEON precondition through an unsafe
// intrinsic function pointer. This keeps their unsafe blocks valid with the
// Rust 1.86 MSRV as well as the current stdarch API, without lint overrides.
// The compiler inlines these constant function pointers.

const VECTOR_WIDTH: usize = 4;
const MAX_FAST_EXP_INPUT: f32 = 80.0;
const _: () = assert!(std::mem::size_of::<GradPair>() == 2 * std::mem::size_of::<f32>());
const _: () = assert!(std::mem::size_of::<GradStats>() == 2 * std::mem::size_of::<f64>());

/// Exponential for finite f32 lanes in [-80, 80]. Range
/// reduction keeps the polynomial input in [-ln(2)/2, ln(2)/2], where a
/// seventh-order Taylor polynomial is within a few f32 ULPs.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn expq_f32<const ESTRIN: bool>(value: float32x4_t) -> float32x4_t {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let scaled = vmulq_n_f32(value, std::f32::consts::LOG2_E);
        let exponent = vcvtnq_s32_f32(scaled);
        let exponent_f32 = vcvtq_f32_s32(exponent);

        // Split ln(2) so the range reduction loses fewer low bits.
        let mut reduced = vfmsq_n_f32(value, exponent_f32, 0.693_359_4);
        reduced = vfmaq_n_f32(reduced, exponent_f32, 2.121_944_4e-4);

        let polynomial = if ESTRIN {
            // Estrin evaluation exposes independent pairs instead of seven dependent FMAs.
            let squared = vmulq_f32(reduced, reduced);
            let fourth = vmulq_f32(squared, squared);
            let pair_0 = vaddq_f32(vdupq_n_f32(1.0), reduced);
            let pair_1 = vfmaq_f32(vdupq_n_f32(0.5), vdupq_n_f32(1.0 / 6.0), reduced);
            let pair_2 = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), vdupq_n_f32(1.0 / 120.0), reduced);
            let pair_3 = vfmaq_f32(
                vdupq_n_f32(1.0 / 720.0),
                vdupq_n_f32(1.0 / 5_040.0),
                reduced,
            );
            let low = vfmaq_f32(pair_0, pair_1, squared);
            let high = vfmaq_f32(pair_2, pair_3, squared);
            vfmaq_f32(low, high, fourth)
        } else {
            // Wide in-place softmax favors Horner evaluation for throughput.
            let mut polynomial = vdupq_n_f32(1.0 / 5_040.0);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0 / 720.0), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(0.5), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0), polynomial, reduced);
            vfmaq_f32(vdupq_n_f32(1.0), polynomial, reduced)
        };

        let exponent_bits = vshlq_n_s32(vaddq_s32(exponent, vdupq_n_s32(127)), 23);
        let multiply: unsafe fn(float32x4_t, float32x4_t) -> float32x4_t = vmulq_f32;
        multiply(polynomial, vreinterpretq_f32_s32(exponent_bits))
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn sigmoidq_f32(value: float32x4_t) -> float32x4_t {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let magnitude = vabsq_f32(value);
        let exp = expq_f32::<true>(vnegq_f32(magnitude));
        let denominator = vaddq_f32(vdupq_n_f32(1.0), exp);
        let positive = vdivq_f32(vdupq_n_f32(1.0), denominator);
        let negative = vdivq_f32(exp, denominator);
        vbslq_f32(vcgeq_f32(value, vdupq_n_f32(0.0)), positive, negative)
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn regular_input(value: float32x4_t) -> bool {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let magnitude = vabsq_f32(value);
        let in_range = vcleq_f32(magnitude, vdupq_n_f32(MAX_FAST_EXP_INPUT));
        let minimum: unsafe fn(uint32x4_t) -> u32 = vminvq_u32;
        minimum(in_range) == u32::MAX
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn finite_input(value: float32x4_t) -> bool {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let finite = vcleq_f32(vabsq_f32(value), vdupq_n_f32(f32::MAX));
        let minimum: unsafe fn(uint32x4_t) -> u32 = vminvq_u32;
        minimum(finite) == u32::MAX
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn finite_min_max(values: &[f32]) -> Option<(f32, f32)> {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let mut maximum = vdupq_n_f32(f32::NEG_INFINITY);
        let mut minimum = vdupq_n_f32(f32::INFINITY);
        let mut index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable values.
            let value = vld1q_f32(values.as_ptr().add(index));
            // These min/max instructions propagate NaNs, so checking the final
            // extrema also rejects non-finite lanes from any earlier block.
            maximum = vmaxq_f32(maximum, value);
            minimum = vminq_f32(minimum, value);
            index += VECTOR_WIDTH;
        }

        let mut max = vmaxvq_f32(maximum);
        let mut min = vminvq_f32(minimum);
        if !min.is_finite() || !max.is_finite() {
            return None;
        }
        for &value in &values[index..] {
            if !value.is_finite() {
                return None;
            }
            max = max.max(value);
            min = min.min(value);
        }
        Some((min, max))
    }
}

/// Natural logarithm for positive, finite, normal f64 lanes. Mantissas are
/// reduced around one and evaluated with an atanh series through z^15.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn logq_f64(value: float64x2_t) -> float64x2_t {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        const MANTISSA_MASK: u64 = (1u64 << 52) - 1;
        const ONE_BITS: u64 = 1023u64 << 52;
        let bits = vreinterpretq_u64_f64(value);
        let raw_exponent = vandq_u64(vshrq_n_u64::<52>(bits), vdupq_n_u64(0x7ff));
        let mut exponent = vsubq_f64(vcvtq_f64_u64(raw_exponent), vdupq_n_f64(1023.0));
        let mut mantissa = vreinterpretq_f64_u64(vorrq_u64(
            vandq_u64(bits, vdupq_n_u64(MANTISSA_MASK)),
            vdupq_n_u64(ONE_BITS),
        ));

        let halve = vcgtq_f64(mantissa, vdupq_n_f64(std::f64::consts::SQRT_2));
        mantissa = vbslq_f64(halve, vmulq_n_f64(mantissa, 0.5), mantissa);
        exponent = vaddq_f64(
            exponent,
            vbslq_f64(halve, vdupq_n_f64(1.0), vdupq_n_f64(0.0)),
        );

        let one = vdupq_n_f64(1.0);
        let z = vdivq_f64(vsubq_f64(mantissa, one), vaddq_f64(mantissa, one));
        let z_squared = vmulq_f64(z, z);
        // Estrin evaluation exposes independent FMAs instead of a seven-FMA chain.
        let z_fourth = vmulq_f64(z_squared, z_squared);
        let z_eighth = vmulq_f64(z_fourth, z_fourth);
        let pair_0 = vfmaq_f64(one, vdupq_n_f64(1.0 / 3.0), z_squared);
        let pair_1 = vfmaq_f64(vdupq_n_f64(1.0 / 5.0), vdupq_n_f64(1.0 / 7.0), z_squared);
        let pair_2 = vfmaq_f64(vdupq_n_f64(1.0 / 9.0), vdupq_n_f64(1.0 / 11.0), z_squared);
        let pair_3 = vfmaq_f64(vdupq_n_f64(1.0 / 13.0), vdupq_n_f64(1.0 / 15.0), z_squared);
        let low = vfmaq_f64(pair_0, pair_1, z_fourth);
        let high = vfmaq_f64(pair_2, pair_3, z_fourth);
        let polynomial = vfmaq_f64(low, high, z_eighth);
        let mantissa_log = vmulq_n_f64(vmulq_f64(z, polynomial), 2.0);
        let result = vfmaq_n_f64(mantissa_log, exponent, 0.693_145_751_953_125);
        let multiply_add: unsafe fn(float64x2_t, float64x2_t, f64) -> float64x2_t = vfmaq_n_f64;
        multiply_add(result, exponent, 1.428_606_820_309_417_3e-6)
    }
}

/// Exponential for the finite f64 range generated by Tweedie powers of f32
/// predictions. Range reduction is followed by a degree-12 polynomial on
/// [-ln(2)/2, ln(2)/2].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn expq_f64(value: float64x2_t) -> float64x2_t {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let scaled = vmulq_n_f64(value, std::f64::consts::LOG2_E);
        let exponent = vcvtnq_s64_f64(scaled);
        let exponent_f64 = vcvtq_f64_s64(exponent);
        let mut reduced = vfmsq_n_f64(value, exponent_f64, 0.693_145_751_953_125);
        reduced = vfmsq_n_f64(reduced, exponent_f64, 1.428_606_820_309_417_3e-6);

        // Independent polynomial pairs shorten the FMA dependency chain.
        let squared = vmulq_f64(reduced, reduced);
        let fourth = vmulq_f64(squared, squared);
        let eighth = vmulq_f64(fourth, fourth);
        let pair_0 = vaddq_f64(vdupq_n_f64(1.0), reduced);
        let pair_1 = vfmaq_f64(vdupq_n_f64(0.5), vdupq_n_f64(1.0 / 6.0), reduced);
        let pair_2 = vfmaq_f64(vdupq_n_f64(1.0 / 24.0), vdupq_n_f64(1.0 / 120.0), reduced);
        let pair_3 = vfmaq_f64(
            vdupq_n_f64(1.0 / 720.0),
            vdupq_n_f64(1.0 / 5_040.0),
            reduced,
        );
        let pair_4 = vfmaq_f64(
            vdupq_n_f64(1.0 / 40_320.0),
            vdupq_n_f64(1.0 / 362_880.0),
            reduced,
        );
        let pair_5 = vfmaq_f64(
            vdupq_n_f64(1.0 / 3_628_800.0),
            vdupq_n_f64(1.0 / 39_916_800.0),
            reduced,
        );
        let low = vfmaq_f64(pair_0, pair_1, squared);
        let middle = vfmaq_f64(pair_2, pair_3, squared);
        let high = vfmaq_f64(pair_4, pair_5, squared);
        let low = vfmaq_f64(low, middle, fourth);
        let high = vfmaq_f64(high, vdupq_n_f64(1.0 / 479_001_600.0), fourth);
        let polynomial = vfmaq_f64(low, high, eighth);

        let exponent_bits = vshlq_n_s64::<52>(vaddq_s64(exponent, vdupq_n_s64(1023)));
        let multiply: unsafe fn(float64x2_t, float64x2_t) -> float64x2_t = vmulq_f64;
        multiply(polynomial, vreinterpretq_f64_s64(exponent_bits))
    }
}

/// Number of the 16 `cuts` that are `<= value`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn count_le_16(cuts: &[f32], value: f32) -> usize {
    debug_assert_eq!(cuts.len(), 16);
    // SAFETY: the caller guarantees NEON support and exactly 16 readable
    // values; each load below covers one aligned quarter of them.
    unsafe {
        let value = vdupq_n_f32(value);
        let ptr = cuts.as_ptr();
        let mut count = vdupq_n_u32(0);
        for quarter in 0..4 {
            let mask = vcleq_f32(vld1q_f32(ptr.add(quarter * VECTOR_WIDTH)), value);
            count = vsubq_u32(count, mask);
        }
        vaddvq_u32(count) as usize
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn sum_grad_stats(values: &[GradStats]) -> GradStats {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let mut gradients = vdupq_n_f64(0.0);
        let mut hessians = vdupq_n_f64(0.0);
        let mut index = 0;
        while index + 2 <= values.len() {
            // SAFETY: GradStats is two adjacent f64 fields and the loop condition
            // leaves two complete values, matching the deinterleaving load.
            let pair = vld2q_f64(values.as_ptr().add(index).cast::<f64>());
            gradients = vaddq_f64(gradients, pair.0);
            hessians = vaddq_f64(hessians, pair.1);
            index += 2;
        }
        let mut sum = GradStats::new(vaddvq_f64(gradients), vaddvq_f64(hessians));
        for &value in &values[index..] {
            sum.add(value);
        }
        sum
    }
}

/// Relative slack applied to the division-free prefilter threshold. Both the
/// cross-multiplied test and the exact quotient test round to within a few
/// ULPs, so this margin guarantees the prefilter never rejects a candidate the
/// exact comparison would accept. False positives merely pay for a division.
const PREFILTER_SLACK: f64 = 1e-9;

/// Ceiling applied to the prefilter bound before the improving comparison. A
/// bound that overflowed to infinity compares as `inf > inf` (false) and would
/// silently drop a candidate the exact comparison accepts, so saturated lanes
/// keep comparing against the ceiling: a `cross` above it goes to the exact
/// check and anything below it loses to a bound that truly dominates it.
const BOUND_CEILING: f64 = 1.0e292;

/// Bounds below this floor take the exact comparison: their products round
/// subnormal, where the relative error behind `PREFILTER_SLACK` is unbounded
/// and an underflowed `cross` or `bound` can hide an acceptable candidate.
const BOUND_FLOOR: f64 = 1.0e-250;

/// Scale of the `target · b_L` intermediate that keeps subnormal rounding
/// below `PREFILTER_SLACK / 100`: products at or above this magnitude round
/// within 3e-17, so the bound they produce stays trustworthy.
const DENOMINATOR_FLOOR_SCALE: f64 = 1.0e-307;

/// First denominator (`b_L`) that keeps the `target · b_L` intermediate above
/// [`DENOMINATOR_FLOOR_SCALE`]. With `lambda` large enough to cover it on its
/// own, every acceptable candidate has `b_L = H_L + lambda >= lambda`, so
/// `lambda` itself is the floor. Otherwise the floor divides out `target`.
#[inline]
fn denominator_floor(target: f64, lambda: f64) -> f64 {
    if target * lambda >= DENOMINATOR_FLOOR_SCALE {
        lambda
    } else if target > 0.0 {
        DENOMINATOR_FLOOR_SCALE / target
    } else {
        f64::INFINITY
    }
}

/// Scaled `gain(L) + gain(R)` a candidate must exceed to beat `best_loss`.
#[inline]
fn prefilter_target(best_loss: f64, comparison_epsilon: f64, parent_gain: f64) -> f64 {
    (best_loss + comparison_epsilon + parent_gain) * (1.0 - PREFILTER_SLACK)
}

/// Scan state shared by the vector prefilter and the exact candidate check.
struct SplitScan<'a> {
    total: GradStats,
    reg: &'a RegParams,
    parent_gain: f64,
    comparison_epsilon: f64,
    best: Option<super::SplitCandidate>,
    best_loss: f64,
    /// See [`prefilter_target`]. Refreshed whenever `best_loss` changes.
    target: f64,
    /// See [`denominator_floor`]. Refreshed together with `target`.
    min_denominator: f64,
}

impl SplitScan<'_> {
    /// Exact scalar check of the candidates at `index` and `index + 1` with
    /// the given left statistics: the same tests, in the same order, as the
    /// histogram builder's scalar scan, so the surviving candidate is
    /// bit-identical to it. The precomputed vector losses are not used: a
    /// child the closed form cannot evaluate (a zero Hessian with no
    /// regularization) contributes a zero gain in the scalar scan, and the
    /// vector quotients would return infinity or NaN for it instead.
    #[inline]
    fn accept_pair(&mut self, index: usize, lefts: [GradStats; 2]) {
        for (lane, left) in lefts.into_iter().enumerate() {
            let right = self.total.sub(left);
            if left.hess < self.reg.min_child_weight || right.hess < self.reg.min_child_weight {
                continue;
            }
            let loss_change =
                calc_gain(left, self.reg) + calc_gain(right, self.reg) - self.parent_gain;
            // Ordered comparison: a NaN loss never improves.
            if loss_change > self.best_loss + self.comparison_epsilon {
                self.best_loss = loss_change;
                self.target =
                    prefilter_target(self.best_loss, self.comparison_epsilon, self.parent_gain);
                self.min_denominator = denominator_floor(self.target, self.reg.lambda);
                self.best = Some(super::SplitCandidate {
                    loss_change,
                    split_offset: index + lane,
                    left,
                    right,
                });
            }
        }
    }
}

/// Closed-form gain terms of both children for two candidates.
struct PairTerms {
    /// `Tα(G_L)²`
    left_numerator: float64x2_t,
    /// `H_L + λ`
    left_denominator: float64x2_t,
    right_numerator: float64x2_t,
    right_denominator: float64x2_t,
}

/// Two lanes of a vector as an array.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn lanes(vector: float64x2_t) -> [f64; 2] {
    // SAFETY: the caller guarantees NEON support; lane extraction only reads
    // registers. The function pointer expresses that precondition on every
    // supported toolchain.
    unsafe {
        let lane_0: unsafe fn(float64x2_t) -> f64 = vgetq_lane_f64::<0>;
        let lane_1: unsafe fn(float64x2_t) -> f64 = vgetq_lane_f64::<1>;
        [lane_0(vector), lane_1(vector)]
    }
}

/// Left statistics of the two candidates in lane-split vectors.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pair_lanes(gradients: float64x2_t, hessians: float64x2_t) -> [GradStats; 2] {
    // SAFETY: the caller guarantees NEON support.
    unsafe {
        let ([grad_0, grad_1], [hess_0, hess_1]) = (lanes(gradients), lanes(hessians));
        [
            GradStats::new(grad_0, hess_0),
            GradStats::new(grad_1, hess_1),
        ]
    }
}

/// Running prefix over the two bins at `*bin` and `*bin + 1`. `accumulated`
/// holds `(grad, hess)` lanes and advances by both bins in sequential order.
/// The result is lane-split into `(gradients, hessians)` of the two candidates.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn prefix_pair(
    accumulated: &mut float64x2_t,
    histogram: *const f64,
    index: usize,
) -> (float64x2_t, float64x2_t) {
    // SAFETY: the caller guarantees NEON support and `index + 1 <
    // histogram.len()`; GradStats is two adjacent f64 fields.
    unsafe {
        let first = vaddq_f64(*accumulated, vld1q_f64(histogram.add(2 * index)));
        let second = vaddq_f64(first, vld1q_f64(histogram.add(2 * index + 2)));
        *accumulated = second;
        (vzip1q_f64(first, second), vzip2q_f64(first, second))
    }
}

/// Closed-form gain terms of two candidates with the given left statistics.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn gain_terms<const L1: bool>(
    left_gradients: float64x2_t,
    left_hessians: float64x2_t,
    total_gradients: float64x2_t,
    total_hessians: float64x2_t,
    alpha: float64x2_t,
    lambda: float64x2_t,
) -> PairTerms {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    // The function pointer expresses that precondition on every supported
    // toolchain.
    unsafe {
        let multiply: unsafe fn(float64x2_t, float64x2_t) -> float64x2_t = vmulq_f64;
        let zero = vdupq_n_f64(0.0);
        // Gain squares the L1-thresholded gradient, so its sign is unnecessary;
        // numeric max maps a NaN gradient to zero like threshold_l1. Without L1
        // the threshold is the identity and the numerator is G².
        let numerator = |gradient: float64x2_t| {
            if L1 {
                let thresholded = vmaxnmq_f64(vsubq_f64(vabsq_f64(gradient), alpha), zero);
                multiply(thresholded, thresholded)
            } else {
                vmaxnmq_f64(multiply(gradient, gradient), zero)
            }
        };
        let right_gradients = vsubq_f64(total_gradients, left_gradients);
        let right_hessians = vsubq_f64(total_hessians, left_hessians);
        PairTerms {
            left_numerator: numerator(left_gradients),
            left_denominator: vaddq_f64(left_hessians, lambda),
            right_numerator: numerator(right_gradients),
            right_denominator: vaddq_f64(right_hessians, lambda),
        }
    }
}

/// Lane mask of candidates whose exact loss change may exceed the incumbent.
///
/// gain = Tα(G)² / (H + λ). With `a_x = Tα(G_x)²` and `b_x = H_x + λ`
/// (`b_x >= 0`), `a_L/b_L + a_R/b_R > target` is equivalent to
/// `a_L·b_R + a_R·b_L > target·b_L·b_R`, which needs no division.
///
/// That equivalence and the relative rounding bound behind `PREFILTER_SLACK`
/// hold only while the products stay in normal range, so three guards keep the
/// prefilter exact-rejecting and otherwise hand the lane to the sequential
/// acceptance: the bound is saturated at [`BOUND_CEILING`] (an overflowed bound
/// compares as `inf > inf`, which is false), it must stay at or above
/// [`BOUND_FLOOR`] (subnormal products round with unbounded relative error),
/// and the first denominator must stay at or above `min_denominator` (a
/// subnormal `target · b_L` intermediate corrupts an otherwise normal bound).
/// Lanes that fail a guard are reported as improving, so they merely pay for
/// the exact check. A `b_L <= 0` lane can only produce a bound at or below
/// zero, which fails the floor and reaches the exact check, and that check
/// scores such children with the scalar scan's zero gain.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn improving_mask(terms: &PairTerms, target: f64, min_denominator: f64) -> uint64x2_t {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    // The function pointer expresses that precondition on every supported
    // toolchain.
    unsafe {
        let multiply: unsafe fn(float64x2_t, float64x2_t) -> float64x2_t = vmulq_f64;
        let cross = vfmaq_f64(
            multiply(terms.left_numerator, terms.right_denominator),
            terms.right_numerator,
            terms.left_denominator,
        );
        let bound = multiply(
            multiply(vdupq_n_f64(target), terms.left_denominator),
            terms.right_denominator,
        );
        // NaNs saturate to the ceiling as well: a NaN bound belongs to a lane
        // the exact check rejects, and only a `cross` beyond the ceiling could
        // pass the comparison against it.
        let bound = vminnmq_f64(bound, vdupq_n_f64(BOUND_CEILING));
        let trusted = vandq_u64(
            vcgeq_f64(bound, vdupq_n_f64(BOUND_FLOOR)),
            vcgeq_f64(terms.left_denominator, vdupq_n_f64(min_denominator)),
        );
        vbslq_u64(trusted, vcgtq_f64(cross, bound), vdupq_n_u64(u64::MAX))
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn dense_unconstrained_best_split(
    histogram: &[GradStats],
    total: GradStats,
    reg: &RegParams,
    parent_gain: f64,
    comparison_epsilon: f64,
) -> Option<super::SplitCandidate> {
    // SAFETY: the caller guarantees NEON support.
    unsafe {
        if reg.alpha == 0.0 {
            scan_dense_splits::<false>(histogram, total, reg, parent_gain, comparison_epsilon)
        } else {
            scan_dense_splits::<true>(histogram, total, reg, parent_gain, comparison_epsilon)
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn scan_dense_splits<const L1: bool>(
    histogram: &[GradStats],
    total: GradStats,
    reg: &RegParams,
    parent_gain: f64,
    comparison_epsilon: f64,
) -> Option<super::SplitCandidate> {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let candidate_count = histogram.len().saturating_sub(1);
        let bins = histogram.as_ptr().cast::<f64>();
        let total_gradients = vdupq_n_f64(total.grad);
        let total_hessians = vdupq_n_f64(total.hess);
        let alpha = vdupq_n_f64(reg.alpha);
        let lambda = vdupq_n_f64(reg.lambda);
        let mut accumulated = vdupq_n_f64(0.0);
        let target = prefilter_target(0.0, comparison_epsilon, parent_gain);
        let mut scan = SplitScan {
            total,
            reg,
            parent_gain,
            comparison_epsilon,
            best: None,
            best_loss: 0.0,
            target,
            min_denominator: denominator_floor(target, reg.lambda),
        };
        let terms = |gradients, hessians| {
            gain_terms::<L1>(
                gradients,
                hessians,
                total_gradients,
                total_hessians,
                alpha,
                lambda,
            )
        };
        let mut index = 0;
        // Four candidates per iteration. Most bins do not improve the
        // incumbent, so they are rejected in vector registers without lane
        // extraction or division; survivors take the exact sequential path.
        while index + 3 < candidate_count {
            // SAFETY: `index + 3 < candidate_count < histogram.len()` covers
            // both pairs.
            let (gradients_0, hessians_0) = prefix_pair(&mut accumulated, bins, index);
            let (gradients_1, hessians_1) = prefix_pair(&mut accumulated, bins, index + 2);
            let terms_0 = terms(gradients_0, hessians_0);
            let terms_1 = terms(gradients_1, hessians_1);
            let mask = vorrq_u64(
                improving_mask(&terms_0, scan.target, scan.min_denominator),
                improving_mask(&terms_1, scan.target, scan.min_denominator),
            );
            if vmaxvq_u32(vreinterpretq_u32_u64(mask)) != 0 {
                scan.accept_pair(index, pair_lanes(gradients_0, hessians_0));
                scan.accept_pair(index + 2, pair_lanes(gradients_1, hessians_1));
            }
            index += 4;
        }
        if index + 1 < candidate_count {
            // SAFETY: `index + 1 < candidate_count < histogram.len()`.
            let (gradients, hessians) = prefix_pair(&mut accumulated, bins, index);
            let terms = terms(gradients, hessians);
            if vmaxvq_u32(vreinterpretq_u32_u64(improving_mask(
                &terms,
                scan.target,
                scan.min_denominator,
            ))) != 0
            {
                scan.accept_pair(index, pair_lanes(gradients, hessians));
            }
            index += 2;
        }

        if index < candidate_count {
            let [grad, hess] = lanes(accumulated);
            let mut left = GradStats::new(grad, hess);
            left.add(histogram[index]);
            let right = total.sub(left);
            if left.hess >= reg.min_child_weight && right.hess >= reg.min_child_weight {
                let loss_change = calc_gain(left, reg) + calc_gain(right, reg) - parent_gain;
                if loss_change > scan.best_loss + comparison_epsilon {
                    scan.best = Some(super::SplitCandidate {
                        loss_change,
                        split_offset: index,
                        left,
                        right,
                    });
                }
            }
        }
        scan.best
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn exp_inplace(values: &mut [f32]) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let mut index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable and writable values.
            let input = vld1q_f32(values.as_ptr().add(index));
            if regular_input(input) {
                let output = expq_f32::<true>(input);
                // SAFETY: the loop condition leaves four writable values.
                vst1q_f32(values.as_mut_ptr().add(index), output);
            } else {
                for value in &mut values[index..index + VECTOR_WIDTH] {
                    *value = value.exp();
                }
            }
            index += VECTOR_WIDTH;
        }
        for value in &mut values[index..] {
            *value = value.exp();
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn sigmoid_inplace(values: &mut [f32]) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let mut index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable and writable values.
            let input = vld1q_f32(values.as_ptr().add(index));
            if regular_input(input) {
                let output = sigmoidq_f32(input);
                // SAFETY: the loop condition leaves four writable values.
                vst1q_f32(values.as_mut_ptr().add(index), output);
            } else {
                for value in &mut values[index..index + VECTOR_WIDTH] {
                    *value = sigmoid_scalar(*value);
                }
            }
            index += VECTOR_WIDTH;
        }
        for value in &mut values[index..] {
            *value = sigmoid_scalar(*value);
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn logistic_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let scale = vdupq_n_f32(scale_pos_weight);
        let min_hess_vector = vdupq_n_f32(min_hess);
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the loop condition and objective length checks leave four
            // readable inputs and four writable GradPair outputs.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            if !regular_input(pred) {
                scalar::logistic_gradient(
                    preds,
                    labels,
                    weights,
                    (scale_pos_weight, min_hess),
                    out,
                    index..index + VECTOR_WIDTH,
                );
                index += VECTOR_WIDTH;
                continue;
            }
            let label = vld1q_f32(labels.as_ptr().add(index));
            let probability = sigmoidq_f32(pred);
            let mut weight = match weights {
                Some(values) => vld1q_f32(values.as_ptr().add(index)),
                None => one,
            };
            weight = vmulq_f32(weight, vbslq_f32(vceqq_f32(label, one), scale, one));
            let grad = vmulq_f32(vsubq_f32(probability, label), weight);
            let hess = vmulq_f32(
                vmaxq_f32(
                    vmulq_f32(probability, vsubq_f32(one, probability)),
                    min_hess_vector,
                ),
                weight,
            );
            // SAFETY: GradPair is repr(C) with two adjacent f32 fields and the loop
            // condition leaves room for four pairs.
            vst2q_f32(
                out.as_mut_ptr().add(index).cast::<f32>(),
                float32x4x2_t(grad, hess),
            );
            index += VECTOR_WIDTH;
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

#[target_feature(enable = "neon")]
pub(super) unsafe fn poisson_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    max_delta_step: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let delta = vdupq_n_f32(max_delta_step);
        let mut index = 0;
        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable inputs.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            let shifted = vaddq_f32(pred, delta);
            if !regular_input(pred) || !regular_input(shifted) {
                scalar::poisson_gradient(
                    preds,
                    labels,
                    weights,
                    max_delta_step,
                    out,
                    index..index + VECTOR_WIDTH,
                );
                index += VECTOR_WIDTH;
                continue;
            }
            let label = vld1q_f32(labels.as_ptr().add(index));
            let weight = match weights {
                Some(values) => vld1q_f32(values.as_ptr().add(index)),
                None => vdupq_n_f32(1.0),
            };
            let grad = vmulq_f32(vsubq_f32(expq_f32::<true>(pred), label), weight);
            let hess = vmulq_f32(expq_f32::<true>(shifted), weight);
            // SAFETY: GradPair is two adjacent f32 fields, with four output slots.
            vst2q_f32(
                out.as_mut_ptr().add(index).cast::<f32>(),
                float32x4x2_t(grad, hess),
            );
            index += VECTOR_WIDTH;
        }
        scalar::poisson_gradient(
            preds,
            labels,
            weights,
            max_delta_step,
            out,
            index..preds.len(),
        );
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn gamma_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let mut index = 0;
        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable inputs.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            let negative = vnegq_f32(pred);
            if !regular_input(negative) {
                scalar::gamma_gradient(preds, labels, weights, out, index..index + VECTOR_WIDTH);
                index += VECTOR_WIDTH;
                continue;
            }
            let label = vld1q_f32(labels.as_ptr().add(index));
            let weight = match weights {
                Some(values) => vld1q_f32(values.as_ptr().add(index)),
                None => one,
            };
            let scaled = vmulq_f32(label, expq_f32::<true>(negative));
            let grad = vmulq_f32(vsubq_f32(one, scaled), weight);
            let hess = vmulq_f32(scaled, weight);
            // SAFETY: GradPair is two adjacent f32 fields, with four output slots.
            vst2q_f32(
                out.as_mut_ptr().add(index).cast::<f32>(),
                float32x4x2_t(grad, hess),
            );
            index += VECTOR_WIDTH;
        }
        scalar::gamma_gradient(preds, labels, weights, out, index..preds.len());
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn tweedie_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let one_minus_rho = vdupq_n_f32(1.0 - rho);
        let two_minus_rho = vdupq_n_f32(2.0 - rho);
        let mut index = 0;
        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable inputs.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            let input_1 = vmulq_f32(one_minus_rho, pred);
            let input_2 = vmulq_f32(two_minus_rho, pred);
            if !regular_input(input_1) || !regular_input(input_2) {
                scalar::tweedie_gradient(
                    preds,
                    labels,
                    weights,
                    rho,
                    out,
                    index..index + VECTOR_WIDTH,
                );
                index += VECTOR_WIDTH;
                continue;
            }
            let label = vld1q_f32(labels.as_ptr().add(index));
            let weight = match weights {
                Some(values) => vld1q_f32(values.as_ptr().add(index)),
                None => vdupq_n_f32(1.0),
            };
            let exp_1 = expq_f32::<true>(input_1);
            let exp_2 = expq_f32::<true>(input_2);
            let label_exp_1 = vmulq_f32(label, exp_1);
            let grad = vmulq_f32(vsubq_f32(exp_2, label_exp_1), weight);
            let hess = vmulq_f32(
                vsubq_f32(
                    vmulq_f32(two_minus_rho, exp_2),
                    vmulq_f32(one_minus_rho, label_exp_1),
                ),
                weight,
            );
            // SAFETY: GradPair is two adjacent f32 fields, with four output slots.
            vst2q_f32(
                out.as_mut_ptr().add(index).cast::<f32>(),
                float32x4x2_t(grad, hess),
            );
            index += VECTOR_WIDTH;
        }
        scalar::tweedie_gradient(preds, labels, weights, rho, out, index..preds.len());
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_inplace(values: &mut [f32]) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let Some((min, max)) = finite_min_max(values) else {
            super::softmax_scalar(values);
            return;
        };
        if max - min > MAX_FAST_EXP_INPUT {
            super::softmax_scalar(values);
            return;
        }

        let max_vector = vdupq_n_f32(max);
        let mut sum_vector = vdupq_n_f32(0.0);
        let mut index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable and writable values.
            let input = vld1q_f32(values.as_ptr().add(index));
            let exp = expq_f32::<false>(vsubq_f32(input, max_vector));
            sum_vector = vaddq_f32(sum_vector, exp);
            // SAFETY: the loop condition leaves four writable values.
            vst1q_f32(values.as_mut_ptr().add(index), exp);
            index += VECTOR_WIDTH;
        }
        let mut sum = vaddvq_f32(sum_vector);
        for value in &mut values[index..] {
            *value = (*value - max).exp();
            sum += *value;
        }

        let inverse = 1.0 / sum;
        let inverse_vector = vdupq_n_f32(inverse);
        index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable and writable values.

            let probability = vld1q_f32(values.as_ptr().add(index));
            vst1q_f32(
                values.as_mut_ptr().add(index),
                vmulq_f32(probability, inverse_vector),
            );

            index += VECTOR_WIDTH;
        }
        for value in &mut values[index..] {
            *value *= inverse;
        }
    }
}

/// Process four independent rows in the lanes, so short rows need neither a
/// horizontal reduction nor a scratch pass through the output matrix.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn short_softmax_batch<const K: usize>(preds: &[f32]) -> Option<[float32x4_t; K]> {
    // SAFETY: callers provide four complete rows, K is 2, 3, or 4, and NEON
    // is available. Interleaved loads transpose row-major inputs into classes.
    unsafe {
        let mut values = [vdupq_n_f32(0.0); K];
        match K {
            2 => {
                let input = vld2q_f32(preds.as_ptr());
                values[0] = input.0;
                values[1] = input.1;
            }
            3 => {
                let input = vld3q_f32(preds.as_ptr());
                values[0] = input.0;
                values[1] = input.1;
                values[2] = input.2;
            }
            4 => {
                let input = vld4q_f32(preds.as_ptr());
                values[0] = input.0;
                values[1] = input.1;
                values[2] = input.2;
                values[3] = input.3;
            }
            _ => unreachable!(),
        }
        let mut minimum = values[0];
        let mut maximum = values[0];
        for &value in &values[1..] {
            minimum = vminq_f32(minimum, value);
            maximum = vmaxq_f32(maximum, value);
        }
        // NaNs propagate through min/max; infinities produce a non-finite
        // range. Either makes this ordered comparison fail for that row.
        if vminvq_u32(vcleq_f32(
            vsubq_f32(maximum, minimum),
            vdupq_n_f32(MAX_FAST_EXP_INPUT),
        )) != u32::MAX
        {
            return None;
        }
        let mut sum = vdupq_n_f32(0.0);
        for value in &mut values {
            *value = expq_f32::<true>(vsubq_f32(*value, maximum));
            sum = vaddq_f32(sum, *value);
        }
        let inverse = vdivq_f32(vdupq_n_f32(1.0), sum);
        for value in &mut values {
            *value = vmulq_f32(*value, inverse);
        }
        Some(values)
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn short_softmax_rows<const K: usize>(values: &mut [f32]) {
    // SAFETY: NEON is available, K is 2, 3, or 4, and chunks bound each
    // interleaved load/store to four complete rows. Remaining rows are scalar.
    unsafe {
        let mut batches = values.chunks_exact_mut(4 * K);
        for batch in &mut batches {
            if let Some(p) = short_softmax_batch::<K>(batch) {
                match K {
                    2 => vst2q_f32(batch.as_mut_ptr(), float32x4x2_t(p[0], p[1])),
                    3 => vst3q_f32(batch.as_mut_ptr(), float32x4x3_t(p[0], p[1], p[2])),
                    4 => vst4q_f32(batch.as_mut_ptr(), float32x4x4_t(p[0], p[1], p[2], p[3])),
                    _ => unreachable!(),
                }
            } else {
                for row in batch.chunks_mut(K) {
                    super::softmax_scalar(row);
                }
            }
        }
        for row in batches.into_remainder().chunks_mut(K) {
            super::softmax_scalar(row);
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn short_softmax_gradient<const K: usize>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: dispatch checks complete matrices and NEON support. Each batch
    // covers four labels/weights and 4*K predictions/output pairs; K is 2..=4.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        let minimum = vdupq_n_f32(min_hess);
        let mut row = 0;
        while row + 4 <= labels.len() {
            let base = row * K;
            if let Some(probabilities) = short_softmax_batch::<K>(&preds[base..base + 4 * K]) {
                // Saturating conversion matches Rust's float-to-usize cast
                // when comparing with class IDs 0..K, including NaN/negatives.
                let target = vcvtq_u32_f32(vld1q_f32(labels.as_ptr().add(row)));
                let weight = match weights {
                    Some(weights) => vld1q_f32(weights.as_ptr().add(row)),
                    None => one,
                };
                let mut low = [vdupq_n_u64(0); K];
                let mut high = low;
                for class in 0..K {
                    let probability = probabilities[class];
                    let indicator =
                        vbslq_f32(vceqq_u32(target, vdupq_n_u32(class as u32)), one, zero);
                    let gradient = vmulq_f32(vsubq_f32(probability, indicator), weight);
                    let hessian = vmaxnmq_f32(
                        vmulq_f32(
                            vmulq_f32(vmulq_n_f32(probability, 2.0), vsubq_f32(one, probability)),
                            weight,
                        ),
                        minimum,
                    );
                    low[class] = vreinterpretq_u64_f32(vzip1q_f32(gradient, hessian));
                    high[class] = vreinterpretq_u64_f32(vzip2q_f32(gradient, hessian));
                }
                // Each u64 lane contains one repr(C) GradPair. Interleaving
                // classes restores row-major order, two complete rows per store.
                for (offset, pairs) in [(0, low), (2 * K, high)] {
                    let dest = out.as_mut_ptr().add(base + offset).cast::<u64>();
                    match K {
                        2 => vst2q_u64(dest, uint64x2x2_t(pairs[0], pairs[1])),
                        3 => {
                            // GradPair guarantees only f32 alignment. Avoid
                            // vst3q_u64: stdarch can implement it using a typed
                            // copy that requires an eight-byte-aligned pointer.
                            // Assemble [a0,b0,c0,a1,b1,c1] in three vectors.
                            let dest = dest.cast::<f32>();
                            vst1q_f32(dest, vreinterpretq_f32_u64(vzip1q_u64(pairs[0], pairs[1])));
                            vst1q_f32(
                                dest.add(4),
                                vreinterpretq_f32_u64(vzip1q_u64(
                                    pairs[2],
                                    vextq_u64::<1>(pairs[0], pairs[0]),
                                )),
                            );
                            vst1q_f32(
                                dest.add(8),
                                vreinterpretq_f32_u64(vzip2q_u64(pairs[1], pairs[2])),
                            );
                        }
                        4 => vst4q_u64(dest, uint64x2x4_t(pairs[0], pairs[1], pairs[2], pairs[3])),
                        _ => unreachable!(),
                    }
                }
            } else {
                for current in row..row + 4 {
                    let base = current * K;
                    super::softmax_gradient_row_scalar(
                        &preds[base..base + K],
                        labels[current] as usize,
                        weights.map_or(1.0, |weights| weights[current]),
                        min_hess,
                        &mut out[base..base + K],
                    );
                }
            }
            row += 4;
        }
        for current in row..labels.len() {
            let base = current * K;
            super::softmax_gradient_row_scalar(
                &preds[base..base + K],
                labels[current] as usize,
                weights.map_or(1.0, |weights| weights[current]),
                min_hess,
                &mut out[base..base + K],
            );
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_rows_inplace(values: &mut [f32], num_class: usize) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        for row in values.chunks_mut(num_class) {
            // SAFETY: this function may only be entered after NEON detection and
            // `chunks_mut` bounds every class row.
            softmax_inplace(row);
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        for row in 0..labels.len() {
            let base = row * num_class;
            // SAFETY: the dispatcher's complete-matrix check ensures these row
            // slices exist, and NEON was detected before this function was called.

            softmax_gradient_row(
                &preds[base..base + num_class],
                labels[row] as usize,
                weights.map_or(1.0, |values| values[row]),
                min_hess,
                &mut out[base..base + num_class],
            );
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn softmax_gradient_row(
    preds: &[f32],
    label: usize,
    weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let Some((min, max)) = finite_min_max(preds) else {
            super::softmax_gradient_row_scalar(preds, label, weight, min_hess, out);
            return;
        };
        if max - min > MAX_FAST_EXP_INPUT {
            super::softmax_gradient_row_scalar(preds, label, weight, min_hess, out);
            return;
        }

        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let max_vector = vdupq_n_f32(max);
        let mut sum_vector = zero;
        let mut index = 0;
        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the loop condition leaves four inputs and four output pairs.
            let input = vld1q_f32(preds.as_ptr().add(index));
            let exp = expq_f32::<true>(vsubq_f32(input, max_vector));
            sum_vector = vaddq_f32(sum_vector, exp);
            // Store the exponent in the gradient field as scratch space. The second
            // pass overwrites both fields with their final values.

            vst2q_f32(
                out.as_mut_ptr().add(index).cast::<f32>(),
                float32x4x2_t(exp, zero),
            );
            index += VECTOR_WIDTH;
        }
        let vector_end = index;
        let mut sum = vaddvq_f32(sum_vector);
        for index in vector_end..preds.len() {
            let exp = (preds[index] - max).exp();
            out[index].grad = exp;
            sum += exp;
        }

        let inverse = vdupq_n_f32(1.0 / sum);
        let row_weight = vdupq_n_f32(weight);
        let min_hessian = vdupq_n_f32(min_hess);
        let class_offsets_data = [0_u32, 1, 2, 3];
        // SAFETY: the local array contains exactly four u32 lanes.
        let class_offsets = vld1q_u32(class_offsets_data.as_ptr());
        let target_class = vdupq_n_u32(label as u32);
        index = 0;
        while index < vector_end {
            // SAFETY: the first pass initialized four GradPairs at this index.
            let scratch = vld2q_f32(out.as_ptr().add(index).cast::<f32>());
            let probability = vmulq_f32(scratch.0, inverse);
            let classes = vaddq_u32(vdupq_n_u32(index as u32), class_offsets);
            let target = vbslq_f32(vceqq_u32(classes, target_class), one, zero);
            let gradient = vmulq_f32(vsubq_f32(probability, target), row_weight);
            let hessian = vmaxq_f32(
                vmulq_f32(
                    vmulq_n_f32(vmulq_f32(probability, vsubq_f32(one, probability)), 2.0),
                    row_weight,
                ),
                min_hessian,
            );
            // SAFETY: the loop bounds leave four writable output pairs.

            vst2q_f32(
                out.as_mut_ptr().add(index).cast::<f32>(),
                float32x4x2_t(gradient, hessian),
            );
            index += VECTOR_WIDTH;
        }
        let inverse = 1.0 / sum;
        for (class, output) in out.iter_mut().enumerate().skip(vector_end) {
            let probability = output.grad * inverse;
            let target = if class == label { 1.0 } else { 0.0 };
            *output = GradPair::new(
                (probability - target) * weight,
                (2.0 * probability * (1.0 - probability) * weight).max(min_hess),
            );
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn distance_sum<const SQUARED: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let mut sum_low = vdupq_n_f64(0.0);
        let mut sum_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable values.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            let label = vld1q_f32(labels.as_ptr().add(index));
            let difference_low = vsubq_f64(
                vcvt_f64_f32(vget_low_f32(pred)),
                vcvt_f64_f32(vget_low_f32(label)),
            );
            let difference_high = vsubq_f64(vcvt_high_f64_f32(pred), vcvt_high_f64_f32(label));
            let distance_low = if SQUARED {
                vmulq_f64(difference_low, difference_low)
            } else {
                vabsq_f64(difference_low)
            };
            let distance_high = if SQUARED {
                vmulq_f64(difference_high, difference_high)
            } else {
                vabsq_f64(difference_high)
            };
            match weights {
                Some(weights) => {
                    let weight = vld1q_f32(weights.as_ptr().add(index));
                    let current_weight_low = vcvt_f64_f32(vget_low_f32(weight));
                    let current_weight_high = vcvt_high_f64_f32(weight);
                    sum_low = vfmaq_f64(sum_low, distance_low, current_weight_low);
                    sum_high = vfmaq_f64(sum_high, distance_high, current_weight_high);
                    weight_low = vaddq_f64(weight_low, current_weight_low);
                    weight_high = vaddq_f64(weight_high, current_weight_high);
                }
                None => {
                    sum_low = vaddq_f64(sum_low, distance_low);
                    sum_high = vaddq_f64(sum_high, distance_high);
                }
            }
            index += VECTOR_WIDTH;
        }

        let mut sum = vaddvq_f64(vaddq_f64(sum_low, sum_high));
        let mut weight_sum = match weights {
            Some(_) => vaddvq_f64(vaddq_f64(weight_low, weight_high)),
            None => index as f64,
        };
        for index in index..preds.len() {
            let weight = weights.map_or(1.0, |values| values[index] as f64);
            let difference = preds[index] as f64 - labels[index] as f64;
            let distance = if SQUARED {
                difference * difference
            } else {
                difference.abs()
            };
            sum += weight * distance;
            weight_sum += weight;
        }
        (sum, weight_sum)
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn classification_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let threshold = vdupq_n_f32(0.5);
        let mut index = 0;
        match weights {
            Some(weights) => {
                let mut wrong_low = vdupq_n_f64(0.0);
                let mut wrong_high = vdupq_n_f64(0.0);
                let mut weight_low = vdupq_n_f64(0.0);
                let mut weight_high = vdupq_n_f64(0.0);
                while index + VECTOR_WIDTH <= preds.len() {
                    // SAFETY: the common-length contract leaves four values in
                    // each input slice.
                    let pred = vld1q_f32(preds.as_ptr().add(index));
                    let label = vld1q_f32(labels.as_ptr().add(index));
                    let weight = vld1q_f32(weights.as_ptr().add(index));
                    let mismatch =
                        veorq_u32(vcgtq_f32(pred, threshold), vcgtq_f32(label, threshold));
                    let wrong_weight = vbslq_f32(mismatch, weight, vdupq_n_f32(0.0));
                    let current_wrong_low = vcvt_f64_f32(vget_low_f32(wrong_weight));
                    let current_wrong_high = vcvt_high_f64_f32(wrong_weight);
                    let current_weight_low = vcvt_f64_f32(vget_low_f32(weight));
                    let current_weight_high = vcvt_high_f64_f32(weight);
                    wrong_low = vaddq_f64(wrong_low, current_wrong_low);
                    wrong_high = vaddq_f64(wrong_high, current_wrong_high);
                    weight_low = vaddq_f64(weight_low, current_weight_low);
                    weight_high = vaddq_f64(weight_high, current_weight_high);
                    index += VECTOR_WIDTH;
                }
                let mut wrong = vaddvq_f64(vaddq_f64(wrong_low, wrong_high));
                let mut weight_sum = vaddvq_f64(vaddq_f64(weight_low, weight_high));
                for index in index..preds.len() {
                    let weight = weights[index] as f64;
                    if (preds[index] > 0.5) != (labels[index] > 0.5) {
                        wrong += weight;
                    }
                    weight_sum += weight;
                }
                (wrong, weight_sum)
            }
            None => {
                let one = vdupq_n_u32(1);
                let mut wrong_low = vdupq_n_u64(0);
                let mut wrong_high = vdupq_n_u64(0);
                while index + VECTOR_WIDTH <= preds.len() {
                    // SAFETY: the common-length contract leaves four values in
                    // each input slice.
                    let pred = vld1q_f32(preds.as_ptr().add(index));
                    let label = vld1q_f32(labels.as_ptr().add(index));
                    let mismatch = vandq_u32(
                        veorq_u32(vcgtq_f32(pred, threshold), vcgtq_f32(label, threshold)),
                        one,
                    );
                    wrong_low = vaddq_u64(wrong_low, vmovl_u32(vget_low_u32(mismatch)));
                    wrong_high = vaddq_u64(wrong_high, vmovl_high_u32(mismatch));
                    index += VECTOR_WIDTH;
                }
                let mut wrong = vaddvq_u64(vaddq_u64(wrong_low, wrong_high)) as f64;
                for index in index..preds.len() {
                    wrong += f64::from((preds[index] > 0.5) != (labels[index] > 0.5));
                }
                (wrong, preds.len() as f64)
            }
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn log_loss_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let one = vdupq_n_f64(1.0);
        let lower = vdupq_n_f64(LOG_LOSS_EPSILON);
        let upper = vdupq_n_f64(1.0 - LOG_LOSS_EPSILON);
        let mut loss_low = vdupq_n_f64(0.0);
        let mut loss_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut fallback = (0.0, 0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable values.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            if !finite_input(pred) {
                let partial = scalar::log_loss(preds, labels, weights, index..index + VECTOR_WIDTH);
                fallback.0 += partial.0;
                fallback.1 += partial.1;
                index += VECTOR_WIDTH;
                continue;
            }
            let label = vld1q_f32(labels.as_ptr().add(index));
            let probability_low =
                vminq_f64(vmaxq_f64(vcvt_f64_f32(vget_low_f32(pred)), lower), upper);
            let probability_high = vminq_f64(vmaxq_f64(vcvt_high_f64_f32(pred), lower), upper);
            let label_low = vcvt_f64_f32(vget_low_f32(label));
            let label_high = vcvt_high_f64_f32(label);
            let value_low = vnegq_f64(vaddq_f64(
                vmulq_f64(label_low, logq_f64(probability_low)),
                vmulq_f64(
                    vsubq_f64(one, label_low),
                    logq_f64(vsubq_f64(one, probability_low)),
                ),
            ));
            let value_high = vnegq_f64(vaddq_f64(
                vmulq_f64(label_high, logq_f64(probability_high)),
                vmulq_f64(
                    vsubq_f64(one, label_high),
                    logq_f64(vsubq_f64(one, probability_high)),
                ),
            ));
            match weights {
                Some(weights) => {
                    let weight = vld1q_f32(weights.as_ptr().add(index));
                    let current_weight_low = vcvt_f64_f32(vget_low_f32(weight));
                    let current_weight_high = vcvt_high_f64_f32(weight);
                    loss_low = vfmaq_f64(loss_low, value_low, current_weight_low);
                    loss_high = vfmaq_f64(loss_high, value_high, current_weight_high);
                    weight_low = vaddq_f64(weight_low, current_weight_low);
                    weight_high = vaddq_f64(weight_high, current_weight_high);
                }
                None => {
                    loss_low = vaddq_f64(loss_low, value_low);
                    loss_high = vaddq_f64(loss_high, value_high);
                }
            }
            index += VECTOR_WIDTH;
        }

        let tail = scalar::log_loss(preds, labels, weights, index..preds.len());
        let loss = vaddvq_f64(vaddq_f64(loss_low, loss_high)) + fallback.0 + tail.0;
        let weight_sum = match weights {
            Some(_) => vaddvq_f64(vaddq_f64(weight_low, weight_high)) + fallback.1 + tail.1,
            None => preds.len() as f64,
        };
        (loss, weight_sum)
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn positive_nloglik_sum<const GAMMA: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let minimum = vdupq_n_f64(MIN_POSITIVE_PREDICTION);
        let mut loss_low = vdupq_n_f64(0.0);
        let mut loss_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut fallback = (0.0, 0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable values.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            if !finite_input(pred) {
                let partial = scalar::positive_nloglik::<GAMMA>(
                    preds,
                    labels,
                    weights,
                    index..index + VECTOR_WIDTH,
                );
                fallback.0 += partial.0;
                fallback.1 += partial.1;
                index += VECTOR_WIDTH;
                continue;
            }
            let label = vld1q_f32(labels.as_ptr().add(index));
            let prediction_low = vmaxq_f64(vcvt_f64_f32(vget_low_f32(pred)), minimum);
            let prediction_high = vmaxq_f64(vcvt_high_f64_f32(pred), minimum);
            let label_low = vcvt_f64_f32(vget_low_f32(label));
            let label_high = vcvt_high_f64_f32(label);
            let log_low = logq_f64(prediction_low);
            let log_high = logq_f64(prediction_high);
            let value_low = if GAMMA {
                vaddq_f64(vdivq_f64(label_low, prediction_low), log_low)
            } else {
                vsubq_f64(prediction_low, vmulq_f64(label_low, log_low))
            };
            let value_high = if GAMMA {
                vaddq_f64(vdivq_f64(label_high, prediction_high), log_high)
            } else {
                vsubq_f64(prediction_high, vmulq_f64(label_high, log_high))
            };
            match weights {
                Some(weights) => {
                    let weight = vld1q_f32(weights.as_ptr().add(index));
                    let current_weight_low = vcvt_f64_f32(vget_low_f32(weight));
                    let current_weight_high = vcvt_high_f64_f32(weight);
                    loss_low = vfmaq_f64(loss_low, value_low, current_weight_low);
                    loss_high = vfmaq_f64(loss_high, value_high, current_weight_high);
                    weight_low = vaddq_f64(weight_low, current_weight_low);
                    weight_high = vaddq_f64(weight_high, current_weight_high);
                }
                None => {
                    loss_low = vaddq_f64(loss_low, value_low);
                    loss_high = vaddq_f64(loss_high, value_high);
                }
            }
            index += VECTOR_WIDTH;
        }

        let tail = scalar::positive_nloglik::<GAMMA>(preds, labels, weights, index..preds.len());
        let loss = vaddvq_f64(vaddq_f64(loss_low, loss_high)) + fallback.0 + tail.0;
        let weight_sum = match weights {
            Some(_) => vaddvq_f64(vaddq_f64(weight_low, weight_high)) + fallback.1 + tail.1,
            None => preds.len() as f64,
        };
        (loss, weight_sum)
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn tweedie_nloglik_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f64,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let minimum = vdupq_n_f64(MIN_POSITIVE_PREDICTION);
        let first_power = vdupq_n_f64(1.0 - rho);
        let second_power = vdupq_n_f64(2.0 - rho);
        let first_denominator = vdupq_n_f64(1.0 - rho);
        let second_denominator = vdupq_n_f64(2.0 - rho);
        let mut loss_low = vdupq_n_f64(0.0);
        let mut loss_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut fallback = (0.0, 0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable values.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            if !finite_input(pred) {
                let end = index + VECTOR_WIDTH;
                let partial = super::tweedie_nloglik_sum_scalar(
                    &preds[index..end],
                    &labels[index..end],
                    weights.map(|values| &values[index..end]),
                    rho,
                );
                fallback.0 += partial.0;
                fallback.1 += partial.1;
                index = end;
                continue;
            }
            let label = vld1q_f32(labels.as_ptr().add(index));
            let prediction_low = vmaxq_f64(vcvt_f64_f32(vget_low_f32(pred)), minimum);
            let prediction_high = vmaxq_f64(vcvt_high_f64_f32(pred), minimum);
            let label_low = vcvt_f64_f32(vget_low_f32(label));
            let label_high = vcvt_high_f64_f32(label);
            let log_low = logq_f64(prediction_low);
            let log_high = logq_f64(prediction_high);
            let first_low = vdivq_f64(
                vmulq_f64(label_low, expq_f64(vmulq_f64(first_power, log_low))),
                first_denominator,
            );
            let first_high = vdivq_f64(
                vmulq_f64(label_high, expq_f64(vmulq_f64(first_power, log_high))),
                first_denominator,
            );
            let second_low = vdivq_f64(
                expq_f64(vmulq_f64(second_power, log_low)),
                second_denominator,
            );
            let second_high = vdivq_f64(
                expq_f64(vmulq_f64(second_power, log_high)),
                second_denominator,
            );
            let value_low = vsubq_f64(second_low, first_low);
            let value_high = vsubq_f64(second_high, first_high);
            match weights {
                Some(weights) => {
                    let weight = vld1q_f32(weights.as_ptr().add(index));
                    let current_weight_low = vcvt_f64_f32(vget_low_f32(weight));
                    let current_weight_high = vcvt_high_f64_f32(weight);
                    loss_low = vfmaq_f64(loss_low, value_low, current_weight_low);
                    loss_high = vfmaq_f64(loss_high, value_high, current_weight_high);
                    weight_low = vaddq_f64(weight_low, current_weight_low);
                    weight_high = vaddq_f64(weight_high, current_weight_high);
                }
                None => {
                    loss_low = vaddq_f64(loss_low, value_low);
                    loss_high = vaddq_f64(loss_high, value_high);
                }
            }
            index += VECTOR_WIDTH;
        }

        let tail = super::tweedie_nloglik_sum_scalar(
            &preds[index..],
            &labels[index..],
            weights.map(|values| &values[index..]),
            rho,
        );
        let loss = vaddvq_f64(vaddq_f64(loss_low, loss_high)) + fallback.0 + tail.0;
        let weight_sum = match weights {
            Some(_) => vaddvq_f64(vaddq_f64(weight_low, weight_high)) + fallback.1 + tail.1,
            None => preds.len() as f64,
        };
        (loss, weight_sum)
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn multiclass_log_loss_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let lower = vdupq_n_f64(LOG_LOSS_EPSILON);
        let upper = vdupq_n_f64(1.0);
        let mut loss_low = vdupq_n_f64(0.0);
        let mut loss_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut fallback = (0.0, 0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= labels.len() {
            let selected = [
                preds[index * num_class + labels[index] as usize],
                preds[(index + 1) * num_class + labels[index + 1] as usize],
                preds[(index + 2) * num_class + labels[index + 2] as usize],
                preds[(index + 3) * num_class + labels[index + 3] as usize],
            ];
            // SAFETY: `selected` contains exactly four f32 lanes.
            let probability = vld1q_f32(selected.as_ptr());
            if !finite_input(probability) {
                let end = index + VECTOR_WIDTH;
                let partial = super::multiclass_log_loss_sum_scalar(
                    &preds[index * num_class..end * num_class],
                    &labels[index..end],
                    weights.map(|values| &values[index..end]),
                    num_class,
                );
                fallback.0 += partial.0;
                fallback.1 += partial.1;
                index = end;
                continue;
            }

            let probability_low = vminq_f64(
                vmaxq_f64(vcvt_f64_f32(vget_low_f32(probability)), lower),
                upper,
            );
            let probability_high =
                vminq_f64(vmaxq_f64(vcvt_high_f64_f32(probability), lower), upper);
            let value_low = vnegq_f64(logq_f64(probability_low));
            let value_high = vnegq_f64(logq_f64(probability_high));
            match weights {
                Some(weights) => {
                    // SAFETY: the dispatcher verified one weight per label.
                    let weight = vld1q_f32(weights.as_ptr().add(index));
                    let current_weight_low = vcvt_f64_f32(vget_low_f32(weight));
                    let current_weight_high = vcvt_high_f64_f32(weight);
                    loss_low = vfmaq_f64(loss_low, value_low, current_weight_low);
                    loss_high = vfmaq_f64(loss_high, value_high, current_weight_high);
                    weight_low = vaddq_f64(weight_low, current_weight_low);
                    weight_high = vaddq_f64(weight_high, current_weight_high);
                }
                None => {
                    loss_low = vaddq_f64(loss_low, value_low);
                    loss_high = vaddq_f64(loss_high, value_high);
                }
            }
            index += VECTOR_WIDTH;
        }

        let tail = super::multiclass_log_loss_sum_scalar(
            &preds[index * num_class..labels.len() * num_class],
            &labels[index..],
            weights.map(|values| &values[index..]),
            num_class,
        );
        let loss = vaddvq_f64(vaddq_f64(loss_low, loss_high)) + fallback.0 + tail.0;
        let weight_sum = match weights {
            Some(_) => vaddvq_f64(vaddq_f64(weight_low, weight_high)) + fallback.1 + tail.1,
            None => labels.len() as f64,
        };
        (loss, weight_sum)
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn multiclass_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let mut wrong = 0.0;
        let mut weight_sum = 0.0;
        for (row_index, &label) in labels.iter().enumerate() {
            let row = &preds[row_index * num_class..(row_index + 1) * num_class];
            // SAFETY: the dispatcher verified `num_class` fits u32 and NEON exists.
            let best = argmax(row);
            let weight = weights.map_or(1.0, |values| values[row_index] as f64);
            if best != label as usize {
                wrong += weight;
            }
            weight_sum += weight;
        }
        (wrong, weight_sum)
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn argmax(values: &[f32]) -> usize {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let offsets_data = [0_u32, 1, 2, 3];
        // SAFETY: the local array contains four u32 lanes.
        let offsets = vld1q_u32(offsets_data.as_ptr());
        let mut maxima = vdupq_n_f32(f32::NEG_INFINITY);
        let mut best_indices = vdupq_n_u32(0);
        let mut index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable values.
            let value = vld1q_f32(values.as_ptr().add(index));
            if !finite_input(value) {
                return super::argmax_scalar(values);
            }
            let better = vcgtq_f32(value, maxima);
            maxima = vbslq_f32(better, value, maxima);
            let indices = vaddq_u32(vdupq_n_u32(index as u32), offsets);
            best_indices = vbslq_u32(better, indices, best_indices);
            index += VECTOR_WIDTH;
        }

        let mut lane_maxima = [0.0_f32; VECTOR_WIDTH];
        let mut lane_indices = [0_u32; VECTOR_WIDTH];
        // SAFETY: each output array contains four writable lanes.

        vst1q_f32(lane_maxima.as_mut_ptr(), maxima);
        vst1q_u32(lane_indices.as_mut_ptr(), best_indices);

        let mut max = lane_maxima[0];
        let mut best = lane_indices[0] as usize;
        for lane in 1..VECTOR_WIDTH {
            if lane_maxima[lane] > max
                || (lane_maxima[lane] == max && (lane_indices[lane] as usize) < best)
            {
                max = lane_maxima[lane];
                best = lane_indices[lane] as usize;
            }
        }
        for (tail_index, &value) in values[index..].iter().enumerate() {
            if !value.is_finite() {
                return super::argmax_scalar(values);
            }
            let class = index + tail_index;
            if value > max {
                max = value;
                best = class;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests;
