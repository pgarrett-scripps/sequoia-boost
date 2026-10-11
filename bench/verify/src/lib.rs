//! Shared helpers for the verification binaries.

/// Deterministic LCG in [0, 1) so every checkout sees identical data.
pub fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32) / ((1u64 << 31) as f32)
}

/// The SIMD path the library's runtime dispatch will take on this CPU. The
/// library uses the same feature-detection macros.
pub fn dispatch_path() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return "aarch64 NEON";
        }
        return "aarch64 scalar (NEON not detected)";
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return "x86_64 AVX2+FMA";
        }
        return "x86_64 scalar/SSE2 (AVX2+FMA not detected)";
    }
    #[allow(unreachable_code)]
    "scalar"
}
