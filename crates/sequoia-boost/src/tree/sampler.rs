//! Column (feature) subsampling shared by the tree builders.
//!
//! XGBoost exposes three cumulative column-sampling ratios: `colsample_bytree`
//! (per tree), `colsample_bylevel` (per level), and `colsample_bynode` (per
//! node). The per-tree sample is drawn by the trainer and passed in here as the
//! *pool*; this sampler then draws the `bylevel` and `bynode` subsets from it.
//!
//! Call granularity differs by builder: the histogram builder samples once per
//! node ([`ColumnSampler::sample`] per node), while the exact builder samples
//! once per level (shared across that level's nodes). With the default ratios of
//! `1.0` every draw returns the full pool, so behavior is unchanged.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Draws feature subsets from a per-tree pool according to the `bylevel` and
/// `bynode` ratios.
#[derive(Debug, Clone)]
pub struct ColumnSampler {
    pool: Vec<u32>,
    bylevel: f64,
    bynode: f64,
    rng: StdRng,
}

impl ColumnSampler {
    /// Build a sampler over `pool` (the per-tree / `colsample_bytree` features).
    pub fn new(pool: Vec<u32>, bylevel: f64, bynode: f64, seed: u64) -> Self {
        ColumnSampler {
            pool,
            bylevel,
            bynode,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// A pass-through sampler over all `n_features` columns (ratios `1.0`),
    /// primarily for tests and callers that do no column sampling.
    pub fn all(n_features: usize) -> Self {
        ColumnSampler::new((0..n_features as u32).collect(), 1.0, 1.0, 0)
    }

    /// The underlying per-tree pool.
    #[inline]
    pub fn pool(&self) -> &[u32] {
        &self.pool
    }

    /// Draw a feature subset: the `bylevel` sample of the pool, then the
    /// `bynode` sample of that. Returned features are sorted ascending.
    pub fn sample(&mut self) -> Vec<u32> {
        let level = subsample(&self.pool, self.bylevel, &mut self.rng);
        subsample(&level, self.bynode, &mut self.rng)
    }
}

/// Sample `round(ratio * len)` features (at least one) without replacement,
/// sorted ascending. Returns a clone of `pool` when `ratio >= 1`.
fn subsample(pool: &[u32], ratio: f64, rng: &mut StdRng) -> Vec<u32> {
    if ratio >= 1.0 || pool.len() <= 1 {
        return pool.to_vec();
    }
    let k = ((ratio * pool.len() as f64).round() as usize).clamp(1, pool.len());
    let mut idx = pool.to_vec();
    idx.shuffle(rng);
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_through_when_ratios_one() {
        let mut s = ColumnSampler::all(10);
        let f = s.sample();
        assert_eq!(f, (0..10u32).collect::<Vec<_>>());
    }

    #[test]
    fn composes_bylevel_and_bynode() {
        // pool of 100, bylevel 0.5, bynode 0.5 -> ~25 features, subset of pool.
        let pool: Vec<u32> = (0..100).collect();
        let mut s = ColumnSampler::new(pool.clone(), 0.5, 0.5, 42);
        let f = s.sample();
        assert_eq!(f.len(), 25); // round(0.5*100)=50, round(0.5*50)=25
        assert!(f.windows(2).all(|w| w[0] < w[1]), "sorted & unique");
        assert!(f.iter().all(|x| pool.contains(x)));
    }

    #[test]
    fn deterministic_for_seed() {
        let pool: Vec<u32> = (0..50).collect();
        let mut a = ColumnSampler::new(pool.clone(), 0.6, 1.0, 7);
        let mut b = ColumnSampler::new(pool, 0.6, 1.0, 7);
        assert_eq!(a.sample(), b.sample());
    }

    #[test]
    fn always_at_least_one() {
        let pool: Vec<u32> = (0..3).collect();
        let mut s = ColumnSampler::new(pool, 0.01, 0.01, 1);
        assert!(!s.sample().is_empty());
    }
}
