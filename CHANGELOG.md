# Changelog

All notable changes to `sequoia-boost` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-10-12

### Added

- Runtime-dispatched AArch64 NEON kernels for objective gradients, prediction
  transforms, metric reductions, and dense numeric histogram split evaluation.
  Scalar fallbacks handle other CPUs, short inputs, and values outside the
  approximation ranges.
- Runtime-dispatched x86-64 kernels: an AVX2+FMA dense split-gain scan whose
  division-free prefilter hands surviving candidates to the exact scalar test
  (split choices are bit-identical), AVX2 exponential, sigmoid, logistic
  gradient, and short softmax (2 and 4 classes) kernels, and an SSE2 16-cut
  search for quantile binning. On x86-64 the histogram bin sums keep the scalar
  summation order.
- Kernel and tree-building benchmarks with reproducible comparisons and
  [documented results](docs/performance.md).
- Public API additions (no existing public item is removed or changed):
  `GHistIndex::dense_stride` and `GHistIndex::column_bins` for the column-major
  bin layout, `data::quantile::BinSearch` (`new`, `n_features`, `bin_of`) for
  two-level cut search, and `RegTree::leaf_id_dense` for routing a dense row
  without a closure.

### Changed

- Skip child histograms and split searches for depthwise leaves at `max_depth`,
  preserving leaf statistics, monotone bounds, and column-sampling order.
- Parallelize quantile cuts, bin assignment, and depthwise node construction.
  Size histogram tasks to the available rows and reuse final training-row
  partitions in smaller thread pools.
- Rewrite prediction around a lazily built branch-free tree layout: breadth-first
  node arena, single-compare split encoding with mirrored default-right
  children, sixteen-row (or sixteen-tree) lockstep traversal, block-parallel
  batches, and per-block densification of sparse rows. `predict`,
  `predict_margin`, `predict_leaf`, and single-row prediction are 10× to 20× faster
  with identical results.
- Speed up TreeSHAP contributions and interactions 8× with an arena-backed
  decision path, precomputed cover fractions, hoisted divisions, a shared
  unwound sum for off-path elements, and row-parallel evaluation.
- Compare prediction splits on monotone `u32` keys (NaN maps to the missing
  direction, mirrored nodes read a precomputed negated key) loaded together
  with the node's feature slot in one 64-bit load. Batch prediction is
  25 to 35% faster with identical results.
- Build the two children's split evaluations concurrently on large nodes and
  reduce partial histograms in parallel by bin range, keeping the per-bin
  addition order. Root histograms over contiguous row ranges are built
  column-wise with one writer per feature, and sibling partitions are written into
  spare capacity instead of zero-filled buffers.
- Compute built-in row-independent gradients (squared error, logistic, softmax)
  over fixed row chunks in parallel, bit-identical to the whole-batch result.
- Evaluate TreeSHAP hot path elements in monomorphized lanes, skip the unit
  `one_fraction` division, and reuse the parent's path region for the cold
  child, for a further ~5% reduction with per-element results unchanged.

### Fixed

- Make the AArch64 NEON dense split scan bit-identical to the scalar scan. Its
  vector prefilter compared a bound that could overflow to infinity (where
  `inf > inf` reads false) or round to a subnormal, silently dropping
  candidates the scalar scan accepts. The prefilter now saturates its bound and routes
  subnormal products and intermediates to the exact check. Surviving
  candidates are also accepted with the scalar `calc_gain` arithmetic, which
  scores zero-Hessian children with a zero gain instead of rejecting them, so
  the chosen split, statistics, and loss match the scalar scan and the x86-64
  kernel bit for bit.

## [0.2.0] - 2026-08-16

### Fixed

- Correct DART tree normalization to use the learning rate in the normalization
  denominator.
- Enforce interaction constraints in exact trees and isolate features omitted
  from configured interaction groups.
- Make `multi:softmax` return one class label per row while preserving the
  probability matrix returned by `multi:softprob`.
- Include DART weights, early-stopping limits, dataset base margins, and linear
  boosters in SHAP contributions and interaction values.
- Reject malformed datasets, invalid objective labels, incompatible evaluation
  sets, invalid early-stopping settings, and prediction feature mismatches with
  structured errors instead of panics or silently ignored inputs.
- Honor per-row base margins in `gblinear` and honor `nthread` with a scoped
  Rayon pool.
- Interpret explicit count, Gamma, and Tweedie base scores in reported-value
  space and pass configured Poisson `max_delta_step` into its Hessian.
- Add query-level ranking weights and use them when averaging NDCG and MAP,
  matching XGBoost's ranking-weight semantics.
- Reject XGBoost JSON export for DART, `gblinear`, and categorical models that
  cannot be represented correctly by the current exporter.
- Replace the unmaintained bincode native serializer with Postcard. Native
  binary blobs from 0.1.0 are not compatible with 0.2.0. JSON remains the
  portable native interchange format across these versions.
- Return `Result` from margin-limited and leaf prediction APIs so feature-count
  and base-margin mismatches cannot silently produce predictions.

### Changed

- Prepare version 0.2.0 citation, Zenodo, and Cargo release metadata. No DOI is
  claimed until the first Zenodo archive has been published.

## [0.1.0]

Initial release: a faithful, pure-Rust reimplementation of XGBoost gradient
boosting.

### Boosters & tree construction
- Boosters: `gbtree`, `dart` (tree dropout), `gblinear` (coordinate descent).
- Tree methods: `exact`, `hist` (histogram + subtraction trick), `approx`
  (hessian-weighted per-round binning), plus `depthwise` and `lossguide` growth.
- Sparsity-aware missing-value handling, row and column subsampling
  (`bytree` / `bylevel` / `bynode`), and multi-core histogram construction (`rayon`).

### Objectives & metrics
- Objectives: `reg:squarederror`, `reg:pseudohubererror`, `reg:gamma`,
  `reg:tweedie`, `count:poisson`, `binary:logistic`, `multi:softmax`,
  `multi:softprob`, learning-to-rank (`rank:pairwise` / `rank:ndcg` / `rank:map`,
  LambdaMART), plus a custom-objective hook.
- Metrics: `rmse`, `mae`, `logloss`, `error`, `auc`, `aucpr`, `mlogloss`,
  `merror`, `ndcg`, `map` (with `@k`), `poisson/gamma/tweedie-nloglik`, plus a
  custom-metric hook.

### Modeling
- Monotone constraints and interaction constraints (both `hist` and `exact`).
- Native categorical splits (both `hist` and `exact`).
- Per-instance `base_margin` (warm-start / stacking), early stopping, feature
  importance (weight / gain / cover / totals).
- TreeSHAP feature contributions (`predict_contribs`) and interaction values
  (`predict_interactions`).

### I/O & tooling
- Loaders: libsvm, CSV, dense/CSR in-memory.
- Model I/O: native binary + JSON, and XGBoost-format JSON import/export.
- K-fold cross-validation.

### Quality
- Unit, property (`proptest`), and doc tests. XGBoost model-quality parity is
  verified in CI against real XGBoost.

[Unreleased]: https://github.com/pgarrett-scripps/sequoia-boost/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/pgarrett-scripps/sequoia-boost/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pgarrett-scripps/sequoia-boost/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pgarrett-scripps/sequoia-boost/releases/tag/v0.1.0
