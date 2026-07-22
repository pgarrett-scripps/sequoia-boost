# Changelog

All notable changes to `sequoia-boost` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

Initial release: a faithful, pure-Rust reimplementation of XGBoost gradient
boosting.

### Boosters & tree construction
- Boosters: `gbtree`, `dart` (tree dropout), `gblinear` (coordinate descent).
- Tree methods: `exact`, `hist` (histogram + subtraction trick), `approx`
  (hessian-weighted per-round binning); `depthwise` and `lossguide` growth.
- Sparsity-aware missing-value handling; row and column subsampling
  (`bytree` / `bylevel` / `bynode`); multi-core histogram construction (`rayon`).

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
- Unit, property (`proptest`), and doc tests; XGBoost model-quality parity is
  verified in CI against real XGBoost.

[Unreleased]: https://github.com/pgarrett-scripps/sequoia-boost/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pgarrett-scripps/sequoia-boost/releases/tag/v0.1.0
