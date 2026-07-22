# AGENTS.md — guide for AI coding agents

Context for automated agents working with **sequoia-boost**, a faithful,
pure-Rust reimplementation of XGBoost gradient boosting (no C/C++, no FFI).
Human docs: `README.md` and [docs.rs](https://docs.rs/sequoia-boost).

## Using it as a dependency

```toml
[dependencies]
sequoia-boost = "0.1"
```

```rust
use sequoia_boost::prelude::*;

fn main() -> Result<()> {
    // features: row-major &[f32] of length n_rows * n_cols; labels: &[f32] of n_rows
    let dtrain = DMatrix::from_dense(&x, n_rows, n_cols)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .objective("reg:squarederror")   // XGBoost-compatible name
        .max_depth(6).eta(0.1).build()?;
    let model = train(&params, &dtrain, 100)?;
    let preds = model.predict(&dtrain)?; // Vec<f32>, length n_rows (n_rows*num_class for multiclass)
    Ok(())
}
```

Everything in the typical workflow is re-exported from `sequoia_boost::prelude`.

## Key entry points

| Function | Purpose |
|---|---|
| `train(&params, &dtrain, num_round) -> Result<BoostedModel>` | Basic training. |
| `train_with_eval(&params, &dtrain, num_round, &[(&DMatrix, "name")], early_stopping_rounds: Option<usize>) -> Result<TrainResult>` | Watch eval sets + early stopping. `TrainResult { model, history }`. |
| `train_with_objective(&params, &dtrain, num_round, Box<dyn Objective>) -> Result<BoostedModel>` | Custom loss (see `CustomObjective`). |
| `train_with_custom_metric(&params, &dtrain, num_round, evals, early, Box<dyn Metric>) -> Result<TrainResult>` | Custom eval metric (see `CustomMetric`). |
| `cv(&params, &data, num_round, nfold, seed) -> Result<Vec<CvResult>>` | K-fold cross-validation. |

## Core types

- **`DMatrix`** — data. `from_dense(&[f32], rows, cols)`, `from_csr(indptr, indices, values, cols)`, `from_libsvm(path)`, `from_csv`/`read_csv(reader, &CsvOptions)`. Chainable: `.with_labels(&[f32])`, `.with_weights`, `.with_base_margin` (warm-start), `.with_group_sizes(&[usize])` (ranking), `.with_feature_types(&[FeatureType])` (categorical).
- **`TrainingParams`** — config via `TrainingParams::builder()...build()?`. Field/method names mirror XGBoost: `objective`, `num_class`, `eta`, `max_depth`, `max_leaves`, `min_child_weight`, `gamma`, `lambda`, `alpha`, `subsample`, `colsample_bytree`/`bylevel`/`bynode`, `max_bin`, `tree_method` (`TreeMethod::{Auto,Hist,Exact,Approx}`), `grow_policy` (`GrowPolicy::{DepthWise,LossGuide}`), `booster` (`BoosterKind::{GbTree,Dart,GbLinear}`), `monotone_constraints(Vec<Monotone>)`, `interaction_constraints(Vec<Vec<u32>>)`, `base_score`, `eval_metric(name)`, `seed`.
- **`BoostedModel`** — trained model. `predict` (reported space, e.g. probabilities), `predict_margin` (raw), `predict_class` (argmax), `predict_leaf`, `predict_contribs` (TreeSHAP), `predict_interactions` (TreeSHAP interactions), `feature_importance(ImportanceType)`, `num_trees`. I/O: `save_binary`/`load_binary`, `to_json`/`from_json`/`save_json`/`load_json`, and `save_xgboost_json`/`load_xgboost_json` (interop with real XGBoost).

## Supported names (strings passed to `.objective(...)` / `.eval_metric(...)`)

- Objectives: `reg:squarederror`, `reg:pseudohubererror`, `reg:gamma`, `reg:tweedie`, `count:poisson`, `binary:logistic`, `multi:softmax`, `multi:softprob` (need `.num_class(k)`), `rank:pairwise`, `rank:ndcg`, `rank:map` (need `.with_group_sizes`).
- Metrics: `rmse`, `mae`, `logloss`, `error`, `auc`, `aucpr`, `mlogloss`, `merror`, `ndcg`, `map` (accept `@k`, e.g. `ndcg@5`), `poisson-nloglik`, `gamma-nloglik`, `tweedie-nloglik`.

## Conventions & gotchas

- **Prediction layout:** single-output → length `n_rows`. Multiclass → `n_rows * num_class`, row-major `[row][class]`. SHAP contribs → `[row][n_features + 1]` (last = bias). SHAP interactions → `[row][(n_features+1)^2]`.
- **Errors:** everything returns `Result<T, SequoiaError>` (`Result` alias is in the prelude). Prefer `?`; don't `unwrap` in library code.
- **Determinism:** identical `(params, data, seed)` ⇒ identical predictions (property-tested).
- **Parity:** matches XGBoost *model quality* (CI-tested via `tests/parity.rs`), not bit-identical predictions.
- **`num_class` is required** for `multi:*`; ranking objectives require `with_group_sizes`.
- No `unsafe` in the public API; `#![forbid(unsafe_op_in_unsafe_fn)]`.

## Runnable examples (`crates/sequoia-boost/examples/`)

`binary_classification`, `multiclass`, `ranking`, `shap`, `model_io`,
`custom_objective`, `constraints`, `train_regression`. Run one with
`cargo run --release --example <name>`. These are the best copy-paste starting
points for each workflow.

## Repo commands (from the workspace root)

```sh
cargo test --workspace                       # unit + property + doc tests
cargo clippy --all-targets -- -D warnings    # lints (CI-enforced)
cargo fmt --all --check                      # formatting (CI-enforced)
cargo run --release --example <name>         # run an example
cargo bench -p sequoia-boost                 # criterion micro-benchmarks
```

## Module map (`crates/sequoia-boost/src/`)

`data/` (DMatrix, quantile binning, ghist) · `config/` (TrainingParams) ·
`objective/` · `metric/` · `tree/` (regtree, gain, constraints, sampler,
`builder/{exact,hist,approx}`, `hist/` backend) · `booster/` (gblinear) ·
`learner/` (train loop, model, cv, shap) · `model/` (XGBoost-JSON I/O).

## Scope

Not yet implemented: UBJSON (binary) XGBoost format, GPU backend,
distributed/external-memory training, and language wrappers (Python/CLI/C-ABI).
Do not assume these exist.
