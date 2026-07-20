# sequoia-boost

A faithful, fast, pure-Rust reimplementation of [XGBoost](https://github.com/dmlc/xgboost)
gradient boosting.

`sequoia-boost` re-implements XGBoost's algorithms from scratch in idiomatic
Rust — the regularized second-order boosting objective, exact and histogram tree
construction, a broad objective/metric catalog, monotone constraints, and native
model persistence — with multi-core (`rayon`) and SIMD-friendly acceleration.

> **Status:** early but functional. The core training/prediction paths are
> implemented and tested; some advanced features are still in progress (see
> [Feature status](#feature-status)).

## Quick start

```rust
use sequoia_boost::prelude::*;

fn main() -> Result<()> {
    // Dense features (row-major) + labels.
    let x: Vec<f32> = /* n_rows * n_cols values */ vec![0.0; 400];
    let y: Vec<f32> = vec![0.0; 100];

    let dtrain = DMatrix::from_dense(&x, 100, 4)?.with_labels(&y)?;

    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)   // fast histogram method
        .max_depth(6)
        .eta(0.1)
        .subsample(0.9)
        .lambda(1.0)
        .build()?;

    let model = train(&params, &dtrain, 200)?;
    let preds = model.predict(&dtrain)?;

    model.save_binary("model.sqb")?;
    Ok(())
}
```

## Feature status

**Implemented & tested**

- **Boosters/trees:** `gbtree`; `tree_method = exact | hist`; `grow_policy =
  depthwise | lossguide`; histogram binning with the parent−child subtraction
  trick; sparsity-aware missing-value handling; row/column subsampling.
- **Regularization:** `lambda`, `alpha`, `gamma`, `min_child_weight`,
  `max_delta_step`, `max_depth`, `max_leaves`, `max_bin`.
- **Objectives:** `reg:squarederror`, `reg:pseudohubererror`, `binary:logistic`,
  `multi:softmax`, `multi:softprob`, `count:poisson`, `reg:gamma`, `reg:tweedie`,
  and a user **custom-objective hook**.
- **Metrics:** `rmse`, `mae`, `logloss`, `error`, `auc`, `mlogloss`, `merror`,
  `poisson/gamma/tweedie-nloglik`.
- **Modeling:** monotone constraints (hist), early stopping, feature importance
  (weight / gain / cover / totals), leaf-index and margin prediction.
- **Ecosystem:** libsvm & CSV loaders, native binary + JSON model I/O, k-fold
  cross-validation, multi-core histogram construction.

**In progress / planned**

- Objectives: learning-to-rank (`rank:*`, LambdaMART).
- `tree_method = approx`; categorical splits; interaction constraints; `dart` and
  `gblinear` boosters; TreeSHAP contributions.
- XGBoost-format (UBJSON/JSON) model import/export for drop-in compatibility.
- GPU histogram backend (the `HistogramBackend` trait is the seam for it).
- Python (PyO3), CLI, and C-ABI wrappers.

## Performance

The histogram method is the fast path. On a synthetic 100k × 30 regression set,
training 50 depth-6 trees:

| Method  | Time   |
|---------|--------|
| `hist`  | ~1.8 s |
| `exact` | ~20 s  |

Run the benchmarks yourself:

```sh
cargo bench -p sequoia-boost
```

## Testing & parity

```sh
cargo test -p sequoia-boost         # unit + integration tests
cargo clippy --all-targets          # lints
```

Numerical parity against upstream XGBoost is checked by a fixture harness:
`scripts/gen_fixtures.py` trains real `xgboost` across objectives and exports
predictions to `fixtures/`; the ignored integration test `tests/parity.rs`
asserts `sequoia-boost` matches within tolerance. See `scripts/README.md`.

## License

Apache-2.0.
