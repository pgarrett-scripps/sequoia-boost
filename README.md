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
  learning-to-rank (`rank:pairwise`, `rank:ndcg`, `rank:map`, LambdaMART), and a
  user **custom-objective hook**.
- **Metrics:** `rmse`, `mae`, `logloss`, `error`, `auc`, `mlogloss`, `merror`,
  `poisson/gamma/tweedie-nloglik`, `ndcg`, `map` (with `@k`).
- **Boosters:** `gbtree` and **`dart`** (tree dropout).
- **Modeling:** monotone constraints, **native categorical splits** (hist),
  **TreeSHAP** exact contributions (`predict_contribs`), early stopping, feature
  importance (weight / gain / cover / totals), leaf-index and margin prediction.
- **Ecosystem:** libsvm & CSV loaders, native binary + JSON model I/O,
  **XGBoost-format JSON model import/export**, k-fold cross-validation,
  multi-core histogram construction.

**In progress / planned**

- `tree_method = approx`; interaction constraints; `gblinear` booster.
- UBJSON (binary) XGBoost model format; categorical splits in the exact builder.
- GPU histogram backend (the `HistogramBackend` trait is the seam for it).
- Python (PyO3), CLI, and C-ABI wrappers.

## Performance

### Head-to-head vs XGBoost

A fair comparison against real XGBoost 3.3 — the **same** little-endian `f32`
bytes fed to both engines, matching `hist` parameters, end-to-end fit timing
(binning + training), best-of-3, **single-threaded**. Dataset: 100k rows × 30
features, 100 rounds, depth 6, `max_bin=256`, `eta=0.1`, `lambda=1`.

| Engine | fit time | train RMSE |
|--------|---------:|-----------:|
| sequoia-boost | **~1.77 s** | 0.05650 |
| XGBoost 3.3   | ~1.30 s     | 0.05685 |

sequoia-boost is roughly **1.35× the wall-clock of XGBoost single-threaded**,
with matching accuracy — a solid result for a pure-Rust engine with **no explicit
SIMD** against XGBoost's heavily hand-optimized C++. Profiling drove a ~26%
speedup in split evaluation (see `examples/profile.rs`); the remaining gap is
largely XGBoost's SIMD and cache-tuned kernels.

Caveats worth stating plainly:

- **Timings are machine-load sensitive** — these are quiet-machine numbers.
- **Multi-core is not benchmarked** here: the sandbox couldn't deliver real
  parallel throughput (both engines regressed identically at high thread counts),
  so honest scaling numbers need bare-metal, isolated cores.
- One dataset shape (moderate width). XGBoost tends to pull further ahead on
  wider data and deeper trees where its SIMD kernels dominate.

Reproduce:

```sh
python scripts/bench_xgb.py <bench_dir> 100000 30 100 1   # writes data + times XGBoost
BENCH_DIR=<bench_dir> RAYON_NUM_THREADS=1 \
  cargo run --release -p sequoia-boost --example bench_compare
# phase-level profiler:
BENCH_DIR=<bench_dir> RAYON_NUM_THREADS=1 \
  cargo run --release -p sequoia-boost --example profile
```

### Internal micro-benchmarks

Criterion benchmarks for the hot paths (histogram build, exact vs hist training):

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
