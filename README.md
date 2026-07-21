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

- **Boosters:** `gbtree`, **`dart`** (tree dropout), and **`gblinear`** (linear
  model via coordinate descent).
- **Trees:** `tree_method = exact | hist | approx` (approx uses hessian-weighted
  per-round binning); `grow_policy = depthwise | lossguide`; histogram binning
  with the parent−child subtraction trick; sparsity-aware missing-value handling;
  row/column subsampling (`bytree`/`bylevel`/`bynode`).
- **Regularization:** `lambda`, `alpha`, `gamma`, `min_child_weight`,
  `max_delta_step`, `max_depth`, `max_leaves`, `max_bin`.
- **Objectives:** `reg:squarederror`, `reg:pseudohubererror`, `binary:logistic`,
  `multi:softmax`, `multi:softprob`, `count:poisson`, `reg:gamma`, `reg:tweedie`,
  learning-to-rank (`rank:pairwise`, `rank:ndcg`, `rank:map`, LambdaMART), and a
  user **custom-objective hook**.
- **Metrics:** `rmse`, `mae`, `logloss`, `error`, `auc`, `aucpr`, `mlogloss`,
  `merror`, `poisson/gamma/tweedie-nloglik`, `ndcg`, `map` (with `@k`), and a
  **custom-metric hook**.
- **Constraints:** monotone constraints and **interaction constraints** —
  supported in **both** the `hist` and `exact` builders.
- **Modeling:** **native categorical splits** (hist and exact), per-instance
  `base_margin` (warm-start), **TreeSHAP** contributions (`predict_contribs`) and
  **interaction values** (`predict_interactions`), early stopping, feature
  importance (weight / gain / cover / totals), leaf-index and margin prediction.
- **Ecosystem:** libsvm & CSV loaders, native binary + JSON model I/O,
  **XGBoost-format JSON model import/export**, k-fold cross-validation,
  multi-core histogram construction.

**In progress / planned**

- UBJSON (binary) XGBoost model format.
- GPU histogram backend (the `HistogramBackend` trait is the seam for it).
- Distributed / external-memory training.
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

## Acknowledgments

sequoia-boost is an independent, from-scratch **reimplementation of
[XGBoost](https://github.com/dmlc/xgboost)** (Copyright the XGBoost Contributors,
Apache-2.0) in Rust. It reimplements XGBoost's algorithms from their public
descriptions and papers and contains no XGBoost source code. "XGBoost" is used
descriptively to indicate algorithmic lineage and result compatibility; this
project is not affiliated with or endorsed by the XGBoost project. See
[`NOTICE`](NOTICE).

## License

Licensed under the **Apache License, Version 2.0** — see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
