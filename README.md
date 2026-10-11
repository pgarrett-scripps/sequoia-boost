# sequoia-boost

[![crates.io](https://img.shields.io/crates/v/sequoia-boost.svg)](https://crates.io/crates/sequoia-boost)
[![docs.rs](https://img.shields.io/docsrs/sequoia-boost)](https://docs.rs/sequoia-boost)
[![CI](https://github.com/pgarrett-scripps/sequoia-boost/actions/workflows/ci.yml/badge.svg)](https://github.com/pgarrett-scripps/sequoia-boost/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21968435.svg)](https://doi.org/10.5281/zenodo.21968435)
[![license](https://img.shields.io/crates/l/sequoia-boost.svg)](LICENSE)

A faithful, fast, pure-Rust reimplementation of [XGBoost](https://github.com/dmlc/xgboost)
gradient boosting with no C/C++ dependency and no FFI.

`sequoia-boost` re-implements XGBoost's algorithms from scratch in idiomatic
Rust. It includes the regularized second-order boosting objective, exact,
histogram, and approximate tree construction, the full objective and metric
catalog, monotone and interaction constraints, categorical splits, DART and
gblinear boosters, TreeSHAP, and numeric-tree XGBoost-format model interop with
multi-core (`rayon`) acceleration.

Objective, metric, and parameter names mirror XGBoost, so configurations
transfer directly.

> **Built with AI.** The implementation was generated with **Claude** (Anthropic's
> AI coding assistant) under Patrick Garrett's direction and review. It is **AI-generated
> code**: it is covered by unit, property, and doc tests plus CI-checked XGBoost
> model-quality parity, but it may still contain bugs, subtle numerical errors, or
> wrong edge-case behavior. **Review and validate it for your own use case. It is
> provided as-is, without warranty** (see [LICENSE](LICENSE)). Issue reports and
> fixes are welcome.

> Using AI coding agents? See [`AGENTS.md`](AGENTS.md) for a task-oriented guide.

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

## Examples

Runnable, self-contained examples live in
[`crates/sequoia-boost/examples/`](crates/sequoia-boost/examples). Run any with
`cargo run --release --example <name>`:

| Example | Shows |
|---|---|
| `binary_classification` | `binary:logistic`, watched eval set, early stopping, AUC |
| `multiclass` | `multi:softprob`, per-class probabilities, `predict_class` |
| `ranking` | LambdaMART `rank:ndcg` over query groups |
| `shap` | `predict_contribs` and `predict_interactions` (TreeSHAP) |
| `model_io` | native binary / JSON and XGBoost-format model save & load |
| `custom_objective` | custom loss and custom eval-metric hooks |
| `constraints` | monotone + interaction constraints and categorical features |
| `train_regression` | end-to-end regression with feature importance |

## Feature status

**Implemented & tested**

- **Boosters:** `gbtree`, **`dart`** (tree dropout), and **`gblinear`** (linear
  model via coordinate descent).
- **Trees:** `tree_method = exact | hist | approx` (approx uses hessian-weighted
  per-round binning), `grow_policy = depthwise | lossguide`, histogram binning
  with the parent−child subtraction trick, sparsity-aware missing-value handling,
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
- **Constraints:** monotone constraints and **interaction constraints**,
  supported in **both** the `hist` and `exact` builders.
- **Modeling:** **native categorical splits** (hist and exact), per-instance
  `base_margin` (warm-start), **TreeSHAP** contributions (`predict_contribs`) and
  **interaction values** (`predict_interactions`), early stopping, feature
  importance (weight / gain / cover / totals), leaf-index and margin prediction.
- **Ecosystem:** libsvm & CSV loaders, native binary + JSON model I/O,
  **XGBoost-format JSON model import/export** for numeric `gbtree` ensembles,
  k-fold cross-validation,
  multi-core histogram construction, and runtime-detected SIMD kernels:
  **AArch64 NEON** for objective and metric kernels, prediction transforms,
  multiclass operations and histogram split evaluation, and **x86-64 AVX2/FMA**
  for objective gradients, prediction transforms, and histogram split
  evaluation.

**In progress / planned**

- UBJSON (binary) XGBoost model format.
- GPU histogram backend (the `HistogramBackend` trait is the seam for it).
- Distributed / external-memory training.
- Python (PyO3), CLI, and C-ABI wrappers.

## Performance

AArch64 builds use runtime-detected NEON kernels for objective gradients,
probability transforms, metric reductions, and dense numeric split evaluation.
x86-64 builds use AVX2+FMA for dense split evaluation, exponential/sigmoid
transforms, logistic and short-softmax gradients, and SSE2 for quantile bin
search. Scalar fallbacks cover other CPUs, short inputs, and values outside the
approximation ranges. Histogram training parallelizes data preparation and
independent nodes, scales histogram tasks to node size, and reuses training-row
partitions when
that reduces prediction work. Leaves at `max_depth` skip histograms and split
searches.

### Compared with XGBoost

Measured on **Apple M3 Max** against [XGBoost 3.4.1](https://pypi.org/project/xgboost/3.4.1/),
the latest stable PyPI release checked on **2026-09-14 UTC**. Both engines use the
same dense `f32` data and CPU `hist` parameters: 100 boosting rounds, depth 6,
256 bins, `eta=0.1`, and `lambda=1`. Times include fresh training-matrix
preparation and training, and report the median of six fits after warmup.

| Workload | Threads | sequoia-boost | XGBoost 3.4.1 |
|---|---:|---:|---:|
| Regression, 100k × 30 | 1 | 0.458 s | 1.054 s |
| Regression, 100k × 30 | 4 | 0.201 s | 0.366 s |
| Regression, 100k × 30 | 16 | 0.264 s | 0.362 s |
| Regression, 50k × 128 | 1 | 1.153 s | 3.200 s |
| Regression, 50k × 128 | 4 | 0.452 s | 0.994 s |
| Regression, 50k × 128 | 16 | 0.428 s | 0.669 s |
| Binary, 100k × 30 | 1 | 0.459 s | 1.044 s |
| Binary, 100k × 30 | 4 | 0.197 s | 0.366 s |
| Binary, 100k × 30 | 16 | 0.256 s | 0.361 s |
| 4-class, 50k × 30 | 1 | 1.094 s | 2.506 s |
| 4-class, 50k × 30 | 4 | 0.523 s | 0.981 s |
| 4-class, 50k × 30 | 16 | 0.746 s | 1.219 s |

Sequoia has lower median fit time in all 12 configurations in this run.
Single-thread speedups are 2.28× to 2.77×, four-thread speedups are 1.82× to
2.20×, and sixteen-thread speedups are 1.37× to 1.63×. Held-out RMSE/log-loss
scores are identical to the previous run on every workload.

See the [full comparison](docs/performance.md#xgboost-comparison) for held-out
quality, sample variability, workload definitions, and reproduction commands.

### Kernel benchmarks

See [Performance](docs/performance.md) for kernel measurements, numerical
behavior, and reproduction commands.

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
predictions to `fixtures/`. The ignored integration test `tests/parity.rs`
asserts `sequoia-boost` matches within tolerance. See `scripts/README.md`.

## Development provenance

Generated with **Claude** (Anthropic's AI coding assistant) under **Patrick
Garrett's** direction and review. The AI system is not an author. See
[`NOTICE`](NOTICE). Because the code is AI-generated, treat it with appropriate
scrutiny. It is tested and parity-checked but not warranted.

## Citation and archiving

Citation metadata is provided in [`CITATION.cff`](CITATION.cff), with matching
deposit metadata in [`.zenodo.json`](.zenodo.json). Cite the project using the
stable [Zenodo concept DOI](https://doi.org/10.5281/zenodo.21968435). The
immutable v0.2.0 archive has release DOI
[10.5281/zenodo.21968436](https://doi.org/10.5281/zenodo.21968436).

## Acknowledgments

sequoia-boost is an independent, from-scratch **reimplementation of
[XGBoost](https://github.com/dmlc/xgboost)** (Copyright the XGBoost Contributors,
Apache-2.0) in Rust. It reimplements XGBoost's algorithms from their public
descriptions and papers and contains no XGBoost source code. "XGBoost" is used
descriptively to indicate algorithmic lineage and result compatibility. This
project is not affiliated with or endorsed by the XGBoost project. See
[`NOTICE`](NOTICE).

## License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
