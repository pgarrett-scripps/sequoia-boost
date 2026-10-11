# bench/verify

Checks that a performance branch trains the same models as a baseline
checkout. `run.sh <baseline> <branch> <out>` builds this harness against each
checkout's `crates/sequoia-boost`, trains 72 configurations (regression,
binary and multiclass objectives; with and without missing values; two seeds;
depth 3, 6 and 9; hist and exact) at 1 and 4 threads, and compares:

- leaf ids, which must equal the baseline's;
- predictions and SHAP contributions, reported as max absolute and ulp
  difference, with predictions required to be within 1e-6 relative;
- the branch's 1-thread and 4-thread outputs, which must be bit-identical.

It also prints rough training and prediction timings. These come from shared
CI runners and are not benchmarks; use `cargo bench` for measurements.
`.github/workflows/arm64-verify.yml` runs it on x86-64, Linux ARM64 and
macOS ARM64.
