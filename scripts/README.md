# Parity fixtures

`gen_fixtures.py` trains **real XGBoost** on standardized synthetic datasets and
writes each dataset + parameters + XGBoost predictions to `../fixtures/*.json`.
The Rust test `crates/sequoia-boost/tests/parity.rs` loads these and asserts
`sequoia-boost` matches within tolerance.

```sh
pip install xgboost numpy
python scripts/gen_fixtures.py
cargo test -p sequoia-boost --test parity -- --ignored
```

Fixtures are intentionally not checked in (regenerate locally). The datasets use
`tree_method=hist` with a fixed `max_bin` so the two histogram implementations
have the best chance of close agreement; tolerances are per-fixture.
