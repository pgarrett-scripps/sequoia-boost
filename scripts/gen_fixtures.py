#!/usr/bin/env python3
"""Generate XGBoost parity fixtures for sequoia-boost.

Trains real XGBoost on standardized synthetic datasets across several
objectives and writes each dataset, the exact training parameters, and
XGBoost's predictions to ``fixtures/*.json``. The Rust integration test
``tests/parity.rs`` (run with ``cargo test -- --ignored``) loads these and
asserts sequoia-boost matches within tolerance.

Usage:
    pip install xgboost numpy
    python scripts/gen_fixtures.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import xgboost as xgb

FIX_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# Parameters shared by every fixture. `tree_method=hist` and a fixed max_bin
# give sequoia-boost's histogram method the best chance of exact agreement.
COMMON = dict(
    tree_method="hist",
    max_depth=4,
    eta=0.1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    min_child_weight=1.0,
    max_bin=256,
    subsample=1.0,
    colsample_bytree=1.0,
    base_score=0.5,
)
NUM_ROUND = 60


def _rng():
    return np.random.default_rng(20260720)


def _write(name, x, y, params, num_class, preds, tol):
    x = np.ascontiguousarray(x, dtype=np.float32)
    fixture = {
        "name": name,
        "objective": params["objective"],
        "num_class": num_class,
        "num_round": NUM_ROUND,
        "n_rows": int(x.shape[0]),
        "n_cols": int(x.shape[1]),
        "x": x.reshape(-1).tolist(),
        "y": np.asarray(y, dtype=np.float32).tolist(),
        "params": {
            "max_depth": params["max_depth"],
            "eta": params["eta"],
            "lambda": params["reg_lambda"],
            "alpha": params["reg_alpha"],
            "gamma": params["gamma"],
            "min_child_weight": params["min_child_weight"],
            "max_bin": params["max_bin"],
            "base_score": params["base_score"],
        },
        "xgb_pred": np.asarray(preds, dtype=np.float32).reshape(-1).tolist(),
        "tolerance": tol,
    }
    os.makedirs(FIX_DIR, exist_ok=True)
    path = os.path.join(FIX_DIR, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(fixture, fh)
    print(f"wrote {path}  ({fixture['n_rows']}x{fixture['n_cols']})")


def _train(x, y, params, num_class):
    dtrain = xgb.DMatrix(x, label=y)
    p = dict(COMMON, **params)
    if num_class:
        p["num_class"] = num_class
    booster = xgb.train(p, dtrain, num_boost_round=NUM_ROUND)
    return booster.predict(dtrain)


def regression():
    rng = _rng()
    x = rng.random((2000, 8), dtype=np.float32)
    y = 2 * x[:, 0] - 3 * x[:, 1] ** 2 + 0.5 * x[:, 2] + 0.1 * rng.standard_normal(2000)
    preds = _train(x, y, dict(objective="reg:squarederror"), 0)
    _write("regression", x, y, dict(COMMON, objective="reg:squarederror"), 0, preds, 1e-3)


def binary():
    rng = _rng()
    x = rng.random((2000, 8), dtype=np.float32)
    logit = 3 * x[:, 0] - 2 * x[:, 1]
    y = (1 / (1 + np.exp(-logit)) > rng.random(2000)).astype(np.float32)
    preds = _train(x, y, dict(objective="binary:logistic"), 0)
    _write("binary", x, y, dict(COMMON, objective="binary:logistic"), 0, preds, 1e-3)


def multiclass():
    rng = _rng()
    x = rng.random((2000, 8), dtype=np.float32)
    y = (x[:, 0] * 3).astype(int).clip(0, 2).astype(np.float32)
    preds = _train(x, y, dict(objective="multi:softprob"), 3)
    _write("multiclass", x, y, dict(COMMON, objective="multi:softprob"), 3, preds, 2e-3)


if __name__ == "__main__":
    regression()
    binary()
    multiclass()
    print("done. run: cargo test -p sequoia-boost --test parity -- --ignored")
