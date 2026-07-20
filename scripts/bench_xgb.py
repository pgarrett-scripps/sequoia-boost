#!/usr/bin/env python3
"""Write a shared benchmark dataset and time real XGBoost (tree_method=hist).

Both engines read the exact same little-endian f32 bytes. Timing is end-to-end
fit-from-arrays (DMatrix construction + train), best-of-3, at the thread counts
given on the command line.

Usage: python scripts/bench_xgb.py <bench_dir> <n_rows> <n_cols> <num_round> [threads...]
"""
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb

bench_dir = sys.argv[1]
n_rows = int(sys.argv[2])
n_cols = int(sys.argv[3])
num_round = int(sys.argv[4])
thread_list = [int(t) for t in sys.argv[5:]] or [1, os.cpu_count()]

MAX_DEPTH, ETA, LAMBDA, MAX_BIN, BASE = 6, 0.1, 1.0, 256, 0.5

os.makedirs(bench_dir, exist_ok=True)
rng = np.random.default_rng(1234)
X = rng.random((n_rows, n_cols), dtype=np.float32)
# Learnable signal in the first 5 features + noise.
y = (
    2 * X[:, 0] - 3 * X[:, 1] ** 2 + 0.5 * X[:, 2] + X[:, 3] * X[:, 4]
    + 0.05 * rng.standard_normal(n_rows).astype(np.float32)
).astype(np.float32)

X.reshape(-1).tofile(os.path.join(bench_dir, "X.bin"))
y.tofile(os.path.join(bench_dir, "y.bin"))
with open(os.path.join(bench_dir, "meta.json"), "w") as f:
    json.dump(
        {"n_rows": n_rows, "n_cols": n_cols, "num_round": num_round,
         "max_depth": MAX_DEPTH, "eta": ETA, "lambda": LAMBDA,
         "max_bin": MAX_BIN, "base_score": BASE},
        f,
    )

print(f"dataset: {n_rows} x {n_cols}, {num_round} rounds, depth {MAX_DEPTH}")

for nthread in thread_list:
    params = dict(
        objective="reg:squarederror", tree_method="hist", max_depth=MAX_DEPTH,
        eta=ETA, reg_lambda=LAMBDA, reg_alpha=0.0, gamma=0.0, min_child_weight=1.0,
        max_bin=MAX_BIN, subsample=1.0, colsample_bytree=1.0, base_score=BASE,
        nthread=nthread,
    )
    best = float("inf")
    rmse = 0.0
    for _ in range(3):
        t = time.perf_counter()
        dtrain = xgb.DMatrix(X, label=y, nthread=nthread)
        booster = xgb.train(params, dtrain, num_boost_round=num_round)
        secs = time.perf_counter() - t
        best = min(best, secs)
        pred = booster.predict(dtrain)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    print(json.dumps(dict(impl="xgboost", threads=nthread, fit_s=round(best, 3), rmse=round(rmse, 5))))
