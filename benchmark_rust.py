#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import lag_feature_matrix  # noqa: E402

def main() -> None:
    s = np.ascontiguousarray(np.sin(np.arange(5000) * 0.01) + 50.0)
    n_lags = 12
    t0 = time.perf_counter()
    for _ in range(200):
        lag_feature_matrix(s, n_lags)
    py_s = time.perf_counter() - t0
    try:
        import time_series_forecasting_for_power_plant_emissions_lstm_xgboost_and_sarima_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(s, n_lags, 2000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        lag_feature_matrix(s, n_lags),
        np.asarray(rs.lag_feature_matrix_py(s, n_lags)),
        rtol=1e-10,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
