"""Lag feature matrix builder (row-major flatten)."""

from __future__ import annotations

import numpy as np


def lag_feature_matrix(series: np.ndarray, n_lags: int) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n <= n_lags:
        return np.empty(0, dtype=float)
    n_samples = n - n_lags
    out = np.zeros(n_samples * n_lags, dtype=float)
    for i in range(n_samples):
        for lag in range(n_lags):
            out[i * n_lags + lag] = s[i + lag]
    return out
