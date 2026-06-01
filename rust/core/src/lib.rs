//! Lag feature matrix builder (row-major flatten).

pub fn lag_feature_matrix(series: &[f64], n_lags: usize) -> Vec<f64> {
    let n = series.len();
    if n <= n_lags {
        return vec![];
    }
    let n_samples = n - n_lags;
    let mut out = vec![0.0; n_samples * n_lags];
    for i in 0..n_samples {
        for l in 0..n_lags {
            out[i * n_lags + l] = series[i + l];
        }
    }
    out
}
