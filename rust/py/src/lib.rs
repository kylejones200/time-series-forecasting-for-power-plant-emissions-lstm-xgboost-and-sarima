use time_series_forecasting_for_power_plant_emissions_lstm_xgboost_and_sarima_core::lag_feature_matrix;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn lag_feature_matrix_py<'py>(py: Python<'py>, series: PyReadonlyArray1<f64>, n_lags: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(lag_feature_matrix(series.as_slice()?, n_lags).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (series, n_lags, iterations=500))]
fn bench_kernel_py(series: PyReadonlyArray1<f64>, n_lags: usize, iterations: usize) -> PyResult<f64> {
    let series_buf = series.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = lag_feature_matrix(&series_buf, n_lags);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn time_series_forecasting_for_power_plant_emissions_lstm_xgboost_and_sarima_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lag_feature_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
