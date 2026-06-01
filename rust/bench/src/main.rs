use time_series_forecasting_for_power_plant_emissions_lstm_xgboost_and_sarima_core::lag_feature_matrix;

fn main() {
    let s: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.01).sin() + 50.0).collect();
    for _ in 0..2000 {
        let _ = lag_feature_matrix(&s, 12);
    }
}
