#!/usr/bin/env python3
"""
Production Time Series Forecasting for Power Plant Emissions
Trains LSTM, XGBoost, SARIMA, and Ensemble models on historical CO2 data
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = Path('../../egrid_all_plants_1996-2023.parquet')
TRAIN_END_YEAR = 2020
TEST_START_YEAR = 2021
FORECAST_HORIZON = 3  # Years to forecast beyond data
RANDOM_STATE = 42

def load_and_prepare_data():
    """Load and aggregate data to yearly time series"""
    print("Loading data...")
    plants = pd.read_parquet(DATA_PATH)
    
    # Aggregate by year
    yearly = plants.groupby('data_year').agg({
        'Plant annual net generation (MWh)': lambda x: pd.to_numeric(x, errors='coerce').sum(),
        'Plant annual CO2 emissions (tons)': lambda x: pd.to_numeric(x, errors='coerce').sum(),
    }).reset_index()
    
    yearly.columns = ['year', 'generation_mwh', 'co2_tons']
    yearly['carbon_intensity'] = yearly['co2_tons'] / yearly['generation_mwh']
    yearly = yearly.sort_values('year').reset_index(drop=True)
    
    print(f"Loaded {len(yearly)} years of data ({yearly['year'].min()}-{yearly['year'].max()})")
    return yearly

def create_sequences(data, lookback=5):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_lstm_model(lookback, n_features=1):
    """Build LSTM model"""
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', return_sequences=True, 
                   input_shape=(lookback, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm(train_data, test_data, lookback=5):
    """Train LSTM model"""
    print("\n[1/4] Training LSTM...")
    
    # Scale data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[['co2_tons']])
    test_scaled = scaler.transform(test_data[['co2_tons']])
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled.flatten(), lookback)
    X_test, y_test = create_sequences(test_scaled.flatten(), lookback)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build and train
    model = build_lstm_model(lookback)
    model.fit(X_train, y_train, epochs=100, batch_size=4, 
             validation_split=0.2, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    
    # Predict
    y_pred = model.predict(X_test, verbose=0)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    
    print(f"  MAE: {mae/1e9:.3f}B tons | RMSE: {rmse/1e9:.3f}B tons")
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': y_pred_unscaled.flatten(),
        'actuals': y_test_unscaled.flatten(),
        'mae': mae,
        'rmse': rmse,
        'lookback': lookback
    }

def create_lag_features(df, target_col, lags=[1, 2, 3, 5]):
    """Create lagged features for XGBoost"""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df['rolling_mean_3'] = df[target_col].rolling(window=3).mean()
    df['rolling_std_3'] = df[target_col].rolling(window=3).std()
    return df.dropna()

def train_xgboost(train_data, test_data):
    """Train XGBoost model"""
    print("\n[2/4] Training XGBoost...")
    
    # Create features
    train_feat = create_lag_features(train_data.copy(), 'co2_tons')
    test_feat = create_lag_features(test_data.copy(), 'co2_tons')
    
    feature_cols = [c for c in train_feat.columns if c not in ['year', 'co2_tons', 'generation_mwh', 'carbon_intensity']]
    
    X_train = train_feat[feature_cols]
    y_train = train_feat['co2_tons']
    X_test = test_feat[feature_cols]
    y_test = test_feat['co2_tons']
    
    # Train
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train, verbose=False)
    
    # Predict
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"  MAE: {mae/1e9:.3f}B tons | RMSE: {rmse/1e9:.3f}B tons")
    
    return {
        'model': model,
        'predictions': y_pred,
        'actuals': y_test.values,
        'mae': mae,
        'rmse': rmse,
        'features': feature_cols
    }

def train_sarima(train_data, test_data):
    """Train SARIMA model"""
    print("\n[3/4] Training SARIMA...")
    
    train_ts = train_data.set_index('year')['co2_tons']
    test_ts = test_data.set_index('year')['co2_tons']
    
    # Train SARIMA(1,1,1)
    model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    fitted = model.fit(disp=False, maxiter=200)
    
    # Forecast
    forecast = fitted.forecast(steps=len(test_ts))
    
    mae = mean_absolute_error(test_ts, forecast)
    rmse = np.sqrt(mean_squared_error(test_ts, forecast))
    
    print(f"  MAE: {mae/1e9:.3f}B tons | RMSE: {rmse/1e9:.3f}B tons")
    
    return {
        'model': fitted,
        'predictions': forecast.values,
        'actuals': test_ts.values,
        'mae': mae,
        'rmse': rmse
    }

def create_ensemble(models, weights=None):
    """Create weighted ensemble of predictions"""
    print("\n[4/4] Creating Ensemble...")
    
    if weights is None:
        # Weight by inverse MAE
        maes = [m['mae'] for m in models]
        inv_maes = [1/mae for mae in maes]
        weights = [inv/sum(inv_maes) for inv in inv_maes]
    
    # Align predictions (they may have different lengths due to lookback)
    min_len = min(len(m['predictions']) for m in models)
    
    ensemble_pred = np.zeros(min_len)
    for model, weight in zip(models, weights):
        ensemble_pred += weight * model['predictions'][-min_len:]
    
    actuals = models[0]['actuals'][-min_len:]
    
    mae = mean_absolute_error(actuals, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(actuals, ensemble_pred))
    
    print(f"  Weights: {[f'{w:.3f}' for w in weights]}")
    print(f"  MAE: {mae/1e9:.3f}B tons | RMSE: {rmse/1e9:.3f}B tons")
    
    return {
        'predictions': ensemble_pred,
        'actuals': actuals,
        'mae': mae,
        'rmse': rmse,
        'weights': weights
    }

def visualize_results(data, models, ensemble, test_start_year):
    """Create comparison visualization"""
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Get test years
    test_data = data[data['year'] >= test_start_year]
    test_years = test_data['year'].values
    
    # Plot 1: All models comparison
    ax1 = axes[0, 0]
    ax1.plot(data['year'], data['co2_tons']/1e9, 'o-', 
            label='Actual', linewidth=2, markersize=6)
    
    # Align years for each model
    for name, model in [('LSTM', models[0]), ('XGBoost', models[1]), ('SARIMA', models[2])]:
        pred_years = test_years[-len(model['predictions']):]
        ax1.plot(pred_years, model['predictions']/1e9, 
                's--', label=name, linewidth=2, markersize=5, alpha=0.7)
    
    # Ensemble
    ens_years = test_years[-len(ensemble['predictions']):]
    ax1.plot(ens_years, ensemble['predictions']/1e9, 
            'D-', label='Ensemble', linewidth=3, markersize=6, color='red')
    
    ax1.axvline(test_start_year-0.5, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CO₂ Emissions (Billion Tons)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model performance
    ax2 = axes[0, 1]
    model_names = ['LSTM', 'XGBoost', 'SARIMA', 'Ensemble']
    maes = [m['mae']/1e9 for m in models] + [ensemble['mae']/1e9]
    
    bars = ax2.bar(model_names, maes, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('MAE (Billion Tons)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.3f}B',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Residuals
    ax3 = axes[1, 0]
    residuals = ensemble['actuals'] - ensemble['predictions']
    ax3.plot(ens_years, residuals/1e9, 'o-', linewidth=2, markersize=8)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Residual (Billion Tons)', fontsize=12, fontweight='bold')
    ax3.set_title('Ensemble Residuals', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals/1e9, bins=10, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residual (Billion Tons)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('01_time_series_results.png', dpi=300, bbox_inches='tight')
    print("  Saved: 01_time_series_results.png")

def main():
    """Main execution"""
    print("=" * 80)
    print("TIME SERIES FORECASTING - PRODUCTION RUN")
    print("=" * 80)
    
    # Load data
    data = load_and_prepare_data()
    
    # Split train/test
    train = data[data['year'] <= TRAIN_END_YEAR]
    test = data[data['year'] >= TEST_START_YEAR]
    
    print(f"\nTrain: {len(train)} years | Test: {len(test)} years")
    
    # Train models
    lstm_results = train_lstm(train, test)
    xgb_results = train_xgboost(train, test)
    sarima_results = train_sarima(train, test)
    
    # Ensemble
    models = [lstm_results, xgb_results, sarima_results]
    ensemble_results = create_ensemble(models)
    
    # Visualize
    visualize_results(data, models, ensemble_results, TEST_START_YEAR)
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'MAE (B tons)':<15} {'RMSE (B tons)':<15} {'Improvement vs Best Single'}")
    print("-" * 80)
    
    best_single_mae = min(m['mae'] for m in models)
    
    for name, model in [('LSTM', lstm_results), ('XGBoost', xgb_results), 
                        ('SARIMA', sarima_results), ('Ensemble', ensemble_results)]:
        improvement = (best_single_mae - model['mae']) / best_single_mae * 100
        rmse = model['rmse'] if 'rmse' in model else np.sqrt(mean_squared_error(model['actuals'], model['predictions']))
        print(f"{name:<15} {model['mae']/1e9:<15.3f} {rmse/1e9:<15.3f} {improvement:+.1f}%")
    
    print("=" * 80)
    print("✓ Complete!")
    
    return {
        'lstm': lstm_results,
        'xgboost': xgb_results,
        'sarima': sarima_results,
        'ensemble': ensemble_results
    }

if __name__ == '__main__':
    results = main()

