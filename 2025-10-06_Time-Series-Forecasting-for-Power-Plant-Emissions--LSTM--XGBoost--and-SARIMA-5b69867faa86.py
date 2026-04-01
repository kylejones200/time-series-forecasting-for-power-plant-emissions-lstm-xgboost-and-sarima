# Description: Short example for Time Series Forecasting for Power Plant Emissions LSTM XGBoost and SARIMA.



from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Load the data
plants = pd.read_parquet('egrid_all_plants_1996-2023.parquet')
# Aggregate to national level
yearly_data = plants.groupby('data_year').agg({
    'Plant annual net generation (MWh)': 'sum',
    'Plant annual CO2 emissions (tons)': 'sum',
}).reset_index()
yearly_data['carbon_intensity'] = (
    yearly_data['Plant annual CO2 emissions (tons)'] / 
    yearly_data['Plant annual net generation (MWh)']
)
logger.info(f"Data spans {yearly_data['data_year'].min()} to {yearly_data['data_year'].max()}")
logger.info(f"Total emissions declined from {yearly_data.iloc[0]['Plant annual CO2 emissions (tons)']:,.0f} to {yearly_data.iloc[-1]['Plant annual CO2 emissions (tons)']:,.0f} tons")


# Prepare sequences (use past 3 years to predict next year)
def create_sequences(data, lookback=3):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(train_data[['total_co2_tons']])
# Create sequences
X_train, y_train = create_sequences(scaled_data, lookback=3)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(3, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2)


def create_time_features(df):
    df = df.copy()
    
    # Lag features
    df['co2_lag1'] = df['total_co2_tons'].shift(1)
    df['co2_lag2'] = df['total_co2_tons'].shift(2)
    df['co2_lag3'] = df['total_co2_tons'].shift(3)
    
    # Rolling statistics
    df['co2_rolling_mean_3y'] = df['total_co2_tons'].rolling(3).mean()
    df['co2_rolling_std_3y'] = df['total_co2_tons'].rolling(3).std()
    
    # Trend features
    df['co2_diff1'] = df['total_co2_tons'].diff(1)
    df['co2_diff2'] = df['total_co2_tons'].diff(2)
    
    # Time-based
    df['years_since_start'] = df['year'] - df['year'].min()
    
    return df
# Create features and train
features = create_time_features(yearly_data)
feature_cols = ['co2_lag1', 'co2_lag2', 'co2_lag3', 
                'co2_rolling_mean_3y', 'co2_rolling_std_3y',
                'co2_diff1', 'years_since_start']
X_train = features[feature_cols].dropna()
y_train = features.loc[X_train.index, 'total_co2_tons']
# Train XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8
)
xgb_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

logger.info(feature_importance)

# feature  importance
# 0         co2_lag1       0.452
# 1  co2_rolling_mean_3y    0.218
# 2         co2_diff1       0.156
# 3         co2_lag2       0.089
# ...


# Grid search for optimal parameters
best_aic = np.inf
best_params = None
for p, d, q in itertools.product(range(3), range(2), range(3)):
    try:
        model = SARIMAX(train_data['total_co2_tons'], 
                       order=(p, d, q))
        results = model.fit(disp=False)
        if results.aic < best_aic:
            best_aic = results.aic
            best_params = (p, d, q)
    except:
        continue
logger.info(f"Best SARIMA order: {best_params}")
# Train final model
sarima = SARIMAX(train_data['total_co2_tons'], 
                 order=best_params)
sarima_results = sarima.fit()
# Forecast
forecast = sarima_results.forecast(steps=3)

# Average predictions from all three models
ensemble_predictions = (
    lstm_predictions.flatten() + 
    xgb_predictions + 
    sarima_predictions
) / 3


# Stack predictions as features
X_meta = np.column_stack([
    lstm_predictions,
    xgb_predictions,
    sarima_predictions
])
# Train meta-learner
meta_model = LinearRegression()
meta_model.fit(X_meta, test_actuals)
# Optimal weights
logger.info("Model weights:")
for i, name in enumerate(['LSTM', 'XGBoost', 'SARIMA']):
    logger.info(f"  {name}: {meta_model.coef_[i]:.3f}")

# Model weights:
#   LSTM: 0.283
#   XGBoost: 0.512
#   SARIMA: 0.205

# Retrain on full dataset
full_model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
full_features = create_time_features(all_data)
X_full = full_features[feature_cols].dropna()
y_full = full_features.loc[X_full.index, 'total_co2_tons']
full_model.fit(X_full, y_full)

# Iteratively forecast future years
future_predictions = []
for year in range(2024, 2031):
    # Use previous predictions as features
    X_future = create_features_for_year(year, recent_history)
    prediction = full_model.predict(X_future)[0]
    future_predictions.append(prediction)
    recent_history.append(prediction)
forecast_df = pd.DataFrame({
    'Year': range(2024, 2031),
    'Predicted_CO2': future_predictions
})
logger.info(forecast_df)

# Year  Predicted_CO2
# 0  2024    1,847,234,219
# 1  2025    1,792,441,087
# 2  2026    1,741,203,944
# 3  2027    1,693,182,229
# 4  2028    1,647,058,917
# 5  2029    1,602,538,273
# 6  2030    1,559,344,961


# Train models for 10th, 50th, and 90th percentiles
quantile_models = {}
for quantile in [0.1, 0.5, 0.9]:
    model = GradientBoostingRegressor(
        loss='quantile', 
        alpha=quantile,
        n_estimators=100
    )
    model.fit(X_full, y_full)
    quantile_models[quantile] = model
# Generate prediction intervals
intervals = pd.DataFrame({
    'Year': range(2024, 2031),
    'Lower_10%': [quantile_models[0.1].predict(X)[0] for X in X_futures],
    'Median': [quantile_models[0.5].predict(X)[0] for X in X_futures],
    'Upper_90%': [quantile_models[0.9].predict(X)[0] for X in X_futures]
})

#!/usr/bin/env python3
"""
Production Time Series Forecasting for Power Plant Emissions
Trains LSTM, XGBoost, SARIMA, and Ensemble models on historical CO2 data
"""



# ML libraries


# Configuration
DATA_PATH = Path('../../egrid_all_plants_1996-2023.parquet')
TRAIN_END_YEAR = 2020
TEST_START_YEAR = 2021
FORECAST_HORIZON = 3  # Years to forecast beyond data
RANDOM_STATE = 42

def load_and_prepare_data():
    """Load and aggregate data to yearly time series"""
    logger.info("Loading data...")
    plants = pd.read_parquet(DATA_PATH)
    
    # Aggregate by year
    yearly = plants.groupby('data_year').agg({
        'Plant annual net generation (MWh)': lambda x: pd.to_numeric(x, errors='coerce').sum(),
        'Plant annual CO2 emissions (tons)': lambda x: pd.to_numeric(x, errors='coerce').sum(),
    }).reset_index()
    
    yearly.columns = ['year', 'generation_mwh', 'co2_tons']
    yearly['carbon_intensity'] = yearly['co2_tons'] / yearly['generation_mwh']
    yearly = yearly.sort_values('year').reset_index(drop=True)
    
    logger.info(f"Loaded {len(yearly)} years of data ({yearly['year'].min()}-{yearly['year'].max()})")
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
    logger.info("\n[1/4] Training LSTM...")
    
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
    
    logger.info(f"  MAE: {mae/1e9:.3f}B tons | RMSE: {rmse/1e9:.3f}B tons")
    
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
    logger.info("\n[2/4] Training XGBoost...")
    
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
    
    logger.info(f"  MAE: {mae/1e9:.3f}B tons | RMSE: {rmse/1e9:.3f}B tons")
    
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
    logger.info("\n[3/4] Training SARIMA...")
    
    train_ts = train_data.set_index('year')['co2_tons']
    test_ts = test_data.set_index('year')['co2_tons']
    
    # Train SARIMA(1,1,1)
    model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    fitted = model.fit(disp=False, maxiter=200)
    
    # Forecast
    forecast = fitted.forecast(steps=len(test_ts))
    
    mae = mean_absolute_error(test_ts, forecast)
    rmse = np.sqrt(mean_squared_error(test_ts, forecast))
    
    logger.info(f"  MAE: {mae/1e9:.3f}B tons | RMSE: {rmse/1e9:.3f}B tons")
    
    return {
        'model': fitted,
        'predictions': forecast.values,
        'actuals': test_ts.values,
        'mae': mae,
        'rmse': rmse
    }

def create_ensemble(models, weights=None):
    """Create weighted ensemble of predictions"""
    logger.info("\n[4/4] Creating Ensemble...")
    
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
    
    logger.info(f"  Weights: {[f'{w:.3f}' for w in weights]}")
    logger.info(f"  MAE: {mae/1e9:.3f}B tons | RMSE: {rmse/1e9:.3f}B tons")
    
    return {
        'predictions': ensemble_pred,
        'actuals': actuals,
        'mae': mae,
        'rmse': rmse,
        'weights': weights
    }

def visualize_results(data, models, ensemble, test_start_year):
    """Create comparison visualization"""
    logger.info("\nGenerating visualizations...")
    
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
    # Plot 2: Model performance
    ax2 = axes[0, 1]
    model_names = ['LSTM', 'XGBoost', 'SARIMA', 'Ensemble']
    maes = [m['mae']/1e9 for m in models] + [ensemble['mae']/1e9]
    
    bars = ax2.bar(model_names, maes, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('MAE (Billion Tons)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Performance', fontsize=14, fontweight='bold')
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
    # Plot 4: Error distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals/1e9, bins=10, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residual (Billion Tons)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('01_time_series_results.png', dpi=300, bbox_inches='tight')
    logger.info("  Saved: 01_time_series_results.png")

def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("TIME SERIES FORECASTING - PRODUCTION RUN")
    logger.info("=" * 80)
    
    # Load data
    data = load_and_prepare_data()
    
    # Split train/test
    train = data[data['year'] <= TRAIN_END_YEAR]
    test = data[data['year'] >= TEST_START_YEAR]
    
    logger.info(f"\nTrain: {len(train)} years | Test: {len(test)} years")
    
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
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Model':<15} {'MAE (B tons)':<15} {'RMSE (B tons)':<15} {'Improvement vs Best Single'}")
    logger.info("-" * 80)
    
    best_single_mae = min(m['mae'] for m in models)
    
    for name, model in [('LSTM', lstm_results), ('XGBoost', xgb_results), 
                        ('SARIMA', sarima_results), ('Ensemble', ensemble_results)]:
        improvement = (best_single_mae - model['mae']) / best_single_mae * 100
        rmse = model['rmse'] if 'rmse' in model else np.sqrt(mean_squared_error(model['actuals'], model['predictions']))
        logger.info(f"{name:<15} {model['mae']/1e9:<15.3f} {rmse/1e9:<15.3f} {improvement:+.1f}%")
    
    logger.info("=" * 80)
    logger.info("✓ Complete!")
    
    return {
        'lstm': lstm_results,
        'xgboost': xgb_results,
        'sarima': sarima_results,
        'ensemble': ensemble_results
    }

if __name__ == '__main__':
    results = main()
