import signalplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.random.seed(42)

signalplot.apply(font_family='serif')


@dataclass
class Config:
    csv_path: str = "emissions.csv"
    time_col: str = "date"
    target_col: str = "emissions"
    feature_cols: Tuple[str, ...] = ()
    freq: str = "D"
    n_splits: int = 5

def load_config(config_path=None) -> 'Config':
    """Build Config from config.yaml, falling back to dataclass defaults."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return Config()
    with open(config_path) as _f:
        import yaml as _yaml
        raw = _yaml.safe_load(_f) or {}
    _d = raw.get('data', {})
    _m = raw.get('model', {})
    _o = raw.get('output', {})
    return Config(
        csv_path=_d.get('input_file', 'emissions.csv'),
        time_col=_m.get('time_col', 'date'),
        target_col=_m.get('target_col', 'emissions'),
        feature_cols=_m.get('feature_cols', ()),
        freq=_d.get('freq', 'D'),
        n_splits=_d.get('n_splits', 5),
    )



def load_data(cfg: Config) -> pd.DataFrame:
    p = Path(cfg.csv_path)
    if not p.exists():
        raise FileNotFoundError("Power plant emissions CSV not found. See DATA_PowerPlant_Emissions.md for required columns and sample.")
    df = pd.read_csv(p)
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    df = df.sort_values(cfg.time_col).set_index(cfg.time_col)
    return df


def chrono_split_index(n: int, test_size: float = 0.2):
    split = int(np.floor((1 - test_size) * n))
    return np.arange(0, split), np.arange(split, n)


def make_lag_features(df: pd.DataFrame, col: str, lags=(1, 7, 14, 28)) -> pd.DataFrame:
    out = df.copy()
    for l in lags:
        out[f"{col}_lag{l}"] = out[col].shift(l)
    return out


def rf_pipeline(cfg: Config):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    ])


def eval_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"mae": float(mae), "rmse": float(rmse)}


def fit_eval_rf(df: pd.DataFrame, cfg: Config, plot: bool = False):
    series = df[[cfg.target_col]].asfreq(cfg.freq)
    feat_df = make_lag_features(series, cfg.target_col).dropna()
    X = feat_df.drop(columns=[cfg.target_col]).values
    y = feat_df[cfg.target_col].values

    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    metrics = []
    last = None
    for tr, te in tscv.split(X):
        model = rf_pipeline(cfg)
        model.fit(X[tr], y[tr])
        yhat = model.predict(X[te])
        metrics.append(eval_regression(y[te], yhat))
        last = (model, tr, te)
    logger.info("RF mean metrics:", {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]})

    if last is not None:
        model, tr, te = last
        idx = feat_df.index
    if plot:
            plt.figure(figsize=(9, 4))
            plt.plot(idx[tr], y[tr], label="Train")
            plt.plot(idx[te], y[te], label="Test")
            plt.plot(idx[te], model.predict(X[te]), label="RF Pred", linestyle="--")
            plt.legend(); plt.title("RandomForest (lag features) – Chrono CV last fold")
            plt.xlabel("Time"); plt.ylabel(cfg.target_col)
            signalplot.save("power_rf_last_fold.png")


def fit_eval_sarimax(df: pd.DataFrame, cfg: Config, order=(1,1,1), seasonal_order=(0,0,0,0)):
    y = df[[cfg.target_col]].asfreq(cfg.freq).dropna()[cfg.target_col]
    tr_idx, te_idx = chrono_split_index(len(y), test_size=0.2)
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    model = sm.tsa.statespace.SARIMAX(y_tr, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    yhat = res.forecast(steps=len(y_te))
    m = eval_regression(y_te.values, yhat.values)
    logger.info("SARIMAX test metrics:", m)

    if plot:
        plt.figure(figsize=(9, 4))
        plt.plot(y_tr.index, y_tr.values, label="Train")
        plt.plot(y_te.index, y_te.values, label="Test")
        plt.plot(y_te.index, yhat.values, label="SARIMAX Forecast", linestyle="--")
        plt.legend(); plt.title("SARIMAX – Chronological holdout")
        plt.xlabel("Time"); plt.ylabel(cfg.target_col)
        signalplot.save("power_sarimax_holdout.png")


def main():
    cfg = load_config()
    df = load_data(cfg)
    fit_eval_rf(df, cfg)
    fit_eval_sarimax(df, cfg)

if __name__ == "__main__":
    main()
