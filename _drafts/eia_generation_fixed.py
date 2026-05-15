import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
np.random.seed(42)

signalplot.apply(font_family="serif")


@dataclass
class Config:
    csv_path: str = "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5


def load_config(config_path=None) -> "Config":
    """Build Config from config.yaml, falling back to dataclass defaults."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        return Config()
    with open(config_path) as _f:
        import yaml as _yaml

        raw = _yaml.safe_load(_f) or {}
    _d = raw.get("data", {})
    _m = raw.get("model", {})
    _o = raw.get("output", {})
    return Config(
        csv_path=_d.get(
            "input_file",
            "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv",
        ),
        freq=_d.get("freq", "MS"),
        horizon=_m.get("horizon", 12),
        n_splits=_d.get("n_splits", 5),
    )


def load_eia_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    if not p.exists():
        raise FileNotFoundError("EIA CSV not found. Update Config.csv_path.")
    # Assume first column is ISO date (YYYY-MM-01), second column numeric value
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    out = df.dropna()
    return out.sort_values("date").set_index("date")["value"].asfreq(cfg.freq)


def rolling_origin_eval(y: pd.Series, horizon: int, n_splits: int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    idx = np.arange(len(y))
    metrics = []
    last_plot = None
    for fold, (tr, te) in enumerate(tscv.split(idx), start=1):
        y_tr = y.iloc[tr]
        y_te = y.iloc[te][:horizon]
        if len(y_te) == 0:
            continue
        # Fit SARIMAX and ETS on TRAIN only (no leakage)
        ets = sm.tsa.ExponentialSmoothing(
            y_tr,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit()
        ets_fore = ets.forecast(len(y_te))
        arima = sm.tsa.statespace.SARIMAX(
            y_tr,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        arima_fore = arima.forecast(len(y_te))
        metrics.append(
            {
                "fold": fold,
                "ETS_MAE": float(mean_absolute_error(y_te, ets_fore)),
                "ARIMA_MAE": float(mean_absolute_error(y_te, arima_fore)),
            }
        )
        last_plot = (
            y_tr.index,
            y_tr.values,
            y_te.index,
            y_te.values,
            ets_fore.values,
            arima_fore.values,
        )
    return metrics, last_plot


def main(plot: bool = False):
    cfg = load_config()
    y = load_eia_series(cfg)
    metrics, last = rolling_origin_eval(y, horizon=cfg.horizon, n_splits=cfg.n_splits)
    ets_mean = np.mean([m["ETS_MAE"] for m in metrics])
    arima_mean = np.mean([m["ARIMA_MAE"] for m in metrics])
    logger.info(f"ETS mean MAE: {ets_mean:.4f}")
    logger.info(f"SARIMAX mean MAE: {arima_mean:.4f}")
    if last:
        tr_idx, tr_vals, te_idx, te_vals, ets_vals, arima_vals = last
    if plot:
        plt.figure(figsize=(9, 4))
        plt.plot(tr_idx, tr_vals, label="Train")
        plt.plot(te_idx, te_vals, label="Test")
        plt.plot(te_idx, ets_vals, label="ETS Forecast", linestyle="--")
        plt.plot(te_idx, arima_vals, label="SARIMAX Forecast", linestyle=":")
        plt.legend()
        plt.title("EIA Generation – Rolling-Origin (last fold)")
        plt.xlabel("Time")
        plt.ylabel("Value")
        signalplot.save("eia_generation_last_fold.png")


if __name__ == "__main__":
    main()
