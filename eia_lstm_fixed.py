import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel

np.random.seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
    input_chunk_length: int = 24
    output_chunk_length: int = 12
    epochs: int = 50


def load_series(cfg: Config) -> TimeSeries:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float32")
    df = df.dropna().sort_values("date")
    ts = TimeSeries.from_dataframe(df, time_col="date", value_cols=["value"], freq=cfg.freq)
    return ts


def rolling_origin_lstm(ts: TimeSeries, cfg: Config):
    s = ts.to_series()
    idx = np.arange(len(s))
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    maes = []
    last_true, last_pred = None, None
    for tr, te in tscv.split(idx):
        end = tr[-1]
        y_tr = ts.drop_after(ts.time_index[end])
        future = ts.split_after(ts.time_index[end])[1]
        y_te = future.drop_after(future.time_index[min(cfg.horizon-1, len(future)-1)])
        if len(y_te) == 0:
            continue
        model = RNNModel(
            model="LSTM",
            input_chunk_length=cfg.input_chunk_length,
            output_chunk_length=cfg.output_chunk_length,
            training_length=24,
            n_rnn_layers=2,
            hidden_dim=64,
            n_epochs=cfg.epochs,
            random_state=42,
            pl_trainer_kwargs={
                "enable_progress_bar": False,
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
            },
        )
        model.fit(y_tr)
        fc = model.predict(len(y_te))
        mae = mean_absolute_error(y_te.values().ravel(), fc.values().ravel())
        print(f"Fold MAE: {mae:.3f}")
        maes.append(mae)
        last_true, last_pred = y_te, fc
    return float(np.mean(maes)), (last_true, last_pred)


def main():
    cfg = Config()
    ts = load_series(cfg)
    mean_mae, _ = rolling_origin_lstm(ts, cfg)
    print(f"LSTM mean MAE: {mean_mae}")

    # Tufte-style: fit on data through Dec 2024, forecast Jan–Aug 2025
    s = ts.to_series()
    start_2024 = pd.Period('2024-01', freq='M').start_time + pd.offsets.MonthBegin(0)
    end_2024 = pd.Period('2024-12', freq='M').start_time + pd.offsets.MonthBegin(0)
    jan_2025 = pd.Period('2025-01', freq='M').start_time + pd.offsets.MonthBegin(0)
    aug_2025 = pd.Period('2025-08', freq='M').start_time + pd.offsets.MonthBegin(0)

    y_hist = s.loc[start_2024:end_2024]
    y_act = s.loc[jan_2025:aug_2025]

    # Refit LSTM on training segment (up to Dec 2024)
    ts_train = TimeSeries.from_times_and_values(s.loc[:end_2024].index, s.loc[:end_2024].values, freq=cfg.freq)
    model = RNNModel(
        model="LSTM",
        input_chunk_length=cfg.input_chunk_length,
        output_chunk_length=1,
        training_length=24,
        n_rnn_layers=2,
        hidden_dim=64,
        n_epochs=cfg.epochs,
        random_state=42,
        pl_trainer_kwargs={
            "enable_progress_bar": False,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
        },
    )
    # Scale training data for stable RNN training
    scaler = Scaler()
    ts_train_s = scaler.fit_transform(ts_train)
    model.fit(ts_train_s)
    fc_s = model.predict(len(pd.period_range('2025-01', '2025-08', freq='M')))
    fc = scaler.inverse_transform(fc_s)

    # Light uncertainty band from seasonal differences std
    seas_diff = s.loc[:end_2024].diff(12).dropna()
    sigma = float(seas_diff.std(ddof=1)) if len(seas_diff) else 0.0
    f_idx = fc.time_index
    upper = TimeSeries.from_times_and_values(f_idx, fc.values().ravel() + 1.96 * sigma, freq=cfg.freq)
    lower = TimeSeries.from_times_and_values(f_idx, fc.values().ravel() - 1.96 * sigma, freq=cfg.freq)

    # Plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_hist.index, y_hist.values, color='#888888', lw=1.5)
    ax.axvline(jan_2025, color='#666666', linestyle='--', lw=1)
    ax.plot(y_act.index, y_act.values, color='#444444', lw=1.8)
    ax.fill_between(f_idx, lower.values().ravel(), upper.values().ravel(), color='#000000', alpha=0.06, linewidth=0)
    ax.plot(f_idx, fc.values().ravel(), color='#000000', lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if len(y_hist):
        ax.annotate('History (2024)', xy=(y_hist.index[-1], y_hist.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#666666')
    if len(y_act):
        ax.annotate('Actual (Jan–Aug 2025)', xy=(y_act.index[-1], y_act.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#444444')
    ax.annotate('Forecast', xy=(f_idx[-1], fc.values().ravel()[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#000000')

    ax.set_title('EIA Net Generation — LSTM forecast Jan–Aug 2025')
    ax.set_xlabel('')
    ax.grid(False)
    save_fig('eia_lstm_last_fold.png')

if __name__ == '__main__':
    main()
