# Time Series Forecasting for Power Plant Emissions LSTM XGBoost and SARIMA

Published: 2025-10-06
Medium: [https://medium.com/@kyle-t-jones/time-series-forecasting-for-power-plant-emissions-lstm-xgboost-and-sarima-5b69867faa86](https://medium.com/@kyle-t-jones/time-series-forecasting-for-power-plant-emissions-lstm-xgboost-and-sarima-5b69867faa86)

## Business context

Predicting future emissions from power plants isn't just an academic exercise --- it's critical for climate policy, regulatory compliance, and grid planning. With 27 years of [EPA eGRID](https://www.epa.gov/egrid) data covering every power plant in the United States, we can build sophisticated models to forecast CO2 emissions through 2030.

But which forecasting method works best? Should you use deep learning LSTMs, gradient boosting with XGBoost, or classical statistical SARIMA models? This article compares all three approaches using real-world data from 108,000+ plant-year observations.

The power sector accounts for approximately 25% of U.S. greenhouse gas emissions. This matters to several groups of decision makers.



## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — lag feature matrix. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p time_series_forecasting_for_power_plant_emissions_lstm_xgboost_and_sarima_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).