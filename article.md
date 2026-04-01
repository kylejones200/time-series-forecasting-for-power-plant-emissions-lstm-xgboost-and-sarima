# Time Series Forecasting for Power Plant Emissions: LSTM, XGBoost, and SARIMA Comparing three state-of-the-art forecasting methods on 27 years of EPA
emissions data to predict the future of energy generation

::::### Time Series Forecasting for Power Plant Emissions: LSTM, XGBoost, and SARIMA 

#### *Comparing three state-of-the-art forecasting methods on 27 years of EPA emissions data to predict the future of energy generation*
Predicting future emissions from power plants isn't just an academic
exercise --- it's critical for climate policy, regulatory compliance,
and grid planning. With 27 years of [EPA
eGRID](https://www.epa.gov/egrid)
data covering every power plant in the United States, we can build
sophisticated models to forecast CO2 emissions through 2030.

But which forecasting method works best? Should you use deep learning
LSTMs, gradient boosting with XGBoost, or classical statistical SARIMA
models? This article compares all three approaches using real-world data
from 108,000+ plant-year observations.

### Why Time Series Forecasting Matters for Energy
The power sector accounts for approximately 25% of U.S. greenhouse gas
emissions. This matters to several groups of decision makers.

Policy Makers: Set realistic emission reduction targets and evaluate
policy effectiveness. Knowing whether emissions will naturally decline
or require intervention drives billion-dollar decisions.

Grid Operators: Plan capacity additions and retirements. If coal plant
emissions are declining faster than expected, renewable capacity needs
to be added sooner to maintain grid reliability.

Investors: Identify opportunities in clean energy. Understanding
emission trends helps value renewable energy projects and assess
stranded asset risk in fossil fuel plants.

Electric utilities: Comply with regulatory requirements and plan capital
investments. Emissions forecasts determine whether new pollution
controls are needed or if transitioning to cleaner fuels makes more
economic sense.

The challenge? Energy time series contain complex patterns: long-term
trends (coal-to-gas switching), cyclical behavior (economic cycles), and
structural breaks (policy changes). Simple extrapolation fails.


### The Dataset: 27 Years of Power Plant Emissions
We're using EPA's eGRID (Emissions & Generation Resource Integrated
Database), which contains:

- 108,129 plant-year records from 1996--2023
- CO2, NOx, SO2, and other emissions
- Generation by fuel type (coal, gas, nuclear, renewables)
- Plant-level efficiency metrics

*The data and code for this project are in*
[*Github*](https://github.com/kylejones200/electric_utilities)*.*

The data shows fascinating trends: total CO2 emissions peaked around
2007, then declined 30%+ due to coal retirements, natural gas switching,
and renewable energy growth. But will this continue? Can we forecast the
trajectory through 2030?


Output shows a 32% decline in absolute emissions despite relatively
stable generation --- efficiency improvements and fuel switching at
work!

### Method 1: LSTM Neural Networks
Long Short-Term Memory networks excel at learning long-term dependencies
in sequential data. Unlike traditional RNNs, LSTMs have "memory cells"
that can retain information across many time steps.

Why LSTMs for Energy Forecasting?

Energy systems have long memory effects: A coal plant retirement today
affects emissions for 40+ years. Natural gas prices from previous
quarters influence generation decisions. Policy changes create lasting
structural shifts.

LSTMs can capture these complex temporal dynamics without manually
specifying lag structures.


The LSTM architecture uses:

- Two LSTM layers (50 units each) to capture temporal patterns
- Dropout layers (20%) to prevent overfitting
- Dense layers for final prediction

Training takes \~5 minutes on a standard laptop. The model learns to
predict emissions with R² = 0.89 on test data.

### Method 2: XGBoost with Feature Engineering
XGBoost (Extreme Gradient Boosting) is a powerful tree-based method.
While it doesn't inherently understand time, we can engineer temporal
features to make it work brilliantly for time series.

Feature Engineering is Key

The magic of XGBoost for time series lies in creating the right
features:


XGBoost achieves R² = 0.93 on test data --- even better than LSTM!

Why XGBoost Outperforms

Three reasons:

1.  [Explicit feature engineering captures domain knowledge about how
    emissions evolve]
2.  [Robust to overfitting through regularization and tree
    pruning]
3.  [Fast training (seconds vs minutes for LSTM) enables rapid
    iteration]

Feature importance analysis reveals insights:


Output:


Last year's emissions are the strongest predictor (45% importance),
followed by the 3-year average trend. This makes intuitive
sense --- power plant emissions change gradually.

### Method 3: SARIMA --- The Statistical Baseline
SARIMA (Seasonal AutoRegressive Integrated Moving Average) is the
classical approach to time series forecasting. While less flashy than
deep learning, it provides:

- Interpretable parameters (you understand *why* it forecasts what it
  does)
- Confidence intervals (quantify uncertainty)
- No need for large datasets (works with limited data)


SARIMA achieves R² = 0.85 on test data --- respectable, though slightly
behind the ML methods.

The optimal parameters (2, 1, 1) indicate:

- AR(2): Use past 2 years to predict
- I(1): First-difference the data (emissions are
  non-stationary)
- MA(1): Include 1 year of error correction

### The Power of Ensembles
Rather than picking one method, we can combine all three. Ensemble
methods often outperform individual models by:

- Reducing variance (averaging smooths predictions)
- Capturing different patterns (LSTM sees long-term trends, XGBoost
  captures recent dynamics, SARIMA provides statistical
  foundation)
- Being more robust to model misspecification

Simple Averaging


Weighted Ensemble (Meta-Learning)

Train a meta-model that learns optimal weights:


Output:


The ensemble learns to weight XGBoost most heavily (51%), with LSTM
secondary (28%) and SARIMA as a stabilizing factor (21%).

Results: R² = 0.96 on test data! The ensemble beats all individual
models.

### Forecasting the Future: 2024--2030
Now the moment of truth: forecasting emissions through 2030.


Output:


The forecast shows a continued 3% annual decline through 2030, reaching
1.56 billion tons --- a 40% reduction from 2007 peak levels.

This trajectory assumes:

- Continued coal retirements (\~5--8 GW/year)
- Natural gas remains economically competitive
- Renewable additions continue at current pace (\~20--30
  GW/year)
- No major policy changes

### Uncertainty Quantification
Point forecasts are useful, but uncertainty matters. We use quantile
regression to generate prediction intervals:


The 80% prediction interval widens over time (±8% by 2030), reflecting
increasing uncertainty. Major policy changes or economic shocks could
shift the trajectory significantly.

### Key Takeaways
1\. Ensemble methods win: Combining LSTM, XGBoost, and SARIMA achieved
R² = 0.96, beating any individual model.

2\. Feature engineering matters: XGBoost with good features (lags,
rolling stats, differences) matched LSTM performance with 100x faster
training.

3\. Domain knowledge helps: Understanding power sector dynamics
(slow-changing infrastructure, policy impacts) improves feature design
and model interpretation.

4\. Quantify uncertainty: Prediction intervals are crucial for
decision-making. The ±8% range by 2030 represents billions of tons of
difference.

5\. Start simple, add complexity: SARIMA provides a strong baseline in
minutes. Only move to complex models if they provide meaningful
improvements

### So What?
Time series forecasting transforms historical patterns into actionable
insights. For the power sector, these forecasts drive:

- \$100B+ in annual capital allocation decisions
- Climate policy development across federal and state
  governments
- Grid reliability planning ensuring lights stay on
- Market strategies for energy traders and investors

As the energy transition accelerates, accurate forecasting becomes even
more critical. The methods shown here --- LSTM for complex patterns,
XGBoost for feature-rich modeling, SARIMA for statistical rigor, and
ensembles for robustness --- provide a comprehensive toolkit for
tackling these challenges.
::::Update (2025--11--10): I refactored the code to run a little better.
