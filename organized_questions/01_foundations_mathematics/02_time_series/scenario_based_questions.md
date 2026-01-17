# Time Series Interview Questions - Scenario_Based Questions

## Question 1

**Discuss the importance oflag selectioninARMA/ARIMAmodels.**

**Why Lag Selection Matters:**

| Problem | Consequence |
|---------|-------------|
| **Underfitting** (p, q too small) | Model too simple; residuals have autocorrelation; forecasts poor |
| **Overfitting** (p, q too large) | Model fits noise; poor generalization; violates parsimony |

**Two Strategies for Lag Selection:**

**Strategy 1: Box-Jenkins Method (Manual)**
```
1. Make series stationary (differencing)
2. Plot ACF and PACF
3. Apply rules:
   - PACF cuts off at p, ACF tails off → AR(p)
   - ACF cuts off at q, PACF tails off → MA(q)
   - Both tail off → ARMA(p,q)
4. Fit model, check residuals
5. Iterate if needed
```

**Strategy 2: Information Criteria (Automated)**
```
1. Fit candidate models: ARIMA(0,d,0), (1,d,0), (0,d,1), (1,d,1), ...
2. Calculate AIC/BIC for each
3. Select model with LOWEST AIC or BIC
   - AIC = -2*log(L) + 2*k  (penalizes complexity)
   - BIC = -2*log(L) + k*log(n)  (stronger penalty)
```

**Validation:**
- Check residuals are white noise (no autocorrelation in ACF)
- If patterns remain → model needs adjustment

**Python Example:**
```python
from pmdarima import auto_arima

# Automatically selects best (p,d,q) using AIC
model = auto_arima(data, seasonal=False, trace=True)
print(model.summary())
```

---

## Question 2

**Discuss the use and considerations ofrolling-window analysisintime series.**

**Definition:**
Rolling-window analysis applies a function (mean, variance, correlation) over a sliding window of fixed size, moving one step at a time.

**Use Cases:**

| Application | What It Shows |
|-------------|---------------|
| Rolling Mean | Trend smoothing (removes noise) |
| Rolling Std | Time-varying volatility |
| Rolling Correlation | How correlation changes over time |
| Rolling Forecast | Walk-forward validation |

**Key Considerations:**

**1. Window Size Selection**
- **Too small:** Noisy, captures short-term fluctuations
- **Too large:** Overly smooth, misses recent changes
- Rule of thumb: Start with period length (7 for daily→weekly, 12 for monthly→yearly)

**2. Edge Effects**
- First (window_size - 1) values are NaN
- Decide: drop, use expanding window, or backfill

**3. Computational Cost**
- Large windows + long series = expensive
- Use efficient implementations (pandas rolling)

**Python Example:**
```python
import pandas as pd
import numpy as np

# Sample data
data = pd.Series(np.random.randn(100).cumsum())

# Rolling statistics
rolling_mean = data.rolling(window=20).mean()
rolling_std = data.rolling(window=20).std()

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(data, alpha=0.5, label='Original')
plt.plot(rolling_mean, label='Rolling Mean (20)')
plt.fill_between(range(len(data)), 
                 rolling_mean - 2*rolling_std,
                 rolling_mean + 2*rolling_std, 
                 alpha=0.2, label='±2 Std')
plt.legend()
plt.show()
```

---

## Question 3

**Discuss the advantage of usingstate-space modelsand theKalman filterfortime series analysis.**

**Definition:**
State-space models represent a time series through two equations:
1. **State Equation:** How hidden state evolves: $x_t = A \cdot x_{t-1} + w_t$
2. **Observation Equation:** How we observe state: $y_t = C \cdot x_t + v_t$

**Kalman Filter:**
Optimal algorithm to estimate hidden states from noisy observations.

**Advantages:**

| Advantage | Explanation |
|-----------|-------------|
| **Unified Framework** | ARIMA, exponential smoothing are special cases |
| **Handle Missing Data** | Naturally imputes missing observations |
| **Time-Varying Parameters** | Parameters can change over time |
| **Multiple Related Series** | Easily model multivariate data |
| **Online Updating** | Update estimates as new data arrives |
| **Uncertainty Quantification** | Provides state uncertainty estimates |

**Use Cases:**
- Tracking objects (navigation, GPS)
- Economic indicators with measurement error
- Combining multiple noisy sensors
- Time-varying regression coefficients

**Python Example:**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMAX uses state-space formulation internally
model = SARIMAX(data, order=(1,1,1))
result = model.fit()

# Kalman filter provides smoothed estimates
smoothed_states = result.smoothed_state
```

---

## Question 4

**How would you approach building atime series modelto forecaststock prices?**

**Important Reality Check:**
Stock prices are nearly random walks - predicting exact prices is extremely hard. Focus on:
- Direction prediction (up/down)
- Volatility prediction (more feasible)
- Risk management

**Approach:**

**Step 1: Data Preparation**
```python
# Work with returns, not prices (more stationary)
returns = np.log(prices).diff().dropna()
```

**Step 2: Feature Engineering**
- Technical indicators: moving averages, RSI, MACD
- Lag features: past returns
- External: market index, sector performance, VIX

**Step 3: Model Selection**

| Approach | Model | What It Predicts |
|----------|-------|------------------|
| Classical | ARIMA | Returns (usually weak) |
| Volatility | GARCH | Volatility (works well) |
| ML | XGBoost, LSTM | Direction or returns |

**Step 4: Walk-Forward Validation**
```python
# Never look into future - strict temporal split
for t in range(train_size, len(data)):
    train = data[:t]
    model.fit(train)
    pred = model.predict(1)
    # Compare with actual data[t]
```

**Step 5: Evaluation**
- For returns: MAE, RMSE
- For direction: Accuracy, F1-score
- For trading: Sharpe ratio, drawdown

**Honest Expectation:**
Most models barely beat naive baseline. Focus on risk management over prediction.

---

## Question 5

**Discuss the challenges and strategies of usingtime series analysisinanomaly detectionforsystem monitoring.**

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| **What is "normal"?** | Normal behavior changes over time |
| **Rare events** | Few anomalies to learn from |
| **Real-time requirements** | Must detect quickly |
| **False positives** | Too many alerts = alert fatigue |
| **Contextual anomalies** | Value normal at some times, anomalous at others |

**Strategies:**

**1. Statistical Methods**
```python
# Z-score method
rolling_mean = data.rolling(window).mean()
rolling_std = data.rolling(window).std()
z_score = (data - rolling_mean) / rolling_std
anomaly = abs(z_score) > threshold  # e.g., 3
```

**2. Forecasting-Based**
```python
# Predict next value, flag if actual deviates too much
model = ARIMA(data, order=(1,1,1)).fit()
forecast = model.forecast(1)
residual = actual - forecast
anomaly = abs(residual) > threshold
```

**3. LSTM Autoencoder**
- Train autoencoder on normal data
- High reconstruction error = anomaly

**4. Isolation Forest**
```python
from sklearn.ensemble import IsolationForest
# Create features from time series
model = IsolationForest(contamination=0.01)
anomalies = model.fit_predict(features)
```

**Best Practices:**
- Adapt threshold dynamically (seasonal patterns)
- Combine multiple methods
- Human-in-the-loop for labeling

---

## Question 6

**How would you usetime series analysisto predictelectricity consumption patterns?**

**Key Characteristics of Electricity Data:**
- Strong daily seasonality (morning/evening peaks)
- Weekly seasonality (weekday vs weekend)
- Annual seasonality (summer AC, winter heating)
- Temperature correlation
- Holiday effects

**Approach:**

**Step 1: Data and Features**
- Hourly consumption data (2+ years)
- Weather data (temperature, humidity)
- Calendar features (hour, day, month, holiday flags)

**Step 2: Model Selection**
```
Best choice: SARIMAX or Prophet
- Handles multiple seasonalities
- Incorporates temperature as exogenous variable
```

**Step 3: Implementation**
```python
from prophet import Prophet

# Prepare data
df = pd.DataFrame({'ds': dates, 'y': consumption})

# Add temperature as regressor
df['temp'] = temperature

# Fit Prophet (handles multiple seasonalities automatically)
model = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True)
model.add_regressor('temp')
model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=168)  # 1 week ahead
future['temp'] = future_temperature
forecast = model.predict(future)
```

**Business Application:**
- Grid operators: Plan generation capacity
- Energy traders: Price forecasting
- Consumers: Optimize usage, reduce costs

---

## Question 7

**Propose a strategy for forecastingtourist arrivalsusingtime series data.**

**Characteristics of Tourism Data:**
- Strong annual seasonality (summer peaks, holiday periods)
- Trend (growing/declining destination)
- External shocks (pandemics, events, exchange rates)
- Marketing campaign effects

**Strategy:**

**Step 1: Data Collection**
- Monthly arrivals (5+ years)
- External variables: GDP of source countries, exchange rates, flight availability
- Events: Olympics, World Cup, festivals

**Step 2: EDA**
```python
# Decompose to understand patterns
from statsmodels.tsa.seasonal import STL
stl = STL(arrivals, period=12)
result = stl.fit()
result.plot()
```

**Step 3: Model - SARIMAX**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Include GDP and exchange rate as exogenous
exog = df[['source_gdp', 'exchange_rate', 'marketing_spend']]

model = SARIMAX(arrivals, 
                exog=exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
result = model.fit()

# Forecast requires future exog values (use forecasts or scenarios)
```

**Step 4: Scenario Planning**
- Optimistic: Strong economic growth
- Baseline: Current trends continue
- Pessimistic: Economic downturn

**Business Use:**
- Hotel capacity planning
- Airline scheduling
- Government tourism board budgeting

---

## Question 8

**How would you analyze and predict theload on a serverusingtime series?**

**Characteristics:**
- High-frequency data (per second/minute)
- Multiple seasonalities (hourly, daily, weekly)
- Spikes from user activity
- Trend from growing user base

**Approach:**

**Step 1: Data Aggregation**
```python
# Raw per-second data → aggregate to 5-minute or hourly
load_hourly = load_raw.resample('H').mean()
```

**Step 2: Features**
- Hour of day, day of week
- Rolling statistics
- Recent lag values
- Special events (deployments, marketing campaigns)

**Step 3: Model Options**

| Model | Best For |
|-------|----------|
| Prophet | Multiple seasonalities, easy to use |
| XGBoost | Many features, non-linear patterns |
| LSTM | Complex sequential patterns |

**Step 4: Real-Time Monitoring**
```python
# Combine forecast with anomaly detection
forecast = model.predict(current_features)
if actual_load > forecast + 3*std:
    alert("Unexpected load spike!")

# Auto-scaling decision
if forecast_next_hour > capacity_threshold:
    scale_up_servers()
```

**Business Impact:**
- Proactive auto-scaling (cost savings)
- Capacity planning for infrastructure purchases
- SLA compliance monitoring

---

## Question 9

**Discuss your approach to evaluating the impact ofpromotional campaignsonsalesusingtime series analysis.**

**Challenge:**
Separate promotion effect from:
- Underlying trend
- Seasonality
- Competitor actions
- Random noise

**Approach 1: Regression with Promotional Dummy**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create promotion indicator
df['promotion'] = (df['date'].isin(promo_dates)).astype(int)

# Fit model
model = SARIMAX(df['sales'], 
                exog=df['promotion'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
result = model.fit()

# Promotion effect = coefficient on promotion variable
print(f"Promotion uplift: {result.params['promotion']:.2f} units")
```

**Approach 2: CausalImpact (Bayesian)**
```python
from causalimpact import CausalImpact

# Pre-period: before promotion
# Post-period: during/after promotion
data = df[['sales', 'competitor_sales']]  # control series

impact = CausalImpact(data, pre_period, post_period)
impact.plot()
impact.summary()

# Output: Estimated causal effect with confidence intervals
```

**Key Considerations:**
- **Control for seasonality:** Don't attribute seasonal peak to promotion
- **Cannibalization:** Promotion may steal sales from future periods
- **Long-term effect:** Does it create loyal customers or just deal-seekers?
- **A/B testing:** If possible, run controlled experiment

---

## Question 10

**Discuss the potential ofrecurrent neural networks (RNNs)intime series forecasting.**

**Why RNNs for Time Series:**
- Designed for sequential data
- Maintain hidden state (memory)
- Learn complex non-linear patterns
- Automatic feature extraction

**Types:**

| Architecture | Strength | Weakness |
|--------------|----------|----------|
| Vanilla RNN | Simple | Vanishing gradient, short memory |
| LSTM | Long-term memory | More parameters |
| GRU | Faster than LSTM | Similar performance |
| Transformer | Attention mechanism | Needs more data |

**When RNNs Excel:**
- Very long sequences with long-range dependencies
- Multivariate with complex interactions
- Large amounts of training data
- Non-linear relationships

**When NOT to Use:**
- Small datasets (ARIMA better)
- Need interpretability
- Simple patterns (overkill)

**Implementation Example:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Prepare sequences
def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, lookback=30)

# Build LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(30, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
```

**Practical Tips:**
- Start with simpler models (ARIMA) as baseline
- Scale data (MinMaxScaler or StandardScaler)
- Use early stopping to prevent overfitting
- Consider attention-based models (Transformers) for state-of-the-art

---