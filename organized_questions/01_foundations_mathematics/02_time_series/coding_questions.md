# Time Series Interview Questions - Coding Questions

## Question 1

**Implement aPython functionto performsimple exponential smoothingon atime series.**

**Simple Exponential Smoothing Formula:**
$$F_{t+1} = \alpha \cdot Y_t + (1-\alpha) \cdot F_t$$

where $\alpha$ is smoothing parameter (0 to 1)

**Code:**
```python
def simple_exponential_smoothing(series, alpha):
    """
    Simple Exponential Smoothing implementation.
    
    Args:
        series: list of values
        alpha: smoothing parameter (0 <= alpha <= 1)
    
    Returns:
        list of smoothed values
    """
    # Validate alpha
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    if len(series) == 0:
        return []
    
    # Initialize: first forecast = first actual value
    result = [series[0]]
    
    # Apply formula for remaining values
    for t in range(1, len(series)):
        forecast = alpha * series[t-1] + (1 - alpha) * result[t-1]
        result.append(forecast)
    
    return result

# Example usage
data = [20, 22, 25, 23, 26, 24, 28, 27, 29, 30]
smoothed = simple_exponential_smoothing(data, alpha=0.3)
print(f"Original: {data}")
print(f"Smoothed: {[round(x, 2) for x in smoothed]}")
```

**Output:** Smoothed series that follows original with less noise

---

## Question 2

**Usingpandas, write ascriptto detectseasonalityin atime series dataset.**

**Approach:** Group by seasonal period → Compare distributions

**Code:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create sample data with weekly seasonality
dates = pd.date_range('2022-01-01', periods=365, freq='D')
base_value = 100
# Higher values on weekends (days 5, 6)
seasonal = [15 if d.dayofweek >= 5 else 0 for d in dates]
noise = np.random.randn(365) * 5
sales = base_value + seasonal + noise

df = pd.DataFrame({'sales': sales}, index=dates)

# Step 2: Add day of week column
df['day_of_week'] = df.index.dayofweek

# Step 3: Method 1 - Group by day, calculate mean
daily_means = df.groupby('day_of_week')['sales'].mean()
print("Mean by day of week:")
print(daily_means)

# Step 4: Method 2 - Visualize with boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar plot of means
daily_means.plot(kind='bar', ax=axes[0])
axes[0].set_title('Mean Sales by Day')
axes[0].set_xlabel('Day (0=Mon, 6=Sun)')

# Box plots by day
df.boxplot(column='sales', by='day_of_week', ax=axes[1])
axes[1].set_title('Sales Distribution by Day')
plt.tight_layout()
plt.show()

# Result: Days 5-6 have higher means → weekly seasonality detected
```

**Output:** Higher means/medians for weekends indicate weekly seasonality

---

## Question 3

**Code anARIMA modelinPythonon a given dataset and visualize the forecasts.**

**Pipeline:** Check stationarity → Fit ARIMA → Forecast → Plot

**Code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Step 1: Create sample data (random walk with drift)
np.random.seed(42)
n = 100
data = np.cumsum(0.5 + np.random.randn(n))

# Step 2: Check stationarity
result = adfuller(data)
print(f"ADF p-value: {result[1]:.4f}")
print("Non-stationary" if result[1] > 0.05 else "Stationary")

# Step 3: Train-test split
train_size = 80
train, test = data[:train_size], data[train_size:]

# Step 4: Fit ARIMA model
# p=1 (AR), d=1 (difference once), q=1 (MA)
model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit()
print(fitted.summary())

# Step 5: Forecast
forecast = fitted.forecast(steps=len(test))

# Step 6: Visualize
plt.figure(figsize=(12, 5))
plt.plot(range(train_size), train, label='Train', color='blue')
plt.plot(range(train_size, n), test, label='Test', color='green')
plt.plot(range(train_size, n), forecast, label='Forecast', 
         color='red', linestyle='--')
plt.axvline(x=train_size, color='black', linestyle=':')
plt.legend()
plt.title('ARIMA(1,1,1) Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Step 7: Calculate error
mae = np.mean(np.abs(test - forecast))
print(f"MAE: {mae:.2f}")
```

---

## Question 4

**Fit aGARCH modelto afinancial time series datasetand interpret the results.**

**Pipeline:** Get returns → Fit GARCH(1,1) → Interpret persistence

**Code:**
```python
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# Step 1: Create sample returns (or use real data)
np.random.seed(42)
n = 1000
# Simulate returns with volatility clustering
returns = np.random.randn(n) * 0.02

# Step 2: Fit GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant')
result = model.fit(disp='off')

# Step 3: Print summary
print(result.summary())

# Step 4: Extract key parameters
omega = result.params['omega']      # constant
alpha = result.params['alpha[1]']   # ARCH term (past shock impact)
beta = result.params['beta[1]']     # GARCH term (persistence)

print(f"\n--- Interpretation ---")
print(f"omega (ω): {omega:.6f}")
print(f"alpha (α): {alpha:.4f} - Impact of past shock")
print(f"beta (β): {beta:.4f} - Persistence of volatility")
print(f"α + β: {alpha + beta:.4f} - Total persistence")
print(f"  (Close to 1 = shocks have long-lasting effect)")

# Step 5: Forecast volatility
forecast = result.forecast(horizon=10)
vol_forecast = np.sqrt(forecast.variance.values[-1])
print(f"\nVolatility forecast (next 10 periods): {vol_forecast}")

# Step 6: Plot conditional volatility
fig, ax = plt.subplots(figsize=(12, 4))
result.conditional_volatility.plot(ax=ax)
ax.set_title('Conditional Volatility (GARCH)')
ax.set_ylabel('Volatility')
plt.show()
```

**Interpretation:** α + β close to 1 means high persistence (shocks last long)

---

## Question 5

**Create aPython scriptthat decomposes atime seriesintotrend,seasonality, andresidualsusingstatsmodels library.**

**Pipeline:** Load data → STL decompose → Plot components

**Code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose

# Step 1: Create sample data with trend + seasonality + noise
np.random.seed(42)
n = 365 * 2  # 2 years of daily data

# Components
time = np.arange(n)
trend = 100 + 0.05 * time  # Upward trend
seasonal = 10 * np.sin(2 * np.pi * time / 365)  # Yearly seasonality
noise = np.random.randn(n) * 3  # Random noise

# Combine
data = trend + seasonal + noise

# Create time series
dates = pd.date_range('2022-01-01', periods=n, freq='D')
ts = pd.Series(data, index=dates)

# Step 2: STL Decomposition (Robust to outliers)
stl = STL(ts, period=365, robust=True)
result = stl.fit()

# Step 3: Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(ts)
axes[0].set_title('Original')

axes[1].plot(result.trend)
axes[1].set_title('Trend')

axes[2].plot(result.seasonal)
axes[2].set_title('Seasonal')

axes[3].plot(result.resid)
axes[3].set_title('Residuals')

plt.tight_layout()
plt.show()

# Step 4: Verify decomposition
# Original ≈ Trend + Seasonal + Residual
reconstructed = result.trend + result.seasonal + result.resid
print(f"Reconstruction error: {np.mean(np.abs(ts - reconstructed)):.6f}")
```

**Output:** Four plots showing original data and extracted components

---

## Question 6

**Write aPython functionto calculate and plot theACFandPACFfor a giventime series.**

**Pipeline:** Input series → Calculate ACF/PACF → Plot with significance bands

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def plot_acf_pacf(series, lags=30, title=""):
    """
    Plot ACF and PACF for a time series.
    
    Args:
        series: array-like time series data
        lags: number of lags to plot
        title: plot title prefix
    
    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # ACF Plot
    plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f'{title} ACF')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Autocorrelation')
    
    # PACF Plot
    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title(f'{title} PACF')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Partial Autocorrelation')
    
    plt.tight_layout()
    return fig

# Example: Create AR(2) process
np.random.seed(42)
n = 300
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.6*y[t-1] - 0.3*y[t-2] + np.random.randn()

# Plot ACF and PACF
fig = plot_acf_pacf(y, lags=20, title="AR(2) Process")
plt.show()

# Interpretation guide
print("--- Interpretation ---")
print("If ACF tails off, PACF cuts off at p → AR(p)")
print("If ACF cuts off at q, PACF tails off → MA(q)")
print("If both tail off → ARMA(p,q)")
print("\nFor AR(2): Expect PACF to cut off after lag 2")
```

**Output:** 
- ACF: Shows total correlation (should tail off for AR)
- PACF: Shows direct correlation (should cut off at lag 2 for AR(2))
- Blue bands = 95% confidence interval

---