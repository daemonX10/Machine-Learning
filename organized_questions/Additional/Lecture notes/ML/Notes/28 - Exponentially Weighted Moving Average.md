# Lecture 28: Exponentially Weighted Moving Average (EWMA)

## 1. What is EWMA?

**Exponentially Weighted Moving Average** is a technique to find **trends** in time series data by computing a running average that gives **more weight to recent observations** and **exponentially decaying weight** to older observations.

---

## 2. Applications

| Domain | Use Case |
|--------|----------|
| Time series forecasting | Identifying trends in temporal data |
| Financial forecasting | Stock price trend analysis |
| Signal processing | Noise filtering |
| **Deep learning** | Building optimizers (Momentum, RMSProp, Adam) |

---

## 3. Formula

$$V_t = \beta \cdot V_{t-1} + (1 - \beta) \cdot \theta_t$$

Where:
- $V_t$ = EWMA at time step $t$
- $V_{t-1}$ = EWMA at previous time step
- $\theta_t$ = actual observed value at time $t$
- $\beta$ = smoothing parameter $(0 < \beta < 1)$

Initial condition: $V_0 = 0$ (or $V_0 = \theta_0$ in some implementations)

---

## 4. Two Key Properties

### Property 1: Recent Points Get Higher Weight
Points that arrive **later in time** have **higher weightage** than older points.

### Property 2: Any Point's Weight Decays Over Time
As new data points arrive, the contribution of any given point **exponentially decreases**.

---

## 5. Calculation Example

Given data: Day 1 (15°C), Day 2 (17°C), Day 3 (20°C), ... with $\beta = 0.9$

| Step | Calculation | Result |
|------|-------------|--------|
| $V_0$ | 0 | 0 |
| $V_1$ | $0.9 \times 0 + 0.1 \times 15$ | 1.5 |
| $V_2$ | $0.9 \times 1.5 + 0.1 \times 17$ | 3.05 |
| $V_3$ | $0.9 \times 3.05 + 0.1 \times 20$ | 4.745 |
| ... | ... | ... |

---

## 6. Mathematical Proof of Exponential Decay

Expanding the recursive formula:

$$V_1 = (1 - \beta) \cdot \theta_1$$

$$V_2 = \beta(1-\beta)\theta_1 + (1-\beta)\theta_2$$

$$V_3 = \beta^2(1-\beta)\theta_1 + \beta(1-\beta)\theta_2 + (1-\beta)\theta_3$$

$$V_4 = \beta^3(1-\beta)\theta_1 + \beta^2(1-\beta)\theta_2 + \beta(1-\beta)\theta_3 + (1-\beta)\theta_4$$

**General pattern for coefficients:**

| Data Point | Coefficient |
|------------|-------------|
| $\theta_1$ (oldest) | $\beta^3(1-\beta)$ |
| $\theta_2$ | $\beta^2(1-\beta)$ |
| $\theta_3$ | $\beta(1-\beta)$ |
| $\theta_4$ (newest) | $(1-\beta)$ |

Since $0 < \beta < 1$, higher powers of $\beta$ produce **smaller** values → **older points carry less weight**.

---

## 7. Effect of β (Smoothing Parameter)

### Intuition
EWMA approximately averages over the last $\frac{1}{1-\beta}$ data points.

| $\beta$ | Averaging Window | Behavior |
|---------|-----------------|----------|
| 0.98 | ~50 days | Very smooth, slow to react |
| 0.9 | ~10 days | **Sweet spot for deep learning** |
| 0.5 | ~2 days | More reactive, noisier |
| 0.1 | ~1.1 days | Very noisy, almost raw data |

### Visual Behavior

| High $\beta$ (e.g., 0.98) | Low $\beta$ (e.g., 0.1) |
|---------------------------|------------------------|
| Smooth curve | Noisy, follows data closely |
| Slow to adapt to changes | Reacts instantly to changes |
| Gives high weight to history | Gives high weight to current value |

> **In deep learning optimizers, $\beta = 0.9$ is the standard choice** — it balances smoothness with responsiveness.

### Understanding via Decomposition

$$V_t = \underbrace{\beta \cdot V_{t-1}}_{\text{history component}} + \underbrace{(1-\beta) \cdot \theta_t}_{\text{current component}}$$

- $\beta$ controls how much **past history** matters
- $(1-\beta)$ controls how much the **current observation** matters

---

## 8. Python Implementation

```python
import pandas as pd

# Load time series data
df = pd.read_csv('daily_climate.csv')

# Calculate EWMA
# alpha = 1 - beta (pandas uses alpha, not beta)
# For beta = 0.9, alpha = 0.1
df['ewma'] = df['mean_temp'].ewm(alpha=0.1).mean()

# Plot
import matplotlib.pyplot as plt
plt.plot(df['date'], df['mean_temp'], label='Raw Data')
plt.plot(df['date'], df['ewma'], label='EWMA', color='black')
plt.legend()
plt.show()
```

> **Note:** Pandas uses `alpha` where `alpha = 1 - β`. So for $\beta = 0.9$, set `alpha = 0.1`.

---

## 9. Key Takeaways

1. EWMA gives **exponentially decaying weights** to older observations
2. The parameter $\beta$ controls the trade-off between smoothness and responsiveness
3. Higher $\beta$ → smoother curve (more memory); lower $\beta$ → noisier curve (more reactive)
4. **Standard value in deep learning: $\beta = 0.9$**
5. EWMA is the mathematical foundation for Momentum, RMSProp, and Adam optimizers
6. Formula: $V_t = \beta \cdot V_{t-1} + (1-\beta) \cdot \theta_t$
