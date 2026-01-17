# Time Series Interview Questions

Comprehensive interview preparation guide for time series analysis and forecasting.

## Question Types Available:

| Type | Count | Description |
|------|-------|-------------|
| **Theory** | 24 | Core concepts: stationarity, ARIMA, GARCH, cointegration |
| **General** | 10 | Broader topics: ML applications, evaluation metrics |
| **Scenario_Based** | 10 | Real-world applications: demand forecasting, anomaly detection |
| **Coding** | 6 | Implementation: SES, ARIMA, decomposition, ACF/PACF |

**Total Questions**: 50

## Key Topics Covered:

### Fundamentals
- Time series components (Trend, Seasonality, Cyclicity, Noise)
- Stationarity and ADF test
- Autocorrelation (ACF) and Partial Autocorrelation (PACF)
- Differencing

### Classical Models
- AR, MA, ARMA, ARIMA, SARIMA
- Exponential Smoothing (SES, Holt, Holt-Winters)
- Box-Jenkins methodology

### Advanced Models
- ARCH/GARCH for volatility
- VAR for multivariate analysis
- Cointegration and ECM
- LSTM and deep learning

### Practical Skills
- Model selection (AIC/BIC)
- Time series cross-validation
- Evaluation metrics (MAE, RMSE, MAPE, MASE)
- Feature engineering for ML models

## Files:

- [theory_questions.md](theory_questions.md) - Core theoretical concepts
- [general_questions.md](general_questions.md) - General/applied questions
- [scenario_based_questions.md](scenario_based_questions.md) - Real-world scenarios
- [coding_questions.md](coding_questions.md) - Python implementations

## Quick Reference - Key Formulas:

| Model | Key Equation |
|-------|-------------|
| AR(p) | $Y_t = c + \phi_1 Y_{t-1} + ... + \phi_p Y_{t-p} + \epsilon_t$ |
| MA(q) | $Y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}$ |
| ARIMA(p,d,q) | Differenced series modeled as ARMA |
| GARCH(1,1) | $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$ |

## Interview Tips:

1. **Always plot the data first** - Visual inspection is crucial
2. **Check stationarity** - ADF test before modeling
3. **Start simple** - Naive baseline → Exponential Smoothing → ARIMA → ML
4. **Never shuffle time series** - Temporal order is sacred
5. **Residuals should be white noise** - If not, model is incomplete
