
## Scenario / case questions

71. How would you assess which factors contribute most to sales in a supermarket chain?

**Answer:**

**Step-by-Step Approach:**

**1. Understand the Data**
- Target variable: Sales (continuous)
- Features: Product type, location, promotions, price, seasonality, customer demographics
- Check data types, missing values, duplicates

**2. Exploratory Data Analysis (EDA)**
```python
# Distribution of sales
df['Sales'].hist()

# Correlation with numeric features
df.corr()['Sales'].sort_values(ascending=False)

# Sales by category
df.groupby('Category')['Sales'].mean().plot(kind='bar')
```

**3. Feature Engineering**
- Create time-based features (day of week, month, holiday flag)
- Encode categorical variables (one-hot or label encoding)
- Create interaction features (price × promotion)

**4. Statistical Analysis**
- **ANOVA**: Compare sales across product categories
- **Correlation analysis**: Identify linear relationships
- **Chi-squared test**: Association between categorical features and sales tiers

**5. Build Predictive Model**
```python
from sklearn.ensemble import RandomForestRegressor

# Fit model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

**6. Key Factors to Investigate:**
- Price elasticity
- Promotional effectiveness
- Store location impact
- Seasonal patterns
- Product placement (shelf position)
- Time of purchase

**7. Validation**
- Cross-validation for model performance
- Business validation with domain experts

---

72. Describe your approach to determining whether a new drug is effective based on clinical trial data.

**Answer:**

**Scenario:** Testing if a new blood pressure medication reduces BP more than placebo.

**Step-by-Step Approach:**

**1. Study Design**
- **Type**: Randomized Controlled Trial (RCT)
- **Groups**: Treatment (drug) vs Control (placebo)
- **Outcome**: Change in systolic BP after 8 weeks
- **Sample size**: Calculate using power analysis (α=0.05, power=0.80, expected effect size)

**2. Hypotheses**
- H₀: μ_drug = μ_placebo (no difference)
- H₁: μ_drug ≠ μ_placebo (drug is effective)

**3. Data Collection**
```python
# Sample data structure
treatment_group = [change_in_bp for patients on drug]
control_group = [change_in_bp for patients on placebo]
```

**4. Statistical Analysis**

**Primary Analysis: Two-sample t-test**
```python
from scipy import stats

# Independent samples t-test
t_stat, p_value = stats.ttest_ind(treatment_group, control_group)

# Effect size (Cohen's d)
cohens_d = (np.mean(treatment_group) - np.mean(control_group)) / pooled_std
```

**5. Interpretation**
- If p < 0.05: Drug shows statistically significant effect
- Cohen's d interpretation: small (0.2), medium (0.5), large (0.8)
- 95% Confidence Interval for mean difference

**6. Additional Analyses**
- **ANCOVA**: Adjust for baseline BP, age, gender
- **ITT vs Per-Protocol**: Intent-to-treat analysis
- **Subgroup analysis**: Effect by age, gender, severity

**7. Report Results**
```
Results: The treatment group showed a mean reduction of 15 mmHg 
(SD=8) compared to 5 mmHg (SD=7) in placebo group.
t(98) = 7.07, p < 0.001, Cohen's d = 1.33 (large effect)
95% CI for difference: [7.2, 12.8] mmHg
Conclusion: Drug is significantly more effective than placebo.
```

**8. Considerations**
- Check normality assumption (Shapiro-Wilk)
- Use non-parametric test (Mann-Whitney U) if violated
- Account for multiple testing if multiple outcomes
- Report both statistical and clinical significance

---

73. Explain how you would evaluate the success of an online advertising campaign with statistical analysis.

**Answer:**

**Framework: A/B Testing + Multi-metric Evaluation**

**1. Define Objectives & KPIs**
- **Primary KPI**: Conversion rate, Revenue, ROI
- **Secondary KPIs**: Click-through rate (CTR), Engagement, Brand awareness
- **Time frame**: Campaign duration + attribution window

**2. Experiment Design**
```
Control Group (A): No ad / Old ad
Treatment Group (B): New advertising campaign
Split: Random 50-50 assignment
```

**3. Data Collection**
```python
# Metrics to track
data = {
    'group': ['A', 'B'],
    'users': [10000, 10000],
    'conversions': [150, 200],
    'revenue': [15000, 22000],
    'cost': [0, 5000]
}
```

**4. Statistical Testing**

**A. Conversion Rate (Chi-squared test)**
```python
from scipy.stats import chi2_contingency

# Contingency table
observed = [[150, 9850],    # Control: converted, not converted
            [200, 9800]]    # Treatment: converted, not converted

chi2, p_value, dof, expected = chi2_contingency(observed)
```

**B. Revenue per User (t-test)**
```python
t_stat, p_value = stats.ttest_ind(revenue_per_user_A, revenue_per_user_B)
```

**5. Calculate Business Metrics**

```python
# Conversion lift
lift = (conversion_B - conversion_A) / conversion_A * 100

# Incremental revenue
incremental_revenue = revenue_B - revenue_A

# ROI
roi = (incremental_revenue - campaign_cost) / campaign_cost * 100
```

**6. Attribution Analysis**
- First-touch vs Last-touch vs Multi-touch attribution
- Time-decay model for longer campaigns
- Control for seasonal effects (year-over-year comparison)

**7. Advanced Analysis**
- **Difference-in-Differences**: Compare pre/post across groups
- **Propensity Score Matching**: For non-randomized campaigns
- **Regression analysis**: Control for confounders

**8. Reporting Framework**
```
Campaign Performance Summary:
- Conversion Rate: +33% lift (1.5% → 2.0%)
- p-value: 0.002 (statistically significant)
- Incremental Revenue: $7,000
- Campaign Cost: $5,000
- ROI: 40%
Recommendation: Scale the campaign
```

---

74. Discuss how you would use time series analysis to forecast stock prices.

**Answer:**

**Comprehensive Time Series Forecasting Pipeline:**

**1. Data Preparation**
```python
import pandas as pd

# Load and preprocess
df = pd.read_csv('stock_prices.csv', parse_dates=['Date'], index_col='Date')
df = df.asfreq('B')  # Business day frequency
df['Price'].fillna(method='ffill', inplace=True)  # Handle missing dates
```

**2. Exploratory Analysis**
```python
# Visualize
df['Price'].plot(title='Stock Price Over Time')

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['Price'], model='multiplicative')
decomposition.plot()
```

**3. Stationarity Check**
```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Price'])
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
# If p > 0.05: Non-stationary, needs differencing
```

**4. Make Stationary**
```python
# Differencing
df['Returns'] = df['Price'].pct_change()  # Log returns often better
# or
df['Diff'] = df['Price'].diff()
```

**5. Identify Model Parameters (ACF/PACF)**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df['Returns'].dropna(), lags=40)
plot_pacf(df['Returns'].dropna(), lags=40)
# ACF cuts off → MA(q)
# PACF cuts off → AR(p)
```

**6. Model Selection & Fitting**

**Option A: ARIMA**
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df['Price'], order=(p, d, q))
fitted = model.fit()
print(fitted.summary())
```

**Option B: Auto ARIMA**
```python
from pmdarima import auto_arima

model = auto_arima(df['Price'], seasonal=False, trace=True)
```

**Option C: GARCH (for volatility)**
```python
from arch import arch_model

model = arch_model(df['Returns'], vol='Garch', p=1, q=1)
fitted = model.fit()
```

**7. Model Validation**
```python
# Train-test split (chronological)
train = df['Price'][:-30]
test = df['Price'][-30:]

# Forecast
forecast = fitted.forecast(steps=30)

# Evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100
```

**8. Generate Forecasts**
```python
# Future predictions
future_forecast = fitted.forecast(steps=30)
confidence_interval = fitted.get_forecast(steps=30).conf_int()

# Plot
plt.plot(train, label='Training')
plt.plot(test, label='Actual')
plt.plot(future_forecast, label='Forecast')
plt.fill_between(range, ci_lower, ci_upper, alpha=0.3)
plt.legend()
```

**9. Important Considerations**
- Stock prices are notoriously hard to predict (Efficient Market Hypothesis)
- Consider external factors (news, earnings, market sentiment)
- Use ensemble methods combining multiple models
- Always include uncertainty estimates (confidence intervals)
- Backtest on out-of-sample data

---

75. How would you design a statistical study to understand customer churn in a subscription-based business?

**Answer:**

**End-to-End Customer Churn Analysis Framework:**

**1. Problem Definition**
- **Target**: Churn (1 = customer left, 0 = retained)
- **Goal**: Identify churn drivers + predict at-risk customers
- **Timeline**: Define churn window (e.g., no activity for 30 days)

**2. Data Collection & Preparation**
```python
# Key features to collect
features = {
    'Demographics': ['age', 'gender', 'location'],
    'Behavior': ['login_frequency', 'feature_usage', 'support_tickets'],
    'Financial': ['monthly_spend', 'payment_method', 'late_payments'],
    'Engagement': ['tenure', 'last_activity_days', 'NPS_score'],
    'Target': ['churned']
}

# Handle imbalanced classes
print(df['churned'].value_counts(normalize=True))
# Typically: 70-90% retained, 10-30% churned
```

**3. Exploratory Data Analysis**
```python
# Churn rate by segment
df.groupby('plan_type')['churned'].mean().plot(kind='bar')

# Correlation with churn
df.corr()['churned'].sort_values()

# Statistical tests for each feature
for col in df.select_dtypes(include=['float', 'int']).columns:
    churned = df[df['churned']==1][col]
    retained = df[df['churned']==0][col]
    t_stat, p_val = stats.ttest_ind(churned, retained)
    print(f"{col}: p-value = {p_val:.4f}")
```

**4. Statistical Analysis**

**A. Hypothesis Testing**
```python
# Chi-squared test: Is churn independent of plan type?
contingency = pd.crosstab(df['plan_type'], df['churned'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
```

**B. Survival Analysis**
```python
from lifelines import KaplanMeierFitter, CoxPHFitter

# Kaplan-Meier: Survival curves
kmf = KaplanMeierFitter()
kmf.fit(df['tenure'], event_observed=df['churned'])
kmf.plot_survival_function()

# Cox Regression: Hazard ratios
cph = CoxPHFitter()
cph.fit(df[['tenure', 'churned', 'age', 'monthly_spend']], 
        duration_col='tenure', event_col='churned')
cph.print_summary()
```

**5. Predictive Modeling**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
```

**6. Feature Importance Analysis**
```python
# Top churn predictors
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Typical top factors:
# 1. Days since last activity
# 2. Support ticket frequency
# 3. Usage decline trend
# 4. Contract type
# 5. Monthly charges
```

**7. Actionable Insights**

**Risk Scoring**
```python
# Score all customers
df['churn_probability'] = model.predict_proba(X)[:, 1]
df['risk_tier'] = pd.cut(df['churn_probability'], 
                         bins=[0, 0.3, 0.7, 1.0],
                         labels=['Low', 'Medium', 'High'])
```

**Intervention Strategy**
| Risk Tier | Action |
|-----------|--------|
| High (>70%) | Personal outreach, special offers |
| Medium (30-70%) | Targeted email campaigns |
| Low (<30%) | Standard engagement |

**8. Key Metrics to Track**
- Churn rate trend
- Customer Lifetime Value (CLV)
- Net Promoter Score (NPS)
- Retention rate by cohort

---

