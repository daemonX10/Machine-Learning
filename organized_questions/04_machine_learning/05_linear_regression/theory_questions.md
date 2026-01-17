# Linear Regression Interview Questions - Theory Questions

## Question 1: What is linear regression and how is it used in predictive modeling?

### Answer

**Definition:**
Linear Regression is a fundamental supervised learning algorithm used for regression tasks. It models the relationship between one or more independent variables (features) and a continuous dependent variable (target) by fitting a linear equation.

**The Model Equation:**

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$$

| Term | Description |
|------|-------------|
| $y$ | Target variable to be predicted |
| $x_1, x_2, ..., x_n$ | Input features |
| $\beta_0$ | Intercept (value of y when all x are zero) |
| $\beta_1, \beta_2, ...$ | Coefficients for each feature |
| $\epsilon$ | Irreducible error term |

**How It's Used in Predictive Modeling:**

| Phase | Description |
|-------|-------------|
| **Training** | Learn optimal coefficients (β) by minimizing sum of squared errors using Ordinary Least Squares (OLS) |
| **Prediction** | Apply learned equation to new data: $\hat{y} = \beta_0 + \beta_1x_1 + ...$ |
| **Inference** | Interpret coefficients to understand feature-target relationships |

**Key Applications:**
- Price prediction (houses, stocks)
- Sales forecasting
- Risk assessment
- Demand estimation
- Understanding variable relationships

---

## Question 2: Can you explain the difference between simple linear regression and multiple linear regression?

### Answer

**Comparison Table:**

| Aspect | Simple Linear Regression | Multiple Linear Regression |
|--------|--------------------------|---------------------------|
| **Predictors** | One input feature | Two or more features |
| **Equation** | $y = \beta_0 + \beta_1x + \epsilon$ | $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \epsilon$ |
| **Geometry** | 2D line | Multi-dimensional hyperplane |
| **Interpretation** | β₁ is direct relationship | βᵢ is relationship holding others constant |
| **Complexity** | Simple, easy to visualize | Deals with multicollinearity |

**Simple Linear Regression:**
- Models relationship between ONE feature and target
- Best-fitting straight line through data points
- Example: Predicting exam score based on hours studied

**Multiple Linear Regression:**
- Models relationship between MULTIPLE features and target
- Best-fitting hyperplane in multi-dimensional space
- Example: Predicting exam score based on hours studied, attendance, and prior GPA
- Key insight: Each coefficient represents effect while holding other features constant

---

## Question 3: What is the role of the intercept term in a linear regression model?

### Answer

**Role of the Intercept (β₀):**

| Function | Description |
|----------|-------------|
| **Baseline Value** | Predicted y when all features equal zero |
| **Line Position** | Allows regression line to shift up/down on y-axis |
| **Model Flexibility** | Without it, line must pass through origin (0,0) |

**Why Include the Intercept:**
- Without intercept, model is forced through origin
- This is rarely a valid assumption
- Omitting leads to biased coefficient estimates

**When Intercept is Meaningful:**
```
Example: Sales = β₀ + β₁ × Ad_Spend
- Intercept = predicted sales when ad spend is zero
- This is a meaningful baseline (organic sales)
```

**When Intercept is NOT Meaningful:**
```
Example: Weight = β₀ + β₁ × Height
- Intercept = predicted weight when height is zero
- Nonsensical extrapolation, but still mathematically necessary
```

**Rule:** Always include the intercept unless you have strong theoretical reasons that y must be zero when all x are zero.

---

## Question 4: What are the common metrics to evaluate a linear regression model's performance?

### Answer

**Evaluation Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error (units²) |
| **RMSE** | $\sqrt{MSE}$ | Average error in original units |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute error |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained (0-1) |
| **Adjusted R²** | $1 - \frac{(1-R²)(n-1)}{n-p-1}$ | R² penalized for extra features |

**Comparison:**

| Metric | Outlier Sensitivity | Units | Use Case |
|--------|---------------------|-------|----------|
| **RMSE** | High (squares errors) | Same as target | Most common, interpretable |
| **MAE** | Low (no squaring) | Same as target | When outliers shouldn't dominate |
| **R²** | N/A | Unitless (0-1) | Comparing models, explaining fit |
| **Adjusted R²** | N/A | Unitless | Comparing models with different features |

**Key Points:**
- R² always increases with more features (even useless ones)
- Adjusted R² only increases if new feature genuinely improves model
- Use Adjusted R² for comparing models with different numbers of features

---

## Question 5: Explain the concept of homoscedasticity. Why is it important?

### Answer

**Definition:**
Homoscedasticity means "constant variance" - the variance of error terms (residuals) is constant across all levels of input features.

**Visual Comparison:**

| Pattern | What You See | Implication |
|---------|-------------|-------------|
| **Homoscedasticity** | Random scatter, uniform spread | Assumption met ✓ |
| **Heteroscedasticity** | Fan/cone shape | Assumption violated ✗ |

**Why It's Important:**

| Consequence of Violation | Impact |
|--------------------------|--------|
| **OLS still unbiased** | Coefficient estimates are still correct on average |
| **OLS not efficient** | No longer BLUE (Best Linear Unbiased Estimator) |
| **Invalid standard errors** | Standard errors are biased |
| **Unreliable hypothesis tests** | t-tests, F-tests, p-values are wrong |
| **Wrong confidence intervals** | May incorrectly conclude significance |

**How to Detect:**
- Residual vs. Fitted plot (look for fan shape)
- Breusch-Pagan test
- White's test

**How to Address:**
1. **Transform target variable** (log, square root)
2. **Use Weighted Least Squares (WLS)**
3. **Use robust standard errors** (heteroscedasticity-consistent)

---

## Question 6: What is multicollinearity and how can it affect a regression model?

### Answer

**Definition:**
Multicollinearity occurs when two or more input features are highly correlated with each other (not with the target).

**Effects on the Model:**

| Problem | Description |
|---------|-------------|
| **Unstable coefficients** | Coefficients swing wildly with small data changes |
| **Large standard errors** | Variance of estimates is inflated |
| **Difficult interpretation** | Can't isolate individual feature effects |
| **Reduced significance** | High p-values even for important features |

**What It Does NOT Affect:**
- Overall predictive accuracy
- Unbiasedness of coefficients (still unbiased, just imprecise)

**Detection Methods:**

| Method | Threshold |
|--------|-----------|
| **Correlation matrix** | \|r\| > 0.8-0.9 |
| **VIF (Variance Inflation Factor)** | VIF > 5 or 10 |

**VIF Formula:**
$$VIF_i = \frac{1}{1 - R_i^2}$$
Where $R_i^2$ is the R² from regressing feature i against all other features.

**Solutions:**
1. Remove one of the correlated features
2. Combine correlated features (e.g., PCA)
3. Use Ridge Regression (shrinks correlated coefficients together)

---

## Question 7: Describe the steps involved in preprocessing data for linear regression analysis.

### Answer

**Preprocessing Pipeline:**

| Step | Action | Why |
|------|--------|-----|
| **1. Handle Missing Values** | Impute with median/mean or drop | LR can't handle NaN |
| **2. Handle Outliers** | Detect (Z-score, IQR) and treat | LR is sensitive to outliers |
| **3. Encode Categoricals** | One-hot encoding | LR needs numerical inputs |
| **4. Check Non-linearity** | Scatter plots, transform if needed | Linearity assumption |
| **5. Feature Scaling** | Standardization/Normalization | Essential for regularization |
| **6. Check Multicollinearity** | Correlation matrix, VIF | Stable coefficients |
| **7. Data Splitting** | Train/test split | Prevent data leakage |

**Code Example:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
    ]), categorical_cols)
])
```

**Key Rule:** Fit preprocessing only on training data, then transform test data.

---

## Question 8: Explain the concept of data splitting into training and test sets.

### Answer

**Purpose:**
Estimate how well the model generalizes to new, unseen data.

**The Split:**

| Set | Purpose | Typical Size |
|-----|---------|--------------|
| **Training** | Learn model parameters | 70-80% |
| **Test** | Final unbiased evaluation | 20-30% |
| **Validation** (optional) | Tune hyperparameters | Part of training |

**Why It's Critical:**
- Evaluating on training data gives optimistic, misleading results
- Model may have memorized training data (overfitting)
- Test set acts as proxy for real-world data

**Three-Way Split Workflow:**
```
1. Train on training set
2. Tune hyperparameters using validation set
3. Final evaluation on test set (only once!)
```

**Code:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Key Rule:** Never use test set for any decisions during model development.

---

## Question 9: What is cross-validation and how is it performed with linear regression?

### Answer

**Definition:**
Cross-validation is a resampling technique that provides more robust performance estimates by training and evaluating on different subsets of data.

**K-Fold Cross-Validation Process:**

```
1. Split data into k equal folds (e.g., k=5)
2. For each fold:
   - Use 1 fold as validation
   - Use remaining k-1 folds for training
   - Record performance score
3. Average all k scores for final estimate
```

**Visual (5-Fold):**
```
Fold 1: [TEST] [Train] [Train] [Train] [Train] → Score 1
Fold 2: [Train] [TEST] [Train] [Train] [Train] → Score 2
Fold 3: [Train] [Train] [TEST] [Train] [Train] → Score 3
Fold 4: [Train] [Train] [Train] [TEST] [Train] → Score 4
Fold 5: [Train] [Train] [Train] [Train] [TEST] → Score 5
                                                → Mean Score
```

**Use Cases in Linear Regression:**

| Use Case | Description |
|----------|-------------|
| **Performance Estimation** | Get reliable R²/RMSE estimate |
| **Hyperparameter Tuning** | Find optimal alpha for Ridge/Lasso |
| **Model Comparison** | Compare different feature sets |

**Code:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring='r2')
print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

---

## Question 10: Can you explain the concept of gradient descent and its importance in linear regression?

### Answer

**What is Gradient Descent?**
An iterative optimization algorithm to find the minimum of a function by following the negative gradient.

**The Analogy:**
Standing on a foggy mountain → feel the slope → step downhill → repeat until reaching valley floor.

**Update Rule:**
$$\beta_{new} = \beta_{old} - \alpha \cdot \frac{\partial Loss}{\partial \beta}$$

Where $\alpha$ is the learning rate (step size).

**Comparison: Normal Equation vs Gradient Descent**

| Aspect | Normal Equation | Gradient Descent |
|--------|-----------------|------------------|
| **Formula** | $\beta = (X^TX)^{-1}X^Ty$ | Iterative updates |
| **Complexity** | O(p³) | O(p) per iteration |
| **Small features** | Fast | Slower |
| **Large features** | Infeasible | Scalable |
| **Memory** | High | Low |

**Types of Gradient Descent:**

| Type | Batch Size | Speed | Stability |
|------|------------|-------|-----------|
| **Batch** | Entire dataset | Slow | Stable |
| **Stochastic (SGD)** | 1 sample | Fast | Noisy |
| **Mini-Batch** | Small batch (32-256) | Medium | Balanced |

**Importance:**
- Scalable to millions of features
- Foundation for training all neural networks
- Works when normal equation is computationally infeasible

---

## Question 11: What is ridge regression and how does it differ from standard linear regression?

### Answer

**Ridge Regression = Linear Regression + L2 Regularization**

**Loss Function Comparison:**

| Model | Loss Function |
|-------|---------------|
| **OLS** | $\sum(y_i - \hat{y}_i)^2$ |
| **Ridge** | $\sum(y_i - \hat{y}_i)^2 + \alpha \sum \beta_j^2$ |

The L2 penalty ($\alpha \sum \beta_j^2$) penalizes large coefficients.

**Key Differences:**

| Aspect | OLS | Ridge |
|--------|-----|-------|
| **Coefficients** | Can be large, unstable | Shrunk towards zero |
| **Overfitting** | Prone with many features | Reduced risk |
| **Multicollinearity** | Unstable coefficients | Robust and stable |
| **Bias-Variance** | Low bias, high variance | Higher bias, lower variance |
| **Feature Selection** | No | No (coefficients ≠ 0) |

**When to Use Ridge:**
- Many features with suspected multicollinearity
- All features believed to be relevant
- Prevent overfitting

**Code:**
```python
from sklearn.linear_model import Ridge, RidgeCV

# With cross-validation for alpha selection
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge.fit(X_train, y_train)
print(f"Best alpha: {ridge.alpha_}")
```

---

## Question 12: Explain the concept of Lasso regression and its benefits.

### Answer

**Lasso = Linear Regression + L1 Regularization**

**Loss Function:**
$$\sum(y_i - \hat{y}_i)^2 + \alpha \sum |\beta_j|$$

The L1 penalty ($\alpha \sum |\beta_j|$) uses absolute values of coefficients.

**Key Benefit: Automatic Feature Selection**

| Regularization | Coefficient Shrinkage | Feature Selection |
|----------------|----------------------|-------------------|
| **Ridge (L2)** | Shrinks towards zero | No (never exactly 0) |
| **Lasso (L1)** | Shrinks to exactly zero | Yes (sparse model) |

**Why Lasso Creates Sparse Models:**
- L1 penalty geometry creates "corners" where coefficients become exactly zero
- Less important features are eliminated completely

**Benefits:**

| Benefit | Description |
|---------|-------------|
| **Feature Selection** | Automatically identifies important features |
| **Sparse Models** | Fewer non-zero coefficients |
| **Interpretability** | Easier to explain with fewer features |
| **Reduced Overfitting** | Regularization reduces variance |

**When to Use:**
- High-dimensional data (many features)
- Suspect many features are irrelevant
- Want interpretable, sparse model

**Code:**
```python
from sklearn.linear_model import Lasso, LassoCV

lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5)
lasso.fit(X_train, y_train)
print(f"Non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
```

---

## Question 13: What is elastic net regression and in what cases would you use it?

### Answer

**Elastic Net = L1 + L2 Regularization Combined**

**Loss Function:**
$$\sum(y_i - \hat{y}_i)^2 + \alpha \left[ \rho \sum|\beta_j| + \frac{(1-\rho)}{2} \sum\beta_j^2 \right]$$

| Parameter | Description |
|-----------|-------------|
| **α (alpha)** | Overall regularization strength |
| **ρ (l1_ratio)** | Mix between L1 and L2 |
| ρ = 1 | Pure Lasso |
| ρ = 0 | Pure Ridge |
| 0 < ρ < 1 | Combination |

**When to Use Elastic Net:**

| Scenario | Why Elastic Net |
|----------|-----------------|
| **Correlated features** | Lasso selects one arbitrarily; Elastic Net selects groups |
| **p >> n** | Lasso can select at most n features; Elastic Net handles better |
| **Unsure Ridge vs Lasso** | Tune l1_ratio to find optimal mix |

**The "Grouping Effect":**
- With correlated features, Lasso picks one randomly
- Elastic Net keeps correlated features together
- More stable feature selection

**Code:**
```python
from sklearn.linear_model import ElasticNetCV

elastic_net = ElasticNetCV(
    alphas=[0.01, 0.1, 1.0],
    l1_ratio=[0.1, 0.5, 0.9],
    cv=5
)
elastic_net.fit(X_train, y_train)
print(f"Best alpha: {elastic_net.alpha_}, l1_ratio: {elastic_net.l1_ratio_}")
```

---

## Question 14: Explain the purpose of residual plots and how to interpret them.

### Answer

**What is a Residual Plot?**
Scatter plot of predicted values (x-axis) vs. residuals (y-axis).

**Residual:** $e_i = y_i - \hat{y}_i$ (actual - predicted)

**Interpretation Guide:**

| Pattern | What You See | Diagnosis | Action |
|---------|-------------|-----------|--------|
| **Random scatter** | No pattern, uniform spread | Good fit ✓ | None needed |
| **U-shape/Curve** | Systematic curve | Non-linearity | Add polynomial features |
| **Fan shape** | Spread increases/decreases | Heteroscedasticity | Transform target |
| **Clusters** | Distinct groups | Missing categorical | Add indicator variables |
| **Outliers** | Points far from 0 line | Influential points | Investigate/remove |

**Visual Examples:**

```
GOOD (Random):          BAD (Curved):           BAD (Fan):
    •  •                    •    •                  •
  •    •  •                  ••                    ••
----•--•-----            --••----••--          --•••-------
  •  •   •                ••    ••                 •••••
    •                    •        •                   ••••••
```

**Code:**
```python
import matplotlib.pyplot as plt

residuals = y_test - model.predict(X_test)
plt.scatter(model.predict(X_test), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

---

## Question 15: What is the adjusted R-squared, and why is it used?

### Answer

**The Problem with R²:**
R² always increases (or stays same) when adding features, even useless ones.

**Adjusted R² Formula:**
$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Where:
- n = number of samples
- p = number of features

**Key Difference:**

| Metric | Behavior with Added Features |
|--------|------------------------------|
| **R²** | Always increases or stays same |
| **Adjusted R²** | Only increases if feature adds genuine value |

**Example:**
```
Model A (5 features): R² = 0.85, Adj R² = 0.84
Model B (8 features): R² = 0.86, Adj R² = 0.82

→ Model A is better! Extra features in B don't add value.
```

**When to Use:**
- Comparing models with different numbers of features
- Avoiding overfitting to training data
- Model selection in multiple regression

**Interpretation:**
- Adj R² can be negative (very poor model)
- Higher is better
- Penalizes unnecessary complexity

---

## Question 16: What are leverage points and how do they affect a regression model?

### Answer

**Definition:**
A leverage point has extreme values in its input features (X values), far from the center of the data cloud.

**The Lever Analogy:**
Like a seesaw - a person far from the center has more influence on the tilt.

**Types of Leverage Points:**

| Type | Description | Effect |
|------|-------------|--------|
| **Good Leverage** | High leverage, follows trend (small residual) | Stabilizes model, reduces standard errors |
| **Bad Leverage** | High leverage, doesn't follow trend (large residual) | Dramatically distorts the regression line |

**Detection:**

| Method | Formula/Threshold |
|--------|-------------------|
| **Leverage statistic (hᵢᵢ)** | hᵢᵢ > 2(p+1)/n |
| **Visualization** | Points far on x-axis |

**Leverage vs. Outlier vs. Influential:**

| Term | Definition |
|------|------------|
| **Leverage point** | Extreme X values |
| **Outlier** | Extreme Y values (large residual) |
| **Influential point** | Changes model significantly if removed |

**Key Insight:**
A point is influential only if it has BOTH high leverage AND large residual. Use Cook's distance to measure overall influence.

---

## Question 17: Describe how you would detect and address outliers in your regression analysis.

### Answer

**Detection Methods:**

| Method | Type | Description |
|--------|------|-------------|
| **Residual plots** | Visual | Points far from y=0 line |
| **Box plots** | Visual | Univariate outliers |
| **Standardized residuals** | Statistical | \|residual\| > 3 |
| **Cook's distance** | Statistical | Dᵢ > 4/n or Dᵢ > 1 |
| **Leverage plot** | Visual | Combines leverage, residual, Cook's D |

**Treatment Decision Tree:**

```
Is it a data entry/measurement error?
├── Yes → Correct or remove
└── No → Is it a genuine extreme value?
    ├── Yes → Consider:
    │   ├── Transformation (log, sqrt)
    │   ├── Robust regression (Huber, RANSAC)
    │   └── Keep and report impact
    └── No → Remove with documentation
```

**Treatment Options:**

| Method | When to Use |
|--------|-------------|
| **Removal** | Clear error, with justification |
| **Transformation** | Skewed data (log reduces outlier impact) |
| **Winsorization** | Cap extreme values at percentiles |
| **Robust regression** | When outliers represent valid variation |

**Code:**
```python
from sklearn.linear_model import HuberRegressor

# Robust to outliers
robust_model = HuberRegressor()
robust_model.fit(X_train, y_train)
```

---

## Question 18: Explain the concept of Cook's distance.

### Answer

**Definition:**
Cook's distance measures the overall influence of a single observation on the entire regression model by combining leverage and residual size.

**What It Measures:**
How much all predicted values change when observation i is removed.

**Combines Two Factors:**

| Factor | Description |
|--------|-------------|
| **Leverage** | How extreme are the X values? |
| **Residual** | How far is the point from the regression line? |

**Key Insight:**
A point needs BOTH high leverage AND large residual to have high Cook's distance.

| Leverage | Residual | Cook's D | Interpretation |
|----------|----------|----------|----------------|
| High | Small | Low | Good leverage point |
| Low | Large | Low | Outlier with little influence |
| High | Large | **High** | **Influential - investigate!** |

**Thresholds:**

| Rule of Thumb | Threshold |
|---------------|-----------|
| Common | Dᵢ > 4/n |
| Conservative | Dᵢ > 1 |

**Code:**
```python
import statsmodels.api as sm

model = sm.OLS(y, sm.add_constant(X)).fit()
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]

# Identify influential points
influential = np.where(cooks_d > 4/len(y))[0]
```

---

## Question 19: Describe the variance inflation factor (VIF) and its significance.

### Answer

**Definition:**
VIF quantifies how much a coefficient's variance is inflated due to multicollinearity with other predictors.

**Formula:**
$$VIF_i = \frac{1}{1 - R_i^2}$$

Where $R_i^2$ is the R² from regressing feature i against all other features.

**Interpretation:**

| VIF Value | Interpretation |
|-----------|----------------|
| VIF = 1 | No correlation with other features (ideal) |
| 1 < VIF < 5 | Moderate, generally acceptable |
| 5 < VIF < 10 | High multicollinearity (concerning) |
| VIF > 10 | Severe multicollinearity (action needed) |

**Example:**
VIF = 10 means the coefficient's variance is 10× larger than if uncorrelated.

**Code:**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])]
print(vif_data.sort_values('VIF', ascending=False))
```

**Actions for High VIF:**
1. Remove one of the correlated features
2. Combine features (PCA)
3. Use Ridge regression

---

## Question 20: How does polynomial regression extend the linear regression model?

### Answer

**Concept:**
Polynomial regression fits non-linear relationships by creating polynomial features, then applying standard linear regression.

**Transformation:**
Original: $y = \beta_0 + \beta_1 x$

Polynomial (degree 2): $y = \beta_0 + \beta_1 x + \beta_2 x^2$

**Key Insight:**
The model is still LINEAR in the coefficients (β), just non-linear in the original feature (x).

**Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_model.fit(X, y)
```

**Trade-offs:**

| Degree | Bias | Variance | Fit |
|--------|------|----------|-----|
| Low (1-2) | High | Low | Underfitting |
| Medium (3-4) | Medium | Medium | Good |
| High (>5) | Low | High | Overfitting |

**Best Practice:**
- Use cross-validation to select optimal degree
- Consider regularization for high degrees
- Watch for overfitting at data edges

---

## Question 21: What are generalized linear models (GLMs), and how do they relate to linear regression?

### Answer

**GLMs extend linear regression to handle different types of target variables.**

**Three Components of a GLM:**

| Component | Description | Linear Regression |
|-----------|-------------|-------------------|
| **Random** | Distribution of y | Gaussian (Normal) |
| **Systematic** | Linear predictor η = β₀ + β₁x₁ + ... | Same |
| **Link Function** | Connects η to E(y) | Identity: μ = η |

**Common GLM Family:**

| Model | Target Type | Distribution | Link Function |
|-------|-------------|--------------|---------------|
| **Linear Regression** | Continuous | Gaussian | Identity |
| **Logistic Regression** | Binary (0/1) | Bernoulli | Logit |
| **Poisson Regression** | Count (0,1,2,...) | Poisson | Log |

**Logistic Regression Example:**
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + ...$$

**Poisson Regression Example:**
$$\log(\mu) = \beta_0 + \beta_1 x_1 + ...$$

**Key Takeaway:**
Linear regression is a special case of GLMs with Gaussian distribution and identity link.

---

## Question 22: Explain how quantile regression differs from ordinary least squares (OLS) regression.

### Answer

**Fundamental Difference:**

| Model | What It Models | Loss Function |
|-------|----------------|---------------|
| **OLS** | Conditional MEAN | Sum of squared errors |
| **Quantile** | Conditional QUANTILES | Asymmetrically weighted absolute errors |

**Why Use Quantile Regression:**

| Scenario | Why Quantile Regression |
|----------|------------------------|
| Outliers present | Median regression (50th quantile) is robust |
| Heteroscedasticity | Model different quantiles to capture spread |
| Risk analysis | 95th percentile for worst-case scenarios |

**Example:**
- Median regression: 50% of data above, 50% below
- 90th percentile: 90% of data below this line

**Comparison:**

| Aspect | OLS | Quantile |
|--------|-----|----------|
| **Output** | One line (mean) | Multiple lines (any quantile) |
| **Outlier sensitivity** | High | Low |
| **Assumptions** | Homoscedasticity | No variance assumptions |
| **Information** | Central tendency | Full distribution shape |

**Code:**
```python
from sklearn.linear_model import QuantileRegressor

# Median regression
median_model = QuantileRegressor(quantile=0.5)
median_model.fit(X, y)

# 90th percentile
upper_model = QuantileRegressor(quantile=0.9)
upper_model.fit(X, y)
```

---

## Question 23: What are mixed models, and where might you use them?

### Answer

**Definition:**
Mixed models (mixed-effects models) combine:
- **Fixed effects:** Standard regression coefficients (population-level)
- **Random effects:** Group-specific variations

**When Data is NOT Independent:**

| Scenario | Example | Random Effect |
|----------|---------|---------------|
| **Repeated measures** | Patient measurements over time | Patient-specific intercept |
| **Clustered data** | Students in schools | School-specific intercept |
| **Hierarchical data** | Employees in companies | Company-specific effects |

**Example: Medical Study**
- Fixed effect: Treatment effect (what we want to estimate)
- Random effect: Patient baseline (each patient has different starting blood pressure)

**Why Not Just OLS?**
- OLS assumes independent observations
- Observations from same patient/school are correlated
- Mixed models correctly account for this structure

**Model Structure:**
$$y_{ij} = \beta_0 + \beta_1 x_{ij} + u_j + \epsilon_{ij}$$

Where:
- $\beta_0, \beta_1$ = fixed effects
- $u_j$ = random effect for group j
- $\epsilon_{ij}$ = residual error

**Code (statsmodels):**
```python
import statsmodels.formula.api as smf

model = smf.mixedlm("score ~ treatment", data, groups=data["patient_id"])
result = model.fit()
```

---

## Question 24: Describe a situation where linear regression could be applied in the finance sector.

### Answer

**Application: Capital Asset Pricing Model (CAPM)**

**Goal:** Estimate a stock's Beta (systematic risk) and Alpha (excess return).

**The Model:**
$$R_{stock} - R_f = \alpha + \beta (R_{market} - R_f) + \epsilon$$

| Term | Meaning |
|------|---------|
| $R_{stock} - R_f$ | Stock's excess return (target) |
| $R_{market} - R_f$ | Market's excess return (feature) |
| $\beta$ | Stock's volatility relative to market |
| $\alpha$ | Stock's risk-adjusted outperformance |

**Interpretation of Beta:**

| Beta Value | Meaning |
|------------|---------|
| β = 1 | Stock moves with market |
| β > 1 | More volatile than market |
| β < 1 | Less volatile than market |
| β < 0 | Moves opposite to market |

**Implementation:**
```python
import yfinance as yf
from sklearn.linear_model import LinearRegression

# Get data
stock = yf.download('AAPL', start='2020-01-01')
market = yf.download('^GSPC', start='2020-01-01')

# Calculate excess returns
stock_returns = stock['Adj Close'].pct_change()
market_returns = market['Adj Close'].pct_change()

# Fit CAPM model
model = LinearRegression()
model.fit(market_returns.dropna().values.reshape(-1,1), 
          stock_returns.dropna().values)

print(f"Beta: {model.coef_[0]:.4f}")
print(f"Alpha: {model.intercept_:.4f}")
```

---

## Question 25: Explain how you might use regression analysis to assess the effect of marketing campaigns.

### Answer

**Application: Marketing Mix Modeling**

**Goal:** Quantify ROI of different marketing channels.

**Model Structure:**
$$\log(Sales) = \beta_0 + \beta_1\log(TV) + \beta_2\log(Radio) + \beta_3\log(Digital) + Controls + \epsilon$$

**Key Feature Engineering:**

| Feature | Why |
|---------|-----|
| **Adstock** | Captures carryover effect (ads impact lingers) |
| **Log transform** | Models diminishing returns |
| **Seasonality** | Control for seasonal patterns |
| **Price** | Control for pricing effects |

**Interpretation:**
- Coefficients represent elasticity
- β₁ = 0.1 means 10% increase in TV spend → 1% sales increase

**Business Applications:**
1. Calculate ROI per channel
2. Optimize budget allocation
3. Simulate "what-if" scenarios

```python
# Marketing Mix Model
model = LinearRegression()
model.fit(X[['log_tv_adstock', 'log_radio_adstock', 
             'log_digital_adstock', 'price', 'season']], 
          y_log_sales)

# Interpret coefficients as elasticities
for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef:.4f} elasticity")
```

---

## Question 26: Describe how linear regression models could be used in predicting real estate prices.

### Answer

**Application: Hedonic Pricing Model**

**Features to Include:**

| Category | Features |
|----------|----------|
| **Size** | SquareFootage, Bedrooms, Bathrooms |
| **Location** | Neighborhood (one-hot), Distance to amenities |
| **Quality** | OverallQuality, YearBuilt, YearRemodeled |
| **Amenities** | HasGarage, HasPool, HasFireplace |

**Common Transformations:**
- Log(Price) - handles skewness
- Log(SquareFootage) - linearizes relationship
- Polynomial features for non-linear effects

**Model Choice:**
```python
from sklearn.linear_model import LassoCV

# Lasso for automatic feature selection
model = LassoCV(cv=5)
model.fit(X_train_scaled, y_log_price)

# Interpret: value contribution of each feature
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
```

**Business Value:**
- Automated valuation models (Zillow Zestimate)
- Feature value estimation (how much is extra bathroom worth?)
- Investment analysis

---

## Question 27: Describe how you might use linear regression to optimize inventory levels in a supply chain context.

### Answer

**Application: Demand Forecasting**

**Goal:** Predict demand to optimize stock levels.

**Features:**

| Category | Examples |
|----------|----------|
| **Time** | DayOfWeek, Month, Holiday flag |
| **Lag** | Sales_last_week, Sales_last_month |
| **Pricing** | Current price, Promotion flag |
| **External** | Weather, Economic indicators |

**Model:**
$$Sales_t = \beta_0 + \beta_1 \cdot trend + \beta_2 \cdot Sales_{t-1} + \beta_3 \cdot Price + \epsilon$$

**Using Forecast for Inventory:**

```python
# 1. Get demand forecast
forecast = model.predict(X_next_week)

# 2. Calculate forecast uncertainty from residuals
forecast_std = np.std(y_train - model.predict(X_train))

# 3. Set reorder point
service_level = 0.95  # 95% service level
z_score = 1.65
safety_stock = z_score * forecast_std * np.sqrt(lead_time)

reorder_point = forecast * lead_time + safety_stock
```

**Output:**
- Point forecast for demand
- Uncertainty estimate for safety stock calculation
- Optimal reorder point

---

## Question 28: What are the latest research trends in regularized regression techniques?

### Answer

**Current Research Trends:**

| Trend | Description |
|-------|-------------|
| **Group Lasso** | Select/discard groups of features together |
| **Adaptive Lasso** | Different penalties for different coefficients |
| **Sparse Additive Models** | Non-linear functions with Lasso-style sparsity |
| **Debiased Lasso** | Valid inference after Lasso selection |
| **Causal Regularization** | Double ML for treatment effect estimation |

**1. Group Lasso:**
```python
# Select entire one-hot encoded groups together
# Not just individual dummies
```

**2. Adaptive Lasso:**
- First-stage: Get initial estimates (e.g., Ridge)
- Second-stage: Penalize inversely to initial estimate size
- Benefits: Better selection consistency

**3. Double/Debiased Machine Learning:**
- Use Lasso for nuisance parameter estimation
- Valid confidence intervals for causal effects
- Handles high-dimensional confounders

**4. Graph Regularization:**
- Incorporate feature relationship structure
- Connected features get similar coefficients

**Key Direction:**
Moving from "just prediction" to valid statistical inference in high-dimensional settings.

---

## Question 29: Describe a situation where logistic regression might be preferred over linear regression.

### Answer

**When Target is Binary (0/1):**

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|---------------------|
| **Target type** | Continuous | Binary categorical |
| **Output** | Any real number | Probability [0, 1] |
| **Example target** | Price ($) | Churn (Yes/No) |

**Why Linear Regression Fails for Binary:**

1. **Unbounded predictions:** Can predict -0.5 or 1.5 (meaningless for probability)
2. **Violates assumptions:** Errors aren't normally distributed
3. **No probability interpretation**

**Example: Customer Churn Prediction**

```python
# Wrong: Linear Regression
from sklearn.linear_model import LinearRegression
# Predictions might be < 0 or > 1

# Correct: Logistic Regression
from sklearn.linear_model import LogisticRegression
# Predictions are probabilities between 0 and 1
```

**Logistic Regression Output:**
- Probability of churn (0.85 = 85% chance)
- Classification via threshold (>0.5 → Churn)

---

## Question 30: Describe a scenario where you'd have to transition from a simple to a multiple linear regression model, and the considerations you'd have to make.

### Answer

**Scenario: Predicting Student Exam Scores**

**Phase 1: Simple Linear Regression**
```python
Score = β₀ + β₁ × Hours_Studied
# R² = 0.30 → Only 30% variance explained
# Underfitting - need more features!
```

**Phase 2: Multiple Linear Regression**
```python
Score = β₀ + β₁×Hours + β₂×GPA + β₃×Attendance + ...
```

**Key Considerations:**

| Consideration | Action |
|---------------|--------|
| **Feature Selection** | Use domain knowledge, not random features |
| **Multicollinearity** | Check VIF (Hours & Assignments may correlate) |
| **Coefficient Interpretation** | Now "holding other factors constant" |
| **Evaluation Metric** | Switch to Adjusted R² |
| **Overfitting Risk** | Consider regularization |

**Interpretation Change:**
- Simple: β₁ = total effect of studying
- Multiple: β₁ = effect of studying, controlling for GPA and attendance

**Validation:**
```python
from sklearn.model_selection import cross_val_score

# Compare models
simple_cv = cross_val_score(simple_model, X_simple, y, cv=5)
multiple_cv = cross_val_score(multiple_model, X_multiple, y, cv=5)

print(f"Simple: {simple_cv.mean():.3f}")
print(f"Multiple: {multiple_cv.mean():.3f}")
```

---

## Question 31: What are the mathematical foundations and assumptions underlying linear regression?

### Answer

**The Model:**
$$y = X\beta + \epsilon$$

**Gauss-Markov Assumptions (for BLUE):**

| Assumption | Mathematical Form | Meaning |
|------------|-------------------|---------|
| **Linearity** | E(y) = Xβ | Mean is linear in X |
| **Exogeneity** | E(ε\|X) = 0 | Errors uncorrelated with features |
| **Homoscedasticity** | Var(ε\|X) = σ²I | Constant error variance |
| **No autocorrelation** | Cov(εᵢ, εⱼ) = 0 | Errors are independent |
| **Full rank** | rank(X) = p | No perfect multicollinearity |

**Additional for Inference:**
- Normality: ε ~ N(0, σ²) for valid t-tests, F-tests

**BLUE Property:**
Under Gauss-Markov assumptions, OLS estimator is:
- **B**est (minimum variance)
- **L**inear (linear function of y)
- **U**nbiased (E(β̂) = β)
- **E**stimator

---

## Question 32: How do you derive the normal equation for linear regression and when is it preferred over gradient descent?

### Answer

**Derivation:**

1. Loss function: $L = (y - X\beta)^T(y - X\beta)$

2. Expand: $L = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta$

3. Derivative: $\frac{\partial L}{\partial \beta} = -2X^Ty + 2X^TX\beta$

4. Set to zero: $X^TX\beta = X^Ty$

5. **Normal Equation:** $\beta = (X^TX)^{-1}X^Ty$

**When to Use Each:**

| Criteria | Normal Equation | Gradient Descent |
|----------|-----------------|------------------|
| **Features (p)** | p < 10,000 | p large |
| **Samples (n)** | Any | Any |
| **Complexity** | O(p³) | O(iterations × p) |
| **Memory** | High (store X^TX) | Low |
| **Convergence** | Exact solution | Iterative |

**Rule of Thumb:**
- p < 10,000: Use Normal Equation (faster)
- p > 10,000 or sparse: Use Gradient Descent (scalable)

---

## Question 33: What is the difference between ordinary least squares (OLS) and other regression estimation methods?

### Answer

**Comparison of Estimation Methods:**

| Method | Minimizes | Assumptions | Use Case |
|--------|-----------|-------------|----------|
| **OLS** | Sum of squared errors | Gauss-Markov | Standard regression |
| **WLS** | Weighted squared errors | Known weights | Heteroscedasticity |
| **GLS** | Generalized squared errors | Known covariance | Correlated errors |
| **MLE** | Negative log-likelihood | Distributional | Known error distribution |
| **LAD** | Sum of absolute errors | Fewer | Outlier robustness |
| **Bayesian** | Posterior | Prior beliefs | Uncertainty quantification |

**OLS Properties:**
- BLUE under Gauss-Markov
- Equivalent to MLE under normality
- Simple and interpretable

**When OLS is NOT Best:**
| Problem | Better Alternative |
|---------|-------------------|
| Heteroscedasticity | WLS |
| Autocorrelated errors | GLS |
| Heavy outliers | LAD (quantile regression) |
| Prior information | Bayesian regression |
| High-dimensional | Regularized (Ridge, Lasso) |

---
