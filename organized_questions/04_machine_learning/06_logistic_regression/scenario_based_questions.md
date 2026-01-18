# Logistic Regression Interview Questions - Scenario-Based Questions

## Question 1

**Discuss the probability interpretations of logistic regression outputs.**

### Answer

**Definition:**
Logistic regression outputs calibrated probabilities representing the likelihood of an observation belonging to the positive class, derived from the sigmoid transformation of the linear combination of features.

**Mathematical Foundation:**
$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + ... + \beta_n X_n)}}$$

**Key Probability Interpretations:**

| Output | Interpretation |
|--------|----------------|
| P = 0.8 | 80% chance of being in positive class |
| P = 0.5 | Decision boundary (equal odds) |
| Odds = P/(1-P) | Ratio of success to failure |
| Log-odds = log(P/(1-P)) | Linear function of features |

**Calibration Considerations:**

1. **Well-Calibrated Outputs:**
   - If model predicts P=0.7 for 1000 samples, ~700 should be positive
   - Logistic regression is inherently well-calibrated under correct model specification
   
2. **Calibration Issues:**
   - Small datasets → unreliable probability estimates
   - Class imbalance → probabilities skewed toward majority class
   - Model misspecification → poor calibration

**Practical Code for Calibration Check:**
```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Get predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Plot calibration curve
fraction_positives, mean_predicted = calibration_curve(y_test, y_proba, n_bins=10)

plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.plot(mean_predicted, fraction_positives, 's-', label='Logistic Regression')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
```

**Interview Tip:**
Unlike many classifiers, logistic regression outputs true probabilities, not just scores. This makes it valuable when you need to communicate uncertainty (e.g., "70% chance of default") rather than just binary predictions.

---

## Question 2

**Discuss the consequences of multicollinearity in logistic regression.**

### Answer

**Definition:**
Multicollinearity occurs when two or more predictor variables are highly correlated, causing instability in coefficient estimation without necessarily affecting predictions.

**Consequences:**

| Issue | Description | Impact |
|-------|-------------|--------|
| **Inflated Standard Errors** | High variance in coefficient estimates | Wide confidence intervals |
| **Unstable Coefficients** | Small data changes → large coefficient swings | Unreliable interpretation |
| **Difficulty Isolating Effects** | Can't determine individual variable impact | Poor attribution |
| **Sign Flipping** | Coefficients may have wrong sign | Misleading interpretation |
| **Predictions Unaffected** | Model still predicts well | Inference compromised, not prediction |

**Detection Methods:**

```python
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Method 1: Correlation matrix
corr_matrix = X.corr()
high_corr = np.where(np.abs(corr_matrix) > 0.7)

# Method 2: Variance Inflation Factor (VIF)
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

# VIF interpretation:
# VIF = 1: No correlation
# VIF 1-5: Moderate (acceptable)
# VIF 5-10: High (concerning)
# VIF > 10: Severe (action needed)
```

**Solutions:**

1. **Remove Redundant Features:** Drop one of the correlated variables
2. **Feature Engineering:** Create composite features or ratios
3. **Regularization:** L1 (Lasso) or L2 (Ridge) automatically handles multicollinearity
4. **PCA:** Transform to uncorrelated principal components
5. **Domain Knowledge:** Choose most interpretable variable

**Interview Tip:**
If goal is prediction, multicollinearity isn't a problem. If goal is coefficient interpretation, it's critical to address. Always check VIF before interpreting coefficients.

---

## Question 3

**How would you assess the goodness-of-fit of a logistic regression model?**

### Answer

**Definition:**
Goodness-of-fit measures how well the model's predicted probabilities match the observed outcomes and whether the model adequately captures the underlying relationship.

**Key Metrics:**

| Metric | Formula/Description | Interpretation |
|--------|---------------------|----------------|
| **Log-Likelihood** | $\sum[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$ | Higher is better (less negative) |
| **Deviance** | $-2 \times \text{Log-Likelihood}$ | Lower is better |
| **McFadden's R²** | $1 - \frac{LL_{model}}{LL_{null}}$ | 0.2-0.4 is good |
| **AIC** | $-2LL + 2k$ | Lower is better (penalizes complexity) |
| **BIC** | $-2LL + k\log(n)$ | Lower is better (stronger penalty) |

**Statistical Tests:**

```python
import statsmodels.api as sm
from scipy import stats

# Fit model with statsmodels
X_const = sm.add_constant(X_train)
model = sm.Logit(y_train, X_const).fit()

# 1. Hosmer-Lemeshow Test (goodness-of-fit)
# Groups observations by predicted probability, compares expected vs observed
# H0: Model fits well; p > 0.05 means adequate fit

# 2. Likelihood Ratio Test (model significance)
ll_null = model.llnull
ll_model = model.llf
lr_stat = 2 * (ll_model - ll_null)
p_value = stats.chi2.sf(lr_stat, df=model.df_model)
print(f"LR Test p-value: {p_value:.4f}")

# 3. Pseudo R-squared values
print(f"McFadden R²: {model.prsquared:.4f}")

# 4. AIC/BIC comparison
print(f"AIC: {model.aic:.2f}")
print(f"BIC: {model.bic:.2f}")
```

**Diagnostic Plots:**

1. **Calibration Plot:** Predicted vs. observed probabilities
2. **Residual Analysis:** Deviance or Pearson residuals
3. **Influence Plots:** Identify high-leverage points
4. **ROC Curve:** Discrimination ability across thresholds

**Hosmer-Lemeshow Test Implementation:**
```python
def hosmer_lemeshow_test(y_true, y_pred_proba, n_groups=10):
    """Hosmer-Lemeshow goodness-of-fit test."""
    data = pd.DataFrame({'y': y_true, 'prob': y_pred_proba})
    data['group'] = pd.qcut(data['prob'], n_groups, duplicates='drop')
    
    grouped = data.groupby('group').agg(
        observed=('y', 'sum'),
        n=('y', 'count'),
        mean_prob=('prob', 'mean')
    )
    grouped['expected'] = grouped['n'] * grouped['mean_prob']
    
    hl_stat = np.sum((grouped['observed'] - grouped['expected'])**2 / 
                     (grouped['expected'] * (1 - grouped['mean_prob']/grouped['n'])))
    p_value = stats.chi2.sf(hl_stat, df=n_groups - 2)
    
    return hl_stat, p_value
```

**Interview Tip:**
No single metric tells the whole story. Use AIC/BIC for model comparison, pseudo-R² for overall fit, Hosmer-Lemeshow for calibration, and ROC-AUC for discrimination.

---

## Question 4

**Discuss the ROC curve and the AUC metric in the context of logistic regression.**

### Answer

**Definition:**
The ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs. False Positive Rate across all classification thresholds. AUC (Area Under Curve) summarizes discriminative ability in a single number.

**Mathematical Definitions:**

$$TPR = \frac{TP}{TP + FN} = \text{Sensitivity/Recall}$$
$$FPR = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

**AUC Interpretation:**

| AUC Value | Interpretation |
|-----------|----------------|
| 0.5 | Random guessing (no discrimination) |
| 0.5-0.6 | Poor |
| 0.6-0.7 | Fair |
| 0.7-0.8 | Good |
| 0.8-0.9 | Very Good |
| 0.9-1.0 | Excellent |
| 1.0 | Perfect separation |

**Probabilistic Interpretation:**
AUC = P(model ranks a random positive instance higher than a random negative instance)

**Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import cross_val_predict

# Get predictions
y_proba = model.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Find optimal threshold (Youden's J)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.3f}")
```

**When AUC is Appropriate:**
- ✅ Binary classification
- ✅ When ranking matters
- ✅ When threshold isn't fixed

**When AUC is NOT Appropriate:**
- ❌ Class imbalance (use AUC-PR instead)
- ❌ Multi-class (use one-vs-rest)
- ❌ When specific threshold performance matters
- ❌ When different costs for FP/FN exist

**AUC-PR for Imbalanced Data:**
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
```

**Interview Tip:**
AUC is threshold-invariant but can be misleading with severe class imbalance. For imbalanced problems, report both AUC-ROC and AUC-PR. Know Youden's J for optimal threshold selection.

---

## Question 5

**How would you approach diagnosing and addressing overfitting in a logistic regression model?**

### Answer

**Definition:**
Overfitting occurs when a model learns the training data too well, including noise, resulting in poor generalization to unseen data.

**Diagnosis Methods:**

| Indicator | Sign of Overfitting |
|-----------|---------------------|
| Train vs Test Gap | Training accuracy >> Test accuracy |
| CV Variance | High variance across CV folds |
| Large Coefficients | Extreme coefficient magnitudes |
| Perfect Separation | Model achieves 100% training accuracy |
| Learning Curves | Training curve flat, validation curve diverges |

**Detection Code:**
```python
from sklearn.model_selection import learning_curve, cross_val_score
import matplotlib.pyplot as plt

# 1. Compare train vs test performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train: {train_score:.3f}, Test: {test_score:.3f}")
print(f"Gap: {train_score - test_score:.3f}")  # > 0.1 is concerning

# 2. Cross-validation variance
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 3. Learning curves
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(), X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
```

**Solutions:**

| Technique | Implementation | When to Use |
|-----------|----------------|-------------|
| **L1 Regularization** | `penalty='l1'` | Too many features, need sparsity |
| **L2 Regularization** | `penalty='l2'` | General regularization, multicollinearity |
| **Elastic Net** | `penalty='elasticnet'` | Combine L1 and L2 benefits |
| **Reduce Features** | Feature selection | High dimensionality |
| **More Data** | Collect more samples | Small sample size |
| **Simplify Model** | Remove polynomial/interaction terms | Complex feature engineering |

**Regularization Implementation:**
```python
from sklearn.linear_model import LogisticRegressionCV

# Automatic regularization tuning
model_cv = LogisticRegressionCV(
    Cs=np.logspace(-4, 4, 20),
    cv=5,
    penalty='l2',
    scoring='roc_auc'
)
model_cv.fit(X_train, y_train)

print(f"Best C: {model_cv.C_[0]:.4f}")
print(f"Regularization strength: {1/model_cv.C_[0]:.4f}")
```

**Interview Tip:**
Logistic regression is a low-variance model and rarely overfits severely. If you see overfitting, suspect too many features or polynomial terms. L2 regularization with proper C tuning usually suffices.

---

## Question 6

**Discuss the use of polynomial and interaction terms in logistic regression.**

### Answer

**Definition:**
Polynomial and interaction terms extend logistic regression to capture non-linear relationships and combined effects between features that a simple linear model cannot represent.

**Types of Extended Terms:**

| Term Type | Example | Captures |
|-----------|---------|----------|
| **Polynomial** | $x^2$, $x^3$ | Non-linear effect of single variable |
| **Interaction** | $x_1 \times x_2$ | Combined effect of two variables |
| **Polynomial + Interaction** | $x_1^2 \times x_2$ | Complex non-linear interactions |

**Mathematical Model:**
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_1 x_2$$

**Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Full polynomial features
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(C=0.1, max_iter=1000))  # Lower C for regularization
])

# Interaction terms only (no squared terms)
interaction_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

# Manual specific interactions
import numpy as np

def add_interactions(X, cols_to_interact):
    """Add specific interaction terms."""
    X_new = X.copy()
    for i, col1 in enumerate(cols_to_interact):
        for col2 in cols_to_interact[i+1:]:
            X_new[f'{col1}_x_{col2}'] = X_new[col1] * X_new[col2]
    return X_new
```

**Feature Count Explosion:**

| Original Features | Degree 2 | Degree 3 |
|-------------------|----------|----------|
| 5 | 20 | 55 |
| 10 | 65 | 285 |
| 20 | 230 | 1770 |

**When to Use:**

✅ **Use Polynomial/Interactions When:**
- Clear non-linear relationship in EDA
- Decision boundary isn't linear
- Domain knowledge suggests combined effects
- Have sufficient data for added complexity

❌ **Avoid When:**
- Small dataset (risk of overfitting)
- High-dimensional data already
- Interpretability is paramount
- No theoretical basis for non-linearity

**Interview Tip:**
Always pair polynomial features with strong regularization (low C). Use `interaction_only=True` to reduce feature explosion. Consider tree-based models if you need many interactions.

---

## Question 7

**Discuss the implications of missing data on logistic regression models.**

### Answer

**Definition:**
Missing data affects logistic regression by reducing sample size, potentially introducing bias, and requiring careful handling strategies based on the missingness mechanism.

**Types of Missing Data:**

| Type | Description | Implications |
|------|-------------|--------------|
| **MCAR** | Missing Completely At Random | Safe to delete, no bias |
| **MAR** | Missing At Random (depends on observed data) | Imputation can work |
| **MNAR** | Missing Not At Random (depends on missing value itself) | Problematic, may need specialized methods |

**Impact on Logistic Regression:**

1. **Reduced Sample Size:** Complete-case analysis drops rows
2. **Biased Estimates:** If missingness is systematic
3. **Loss of Power:** Fewer observations → wider confidence intervals
4. **Feature Issues:** Some features may be mostly missing

**Handling Strategies:**

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. Complete Case Analysis (listwise deletion)
X_complete = X.dropna()  # Only if MCAR and few missing

# 2. Simple Imputation
imputer_mean = SimpleImputer(strategy='mean')  # For numeric
imputer_mode = SimpleImputer(strategy='most_frequent')  # For categorical

# 3. KNN Imputation (considers relationships between features)
imputer_knn = KNNImputer(n_neighbors=5)

# 4. Pipeline with imputation
numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

# 5. Missing Indicator (add flag for missingness)
from sklearn.impute import SimpleImputer, MissingIndicator

imputer_with_indicator = ColumnTransformer([
    ('imputer', SimpleImputer(strategy='mean'), numeric_features),
    ('indicator', MissingIndicator(), numeric_features)
])
```

**Advanced: Multiple Imputation:**
```python
# Using fancyimpute or sklearn's IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# MICE-like approach
imputer_iterative = IterativeImputer(
    max_iter=10,
    random_state=42
)
X_imputed = imputer_iterative.fit_transform(X)
```

**Decision Framework:**

| Missing % | Recommendation |
|-----------|----------------|
| < 5% | Mean/median imputation usually fine |
| 5-20% | Consider KNN or iterative imputation |
| 20-50% | Add missing indicator, multiple imputation |
| > 50% | Consider dropping feature or specialized methods |

**Interview Tip:**
Always investigate why data is missing before choosing a strategy. Adding a missing indicator feature can help the model learn from missingness patterns. For critical applications, use multiple imputation to quantify uncertainty.

---

## Question 8

**How would you apply logistic regression to a marketing campaign to predict customer conversion?**

### Answer

**Definition:**
Customer conversion prediction models the probability that a prospect will take a desired action (purchase, sign-up, etc.) based on demographic, behavioral, and campaign-related features.

**End-to-End Implementation:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Sample marketing data
data = pd.DataFrame({
    'age': np.random.randint(18, 65, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'website_visits': np.random.poisson(5, 1000),
    'email_opens': np.random.poisson(3, 1000),
    'time_on_site': np.random.exponential(10, 1000),
    'previous_purchases': np.random.poisson(1, 1000),
    'channel': np.random.choice(['Email', 'Social', 'Search', 'Direct'], 1000),
    'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 1000),
    'converted': np.random.binomial(1, 0.15, 1000)  # 15% conversion rate
})

# Feature engineering
data['engagement_score'] = data['website_visits'] + data['email_opens'] * 2
data['is_returning'] = (data['previous_purchases'] > 0).astype(int)

# Define features
numeric_features = ['age', 'income', 'website_visits', 'email_opens', 
                    'time_on_site', 'engagement_score']
categorical_features = ['channel', 'device', 'is_returning']

X = data[numeric_features + categorical_features]
y = data['converted']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Fit and evaluate
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print(classification_report(y_test, pipeline.predict(X_test)))

# Feature importance (coefficients)
feature_names = numeric_features + list(
    pipeline.named_steps['prep']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features)
)
coefficients = pipeline.named_steps['clf'].coef_[0]

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds_Ratio': np.exp(coefficients)
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop Conversion Drivers:")
print(coef_df.head(10))
```

**Business Application:**

1. **Targeting:** Score all prospects, focus on high-probability
2. **Budget Allocation:** Invest more in channels with high conversion impact
3. **Personalization:** Tailor messaging based on key features
4. **Threshold Optimization:** Balance acquisition cost vs. volume

```python
# Optimal threshold for business constraint
# If cost per contact = $5, value per conversion = $100
cost_per_contact = 5
value_per_conversion = 100

# Find threshold that maximizes profit
thresholds = np.linspace(0.05, 0.5, 50)
profits = []

for thresh in thresholds:
    predicted_convert = y_proba >= thresh
    n_contacted = predicted_convert.sum()
    actual_conversions = (predicted_convert & y_test).sum()
    
    profit = actual_conversions * value_per_conversion - n_contacted * cost_per_contact
    profits.append(profit)

optimal_threshold = thresholds[np.argmax(profits)]
print(f"Optimal threshold for profit: {optimal_threshold:.3f}")
```

**Interview Tip:**
Marketing models need interpretability (odds ratios for stakeholders) and calibrated probabilities (for ROI calculations). Always present results in business terms: "Customers from social media are 2.3x more likely to convert."

---

## Question 9

**Discuss how logistic regression can be used for credit scoring in the financial industry.**

### Answer

**Definition:**
Credit scoring uses logistic regression to predict the probability of loan default, enabling lenders to make risk-based decisions on loan approval and pricing.

**Why Logistic Regression for Credit Scoring:**

| Advantage | Business Value |
|-----------|---------------|
| **Interpretability** | Regulatory requirement (explainable decisions) |
| **Calibrated Probabilities** | Direct PD (Probability of Default) estimates |
| **Stable** | Consistent predictions, low variance |
| **Coefficients as Scorecards** | Easy to deploy as point-based system |

**Implementation:**

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import roc_auc_score

# Typical credit features
credit_data = pd.DataFrame({
    'income': np.random.lognormal(10.5, 0.5, 1000),
    'debt_to_income': np.random.uniform(0.1, 0.6, 1000),
    'credit_utilization': np.random.uniform(0.1, 0.9, 1000),
    'num_credit_lines': np.random.poisson(5, 1000),
    'delinquencies_2yr': np.random.poisson(0.5, 1000),
    'credit_age_months': np.random.exponential(60, 1000),
    'inquiries_6mo': np.random.poisson(1, 1000),
    'default': np.random.binomial(1, 0.08, 1000)  # 8% default rate
})

# Feature engineering for credit
credit_data['log_income'] = np.log1p(credit_data['income'])
credit_data['debt_burden'] = credit_data['debt_to_income'] * credit_data['credit_utilization']

features = ['log_income', 'debt_to_income', 'credit_utilization', 
            'num_credit_lines', 'delinquencies_2yr', 'credit_age_months',
            'inquiries_6mo', 'debt_burden']

X = credit_data[features]
y = credit_data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Probability of Default (PD)
pd_scores = model.predict_proba(X_test_scaled)[:, 1]
print(f"AUC-ROC (Gini): {roc_auc_score(y_test, pd_scores):.3f}")
print(f"Gini Coefficient: {2 * roc_auc_score(y_test, pd_scores) - 1:.3f}")
```

**Converting to Scorecard:**
```python
def create_scorecard(model, scaler, feature_names, base_score=600, pdo=20, base_odds=50):
    """
    Convert logistic regression to points-based scorecard.
    PDO = Points to Double the Odds
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    means = scaler.mean_
    stds = scaler.scale_
    
    # Points for each feature (per unit change)
    points_per_unit = -factor * coefficients / stds
    
    scorecard = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Points_per_unit': points_per_unit
    })
    
    return scorecard, offset, factor

scorecard, offset, factor = create_scorecard(model, scaler, features)
print("\nScorecard Points:")
print(scorecard)
```

**Regulatory Considerations:**

1. **Fair Lending:** Cannot use protected attributes (race, gender)
2. **Adverse Action:** Must explain why applicant was denied
3. **Model Documentation:** Full model development documentation required
4. **Validation:** Independent validation before deployment
5. **Monitoring:** Ongoing performance monitoring required

**Interview Tip:**
Credit scoring requires Gini coefficient (2*AUC - 1) for performance measurement. Models must be explainable for regulatory compliance. Always use scaled coefficients for scorecard conversion.

---

## Question 10

**How would you use logistic regression to analyze the impact of various factors on employee attrition?**

### Answer

**Definition:**
Employee attrition modeling predicts the probability of an employee leaving the organization based on demographic, job-related, and engagement factors, enabling proactive retention strategies.

**Implementation:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import statsmodels.api as sm

# Sample HR data
np.random.seed(42)
hr_data = pd.DataFrame({
    'age': np.random.randint(22, 60, 1000),
    'tenure_years': np.random.exponential(4, 1000),
    'salary': np.random.lognormal(10.8, 0.4, 1000),
    'satisfaction_score': np.random.uniform(1, 5, 1000),
    'performance_rating': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.05, 0.1, 0.5, 0.25, 0.1]),
    'overtime_hours': np.random.exponential(5, 1000),
    'promotions_last_5yr': np.random.poisson(0.5, 1000),
    'training_hours': np.random.exponential(20, 1000),
    'department': np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing', 'Operations'], 1000),
    'manager_rating': np.random.uniform(1, 5, 1000),
    'work_from_home': np.random.choice([0, 1], 1000, p=[0.6, 0.4]),
    'left': np.random.binomial(1, 0.18, 1000)  # 18% attrition
})

# Feature engineering
hr_data['salary_ratio'] = hr_data['salary'] / hr_data.groupby('department')['salary'].transform('median')
hr_data['tenure_bucket'] = pd.cut(hr_data['tenure_years'], bins=[0, 1, 3, 5, 100], 
                                   labels=['0-1yr', '1-3yr', '3-5yr', '5+yr'])
hr_data['engagement_index'] = (hr_data['satisfaction_score'] + hr_data['manager_rating']) / 2

numeric_features = ['age', 'tenure_years', 'salary_ratio', 'satisfaction_score',
                    'overtime_hours', 'promotions_last_5yr', 'engagement_index']
categorical_features = ['department', 'tenure_bucket', 'work_from_home']

X = hr_data[numeric_features + categorical_features]
y = hr_data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Build pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Fit and evaluate
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print(classification_report(y_test, pipeline.predict(X_test)))

# Feature importance with confidence intervals (statsmodels)
X_train_processed = pipeline.named_steps['prep'].fit_transform(X_train)
feature_names = numeric_features + list(
    pipeline.named_steps['prep']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features)
)

X_sm = sm.add_constant(X_train_processed)
model_sm = sm.Logit(y_train, X_sm).fit(disp=0)

# Results with statistical significance
results = pd.DataFrame({
    'Feature': ['Intercept'] + feature_names,
    'Coefficient': model_sm.params,
    'Odds_Ratio': np.exp(model_sm.params),
    'p_value': model_sm.pvalues,
    'Significant': model_sm.pvalues < 0.05
})
results = results.sort_values('p_value')

print("\nSignificant Attrition Factors:")
print(results[results['Significant'] == True][['Feature', 'Odds_Ratio', 'p_value']])
```

**Key Insights Interpretation:**

```python
def interpret_for_hr(results):
    """Generate HR-friendly interpretations."""
    significant = results[results['Significant'] == True].copy()
    
    interpretations = []
    for _, row in significant.iterrows():
        feature = row['Feature']
        odds_ratio = row['Odds_Ratio']
        
        if odds_ratio > 1:
            change = f"{(odds_ratio - 1) * 100:.1f}% higher"
        else:
            change = f"{(1 - odds_ratio) * 100:.1f}% lower"
        
        interpretations.append(f"• {feature}: {change} odds of leaving")
    
    return interpretations

print("\nActionable Insights:")
for insight in interpret_for_hr(results):
    print(insight)
```

**Business Application:**

1. **Risk Scoring:** Flag high-risk employees for proactive retention
2. **Intervention Targeting:** Focus on modifiable factors (satisfaction, overtime)
3. **Budget Allocation:** Prioritize retention spend on high-value employees
4. **Policy Changes:** Address systemic issues (overtime, promotion rates)

**Interview Tip:**
For HR analytics, focus on actionable factors HR can influence (satisfaction, training, overtime) rather than fixed demographics. Present results as "X% increase in overtime is associated with Y% higher attrition odds" for stakeholder understanding.

---
