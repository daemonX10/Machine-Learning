# Scikit-Learn Interview Questions - Theory Questions

## Question 1

**What is Scikit-Learn, and why is it popular in Machine Learning?**

### Definition
Scikit-learn (sklearn) is an open-source Python library for traditional machine learning. Built on NumPy, SciPy, and Matplotlib, it provides tools for classification, regression, clustering, dimensionality reduction, model selection, and preprocessing.

### Why It's Popular

| Feature | Description |
|---------|-------------|
| **Consistent API** | All models use `fit()`, `predict()`, `transform()` |
| **Comprehensive** | Classification, regression, clustering, preprocessing |
| **Well-Documented** | Excellent documentation with examples |
| **Integration** | Works with NumPy, Pandas, Matplotlib |
| **BSD License** | Free for commercial use |

### Core API Pattern
```python
from sklearn.ensemble import RandomForestClassifier

# All models follow the same pattern
model = RandomForestClassifier()
model.fit(X_train, y_train)      # Train
predictions = model.predict(X_test)  # Predict
score = model.score(X_test, y_test)  # Evaluate
```

---

## Question 2

**Explain the design principles behind Scikit-Learn's API.**

### Five Core Principles

| Principle | Description |
|-----------|-------------|
| **Consistency** | Unified interface: `fit()`, `predict()`, `transform()` |
| **Inspection** | Learned parameters accessible as `model.attribute_` |
| **Non-proliferation** | Uses NumPy arrays, not custom data types |
| **Composition** | Pipeline chains transformers and estimators |
| **Sensible Defaults** | Works well out-of-the-box |

### Code Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Switching models is trivial due to consistent API
model1 = LogisticRegression()
model2 = RandomForestClassifier()

# Both use identical syntax
for model in [model1, model2]:
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    
# Inspection: access learned parameters
print(model1.coef_)           # Coefficients
print(model2.feature_importances_)  # Feature importances
```

---

## Question 3

**Describe the role of transformers and estimators in Scikit-Learn.**

### Definitions

| Type | Role | Key Methods |
|------|------|-------------|
| **Estimator** | Any object that learns from data | `fit()` |
| **Transformer** | Modifies/transforms data | `fit()`, `transform()`, `fit_transform()` |
| **Predictor** | Makes predictions | `predict()`, `predict_proba()` |

### Examples
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Transformer: learns parameters, transforms data
scaler = StandardScaler()
scaler.fit(X_train)              # Learn mean, std
X_scaled = scaler.transform(X_train)  # Transform data

# Predictor: learns parameters, makes predictions
model = LogisticRegression()
model.fit(X_scaled, y_train)     # Learn weights
predictions = model.predict(X_test)  # Predict
```

### Relationship
- All transformers are estimators
- All predictors are estimators
- Not all estimators are transformers or predictors

---

## Question 4

**What is the typical workflow for building a predictive model?**

### 8-Step Workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create Preprocessing Pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 4. Create Full Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Train Model
pipeline.fit(X_train, y_train)

# 6. Make Predictions
y_pred = pipeline.predict(X_test)

# 7. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 8. (Optional) Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'classifier__n_estimators': [50, 100, 200]}
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)
```

---

## Question 5

**Explain the concept of a Pipeline in Scikit-Learn.**

### Definition
A Pipeline chains multiple transformers and a final estimator into a single object. It ensures proper preprocessing during cross-validation and prevents data leakage.

### Why Pipelines are Essential

| Benefit | Description |
|---------|-------------|
| **Prevents Data Leakage** | Transformers fit only on training data |
| **Simplifies Code** | One `fit()` call instead of many |
| **Enables Tuning** | Tune all hyperparameters together |
| **Reproducible** | Easy to save and deploy |

### Code Example
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Tune hyperparameters of all steps
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

### Data Leakage Prevention
```python
# WRONG: Fit on all data, then split (LEAKAGE!)
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)

# CORRECT: Pipeline handles this automatically
pipe.fit(X_train, y_train)  # Scaler fits only on X_train
pipe.predict(X_test)        # Scaler transforms X_test
```

---

## Question 6

**What are the main categories of algorithms in Scikit-Learn?**

### Algorithm Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| **Classification** | Predict discrete labels | LogisticRegression, SVC, RandomForest |
| **Regression** | Predict continuous values | LinearRegression, Ridge, Lasso, SVR |
| **Clustering** | Group similar data | KMeans, DBSCAN, Hierarchical |
| **Dimensionality Reduction** | Reduce features | PCA, t-SNE, LDA |
| **Preprocessing** | Transform data | StandardScaler, OneHotEncoder |
| **Model Selection** | Evaluate models | GridSearchCV, cross_val_score |
| **Ensemble** | Combine models | RandomForest, GradientBoosting |

---

## Question 7

**What strategies does Scikit-Learn provide for imbalanced datasets?**

### Strategies

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Class Weights** | Penalize minority errors more | `class_weight='balanced'` |
| **Undersampling** | Reduce majority class | `imbalanced-learn` library |
| **Oversampling** | Increase minority class | `RandomOverSampler` |
| **SMOTE** | Synthetic minority samples | `SMOTE()` from imblearn |

### Code Example
```python
from sklearn.ensemble import RandomForestClassifier

# Method 1: Class weights (built-in)
model = RandomForestClassifier(class_weight='balanced')

# Method 2: SMOTE (requires imbalanced-learn)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier())
])
```

---

## Question 8

**Describe the use of ColumnTransformer in Scikit-Learn.**

### Definition
ColumnTransformer applies different transformations to different columns. Essential for datasets with mixed types (numeric + categorical).

### Code Example
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Define column groups
numeric_features = ['age', 'salary']
categorical_features = ['city', 'gender']

# Create transformers for each type
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

---

## Question 9

**How does cross-validation work in Scikit-Learn?**

### Definition
Cross-validation splits data into k folds, trains on k-1, validates on 1, and repeats k times. Provides robust performance estimate.

### Types

| Type | Use Case |
|------|----------|
| `KFold` | Regression or balanced classification |
| `StratifiedKFold` | Imbalanced classification |
| `LeaveOneOut` | Small datasets |
| `TimeSeriesSplit` | Time series data |

### Code Example
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

model = RandomForestClassifier()

# Simple cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Stratified K-Fold (maintains class proportions)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)

# Multiple metrics
from sklearn.model_selection import cross_validate
results = cross_validate(model, X, y, cv=5, 
                         scoring=['accuracy', 'f1', 'roc_auc'])
```

---

## Question 10

**Explain GridSearchCV vs RandomizedSearchCV.**

### Comparison

| Aspect | GridSearchCV | RandomizedSearchCV |
|--------|-------------|-------------------|
| Search | Exhaustive (all combinations) | Random sampling |
| Speed | Slow for large grids | Faster |
| Use Case | Small parameter space | Large parameter space |

### Code Example
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# GridSearchCV: tests ALL combinations
grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}
grid_search = GridSearchCV(model, grid, cv=5)  # 9 combinations

# RandomizedSearchCV: samples N combinations
from scipy.stats import randint
random_grid = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50)
}
random_search = RandomizedSearchCV(model, random_grid, 
                                   n_iter=20, cv=5)  # 20 random combos

# Usage is identical
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

---

## Question 11

**What evaluation metrics does Scikit-Learn provide?**

### Classification Metrics

| Metric | Use Case | Function |
|--------|----------|----------|
| Accuracy | Balanced classes | `accuracy_score()` |
| Precision | Minimize false positives | `precision_score()` |
| Recall | Minimize false negatives | `recall_score()` |
| F1-Score | Balance precision/recall | `f1_score()` |
| ROC-AUC | Overall performance | `roc_auc_score()` |
| Confusion Matrix | Detailed breakdown | `confusion_matrix()` |

### Regression Metrics

| Metric | Description | Function |
|--------|-------------|----------|
| MSE | Mean Squared Error | `mean_squared_error()` |
| RMSE | Root MSE | `sqrt(mean_squared_error())` |
| MAE | Mean Absolute Error | `mean_absolute_error()` |
| R² | Explained variance | `r2_score()` |

### Code Example
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Quick summary
print(classification_report(y_test, y_pred))

# Individual metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")
```

