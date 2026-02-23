# Scikit-Learn Interview Questions - Theory Questions

## Question 1

**What is Scikit-Learn , and why is it popular in the field of Machine Learning ?**

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

**Explain the design principles behind Scikit-Learn’s API**

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

**What is the typical workflow for building a predictive model using Scikit-Learn ?**

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

**What are some of the main categories of algorithms included in Scikit-Learn ?**

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



---

## Question 12

**Describe how a decision tree is constructed in Scikit-Learn.**

### Definition
A decision tree recursively splits data by selecting the feature and threshold that best separates target classes (classification) or minimizes prediction error (regression). Scikit-Learn implements the CART (Classification and Regression Trees) algorithm, which builds binary trees using greedy, top-down splitting.

### How the Tree is Built

| Step | Description |
|------|-------------|
| **1. Select Best Split** | Evaluate every feature and threshold; pick the one that maximizes information gain (or minimizes impurity) |
| **2. Split the Node** | Partition data into left and right child nodes |
| **3. Recurse** | Repeat on each child until a stopping criterion is met |
| **4. Assign Leaf Value** | Majority class (classification) or mean value (regression) |

### Splitting Criteria

| Criterion | Task | Formula Intuition |
|-----------|------|-------------------|
| **Gini Impurity** (default) | Classification | Probability of misclassification |
| **Entropy** | Classification | Information gain (Shannon entropy) |
| **Squared Error** (default) | Regression | Minimizes MSE |

### Key Hyperparameters

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `max_depth` | Maximum tree depth | 3–20 |
| `min_samples_split` | Min samples to split a node | 2–20 |
| `min_samples_leaf` | Min samples in a leaf | 1–10 |
| `max_features` | Features considered per split | `'sqrt'`, `'log2'`, or int |
| `criterion` | Splitting quality measure | `'gini'`, `'entropy'` |

### Code Example
```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build tree with controlled depth to prevent overfitting
tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42
)
tree.fit(X_train, y_train)
print(f"Accuracy: {tree.score(X_test, y_test):.4f}")

# Inspect the tree structure
print(export_text(tree, feature_names=load_iris().feature_names))

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=load_iris().feature_names,
          class_names=load_iris().target_names, filled=True)
plt.show()

# Feature importance
for name, imp in zip(load_iris().feature_names, tree.feature_importances_):
    print(f"{name}: {imp:.4f}")
```

### Interview Tips
- Decision trees are prone to **overfitting**; always control depth or use pruning.
- `feature_importances_` shows which features drive splits.
- CART produces **binary splits only**, unlike some other algorithms (e.g., ID3/C4.5).
- Mention that ensembles (Random Forest, Gradient Boosting) improve upon single trees.

---

## Question 13

**Explain the differences between RandomForestClassifier and GradientBoostingClassifier in Scikit-Learn.**

### Definition
Both are ensemble methods that combine multiple decision trees, but they differ fundamentally in **how** they build and combine those trees.

### Key Differences

| Aspect | RandomForestClassifier | GradientBoostingClassifier |
|--------|----------------------|---------------------------|
| **Strategy** | Bagging (parallel trees) | Boosting (sequential trees) |
| **Tree Building** | Trees built independently | Each tree corrects errors of the previous |
| **Randomness** | Random feature subsets + bootstrap samples | No bootstrapping; fits residuals |
| **Bias-Variance** | Reduces variance | Reduces bias |
| **Overfitting Risk** | Low (more trees rarely overfit) | Higher (can overfit with too many trees) |
| **Speed** | Faster (parallelizable) | Slower (sequential) |
| **Key Params** | `n_estimators`, `max_features` | `n_estimators`, `learning_rate`, `max_depth` |

### How Each Works

**Random Forest:**
1. Draw N bootstrap samples from the training data
2. Build a tree on each sample using a random subset of features per split
3. Aggregate predictions by majority vote (classification) or averaging (regression)

**Gradient Boosting:**
1. Start with a simple prediction (e.g., mean)
2. Compute residuals (errors)
3. Fit a shallow tree to the residuals
4. Update predictions by adding the new tree scaled by `learning_rate`
5. Repeat until `n_estimators` trees are built

### Code Example
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"RF Accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1,
    max_depth=3, random_state=42
)
gb_scores = cross_val_score(gb, X, y, cv=5, scoring='accuracy')
print(f"GB Accuracy: {gb_scores.mean():.4f} (+/- {gb_scores.std()*2:.4f})")

# Feature importances comparison
rf.fit(X, y)
gb.fit(X, y)
print("Top RF features:", rf.feature_importances_[:5])
print("Top GB features:", gb.feature_importances_[:5])
```

### When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Quick baseline, minimal tuning needed | Random Forest |
| Maximum accuracy with careful tuning | Gradient Boosting |
| Parallelizable training required | Random Forest |
| Small-to-medium dataset | Either works well |
| Need to avoid overfitting with default settings | Random Forest |

### Interview Tips
- Random Forest is easier to tune and harder to overfit; Gradient Boosting often yields higher accuracy but requires careful hyperparameter tuning.
- Mention **HistGradientBoostingClassifier** as a faster alternative for large datasets (inspired by LightGBM).
- `learning_rate` in gradient boosting controls the contribution of each tree — lower values need more trees but generalize better.

---

## Question 14

**How does Scikit-Learn’s SVM handle non-linear data?**

### Definition
Support Vector Machines (SVM) find the optimal hyperplane that separates classes with maximum margin. For non-linear data, Scikit-Learn uses the **kernel trick** to implicitly map data into a higher-dimensional space where a linear separator exists, without actually computing the transformation.

### The Kernel Trick

| Kernel | Formula Intuition | Use Case |
|--------|-------------------|----------|
| **linear** | Dot product in original space | Linearly separable data |
| **rbf** (default) | Gaussian similarity; infinite-dimensional mapping | Most non-linear problems |
| **poly** | Polynomial feature interactions | Moderate non-linearity |
| **sigmoid** | Tanh-based similarity | Neural-network-like behavior |

### How RBF Kernel Works
1. Compute pairwise similarity between data points using K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
2. Points close together have high similarity (near 1), far apart have low similarity (near 0)
3. The `gamma` parameter controls the influence radius:
   - **High gamma**: Tight radius → complex boundary → risk of overfitting
   - **Low gamma**: Wide radius → smooth boundary → risk of underfitting

### Key Hyperparameters

| Parameter | Role | Effect |
|-----------|------|--------|
| `C` | Regularization | High C → less regularization, tighter fit |
| `gamma` | Kernel coefficient | High gamma → more complex boundary |
| `kernel` | Kernel function | `'rbf'`, `'poly'`, `'linear'`, `'sigmoid'` |
| `degree` | Degree for poly kernel | Higher → more complex boundary |

### Code Example
```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create non-linear dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM requires feature scaling
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])

# Tune C and gamma
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 1]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Test accuracy: {grid.score(X_test, y_test):.4f}")

# Compare kernels
for kernel in ['linear', 'rbf', 'poly']:
    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel=kernel))])
    svm.fit(X_train, y_train)
    print(f"{kernel}: {svm.score(X_test, y_test):.4f}")
```

### Interview Tips
- **Always scale features** before training SVM — it is distance-based and sensitive to feature magnitudes.
- The kernel trick avoids explicitly computing high-dimensional features, making SVM computationally feasible.
- RBF is the default and works well for most non-linear problems; start there before trying others.
- SVM training complexity is O(n^2) to O(n^3), so it struggles with very large datasets.

---

## Question 15

**What is a support vector machine, and how can it be used for both classification and regression tasks?**

### Definition
A Support Vector Machine (SVM) is a supervised learning algorithm that finds the optimal hyperplane maximizing the margin between classes. The data points closest to the hyperplane are called **support vectors** — they define the decision boundary.

### SVM for Classification vs Regression

| Aspect | SVC (Classification) | SVR (Regression) |
|--------|---------------------|-------------------|
| **Goal** | Maximize margin between classes | Fit data within an epsilon-tube |
| **Output** | Discrete class labels | Continuous values |
| **Loss** | Hinge loss | Epsilon-insensitive loss |
| **Key Param** | `C` (regularization) | `C`, `epsilon` (tube width) |
| **Class** | `sklearn.svm.SVC` | `sklearn.svm.SVR` |

### How SVR Works
Instead of maximizing the margin between classes, SVR defines an **epsilon-tube** around the predicted function. Points inside the tube incur no penalty; points outside contribute to the loss proportionally to their distance from the tube boundary.

### Code Example
```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, fetch_california_housing

# --- Classification with SVC ---
X_clf, y_clf = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=1.0, probability=True))
])
clf_pipe.fit(X_train, y_train)
print(f"Classification Accuracy: {clf_pipe.score(X_test, y_test):.4f}")
print(f"Probabilities: {clf_pipe.predict_proba(X_test[:3])}")

# --- Regression with SVR ---
X_reg, y_reg = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
])
reg_pipe.fit(X_train, y_train)
print(f"Regression R2: {reg_pipe.score(X_test, y_test):.4f}")
```

### Multi-Class Classification
```python
# SVC handles multi-class automatically via one-vs-one (default) or one-vs-rest
from sklearn.svm import SVC, LinearSVC

# One-vs-One (default for SVC): trains C(n,2) classifiers
svc_ovo = SVC(kernel='rbf', decision_function_shape='ovo')

# One-vs-Rest (default for LinearSVC): trains n classifiers
svc_ovr = LinearSVC(multi_class='ovr')
```

### Interview Tips
- SVC uses **hinge loss** and maximizes margin; SVR uses **epsilon-insensitive loss** and fits a tube.
- Set `probability=True` on SVC to enable `predict_proba()`, but it adds computation (uses Platt scaling internally).
- For large datasets, prefer `LinearSVC` or `SGDClassifier(loss='hinge')` over `SVC` for scalability.
- The `epsilon` parameter in SVR controls the width of the no-penalty zone — larger epsilon means fewer support vectors and a smoother fit.

---

## Question 16

**Describe the process of deploying a Scikit-Learn model into a production environment.**

### Definition
Model deployment is the process of making a trained machine-learning model available for inference in a production system. Scikit-Learn models can be serialized, containerized, and served via APIs or embedded directly into applications.

### Deployment Pipeline

| Step | Description | Tools |
|------|-------------|-------|
| **1. Train & Validate** | Build and evaluate the model | Scikit-Learn, cross-validation |
| **2. Serialize** | Save model to disk | `joblib`, `pickle` |
| **3. Create API** | Wrap model in a web service | Flask, FastAPI, Django |
| **4. Containerize** | Package app + dependencies | Docker |
| **5. Deploy** | Host on infrastructure | AWS, GCP, Azure, Kubernetes |
| **6. Monitor** | Track performance over time | Prometheus, custom logging |

### Step-by-Step Code Example

**Saving and Loading the Model:**
```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Train a complete pipeline (includes preprocessing)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

# Save the entire pipeline (not just the model!)
joblib.dump(pipeline, 'model_pipeline.joblib')

# Load in production
loaded_pipeline = joblib.load('model_pipeline.joblib')
predictions = loaded_pipeline.predict(X_new)
```

**Serving via FastAPI:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
pipeline = joblib.load('model_pipeline.joblib')

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0].tolist()
    return {"prediction": int(prediction), "probabilities": probability}
```

**Dockerfile:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model_pipeline.joblib .
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Best Practices

| Practice | Why |
|----------|-----|
| **Save the full Pipeline** | Ensures preprocessing is always applied consistently |
| **Version your models** | Track which model is in production (e.g., `model_v2.1.joblib`) |
| **Validate inputs** | Check for missing features, data types, and value ranges |
| **Log predictions** | Store inputs and outputs for monitoring and debugging |
| **Pin dependencies** | Use `requirements.txt` with exact versions |

### Interview Tips
- Always serialize the **entire pipeline**, not just the estimator — this guarantees consistent preprocessing.
- Use `joblib` over `pickle` for Scikit-Learn models because it handles large NumPy arrays more efficiently.
- Mention **model versioning** and **A/B testing** for production maturity.
- Be aware that Scikit-Learn models must match the library version used during training — version mismatches can cause errors.

---

## Question 17

**Explain how you would update a Scikit-Learn model with new data over time.**

### Definition
Updating a model with new data (also called **incremental learning** or **online learning**) means incorporating fresh observations without retraining from scratch. Scikit-Learn supports this through `partial_fit()` for compatible estimators, or through periodic full retraining.

### Two Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Full Retrain** | Retrain on old + new data combined | Small/medium datasets; model drift detected |
| **Incremental (`partial_fit`)** | Update model parameters with new batch | Streaming data; large datasets that don't fit in memory |

### Estimators Supporting `partial_fit()`

| Category | Estimators |
|----------|-----------|
| **Classification** | `SGDClassifier`, `Perceptron`, `MultinomialNB`, `BernoulliNB`, `PassiveAggressiveClassifier` |
| **Regression** | `SGDRegressor`, `PassiveAggressiveRegressor` |
| **Clustering** | `MiniBatchKMeans`, `Birch` |
| **Decomposition** | `IncrementalPCA`, `MiniBatchDictionaryLearning` |

### Code Example: Incremental Learning
```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np

# Initial training data
X_initial, y_initial = make_classification(n_samples=1000, n_features=20, random_state=42)

# Incrementally trainable model
model = SGDClassifier(loss='log_loss', random_state=42)
scaler = StandardScaler()

# Initial fit
X_scaled = scaler.fit_transform(X_initial)
model.partial_fit(X_scaled, y_initial, classes=np.unique(y_initial))
print(f"Initial accuracy: {model.score(X_scaled, y_initial):.4f}")

# Simulate new data arriving in batches
for batch_num in range(5):
    X_new, y_new = make_classification(n_samples=200, n_features=20, random_state=batch_num)
    X_new_scaled = scaler.transform(X_new)  # Use existing scaler
    model.partial_fit(X_new_scaled, y_new)
    print(f"Batch {batch_num + 1} accuracy: {model.score(X_new_scaled, y_new):.4f}")
```

### Code Example: Full Retrain with Versioning
```python
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

def retrain_model(X_all, y_all, model_dir='models/'):
    """Retrain model on full dataset and save with timestamp."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_all, y_all)
    
    # Version the model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f"{model_dir}model_{timestamp}.joblib"
    joblib.dump(model, path)
    print(f"Model saved: {path}")
    return model

# Schedule this function to run periodically (e.g., weekly)
# Combine historical + new data before calling
```

### Handling Concept Drift

| Technique | Description |
|-----------|-------------|
| **Monitor metrics** | Track accuracy/F1 on recent data; retrain when performance drops |
| **Sliding window** | Train only on the most recent N observations |
| **Weighted retraining** | Give higher weight to recent samples using `sample_weight` |
| **Scheduled retraining** | Retrain on a fixed schedule (daily, weekly) |

### Interview Tips
- Most Scikit-Learn models (e.g., Random Forest, SVM) do **not** support `partial_fit()` — they must be retrained from scratch.
- `partial_fit()` requires passing `classes=` on the first call for classifiers so the model knows all possible labels.
- For production systems, discuss **concept drift detection** — model performance degrades when the data distribution changes.
- Mention the trade-off: full retraining gives optimal accuracy, incremental learning saves time and memory.

---

## Question 18

**What are some of the limitations of Scikit-Learn when dealing with very large datasets?**

### Definition
Scikit-Learn is designed for in-memory computation on single machines. While excellent for small-to-medium datasets, it faces significant challenges with very large datasets (millions of rows or high-dimensional feature spaces).

### Key Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| **In-Memory Only** | All data must fit in RAM | Cannot process datasets larger than available memory |
| **Single-Machine** | No native distributed computing | Cannot scale horizontally across clusters |
| **No GPU Support** | CPU-only computation | Slower training on large datasets vs GPU-based libraries |
| **Limited Incremental Learning** | Few estimators support `partial_fit()` | Most models require full retraining |
| **No Deep Learning** | No neural network architectures | Must use TensorFlow/PyTorch for deep learning tasks |
| **Sparse Data Handling** | Some algorithms struggle with sparse matrices | High-dimensional text/NLP data can be slow |

### Workarounds and Alternatives

| Problem | Workaround | Alternative Library |
|---------|------------|-------------------|
| Data too large for RAM | Use `partial_fit()` with batches | Dask-ML, Vaex |
| Need distributed training | N/A | Apache Spark MLlib, Dask-ML |
| Slow training | Use `n_jobs=-1` for parallelism | XGBoost, LightGBM (faster boosting) |
| Need GPU acceleration | N/A | cuML (RAPIDS), TensorFlow, PyTorch |
| Very large feature space | Use `HashingVectorizer`, dimensionality reduction | Vowpal Wabbit |

### Code Example: Scaling Strategies
```python
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import make_classification
import numpy as np

# Strategy 1: Use partial_fit for out-of-core learning
model = SGDClassifier(loss='log_loss')

def stream_data(file_path, batch_size=1000):
    """Generator that yields batches from a large file."""
    import pandas as pd
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        X = chunk.drop('target', axis=1).values
        y = chunk['target'].values
        yield X, y

# Strategy 2: Use n_jobs for parallel training
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # Use all CPU cores

# Strategy 3: Subsample for prototyping, then scale
from sklearn.model_selection import train_test_split
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)
model.fit(X_sample, y_sample)  # Fast prototyping on 10% of data

# Strategy 4: HashingVectorizer for large text datasets (fixed memory)
vectorizer = HashingVectorizer(n_features=2**16)  # No fit needed
X_hashed = vectorizer.transform(text_data)
```

### When to Move Beyond Scikit-Learn

| Scenario | Recommended Tool |
|----------|-----------------|
| Datasets > 10GB | Dask-ML, Spark MLlib |
| GPU-accelerated ML | RAPIDS cuML |
| Gradient boosting at scale | XGBoost, LightGBM, CatBoost |
| Deep learning / neural nets | TensorFlow, PyTorch |
| Real-time streaming ML | River, Vowpal Wabbit |

### Interview Tips
- Scikit-Learn is **not** designed for big data — acknowledge this and pivot to discussing alternatives.
- The `n_jobs=-1` parameter is the simplest scaling trick; it parallelizes across CPU cores for models like Random Forest and GridSearchCV.
- For interviews, show awareness of the ecosystem: Scikit-Learn for prototyping, then XGBoost/Spark/cuML for production scale.
- `partial_fit()` enables out-of-core learning but is limited to a subset of algorithms (SGD, Naive Bayes, MiniBatchKMeans).

---

## Question 19

**How can you scale features in a dataset using Scikit-Learn ?**

*Answer to be added.*

---

## Question 20

**What are the strategies provided by Scikit-Learn to handle imbalanced datasets ?**

*Answer to be added.*

---

## Question 21

**How do you split a dataset into training and testing sets using Scikit-Learn ?**

*Answer to be added.*

---

## Question 22

**Explain how Imputer works in Scikit-Learn for dealing with missing data**

*Answer to be added.*

---

## Question 23

**How do you normalize or standardize data with Scikit-Learn ?**

*Answer to be added.*

---

## Question 24

**Explain the process of training a supervised machine learning model using Scikit-Learn**

*Answer to be added.*

---

## Question 25

**Explain the GridSearchCV function and its purpose**

*Answer to be added.*

---

## Question 26

**What is the difference between .fit() , .predict() , and .transform() methods?**

*Answer to be added.*

---

## Question 27

**How would you explain the concept of overfitting , and how can it be identified using Scikit-Learn tools?**

*Answer to be added.*

---

## Question 28

**How do you use Scikit-Learn to build ensemble models ?**

*Answer to be added.*

---

## Question 29

**Describe the k-means clustering process as implemented in Scikit-Learn**

*Answer to be added.*

---

## Question 30

**How can you implement custom transformers in Scikit-Learn ?**

*Answer to be added.*

---

## Question 31

**Discuss the integration of Scikit-Learn with other popular machine learning libraries like TensorFlow and PyTorch**

*Answer to be added.*

---
## Question 32

**How does Scikit-Learn implement logistic regression differently from linear regression?**

**Answer:**

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|--------------------|
| Task | Regression (continuous) | Classification (discrete) |
| Output | $\hat{y} = Xw + b$ | $P(y=1) = \sigma(Xw + b)$ |
| Loss function | MSE: $\frac{1}{n}\sum(y - \hat{y})^2$ | Log loss: $-\frac{1}{n}\sum[y\log(p) + (1-y)\log(1-p)]$ |
| Activation | None (identity) | Sigmoid: $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| Solver | Closed-form (OLS) or SGD | Iterative (LBFGS, Newton, SAG) |
| Regularization | Optional (Ridge, Lasso) | Always applied (`C` parameter) |

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

# Linear Regression
lin_reg = LinearRegression()  # No regularization by default
lin_reg.fit(X_train, y_continuous)
y_pred = lin_reg.predict(X_test)  # Continuous values
print(lin_reg.coef_)              # Feature weights
print(lin_reg.intercept_)         # Bias term

# Logistic Regression
log_reg = LogisticRegression(
    C=1.0,              # Inverse regularization (higher = less regularization)
    penalty='l2',       # Regularization type
    solver='lbfgs',     # Optimization algorithm
    max_iter=100,
    multi_class='auto'  # 'ovr' or 'multinomial'
)
log_reg.fit(X_train, y_class)
y_pred = log_reg.predict(X_test)          # Class labels
y_prob = log_reg.predict_proba(X_test)    # Probabilities
print(log_reg.classes_)                    # [0, 1]

# Key difference: LogisticRegression ALWAYS regularizes
# C=1e10 ≈ no regularization
# C=0.01 = strong regularization
```

> **Interview Tip:** Despite the name, logistic regression is a **classifier**. The key API difference is `predict_proba()` — linear regression doesn't have it. Logistic regression always uses regularization (`C` parameter).

---

