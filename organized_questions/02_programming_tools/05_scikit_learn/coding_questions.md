# Scikit-Learn Interview Questions - Coding Questions

## Question 1

**Implement K-Means clustering with Scikit-Learn.**

### Solution
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# Generate sample data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# 1. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 3. Evaluate
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")

# 4. Elbow method to find optimal k
def find_optimal_k(X, max_k=10):
    inertias = []
    silhouettes = []
    
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))
    
    return inertias, silhouettes

inertias, silhouettes = find_optimal_k(X_scaled)
optimal_k = np.argmax(silhouettes) + 2
print(f"Optimal k: {optimal_k}")
```

---

## Question 2

**Implement a complete classification pipeline with preprocessing.**

### Solution
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np

# Sample data
data = {
    'age': [25, 30, np.nan, 45, 35],
    'salary': [50000, 60000, 55000, np.nan, 70000],
    'city': ['NY', 'LA', 'NY', 'LA', 'SF'],
    'target': [0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df.drop('target', axis=1)
y = df['target']

# Define column types
numeric_features = ['age', 'salary']
categorical_features = ['city']

# Create transformers
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
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)

print(f"Test accuracy: {pipeline.score(X_test, y_test):.4f}")
```

---

## Question 3

**Implement hyperparameter tuning with GridSearchCV.**

### Solution
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from scipy.stats import randint

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define model
rf = RandomForestClassifier(random_state=42)

# GridSearchCV: exhaustive search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    rf, param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# RandomizedSearchCV: faster for large search spaces
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    rf, param_dist,
    n_iter=50,  # Number of random combinations
    cv=5,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
```

---

## Question 4

**Implement cross-validation with multiple metrics.**

### Solution
```python
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

# Define custom scorers
scorers = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'roc_auc': 'roc_auc'
}

# Stratified K-Fold (maintains class proportions)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validate with multiple metrics
model = RandomForestClassifier(random_state=42)
results = cross_validate(
    model, X, y,
    cv=cv,
    scoring=scorers,
    return_train_score=True
)

# Print results
for metric in scorers.keys():
    train_score = results[f'train_{metric}'].mean()
    test_score = results[f'test_{metric}'].mean()
    print(f"{metric}: Train={train_score:.4f}, Test={test_score:.4f}")
```

---

## Question 5

**Implement a custom transformer for log transformation.**

### Solution
```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log transformation to skewed features."""
    
    def __init__(self, columns=None, threshold=1.0):
        self.columns = columns
        self.threshold = threshold
        
    def fit(self, X, y=None):
        if self.columns is None:
            # Auto-detect skewed columns
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                skewness = X.skew()
                self.columns_ = skewness[skewness.abs() > self.threshold].index.tolist()
            else:
                self.columns_ = list(range(X.shape[1]))
        else:
            self.columns_ = self.columns
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns_:
            if isinstance(X_copy, pd.DataFrame):
                X_copy[col] = np.log1p(X_copy[col])
            else:
                X_copy[:, col] = np.log1p(X_copy[:, col])
        return X_copy

# Use in pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('log', LogTransformer(threshold=0.5)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
```

---

## Question 6

**Implement model evaluation with confusion matrix and ROC curve.**

### Solution
```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```

---

## Question 7

**Implement feature importance analysis.**

### Solution
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Method 1: Built-in feature importance (tree-based)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature Ranking (Built-in):")
for i, idx in enumerate(indices[:10]):
    print(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")

# Method 2: Permutation Importance (model-agnostic)
perm_importance = permutation_importance(
    model, X_test, y_test, 
    n_repeats=10, 
    random_state=42
)

print("\nPermutation Importance:")
sorted_idx = perm_importance.importances_mean.argsort()[::-1]
for idx in sorted_idx[:10]:
    print(f"Feature {idx}: {perm_importance.importances_mean[idx]:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[indices[:10]])
plt.yticks(range(10), [f'Feature {i}' for i in indices[:10]])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
```

---

## Question 8

**Implement train-test split with stratification for imbalanced data.**

### Solution
```python
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from collections import Counter

# Check class distribution
print(f"Class distribution: {Counter(y)}")

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Maintains class proportions
    random_state=42
)

print(f"Train distribution: {Counter(y_train)}")
print(f"Test distribution: {Counter(y_test)}")

# Stratified cross-validation
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(class_weight='balanced', random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

print(f"Cross-val F1 scores: {scores}")
print(f"Mean F1: {scores.mean():.4f}")
```

---

## Question 9

**Implement a regression model with regularization.**

### Solution
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate regression data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Compare regularized models
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE={rmse:.4f}, R²={r2:.4f}")

# Tune regularization strength
param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

print(f"\nBest alpha: {grid.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-grid.best_score_):.4f}")
```

---

## Question 10

**Implement learning curve analysis to diagnose bias/variance.**

### Solution
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, val_mean, 's-', color='red', label='Validation')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Diagnose
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.1:
        print("⚠️ High variance (overfitting): Try more data, simpler model, regularization")
    elif val_mean[-1] < 0.7:
        print("⚠️ High bias (underfitting): Try more features, complex model")
    else:
        print("✅ Good fit!")
    
    plt.show()

# Use
model = RandomForestClassifier(random_state=42)
plot_learning_curve(model, X, y)
```

