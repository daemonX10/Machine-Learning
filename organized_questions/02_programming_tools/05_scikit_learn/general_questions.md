# Scikit-Learn Interview Questions - General Questions

## Question 1

**How do you handle missing values using Scikit-Learn?**

### SimpleImputer Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `mean` | Replace with column mean | Numeric, normal distribution |
| `median` | Replace with column median | Numeric, with outliers |
| `most_frequent` | Replace with mode | Categorical |
| `constant` | Replace with specified value | Custom fill |

### Code Example
```python
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

X = np.array([[1, 2], [np.nan, 3], [7, np.nan]])

# Mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Median imputation (robust to outliers)
imputer = SimpleImputer(strategy='median')

# KNN imputation (uses similar samples)
imputer = KNNImputer(n_neighbors=3)
X_imputed = imputer.fit_transform(X)
```

### Best Practice: Use in Pipeline
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
```

---

## Question 2

**How do you perform feature scaling in Scikit-Learn?**

### Scaling Methods

| Scaler | Formula | Use Case |
|--------|---------|----------|
| `StandardScaler` | (x - mean) / std | Most algorithms, normal dist |
| `MinMaxScaler` | (x - min) / (max - min) | Fixed range [0, 1] |
| `RobustScaler` | (x - median) / IQR | Data with outliers |
| `MaxAbsScaler` | x / max(abs(x)) | Sparse data |

### Code Example
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same parameters!

# MinMaxScaler: [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))

# RobustScaler: resistant to outliers
scaler = RobustScaler()
```

### Important: Fit only on training data!
```python
# CORRECT
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# WRONG (data leakage)
scaler.fit(X)  # Don't fit on all data!
```

---

## Question 3

**How do you encode categorical variables?**

### Encoding Methods

| Encoder | Output | Use Case |
|---------|--------|----------|
| `OneHotEncoder` | Binary columns | Nominal (no order) |
| `OrdinalEncoder` | Integer codes | Ordinal (has order) |
| `LabelEncoder` | Integer codes | Target variable |

### Code Example
```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# One-Hot Encoding (for features)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[['city']])

# Ordinal Encoding (for ordered categories)
encoder = OrdinalEncoder(categories=[['small', 'medium', 'large']])
X_encoded = encoder.fit_transform(X[['size']])

# Label Encoding (for target variable)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

### In ColumnTransformer
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])
```

---

## Question 4

**How do you split data for training and testing?**

### train_test_split
```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    stratify=y          # Maintain class proportions
)

# With validation set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
# Result: 60% train, 20% val, 20% test
```

### Cross-Validation (Better)
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Mean accuracy: {scores.mean():.4f}")
```

---

## Question 5

**How do you perform feature selection?**

### Feature Selection Methods

| Method | Type | Use Case |
|--------|------|----------|
| `SelectKBest` | Filter | Quick, based on statistics |
| `RFE` | Wrapper | Uses model performance |
| `SelectFromModel` | Embedded | Uses feature importances |

### Code Example
```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier

# Method 1: SelectKBest (statistical test)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
print(f"Selected features: {selector.get_support()}")

# Method 2: RFE (Recursive Feature Elimination)
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Method 3: SelectFromModel (based on importance)
model = RandomForestClassifier().fit(X, y)
selector = SelectFromModel(model, prefit=True)
X_selected = selector.transform(X)

# Get feature importances
importances = model.feature_importances_
```

---

## Question 6

**How do you save and load a trained model?**

### Using joblib (Recommended)
```python
import joblib

# Save model
joblib.dump(model, 'model.joblib')

# Load model
loaded_model = joblib.load('model.joblib')
predictions = loaded_model.predict(X_new)
```

### Using pickle
```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### Save Complete Pipeline
```python
from sklearn.pipeline import Pipeline

# Create and train pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# Save entire pipeline (includes preprocessing!)
joblib.dump(pipeline, 'full_pipeline.joblib')

# Load and use directly on raw data
pipeline = joblib.load('full_pipeline.joblib')
predictions = pipeline.predict(X_new)  # Handles scaling automatically
```

---

## Question 7

**How do you implement custom transformers?**

### Creating Custom Transformer
```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    """Custom transformer that scales to [0, 100] range."""
    
    def __init__(self, multiply_by=100):
        self.multiply_by = multiply_by
    
    def fit(self, X, y=None):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self
    
    def transform(self, X):
        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        return X_scaled * self.multiply_by

# Use in pipeline
pipeline = Pipeline([
    ('custom_scaler', CustomScaler(multiply_by=100)),
    ('model', LogisticRegression())
])
```

### FunctionTransformer (Simple Cases)
```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Wrap any function
log_transformer = FunctionTransformer(np.log1p)
X_log = log_transformer.fit_transform(X)
```

---

## Question 8

**How do you handle multiclass classification?**

### Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| One-vs-Rest (OvR) | N binary classifiers | Default for most |
| One-vs-One (OvO) | N(N-1)/2 classifiers | SVM with many classes |
| Native | Algorithm handles natively | Decision Trees, RF |

### Code Example
```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC

# One-vs-Rest
ovr = OneVsRestClassifier(SVC(kernel='linear'))
ovr.fit(X_train, y_train)

# One-vs-One
ovo = OneVsOneClassifier(SVC(kernel='rbf'))
ovo.fit(X_train, y_train)

# Native multiclass (no wrapper needed)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()  # Handles multiclass natively
rf.fit(X_train, y_train)
```

### Multiclass Metrics
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
# Shows precision, recall, F1 for each class
```

---

## Question 9

**How do you implement ensemble methods?**

### Ensemble Types

| Type | Method | Key Parameter |
|------|--------|---------------|
| **Bagging** | Random Forest | `n_estimators` |
| **Boosting** | GradientBoosting, AdaBoost | `learning_rate` |
| **Voting** | VotingClassifier | `voting='hard'/'soft'` |
| **Stacking** | StackingClassifier | `final_estimator` |

### Code Example
```python
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression

# Bagging: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Boosting: Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# Voting: Combine multiple models
voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('lr', LogisticRegression())
    ],
    voting='soft'  # Use probabilities
)

# Stacking: Use meta-learner
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    final_estimator=LogisticRegression()
)
```

---

## Question 10

**How do you perform dimensionality reduction?**

### Methods

| Method | Type | Use Case |
|--------|------|----------|
| PCA | Linear | General reduction |
| t-SNE | Non-linear | Visualization |
| LDA | Supervised | Classification |

### Code Example
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# PCA: Reduce to n components
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)
print(f"Components kept: {pca.n_components_}")

# Explained variance
print(f"Variance ratio: {pca.explained_variance_ratio_}")

# t-SNE: For visualization (2D)
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)

# LDA: Supervised dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

### PCA in Pipeline
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('model', LogisticRegression())
])
```

