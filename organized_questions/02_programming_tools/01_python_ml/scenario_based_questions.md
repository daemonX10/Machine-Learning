# Python ML Interview Questions - Scenario-Based Questions

## Question 1

**Discuss the difference between a list, a tuple, and a set in Python.**

### Definition
Lists, tuples, and sets are Python's core collection data structures, each with distinct characteristics for mutability, ordering, and uniqueness.

### Comparison Table

| Feature | List | Tuple | Set |
|---------|------|-------|-----|
| Syntax | `[1, 2, 3]` | `(1, 2, 3)` | `{1, 2, 3}` |
| Mutable | Yes | No | Yes |
| Ordered | Yes | Yes | No (Python 3.7+ maintains insertion order) |
| Duplicates | Allowed | Allowed | Not Allowed |
| Indexing | Yes | Yes | No |
| Hashable | No | Yes | No (but elements must be hashable) |
| Use Case | General purpose collection | Fixed data, dictionary keys | Unique elements, membership testing |

### When to Use Each

**List** - When you need:
- Ordered collection that can change
- Allow duplicates
- Index-based access

```python
features = ['age', 'income', 'score']
features.append('education')  # Modifiable
```

**Tuple** - When you need:
- Immutable data (coordinates, RGB values)
- Dictionary keys
- Return multiple values from function
- Memory efficiency

```python
point = (10, 20)  # Coordinates
rgb = (255, 128, 0)  # Color
# point[0] = 5  # Error! Immutable
```

**Set** - When you need:
- Unique elements only
- Fast membership testing O(1)
- Set operations (union, intersection)

```python
unique_labels = {1, 2, 3, 2, 1}  # Becomes {1, 2, 3}
print(2 in unique_labels)  # O(1) lookup
```

### ML Use Case
```python
# List: Store features
features = ['f1', 'f2', 'f3']

# Tuple: Return train/test split
def split_data(X, y):
    return X_train, X_test, y_train, y_test  # Returns tuple

# Set: Find unique classes
unique_classes = set(y_labels)
```

---

## Question 2

**Discuss the usage of *args and **kwargs in function definitions.**

### Definition
- `*args`: Allows passing variable number of **positional arguments** (collected as tuple)
- `**kwargs`: Allows passing variable number of **keyword arguments** (collected as dictionary)

### Code Examples

```python
# *args example
def sum_all(*args):
    """Accept any number of positional arguments."""
    print(f"args is a tuple: {args}")
    return sum(args)

result = sum_all(1, 2, 3, 4, 5)
# args is a tuple: (1, 2, 3, 4, 5)
# result = 15


# **kwargs example
def create_model(**kwargs):
    """Accept any number of keyword arguments."""
    print(f"kwargs is a dict: {kwargs}")
    for key, value in kwargs.items():
        print(f"  {key} = {value}")

create_model(learning_rate=0.01, epochs=100, batch_size=32)
# kwargs is a dict: {'learning_rate': 0.01, 'epochs': 100, 'batch_size': 32}


# Combined example
def flexible_function(required, *args, default=10, **kwargs):
    """Shows argument order: required -> *args -> defaults -> **kwargs"""
    print(f"required: {required}")
    print(f"args: {args}")
    print(f"default: {default}")
    print(f"kwargs: {kwargs}")

flexible_function("must have", 1, 2, 3, default=20, extra="value")
```

### ML Use Case
```python
# Wrapper function for model training
def train_model(model_class, X, y, *preprocessing_steps, **hyperparams):
    """
    model_class: The ML model class
    *preprocessing_steps: Variable preprocessing functions
    **hyperparams: Model hyperparameters
    """
    # Apply preprocessing
    for step in preprocessing_steps:
        X = step(X)
    
    # Create and train model with hyperparameters
    model = model_class(**hyperparams)
    model.fit(X, y)
    return model
```

---

## Question 3

**Discuss the benefits of using Jupyter Notebooks for machine learning projects.**

### Key Benefits

**1. Interactive Development**
- Execute code cell by cell
- See immediate output/visualizations
- Experiment and iterate quickly

**2. Documentation + Code Together**
- Markdown cells for explanations
- Code cells for implementation
- Creates reproducible research

**3. Visualization Integration**
- Inline plots with Matplotlib/Seaborn
- Interactive widgets
- Rich output (images, HTML, LaTeX)

**4. Exploratory Data Analysis (EDA)**
- Display DataFrames directly
- Quick statistical summaries
- Iterative data exploration

**5. Prototyping and Experimentation**
- Test different approaches quickly
- Share results with stakeholders
- Export to various formats (HTML, PDF, Python script)

### Limitations
- Version control is difficult (JSON format)
- Not ideal for production code
- Can lead to hidden state issues
- Transition to .py files for deployment

---

## Question 4

**Discuss the use of pipelines in Scikit-learn for streamlining preprocessing steps.**

### Definition
A Pipeline chains multiple preprocessing steps and a model into a single object, ensuring consistent data flow and preventing data leakage.

### Benefits
1. **Prevents Data Leakage**: Fit preprocessing only on training data
2. **Cleaner Code**: Single object instead of multiple steps
3. **Easy Cross-Validation**: CV properly handles preprocessing
4. **Simpler Deployment**: One object to save and load

### Code Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Define column types
numeric_features = ['age', 'income']
categorical_features = ['city', 'gender']

# Create preprocessing pipelines for each type
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create full pipeline with model
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Use the pipeline
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

### Interview Tip
Always mention that pipelines prevent data leakage - preprocessing is fit only on training folds during cross-validation.

---

## Question 5

**Discuss how ensemble methods work and give an example where they might be useful.**

### Definition
Ensemble methods combine multiple models to create a stronger predictor. The key insight: diverse models make different errors that can cancel out.

### Types of Ensemble Methods

| Method | How It Works | Reduces |
|--------|-------------|---------|
| **Bagging** | Train models on random subsets (with replacement) | Variance |
| **Boosting** | Train sequentially, focus on previous errors | Bias |
| **Stacking** | Use meta-model to combine base model predictions | Both |

### Scenario: Credit Card Fraud Detection

**Why Ensemble Works Here:**
- High-stakes decision (false negatives are costly)
- Complex patterns in data
- Need robust predictions

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

# Create diverse base models
models = [
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100))
]

# Combine with voting
ensemble = VotingClassifier(estimators=models, voting='soft')
ensemble.fit(X_train, y_train)

# Ensemble often outperforms individual models
print(f"Ensemble Accuracy: {ensemble.score(X_test, y_test):.3f}")
```

---

## Question 6

**How would you assess a model's performance? Mention at least three metrics.**

### Classification Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | $(TP + TN) / Total$ | Balanced classes |
| Precision | $TP / (TP + FP)$ | High cost of false positive |
| Recall | $TP / (TP + FN)$ | High cost of false negative |
| F1-Score | $2 \times \frac{P \times R}{P + R}$ | Imbalanced data |
| ROC-AUC | Area under TPR vs FPR curve | Threshold-independent |

### Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | Penalizes large errors |
| MAE | $\frac{1}{n}\sum|y - \hat{y}|$ | Robust to outliers |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Variance explained |

### Code Example
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
print(f"F1:        {f1_score(y_true, y_pred):.3f}")
```

### Interview Tip
Always ask: "What is the business cost of different types of errors?" This determines which metric to prioritize.

