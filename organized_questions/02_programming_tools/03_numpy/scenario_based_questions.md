# NumPy Interview Questions - Scenario-Based Questions

## Scenario 1

**You need to normalize a dataset for a neural network. How would you use NumPy to perform Z-score standardization on a feature matrix?**

### Scenario Context
Neural networks train better when features are on similar scales. Z-score standardization transforms features to have mean=0 and std=1.

### Solution
```python
import numpy as np

# Dataset: 100 samples, 4 features
np.random.seed(42)
X = np.random.randn(100, 4) * [10, 100, 0.1, 1000] + [50, 500, 5, 5000]

print(f"Before - Mean: {X.mean(axis=0).round(2)}")
print(f"Before - Std: {X.std(axis=0).round(2)}")

# Z-score standardization
mean = X.mean(axis=0)  # Mean per feature (column)
std = X.std(axis=0)    # Std per feature (column)

X_standardized = (X - mean) / std

print(f"\nAfter - Mean: {X_standardized.mean(axis=0).round(2)}")  # ~[0, 0, 0, 0]
print(f"After - Std: {X_standardized.std(axis=0).round(2)}")      # ~[1, 1, 1, 1]
```

### Important: Handling Test Data
```python
# Train set: calculate mean and std
X_train = X[:80]
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

# Standardize train
X_train_scaled = (X_train - train_mean) / train_std

# Test set: use TRAIN statistics!
X_test = X[80:]
X_test_scaled = (X_test - train_mean) / train_std
```

---

## Scenario 2

**Your dataset contains outliers. How would you use NumPy to cap extreme values at the 5th and 95th percentiles?**

### Solution
```python
import numpy as np

# Data with outliers
np.random.seed(42)
data = np.random.randn(1000)
data[0:5] = [100, -100, 50, -50, 75]  # Add outliers

print(f"Before - Min: {data.min():.2f}, Max: {data.max():.2f}")

# Calculate percentiles
p5 = np.percentile(data, 5)
p95 = np.percentile(data, 95)

print(f"5th percentile: {p5:.2f}")
print(f"95th percentile: {p95:.2f}")

# Clip values
data_capped = np.clip(data, p5, p95)

print(f"After - Min: {data_capped.min():.2f}, Max: {data_capped.max():.2f}")
```

### Alternative: Boolean Indexing
```python
data_capped = data.copy()
data_capped[data_capped < p5] = p5
data_capped[data_capped > p95] = p95
```

---

## Scenario 3

**You're implementing a simple linear regression from scratch. Use NumPy to compute the closed-form solution.**

### Solution
The closed-form solution: $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

```python
import numpy as np

# Generate sample data
np.random.seed(42)
n_samples = 100
X = np.random.rand(n_samples, 1) * 10  # Single feature

# True relationship: y = 3x + 5 + noise
y = 3 * X.squeeze() + 5 + np.random.randn(n_samples) * 2

# Add bias column (column of 1s)
X_bias = np.c_[np.ones(n_samples), X]  # Shape: (100, 2)

# Closed-form solution
# w = (X^T X)^(-1) X^T y
XtX = X_bias.T @ X_bias           # (2, 2)
XtX_inv = np.linalg.inv(XtX)      # (2, 2)
Xty = X_bias.T @ y                # (2,)
w = XtX_inv @ Xty                 # (2,)

print(f"Intercept (b): {w[0]:.2f}")  # ~5
print(f"Slope (w): {w[1]:.2f}")       # ~3

# Make predictions
y_pred = X_bias @ w
mse = np.mean((y - y_pred) ** 2)
print(f"MSE: {mse:.4f}")
```

---

## Scenario 4

**You need to implement one-hot encoding using only NumPy. How would you do it?**

### Solution
```python
import numpy as np

# Categorical labels (0, 1, 2 representing 3 classes)
labels = np.array([0, 1, 2, 1, 0, 2, 2, 0])

# Get number of classes
n_samples = len(labels)
n_classes = labels.max() + 1

# Method 1: Using np.eye
one_hot = np.eye(n_classes)[labels]

print(one_hot)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [1. 0. 0.]]

# Method 2: Manual creation
one_hot_manual = np.zeros((n_samples, n_classes))
one_hot_manual[np.arange(n_samples), labels] = 1
```

### Explanation
- `np.eye(n_classes)` creates identity matrix
- Indexing with `labels` selects rows corresponding to each class

---

## Scenario 5

**How would you implement the softmax function using NumPy?**

### Solution
Softmax: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$

```python
import numpy as np

def softmax(x):
    """
    Numerically stable softmax.
    Subtracting max prevents overflow.
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Single sample (logits)
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(f"Logits: {logits}")
print(f"Probabilities: {probs.round(3)}")  # [0.659, 0.242, 0.099]
print(f"Sum: {probs.sum():.3f}")           # 1.000

# Batch of samples
logits_batch = np.array([[2.0, 1.0, 0.1],
                          [1.0, 2.0, 3.0]])
probs_batch = softmax(logits_batch)
print(f"\nBatch probabilities:\n{probs_batch.round(3)}")
```

### Why subtract max?
- Without: `exp(1000)` causes overflow → inf
- With: `exp(1000 - 1000) = exp(0) = 1` → safe

---

## Scenario 6

**Implement a function to calculate cosine similarity between two vectors using NumPy.**

### Solution
Cosine similarity: $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \times ||\mathbf{b}||}$

```python
import numpy as np

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    Returns value between -1 and 1.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Example: word embeddings
vec1 = np.array([1, 2, 3])
vec2 = np.array([1, 2, 3.5])  # Similar
vec3 = np.array([-1, -2, -3]) # Opposite

print(f"Similarity (vec1, vec2): {cosine_similarity(vec1, vec2):.4f}")  # ~0.998
print(f"Similarity (vec1, vec3): {cosine_similarity(vec1, vec3):.4f}")  # -1.0
```

### For matrices (all pairs):
```python
def cosine_similarity_matrix(X):
    """Cosine similarity between all row pairs."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T
```

---

## Scenario 7

**You're implementing k-means clustering. How would you use NumPy to calculate distances between data points and centroids?**

### Solution
```python
import numpy as np

def euclidean_distances(X, centroids):
    """
    Calculate distances from each point to each centroid.
    X: (n_samples, n_features)
    centroids: (k, n_features)
    Returns: (n_samples, k)
    """
    # Method: Using broadcasting
    # X[:, np.newaxis] shape: (n_samples, 1, n_features)
    # centroids shape: (k, n_features)
    # Difference shape: (n_samples, k, n_features)
    diff = X[:, np.newaxis] - centroids
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return distances

# Example
np.random.seed(42)
X = np.random.randn(100, 2)  # 100 points, 2D
centroids = np.array([[0, 0], [3, 3], [-3, -3]])  # 3 centroids

distances = euclidean_distances(X, centroids)
print(f"Distances shape: {distances.shape}")  # (100, 3)

# Assign each point to nearest centroid
labels = np.argmin(distances, axis=1)
print(f"Labels: {labels[:10]}")  # First 10 assignments
```

---

## Scenario 8

**How would you implement a train/test split using only NumPy?**

### Solution
```python
import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split data into train and test sets.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Example usage
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {len(X_train)}")  # 80
print(f"Test size: {len(X_test)}")    # 20
```

---

## Scenario 9

**Implement batch gradient descent for linear regression using NumPy.**

### Solution
```python
import numpy as np

def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Perform batch gradient descent for linear regression.
    """
    n_samples, n_features = X.shape
    
    # Initialize weights
    weights = np.zeros(n_features)
    bias = 0
    
    losses = []
    
    for i in range(n_iterations):
        # Forward pass
        y_pred = X @ weights + bias
        
        # Compute loss (MSE)
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)
        
        # Compute gradients
        dw = (-2/n_samples) * (X.T @ (y - y_pred))
        db = (-2/n_samples) * np.sum(y - y_pred)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if i % 200 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")
    
    return weights, bias, losses

# Example
np.random.seed(42)
X = np.random.randn(100, 3)
true_weights = np.array([2, -1, 0.5])
y = X @ true_weights + 3 + np.random.randn(100) * 0.1

weights, bias, losses = batch_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)

print(f"\nLearned weights: {weights.round(3)}")  # ~[2, -1, 0.5]
print(f"Learned bias: {bias:.3f}")               # ~3
```

---

## Scenario 10

**Calculate precision, recall, and F1-score using only NumPy given predictions and true labels.**

### Solution
```python
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate precision, recall, and F1-score for binary classification.
    """
    # True positives, false positives, false negatives
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    
    # Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / len(y_true)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Example
y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1])
y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])

metrics = calculate_metrics(y_true, y_pred)
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")
```

### Confusion Matrix
```python
def confusion_matrix(y_true, y_pred):
    """Binary confusion matrix."""
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    
    return np.array([[TN, FP],
                     [FN, TP]])

cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{cm}")
#      Pred 0  Pred 1
# True 0: TN    FP
# True 1: FN    TP
```
