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

---

## Scenario 11

**Discuss the performance benefits of using NumPy's in-place operations.**

**Answer:**

In-place operations modify arrays directly without allocating new memory, providing significant performance benefits.

### Memory & Speed Comparison

```python
import numpy as np

arr = np.random.rand(10_000_000)

# Out-of-place (creates new array)
# arr = arr * 2  # Allocates new 80MB array, copies, deallocates old

# In-place (modifies existing array)
# arr *= 2       # No allocation, modifies memory directly
# np.multiply(arr, 2, out=arr)  # Explicit in-place
```

### In-place Operations

```python
# Arithmetic
arr += 5           # np.add(arr, 5, out=arr)
arr -= 3           # np.subtract(arr, 3, out=arr)
arr *= 2           # np.multiply(arr, 2, out=arr)
arr /= 4           # np.divide(arr, 4, out=arr)

# Element assignment
arr[arr > 0.5] = 0       # In-place conditional
np.clip(arr, 0, 1, out=arr)  # In-place clipping
np.sqrt(arr, out=arr)         # In-place sqrt

# Sorting
arr.sort()                     # In-place (vs np.sort() which copies)
```

### Performance Benefits

| Aspect | Out-of-place | In-place |
|--------|-------------|----------|
| Memory allocation | New array each time | Zero allocation |
| Garbage collection | Triggers GC | No GC overhead |
| Cache efficiency | Poor (new memory) | Excellent (same cache lines) |
| Speed (10M elements) | ~30ms | ~15ms |
| Memory usage | 2× array size | 1× array size |

### When NOT to Use

```python
# Danger: views share memory
a = np.array([1, 2, 3, 4])
b = a[1:3]   # View of a
b += 10       # Also modifies a! a = [1, 12, 13, 4]

# Safe: ensure independent copy first
b = a[1:3].copy()
b += 10  # Only modifies b
```

> **Interview Tip:** In-place ops are crucial in **deep learning** (gradient updates, normalization) and **real-time systems** where allocation latency matters. Always be aware of **views** sharing memory.

---

## Scenario 12

**How would you use NumPy to process image data for a convolutional neural network?**

**Answer:**

```python
import numpy as np

# === 1. Load & Represent Images ===
# Images as NumPy arrays: (H, W, C) or batches: (N, H, W, C)
image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
batch = np.random.randint(0, 256, (32, 224, 224, 3), dtype=np.uint8)

# === 2. Preprocessing ===

# Normalize to [0, 1]
image_norm = image.astype(np.float32) / 255.0

# Standardize (ImageNet mean/std)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_std = (image_norm - mean) / std

# Resize (simple nearest neighbor)
def resize_image(img, new_h, new_w):
    h, w = img.shape[:2]
    row_idx = (np.arange(new_h) * h / new_h).astype(int)
    col_idx = (np.arange(new_w) * w / new_w).astype(int)
    return img[np.ix_(row_idx, col_idx)]

# === 3. Channel Format Conversion ===
# HWC (channels last) -> CHW (channels first, PyTorch format)
image_chw = np.transpose(image, (2, 0, 1))  # (3, 224, 224)

# Batch: NHWC -> NCHW
batch_chw = np.transpose(batch, (0, 3, 1, 2))  # (32, 3, 224, 224)

# === 4. Data Augmentation ===
# Horizontal flip
flipped = image[:, ::-1, :]

# Random crop
def random_crop(img, crop_h, crop_w):
    h, w = img.shape[:2]
    top = np.random.randint(0, h - crop_h)
    left = np.random.randint(0, w - crop_w)
    return img[top:top+crop_h, left:left+crop_w]

# Add Gaussian noise
noisy = image_norm + np.random.normal(0, 0.05, image_norm.shape)
noisy = np.clip(noisy, 0, 1)

# === 5. Convolution Operation ===
def conv2d_numpy(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))
    h, w, c = image.shape
    kh, kw = kernel.shape[:2]
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            region = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(region * kernel)
    return output
```

> **Interview Tip:** For production, use PIL/OpenCV for loading and augmentation. NumPy handles the **array manipulation** (transpose, normalize, batch). Know the difference between **NHWC** (TensorFlow) and **NCHW** (PyTorch) formats.

---

## Scenario 13

**Discuss the role of NumPy in managing data for training a machine learning model.**

**Answer:**

NumPy serves as the **foundational data layer** for virtually all ML frameworks.

### Data Pipeline

```python
import numpy as np

# === 1. Data Loading & Storage ===
# Load from CSV
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1]  # Features and labels

# === 2. Train/Test Split ===
def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# === 3. Feature Preprocessing ===
# Standardization
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train_std = (X_train - mean) / (std + 1e-8)
X_test_std = (X_test - mean) / (std + 1e-8)  # Use training stats!

# One-hot encoding
def one_hot(labels, n_classes):
    return np.eye(n_classes)[labels.astype(int)]

# === 4. Batch Generation ===
def batch_generator(X, y, batch_size=32, shuffle=True):
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], y[idx]

# === 5. During Training ===
# Matrix operations (forward pass)
z = X_batch @ weights + bias    # Linear transformation
a = 1 / (1 + np.exp(-z))        # Sigmoid activation
loss = -np.mean(y * np.log(a) + (1-y) * np.log(1-a))  # Cross-entropy

# Gradient computation (backward pass)
dw = (1/m) * X_batch.T @ (a - y)
db = (1/m) * np.sum(a - y)

# === 6. Evaluation ===
predictions = (a > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
confusion = np.array([
    [np.sum((predictions == 0) & (y_test == 0)), np.sum((predictions == 1) & (y_test == 0))],
    [np.sum((predictions == 0) & (y_test == 1)), np.sum((predictions == 1) & (y_test == 1))]
])
```

| ML Stage | NumPy Role |
|----------|------------|
| Data loading | `genfromtxt`, `loadtxt`, `load` |
| Preprocessing | Normalization, encoding, imputation |
| Training | Matrix math, gradient computation |
| Batching | Indexing, shuffling, slicing |
| Evaluation | Metrics calculation |

> **Interview Tip:** NumPy is used **under the hood** by Scikit-learn, TensorFlow, and PyTorch. Understanding NumPy operations means understanding what these frameworks do internally.

---

## Scenario 14

**Discuss the potential issues when importing large datasets into NumPy arrays.**

**Answer:**

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Memory overflow** | Array exceeds RAM | Use `memmap`, chunked loading, or Dask |
| **Slow loading** | Large CSV parsing | Use binary formats (`.npy`, `.npz`, HDF5) |
| **dtype explosion** | Default float64 | Specify `dtype=np.float32` to halve memory |
| **String columns** | Mixed types | Separate numeric/string data |
| **Missing values** | NaN in int arrays | Use float dtype or masked arrays |

### Memory Calculation

```python
import numpy as np

# Memory usage: rows × cols × bytes_per_element
rows, cols = 10_000_000, 100
memory_f64 = rows * cols * 8 / (1024**3)   # 7.45 GB (float64)
memory_f32 = rows * cols * 4 / (1024**3)   # 3.73 GB (float32)
memory_f16 = rows * cols * 2 / (1024**3)   # 1.86 GB (float16)
```

### Solutions

```python
# 1. Memory-mapped files (load on demand)
mmap = np.memmap('data.dat', dtype='float32', mode='r', shape=(10_000_000, 100))
chunk = mmap[0:1000]  # Only loads 1000 rows into RAM

# 2. Chunked loading
def load_chunked(filename, chunk_size=10000):
    chunks = []
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        chunks.append(chunk.values.astype(np.float32))
    return np.vstack(chunks)

# 3. Use smaller dtypes
data = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)  # Not float64

# 4. Binary format (10-100× faster than CSV)
np.save('data.npy', data)       # Save binary
data = np.load('data.npy')      # Load binary (instant)

# 5. HDF5 for very large data
import h5py
with h5py.File('data.h5', 'w') as f:
    f.create_dataset('features', data=X, chunks=True, compression='gzip')

# 6. Dask for out-of-core computation
import dask.array as da
dask_arr = da.from_npy_stack('data/')  # Lazy loading
result = dask_arr.mean(axis=0).compute()  # Compute on chunks
```

> **Interview Tip:** Always check dataset size vs available RAM first. Use `float32` instead of `float64` (sufficient for most ML). For datasets larger than RAM, use **memory-mapped files**, **HDF5**, or **Dask**.

---

## Scenario 15

**Discuss the use of NumPy for operations on polynomials.**

**Answer:**

NumPy provides comprehensive polynomial support through two APIs.

### Legacy API: np.poly1d

```python
import numpy as np

# Create polynomial: 2x³ + 3x² - x + 5
p = np.poly1d([2, 3, -1, 5])  # Coefficients highest to lowest degree

# Evaluate
p(2)       # 2(8) + 3(4) - 2 + 5 = 31
p([1, 2, 3])  # Evaluate at multiple points

# Operations
q = np.poly1d([1, -1])  # x - 1
print(p + q)     # Addition
print(p * q)     # Multiplication
print(p.deriv()) # Derivative: 6x² + 6x - 1
print(p.integ()) # Integral: 0.5x⁴ + x³ - 0.5x² + 5x

# Roots
roots = np.roots([1, -3, 2])  # x² - 3x + 2 = 0  → [2, 1]

# Polynomial from roots
np.poly([2, 1])  # [1, -3, 2] — coefficients of (x-2)(x-1)
```

### Modern API: np.polynomial (Preferred)

```python
from numpy.polynomial import polynomial as P

# Coefficients: lowest to highest degree
coeffs = [5, -1, 3, 2]  # 5 - x + 3x² + 2x³

# Evaluate
P.polyval(2, coeffs)  # 31

# Fit polynomial to data (polynomial regression)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1, 2.5, 7, 15, 30, 50])

# Fit degree-2 polynomial
coeffs = np.polyfit(x, y, deg=2)
poly_fn = np.poly1d(coeffs)
y_pred = poly_fn(x)

# Using modern API
from numpy.polynomial import Polynomial
p = Polynomial.fit(x, y, deg=2)
y_pred = p(x)

# Chebyshev polynomials (better numerical stability)
from numpy.polynomial import chebyshev as C
coeffs_cheb = C.chebfit(x, y, deg=3)
y_pred = C.chebval(x, coeffs_cheb)
```

### ML Application: Polynomial Features

```python
# Create polynomial features for linear regression
def polynomial_features(X, degree):
    n_samples = X.shape[0]
    features = [np.ones(n_samples)]  # Bias
    for d in range(1, degree + 1):
        features.append(X ** d)
    return np.column_stack(features)

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
X_poly = polynomial_features(X.ravel(), degree=3)
# [[1, 1, 1, 1], [1, 2, 4, 8], [1, 3, 9, 27], ...]
```

> **Interview Tip:** Use the **modern `np.polynomial`** API (lowest-degree-first coefficients, better numerical stability). `np.polyfit` is still commonly used for polynomial regression. High-degree polynomials overfit — use regularization.

## Question 1

**How would you use NumPy to process image data for a convolutional neural network ?**

*Answer to be added.*

---

## Question 2

**Discuss the role of NumPy in managing data for training a machine learning model**

*Answer to be added.*

---

## Question 3

**Discuss the potential issues when importing large datasets into NumPy arrays**

*Answer to be added.*

---
