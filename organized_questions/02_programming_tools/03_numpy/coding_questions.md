# NumPy Interview Questions - Coding Questions

## Question 1

**How do you create a NumPy array from a Python list?**

### Solution
```python
import numpy as np

# From 1D list
py_list = [1, 2, 3, 4, 5]
arr_1d = np.array(py_list)
print(arr_1d)  # [1 2 3 4 5]

# From 2D list (nested)
py_list_2d = [[1, 2, 3],
              [4, 5, 6]]
arr_2d = np.array(py_list_2d)
print(arr_2d)
# [[1 2 3]
#  [4 5 6]]

# With specific dtype
arr_float = np.array([1, 2, 3], dtype=np.float32)
print(arr_float)  # [1. 2. 3.]
```

---

## Question 2

**How do you reshape a NumPy array?**

### Solution
```python
import numpy as np

arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape to 3x4
arr_3x4 = arr.reshape(3, 4)
print(arr_3x4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Use -1 for automatic dimension calculation
arr_auto = arr.reshape(4, -1)  # 4 rows, auto calculate cols (3)
print(arr_auto.shape)  # (4, 3)

# Flatten back to 1D
arr_flat = arr_3x4.flatten()    # Returns copy
arr_ravel = arr_3x4.ravel()     # Returns view (if possible)
```

### Important
- Total elements must match: `3 × 4 = 12`
- Use `-1` for one dimension to auto-calculate

---

## Question 3

**How do you slice and index NumPy arrays?**

### Solution
```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# --- Basic Slicing (returns VIEW) ---
# arr[row_start:row_end, col_start:col_end]
print(arr[0, :])      # First row: [1 2 3 4]
print(arr[:, 0])      # First column: [1 5 9]
print(arr[:2, 1:3])   # First 2 rows, cols 1-2
# [[2 3]
#  [6 7]]

# --- Boolean Indexing (returns COPY) ---
mask = arr > 5
print(arr[mask])      # [6 7 8 9 10 11 12]

# Conditional replacement
arr_copy = arr.copy()
arr_copy[arr_copy > 10] = 10  # Cap at 10

# --- Integer Array Indexing (returns COPY) ---
indices = [0, 2]
print(arr[indices])   # Rows 0 and 2
```

### Key Difference
- **Basic slicing → View** (modifications affect original)
- **Boolean/Integer indexing → Copy** (independent)

---

## Question 4

**How do you perform matrix multiplication in NumPy?**

### Solution
```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])    # Shape: (2, 2)

B = np.array([[5, 6],
              [7, 8]])    # Shape: (2, 2)

# Method 1: @ operator (recommended)
C = A @ B

# Method 2: np.dot()
C = np.dot(A, B)

# Method 3: np.matmul()
C = np.matmul(A, B)

print(C)
# [[19 22]
#  [43 50]]

# --- For vectors ---
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product (scalar result)
dot = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
```

### Warning: Don't confuse with element-wise multiplication!
```python
print(A * B)  # Element-wise: [[ 5 12], [21 32]]
print(A @ B)  # Matrix mult:  [[19 22], [43 50]]
```

---

## Question 5

**How do you concatenate and stack arrays?**

### Solution
```python
import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])

# --- Concatenate (join along existing axis) ---
# Vertically (axis=0) - adds rows
vert = np.concatenate([a, b], axis=0)
print(vert.shape)  # (4, 2)

# Horizontally (axis=1) - adds columns
horiz = np.concatenate([a, b], axis=1)
print(horiz.shape)  # (2, 4)

# --- Stack (creates new axis) ---
# Stacks along new first axis
stacked = np.stack([a, b], axis=0)
print(stacked.shape)  # (2, 2, 2)

# --- Shortcuts ---
np.vstack([a, b])  # Same as concatenate axis=0
np.hstack([a, b])  # Same as concatenate axis=1
```

---

## Question 6

**How do you generate random numbers with NumPy?**

### Solution
```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# --- Uniform distribution [0, 1) ---
uniform = np.random.rand(3, 4)      # Shape (3, 4)

# --- Normal distribution (mean=0, std=1) ---
normal = np.random.randn(3, 4)      # Shape (3, 4)

# --- Integers ---
integers = np.random.randint(0, 10, size=(3, 4))  # [0, 10)

# --- From specific distribution ---
# Normal with custom mean and std
custom_normal = np.random.normal(loc=50, scale=10, size=(100,))

# Uniform with custom range
custom_uniform = np.random.uniform(low=5, high=15, size=(100,))

# --- Shuffle and choice ---
arr = np.arange(10)
np.random.shuffle(arr)              # In-place shuffle
sample = np.random.choice(arr, 3)   # Random sample of 3
```

### New Generator API (Recommended)
```python
rng = np.random.default_rng(seed=42)
uniform = rng.random((3, 4))
normal = rng.standard_normal((3, 4))
integers = rng.integers(0, 10, size=(3, 4))
```

---

## Question 7

**How do you find unique values and their counts?**

### Solution
```python
import numpy as np

arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

# --- Unique values ---
unique = np.unique(arr)
print(unique)  # [1 2 3 4]

# --- Unique with counts ---
unique, counts = np.unique(arr, return_counts=True)
print(dict(zip(unique, counts)))  # {1: 1, 2: 2, 3: 3, 4: 4}

# --- Unique with indices ---
unique, indices = np.unique(arr, return_index=True)
print(indices)  # [0 1 3 6] - first occurrence indices
```

---

## Question 8

**How do you sort arrays in NumPy?**

### Solution
```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# --- Sort (returns copy) ---
sorted_arr = np.sort(arr)
print(sorted_arr)  # [1 1 2 3 4 5 6 9]

# --- In-place sort ---
arr.sort()  # Modifies original

# --- Get sorted indices (argsort) ---
arr = np.array([3, 1, 4, 1, 5])
indices = np.argsort(arr)
print(indices)         # [1 3 0 2 4]
print(arr[indices])    # Sorted: [1 1 3 4 5]

# --- 2D array sorting ---
arr_2d = np.array([[3, 1, 2],
                   [6, 4, 5]])

# Sort each row
print(np.sort(arr_2d, axis=1))
# [[1 2 3]
#  [4 5 6]]

# Sort each column
print(np.sort(arr_2d, axis=0))
# [[3 1 2]
#  [6 4 5]]
```

---

## Question 9

**How do you implement standardization (Z-score normalization)?**

### Solution
```python
import numpy as np

# Sample data: 5 samples, 3 features
X = np.array([[1, 200, 0.5],
              [2, 300, 0.8],
              [3, 250, 0.6],
              [4, 350, 0.9],
              [5, 400, 0.7]])

# Calculate mean and std for each feature (column)
mean = X.mean(axis=0)
std = X.std(axis=0)

# Standardize: Z = (X - mean) / std
X_standardized = (X - mean) / std

print(f"Mean before: {X.mean(axis=0)}")
print(f"Mean after: {X_standardized.mean(axis=0).round(2)}")  # [0, 0, 0]
print(f"Std after: {X_standardized.std(axis=0).round(2)}")    # [1, 1, 1]
```

---

## Question 10

**How do you implement Min-Max normalization?**

### Solution
```python
import numpy as np

X = np.array([[1, 200],
              [2, 300],
              [3, 400],
              [4, 500],
              [5, 600]])

# Calculate min and max per feature
X_min = X.min(axis=0)
X_max = X.max(axis=0)

# Normalize to [0, 1]
X_normalized = (X - X_min) / (X_max - X_min)

print(X_normalized)
# [[0.   0.  ]
#  [0.25 0.25]
#  [0.5  0.5 ]
#  [0.75 0.75]
#  [1.   1.  ]]

# Verify range is [0, 1]
print(f"Min: {X_normalized.min(axis=0)}")  # [0, 0]
print(f"Max: {X_normalized.max(axis=0)}")  # [1, 1]
```

---

## Question 11

**How do you compute the dot product and cross product?**

### Solution
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# --- Dot Product (scalar) ---
# Formula: a·b = a1*b1 + a2*b2 + a3*b3
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot product: {dot}")

# Alternative: a @ b

# --- Cross Product (vector, 3D only) ---
# Returns vector perpendicular to both inputs
cross = np.cross(a, b)
print(f"Cross product: {cross}")  # [-3, 6, -3]
```

---

## Question 12

**How do you compute eigenvalues and eigenvectors?**

### Solution
```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

# Get eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")     # [5. 2.]
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: A @ v = λ * v
v = eigenvectors[:, 0]  # First eigenvector
lambda_1 = eigenvalues[0]

print(f"A @ v: {A @ v}")
print(f"λ * v: {lambda_1 * v}")
# Both should be equal
```

### Use Case in ML
- PCA (Principal Component Analysis) uses eigendecomposition
- Eigenvectors of covariance matrix = principal components
