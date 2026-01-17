# NumPy Interview Questions - General Questions

## Question 1

**How do you inspect the shape and size of a NumPy array?**

### Key Attributes

| Attribute | Returns | Example for (3, 4) array |
|-----------|---------|--------------------------|
| `arr.shape` | Tuple of dimensions | `(3, 4)` |
| `arr.size` | Total elements | `12` |
| `arr.ndim` | Number of dimensions | `2` |

### Code Example
```python
import numpy as np

arr = np.arange(24).reshape(2, 3, 4)  # 3D array

print(f"Shape: {arr.shape}")     # (2, 3, 4)
print(f"Size: {arr.size}")       # 24
print(f"Dimensions: {arr.ndim}") # 3
```

---

## Question 2

**How do you perform element-wise operations in NumPy?**

### Definition
Element-wise operations apply to each element independently. Use arithmetic operators or ufuncs.

### Code Example
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# Using operators (element-wise)
print(a + b)   # [11, 22, 33]
print(a * b)   # [10, 40, 90]
print(a ** 2)  # [1, 4, 9]

# Using ufuncs
print(np.add(a, b))      # [11, 22, 33]
print(np.sqrt(a))        # [1.0, 1.41, 1.73]
print(np.exp(a))         # [2.72, 7.39, 20.09]

# With scalar (broadcasting)
print(a * 10)  # [10, 20, 30]
```

---

## Question 3

**How do you compute mean, median, and standard deviation?**

### Key Functions

| Function | Description |
|----------|-------------|
| `np.mean(arr)` | Arithmetic average |
| `np.median(arr)` | Middle value (robust to outliers) |
| `np.std(arr)` | Standard deviation |

### Code Example
```python
import numpy as np

data = np.array([[1, 2, 3],
                 [4, 5, 6]])

# Overall statistics
print(f"Mean: {np.mean(data)}")      # 3.5
print(f"Median: {np.median(data)}")  # 3.5
print(f"Std: {np.std(data)}")        # 1.71

# Axis-wise (important for ML)
print(f"Mean per column: {data.mean(axis=0)}")  # [2.5, 3.5, 4.5]
print(f"Mean per row: {data.mean(axis=1)}")     # [2.0, 5.0]
```

### ML Use Case
Feature normalization: `X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)`

---

## Question 4

**Why is NumPy more efficient than pure Python for numerical computations?**

### Four Key Reasons

| Reason | Python List | NumPy Array |
|--------|-------------|-------------|
| **Implementation** | Python interpreter | Compiled C/Fortran |
| **Operations** | Explicit for loops | Vectorized (single call to C) |
| **Memory** | Pointers to objects | Contiguous block |
| **Type Checking** | Every operation | Once per array |

### Performance Example
```python
import numpy as np
import time

size = 1_000_000
py_list = list(range(size))
np_arr = np.arange(size)

# Python: ~200ms
start = time.time()
[x * 2 for x in py_list]
print(f"Python: {time.time() - start:.4f}s")

# NumPy: ~2ms (100x faster!)
start = time.time()
np_arr * 2
print(f"NumPy: {time.time() - start:.4f}s")
```

---

## Question 5

**How do you check the memory size of a NumPy array?**

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| `arr.nbytes` | Total bytes for data |
| `arr.itemsize` | Bytes per element |
| `arr.size` | Number of elements |

### Code Example
```python
import numpy as np

arr = np.zeros((1000, 1000), dtype=np.float64)

print(f"Total bytes: {arr.nbytes}")           # 8,000,000
print(f"Bytes per element: {arr.itemsize}")   # 8
print(f"Elements: {arr.size}")                # 1,000,000

# Verify: nbytes = size × itemsize
assert arr.nbytes == arr.size * arr.itemsize

# Convert to MB
print(f"Memory: {arr.nbytes / (1024**2):.2f} MB")  # 7.63 MB
```

### Memory Optimization Tip
```python
# float64 (default): 8 bytes
arr64 = np.zeros(1000000, dtype=np.float64)
print(f"float64: {arr64.nbytes / 1e6:.1f} MB")  # 8.0 MB

# float32: 4 bytes (half the memory!)
arr32 = np.zeros(1000000, dtype=np.float32)
print(f"float32: {arr32.nbytes / 1e6:.1f} MB")  # 4.0 MB
```

---

## Question 6

**How do you handle NaN or infinite values in NumPy?**

### Detection Functions

| Function | Detects |
|----------|---------|
| `np.isnan(arr)` | NaN values |
| `np.isinf(arr)` | Infinite values |
| `np.isfinite(arr)` | Finite (not NaN or inf) |

### NaN-Safe Functions

| Standard | NaN-Safe | Description |
|----------|----------|-------------|
| `np.mean()` | `np.nanmean()` | Mean ignoring NaN |
| `np.sum()` | `np.nansum()` | Sum treating NaN as 0 |
| `np.std()` | `np.nanstd()` | Std ignoring NaN |
| `np.min()` | `np.nanmin()` | Min ignoring NaN |

### Code Example
```python
import numpy as np

arr = np.array([1.0, 2.0, np.nan, 4.0, np.inf])

# Detection
print(np.isnan(arr))     # [False, False, True, False, False]
print(np.isnan(arr).sum())  # Count: 1

# NaN-safe computation
print(np.mean(arr))      # nan (propagates)
print(np.nanmean(arr))   # 2.33 (ignores NaN)

# Replacement
arr_clean = arr.copy()
arr_clean[np.isnan(arr_clean)] = 0  # Replace NaN with 0
# OR
arr_clean = np.nan_to_num(arr)      # Replaces NaN→0, inf→large number
```

### Important
```python
# DON'T use == to check for NaN
np.nan == np.nan  # Returns False!

# DO use np.isnan()
np.isnan(np.nan)  # Returns True
```

