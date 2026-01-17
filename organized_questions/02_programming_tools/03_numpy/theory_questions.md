# NumPy Interview Questions - Theory Questions

## Question 1

**What is NumPy, and why is it important in Machine Learning?**

### Definition
NumPy (Numerical Python) is the foundational library for numerical computing in Python. Its core feature is the **ndarray** - a fast, memory-efficient multi-dimensional array object.

### Why NumPy is Important for ML

**1. Performance**
- **Vectorization**: Operations on entire arrays without Python loops
- **C Backend**: Operations implemented in optimized C/Fortran code
- **10-100x faster** than Python lists for numerical operations

**2. Universal Data Format**
- All ML libraries (Pandas, Scikit-learn, TensorFlow, PyTorch) use NumPy arrays
- Dataset → 2D array (samples × features)
- Image → 3D array (height × width × channels)
- Model weights → ndarray

**3. Mathematical Tools**
- Linear algebra (matrix multiplication, SVD, eigenvalues)
- Statistics (mean, std, variance)
- Random number generation (weight initialization)

---

## Question 2

**Explain how NumPy arrays differ from Python lists.**

### Comparison Table

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Data Types | Heterogeneous (mixed) | Homogeneous (same type) |
| Memory | Pointers to objects | Contiguous block |
| Performance | Slow (Python loops) | Fast (vectorized C) |
| Functionality | General purpose | Numerical computing |

### Code Example
```python
import numpy as np
import sys

# Memory comparison
py_list = list(range(1000))
np_array = np.arange(1000)

print(f"List size: {sys.getsizeof(py_list)} bytes")   # ~9000 bytes
print(f"Array size: {np_array.nbytes} bytes")          # ~8000 bytes

# Performance: NumPy is 10-100x faster
# Python: [x*2 for x in py_list]  
# NumPy:  np_array * 2  (vectorized)
```

---

## Question 3

**What are the main attributes of a NumPy ndarray?**

### Key Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `ndim` | Number of dimensions | `2` for matrix |
| `shape` | Size of each dimension | `(3, 4)` = 3 rows, 4 cols |
| `size` | Total elements | `12` (3 × 4) |
| `dtype` | Data type | `float64`, `int32` |
| `itemsize` | Bytes per element | `8` for float64 |
| `nbytes` | Total bytes | `96` (12 × 8) |
| `T` | Transpose | Swaps axes |

### Code Example
```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]], dtype=np.float32)

print(f"ndim: {arr.ndim}")        # 2
print(f"shape: {arr.shape}")      # (2, 4)
print(f"size: {arr.size}")        # 8
print(f"dtype: {arr.dtype}")      # float32
print(f"itemsize: {arr.itemsize}")  # 4 bytes
print(f"nbytes: {arr.nbytes}")    # 32 bytes
```

---

## Question 4

**Explain the concept of broadcasting in NumPy.**

### Definition
Broadcasting allows operations on arrays of different shapes by "stretching" smaller arrays to match larger ones (virtually, without copying data).

### Broadcasting Rules
Compare shapes from right to left. Dimensions are compatible if:
1. They are equal, OR
2. One of them is 1

### Examples
```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])  # Shape (3, 3)

b = np.array([10, 20, 30])  # Shape (3,)

# b is broadcast across each row
result = A + b
# [[11, 22, 33],
#  [14, 25, 36],
#  [17, 28, 39]]

# Column vector broadcast
c = np.array([[100], [200], [300]])  # Shape (3, 1)
result2 = A + c  # c broadcast across columns
```

### Shape Compatibility
```
A: (3, 3)       A: (4, 3)       A: (3, 3)
b: (3,)    ✓    b: (3,)    ✓    b: (4,)    ✗
```

---

## Question 5

**What are the data types (dtypes) supported by NumPy?**

### Common dtypes

| Category | Types | Notes |
|----------|-------|-------|
| Integers | `int8`, `int16`, `int32`, `int64` | Signed |
| Unsigned | `uint8`, `uint16`, `uint32`, `uint64` | Non-negative (images use uint8) |
| Floats | `float16`, `float32`, `float64` | float32 for GPU/ML |
| Boolean | `bool_` | True/False |
| Complex | `complex64`, `complex128` | Real + imaginary |

### Code Example
```python
import numpy as np

# Specify dtype at creation
arr_int8 = np.array([1, 2, 3], dtype=np.int8)
arr_float32 = np.array([1.0, 2.0], dtype=np.float32)

# Type casting
arr = np.array([1, 2, 3])
arr_float = arr.astype(np.float64)
```

### ML Tip
Use `float32` instead of `float64` to save memory and speed up GPU training.

---

## Question 6

**What is the difference between a deep copy and a shallow copy (view)?**

### Comparison

| Type | Creates New Data? | Affects Original? |
|------|------------------|-------------------|
| Assignment (`b = a`) | No | Yes (same object) |
| View/Slice (`b = a[1:3]`) | No | Yes (shares data) |
| Copy (`b = a.copy()`) | Yes | No (independent) |

### Code Example
```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

# View (shallow copy) - shares memory
view = a[1:4]
view[0] = 99
print(a)  # [1, 99, 3, 4, 5] - original changed!

# Copy (deep copy) - independent
a = np.array([1, 2, 3, 4, 5])
copy = a.copy()
copy[0] = 77
print(a)  # [1, 2, 3, 4, 5] - original unchanged
```

### Check if View
```python
print(view.base is a)  # True = view
print(copy.base)       # None = copy
```

---

## Question 7

**What are universal functions (ufuncs) in NumPy?**

### Definition
Ufuncs are vectorized functions that operate element-by-element on arrays, implemented in fast C code.

### Types

| Type | Examples | Description |
|------|----------|-------------|
| Unary | `np.sqrt()`, `np.exp()`, `np.sin()` | One input array |
| Binary | `np.add()`, `np.multiply()`, `np.maximum()` | Two input arrays |

### Code Example
```python
import numpy as np

a = np.array([1, 4, 9, 16])

# Unary ufuncs
print(np.sqrt(a))  # [1., 2., 3., 4.]
print(np.exp(a))   # Exponential

# Binary ufuncs
b = np.array([1, 2, 3, 4])
print(np.add(a, b))      # Same as a + b
print(np.maximum(a, b))  # Element-wise max
```

---

## Question 8

**What is the use of the axis parameter in NumPy functions?**

### Definition
`axis` specifies which dimension to operate along. That dimension is "collapsed."

### For 2D Arrays
- `axis=0`: Down rows (column-wise) → result has shape (n_cols,)
- `axis=1`: Across columns (row-wise) → result has shape (n_rows,)
- `axis=None`: All elements → scalar

### Code Example
```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])  # Shape (2, 3)

print(arr.sum())         # 21 (all elements)
print(arr.sum(axis=0))   # [5, 7, 9] (sum each column)
print(arr.sum(axis=1))   # [6, 15] (sum each row)

# Mean per column (useful for feature normalization)
print(arr.mean(axis=0))  # [2.5, 3.5, 4.5]
```

### ML Use Case
```python
# Normalize features (subtract column mean)
X_normalized = X - X.mean(axis=0)
```

