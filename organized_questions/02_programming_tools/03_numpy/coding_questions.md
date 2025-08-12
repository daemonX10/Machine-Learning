# Numpy Interview Questions - Coding Questions

## Question 1

**How do you create a NumPy array from a regular Python list?**

**Answer:**

### Theory
Creating NumPy arrays from Python lists is the most common way to convert existing Python data into NumPy's optimized array format. NumPy provides several methods for this conversion, each with specific use cases and performance characteristics.

### Basic Array Creation

#### Code Example

```python
import numpy as np

print("=== Creating NumPy Arrays from Python Lists ===")

# Basic 1D array creation
python_list = [1, 2, 3, 4, 5]
numpy_array = np.array(python_list)

print(f"Python list: {python_list}")
print(f"NumPy array: {numpy_array}")
print(f"Array dtype: {numpy_array.dtype}")
print(f"Array shape: {numpy_array.shape}")
```

#### 1. **Different Data Types**

```python
print("\n=== Different Data Types ===")

# Integer list
int_list = [1, 2, 3, 4, 5]
int_array = np.array(int_list)
print(f"Integer array: {int_array}, dtype: {int_array.dtype}")

# Float list
float_list = [1.1, 2.2, 3.3, 4.4, 5.5]
float_array = np.array(float_list)
print(f"Float array: {float_array}, dtype: {float_array.dtype}")

# Mixed types (automatic casting)
mixed_list = [1, 2.5, 3, 4.7]
mixed_array = np.array(mixed_list)
print(f"Mixed array: {mixed_array}, dtype: {mixed_array.dtype}")

# String list
string_list = ['apple', 'banana', 'cherry']
string_array = np.array(string_list)
print(f"String array: {string_array}, dtype: {string_array.dtype}")

# Boolean list
bool_list = [True, False, True, False]
bool_array = np.array(bool_list)
print(f"Boolean array: {bool_array}, dtype: {bool_array.dtype}")
```

#### 2. **Multidimensional Arrays**

```python
print("\n=== Multidimensional Arrays ===")

# 2D array from nested lists
list_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
array_2d = np.array(list_2d)
print(f"2D list: {list_2d}")
print(f"2D array:\n{array_2d}")
print(f"Shape: {array_2d.shape}")

# 3D array from nested lists
list_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
array_3d = np.array(list_3d)
print(f"3D array:\n{array_3d}")
print(f"Shape: {array_3d.shape}")

# Irregular nested lists (creates object array)
irregular_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
irregular_array = np.array(irregular_list, dtype=object)
print(f"Irregular array: {irregular_array}")
print(f"Dtype: {irregular_array.dtype}")
```

#### 3. **Explicit Data Type Specification**

```python
print("\n=== Explicit Data Type Specification ===")

# Specify dtype during creation
int_list = [1, 2, 3, 4, 5]

# Different integer types
int8_array = np.array(int_list, dtype=np.int8)
int32_array = np.array(int_list, dtype=np.int32)
int64_array = np.array(int_list, dtype=np.int64)

print(f"int8 array: {int8_array}, dtype: {int8_array.dtype}")
print(f"int32 array: {int32_array}, dtype: {int32_array.dtype}")
print(f"int64 array: {int64_array}, dtype: {int64_array.dtype}")

# Float types
float_list = [1.1, 2.2, 3.3]
float32_array = np.array(float_list, dtype=np.float32)
float64_array = np.array(float_list, dtype=np.float64)

print(f"float32 array: {float32_array}, dtype: {float32_array.dtype}")
print(f"float64 array: {float64_array}, dtype: {float64_array.dtype}")

# Complex numbers
complex_list = [1+2j, 3+4j, 5+6j]
complex_array = np.array(complex_list)
print(f"Complex array: {complex_array}, dtype: {complex_array.dtype}")
```

#### 4. **Alternative Creation Methods**

```python
print("\n=== Alternative Creation Methods ===")

# Using np.asarray (no copy if already array)
python_list = [1, 2, 3, 4, 5]
array1 = np.array(python_list)      # Always creates new array
array2 = np.asarray(python_list)    # Creates array if needed

print(f"np.array result: {array1}")
print(f"np.asarray result: {array2}")
print(f"Same data: {np.array_equal(array1, array2)}")

# Performance difference with existing arrays
existing_array = np.array([1, 2, 3, 4, 5])
copy_array = np.array(existing_array)     # Creates copy
view_array = np.asarray(existing_array)   # Returns same array

print(f"Copy shares memory: {np.shares_memory(existing_array, copy_array)}")
print(f"Asarray shares memory: {np.shares_memory(existing_array, view_array)}")

# Using np.fromiter for iterators
def number_generator():
    for i in range(5):
        yield i ** 2

iter_array = np.fromiter(number_generator(), dtype=int)
print(f"From iterator: {iter_array}")

# List of lists to array
list_of_lists = [[1, 2], [3, 4], [5, 6]]
concatenated = np.array([item for sublist in list_of_lists for item in sublist])
print(f"Flattened: {concatenated}")
```

#### Explanation

**Key Methods:**
1. **np.array()**: Most common method, always creates new array
2. **np.asarray()**: Creates array only if input isn't already an array
3. **np.fromiter()**: Efficient for iterator objects
4. **dtype parameter**: Explicit control over data type

**Automatic Type Detection:**
- NumPy automatically determines the most appropriate dtype
- Mixed types promote to most general type (int + float → float)
- All elements must fit the chosen dtype

#### Use Cases

**Data Import:**
- Converting parsed data from files
- Processing user input
- Integration with other Python libraries

**Data Science:**
- Converting pandas data to NumPy format
- Preprocessing data for machine learning
- Creating test datasets

#### Best Practices

1. **Specify dtype explicitly** when you know the required precision
2. **Use np.asarray()** when input might already be an array
3. **Be aware of memory implications** for large lists
4. **Consider dtype promotion** with mixed-type data
5. **Validate data consistency** before conversion

#### Pitfalls

```python
print("\n=== Common Pitfalls ===")

# Pitfall 1: Unintended dtype promotion
int_list = [1, 2, 3, 4, 5]
float_value = 3.14
mixed_with_float = int_list + [float_value]
mixed_array = np.array(mixed_with_float)
print(f"Unintended float promotion: {mixed_array.dtype}")

# Pitfall 2: Memory usage with nested lists
large_nested = [[1, 2, 3] for _ in range(1000)]
# This creates many intermediate Python objects before conversion
large_array = np.array(large_nested)
print(f"Large array shape: {large_array.shape}")

# Better approach for large data
better_array = np.zeros((1000, 3), dtype=int)
for i in range(1000):
    better_array[i] = [1, 2, 3]

# Pitfall 3: Irregular nested lists
try:
    irregular = [[1, 2, 3], [4, 5], [6]]
    regular_array = np.array(irregular)  # This works but creates object array
    print(f"Irregular becomes object array: {regular_array.dtype}")
except ValueError as e:
    print(f"Error with irregular lists: {e}")

# Pitfall 4: Large integer overflow
large_int_list = [2**63 - 1, 2**63]  # Second value overflows int64
try:
    overflow_array = np.array(large_int_list, dtype=np.int64)
    print(f"Overflow result: {overflow_array}")
except OverflowError as e:
    print(f"Overflow error: {e}")
```

#### Optimization Tips

```python
print("\n=== Optimization Tips ===")

import time

# Performance comparison for large lists
size = 1000000
large_list = list(range(size))

# Method 1: Direct conversion
start = time.time()
array1 = np.array(large_list)
time1 = time.time() - start

# Method 2: Pre-allocate and fill
start = time.time()
array2 = np.empty(size, dtype=int)
array2[:] = large_list
time2 = time.time() - start

# Method 3: Using np.arange (when applicable)
start = time.time()
array3 = np.arange(size)
time3 = time.time() - start

print(f"Direct conversion: {time1:.6f}s")
print(f"Pre-allocate and fill: {time2:.6f}s")
print(f"np.arange: {time3:.6f}s")

# Memory-efficient creation for known patterns
# Instead of: np.array([[i, i**2, i**3] for i in range(1000)])
# Use vectorized operations:
indices = np.arange(1000)
efficient_array = np.column_stack([indices, indices**2, indices**3])
print(f"Efficient array shape: {efficient_array.shape}")

# For very large datasets, consider memmap
# large_array = np.memmap('large_data.dat', dtype='float32', mode='w+', shape=(1000000,))
```

---

## Question 2

**How do you perform matrix multiplication using NumPy?**

**Answer:**

### Theory
NumPy provides multiple methods for matrix multiplication, each optimized for different use cases. Understanding when to use each method is crucial for performance and correctness.

### Key Methods:

#### 1. **Element-wise Multiplication (`*`)**
- Multiplies corresponding elements
- Arrays must have compatible shapes (broadcasting applies)
- Not true matrix multiplication

#### 2. **Matrix Multiplication (`@` operator)**
- Python 3.5+ recommended approach
- Performs mathematical matrix multiplication
- Most readable and Pythonic

#### 3. **`np.dot()` Function**
- Traditional matrix multiplication
- Widely supported across NumPy versions
- Can handle various input types

#### 4. **`np.matmul()` Function**
- Explicit matrix multiplication
- Handles batch operations and broadcasting
- Equivalent to `@` operator

#### Code Example

```python
import numpy as np
import time

# Create sample matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])  # Shape: (3, 2)

C = np.array([[1, 2],
              [3, 4]])    # Shape: (2, 2)

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nMatrix C:")
print(C)

# Method 1: @ operator (Recommended)
result_at = A @ B
print("\nA @ B (@ operator):")
print(result_at)

# Method 2: np.dot()
result_dot = np.dot(A, B)
print("\nnp.dot(A, B):")
print(result_dot)

# Method 3: np.matmul()
result_matmul = np.matmul(A, B)
print("\nnp.matmul(A, B):")
print(result_matmul)

# Verify all methods give same result
print(f"\nAll methods equal: {np.array_equal(result_at, result_dot) and np.array_equal(result_dot, result_matmul)}")

# Element-wise multiplication (NOT matrix multiplication)
D = np.array([[1, 2],
              [3, 4]])
E = np.array([[5, 6],
              [7, 8]])

element_wise = D * E
print(f"\nElement-wise multiplication D * E:")
print(element_wise)

matrix_mult = D @ E
print(f"\nMatrix multiplication D @ E:")
print(matrix_mult)

# Advanced matrix operations
print("\nAdvanced Matrix Operations:")

# Matrix chain multiplication
F = np.random.randn(100, 50)
G = np.random.randn(50, 75)
H = np.random.randn(75, 25)

# Efficient chain multiplication
result_chain = F @ G @ H
print(f"Chain multiplication shape: {F.shape} @ {G.shape} @ {H.shape} = {result_chain.shape}")

# Batch matrix multiplication
batch_A = np.random.randn(10, 3, 4)  # 10 matrices of shape (3, 4)
batch_B = np.random.randn(10, 4, 2)  # 10 matrices of shape (4, 2)

batch_result = batch_A @ batch_B  # Result: (10, 3, 2)
print(f"Batch multiplication: {batch_A.shape} @ {batch_B.shape} = {batch_result.shape}")

# Broadcasting in matrix multiplication
vector = np.array([1, 2, 3])
matrix = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 1, 1]])

# Vector-matrix multiplication
vec_result = vector @ matrix
print(f"\nVector @ Matrix: {vec_result}")

# Matrix-vector multiplication
mat_vec_result = matrix @ vector
print(f"Matrix @ Vector: {mat_vec_result}")

# Performance comparison
def performance_test():
    size = 1000
    A_large = np.random.randn(size, size)
    B_large = np.random.randn(size, size)
    
    # Time @ operator
    start = time.time()
    result_at = A_large @ B_large
    time_at = time.time() - start
    
    # Time np.dot
    start = time.time()
    result_dot = np.dot(A_large, B_large)
    time_dot = time.time() - start
    
    # Time np.matmul
    start = time.time()
    result_matmul = np.matmul(A_large, B_large)
    time_matmul = time.time() - start
    
    print(f"\nPerformance comparison ({size}x{size} matrices):")
    print(f"@ operator: {time_at:.4f} seconds")
    print(f"np.dot(): {time_dot:.4f} seconds")
    print(f"np.matmul(): {time_matmul:.4f} seconds")

performance_test()

# Specialized matrix operations
print("\nSpecialized Operations:")

# Matrix power
square_matrix = np.array([[2, 1],
                          [1, 2]])
matrix_squared = np.linalg.matrix_power(square_matrix, 2)
print(f"Matrix power (A^2):")
print(matrix_squared)

# Kronecker product
kron_result = np.kron(np.array([[1, 2], [3, 4]]), np.array([[0, 5], [6, 7]]))
print(f"\nKronecker product:")
print(kron_result)
```

#### Explanation

1. **Method Selection**: Choose `@` operator for readability and performance
2. **Shape Compatibility**: Inner dimensions must match (A: m×n, B: n×p → Result: m×p)
3. **Broadcasting**: NumPy handles batch operations automatically
4. **Performance**: All methods are optimized, `@` operator is preferred

#### Use Cases

1. **Linear Algebra Systems**:
   ```python
   # Solving Ax = b
   A = np.array([[2, 1], [1, 3]])
   b = np.array([1, 2])
   x = np.linalg.solve(A, b)
   verification = A @ x
   ```

2. **Neural Network Operations**:
   ```python
   # Forward pass in neural network
   inputs = np.random.randn(32, 784)  # Batch of 32, 784 features
   weights = np.random.randn(784, 128)  # 784 → 128
   hidden = inputs @ weights  # Result: (32, 128)
   ```

3. **Computer Graphics Transformations**:
   ```python
   # 3D rotation matrix
   theta = np.pi / 4
   rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
   points = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]]).T
   rotated_points = rotation_matrix @ points
   ```

#### Best Practices

1. **Use `@` Operator**: Most readable and consistent with mathematical notation
2. **Check Dimensions**: Verify matrix shapes before multiplication
3. **Consider Memory**: Large matrices may require memory optimization
4. **Leverage Broadcasting**: Use NumPy's broadcasting for batch operations
5. **Profile Performance**: Test different methods for your specific use case

#### Pitfalls

1. **Dimension Mismatch**: Always verify compatible shapes
2. **Element-wise vs Matrix**: Don't confuse `*` with `@`
3. **Memory Usage**: Large matrix products can consume significant memory
4. **Numerical Stability**: Be aware of floating-point precision issues
5. **Copy vs View**: Matrix operations typically create new arrays

#### Debugging

```python
def debug_matrix_multiplication(A, B):
    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")
    
    if A.shape[-1] != B.shape[-2]:
        print(f"Error: Incompatible shapes for multiplication")
        print(f"A columns ({A.shape[-1]}) ≠ B rows ({B.shape[-2]})")
        return None
    
    result = A @ B
    print(f"Result shape: {result.shape}")
    return result

# Memory usage check
def check_memory_usage(shape1, shape2):
    memory_A = np.prod(shape1) * 8  # 8 bytes per float64
    memory_B = np.prod(shape2) * 8
    memory_result = shape1[0] * shape2[1] * 8
    total_memory = memory_A + memory_B + memory_result
    
    print(f"Memory usage: {total_memory / 1024**2:.2f} MB")
```

#### Optimization

1. **BLAS Libraries**: NumPy uses optimized BLAS implementations
2. **Memory Layout**: Ensure matrices are C-contiguous for best performance
3. **Batch Processing**: Use broadcasting instead of loops
4. **Data Types**: Consider using float32 for reduced memory usage
5. **Parallel Processing**: NumPy automatically uses multiple cores for large matrices

---

## Question 3

**Explain how to invert a matrix in NumPy.**

**Answer:**

### Theory
Matrix inversion finds the multiplicative inverse of a square matrix A, denoted as A⁻¹, such that A × A⁻¹ = A⁻¹ × A = I (identity matrix). Not all matrices are invertible; only non-singular matrices (det(A) ≠ 0) have inverses.

### Key Concepts:

#### 1. **Invertibility Conditions**
- Matrix must be square (n × n)
- Determinant must be non-zero
- Matrix must have full rank
- All eigenvalues must be non-zero

#### 2. **NumPy Inversion Methods**
- `np.linalg.inv()`: Direct matrix inversion
- `np.linalg.pinv()`: Moore-Penrose pseudoinverse
- `np.linalg.solve()`: Solve linear systems without explicit inversion

#### Code Example

```python
import numpy as np
import warnings

# Suppress warnings for demonstration
warnings.filterwarnings('ignore')

# Create invertible matrices
A = np.array([[2, 1],
              [1, 1]], dtype=float)

B = np.array([[4, 2, 1],
              [3, 1, 2],
              [1, 0, 1]], dtype=float)

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Method 1: np.linalg.inv() - Direct inversion
try:
    A_inv = np.linalg.inv(A)
    print(f"\nA inverse using np.linalg.inv():")
    print(A_inv)
    
    # Verify inversion
    identity_check = A @ A_inv
    print(f"\nA @ A_inv (should be identity):")
    print(identity_check)
    print(f"Is identity matrix: {np.allclose(identity_check, np.eye(A.shape[0]))}")
    
except np.linalg.LinAlgError as e:
    print(f"Matrix inversion failed: {e}")

# Method 2: Check determinant before inversion
def safe_invert(matrix):
    """Safely invert a matrix with error checking"""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    
    det = np.linalg.det(matrix)
    print(f"Determinant: {det}")
    
    if abs(det) < 1e-10:
        print("Warning: Matrix is nearly singular, inversion may be unstable")
        return None
    
    try:
        inverse = np.linalg.inv(matrix)
        # Verify the inversion
        verification = matrix @ inverse
        if np.allclose(verification, np.eye(matrix.shape[0]), atol=1e-10):
            print("Inversion successful and verified")
            return inverse
        else:
            print("Warning: Inversion verification failed")
            return inverse
    except np.linalg.LinAlgError:
        print("Matrix is singular and cannot be inverted")
        return None

print(f"\nSafe inversion of matrix B:")
B_inv = safe_invert(B)
if B_inv is not None:
    print("B inverse:")
    print(B_inv)

# Method 3: Moore-Penrose pseudoinverse (for non-square or singular matrices)
print(f"\nPseudoinverse examples:")

# Non-square matrix
C = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3 matrix

C_pinv = np.linalg.pinv(C)
print(f"Non-square matrix C:")
print(C)
print(f"Pseudoinverse of C:")
print(C_pinv)

# Verification for pseudoinverse
verification_left = C @ C_pinv @ C
verification_right = C_pinv @ C @ C_pinv
print(f"C @ C_pinv @ C ≈ C: {np.allclose(verification_left, C)}")
print(f"C_pinv @ C @ C_pinv ≈ C_pinv: {np.allclose(verification_right, C_pinv)}")

# Singular matrix example
singular_matrix = np.array([[1, 2, 3],
                           [2, 4, 6],
                           [1, 2, 3]], dtype=float)

print(f"\nSingular matrix:")
print(singular_matrix)
print(f"Determinant: {np.linalg.det(singular_matrix)}")

try:
    singular_inv = np.linalg.inv(singular_matrix)
    print("Unexpected: Singular matrix was inverted")
except np.linalg.LinAlgError:
    print("Expected: Cannot invert singular matrix with np.linalg.inv()")

# Use pseudoinverse for singular matrix
singular_pinv = np.linalg.pinv(singular_matrix)
print(f"Pseudoinverse of singular matrix:")
print(singular_pinv)

# Practical application: Solving linear systems
print(f"\nSolving linear systems:")

# System: Ax = b
A_system = np.array([[3, 2, 1],
                     [2, 3, 2],
                     [1, 2, 3]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

print("System: A @ x = b")
print(f"A:\n{A_system}")
print(f"b: {b}")

# Method 1: Using matrix inversion (less efficient)
A_inv_system = np.linalg.inv(A_system)
x_inv = A_inv_system @ b
print(f"\nSolution using inversion: x = A_inv @ b")
print(f"x = {x_inv}")

# Method 2: Using solve (more efficient and numerically stable)
x_solve = np.linalg.solve(A_system, b)
print(f"Solution using solve: x = {x_solve}")

# Verify both solutions
verification_inv = A_system @ x_inv
verification_solve = A_system @ x_solve
print(f"\nVerification:")
print(f"A @ x_inv = {verification_inv}")
print(f"A @ x_solve = {verification_solve}")
print(f"Both close to b: {np.allclose(verification_inv, b) and np.allclose(verification_solve, b)}")

# Performance comparison
import time

def performance_comparison():
    size = 500
    A_large = np.random.randn(size, size)
    A_large = A_large @ A_large.T  # Make positive definite (invertible)
    b_large = np.random.randn(size)
    
    # Time matrix inversion method
    start = time.time()
    A_inv_large = np.linalg.inv(A_large)
    x_inv_large = A_inv_large @ b_large
    time_inv = time.time() - start
    
    # Time direct solve method
    start = time.time()
    x_solve_large = np.linalg.solve(A_large, b_large)
    time_solve = time.time() - start
    
    print(f"\nPerformance comparison ({size}x{size} system):")
    print(f"Inversion method: {time_inv:.4f} seconds")
    print(f"Direct solve: {time_solve:.4f} seconds")
    print(f"Solve is {time_inv/time_solve:.1f}x faster")

performance_comparison()

# Condition number analysis
print(f"\nCondition number analysis:")

# Well-conditioned matrix
well_conditioned = np.array([[4, 1],
                             [1, 3]], dtype=float)

# Ill-conditioned matrix
ill_conditioned = np.array([[1, 1],
                           [1, 1.0001]], dtype=float)

for name, matrix in [("Well-conditioned", well_conditioned), 
                     ("Ill-conditioned", ill_conditioned)]:
    cond_num = np.linalg.cond(matrix)
    print(f"{name} matrix condition number: {cond_num:.2e}")
    if cond_num > 1e12:
        print(f"  Warning: Matrix is ill-conditioned, inversion may be unstable")
```

#### Explanation

1. **Direct Inversion**: Use `np.linalg.inv()` for well-conditioned square matrices
2. **Pseudoinverse**: Use `np.linalg.pinv()` for non-square or singular matrices
3. **Linear Systems**: Prefer `np.linalg.solve()` over explicit inversion for solving Ax=b
4. **Error Checking**: Always verify determinant and condition number

#### Use Cases

1. **Linear System Solving**:
   ```python
   # Solving multiple systems with same matrix
   A = np.random.randn(100, 100)
   A = A @ A.T  # Make positive definite
   B = np.random.randn(100, 50)  # 50 different right-hand sides
   
   # Method 1: Multiple solves (preferred)
   X_solve = np.linalg.solve(A, B)
   
   # Method 2: Single inversion
   A_inv = np.linalg.inv(A)
   X_inv = A_inv @ B
   ```

2. **Least Squares Problems**:
   ```python
   # Overdetermined system (more equations than unknowns)
   A = np.random.randn(100, 50)
   b = np.random.randn(100)
   
   # Normal equation: x = (A^T A)^(-1) A^T b
   x_normal = np.linalg.inv(A.T @ A) @ A.T @ b
   
   # Using pseudoinverse (more stable)
   x_pinv = np.linalg.pinv(A) @ b
   
   # Using least squares (most stable)
   x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
   ```

3. **Covariance Matrix Inversion**:
   ```python
   # In statistics and machine learning
   data = np.random.randn(1000, 10)
   cov_matrix = np.cov(data.T)
   precision_matrix = np.linalg.inv(cov_matrix)
   ```

#### Best Practices

1. **Avoid Explicit Inversion**: Use `solve()` instead of `inv()` when possible
2. **Check Condition Number**: High condition numbers indicate numerical instability
3. **Use Appropriate Method**: Choose between `inv()`, `pinv()`, and `solve()` based on problem type
4. **Handle Exceptions**: Wrap inversion in try-catch blocks
5. **Verify Results**: Always check A @ A_inv ≈ I

#### Pitfalls

1. **Singular Matrices**: Cannot be inverted with `np.linalg.inv()`
2. **Numerical Instability**: Ill-conditioned matrices produce inaccurate results
3. **Performance Issues**: Explicit inversion is slower than direct solving
4. **Memory Usage**: Storing inverse matrices uses extra memory
5. **Floating Point Errors**: Small determinants lead to precision loss

#### Debugging

```python
def diagnose_matrix(A):
    """Comprehensive matrix analysis for inversion"""
    print(f"Matrix shape: {A.shape}")
    print(f"Is square: {A.shape[0] == A.shape[1]}")
    
    if A.shape[0] == A.shape[1]:
        det = np.linalg.det(A)
        cond = np.linalg.cond(A)
        rank = np.linalg.matrix_rank(A)
        
        print(f"Determinant: {det:.2e}")
        print(f"Condition number: {cond:.2e}")
        print(f"Rank: {rank} (full rank: {rank == min(A.shape)})")
        
        if abs(det) < 1e-10:
            print("WARNING: Matrix is singular or nearly singular")
        if cond > 1e12:
            print("WARNING: Matrix is ill-conditioned")
```

#### Optimization

1. **Factorization Methods**: Use LU, Cholesky, or QR decomposition for repeated solves
2. **Sparse Matrices**: Use `scipy.sparse.linalg` for sparse matrix operations
3. **Batch Operations**: Vectorize operations when possible
4. **Memory Management**: Consider in-place operations for large matrices
5. **Specialized Algorithms**: Use domain-specific methods when available

---

## Question 4

**How do you calculate the determinant of a matrix?**

**Answer:**

### Theory
The determinant is a scalar value that provides important information about a square matrix, including whether it's invertible, its scaling factor for transformations, and its orientation properties. In NumPy, determinants are calculated using optimized linear algebra routines.

### Key Concepts:

#### 1. **Mathematical Properties**
- Only defined for square matrices
- det(A) = 0 indicates singular (non-invertible) matrix
- det(A) ≠ 0 indicates non-singular (invertible) matrix
- det(AB) = det(A) × det(B)
- det(A^T) = det(A)

#### 2. **Geometric Interpretation**
- Represents volume scaling factor in linear transformations
- Sign indicates orientation preservation/reversal
- Absolute value gives volume of parallelepiped formed by matrix rows/columns

#### Code Example

```python
import numpy as np
import time

# Basic determinant calculation
print("Basic Determinant Examples:")

# 2x2 matrix
A_2x2 = np.array([[3, 1],
                  [2, 4]], dtype=float)

det_A = np.linalg.det(A_2x2)
print(f"2x2 Matrix A:")
print(A_2x2)
print(f"Determinant: {det_A}")

# Manual verification for 2x2: det = ad - bc
manual_det = A_2x2[0,0] * A_2x2[1,1] - A_2x2[0,1] * A_2x2[1,0]
print(f"Manual calculation: {A_2x2[0,0]} × {A_2x2[1,1]} - {A_2x2[0,1]} × {A_2x2[1,0]} = {manual_det}")

# 3x3 matrix
A_3x3 = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]], dtype=float)

det_A_3x3 = np.linalg.det(A_3x3)
print(f"\n3x3 Matrix A:")
print(A_3x3)
print(f"Determinant: {det_A_3x3}")

# Larger matrix
A_4x4 = np.array([[2, 1, 3, 4],
                  [1, 0, 1, 2],
                  [3, 1, 0, 1],
                  [1, 2, 1, 0]], dtype=float)

det_A_4x4 = np.linalg.det(A_4x4)
print(f"\n4x4 Matrix A:")
print(A_4x4)
print(f"Determinant: {det_A_4x4}")

# Special matrices and their determinants
print(f"\nSpecial Matrix Determinants:")

# Identity matrix
I = np.eye(3)
print(f"Identity matrix determinant: {np.linalg.det(I)}")

# Zero matrix
Z = np.zeros((3, 3))
print(f"Zero matrix determinant: {np.linalg.det(Z)}")

# Singular matrix (linearly dependent rows)
singular = np.array([[1, 2, 3],
                     [2, 4, 6],
                     [3, 6, 9]], dtype=float)
print(f"Singular matrix determinant: {np.linalg.det(singular):.2e}")

# Upper triangular matrix
upper_tri = np.array([[2, 1, 3],
                      [0, 4, 1],
                      [0, 0, 5]], dtype=float)
print(f"Upper triangular determinant: {np.linalg.det(upper_tri)}")
print(f"Product of diagonal elements: {np.prod(np.diag(upper_tri))}")

# Lower triangular matrix
lower_tri = np.array([[3, 0, 0],
                      [1, 2, 0],
                      [4, 1, 5]], dtype=float)
print(f"Lower triangular determinant: {np.linalg.det(lower_tri)}")

# Determinant properties demonstration
print(f"\nDeterminant Properties:")

# Property: det(A^T) = det(A)
A = np.random.randn(3, 3)
det_A = np.linalg.det(A)
det_AT = np.linalg.det(A.T)
print(f"det(A) = {det_A:.6f}")
print(f"det(A^T) = {det_AT:.6f}")
print(f"Equal: {np.isclose(det_A, det_AT)}")

# Property: det(AB) = det(A) × det(B)
B = np.random.randn(3, 3)
det_B = np.linalg.det(B)
det_AB = np.linalg.det(A @ B)
det_product = det_A * det_B
print(f"\ndet(A) = {det_A:.6f}")
print(f"det(B) = {det_B:.6f}")
print(f"det(AB) = {det_AB:.6f}")
print(f"det(A) × det(B) = {det_product:.6f}")
print(f"Property holds: {np.isclose(det_AB, det_product)}")

# Property: det(cA) = c^n × det(A) for n×n matrix
c = 2.5
A_scaled = c * A
det_A_scaled = np.linalg.det(A_scaled)
expected_det = (c ** A.shape[0]) * det_A
print(f"\nScaling by {c}:")
print(f"det({c}A) = {det_A_scaled:.6f}")
print(f"{c}^{A.shape[0]} × det(A) = {expected_det:.6f}")
print(f"Property holds: {np.isclose(det_A_scaled, expected_det)}")

# Numerical stability analysis
print(f"\nNumerical Stability Analysis:")

def analyze_determinant_stability(matrix, description):
    det = np.linalg.det(matrix)
    cond = np.linalg.cond(matrix)
    print(f"\n{description}:")
    print(f"  Determinant: {det:.2e}")
    print(f"  Condition number: {cond:.2e}")
    
    if abs(det) < 1e-12:
        print("  Status: Nearly singular (determinant ≈ 0)")
    elif cond > 1e12:
        print("  Status: Ill-conditioned (high condition number)")
    else:
        print("  Status: Well-conditioned")

# Well-conditioned matrix
well_cond = np.array([[4, 1],
                      [1, 3]], dtype=float)
analyze_determinant_stability(well_cond, "Well-conditioned matrix")

# Ill-conditioned matrix
ill_cond = np.array([[1, 1],
                     [1, 1.0000001]], dtype=float)
analyze_determinant_stability(ill_cond, "Ill-conditioned matrix")

# Performance comparison for different matrix sizes
def determinant_performance():
    sizes = [10, 50, 100, 200, 500]
    print(f"\nPerformance Analysis:")
    print("Size\tTime (ms)\tDeterminant")
    print("-" * 35)
    
    for size in sizes:
        # Create random well-conditioned matrix
        A = np.random.randn(size, size)
        A = A @ A.T  # Make positive definite
        
        start = time.time()
        det = np.linalg.det(A)
        elapsed = (time.time() - start) * 1000
        
        print(f"{size}\t{elapsed:.2f}\t\t{det:.2e}")

determinant_performance()

# Applications of determinants
print(f"\nPractical Applications:")

# 1. Volume calculation
print("1. Volume Calculation:")
# Vectors forming a parallelepiped
v1 = np.array([1, 0, 0])
v2 = np.array([0, 2, 0])
v3 = np.array([0, 0, 3])

volume_matrix = np.column_stack([v1, v2, v3])
volume = abs(np.linalg.det(volume_matrix))
print(f"   Parallelepiped volume: {volume}")

# 2. Linear independence check
print("\n2. Linear Independence:")
vectors = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
det_vectors = np.linalg.det(vectors)
print(f"   Vectors matrix determinant: {det_vectors:.2e}")
print(f"   Linearly independent: {abs(det_vectors) > 1e-10}")

# 3. Transformation scaling
print("\n3. Transformation Scaling:")
# 2D rotation matrix
theta = np.pi / 4
rotation = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
print(f"   Rotation matrix determinant: {np.linalg.det(rotation):.6f}")
print("   (Should be 1 for pure rotation)")

# 2D scaling matrix
scale_x, scale_y = 2, 3
scaling = np.array([[scale_x, 0],
                    [0, scale_y]])
print(f"   Scaling matrix determinant: {np.linalg.det(scaling)}")
print(f"   Area scaling factor: {scale_x * scale_y}")

# Batch determinant calculation
print(f"\nBatch Determinant Calculation:")
# Multiple matrices
batch_matrices = np.random.randn(5, 3, 3)  # 5 matrices of size 3x3

# Calculate determinants for all matrices
batch_dets = np.array([np.linalg.det(matrix) for matrix in batch_matrices])
print(f"Batch determinants: {batch_dets}")

# Alternative using np.linalg.det with broadcasting
batch_dets_alt = np.linalg.det(batch_matrices)
print(f"Alternative method: {batch_dets_alt}")
print(f"Results match: {np.allclose(batch_dets, batch_dets_alt)}")

# Error handling and edge cases
def safe_determinant(matrix):
    """Calculate determinant with comprehensive error checking"""
    try:
        # Check if matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
        
        # Check for NaN or infinite values
        if not np.isfinite(matrix).all():
            raise ValueError("Matrix contains NaN or infinite values")
        
        # Calculate determinant
        det = np.linalg.det(matrix)
        
        # Check result validity
        if not np.isfinite(det):
            print("Warning: Determinant calculation resulted in NaN or infinity")
        
        return det
        
    except Exception as e:
        print(f"Error calculating determinant: {e}")
        return None

# Test error handling
print(f"\nError Handling Examples:")

# Non-square matrix
try:
    non_square = np.random.randn(2, 3)
    det = safe_determinant(non_square)
except Exception as e:
    print(f"Non-square matrix error handled: {e}")

# Matrix with NaN
nan_matrix = np.array([[1, 2], [np.nan, 4]])
det_nan = safe_determinant(nan_matrix)
print(f"Matrix with NaN determinant: {det_nan}")
```

#### Explanation

1. **Basic Calculation**: Use `np.linalg.det()` for any square matrix
2. **Numerical Precision**: Be aware of floating-point limitations for small determinants
3. **Performance**: O(n³) complexity using LU decomposition
4. **Stability**: Condition number indicates numerical reliability

#### Use Cases

1. **Matrix Invertibility**:
   ```python
   def is_invertible(matrix, tolerance=1e-12):
       return abs(np.linalg.det(matrix)) > tolerance
   ```

2. **Volume Calculations**:
   ```python
   # Volume of n-dimensional parallelepiped
   def parallelepiped_volume(vectors):
       return abs(np.linalg.det(np.column_stack(vectors)))
   ```

3. **Linear System Analysis**:
   ```python
   # Check if system Ax = b has unique solution
   def has_unique_solution(A):
       return np.linalg.det(A) != 0
   ```

4. **Coordinate Transformations**:
   ```python
   # Area/volume scaling factor
   transformation = np.array([[2, 0], [0, 3]])
   scaling_factor = abs(np.linalg.det(transformation))
   ```

#### Best Practices

1. **Check Matrix Validity**: Ensure matrix is square before calculation
2. **Handle Small Determinants**: Use appropriate tolerance for zero-checking
3. **Consider Condition Number**: High condition numbers indicate numerical instability
4. **Use Appropriate Data Types**: Float64 provides better precision than float32
5. **Batch Processing**: Leverage vectorization for multiple matrices

#### Pitfalls

1. **Floating-Point Precision**: Small determinants may be computed as zero
2. **Numerical Instability**: Ill-conditioned matrices produce unreliable results
3. **Singular Detection**: Use tolerance when checking for zero determinant
4. **Performance**: O(n³) complexity can be slow for very large matrices
5. **Memory Usage**: Large matrices require significant memory

#### Debugging

```python
def debug_determinant(matrix):
    """Comprehensive determinant analysis"""
    print(f"Matrix shape: {matrix.shape}")
    print(f"Data type: {matrix.dtype}")
    print(f"Contains NaN: {np.isnan(matrix).any()}")
    print(f"Contains infinity: {np.isinf(matrix).any()}")
    
    if matrix.shape[0] == matrix.shape[1]:
        det = np.linalg.det(matrix)
        cond = np.linalg.cond(matrix)
        rank = np.linalg.matrix_rank(matrix)
        
        print(f"Determinant: {det:.2e}")
        print(f"Condition number: {cond:.2e}")
        print(f"Matrix rank: {rank}")
        print(f"Full rank: {rank == matrix.shape[0]}")
        
        if abs(det) < 1e-12:
            print("WARNING: Matrix appears singular")
        if cond > 1e12:
            print("WARNING: Matrix is ill-conditioned")
```

#### Optimization

1. **Specialized Methods**: Use Cholesky decomposition for positive definite matrices
2. **Sparse Matrices**: Use `scipy.sparse` for sparse matrix determinants
3. **Parallel Processing**: NumPy automatically uses BLAS libraries
4. **Memory Management**: Consider in-place operations for memory efficiency
5. **Early Detection**: Check for obvious cases (triangular, diagonal matrices)

---

## Question 5

**How do you concatenate two arrays in NumPy?**

**Answer:**

### Theory
Array concatenation in NumPy joins arrays along existing axes or creates new axes. NumPy provides multiple functions for different concatenation patterns, each optimized for specific use cases and dimensional requirements.

### Key Functions:

#### 1. **`np.concatenate()`** - General Purpose
- Most flexible concatenation function
- Works along any existing axis
- Handles multiple arrays simultaneously

#### 2. **`np.vstack()` / `np.r_[]`** - Vertical Stacking
- Stacks arrays vertically (along axis 0)
- Equivalent to concatenate with axis=0

#### 3. **`np.hstack()` / `np.c_[]`** - Horizontal Stacking
- Stacks arrays horizontally (along axis 1)
- Equivalent to concatenate with axis=1

#### 4. **`np.dstack()`** - Depth Stacking
- Stacks arrays along axis 2 (depth)
- Creates 3D arrays from 2D inputs

#### 5. **`np.stack()`** - New Axis Creation
- Creates new axis for concatenation
- More control over stacking dimension

#### Code Example

```python
import numpy as np

# Create sample arrays
a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

c = np.array([[9, 10, 11],
              [12, 13, 14]])

print("Array a:")
print(a)
print("\nArray b:")
print(b)
print("\nArray c:")
print(c)

# Method 1: np.concatenate() - Most flexible
print("\n1. Using np.concatenate():")

# Concatenate along axis 0 (rows)
concat_axis0 = np.concatenate([a, b], axis=0)
print("Concatenate along axis 0 (vertical):")
print(concat_axis0)

# Concatenate along axis 1 (columns)
concat_axis1 = np.concatenate([a, b], axis=1)
print("\nConcatenate along axis 1 (horizontal):")
print(concat_axis1)

# Multiple arrays
multiple_concat = np.concatenate([a, b, [[9, 10], [11, 12]]], axis=0)
print("\nMultiple arrays concatenation:")
print(multiple_concat)

# Method 2: np.vstack() - Vertical stacking
print("\n2. Using np.vstack():")
vstack_result = np.vstack([a, b])
print("Vertical stack:")
print(vstack_result)

# Equivalent using np.r_[]
r_result = np.r_[a, b]
print("\nUsing np.r_[] (equivalent to vstack):")
print(r_result)

# Method 3: np.hstack() - Horizontal stacking
print("\n3. Using np.hstack():")
hstack_result = np.hstack([a, b])
print("Horizontal stack:")
print(hstack_result)

# Equivalent using np.c_[]
c_result = np.c_[a, b]
print("\nUsing np.c_[] (equivalent to hstack):")
print(c_result)

# Method 4: np.dstack() - Depth stacking
print("\n4. Using np.dstack():")
dstack_result = np.dstack([a, b])
print("Depth stack shape:", dstack_result.shape)
print("Depth stack:")
print(dstack_result)

# Method 5: np.stack() - New axis creation
print("\n5. Using np.stack():")

# Stack along new axis 0
stack_axis0 = np.stack([a, b], axis=0)
print("Stack along new axis 0, shape:", stack_axis0.shape)
print(stack_axis0)

# Stack along new axis 2
stack_axis2 = np.stack([a, b], axis=2)
print("\nStack along new axis 2, shape:", stack_axis2.shape)
print(stack_axis2)

# Working with 1D arrays
print("\nWorking with 1D arrays:")
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(f"x: {x}")
print(f"y: {y}")

# Concatenate 1D arrays
concat_1d = np.concatenate([x, y])
print(f"Concatenated: {concat_1d}")

# Stack 1D arrays to create 2D
stack_1d_rows = np.stack([x, y], axis=0)  # Stack as rows
stack_1d_cols = np.stack([x, y], axis=1)  # Stack as columns
print(f"Stacked as rows:\n{stack_1d_rows}")
print(f"Stacked as columns:\n{stack_1d_cols}")

# Advanced concatenation examples
print("\nAdvanced Examples:")

# Concatenating arrays with different dimensions (broadcasting)
arr_2d = np.array([[1, 2, 3]])
arr_1d = np.array([4, 5, 6])

# Need to reshape for proper concatenation
reshaped_1d = arr_1d.reshape(1, -1)
advanced_concat = np.concatenate([arr_2d, reshaped_1d], axis=0)
print("Concatenating different dimensional arrays:")
print(advanced_concat)

# Concatenating along new axis using np.newaxis
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Add new axis and concatenate
expanded_concat = np.concatenate([arr1[:, np.newaxis], arr2[:, np.newaxis]], axis=1)
print(f"\nConcatenating with new axis:\n{expanded_concat}")

# Practical applications
print("\nPractical Applications:")

# 1. Image processing - combining image channels
red_channel = np.random.randint(0, 256, (100, 100))
green_channel = np.random.randint(0, 256, (100, 100))
blue_channel = np.random.randint(0, 256, (100, 100))

# Combine channels to create RGB image
rgb_image = np.dstack([red_channel, green_channel, blue_channel])
print(f"RGB image shape: {rgb_image.shape}")

# 2. Data analysis - combining datasets
dataset1 = np.random.randn(100, 5)  # 100 samples, 5 features
dataset2 = np.random.randn(50, 5)   # 50 samples, 5 features

combined_dataset = np.concatenate([dataset1, dataset2], axis=0)
print(f"Combined dataset shape: {combined_dataset.shape}")

# 3. Time series - appending new data
existing_data = np.random.randn(365, 3)  # 365 days, 3 metrics
new_data = np.random.randn(30, 3)        # 30 more days

extended_data = np.concatenate([existing_data, new_data], axis=0)
print(f"Extended time series shape: {extended_data.shape}")

# Performance comparison
import time

def performance_test():
    # Generate large arrays for performance testing
    large_a = np.random.randn(1000, 1000)
    large_b = np.random.randn(1000, 1000)
    
    # Test concatenate
    start = time.time()
    result_concat = np.concatenate([large_a, large_b], axis=0)
    time_concat = time.time() - start
    
    # Test vstack
    start = time.time()
    result_vstack = np.vstack([large_a, large_b])
    time_vstack = time.time() - start
    
    # Test np.r_[]
    start = time.time()
    result_r = np.r_[large_a, large_b]
    time_r = time.time() - start
    
    print(f"\nPerformance comparison (1000x1000 arrays):")
    print(f"np.concatenate: {time_concat:.4f} seconds")
    print(f"np.vstack: {time_vstack:.4f} seconds")
    print(f"np.r_[]: {time_r:.4f} seconds")
    
    # Verify results are identical
    print(f"Results identical: {np.array_equal(result_concat, result_vstack) and np.array_equal(result_vstack, result_r)}")

performance_test()

# Error handling and edge cases
print("\nError Handling:")

def safe_concatenate(arrays, axis=0):
    """Safely concatenate arrays with error checking"""
    try:
        if len(arrays) == 0:
            raise ValueError("No arrays to concatenate")
        
        # Check if all arrays have compatible shapes
        first_shape = arrays[0].shape
        for i, arr in enumerate(arrays[1:], 1):
            compatible = True
            for dim in range(len(first_shape)):
                if dim != axis and first_shape[dim] != arr.shape[dim]:
                    compatible = False
                    break
            
            if not compatible:
                raise ValueError(f"Array {i} has incompatible shape {arr.shape} for concatenation along axis {axis}")
        
        return np.concatenate(arrays, axis=axis)
    
    except Exception as e:
        print(f"Concatenation error: {e}")
        return None

# Test error handling
incompatible_a = np.array([[1, 2], [3, 4]])
incompatible_b = np.array([[5, 6, 7], [8, 9, 10]])

result = safe_concatenate([incompatible_a, incompatible_b], axis=0)

# Memory-efficient concatenation for large arrays
print("\nMemory-Efficient Concatenation:")

def memory_efficient_concat(arrays, axis=0, chunk_size=1000):
    """Concatenate large arrays in chunks to manage memory"""
    if len(arrays) <= chunk_size:
        return np.concatenate(arrays, axis=axis)
    
    # Process in chunks
    result = arrays[0]
    for i in range(1, len(arrays), chunk_size):
        chunk = arrays[i:i+chunk_size]
        chunk_result = np.concatenate(chunk, axis=axis)
        result = np.concatenate([result, chunk_result], axis=axis)
    
    return result

# Specialized concatenation functions
print("\nSpecialized Functions:")

# np.append() - appends values to end of array
arr = np.array([1, 2, 3])
appended = np.append(arr, [4, 5, 6])
print(f"np.append result: {appended}")

# np.insert() - inserts values at specified indices
inserted = np.insert(arr, 1, [10, 11])
print(f"np.insert result: {inserted}")

# Block assembly for complex structures
print("\nBlock Assembly:")

# Create block matrices
top_left = np.ones((2, 2))
top_right = np.zeros((2, 3))
bottom_left = np.eye(3, 2)
bottom_right = np.full((3, 3), 5)

# Method 1: Manual concatenation
top_row = np.concatenate([top_left, top_right], axis=1)
bottom_row = np.concatenate([bottom_left, bottom_right], axis=1)
block_matrix = np.concatenate([top_row, bottom_row], axis=0)

print("Block matrix:")
print(block_matrix)

# Method 2: Using np.block() (more readable)
block_matrix_v2 = np.block([[top_left, top_right],
                            [bottom_left, bottom_right]])
print("\nUsing np.block():")
print(block_matrix_v2)

print(f"Results identical: {np.array_equal(block_matrix, block_matrix_v2)}")
```

#### Explanation

1. **Axis Selection**: Choose appropriate axis for concatenation direction
2. **Shape Compatibility**: All dimensions except concatenation axis must match
3. **Function Choice**: Use specific functions (vstack, hstack) for common patterns
4. **Memory Efficiency**: Consider memory usage for large array operations

#### Use Cases

1. **Data Processing**:
   ```python
   # Combining batch results
   batch_results = []
   for batch in data_batches:
       result = process_batch(batch)
       batch_results.append(result)
   final_result = np.concatenate(batch_results, axis=0)
   ```

2. **Image Processing**:
   ```python
   # Creating image mosaics
   row1 = np.hstack([img1, img2])
   row2 = np.hstack([img3, img4])
   mosaic = np.vstack([row1, row2])
   ```

3. **Feature Engineering**:
   ```python
   # Combining feature vectors
   numerical_features = np.array([age, income, score])
   categorical_features = np.array([gender_encoded, category_encoded])
   all_features = np.concatenate([numerical_features, categorical_features])
   ```

#### Best Practices

1. **Choose Right Function**: Use specific functions for clarity (vstack vs concatenate)
2. **Check Shapes**: Verify array compatibility before concatenation
3. **Memory Management**: Consider memory usage for large arrays
4. **Vectorization**: Prefer NumPy functions over Python loops
5. **Error Handling**: Implement checks for edge cases

#### Pitfalls

1. **Shape Mismatch**: Incompatible array shapes cause errors
2. **Memory Usage**: Large concatenations can exhaust memory
3. **Copy vs View**: Concatenation always creates new arrays
4. **Data Type Mixing**: Mixed types may cause unexpected results
5. **Axis Confusion**: Wrong axis selection leads to incorrect results

#### Debugging

```python
def debug_concatenation(arrays, axis=0):
    """Debug concatenation issues"""
    print(f"Number of arrays: {len(arrays)}")
    print(f"Concatenation axis: {axis}")
    
    for i, arr in enumerate(arrays):
        print(f"Array {i}: shape={arr.shape}, dtype={arr.dtype}")
    
    # Check compatibility
    if len(arrays) > 1:
        ref_shape = list(arrays[0].shape)
        for i, arr in enumerate(arrays[1:], 1):
            curr_shape = list(arr.shape)
            
            # Check all dimensions except concatenation axis
            for dim in range(len(ref_shape)):
                if dim != axis and ref_shape[dim] != curr_shape[dim]:
                    print(f"ERROR: Dimension {dim} mismatch between array 0 and {i}")
                    return False
    
    print("All arrays compatible for concatenation")
    return True
```

#### Optimization

1. **Pre-allocate**: When possible, pre-allocate result arrays
2. **Batch Processing**: Process multiple concatenations together
3. **Memory Layout**: Ensure arrays are contiguous for better performance
4. **Data Types**: Use consistent data types to avoid conversions
5. **Chunked Processing**: Break large operations into smaller chunks

---

## Question 6

**Describe how you would flatten a multi-dimensional array.**

### Theory

Flattening a multi-dimensional array means converting it into a 1-dimensional array while preserving all elements. NumPy provides several methods for flattening arrays, each with different characteristics regarding memory usage and data copying behavior.

### Code Example

```python
import numpy as np

# Create a multi-dimensional array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("Original 2D array:")
print(arr_2d)
print(f"Shape: {arr_2d.shape}")

print("\nOriginal 3D array:")
print(arr_3d)
print(f"Shape: {arr_3d.shape}")

# Method 1: flatten() - always returns a copy
flattened_copy = arr_2d.flatten()
print(f"\nUsing flatten(): {flattened_copy}")
print(f"Shape: {flattened_copy.shape}")

# Method 2: ravel() - returns view when possible, copy otherwise
flattened_view = arr_2d.ravel()
print(f"\nUsing ravel(): {flattened_view}")
print(f"Shape: {flattened_view.shape}")

# Method 3: reshape(-1) - returns view when possible
reshaped_flat = arr_2d.reshape(-1)
print(f"\nUsing reshape(-1): {reshaped_flat}")
print(f"Shape: {reshaped_flat.shape}")

# Method 4: Using np.concatenate for complex flattening
complex_arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
flat_concat = np.concatenate(complex_arr.flat)
print(f"\nComplex array flattened with concatenate: {flat_concat}")

# Demonstrating order parameter (C vs Fortran order)
arr_2d_ordered = np.array([[1, 2, 3], [4, 5, 6]])

# C-order (row-major, default)
flat_c = arr_2d_ordered.flatten(order='C')
print(f"\nC-order flattening: {flat_c}")

# Fortran-order (column-major)
flat_f = arr_2d_ordered.flatten(order='F')
print(f"Fortran-order flattening: {flat_f}")

# Performance comparison function
def compare_flattening_methods(arr, iterations=1000):
    """Compare performance of different flattening methods"""
    import time
    
    # Test flatten()
    start = time.time()
    for _ in range(iterations):
        _ = arr.flatten()
    flatten_time = time.time() - start
    
    # Test ravel()
    start = time.time()
    for _ in range(iterations):
        _ = arr.ravel()
    ravel_time = time.time() - start
    
    # Test reshape(-1)
    start = time.time()
    for _ in range(iterations):
        _ = arr.reshape(-1)
    reshape_time = time.time() - start
    
    print(f"flatten() time: {flatten_time:.6f}s")
    print(f"ravel() time: {ravel_time:.6f}s")
    print(f"reshape(-1) time: {reshape_time:.6f}s")

# Memory usage demonstration
print("\nMemory behavior:")
original = np.array([[1, 2, 3], [4, 5, 6]])

# Check if arrays share memory
flattened = original.flatten()  # Always copy
raveled = original.ravel()      # View when possible
reshaped = original.reshape(-1)  # View when possible

print(f"flatten() shares memory: {np.shares_memory(original, flattened)}")
print(f"ravel() shares memory: {np.shares_memory(original, raveled)}")
print(f"reshape(-1) shares memory: {np.shares_memory(original, reshaped)}")

# Modify original to see effect on views
original[0, 0] = 999

print(f"\nAfter modifying original[0,0] to 999:")
print(f"Original: {original.ravel()}")
print(f"Flattened copy: {flattened}")
print(f"Raveled view: {raveled}")
print(f"Reshaped view: {reshaped}")
```

### Explanation

1. **Method Selection**: Choose method based on memory requirements and performance needs
2. **Memory Behavior**: Views share memory with original array, copies don't
3. **Order Parameter**: Controls whether flattening follows C or Fortran ordering
4. **Performance**: Views are faster than copies for large arrays

### Use Cases

1. **Machine Learning**: Flattening image data for neural networks
2. **Data Processing**: Converting matrices to vectors for algorithms
3. **Serialization**: Preparing arrays for storage or transmission
4. **Statistical Analysis**: Converting multi-dimensional data to 1D for certain functions

### Best Practices

1. **Use ravel()** when you need a view and memory efficiency is important
2. **Use flatten()** when you explicitly need a copy
3. **Use reshape(-1)** for clarity and consistency with other reshape operations
4. **Consider order parameter** for performance-critical applications
5. **Check memory sharing** when modifying arrays after flattening

### Pitfalls

1. **Unexpected Views**: Modifications to views affect original arrays
2. **Memory Overhead**: flatten() always creates copies, using more memory
3. **Order Confusion**: Different ordering can produce different results
4. **Performance Issues**: Unnecessary copying in memory-constrained environments
5. **Type Preservation**: All methods preserve data types but not array structure

### Debugging

```python
def debug_flattening(arr):
    """Debug flattening operations"""
    print(f"Original array shape: {arr.shape}")
    print(f"Original array size: {arr.size}")
    print(f"Original array dtype: {arr.dtype}")
    
    flattened = arr.flatten()
    raveled = arr.ravel()
    reshaped = arr.reshape(-1)
    
    print(f"All methods produce same result: {np.array_equal(flattened, raveled) and np.array_equal(raveled, reshaped)}")
    print(f"flatten() is copy: {not np.shares_memory(arr, flattened)}")
    print(f"ravel() is view: {np.shares_memory(arr, raveled)}")
    print(f"reshape(-1) is view: {np.shares_memory(arr, reshaped)}")
```

### Optimization

1. **Prefer Views**: Use ravel() or reshape(-1) for better memory efficiency
2. **Batch Operations**: Flatten multiple arrays together when possible
3. **In-place Operations**: Consider if flattening is necessary or can be avoided
4. **Memory Layout**: Ensure arrays are contiguous for optimal performance
5. **Vectorized Operations**: Often flattening can be avoided with proper indexing

---

## Question 7

**How do you calculate the eigenvalues and eigenvectors of a matrix in NumPy?**

### Theory

Eigenvalues and eigenvectors are fundamental concepts in linear algebra. For a square matrix A, an eigenvector v and eigenvalue λ satisfy the equation: Av = λv. NumPy provides efficient functions in the `numpy.linalg` module to compute these values, which are essential for Principal Component Analysis (PCA), stability analysis, and many machine learning algorithms.

### Code Example

```python
import numpy as np
from numpy.linalg import eig, eigvals, eigh

# Create sample matrices
symmetric_matrix = np.array([[4, 2, 1],
                            [2, 3, 0],
                            [1, 0, 2]])

general_matrix = np.array([[1, 2, 3],
                          [0, 4, 5],
                          [0, 0, 6]])

complex_matrix = np.array([[1, 2j],
                          [-2j, 3]])

print("Symmetric Matrix:")
print(symmetric_matrix)

# Method 1: numpy.linalg.eig() - for general matrices
eigenvalues, eigenvectors = eig(symmetric_matrix)

print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Method 2: numpy.linalg.eigvals() - eigenvalues only
eigenvals_only = eigvals(symmetric_matrix)
print(f"\nEigenvalues only: {eigenvals_only}")

# Method 3: numpy.linalg.eigh() - for Hermitian/symmetric matrices (more efficient)
eig_vals_h, eig_vecs_h = eigh(symmetric_matrix)
print(f"\nUsing eigh() for symmetric matrix:")
print(f"Eigenvalues: {eig_vals_h}")
print(f"Eigenvectors:\n{eig_vecs_h}")

# Verification of eigenvalue equation: Av = λv
print("\nVerification of eigenvalue equation:")
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]
    
    # Calculate Av
    Av = symmetric_matrix @ v_i
    # Calculate λv
    lambda_v = lambda_i * v_i
    
    print(f"Eigenvalue {i+1}: λ = {lambda_i:.6f}")
    print(f"Av = {Av}")
    print(f"λv = {lambda_v}")
    print(f"Are they equal? {np.allclose(Av, lambda_v)}\n")

# Advanced eigenvalue computations
print("Advanced Computations:")

# Generalized eigenvalue problem: Av = λBv
A = np.array([[2, 1], [1, 2]])
B = np.array([[1, 0], [0, 1]])  # Identity matrix

from scipy.linalg import eig as scipy_eig
gen_eigenvals, gen_eigenvecs = scipy_eig(A, B)
print(f"Generalized eigenvalues: {gen_eigenvals}")

# Sort eigenvalues and eigenvectors
indices = np.argsort(eigenvalues)[::-1]  # Descending order
sorted_eigenvals = eigenvalues[indices]
sorted_eigenvecs = eigenvectors[:, indices]

print(f"\nSorted eigenvalues (descending): {sorted_eigenvals}")

# Compute condition number using eigenvalues
def condition_number_from_eigenvals(eigenvals):
    """Compute condition number from eigenvalues"""
    return np.max(np.real(eigenvals)) / np.min(np.real(eigenvals))

cond_num = condition_number_from_eigenvals(eigenvalues)
print(f"Condition number: {cond_num:.6f}")

# Principal Component Analysis example
def pca_using_eigenvalues(data, n_components=None):
    """Perform PCA using eigenvalue decomposition"""
    # Center the data
    centered_data = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered_data.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = eigh(cov_matrix)
    
    # Sort in descending order
    indices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[indices]
    eigenvecs = eigenvecs[:, indices]
    
    # Select number of components
    if n_components is not None:
        eigenvals = eigenvals[:n_components]
        eigenvecs = eigenvecs[:, :n_components]
    
    # Compute explained variance ratio
    explained_variance_ratio = eigenvals / np.sum(eigenvals)
    
    # Transform data
    transformed_data = centered_data @ eigenvecs
    
    return {
        'eigenvalues': eigenvals,
        'eigenvectors': eigenvecs,
        'explained_variance_ratio': explained_variance_ratio,
        'transformed_data': transformed_data
    }

# Example PCA application
np.random.seed(42)
sample_data = np.random.randn(100, 4)  # 100 samples, 4 features

pca_result = pca_using_eigenvalues(sample_data, n_components=2)
print(f"\nPCA Results:")
print(f"Explained variance ratio: {pca_result['explained_variance_ratio']}")
print(f"Cumulative explained variance: {np.cumsum(pca_result['explained_variance_ratio'])}")

# Stability analysis using eigenvalues
def stability_analysis(matrix):
    """Analyze system stability using eigenvalues"""
    eigenvals = eigvals(matrix)
    
    # For continuous systems: all real parts should be negative
    real_parts = np.real(eigenvals)
    max_real_part = np.max(real_parts)
    
    is_stable = max_real_part < 0
    
    print(f"Eigenvalues: {eigenvals}")
    print(f"Maximum real part: {max_real_part:.6f}")
    print(f"System is stable: {is_stable}")
    
    return is_stable, eigenvals

# Example stability analysis
system_matrix = np.array([[-1, 2], [-2, -1]])
print(f"\nStability Analysis:")
stability_analysis(system_matrix)

# Error handling and edge cases
def robust_eigenvalue_computation(matrix):
    """Robust eigenvalue computation with error handling"""
    try:
        # Check if matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square for eigenvalue computation")
        
        # Check for NaN or infinite values
        if not np.isfinite(matrix).all():
            raise ValueError("Matrix contains NaN or infinite values")
        
        # Compute eigenvalues
        if np.allclose(matrix, matrix.T):
            # Use more efficient method for symmetric matrices
            eigenvals, eigenvecs = eigh(matrix)
        else:
            eigenvals, eigenvecs = eig(matrix)
        
        return eigenvals, eigenvecs
    
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error: {e}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Test robust computation
test_matrices = [
    np.array([[1, 2], [3, 4]]),      # Valid matrix
    np.array([[1, 2, 3], [4, 5]]),   # Non-square matrix
    np.array([[np.inf, 1], [1, 2]])  # Matrix with infinite values
]

for i, test_matrix in enumerate(test_matrices):
    print(f"\nTest matrix {i+1}:")
    eigenvals, eigenvecs = robust_eigenvalue_computation(test_matrix)
    if eigenvals is not None:
        print(f"Successfully computed eigenvalues: {eigenvals}")
```

### Explanation

1. **Function Selection**: Use `eig()` for general matrices, `eigh()` for symmetric/Hermitian matrices
2. **Verification**: Always verify that computed eigenvalues satisfy Av = λv
3. **Sorting**: Sort eigenvalues/eigenvectors based on application needs
4. **Numerical Stability**: Use appropriate functions for matrix types

### Use Cases

1. **Principal Component Analysis**: Dimensionality reduction in machine learning
2. **Stability Analysis**: Determining system stability in control theory
3. **Spectral Analysis**: Analyzing frequency components in signal processing
4. **Graph Theory**: Computing graph properties using adjacency matrices
5. **Quantum Mechanics**: Finding energy states in quantum systems

### Best Practices

1. **Use eigh() for symmetric matrices** - more numerically stable and efficient
2. **Check matrix properties** before computation (square, finite values)
3. **Sort results** based on eigenvalue magnitude when needed
4. **Verify results** by checking the eigenvalue equation
5. **Handle complex eigenvalues** appropriately in applications

### Pitfalls

1. **Non-square Matrices**: Only square matrices have eigenvalues
2. **Numerical Precision**: Small eigenvalues may have precision issues
3. **Complex Results**: Real matrices can have complex eigenvalues
4. **Normalization**: Eigenvectors may not be normalized as expected
5. **Ordering**: Eigenvalues/eigenvectors may not be sorted

### Debugging

```python
def debug_eigenvalue_computation(matrix):
    """Debug eigenvalue computation issues"""
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix dtype: {matrix.dtype}")
    print(f"Is square: {matrix.shape[0] == matrix.shape[1]}")
    print(f"Is symmetric: {np.allclose(matrix, matrix.T)}")
    print(f"Has finite values: {np.isfinite(matrix).all()}")
    print(f"Condition number: {np.linalg.cond(matrix):.2e}")
    
    if matrix.shape[0] == matrix.shape[1]:
        eigenvals, eigenvecs = eig(matrix)
        print(f"Eigenvalues computed successfully")
        print(f"Are eigenvalues real: {np.isreal(eigenvals).all()}")
        print(f"Eigenvector norms: {np.linalg.norm(eigenvecs, axis=0)}")
```

### Optimization

1. **Matrix Type Selection**: Use specialized functions for matrix properties
2. **Memory Management**: Compute only eigenvalues when eigenvectors not needed
3. **Parallel Computation**: Use threaded BLAS libraries for large matrices
4. **Sparse Matrices**: Use scipy.sparse.linalg for sparse matrices
5. **Iterative Methods**: Consider iterative solvers for largest/smallest eigenvalues

---

## Question 8

**How can you reverse an array in NumPy?**

### Theory

Reversing arrays in NumPy can be accomplished through multiple approaches, each with different performance characteristics and use cases. The most common methods include slicing with negative steps, using `np.flip()`, and `np.flipud()`/`np.fliplr()` for specific axis operations. Understanding when to use each method is crucial for writing efficient NumPy code.

### Code Example

```python
import numpy as np

# Create sample arrays for demonstration
arr_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]],
                   [[9, 10], [11, 12]]])

print("Original 1D array:", arr_1d)
print("Original 2D array:")
print(arr_2d)

# Method 1: Slicing with negative step (most common)
print("\n1. Using slicing with negative step:")
reversed_1d = arr_1d[::-1]
print(f"1D reversed: {reversed_1d}")

# Reverse along different axes in 2D
reversed_2d_rows = arr_2d[::-1]  # Reverse rows
reversed_2d_cols = arr_2d[:, ::-1]  # Reverse columns
reversed_2d_both = arr_2d[::-1, ::-1]  # Reverse both

print(f"2D reversed rows:\n{reversed_2d_rows}")
print(f"2D reversed columns:\n{reversed_2d_cols}")
print(f"2D reversed both:\n{reversed_2d_both}")

# Method 2: Using np.flip()
print("\n2. Using np.flip():")
flip_1d = np.flip(arr_1d)
print(f"1D flipped: {flip_1d}")

flip_2d_axis0 = np.flip(arr_2d, axis=0)  # Flip along axis 0 (rows)
flip_2d_axis1 = np.flip(arr_2d, axis=1)  # Flip along axis 1 (columns)
flip_2d_all = np.flip(arr_2d)  # Flip along all axes

print(f"2D flipped axis 0:\n{flip_2d_axis0}")
print(f"2D flipped axis 1:\n{flip_2d_axis1}")
print(f"2D flipped all axes:\n{flip_2d_all}")

# Method 3: Using np.flipud() and np.fliplr()
print("\n3. Using np.flipud() and np.fliplr():")
flipud_2d = np.flipud(arr_2d)  # Flip up-down (equivalent to flip(axis=0))
fliplr_2d = np.fliplr(arr_2d)  # Flip left-right (equivalent to flip(axis=1))

print(f"2D flip up-down:\n{flipud_2d}")
print(f"2D flip left-right:\n{fliplr_2d}")

# Method 4: Using advanced indexing
print("\n4. Using advanced indexing:")
indices_reversed = np.arange(len(arr_1d)-1, -1, -1)
advanced_reversed = arr_1d[indices_reversed]
print(f"Advanced indexing result: {advanced_reversed}")

# Complex multi-dimensional reversing
print("\n5. Complex multi-dimensional operations:")
print(f"Original 3D shape: {arr_3d.shape}")
print(f"Original 3D:\n{arr_3d}")

# Reverse along specific axes
reverse_axis0 = np.flip(arr_3d, axis=0)
reverse_axis1 = np.flip(arr_3d, axis=1)
reverse_axis2 = np.flip(arr_3d, axis=2)

print(f"Reversed along axis 0:\n{reverse_axis0}")
print(f"Reversed along axis 1:\n{reverse_axis1}")
print(f"Reversed along axis 2:\n{reverse_axis2}")

# Performance comparison
def compare_reverse_methods(arr, iterations=10000):
    """Compare performance of different reversal methods"""
    import time
    
    # Method 1: Slicing
    start = time.time()
    for _ in range(iterations):
        _ = arr[::-1]
    slice_time = time.time() - start
    
    # Method 2: np.flip()
    start = time.time()
    for _ in range(iterations):
        _ = np.flip(arr)
    flip_time = time.time() - start
    
    print(f"Slicing time: {slice_time:.6f}s")
    print(f"np.flip() time: {flip_time:.6f}s")
    print(f"Slicing is {flip_time/slice_time:.2f}x faster")

print("\nPerformance comparison:")
large_array = np.random.randn(10000)
compare_reverse_methods(large_array)

# Memory behavior analysis
print("\nMemory behavior:")
original = np.array([1, 2, 3, 4, 5])

# Check if operations create views or copies
sliced_reverse = original[::-1]
flipped_reverse = np.flip(original)

print(f"Original: {original}")
print(f"Sliced reverse: {sliced_reverse}")
print(f"Flipped reverse: {flipped_reverse}")

# Modify original to see effect
original[0] = 999

print(f"\nAfter modifying original[0] to 999:")
print(f"Original: {original}")
print(f"Sliced reverse: {sliced_reverse}")  # This will change (view)
print(f"Flipped reverse: {flipped_reverse}")  # This won't change (copy)

print(f"Slicing creates view: {np.shares_memory(original, sliced_reverse)}")
print(f"np.flip creates copy: {not np.shares_memory(original, flipped_reverse)}")

# Practical applications
def reverse_image_processing():
    """Demonstrate array reversal in image processing context"""
    # Simulate an image (3D array: height, width, channels)
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Image transformations using array reversal
    horizontal_flip = np.flip(image, axis=1)  # Mirror horizontally
    vertical_flip = np.flip(image, axis=0)    # Mirror vertically
    rotate_180 = np.flip(image, axis=(0, 1))  # Rotate 180 degrees
    
    return {
        'original': image,
        'horizontal_flip': horizontal_flip,
        'vertical_flip': vertical_flip,
        'rotate_180': rotate_180
    }

# Time series data reversal
def reverse_time_series(data, window_size=None):
    """Reverse time series data with optional windowed approach"""
    if window_size is None:
        return np.flip(data)
    
    # Reverse in windows
    n_windows = len(data) // window_size
    remainder = len(data) % window_size
    
    result = np.empty_like(data)
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        result[start_idx:end_idx] = np.flip(data[start_idx:end_idx])
    
    # Handle remainder
    if remainder > 0:
        result[-remainder:] = np.flip(data[-remainder:])
    
    return result

# Example usage
time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
windowed_reverse = reverse_time_series(time_series, window_size=3)
print(f"\nOriginal time series: {time_series}")
print(f"Windowed reverse (window=3): {windowed_reverse}")

# In-place reversal for memory efficiency
def inplace_reverse_1d(arr):
    """Reverse 1D array in-place"""
    n = len(arr)
    for i in range(n // 2):
        arr[i], arr[n-1-i] = arr[n-1-i], arr[i]
    return arr

def inplace_reverse_axis(arr, axis=0):
    """Reverse array along specified axis in-place"""
    # Use advanced indexing for in-place reversal
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None, None, -1)
    arr[:] = arr[tuple(slices)]
    return arr

# Test in-place operations
test_array = np.array([1, 2, 3, 4, 5])
print(f"\nBefore in-place reverse: {test_array}")
inplace_reverse_1d(test_array)
print(f"After in-place reverse: {test_array}")

# Error handling and edge cases
def robust_array_reverse(arr, axis=None, method='slice'):
    """Robust array reversal with error handling"""
    try:
        if arr.size == 0:
            return arr.copy()
        
        if method == 'slice':
            if axis is None:
                return arr[::-1] if arr.ndim == 1 else np.flip(arr)
            else:
                slices = [slice(None)] * arr.ndim
                slices[axis] = slice(None, None, -1)
                return arr[tuple(slices)]
        elif method == 'flip':
            return np.flip(arr, axis=axis)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    except Exception as e:
        print(f"Error reversing array: {e}")
        return None

# Test edge cases
edge_cases = [
    np.array([]),           # Empty array
    np.array([1]),          # Single element
    np.array([[1, 2]]),     # Single row 2D
    np.array([[1], [2]])    # Single column 2D
]

print("\nEdge case testing:")
for i, test_case in enumerate(edge_cases):
    print(f"Test case {i+1}: {test_case.shape}")
    result = robust_array_reverse(test_case)
    print(f"Result: {result}")
```

### Explanation

1. **Method Selection**: Choose between slicing, `np.flip()`, and specialized functions based on needs
2. **Memory Considerations**: Slicing creates views, `np.flip()` creates copies
3. **Axis Specification**: Use axis parameter for multi-dimensional operations
4. **Performance**: Slicing is typically faster than function calls

### Use Cases

1. **Image Processing**: Flipping images horizontally or vertically
2. **Signal Processing**: Reversing time series data
3. **Matrix Operations**: Reversing rows/columns for mathematical computations
4. **Data Augmentation**: Creating reversed versions for machine learning
5. **Game Development**: Implementing mirror effects or rotations

### Best Practices

1. **Use slicing for simple reversals** - fastest and most readable
2. **Use np.flip() for axis-specific operations** - more explicit and flexible
3. **Consider memory implications** - views vs copies
4. **Handle edge cases** like empty arrays
5. **Use in-place operations** for memory-constrained environments

### Pitfalls

1. **View vs Copy Confusion**: Slicing creates views that can be modified
2. **Axis Misunderstanding**: Wrong axis can lead to unexpected results
3. **Memory Usage**: `np.flip()` always creates copies
4. **Performance Overhead**: Function calls slower than slicing
5. **Multidimensional Complexity**: Multiple axes can be confusing

### Debugging

```python
def debug_array_reversal(arr):
    """Debug array reversal operations"""
    print(f"Array shape: {arr.shape}")
    print(f"Array dtype: {arr.dtype}")
    print(f"Array size: {arr.size}")
    
    if arr.size > 0:
        # Test different reversal methods
        try:
            slice_result = arr[::-1] if arr.ndim == 1 else None
            flip_result = np.flip(arr)
            
            print(f"Slicing works: {slice_result is not None}")
            print(f"np.flip works: True")
            
            if slice_result is not None and arr.ndim == 1:
                print(f"Results equal: {np.array_equal(slice_result, flip_result)}")
        except Exception as e:
            print(f"Error in reversal: {e}")
```

### Optimization

1. **Prefer Slicing**: Use `arr[::-1]` for 1D arrays and simple cases
2. **Memory Management**: Use views when possible to avoid copying
3. **Batch Operations**: Reverse multiple arrays together when applicable
4. **Vectorized Operations**: Combine reversal with other operations
5. **Specialized Functions**: Use `flipud`/`fliplr` for clarity in 2D operations

---

## Question 9

**How do you apply a conditional filter to a NumPy array?**

### Theory

Conditional filtering in NumPy involves creating boolean masks that identify elements meeting specific criteria, then using these masks to extract, modify, or operate on array subsets. This technique is fundamental for data analysis, preprocessing, and mathematical operations. NumPy provides multiple approaches including boolean indexing, `np.where()`, `np.select()`, and conditional functions.

### Code Example

```python
import numpy as np

# Create sample data for demonstration
data = np.array([1, 5, 3, 8, 2, 7, 9, 4, 6, 10])
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

print("Original 1D array:", data)
print("Original 2D matrix:")
print(matrix)

# Method 1: Boolean indexing (most common)
print("\n1. Boolean Indexing:")

# Simple condition
mask = data > 5
filtered_data = data[mask]
print(f"Elements > 5: {filtered_data}")
print(f"Boolean mask: {mask}")

# Multiple conditions using logical operators
complex_mask = (data > 3) & (data < 8)
complex_filtered = data[complex_mask]
print(f"Elements between 3 and 8: {complex_filtered}")

# Using OR condition
or_mask = (data < 3) | (data > 8)
or_filtered = data[or_mask]
print(f"Elements < 3 OR > 8: {or_filtered}")

# Method 2: Using np.where()
print("\n2. Using np.where():")

# Extract indices where condition is true
indices = np.where(data > 5)
print(f"Indices where data > 5: {indices[0]}")
print(f"Values at those indices: {data[indices]}")

# Conditional replacement
replaced = np.where(data > 5, data, 0)  # Keep values > 5, replace others with 0
print(f"Replace values <= 5 with 0: {replaced}")

# Multiple conditions with np.where
multi_replace = np.where(data > 7, 100, np.where(data < 3, -100, data))
print(f"Multi-condition replacement: {multi_replace}")

# Method 3: Using np.select()
print("\n3. Using np.select():")

# Define conditions and choices
conditions = [
    data < 3,
    (data >= 3) & (data <= 7),
    data > 7
]
choices = ['Low', 'Medium', 'High']
default = 'Unknown'

categories = np.select(conditions, choices, default=default)
print(f"Categorized data: {categories}")

# Numerical example with np.select
num_conditions = [data < 4, data < 8, data >= 8]
num_choices = [data * 0.1, data * 0.5, data * 1.0]
scaled_data = np.select(num_conditions, num_choices)
print(f"Scaled data: {scaled_data}")

# Method 4: Advanced filtering techniques
print("\n4. Advanced Filtering:")

# Multi-dimensional filtering
print("2D Matrix filtering:")
matrix_mask = matrix % 2 == 0  # Even numbers
even_elements = matrix[matrix_mask]
print(f"Even elements: {even_elements}")

# Row-wise filtering
row_sums = np.sum(matrix, axis=1)
high_sum_rows = matrix[row_sums > 30]
print(f"Rows with sum > 30:")
print(high_sum_rows)

# Column-wise filtering
col_means = np.mean(matrix, axis=0)
high_mean_cols = matrix[:, col_means > 7]
print(f"Columns with mean > 7:")
print(high_mean_cols)

# Method 5: Filtering with custom functions
print("\n5. Custom Function Filtering:")

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Vectorize the function
vectorized_is_prime = np.vectorize(is_prime)
prime_mask = vectorized_is_prime(data)
prime_numbers = data[prime_mask]
print(f"Prime numbers: {prime_numbers}")

# Method 6: Statistical filtering
print("\n6. Statistical Filtering:")

# Remove outliers using IQR method
def remove_outliers_iqr(arr):
    """Remove outliers using Interquartile Range method"""
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (arr >= lower_bound) & (arr <= upper_bound)
    return arr[mask], mask

# Generate data with outliers
data_with_outliers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 100])
clean_data, outlier_mask = remove_outliers_iqr(data_with_outliers)
print(f"Original data: {data_with_outliers}")
print(f"Data without outliers: {clean_data}")

# Z-score filtering
def filter_by_zscore(arr, threshold=2):
    """Filter data based on Z-score threshold"""
    z_scores = np.abs((arr - np.mean(arr)) / np.std(arr))
    mask = z_scores < threshold
    return arr[mask], mask

zscore_filtered, zscore_mask = filter_by_zscore(data_with_outliers)
print(f"Z-score filtered data: {zscore_filtered}")

# Method 7: Conditional operations
print("\n7. Conditional Operations:")

# Conditional assignment
data_copy = data.copy()
data_copy[data_copy > 5] *= 2  # Double values > 5
data_copy[data_copy <= 5] += 10  # Add 10 to values <= 5
print(f"Conditional operations result: {data_copy}")

# Using np.clip for range filtering
clipped = np.clip(data, 3, 8)  # Clip values to range [3, 8]
print(f"Clipped to [3, 8]: {clipped}")

# Method 8: Complex real-world filtering
print("\n8. Real-world Example - Data Cleaning:")

# Simulate sensor data with missing values (NaN) and outliers
np.random.seed(42)
sensor_data = np.random.normal(25, 5, 100)  # Temperature data
sensor_data[::10] = np.nan  # Add missing values
sensor_data[5] = 100  # Add outlier
sensor_data[15] = -50  # Add another outlier

def clean_sensor_data(data, temp_range=(0, 50)):
    """Clean sensor data by removing NaN and outliers"""
    # Remove NaN values
    valid_mask = ~np.isnan(data)
    
    # Remove outliers based on reasonable temperature range
    temp_mask = (data >= temp_range[0]) & (data <= temp_range[1])
    
    # Combine masks
    clean_mask = valid_mask & temp_mask
    
    return data[clean_mask], clean_mask

clean_temps, clean_mask = clean_sensor_data(sensor_data)
print(f"Original data points: {len(sensor_data)}")
print(f"Clean data points: {len(clean_temps)}")
print(f"Removed {len(sensor_data) - len(clean_temps)} invalid points")

# Method 9: Performance optimization
def compare_filtering_methods(data, iterations=10000):
    """Compare performance of different filtering methods"""
    import time
    
    # Boolean indexing
    start = time.time()
    for _ in range(iterations):
        _ = data[data > 5]
    bool_time = time.time() - start
    
    # np.where
    start = time.time()
    for _ in range(iterations):
        indices = np.where(data > 5)[0]
        _ = data[indices]
    where_time = time.time() - start
    
    # List comprehension (for comparison)
    start = time.time()
    for _ in range(iterations):
        _ = np.array([x for x in data if x > 5])
    list_time = time.time() - start
    
    print(f"Boolean indexing: {bool_time:.6f}s")
    print(f"np.where: {where_time:.6f}s")
    print(f"List comprehension: {list_time:.6f}s")
    print(f"Boolean indexing is {where_time/bool_time:.2f}x faster than np.where")
    print(f"Boolean indexing is {list_time/bool_time:.2f}x faster than list comprehension")

print("\nPerformance Comparison:")
large_data = np.random.randint(1, 100, 10000)
compare_filtering_methods(large_data)

# Method 10: Error handling and edge cases
def robust_filter(arr, condition_func, default_value=None):
    """Robust filtering with error handling"""
    try:
        if arr.size == 0:
            return np.array([])
        
        mask = condition_func(arr)
        
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            raise ValueError("Condition function must return boolean array")
        
        if mask.shape != arr.shape:
            raise ValueError("Mask shape must match array shape")
        
        filtered = arr[mask]
        
        if filtered.size == 0 and default_value is not None:
            return np.array([default_value])
        
        return filtered
    
    except Exception as e:
        print(f"Filtering error: {e}")
        return np.array([]) if default_value is None else np.array([default_value])

# Test robust filtering
test_cases = [
    (np.array([]), lambda x: x > 5),  # Empty array
    (np.array([1, 2, 3]), lambda x: x > 10),  # No matches
    (np.array([1, 2, np.nan, 4]), lambda x: ~np.isnan(x))  # NaN handling
]

print("\nRobust filtering tests:")
for i, (test_arr, condition) in enumerate(test_cases):
    result = robust_filter(test_arr, condition, default_value=0)
    print(f"Test {i+1}: {result}")
```

### Explanation

1. **Boolean Masking**: Create boolean arrays that identify elements meeting conditions
2. **Logical Operators**: Combine multiple conditions using &, |, and ~ operators
3. **Function Selection**: Choose appropriate method based on complexity and performance needs
4. **Edge Case Handling**: Consider empty results, NaN values, and data type consistency

### Use Cases

1. **Data Cleaning**: Removing outliers, missing values, and invalid entries
2. **Feature Selection**: Filtering data based on statistical criteria
3. **Image Processing**: Thresholding and segmentation operations
4. **Financial Analysis**: Filtering stocks based on performance criteria
5. **Scientific Computing**: Selecting data points meeting experimental conditions

### Best Practices

1. **Use boolean indexing** for simple, fast filtering operations
2. **Parenthesize complex conditions** to ensure correct operator precedence
3. **Consider memory usage** when filtering large arrays
4. **Validate filter results** to ensure expected behavior
5. **Handle edge cases** like empty arrays and no matches

### Pitfalls

1. **Operator Precedence**: Incorrect precedence with &, |, and ~ operators
2. **Broadcasting Issues**: Mismatched shapes in conditional operations
3. **Data Type Confusion**: Mixed types can cause unexpected filtering results
4. **Memory Overhead**: Large boolean masks consume significant memory
5. **NaN Handling**: NaN values in conditions can produce unexpected results

### Debugging

```python
def debug_filtering(arr, condition_func):
    """Debug conditional filtering operations"""
    print(f"Array shape: {arr.shape}")
    print(f"Array dtype: {arr.dtype}")
    
    try:
        mask = condition_func(arr)
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"True values in mask: {np.sum(mask)}")
        print(f"Percentage filtered: {np.sum(mask)/len(mask)*100:.1f}%")
        
        filtered = arr[mask]
        print(f"Filtered array shape: {filtered.shape}")
        
        return filtered
    except Exception as e:
        print(f"Error in filtering: {e}")
        return None
```

### Optimization

1. **Pre-compute Conditions**: Store complex conditions in variables
2. **Use Vectorized Operations**: Prefer NumPy functions over Python loops
3. **Memory-Efficient Filtering**: Process large arrays in chunks
4. **Combine Operations**: Merge filtering with other array operations
5. **Choose Right Method**: Boolean indexing for simple cases, `np.where()` for complex logic

---

## Question 10

**Explain how to generate random data with NumPy.**

### Theory

NumPy's random number generation is built on the concept of pseudorandom number generators (PRNGs) that produce sequences of numbers that appear random but are actually deterministic. NumPy provides two interfaces: the legacy `numpy.random` module and the modern `numpy.random.Generator` class. Understanding different distributions, seeding for reproducibility, and performance considerations is crucial for scientific computing and simulation work.

### Code Example

```python
import numpy as np
from numpy.random import default_rng

# Set seed for reproducibility (legacy method)
np.random.seed(42)

print("=== BASIC RANDOM GENERATION ===")

# Legacy numpy.random interface
print("1. Legacy numpy.random methods:")
random_floats = np.random.random(5)  # Uniform [0, 1)
random_integers = np.random.randint(1, 10, size=5)  # Integers [1, 10)
normal_distribution = np.random.normal(0, 1, 5)  # Normal(mean=0, std=1)

print(f"Random floats [0,1): {random_floats}")
print(f"Random integers [1,10): {random_integers}")
print(f"Normal distribution: {normal_distribution}")

# Modern Generator interface (recommended)
print("\n2. Modern Generator interface:")
rng = default_rng(seed=42)  # Create generator with seed

gen_floats = rng.random(5)
gen_integers = rng.integers(1, 10, size=5)
gen_normal = rng.normal(0, 1, 5)

print(f"Generator floats: {gen_floats}")
print(f"Generator integers: {gen_integers}")
print(f"Generator normal: {gen_normal}")

print("\n=== DISTRIBUTION TYPES ===")

# Uniform distributions
print("3. Uniform Distributions:")
uniform_01 = rng.random(5)  # [0, 1)
uniform_range = rng.uniform(-5, 5, 5)  # Custom range
discrete_uniform = rng.integers(1, 7, 10)  # Dice rolls

print(f"Uniform [0,1): {uniform_01}")
print(f"Uniform [-5,5): {uniform_range}")
print(f"Discrete uniform (dice): {discrete_uniform}")

# Normal (Gaussian) distributions
print("\n4. Normal Distributions:")
standard_normal = rng.standard_normal(5)
custom_normal = rng.normal(loc=100, scale=15, size=5)  # IQ scores
multivariate_normal = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 3)

print(f"Standard normal: {standard_normal}")
print(f"Custom normal (μ=100, σ=15): {custom_normal}")
print(f"Multivariate normal:\n{multivariate_normal}")

# Other common distributions
print("\n5. Other Distributions:")

# Exponential (waiting times)
exponential = rng.exponential(scale=2.0, size=5)
print(f"Exponential (scale=2): {exponential}")

# Poisson (count data)
poisson = rng.poisson(lam=3.0, size=5)
print(f"Poisson (λ=3): {poisson}")

# Binomial (number of successes)
binomial = rng.binomial(n=10, p=0.3, size=5)
print(f"Binomial (n=10, p=0.3): {binomial}")

# Beta distribution
beta = rng.beta(2, 5, 5)
print(f"Beta (α=2, β=5): {beta}")

# Gamma distribution
gamma = rng.gamma(2, 2, 5)
print(f"Gamma (shape=2, scale=2): {gamma}")

print("\n=== ARRAY GENERATION ===")

# Multi-dimensional arrays
print("6. Multi-dimensional Arrays:")
matrix_2d = rng.random((3, 4))
tensor_3d = rng.integers(0, 100, (2, 3, 4))
normal_matrix = rng.normal(0, 1, (4, 4))

print(f"2D random matrix:\n{matrix_2d}")
print(f"3D tensor shape: {tensor_3d.shape}")
print(f"Normal matrix:\n{normal_matrix}")

# Structured arrays
print("\n7. Structured Arrays:")
def generate_sample_data(n_samples):
    """Generate structured sample data"""
    names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
    
    data = {
        'name': rng.choice(names, n_samples),
        'age': rng.integers(18, 80, n_samples),
        'salary': rng.normal(50000, 15000, n_samples),
        'performance': rng.beta(2, 2, n_samples)  # Performance score [0,1]
    }
    return data

sample_data = generate_sample_data(5)
print("Sample structured data:")
for key, values in sample_data.items():
    print(f"{key}: {values}")

print("\n=== SAMPLING AND PERMUTATIONS ===")

# Sampling without replacement
population = np.arange(1, 101)  # Numbers 1-100
sample_without_replacement = rng.choice(population, size=10, replace=False)
print(f"Sample without replacement: {sample_without_replacement}")

# Sampling with weights
items = ['A', 'B', 'C', 'D']
weights = [0.1, 0.2, 0.3, 0.4]
weighted_sample = rng.choice(items, size=10, p=weights)
print(f"Weighted sample: {weighted_sample}")

# Permutations and shuffling
original_array = np.arange(10)
shuffled = rng.permutation(original_array)
print(f"Original: {original_array}")
print(f"Shuffled: {shuffled}")

# In-place shuffle
to_shuffle = np.arange(10)
rng.shuffle(to_shuffle)
print(f"In-place shuffled: {to_shuffle}")

print("\n=== ADVANCED TECHNIQUES ===")

# Custom probability distributions
def generate_custom_distribution(n_samples):
    """Generate samples from custom distribution using inverse transform"""
    # Generate exponential distribution using uniform random numbers
    u = rng.random(n_samples)
    custom_exp = -np.log(1 - u) / 2.0  # Exponential with rate=2
    return custom_exp

custom_samples = generate_custom_distribution(5)
print(f"Custom exponential samples: {custom_samples}")

# Acceptance-rejection sampling
def acceptance_rejection_normal(n_samples, target_mean=0, target_std=1):
    """Generate normal samples using acceptance-rejection method"""
    samples = []
    
    while len(samples) < n_samples:
        # Generate candidate from uniform distribution
        x = rng.uniform(-4, 4)  # Proposal distribution
        
        # Calculate acceptance probability
        target_density = np.exp(-0.5 * ((x - target_mean) / target_std)**2)
        proposal_density = 1/8  # Uniform density over [-4, 4]
        
        # Accept or reject
        if rng.random() < target_density / (proposal_density * np.sqrt(2 * np.pi) * target_std * 2.5):
            samples.append(x)
    
    return np.array(samples)

ar_samples = acceptance_rejection_normal(5)
print(f"Acceptance-rejection normal: {ar_samples}")

# Monte Carlo simulation example
def monte_carlo_pi_estimation(n_samples):
    """Estimate π using Monte Carlo method"""
    # Generate random points in unit square
    x = rng.uniform(-1, 1, n_samples)
    y = rng.uniform(-1, 1, n_samples)
    
    # Check if points are inside unit circle
    inside_circle = (x**2 + y**2) <= 1
    
    # Estimate π
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    return pi_estimate

pi_est = monte_carlo_pi_estimation(100000)
print(f"Monte Carlo π estimation: {pi_est:.6f} (error: {abs(pi_est - np.pi):.6f})")

print("\n=== REPRODUCIBILITY AND SEEDING ===")

# Demonstrate seeding effects
def demonstrate_seeding():
    """Show how seeding affects reproducibility"""
    # Same seed produces same results
    rng1 = default_rng(seed=123)
    rng2 = default_rng(seed=123)
    
    sample1 = rng1.random(5)
    sample2 = rng2.random(5)
    
    print(f"RNG1 with seed 123: {sample1}")
    print(f"RNG2 with seed 123: {sample2}")
    print(f"Are identical: {np.array_equal(sample1, sample2)}")
    
    # Different seeds produce different results
    rng3 = default_rng(seed=456)
    sample3 = rng3.random(5)
    print(f"RNG3 with seed 456: {sample3}")
    print(f"Different from seed 123: {not np.array_equal(sample1, sample3)}")

demonstrate_seeding()

# State management
print("\n8. State Management:")
rng_state = rng.__getstate__()  # Save current state
before_state = rng.random(3)
rng.__setstate__(rng_state)  # Restore state
after_state = rng.random(3)

print(f"Before state save: {before_state}")
print(f"After state restore: {after_state}")
print(f"States identical: {np.array_equal(before_state, after_state)}")

print("\n=== PERFORMANCE OPTIMIZATION ===")

# Performance comparison
def compare_random_generation_performance():
    """Compare performance of different random generation methods"""
    import time
    
    n_samples = 1000000
    
    # Legacy method
    start = time.time()
    _ = np.random.random(n_samples)
    legacy_time = time.time() - start
    
    # Generator method
    rng = default_rng()
    start = time.time()
    _ = rng.random(n_samples)
    generator_time = time.time() - start
    
    print(f"Legacy method: {legacy_time:.6f}s")
    print(f"Generator method: {generator_time:.6f}s")
    print(f"Speed ratio: {legacy_time/generator_time:.2f}x")

compare_random_generation_performance()

# Memory-efficient generation for large datasets
def generate_large_dataset_chunked(total_size, chunk_size=1000000):
    """Generate large random datasets in chunks to manage memory"""
    rng = default_rng(seed=42)
    
    n_chunks = total_size // chunk_size
    remainder = total_size % chunk_size
    
    print(f"Generating {total_size} samples in {n_chunks} chunks of {chunk_size}")
    
    # Process chunks
    for i in range(n_chunks):
        chunk = rng.random(chunk_size)
        # Process chunk (save to file, compute statistics, etc.)
        chunk_mean = np.mean(chunk)
        print(f"Chunk {i+1}/{n_chunks}: mean = {chunk_mean:.6f}")
    
    # Handle remainder
    if remainder > 0:
        final_chunk = rng.random(remainder)
        final_mean = np.mean(final_chunk)
        print(f"Final chunk: mean = {final_mean:.6f}")

# Demonstrate chunked generation
generate_large_dataset_chunked(5000000, chunk_size=1000000)

# Error handling and validation
def robust_random_generation(distribution, params, size, validation_func=None):
    """Robust random number generation with validation"""
    try:
        rng = default_rng()
        
        # Generate samples based on distribution type
        if distribution == 'normal':
            samples = rng.normal(params['loc'], params['scale'], size)
        elif distribution == 'uniform':
            samples = rng.uniform(params['low'], params['high'], size)
        elif distribution == 'exponential':
            samples = rng.exponential(params['scale'], size)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        # Apply validation if provided
        if validation_func:
            valid_samples = validation_func(samples)
            print(f"Generated {len(samples)} samples, {len(valid_samples)} valid")
            return valid_samples
        
        return samples
    
    except Exception as e:
        print(f"Error in random generation: {e}")
        return np.array([])

# Test robust generation
def validate_positive(samples):
    """Validation function to keep only positive values"""
    return samples[samples > 0]

valid_samples = robust_random_generation(
    'normal', 
    {'loc': 0, 'scale': 1}, 
    1000, 
    validation_func=validate_positive
)
print(f"Valid positive normal samples: {len(valid_samples)}")
```

### Explanation

1. **Generator vs Legacy**: Modern Generator interface provides better performance and features
2. **Distribution Selection**: Choose appropriate distributions based on data characteristics
3. **Seeding Strategy**: Use seeds for reproducible results in scientific applications
4. **Memory Management**: Consider memory usage for large random datasets

### Use Cases

1. **Monte Carlo Simulations**: Financial modeling, physics simulations
2. **Machine Learning**: Data augmentation, bootstrap sampling, random initialization
3. **Statistical Testing**: Hypothesis testing, confidence intervals
4. **Game Development**: Procedural generation, AI behavior randomization
5. **Cryptography**: Generating random keys and nonces (with secure generators)

### Best Practices

1. **Use the modern Generator interface** for new code
2. **Set seeds for reproducibility** in scientific applications
3. **Choose appropriate distributions** based on domain knowledge
4. **Validate generated data** for edge cases and constraints
5. **Consider memory usage** for large-scale random generation

### Pitfalls

1. **Predictable Seeds**: Using simple or time-based seeds in security contexts
2. **Distribution Misunderstanding**: Wrong distribution choice for the problem
3. **Memory Exhaustion**: Generating very large arrays without chunking
4. **State Management**: Losing reproducibility due to unmanaged generator state
5. **Performance Issues**: Using legacy interface when performance matters

### Debugging

```python
def debug_random_generation(rng, distribution_func, *args, **kwargs):
    """Debug random number generation"""
    try:
        samples = distribution_func(*args, **kwargs)
        
        print(f"Generated {len(samples)} samples")
        print(f"Data type: {samples.dtype}")
        print(f"Shape: {samples.shape}")
        print(f"Min: {np.min(samples):.6f}")
        print(f"Max: {np.max(samples):.6f}")
        print(f"Mean: {np.mean(samples):.6f}")
        print(f"Std: {np.std(samples):.6f}")
        print(f"Has NaN: {np.isnan(samples).any()}")
        print(f"Has Inf: {np.isinf(samples).any()}")
        
        return samples
    except Exception as e:
        print(f"Error in random generation: {e}")
        return None
```

### Optimization

1. **Pre-allocate Arrays**: Use size parameter instead of appending
2. **Vectorized Operations**: Generate multiple samples at once
3. **Choose Efficient Distributions**: Some distributions are faster than others
4. **Chunked Processing**: Process large datasets in manageable chunks
5. **Avoid Loops**: Use NumPy's vectorized random functions

---

## Question 11

**How do you normalize an array in NumPy?**

### Theory

Array normalization is the process of scaling array values to a specific range or distribution. Common normalization techniques include min-max scaling (0-1 range), z-score standardization (mean=0, std=1), unit vector normalization, and robust scaling. The choice depends on data distribution, outliers presence, and application requirements. Proper normalization is crucial for machine learning, data analysis, and numerical stability.

### Code Example

```python
import numpy as np
from scipy import stats

# Create sample data for demonstration
np.random.seed(42)
data = np.random.normal(50, 15, 20)  # Mean=50, std=15
outlier_data = np.append(data, [120, -10])  # Add outliers

print("Original data:", data[:10])  # Show first 10 values
print(f"Original stats - Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")
print(f"Original range: [{np.min(data):.2f}, {np.max(data):.2f}]")

print("\n=== NORMALIZATION METHODS ===")

# Method 1: Min-Max Normalization (0-1 scaling)
print("1. Min-Max Normalization:")

def min_max_normalize(arr, target_min=0, target_max=1):
    """Normalize array to [target_min, target_max] range"""
    arr_min, arr_max = np.min(arr), np.max(arr)
    
    # Handle case where all values are the same
    if arr_max == arr_min:
        return np.full_like(arr, (target_min + target_max) / 2)
    
    # Scale to [0, 1] then to target range
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return normalized * (target_max - target_min) + target_min

min_max_norm = min_max_normalize(data)
print(f"Min-Max normalized: {min_max_norm[:5]}")
print(f"Range: [{np.min(min_max_norm):.2f}, {np.max(min_max_norm):.2f}]")

# Custom range normalization
custom_range = min_max_normalize(data, target_min=-1, target_max=1)
print(f"Custom range [-1, 1]: {custom_range[:5]}")

# Method 2: Z-Score Standardization
print("\n2. Z-Score Standardization:")

def z_score_normalize(arr):
    """Standardize array to mean=0, std=1"""
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # Use sample std
    
    # Handle case where std is 0
    if std == 0:
        return np.zeros_like(arr)
    
    return (arr - mean) / std

z_normalized = z_score_normalize(data)
print(f"Z-score normalized: {z_normalized[:5]}")
print(f"Mean: {np.mean(z_normalized):.6f}, Std: {np.std(z_normalized, ddof=1):.6f}")

# Using scipy.stats for comparison
scipy_z = stats.zscore(data)
print(f"Scipy z-score: {scipy_z[:5]}")
print(f"Results match: {np.allclose(z_normalized, scipy_z)}")

# Method 3: Unit Vector Normalization (L2 norm)
print("\n3. Unit Vector Normalization:")

def unit_vector_normalize(arr):
    """Normalize array to unit vector (L2 norm = 1)"""
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm

unit_normalized = unit_vector_normalize(data)
print(f"Unit vector normalized: {unit_normalized[:5]}")
print(f"L2 norm: {np.linalg.norm(unit_normalized):.6f}")

# Method 4: Robust Scaling (using median and IQR)
print("\n4. Robust Scaling:")

def robust_scale(arr):
    """Scale using median and interquartile range"""
    median = np.median(arr)
    q75 = np.percentile(arr, 75)
    q25 = np.percentile(arr, 25)
    iqr = q75 - q25
    
    if iqr == 0:
        return np.zeros_like(arr)
    
    return (arr - median) / iqr

robust_scaled = robust_scale(data)
print(f"Robust scaled: {robust_scaled[:5]}")
print(f"Median: {np.median(robust_scaled):.6f}")

# Method 5: Decimal Scaling
print("\n5. Decimal Scaling:")

def decimal_scale(arr):
    """Scale by dividing by 10^k where k is number of digits"""
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        return arr
    
    k = len(str(int(max_val)))
    return arr / (10 ** k)

decimal_scaled = decimal_scale(data)
print(f"Decimal scaled: {decimal_scaled[:5]}")
print(f"Max absolute value: {np.max(np.abs(decimal_scaled)):.6f}")

print("\n=== MULTI-DIMENSIONAL NORMALIZATION ===")

# 2D array normalization
matrix = np.random.normal(0, 1, (4, 5))
print("Original 2D matrix:")
print(matrix)

# Normalize entire matrix
print("\n6. Entire Matrix Normalization:")
matrix_normalized = min_max_normalize(matrix)
print("Entire matrix min-max normalized:")
print(matrix_normalized)

# Normalize along axes
print("\n7. Axis-wise Normalization:")

def normalize_along_axis(arr, axis=None, method='min_max'):
    """Normalize array along specified axis"""
    if method == 'min_max':
        min_vals = np.min(arr, axis=axis, keepdims=True)
        max_vals = np.max(arr, axis=axis, keepdims=True)
        
        # Handle division by zero
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)
        
        return (arr - min_vals) / range_vals
    
    elif method == 'z_score':
        mean_vals = np.mean(arr, axis=axis, keepdims=True)
        std_vals = np.std(arr, axis=axis, keepdims=True)
        
        # Handle division by zero
        std_vals = np.where(std_vals == 0, 1, std_vals)
        
        return (arr - mean_vals) / std_vals

# Normalize along rows (axis=1)
row_normalized = normalize_along_axis(matrix, axis=1, method='min_max')
print("Row-wise normalized (each row [0,1]):")
print(row_normalized)

# Normalize along columns (axis=0)
col_normalized = normalize_along_axis(matrix, axis=0, method='z_score')
print("Column-wise z-score normalized:")
print(col_normalized)

print("\n=== OUTLIER-ROBUST NORMALIZATION ===")

print("8. Handling Outliers:")
print(f"Data with outliers: min={np.min(outlier_data):.1f}, max={np.max(outlier_data):.1f}")

# Standard min-max (affected by outliers)
standard_minmax = min_max_normalize(outlier_data)
print(f"Standard min-max range: [{np.min(standard_minmax):.3f}, {np.max(standard_minmax):.3f}]")

# Robust percentile-based normalization
def percentile_normalize(arr, lower_percentile=5, upper_percentile=95):
    """Normalize using percentiles instead of min/max"""
    lower = np.percentile(arr, lower_percentile)
    upper = np.percentile(arr, upper_percentile)
    
    if upper == lower:
        return np.full_like(arr, 0.5)
    
    # Clip outliers to percentile range
    clipped = np.clip(arr, lower, upper)
    return (clipped - lower) / (upper - lower)

percentile_norm = percentile_normalize(outlier_data)
print(f"Percentile-based normalization (5th-95th percentile):")
print(f"Range: [{np.min(percentile_norm):.3f}, {np.max(percentile_norm):.3f}]")

# Winsorization + normalization
def winsorize_normalize(arr, limits=(0.05, 0.05)):
    """Winsorize then normalize"""
    winsorized = stats.mstats.winsorize(arr, limits=limits)
    return min_max_normalize(winsorized)

winsorized_norm = winsorize_normalize(outlier_data)
print(f"Winsorized + normalized:")
print(f"Range: [{np.min(winsorized_norm):.3f}, {np.max(winsorized_norm):.3f}]")

print("\n=== SPECIALIZED NORMALIZATION ===")

# Log normalization for skewed data
print("9. Log Normalization:")
skewed_data = np.random.exponential(2, 100)  # Exponentially distributed data

def log_normalize(arr, method='log1p'):
    """Apply log transformation then normalize"""
    if method == 'log1p':
        log_data = np.log1p(arr)  # log(1 + x), handles zeros
    elif method == 'log':
        log_data = np.log(arr + 1e-8)  # Add small constant to avoid log(0)
    else:
        raise ValueError("Method must be 'log' or 'log1p'")
    
    return min_max_normalize(log_data)

log_normalized = log_normalize(skewed_data)
print(f"Original skewness: {stats.skew(skewed_data):.2f}")
print(f"Log-normalized skewness: {stats.skew(log_normalized):.2f}")

# Box-Cox transformation
print("\n10. Box-Cox Normalization:")

def boxcox_normalize(arr):
    """Apply Box-Cox transformation then normalize"""
    # Ensure all values are positive
    if np.min(arr) <= 0:
        arr = arr - np.min(arr) + 1e-8
    
    transformed, lambda_param = stats.boxcox(arr)
    normalized = min_max_normalize(transformed)
    
    return normalized, lambda_param

if np.min(skewed_data) > 0:  # Box-Cox requires positive values
    boxcox_norm, lambda_val = boxcox_normalize(skewed_data)
    print(f"Box-Cox parameter λ: {lambda_val:.4f}")
    print(f"Box-Cox normalized skewness: {stats.skew(boxcox_norm):.2f}")

print("\n=== BATCH NORMALIZATION ===")

# Simulate batch processing
print("11. Batch Normalization:")

def batch_normalize(batch, running_mean=None, running_var=None, momentum=0.1, epsilon=1e-8):
    """Implement batch normalization as used in deep learning"""
    batch_mean = np.mean(batch, axis=0)
    batch_var = np.var(batch, axis=0)
    
    # Update running statistics
    if running_mean is not None:
        running_mean = momentum * batch_mean + (1 - momentum) * running_mean
    else:
        running_mean = batch_mean
    
    if running_var is not None:
        running_var = momentum * batch_var + (1 - momentum) * running_var
    else:
        running_var = batch_var
    
    # Normalize
    normalized = (batch - batch_mean) / np.sqrt(batch_var + epsilon)
    
    return normalized, running_mean, running_var

# Simulate multiple batches
batch_size, features = 32, 10
batches = [np.random.normal(0, 1, (batch_size, features)) for _ in range(5)]

running_mean = None
running_var = None

for i, batch in enumerate(batches):
    normalized_batch, running_mean, running_var = batch_normalize(
        batch, running_mean, running_var
    )
    print(f"Batch {i+1} - Mean: {np.mean(normalized_batch):.6f}, Std: {np.std(normalized_batch):.6f}")

print("\n=== NORMALIZATION PIPELINE ===")

# Complete normalization pipeline
class NormalizationPipeline:
    """Complete normalization pipeline with multiple methods"""
    
    def __init__(self):
        self.method = None
        self.params = {}
    
    def fit(self, data, method='min_max', **kwargs):
        """Fit normalization parameters"""
        self.method = method
        
        if method == 'min_max':
            self.params = {
                'min': np.min(data),
                'max': np.max(data),
                'target_min': kwargs.get('target_min', 0),
                'target_max': kwargs.get('target_max', 1)
            }
        
        elif method == 'z_score':
            self.params = {
                'mean': np.mean(data),
                'std': np.std(data, ddof=1)
            }
        
        elif method == 'robust':
            self.params = {
                'median': np.median(data),
                'iqr': np.percentile(data, 75) - np.percentile(data, 25)
            }
        
        return self
    
    def transform(self, data):
        """Apply normalization"""
        if self.method == 'min_max':
            if self.params['max'] == self.params['min']:
                return np.full_like(data, (self.params['target_min'] + self.params['target_max']) / 2)
            
            normalized = (data - self.params['min']) / (self.params['max'] - self.params['min'])
            return normalized * (self.params['target_max'] - self.params['target_min']) + self.params['target_min']
        
        elif self.method == 'z_score':
            if self.params['std'] == 0:
                return np.zeros_like(data)
            return (data - self.params['mean']) / self.params['std']
        
        elif self.method == 'robust':
            if self.params['iqr'] == 0:
                return np.zeros_like(data)
            return (data - self.params['median']) / self.params['iqr']
    
    def fit_transform(self, data, method='min_max', **kwargs):
        """Fit and transform in one step"""
        return self.fit(data, method, **kwargs).transform(data)
    
    def inverse_transform(self, normalized_data):
        """Reverse the normalization"""
        if self.method == 'min_max':
            denorm = (normalized_data - self.params['target_min']) / (self.params['target_max'] - self.params['target_min'])
            return denorm * (self.params['max'] - self.params['min']) + self.params['min']
        
        elif self.method == 'z_score':
            return normalized_data * self.params['std'] + self.params['mean']
        
        elif self.method == 'robust':
            return normalized_data * self.params['iqr'] + self.params['median']

# Test the pipeline
pipeline = NormalizationPipeline()

# Train data normalization
train_data = np.random.normal(100, 20, 1000)
normalized_train = pipeline.fit_transform(train_data, method='z_score')

# Test data normalization (using train parameters)
test_data = np.random.normal(95, 25, 100)
normalized_test = pipeline.transform(test_data)

print("Pipeline normalization:")
print(f"Train data - Mean: {np.mean(normalized_train):.6f}, Std: {np.std(normalized_train, ddof=1):.6f}")
print(f"Test data - Mean: {np.mean(normalized_test):.6f}, Std: {np.std(normalized_test, ddof=1):.6f}")

# Inverse transformation
recovered_data = pipeline.inverse_transform(normalized_train[:10])
original_sample = train_data[:10]
print(f"Inverse transform error: {np.mean(np.abs(recovered_data - original_sample)):.10f}")

print("\n=== PERFORMANCE OPTIMIZATION ===")

# Performance comparison
def compare_normalization_performance():
    """Compare performance of different normalization methods"""
    import time
    
    large_data = np.random.normal(0, 1, 1000000)
    iterations = 100
    
    # Min-max normalization
    start = time.time()
    for _ in range(iterations):
        _ = min_max_normalize(large_data)
    minmax_time = time.time() - start
    
    # Z-score normalization
    start = time.time()
    for _ in range(iterations):
        _ = z_score_normalize(large_data)
    zscore_time = time.time() - start
    
    # Robust scaling
    start = time.time()
    for _ in range(iterations):
        _ = robust_scale(large_data)
    robust_time = time.time() - start
    
    print(f"Min-max time: {minmax_time:.6f}s")
    print(f"Z-score time: {zscore_time:.6f}s")
    print(f"Robust scaling time: {robust_time:.6f}s")

compare_normalization_performance()
```

### Explanation

1. **Method Selection**: Choose normalization based on data distribution and application needs
2. **Outlier Handling**: Use robust methods when outliers are present
3. **Axis Considerations**: Apply normalization along appropriate axes for multi-dimensional data
4. **Parameter Preservation**: Save normalization parameters for consistent test data processing

### Use Cases

1. **Machine Learning**: Feature scaling for algorithms sensitive to scale
2. **Deep Learning**: Batch normalization for training stability
3. **Data Visualization**: Scaling data for consistent plotting ranges
4. **Signal Processing**: Normalizing signals for comparison and analysis
5. **Image Processing**: Pixel value normalization for computer vision

### Best Practices

1. **Fit on training data only** - apply same parameters to test data
2. **Choose method based on data distribution** - z-score for normal, robust for skewed
3. **Handle edge cases** like constant values and zeros
4. **Preserve normalization parameters** for inverse transformations
5. **Consider outliers** when selecting normalization method

### Pitfalls

1. **Data Leakage**: Using test data statistics for normalization
2. **Outlier Sensitivity**: Min-max scaling affected by extreme values
3. **Zero Division**: Constant arrays causing division by zero
4. **Method Mismatch**: Wrong normalization method for data distribution
5. **Scale Inconsistency**: Different scaling between training and test data

### Debugging

```python
def debug_normalization(original, normalized, method='unknown'):
    """Debug normalization results"""
    print(f"Normalization method: {method}")
    print(f"Original - Mean: {np.mean(original):.6f}, Std: {np.std(original):.6f}")
    print(f"Original - Min: {np.min(original):.6f}, Max: {np.max(original):.6f}")
    print(f"Normalized - Mean: {np.mean(normalized):.6f}, Std: {np.std(normalized):.6f}")
    print(f"Normalized - Min: {np.min(normalized):.6f}, Max: {np.max(normalized):.6f}")
    print(f"Has NaN: {np.isnan(normalized).any()}")
    print(f"Has Inf: {np.isinf(normalized).any()}")
```

### Optimization

1. **Vectorized Operations**: Use NumPy's built-in functions for speed
2. **In-place Operations**: Modify arrays in-place when possible
3. **Batch Processing**: Normalize multiple arrays together
4. **Memory Efficiency**: Avoid unnecessary copies during normalization
5. **Precomputed Statistics**: Cache statistics for repeated normalizations

---

## Question 12

**How can you compute percentiles with NumPy?**

### Theory

Percentiles are values below which a certain percentage of data falls. They're essential for statistical analysis, outlier detection, and data distribution understanding. NumPy provides `np.percentile()`, `np.quantile()`, and `np.nanpercentile()` functions with various interpolation methods and axis handling capabilities.

### Code Example

```python
import numpy as np

# Sample data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Data: {data}")

# Basic percentile computation
p25 = np.percentile(data, 25)  # 25th percentile (Q1)
p50 = np.percentile(data, 50)  # 50th percentile (median, Q2)
p75 = np.percentile(data, 75)  # 75th percentile (Q3)

print(f"25th percentile: {p25}")
print(f"50th percentile: {p50}")
print(f"75th percentile: {p75}")

# Multiple percentiles at once
percentiles = np.percentile(data, [25, 50, 75, 90, 95])
print(f"Multiple percentiles: {percentiles}")

# Using quantiles (0-1 scale)
q25 = np.quantile(data, 0.25)
print(f"25th quantile: {q25}")

# 2D array percentiles
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
axis0_percentiles = np.percentile(matrix, 50, axis=0)  # Along columns
axis1_percentiles = np.percentile(matrix, 50, axis=1)  # Along rows
print(f"50th percentile along axis 0: {axis0_percentiles}")
print(f"50th percentile along axis 1: {axis1_percentiles}")
```

### Use Cases
- Statistical analysis and data exploration
- Outlier detection and removal
- Performance benchmarking and SLA monitoring
- Risk assessment in finance

---

## Question 13

**How do you calculate the correlation coefficient using NumPy?**

### Theory

Correlation coefficients measure the linear relationship between variables. The Pearson correlation coefficient ranges from -1 to 1, where -1 indicates perfect negative correlation, 0 indicates no linear correlation, and 1 indicates perfect positive correlation.

### Code Example

```python
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # Perfect positive correlation

# Correlation coefficient
correlation_matrix = np.corrcoef(x, y)
print(f"Correlation matrix:\n{correlation_matrix}")
print(f"Correlation coefficient: {correlation_matrix[0, 1]}")

# Multiple variables
z = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # Negative correlation with x
multi_corr = np.corrcoef([x, y, z])
print(f"Multi-variable correlation matrix:\n{multi_corr}")

# Manual calculation
def pearson_correlation(x, y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    return numerator / denominator

manual_corr = pearson_correlation(x, y)
print(f"Manual correlation: {manual_corr}")
```

### Use Cases
- Feature selection in machine learning
- Financial portfolio analysis
- Scientific data analysis
- Quality control and process monitoring

---

## Question 14

**Explain the use of the np.cumsum() and np.cumprod() functions.**

### Theory

Cumulative functions compute running totals or products along arrays. `np.cumsum()` calculates cumulative sums, while `np.cumprod()` calculates cumulative products. These are essential for time series analysis, financial calculations, and algorithmic implementations.

### Code Example

```python
import numpy as np

# Sample data
data = np.array([1, 2, 3, 4, 5])
print(f"Original data: {data}")

# Cumulative sum
cumsum_result = np.cumsum(data)
print(f"Cumulative sum: {cumsum_result}")

# Cumulative product
cumprod_result = np.cumprod(data)
print(f"Cumulative product: {cumprod_result}")

# 2D arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D matrix:\n{matrix}")

# Cumulative sum along different axes
cumsum_axis0 = np.cumsum(matrix, axis=0)
cumsum_axis1 = np.cumsum(matrix, axis=1)
print(f"Cumsum along axis 0:\n{cumsum_axis0}")
print(f"Cumsum along axis 1:\n{cumsum_axis1}")

# Financial application - compound returns
returns = np.array([0.02, 0.03, -0.01, 0.04, 0.02])  # Daily returns
compound_growth = np.cumprod(1 + returns)
print(f"Compound growth: {compound_growth}")

# Running totals for analysis
sales = np.array([100, 150, 200, 120, 180])
running_total = np.cumsum(sales)
print(f"Running sales total: {running_total}")
```

### Use Cases
- Financial modeling (compound interest, portfolio returns)
- Time series analysis and trend detection
- Algorithmic trading strategies
- Inventory and supply chain management

---

## Question 15

**How do you stack multiple arrays vertically and horizontally?**

### Theory

Array stacking combines multiple arrays along specified axes. NumPy provides `np.vstack()` for vertical stacking, `np.hstack()` for horizontal stacking, and `np.concatenate()` for general-purpose stacking along any axis.

### Code Example

```python
import numpy as np

# Sample arrays
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
arr3 = np.array([[13, 14, 15]])

print(f"Array 1:\n{arr1}")
print(f"Array 2:\n{arr2}")

# Vertical stacking (along rows)
vstack_result = np.vstack([arr1, arr2])
print(f"Vertical stack:\n{vstack_result}")

# Horizontal stacking (along columns)
hstack_result = np.hstack([arr1, arr2])
print(f"Horizontal stack:\n{hstack_result}")

# Using concatenate for more control
concat_axis0 = np.concatenate([arr1, arr2], axis=0)  # Same as vstack
concat_axis1 = np.concatenate([arr1, arr2], axis=1)  # Same as hstack
print(f"Concatenate axis 0:\n{concat_axis0}")

# 3D stacking
depth_stack = np.dstack([arr1, arr2])
print(f"Depth stack shape: {depth_stack.shape}")

# Column and row stacking for 1D arrays
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

column_stack = np.column_stack([vec1, vec2])
row_stack = np.row_stack([vec1, vec2])
print(f"Column stack:\n{column_stack}")
print(f"Row stack:\n{row_stack}")
```

### Use Cases
- Data preprocessing and feature engineering
- Image processing and computer vision
- Batch processing in machine learning
- Signal processing and data fusion

---

## Question 16

**Describe the process for creating a structured array in NumPy.**

### Theory

Structured arrays are NumPy arrays with named fields, similar to database records or C structs. They allow heterogeneous data types within a single array, making them ideal for representing tabular data with mixed types.

### Code Example

```python
import numpy as np

# Define data type for structured array
dtype = [('name', 'U20'), ('age', 'i4'), ('salary', 'f8'), ('active', '?')]

# Create structured array
employees = np.array([
    ('Alice', 25, 50000.0, True),
    ('Bob', 30, 60000.0, True),
    ('Charlie', 35, 70000.0, False)
], dtype=dtype)

print(f"Structured array:\n{employees}")
print(f"Names: {employees['name']}")
print(f"Ages: {employees['age']}")

# Alternative creation method
dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
points = np.zeros(3, dtype=dt)
points['x'] = [1.0, 2.0, 3.0]
points['y'] = [4.0, 5.0, 6.0]
points['z'] = [7.0, 8.0, 9.0]
print(f"3D points:\n{points}")

# Nested structures
nested_dtype = [('person', [('name', 'U20'), ('age', 'i4')]), 
                ('address', [('street', 'U30'), ('city', 'U20')])]

records = np.array([
    (('Alice', 25), ('123 Main St', 'New York')),
    (('Bob', 30), ('456 Oak Ave', 'Boston'))
], dtype=nested_dtype)

print(f"Nested structure:\n{records}")
print(f"Names: {records['person']['name']}")
```

### Use Cases
- Scientific data with heterogeneous measurements
- Database-like operations in NumPy
- File I/O with complex data formats
- Interfacing with C/Fortran code

---

## Question 17

**How do you save and load NumPy arrays to and from disk?**

### Theory

NumPy provides multiple methods for array persistence: `.npy` format for single arrays, `.npz` for multiple arrays, and text formats for human-readable storage. The choice depends on performance requirements, file size constraints, and interoperability needs.

### Code Example

```python
import numpy as np

# Sample data
data = np.random.randn(1000, 100)
labels = np.random.randint(0, 10, 1000)

# Save single array (binary format)
np.save('data.npy', data)
loaded_data = np.load('data.npy')
print(f"Loaded data shape: {loaded_data.shape}")

# Save multiple arrays
np.savez('arrays.npz', data=data, labels=labels)
loaded_arrays = np.load('arrays.npz')
print(f"Loaded arrays: {list(loaded_arrays.keys())}")

# Save compressed
np.savez_compressed('compressed.npz', data=data, labels=labels)

# Save as text
small_array = np.array([[1, 2, 3], [4, 5, 6]])
np.savetxt('array.txt', small_array, delimiter=',')
loaded_text = np.loadtxt('array.txt', delimiter=',')
print(f"Text loaded array:\n{loaded_text}")

# Memory mapping for large files
large_array = np.random.randn(10000, 1000)
np.save('large_array.npy', large_array)
mmap_array = np.load('large_array.npy', mmap_mode='r')
print(f"Memory mapped array shape: {mmap_array.shape}")

# Custom binary format
def save_custom(filename, arrays_dict):
    with open(filename, 'wb') as f:
        for name, array in arrays_dict.items():
            name_bytes = name.encode('utf-8')
            f.write(len(name_bytes).to_bytes(4, 'little'))
            f.write(name_bytes)
            f.write(array.shape[0].to_bytes(4, 'little'))
            f.write(array.shape[1].to_bytes(4, 'little'))
            f.write(array.tobytes())

save_custom('custom.bin', {'features': data[:10], 'targets': labels[:10]})
```

### Use Cases
- Machine learning model checkpoints
- Scientific data archival
- Distributed computing data exchange
- Application state persistence

---

## Question 18

**Write a NumPy code to create a 3x3 identity matrix.**

### Theory

An identity matrix is a square matrix with ones on the main diagonal and zeros elsewhere. It acts as the multiplicative identity in matrix operations. NumPy provides several methods to create identity matrices efficiently.

### Code Example

```python
import numpy as np

# Method 1: Using np.eye() - most common
identity_3x3 = np.eye(3)
print("3x3 Identity matrix using np.eye():")
print(identity_3x3)

# Method 2: Using np.identity()
identity_alt = np.identity(3)
print("\n3x3 Identity matrix using np.identity():")
print(identity_alt)

# Method 3: Manual creation
manual_identity = np.zeros((3, 3))
np.fill_diagonal(manual_identity, 1)
print("\nManual 3x3 identity matrix:")
print(manual_identity)

# Method 4: Using broadcasting
indices = np.arange(3)
broadcast_identity = (indices[:, None] == indices).astype(int)
print("\nBroadcast 3x3 identity matrix:")
print(broadcast_identity)

# Different sizes and types
identity_5x5 = np.eye(5)
identity_float32 = np.eye(3, dtype=np.float32)
identity_complex = np.eye(3, dtype=complex)

print(f"\n5x5 identity shape: {identity_5x5.shape}")
print(f"Float32 identity dtype: {identity_float32.dtype}")
print(f"Complex identity:\n{identity_complex}")

# Non-square identity (rectangular)
rect_identity = np.eye(3, 4)  # 3 rows, 4 columns
print(f"\nRectangular identity (3x4):\n{rect_identity}")

# Shifted diagonal
shifted_identity = np.eye(3, k=1)  # Diagonal shifted right
print(f"\nShifted diagonal (k=1):\n{shifted_identity}")
```

### Use Cases
- Linear algebra operations and matrix inversions
- Neural network weight initialization
- Solving systems of linear equations
- Computer graphics transformations

---

## Question 19

**Code a function in NumPy to compute the moving average of a 1D array.**

### Theory

Moving average smooths data by computing the average of a sliding window. It's used for trend analysis, noise reduction, and signal processing. Different window types and edge handling methods affect the results.

### Code Example

```python
import numpy as np

def moving_average(data, window_size, mode='valid'):
    """
    Compute moving average of 1D array
    
    Parameters:
    - data: Input 1D array
    - window_size: Size of moving window
    - mode: 'valid', 'same', or 'full'
    """
    if window_size > len(data):
        raise ValueError("Window size cannot be larger than data length")
    
    # Method 1: Using np.convolve
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode=mode)

def weighted_moving_average(data, weights):
    """Compute weighted moving average"""
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize weights
    return np.convolve(data, weights[::-1], mode='valid')

def exponential_moving_average(data, alpha=0.1):
    """Compute exponential moving average"""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    
    return ema

# Test data
np.random.seed(42)
noisy_data = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.3 * np.random.randn(100)

# Simple moving average
sma_5 = moving_average(noisy_data, 5, mode='valid')
sma_10 = moving_average(noisy_data, 10, mode='valid')

print(f"Original data length: {len(noisy_data)}")
print(f"5-point moving average length: {len(sma_5)}")
print(f"10-point moving average length: {len(sma_10)}")

# Weighted moving average (more weight to recent values)
weights = [0.1, 0.2, 0.3, 0.4]  # Recent values get higher weight
wma = weighted_moving_average(noisy_data, weights)
print(f"Weighted moving average length: {len(wma)}")

# Exponential moving average
ema = exponential_moving_average(noisy_data, alpha=0.3)
print(f"Exponential moving average length: {len(ema)}")

# Centered moving average (for better alignment)
def centered_moving_average(data, window_size):
    """Centered moving average with padding"""
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    weights = np.ones(window_size) / window_size
    
    result = np.convolve(padded_data, weights, mode='valid')
    return result

centered_ma = centered_moving_average(noisy_data, 5)
print(f"Centered moving average length: {len(centered_ma)}")

# 2D moving average for matrices
def moving_average_2d(matrix, window_shape):
    """2D moving average using sliding window"""
    from scipy.ndimage import uniform_filter
    return uniform_filter(matrix, size=window_shape, mode='constant')

# Performance comparison
def compare_ma_methods(data, window_size=5, iterations=1000):
    """Compare performance of different moving average methods"""
    import time
    
    # NumPy convolve method
    start = time.time()
    weights = np.ones(window_size) / window_size
    for _ in range(iterations):
        _ = np.convolve(data, weights, mode='valid')
    convolve_time = time.time() - start
    
    # Manual loop method
    start = time.time()
    for _ in range(iterations):
        result = np.zeros(len(data) - window_size + 1)
        for i in range(len(result)):
            result[i] = np.mean(data[i:i+window_size])
    manual_time = time.time() - start
    
    print(f"Convolve method: {convolve_time:.6f}s")
    print(f"Manual method: {manual_time:.6f}s")
    print(f"Speedup: {manual_time/convolve_time:.2f}x")

compare_ma_methods(noisy_data)
```

### Use Cases
- Financial time series analysis (stock prices, trading indicators)
- Signal processing and noise reduction
- Weather data smoothing
- Quality control in manufacturing

---

## Question 20

**Generate a 2D NumPy array of random integers and normalize it between 0 and 1.**

### Code Example

```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 2D array of random integers
rows, cols = 5, 4
min_val, max_val = 10, 50

random_integers = np.random.randint(min_val, max_val, size=(rows, cols))
print("Original random integer array:")
print(random_integers)
print(f"Original range: [{np.min(random_integers)}, {np.max(random_integers)}]")

# Method 1: Min-Max normalization
def normalize_min_max(arr):
    """Normalize array to [0, 1] using min-max scaling"""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=float)
    
    return (arr - arr_min) / (arr_max - arr_min)

normalized = normalize_min_max(random_integers)
print("\nNormalized array (min-max):")
print(normalized)
print(f"Normalized range: [{np.min(normalized):.6f}, {np.max(normalized):.6f}]")

# Method 2: Using sklearn-style normalization
def normalize_feature_scale(arr):
    """Feature scaling normalization"""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# Method 3: Axis-wise normalization
def normalize_by_axis(arr, axis=None):
    """Normalize along specified axis"""
    if axis is None:
        return normalize_min_max(arr)
    
    min_vals = np.min(arr, axis=axis, keepdims=True)
    max_vals = np.max(arr, axis=axis, keepdims=True)
    
    return (arr - min_vals) / (max_vals - min_vals)

# Normalize each row independently
row_normalized = normalize_by_axis(random_integers, axis=1)
print("\nRow-wise normalized array:")
print(row_normalized)

# Normalize each column independently
col_normalized = normalize_by_axis(random_integers, axis=0)
print("\nColumn-wise normalized array:")
print(col_normalized)

# Alternative using different ranges
def normalize_to_range(arr, new_min=0, new_max=1):
    """Normalize to custom range"""
    normalized = normalize_min_max(arr)
    return normalized * (new_max - new_min) + new_min

custom_range = normalize_to_range(random_integers, -1, 1)
print(f"\nNormalized to [-1, 1]:")
print(custom_range)
print(f"Range: [{np.min(custom_range):.6f}, {np.max(custom_range):.6f}]")
```

---

## Question 21

**Create a NumPy code snippet to extract all odd numbers from an array.**

### Code Example

```python
import numpy as np

# Sample array with mixed numbers
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print(f"Original array: {data}")

# Method 1: Boolean indexing (most efficient)
odd_mask = data % 2 == 1
odd_numbers = data[odd_mask]
print(f"Odd numbers (boolean indexing): {odd_numbers}")

# Method 2: Using np.where
odd_indices = np.where(data % 2 == 1)[0]
odd_with_where = data[odd_indices]
print(f"Odd numbers (np.where): {odd_with_where}")

# Method 3: List comprehension (less efficient for large arrays)
odd_list_comp = np.array([x for x in data if x % 2 == 1])
print(f"Odd numbers (list comprehension): {odd_list_comp}")

# Method 4: Using np.extract
odd_extracted = np.extract(data % 2 == 1, data)
print(f"Odd numbers (np.extract): {odd_extracted}")

# For 2D arrays
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"\n2D matrix:\n{matrix}")

# Extract odd numbers from 2D array
odd_2d = matrix[matrix % 2 == 1]
print(f"Odd numbers from 2D array: {odd_2d}")

# Get positions of odd numbers
odd_positions = np.where(matrix % 2 == 1)
print(f"Positions of odd numbers: {list(zip(odd_positions[0], odd_positions[1]))}")

# Advanced filtering with multiple conditions
def extract_custom_odds(arr, min_val=None, max_val=None):
    """Extract odd numbers with optional range filtering"""
    mask = arr % 2 == 1
    
    if min_val is not None:
        mask &= arr >= min_val
    if max_val is not None:
        mask &= arr <= max_val
    
    return arr[mask]

# Extract odd numbers between 5 and 12
filtered_odds = extract_custom_odds(data, min_val=5, max_val=12)
print(f"Odd numbers between 5 and 12: {filtered_odds}")

# Performance comparison
def compare_odd_extraction_methods(arr, iterations=10000):
    """Compare performance of different odd extraction methods"""
    import time
    
    # Boolean indexing
    start = time.time()
    for _ in range(iterations):
        _ = arr[arr % 2 == 1]
    bool_time = time.time() - start
    
    # np.where
    start = time.time()
    for _ in range(iterations):
        indices = np.where(arr % 2 == 1)[0]
        _ = arr[indices]
    where_time = time.time() - start
    
    # np.extract
    start = time.time()
    for _ in range(iterations):
        _ = np.extract(arr % 2 == 1, arr)
    extract_time = time.time() - start
    
    print(f"Boolean indexing: {bool_time:.6f}s")
    print(f"np.where: {where_time:.6f}s")
    print(f"np.extract: {extract_time:.6f}s")

large_array = np.random.randint(1, 1000, 10000)
compare_odd_extraction_methods(large_array)
```

---

## Question 22

**Implement a routine to calculate the outer product of two vectors in NumPy.**

### Code Example

```python
import numpy as np

def outer_product(a, b):
    """
    Calculate outer product of two vectors
    For vectors a and b, outer product C[i,j] = a[i] * b[j]
    """
    return np.outer(a, b)

def outer_product_manual(a, b):
    """Manual implementation using broadcasting"""
    a = np.array(a).reshape(-1, 1)  # Column vector
    b = np.array(b).reshape(1, -1)  # Row vector
    return a * b

def outer_product_einsum(a, b):
    """Using Einstein summation notation"""
    return np.einsum('i,j->ij', a, b)

# Test vectors
vector_a = np.array([1, 2, 3, 4])
vector_b = np.array([5, 6, 7])

print(f"Vector a: {vector_a}")
print(f"Vector b: {vector_b}")

# Method 1: Using np.outer
outer_np = outer_product(vector_a, vector_b)
print(f"\nOuter product (np.outer):\n{outer_np}")

# Method 2: Manual broadcasting
outer_manual = outer_product_manual(vector_a, vector_b)
print(f"\nOuter product (manual):\n{outer_manual}")

# Method 3: Einstein summation
outer_einsum = outer_product_einsum(vector_a, vector_b)
print(f"\nOuter product (einsum):\n{outer_einsum}")

# Verify all methods give same result
print(f"All methods equal: {np.allclose(outer_np, outer_manual) and np.allclose(outer_np, outer_einsum)}")

# Applications
print("\n=== Applications ===")

# 1. Correlation matrix from standardized vectors
def correlation_from_outer(x, y):
    """Calculate correlation using outer product"""
    x_std = (x - np.mean(x)) / np.std(x)
    y_std = (y - np.mean(y)) / np.std(y)
    return np.sum(np.outer(x_std, y_std)) / len(x)

# 2. Rank-1 matrix approximation
def rank_1_approximation(matrix):
    """Find best rank-1 approximation using SVD and outer product"""
    U, s, Vt = np.linalg.svd(matrix)
    return s[0] * np.outer(U[:, 0], Vt[0, :])

# 3. Tensor product
def tensor_product_2d(A, B):
    """2D tensor product using outer products"""
    m, n = A.shape
    p, q = B.shape
    result = np.zeros((m*p, n*q))
    
    for i in range(m):
        for j in range(n):
            result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B
    
    return result

# Example matrix for rank-1 approximation
test_matrix = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
rank1_approx = rank_1_approximation(test_matrix)
print(f"Original matrix:\n{test_matrix}")
print(f"Rank-1 approximation:\n{rank1_approx}")

# Performance comparison
def compare_outer_methods(a, b, iterations=10000):
    """Compare performance of outer product methods"""
    import time
    
    # np.outer
    start = time.time()
    for _ in range(iterations):
        _ = np.outer(a, b)
    outer_time = time.time() - start
    
    # Manual broadcasting
    start = time.time()
    for _ in range(iterations):
        a_col = a.reshape(-1, 1)
        b_row = b.reshape(1, -1)
        _ = a_col * b_row
    manual_time = time.time() - start
    
    # Einstein summation
    start = time.time()
    for _ in range(iterations):
        _ = np.einsum('i,j->ij', a, b)
    einsum_time = time.time() - start
    
    print(f"np.outer: {outer_time:.6f}s")
    print(f"Manual: {manual_time:.6f}s")
    print(f"einsum: {einsum_time:.6f}s")

compare_outer_methods(vector_a, vector_b)
```

### Use Cases
- Machine learning (feature interactions, attention mechanisms)
- Signal processing (correlation analysis)
- Linear algebra (matrix decompositions)
- Physics simulations (tensor operations)

---

## Question 23

**Write a NumPy program to create a checkerboard 8x8 matrix using the tile function.**

### Code Example

```python
import numpy as np

# Method 1: Using np.tile with 2x2 base pattern
def checkerboard_tile(size=8):
    """Create checkerboard using np.tile"""
    # Create 2x2 base pattern
    base_pattern = np.array([[0, 1], [1, 0]])
    
    # Tile the pattern to create full checkerboard
    repetitions = size // 2
    return np.tile(base_pattern, (repetitions, repetitions))

checkerboard = checkerboard_tile(8)
print("8x8 Checkerboard using np.tile:")
print(checkerboard)

# Method 2: Alternative implementation with different values
def checkerboard_custom(size=8, val1=0, val2=1):
    """Create checkerboard with custom values"""
    base = np.array([[val1, val2], [val2, val1]])
    return np.tile(base, (size//2, size//2))

# Chess-style checkerboard (0 and 1)
chess_board = checkerboard_custom(8, 0, 1)
print(f"\nChess-style checkerboard:\n{chess_board}")

# Colored checkerboard (using different values)
colored_board = checkerboard_custom(8, 255, 0)  # White and black pixels
print(f"\nColored checkerboard (255, 0):\n{colored_board}")

# Method 3: Mathematical approach without tile
def checkerboard_math(size=8):
    """Create checkerboard using mathematical operations"""
    indices = np.arange(size)
    x, y = np.meshgrid(indices, indices)
    return (x + y) % 2

math_board = checkerboard_math(8)
print(f"\nMathematical checkerboard:\n{math_board}")

# Method 4: Broadcasting approach
def checkerboard_broadcast(size=8):
    """Create checkerboard using broadcasting"""
    row = np.arange(size) % 2
    col = np.arange(size) % 2
    return (row[:, None] + col) % 2

broadcast_board = checkerboard_broadcast(8)
print(f"\nBroadcast checkerboard:\n{broadcast_board}")

# Verify all methods produce same result
print(f"All methods equal: {np.array_equal(checkerboard, math_board) and np.array_equal(checkerboard, broadcast_board)}")

# Performance comparison
def compare_checkerboard_methods(size=64):
    """Compare performance of different checkerboard creation methods"""
    import time
    iterations = 1000
    
    # np.tile method
    start = time.time()
    for _ in range(iterations):
        base = np.array([[0, 1], [1, 0]])
        _ = np.tile(base, (size//2, size//2))
    tile_time = time.time() - start
    
    # Mathematical method
    start = time.time()
    for _ in range(iterations):
        indices = np.arange(size)
        x, y = np.meshgrid(indices, indices)
        _ = (x + y) % 2
    math_time = time.time() - start
    
    print(f"Tile method: {tile_time:.6f}s")
    print(f"Math method: {math_time:.6f}s")

compare_checkerboard_methods()
```

---

## Question 24

**Code a NumPy snippet to create a border around an existing array.**

### Code Example

```python
import numpy as np

def add_border(arr, border_width=1, border_value=0):
    """Add border around existing array"""
    return np.pad(arr, border_width, mode='constant', constant_values=border_value)

def add_custom_border(arr, top=1, bottom=1, left=1, right=1, value=0):
    """Add custom border with different widths on each side"""
    return np.pad(arr, ((top, bottom), (left, right)), mode='constant', constant_values=value)

# Original array
original = np.ones((3, 3)) * 5
print("Original array:")
print(original)

# Method 1: Simple border
bordered = add_border(original, border_width=1, border_value=0)
print("\nArray with border (width=1, value=0):")
print(bordered)

# Method 2: Thick border
thick_border = add_border(original, border_width=2, border_value=-1)
print("\nArray with thick border (width=2, value=-1):")
print(thick_border)

# Method 3: Custom border widths
custom_bordered = add_custom_border(original, top=2, bottom=1, left=1, right=3, value=9)
print("\nArray with custom border widths:")
print(custom_bordered)

# Method 4: Multiple border layers
def multi_layer_border(arr, layers):
    """Create multiple border layers with different values"""
    result = arr.copy()
    for width, value in layers:
        result = add_border(result, width, value)
    return result

multi_border = multi_layer_border(original, [(1, 0), (1, 8), (1, -2)])
print("\nArray with multi-layer border:")
print(multi_border)

# Method 5: Gradient border
def gradient_border(arr, border_width=2):
    """Create gradient border"""
    result = np.pad(arr, border_width, mode='constant', constant_values=0)
    
    # Create gradient effect
    for i in range(border_width):
        intensity = (i + 1) / border_width
        # Top and bottom borders
        result[i, :] = intensity
        result[-(i+1), :] = intensity
        # Left and right borders
        result[:, i] = intensity
        result[:, -(i+1)] = intensity
    
    return result

gradient_bordered = gradient_border(original, 3)
print("\nArray with gradient border:")
print(gradient_bordered)

# Method 6: Different padding modes
print("\nDifferent padding modes:")

# Edge padding (repeat edge values)
edge_padded = np.pad(original, 1, mode='edge')
print("Edge padding:")
print(edge_padded)

# Reflection padding
reflect_padded = np.pad(original, 1, mode='reflect')
print("\nReflection padding:")
print(reflect_padded)

# Wrap padding (circular)
wrap_padded = np.pad(original, 1, mode='wrap')
print("\nWrap padding:")
print(wrap_padded)

# For 3D arrays
array_3d = np.random.randint(1, 10, (2, 3, 3))
print(f"\nOriginal 3D array shape: {array_3d.shape}")

bordered_3d = np.pad(array_3d, ((1, 1), (2, 2), (1, 1)), mode='constant', constant_values=0)
print(f"3D array with border shape: {bordered_3d.shape}")
```

---

## Question 25

**Write a function to compute the convolution of two matrices in NumPy.**

### Code Example

```python
import numpy as np
from scipy import signal

def convolution_2d(matrix, kernel, mode='valid'):
    """
    Compute 2D convolution of matrix with kernel
    
    Parameters:
    - matrix: Input 2D array
    - kernel: Convolution kernel/filter
    - mode: 'valid', 'same', or 'full'
    """
    return signal.convolve2d(matrix, kernel, mode=mode, boundary='fill')

def manual_convolution_2d(matrix, kernel):
    """Manual implementation of 2D convolution"""
    m_rows, m_cols = matrix.shape
    k_rows, k_cols = kernel.shape
    
    # Output dimensions for 'valid' convolution
    out_rows = m_rows - k_rows + 1
    out_cols = m_cols - k_cols + 1
    
    result = np.zeros((out_rows, out_cols))
    
    for i in range(out_rows):
        for j in range(out_cols):
            result[i, j] = np.sum(matrix[i:i+k_rows, j:j+k_cols] * kernel)
    
    return result

# Sample matrix and kernels
matrix = np.array([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]])

# Edge detection kernel (Sobel)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Gaussian blur kernel
gaussian = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]]) / 16

print("Original matrix:")
print(matrix)

# Convolution with different kernels
sobel_x_result = convolution_2d(matrix, sobel_x, mode='valid')
sobel_y_result = convolution_2d(matrix, sobel_y, mode='valid')
gaussian_result = convolution_2d(matrix, gaussian, mode='valid')

print(f"\nSobel X convolution:\n{sobel_x_result}")
print(f"\nSobel Y convolution:\n{sobel_y_result}")
print(f"\nGaussian blur convolution:\n{gaussian_result}")

# Different modes
print(f"\nConvolution modes comparison:")
valid_conv = convolution_2d(matrix, sobel_x, mode='valid')
same_conv = convolution_2d(matrix, sobel_x, mode='same')
full_conv = convolution_2d(matrix, sobel_x, mode='full')

print(f"Original shape: {matrix.shape}")
print(f"Valid mode shape: {valid_conv.shape}")
print(f"Same mode shape: {same_conv.shape}")
print(f"Full mode shape: {full_conv.shape}")

# Verify manual implementation
manual_result = manual_convolution_2d(matrix, sobel_x)
scipy_result = convolution_2d(matrix, sobel_x, mode='valid')

print(f"\nManual vs SciPy convolution equal: {np.allclose(manual_result, scipy_result)}")

# Image processing example
def apply_image_filters(image, filters):
    """Apply multiple filters to an image"""
    results = {}
    for name, kernel in filters.items():
        results[name] = convolution_2d(image, kernel, mode='same')
    return results

# Common image processing kernels
filters = {
    'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'edge_detect': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
}

# Apply filters to sample image
sample_image = np.random.rand(8, 8) * 255
filtered_results = apply_image_filters(sample_image, filters)

print(f"\nFilter results shapes:")
for name, result in filtered_results.items():
    print(f"{name}: {result.shape}")
```

---

## Question 26

**Implement a script that computes the Fibonacci sequence using a NumPy matrix.**

### Code Example

```python
import numpy as np

def fibonacci_matrix_method(n):
    """
    Compute nth Fibonacci number using matrix exponentiation
    Based on: [[1,1],[1,0]]^n = [[F(n+1),F(n)],[F(n),F(n-1)]]
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Fibonacci matrix
    fib_matrix = np.array([[1, 1], [1, 0]], dtype=object)
    
    # Compute matrix power
    result_matrix = matrix_power(fib_matrix, n)
    
    return int(result_matrix[0, 1])

def matrix_power(matrix, power):
    """Fast matrix exponentiation using binary exponentiation"""
    if power == 0:
        return np.eye(matrix.shape[0], dtype=object)
    if power == 1:
        return matrix.copy()
    
    result = np.eye(matrix.shape[0], dtype=object)
    base = matrix.copy()
    
    while power > 0:
        if power % 2 == 1:
            result = np.dot(result, base)
        base = np.dot(base, base)
        power //= 2
    
    return result

def fibonacci_sequence_matrix(count):
    """Generate Fibonacci sequence using matrix method"""
    sequence = []
    for i in range(count):
        sequence.append(fibonacci_matrix_method(i))
    return np.array(sequence)

def fibonacci_vectorized(n):
    """Vectorized Fibonacci computation for multiple values"""
    if isinstance(n, int):
        n = np.array([n])
    elif not isinstance(n, np.ndarray):
        n = np.array(n)
    
    result = np.zeros_like(n, dtype=object)
    
    for i, val in enumerate(n):
        result[i] = fibonacci_matrix_method(val)
    
    return result

# Generate Fibonacci sequence
print("Fibonacci sequence using matrix method:")
fib_sequence = fibonacci_sequence_matrix(15)
print(fib_sequence)

# Compare with traditional method
def fibonacci_traditional(n):
    """Traditional iterative Fibonacci"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Verification
print("\nVerification against traditional method:")
for i in range(10):
    matrix_result = fibonacci_matrix_method(i)
    traditional_result = fibonacci_traditional(i)
    print(f"F({i}): Matrix={matrix_result}, Traditional={traditional_result}, Match={matrix_result == traditional_result}")

# Performance comparison for large numbers
def compare_fibonacci_methods(n):
    """Compare performance of different Fibonacci methods"""
    import time
    
    # Matrix method
    start = time.time()
    matrix_result = fibonacci_matrix_method(n)
    matrix_time = time.time() - start
    
    # Traditional method (for comparison, if n is not too large)
    if n < 1000:  # Avoid long computation for traditional method
        start = time.time()
        traditional_result = fibonacci_traditional(n)
        traditional_time = time.time() - start
        
        print(f"F({n}):")
        print(f"Matrix method: {matrix_result} (Time: {matrix_time:.6f}s)")
        print(f"Traditional method: {traditional_result} (Time: {traditional_time:.6f}s)")
        print(f"Results match: {matrix_result == traditional_result}")
    else:
        print(f"F({n}) using matrix method: {matrix_result} (Time: {matrix_time:.6f}s)")

# Test with large numbers
print(f"\nLarge Fibonacci numbers:")
compare_fibonacci_methods(50)
compare_fibonacci_methods(100)

# Golden ratio verification
def golden_ratio_fibonacci(n_terms=20):
    """Verify that F(n)/F(n-1) approaches golden ratio"""
    fib_seq = fibonacci_sequence_matrix(n_terms)
    ratios = []
    
    for i in range(2, len(fib_seq)):
        if fib_seq[i-1] != 0:
            ratio = float(fib_seq[i]) / float(fib_seq[i-1])
            ratios.append(ratio)
    
    golden_ratio = (1 + np.sqrt(5)) / 2
    print(f"\nGolden ratio convergence:")
    print(f"Theoretical golden ratio: {golden_ratio:.10f}")
    
    for i, ratio in enumerate(ratios[-5:], len(ratios)-4):
        print(f"F({i+2})/F({i+1}) = {ratio:.10f}")
    
    return ratios

ratios = golden_ratio_fibonacci(25)

# Matrix properties analysis
def analyze_fibonacci_matrix():
    """Analyze properties of the Fibonacci matrix"""
    F = np.array([[1, 1], [1, 0]], dtype=float)
    
    print(f"\nFibonacci matrix F:")
    print(F)
    
    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(F)
    print(f"\nEigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    # Verify Binet's formula connection
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    psi = (1 - np.sqrt(5)) / 2  # Conjugate
    
    print(f"Golden ratio φ: {phi:.10f}")
    print(f"Conjugate ψ: {psi:.10f}")
    print(f"Eigenvalue 1: {eigenvals[0]:.10f}")
    print(f"Eigenvalue 2: {eigenvals[1]:.10f}")

analyze_fibonacci_matrix()

# Optimized version for sequences
def fibonacci_sequence_optimized(n):
    """Optimized Fibonacci sequence generation"""
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])
    
    # Use recurrence relation for efficiency
    fib = np.zeros(n, dtype=object)
    fib[0], fib[1] = 0, 1
    
    for i in range(2, n):
        fib[i] = fib[i-1] + fib[i-2]
    
    return fib

# Generate large sequence efficiently
large_sequence = fibonacci_sequence_optimized(30)
print(f"\nLarge Fibonacci sequence (first 30 terms):")
print(large_sequence)
```

---

## Question 27

**Write a code to replace all elements greater than a certain threshold in a NumPy array with a specific value.**

### Code Example

```python
import numpy as np

def replace_threshold(arr, threshold, replacement_value, condition='greater'):
    """
    Replace elements based on threshold condition
    
    Parameters:
    - arr: Input array
    - threshold: Threshold value
    - replacement_value: Value to replace with
    - condition: 'greater', 'less', 'greater_equal', 'less_equal', 'equal', 'not_equal'
    """
    arr_copy = arr.copy()
    
    if condition == 'greater':
        mask = arr_copy > threshold
    elif condition == 'less':
        mask = arr_copy < threshold
    elif condition == 'greater_equal':
        mask = arr_copy >= threshold
    elif condition == 'less_equal':
        mask = arr_copy <= threshold
    elif condition == 'equal':
        mask = arr_copy == threshold
    elif condition == 'not_equal':
        mask = arr_copy != threshold
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    arr_copy[mask] = replacement_value
    return arr_copy

# Sample data
np.random.seed(42)
data = np.random.randint(1, 20, (4, 5))
print("Original array:")
print(data)

# Basic threshold replacement
threshold_val = 10
replacement_val = 999

# Replace elements > 10 with 999
result_greater = replace_threshold(data, threshold_val, replacement_val, 'greater')
print(f"\nReplace elements > {threshold_val} with {replacement_val}:")
print(result_greater)

# Replace elements <= 5 with 0
result_less_equal = replace_threshold(data, 5, 0, 'less_equal')
print(f"\nReplace elements <= 5 with 0:")
print(result_less_equal)

# Multiple threshold replacement
def multi_threshold_replace(arr, thresholds_replacements):
    """
    Replace elements based on multiple threshold conditions
    
    Parameters:
    - arr: Input array
    - thresholds_replacements: List of tuples (threshold, replacement, condition)
    """
    result = arr.copy()
    
    for threshold, replacement, condition in thresholds_replacements:
        result = replace_threshold(result, threshold, replacement, condition)
    
    return result

# Apply multiple replacements
multi_rules = [
    (15, -1, 'greater'),      # Replace > 15 with -1
    (5, 0, 'less'),           # Replace < 5 with 0
    (10, 100, 'equal')        # Replace == 10 with 100
]

multi_result = multi_threshold_replace(data, multi_rules)
print(f"\nMultiple threshold replacements:")
print(multi_result)

# Conditional replacement with np.where
def conditional_replace_where(arr, condition_func, replacement_func):
    """
    Replace elements using np.where with custom functions
    
    Parameters:
    - arr: Input array
    - condition_func: Function that returns boolean mask
    - replacement_func: Function that returns replacement values
    """
    condition = condition_func(arr)
    replacement = replacement_func(arr)
    return np.where(condition, replacement, arr)

# Example: Replace outliers with median
def replace_outliers_median(arr, std_threshold=2):
    """Replace outliers (> mean + std_threshold*std) with median"""
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    median_val = np.median(arr)
    
    outlier_mask = np.abs(arr - mean_val) > std_threshold * std_val
    result = arr.copy()
    result[outlier_mask] = median_val
    
    return result, outlier_mask

# Apply outlier replacement
outlier_data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 200])
cleaned_data, outlier_mask = replace_outliers_median(outlier_data, std_threshold=2)

print(f"\nOutlier replacement:")
print(f"Original: {outlier_data}")
print(f"Cleaned: {cleaned_data}")
print(f"Outlier positions: {np.where(outlier_mask)[0]}")

# Range-based replacement
def replace_range(arr, min_val, max_val, replacement):
    """Replace values in range [min_val, max_val] with replacement"""
    mask = (arr >= min_val) & (arr <= max_val)
    result = arr.copy()
    result[mask] = replacement
    return result

range_result = replace_range(data, 8, 12, 888)
print(f"\nReplace values in range [8, 12] with 888:")
print(range_result)

# Percentile-based replacement
def replace_percentile_based(arr, lower_percentile=5, upper_percentile=95, 
                           lower_replacement=None, upper_replacement=None):
    """Replace values based on percentiles"""
    lower_bound = np.percentile(arr, lower_percentile)
    upper_bound = np.percentile(arr, upper_percentile)
    
    result = arr.copy()
    
    if lower_replacement is not None:
        result[arr < lower_bound] = lower_replacement
    
    if upper_replacement is not None:
        result[arr > upper_bound] = upper_replacement
    
    return result

percentile_result = replace_percentile_based(data, 10, 90, -99, 99)
print(f"\nReplace values below 10th percentile with -99 and above 90th with 99:")
print(percentile_result)

# In-place replacement for memory efficiency
def replace_threshold_inplace(arr, threshold, replacement_value, condition='greater'):
    """In-place threshold replacement"""
    if condition == 'greater':
        arr[arr > threshold] = replacement_value
    elif condition == 'less':
        arr[arr < threshold] = replacement_value
    elif condition == 'greater_equal':
        arr[arr >= threshold] = replacement_value
    elif condition == 'less_equal':
        arr[arr <= threshold] = replacement_value
    
    return arr

# Test in-place replacement
test_array = data.copy()
print(f"\nBefore in-place replacement:")
print(test_array)

replace_threshold_inplace(test_array, 10, 777, 'greater')
print(f"After in-place replacement (> 10 with 777):")
print(test_array)

# Conditional replacement with mathematical operations
def replace_with_operation(arr, condition_func, operation_func):
    """Replace elements with result of operation"""
    mask = condition_func(arr)
    result = arr.copy()
    result[mask] = operation_func(arr[mask])
    return result

# Example: Square all values greater than 10
squared_result = replace_with_operation(
    data, 
    lambda x: x > 10, 
    lambda x: x ** 2
)
print(f"\nSquare all values > 10:")
print(squared_result)

# Performance comparison
def compare_replacement_methods(arr, threshold, replacement):
    """Compare performance of different replacement methods"""
    import time
    iterations = 10000
    
    # Boolean indexing
    start = time.time()
    for _ in range(iterations):
        result = arr.copy()
        result[arr > threshold] = replacement
    bool_time = time.time() - start
    
    # np.where
    start = time.time()
    for _ in range(iterations):
        result = np.where(arr > threshold, replacement, arr)
    where_time = time.time() - start
    
    print(f"Boolean indexing: {bool_time:.6f}s")
    print(f"np.where: {where_time:.6f}s")
    print(f"Boolean indexing is {where_time/bool_time:.2f}x faster")

large_array = np.random.randint(1, 100, 10000)
compare_replacement_methods(large_array, 50, 999)
```

---

## Question 28

**Implement an efficient rolling window calculation for a 1D array using NumPy.**

### Code Example

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def rolling_window_stride(arr, window_size):
    """
    Efficient rolling window using stride tricks
    Returns a 2D array where each row is a window
    """
    return sliding_window_view(arr, window_size)

def rolling_mean(arr, window_size):
    """Calculate rolling mean using stride tricks"""
    windows = rolling_window_stride(arr, window_size)
    return np.mean(windows, axis=1)

def rolling_statistics(arr, window_size, stats=['mean', 'std', 'min', 'max']):
    """Calculate multiple rolling statistics efficiently"""
    windows = rolling_window_stride(arr, window_size)
    results = {}
    
    if 'mean' in stats:
        results['mean'] = np.mean(windows, axis=1)
    if 'std' in stats:
        results['std'] = np.std(windows, axis=1)
    if 'min' in stats:
        results['min'] = np.min(windows, axis=1)
    if 'max' in stats:
        results['max'] = np.max(windows, axis=1)
    if 'median' in stats:
        results['median'] = np.median(windows, axis=1)
    if 'sum' in stats:
        results['sum'] = np.sum(windows, axis=1)
    
    return results

# Sample data
np.random.seed(42)
data = np.random.randn(100) * 10 + 50  # Random walk around 50
print(f"Data shape: {data.shape}")
print(f"First 10 values: {data[:10]}")

# Basic rolling window
window_size = 5
windows = rolling_window_stride(data, window_size)
print(f"\nRolling windows shape: {windows.shape}")
print(f"First window: {windows[0]}")
print(f"Second window: {windows[1]}")

# Rolling mean
roll_mean = rolling_mean(data, window_size)
print(f"\nRolling mean shape: {roll_mean.shape}")
print(f"First 5 rolling means: {roll_mean[:5]}")

# Multiple rolling statistics
roll_stats = rolling_statistics(data, window_size, ['mean', 'std', 'min', 'max'])
print(f"\nRolling statistics:")
for stat_name, values in roll_stats.items():
    print(f"{stat_name}: {values[:5]} (first 5 values)")

# Custom rolling functions
def rolling_custom(arr, window_size, func):
    """Apply custom function to rolling windows"""
    windows = rolling_window_stride(arr, window_size)
    return np.apply_along_axis(func, axis=1, arr=windows)

# Example custom functions
def range_func(window):
    """Calculate range (max - min) of window"""
    return np.max(window) - np.min(window)

def skewness_func(window):
    """Calculate skewness of window"""
    mean = np.mean(window)
    std = np.std(window)
    if std == 0:
        return 0
    return np.mean(((window - mean) / std) ** 3)

# Apply custom functions
roll_range = rolling_custom(data, window_size, range_func)
roll_skew = rolling_custom(data, window_size, skewness_func)

print(f"\nCustom rolling functions:")
print(f"Rolling range: {roll_range[:5]}")
print(f"Rolling skewness: {roll_skew[:5]}")

# Memory-efficient rolling for large datasets
def rolling_large_dataset(arr, window_size, func, chunk_size=10000):
    """
    Memory-efficient rolling calculation for large datasets
    Processes data in chunks to manage memory usage
    """
    if len(arr) <= chunk_size:
        return rolling_custom(arr, window_size, func)
    
    results = []
    overlap = window_size - 1
    
    for start in range(0, len(arr), chunk_size - overlap):
        end = min(start + chunk_size, len(arr))
        chunk = arr[start:end]
        
        if len(chunk) >= window_size:
            chunk_result = rolling_custom(chunk, window_size, func)
            
            # Remove overlap from all chunks except the first
            if start > 0:
                chunk_result = chunk_result[overlap:]
            
            results.append(chunk_result)
    
    return np.concatenate(results)

# Test with large dataset simulation
large_data = np.random.randn(50000)
large_roll_mean = rolling_large_dataset(large_data, 10, np.mean, chunk_size=5000)
print(f"\nLarge dataset rolling mean shape: {large_roll_mean.shape}")

# Centered rolling window
def centered_rolling(arr, window_size, func=np.mean, fill_value=np.nan):
    """
    Centered rolling window calculation
    Places result at center of window for better alignment
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size for centering
    
    pad_size = window_size // 2
    padded_arr = np.pad(arr, pad_size, mode='constant', constant_values=fill_value)
    
    windows = rolling_window_stride(padded_arr, window_size)
    
    # Filter out windows containing fill_value
    valid_windows = []
    for window in windows:
        if not np.isnan(window).any():
            valid_windows.append(func(window))
        else:
            valid_windows.append(fill_value)
    
    return np.array(valid_windows)

# Centered rolling mean
centered_mean = centered_rolling(data[:20], 5, np.mean)
print(f"\nCentered rolling mean (first 20 points): {centered_mean}")

# Rolling correlation between two series
def rolling_correlation(x, y, window_size):
    """Calculate rolling correlation between two arrays"""
    if len(x) != len(y):
        raise ValueError("Arrays must have same length")
    
    x_windows = rolling_window_stride(x, window_size)
    y_windows = rolling_window_stride(y, window_size)
    
    correlations = []
    for x_win, y_win in zip(x_windows, y_windows):
        corr = np.corrcoef(x_win, y_win)[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0)
    
    return np.array(correlations)

# Test rolling correlation
y_data = data + np.random.randn(len(data)) * 5  # Correlated with noise
roll_corr = rolling_correlation(data, y_data, 10)
print(f"\nRolling correlation (window=10): {roll_corr[:5]}")

# Performance comparison
def compare_rolling_methods(arr, window_size):
    """Compare performance of different rolling window methods"""
    import time
    iterations = 100
    
    # Stride tricks method
    start = time.time()
    for _ in range(iterations):
        windows = sliding_window_view(arr, window_size)
        _ = np.mean(windows, axis=1)
    stride_time = time.time() - start
    
    # Manual convolution method
    start = time.time()
    for _ in range(iterations):
        _ = np.convolve(arr, np.ones(window_size)/window_size, mode='valid')
    convolve_time = time.time() - start
    
    # Loop-based method
    start = time.time()
    for _ in range(iterations):
        result = []
        for i in range(len(arr) - window_size + 1):
            result.append(np.mean(arr[i:i+window_size]))
    loop_time = time.time() - start
    
    print(f"Stride tricks method: {stride_time:.6f}s")
    print(f"Convolution method: {convolve_time:.6f}s")
    print(f"Loop method: {loop_time:.6f}s")
    print(f"Stride tricks is {loop_time/stride_time:.1f}x faster than loops")

test_data = np.random.randn(1000)
compare_rolling_methods(test_data, 10)

# Advanced rolling window applications
def bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands using rolling statistics"""
    rolling_stats = rolling_statistics(prices, window, ['mean', 'std'])
    
    middle_band = rolling_stats['mean']
    std_dev = rolling_stats['std']
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }

# Simulate stock prices
stock_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
bands = bollinger_bands(stock_prices, window=10)

print(f"\nBollinger Bands example (last 5 values):")
print(f"Upper: {bands['upper'][-5:]}")
print(f"Middle: {bands['middle'][-5:]}")
print(f"Lower: {bands['lower'][-5:]}")
```

---

## Question 29

**Explain how you would implement gradient descent optimization with NumPy.**

### Code Example

```python
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    """
    Gradient Descent optimizer implementation
    Supports various optimization algorithms and features
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cost_history = []
        self.parameter_history = []
    
    def optimize(self, cost_function, gradient_function, initial_params, 
                 method='standard', **kwargs):
        """
        Main optimization method supporting different algorithms
        
        Parameters:
        - cost_function: Function to minimize
        - gradient_function: Function that computes gradients
        - initial_params: Starting parameters
        - method: 'standard', 'momentum', 'adam', 'adagrad', 'rmsprop'
        """
        
        if method == 'standard':
            return self._standard_gd(cost_function, gradient_function, initial_params)
        elif method == 'momentum':
            return self._momentum_gd(cost_function, gradient_function, initial_params, 
                                   kwargs.get('momentum', 0.9))
        elif method == 'adam':
            return self._adam_gd(cost_function, gradient_function, initial_params,
                               kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999))
        elif method == 'adagrad':
            return self._adagrad_gd(cost_function, gradient_function, initial_params)
        elif method == 'rmsprop':
            return self._rmsprop_gd(cost_function, gradient_function, initial_params,
                                  kwargs.get('decay_rate', 0.9))
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _standard_gd(self, cost_function, gradient_function, params):
        """Standard gradient descent"""
        self.cost_history = []
        self.parameter_history = []
        
        for i in range(self.max_iterations):
            cost = cost_function(params)
            gradient = gradient_function(params)
            
            self.cost_history.append(cost)
            self.parameter_history.append(params.copy())
            
            # Update parameters
            params = params - self.learning_rate * gradient
            
            # Check convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
        
        return params
    
    def _momentum_gd(self, cost_function, gradient_function, params, momentum=0.9):
        """Gradient descent with momentum"""
        self.cost_history = []
        self.parameter_history = []
        velocity = np.zeros_like(params)
        
        for i in range(self.max_iterations):
            cost = cost_function(params)
            gradient = gradient_function(params)
            
            self.cost_history.append(cost)
            self.parameter_history.append(params.copy())
            
            # Update velocity and parameters
            velocity = momentum * velocity - self.learning_rate * gradient
            params = params + velocity
            
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Momentum GD converged after {i+1} iterations")
                break
        
        return params
    
    def _adam_gd(self, cost_function, gradient_function, params, beta1=0.9, beta2=0.999):
        """Adam optimization algorithm"""
        self.cost_history = []
        self.parameter_history = []
        
        m = np.zeros_like(params)  # First moment estimate
        v = np.zeros_like(params)  # Second moment estimate
        epsilon = 1e-8
        
        for i in range(self.max_iterations):
            cost = cost_function(params)
            gradient = gradient_function(params)
            
            self.cost_history.append(cost)
            self.parameter_history.append(params.copy())
            
            # Update biased first and second moment estimates
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            # Compute bias-corrected estimates
            m_corrected = m / (1 - beta1 ** (i + 1))
            v_corrected = v / (1 - beta2 ** (i + 1))
            
            # Update parameters
            params = params - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
            
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Adam converged after {i+1} iterations")
                break
        
        return params
    
    def _adagrad_gd(self, cost_function, gradient_function, params):
        """Adagrad optimization algorithm"""
        self.cost_history = []
        self.parameter_history = []
        
        G = np.zeros_like(params)  # Sum of squared gradients
        epsilon = 1e-8
        
        for i in range(self.max_iterations):
            cost = cost_function(params)
            gradient = gradient_function(params)
            
            self.cost_history.append(cost)
            self.parameter_history.append(params.copy())
            
            # Accumulate squared gradients
            G += gradient ** 2
            
            # Update parameters
            params = params - self.learning_rate * gradient / (np.sqrt(G) + epsilon)
            
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Adagrad converged after {i+1} iterations")
                break
        
        return params
    
    def _rmsprop_gd(self, cost_function, gradient_function, params, decay_rate=0.9):
        """RMSprop optimization algorithm"""
        self.cost_history = []
        self.parameter_history = []
        
        E_g2 = np.zeros_like(params)  # Running average of squared gradients
        epsilon = 1e-8
        
        for i in range(self.max_iterations):
            cost = cost_function(params)
            gradient = gradient_function(params)
            
            self.cost_history.append(cost)
            self.parameter_history.append(params.copy())
            
            # Update running average of squared gradients
            E_g2 = decay_rate * E_g2 + (1 - decay_rate) * (gradient ** 2)
            
            # Update parameters
            params = params - self.learning_rate * gradient / (np.sqrt(E_g2) + epsilon)
            
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"RMSprop converged after {i+1} iterations")
                break
        
        return params

# Example 1: Linear Regression
def linear_regression_example():
    """Implement linear regression using gradient descent"""
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_params = np.array([3.0, -2.0, 1.5])  # [w1, w2, bias]
    y = X @ true_params[:2] + true_params[2] + 0.1 * np.random.randn(100)
    
    # Add bias column to X
    X_with_bias = np.column_stack([X, np.ones(len(X))])
    
    def cost_function(params):
        """Mean squared error cost function"""
        predictions = X_with_bias @ params
        return np.mean((predictions - y) ** 2)
    
    def gradient_function(params):
        """Gradient of MSE cost function"""
        predictions = X_with_bias @ params
        errors = predictions - y
        return 2 * X_with_bias.T @ errors / len(y)
    
    # Initialize parameters
    initial_params = np.random.randn(3) * 0.1
    
    # Test different optimization methods
    methods = ['standard', 'momentum', 'adam']
    results = {}
    
    for method in methods:
        print(f"\n=== {method.upper()} ===")
        optimizer = GradientDescent(learning_rate=0.01, max_iterations=1000)
        final_params = optimizer.optimize(cost_function, gradient_function, 
                                        initial_params.copy(), method=method)
        
        results[method] = {
            'params': final_params,
            'cost_history': optimizer.cost_history,
            'final_cost': optimizer.cost_history[-1]
        }
        
        print(f"True parameters: {true_params}")
        print(f"Estimated parameters: {final_params}")
        print(f"Final cost: {optimizer.cost_history[-1]:.6f}")
    
    return results

# Example 2: Logistic Regression
def logistic_regression_example():
    """Implement logistic regression using gradient descent"""
    
    # Generate binary classification data
    np.random.seed(42)
    X = np.random.randn(200, 2)
    true_params = np.array([2.0, -1.5, 0.5])
    linear_combination = X @ true_params[:2] + true_params[2]
    probabilities = 1 / (1 + np.exp(-linear_combination))
    y = (np.random.rand(200) < probabilities).astype(int)
    
    # Add bias column
    X_with_bias = np.column_stack([X, np.ones(len(X))])
    
    def sigmoid(z):
        """Sigmoid activation function with numerical stability"""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)), 
                       np.exp(z) / (1 + np.exp(z)))
    
    def cost_function(params):
        """Logistic regression cost function (cross-entropy)"""
        z = X_with_bias @ params
        predictions = sigmoid(z)
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    
    def gradient_function(params):
        """Gradient of logistic regression cost function"""
        z = X_with_bias @ params
        predictions = sigmoid(z)
        errors = predictions - y
        return X_with_bias.T @ errors / len(y)
    
    # Initialize and optimize
    initial_params = np.random.randn(3) * 0.1
    optimizer = GradientDescent(learning_rate=0.1, max_iterations=1000)
    final_params = optimizer.optimize(cost_function, gradient_function, 
                                    initial_params, method='adam')
    
    print(f"\n=== LOGISTIC REGRESSION ===")
    print(f"True parameters: {true_params}")
    print(f"Estimated parameters: {final_params}")
    print(f"Final cost: {optimizer.cost_history[-1]:.6f}")
    
    # Calculate accuracy
    final_predictions = sigmoid(X_with_bias @ final_params) > 0.5
    accuracy = np.mean(final_predictions == y)
    print(f"Accuracy: {accuracy:.3f}")
    
    return final_params, optimizer.cost_history

# Example 3: Neural Network (Simple Multi-layer Perceptron)
def neural_network_example():
    """Simple neural network implementation with gradient descent"""
    
    # Generate XOR dataset (classic non-linearly separable problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR truth table
    
    class SimpleNeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size):
            # Initialize weights randomly
            self.W1 = np.random.randn(input_size, hidden_size) * 0.5
            self.b1 = np.zeros((1, hidden_size))
            self.W2 = np.random.randn(hidden_size, output_size) * 0.5
            self.b2 = np.zeros((1, output_size))
        
        def forward(self, X):
            """Forward propagation"""
            self.z1 = X @ self.W1 + self.b1
            self.a1 = np.tanh(self.z1)  # Hidden layer activation
            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = 1 / (1 + np.exp(-self.z2))  # Output layer (sigmoid)
            return self.a2
        
        def cost_function(self, X, y):
            """Compute cost function"""
            predictions = self.forward(X)
            return np.mean((predictions - y) ** 2)
        
        def backward(self, X, y):
            """Backward propagation to compute gradients"""
            m = X.shape[0]
            predictions = self.forward(X)
            
            # Output layer gradients
            dz2 = predictions - y
            dW2 = (1/m) * self.a1.T @ dz2
            db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
            
            # Hidden layer gradients
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (1 - self.a1 ** 2)  # tanh derivative
            dW1 = (1/m) * X.T @ dz1
            db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
            
            return dW1, db1, dW2, db2
        
        def get_params(self):
            """Get all parameters as a single vector"""
            return np.concatenate([
                self.W1.flatten(), self.b1.flatten(),
                self.W2.flatten(), self.b2.flatten()
            ])
        
        def set_params(self, params):
            """Set parameters from a single vector"""
            idx = 0
            
            # W1
            w1_size = self.W1.size
            self.W1 = params[idx:idx+w1_size].reshape(self.W1.shape)
            idx += w1_size
            
            # b1
            b1_size = self.b1.size
            self.b1 = params[idx:idx+b1_size].reshape(self.b1.shape)
            idx += b1_size
            
            # W2
            w2_size = self.W2.size
            self.W2 = params[idx:idx+w2_size].reshape(self.W2.shape)
            idx += w2_size
            
            # b2
            b2_size = self.b2.size
            self.b2 = params[idx:idx+b2_size].reshape(self.b2.shape)
    
    # Create network
    nn = SimpleNeuralNetwork(2, 4, 1)
    
    def cost_function(params):
        nn.set_params(params)
        return nn.cost_function(X, y)
    
    def gradient_function(params):
        nn.set_params(params)
        dW1, db1, dW2, db2 = nn.backward(X, y)
        return np.concatenate([
            dW1.flatten(), db1.flatten(),
            dW2.flatten(), db2.flatten()
        ])
    
    # Optimize
    initial_params = nn.get_params()
    optimizer = GradientDescent(learning_rate=1.0, max_iterations=5000)
    final_params = optimizer.optimize(cost_function, gradient_function, 
                                    initial_params, method='adam')
    
    # Test final network
    nn.set_params(final_params)
    final_predictions = nn.forward(X)
    
    print(f"\n=== NEURAL NETWORK (XOR) ===")
    print(f"Input -> Target -> Prediction")
    for i in range(len(X)):
        print(f"{X[i]} -> {y[i][0]} -> {final_predictions[i][0]:.3f}")
    
    print(f"Final cost: {optimizer.cost_history[-1]:.6f}")
    
    return nn, optimizer.cost_history

# Run examples
print("=== GRADIENT DESCENT OPTIMIZATION EXAMPLES ===")

# Linear regression
linear_results = linear_regression_example()

# Logistic regression
logistic_params, logistic_history = logistic_regression_example()

# Neural network
neural_network, nn_history = neural_network_example()

# Comparison of convergence rates
print(f"\n=== CONVERGENCE COMPARISON ===")
print(f"Linear regression (Standard GD): {len(linear_results['standard']['cost_history'])} iterations")
print(f"Linear regression (Momentum): {len(linear_results['momentum']['cost_history'])} iterations")
print(f"Linear regression (Adam): {len(linear_results['adam']['cost_history'])} iterations")
print(f"Logistic regression (Adam): {len(logistic_history)} iterations")
print(f"Neural network (Adam): {len(nn_history)} iterations")
```

### Use Cases
- Machine learning model training (linear/logistic regression, neural networks)
- Deep learning optimization
- Parameter estimation in scientific computing
- Economic modeling and optimization problems
- Signal processing and control systems

---

