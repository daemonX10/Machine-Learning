# Numpy Interview Questions - General Questions

## Question 1

**How do you inspect the shape and size of a NumPy array?**

**Answer:**

### Theory
Array inspection is fundamental to NumPy programming. Understanding array properties like shape, size, dimensions, and data types is crucial for debugging, optimization, and ensuring correct operations. NumPy provides comprehensive attributes and functions for array introspection.

### Key Inspection Attributes:

#### 1. **Shape and Dimensions**
- `.shape`: Tuple of array dimensions
- `.ndim`: Number of dimensions
- `.size`: Total number of elements

#### 2. **Data Properties**
- `.dtype`: Data type of elements
- `.itemsize`: Size of each element in bytes
- `.nbytes`: Total bytes consumed

#### 3. **Memory Layout**
- `.strides`: Bytes to skip for each dimension
- `.flags`: Memory layout information

#### Code Example

```python
import numpy as np

# Create various arrays for inspection
array_1d = np.array([1, 2, 3, 4, 5])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
array_mixed = np.array([1.5, 2.7, 3.9])

print("=== Basic Shape and Size Inspection ===")

def inspect_array(arr, name):
    """Comprehensive array inspection function"""
    print(f"\n{name}:")
    print(f"  Array: {arr}")
    print(f"  Shape: {arr.shape}")
    print(f"  Size: {arr.size}")
    print(f"  Dimensions: {arr.ndim}")
    print(f"  Data type: {arr.dtype}")
    print(f"  Item size: {arr.itemsize} bytes")
    print(f"  Total bytes: {arr.nbytes} bytes")

inspect_array(array_1d, "1D Array")
inspect_array(array_2d, "2D Array")
inspect_array(array_3d, "3D Array")
inspect_array(array_mixed, "Float Array")

# Detailed shape analysis
print("\n=== Detailed Shape Analysis ===")

# Understanding shape tuple
print(f"2D array shape: {array_2d.shape}")
print(f"  - Rows (axis 0): {array_2d.shape[0]}")
print(f"  - Columns (axis 1): {array_2d.shape[1]}")

print(f"\n3D array shape: {array_3d.shape}")
print(f"  - Depth (axis 0): {array_3d.shape[0]}")
print(f"  - Rows (axis 1): {array_3d.shape[1]}")
print(f"  - Columns (axis 2): {array_3d.shape[2]}")

# Shape manipulation verification
reshaped = array_1d.reshape(5, 1)
print(f"\nOriginal 1D shape: {array_1d.shape}")
print(f"Reshaped to 2D: {reshaped.shape}")
print(f"Same data: {np.array_equal(array_1d.flatten(), reshaped.flatten())}")

# Advanced inspection techniques
print("\n=== Advanced Inspection ===")

# Memory layout inspection
def memory_inspection(arr, name):
    print(f"\n{name} Memory Layout:")
    print(f"  Strides: {arr.strides}")
    print(f"  C-contiguous: {arr.flags['C_CONTIGUOUS']}")
    print(f"  Fortran-contiguous: {arr.flags['F_CONTIGUOUS']}")
    print(f"  Memory aligned: {arr.flags['ALIGNED']}")
    print(f"  Writable: {arr.flags['WRITEABLE']}")

memory_inspection(array_2d, "2D Array")

# Create Fortran-order array for comparison
array_fortran = np.array([[1, 2, 3], [4, 5, 6]], order='F')
memory_inspection(array_fortran, "Fortran-order Array")

# Transpose memory layout
array_transposed = array_2d.T
memory_inspection(array_transposed, "Transposed Array")

# Data type inspection
print("\n=== Data Type Analysis ===")

arrays_different_types = {
    'int8': np.array([1, 2, 3], dtype=np.int8),
    'int32': np.array([1, 2, 3], dtype=np.int32),
    'int64': np.array([1, 2, 3], dtype=np.int64),
    'float32': np.array([1.0, 2.0, 3.0], dtype=np.float32),
    'float64': np.array([1.0, 2.0, 3.0], dtype=np.float64),
    'complex': np.array([1+2j, 3+4j], dtype=np.complex128),
    'bool': np.array([True, False, True], dtype=bool)
}

for type_name, arr in arrays_different_types.items():
    print(f"\n{type_name.upper()} Array:")
    print(f"  Data type: {arr.dtype}")
    print(f"  Item size: {arr.itemsize} bytes")
    print(f"  Total memory: {arr.nbytes} bytes")
    print(f"  Type info: {np.iinfo(arr.dtype) if 'int' in str(arr.dtype) else np.finfo(arr.dtype) if 'float' in str(arr.dtype) else 'N/A'}")

# Size efficiency comparison
print("\n=== Memory Efficiency Comparison ===")

sizes = [1000, 10000, 100000]
for size in sizes:
    arr_int32 = np.ones(size, dtype=np.int32)
    arr_int64 = np.ones(size, dtype=np.int64)
    arr_float32 = np.ones(size, dtype=np.float32)
    arr_float64 = np.ones(size, dtype=np.float64)
    
    print(f"\nArray size: {size} elements")
    print(f"  int32:   {arr_int32.nbytes:8} bytes ({arr_int32.nbytes/1024:.1f} KB)")
    print(f"  int64:   {arr_int64.nbytes:8} bytes ({arr_int64.nbytes/1024:.1f} KB)")
    print(f"  float32: {arr_float32.nbytes:8} bytes ({arr_float32.nbytes/1024:.1f} KB)")
    print(f"  float64: {arr_float64.nbytes:8} bytes ({arr_float64.nbytes/1024:.1f} KB)")

# Practical inspection utilities
print("\n=== Practical Utilities ===")

def array_summary(arr, name="Array"):
    """Comprehensive array summary for debugging"""
    print(f"\n{name} Summary:")
    print(f"  {'Shape:':<15} {arr.shape}")
    print(f"  {'Size:':<15} {arr.size:,} elements")
    print(f"  {'Dimensions:':<15} {arr.ndim}")
    print(f"  {'Data type:':<15} {arr.dtype}")
    print(f"  {'Memory:':<15} {arr.nbytes:,} bytes ({arr.nbytes/1024/1024:.2f} MB)")
    print(f"  {'Min value:':<15} {arr.min()}")
    print(f"  {'Max value:':<15} {arr.max()}")
    print(f"  {'Mean:':<15} {arr.mean():.2f}")
    
    if arr.ndim > 1:
        print(f"  {'Strides:':<15} {arr.strides}")
        print(f"  {'C-contiguous:':<15} {arr.flags['C_CONTIGUOUS']}")

# Test with various arrays
test_arrays = {
    "Small 2D": np.random.randn(3, 4),
    "Large 1D": np.random.randn(100000),
    "3D Tensor": np.random.randn(10, 20, 30),
    "Integer Matrix": np.random.randint(0, 100, (50, 50))
}

for name, arr in test_arrays.items():
    array_summary(arr, name)

# Shape validation utilities
print("\n=== Shape Validation Utilities ===")

def validate_shapes_for_operation(arr1, arr2, operation="element-wise"):
    """Validate array shapes for different operations"""
    print(f"\nValidating shapes for {operation} operation:")
    print(f"  Array 1 shape: {arr1.shape}")
    print(f"  Array 2 shape: {arr2.shape}")
    
    if operation == "element-wise":
        compatible = arr1.shape == arr2.shape
        print(f"  Compatible: {compatible}")
        if not compatible:
            print("  Note: Broadcasting may still work")
            try:
                result_shape = np.broadcast_shapes(arr1.shape, arr2.shape)
                print(f"  Broadcast result shape: {result_shape}")
            except ValueError:
                print("  Broadcasting not possible")
    
    elif operation == "matrix_multiplication":
        if arr1.ndim >= 2 and arr2.ndim >= 2:
            compatible = arr1.shape[-1] == arr2.shape[-2]
            print(f"  Compatible: {compatible}")
            if compatible:
                result_shape = arr1.shape[:-1] + arr2.shape[-1:]
                print(f"  Result shape: {result_shape}")
        else:
            print("  Need at least 2D arrays for matrix multiplication")

# Test shape validation
arr_a = np.random.randn(3, 4)
arr_b = np.random.randn(4, 5)
arr_c = np.random.randn(3, 4)

validate_shapes_for_operation(arr_a, arr_c, "element-wise")
validate_shapes_for_operation(arr_a, arr_b, "matrix_multiplication")

# Dynamic array inspection during processing
print("\n=== Dynamic Inspection Example ===")

def process_with_inspection(data, operation_name):
    """Process data with inspection at each step"""
    print(f"\n{operation_name}:")
    print(f"  Input shape: {data.shape}, size: {data.size}")
    
    # Simulate processing steps
    step1 = data * 2
    print(f"  After scaling: shape={step1.shape}, min={step1.min():.2f}, max={step1.max():.2f}")
    
    step2 = step1.reshape(-1)
    print(f"  After flattening: shape={step2.shape}")
    
    step3 = step2[step2 > 0]
    print(f"  After filtering: shape={step3.shape} ({step3.size/step2.size*100:.1f}% remaining)")
    
    return step3

# Example processing pipeline
sample_data = np.random.randn(5, 6) * 10
result = process_with_inspection(sample_data, "Data Processing Pipeline")

# Comparison utilities
print("\n=== Array Comparison Utilities ===")

def compare_arrays(arr1, arr2, name1="Array 1", name2="Array 2"):
    """Compare two arrays across multiple dimensions"""
    print(f"\nComparing {name1} vs {name2}:")
    print(f"  {name1:<12} shape: {arr1.shape}, size: {arr1.size}, dtype: {arr1.dtype}")
    print(f"  {name2:<12} shape: {arr2.shape}, size: {arr2.size}, dtype: {arr2.dtype}")
    print(f"  Memory ratio: {arr2.nbytes / arr1.nbytes:.2f}x")
    
    if arr1.shape == arr2.shape:
        print(f"  Element-wise equal: {np.array_equal(arr1, arr2)}")
        if arr1.dtype == arr2.dtype and np.issubdtype(arr1.dtype, np.number):
            print(f"  Correlation: {np.corrcoef(arr1.flatten(), arr2.flatten())[0,1]:.3f}")

# Test array comparison
original = np.random.randn(100, 100)
compressed = original.astype(np.float32)
reshaped = original.reshape(200, 50)

compare_arrays(original, compressed, "Original (float64)", "Compressed (float32)")
compare_arrays(original, reshaped, "Original", "Reshaped")
```

#### Explanation

1. **Basic Inspection**: Use `.shape`, `.size`, `.ndim` for fundamental properties
2. **Memory Analysis**: Check `.nbytes`, `.itemsize`, `.strides` for memory usage
3. **Data Type Info**: Use `.dtype` and type-specific info functions
4. **Layout Information**: Examine `.flags` for memory organization

#### Use Cases

1. **Debugging Shape Issues**:
   ```python
   def debug_shape_error(func, *arrays):
       try:
           return func(*arrays)
       except ValueError as e:
           for i, arr in enumerate(arrays):
               print(f"Array {i}: shape={arr.shape}")
           raise e
   ```

2. **Memory Optimization**:
   ```python
   def suggest_dtype_optimization(arr):
       if np.issubdtype(arr.dtype, np.integer):
           if arr.min() >= 0 and arr.max() < 256:
               return np.uint8
           elif arr.min() >= -128 and arr.max() < 128:
               return np.int8
       return arr.dtype
   ```

3. **Performance Monitoring**:
   ```python
   def track_memory_usage(arrays_dict):
       total_memory = sum(arr.nbytes for arr in arrays_dict.values())
       for name, arr in arrays_dict.items():
           percentage = (arr.nbytes / total_memory) * 100
           print(f"{name}: {percentage:.1f}% of total memory")
   ```

#### Best Practices

1. **Regular Inspection**: Check array properties during development
2. **Memory Awareness**: Monitor memory usage for large arrays
3. **Data Type Optimization**: Choose appropriate dtypes for memory efficiency
4. **Shape Validation**: Verify shapes before operations
5. **Documentation**: Document expected array shapes in functions

#### Pitfalls

1. **Shape Assumptions**: Don't assume array shapes without checking
2. **Memory Overhead**: Large arrays can consume unexpected memory
3. **Data Type Surprises**: Operations may change data types unexpectedly
4. **Copy vs View**: Shape operations may create copies
5. **Broadcasting Confusion**: Broadcasting can mask shape errors

#### Debugging

```python
def comprehensive_debug(arr, context=""):
    """Complete debugging information for arrays"""
    print(f"\n=== DEBUG: {context} ===")
    print(f"Type: {type(arr)}")
    print(f"Shape: {arr.shape}")
    print(f"Size: {arr.size}")
    print(f"Dtype: {arr.dtype}")
    print(f"Memory: {arr.nbytes} bytes")
    print(f"Contiguous: C={arr.flags['C_CONTIGUOUS']}, F={arr.flags['F_CONTIGUOUS']}")
    print(f"Min/Max: {arr.min()}/{arr.max()}")
    print(f"Has NaN: {np.isnan(arr).any()}")
    print(f"Has Inf: {np.isinf(arr).any()}")
```

#### Optimization

1. **Efficient Inspection**: Use properties instead of functions when possible
2. **Batch Inspection**: Inspect multiple arrays together
3. **Conditional Checks**: Only perform expensive checks when necessary
4. **Caching Results**: Store frequently accessed properties
5. **Memory Profiling**: Use memory profilers for detailed analysis

---

## Question 2

**How do you perform element-wise operations in NumPy?**

**Answer:**

### Theory
Element-wise operations (also called vectorized operations) apply functions to each element of an array without explicit loops. This is one of NumPy's core features, providing both performance benefits and clean, readable code. NumPy's broadcasting rules enable element-wise operations between arrays of different shapes.

### Key Concepts:

#### 1. **Basic Arithmetic Operations**
- `+`, `-`, `*`, `/`: Standard arithmetic
- `**`: Exponentiation
- `%`: Modulo
- `//`: Floor division

#### 2. **Universal Functions (ufuncs)**
- Vectorized functions operating element-wise
- Optimized C implementations
- Support broadcasting and multiple outputs

#### 3. **Broadcasting Rules**
- Enables operations between different-shaped arrays
- Arrays aligned from trailing dimensions
- Dimensions of size 1 are stretched

#### Code Example

```python
import numpy as np

# Create sample arrays for demonstrations
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])
matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
matrix_b = np.array([[10, 20, 30], [40, 50, 60]])

print("=== Basic Element-wise Arithmetic ===")
print(f"Array a: {a}")
print(f"Array b: {b}")

# Basic arithmetic operations
print(f"\nAddition (a + b): {a + b}")
print(f"Subtraction (a - b): {a - b}")
print(f"Multiplication (a * b): {a * b}")
print(f"Division (a / b): {a / b}")
print(f"Power (a ** 2): {a ** 2}")
print(f"Modulo (b % 3): {b % 3}")
print(f"Floor division (b // 3): {b // 3}")

# Matrix element-wise operations
print(f"\n=== Matrix Element-wise Operations ===")
print(f"Matrix A:\n{matrix_a}")
print(f"Matrix B:\n{matrix_b}")
print(f"\nElement-wise multiplication:\n{matrix_a * matrix_b}")
print(f"Element-wise division:\n{matrix_a / matrix_b}")

# Comparison with scalar operations
scalar = 5
print(f"\n=== Scalar Operations ===")
print(f"Array: {a}")
print(f"Scalar: {scalar}")
print(f"Array + scalar: {a + scalar}")
print(f"Array * scalar: {a * scalar}")
print(f"Array > scalar: {a > scalar}")

# Universal functions (ufuncs)
print(f"\n=== Universal Functions ===")
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"Angles: {angles}")
print(f"sin(angles): {np.sin(angles)}")
print(f"cos(angles): {np.cos(angles)}")
print(f"tan(angles): {np.tan(angles)}")

# Mathematical functions
data = np.array([1, 4, 9, 16, 25])
print(f"\nData: {data}")
print(f"Square root: {np.sqrt(data)}")
print(f"Natural log: {np.log(data)}")
print(f"Exponential: {np.exp([1, 2, 3])}")

# Logical operations
bool_a = np.array([True, False, True, False])
bool_b = np.array([True, True, False, False])
print(f"\n=== Logical Operations ===")
print(f"Boolean array A: {bool_a}")
print(f"Boolean array B: {bool_b}")
print(f"Logical AND: {np.logical_and(bool_a, bool_b)}")
print(f"Logical OR: {np.logical_or(bool_a, bool_b)}")
print(f"Logical NOT: {np.logical_not(bool_a)}")
print(f"Logical XOR: {np.logical_xor(bool_a, bool_b)}")

# Comparison operations
print(f"\n=== Comparison Operations ===")
x = np.array([1, 5, 3, 8, 2])
y = np.array([2, 4, 3, 7, 6])
print(f"Array X: {x}")
print(f"Array Y: {y}")
print(f"X == Y: {x == y}")
print(f"X > Y: {x > y}")
print(f"X <= Y: {x <= y}")
print(f"X != Y: {x != y}")

# Broadcasting examples
print(f"\n=== Broadcasting Examples ===")

# 1D array with 2D array
arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[10], [20], [30]])

print(f"1D array: {arr_1d}")
print(f"2D array:\n{arr_2d}")
print(f"Broadcasting addition:\n{arr_1d + arr_2d}")

# Different broadcasting scenarios
print(f"\nBroadcasting scenarios:")

# Scalar with array
scalar_broadcast = 5 + np.array([[1, 2], [3, 4]])
print(f"Scalar + 2D array:\n{scalar_broadcast}")

# Row vector with column vector
row = np.array([[1, 2, 3]])
col = np.array([[10], [20]])
print(f"Row vector: {row}")
print(f"Column vector:\n{col}")
print(f"Row + Column broadcasting:\n{row + col}")

# Advanced element-wise operations
print(f"\n=== Advanced Element-wise Operations ===")

# Conditional operations
data_with_negatives = np.array([-2, -1, 0, 1, 2])
print(f"Original data: {data_with_negatives}")
print(f"Absolute values: {np.abs(data_with_negatives)}")
print(f"Clip to [0, 1]: {np.clip(data_with_negatives, 0, 1)}")

# Where function for conditional replacement
condition = data_with_negatives > 0
result = np.where(condition, data_with_negatives, 0)
print(f"Replace negatives with 0: {result}")

# Multiple conditions
complex_condition = np.where(data_with_negatives > 0, 
                           data_with_negatives * 2,  # if positive, multiply by 2
                           data_with_negatives / 2)  # if negative, divide by 2
print(f"Complex conditional operation: {complex_condition}")

# Performance comparison
print(f"\n=== Performance Comparison ===")

import time

# Create large arrays for performance testing
large_a = np.random.randn(1000000)
large_b = np.random.randn(1000000)

# NumPy vectorized operation
start_time = time.time()
numpy_result = large_a * large_b + np.sin(large_a)
numpy_time = time.time() - start_time

# Pure Python equivalent (slower)
start_time = time.time()
python_result = []
for i in range(len(large_a)):
    python_result.append(large_a[i] * large_b[i] + np.sin(large_a[i]))
python_result = np.array(python_result)
python_time = time.time() - start_time

print(f"NumPy vectorized time: {numpy_time:.4f} seconds")
print(f"Pure Python time: {python_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster")
print(f"Results equal: {np.allclose(numpy_result, python_result)}")

# Custom element-wise functions
print(f"\n=== Custom Element-wise Functions ===")

# Using np.vectorize for custom functions
def custom_function(x):
    """Custom function to be applied element-wise"""
    if x < 0:
        return x ** 2
    elif x < 5:
        return x * 2
    else:
        return x / 2

# Vectorize the function
vectorized_custom = np.vectorize(custom_function)

test_array = np.array([-3, -1, 0, 2, 5, 8, 10])
print(f"Input array: {test_array}")
print(f"Custom function applied: {vectorized_custom(test_array)}")

# More efficient approach using numpy operations
def efficient_custom(x):
    """Efficient vectorized version using NumPy operations"""
    result = np.empty_like(x, dtype=float)
    mask1 = x < 0
    mask2 = (x >= 0) & (x < 5)
    mask3 = x >= 5
    
    result[mask1] = x[mask1] ** 2
    result[mask2] = x[mask2] * 2
    result[mask3] = x[mask3] / 2
    
    return result

efficient_result = efficient_custom(test_array.astype(float))
print(f"Efficient version result: {efficient_result}")

# Chaining operations
print(f"\n=== Chaining Element-wise Operations ===")

# Complex mathematical expressions
x = np.linspace(-5, 5, 11)
print(f"Input x: {x}")

# Chain multiple operations
complex_result = np.sqrt(np.abs(x)) * np.sin(x**2) + np.cos(x/2)
print(f"Complex expression result: {complex_result}")

# Step-by-step breakdown for debugging
step1 = np.abs(x)
step2 = np.sqrt(step1)
step3 = x**2
step4 = np.sin(step3)
step5 = step2 * step4
step6 = x/2
step7 = np.cos(step6)
final = step5 + step7

print(f"Verification (should be same): {np.allclose(complex_result, final)}")

# Memory-efficient operations
print(f"\n=== Memory-Efficient Operations ===")

# In-place operations to save memory
large_array = np.random.randn(100000)
original_id = id(large_array)

# In-place addition
large_array += 5
print(f"In-place operation preserves memory: {id(large_array) == original_id}")

# Out-of-place operation creates new array
new_array = large_array + 5
print(f"Out-of-place operation creates new array: {id(new_array) == original_id}")

# Error handling and edge cases
print(f"\n=== Error Handling ===")

# Division by zero
safe_division = np.array([1, 2, 3, 4])
divisor = np.array([1, 0, 2, 0])

# Using np.divide with where parameter
safe_result = np.divide(safe_division, divisor, 
                       out=np.zeros_like(safe_division, dtype=float),
                       where=divisor!=0)
print(f"Safe division result: {safe_result}")

# Using np.errstate for error control
with np.errstate(divide='ignore', invalid='ignore'):
    division_result = safe_division / divisor
    division_result[np.isinf(division_result)] = 0  # Replace inf with 0
    division_result[np.isnan(division_result)] = 0  # Replace nan with 0
print(f"Controlled division result: {division_result}")

# Broadcasting validation
def validate_broadcasting(arr1, arr2):
    """Check if two arrays can be broadcast together"""
    try:
        np.broadcast_arrays(arr1, arr2)
        return True, "Compatible for broadcasting"
    except ValueError as e:
        return False, str(e)

# Test broadcasting validation
test_cases = [
    (np.array([1, 2, 3]), np.array([[1], [2]])),
    (np.array([[1, 2]]), np.array([[1], [2], [3]])),
    (np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
]

for i, (a1, a2) in enumerate(test_cases, 1):
    is_compatible, message = validate_broadcasting(a1, a2)
    print(f"Test case {i}: {is_compatible} - {message}")

# Practical applications
print(f"\n=== Practical Applications ===")

# Image processing simulation
def apply_brightness_contrast(image, brightness=0, contrast=1):
    """Apply brightness and contrast adjustments element-wise"""
    # Simulate image as 2D array
    return np.clip(image * contrast + brightness, 0, 255)

# Simulate grayscale image
image = np.random.randint(0, 256, (5, 5))
print(f"Original image:\n{image}")

# Apply transformations
bright_image = apply_brightness_contrast(image, brightness=50)
high_contrast = apply_brightness_contrast(image, contrast=1.5)

print(f"Brightened image:\n{bright_image}")
print(f"High contrast image:\n{high_contrast}")

# Statistical operations element-wise
def normalize_features(data, axis=0):
    """Normalize features using element-wise operations"""
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

# Example data normalization
feature_data = np.random.randn(100, 5) * 10 + 5  # 100 samples, 5 features
normalized_data = normalize_features(feature_data)

print(f"Original data mean per feature: {np.mean(feature_data, axis=0)}")
print(f"Original data std per feature: {np.std(feature_data, axis=0)}")
print(f"Normalized data mean per feature: {np.mean(normalized_data, axis=0)}")
print(f"Normalized data std per feature: {np.std(normalized_data, axis=0)}")
```

#### Explanation

1. **Vectorization**: Operations apply to entire arrays without explicit loops
2. **Broadcasting**: Enables operations between different-shaped arrays
3. **Performance**: C-optimized implementations are much faster than Python loops
4. **Memory Efficiency**: In-place operations modify arrays without creating copies

#### Use Cases

1. **Data Processing**:
   ```python
   # Normalize dataset
   normalized = (data - data.mean()) / data.std()
   
   # Apply transformations
   log_transformed = np.log1p(data[data > 0])
   ```

2. **Image Processing**:
   ```python
   # Adjust image properties
   enhanced = np.clip(image * 1.2 + 30, 0, 255)
   
   # Apply filters
   smoothed = (image[:-2, :-2] + image[1:-1, 1:-1] + image[2:, 2:]) / 3
   ```

3. **Financial Calculations**:
   ```python
   # Calculate returns
   returns = (prices[1:] - prices[:-1]) / prices[:-1]
   
   # Risk metrics
   volatility = np.std(returns) * np.sqrt(252)
   ```

#### Best Practices

1. **Use Vectorization**: Prefer NumPy operations over Python loops
2. **Leverage Broadcasting**: Understand broadcasting rules for flexible operations
3. **Handle Edge Cases**: Consider division by zero, overflow, and underflow
4. **Memory Management**: Use in-place operations when appropriate
5. **Profile Performance**: Compare vectorized vs loop-based approaches

#### Pitfalls

1. **Broadcasting Errors**: Incompatible array shapes cause errors
2. **Memory Usage**: Large temporary arrays can cause memory issues
3. **Precision Loss**: Floating-point operations may accumulate errors
4. **Type Promotion**: Operations may change array data types unexpectedly
5. **Overflow/Underflow**: Large numbers may exceed data type limits

#### Debugging

```python
def debug_element_wise_op(arr1, arr2, operation):
    """Debug element-wise operations"""
    print(f"Array 1 shape: {arr1.shape}, dtype: {arr1.dtype}")
    print(f"Array 2 shape: {arr2.shape}, dtype: {arr2.dtype}")
    
    try:
        result = operation(arr1, arr2)
        print(f"Operation successful, result shape: {result.shape}")
        return result
    except Exception as e:
        print(f"Operation failed: {e}")
        return None
```

#### Optimization

1. **Use Appropriate Data Types**: Choose optimal dtypes for memory and speed
2. **Minimize Temporary Arrays**: Chain operations to reduce memory usage
3. **Leverage SIMD**: NumPy automatically uses vectorized instructions
4. **Consider Memory Layout**: Contiguous arrays perform better
5. **Profile Bottlenecks**: Identify and optimize slow operations

---

## Question 3

**How do you compute the mean, median, and standard deviation with NumPy?**

### Theory

Statistical measures are fundamental for data analysis and understanding data distributions. NumPy provides efficient functions for computing various statistical measures including central tendency (mean, median) and dispersion (standard deviation, variance). These functions support axis-wise operations and handle missing data appropriately.

### Code Example

```python
import numpy as np

# Sample data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"1D Data: {data}")
print(f"2D Matrix:\n{matrix}")

# Basic statistics for 1D array
mean_val = np.mean(data)
median_val = np.median(data)
std_val = np.std(data)
var_val = np.var(data)

print(f"\n1D Array Statistics:")
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Standard Deviation: {std_val}")
print(f"Variance: {var_val}")

# 2D array statistics along different axes
print(f"\n2D Array Statistics:")
print(f"Overall mean: {np.mean(matrix)}")
print(f"Mean along axis 0 (columns): {np.mean(matrix, axis=0)}")
print(f"Mean along axis 1 (rows): {np.mean(matrix, axis=1)}")

# Additional statistical functions
print(f"\nAdditional Statistics:")
print(f"Min: {np.min(data)}, Max: {np.max(data)}")
print(f"25th percentile: {np.percentile(data, 25)}")
print(f"75th percentile: {np.percentile(data, 75)}")
print(f"Range: {np.ptp(data)}")  # Peak to peak (max - min)

# Handling missing data
data_with_nan = np.array([1, 2, np.nan, 4, 5])
print(f"\nData with NaN: {data_with_nan}")
print(f"Mean (ignoring NaN): {np.nanmean(data_with_nan)}")
print(f"Median (ignoring NaN): {np.nanmedian(data_with_nan)}")
print(f"Std (ignoring NaN): {np.nanstd(data_with_nan)}")
```

### Use Cases
- Data exploration and exploratory data analysis
- Quality control and process monitoring
- Feature engineering in machine learning
- Financial analysis and risk assessment

---

## Question 4

**Why is NumPy more efficient for numerical computations than pure Python?**

### Theory

NumPy's efficiency comes from several key factors: vectorized operations implemented in C, contiguous memory layout, elimination of Python's interpreted overhead, optimized BLAS/LAPACK libraries, and efficient broadcasting mechanisms. Understanding these factors is crucial for writing high-performance numerical code.

### Code Example

```python
import numpy as np
import time

# Performance comparison example
def pure_python_operations():
    # Pure Python list operations
    python_list1 = list(range(1000000))
    python_list2 = list(range(1000000, 2000000))
    
    start_time = time.time()
    result = [a + b for a, b in zip(python_list1, python_list2)]
    python_time = time.time() - start_time
    
    return python_time

def numpy_operations():
    # NumPy array operations
    numpy_array1 = np.arange(1000000)
    numpy_array2 = np.arange(1000000, 2000000)
    
    start_time = time.time()
    result = numpy_array1 + numpy_array2
    numpy_time = time.time() - start_time
    
    return numpy_time

python_time = pure_python_operations()
numpy_time = numpy_operations()

print(f"Pure Python time: {python_time:.6f} seconds")
print(f"NumPy time: {numpy_time:.6f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster")

# Memory efficiency comparison
python_list = [1.0] * 1000000
numpy_array = np.ones(1000000, dtype=np.float64)

import sys
python_memory = sys.getsizeof(python_list) + sum(sys.getsizeof(x) for x in python_list)
numpy_memory = numpy_array.nbytes

print(f"\nMemory usage:")
print(f"Python list: {python_memory:,} bytes")
print(f"NumPy array: {numpy_memory:,} bytes")
print(f"NumPy uses {python_memory/numpy_memory:.1f}x less memory")
```

### Key Efficiency Factors
1. **C Implementation**: Core operations written in C/Fortran
2. **Vectorization**: Operations applied to entire arrays at once
3. **Memory Layout**: Contiguous memory storage reduces cache misses
4. **No Python Overhead**: Bypasses Python's interpretation for inner loops
5. **Optimized Libraries**: Uses BLAS/LAPACK for linear algebra operations

---

## Question 5

**How do you check the memory size of a NumPy array?**

### Code Example

```python
import numpy as np

# Create sample arrays
int_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
float_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
large_array = np.random.randn(1000, 1000)

print("Array Memory Information:")
print(f"Int array shape: {int_array.shape}")
print(f"Int array dtype: {int_array.dtype}")
print(f"Int array itemsize: {int_array.itemsize} bytes per element")
print(f"Int array nbytes: {int_array.nbytes} bytes total")
print(f"Int array size: {int_array.size} elements")

print(f"\nFloat array nbytes: {float_array.nbytes} bytes")
print(f"Large array nbytes: {large_array.nbytes:,} bytes ({large_array.nbytes/1024/1024:.2f} MB)")

# Memory usage by data type
dtypes_to_test = [np.int8, np.int16, np.int32, np.int64, 
                  np.float16, np.float32, np.float64, np.complex128]

print(f"\nMemory usage by data type (1000 elements):")
for dtype in dtypes_to_test:
    arr = np.ones(1000, dtype=dtype)
    print(f"{str(dtype):15}: {arr.nbytes:5} bytes ({arr.itemsize} bytes per element)")

# Memory layout information
print(f"\nMemory layout information:")
print(f"Array flags: {large_array.flags}")
print(f"Is C-contiguous: {large_array.flags['C_CONTIGUOUS']}")
print(f"Is Fortran-contiguous: {large_array.flags['F_CONTIGUOUS']}")
print(f"Array base: {large_array.base}")
```

### Memory Optimization Tips
- Choose appropriate data types (int8 vs int64)
- Use views instead of copies when possible
- Consider memory layout for performance
- Monitor memory usage in data processing pipelines

---

## Question 6

**How do you create a record array in NumPy?**

### Code Example

```python
import numpy as np

# Method 1: Using structured dtype
dtype = [('name', 'U20'), ('age', 'i4'), ('salary', 'f8'), ('active', '?')]
employees = np.array([
    ('Alice Johnson', 25, 75000.0, True),
    ('Bob Smith', 30, 85000.0, True),
    ('Charlie Brown', 35, 95000.0, False),
    ('Diana Prince', 28, 80000.0, True)
], dtype=dtype)

print("Structured Array:")
print(employees)
print(f"Names: {employees['name']}")
print(f"Ages: {employees['age']}")

# Method 2: Using np.rec.array (record array)
rec_array = np.rec.array([
    ('Alice', 25, 75000.0),
    ('Bob', 30, 85000.0),
    ('Charlie', 35, 95000.0)
], dtype=[('name', 'U20'), ('age', 'i4'), ('salary', 'f8')])

print(f"\nRecord Array:")
print(rec_array)
print(f"Access by attribute: {rec_array.name}")
print(f"Access by index: {rec_array[0].name}")

# Method 3: Creating from existing arrays
names = np.array(['Alice', 'Bob', 'Charlie'])
ages = np.array([25, 30, 35])
salaries = np.array([75000.0, 85000.0, 95000.0])

# Combine into record array
combined_rec = np.rec.fromarrays([names, ages, salaries], 
                                names=['name', 'age', 'salary'])
print(f"\nFrom arrays:")
print(combined_rec)

# Method 4: Reading from CSV-like data
csv_data = """Alice,25,75000
Bob,30,85000
Charlie,35,95000"""

# Save to temporary file and read
with open('temp_data.csv', 'w') as f:
    f.write(csv_data)

csv_rec = np.rec.fromrecords(
    np.genfromtxt('temp_data.csv', delimiter=',', dtype=None, encoding='utf-8'),
    names=['name', 'age', 'salary']
)
print(f"\nFrom CSV:")
print(csv_rec)

# Advanced operations on record arrays
print(f"\nAdvanced Operations:")
print(f"Average age: {np.mean(employees['age'])}")
print(f"High earners (>80k): {employees[employees['salary'] > 80000]['name']}")
print(f"Active employees: {employees[employees['active']]['name']}")

import os
os.remove('temp_data.csv')  # Cleanup
```

### Use Cases
- Database-like operations in NumPy
- Scientific data with heterogeneous fields
- Reading structured data from files
- Time series data with multiple attributes

---

## Question 7

**How can NumPy be used for audio signal processing?**

### Code Example

```python
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

# Generate synthetic audio signals
sample_rate = 44100  # Standard audio sample rate
duration = 2.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))

# Generate different waveforms
sine_wave = np.sin(2 * np.pi * 440 * t)  # A4 note (440 Hz)
square_wave = signal.square(2 * np.pi * 440 * t)
sawtooth_wave = signal.sawtooth(2 * np.pi * 440 * t)

print(f"Audio signal properties:")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {duration} seconds")
print(f"Signal length: {len(sine_wave)} samples")

# Audio effects using NumPy
def add_reverb(audio, delay_samples, decay_factor):
    """Add simple reverb effect"""
    reverb = np.zeros_like(audio)
    reverb[delay_samples:] = audio[:-delay_samples] * decay_factor
    return audio + reverb

def apply_fade(audio, fade_in_samples, fade_out_samples):
    """Apply fade in/out effect"""
    result = audio.copy()
    
    # Fade in
    fade_in_curve = np.linspace(0, 1, fade_in_samples)
    result[:fade_in_samples] *= fade_in_curve
    
    # Fade out
    fade_out_curve = np.linspace(1, 0, fade_out_samples)
    result[-fade_out_samples:] *= fade_out_curve
    
    return result

def low_pass_filter(audio, cutoff_freq, sample_rate):
    """Apply simple low-pass filter"""
    nyquist = sample_rate // 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, audio)

# Apply effects
reverb_audio = add_reverb(sine_wave, 4410, 0.3)  # 100ms delay
faded_audio = apply_fade(sine_wave, 2205, 2205)  # 50ms fade in/out
filtered_audio = low_pass_filter(sine_wave, 2000, sample_rate)

# Frequency domain analysis
def analyze_frequency_spectrum(audio, sample_rate):
    """Analyze frequency spectrum using FFT"""
    fft_result = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(len(audio), 1/sample_rate)
    
    # Only take positive frequencies
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    magnitude = np.abs(fft_result[positive_freq_idx])
    
    return frequencies, magnitude

# Analyze spectrum
freqs, magnitude = analyze_frequency_spectrum(sine_wave[:4410], sample_rate)
peak_freq_idx = np.argmax(magnitude)
peak_frequency = freqs[peak_freq_idx]

print(f"\nFrequency analysis:")
print(f"Peak frequency detected: {peak_frequency:.1f} Hz")
print(f"Expected frequency: 440 Hz")

# Audio mixing
def mix_audio_signals(*signals, weights=None):
    """Mix multiple audio signals"""
    if weights is None:
        weights = [1.0] * len(signals)
    
    # Ensure all signals have same length
    min_length = min(len(sig) for sig in signals)
    signals = [sig[:min_length] for sig in signals]
    
    mixed = np.zeros(min_length)
    for signal_data, weight in zip(signals, weights):
        mixed += signal_data * weight
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(mixed))
    if max_amplitude > 1.0:
        mixed = mixed / max_amplitude
    
    return mixed

# Create chord (multiple frequencies)
frequencies = [440, 554.37, 659.25]  # A major chord (A, C#, E)
chord_signals = [np.sin(2 * np.pi * freq * t) for freq in frequencies]
chord = mix_audio_signals(*chord_signals, weights=[0.4, 0.3, 0.3])

print(f"\nChord mixing:")
print(f"Individual frequencies: {frequencies} Hz")
print(f"Mixed chord amplitude range: [{np.min(chord):.3f}, {np.max(chord):.3f}]")

# Real-time audio processing simulation
def sliding_window_rms(audio, window_size):
    """Calculate RMS energy in sliding windows"""
    rms_values = []
    for i in range(0, len(audio) - window_size + 1, window_size // 4):
        window = audio[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)
    return np.array(rms_values)

# Calculate RMS energy for volume detection
window_size = 1024  # ~23ms at 44.1kHz
rms_values = sliding_window_rms(sine_wave, window_size)

print(f"\nAudio energy analysis:")
print(f"RMS window size: {window_size} samples ({window_size/sample_rate*1000:.1f} ms)")
print(f"Number of RMS values: {len(rms_values)}")
print(f"Average RMS energy: {np.mean(rms_values):.6f}")

# Save audio to file (example)
def save_audio_example():
    """Example of saving audio to WAV file"""
    # Normalize audio to 16-bit integer range
    audio_int16 = (sine_wave * 32767).astype(np.int16)
    
    # Save to WAV file
    wav.write('example_audio.wav', sample_rate, audio_int16)
    print(f"Audio saved to example_audio.wav")

# Cleanup function would go here
print(f"\nAudio processing complete!")
```

### Use Cases
- Music production and audio effects
- Speech processing and recognition
- Acoustic analysis and measurement
- Digital signal processing research
- Real-time audio applications

---

## Question 8

**How do you handle NaN or infinite values in a NumPy array?**

### Code Example

```python
import numpy as np

# Create array with problematic values
data = np.array([1, 2, np.nan, 4, np.inf, 6, -np.inf, 8, np.nan, 10])
matrix = np.array([[1, 2, np.nan], [4, np.inf, 6], [7, 8, -np.inf]])

print(f"Original array: {data}")
print(f"Original matrix:\n{matrix}")

# Detection functions
print(f"\nDetection:")
print(f"Has NaN: {np.isnan(data).any()}")
print(f"Has infinite: {np.isinf(data).any()}")
print(f"Is finite: {np.isfinite(data)}")

# Count problematic values
nan_count = np.sum(np.isnan(data))
inf_count = np.sum(np.isinf(data))
finite_count = np.sum(np.isfinite(data))

print(f"NaN count: {nan_count}")
print(f"Infinite count: {inf_count}")
print(f"Finite count: {finite_count}")

# Removal strategies
clean_data = data[np.isfinite(data)]
print(f"\nClean data (finite only): {clean_data}")

# Replacement strategies
data_replaced = data.copy()
data_replaced[np.isnan(data_replaced)] = 0  # Replace NaN with 0
data_replaced[np.isinf(data_replaced)] = np.nanmax(data)  # Replace inf with max finite value
print(f"Data with replacements: {data_replaced}")

# Using numpy's nan-aware functions
print(f"\nNaN-aware computations:")
print(f"Mean (ignoring NaN): {np.nanmean(data)}")
print(f"Sum (ignoring NaN): {np.nansum(data)}")
print(f"Standard deviation (ignoring NaN): {np.nanstd(data)}")
print(f"Min (ignoring NaN): {np.nanmin(data)}")
print(f"Max (ignoring NaN): {np.nanmax(data)}")

# Advanced cleaning function
def clean_array(arr, strategy='remove', fill_value=0):
    """Clean array by handling NaN and infinite values"""
    if strategy == 'remove':
        return arr[np.isfinite(arr)]
    elif strategy == 'fill':
        result = arr.copy()
        result[~np.isfinite(result)] = fill_value
        return result
    elif strategy == 'interpolate':
        # Simple linear interpolation for 1D arrays
        result = arr.copy()
        mask = np.isfinite(result)
        if np.any(mask):
            indices = np.arange(len(result))
            result[~mask] = np.interp(indices[~mask], indices[mask], result[mask])
        return result

cleaned_remove = clean_array(data, 'remove')
cleaned_fill = clean_array(data, 'fill', fill_value=-999)
cleaned_interp = clean_array(data, 'interpolate')

print(f"\nCleaning strategies:")
print(f"Remove: {cleaned_remove}")
print(f"Fill with -999: {cleaned_fill}")
print(f"Interpolate: {cleaned_interp}")
```

### Best Practices
- Always check for NaN/inf values in input data
- Use appropriate nan-aware functions for computations
- Choose replacement strategy based on domain knowledge
- Document how missing values are handled

---

## Question 9

**What methods are there in NumPy to deal with missing data?**

### Code Example

```python
import numpy as np

# Create dataset with missing values
data = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0, 7.0, np.nan, 9.0, 10.0])
matrix = np.array([[1, 2, np.nan], [np.nan, 5, 6], [7, np.nan, 9]])

print(f"Data with missing values: {data}")
print(f"Matrix with missing values:\n{matrix}")

# Method 1: Detection
missing_mask = np.isnan(data)
print(f"\nMissing value mask: {missing_mask}")
print(f"Indices of missing values: {np.where(missing_mask)[0]}")

# Method 2: Removal (listwise deletion)
complete_cases = data[~np.isnan(data)]
print(f"Complete cases only: {complete_cases}")

# Method 3: NaN-aware statistical functions
print(f"\nNaN-aware statistics:")
print(f"Count of non-NaN values: {np.count_nonzero(~np.isnan(data))}")
print(f"Mean: {np.nanmean(data)}")
print(f"Median: {np.nanmedian(data)}")
print(f"Standard deviation: {np.nanstd(data)}")
print(f"Percentiles: {np.nanpercentile(data, [25, 50, 75])}")

# Method 4: Simple imputation
def simple_imputation(arr, method='mean'):
    """Simple imputation strategies"""
    result = arr.copy()
    mask = np.isnan(result)
    
    if method == 'mean':
        fill_value = np.nanmean(arr)
    elif method == 'median':
        fill_value = np.nanmedian(arr)
    elif method == 'mode':
        # For mode, we'll use the most frequent value
        finite_values = arr[~mask]
        unique, counts = np.unique(finite_values, return_counts=True)
        fill_value = unique[np.argmax(counts)]
    elif method == 'zero':
        fill_value = 0
    elif method == 'forward_fill':
        # Forward fill (carry last observation forward)
        result = forward_fill_1d(arr)
        return result
    elif method == 'backward_fill':
        # Backward fill
        result = backward_fill_1d(arr)
        return result
    else:
        raise ValueError(f"Unknown method: {method}")
    
    result[mask] = fill_value
    return result

def forward_fill_1d(arr):
    """Forward fill missing values"""
    result = arr.copy()
    mask = np.isnan(result)
    
    # Forward fill
    for i in range(1, len(result)):
        if mask[i] and not mask[i-1]:
            result[i] = result[i-1]
    
    return result

def backward_fill_1d(arr):
    """Backward fill missing values"""
    result = arr.copy()
    mask = np.isnan(result)
    
    # Backward fill
    for i in range(len(result)-2, -1, -1):
        if mask[i] and not mask[i+1]:
            result[i] = result[i+1]
    
    return result

# Test different imputation methods
imputation_methods = ['mean', 'median', 'zero', 'forward_fill', 'backward_fill']

print(f"\nImputation methods:")
for method in imputation_methods:
    imputed = simple_imputation(data, method)
    print(f"{method:12}: {imputed}")

# Method 5: Linear interpolation
def linear_interpolation_1d(arr):
    """Linear interpolation for missing values"""
    result = arr.copy()
    mask = np.isnan(result)
    
    if np.all(mask):
        return result  # All values are NaN
    
    # Get indices of finite values
    finite_indices = np.where(~mask)[0]
    finite_values = result[finite_indices]
    
    # Interpolate missing values
    all_indices = np.arange(len(result))
    interpolated = np.interp(all_indices, finite_indices, finite_values)
    
    result[mask] = interpolated[mask]
    return result

interpolated_data = linear_interpolation_1d(data)
print(f"\nLinear interpolation: {interpolated_data}")

# Method 6: Advanced imputation using neighboring values
def knn_imputation_1d(arr, k=2):
    """Simple K-nearest neighbors imputation for 1D array"""
    result = arr.copy()
    mask = np.isnan(result)
    
    for i in np.where(mask)[0]:
        # Find k nearest non-missing neighbors
        distances = []
        for j in range(len(arr)):
            if not mask[j]:  # Non-missing value
                distance = abs(i - j)
                distances.append((distance, arr[j]))
        
        # Sort by distance and take k nearest
        distances.sort()
        nearest_values = [val for _, val in distances[:k]]
        
        # Impute with mean of nearest neighbors
        if nearest_values:
            result[i] = np.mean(nearest_values)
    
    return result

knn_imputed = knn_imputation_1d(data, k=2)
print(f"KNN imputation (k=2): {knn_imputed}")

# Method 7: Multiple imputation simulation
def multiple_imputation_simple(arr, n_imputations=5):
    """Simple multiple imputation using random sampling"""
    finite_values = arr[~np.isnan(arr)]
    if len(finite_values) == 0:
        return [arr] * n_imputations
    
    imputations = []
    for _ in range(n_imputations):
        result = arr.copy()
        mask = np.isnan(result)
        
        # Random imputation from finite values
        n_missing = np.sum(mask)
        random_values = np.random.choice(finite_values, n_missing, replace=True)
        result[mask] = random_values
        
        imputations.append(result)
    
    return imputations

# Generate multiple imputations
multiple_imputations = multiple_imputation_simple(data, n_imputations=3)
print(f"\nMultiple imputations:")
for i, imp in enumerate(multiple_imputations):
    print(f"Imputation {i+1}: {imp}")

# Combine results from multiple imputations
combined_result = np.mean(multiple_imputations, axis=0)
print(f"Combined result: {combined_result}")

# Method 8: Handling missing data in 2D arrays
def impute_2d_matrix(matrix, method='mean', axis=None):
    """Impute missing values in 2D matrix"""
    result = matrix.copy()
    
    if method == 'mean':
        if axis is None:
            fill_value = np.nanmean(matrix)
            result[np.isnan(result)] = fill_value
        elif axis == 0:  # Column-wise imputation
            for col in range(matrix.shape[1]):
                col_mean = np.nanmean(matrix[:, col])
                result[np.isnan(result[:, col]), col] = col_mean
        elif axis == 1:  # Row-wise imputation
            for row in range(matrix.shape[0]):
                row_mean = np.nanmean(matrix[row, :])
                result[row, np.isnan(result[row, :])] = row_mean
    
    return result

# Test 2D imputation
print(f"\n2D Matrix imputation:")
print(f"Original matrix:\n{matrix}")
print(f"Overall mean imputation:\n{impute_2d_matrix(matrix, 'mean')}")
print(f"Column-wise mean imputation:\n{impute_2d_matrix(matrix, 'mean', axis=0)}")
print(f"Row-wise mean imputation:\n{impute_2d_matrix(matrix, 'mean', axis=1)}")
```

### Missing Data Strategies
1. **Complete Case Analysis**: Remove all rows with missing values
2. **Mean/Median Imputation**: Replace with average values
3. **Forward/Backward Fill**: Use adjacent values
4. **Interpolation**: Linear or polynomial interpolation
5. **Multiple Imputation**: Generate multiple complete datasets
6. **Model-based Imputation**: Use statistical models to predict missing values

---

## Question 10

**How do you find unique values and their counts in a NumPy array?**

### Code Example

```python
import numpy as np

# Sample data
data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])
text_data = np.array(['apple', 'banana', 'apple', 'cherry', 'banana', 'apple'])

print(f"Numeric data: {data}")
print(f"Text data: {text_data}")

# Basic unique values
unique_values = np.unique(data)
print(f"\nUnique values: {unique_values}")

# Unique values with counts
unique_vals, counts = np.unique(data, return_counts=True)
print(f"Unique values: {unique_vals}")
print(f"Counts: {counts}")

# Create frequency table
print(f"\nFrequency table:")
for val, count in zip(unique_vals, counts):
    print(f"Value {val}: {count} occurrences")

# Unique with additional information
unique_vals, indices, inverse, counts = np.unique(data, return_index=True, 
                                                 return_inverse=True, 
                                                 return_counts=True)

print(f"\nDetailed unique analysis:")
print(f"Unique values: {unique_vals}")
print(f"First occurrence indices: {indices}")
print(f"Inverse mapping: {inverse}")
print(f"Counts: {counts}")

# For 2D arrays
matrix = np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3]])
print(f"\n2D Matrix:\n{matrix}")

# Unique elements (flattened)
unique_2d = np.unique(matrix)
print(f"Unique elements in 2D array: {unique_2d}")

# Unique rows
unique_rows = np.unique(matrix, axis=0)
print(f"Unique rows:\n{unique_rows}")

# Most/least frequent values
most_frequent_idx = np.argmax(counts)
least_frequent_idx = np.argmin(counts)

print(f"\nFrequency analysis:")
print(f"Most frequent value: {unique_vals[most_frequent_idx]} (appears {counts[most_frequent_idx]} times)")
print(f"Least frequent value: {unique_vals[least_frequent_idx]} (appears {counts[least_frequent_idx]} times)")

# Percentage distribution
percentages = counts / len(data) * 100
print(f"\nPercentage distribution:")
for val, pct in zip(unique_vals, percentages):
    print(f"Value {val}: {pct:.1f}%")

# Working with text data
text_unique, text_counts = np.unique(text_data, return_counts=True)
print(f"\nText data analysis:")
print(f"Unique fruits: {text_unique}")
print(f"Counts: {text_counts}")

# Custom counting function
def value_counts(arr):
    """Custom function similar to pandas value_counts"""
    unique_vals, counts = np.unique(arr, return_counts=True)
    
    # Sort by count (descending)
    sort_indices = np.argsort(counts)[::-1]
    
    return unique_vals[sort_indices], counts[sort_indices]

sorted_vals, sorted_counts = value_counts(data)
print(f"\nSorted by frequency:")
for val, count in zip(sorted_vals, sorted_counts):
    print(f"{val}: {count}")
```

### Use Cases
- Data exploration and frequency analysis
- Categorical data analysis
- Duplicate detection and removal
- Statistical distribution analysis
- Data quality assessment

---

## Question 11

**How can you use NumPy arrays with Cython for performance optimization?**

### Theory

Cython allows writing C extensions for Python using Python-like syntax. When combined with NumPy arrays, it can provide significant performance improvements for numerical computations by eliminating Python overhead and enabling optimized memory access patterns.

### Code Example

```python
# This would typically be in a .pyx file for Cython compilation
"""
# cython_numpy_example.pyx

import numpy as np
cimport numpy as cnp
cimport cython

# Define numpy array types
ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_sum_array(cnp.ndarray[DTYPE_t, ndim=1] arr):
    '''Fast summation using Cython'''
    cdef int i
    cdef int n = arr.shape[0]
    cdef DTYPE_t total = 0.0
    
    for i in range(n):
        total += arr[i]
    
    return total

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_matrix_multiply(cnp.ndarray[DTYPE_t, ndim=2] A,
                          cnp.ndarray[DTYPE_t, ndim=2] B):
    '''Matrix multiplication using Cython'''
    cdef int i, j, k
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int p = B.shape[1]
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] C = np.zeros((m, p), dtype=np.float64)
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_moving_average(cnp.ndarray[DTYPE_t, ndim=1] arr, int window_size):
    '''Moving average calculation using Cython'''
    cdef int i, j
    cdef int n = arr.shape[0]
    cdef int out_size = n - window_size + 1
    cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.zeros(out_size, dtype=np.float64)
    cdef DTYPE_t window_sum
    
    for i in range(out_size):
        window_sum = 0.0
        for j in range(window_size):
            window_sum += arr[i + j]
        result[i] = window_sum / window_size
    
    return result
"""

# Python comparison and demonstration
import numpy as np

# Pure Python implementations for comparison
def python_sum_array(arr):
    """Pure Python array summation"""
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total

def python_matrix_multiply(A, B):
    """Pure Python matrix multiplication"""
    m, n = A.shape
    n2, p = B.shape
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

def python_moving_average(arr, window_size):
    """Pure Python moving average"""
    result = []
    for i in range(len(arr) - window_size + 1):
        window_sum = sum(arr[i:i+window_size])
        result.append(window_sum / window_size)
    return np.array(result)

# Performance comparison
def compare_performance():
    """Compare Python, NumPy, and Cython performance"""
    import time
    
    # Create test data
    large_array = np.random.randn(100000).astype(np.float64)
    matrix_a = np.random.randn(200, 200).astype(np.float64)
    matrix_b = np.random.randn(200, 200).astype(np.float64)
    
    print("Performance Comparison Results:")
    print("=" * 50)
    
    # Array summation comparison
    print("\nArray Summation (100,000 elements):")
    
    # Pure Python
    start = time.time()
    python_result = python_sum_array(large_array)
    python_time = time.time() - start
    
    # NumPy
    start = time.time()
    numpy_result = np.sum(large_array)
    numpy_time = time.time() - start
    
    print(f"Pure Python: {python_time:.6f}s (result: {python_result:.6f})")
    print(f"NumPy:       {numpy_time:.6f}s (result: {numpy_result:.6f})")
    print(f"NumPy speedup: {python_time/numpy_time:.1f}x")
    
    # Matrix multiplication comparison
    print(f"\nMatrix Multiplication (200x200):")
    
    # Pure Python
    start = time.time()
    python_mat_result = python_matrix_multiply(matrix_a, matrix_b)
    python_mat_time = time.time() - start
    
    # NumPy
    start = time.time()
    numpy_mat_result = np.dot(matrix_a, matrix_b)
    numpy_mat_time = time.time() - start
    
    print(f"Pure Python: {python_mat_time:.6f}s")
    print(f"NumPy:       {numpy_mat_time:.6f}s")
    print(f"NumPy speedup: {python_mat_time/numpy_mat_time:.1f}x")
    
    # Moving average comparison
    test_array = np.random.randn(10000).astype(np.float64)
    window_size = 100
    
    print(f"\nMoving Average (10,000 elements, window=100):")
    
    # Pure Python
    start = time.time()
    python_ma_result = python_moving_average(test_array, window_size)
    python_ma_time = time.time() - start
    
    # NumPy (using convolution)
    start = time.time()
    numpy_ma_result = np.convolve(test_array, np.ones(window_size)/window_size, mode='valid')
    numpy_ma_time = time.time() - start
    
    print(f"Pure Python: {python_ma_time:.6f}s")
    print(f"NumPy:       {numpy_ma_time:.6f}s")
    print(f"NumPy speedup: {python_ma_time/numpy_ma_time:.1f}x")

# Cython optimization tips and best practices
def cython_optimization_tips():
    """Guidelines for optimizing NumPy code with Cython"""
    tips = [
        "1. Use typed memoryviews for efficient array access",
        "2. Disable bounds checking with @cython.boundscheck(False)",
        "3. Disable negative indexing with @cython.wraparound(False)",
        "4. Use cdef for variables and function parameters when possible",
        "5. Specify array types explicitly (e.g., cnp.ndarray[DTYPE_t, ndim=2])",
        "6. Use C data types (int, double) instead of Python objects",
        "7. Avoid Python function calls in tight loops",
        "8. Use inline functions for small, frequently called operations",
        "9. Profile code to identify bottlenecks before optimization",
        "10. Consider using parallel processing with OpenMP"
    ]
    
    print("Cython Optimization Tips:")
    print("=" * 40)
    for tip in tips:
        print(tip)

# Example of setup.py for compiling Cython code
setup_py_content = '''
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("cython_numpy_example.pyx"),
    include_dirs=[numpy.get_include()]
)
'''

print("Example setup.py for Cython compilation:")
print("=" * 45)
print(setup_py_content)

print("\nCompilation command:")
print("python setup.py build_ext --inplace")

# Run performance comparison
compare_performance()

# Show optimization tips
print("\n")
cython_optimization_tips()

# Example of memory-efficient array processing
def memory_efficient_processing():
    """Example of memory-efficient array processing techniques"""
    print(f"\nMemory-Efficient Processing Techniques:")
    print("=" * 45)
    
    # Use views instead of copies
    large_array = np.random.randn(1000000)
    
    # Good: Use view (no memory copy)
    subset_view = large_array[::2]  # Every other element
    print(f"View shares memory: {np.shares_memory(large_array, subset_view)}")
    
    # In-place operations
    print(f"Original array mean: {np.mean(large_array):.6f}")
    large_array *= 2  # In-place multiplication
    print(f"After in-place doubling: {np.mean(large_array):.6f}")
    
    # Memory mapping for large files
    # This would save/load large arrays without loading into memory
    temp_filename = 'temp_large_array.npy'
    np.save(temp_filename, large_array)
    
    # Memory-mapped access
    mmap_array = np.load(temp_filename, mmap_mode='r')
    print(f"Memory-mapped array shape: {mmap_array.shape}")
    print(f"Memory-mapped array type: {type(mmap_array)}")
    
    # Cleanup
    import os
    os.remove(temp_filename)

memory_efficient_processing()
```

### Key Benefits of Cython + NumPy
1. **Speed**: 10-1000x performance improvements for numerical code
2. **Memory Efficiency**: Direct access to array data without Python overhead
3. **Compatibility**: Seamless integration with existing NumPy code
4. **Gradual Optimization**: Can optimize only critical sections
5. **Type Safety**: Static typing catches errors at compile time

### Use Cases
- Scientific computing with tight loops
- Image and signal processing algorithms
- Machine learning model implementations
- Real-time data processing systems
- High-performance numerical libraries

---

