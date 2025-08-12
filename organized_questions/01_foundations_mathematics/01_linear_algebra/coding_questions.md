# Linear Algebra Interview Questions - Coding Questions

## Question 1

**Write code to add, subtract, and multiply two matrices without using external libraries.**

**Answer:** Here's a comprehensive implementation of basic matrix operations from scratch:

```python
def matrix_add(A, B):
    """
    Add two matrices A and B.
    
    Args:
        A, B: Lists of lists representing matrices
    
    Returns:
        Result matrix as list of lists
    
    Raises:
        ValueError: If matrices have incompatible dimensions
    """
    # Check dimensions
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have same dimensions for addition")
    
    rows, cols = len(A), len(A[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] + B[i][j]
    
    return result


def matrix_subtract(A, B):
    """
    Subtract matrix B from matrix A.
    
    Args:
        A, B: Lists of lists representing matrices
    
    Returns:
        Result matrix as list of lists
    """
    # Check dimensions
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have same dimensions for subtraction")
    
    rows, cols = len(A), len(A[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] - B[i][j]
    
    return result


def matrix_multiply(A, B):
    """
    Multiply two matrices A and B.
    
    Args:
        A: Matrix of size (m x n)
        B: Matrix of size (n x p)
    
    Returns:
        Result matrix of size (m x p)
    
    Raises:
        ValueError: If matrices have incompatible dimensions
    """
    # Check compatibility: A cols must equal B rows
    if len(A[0]) != len(B):
        raise ValueError(f"Cannot multiply {len(A)}x{len(A[0])} with {len(B)}x{len(B[0])}")
    
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    # Initialize result matrix
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    # Perform multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result


def print_matrix(matrix, title="Matrix"):
    """Helper function to print matrices nicely"""
    print(f"\n{title}:")
    for row in matrix:
        print([f"{val:8.3f}" if isinstance(val, float) else f"{val:8}" for val in row])


# Example usage and testing
if __name__ == "__main__":
    # Test matrices
    A = [[1, 2, 3],
         [4, 5, 6]]
    
    B = [[7, 8, 9],
         [10, 11, 12]]
    
    C = [[1, 2],
         [3, 4],
         [5, 6]]
    
    # Test addition
    try:
        result_add = matrix_add(A, B)
        print_matrix(A, "Matrix A")
        print_matrix(B, "Matrix B")
        print_matrix(result_add, "A + B")
    except ValueError as e:
        print(f"Addition error: {e}")
    
    # Test subtraction
    try:
        result_sub = matrix_subtract(A, B)
        print_matrix(result_sub, "A - B")
    except ValueError as e:
        print(f"Subtraction error: {e}")
    
    # Test multiplication
    try:
        result_mult = matrix_multiply(A, C)
        print_matrix(C, "Matrix C")
        print_matrix(result_mult, "A × C")
    except ValueError as e:
        print(f"Multiplication error: {e}")
    
    # Test invalid operations
    try:
        invalid = matrix_multiply(A, B)  # Should fail
    except ValueError as e:
        print(f"\nExpected error: {e}")


# Advanced version with optimizations
class Matrix:
    """
    A more advanced matrix class with operator overloading
    """
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions")
        
        result = [[self.data[i][j] + other.data[i][j] 
                  for j in range(self.cols)] 
                 for i in range(self.rows)]
        return Matrix(result)
    
    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions")
        
        result = [[self.data[i][j] - other.data[i][j] 
                  for j in range(self.cols)] 
                 for i in range(self.rows)]
        return Matrix(result)
    
    def __matmul__(self, other):
        """Matrix multiplication using @ operator"""
        if self.cols != other.rows:
            raise ValueError("Incompatible dimensions for multiplication")
        
        result = [[sum(self.data[i][k] * other.data[k][j] 
                      for k in range(self.cols))
                  for j in range(other.cols)]
                 for i in range(self.rows)]
        return Matrix(result)
    
    def __str__(self):
        return '\n'.join([' '.join(f"{val:8.3f}" for val in row) 
                         for row in self.data])
```

**Key Features:**
- **Error Handling**: Comprehensive dimension checking
- **Performance**: O(n³) for multiplication, O(n²) for addition/subtraction
- **Memory Efficiency**: Minimal space overhead
- **Clean API**: Both functional and object-oriented approaches
- **Extensibility**: Easy to add more operations

**Time Complexities:**
- Addition/Subtraction: O(m×n)
- Multiplication: O(m×n×p) where A is m×n and B is n×p

**Use Cases:**
- Educational purposes to understand matrix operations
- Embedded systems where external libraries aren't available
- Custom optimization where you need full control over operations

---

## Question 2

**Implement a function to calculate the transpose of a given matrix.**

**Answer:** Here are multiple implementations for matrix transpose with different approaches:

```python
def matrix_transpose_basic(matrix):
    """
    Basic implementation of matrix transpose.
    
    Args:
        matrix: List of lists representing the matrix
    
    Returns:
        Transposed matrix as list of lists
    """
    if not matrix or not matrix[0]:
        return []
    
    rows, cols = len(matrix), len(matrix[0])
    
    # Create transposed matrix: swap rows and columns
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    
    return transposed


def matrix_transpose_comprehension(matrix):
    """
    Pythonic implementation using list comprehension.
    
    Args:
        matrix: List of lists representing the matrix
    
    Returns:
        Transposed matrix
    """
    if not matrix or not matrix[0]:
        return []
    
    return [[matrix[i][j] for i in range(len(matrix))] 
            for j in range(len(matrix[0]))]


def matrix_transpose_zip(matrix):
    """
    Elegant implementation using zip function.
    
    Args:
        matrix: List of lists representing the matrix
    
    Returns:
        Transposed matrix
    """
    if not matrix:
        return []
    
    return [list(row) for row in zip(*matrix)]


def matrix_transpose_inplace(matrix):
    """
    In-place transpose for square matrices only.
    
    Args:
        matrix: Square matrix (n x n) as list of lists
    
    Returns:
        None (modifies matrix in place)
    
    Raises:
        ValueError: If matrix is not square
    """
    n = len(matrix)
    
    # Check if matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("In-place transpose only works for square matrices")
    
    for i in range(n):
        for j in range(i + 1, n):
            # Swap elements across the diagonal
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


class MatrixTransposer:
    """
    Advanced matrix transpose class with additional features.
    """
    
    @staticmethod
    def transpose(matrix):
        """Standard transpose operation"""
        return MatrixTransposer._validate_and_transpose(matrix)
    
    @staticmethod
    def conjugate_transpose(matrix):
        """
        Conjugate transpose (Hermitian transpose) for complex matrices.
        For real matrices, this is the same as regular transpose.
        """
        if not matrix:
            return []
        
        transposed = MatrixTransposer._validate_and_transpose(matrix)
        
        # Apply complex conjugate if needed
        for i in range(len(transposed)):
            for j in range(len(transposed[0])):
                if isinstance(transposed[i][j], complex):
                    transposed[i][j] = transposed[i][j].conjugate()
        
        return transposed
    
    @staticmethod
    def _validate_and_transpose(matrix):
        """Helper method with validation"""
        if not matrix:
            return []
        
        if not all(len(row) == len(matrix[0]) for row in matrix):
            raise ValueError("All rows must have the same length")
        
        return [[matrix[i][j] for i in range(len(matrix))] 
                for j in range(len(matrix[0]))]


# Performance comparison and testing
def benchmark_transpose_methods():
    """Compare performance of different transpose methods"""
    import time
    
    # Create test matrix
    n = 1000
    large_matrix = [[i * n + j for j in range(n)] for i in range(n)]
    
    methods = [
        ("Basic Implementation", matrix_transpose_basic),
        ("List Comprehension", matrix_transpose_comprehension),
        ("Zip Method", matrix_transpose_zip),
    ]
    
    for name, func in methods:
        start_time = time.time()
        result = func(large_matrix)
        end_time = time.time()
        print(f"{name}: {end_time - start_time:.4f} seconds")


# Example usage and demonstrations
if __name__ == "__main__":
    # Test matrix
    test_matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    
    print("Original Matrix:")
    for row in test_matrix:
        print(row)
    
    # Test different methods
    methods = [
        ("Basic", matrix_transpose_basic),
        ("Comprehension", matrix_transpose_comprehension),
        ("Zip", matrix_transpose_zip),
        ("Class Method", MatrixTransposer.transpose)
    ]
    
    for name, method in methods:
        result = method(test_matrix)
        print(f"\nTranspose using {name}:")
        for row in result:
            print(row)
    
    # Test in-place transpose for square matrix
    square_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    print(f"\nOriginal square matrix:")
    for row in square_matrix:
        print(row)
    
    matrix_transpose_inplace(square_matrix)
    print(f"\nAfter in-place transpose:")
    for row in square_matrix:
        print(row)
    
    # Test with complex numbers
    complex_matrix = [
        [1+2j, 3+4j],
        [5+6j, 7+8j]
    ]
    
    print(f"\nComplex matrix:")
    for row in complex_matrix:
        print(row)
    
    conjugate_transposed = MatrixTransposer.conjugate_transpose(complex_matrix)
    print(f"\nConjugate transpose:")
    for row in conjugate_transposed:
        print(row)


# NumPy comparison for verification
def verify_with_numpy():
    """Verify results against NumPy implementation"""
    try:
        import numpy as np
        
        matrix = [[1, 2, 3], [4, 5, 6]]
        
        # Our implementation
        our_result = matrix_transpose_zip(matrix)
        
        # NumPy implementation
        numpy_result = np.array(matrix).T.tolist()
        
        print("Our result:", our_result)
        print("NumPy result:", numpy_result)
        print("Results match:", our_result == numpy_result)
        
    except ImportError:
        print("NumPy not available for verification")


# Special case handlers
def transpose_jagged_array(jagged_matrix):
    """
    Handle transpose of jagged arrays (different row lengths).
    Missing elements are filled with None.
    """
    if not jagged_matrix:
        return []
    
    max_cols = max(len(row) for row in jagged_matrix)
    
    result = []
    for j in range(max_cols):
        column = []
        for i in range(len(jagged_matrix)):
            if j < len(jagged_matrix[i]):
                column.append(jagged_matrix[i][j])
            else:
                column.append(None)
        result.append(column)
    
    return result
```

**Key Features:**
- **Multiple Approaches**: Basic, comprehension, zip, and in-place methods
- **Performance Optimized**: Different methods for different use cases
- **Complex Number Support**: Handles conjugate transpose
- **Error Handling**: Comprehensive validation
- **Memory Efficiency**: In-place option for square matrices

**Time Complexity**: O(m×n) where m and n are matrix dimensions
**Space Complexity**: O(m×n) for new matrix, O(1) for in-place

**Best Practices:**
- Use `zip(*matrix)` for clean, readable code
- Use in-place transpose for memory-constrained environments
- Consider conjugate transpose for complex matrices
- Validate input dimensions before processing

---

## Question 3

**Code to find the determinant of a matrix using recursion.**

**Answer:** Here's a comprehensive implementation of determinant calculation using multiple approaches:

```python
def determinant_recursive(matrix):
    """
    Calculate determinant using recursive cofactor expansion.
    
    Args:
        matrix: Square matrix as list of lists
    
    Returns:
        Determinant value as float
    
    Raises:
        ValueError: If matrix is not square or empty
    """
    # Validate input
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")
    
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")
    
    # Base cases
    if n == 1:
        return matrix[0][0]
    
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Recursive case: cofactor expansion along first row
    det = 0
    for j in range(n):
        # Calculate cofactor
        cofactor = (-1) ** j * matrix[0][j]
        
        # Create minor matrix (remove row 0 and column j)
        minor = get_minor_matrix(matrix, 0, j)
        
        # Recursive call
        det += cofactor * determinant_recursive(minor)
    
    return det


def get_minor_matrix(matrix, row_to_remove, col_to_remove):
    """
    Create minor matrix by removing specified row and column.
    
    Args:
        matrix: Original matrix
        row_to_remove: Row index to remove
        col_to_remove: Column index to remove
    
    Returns:
        Minor matrix
    """
    n = len(matrix)
    minor = []
    
    for i in range(n):
        if i == row_to_remove:
            continue
        
        row = []
        for j in range(n):
            if j == col_to_remove:
                continue
            row.append(matrix[i][j])
        
        minor.append(row)
    
    return minor


def determinant_iterative(matrix):
    """
    Calculate determinant using Gaussian elimination (more efficient).
    
    Args:
        matrix: Square matrix as list of lists
    
    Returns:
        Determinant value
    """
    if not matrix or len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square and non-empty")
    
    n = len(matrix)
    
    # Create a copy to avoid modifying original
    mat = [row[:] for row in matrix]
    
    det = 1
    
    for i in range(n):
        # Find pivot (partial pivoting for numerical stability)
        max_row = i
        for k in range(i + 1, n):
            if abs(mat[k][i]) > abs(mat[max_row][i]):
                max_row = k
        
        # Swap rows if needed
        if max_row != i:
            mat[i], mat[max_row] = mat[max_row], mat[i]
            det *= -1  # Row swap changes sign
        
        # Check for zero pivot (singular matrix)
        if abs(mat[i][i]) < 1e-10:
            return 0
        
        # Update determinant with diagonal element
        det *= mat[i][i]
        
        # Eliminate below pivot
        for k in range(i + 1, n):
            if mat[i][i] != 0:
                factor = mat[k][i] / mat[i][i]
                for j in range(i, n):
                    mat[k][j] -= factor * mat[i][j]
    
    return det


def determinant_optimized_recursive(matrix, memo=None):
    """
    Optimized recursive determinant with memoization.
    
    Args:
        matrix: Square matrix
        memo: Memoization dictionary
    
    Returns:
        Determinant value
    """
    if memo is None:
        memo = {}
    
    # Create hashable key for memoization
    key = tuple(tuple(row) for row in matrix)
    if key in memo:
        return memo[key]
    
    n = len(matrix)
    
    # Base cases
    if n == 1:
        result = matrix[0][0]
    elif n == 2:
        result = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        # Find best row/column for expansion (most zeros)
        best_row, best_col, best_zeros = find_best_expansion_line(matrix)
        
        result = 0
        if best_row is not None:
            # Expand along row
            for j in range(n):
                if matrix[best_row][j] != 0:
                    cofactor = (-1) ** (best_row + j) * matrix[best_row][j]
                    minor = get_minor_matrix(matrix, best_row, j)
                    result += cofactor * determinant_optimized_recursive(minor, memo)
        else:
            # Expand along column
            for i in range(n):
                if matrix[i][best_col] != 0:
                    cofactor = (-1) ** (i + best_col) * matrix[i][best_col]
                    minor = get_minor_matrix(matrix, i, best_col)
                    result += cofactor * determinant_optimized_recursive(minor, memo)
    
    memo[key] = result
    return result


def find_best_expansion_line(matrix):
    """
    Find the row or column with the most zeros for efficient expansion.
    
    Returns:
        (best_row, best_col, max_zeros)
    """
    n = len(matrix)
    max_zeros = -1
    best_row, best_col = None, None
    
    # Check rows
    for i in range(n):
        zeros = sum(1 for val in matrix[i] if val == 0)
        if zeros > max_zeros:
            max_zeros = zeros
            best_row, best_col = i, None
    
    # Check columns
    for j in range(n):
        zeros = sum(1 for i in range(n) if matrix[i][j] == 0)
        if zeros > max_zeros:
            max_zeros = zeros
            best_row, best_col = None, j
    
    return best_row, best_col, max_zeros


def determinant_laplace_expansion(matrix, row=0):
    """
    Determinant using Laplace expansion along specified row.
    
    Args:
        matrix: Square matrix
        row: Row to expand along (default: 0)
    
    Returns:
        Determinant value
    """
    n = len(matrix)
    
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for j in range(n):
        if matrix[row][j] != 0:  # Skip zero elements for efficiency
            cofactor = (-1) ** (row + j) * matrix[row][j]
            minor = get_minor_matrix(matrix, row, j)
            det += cofactor * determinant_laplace_expansion(minor, 0)
    
    return det


# Performance testing and comparison
def benchmark_determinant_methods():
    """Compare performance of different determinant methods"""
    import time
    import random
    
    # Generate test matrices of different sizes
    sizes = [3, 4, 5, 6]
    
    for n in sizes:
        # Create random matrix
        matrix = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
        
        print(f"\nMatrix size: {n}x{n}")
        
        methods = [
            ("Recursive", determinant_recursive),
            ("Iterative (Gaussian)", determinant_iterative),
            ("Optimized Recursive", determinant_optimized_recursive),
        ]
        
        results = []
        for name, method in methods:
            start_time = time.time()
            try:
                result = method(matrix)
                end_time = time.time()
                results.append(result)
                print(f"{name:20}: {end_time - start_time:.6f}s, det = {result:.6f}")
            except Exception as e:
                print(f"{name:20}: Error - {e}")
        
        # Check if results are consistent
        if len(set(f"{r:.6f}" for r in results)) <= 1:
            print("✓ All methods agree on result")
        else:
            print("⚠ Methods disagree on result")


# Example usage and testing
if __name__ == "__main__":
    # Test matrices
    test_matrices = [
        # 2x2 matrix
        [[2, 3],
         [1, 4]],
        
        # 3x3 matrix
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        
        # 3x3 non-singular matrix
        [[2, -3, 1],
         [2, 0, -1],
         [1, 4, 5]],
        
        # 4x4 matrix with some zeros
        [[1, 0, 2, -1],
         [3, 0, 0, 5],
         [2, 1, 4, -3],
         [1, 0, 5, 0]]
    ]
    
    for i, matrix in enumerate(test_matrices):
        print(f"\n--- Test Matrix {i+1} ---")
        print("Matrix:")
        for row in matrix:
            print(row)
        
        try:
            det_recursive = determinant_recursive(matrix)
            det_iterative = determinant_iterative(matrix)
            det_optimized = determinant_optimized_recursive(matrix)
            
            print(f"Determinant (recursive): {det_recursive:.6f}")
            print(f"Determinant (iterative): {det_iterative:.6f}")
            print(f"Determinant (optimized): {det_optimized:.6f}")
            
            # Check if singular
            if abs(det_recursive) < 1e-10:
                print("Matrix is singular (non-invertible)")
            else:
                print("Matrix is non-singular (invertible)")
                
        except ValueError as e:
            print(f"Error: {e}")
    
    # Verify with NumPy if available
    try:
        import numpy as np
        print("\n--- NumPy Verification ---")
        for i, matrix in enumerate(test_matrices):
            our_det = determinant_iterative(matrix)
            numpy_det = np.linalg.det(matrix)
            print(f"Matrix {i+1}: Our={our_det:.6f}, NumPy={numpy_det:.6f}, Diff={abs(our_det-numpy_det):.10f}")
    except ImportError:
        print("NumPy not available for verification")


# Special matrices for testing
def test_special_matrices():
    """Test determinant calculation on special matrices"""
    
    # Identity matrix
    identity_3x3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    # Zero matrix
    zero_3x3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    # Upper triangular
    upper_triangular = [[2, 3, 4], [0, 5, 6], [0, 0, 7]]
    
    # Lower triangular  
    lower_triangular = [[2, 0, 0], [3, 5, 0], [4, 6, 7]]
    
    special_matrices = [
        ("Identity", identity_3x3),
        ("Zero", zero_3x3),
        ("Upper Triangular", upper_triangular),
        ("Lower Triangular", lower_triangular)
    ]
    
    print("\n--- Special Matrices ---")
    for name, matrix in special_matrices:
        det = determinant_recursive(matrix)
        print(f"{name:15}: det = {det:.6f}")
```

**Key Features:**
- **Multiple Algorithms**: Recursive, iterative, and optimized approaches
- **Performance Optimization**: Memoization and smart expansion choices
- **Numerical Stability**: Partial pivoting in Gaussian elimination
- **Comprehensive Testing**: Benchmarking and verification capabilities
- **Special Case Handling**: Identity, zero, and triangular matrices

**Complexity Analysis:**
- **Recursive**: O(n!) - factorial time complexity
- **Iterative (Gaussian)**: O(n³) - much more efficient
- **Optimized Recursive**: Better than basic recursive due to smart expansion

**Best Practices:**
- Use iterative method for large matrices (n > 4)
- Recursive method good for educational purposes and small matrices
- Always check for numerical stability in practical applications

---

## Question 4

**Develop a Python function to compute the inverse of a matrix.**

**Answer:** Here's a comprehensive implementation of matrix inverse using multiple methods:

```python
import copy
import math

def matrix_inverse_gauss_jordan(matrix):
    """
    Calculate matrix inverse using Gauss-Jordan elimination.
    
    Args:
        matrix: Square matrix as list of lists
    
    Returns:
        Inverse matrix as list of lists
    
    Raises:
        ValueError: If matrix is singular or not square
    """
    n = len(matrix)
    
    # Validate input
    if not matrix or any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square and non-empty")
    
    # Create augmented matrix [A|I]
    augmented = []
    for i in range(n):
        row = matrix[i][:] + [0] * n  # Copy original row + identity row
        row[n + i] = 1  # Set diagonal element of identity part to 1
        augmented.append(row)
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Swap rows if needed
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Check for singular matrix
        if abs(augmented[i][i]) < 1e-10:
            raise ValueError("Matrix is singular (non-invertible)")
        
        # Scale pivot row
        pivot = augmented[i][i]
        for j in range(2 * n):
            augmented[i][j] /= pivot
        
        # Eliminate column
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Extract inverse matrix from right half
    inverse = []
    for i in range(n):
        inverse.append(augmented[i][n:])
    
    return inverse


def matrix_inverse_adjugate(matrix):
    """
    Calculate matrix inverse using adjugate (classical adjoint) method.
    
    Args:
        matrix: Square matrix as list of lists
    
    Returns:
        Inverse matrix
    
    Note: Less efficient for large matrices but good for understanding
    """
    n = len(matrix)
    
    if n != len(matrix[0]):
        raise ValueError("Matrix must be square")
    
    # Calculate determinant
    det = determinant_recursive(matrix)
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular (determinant is zero)")
    
    # Calculate adjugate matrix
    adjugate = calculate_adjugate(matrix)
    
    # Inverse = (1/det) * adjugate
    inverse = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adjugate[i][j] / det)
        inverse.append(row)
    
    return inverse


def calculate_adjugate(matrix):
    """
    Calculate the adjugate (adjoint) matrix.
    
    Args:
        matrix: Square matrix
    
    Returns:
        Adjugate matrix
    """
    n = len(matrix)
    adjugate = []
    
    for i in range(n):
        row = []
        for j in range(n):
            # Calculate cofactor
            minor = get_minor_matrix(matrix, i, j)
            cofactor = ((-1) ** (i + j)) * determinant_recursive(minor)
            row.append(cofactor)
        adjugate.append(row)
    
    # Transpose the cofactor matrix
    return matrix_transpose(adjugate)


def matrix_inverse_lu_decomposition(matrix):
    """
    Calculate matrix inverse using LU decomposition.
    
    Args:
        matrix: Square matrix
    
    Returns:
        Inverse matrix
    """
    n = len(matrix)
    
    # Get LU decomposition
    L, U, P = lu_decomposition(matrix)
    
    # Calculate inverse by solving AX = I
    # Where A = PLU, so we solve PLU * X = I
    # This becomes LU * X = P^T * I = P^T
    
    inverse = []
    
    for i in range(n):
        # Create i-th column of identity matrix
        e_i = [0] * n
        e_i[i] = 1
        
        # Apply permutation
        b = apply_permutation_transpose(e_i, P)
        
        # Solve Ly = b
        y = forward_substitution(L, b)
        
        # Solve Ux = y
        x = backward_substitution(U, y)
        
        inverse.append(x)
    
    # Transpose to get final inverse
    return matrix_transpose(inverse)


def lu_decomposition(matrix):
    """
    Perform LU decomposition with partial pivoting.
    
    Returns:
        L, U, P matrices where PA = LU
    """
    n = len(matrix)
    
    # Initialize matrices
    L = [[0.0] * n for _ in range(n)]
    U = [row[:] for row in matrix]  # Copy of original matrix
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # Set diagonal of L to 1
    for i in range(n):
        L[i][i] = 1.0
    
    # Perform decomposition
    for i in range(n):
        # Partial pivoting
        max_row = i
        for k in range(i + 1, n):
            if abs(U[k][i]) > abs(U[max_row][i]):
                max_row = k
        
        # Swap rows in U and P
        if max_row != i:
            U[i], U[max_row] = U[max_row], U[i]
            P[i], P[max_row] = P[max_row], P[i]
            
            # Swap rows in L (only the computed part)
            for k in range(i):
                L[i][k], L[max_row][k] = L[max_row][k], L[i][k]
        
        # Check for zero pivot
        if abs(U[i][i]) < 1e-10:
            raise ValueError("Matrix is singular")
        
        # Eliminate below pivot
        for k in range(i + 1, n):
            factor = U[k][i] / U[i][i]
            L[k][i] = factor
            for j in range(i, n):
                U[k][j] -= factor * U[i][j]
    
    return L, U, P


def forward_substitution(L, b):
    """Solve Ly = b for y using forward substitution"""
    n = len(L)
    y = [0.0] * n
    
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
        y[i] /= L[i][i]
    
    return y


def backward_substitution(U, y):
    """Solve Ux = y for x using backward substitution"""
    n = len(U)
    x = [0.0] * n
    
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    
    return x


def apply_permutation_transpose(vector, P):
    """Apply transpose of permutation matrix to vector"""
    n = len(vector)
    result = [0.0] * n
    
    for i in range(n):
        for j in range(n):
            result[i] += P[j][i] * vector[j]
    
    return result


def matrix_inverse_iterative_refinement(matrix, max_iterations=5):
    """
    Calculate matrix inverse with iterative refinement for better accuracy.
    
    Args:
        matrix: Square matrix
        max_iterations: Maximum refinement iterations
    
    Returns:
        Improved inverse matrix
    """
    # Get initial inverse
    X = matrix_inverse_gauss_jordan(matrix)
    
    n = len(matrix)
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    for iteration in range(max_iterations):
        # Calculate residual: R = I - A*X
        AX = matrix_multiply(matrix, X)
        R = matrix_subtract(I, AX)
        
        # Check convergence
        residual_norm = matrix_frobenius_norm(R)
        if residual_norm < 1e-12:
            break
        
        # Solve A*dX = R for correction dX
        try:
            dX = matrix_inverse_gauss_jordan_solve(matrix, R)
            
            # Update: X = X + dX
            X = matrix_add(X, dX)
        except:
            break  # Stop if refinement fails
    
    return X


def matrix_frobenius_norm(matrix):
    """Calculate Frobenius norm of matrix"""
    norm = 0.0
    for row in matrix:
        for val in row:
            norm += val * val
    return math.sqrt(norm)


def verify_inverse(matrix, inverse, tolerance=1e-10):
    """
    Verify that A * A^(-1) = I
    
    Args:
        matrix: Original matrix
        inverse: Computed inverse
        tolerance: Numerical tolerance
    
    Returns:
        True if verification passes
    """
    n = len(matrix)
    
    # Calculate A * A^(-1)
    product = matrix_multiply(matrix, inverse)
    
    # Check if result is identity matrix
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(product[i][j] - expected) > tolerance:
                return False
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Test matrices
    test_matrices = [
        # 2x2 invertible matrix
        [[2, 1],
         [1, 1]],
        
        # 3x3 invertible matrix
        [[2, -1, 0],
         [-1, 2, -1],
         [0, -1, 2]],
        
        # 3x3 well-conditioned matrix
        [[4, 7, 2],
         [3, 6, 1],
         [5, 1, 9]]
    ]
    
    for i, matrix in enumerate(test_matrices):
        print(f"\n--- Test Matrix {i+1} ---")
        print("Original Matrix:")
        for row in matrix:
            print([f"{val:8.3f}" for val in row])
        
        try:
            # Test different methods
            methods = [
                ("Gauss-Jordan", matrix_inverse_gauss_jordan),
                ("Adjugate", matrix_inverse_adjugate),
                ("LU Decomposition", matrix_inverse_lu_decomposition),
            ]
            
            for name, method in methods:
                try:
                    inverse = method(matrix)
                    
                    print(f"\nInverse using {name}:")
                    for row in inverse:
                        print([f"{val:8.3f}" for val in row])
                    
                    # Verify the inverse
                    if verify_inverse(matrix, inverse):
                        print(f"✓ {name} verification passed")
                    else:
                        print(f"✗ {name} verification failed")
                
                except Exception as e:
                    print(f"✗ {name} failed: {e}")
        
        except Exception as e:
            print(f"Error processing matrix: {e}")
    
    # Test singular matrix
    print(f"\n--- Singular Matrix Test ---")
    singular_matrix = [[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]
    
    try:
        inverse = matrix_inverse_gauss_jordan(singular_matrix)
        print("Unexpected: Singular matrix was inverted!")
    except ValueError as e:
        print(f"✓ Correctly detected singular matrix: {e}")


# Utility functions (assuming these exist from previous questions)
def matrix_multiply(A, B):
    """Matrix multiplication function from Question 1"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Incompatible dimensions")
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result


def matrix_add(A, B):
    """Matrix addition from Question 1"""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] 
            for i in range(len(A))]


def matrix_subtract(A, B):
    """Matrix subtraction from Question 1"""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] 
            for i in range(len(A))]


def matrix_transpose(matrix):
    """Matrix transpose from Question 2"""
    return [[matrix[i][j] for i in range(len(matrix))] 
            for j in range(len(matrix[0]))]


def get_minor_matrix(matrix, row_to_remove, col_to_remove):
    """Get minor matrix (from Question 3)"""
    return [[matrix[i][j] for j in range(len(matrix[0])) if j != col_to_remove]
            for i in range(len(matrix)) if i != row_to_remove]


def determinant_recursive(matrix):
    """Determinant calculation (from Question 3)"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for j in range(n):
        cofactor = (-1) ** j * matrix[0][j]
        minor = get_minor_matrix(matrix, 0, j)
        det += cofactor * determinant_recursive(minor)
    
    return det
```

**Key Features:**
- **Multiple Methods**: Gauss-Jordan, adjugate, and LU decomposition
- **Numerical Stability**: Partial pivoting and iterative refinement
- **Error Handling**: Singular matrix detection
- **Verification**: Automatic inverse verification
- **Performance**: Different methods for different use cases

**Method Comparison:**
- **Gauss-Jordan**: Most straightforward, O(n³) complexity
- **Adjugate**: Good for small matrices, O(n⁴) complexity
- **LU Decomposition**: Most efficient for multiple solves

**Best Practices:**
- Always verify the computed inverse
- Use appropriate method based on matrix size
- Handle numerical precision carefully
- Check for singular matrices before computation

---

## Question 5

**Write an algorithm to perform eigenvalue and eigenvector decomposition.**

**Answer:** Here's a comprehensive implementation of eigenvalue and eigenvector decomposition using multiple methods:

```python
import math
import copy

def power_iteration(matrix, max_iterations=1000, tolerance=1e-10):
    """
    Find the dominant eigenvalue and eigenvector using power iteration.
    
    Args:
        matrix: Square matrix as list of lists
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
    
    Returns:
        (eigenvalue, eigenvector) tuple
    """
    n = len(matrix)
    
    # Initialize random vector
    x = [1.0] + [0.0] * (n - 1)  # Start with [1, 0, 0, ...]
    
    eigenvalue = 0
    
    for iteration in range(max_iterations):
        # Matrix-vector multiplication: y = Ax
        y = matrix_vector_multiply(matrix, x)
        
        # Find the largest component (in absolute value)
        max_component = max(abs(val) for val in y)
        
        if max_component < tolerance:
            raise ValueError("Matrix appears to be singular")
        
        # Normalize the vector
        y_normalized = [val / max_component for val in y]
        
        # Calculate Rayleigh quotient for eigenvalue
        numerator = vector_dot_product(x, matrix_vector_multiply(matrix, x))
        denominator = vector_dot_product(x, x)
        new_eigenvalue = numerator / denominator if denominator != 0 else 0
        
        # Check convergence
        if abs(new_eigenvalue - eigenvalue) < tolerance:
            return new_eigenvalue, y_normalized
        
        eigenvalue = new_eigenvalue
        x = y_normalized
    
    return eigenvalue, x


def inverse_power_iteration(matrix, shift=0, max_iterations=1000, tolerance=1e-10):
    """
    Find eigenvalue closest to shift using inverse power iteration.
    
    Args:
        matrix: Square matrix
        shift: Shift value (finds eigenvalue closest to this)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        (eigenvalue, eigenvector) tuple
    """
    n = len(matrix)
    
    # Create shifted matrix: A - shift*I
    shifted_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(matrix[i][j] - shift)
            else:
                row.append(matrix[i][j])
        shifted_matrix.append(row)
    
    # Try to invert the shifted matrix
    try:
        shifted_inverse = matrix_inverse_gauss_jordan(shifted_matrix)
    except ValueError:
        # If shift is exactly an eigenvalue, use a small perturbation
        shift += 1e-6
        for i in range(n):
            shifted_matrix[i][i] = matrix[i][i] - shift
        shifted_inverse = matrix_inverse_gauss_jordan(shifted_matrix)
    
    # Apply power iteration to the inverse
    eigenvalue_inv, eigenvector = power_iteration(shifted_inverse, max_iterations, tolerance)
    
    # Convert back to original eigenvalue
    eigenvalue = 1.0 / eigenvalue_inv + shift
    
    return eigenvalue, eigenvector


def qr_algorithm(matrix, max_iterations=1000, tolerance=1e-10):
    """
    Find all eigenvalues using QR algorithm.
    
    Args:
        matrix: Square matrix
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        List of eigenvalues
    """
    n = len(matrix)
    A = [row[:] for row in matrix]  # Copy matrix
    
    for iteration in range(max_iterations):
        # QR decomposition
        Q, R = qr_decomposition(A)
        
        # Update A = RQ
        A = matrix_multiply(R, Q)
        
        # Check for convergence (off-diagonal elements should be small)
        converged = True
        for i in range(n):
            for j in range(n):
                if i != j and abs(A[i][j]) > tolerance:
                    converged = False
                    break
            if not converged:
                break
        
        if converged:
            break
    
    # Extract eigenvalues from diagonal
    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues


def qr_decomposition(matrix):
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        matrix: Square matrix
    
    Returns:
        (Q, R) where Q is orthogonal and R is upper triangular
    """
    n = len(matrix)
    
    # Initialize Q and R
    Q = [[0.0] * n for _ in range(n)]
    R = [[0.0] * n for _ in range(n)]
    
    # Extract columns of matrix
    columns = []
    for j in range(n):
        column = [matrix[i][j] for i in range(n)]
        columns.append(column)
    
    # Gram-Schmidt process
    for j in range(n):
        # Start with original column
        q_j = columns[j][:]
        
        # Subtract projections onto previous orthogonal vectors
        for i in range(j):
            q_i = [Q[k][i] for k in range(n)]
            projection_coeff = vector_dot_product(columns[j], q_i)
            R[i][j] = projection_coeff
            
            # Subtract projection
            for k in range(n):
                q_j[k] -= projection_coeff * q_i[k]
        
        # Normalize
        norm = vector_norm(q_j)
        R[j][j] = norm
        
        if norm > 1e-10:
            for k in range(n):
                Q[k][j] = q_j[k] / norm
        else:
            # Handle near-zero vector
            Q[j][j] = 1.0
    
    return Q, R


def jacobi_eigenvalue(matrix, max_iterations=1000, tolerance=1e-10):
    """
    Find eigenvalues and eigenvectors using Jacobi method (for symmetric matrices).
    
    Args:
        matrix: Symmetric square matrix
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        (eigenvalues, eigenvectors) where eigenvectors are columns of the matrix
    """
    n = len(matrix)
    
    # Check if matrix is symmetric
    for i in range(n):
        for j in range(n):
            if abs(matrix[i][j] - matrix[j][i]) > tolerance:
                raise ValueError("Jacobi method requires symmetric matrix")
    
    # Initialize
    A = [row[:] for row in matrix]  # Copy matrix
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # Identity matrix
    
    for iteration in range(max_iterations):
        # Find largest off-diagonal element
        max_val = 0
        p, q = 0, 1
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > max_val:
                    max_val = abs(A[i][j])
                    p, q = i, j
        
        # Check convergence
        if max_val < tolerance:
            break
        
        # Calculate rotation angle
        if abs(A[p][p] - A[q][q]) < tolerance:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan(2 * A[p][q] / (A[p][p] - A[q][q]))
        
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Apply Jacobi rotation
        apply_jacobi_rotation(A, V, p, q, cos_theta, sin_theta)
    
    # Extract eigenvalues from diagonal
    eigenvalues = [A[i][i] for i in range(n)]
    
    # Eigenvectors are columns of V
    eigenvectors = []
    for j in range(n):
        eigenvector = [V[i][j] for i in range(n)]
        eigenvectors.append(eigenvector)
    
    return eigenvalues, eigenvectors


def apply_jacobi_rotation(A, V, p, q, cos_theta, sin_theta):
    """Apply Jacobi rotation to matrices A and V"""
    n = len(A)
    
    # Store original values that will be modified
    App = A[p][p]
    Aqq = A[q][q]
    Apq = A[p][q]
    
    # Update diagonal elements
    A[p][p] = cos_theta * cos_theta * App + sin_theta * sin_theta * Aqq - 2 * cos_theta * sin_theta * Apq
    A[q][q] = sin_theta * sin_theta * App + cos_theta * cos_theta * Aqq + 2 * cos_theta * sin_theta * Apq
    A[p][q] = A[q][p] = 0
    
    # Update other elements in row/column p and q
    for i in range(n):
        if i != p and i != q:
            Aip = A[i][p]
            Aiq = A[i][q]
            A[i][p] = A[p][i] = cos_theta * Aip - sin_theta * Aiq
            A[i][q] = A[q][i] = sin_theta * Aip + cos_theta * Aiq
    
    # Update eigenvector matrix V
    for i in range(n):
        Vip = V[i][p]
        Viq = V[i][q]
        V[i][p] = cos_theta * Vip - sin_theta * Viq
        V[i][q] = sin_theta * Vip + cos_theta * Viq


def characteristic_polynomial_roots(matrix):
    """
    Find eigenvalues by solving characteristic polynomial (for small matrices).
    
    Args:
        matrix: Square matrix (preferably 2x2 or 3x3)
    
    Returns:
        List of eigenvalues
    """
    n = len(matrix)
    
    if n == 1:
        return [matrix[0][0]]
    
    elif n == 2:
        # For 2x2 matrix: λ² - trace(A)λ + det(A) = 0
        trace = matrix[0][0] + matrix[1][1]
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        
        # Quadratic formula
        discriminant = trace * trace - 4 * det
        if discriminant < 0:
            # Complex eigenvalues
            real_part = trace / 2
            imag_part = math.sqrt(-discriminant) / 2
            return [complex(real_part, imag_part), complex(real_part, -imag_part)]
        else:
            sqrt_disc = math.sqrt(discriminant)
            lambda1 = (trace + sqrt_disc) / 2
            lambda2 = (trace - sqrt_disc) / 2
            return [lambda1, lambda2]
    
    else:
        # For larger matrices, use QR algorithm
        return qr_algorithm(matrix)


def find_eigenvector(matrix, eigenvalue, tolerance=1e-10):
    """
    Find eigenvector corresponding to a given eigenvalue.
    
    Args:
        matrix: Square matrix
        eigenvalue: Known eigenvalue
        tolerance: Numerical tolerance
    
    Returns:
        Corresponding eigenvector
    """
    n = len(matrix)
    
    # Create matrix (A - λI)
    shifted_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(matrix[i][j] - eigenvalue)
            else:
                row.append(matrix[i][j])
        shifted_matrix.append(row)
    
    # Find null space of (A - λI) by Gaussian elimination
    # Convert to row echelon form
    augmented = [row[:] for row in shifted_matrix]
    
    # Gaussian elimination
    for i in range(min(n, n)):
        # Find pivot
        pivot_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[pivot_row][i]):
                pivot_row = k
        
        # Swap rows
        if pivot_row != i:
            augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]
        
        # Skip if pivot is too small
        if abs(augmented[i][i]) < tolerance:
            continue
        
        # Eliminate below pivot
        for k in range(i + 1, n):
            if abs(augmented[i][i]) > tolerance:
                factor = augmented[k][i] / augmented[i][i]
                for j in range(n):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Back substitution to find a solution
    eigenvector = [0.0] * n
    eigenvector[-1] = 1.0  # Set last component to 1
    
    for i in range(n - 2, -1, -1):
        if abs(augmented[i][i]) > tolerance:
            sum_val = sum(augmented[i][j] * eigenvector[j] for j in range(i + 1, n))
            eigenvector[i] = -sum_val / augmented[i][i]
    
    # Normalize eigenvector
    norm = vector_norm(eigenvector)
    if norm > tolerance:
        eigenvector = [x / norm for x in eigenvector]
    
    return eigenvector


# Utility functions
def matrix_vector_multiply(matrix, vector):
    """Multiply matrix by vector"""
    return [sum(matrix[i][j] * vector[j] for j in range(len(vector))) 
            for i in range(len(matrix))]


def vector_dot_product(v1, v2):
    """Calculate dot product of two vectors"""
    return sum(v1[i] * v2[i] for i in range(len(v1)))


def vector_norm(vector):
    """Calculate L2 norm of vector"""
    return math.sqrt(sum(x * x for x in vector))


def verify_eigendecomposition(matrix, eigenvalues, eigenvectors, tolerance=1e-10):
    """
    Verify that Av = λv for each eigenvalue-eigenvector pair.
    
    Args:
        matrix: Original matrix
        eigenvalues: List of eigenvalues
        eigenvectors: List of corresponding eigenvectors
        tolerance: Numerical tolerance
    
    Returns:
        True if verification passes
    """
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors)):
        # Calculate Av
        Av = matrix_vector_multiply(matrix, eigenvec)
        
        # Calculate λv
        lambda_v = [eigenval * x for x in eigenvec]
        
        # Check if Av ≈ λv
        error = vector_norm([Av[j] - lambda_v[j] for j in range(len(Av))])
        
        if error > tolerance:
            print(f"Verification failed for eigenvalue {i}: error = {error}")
            return False
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Test matrices
    test_matrices = [
        # 2x2 symmetric matrix
        [[4, 1],
         [1, 3]],
        
        # 3x3 symmetric matrix
        [[2, -1, 0],
         [-1, 2, -1],
         [0, -1, 2]],
        
        # 3x3 general matrix
        [[1, 2, 3],
         [0, 4, 5],
         [0, 0, 6]]
    ]
    
    for i, matrix in enumerate(test_matrices):
        print(f"\n--- Test Matrix {i+1} ---")
        print("Matrix:")
        for row in matrix:
            print([f"{val:8.3f}" for val in row])
        
        n = len(matrix)
        
        try:
            # Method 1: Power iteration for dominant eigenvalue
            print(f"\n1. Power Iteration:")
            dom_eigenval, dom_eigenvec = power_iteration(matrix)
            print(f"   Dominant eigenvalue: {dom_eigenval:.6f}")
            print(f"   Corresponding eigenvector: {[f'{x:.6f}' for x in dom_eigenvec]}")
            
            # Method 2: QR algorithm for all eigenvalues
            print(f"\n2. QR Algorithm:")
            eigenvalues_qr = qr_algorithm(matrix)
            print(f"   All eigenvalues: {[f'{x:.6f}' for x in eigenvalues_qr]}")
            
            # Method 3: Jacobi method (for symmetric matrices)
            is_symmetric = all(abs(matrix[i][j] - matrix[j][i]) < 1e-10 
                             for i in range(n) for j in range(n))
            
            if is_symmetric:
                print(f"\n3. Jacobi Method (Symmetric):")
                eigenvals_jac, eigenvecs_jac = jacobi_eigenvalue(matrix)
                print(f"   Eigenvalues: {[f'{x:.6f}' for x in eigenvals_jac]}")
                print(f"   Eigenvectors:")
                for j, vec in enumerate(eigenvecs_jac):
                    print(f"     λ{j+1}: {[f'{x:.6f}' for x in vec]}")
                
                # Verify eigendecomposition
                if verify_eigendecomposition(matrix, eigenvals_jac, eigenvecs_jac):
                    print("   ✓ Eigendecomposition verified")
                else:
                    print("   ✗ Eigendecomposition verification failed")
            
            # Method 4: Characteristic polynomial (for small matrices)
            if n <= 3:
                print(f"\n4. Characteristic Polynomial:")
                eigenvals_char = characteristic_polynomial_roots(matrix)
                print(f"   Eigenvalues: {[f'{x:.6f}' if isinstance(x, (int, float)) else f'{x}' for x in eigenvals_char]}")
        
        except Exception as e:
            print(f"Error processing matrix: {e}")
    
    # Special test: Diagonal matrix (eigenvalues should be diagonal elements)
    print(f"\n--- Diagonal Matrix Test ---")
    diagonal_matrix = [[5, 0, 0],
                      [0, 3, 0],
                      [0, 0, 1]]
    
    print("Diagonal Matrix:")
    for row in diagonal_matrix:
        print(row)
    
    eigenvals_diag = qr_algorithm(diagonal_matrix)
    print(f"Eigenvalues: {[f'{x:.6f}' for x in sorted(eigenvals_diag, reverse=True)]}")
    print("Expected: [5.000, 3.000, 1.000]")
    
    # Compare with NumPy if available
    try:
        import numpy as np
        print(f"\n--- NumPy Verification ---")
        
        for i, matrix in enumerate(test_matrices):
            our_eigenvals = sorted(qr_algorithm(matrix), reverse=True)
            numpy_eigenvals = sorted(np.linalg.eigvals(matrix), reverse=True)
            
            print(f"Matrix {i+1}:")
            print(f"  Our result:   {[f'{x:.6f}' for x in our_eigenvals]}")
            print(f"  NumPy result: {[f'{x:.6f}' for x in numpy_eigenvals]}")
            
            # Calculate difference
            if len(our_eigenvals) == len(numpy_eigenvals):
                max_diff = max(abs(our_eigenvals[j] - numpy_eigenvals[j]) 
                             for j in range(len(our_eigenvals)))
                print(f"  Max difference: {max_diff:.10f}")
    
    except ImportError:
        print("NumPy not available for verification")


# Advanced eigendecomposition class
class EigenDecomposer:
    """
    Advanced eigenvalue decomposition with multiple algorithms.
    """
    
    def __init__(self, matrix):
        self.matrix = [row[:] for row in matrix]  # Copy
        self.n = len(matrix)
        self.eigenvalues = None
        self.eigenvectors = None
    
    def decompose(self, method="auto", **kwargs):
        """
        Perform eigendecomposition using specified method.
        
        Args:
            method: "auto", "qr", "jacobi", "power", or "characteristic"
        """
        if method == "auto":
            # Choose best method based on matrix properties
            if self.n <= 2:
                method = "characteristic"
            elif self._is_symmetric():
                method = "jacobi"
            else:
                method = "qr"
        
        if method == "qr":
            self.eigenvalues = qr_algorithm(self.matrix, **kwargs)
            self.eigenvectors = [find_eigenvector(self.matrix, val) 
                               for val in self.eigenvalues]
        
        elif method == "jacobi":
            if not self._is_symmetric():
                raise ValueError("Jacobi method requires symmetric matrix")
            self.eigenvalues, self.eigenvectors = jacobi_eigenvalue(self.matrix, **kwargs)
        
        elif method == "power":
            # Only finds dominant eigenvalue
            eigenval, eigenvec = power_iteration(self.matrix, **kwargs)
            self.eigenvalues = [eigenval]
            self.eigenvectors = [eigenvec]
        
        elif method == "characteristic":
            if self.n > 3:
                raise ValueError("Characteristic polynomial method only for small matrices")
            self.eigenvalues = characteristic_polynomial_roots(self.matrix)
            self.eigenvectors = [find_eigenvector(self.matrix, val) 
                               for val in self.eigenvalues]
        
        return self.eigenvalues, self.eigenvectors
    
    def _is_symmetric(self, tolerance=1e-10):
        """Check if matrix is symmetric"""
        for i in range(self.n):
            for j in range(self.n):
                if abs(self.matrix[i][j] - self.matrix[j][i]) > tolerance:
                    return False
        return True
    
    def get_spectral_radius(self):
        """Get spectral radius (largest absolute eigenvalue)"""
        if self.eigenvalues is None:
            self.decompose()
        return max(abs(val) for val in self.eigenvalues)
    
    def get_condition_number(self):
        """Get condition number (ratio of largest to smallest eigenvalue)"""
        if self.eigenvalues is None:
            self.decompose()
        
        abs_eigenvals = [abs(val) for val in self.eigenvalues if abs(val) > 1e-10]
        if not abs_eigenvals:
            return float('inf')
        
        return max(abs_eigenvals) / min(abs_eigenvals)
```

**Key Features:**
- **Multiple Algorithms**: Power iteration, QR algorithm, Jacobi method, characteristic polynomial
- **Comprehensive**: Handles both symmetric and general matrices
- **Robust**: Includes numerical stability features and error handling
- **Verification**: Built-in verification of eigendecomposition results
- **Performance**: Different methods optimized for different matrix types

**Algorithm Comparison:**
- **Power Iteration**: O(n²) per iteration, finds dominant eigenvalue only
- **QR Algorithm**: O(n³) per iteration, finds all eigenvalues
- **Jacobi Method**: O(n³) per iteration, for symmetric matrices, finds eigenvalues and eigenvectors
- **Characteristic Polynomial**: Analytical for small matrices (n ≤ 3)

**Best Practices:**
- Use Jacobi method for symmetric matrices
- Use QR algorithm for general matrices
- Use power iteration when only dominant eigenvalue is needed
- Always verify results when numerical precision is critical

**Applications:**
- Principal Component Analysis (PCA)
- Stability analysis of dynamical systems
- Google PageRank algorithm
- Quantum mechanics and vibration analysis

---

## Question 6

**Create a Python script to solve a system of linear equations using NumPy.**

**Answer:** Here's a comprehensive Python script with multiple methods for solving linear equation systems both with and without NumPy:

```python
import numpy as np
import math

def solve_linear_system_numpy(A, b):
    """
    Solve linear system Ax = b using NumPy.
    
    Args:
        A: Coefficient matrix (2D array)
        b: Right-hand side vector (1D array)
    
    Returns:
        Solution vector x
    """
    # Convert to numpy arrays if needed
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    
    try:
        # Method 1: Direct solve using numpy.linalg.solve
        x = np.linalg.solve(A_np, b_np)
        return x
    
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError: {e}")
        # Try alternative methods for singular or ill-conditioned matrices
        return solve_with_pseudoinverse_numpy(A_np, b_np)


def solve_with_pseudoinverse_numpy(A, b):
    """
    Solve using Moore-Penrose pseudoinverse for overdetermined/underdetermined systems.
    """
    try:
        # Calculate pseudoinverse
        A_pinv = np.linalg.pinv(A)
        x = A_pinv @ b
        return x
    except Exception as e:
        print(f"Pseudoinverse failed: {e}")
        return None


def analyze_system_numpy(A, b):
    """
    Comprehensive analysis of linear system using NumPy.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
    
    Returns:
        Dictionary with analysis results
    """
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    
    analysis = {}
    
    # Basic properties
    analysis['matrix_shape'] = A_np.shape
    analysis['vector_length'] = len(b_np)
    analysis['is_square'] = A_np.shape[0] == A_np.shape[1]
    
    # Determinant (for square matrices)
    if analysis['is_square']:
        analysis['determinant'] = np.linalg.det(A_np)
        analysis['is_singular'] = abs(analysis['determinant']) < 1e-10
    
    # Rank and condition number
    analysis['rank_A'] = np.linalg.matrix_rank(A_np)
    analysis['rank_augmented'] = np.linalg.matrix_rank(np.column_stack([A_np, b_np]))
    
    if analysis['is_square'] and not analysis['is_singular']:
        analysis['condition_number'] = np.linalg.cond(A_np)
        analysis['is_well_conditioned'] = analysis['condition_number'] < 1e12
    
    # System classification
    m, n = A_np.shape
    if m == n:
        if analysis['rank_A'] == n:
            analysis['system_type'] = 'unique_solution'
        else:
            analysis['system_type'] = 'no_solution_or_infinite'
    elif m > n:
        analysis['system_type'] = 'overdetermined'
    else:
        analysis['system_type'] = 'underdetermined'
    
    # Consistency check (Rouché-Capelli theorem)
    if analysis['rank_A'] == analysis['rank_augmented']:
        if analysis['rank_A'] == A_np.shape[1]:
            analysis['solution_type'] = 'unique'
        else:
            analysis['solution_type'] = 'infinite'
    else:
        analysis['solution_type'] = 'no_solution'
    
    return analysis


def solve_multiple_methods_numpy(A, b):
    """
    Solve linear system using multiple NumPy methods and compare results.
    """
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    
    results = {}
    
    # Method 1: Direct solve
    try:
        x1 = np.linalg.solve(A_np, b_np)
        results['direct_solve'] = x1
    except Exception as e:
        results['direct_solve'] = f"Failed: {e}"
    
    # Method 2: Matrix inverse
    try:
        A_inv = np.linalg.inv(A_np)
        x2 = A_inv @ b_np
        results['matrix_inverse'] = x2
    except Exception as e:
        results['matrix_inverse'] = f"Failed: {e}"
    
    # Method 3: LU decomposition
    try:
        from scipy.linalg import lu_solve, lu_factor
        lu, piv = lu_factor(A_np)
        x3 = lu_solve((lu, piv), b_np)
        results['lu_decomposition'] = x3
    except Exception as e:
        results['lu_decomposition'] = f"Failed: {e}"
    
    # Method 4: QR decomposition
    try:
        Q, R = np.linalg.qr(A_np)
        x4 = np.linalg.solve(R, Q.T @ b_np)
        results['qr_decomposition'] = x4
    except Exception as e:
        results['qr_decomposition'] = f"Failed: {e}"
    
    # Method 5: SVD (Singular Value Decomposition)
    try:
        U, s, Vt = np.linalg.svd(A_np)
        # Solve using SVD
        c = U.T @ b_np
        y = c / s
        x5 = Vt.T @ y
        results['svd'] = x5
    except Exception as e:
        results['svd'] = f"Failed: {e}"
    
    # Method 6: Least squares (for overdetermined systems)
    try:
        x6, residuals, rank, s = np.linalg.lstsq(A_np, b_np, rcond=None)
        results['least_squares'] = x6
        results['least_squares_residuals'] = residuals
    except Exception as e:
        results['least_squares'] = f"Failed: {e}"
    
    return results


def solve_iterative_methods_numpy(A, b, max_iterations=1000, tolerance=1e-10):
    """
    Solve using iterative methods (Jacobi, Gauss-Seidel).
    """
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    n = len(b_np)
    
    results = {}
    
    # Jacobi method
    try:
        x_jacobi = jacobi_method_numpy(A_np, b_np, max_iterations, tolerance)
        results['jacobi'] = x_jacobi
    except Exception as e:
        results['jacobi'] = f"Failed: {e}"
    
    # Gauss-Seidel method
    try:
        x_gauss_seidel = gauss_seidel_method_numpy(A_np, b_np, max_iterations, tolerance)
        results['gauss_seidel'] = x_gauss_seidel
    except Exception as e:
        results['gauss_seidel'] = f"Failed: {e}"
    
    return results


def jacobi_method_numpy(A, b, max_iterations=1000, tolerance=1e-10):
    """
    Solve linear system using Jacobi iteration method.
    """
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    
    # Check diagonal dominance
    for i in range(n):
        if abs(A[i, i]) <= sum(abs(A[i, j]) for j in range(n) if j != i):
            print("Warning: Matrix may not be diagonally dominant. Convergence not guaranteed.")
            break
    
    for iteration in range(max_iterations):
        for i in range(n):
            if abs(A[i, i]) < tolerance:
                raise ValueError(f"Zero diagonal element at position {i}")
            
            sum_ax = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_ax) / A[i, i]
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tolerance:
            return x_new
        
        x = x_new.copy()
    
    print(f"Warning: Jacobi method did not converge after {max_iterations} iterations")
    return x


def gauss_seidel_method_numpy(A, b, max_iterations=1000, tolerance=1e-10):
    """
    Solve linear system using Gauss-Seidel iteration method.
    """
    n = len(b)
    x = np.zeros(n)
    
    for iteration in range(max_iterations):
        x_old = x.copy()
        
        for i in range(n):
            if abs(A[i, i]) < tolerance:
                raise ValueError(f"Zero diagonal element at position {i}")
            
            sum1 = sum(A[i, j] * x[j] for j in range(i))
            sum2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Check convergence
        if np.linalg.norm(x - x_old) < tolerance:
            return x
    
    print(f"Warning: Gauss-Seidel method did not converge after {max_iterations} iterations")
    return x


def solve_special_systems_numpy(A, b):
    """
    Handle special types of linear systems.
    """
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    
    results = {}
    
    # Check if matrix is triangular
    is_upper_triangular = np.allclose(A_np, np.triu(A_np))
    is_lower_triangular = np.allclose(A_np, np.tril(A_np))
    
    if is_upper_triangular:
        results['type'] = 'upper_triangular'
        results['solution'] = backward_substitution_numpy(A_np, b_np)
    elif is_lower_triangular:
        results['type'] = 'lower_triangular'
        results['solution'] = forward_substitution_numpy(A_np, b_np)
    else:
        results['type'] = 'general'
        results['solution'] = np.linalg.solve(A_np, b_np)
    
    # Check if matrix is symmetric
    if np.allclose(A_np, A_np.T):
        results['is_symmetric'] = True
        # Use Cholesky decomposition if positive definite
        try:
            L = np.linalg.cholesky(A_np)
            y = forward_substitution_numpy(L, b_np)
            x_cholesky = backward_substitution_numpy(L.T, y)
            results['cholesky_solution'] = x_cholesky
        except np.linalg.LinAlgError:
            results['cholesky_solution'] = "Not positive definite"
    
    return results


def forward_substitution_numpy(L, b):
    """Forward substitution for lower triangular matrix."""
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    
    return x


def backward_substitution_numpy(U, b):
    """Backward substitution for upper triangular matrix."""
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x


def verify_solution_numpy(A, b, x, tolerance=1e-10):
    """
    Verify that Ax = b for the computed solution.
    """
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    x_np = np.array(x, dtype=float)
    
    # Calculate Ax
    Ax = A_np @ x_np
    
    # Calculate residual
    residual = b_np - Ax
    residual_norm = np.linalg.norm(residual)
    
    verification = {
        'is_solution': residual_norm < tolerance,
        'residual_norm': residual_norm,
        'relative_error': residual_norm / np.linalg.norm(b_np) if np.linalg.norm(b_np) > 0 else 0,
        'residual_vector': residual
    }
    
    return verification


# Comprehensive demonstration script
def main():
    """
    Main demonstration script showing various linear system solving methods.
    """
    print("="*60)
    print("LINEAR EQUATION SYSTEM SOLVER WITH NUMPY")
    print("="*60)
    
    # Example 1: Well-conditioned square system
    print("\n1. WELL-CONDITIONED SQUARE SYSTEM")
    print("-" * 40)
    
    A1 = [[3, 2, -1],
          [2, -2, 4],
          [-1, 0.5, -1]]
    
    b1 = [1, -2, 0]
    
    print("System: Ax = b")
    print("A =", np.array(A1))
    print("b =", np.array(b1))
    
    # Analyze system
    analysis1 = analyze_system_numpy(A1, b1)
    print(f"\nSystem Analysis:")
    print(f"  System type: {analysis1['system_type']}")
    print(f"  Solution type: {analysis1['solution_type']}")
    print(f"  Determinant: {analysis1.get('determinant', 'N/A'):.6f}")
    print(f"  Condition number: {analysis1.get('condition_number', 'N/A'):.2e}")
    
    # Solve using multiple methods
    solutions1 = solve_multiple_methods_numpy(A1, b1)
    print(f"\nSolutions:")
    for method, solution in solutions1.items():
        if isinstance(solution, np.ndarray):
            print(f"  {method:15}: {solution}")
        else:
            print(f"  {method:15}: {solution}")
    
    # Verify solution
    if isinstance(solutions1.get('direct_solve'), np.ndarray):
        verification1 = verify_solution_numpy(A1, b1, solutions1['direct_solve'])
        print(f"\nVerification:")
        print(f"  Is solution: {verification1['is_solution']}")
        print(f"  Residual norm: {verification1['residual_norm']:.2e}")
    
    # Example 2: Overdetermined system (least squares)
    print("\n\n2. OVERDETERMINED SYSTEM (LEAST SQUARES)")
    print("-" * 50)
    
    A2 = [[1, 1],
          [1, 2],
          [1, 3],
          [1, 4]]
    
    b2 = [6, 8, 10, 12]
    
    print("System: Ax = b (overdetermined)")
    print("A =", np.array(A2))
    print("b =", np.array(b2))
    
    analysis2 = analyze_system_numpy(A2, b2)
    print(f"\nSystem Analysis:")
    print(f"  System type: {analysis2['system_type']}")
    print(f"  Solution type: {analysis2['solution_type']}")
    print(f"  Rank of A: {analysis2['rank_A']}")
    print(f"  Rank of [A|b]: {analysis2['rank_augmented']}")
    
    # Solve using least squares
    x2_lstsq, residuals, rank, s = np.linalg.lstsq(A2, b2, rcond=None)
    print(f"\nLeast squares solution: {x2_lstsq}")
    print(f"Residuals: {residuals}")
    print(f"Rank: {rank}")
    
    # Example 3: Ill-conditioned system
    print("\n\n3. ILL-CONDITIONED SYSTEM")
    print("-" * 35)
    
    # Hilbert matrix (notoriously ill-conditioned)
    n = 4
    A3 = [[1/(i+j+1) for j in range(n)] for i in range(n)]
    b3 = [sum(A3[i]) for i in range(n)]  # Solution should be [1, 1, 1, 1]
    
    print("Hilbert matrix system")
    print("A =", np.array(A3))
    print("b =", np.array(b3))
    
    analysis3 = analyze_system_numpy(A3, b3)
    print(f"\nSystem Analysis:")
    print(f"  Condition number: {analysis3.get('condition_number', 'N/A'):.2e}")
    print(f"  Is well-conditioned: {analysis3.get('is_well_conditioned', False)}")
    
    # Solve and show numerical issues
    x3 = np.linalg.solve(A3, b3)
    print(f"Computed solution: {x3}")
    print(f"Expected solution: [1, 1, 1, 1]")
    print(f"Error: {np.linalg.norm(x3 - np.ones(n)):.2e}")
    
    # Example 4: Iterative methods comparison
    print("\n\n4. ITERATIVE METHODS COMPARISON")
    print("-" * 40)
    
    # Create diagonally dominant system for convergence
    A4 = [[10, -1, 2, 0],
          [-1, 11, -1, 3],
          [2, -1, 10, -1],
          [0, 3, -1, 8]]
    
    b4 = [6, 25, -11, 15]
    
    print("Diagonally dominant system")
    print("A =", np.array(A4))
    print("b =", np.array(b4))
    
    # Direct solution
    x4_direct = np.linalg.solve(A4, b4)
    print(f"Direct solution: {x4_direct}")
    
    # Iterative solutions
    iterative_solutions = solve_iterative_methods_numpy(A4, b4)
    for method, solution in iterative_solutions.items():
        if isinstance(solution, np.ndarray):
            error = np.linalg.norm(solution - x4_direct)
            print(f"{method.capitalize():15}: {solution}, Error: {error:.2e}")
        else:
            print(f"{method.capitalize():15}: {solution}")
    
    # Example 5: Special matrices
    print("\n\n5. SPECIAL MATRIX TYPES")
    print("-" * 30)
    
    # Symmetric positive definite matrix
    A5_spd = [[4, -1, 1],
              [-1, 4, -2],
              [1, -2, 4]]
    
    b5 = [1, 5, 0]
    
    print("Symmetric positive definite matrix")
    special_results = solve_special_systems_numpy(A5_spd, b5)
    print(f"Matrix type: {special_results['type']}")
    print(f"Is symmetric: {special_results.get('is_symmetric', False)}")
    print(f"Standard solution: {special_results['solution']}")
    
    if 'cholesky_solution' in special_results:
        if isinstance(special_results['cholesky_solution'], np.ndarray):
            print(f"Cholesky solution: {special_results['cholesky_solution']}")
        else:
            print(f"Cholesky: {special_results['cholesky_solution']}")


if __name__ == "__main__":
    main()


# Additional utility functions for custom implementations
def solve_without_numpy(A, b):
    """
    Solve linear system without NumPy (using custom implementations).
    """
    # Convert to lists if numpy arrays
    if hasattr(A, 'tolist'):
        A = A.tolist()
    if hasattr(b, 'tolist'):
        b = b.tolist()
    
    # Use Gaussian elimination
    return gaussian_elimination(A, b)


def gaussian_elimination(A, b):
    """
    Solve linear system using Gaussian elimination with partial pivoting.
    """
    n = len(A)
    
    # Create augmented matrix
    augmented = []
    for i in range(n):
        row = A[i][:] + [b[i]]
        augmented.append(row)
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Swap rows
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Check for singular matrix
        if abs(augmented[i][i]) < 1e-10:
            raise ValueError("Matrix is singular")
        
        # Eliminate below pivot
        for k in range(i + 1, n):
            factor = augmented[k][i] / augmented[i][i]
            for j in range(i, n + 1):
                augmented[k][j] -= factor * augmented[i][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
        x[i] /= augmented[i][i]
    
    return x


# Testing and validation functions
def comprehensive_test():
    """
    Comprehensive test suite for all implemented methods.
    """
    print("Running comprehensive test suite...")
    
    test_cases = [
        {
            'name': 'Simple 2x2 system',
            'A': [[2, 1], [1, 1]],
            'b': [3, 2],
            'expected': [1, 1]
        },
        {
            'name': 'Simple 3x3 system',
            'A': [[1, 2, 3], [0, 1, 4], [5, 6, 0]],
            'b': [14, 8, 11],
            'expected': [1, 1, 1]
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['name']}")
        
        # Test NumPy solution
        try:
            x_numpy = solve_linear_system_numpy(test['A'], test['b'])
            error_numpy = np.linalg.norm(x_numpy - test['expected'])
            print(f"  NumPy solution: {x_numpy}, Error: {error_numpy:.2e}")
        except Exception as e:
            print(f"  NumPy failed: {e}")
        
        # Test custom solution
        try:
            x_custom = solve_without_numpy(test['A'], test['b'])
            error_custom = math.sqrt(sum((x_custom[j] - test['expected'][j])**2 
                                       for j in range(len(x_custom))))
            print(f"  Custom solution: {x_custom}, Error: {error_custom:.2e}")
        except Exception as e:
            print(f"  Custom failed: {e}")
```

**Key Features:**
- **Multiple NumPy Methods**: Direct solve, matrix inverse, LU, QR, SVD, least squares
- **Iterative Methods**: Jacobi and Gauss-Seidel implementations
- **System Analysis**: Comprehensive analysis including condition number, rank, system type
- **Special Cases**: Handling of triangular, symmetric, and positive definite matrices
- **Error Handling**: Robust error detection and alternative methods
- **Verification**: Automatic solution verification and residual analysis

**Method Comparison:**
- **np.linalg.solve()**: Most reliable general-purpose method
- **Matrix Inverse**: Less stable but educational
- **LU Decomposition**: Efficient for multiple right-hand sides
- **QR Decomposition**: More stable for ill-conditioned systems
- **SVD**: Most robust for rank-deficient systems
- **Iterative Methods**: Memory efficient for large sparse systems

**Best Practices:**
- Always analyze the system before choosing a method
- Check condition number for numerical stability
- Use least squares for overdetermined systems
- Verify solutions when numerical precision is critical
- Use appropriate method based on matrix properties (symmetric, sparse, etc.)

---

## Question 7

**Implement a function to calculate the L2 norm of a vector.**

**Answer:** Here's a comprehensive implementation of L2 norm calculation with multiple approaches and optimizations:

```python
import math
import numpy as np

def l2_norm_basic(vector):
    """
    Calculate L2 norm using basic Python implementation.
    
    Args:
        vector: List or array of numbers
    
    Returns:
        L2 norm (Euclidean norm) of the vector
    
    Formula: ||v||₂ = √(v₁² + v₂² + ... + vₙ²)
    """
    if not vector:
        return 0.0
    
    # Calculate sum of squares
    sum_of_squares = sum(x * x for x in vector)
    
    # Return square root
    return math.sqrt(sum_of_squares)


def l2_norm_robust(vector):
    """
    Numerically robust L2 norm calculation to avoid overflow/underflow.
    
    Args:
        vector: List or array of numbers
    
    Returns:
        L2 norm with improved numerical stability
    """
    if not vector:
        return 0.0
    
    # Find maximum absolute value to scale the vector
    max_val = max(abs(x) for x in vector)
    
    if max_val == 0:
        return 0.0
    
    # Scale vector by max value to prevent overflow
    scaled_sum = sum((x / max_val) ** 2 for x in vector)
    
    # Scale back the result
    return max_val * math.sqrt(scaled_sum)


def l2_norm_iterative(vector):
    """
    Iterative approach for very large vectors to save memory.
    
    Args:
        vector: Iterable of numbers (can be generator)
    
    Returns:
        L2 norm calculated iteratively
    """
    sum_of_squares = 0.0
    count = 0
    
    for x in vector:
        sum_of_squares += x * x
        count += 1
    
    if count == 0:
        return 0.0
    
    return math.sqrt(sum_of_squares)


def l2_norm_numpy(vector):
    """
    Calculate L2 norm using NumPy (most efficient for large vectors).
    
    Args:
        vector: Array-like input
    
    Returns:
        L2 norm using NumPy's optimized implementation
    """
    return np.linalg.norm(vector, ord=2)


def l2_norm_manual_numpy(vector):
    """
    Manual L2 norm calculation using NumPy operations.
    
    Args:
        vector: Array-like input
    
    Returns:
        L2 norm calculated manually with NumPy
    """
    v = np.array(vector)
    return np.sqrt(np.sum(v * v))


def l2_norm_optimized(vector):
    """
    Optimized L2 norm with special case handling.
    
    Args:
        vector: List, tuple, or array of numbers
    
    Returns:
        L2 norm with optimizations for common cases
    """
    # Handle empty vector
    if not vector:
        return 0.0
    
    # Convert to list if needed
    if hasattr(vector, 'tolist'):
        vector = vector.tolist()
    
    n = len(vector)
    
    # Special cases for small vectors
    if n == 1:
        return abs(vector[0])
    elif n == 2:
        return math.sqrt(vector[0]**2 + vector[1]**2)
    elif n == 3:
        return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    
    # General case with overflow protection
    return l2_norm_robust(vector)


def l2_norm_vectorized(vectors):
    """
    Calculate L2 norm for multiple vectors efficiently.
    
    Args:
        vectors: 2D array where each row is a vector
    
    Returns:
        Array of L2 norms for each vector
    """
    vectors = np.array(vectors)
    
    if vectors.ndim == 1:
        # Single vector case
        return np.linalg.norm(vectors)
    else:
        # Multiple vectors case
        return np.linalg.norm(vectors, axis=1)


def l2_norm_complex(vector):
    """
    Calculate L2 norm for complex vectors.
    
    Args:
        vector: List or array of complex numbers
    
    Returns:
        L2 norm of complex vector
    """
    if not vector:
        return 0.0
    
    # For complex numbers: |z|² = z * conjugate(z) = real² + imag²
    sum_of_squares = sum(abs(x)**2 for x in vector)
    
    return math.sqrt(sum_of_squares)


def l2_norm_with_weights(vector, weights):
    """
    Calculate weighted L2 norm.
    
    Args:
        vector: Input vector
        weights: Weight vector (same length as vector)
    
    Returns:
        Weighted L2 norm: √(Σ wᵢ * vᵢ²)
    """
    if len(vector) != len(weights):
        raise ValueError("Vector and weights must have same length")
    
    if not vector:
        return 0.0
    
    weighted_sum = sum(w * x * x for w, x in zip(weights, vector))
    
    return math.sqrt(weighted_sum)


class VectorNormCalculator:
    """
    Advanced vector norm calculator supporting multiple norm types.
    """
    
    @staticmethod
    def l2_norm(vector, robust=True):
        """Calculate L2 norm with optional robustness."""
        if robust:
            return l2_norm_robust(vector)
        else:
            return l2_norm_basic(vector)
    
    @staticmethod
    def lp_norm(vector, p=2):
        """
        Calculate Lp norm: (Σ |xᵢ|^p)^(1/p)
        
        Args:
            vector: Input vector
            p: Norm order (default: 2 for L2 norm)
        
        Returns:
            Lp norm of the vector
        """
        if not vector:
            return 0.0
        
        if p == 1:
            # L1 norm (Manhattan distance)
            return sum(abs(x) for x in vector)
        elif p == 2:
            # L2 norm (Euclidean distance)
            return l2_norm_robust(vector)
        elif p == float('inf'):
            # L∞ norm (Maximum norm)
            return max(abs(x) for x in vector)
        else:
            # General Lp norm
            sum_powers = sum(abs(x)**p for x in vector)
            return sum_powers**(1.0/p)
    
    @staticmethod
    def frobenius_norm(matrix):
        """
        Calculate Frobenius norm of a matrix (L2 norm of flattened matrix).
        
        Args:
            matrix: 2D array or list of lists
        
        Returns:
            Frobenius norm
        """
        flat_vector = []
        for row in matrix:
            flat_vector.extend(row)
        
        return l2_norm_robust(flat_vector)


def compare_norm_methods(vector, iterations=1000):
    """
    Compare performance and accuracy of different L2 norm methods.
    
    Args:
        vector: Test vector
        iterations: Number of iterations for timing
    
    Returns:
        Dictionary with results and timing information
    """
    import time
    
    methods = [
        ("Basic Python", l2_norm_basic),
        ("Robust Python", l2_norm_robust),
        ("Optimized", l2_norm_optimized),
        ("NumPy linalg.norm", l2_norm_numpy),
        ("Manual NumPy", l2_norm_manual_numpy),
    ]
    
    results = {}
    
    for name, method in methods:
        try:
            # Test correctness
            result = method(vector)
            
            # Time performance
            start_time = time.time()
            for _ in range(iterations):
                method(vector)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations
            
            results[name] = {
                'result': result,
                'avg_time_ms': avg_time * 1000,
                'status': 'Success'
            }
            
        except Exception as e:
            results[name] = {
                'result': None,
                'avg_time_ms': None,
                'status': f'Error: {e}'
            }
    
    return results


def verify_norm_properties(vector):
    """
    Verify mathematical properties of L2 norm.
    
    Args:
        vector: Test vector
    
    Returns:
        Dictionary with property verification results
    """
    norm = l2_norm_robust(vector)
    
    properties = {}
    
    # Property 1: Non-negativity (||v|| ≥ 0)
    properties['non_negative'] = norm >= 0
    
    # Property 2: Zero vector has zero norm
    if all(x == 0 for x in vector):
        properties['zero_vector'] = norm == 0
    else:
        properties['zero_vector'] = norm > 0
    
    # Property 3: Scaling property (||cv|| = |c| * ||v||)
    if vector:
        c = 3.14159
        scaled_vector = [c * x for x in vector]
        scaled_norm = l2_norm_robust(scaled_vector)
        expected_scaled_norm = abs(c) * norm
        properties['scaling'] = abs(scaled_norm - expected_scaled_norm) < 1e-10
    else:
        properties['scaling'] = True
    
    # Property 4: Triangle inequality (||u + v|| ≤ ||u|| + ||v||)
    if len(vector) >= 2:
        # Split vector into two parts
        mid = len(vector) // 2
        u = vector[:mid] + [0] * (len(vector) - mid)
        v = [0] * mid + vector[mid:]
        
        norm_u = l2_norm_robust(u)
        norm_v = l2_norm_robust(v)
        norm_sum = l2_norm_robust([u[i] + v[i] for i in range(len(vector))])
        
        properties['triangle_inequality'] = norm_sum <= norm_u + norm_v + 1e-10
    else:
        properties['triangle_inequality'] = True
    
    return properties


# Example usage and demonstrations
if __name__ == "__main__":
    print("="*60)
    print("L2 NORM CALCULATION IMPLEMENTATIONS")
    print("="*60)
    
    # Test vectors
    test_vectors = [
        # Simple cases
        [3, 4],                    # Should be 5
        [1, 1, 1],                 # Should be √3
        [0, 0, 0],                 # Should be 0
        [5],                       # Should be 5
        
        # Larger vectors
        list(range(1, 11)),        # [1, 2, 3, ..., 10]
        [1e10, 1e10, 1e10],        # Large numbers (test overflow)
        [1e-10, 1e-10, 1e-10],     # Small numbers (test underflow)
        
        # Random vector
        [2.5, -1.7, 3.14, -0.8, 1.2],
    ]
    
    for i, vector in enumerate(test_vectors):
        print(f"\n--- Test Vector {i+1}: {vector} ---")
        
        # Calculate using different methods
        basic_norm = l2_norm_basic(vector)
        robust_norm = l2_norm_robust(vector)
        optimized_norm = l2_norm_optimized(vector)
        
        print(f"Basic implementation:     {basic_norm:.10f}")
        print(f"Robust implementation:    {robust_norm:.10f}")
        print(f"Optimized implementation: {optimized_norm:.10f}")
        
        # NumPy comparison
        try:
            numpy_norm = l2_norm_numpy(vector)
            print(f"NumPy implementation:     {numpy_norm:.10f}")
            
            # Check consistency
            max_diff = max(
                abs(basic_norm - numpy_norm),
                abs(robust_norm - numpy_norm),
                abs(optimized_norm - numpy_norm)
            )
            
            if max_diff < 1e-10:
                print("✓ All methods agree")
            else:
                print(f"⚠ Methods disagree, max difference: {max_diff:.2e}")
                
        except Exception as e:
            print(f"NumPy failed: {e}")
        
        # Verify mathematical properties
        properties = verify_norm_properties(vector)
        print(f"Properties: {properties}")
    
    # Performance comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Large vector for timing
    large_vector = list(range(10000))
    
    performance_results = compare_norm_methods(large_vector, iterations=100)
    
    print(f"Vector size: {len(large_vector)}")
    print(f"{'Method':<20} {'Result':<15} {'Time (ms)':<12} {'Status'}")
    print("-" * 60)
    
    for method, data in performance_results.items():
        result_str = f"{data['result']:.6f}" if data['result'] is not None else "N/A"
        time_str = f"{data['avg_time_ms']:.3f}" if data['avg_time_ms'] is not None else "N/A"
        print(f"{method:<20} {result_str:<15} {time_str:<12} {data['status']}")
    
    # Special cases
    print(f"\n{'='*60}")
    print("SPECIAL CASES")
    print("="*60)
    
    # Complex numbers
    complex_vector = [1+2j, 3-4j, 2+1j]
    complex_norm = l2_norm_complex(complex_vector)
    print(f"Complex vector {complex_vector}: norm = {complex_norm:.6f}")
    
    # Weighted norm
    vector = [1, 2, 3, 4]
    weights = [2, 1, 3, 1]
    weighted_norm = l2_norm_with_weights(vector, weights)
    print(f"Weighted norm of {vector} with weights {weights}: {weighted_norm:.6f}")
    
    # Multiple vectors
    vectors_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    norms_vectorized = l2_norm_vectorized(vectors_2d)
    print(f"Multiple vectors {vectors_2d}:")
    for i, norm in enumerate(norms_vectorized):
        print(f"  Vector {i+1} norm: {norm:.6f}")
    
    # Different Lp norms
    print(f"\n{'='*60}")
    print("DIFFERENT LP NORMS")
    print("="*60)
    
    test_vector = [1, -2, 3, -4]
    calc = VectorNormCalculator()
    
    for p in [1, 2, 3, float('inf')]:
        lp_norm = calc.lp_norm(test_vector, p)
        print(f"L{p} norm of {test_vector}: {lp_norm:.6f}")
    
    # Matrix Frobenius norm
    test_matrix = [[1, 2, 3], [4, 5, 6]]
    frobenius = calc.frobenius_norm(test_matrix)
    print(f"Frobenius norm of {test_matrix}: {frobenius:.6f}")
    
    # Verify with known examples
    print(f"\n{'='*60}")
    print("VERIFICATION WITH KNOWN EXAMPLES")
    print("="*60)
    
    known_examples = [
        ([3, 4], 5.0, "3-4-5 triangle"),
        ([1, 1], math.sqrt(2), "Unit square diagonal"),
        ([1, 1, 1], math.sqrt(3), "Unit cube diagonal"),
        ([0, 0, 0, 0], 0.0, "Zero vector"),
        ([5], 5.0, "Single element"),
    ]
    
    for vector, expected, description in known_examples:
        calculated = l2_norm_robust(vector)
        error = abs(calculated - expected)
        status = "✓" if error < 1e-10 else "✗"
        print(f"{status} {description}: {vector} → {calculated:.10f} (expected: {expected:.10f})")


# Advanced applications
def distance_between_points(point1, point2):
    """
    Calculate Euclidean distance between two points using L2 norm.
    
    Args:
        point1, point2: Points as lists/tuples of coordinates
    
    Returns:
        Euclidean distance between the points
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same dimension")
    
    difference_vector = [p1 - p2 for p1, p2 in zip(point1, point2)]
    return l2_norm_robust(difference_vector)


def normalize_vector(vector, target_norm=1.0):
    """
    Normalize vector to have specified L2 norm.
    
    Args:
        vector: Input vector
        target_norm: Desired norm (default: 1.0 for unit vector)
    
    Returns:
        Normalized vector
    """
    current_norm = l2_norm_robust(vector)
    
    if current_norm == 0:
        raise ValueError("Cannot normalize zero vector")
    
    scaling_factor = target_norm / current_norm
    return [x * scaling_factor for x in vector]


def vector_similarity_cosine(v1, v2):
    """
    Calculate cosine similarity between two vectors using L2 norms.
    
    Args:
        v1, v2: Input vectors
    
    Returns:
        Cosine similarity: (v1 · v2) / (||v1|| * ||v2||)
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same length")
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = l2_norm_robust(v1)
    norm2 = l2_norm_robust(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
```

**Key Features:**
- **Multiple Implementations**: Basic, robust, optimized, and NumPy versions
- **Numerical Stability**: Overflow/underflow protection for extreme values
- **Performance Optimization**: Special cases for small vectors and vectorized operations
- **Complex Number Support**: Handles complex vectors correctly
- **Weighted Norms**: Support for weighted L2 norm calculations
- **Mathematical Verification**: Built-in property verification functions

**Implementation Comparison:**
- **Basic**: Simple and readable, suitable for small vectors
- **Robust**: Numerically stable, handles extreme values
- **NumPy**: Fastest for large vectors, highly optimized
- **Optimized**: Good balance of speed and stability

**Applications:**
- **Machine Learning**: Feature normalization, distance calculations
- **Computer Graphics**: Vector normalization, length calculations
- **Signal Processing**: Energy calculations, RMS values
- **Statistics**: Standard deviation, variance calculations

**Best Practices:**
- Use robust implementation for numerical stability
- Use NumPy for large vectors and performance-critical applications
- Always handle zero vectors appropriately
- Consider weighted norms for specialized applications

---

## Question 8

**Write a program to verify if a given square matrix is orthogonal.**

**Answer:** Here's a comprehensive program to verify orthogonal matrices with multiple verification methods:

```python
import math
import numpy as np

def is_orthogonal_basic(matrix, tolerance=1e-10):
    """
    Basic verification: A matrix is orthogonal if A^T * A = I
    
    Args:
        matrix: Square matrix as list of lists
        tolerance: Numerical tolerance for comparison
    
    Returns:
        Boolean indicating if matrix is orthogonal
    """
    n = len(matrix)
    
    # Verify square matrix
    if any(len(row) != n for row in matrix):
        return False
    
    # Calculate transpose
    transpose = [[matrix[j][i] for j in range(n)] for i in range(n)]
    
    # Calculate A^T * A
    product = matrix_multiply(transpose, matrix)
    
    # Check if result is identity matrix
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(product[i][j] - expected) > tolerance:
                return False
    
    return True


def is_orthogonal_detailed(matrix, tolerance=1e-10):
    """
    Detailed orthogonal matrix verification with comprehensive analysis.
    
    Args:
        matrix: Square matrix
        tolerance: Numerical tolerance
    
    Returns:
        Dictionary with detailed verification results
    """
    result = {
        'is_orthogonal': False,
        'is_square': False,
        'determinant': None,
        'condition_number': None,
        'orthogonality_error': None,
        'column_norms': [],
        'row_norms': [],
        'dot_products': [],
        'verification_methods': {}
    }
    
    n = len(matrix)
    
    # Check if square
    result['is_square'] = all(len(row) == n for row in matrix)
    if not result['is_square']:
        return result
    
    # Method 1: A^T * A = I
    transpose = [[matrix[j][i] for j in range(n)] for i in range(n)]
    ata_product = matrix_multiply(transpose, matrix)
    
    orthogonality_error = 0.0
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            error = abs(ata_product[i][j] - expected)
            orthogonality_error = max(orthogonality_error, error)
    
    result['orthogonality_error'] = orthogonality_error
    result['verification_methods']['transpose_product'] = orthogonality_error < tolerance
    
    # Method 2: A * A^T = I
    aat_product = matrix_multiply(matrix, transpose)
    aat_error = 0.0
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            error = abs(aat_product[i][j] - expected)
            aat_error = max(aat_error, error)
    
    result['verification_methods']['product_transpose'] = aat_error < tolerance
    
    # Method 3: Check column orthonormality
    columns = [[matrix[i][j] for i in range(n)] for j in range(n)]
    
    # Column norms should be 1
    column_norms = [vector_norm(col) for col in columns]
    result['column_norms'] = column_norms
    
    norms_ok = all(abs(norm - 1.0) < tolerance for norm in column_norms)
    result['verification_methods']['column_norms'] = norms_ok
    
    # Column dot products should be 0 (except with themselves)
    dot_products = []
    for i in range(n):
        for j in range(i + 1, n):
            dot_prod = vector_dot_product(columns[i], columns[j])
            dot_products.append(abs(dot_prod))
    
    result['dot_products'] = dot_products
    orthogonal_columns = all(dot_prod < tolerance for dot_prod in dot_products)
    result['verification_methods']['column_orthogonality'] = orthogonal_columns
    
    # Method 4: Check row orthonormality
    row_norms = [vector_norm(row) for row in matrix]
    result['row_norms'] = row_norms
    
    row_norms_ok = all(abs(norm - 1.0) < tolerance for norm in row_norms)
    result['verification_methods']['row_norms'] = row_norms_ok
    
    # Method 5: Determinant should be ±1
    try:
        det = determinant(matrix)
        result['determinant'] = det
        result['verification_methods']['determinant'] = abs(abs(det) - 1.0) < tolerance
    except:
        result['verification_methods']['determinant'] = False
    
    # Method 6: Condition number should be 1
    try:
        cond = condition_number(matrix)
        result['condition_number'] = cond
        result['verification_methods']['condition_number'] = abs(cond - 1.0) < tolerance * 1000
    except:
        result['verification_methods']['condition_number'] = False
    
    # Overall result
    result['is_orthogonal'] = all(result['verification_methods'].values())
    
    return result


def is_orthogonal_numpy(matrix, tolerance=1e-10):
    """
    Verify orthogonal matrix using NumPy for better numerical stability.
    
    Args:
        matrix: Input matrix
        tolerance: Numerical tolerance
    
    Returns:
        Boolean and detailed analysis
    """
    A = np.array(matrix, dtype=float)
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        return False, "Matrix is not square"
    
    # Method 1: A^T @ A should be identity
    AtA = A.T @ A
    I = np.eye(n)
    error_AtA = np.linalg.norm(AtA - I, 'fro')
    
    # Method 2: A @ A^T should be identity
    AAt = A @ A.T
    error_AAt = np.linalg.norm(AAt - I, 'fro')
    
    # Method 3: Determinant should be ±1
    det = np.linalg.det(A)
    
    # Method 4: All singular values should be 1
    singular_values = np.linalg.svd(A, compute_uv=False)
    svd_error = np.linalg.norm(singular_values - 1.0)
    
    analysis = {
        'error_AtA': error_AtA,
        'error_AAt': error_AAt,
        'determinant': det,
        'svd_error': svd_error,
        'singular_values': singular_values
    }
    
    is_orthogonal = (error_AtA < tolerance and 
                    error_AAt < tolerance and 
                    abs(abs(det) - 1.0) < tolerance and
                    svd_error < tolerance)
    
    return is_orthogonal, analysis


def generate_orthogonal_matrix(n, method="qr"):
    """
    Generate a random orthogonal matrix for testing.
    
    Args:
        n: Size of the matrix
        method: Generation method ("qr", "rotation", "householder")
    
    Returns:
        n×n orthogonal matrix
    """
    if method == "qr":
        # QR decomposition of random matrix
        random_matrix = np.random.randn(n, n)
        Q, R = np.linalg.qr(random_matrix)
        # Ensure proper orthogonal matrix (det = +1)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q.tolist()
    
    elif method == "rotation":
        # Generate rotation matrix (for 2D case)
        if n != 2:
            raise ValueError("Rotation method only works for 2×2 matrices")
        
        angle = np.random.uniform(0, 2 * np.pi)
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        
        return [[cos_theta, -sin_theta],
                [sin_theta, cos_theta]]
    
    elif method == "householder":
        # Start with identity and apply random Householder reflections
        Q = np.eye(n)
        
        for _ in range(n):
            # Random unit vector
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            
            # Householder matrix H = I - 2vv^T
            H = np.eye(n) - 2 * np.outer(v, v)
            Q = Q @ H
        
        return Q.tolist()


def create_test_matrices():
    """
    Create a variety of test matrices for verification.
    
    Returns:
        List of (matrix, expected_result, description) tuples
    """
    test_cases = []
    
    # Identity matrix (trivially orthogonal)
    identity_2x2 = [[1, 0], [0, 1]]
    test_cases.append((identity_2x2, True, "2×2 Identity matrix"))
    
    identity_3x3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    test_cases.append((identity_3x3, True, "3×3 Identity matrix"))
    
    # Basic rotation matrix
    cos45 = math.cos(math.pi/4)
    sin45 = math.sin(math.pi/4)
    rotation_45 = [[cos45, -sin45], [sin45, cos45]]
    test_cases.append((rotation_45, True, "45° rotation matrix"))
    
    # Reflection matrix
    reflection = [[1, 0], [0, -1]]
    test_cases.append((reflection, True, "Reflection matrix"))
    
    # Permutation matrix
    permutation = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    test_cases.append((permutation, True, "Permutation matrix"))
    
    # Hadamard matrix (scaled)
    hadamard_2x2 = [[1/math.sqrt(2), 1/math.sqrt(2)], 
                    [1/math.sqrt(2), -1/math.sqrt(2)]]
    test_cases.append((hadamard_2x2, True, "2×2 Hadamard matrix"))
    
    # Non-orthogonal matrices
    non_orthogonal_1 = [[1, 1], [0, 1]]
    test_cases.append((non_orthogonal_1, False, "Upper triangular (non-orthogonal)"))
    
    non_orthogonal_2 = [[2, 0], [0, 1]]
    test_cases.append((non_orthogonal_2, False, "Scaled identity (non-orthogonal)"))
    
    non_orthogonal_3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    test_cases.append((non_orthogonal_3, False, "Random matrix (non-orthogonal)"))
    
    # Nearly orthogonal (with numerical errors)
    nearly_orthogonal = [[1.0001, 0], [0, 0.9999]]
    test_cases.append((nearly_orthogonal, False, "Nearly orthogonal matrix"))
    
    return test_cases


class OrthogonalMatrixVerifier:
    """
    Advanced orthogonal matrix verification class with multiple methods.
    """
    
    def __init__(self, tolerance=1e-10):
        self.tolerance = tolerance
        self.verification_history = []
    
    def verify(self, matrix, method="comprehensive"):
        """
        Verify if matrix is orthogonal using specified method.
        
        Args:
            matrix: Input matrix
            method: "basic", "detailed", "numpy", or "comprehensive"
        
        Returns:
            Verification result
        """
        if method == "basic":
            result = is_orthogonal_basic(matrix, self.tolerance)
        elif method == "detailed":
            result = is_orthogonal_detailed(matrix, self.tolerance)
        elif method == "numpy":
            result = is_orthogonal_numpy(matrix, self.tolerance)
        elif method == "comprehensive":
            result = self._comprehensive_verify(matrix)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store in history
        self.verification_history.append({
            'matrix_hash': hash(str(matrix)),
            'method': method,
            'result': result,
            'timestamp': self._get_timestamp()
        })
        
        return result
    
    def _comprehensive_verify(self, matrix):
        """
        Comprehensive verification using all available methods.
        """
        results = {}
        
        results['basic'] = is_orthogonal_basic(matrix, self.tolerance)
        results['detailed'] = is_orthogonal_detailed(matrix, self.tolerance)
        
        try:
            numpy_result, numpy_analysis = is_orthogonal_numpy(matrix, self.tolerance)
            results['numpy'] = {'is_orthogonal': numpy_result, 'analysis': numpy_analysis}
        except Exception as e:
            results['numpy'] = {'error': str(e)}
        
        # Consensus result
        orthogonal_votes = []
        
        if isinstance(results['basic'], bool):
            orthogonal_votes.append(results['basic'])
        
        if isinstance(results['detailed'], dict):
            orthogonal_votes.append(results['detailed']['is_orthogonal'])
        
        if 'is_orthogonal' in results.get('numpy', {}):
            orthogonal_votes.append(results['numpy']['is_orthogonal'])
        
        consensus = sum(orthogonal_votes) > len(orthogonal_votes) / 2
        
        return {
            'consensus': consensus,
            'individual_results': results,
            'vote_count': sum(orthogonal_votes),
            'total_methods': len(orthogonal_votes)
        }
    
    def _get_timestamp(self):
        """Get current timestamp for logging."""
        import time
        return time.time()
    
    def get_statistics(self):
        """Get verification statistics."""
        if not self.verification_history:
            return "No verifications performed yet."
        
        total_verifications = len(self.verification_history)
        methods_used = set(entry['method'] for entry in self.verification_history)
        
        return {
            'total_verifications': total_verifications,
            'methods_used': list(methods_used),
            'recent_results': self.verification_history[-5:]  # Last 5 results
        }


# Utility functions
def matrix_multiply(A, B):
    """Multiply two matrices."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result


def vector_norm(vector):
    """Calculate L2 norm of vector."""
    return math.sqrt(sum(x * x for x in vector))


def vector_dot_product(v1, v2):
    """Calculate dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def determinant(matrix):
    """Calculate determinant using recursive expansion."""
    n = len(matrix)
    
    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for j in range(n):
            minor = get_minor(matrix, 0, j)
            cofactor = ((-1) ** j) * matrix[0][j] * determinant(minor)
            det += cofactor
        return det


def get_minor(matrix, row, col):
    """Get minor matrix by removing specified row and column."""
    return [[matrix[i][j] for j in range(len(matrix[0])) if j != col]
            for i in range(len(matrix)) if i != row]


def condition_number(matrix):
    """Calculate condition number (rough estimate)."""
    # This is a simplified version - in practice, use NumPy
    try:
        A = np.array(matrix)
        return np.linalg.cond(A)
    except:
        return float('inf')


# Main demonstration program
def main():
    """
    Main program demonstrating orthogonal matrix verification.
    """
    print("="*70)
    print("ORTHOGONAL MATRIX VERIFICATION PROGRAM")
    print("="*70)
    
    # Create verifier instance
    verifier = OrthogonalMatrixVerifier(tolerance=1e-10)
    
    # Test with predefined matrices
    test_matrices = create_test_matrices()
    
    print("\n1. TESTING KNOWN MATRICES")
    print("-" * 50)
    
    for matrix, expected, description in test_matrices:
        print(f"\nTesting: {description}")
        print(f"Matrix: {matrix}")
        
        # Basic verification
        basic_result = is_orthogonal_basic(matrix)
        
        # Detailed verification
        detailed_result = is_orthogonal_detailed(matrix)
        
        # NumPy verification
        try:
            numpy_result, numpy_analysis = is_orthogonal_numpy(matrix)
            
            print(f"Expected: {expected}")
            print(f"Basic method: {basic_result}")
            print(f"Detailed method: {detailed_result['is_orthogonal']}")
            print(f"NumPy method: {numpy_result}")
            
            # Show detailed analysis for interesting cases
            if not detailed_result['is_orthogonal']:
                print(f"  Orthogonality error: {detailed_result['orthogonality_error']:.2e}")
                if detailed_result['determinant'] is not None:
                    print(f"  Determinant: {detailed_result['determinant']:.6f}")
            
            # Verify against expected result
            status = "✓" if basic_result == expected else "✗"
            print(f"Result: {status}")
            
        except Exception as e:
            print(f"Error in verification: {e}")
    
    # Test with randomly generated orthogonal matrices
    print(f"\n\n2. TESTING RANDOMLY GENERATED ORTHOGONAL MATRICES")
    print("-" * 60)
    
    for size in [2, 3, 4]:
        print(f"\nGenerating {size}×{size} orthogonal matrix:")
        
        try:
            # Generate using QR decomposition
            ortho_matrix = generate_orthogonal_matrix(size, method="qr")
            
            # Verify
            numpy_result, numpy_analysis = is_orthogonal_numpy(ortho_matrix)
            
            print(f"Generated matrix (first row): {ortho_matrix[0]}")
            print(f"Is orthogonal: {numpy_result}")
            print(f"A^T*A error: {numpy_analysis['error_AtA']:.2e}")
            print(f"Determinant: {numpy_analysis['determinant']:.6f}")
            print(f"Singular values: {[f'{sv:.6f}' for sv in numpy_analysis['singular_values']]}")
            
        except Exception as e:
            print(f"Error generating/verifying {size}×{size} matrix: {e}")
    
    # Performance comparison
    print(f"\n\n3. PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Large matrix for timing
    large_matrix = generate_orthogonal_matrix(10, method="qr")
    
    import time
    
    methods = [
        ("Basic Python", lambda m: is_orthogonal_basic(m)),
        ("Detailed Analysis", lambda m: is_orthogonal_detailed(m)),
        ("NumPy Method", lambda m: is_orthogonal_numpy(m)[0]),
    ]
    
    iterations = 100
    
    for method_name, method_func in methods:
        start_time = time.time()
        
        for _ in range(iterations):
            try:
                result = method_func(large_matrix)
            except:
                result = None
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations * 1000  # ms
        
        print(f"{method_name:20}: {avg_time:.3f} ms/call")
    
    # Special orthogonal matrices
    print(f"\n\n4. SPECIAL ORTHOGONAL MATRICES")
    print("-" * 45)
    
    # Rotation matrices for different angles
    angles = [0, 30, 45, 90, 180]
    
    for angle in angles:
        theta = math.radians(angle)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        rotation_matrix = [[cos_theta, -sin_theta],
                          [sin_theta, cos_theta]]
        
        is_ortho = is_orthogonal_basic(rotation_matrix)
        det = determinant(rotation_matrix)
        
        print(f"Rotation {angle:3d}°: orthogonal={is_ortho}, det={det:.6f}")
    
    # Householder reflectors
    print(f"\nHouseholder reflectors:")
    
    for i in range(2, 5):
        # Random unit vector
        v = [1.0] + [0.1 * j for j in range(i-1)]
        norm_v = vector_norm(v)
        v = [x / norm_v for x in v]  # Normalize
        
        # Householder matrix H = I - 2vv^T
        H = [[0.0 for _ in range(i)] for _ in range(i)]
        
        for row in range(i):
            for col in range(i):
                if row == col:
                    H[row][col] = 1.0 - 2 * v[row] * v[col]
                else:
                    H[row][col] = -2 * v[row] * v[col]
        
        is_ortho = is_orthogonal_basic(H)
        det = determinant(H)
        
        print(f"  {i}×{i} Householder: orthogonal={is_ortho}, det={det:.6f}")
    
    # Summary
    print(f"\n\n5. VERIFICATION SUMMARY")
    print("-" * 35)
    
    stats = verifier.get_statistics()
    print(f"Total verifications performed: {stats['total_verifications']}")
    print(f"Methods used: {', '.join(stats['methods_used'])}")


if __name__ == "__main__":
    main()


# Additional specialized functions
def is_special_orthogonal(matrix, tolerance=1e-10):
    """
    Check if matrix is special orthogonal (orthogonal with determinant +1).
    
    Args:
        matrix: Square matrix
        tolerance: Numerical tolerance
    
    Returns:
        Boolean indicating if matrix is special orthogonal
    """
    if not is_orthogonal_basic(matrix, tolerance):
        return False
    
    det = determinant(matrix)
    return abs(det - 1.0) < tolerance


def orthogonalize_matrix(matrix, method="gram_schmidt"):
    """
    Orthogonalize a matrix using specified method.
    
    Args:
        matrix: Input matrix
        method: "gram_schmidt" or "qr"
    
    Returns:
        Orthogonalized matrix
    """
    if method == "gram_schmidt":
        return gram_schmidt_orthogonalization(matrix)
    elif method == "qr":
        # Use QR decomposition to get orthogonal matrix
        A = np.array(matrix)
        Q, R = np.linalg.qr(A)
        return Q.tolist()
    else:
        raise ValueError(f"Unknown orthogonalization method: {method}")


def gram_schmidt_orthogonalization(matrix):
    """
    Perform Gram-Schmidt orthogonalization on matrix columns.
    
    Args:
        matrix: Input matrix (columns will be orthogonalized)
    
    Returns:
        Matrix with orthogonal columns
    """
    n = len(matrix)
    m = len(matrix[0])
    
    # Extract columns
    columns = [[matrix[i][j] for i in range(n)] for j in range(m)]
    
    # Orthogonalize columns
    orthogonal_columns = []
    
    for j in range(m):
        # Start with original column
        v = columns[j][:]
        
        # Subtract projections onto previous orthogonal vectors
        for k in range(j):
            u_k = orthogonal_columns[k]
            projection_coeff = vector_dot_product(v, u_k) / vector_dot_product(u_k, u_k)
            
            for i in range(n):
                v[i] -= projection_coeff * u_k[i]
        
        # Normalize
        norm_v = vector_norm(v)
        if norm_v > 1e-10:
            v = [x / norm_v for x in v]
        
        orthogonal_columns.append(v)
    
    # Reconstruct matrix
    result = [[orthogonal_columns[j][i] for j in range(m)] for i in range(n)]
    
    return result


def matrix_properties_analysis(matrix):
    """
    Comprehensive analysis of matrix properties related to orthogonality.
    
    Args:
        matrix: Input matrix
    
    Returns:
        Dictionary with various matrix properties
    """
    properties = {}
    
    try:
        A = np.array(matrix, dtype=float)
        
        properties['shape'] = A.shape
        properties['is_square'] = A.shape[0] == A.shape[1]
        
        if properties['is_square']:
            properties['determinant'] = np.linalg.det(A)
            properties['is_singular'] = abs(properties['determinant']) < 1e-10
            
            if not properties['is_singular']:
                properties['condition_number'] = np.linalg.cond(A)
                properties['is_well_conditioned'] = properties['condition_number'] < 1e12
        
        # Singular value decomposition
        U, s, Vt = np.linalg.svd(A)
        properties['singular_values'] = s.tolist()
        properties['rank'] = np.sum(s > 1e-10)
        properties['is_full_rank'] = properties['rank'] == min(A.shape)
        
        # Frobenius norm
        properties['frobenius_norm'] = np.linalg.norm(A, 'fro')
        
        # Check if symmetric
        if properties['is_square']:
            properties['is_symmetric'] = np.allclose(A, A.T, atol=1e-10)
        
        # Check if orthogonal
        properties['orthogonal_analysis'] = is_orthogonal_detailed(matrix)
        
    except Exception as e:
        properties['error'] = str(e)
    
    return properties
```

**Key Features:**
- **Multiple Verification Methods**: Basic, detailed, NumPy-based, and comprehensive approaches
- **Numerical Stability**: Robust handling of floating-point precision issues
- **Comprehensive Analysis**: Detailed breakdown of orthogonality properties
- **Test Matrix Generation**: Create orthogonal matrices for testing
- **Performance Comparison**: Benchmark different verification methods
- **Special Cases**: Handle rotation, reflection, permutation, and Householder matrices

**Orthogonal Matrix Properties Verified:**
1. **A^T × A = I**: Transpose times original equals identity
2. **A × A^T = I**: Original times transpose equals identity  
3. **Column Orthonormality**: All columns have unit norm and are orthogonal
4. **Row Orthonormality**: All rows have unit norm and are orthogonal
5. **Determinant**: Must be ±1 for orthogonal matrices
6. **Singular Values**: All equal to 1 for orthogonal matrices

**Applications:**
- **Computer Graphics**: Rotation and reflection transformations
- **Signal Processing**: Orthogonal transforms (FFT, DCT)
- **Machine Learning**: PCA, whitening transformations
- **Numerical Analysis**: Stable matrix decompositions

**Best Practices:**
- Use appropriate numerical tolerance for floating-point comparisons
- Verify multiple orthogonality conditions for robustness
- Consider special types (rotation, reflection, permutation matrices)
- Use NumPy for better numerical stability with large matrices

---

## Question 9

**How would you implement a linear algebra-based algorithm to classify text documents?**

**Answer:** Here's a comprehensive implementation of linear algebra-based text classification using multiple approaches:

```python
import math
import numpy as np
from collections import defaultdict, Counter
import re

class LinearAlgebraTextClassifier:
    """
    Linear algebra-based text classification using multiple methods:
    1. TF-IDF with Cosine Similarity
    2. Naive Bayes with Linear Algebra
    3. Linear Discriminant Analysis (LDA)
    4. Principal Component Analysis (PCA) + Classification
    """
    
    def __init__(self, method="tfidf_cosine", max_features=1000):
        self.method = method
        self.max_features = max_features
        self.vocabulary = {}
        self.word_to_index = {}
        self.index_to_word = {}
        self.class_labels = []
        self.trained = False
        
        # Method-specific parameters
        self.tfidf_matrix = None
        self.class_centroids = {}
        self.class_priors = {}
        self.feature_probs = {}
        self.pca_components = None
        self.lda_weights = None
    
    def preprocess_text(self, text):
        """
        Preprocess text: lowercase, remove punctuation, tokenize.
        
        Args:
            text: Input text string
        
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Simple stopword removal
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
        
        return tokens
    
    def build_vocabulary(self, documents):
        """
        Build vocabulary from training documents.
        
        Args:
            documents: List of text documents
        """
        word_counts = Counter()
        
        for doc in documents:
            tokens = self.preprocess_text(doc)
            word_counts.update(tokens)
        
        # Select top features
        most_common = word_counts.most_common(self.max_features)
        
        self.vocabulary = {word: count for word, count in most_common}
        self.word_to_index = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def text_to_vector(self, text, method="tf"):
        """
        Convert text to numerical vector.
        
        Args:
            text: Input text
            method: "tf" (term frequency), "binary", or "tfidf"
        
        Returns:
            Numerical vector representation
        """
        tokens = self.preprocess_text(text)
        vector = [0.0] * len(self.vocabulary)
        
        if method == "tf":
            # Term frequency
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token in self.word_to_index:
                    vector[self.word_to_index[token]] = count
        
        elif method == "binary":
            # Binary representation
            for token in set(tokens):
                if token in self.word_to_index:
                    vector[self.word_to_index[token]] = 1.0
        
        elif method == "tfidf":
            # TF-IDF (requires pre-computed IDF values)
            token_counts = Counter(tokens)
            total_tokens = len(tokens)
            
            for token, count in token_counts.items():
                if token in self.word_to_index:
                    tf = count / total_tokens
                    idf = self.idf_values.get(token, 0)
                    vector[self.word_to_index[token]] = tf * idf
        
        return vector
    
    def compute_idf(self, documents):
        """
        Compute Inverse Document Frequency for all terms.
        
        Args:
            documents: List of training documents
        """
        self.idf_values = {}
        n_docs = len(documents)
        
        # Count document frequency for each term
        doc_freq = defaultdict(int)
        
        for doc in documents:
            tokens = set(self.preprocess_text(doc))
            for token in tokens:
                if token in self.vocabulary:
                    doc_freq[token] += 1
        
        # Compute IDF
        for word in self.vocabulary:
            df = doc_freq.get(word, 0)
            if df > 0:
                self.idf_values[word] = math.log(n_docs / df)
            else:
                self.idf_values[word] = 0
    
    def train_tfidf_cosine(self, documents, labels):
        """
        Train TF-IDF with cosine similarity classifier.
        
        Args:
            documents: Training documents
            labels: Training labels
        """
        print("Training TF-IDF Cosine Similarity classifier...")
        
        # Build vocabulary and compute IDF
        self.build_vocabulary(documents)
        self.compute_idf(documents)
        
        # Convert documents to TF-IDF vectors
        self.tfidf_matrix = []
        for doc in documents:
            vector = self.text_to_vector(doc, method="tfidf")
            self.tfidf_matrix.append(vector)
        
        self.tfidf_matrix = np.array(self.tfidf_matrix)
        self.training_labels = labels
        self.class_labels = list(set(labels))
        
        # Compute class centroids
        for class_label in self.class_labels:
            class_indices = [i for i, label in enumerate(labels) if label == class_label]
            class_vectors = self.tfidf_matrix[class_indices]
            centroid = np.mean(class_vectors, axis=0)
            self.class_centroids[class_label] = centroid
    
    def train_naive_bayes(self, documents, labels):
        """
        Train Naive Bayes classifier using linear algebra operations.
        
        Args:
            documents: Training documents
            labels: Training labels
        """
        print("Training Naive Bayes classifier...")
        
        self.build_vocabulary(documents)
        self.class_labels = list(set(labels))
        
        # Compute class priors
        total_docs = len(documents)
        for class_label in self.class_labels:
            class_count = sum(1 for label in labels if label == class_label)
            self.class_priors[class_label] = class_count / total_docs
        
        # Compute feature probabilities for each class
        for class_label in self.class_labels:
            class_docs = [doc for doc, label in zip(documents, labels) if label == class_label]
            
            # Aggregate word counts for this class
            class_word_counts = Counter()
            for doc in class_docs:
                tokens = self.preprocess_text(doc)
                class_word_counts.update(tokens)
            
            # Compute probabilities with Laplace smoothing
            total_words = sum(class_word_counts.values())
            vocab_size = len(self.vocabulary)
            
            feature_probs = {}
            for word in self.vocabulary:
                count = class_word_counts.get(word, 0)
                # Laplace smoothing
                feature_probs[word] = (count + 1) / (total_words + vocab_size)
            
            self.feature_probs[class_label] = feature_probs
    
    def train_lda(self, documents, labels):
        """
        Train Linear Discriminant Analysis classifier.
        
        Args:
            documents: Training documents
            labels: Training labels
        """
        print("Training Linear Discriminant Analysis classifier...")
        
        self.build_vocabulary(documents)
        self.class_labels = list(set(labels))
        
        # Convert documents to TF vectors
        X = []
        for doc in documents:
            vector = self.text_to_vector(doc, method="tf")
            X.append(vector)
        
        X = np.array(X)
        y = np.array(labels)
        
        # Compute class means
        class_means = {}
        for class_label in self.class_labels:
            class_mask = y == class_label
            class_means[class_label] = np.mean(X[class_mask], axis=0)
        
        # Compute overall mean
        overall_mean = np.mean(X, axis=0)
        
        # Compute within-class scatter matrix
        S_w = np.zeros((X.shape[1], X.shape[1]))
        for class_label in self.class_labels:
            class_mask = y == class_label
            class_data = X[class_mask]
            class_mean = class_means[class_label]
            
            for sample in class_data:
                diff = (sample - class_mean).reshape(-1, 1)
                S_w += diff @ diff.T
        
        # Compute between-class scatter matrix
        S_b = np.zeros((X.shape[1], X.shape[1]))
        for class_label in self.class_labels:
            n_class = np.sum(y == class_label)
            class_mean = class_means[class_label]
            diff = (class_mean - overall_mean).reshape(-1, 1)
            S_b += n_class * (diff @ diff.T)
        
        # Solve generalized eigenvalue problem: S_b * w = λ * S_w * w
        try:
            # Add regularization to avoid singular matrix
            S_w += 1e-6 * np.eye(S_w.shape[0])
            eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(S_w) @ S_b)
            
            # Sort by eigenvalues
            idx = np.argsort(eigenvals)[::-1]
            self.lda_weights = eigenvecs[:, idx]
            
        except np.linalg.LinAlgError:
            print("Warning: LDA failed, using PCA instead")
            self.train_pca_classification(documents, labels)
    
    def train_pca_classification(self, documents, labels):
        """
        Train PCA + nearest centroid classifier.
        
        Args:
            documents: Training documents
            labels: Training labels
        """
        print("Training PCA + Classification...")
        
        self.build_vocabulary(documents)
        self.class_labels = list(set(labels))
        
        # Convert documents to TF vectors
        X = []
        for doc in documents:
            vector = self.text_to_vector(doc, method="tf")
            X.append(vector)
        
        X = np.array(X)
        
        # Perform PCA
        # Center the data
        self.data_mean = np.mean(X, axis=0)
        X_centered = X - self.data_mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select top components (95% variance)
        cumsum = np.cumsum(eigenvals)
        total_var = cumsum[-1]
        n_components = np.argmax(cumsum >= 0.95 * total_var) + 1
        n_components = min(n_components, 100)  # Limit to 100 components
        
        self.pca_components = eigenvecs[:, :n_components]
        
        # Transform training data
        X_pca = X_centered @ self.pca_components
        
        # Compute class centroids in PCA space
        for class_label in self.class_labels:
            class_mask = np.array(labels) == class_label
            class_data_pca = X_pca[class_mask]
            self.class_centroids[class_label] = np.mean(class_data_pca, axis=0)
    
    def train(self, documents, labels):
        """
        Train the classifier using the specified method.
        
        Args:
            documents: List of training documents
            labels: List of training labels
        """
        if self.method == "tfidf_cosine":
            self.train_tfidf_cosine(documents, labels)
        elif self.method == "naive_bayes":
            self.train_naive_bayes(documents, labels)
        elif self.method == "lda":
            self.train_lda(documents, labels)
        elif self.method == "pca":
            self.train_pca_classification(documents, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.trained = True
        print(f"Training completed using {self.method} method.")
    
    def cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1, vec2: Input vectors
        
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def predict_tfidf_cosine(self, text):
        """
        Predict using TF-IDF and cosine similarity.
        
        Args:
            text: Input text to classify
        
        Returns:
            Predicted class and confidence scores
        """
        # Convert text to TF-IDF vector
        test_vector = self.text_to_vector(text, method="tfidf")
        
        # Calculate similarities to class centroids
        similarities = {}
        for class_label, centroid in self.class_centroids.items():
            similarity = self.cosine_similarity(test_vector, centroid)
            similarities[class_label] = similarity
        
        # Return class with highest similarity
        best_class = max(similarities, key=similarities.get)
        
        return best_class, similarities
    
    def predict_naive_bayes(self, text):
        """
        Predict using Naive Bayes.
        
        Args:
            text: Input text to classify
        
        Returns:
            Predicted class and log probabilities
        """
        tokens = self.preprocess_text(text)
        
        # Calculate log probabilities for each class
        log_probs = {}
        
        for class_label in self.class_labels:
            # Start with log prior
            log_prob = math.log(self.class_priors[class_label])
            
            # Add log probabilities of features
            for token in tokens:
                if token in self.feature_probs[class_label]:
                    log_prob += math.log(self.feature_probs[class_label][token])
            
            log_probs[class_label] = log_prob
        
        # Return class with highest log probability
        best_class = max(log_probs, key=log_probs.get)
        
        return best_class, log_probs
    
    def predict_lda(self, text):
        """
        Predict using Linear Discriminant Analysis.
        
        Args:
            text: Input text to classify
        
        Returns:
            Predicted class and scores
        """
        # Convert text to TF vector
        test_vector = np.array(self.text_to_vector(text, method="tf"))
        
        # Project to LDA space
        if self.lda_weights is not None:
            projected = test_vector @ self.lda_weights
            
            # Find nearest class centroid in projected space
            distances = {}
            for class_label in self.class_labels:
                # Project class centroid
                class_vector = self.class_centroids.get(class_label, np.zeros_like(test_vector))
                projected_centroid = class_vector @ self.lda_weights
                
                # Calculate distance
                distance = np.linalg.norm(projected - projected_centroid)
                distances[class_label] = distance
            
            best_class = min(distances, key=distances.get)
            return best_class, distances
        else:
            # Fallback to nearest centroid in original space
            return self.predict_tfidf_cosine(text)
    
    def predict_pca(self, text):
        """
        Predict using PCA + nearest centroid.
        
        Args:
            text: Input text to classify
        
        Returns:
            Predicted class and distances
        """
        # Convert text to TF vector
        test_vector = np.array(self.text_to_vector(text, method="tf"))
        
        # Center and project to PCA space
        test_vector_centered = test_vector - self.data_mean
        test_vector_pca = test_vector_centered @ self.pca_components
        
        # Find nearest class centroid
        distances = {}
        for class_label, centroid in self.class_centroids.items():
            distance = np.linalg.norm(test_vector_pca - centroid)
            distances[class_label] = distance
        
        best_class = min(distances, key=distances.get)
        
        return best_class, distances
    
    def predict(self, text):
        """
        Predict the class of input text.
        
        Args:
            text: Input text to classify
        
        Returns:
            Predicted class and confidence/score information
        """
        if not self.trained:
            raise ValueError("Classifier must be trained before prediction")
        
        if self.method == "tfidf_cosine":
            return self.predict_tfidf_cosine(text)
        elif self.method == "naive_bayes":
            return self.predict_naive_bayes(text)
        elif self.method == "lda":
            return self.predict_lda(text)
        elif self.method == "pca":
            return self.predict_pca(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def evaluate(self, test_documents, test_labels):
        """
        Evaluate classifier on test data.
        
        Args:
            test_documents: Test documents
            test_labels: True test labels
        
        Returns:
            Evaluation metrics
        """
        predictions = []
        confidences = []
        
        for doc in test_documents:
            pred_class, scores = self.predict(doc)
            predictions.append(pred_class)
            
            # Convert scores to confidence (method-dependent)
            if self.method == "tfidf_cosine":
                confidence = max(scores.values())
            elif self.method == "naive_bayes":
                # Convert log probabilities to probabilities
                log_probs = list(scores.values())
                max_log_prob = max(log_probs)
                probs = [math.exp(lp - max_log_prob) for lp in log_probs]
                confidence = max(probs) / sum(probs)
            else:
                confidence = 1.0 / (1.0 + min(scores.values()))  # Distance-based
            
            confidences.append(confidence)
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
        accuracy = correct / len(test_labels)
        
        # Calculate per-class metrics
        class_metrics = {}
        for class_label in self.class_labels:
            true_positives = sum(1 for pred, true in zip(predictions, test_labels) 
                               if pred == class_label and true == class_label)
            false_positives = sum(1 for pred, true in zip(predictions, test_labels) 
                                if pred == class_label and true != class_label)
            false_negatives = sum(1 for pred, true in zip(predictions, test_labels) 
                                if pred != class_label and true == class_label)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'confidences': confidences,
            'class_metrics': class_metrics
        }


# Example usage and demonstration
def create_sample_dataset():
    """
    Create a sample text classification dataset.
    
    Returns:
        (documents, labels) tuple
    """
    # Sample documents for different categories
    tech_docs = [
        "Machine learning algorithms use statistical methods to enable computers to learn from data without explicit programming.",
        "Neural networks are computing systems inspired by biological neural networks that process information.",
        "Artificial intelligence involves creating computer systems that can perform tasks requiring human intelligence.",
        "Deep learning uses multiple layers of neural networks to model and understand complex patterns in data.",
        "Python programming language is widely used for data science and machine learning applications.",
        "Software engineering principles help developers create robust and maintainable computer programs.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Database management systems store and organize large amounts of structured information efficiently."
    ]
    
    sports_docs = [
        "Football players train rigorously to improve their physical conditioning and tactical skills.",
        "Basketball requires excellent hand-eye coordination and strategic team play to win games.",
        "Tennis players must develop powerful serves and precise shot placement to compete effectively.",
        "Soccer matches involve continuous running and precise ball control techniques throughout the game.",
        "Swimming competitions test athletes' speed, endurance, and technical stroke mechanics.",
        "Baseball strategies include bunting, stealing bases, and managing pitching rotations carefully.",
        "Golf requires mental focus, precise swing mechanics, and course management skills.",
        "Hockey players need excellent skating skills and quick decision-making under pressure."
    ]
    
    finance_docs = [
        "Stock market investments require careful analysis of company fundamentals and market trends.",
        "Portfolio diversification helps reduce investment risk by spreading assets across different sectors.",
        "Interest rates significantly impact bond prices and overall market liquidity conditions.",
        "Cryptocurrency trading involves high volatility and requires understanding blockchain technology fundamentals.",
        "Financial planning helps individuals achieve long-term wealth accumulation and retirement goals.",
        "Corporate earnings reports provide insights into company performance and future growth prospects.",
        "Economic indicators like GDP and inflation influence monetary policy decisions by central banks.",
        "Investment banking services include underwriting securities and providing merger acquisition advice."
    ]
    
    # Combine documents and create labels
    documents = tech_docs + sports_docs + finance_docs
    labels = (['technology'] * len(tech_docs) + 
             ['sports'] * len(sports_docs) + 
             ['finance'] * len(finance_docs))
    
    return documents, labels


def demonstrate_text_classification():
    """
    Comprehensive demonstration of linear algebra-based text classification.
    """
    print("="*70)
    print("LINEAR ALGEBRA-BASED TEXT CLASSIFICATION DEMONSTRATION")
    print("="*70)
    
    # Create sample dataset
    documents, labels = create_sample_dataset()
    
    print(f"\nDataset Statistics:")
    print(f"Total documents: {len(documents)}")
    print(f"Classes: {set(labels)}")
    print(f"Class distribution: {Counter(labels)}")
    
    # Split into train/test (simple split for demo)
    train_size = int(0.7 * len(documents))
    train_docs = documents[:train_size]
    train_labels = labels[:train_size]
    test_docs = documents[train_size:]
    test_labels = labels[train_size:]
    
    print(f"\nTrain/Test Split:")
    print(f"Training documents: {len(train_docs)}")
    print(f"Test documents: {len(test_docs)}")
    
    # Test different methods
    methods = ["tfidf_cosine", "naive_bayes", "pca"]
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"TESTING METHOD: {method.upper()}")
        print(f"{'='*50}")
        
        try:
            # Create and train classifier
            classifier = LinearAlgebraTextClassifier(method=method, max_features=200)
            classifier.train(train_docs, train_labels)
            
            # Evaluate on test set
            evaluation = classifier.evaluate(test_docs, test_labels)
            results[method] = evaluation
            
            print(f"\nAccuracy: {evaluation['accuracy']:.3f}")
            
            print(f"\nPer-class metrics:")
            for class_label, metrics in evaluation['class_metrics'].items():
                print(f"  {class_label:12}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
            
            # Test on new examples
            print(f"\nTesting on new examples:")
            
            test_examples = [
                "Deep neural networks process complex data using multiple hidden layers.",
                "The basketball team won the championship with excellent defensive play.",
                "Stock prices fluctuated wildly due to economic uncertainty and market volatility."
            ]
            
            expected_classes = ["technology", "sports", "finance"]
            
            for i, (example, expected) in enumerate(zip(test_examples, expected_classes)):
                pred_class, scores = classifier.predict(example)
                
                print(f"\nExample {i+1}: {example[:60]}...")
                print(f"  Expected: {expected}")
                print(f"  Predicted: {pred_class}")
                print(f"  Correct: {'✓' if pred_class == expected else '✗'}")
                
                # Show top scores
                if isinstance(scores, dict):
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    print(f"  Scores: {sorted_scores}")
        
        except Exception as e:
            print(f"Error with method {method}: {e}")
            results[method] = None
    
    # Compare methods
    print(f"\n{'='*50}")
    print("METHOD COMPARISON")
    print(f"{'='*50}")
    
    print(f"{'Method':<15} {'Accuracy':<10} {'Status'}")
    print("-" * 35)
    
    for method, result in results.items():
        if result is not None:
            accuracy = result['accuracy']
            status = "Success"
            print(f"{method:<15} {accuracy:<10.3f} {status}")
        else:
            print(f"{method:<15} {'N/A':<10} Failed")
    
    # Demonstrate feature analysis
    print(f"\n{'='*50}")
    print("FEATURE ANALYSIS")
    print(f"{'='*50}")
    
    # Use the TF-IDF classifier for feature analysis
    tfidf_classifier = LinearAlgebraTextClassifier(method="tfidf_cosine", max_features=50)
    tfidf_classifier.train(train_docs, train_labels)
    
    print(f"\nTop features in vocabulary:")
    top_features = list(tfidf_classifier.vocabulary.keys())[:20]
    print(f"  {top_features}")
    
    print(f"\nClass centroids analysis:")
    for class_label, centroid in tfidf_classifier.class_centroids.items():
        # Find features with highest values in centroid
        feature_values = [(tfidf_classifier.index_to_word[i], centroid[i]) 
                         for i in range(len(centroid)) if centroid[i] > 0]
        feature_values.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_values[:5]
        
        print(f"  {class_label}: {[(feat, f'{val:.3f}') for feat, val in top_features]}")


# Mathematical foundations demonstration
def demonstrate_mathematical_foundations():
    """
    Demonstrate the mathematical foundations of linear algebra in text classification.
    """
    print(f"\n{'='*70}")
    print("MATHEMATICAL FOUNDATIONS DEMONSTRATION")
    print(f"{'='*70}")
    
    # Simple example with small vocabulary
    documents = [
        "machine learning algorithms",
        "learning artificial intelligence",
        "football game strategy",
        "game artificial intelligence"
    ]
    
    labels = ["tech", "tech", "sports", "tech"]
    
    # Build simple vocabulary
    vocabulary = ["machine", "learning", "algorithms", "artificial", "intelligence", "football", "game", "strategy"]
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    
    print(f"Vocabulary: {vocabulary}")
    print(f"Documents: {documents}")
    print(f"Labels: {labels}")
    
    # Convert to term frequency matrix
    tf_matrix = []
    for doc in documents:
        tokens = doc.split()
        tf_vector = [0] * len(vocabulary)
        for token in tokens:
            if token in word_to_index:
                tf_vector[word_to_index[token]] += 1
        tf_matrix.append(tf_vector)
    
    tf_matrix = np.array(tf_matrix)
    
    print(f"\nTerm Frequency Matrix:")
    print(f"{'Doc':<5} {' '.join(f'{word[:4]:>4}' for word in vocabulary)}")
    print("-" * 50)
    for i, (doc_vector, label) in enumerate(zip(tf_matrix, labels)):
        print(f"{i:<5} {' '.join(f'{val:>4}' for val in doc_vector)} ({label})")
    
    # Compute class centroids
    tech_docs = tf_matrix[[i for i, label in enumerate(labels) if label == "tech"]]
    sports_docs = tf_matrix[[i for i, label in enumerate(labels) if label == "sports"]]
    
    tech_centroid = np.mean(tech_docs, axis=0)
    sports_centroid = np.mean(sports_docs, axis=0)
    
    print(f"\nClass Centroids:")
    print(f"Tech:   {' '.join(f'{val:>4.1f}' for val in tech_centroid)}")
    print(f"Sports: {' '.join(f'{val:>4.1f}' for val in sports_centroid)}")
    
    # Test classification on new document
    test_doc = "machine intelligence algorithms"
    test_tokens = test_doc.split()
    test_vector = np.array([1 if word in test_tokens else 0 for word in vocabulary])
    
    print(f"\nTest document: '{test_doc}'")
    print(f"Test vector: {' '.join(f'{val:>4}' for val in test_vector)}")
    
    # Calculate cosine similarities
    def cosine_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot_product / norms if norms > 0 else 0
    
    tech_similarity = cosine_similarity(test_vector, tech_centroid)
    sports_similarity = cosine_similarity(test_vector, sports_centroid)
    
    print(f"\nCosine Similarities:")
    print(f"Tech: {tech_similarity:.3f}")
    print(f"Sports: {sports_similarity:.3f}")
    print(f"Predicted class: {'Tech' if tech_similarity > sports_similarity else 'Sports'}")


if __name__ == "__main__":
    # Run comprehensive demonstration
    demonstrate_text_classification()
    
    # Run mathematical foundations demo
    demonstrate_mathematical_foundations()


# Advanced features and extensions
class AdvancedTextClassifier(LinearAlgebraTextClassifier):
    """
    Extended classifier with advanced linear algebra techniques.
    """
    
    def __init__(self, method="tfidf_cosine", max_features=1000, use_svd=False, n_components=100):
        super().__init__(method, max_features)
        self.use_svd = use_svd
        self.n_components = n_components
        self.svd_components = None
    
    def apply_svd_dimensionality_reduction(self, matrix):
        """
        Apply SVD for dimensionality reduction.
        
        Args:
            matrix: Document-term matrix
        
        Returns:
            Reduced dimension matrix
        """
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Keep top k components
        k = min(self.n_components, len(s))
        self.svd_components = {
            'U': U[:, :k],
            's': s[:k],
            'Vt': Vt[:k, :]
        }
        
        # Reduced matrix
        return U[:, :k] @ np.diag(s[:k])
    
    def compute_document_similarity_matrix(self, documents):
        """
        Compute pairwise document similarity matrix.
        
        Args:
            documents: List of documents
        
        Returns:
            Similarity matrix
        """
        n = len(documents)
        similarity_matrix = np.zeros((n, n))
        
        # Convert documents to vectors
        vectors = []
        for doc in documents:
            vector = self.text_to_vector(doc, method="tfidf")
            vectors.append(vector)
        
        # Compute pairwise similarities
        for i in range(n):
            for j in range(i, n):
                sim = self.cosine_similarity(vectors[i], vectors[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return similarity_matrix
    
    def perform_spectral_clustering(self, similarity_matrix, n_clusters):
        """
        Perform spectral clustering using eigendecomposition.
        
        Args:
            similarity_matrix: Document similarity matrix
            n_clusters: Number of clusters
        
        Returns:
            Cluster assignments
        """
        # Compute Laplacian matrix
        degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
        laplacian = degree_matrix - similarity_matrix
        
        # Normalize Laplacian
        sqrt_degree = np.sqrt(degree_matrix)
        sqrt_degree_inv = np.linalg.pinv(sqrt_degree)
        normalized_laplacian = sqrt_degree_inv @ laplacian @ sqrt_degree_inv
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eig(normalized_laplacian)
        
        # Sort by eigenvalues (smallest first)
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Use first k eigenvectors for clustering
        embedding = eigenvecs[:, :n_clusters]
        
        # Simple k-means clustering in eigenspace
        cluster_assignments = self.simple_kmeans(embedding, n_clusters)
        
        return cluster_assignments
    
    def simple_kmeans(self, data, k, max_iterations=100):
        """
        Simple k-means clustering implementation.
        
        Args:
            data: Data matrix
            k: Number of clusters
            max_iterations: Maximum iterations
        
        Returns:
            Cluster assignments
        """
        n, d = data.shape
        
        # Initialize centroids randomly
        centroids = data[np.random.choice(n, k, replace=False)]
        
        for iteration in range(max_iterations):
            # Assign points to closest centroids
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = data[assignments == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return assignments


# Performance analysis tools
def analyze_classifier_performance():
    """
    Analyze performance characteristics of different methods.
    """
    print(f"\n{'='*70}")
    print("CLASSIFIER PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    
    # Create larger synthetic dataset
    np.random.seed(42)
    
    # Generate more diverse dataset
    categories = ['technology', 'sports', 'finance', 'health', 'travel']
    
    word_pools = {
        'technology': ['computer', 'software', 'algorithm', 'data', 'programming', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'network'],
        'sports': ['game', 'player', 'team', 'score', 'championship', 'training', 'athletics', 'competition', 'stadium', 'coach', 'strategy'],
        'finance': ['money', 'investment', 'stock', 'market', 'portfolio', 'trading', 'profit', 'economy', 'banking', 'financial', 'capital'],
        'health': ['medical', 'treatment', 'patient', 'doctor', 'hospital', 'medicine', 'therapy', 'diagnosis', 'healthcare', 'wellness', 'fitness'],
        'travel': ['vacation', 'destination', 'journey', 'hotel', 'flight', 'tourism', 'adventure', 'culture', 'explore', 'passport', 'luggage']
    }
    
    # Generate synthetic documents
    def generate_document(category, length=15):
        words = word_pools[category]
        # Add some noise from other categories
        noise_words = []
        for other_cat in categories:
            if other_cat != category:
                noise_words.extend(word_pools[other_cat][:2])  # Add 2 noise words per category
        
        # 80% category words, 20% noise
        category_count = int(0.8 * length)
        noise_count = length - category_count
        
        doc_words = (np.random.choice(words, category_count).tolist() + 
                     np.random.choice(noise_words, noise_count).tolist())
        
        np.random.shuffle(doc_words)
        return ' '.join(doc_words)
    
    # Generate dataset
    docs_per_category = 50
    documents = []
    labels = []
    
    for category in categories:
        for _ in range(docs_per_category):
            doc = generate_document(category)
            documents.append(doc)
            labels.append(category)
    
    # Split data
    from random import shuffle
    combined = list(zip(documents, labels))
    shuffle(combined)
    documents, labels = zip(*combined)
    
    split_point = int(0.8 * len(documents))
    train_docs = documents[:split_point]
    train_labels = labels[:split_point]
    test_docs = documents[split_point:]
    test_labels = labels[split_point:]
    
    print(f"Dataset: {len(documents)} documents, {len(categories)} categories")
    print(f"Training: {len(train_docs)}, Testing: {len(test_docs)}")
    
    # Test all methods
    methods_to_test = ["tfidf_cosine", "naive_bayes", "pca"]
    feature_sizes = [100, 500, 1000]
    
    results_table = []
    
    for method in methods_to_test:
        for max_features in feature_sizes:
            print(f"\nTesting {method} with {max_features} features...")
            
            try:
                import time
                start_time = time.time()
                
                classifier = LinearAlgebraTextClassifier(method=method, max_features=max_features)
                classifier.train(train_docs, train_labels)
                
                train_time = time.time() - start_time
                
                start_time = time.time()
                evaluation = classifier.evaluate(test_docs, test_labels)
                test_time = time.time() - start_time
                
                results_table.append({
                    'method': method,
                    'features': max_features,
                    'accuracy': evaluation['accuracy'],
                    'train_time': train_time,
                    'test_time': test_time
                })
                
                print(f"  Accuracy: {evaluation['accuracy']:.3f}")
                print(f"  Train time: {train_time:.2f}s")
                print(f"  Test time: {test_time:.3f}s")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Features':<10} {'Accuracy':<10} {'Train(s)':<10} {'Test(s)':<10}")
    print("-" * 80)
    
    for result in results_table:
        print(f"{result['method']:<15} {result['features']:<10} {result['accuracy']:<10.3f} {result['train_time']:<10.2f} {result['test_time']:<10.3f}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_text_classification()
    demonstrate_mathematical_foundations()
    analyze_classifier_performance()
```

**Key Features:**
- **Multiple Linear Algebra Methods**: TF-IDF with cosine similarity, Naive Bayes, LDA, PCA
- **Comprehensive Text Processing**: Tokenization, stopword removal, vocabulary building
- **Mathematical Foundations**: Vector space models, matrix operations, eigendecomposition
- **Performance Analysis**: Accuracy, precision, recall, F1-score evaluation
- **Advanced Techniques**: SVD dimensionality reduction, spectral clustering

**Linear Algebra Components:**
1. **Vector Space Model**: Documents represented as high-dimensional vectors
2. **TF-IDF Matrix**: Term frequency-inverse document frequency weighting
3. **Cosine Similarity**: Angle-based document similarity using dot products
4. **Principal Component Analysis**: Eigendecomposition for dimensionality reduction
5. **Linear Discriminant Analysis**: Between/within-class scatter matrices
6. **Singular Value Decomposition**: Matrix factorization for feature reduction

**Classification Methods:**
- **TF-IDF + Cosine Similarity**: Nearest centroid in vector space
- **Naive Bayes**: Probabilistic classification with linear algebra operations
- **LDA**: Optimal linear projection for class separation
- **PCA + Classification**: Dimensionality reduction followed by nearest centroid

**Applications:**
- **Document Classification**: News categorization, email filtering
- **Sentiment Analysis**: Product reviews, social media analysis
- **Information Retrieval**: Search engines, recommendation systems
- **Content Organization**: Library systems, knowledge management

**Best Practices:**
- Preprocess text consistently (tokenization, normalization)
- Use appropriate feature selection (vocabulary size, TF-IDF weighting)
- Apply dimensionality reduction for high-dimensional data
- Evaluate multiple methods and choose based on dataset characteristics
- Consider computational complexity for large-scale applications

---

