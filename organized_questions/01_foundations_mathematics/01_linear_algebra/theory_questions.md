# Linear Algebra Interview Questions - Theory Questions

## Question 1

**What is a vector and how is it used in machine learning?**

**Answer:** A vector is a mathematical object that has both magnitude and direction, represented as an ordered collection of numbers (components). In machine learning:

- **Feature Representation**: Each data point is represented as a feature vector where each component represents a specific attribute or feature
- **Model Parameters**: Weight vectors store the learned parameters of models
- **Embeddings**: Words, images, or other data are converted into vector representations in high-dimensional spaces
- **Computations**: Operations like dot products, distance calculations, and transformations are performed using vector arithmetic
- **Examples**: A house might be represented as [bedrooms=3, bathrooms=2, sqft=1500, price=300000]

---

## Question 2

**Explain the difference between a scalar and a vector.**

**Answer:** 

**Scalar:**
- A single numerical value with magnitude only
- Has no direction (0-dimensional)
- Examples: temperature (25°C), mass (5kg), speed (60 mph)
- Represented by simple numbers: 5, -3.14, 100

**Vector:**
- An ordered collection of numbers with both magnitude and direction
- Multi-dimensional (1D, 2D, 3D, or higher)
- Examples: velocity (60 mph northeast), force (10N at 45°), position coordinates (x=3, y=4)
- Represented as arrays: [3, 4], [-1, 2, 5]

**Key Differences:**
- **Dimensionality**: Scalars are 0D, vectors are 1D or higher
- **Operations**: Scalars use basic arithmetic, vectors use specialized operations (dot product, cross product)
- **Representation**: Scalars are single values, vectors are arrays/matrices

---

## Question 3

**What is a matrix and why is it central to linear algebra?**

**Answer:** A matrix is a 2D array of numbers arranged in rows and columns, represented as:

```
A = [a₁₁  a₁₂  a₁₃]
    [a₂₁  a₂₂  a₂₃]
    [a₃₁  a₃₂  a₃₃]
```

**Why matrices are central to linear algebra:**

1. **Linear Transformations**: Matrices represent linear transformations between vector spaces
2. **System of Equations**: Solve multiple linear equations simultaneously (Ax = b)
3. **Data Organization**: Store and manipulate large datasets efficiently
4. **Composition**: Combine multiple transformations through matrix multiplication
5. **Eigenvalue Problems**: Find characteristic vectors and values
6. **Dimensionality**: Work with high-dimensional spaces

**Applications in ML:**
- **Dataset Representation**: Each row = sample, each column = feature
- **Neural Networks**: Weight matrices connect layers
- **PCA**: Covariance matrices for dimensionality reduction
- **Transformations**: Rotation, scaling, translation operations

---

## Question 4

**Explain the concept of a tensor in the context of machine learning.**

**Answer:** A tensor is a generalization of scalars, vectors, and matrices to arbitrary dimensions:

**Tensor Hierarchy:**
- **0D Tensor**: Scalar (single number)
- **1D Tensor**: Vector (array of numbers)
- **2D Tensor**: Matrix (2D array)
- **3D Tensor**: Cube of numbers (height × width × depth)
- **nD Tensor**: n-dimensional array

**In Machine Learning:**

1. **Data Representation**:
   - **Images**: 3D tensors (height × width × channels)
   - **Video**: 4D tensors (time × height × width × channels)
   - **Batch Processing**: Add batch dimension (batch × features)

2. **Deep Learning**:
   - **Input**: Multi-dimensional data tensors
   - **Weights**: Parameter tensors of various shapes
   - **Activations**: Feature maps as tensors

3. **Operations**:
   - **Tensor Addition**: Element-wise operations
   - **Tensor Multiplication**: Generalized matrix multiplication
   - **Reshaping**: Change dimensions while preserving data

**Example**: A batch of 32 RGB images (224×224) = tensor shape [32, 224, 224, 3]

---

## Question 5

**What are the properties of matrix multiplication?**

**Answer:** Matrix multiplication is a fundamental operation in linear algebra with specific properties that govern how matrices interact. Understanding these properties is crucial for efficient computation and mathematical reasoning.

**Basic Definition:**
For matrices A (m×n) and B (n×p), the product AB results in an (m×p) matrix where:
```
(AB)ᵢⱼ = Σₖ aᵢₖ × bₖⱼ
```

**Key Properties:**

1. **Associativity**: (AB)C = A(BC)
   - Order of operations doesn't matter for parentheses
   - Allows efficient computation strategies

2. **Non-Commutativity**: AB ≠ BA (generally)
   - Matrix multiplication order matters
   - AB may exist while BA doesn't (dimension mismatch)

3. **Distributivity**: 
   - Left: A(B + C) = AB + AC
   - Right: (A + B)C = AC + BC

4. **Identity Element**: AI = IA = A
   - Identity matrix I acts as multiplicative identity
   - Preserves original matrix dimensions and values

5. **Zero Element**: A × 0 = 0 × A = 0
   - Multiplication by zero matrix gives zero matrix

6. **Scalar Multiplication**: (cA)B = c(AB) = A(cB)
   - Scalar factors can be moved around freely

**Computational Properties:**

**Time Complexity**: O(mnp) for A(m×n) × B(n×p)
**Memory Requirements**: O(mp) for result storage
**Parallelization**: Highly parallelizable operation

**Machine Learning Applications:**
- **Neural Networks**: Forward propagation (X × W + b)
- **Data Transformation**: Apply learned transformations
- **Feature Engineering**: Combine and transform features
- **Optimization**: Gradient computations and updates

**Implementation Example:**
```python
import numpy as np

# Basic matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # Modern Python syntax
# or: C = np.dot(A, B)

# Verify non-commutativity
print("AB =", A @ B)
print("BA =", B @ A)
print("AB == BA:", np.array_equal(A @ B, B @ A))
```

**Important Notes:**
- Dimensions must be compatible: (m×n) × (n×p) → (m×p)
- Inner dimensions must match (n in both cases)
- Result dimensions are outer dimensions (m×p)
- Efficient algorithms like Strassen's can reduce complexity

---

## Question 6

**Explain the dot product of two vectors and its significance in machine learning.**

**Answer:** The dot product (also called scalar product or inner product) is a fundamental operation that combines two vectors to produce a scalar value. It measures the degree of alignment between vectors and has extensive applications in machine learning.

**Mathematical Definition:**
For vectors **a** = [a₁, a₂, ..., aₙ] and **b** = [b₁, b₂, ..., bₙ]:

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = Σᵢ aᵢbᵢ
```

**Alternative Formula:**
```
a · b = |a| × |b| × cos(θ)
```
where θ is the angle between vectors a and b.

**Geometric Interpretation:**
- **Positive dot product**: Vectors point in similar directions (θ < 90°)
- **Zero dot product**: Vectors are orthogonal/perpendicular (θ = 90°)
- **Negative dot product**: Vectors point in opposite directions (θ > 90°)

**Key Properties:**
1. **Commutativity**: a · b = b · a
2. **Distributivity**: a · (b + c) = a · b + a · c
3. **Scalar multiplication**: (ka) · b = k(a · b)
4. **Self dot product**: a · a = |a|² (magnitude squared)

**Machine Learning Applications:**

1. **Similarity Measures**:
   - **Cosine Similarity**: cos(θ) = (a · b) / (|a| × |b|)
   - **Document similarity** in NLP
   - **User similarity** in recommendation systems

2. **Neural Networks**:
   - **Linear layers**: output = input · weights + bias
   - **Attention mechanisms**: query · key operations
   - **Feature interactions**: measuring feature relationships

3. **Distance Computations**:
   - **Euclidean distance**: √((a-b) · (a-b))
   - **Kernel methods**: RBF kernels use dot products
   - **Support Vector Machines**: decision boundaries

4. **Optimization**:
   - **Gradient descent**: gradient · direction vectors
   - **Momentum**: previous update · current gradient
   - **Convergence checking**: gradient · gradient < threshold

5. **Dimensionality Reduction**:
   - **PCA**: principal components via eigenvector dot products
   - **Projections**: project data onto lower dimensions
   - **Feature selection**: correlation measurements

**Implementation Examples:**

```python
import numpy as np

# Basic dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # or a @ b
print(f"Dot product: {dot_product}")  # Output: 32

# Cosine similarity
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Neural network linear layer (simplified)
def linear_layer(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

# Batch processing
X = np.random.rand(100, 10)  # 100 samples, 10 features
W = np.random.rand(10, 5)    # 10 input, 5 output neurons
output = X @ W  # Efficient batch dot product
```

**Performance Considerations:**
- **Vectorization**: Use numpy/tensor operations for efficiency
- **Memory**: O(n) space complexity for vectors of length n
- **Time**: O(n) time complexity for computation
- **Parallelization**: Highly parallelizable across vector elements

**Common Pitfalls:**
- **Dimension mismatch**: Ensure vectors have same length
- **Normalization**: Consider normalizing vectors for cosine similarity
- **Numerical stability**: Use appropriate data types for precision

---

## Question 7

**What is the cross product of vectors and when is it used?**

**Answer:** The cross product (also called vector product) is a binary operation on two vectors in 3D space that produces a vector perpendicular to both input vectors. Unlike the dot product which yields a scalar, the cross product returns a vector with specific geometric properties.

**Mathematical Definition:**
For vectors **a** = [a₁, a₂, a₃] and **b** = [b₁, b₂, b₃]:

```
a × b = [a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁]
```

**Determinant Form:**
```
a × b = | i   j   k  |
        | a₁  a₂  a₃ |
        | b₁  b₂  b₃ |
```

**Geometric Properties:**
1. **Direction**: Perpendicular to both a and b (follows right-hand rule)
2. **Magnitude**: |a × b| = |a| × |b| × sin(θ)
3. **Area**: |a × b| equals the area of parallelogram formed by a and b

**Key Properties:**
- **Anti-commutativity**: a × b = -(b × a)
- **Distributivity**: a × (b + c) = a × b + a × c
- **Scalar multiplication**: (ka) × b = k(a × b)
- **Zero cross product**: If a × b = 0, then a and b are parallel
- **Self cross product**: a × a = 0

**When Cross Products Are Used:**

1. **Computer Graphics & 3D Modeling**:
   - **Surface normals**: Calculate perpendicular vectors to surfaces
   - **Lighting calculations**: Determine surface orientation for shading
   - **Camera transformations**: Compute viewing directions

2. **Physics & Engineering**:
   - **Torque calculations**: τ = r × F (force × distance)
   - **Magnetic fields**: F = q(v × B) (Lorentz force)
   - **Angular momentum**: L = r × p

3. **Computer Vision**:
   - **3D reconstruction**: Camera calibration and stereo vision
   - **Object orientation**: Determine object pose and rotation
   - **Feature matching**: Geometric consistency checks

4. **Robotics**:
   - **End-effector orientation**: Robot arm positioning
   - **Path planning**: Obstacle avoidance in 3D space
   - **Control systems**: Angular velocity calculations

5. **Machine Learning Applications**:
   - **Data augmentation**: 3D rotations for training data
   - **Geometric deep learning**: Graph neural networks with 3D data
   - **Point cloud processing**: Normal estimation for 3D objects

**Implementation Examples:**

```python
import numpy as np

# Basic cross product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
cross_product = np.cross(a, b)
print(f"Cross product: {cross_product}")  # [-3, 6, -3]

# Surface normal calculation
def calculate_surface_normal(p1, p2, p3):
    """Calculate normal vector for triangle defined by 3 points"""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)  # Normalize

# 3D rotation using cross product
def rotate_vector_around_axis(vector, axis, angle):
    """Rodrigues' rotation formula using cross product"""
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rotated = (vector * cos_angle + 
               np.cross(axis, vector) * sin_angle +
               axis * np.dot(axis, vector) * (1 - cos_angle))
    return rotated

# Area calculation
def triangle_area(a, b, c):
    """Calculate triangle area using cross product"""
    ab = b - a
    ac = c - a
    return 0.5 * np.linalg.norm(np.cross(ab, ac))
```

**Limitations & Considerations:**
- **3D specific**: Cross product is only defined for 3D vectors
- **Non-associative**: (a × b) × c ≠ a × (b × c)
- **Coordinate system dependent**: Results depend on right-hand vs left-hand systems
- **Numerical precision**: Can suffer from floating-point errors in parallel vectors

**Alternative in Higher Dimensions:**
For higher dimensions, the **wedge product** or **exterior product** generalizes the concept of cross products, used in differential geometry and advanced physics applications.

---

## Question 8

**What is the determinant of a matrix and what information does it provide?**

**Answer:** The determinant is a scalar value calculated from a square matrix that provides crucial information about the matrix's properties and the linear transformation it represents. It's one of the most important concepts in linear algebra with deep geometric and algebraic significance.

**Mathematical Definition:**

**For 2×2 matrix:**
```
det(A) = |a  b| = ad - bc
         |c  d|
```

**For 3×3 matrix:**
```
det(A) = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)
```

**For n×n matrix (general):**
Using cofactor expansion along any row or column:
```
det(A) = Σⱼ aᵢⱼ × (-1)ⁱ⁺ʲ × Mᵢⱼ
```
where Mᵢⱼ is the minor (determinant of submatrix).

**What the Determinant Tells Us:**

1. **Matrix Invertibility**:
   - **det(A) ≠ 0**: Matrix is invertible (non-singular)
   - **det(A) = 0**: Matrix is not invertible (singular)

2. **Geometric Interpretation**:
   - **Absolute value**: Volume scaling factor of linear transformation
   - **Sign**: Orientation preservation (+) or reversal (-)
   - **Zero**: Transformation collapses space to lower dimension

3. **System of Linear Equations**:
   - **det(A) ≠ 0**: Unique solution exists
   - **det(A) = 0**: No solution or infinitely many solutions

4. **Linear Independence**:
   - **det(A) ≠ 0**: Column/row vectors are linearly independent
   - **det(A) = 0**: Column/row vectors are linearly dependent

**Key Properties:**

1. **Multiplicativity**: det(AB) = det(A) × det(B)
2. **Transpose**: det(Aᵀ) = det(A)
3. **Inverse**: det(A⁻¹) = 1/det(A) (when inverse exists)
4. **Scalar multiplication**: det(kA) = kⁿdet(A) for n×n matrix
5. **Row operations**:
   - Row swap: changes sign
   - Row multiplication by k: multiplies determinant by k
   - Row addition: doesn't change determinant

**Machine Learning Applications:**

1. **Principal Component Analysis (PCA)**:
   - **Covariance matrix determinant**: Measure of data spread
   - **Eigenvalue computation**: Characteristic polynomial

2. **Gaussian Distributions**:
   - **Multivariate normal**: |2πΣ|^(-1/2) in probability density
   - **Covariance matrix**: det(Σ) indicates data concentration

3. **Optimization**:
   - **Hessian determinant**: Second-order optimization conditions
   - **Convexity checking**: Positive definite matrices

4. **Regularization**:
   - **Ridge regression**: det(XᵀX + λI) for numerical stability
   - **Condition number**: Related to determinant magnitude

5. **Computer Graphics**:
   - **Transformation matrices**: Volume preservation/scaling
   - **3D rendering**: Backface culling using determinant signs

**Computational Methods:**

```python
import numpy as np
from scipy.linalg import det

# Basic determinant calculation
A = np.array([[1, 2, 3],
              [4, 5, 6], 
              [7, 8, 9]])
determinant = np.linalg.det(A)
print(f"Determinant: {determinant:.6f}")  # Close to 0 (singular)

# Check matrix invertibility
def is_invertible(matrix, tolerance=1e-10):
    return abs(np.linalg.det(matrix)) > tolerance

# Volume scaling in transformation
def transformation_volume_factor(transformation_matrix):
    return abs(np.linalg.det(transformation_matrix))

# Condition number (related to determinant)
def condition_number(matrix):
    return np.linalg.cond(matrix)

# Example: 2D area scaling
def area_scaling_2d(transform_2x2):
    return abs(np.linalg.det(transform_2x2))

# Rotation matrix (determinant = 1)
angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
print(f"Rotation determinant: {np.linalg.det(rotation_matrix):.6f}")  # = 1
```

**Computational Complexity:**
- **Naive expansion**: O(n!) - extremely expensive
- **LU decomposition**: O(n³) - practical for large matrices
- **Gaussian elimination**: O(n³) - most common approach

**Special Cases:**

1. **Triangular matrices**: Product of diagonal elements
2. **Diagonal matrices**: Product of diagonal elements  
3. **Orthogonal matrices**: det(Q) = ±1
4. **Identity matrix**: det(I) = 1
5. **Zero matrix**: det(0) = 0

**Common Applications in Data Science:**
- **Feature correlation**: Determinant of correlation matrix
- **Dimensionality assessment**: Near-zero determinants indicate redundancy
- **Numerical stability**: Monitor determinant magnitude during iterations
- **Model selection**: Compare determinants of different covariance structures

---

## Question 9

**Can you explain what an eigenvector and eigenvalue are?**

**Answer:** Eigenvectors and eigenvalues are fundamental concepts in linear algebra that describe special directions and scaling factors for linear transformations. They reveal the intrinsic geometric properties of matrices and have profound applications across machine learning, data science, and engineering.

**Mathematical Definition:**
For a square matrix A and non-zero vector **v**, if:
```
A**v** = λ**v**
```
Then:
- **v** is an **eigenvector** of A
- λ (lambda) is the corresponding **eigenvalue**

**Intuitive Understanding:**
- **Eigenvector**: A direction that doesn't change when the matrix transformation is applied
- **Eigenvalue**: The factor by which the eigenvector is scaled during transformation

**Key Properties:**

1. **Direction Preservation**: Eigenvectors maintain their direction under transformation
2. **Scaling**: Eigenvalues determine how much the eigenvector is stretched/compressed
3. **Multiple Eigenvectors**: An n×n matrix has up to n linearly independent eigenvectors
4. **Complex Values**: Eigenvalues can be complex numbers (especially for rotation matrices)

**Geometric Interpretation:**

```
Original vector:    v = [3, 1]
After transformation Av:
- If λ = 2: Result is [6, 2] (same direction, doubled magnitude)
- If λ = -1: Result is [-3, -1] (opposite direction, same magnitude)
- If λ = 0.5: Result is [1.5, 0.5] (same direction, halved magnitude)
```

**Finding Eigenvalues and Eigenvectors:**

1. **Characteristic Equation**: det(A - λI) = 0
2. **Solve for λ**: Roots give eigenvalues
3. **For each λ**: Solve (A - λI)**v** = **0** for eigenvectors

**Machine Learning Applications:**

1. **Principal Component Analysis (PCA)**:
   - **Eigenvectors**: Principal components (directions of maximum variance)
   - **Eigenvalues**: Amount of variance explained by each component
   - **Dimensionality reduction**: Keep top k eigenvectors

2. **Spectral Clustering**:
   - **Graph Laplacian**: Eigenvalues reveal cluster structure
   - **Eigenvectors**: Used to embed data for clustering
   - **Connectivity**: Second smallest eigenvalue (algebraic connectivity)

3. **Google's PageRank Algorithm**:
   - **Dominant eigenvector**: Represents page importance scores
   - **Eigenvalue = 1**: Steady-state of random walk process
   - **Web graph**: Matrix represents link structure

4. **Neural Network Analysis**:
   - **Weight matrices**: Eigenvalues indicate gradient flow properties
   - **Stability analysis**: Eigenvalue magnitudes determine convergence
   - **Activation landscapes**: Hessian eigenvalues for optimization

5. **Markov Chains**:
   - **Transition matrices**: Eigenvalue 1 corresponds to stationary distribution
   - **Convergence rate**: Second largest eigenvalue determines mixing time
   - **Steady state**: Dominant eigenvector gives long-term probabilities

**Implementation Examples:**

```python
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

# Basic eigenvalue/eigenvector computation
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verify the eigenvalue equation
for i, (λ, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
    Av = A @ v
    λv = λ * v
    print(f"λ{i+1}: {λ:.3f}")
    print(f"Av = {Av}")
    print(f"λv = {λv}")
    print(f"Equal: {np.allclose(Av, λv)}\n")

# PCA example
def pca_eigendecomposition(data, n_components=2):
    """Perform PCA using eigendecomposition"""
    # Center the data
    centered_data = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered_data.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    principal_components = eigenvectors[:, :n_components]
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return principal_components, explained_variance_ratio

# Power iteration for dominant eigenvalue
def power_iteration(A, num_iterations=100):
    """Find dominant eigenvalue using power iteration"""
    # Random initial vector
    v = np.random.rand(A.shape[0])
    
    for _ in range(num_iterations):
        # Matrix-vector multiplication
        v = A @ v
        # Normalize
        v = v / np.linalg.norm(v)
    
    # Compute eigenvalue
    eigenvalue = v.T @ A @ v
    return eigenvalue, v

# Spectral analysis of graphs
def graph_spectral_analysis(adjacency_matrix):
    """Analyze graph using eigenvalues of Laplacian"""
    # Degree matrix
    D = np.diag(np.sum(adjacency_matrix, axis=1))
    # Laplacian matrix
    L = D - adjacency_matrix
    
    eigenvalues, eigenvectors = np.linalg.eig(L)
    
    # Sort eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Algebraic connectivity (second smallest eigenvalue)
    algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0
    
    return eigenvalues, eigenvectors, algebraic_connectivity
```

**Special Cases:**

1. **Symmetric Matrices**: Always have real eigenvalues and orthogonal eigenvectors
2. **Positive Definite**: All eigenvalues are positive
3. **Orthogonal Matrices**: All eigenvalues have magnitude 1
4. **Diagonal Matrices**: Diagonal elements are eigenvalues
5. **Identity Matrix**: All eigenvalues equal 1

**Computational Considerations:**
- **Time Complexity**: O(n³) for general matrices
- **Iterative Methods**: Power iteration, Lanczos for large sparse matrices
- **Numerical Stability**: Use specialized algorithms (QR, Jacobi) for better precision
- **Memory**: O(n²) storage for dense matrices

**Practical Tips:**
- **Condition Number**: Ratio of largest to smallest eigenvalue indicates numerical stability
- **Rank Deficiency**: Zero eigenvalues indicate singular matrix
- **Clustering Applications**: Use eigenvector components as features for clustering
- **Visualization**: Plot eigenvectors to understand transformation directions

---

## Question 10

**How is the trace of a matrix defined and what is its relevance?**

**Answer:** The trace of a matrix is the sum of its diagonal elements. Despite its simple definition, the trace is a fundamental invariant with important theoretical properties and practical applications in machine learning, optimization, and linear algebra.

**Mathematical Definition:**
For an n×n square matrix A:
```
tr(A) = a₁₁ + a₂₂ + a₃₃ + ... + aₙₙ = Σᵢ aᵢᵢ
```

**Example:**
```
A = [2  3  1]    →    tr(A) = 2 + 5 + 9 = 16
    [4  5  6]
    [7  8  9]
```

**Key Properties:**

1. **Linearity**: tr(A + B) = tr(A) + tr(B)
2. **Scalar multiplication**: tr(cA) = c·tr(A)
3. **Transpose invariance**: tr(Aᵀ) = tr(A)
4. **Cyclic property**: tr(ABC) = tr(BCA) = tr(CAB)
5. **Similarity invariance**: tr(P⁻¹AP) = tr(A) for any invertible P

**Eigenvalue Connection:**
```
tr(A) = λ₁ + λ₂ + ... + λₙ
```
The trace equals the sum of all eigenvalues (counting multiplicities).

**Machine Learning Applications:**

1. **Neural Network Regularization**:
   - **Weight matrices**: tr(WᵀW) for Frobenius norm regularization
   - **Nuclear norm**: Sum of singular values (related to trace)
   - **Spectral regularization**: Control eigenvalue magnitudes

2. **Covariance Analysis**:
   - **Total variance**: tr(Σ) gives sum of variances across all dimensions
   - **Data concentration**: Higher trace indicates more spread
   - **Dimensionality assessment**: Compare traces of different covariance matrices

3. **Optimization**:
   - **Gradient computation**: tr(AᵀB) appears in matrix derivatives
   - **Loss functions**: Many ML objectives involve trace operations
   - **Hessian analysis**: tr(H) provides second-order information

4. **Principal Component Analysis**:
   - **Explained variance**: tr(Λ) where Λ is diagonal eigenvalue matrix
   - **Compression ratio**: Ratio of retained to total trace
   - **Quality metric**: Trace of reconstructed vs original covariance

5. **Matrix Completion**:
   - **Nuclear norm minimization**: Minimize sum of singular values
   - **Low-rank approximation**: Trace-based constraints
   - **Recommendation systems**: Matrix factorization with trace regularization

**Implementation Examples:**

```python
import numpy as np

# Basic trace calculation
A = np.array([[2, 3, 1],
              [4, 5, 6],
              [7, 8, 9]])
trace_A = np.trace(A)
print(f"Trace of A: {trace_A}")  # Output: 16

# Alternative calculation
trace_manual = np.sum(np.diag(A))
print(f"Manual trace: {trace_manual}")

# Trace properties demonstration
B = np.random.rand(3, 3)
C = np.random.rand(3, 3)

# Linearity
print(f"tr(A+B) = {np.trace(A+B):.3f}")
print(f"tr(A)+tr(B) = {np.trace(A)+np.trace(B):.3f}")

# Cyclic property
print(f"tr(ABC) = {np.trace(A @ B @ C):.3f}")
print(f"tr(BCA) = {np.trace(B @ C @ A):.3f}")
print(f"tr(CAB) = {np.trace(C @ A @ B):.3f}")

# Eigenvalue connection
eigenvalues = np.linalg.eigvals(A)
sum_eigenvalues = np.sum(eigenvalues)
print(f"tr(A) = {np.trace(A):.3f}")
print(f"Sum of eigenvalues = {sum_eigenvalues:.3f}")

# Covariance analysis
def analyze_data_spread(data):
    """Analyze data spread using trace of covariance matrix"""
    cov_matrix = np.cov(data.T)
    total_variance = np.trace(cov_matrix)
    return total_variance, cov_matrix

# PCA with trace analysis
def pca_with_trace_analysis(data):
    """PCA with trace-based variance analysis"""
    # Center data
    centered_data = data - np.mean(data, axis=0)
    
    # Covariance matrix
    cov_matrix = np.cov(centered_data.T)
    total_variance = np.trace(cov_matrix)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Cumulative explained variance
    cumulative_variance = np.cumsum(eigenvalues)
    explained_variance_ratio = cumulative_variance / total_variance
    
    return eigenvalues, eigenvectors, explained_variance_ratio, total_variance

# Nuclear norm using trace
def nuclear_norm(matrix):
    """Compute nuclear norm (sum of singular values)"""
    U, s, Vt = np.linalg.svd(matrix)
    return np.sum(s)  # This is tr(sqrt(A^T A))

# Frobenius norm using trace  
def frobenius_norm_squared(matrix):
    """Compute squared Frobenius norm using trace"""
    return np.trace(matrix.T @ matrix)

# Regularization example
def ridge_regression_with_trace(X, y, lambda_reg):
    """Ridge regression highlighting trace in regularization"""
    # Normal equation with regularization
    XtX = X.T @ X
    regularization_term = lambda_reg * np.eye(X.shape[1])
    
    # The regularization adds lambda * tr(I) = lambda * n to the objective
    coefficients = np.linalg.solve(XtX + regularization_term, X.T @ y)
    
    # Effective degrees of freedom (involves trace)
    H = X @ np.linalg.solve(XtX + regularization_term, X.T)
    effective_dof = np.trace(H)
    
    return coefficients, effective_dof

# Matrix similarity and trace invariance
def demonstrate_similarity_invariance():
    """Show that trace is invariant under similarity transformations"""
    A = np.random.rand(4, 4)
    P = np.random.rand(4, 4)
    
    # Ensure P is invertible
    while np.abs(np.linalg.det(P)) < 1e-10:
        P = np.random.rand(4, 4)
    
    P_inv = np.linalg.inv(P)
    similar_matrix = P_inv @ A @ P
    
    print(f"tr(A) = {np.trace(A):.6f}")
    print(f"tr(P⁻¹AP) = {np.trace(similar_matrix):.6f}")
    print(f"Difference: {abs(np.trace(A) - np.trace(similar_matrix)):.10f}")
```

**Special Cases:**

1. **Identity Matrix**: tr(I) = n (dimension of matrix)
2. **Zero Matrix**: tr(0) = 0
3. **Diagonal Matrix**: tr(D) = sum of diagonal elements
4. **Symmetric Matrix**: tr(A) = tr(Aᵀ) (always true, but eigenvalues are real)
5. **Orthogonal Matrix**: tr(Q) can vary, but |tr(Q)| ≤ n

**Advanced Applications:**

1. **Spectral Learning**: Use trace to monitor eigenvalue distributions
2. **Graph Analysis**: tr(Aᵏ) counts closed walks of length k
3. **Quantum Computing**: Trace operations in density matrices
4. **Signal Processing**: Trace of autocorrelation matrices
5. **Optimization**: Trace-based constraints in semidefinite programming

**Computational Efficiency:**
- **Time Complexity**: O(n) for trace computation
- **Memory**: No additional storage needed beyond matrix
- **Numerical Stability**: Generally stable operation
- **Parallelization**: Diagonal elements can be summed in parallel

**Relationship to Other Concepts:**
- **Determinant**: Both are matrix invariants, but trace is linear while determinant is multiplicative
- **Norm**: Frobenius norm squared = tr(AᵀA)
- **Rank**: No direct relationship, but both provide matrix information
- **Condition Number**: Trace can help assess numerical conditioning

---

## Question 11

**What is a diagonal matrix and how is it used in linear algebra?**

**Answer:** A diagonal matrix is a square matrix where all non-diagonal elements are zero. Only the main diagonal (from top-left to bottom-right) can contain non-zero values. It's one of the most computationally efficient and theoretically important matrix types.

**Definition:**
```
D = [d₁  0   0  ]
    [0   d₂  0  ]
    [0   0   d₃ ]
```

**Key Properties:**
- **Multiplication**: Very fast O(n) operations
- **Inverse**: D⁻¹ has diagonal elements 1/dᵢ (if dᵢ ≠ 0)
- **Powers**: Dᵏ has diagonal elements dᵢᵏ
- **Eigenvalues**: Diagonal elements are the eigenvalues
- **Determinant**: Product of diagonal elements

**Applications:**
- **Scaling transformations**: Each axis scaled independently
- **Eigenvalue decomposition**: A = QDQ⁻¹ for symmetric matrices
- **Principal Component Analysis**: Eigenvalue matrix in PCA
- **Neural networks**: Efficient computation in certain layers

---

## Question 12

**Explain the properties of an identity matrix.**

**Answer:** The identity matrix is a special diagonal matrix with all diagonal elements equal to 1. It serves as the multiplicative identity in matrix algebra, analogous to the number 1 in scalar arithmetic.

**Definition:**
```
I₃ = [1  0  0]
     [0  1  0]
     [0  0  1]
```

**Key Properties:**
- **Multiplicative identity**: AI = IA = A for any compatible matrix A
- **Inverse**: I⁻¹ = I (self-inverse)
- **Determinant**: det(I) = 1
- **Trace**: tr(I) = n (matrix dimension)
- **Eigenvalues**: All eigenvalues equal 1
- **Rank**: rank(I) = n (full rank)

**Applications:**
- **System solving**: Converting Ax = b to x = A⁻¹b
- **Regularization**: Ridge regression uses (XᵀX + λI)
- **Initialization**: Neural network weight initialization
- **Coordinate systems**: Standard basis representation

---

## Question 13

**What is a unit vector and how do you find it?**

**Answer:** A unit vector is a vector with magnitude (length) equal to 1. Unit vectors are fundamental in linear algebra as they provide direction information without magnitude bias, making them essential for normalization, coordinate systems, and many geometric computations.

**Mathematical Definition:**
A vector **û** is a unit vector if:
```
||û|| = 1
```
where ||û|| denotes the magnitude (norm) of the vector.

**How to Find a Unit Vector:**

**1. From Any Non-Zero Vector:**
Given a vector **v** ≠ **0**, the unit vector **û** in the same direction is:
```
û = v / ||v||
```

**2. Magnitude Calculation:**
For vector **v** = [v₁, v₂, ..., vₙ]:
```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

**3. Normalization Process:**
```
û = [v₁/||v||, v₂/||v||, ..., vₙ/||v||]
```

**Examples:**

**2D Vector:**
```
v = [3, 4]
||v|| = √(3² + 4²) = √(9 + 16) = 5
û = [3/5, 4/5] = [0.6, 0.8]
Verification: ||û|| = √(0.6² + 0.8²) = √(0.36 + 0.64) = 1 ✓
```

**3D Vector:**
```
v = [1, 2, 2]
||v|| = √(1² + 2² + 2²) = √9 = 3
û = [1/3, 2/3, 2/3]
```

**Properties of Unit Vectors:**

1. **Direction Preservation**: Unit vector points in same direction as original
2. **Magnitude**: Always has length 1
3. **Dot Product**: û · û = 1
4. **Scaling**: Any vector v = ||v|| × û (magnitude × direction)
5. **Orthogonality**: Two unit vectors are orthogonal if their dot product is 0

**Types of Unit Vectors:**

**1. Standard Basis Vectors:**
- **2D**: î = [1, 0], ĵ = [0, 1]
- **3D**: î = [1, 0, 0], ĵ = [0, 1, 0], k̂ = [0, 0, 1]
- **nD**: eᵢ has 1 in position i, 0 elsewhere

**2. Normalized Vectors:**
Any vector divided by its magnitude becomes a unit vector

**3. Direction Vectors:**
Unit vectors representing specific directions (e.g., surface normals)

**Applications in Machine Learning:**

**1. Feature Normalization:**
- **L2 Normalization**: Convert feature vectors to unit vectors
- **Prevents scale bias**: Features with different scales don't dominate
- **Cosine Similarity**: Compute angles between data points

**2. Gradient Normalization:**
- **Optimization**: Normalize gradients to unit vectors for stable training
- **Gradient Clipping**: Prevent exploding gradients in deep networks
- **Direction Focus**: Emphasize gradient direction over magnitude

**3. Principal Component Analysis:**
- **Principal Components**: Eigenvectors are normalized to unit vectors
- **Orthonormal Basis**: Set of orthogonal unit vectors spanning space
- **Projection**: Project data onto unit principal component vectors

**4. Neural Networks:**
- **Weight Normalization**: Normalize weight vectors to improve training
- **Activation Functions**: Some activations benefit from unit input vectors
- **Attention Mechanisms**: Query/key vectors often normalized

**5. Computer Graphics & Vision:**
- **Surface Normals**: Unit vectors perpendicular to surfaces
- **Lighting Calculations**: Direction vectors for light sources
- **Camera Orientations**: Unit vectors for viewing directions

**Implementation Examples:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Basic unit vector calculation
def make_unit_vector(vector):
    """Convert vector to unit vector"""
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        raise ValueError("Cannot normalize zero vector")
    return vector / magnitude

# Example calculations
v1 = np.array([3, 4])
u1 = make_unit_vector(v1)
print(f"Original: {v1}, Magnitude: {np.linalg.norm(v1):.3f}")
print(f"Unit vector: {u1}, Magnitude: {np.linalg.norm(u1):.3f}")

# Batch normalization for multiple vectors
def normalize_rows(matrix):
    """Normalize each row to unit vector"""
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero
    row_norms = np.where(row_norms == 0, 1, row_norms)
    return matrix / row_norms

# Feature normalization example
def l2_normalize_features(X):
    """L2 normalize features for ML"""
    return X / np.linalg.norm(X, axis=1, keepdims=True)

# Generate random unit vectors
def random_unit_vector(dimension):
    """Generate random unit vector in n dimensions"""
    # Generate random vector
    random_vector = np.random.randn(dimension)
    # Normalize to unit length
    return random_vector / np.linalg.norm(random_vector)

# Cosine similarity using unit vectors
def cosine_similarity(u, v):
    """Compute cosine similarity between vectors"""
    u_unit = make_unit_vector(u)
    v_unit = make_unit_vector(v)
    return np.dot(u_unit, v_unit)

# Orthonormal basis generation
def gram_schmidt_orthonormalization(vectors):
    """Create orthonormal basis using Gram-Schmidt process"""
    orthonormal_basis = []
    
    for v in vectors:
        # Start with current vector
        u = v.copy()
        
        # Subtract projections onto previous orthonormal vectors
        for basis_vector in orthonormal_basis:
            projection = np.dot(u, basis_vector) * basis_vector
            u = u - projection
        
        # Normalize to unit vector
        if np.linalg.norm(u) > 1e-10:  # Avoid numerical issues
            u = u / np.linalg.norm(u)
            orthonormal_basis.append(u)
    
    return np.array(orthonormal_basis)

# Unit vector in spherical coordinates (3D)
def spherical_to_unit_vector(theta, phi):
    """Convert spherical coordinates to 3D unit vector"""
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])

# Demonstration with various applications
def demonstrate_unit_vectors():
    """Comprehensive demonstration of unit vector applications"""
    
    print("="*60)
    print("UNIT VECTOR DEMONSTRATIONS")
    print("="*60)
    
    # 1. Basic normalization
    print("\n1. Basic Vector Normalization:")
    vectors = [
        [3, 4],
        [1, 1, 1],
        [5, 0, 12],
        [-2, -3, 6]
    ]
    
    for v in vectors:
        v_array = np.array(v)
        u = make_unit_vector(v_array)
        print(f"  {v} → {u} (magnitude: {np.linalg.norm(u):.6f})")
    
    # 2. Standard basis vectors
    print("\n2. Standard Basis Vectors:")
    for dim in [2, 3, 4]:
        basis = np.eye(dim)
        print(f"  {dim}D basis: {basis.tolist()}")
    
    # 3. Feature normalization example
    print("\n3. Feature Normalization Example:")
    # Synthetic data with different scales
    data = np.array([
        [1000, 5, 0.1],    # Income, years, ratio
        [2000, 3, 0.2],
        [1500, 7, 0.15],
        [3000, 2, 0.3]
    ])
    
    print(f"  Original data:\n{data}")
    normalized_data = l2_normalize_features(data)
    print(f"  L2 normalized:\n{normalized_data}")
    print(f"  Row magnitudes: {np.linalg.norm(normalized_data, axis=1)}")
    
    # 4. Orthonormal basis creation
    print("\n4. Orthonormal Basis Creation:")
    original_vectors = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])
    
    orthonormal = gram_schmidt_orthonormalization(original_vectors)
    print(f"  Original vectors:\n{original_vectors}")
    print(f"  Orthonormal basis:\n{orthonormal}")
    
    # Verify orthonormality
    dot_products = orthonormal @ orthonormal.T
    print(f"  Gram matrix (should be identity):\n{dot_products}")
    
    # 5. Random unit vectors for Monte Carlo
    print("\n5. Random Unit Vectors (Monte Carlo sampling):")
    random_units = [random_unit_vector(3) for _ in range(5)]
    for i, u in enumerate(random_units):
        print(f"  Vector {i+1}: {u} (magnitude: {np.linalg.norm(u):.6f})")

# Special cases and error handling
def safe_normalize(vector, epsilon=1e-12):
    """Safely normalize vector with numerical stability"""
    magnitude = np.linalg.norm(vector)
    
    if magnitude < epsilon:
        # Return zero vector or raise warning
        print(f"Warning: Vector magnitude {magnitude} is too small to normalize safely")
        return np.zeros_like(vector)
    
    return vector / magnitude

# Unit vector derivatives (for optimization)
def unit_vector_derivative(vector):
    """Compute derivative of unit vector with respect to original vector"""
    v = np.array(vector)
    norm_v = np.linalg.norm(v)
    
    if norm_v == 0:
        raise ValueError("Cannot compute derivative of zero vector normalization")
    
    # d(v/||v||)/dv = (I - (v⊗v)/||v||²) / ||v||
    I = np.eye(len(v))
    outer_product = np.outer(v, v)
    
    derivative = (I - outer_product / (norm_v**2)) / norm_v
    return derivative

if __name__ == "__main__":
    demonstrate_unit_vectors()
```

**Important Considerations:**

**1. Numerical Stability:**
- **Zero vectors**: Cannot be normalized (division by zero)
- **Very small vectors**: May cause numerical instability
- **Floating-point precision**: Use appropriate tolerance for comparisons

**2. Computational Efficiency:**
- **Batch operations**: Normalize multiple vectors simultaneously
- **In-place operations**: Modify vectors directly to save memory
- **Vectorization**: Use NumPy/tensor operations for speed

**3. Geometric Interpretation:**
- **Unit sphere**: All unit vectors lie on the unit sphere in n-dimensional space
- **Direction only**: Unit vectors represent pure direction information
- **Angle preservation**: Angles between unit vectors equal angles between original vectors

**Common Use Cases:**
- **Data preprocessing**: Normalize features before machine learning
- **Optimization**: Gradient direction in gradient descent
- **Computer graphics**: Surface normals, lighting calculations
- **Physics**: Velocity directions, force directions
- **Statistics**: Principal component directions in PCA

**Best Practices:**
- Always check for zero vectors before normalization
- Use appropriate numerical tolerance for floating-point comparisons
- Consider the geometric meaning of normalization in your application
- Use vectorized operations for efficiency with large datasets

---

## Question 14

**Explain the concept of an orthogonal matrix.**

**Answer:** An orthogonal matrix is a square matrix whose columns (and rows) form an orthonormal set of vectors. This means the columns are mutually orthogonal (perpendicular) and each has unit length. Orthogonal matrices preserve angles and distances, making them fundamental in linear algebra and geometry.

**Mathematical Definition:**
A matrix Q is orthogonal if:
```
Q^T Q = Q Q^T = I
```
where Q^T is the transpose of Q and I is the identity matrix.

**Equivalent Conditions:**
1. **Columns are orthonormal**: qᵢ · qⱼ = δᵢⱼ (Kronecker delta)
2. **Rows are orthonormal**: Row vectors satisfy same orthonormality
3. **Inverse equals transpose**: Q⁻¹ = Q^T
4. **Preserves dot products**: (Qx) · (Qy) = x · y

**Key Properties:**

**1. Determinant:**
```
det(Q) = ±1
```
- det(Q) = +1: Special orthogonal (pure rotation)
- det(Q) = -1: Includes reflection

**2. Eigenvalues:**
- All eigenvalues have magnitude 1
- For real matrices: eigenvalues are ±1 or complex conjugate pairs on unit circle

**3. Preservation Properties:**
- **Norm preservation**: ||Qx|| = ||x|| for any vector x
- **Angle preservation**: Angle between Qx and Qy equals angle between x and y
- **Distance preservation**: ||Qx - Qy|| = ||x - y||

**4. Composition:**
- Product of orthogonal matrices is orthogonal
- Inverse of orthogonal matrix is orthogonal

**Types of Orthogonal Matrices:**

**1. Rotation Matrices (2D):**
```
R(θ) = [cos θ  -sin θ]
       [sin θ   cos θ]
```

**2. Reflection Matrices:**
```
Reflection across x-axis: [1   0]
                         [0  -1]
```

**3. Permutation Matrices:**
```
P = [0  1  0]  (swaps first two coordinates)
    [1  0  0]
    [0  0  1]
```

**4. Householder Reflectors:**
```
H = I - 2vv^T  (where v is unit vector)
```

**Construction Methods:**

**1. Gram-Schmidt Process:**
Given linearly independent vectors, create orthonormal basis

**2. QR Decomposition:**
Any matrix A = QR where Q is orthogonal, R is upper triangular

**3. Rotation Composition:**
Combine elementary rotations in different planes

**4. Rodrigues' Formula:**
Rotation around arbitrary axis by angle θ

**Applications in Machine Learning:**

**1. Principal Component Analysis (PCA):**
- **Eigenvector matrix**: Columns are principal components (orthonormal)
- **Data transformation**: Rotate data to principal component coordinates
- **Dimensionality reduction**: Project onto subspace spanned by top components

**2. Singular Value Decomposition (SVD):**
- **A = UΣV^T**: Both U and V are orthogonal matrices
- **Applications**: Matrix completion, recommendation systems, image compression
- **Pseudoinverse**: Use orthogonal matrices to compute Moore-Penrose inverse

**3. Neural Network Initialization:**
- **Orthogonal weight initialization**: Preserve gradient magnitudes
- **Recurrent networks**: Prevent vanishing/exploding gradients
- **Skip connections**: Maintain signal flow in deep networks

**4. Computer Vision:**
- **Camera calibration**: Rotation matrices for camera orientation
- **3D transformations**: Object pose estimation and tracking
- **Image registration**: Align images using orthogonal transformations

**5. Natural Language Processing:**
- **Word embeddings**: Orthogonal transformations preserve semantic relationships
- **Attention mechanisms**: Orthogonal projections in transformer models
- **Language model fine-tuning**: Orthogonal adapters for domain adaptation

**Implementation Examples:**

```python
import numpy as np
from scipy.linalg import qr, svd

# Check if matrix is orthogonal
def is_orthogonal(matrix, tolerance=1e-10):
    """Check if matrix is orthogonal"""
    Q = np.array(matrix)
    if Q.shape[0] != Q.shape[1]:
        return False
    
    # Check Q^T Q = I
    product = Q.T @ Q
    identity = np.eye(Q.shape[0])
    
    return np.allclose(product, identity, atol=tolerance)

# Generate orthogonal matrix using QR decomposition
def random_orthogonal_matrix(n):
    """Generate random orthogonal matrix"""
    # Start with random matrix
    A = np.random.randn(n, n)
    # QR decomposition gives orthogonal Q
    Q, R = qr(A)
    return Q

# Gram-Schmidt orthogonalization
def gram_schmidt(vectors):
    """Convert vectors to orthonormal basis"""
    vectors = np.array(vectors, dtype=float)
    n_vectors, dimension = vectors.shape
    
    orthonormal = np.zeros_like(vectors)
    
    for i in range(n_vectors):
        # Start with current vector
        vector = vectors[i].copy()
        
        # Subtract projections onto previous orthonormal vectors
        for j in range(i):
            projection = np.dot(vector, orthonormal[j]) * orthonormal[j]
            vector -= projection
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            orthonormal[i] = vector / norm
        else:
            print(f"Warning: Vector {i} is linearly dependent")
    
    return orthonormal

# Householder reflector
def householder_reflector(x):
    """Create Householder reflector that maps x to ||x||e_1"""
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Target vector: ||x|| * e_1
    alpha = np.linalg.norm(x)
    e1 = np.zeros(n)
    e1[0] = 1
    
    # Choose sign to avoid cancellation
    if x[0] >= 0:
        alpha = -alpha
    
    # Householder vector
    v = x - alpha * e1
    v = v / np.linalg.norm(v)
    
    # Householder matrix H = I - 2vv^T
    H = np.eye(n) - 2 * np.outer(v, v)
    return H

# Rotation matrix generation
def rotation_matrix_2d(angle):
    """Generate 2D rotation matrix"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([[cos_a, -sin_a],
                     [sin_a,  cos_a]])

def rotation_matrix_3d(axis, angle):
    """Generate 3D rotation matrix using Rodrigues' formula"""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    return R

# Orthogonal Procrustes problem
def orthogonal_procrustes(A, B):
    """Find orthogonal matrix Q that best approximates B ≈ QA"""
    # SVD of A^T B
    U, s, Vt = svd(A.T @ B)
    
    # Optimal orthogonal matrix
    Q = U @ Vt
    
    # Ensure proper rotation (det = 1) if needed
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    
    return Q

# Demonstration of orthogonal matrix properties
def demonstrate_orthogonal_properties():
    """Demonstrate key properties of orthogonal matrices"""
    
    print("="*60)
    print("ORTHOGONAL MATRIX DEMONSTRATIONS")
    print("="*60)
    
    # Generate random orthogonal matrix
    Q = random_orthogonal_matrix(3)
    print(f"\n1. Random 3×3 Orthogonal Matrix:")
    print(f"{Q}")
    
    # Verify orthogonality
    print(f"\n2. Verification of Orthogonality:")
    print(f"Q^T Q =\n{Q.T @ Q}")
    print(f"Is orthogonal: {is_orthogonal(Q)}")
    print(f"Determinant: {np.linalg.det(Q):.6f}")
    
    # Test preservation properties
    print(f"\n3. Preservation Properties:")
    x = np.random.randn(3)
    y = np.random.randn(3)
    
    Qx = Q @ x
    Qy = Q @ y
    
    print(f"Original vectors:")
    print(f"  x = {x}")
    print(f"  y = {y}")
    print(f"  ||x|| = {np.linalg.norm(x):.6f}")
    print(f"  ||y|| = {np.linalg.norm(y):.6f}")
    print(f"  x·y = {np.dot(x, y):.6f}")
    
    print(f"After transformation:")
    print(f"  Qx = {Qx}")
    print(f"  Qy = {Qy}")
    print(f"  ||Qx|| = {np.linalg.norm(Qx):.6f}")
    print(f"  ||Qy|| = {np.linalg.norm(Qy):.6f}")
    print(f"  (Qx)·(Qy) = {np.dot(Qx, Qy):.6f}")
    
    # Different types of orthogonal matrices
    print(f"\n4. Types of Orthogonal Matrices:")
    
    # 2D rotation
    theta = np.pi / 4
    R2D = rotation_matrix_2d(theta)
    print(f"2D Rotation (45°):\n{R2D}")
    print(f"Determinant: {np.linalg.det(R2D):.6f}")
    
    # 3D rotation
    axis = [0, 0, 1]  # z-axis
    R3D = rotation_matrix_3d(axis, theta)
    print(f"\n3D Rotation around z-axis (45°):\n{R3D}")
    print(f"Determinant: {np.linalg.det(R3D):.6f}")
    
    # Reflection
    reflection = np.diag([1, -1, 1])  # Reflect across y-axis
    print(f"\nReflection matrix:\n{reflection}")
    print(f"Determinant: {np.linalg.det(reflection):.6f}")
    
    # Householder reflector
    v = np.array([1, 2, 3])
    H = householder_reflector(v)
    print(f"\nHouseholder reflector for v = {v}:")
    print(f"H =\n{H}")
    print(f"Hv = {H @ v}")
    print(f"Expected: {np.linalg.norm(v) * np.array([1, 0, 0])}")

# PCA using orthogonal matrices
def pca_with_orthogonal_matrices(data, n_components=2):
    """Perform PCA demonstrating orthogonal matrix usage"""
    # Center the data
    centered_data = data - np.mean(data, axis=0)
    
    # Covariance matrix
    cov_matrix = np.cov(centered_data.T)
    
    # Eigendecomposition (eigenvectors form orthogonal matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Verify orthogonality of eigenvector matrix
    is_ortho = is_orthogonal(eigenvectors)
    print(f"Eigenvector matrix is orthogonal: {is_ortho}")
    
    # Select top components
    principal_components = eigenvectors[:, :n_components]
    
    # Transform data (projection onto orthogonal subspace)
    transformed_data = centered_data @ principal_components
    
    return transformed_data, principal_components, eigenvalues

if __name__ == "__main__":
    demonstrate_orthogonal_properties()
    
    # Example with PCA
    print(f"\n{'='*60}")
    print("PCA WITH ORTHOGONAL MATRICES")
    print(f"{'='*60}")
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(100, 4)
    
    transformed, components, values = pca_with_orthogonal_matrices(data)
    print(f"Principal components (orthogonal matrix):\n{components}")
    print(f"Explained variance ratios: {values/np.sum(values)}")
```

**Special Cases and Variations:**

**1. Special Orthogonal Group SO(n):**
- Orthogonal matrices with determinant +1
- Pure rotations without reflections
- Forms a group under matrix multiplication

**2. Orthogonal Matrices in Complex Space:**
- **Unitary matrices**: U†U = I (conjugate transpose)
- Generalization to complex numbers
- Important in quantum mechanics and signal processing

**3. Block Orthogonal Matrices:**
- Block diagonal structure with orthogonal blocks
- Useful for structured transformations
- Applications in parallel processing

**Numerical Considerations:**
- **Precision**: Orthogonality can degrade due to floating-point errors
- **Reorthogonalization**: Periodically recompute to maintain orthogonality
- **Condition numbers**: Well-conditioned for numerical computations
- **Storage**: Can exploit structure for memory efficiency

**Best Practices:**
- Use specialized algorithms (QR, SVD) for constructing orthogonal matrices
- Verify orthogonality numerically with appropriate tolerance
- Leverage orthogonality for efficient computations (Q⁻¹ = Q^T)
- Consider numerical stability when composing multiple orthogonal transformations

---

## Question 15

**What is the rank of a matrix and why is it important?**

**Answer:** The rank of a matrix is the maximum number of linearly independent rows or columns in the matrix. It measures the dimension of the vector space spanned by the matrix's rows (or columns) and provides crucial information about the matrix's properties and the solvability of linear systems.

**Mathematical Definition:**
For matrix A, the rank (denoted rank(A) or rk(A)) is:
```
rank(A) = dim(span(columns of A)) = dim(span(rows of A))
```

**Equivalent Definitions:**
1. **Maximum linearly independent columns/rows**
2. **Dimension of column space (range) of A**
3. **Dimension of row space of A**
4. **Number of non-zero singular values in SVD**
5. **Number of non-zero eigenvalues for positive semidefinite matrices**

**Properties of Matrix Rank:**

**1. Basic Properties:**
- **0 ≤ rank(A) ≤ min(m,n)** for m×n matrix
- **rank(A) = rank(A^T)** (row rank = column rank)
- **rank(AB) ≤ min(rank(A), rank(B))**
- **rank(A + B) ≤ rank(A) + rank(B)**

**2. Special Cases:**
- **Full rank**: rank(A) = min(m,n) (maximum possible)
- **Rank deficient**: rank(A) < min(m,n)
- **Zero matrix**: rank(0) = 0
- **Identity matrix**: rank(I_n) = n

**3. Geometric Interpretation:**
- **2D**: rank = 1 means all vectors lie on a line through origin
- **3D**: rank = 2 means all vectors lie in a plane through origin
- **General**: rank = k means vectors span a k-dimensional subspace

**Methods to Compute Rank:**

**1. Row Reduction (Gaussian Elimination):**
- Convert to row echelon form
- Count non-zero rows

**2. Singular Value Decomposition (SVD):**
- A = UΣV^T
- Count non-zero singular values

**3. Eigenvalue Decomposition:**
- For symmetric matrices: count non-zero eigenvalues

**4. LU Decomposition:**
- Count non-zero diagonal elements in U

**Why Rank is Important:**

**1. Linear System Solvability:**
- **Ax = b consistent ⟺ rank(A) = rank([A|b])**
- **Unique solution ⟺ rank(A) = n (number of variables)**
- **Infinite solutions ⟺ rank(A) < n**

**2. Matrix Invertibility:**
- **A is invertible ⟺ rank(A) = n (for n×n matrix)**
- **Singular matrix ⟺ rank(A) < n**

**3. Dimension of Solution Space:**
- **Null space dimension = n - rank(A)**
- **Number of free variables = n - rank(A)**

**4. Data Analysis:**
- **Feature redundancy**: Low rank indicates correlated features
- **Dimensionality**: Effective dimension of data
- **Compression**: Low-rank approximations for data compression

**Machine Learning Applications:**

**1. Dimensionality Reduction:**
- **PCA**: Find low-rank approximation of covariance matrix
- **Matrix Factorization**: Decompose data into low-rank components
- **Feature Selection**: Remove linearly dependent features

**2. Recommender Systems:**
- **Matrix Completion**: Fill missing entries assuming low-rank structure
- **Collaborative Filtering**: User-item matrix typically low-rank
- **SVD**: Use rank to determine number of latent factors

**3. Computer Vision:**
- **Structure from Motion**: Camera matrices have specific rank constraints
- **Fundamental Matrix**: Rank-2 constraint in epipolar geometry
- **Image Compression**: Low-rank approximation of image matrices

**4. Natural Language Processing:**
- **Term-Document Matrices**: Often low-rank due to semantic structure
- **Word Embeddings**: Dimensionality determines embedding space rank
- **Latent Semantic Analysis**: Use SVD to find low-rank semantic space

**5. Optimization:**
- **Regularization**: Nuclear norm promotes low-rank solutions
- **Convex Relaxation**: Rank minimization → nuclear norm minimization
- **Matrix Sensing**: Recover low-rank matrices from linear measurements

**Implementation Examples:**

```python
import numpy as np
from scipy.linalg import svd, qr
import matplotlib.pyplot as plt

def matrix_rank_methods(A, tolerance=1e-10):
    """Compute matrix rank using different methods"""
    results = {}
    
    # Method 1: NumPy's built-in rank function
    results['numpy'] = np.linalg.matrix_rank(A, tol=tolerance)
    
    # Method 2: SVD-based computation
    U, s, Vt = svd(A)
    results['svd'] = np.sum(s > tolerance)
    results['singular_values'] = s
    
    # Method 3: QR decomposition with pivoting
    Q, R, P = qr(A, pivoting=True)
    # Count non-zero diagonal elements in R
    diag_R = np.abs(np.diag(R))
    results['qr'] = np.sum(diag_R > tolerance)
    
    # Method 4: Eigenvalues (for square matrices)
    if A.shape[0] == A.shape[1]:
        eigenvals = np.linalg.eigvals(A)
        results['eigenvalues'] = np.sum(np.abs(eigenvals) > tolerance)
    
    return results

def demonstrate_rank_properties():
    """Demonstrate rank properties with examples"""
    
    print("="*60)
    print("MATRIX RANK DEMONSTRATIONS")
    print("="*60)
    
    # Example 1: Full rank matrix
    print("\n1. Full Rank Matrix:")
    A_full = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 10]])  # Modified last element for full rank
    
    rank_info = matrix_rank_methods(A_full)
    print(f"Matrix A:\n{A_full}")
    print(f"Rank: {rank_info['numpy']} (out of max {min(A_full.shape)})")
    print(f"Singular values: {rank_info['singular_values']}")
    
    # Example 2: Rank deficient matrix
    print("\n2. Rank Deficient Matrix:")
    A_deficient = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])  # Third row = 2*(second row) - (first row)
    
    rank_info = matrix_rank_methods(A_deficient)
    print(f"Matrix A:\n{A_deficient}")
    print(f"Rank: {rank_info['numpy']} (out of max {min(A_deficient.shape)})")
    print(f"Singular values: {rank_info['singular_values']}")
    
    # Example 3: Effect of linear combinations
    print("\n3. Linear Dependence Example:")
    # Create matrix where third column = first + second column
    B = np.array([[1, 2, 3],
                  [4, 5, 9],
                  [7, 8, 15]])
    
    rank_info = matrix_rank_methods(B)
    print(f"Matrix B (col3 = col1 + col2):\n{B}")
    print(f"Rank: {rank_info['numpy']}")
    print("Column 3 should equal Column 1 + Column 2:")
    print(f"Col1 + Col2 = {B[:, 0] + B[:, 1]}")
    print(f"Col3 = {B[:, 2]}")

# Low-rank approximation example
def low_rank_approximation(matrix, rank):
    """Compute best rank-r approximation using SVD"""
    U, s, Vt = svd(matrix, full_matrices=False)
    
    # Keep only top 'rank' components
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vt_r = Vt[:rank, :]
    
    # Reconstruct
    approximation = U_r @ np.diag(s_r) @ Vt_r
    
    # Compute approximation error
    error = np.linalg.norm(matrix - approximation, 'fro')
    
    return approximation, error

# Rank and linear systems
def analyze_linear_system(A, b):
    """Analyze solvability of linear system Ax = b"""
    rank_A = np.linalg.matrix_rank(A)
    
    # Augmented matrix [A | b]
    augmented = np.column_stack([A, b])
    rank_augmented = np.linalg.matrix_rank(augmented)
    
    m, n = A.shape
    
    print(f"System Analysis:")
    print(f"Matrix A shape: {A.shape}")
    print(f"rank(A) = {rank_A}")
    print(f"rank([A|b]) = {rank_augmented}")
    
    if rank_A != rank_augmented:
        solution_type = "No solution (inconsistent)"
    elif rank_A == n:
        solution_type = "Unique solution"
    else:
        solution_type = f"Infinite solutions ({n - rank_A} free variables)"
    
    print(f"Solution type: {solution_type}")
    return solution_type

# PCA and rank
def pca_rank_analysis(data, explained_variance_threshold=0.95):
    """Analyze effective rank using PCA"""
    # Center the data
    centered_data = data - np.mean(data, axis=0)
    
    # SVD
    U, s, Vt = svd(centered_data, full_matrices=False)
    
    # Explained variance ratios
    explained_variance = s**2 / np.sum(s**2)
    cumulative_variance = np.cumsum(explained_variance)
    
    # Effective rank for given threshold
    effective_rank = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
    
    print(f"Data matrix shape: {data.shape}")
    print(f"Mathematical rank: {np.linalg.matrix_rank(centered_data)}")
    print(f"Effective rank (95% variance): {effective_rank}")
    print(f"Singular values: {s[:min(10, len(s))]}")  # Show first 10
    
    return effective_rank, explained_variance

# Rank and matrix completion
def matrix_completion_demo():
    """Demonstrate rank constraint in matrix completion"""
    # Create low-rank matrix
    np.random.seed(42)
    true_rank = 3
    m, n = 20, 15
    
    # Generate low-rank matrix
    U_true = np.random.randn(m, true_rank)
    V_true = np.random.randn(n, true_rank)
    M_true = U_true @ V_true.T
    
    print(f"True matrix rank: {np.linalg.matrix_rank(M_true)}")
    
    # Create observed matrix with missing entries
    observation_prob = 0.3
    mask = np.random.rand(m, n) < observation_prob
    M_observed = M_true.copy()
    M_observed[~mask] = 0  # Set unobserved entries to 0
    
    print(f"Observed entries: {np.sum(mask)} out of {m*n} ({100*np.sum(mask)/(m*n):.1f}%)")
    
    # Simple completion: low-rank approximation of observed entries
    U, s, Vt = svd(M_observed, full_matrices=False)
    
    # Reconstruct using true rank
    M_completed = U[:, :true_rank] @ np.diag(s[:true_rank]) @ Vt[:true_rank, :]
    
    # Compute error on observed entries
    observed_error = np.linalg.norm((M_true - M_completed)[mask])
    total_error = np.linalg.norm(M_true - M_completed)
    
    print(f"Completion error (observed): {observed_error:.4f}")
    print(f"Completion error (total): {total_error:.4f}")

# Demonstrate with real applications
def demonstrate_rank_applications():
    """Show rank applications in different domains"""
    
    print("\n" + "="*60)
    print("RANK APPLICATIONS")
    print("="*60)
    
    # 1. Image compression
    print("\n1. Image Compression with Low-Rank Approximation:")
    # Create a simple synthetic "image" (low-rank structure)
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    
    # Low-rank image: combination of simple patterns
    image = (np.sin(5*X) * np.cos(5*Y) + 
             0.5 * np.sin(10*X) + 
             0.3 * np.cos(8*Y))
    
    original_rank = np.linalg.matrix_rank(image)
    print(f"Original image rank: {original_rank}")
    
    # Compress with different ranks
    for rank in [1, 5, 10, 20]:
        compressed, error = low_rank_approximation(image, rank)
        compression_ratio = (rank * (image.shape[0] + image.shape[1])) / (image.shape[0] * image.shape[1])
        print(f"Rank {rank:2d}: Error = {error:.4f}, Compression = {compression_ratio:.2f}")
    
    # 2. Feature correlation analysis
    print("\n2. Feature Correlation Analysis:")
    # Generate correlated features
    np.random.seed(123)
    n_samples, n_features = 100, 8
    
    # Create some base features
    base_features = np.random.randn(n_samples, 3)
    
    # Create correlated features
    features = np.column_stack([
        base_features[:, 0],                           # Original
        base_features[:, 1],                           # Original  
        base_features[:, 2],                           # Original
        base_features[:, 0] + 0.1 * np.random.randn(n_samples),  # Almost same as first
        base_features[:, 1] + base_features[:, 2],     # Linear combination
        2 * base_features[:, 0] - base_features[:, 1], # Linear combination
        np.random.randn(n_samples),                    # Independent
        np.random.randn(n_samples)                     # Independent
    ])
    
    feature_rank = np.linalg.matrix_rank(features)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature matrix rank: {feature_rank}")
    print(f"Redundant features: {n_features - feature_rank}")
    
    # 3. Linear system analysis
    print("\n3. Linear System Analysis:")
    examples = [
        # Unique solution
        (np.array([[1, 2], [3, 4]]), np.array([1, 2])),
        # No solution
        (np.array([[1, 2], [2, 4]]), np.array([1, 3])),
        # Infinite solutions
        (np.array([[1, 2], [2, 4]]), np.array([1, 2]))
    ]
    
    for i, (A, b) in enumerate(examples):
        print(f"\nExample {i+1}:")
        analyze_linear_system(A, b)

if __name__ == "__main__":
    demonstrate_rank_properties()
    demonstrate_rank_applications()
    
    print("\n" + "="*60)
    print("PCA RANK ANALYSIS")
    print("="*60)
    
    # Generate sample data with inherent low-rank structure
    np.random.seed(42)
    true_components = 3
    data = np.random.randn(100, true_components) @ np.random.randn(true_components, 10)
    data += 0.1 * np.random.randn(100, 10)  # Add noise
    
    effective_rank, variance_ratios = pca_rank_analysis(data)
    
    print("\n" + "="*60)
    print("MATRIX COMPLETION DEMO")
    print("="*60)
    
    matrix_completion_demo()
```

**Practical Considerations:**

**1. Numerical Rank:**
- **Tolerance**: Use appropriate threshold for near-zero values
- **Condition Number**: High condition number indicates near rank deficiency
- **Numerical Stability**: SVD provides most stable rank computation

**2. Computational Complexity:**
- **SVD**: O(min(m²n, mn²)) for m×n matrix
- **QR with pivoting**: More efficient for rank estimation
- **Randomized methods**: Faster for large matrices

**3. Applications in Optimization:**
- **Rank constraints**: Often non-convex, replaced by nuclear norm
- **Matrix completion**: Recover missing entries assuming low rank
- **Robust PCA**: Separate low-rank and sparse components

**Real-World Impact:**
- **Data compression**: Remove redundant information
- **Feature engineering**: Identify and remove correlated features
- **Dimensionality reduction**: Project to lower-dimensional space
- **System identification**: Determine model complexity
- **Signal processing**: Separate signal from noise

**Best Practices:**
- Use SVD for accurate rank computation
- Consider numerical tolerance appropriate for your data scale
- Monitor rank during iterative algorithms
- Use rank for model selection and complexity control
- Combine with domain knowledge for interpretation

---

## Question 16

**What is the method of Gaussian elimination?**

**Answer:** Gaussian elimination is a systematic algorithm for solving systems of linear equations by transforming the system into row echelon form through a sequence of elementary row operations. It's one of the most fundamental and widely used methods in linear algebra for solving linear systems, finding matrix rank, and computing determinants.

**Basic Concept:**
Transform the augmented matrix [A|b] representing the system Ax = b into an equivalent system that's easy to solve through back substitution.

**Elementary Row Operations:**
1. **Row swapping**: Rᵢ ↔ Rⱼ (interchange two rows)
2. **Row scaling**: Rᵢ → kRᵢ where k ≠ 0 (multiply row by non-zero scalar)
3. **Row addition**: Rᵢ → Rᵢ + kRⱼ (add multiple of one row to another)

**Algorithm Steps:**

**Forward Elimination:**
1. **Choose pivot**: Select non-zero element (preferably largest) in current column
2. **Pivot to diagonal**: Swap rows if necessary to place pivot on diagonal
3. **Eliminate below**: Use row operations to make all elements below pivot zero
4. **Move to next column**: Repeat for remaining columns

**Back Substitution:**
1. **Start from bottom**: Solve for last variable from bottom equation
2. **Substitute upward**: Use solved variables to find remaining unknowns
3. **Complete solution**: Continue until all variables are determined

**Mathematical Example:**

**System:**
```
2x + 3y - z = 1
4x + 4y + z = 7  
2x + y + 3z = 11
```

**Augmented Matrix:**
```
[2  3  -1 |  1]
[4  4   1 |  7]
[2  1   3 | 11]
```

**Step 1: Eliminate first column below pivot**
```
R₂ → R₂ - 2R₁:  [2  3  -1 |  1]
                [0 -2   3 |  5]
                [2  1   3 | 11]

R₃ → R₃ - R₁:   [2  3  -1 |  1]
                [0 -2   3 |  5]
                [0 -2   4 | 10]
```

**Step 2: Eliminate second column below pivot**
```
R₃ → R₃ - R₂:   [2  3  -1 |  1]
                [0 -2   3 |  5]
                [0  0   1 |  5]
```

**Back Substitution:**
```
From equation 3: z = 5
From equation 2: -2y + 3(5) = 5 → y = 5
From equation 1: 2x + 3(5) - 5 = 1 → x = -4.5
```

**Variants of Gaussian Elimination:**

**1. Gaussian Elimination with Partial Pivoting:**
- Choose largest element in column as pivot
- Improves numerical stability
- Standard in most implementations

**2. Gaussian Elimination with Complete Pivoting:**
- Choose largest element in remaining submatrix
- Maximum numerical stability
- More computationally expensive

**3. Gauss-Jordan Elimination:**
- Continues elimination above pivots
- Produces reduced row echelon form (RREF)
- Solution can be read directly

**Applications:**

**1. Solving Linear Systems:**
- **Unique solution**: System has solution when rank(A) = rank([A|b]) = n
- **No solution**: When rank(A) < rank([A|b])
- **Infinite solutions**: When rank(A) = rank([A|b]) < n

**2. Matrix Inversion:**
- Apply Gauss-Jordan to [A|I] to get [I|A⁻¹]
- Verify invertibility during process

**3. Determinant Calculation:**
- det(A) = (-1)ᵖ × (product of pivots)
- p = number of row swaps performed

**4. Rank Computation:**
- Count number of non-zero rows in echelon form
- Identify linearly independent rows/columns

**5. LU Factorization:**
- Record elimination operations to construct L and U matrices
- A = LU where L is lower triangular, U is upper triangular

**Implementation Examples:**

```python
import numpy as np

def gaussian_elimination(A, b, pivoting=True):
    """
    Solve Ax = b using Gaussian elimination with optional pivoting
    
    Args:
        A: Coefficient matrix (n×n)
        b: Right-hand side vector (n×1)
        pivoting: Whether to use partial pivoting
    
    Returns:
        x: Solution vector
        steps: List of elimination steps for educational purposes
    """
    n = len(A)
    # Create augmented matrix
    augmented = np.column_stack([A.astype(float), b.astype(float)])
    steps = []
    
    # Forward elimination
    for i in range(n):
        # Partial pivoting
        if pivoting:
            # Find pivot (largest element in column)
            max_row = i + np.argmax(np.abs(augmented[i:, i]))
            if max_row != i:
                # Swap rows
                augmented[[i, max_row]] = augmented[[max_row, i]]
                steps.append(f"Swap R{i+1} ↔ R{max_row+1}")
        
        # Check for zero pivot
        if abs(augmented[i, i]) < 1e-10:
            raise ValueError(f"Zero pivot encountered at position ({i}, {i})")
        
        # Eliminate below pivot
        for j in range(i + 1, n):
            if augmented[j, i] != 0:
                factor = augmented[j, i] / augmented[i, i]
                augmented[j] -= factor * augmented[i]
                steps.append(f"R{j+1} → R{j+1} - {factor:.3f}×R{i+1}")
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i, -1]
        for j in range(i + 1, n):
            x[i] -= augmented[i, j] * x[j]
        x[i] /= augmented[i, i]
    
    return x, steps, augmented

def gauss_jordan_elimination(A, b=None):
    """
    Perform Gauss-Jordan elimination to get reduced row echelon form
    
    Args:
        A: Matrix to reduce
        b: Optional right-hand side (for solving systems)
    
    Returns:
        Reduced matrix, solution (if b provided)
    """
    if b is not None:
        augmented = np.column_stack([A.astype(float), b.astype(float)])
    else:
        augmented = A.astype(float)
    
    rows, cols = augmented.shape
    
    for i in range(min(rows, cols - (1 if b is not None else 0))):
        # Find pivot
        pivot_row = i + np.argmax(np.abs(augmented[i:, i]))
        
        # Swap if necessary
        if pivot_row != i:
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
        
        # Check for zero pivot
        if abs(augmented[i, i]) < 1e-10:
            continue
        
        # Scale pivot row to make pivot = 1
        augmented[i] /= augmented[i, i]
        
        # Eliminate entire column (above and below)
        for j in range(rows):
            if j != i and augmented[j, i] != 0:
                augmented[j] -= augmented[j, i] * augmented[i]
    
    if b is not None:
        return augmented[:, :-1], augmented[:, -1]
    else:
        return augmented

def compute_determinant_gaussian(A):
    """Compute determinant using Gaussian elimination"""
    A = A.astype(float)
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    
    det = 1.0
    sign = 1
    
    for i in range(n):
        # Find pivot
        max_row = i + np.argmax(np.abs(A[i:, i]))
        
        # Swap rows if necessary
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            sign *= -1  # Row swap changes sign
        
        # Check for zero pivot (singular matrix)
        if abs(A[i, i]) < 1e-10:
            return 0.0
        
        # Update determinant
        det *= A[i, i]
        
        # Eliminate below
        for j in range(i + 1, n):
            if A[j, i] != 0:
                factor = A[j, i] / A[i, i]
                A[j] -= factor * A[i]
    
    return sign * det

def matrix_rank_gaussian(A):
    """Compute matrix rank using Gaussian elimination"""
    A = A.astype(float)
    rows, cols = A.shape
    
    rank = 0
    for i in range(min(rows, cols)):
        # Find pivot in remaining matrix
        remaining = A[rank:, i:]
        if remaining.size == 0:
            break
            
        pivot_pos = np.unravel_index(np.argmax(np.abs(remaining)), remaining.shape)
        pivot_row = rank + pivot_pos[0]
        pivot_col = i + pivot_pos[1]
        
        # Check if pivot is effectively zero
        if abs(A[pivot_row, pivot_col]) < 1e-10:
            continue
        
        # Swap rows to bring pivot to diagonal position
        if pivot_row != rank:
            A[[rank, pivot_row]] = A[[pivot_row, rank]]
        
        # Eliminate below pivot
        for j in range(rank + 1, rows):
            if abs(A[j, pivot_col]) > 1e-10:
                factor = A[j, pivot_col] / A[rank, pivot_col]
                A[j] -= factor * A[rank]
        
        rank += 1
    
    return rank

def lu_factorization_gaussian(A):
    """Perform LU factorization using Gaussian elimination"""
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float)
    P = np.eye(n)  # Permutation matrix
    
    for i in range(n - 1):
        # Partial pivoting
        max_row = i + np.argmax(np.abs(U[i:, i]))
        if max_row != i:
            # Swap rows in U and P
            U[[i, max_row]] = U[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
            # Swap corresponding rows in L (for columns < i)
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]
        
        # Eliminate
        for j in range(i + 1, n):
            if U[i, i] != 0:
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j] -= factor * U[i]
    
    return P, L, U

# Demonstration and educational examples
def demonstrate_gaussian_elimination():
    """Comprehensive demonstration of Gaussian elimination"""
    
    print("="*60)
    print("GAUSSIAN ELIMINATION DEMONSTRATION")
    print("="*60)
    
    # Example 1: Basic system
    print("\n1. Solving Linear System:")
    A1 = np.array([[2, 3, -1],
                   [4, 4, 1],
                   [2, 1, 3]], dtype=float)
    b1 = np.array([1, 7, 11], dtype=float)
    
    print(f"System: Ax = b")
    print(f"A =\n{A1}")
    print(f"b = {b1}")
    
    try:
        x, steps, final_matrix = gaussian_elimination(A1, b1)
        print(f"\nSolution: x = {x}")
        print(f"Verification: Ax = {A1 @ x}")
        print(f"Expected: b = {b1}")
        print(f"Error: {np.linalg.norm(A1 @ x - b1):.2e}")
        
        print(f"\nElimination steps:")
        for step in steps[:5]:  # Show first few steps
            print(f"  {step}")
        
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 2: Singular system
    print(f"\n2. Singular System (No Unique Solution):")
    A2 = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [1, 2, 4]], dtype=float)
    b2 = np.array([1, 2, 2], dtype=float)
    
    print(f"A =\n{A2}")
    print(f"b = {b2}")
    print(f"Matrix rank: {matrix_rank_gaussian(A2.copy())}")
    
    # Example 3: Determinant calculation
    print(f"\n3. Determinant Calculation:")
    A3 = np.array([[4, 2, 1],
                   [3, 1, 2],
                   [1, 3, 4]], dtype=float)
    
    det_gaussian = compute_determinant_gaussian(A3.copy())
    det_numpy = np.linalg.det(A3)
    
    print(f"Matrix A =\n{A3}")
    print(f"Determinant (Gaussian): {det_gaussian:.6f}")
    print(f"Determinant (NumPy): {det_numpy:.6f}")
    print(f"Difference: {abs(det_gaussian - det_numpy):.2e}")
    
    # Example 4: LU Factorization
    print(f"\n4. LU Factorization:")
    P, L, U = lu_factorization_gaussian(A3.copy())
    
    print(f"P =\n{P}")
    print(f"L =\n{L}")
    print(f"U =\n{U}")
    print(f"PA =\n{P @ A3}")
    print(f"LU =\n{L @ U}")
    print(f"Reconstruction error: {np.linalg.norm(P @ A3 - L @ U):.2e}")

# Performance comparison
def compare_methods():
    """Compare Gaussian elimination with other methods"""
    
    print(f"\n{'='*60}")
    print("METHOD COMPARISON")
    print(f"{'='*60}")
    
    sizes = [10, 50, 100]
    methods = [
        ("Gaussian Elimination", gaussian_elimination),
        ("NumPy solve", lambda A, b: (np.linalg.solve(A, b), [], None)),
        ("LU solve", lambda A, b: (np.linalg.solve(A, b), [], None))
    ]
    
    import time
    
    for n in sizes:
        print(f"\nMatrix size: {n}×{n}")
        
        # Generate random well-conditioned system
        np.random.seed(42)
        A = np.random.randn(n, n) + n * np.eye(n)  # Well-conditioned
        b = np.random.randn(n)
        
        for method_name, method_func in methods:
            try:
                start_time = time.time()
                x, _, _ = method_func(A.copy(), b.copy())
                end_time = time.time()
                
                # Verify solution
                residual = np.linalg.norm(A @ x - b)
                
                print(f"  {method_name:20}: {end_time - start_time:.4f}s, residual: {residual:.2e}")
                
            except Exception as e:
                print(f"  {method_name:20}: Error - {e}")

# Educational tool for step-by-step demonstration
def step_by_step_gaussian(A, b, show_steps=True):
    """Show detailed step-by-step Gaussian elimination"""
    
    print(f"Starting system:")
    print(f"Augmented matrix [A|b]:")
    augmented = np.column_stack([A.astype(float), b.astype(float)])
    print(augmented)
    
    n = len(A)
    step_count = 0
    
    # Forward elimination
    for i in range(n):
        if show_steps:
            print(f"\n--- Step {step_count + 1}: Working on column {i + 1} ---")
        
        # Find and apply pivot
        max_row = i + np.argmax(np.abs(augmented[i:, i]))
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
            if show_steps:
                print(f"Swap rows {i + 1} and {max_row + 1}")
                print(augmented)
        
        # Eliminate below pivot
        for j in range(i + 1, n):
            if augmented[j, i] != 0:
                factor = augmented[j, i] / augmented[i, i]
                augmented[j] -= factor * augmented[i]
                if show_steps:
                    print(f"R{j + 1} ← R{j + 1} - {factor:.3f} × R{i + 1}")
                    print(augmented)
        
        step_count += 1
    
    # Back substitution
    if show_steps:
        print(f"\n--- Back Substitution ---")
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i, -1]
        for j in range(i + 1, n):
            x[i] -= augmented[i, j] * x[j]
        x[i] /= augmented[i, i]
        
        if show_steps:
            print(f"x{i + 1} = {x[i]:.4f}")
    
    return x

if __name__ == "__main__":
    demonstrate_gaussian_elimination()
    compare_methods()
    
    print(f"\n{'='*60}")
    print("STEP-BY-STEP EXAMPLE")
    print(f"{'='*60}")
    
    # Small example for detailed walkthrough
    A_demo = np.array([[2, 1, -1],
                       [-3, -1, 2],
                       [-2, 1, 2]], dtype=float)
    b_demo = np.array([8, -11, -3], dtype=float)
    
    x_demo = step_by_step_gaussian(A_demo, b_demo, show_steps=True)
    print(f"\nFinal solution: {x_demo}")
    print(f"Verification: Ax = {A_demo @ x_demo}")
    print(f"Expected: b = {b_demo}")
```

**Computational Complexity:**
- **Time**: O(n³) for n×n system
- **Space**: O(n²) for storing matrix
- **Pivoting overhead**: Minimal additional cost

**Numerical Considerations:**

**1. Pivoting Strategies:**
- **No pivoting**: Fast but numerically unstable
- **Partial pivoting**: Good balance of speed and stability
- **Complete pivoting**: Maximum stability, higher cost

**2. Numerical Stability:**
- **Condition number**: κ(A) = ||A|| ||A⁻¹|| indicates sensitivity
- **Growth factor**: How much elements grow during elimination
- **Backward stability**: Small changes in input produce small changes in computed solution

**3. Error Sources:**
- **Round-off errors**: Accumulate during elimination
- **Catastrophic cancellation**: When subtracting nearly equal numbers
- **Pivot magnitude**: Small pivots amplify errors

**Advantages:**
- **Systematic**: Well-defined algorithm with clear steps
- **General**: Works for any non-singular system
- **Educational**: Easy to understand and implement
- **Foundation**: Basis for many other algorithms (LU, QR)

**Disadvantages:**
- **Numerical instability**: Without pivoting
- **Fill-in**: Can destroy sparsity structure
- **Not parallelizable**: Sequential nature of elimination
- **Memory**: Requires storage of full matrix

**Modern Variations:**
- **Block Gaussian elimination**: For large matrices
- **Iterative refinement**: Improve accuracy of solution
- **Sparse Gaussian elimination**: Preserve sparsity
- **Parallel implementations**: Distribute computation across processors

---

## Question 17

**Explain the concept of linear dependence and independence.**

**Answer:** Linear dependence and independence are fundamental concepts that describe the relationships between vectors in a vector space. They determine whether vectors provide unique directional information and are crucial for understanding dimensions, basis vectors, matrix properties, and solving linear systems.

**Linear Independence Definition:**
A set of vectors {v₁, v₂, ..., vₙ} is **linearly independent** if the only solution to:
```
c₁v₁ + c₂v₂ + ... + cₙvₙ = 0
```
is the trivial solution: c₁ = c₂ = ... = cₙ = 0.

**Linear Dependence Definition:**
A set of vectors is **linearly dependent** if there exist scalars c₁, c₂, ..., cₙ (not all zero) such that:
```
c₁v₁ + c₂v₂ + ... + cₙvₙ = 0
```

**Geometric Interpretation:**

**2D Space:**
- **Independent**: Two vectors not on the same line through origin
- **Dependent**: Two vectors on the same line (one is scalar multiple of other)

**3D Space:**
- **Independent**: Three vectors not in the same plane through origin
- **Dependent**: Three vectors lie in the same plane

**General n-D:**
- **Independent**: Vectors span the full dimensional space they live in
- **Dependent**: Vectors lie in a lower-dimensional subspace

**Key Properties:**

**1. Maximum Independent Vectors:**
- In n-dimensional space, maximum n linearly independent vectors
- Any set of more than n vectors in n-D space is linearly dependent

**2. Subset Property:**
- If a set is linearly independent, all subsets are linearly independent
- If a set is linearly dependent, all supersets are linearly dependent

**3. Zero Vector:**
- Any set containing the zero vector is linearly dependent
- The zero vector can always be written as 0 = 1·0 + 0·v₁ + ... + 0·vₙ

**4. Single Vector:**
- One non-zero vector is always linearly independent
- One zero vector is linearly dependent

**Methods to Test Linear Independence:**

**1. Matrix Method:**
Form matrix with vectors as columns and check rank:
- **rank(A) = number of columns → independent**
- **rank(A) < number of columns → dependent**

**2. Determinant Method (for square matrices):**
- **det(A) ≠ 0 → independent**
- **det(A) = 0 → dependent**

**3. Row Reduction:**
Apply Gaussian elimination:
- **No zero rows → independent**
- **Zero rows appear → dependent**

**4. Null Space:**
- **Only trivial solution to Ax = 0 → independent**
- **Non-trivial solutions exist → dependent**

**Applications in Machine Learning:**

**1. Feature Engineering:**
- **Redundant features**: Linearly dependent features provide no new information
- **Feature selection**: Remove dependent features to reduce overfitting
- **Multicollinearity**: Detect and handle correlated features

**2. Dimensionality Reduction:**
- **Principal Component Analysis**: Find linearly independent directions of maximum variance
- **Basis selection**: Choose independent components for data representation
- **Rank reduction**: Project to space spanned by independent vectors

**3. Neural Networks:**
- **Weight initialization**: Ensure initial weights span desired space
- **Hidden representations**: Encourage diverse, independent feature learning
- **Regularization**: Penalties to prevent weight dependencies

**4. Computer Vision:**
- **Camera calibration**: Independent view directions for 3D reconstruction
- **Feature matching**: Independent descriptors for robust matching
- **Basis functions**: Independent components for image representation

**5. Natural Language Processing:**
- **Word embeddings**: Independent semantic dimensions
- **Topic modeling**: Independent topic directions
- **Vocabulary pruning**: Remove linearly dependent word vectors

**Implementation Examples:**

```python
import numpy as np
from scipy.linalg import null_space

def check_linear_independence(vectors, tolerance=1e-10):
    """
    Check if vectors are linearly independent using multiple methods
    
    Args:
        vectors: List of vectors or matrix with vectors as columns
        tolerance: Numerical tolerance for zero checking
    
    Returns:
        Dictionary with results from different methods
    """
    # Convert to matrix format (vectors as columns)
    if isinstance(vectors, list):
        A = np.column_stack(vectors)
    else:
        A = np.array(vectors)
    
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    
    results = {}
    
    # Method 1: Rank-based check
    rank = np.linalg.matrix_rank(A, tol=tolerance)
    num_vectors = A.shape[1]
    results['rank_method'] = {
        'rank': rank,
        'num_vectors': num_vectors,
        'independent': rank == num_vectors
    }
    
    # Method 2: Determinant (for square matrices)
    if A.shape[0] == A.shape[1]:
        det = np.linalg.det(A)
        results['determinant_method'] = {
            'determinant': det,
            'independent': abs(det) > tolerance
        }
    
    # Method 3: Null space
    null_space_vectors = null_space(A)
    null_space_dim = null_space_vectors.shape[1]
    results['null_space_method'] = {
        'null_space_dimension': null_space_dim,
        'independent': null_space_dim == 0
    }
    
    # Method 4: Solve homogeneous system
    try:
        # For overdetermined systems, use least squares
        if A.shape[0] >= A.shape[1]:
            solution = np.linalg.lstsq(A, np.zeros(A.shape[0]), rcond=None)[0]
        else:
            # For underdetermined systems, there are always non-trivial solutions
            solution = np.ones(A.shape[1])  # Dummy solution
        
        is_trivial = np.allclose(solution, 0, atol=tolerance)
        results['homogeneous_system'] = {
            'solution': solution,
            'trivial_only': is_trivial,
            'independent': is_trivial or A.shape[0] < A.shape[1]
        }
    except np.linalg.LinAlgError:
        results['homogeneous_system'] = {'error': 'Could not solve system'}
    
    return results

def find_linear_combination(vectors, target_vector, tolerance=1e-10):
    """
    Find linear combination coefficients if target vector is dependent on given vectors
    
    Args:
        vectors: List of basis vectors
        target_vector: Vector to express as linear combination
        tolerance: Numerical tolerance
    
    Returns:
        Coefficients if combination exists, None otherwise
    """
    A = np.column_stack(vectors)
    
    try:
        # Solve A @ coefficients = target_vector
        coefficients = np.linalg.lstsq(A, target_vector, rcond=None)[0]
        
        # Verify the solution
        reconstructed = A @ coefficients
        error = np.linalg.norm(reconstructed - target_vector)
        
        if error < tolerance:
            return coefficients, error
        else:
            return None, error
    
    except np.linalg.LinAlgError:
        return None, float('inf')

def gram_schmidt_independence_check(vectors):
    """
    Check independence using Gram-Schmidt process
    
    Args:
        vectors: List of vectors to check
    
    Returns:
        Orthogonalized vectors and independence status
    """
    vectors = [np.array(v, dtype=float) for v in vectors]
    orthogonal_vectors = []
    dependent_indices = []
    
    for i, v in enumerate(vectors):
        # Start with current vector
        u = v.copy()
        
        # Subtract projections onto previous orthogonal vectors
        for orth_v in orthogonal_vectors:
            projection = np.dot(u, orth_v) / np.dot(orth_v, orth_v) * orth_v
            u = u - projection
        
        # Check if resulting vector is (nearly) zero
        if np.linalg.norm(u) < 1e-10:
            dependent_indices.append(i)
        else:
            orthogonal_vectors.append(u)
    
    is_independent = len(dependent_indices) == 0
    
    return orthogonal_vectors, is_independent, dependent_indices

def demonstrate_linear_independence():
    """Demonstrate linear independence concepts with examples"""
    
    print("="*60)
    print("LINEAR INDEPENDENCE DEMONSTRATIONS")
    print("="*60)
    
    # Example 1: Independent vectors in 2D
    print("\n1. Independent Vectors in 2D:")
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    vectors_2d_indep = [v1, v2]
    results = check_linear_independence(vectors_2d_indep)
    
    print(f"Vectors: {[v.tolist() for v in vectors_2d_indep]}")
    print(f"Rank: {results['rank_method']['rank']}")
    print(f"Determinant: {results['determinant_method']['determinant']:.6f}")
    print(f"Independent: {results['rank_method']['independent']}")
    
    # Example 2: Dependent vectors in 2D
    print("\n2. Dependent Vectors in 2D:")
    v1 = np.array([1, 2])
    v2 = np.array([2, 4])  # v2 = 2*v1
    
    vectors_2d_dep = [v1, v2]
    results = check_linear_independence(vectors_2d_dep)
    
    print(f"Vectors: {[v.tolist() for v in vectors_2d_dep]}")
    print(f"Rank: {results['rank_method']['rank']}")
    print(f"Determinant: {results['determinant_method']['determinant']:.6f}")
    print(f"Independent: {results['rank_method']['independent']}")
    
    # Find linear combination
    coeffs, error = find_linear_combination([v1], v2)
    if coeffs is not None:
        print(f"Linear combination: v2 = {coeffs[0]:.3f} × v1")
    
    # Example 3: Three vectors in 3D (independent)
    print("\n3. Independent Vectors in 3D:")
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 1, 0])
    v3 = np.array([1, 1, 1])
    
    vectors_3d_indep = [v1, v2, v3]
    results = check_linear_independence(vectors_3d_indep)
    
    print(f"Vectors: {[v.tolist() for v in vectors_3d_indep]}")
    print(f"Rank: {results['rank_method']['rank']}")
    print(f"Determinant: {results['determinant_method']['determinant']:.6f}")
    print(f"Independent: {results['rank_method']['independent']}")
    
    # Example 4: Four vectors in 3D (must be dependent)
    print("\n4. Four Vectors in 3D (Must be Dependent):")
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    v4 = np.array([1, 1, 1])
    
    vectors_4_in_3d = [v1, v2, v3, v4]
    results = check_linear_independence(vectors_4_in_3d)
    
    print(f"Vectors: {[v.tolist() for v in vectors_4_in_3d]}")
    print(f"Rank: {results['rank_method']['rank']}")
    print(f"Independent: {results['rank_method']['independent']}")
    
    # Find how v4 depends on v1, v2, v3
    coeffs, error = find_linear_combination([v1, v2, v3], v4)
    if coeffs is not None:
        print(f"Linear combination: v4 = {coeffs[0]:.3f}×v1 + {coeffs[1]:.3f}×v2 + {coeffs[2]:.3f}×v3")
    
    # Example 5: Gram-Schmidt process
    print("\n5. Gram-Schmidt Independence Check:")
    original_vectors = [
        np.array([1, 1, 0]),
        np.array([1, 0, 1]),
        np.array([2, 1, 1])  # This should be dependent on first two
    ]
    
    orthogonal, independent, dep_indices = gram_schmidt_independence_check(original_vectors)
    
    print(f"Original vectors: {[v.tolist() for v in original_vectors]}")
    print(f"Independent: {independent}")
    print(f"Dependent indices: {dep_indices}")
    print(f"Number of orthogonal vectors: {len(orthogonal)}")

def feature_independence_analysis(data_matrix):
    """Analyze feature independence in data matrix"""
    
    print(f"\nFeature Independence Analysis:")
    print(f"Data shape: {data_matrix.shape}")
    
    # Check column independence (features)
    column_results = check_linear_independence(data_matrix)
    print(f"Feature rank: {column_results['rank_method']['rank']} out of {data_matrix.shape[1]}")
    print(f"Features independent: {column_results['rank_method']['independent']}")
    
    if not column_results['rank_method']['independent']:
        redundant_features = data_matrix.shape[1] - column_results['rank_method']['rank']
        print(f"Redundant features: {redundant_features}")
    
    # Compute correlation matrix to identify dependencies
    correlation_matrix = np.corrcoef(data_matrix.T)
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    n_features = data_matrix.shape[1]
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(correlation_matrix[i, j]) > 0.95:  # High correlation threshold
                high_corr_pairs.append((i, j, correlation_matrix[i, j]))
    
    if high_corr_pairs:
        print(f"Highly correlated feature pairs (>0.95):")
        for i, j, corr in high_corr_pairs:
            print(f"  Features {i} and {j}: correlation = {corr:.3f}")
    
    return column_results, correlation_matrix

def basis_construction_example():
    """Demonstrate constructing basis from linearly independent vectors"""
    
    print(f"\n{'='*60}")
    print("BASIS CONSTRUCTION EXAMPLE")
    print(f"{'='*60}")
    
    # Start with some vectors in 3D
    vectors = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9]),
        np.array([1, 0, 1])
    ]
    
    print(f"Starting vectors: {[v.tolist() for v in vectors]}")
    
    # Use Gram-Schmidt to find orthogonal basis
    orthogonal, independent, dep_indices = gram_schmidt_independence_check(vectors)
    
    print(f"Linear independence: {independent}")
    print(f"Dependent vector indices: {dep_indices}")
    print(f"Orthogonal basis vectors:")
    
    for i, v in enumerate(orthogonal):
        normalized = v / np.linalg.norm(v)
        print(f"  e{i+1} = {v} (normalized: {normalized})")
    
    # Verify orthogonality
    if len(orthogonal) > 1:
        print(f"\nOrthogonality verification:")
        for i in range(len(orthogonal)):
            for j in range(i + 1, len(orthogonal)):
                dot_product = np.dot(orthogonal[i], orthogonal[j])
                print(f"  e{i+1} · e{j+1} = {dot_product:.6f}")

# Real-world application examples
def ml_independence_examples():
    """Machine learning examples of linear independence"""
    
    print(f"\n{'='*60}")
    print("MACHINE LEARNING APPLICATIONS")
    print(f"{'='*60}")
    
    # Example 1: Feature matrix with redundant features
    print("\n1. Feature Matrix Analysis:")
    np.random.seed(42)
    
    # Create data with some dependent features
    n_samples = 100
    independent_features = np.random.randn(n_samples, 3)
    
    # Add dependent features
    feature_matrix = np.column_stack([
        independent_features[:, 0],                    # Original feature 1
        independent_features[:, 1],                    # Original feature 2
        independent_features[:, 2],                    # Original feature 3
        2 * independent_features[:, 0] + independent_features[:, 1],  # Linear combination
        independent_features[:, 0] + 0.1 * np.random.randn(n_samples),  # Nearly dependent
        np.random.randn(n_samples)                     # Independent feature
    ])
    
    analysis_results, corr_matrix = feature_independence_analysis(feature_matrix)
    
    # Example 2: Principal Component Analysis
    print(f"\n2. PCA and Independence:")
    # PCA finds orthogonal (independent) directions
    centered_data = feature_matrix - np.mean(feature_matrix, axis=0)
    cov_matrix = np.cov(centered_data.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Check if eigenvectors are orthogonal (independent)
    eigenvector_independence = check_linear_independence(eigenvectors)
    
    print(f"Eigenvector matrix rank: {eigenvector_independence['rank_method']['rank']}")
    print(f"Eigenvectors independent: {eigenvector_independence['rank_method']['independent']}")
    print(f"Eigenvalues: {eigenvalues[:3]}")  # Show first 3

if __name__ == "__main__":
    demonstrate_linear_independence()
    basis_construction_example()
    ml_independence_examples()
    
    # Additional educational example
    print(f"\n{'='*60}")
    print("EDUCATIONAL EXAMPLE: STEP-BY-STEP ANALYSIS")
    print(f"{'='*60}")
    
    # Simple 2D example for detailed analysis
    v1 = np.array([3, 2])
    v2 = np.array([6, 4])  # v2 = 2*v1 (dependent)
    v3 = np.array([1, 3])  # Independent of v1
    
    print(f"\nAnalyzing vectors:")
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v3 = {v3}")
    
    # Check each pair
    pairs = [([v1, v2], "v1, v2"), ([v1, v3], "v1, v3"), ([v2, v3], "v2, v3")]
    
    for vectors, label in pairs:
        results = check_linear_independence(vectors)
        print(f"\n{label}:")
        print(f"  Independent: {results['rank_method']['independent']}")
        print(f"  Rank: {results['rank_method']['rank']}")
        print(f"  Determinant: {results['determinant_method']['determinant']:.3f}")
```

**Important Theorems:**

**1. Fundamental Theorem of Linear Algebra:**
- For m×n matrix A: rank(A) + dim(null(A)) = n
- Number of independent columns + dimension of null space = total columns

**2. Basis Theorem:**
- Any linearly independent set in n-dimensional space has at most n vectors
- Any set of n linearly independent vectors in n-dimensional space forms a basis

**3. Dimension Theorem:**
- All bases of a vector space have the same number of elements
- This number is the dimension of the space

**Practical Implications:**

**1. Data Analysis:**
- **Redundancy detection**: Identify and remove correlated features
- **Dimension estimation**: Determine intrinsic dimensionality of data
- **Quality control**: Ensure data collection provides independent information

**2. Model Design:**
- **Feature selection**: Choose independent features for better model performance
- **Regularization**: Penalties to encourage weight independence
- **Architecture design**: Ensure neural network layers learn independent representations

**3. Optimization:**
- **Constraint satisfaction**: Independent constraints define feasible region
- **Convergence analysis**: Independent directions for optimization algorithms
- **Conditioning**: Independent variables lead to better-conditioned systems

**Common Misconceptions:**
- **Orthogonal ⟹ Independent**: True, but independence doesn't require orthogonality
- **Zero correlation ⟹ Independent**: Only true for linear relationships
- **More vectors ⟹ Better**: Dependent vectors add noise, not information

**Best Practices:**
- Always check for linear dependencies in data preprocessing
- Use numerical tolerance when testing independence with floating-point arithmetic
- Consider both exact and approximate independence in practical applications
- Combine multiple methods for robust independence testing

---

## Question 18

**What is the meaning of the solution space of a system of linear equations?**

**Answer:** The solution space (also called solution set) of a system of linear equations is the set of all vectors that satisfy every equation in the system simultaneously. It represents the geometric object formed by the intersection of hyperplanes defined by each equation and provides complete information about the system's solvability and structure.

**Mathematical Definition:**
For the system Ax = b, the solution space is:
```
S = {x ∈ ℝⁿ : Ax = b}
```

**Types of Solution Spaces:**

**1. Empty Set (No Solution):**
- **Condition**: rank(A) < rank([A|b])
- **Geometry**: Hyperplanes don't intersect
- **Example**: Parallel planes in 3D

**2. Single Point (Unique Solution):**
- **Condition**: rank(A) = rank([A|b]) = n (number of variables)
- **Geometry**: Hyperplanes intersect at exactly one point
- **Structure**: S = {x₀} where x₀ is the unique solution

**3. Infinite Solutions (Solution Subspace):**
- **Condition**: rank(A) = rank([A|b]) < n
- **Geometry**: Hyperplanes intersect in higher-dimensional object
- **Structure**: S = x₀ + null(A) (affine subspace)

**Structure of Solution Space:**

**General Form for Consistent Systems:**
```
S = x_particular + span{v₁, v₂, ..., vₖ}
```
where:
- **x_particular**: Any particular solution to Ax = b
- **{v₁, v₂, ..., vₖ}**: Basis for null space of A
- **k = n - rank(A)**: Number of free variables

**Homogeneous vs Non-Homogeneous Systems:**

**Homogeneous System (Ax = 0):**
- **Always consistent** (x = 0 is always a solution)
- **Solution space is a subspace** (vector space structure)
- **Dimension = n - rank(A)**

**Non-Homogeneous System (Ax = b, b ≠ 0):**
- **May or may not be consistent**
- **Solution space is affine subspace** (translated vector subspace)
- **If consistent: dimension = n - rank(A)**

**Geometric Interpretation:**

**2D Examples:**
```
x + y = 1    (line)
2x + y = 3   (another line)
```
- **Intersection**: Single point (unique solution)
- **Parallel lines**: No solution
- **Same line**: Infinite solutions (1D line)

**3D Examples:**
```
x + y + z = 1    (plane)
2x + y - z = 0   (plane)
x - y + 2z = 3   (plane)
```
- **Three planes meeting at point**: Unique solution
- **Parallel planes**: No solution
- **Planes intersecting in line**: Infinite solutions (1D line)
- **All same plane**: Infinite solutions (2D plane)

**Parametric Representation:**

**Free Variables Approach:**
1. **Identify pivot and free variables** from row echelon form
2. **Express pivot variables** in terms of free variables
3. **Set free variables as parameters**: t₁, t₂, ..., tₖ
4. **Write solution as vector equation**

**Example:**
```
System: x + 2y - z = 1
        2x + 4y - 2z = 2

Row reduction gives: x + 2y - z = 1
                    0 = 0

Solution: x = 1 - 2y + z
         y = t₁ (free)
         z = t₂ (free)

Parametric form: [x]   [1]      [-2]     [1]
                [y] = [0] + t₁ [ 1] + t₂[0]
                [z]   [0]      [ 0]     [1]
```

**Applications in Machine Learning:**

**1. Linear Regression:**
- **Normal equations**: (XᵀX)β = Xᵀy
- **Solution space**: Depends on rank of XᵀX
- **Regularization**: Ensures unique solution when system is underdetermined

**2. Neural Network Training:**
- **Weight optimization**: Find weights that minimize loss
- **Solution manifold**: Set of optimal weight configurations
- **Local minima**: Multiple solutions with same loss value

**3. Constraint Optimization:**
- **Feasible region**: Solution space of constraint equations
- **Equality constraints**: Intersection of hyperplanes
- **Lagrange multipliers**: Find optimal points in solution space

**4. Computer Vision:**
- **Camera calibration**: Solution space of projection equations
- **3D reconstruction**: Intersection of epipolar constraints
- **Bundle adjustment**: Large system with geometric constraints

**5. Collaborative Filtering:**
- **Matrix completion**: Find low-rank matrices satisfying observed entries
- **Solution space**: Set of all completions with given rank constraint
- **Regularization**: Prefer solutions with desired properties

**Implementation Examples:**

```python
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt

def analyze_solution_space(A, b, tolerance=1e-10):
    """
    Analyze the solution space of linear system Ax = b
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        tolerance: Numerical tolerance
    
    Returns:
        Dictionary with solution space analysis
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Compute ranks
    rank_A = np.linalg.matrix_rank(A, tol=tolerance)
    augmented = np.column_stack([A, b])
    rank_augmented = np.linalg.matrix_rank(augmented, tol=tolerance)
    
    m, n = A.shape  # m equations, n variables
    
    analysis = {
        'coefficient_matrix_rank': rank_A,
        'augmented_matrix_rank': rank_augmented,
        'num_equations': m,
        'num_variables': n,
        'num_free_variables': n - rank_A
    }
    
    # Determine solution type
    if rank_A < rank_augmented:
        analysis['solution_type'] = 'no_solution'
        analysis['consistent'] = False
        analysis['description'] = 'Inconsistent system - no solution exists'
        
    elif rank_A == rank_augmented == n:
        analysis['solution_type'] = 'unique_solution'
        analysis['consistent'] = True
        analysis['description'] = 'Unique solution exists'
        
        # Find the unique solution
        try:
            solution = np.linalg.solve(A, b)
            analysis['particular_solution'] = solution
            analysis['null_space_dimension'] = 0
        except np.linalg.LinAlgError:
            # Use least squares if system is overdetermined
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            analysis['particular_solution'] = solution
            
    else:  # rank_A == rank_augmented < n
        analysis['solution_type'] = 'infinite_solutions'
        analysis['consistent'] = True
        analysis['description'] = f'Infinite solutions with {n - rank_A} free variables'
        
        # Find particular solution
        try:
            particular_solution = np.linalg.lstsq(A, b, rcond=None)[0]
            analysis['particular_solution'] = particular_solution
        except np.linalg.LinAlgError:
            analysis['particular_solution'] = None
        
        # Find null space basis
        null_basis = null_space(A)
        analysis['null_space_basis'] = null_basis
        analysis['null_space_dimension'] = null_basis.shape[1]
        
        # Generate parametric solution
        if analysis['particular_solution'] is not None and null_basis.shape[1] > 0:
            analysis['parametric_form'] = {
                'particular_solution': particular_solution,
                'null_space_basis': null_basis
            }
    
    return analysis

def parametric_solution_generator(analysis, parameter_values=None):
    """
    Generate solutions using parametric representation
    
    Args:
        analysis: Result from analyze_solution_space
        parameter_values: Values for free variables
    
    Returns:
        Generated solution vector
    """
    if analysis['solution_type'] != 'infinite_solutions':
        if analysis['solution_type'] == 'unique_solution':
            return analysis['particular_solution']
        else:
            return None
    
    particular = analysis['particular_solution']
    null_basis = analysis['null_space_basis']
    
    if parameter_values is None:
        # Generate random parameters
        num_params = null_basis.shape[1]
        parameter_values = np.random.randn(num_params)
    
    # Construct solution: x = x_p + c₁v₁ + c₂v₂ + ... + cₖvₖ
    solution = particular.copy()
    for i, param in enumerate(parameter_values):
        solution += param * null_basis[:, i]
    
    return solution

def visualize_solution_space_2d(A, b):
    """Visualize 2D solution space (lines and their intersections)"""
    
    if A.shape[1] != 2:
        print("Visualization only available for 2D systems")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    x_range = np.linspace(-5, 5, 100)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot each equation as a line
    for i in range(A.shape[0]):
        a, b_coef = A[i, 0], A[i, 1]
        c = b[i]
        
        if abs(b_coef) > 1e-10:  # Not vertical line
            y_values = (c - a * x_range) / b_coef
            ax.plot(x_range, y_values, color=colors[i % len(colors)], 
                   label=f'Equation {i+1}: {a:.2f}x + {b_coef:.2f}y = {c:.2f}')
        else:  # Vertical line
            if abs(a) > 1e-10:
                x_val = c / a
                ax.axvline(x=x_val, color=colors[i % len(colors)], 
                          label=f'Equation {i+1}: x = {x_val:.2f}')
    
    # Analyze and mark solution
    analysis = analyze_solution_space(A, b)
    
    if analysis['solution_type'] == 'unique_solution':
        sol = analysis['particular_solution']
        ax.plot(sol[0], sol[1], 'ko', markersize=10, label='Unique Solution')
        ax.annotate(f'({sol[0]:.2f}, {sol[1]:.2f})', 
                   (sol[0], sol[1]), xytext=(10, 10), 
                   textcoords='offset points')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Solution Space Visualization - {analysis["description"]}')
    
    plt.tight_layout()
    plt.show()
    
    return analysis

def demonstrate_solution_spaces():
    """Demonstrate different types of solution spaces"""
    
    print("="*60)
    print("SOLUTION SPACE DEMONSTRATIONS")
    print("="*60)
    
    # Example 1: Unique solution
    print("\n1. UNIQUE SOLUTION:")
    A1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([5, 6])
    
    analysis1 = analyze_solution_space(A1, b1)
    print(f"System: Ax = b where")
    print(f"A = \n{A1}")
    print(f"b = {b1}")
    print(f"Analysis: {analysis1['description']}")
    print(f"Solution: {analysis1['particular_solution']}")
    
    # Example 2: No solution
    print("\n2. NO SOLUTION:")
    A2 = np.array([[1, 2], [2, 4]])
    b2 = np.array([1, 3])  # Inconsistent
    
    analysis2 = analyze_solution_space(A2, b2)
    print(f"System: Ax = b where")
    print(f"A = \n{A2}")
    print(f"b = {b2}")
    print(f"Analysis: {analysis2['description']}")
    print(f"rank(A) = {analysis2['coefficient_matrix_rank']}")
    print(f"rank([A|b]) = {analysis2['augmented_matrix_rank']}")
    
    # Example 3: Infinite solutions
    print("\n3. INFINITE SOLUTIONS:")
    A3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    b3 = np.array([3, 6, 3])
    
    analysis3 = analyze_solution_space(A3, b3)
    print(f"System: Ax = b where")
    print(f"A = \n{A3}")
    print(f"b = {b3}")
    print(f"Analysis: {analysis3['description']}")
    print(f"Free variables: {analysis3['num_free_variables']}")
    
    if 'null_space_basis' in analysis3:
        print(f"Null space basis:")
        print(f"{analysis3['null_space_basis']}")
    
    # Generate some solutions
    print(f"\nSample solutions:")
    for i in range(3):
        params = np.random.randn(analysis3['num_free_variables'])
        solution = parametric_solution_generator(analysis3, params)
        if solution is not None:
            verification = A3 @ solution
            print(f"  Solution {i+1}: {solution}")
            print(f"    Verification Ax = {verification} (should equal {b3})")

def homogeneous_vs_nonhomogeneous():
    """Compare homogeneous and non-homogeneous systems"""
    
    print(f"\n{'='*60}")
    print("HOMOGENEOUS VS NON-HOMOGENEOUS SYSTEMS")
    print(f"{'='*60}")
    
    # Define matrix
    A = np.array([[1, 2, -1], 
                  [2, 1, 1], 
                  [1, -1, 2]], dtype=float)
    
    # Homogeneous system: Ax = 0
    print("\n1. HOMOGENEOUS SYSTEM (Ax = 0):")
    b_homogeneous = np.zeros(3)
    
    analysis_homo = analyze_solution_space(A, b_homogeneous)
    print(f"Analysis: {analysis_homo['description']}")
    print(f"Null space dimension: {analysis_homo.get('null_space_dimension', 0)}")
    
    if 'null_space_basis' in analysis_homo:
        print(f"Null space basis:")
        for i, vec in enumerate(analysis_homo['null_space_basis'].T):
            print(f"  v{i+1} = {vec}")
    
    # Non-homogeneous system: Ax = b
    print("\n2. NON-HOMOGENEOUS SYSTEM (Ax = b):")
    b_nonhomogeneous = np.array([1, 2, 3])
    
    analysis_nonhomo = analyze_solution_space(A, b_nonhomogeneous)
    print(f"Analysis: {analysis_nonhomo['description']}")
    
    if analysis_nonhomo['consistent']:
        print(f"Particular solution: {analysis_nonhomo['particular_solution']}")
        
        if 'null_space_basis' in analysis_nonhomo:
            print(f"General solution structure:")
            print(f"  x = x_particular + linear_combination(null_space_basis)")
            print(f"  x = {analysis_nonhomo['particular_solution']} + c₁v₁ + c₂v₂ + ...")

def machine_learning_applications():
    """Show ML applications of solution space concepts"""
    
    print(f"\n{'='*60}")
    print("MACHINE LEARNING APPLICATIONS")
    print(f"{'='*60}")
    
    # Example 1: Linear regression with more features than samples
    print("\n1. UNDERDETERMINED LINEAR REGRESSION:")
    np.random.seed(42)
    
    n_samples = 5
    n_features = 8
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Normal equations: X^T X β = X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    
    analysis = analyze_solution_space(XtX, Xty)
    print(f"Design matrix shape: {X.shape}")
    print(f"Analysis: {analysis['description']}")
    print(f"Free variables (degrees of freedom): {analysis['num_free_variables']}")
    
    # This is why regularization is needed!
    print(f"→ This is why regularization (Ridge, Lasso) is needed for underdetermined systems!")
    
    # Example 2: Matrix completion
    print(f"\n2. MATRIX COMPLETION EXAMPLE:")
    # Simplified matrix completion: find entries that satisfy observed constraints
    
    # Create partial observation constraints
    M_shape = (3, 4)
    observed_entries = [(0, 0, 1), (0, 2, 3), (1, 1, 2), (2, 3, 4)]  # (row, col, value)
    
    # Set up linear system for matrix entries
    n_entries = M_shape[0] * M_shape[1]
    n_constraints = len(observed_entries)
    
    A_constraints = np.zeros((n_constraints, n_entries))
    b_constraints = np.zeros(n_constraints)
    
    for i, (row, col, value) in enumerate(observed_entries):
        entry_index = row * M_shape[1] + col
        A_constraints[i, entry_index] = 1
        b_constraints[i] = value
    
    constraint_analysis = analyze_solution_space(A_constraints, b_constraints)
    print(f"Matrix shape: {M_shape}")
    print(f"Observed entries: {len(observed_entries)}")
    print(f"Total entries: {n_entries}")
    print(f"Analysis: {constraint_analysis['description']}")
    print(f"Free variables: {constraint_analysis['num_free_variables']}")
    print(f"→ Need additional constraints (like low-rank) for unique solution!")

if __name__ == "__main__":
    demonstrate_solution_spaces()
    homogeneous_vs_nonhomogeneous()
    machine_learning_applications()
    
    print(f"\n{'='*60}")
    print("PARAMETRIC SOLUTION EXAMPLE")
    print(f"{'='*60}")
    
    # Detailed parametric example
    A_param = np.array([[1, 2, -1, 3],
                       [2, 4, 1, 1],
                       [3, 6, 0, 4]], dtype=float)
    b_param = np.array([1, 3, 4], dtype=float)
    
    analysis_param = analyze_solution_space(A_param, b_param)
    print(f"System with {A_param.shape[1]} variables and {A_param.shape[0]} equations:")
    print(f"Analysis: {analysis_param['description']}")
    
    if analysis_param['solution_type'] == 'infinite_solutions':
        print(f"\nParametric solution form:")
        print(f"x = x_particular + t₁v₁ + t₂v₂ + ... + tₖvₖ")
        print(f"where:")
        print(f"  x_particular = {analysis_param['particular_solution']}")
        print(f"  Null space basis vectors:")
        
        null_basis = analysis_param['null_space_basis']
        for i in range(null_basis.shape[1]):
            print(f"    v{i+1} = {null_basis[:, i]}")
        
        print(f"\nGenerate specific solutions:")
        for i in range(3):
            params = [1, 0] if i == 0 else [0, 1] if i == 1 else [2, -1]
            if len(params) <= analysis_param['num_free_variables']:
                # Pad with zeros if needed
                while len(params) < analysis_param['num_free_variables']:
                    params.append(0)
                
                solution = parametric_solution_generator(analysis_param, params[:analysis_param['num_free_variables']])
                verification = A_param @ solution
                print(f"  Parameters {params[:analysis_param['num_free_variables']]}: x = {solution}")
                print(f"    Check: Ax = {verification} ≈ {b_param}")
```

**Key Properties of Solution Spaces:**

**1. Linearity of Solutions:**
- If x₁ and x₂ are solutions to Ax = b, then x₁ - x₂ is in null(A)
- Solution space has structure: particular solution + null space

**2. Dimension Formula:**
- For consistent system: dim(solution space) = n - rank(A)
- This equals the number of free variables

**3. Uniqueness Conditions:**
- Unique solution ⟺ rank(A) = n
- Infinite solutions ⟺ rank(A) < n (when consistent)

**4. Geometric Structure:**
- **Point**: Unique solution
- **Line**: 1 free variable
- **Plane**: 2 free variables
- **Hyperplane**: n-1 free variables

**Practical Considerations:**

**1. Numerical Stability:**
- Use appropriate tolerance for rank determination
- Consider condition number for near-singular systems
- Regularization for ill-posed problems

**2. Computational Methods:**
- **Direct methods**: Gaussian elimination, LU decomposition
- **Iterative methods**: For large sparse systems
- **Least squares**: For overdetermined systems

**3. Applications:**
- **Optimization**: Solution space defines feasible region
- **Machine learning**: Parameter space exploration
- **Computer graphics**: Intersection computations
- **Physics**: Constraint satisfaction in mechanical systems

**Best Practices:**
- Always check consistency before seeking solutions
- Use parametric form for infinite solution sets
- Consider regularization for underdetermined systems
- Visualize low-dimensional cases for intuition
- Apply domain knowledge to choose appropriate solutions from infinite sets

---

## Question 19

**Describe the conditions for consistency in linear equations.**

**Answer:** Consistency in linear equations refers to whether a system of linear equations has at least one solution. Understanding consistency conditions is crucial for determining solvability, designing algorithms, and ensuring well-posed problems in machine learning and engineering applications.

**Mathematical Definition:**
A system Ax = b is **consistent** (or compatible) if there exists at least one vector x that satisfies all equations simultaneously. Otherwise, the system is **inconsistent**.

**Fundamental Consistency Theorem:**
The system Ax = b is consistent if and only if:
```
rank(A) = rank([A|b])
```
where [A|b] is the augmented matrix formed by appending b as the last column of A.

**Detailed Consistency Conditions:**

**1. Consistent with Unique Solution:**
- **Condition**: rank(A) = rank([A|b]) = n (number of variables)
- **Interpretation**: Exactly enough independent equations to determine all variables
- **Geometry**: Hyperplanes intersect at exactly one point

**2. Consistent with Infinite Solutions:**
- **Condition**: rank(A) = rank([A|b]) < n
- **Interpretation**: Fewer independent equations than variables
- **Geometry**: Hyperplanes intersect in higher-dimensional subspace
- **Free variables**: n - rank(A)

**3. Inconsistent (No Solution):**
- **Condition**: rank(A) < rank([A|b])
- **Interpretation**: Augmenting with b adds new information (contradiction)
- **Geometry**: Hyperplanes don't intersect

**Geometric Interpretation:**

**2D Examples:**
```
Consistent (unique):     x + y = 1, x - y = 1     (intersecting lines)
Consistent (infinite):   x + y = 1, 2x + 2y = 2   (same line)
Inconsistent:           x + y = 1, x + y = 2     (parallel lines)
```

**3D Examples:**
```
Three planes can:
- Meet at a point (unique solution)
- Meet along a line (infinite solutions)
- Have common intersection region (infinite solutions)
- Have no common intersection (inconsistent)
```

**Row Echelon Form Analysis:**

**After Gaussian elimination, inconsistency appears as:**
```
[1  2  3 | 4 ]
[0  1  2 | 1 ]
[0  0  0 | 5 ]  ← This row indicates inconsistency
```
The last row represents 0 = 5, which is impossible.

**Consistency Testing Methods:**

**1. Rank Comparison Method:**
```python
def is_consistent(A, b):
    rank_A = np.linalg.matrix_rank(A)
    augmented = np.column_stack([A, b])
    rank_augmented = np.linalg.matrix_rank(augmented)
    return rank_A == rank_augmented
```

**2. Gaussian Elimination Method:**
- Reduce [A|b] to row echelon form
- Check for rows of form [0 0 ... 0 | c] where c ≠ 0

**3. Least Squares Residual:**
- Compute minimum residual ||Ax - b||₂
- System consistent ⟺ minimum residual = 0

**Special Cases:**

**1. Homogeneous Systems (Ax = 0):**
- **Always consistent** (x = 0 is always a solution)
- Question is whether non-trivial solutions exist
- Non-trivial solutions ⟺ rank(A) < n

**2. Square Systems (m = n):**
- **Consistent with unique solution ⟺ det(A) ≠ 0**
- **Inconsistent or infinite solutions ⟺ det(A) = 0**

**3. Overdetermined Systems (m > n):**
- More equations than unknowns
- Usually inconsistent unless b is in column space of A
- Often solved using least squares approximation

**4. Underdetermined Systems (m < n):**
- Fewer equations than unknowns
- If consistent, infinite solutions exist
- Always has infinite solutions if rank(A) = m

**Applications in Machine Learning:**

**1. Linear Regression:**
- **Normal equations**: X^T X β = X^T y
- **Consistency**: Always consistent for least squares
- **Overdetermined**: More data points than features (typical case)

**2. Neural Network Training:**
- **Gradient equations**: ∇L = 0 for optimal weights
- **Consistency**: Determines existence of global minima
- **Regularization**: Ensures consistency by adding constraints

**3. Computer Vision:**
- **Camera calibration**: Projection equations must be consistent
- **3D reconstruction**: Multiple view constraints
- **Bundle adjustment**: Large consistent system of geometric constraints

**4. Optimization with Constraints:**
- **Equality constraints**: Must form consistent system
- **Feasible region**: Set of points satisfying all constraints
- **KKT conditions**: Consistency determines optimality

**5. Collaborative Filtering:**
- **Matrix completion**: Observed ratings provide consistency constraints
- **Low-rank constraints**: Additional structure for consistency
- **Regularization**: Ensures well-posed problem

**Implementation Examples:**

```python
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

def comprehensive_consistency_analysis(A, b, tolerance=1e-10):
    """
    Comprehensive analysis of system consistency with multiple methods
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        tolerance: Numerical tolerance for rank computation
    
    Returns:
        Dictionary with consistency analysis
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    analysis = {}
    
    # Basic information
    m, n = A.shape
    analysis['num_equations'] = m
    analysis['num_variables'] = n
    analysis['system_type'] = 'square' if m == n else 'overdetermined' if m > n else 'underdetermined'
    
    # Method 1: Rank comparison
    rank_A = np.linalg.matrix_rank(A, tol=tolerance)
    augmented = np.column_stack([A, b])
    rank_augmented = np.linalg.matrix_rank(augmented, tol=tolerance)
    
    analysis['rank_A'] = rank_A
    analysis['rank_augmented'] = rank_augmented
    analysis['consistent_rank_test'] = rank_A == rank_augmented
    
    # Method 2: Gaussian elimination approach
    try:
        # Try to solve using least squares (works for any system)
        solution, residuals, rank_lstsq, singular_values = lstsq(A, b)
        
        analysis['lstsq_solution'] = solution
        analysis['lstsq_rank'] = rank_lstsq
        analysis['singular_values'] = singular_values
        
        # Check residual for consistency
        if len(residuals) > 0:
            residual_norm = np.sqrt(residuals[0]) if residuals[0] > 0 else 0
        else:
            residual_norm = np.linalg.norm(A @ solution - b)
        
        analysis['residual_norm'] = residual_norm
        analysis['consistent_residual_test'] = residual_norm < tolerance
        
    except np.linalg.LinAlgError as e:
        analysis['lstsq_error'] = str(e)
    
    # Method 3: Row reduction analysis
    try:
        augmented_copy = augmented.copy()
        analysis['row_reduction'] = analyze_row_reduction_consistency(augmented_copy, tolerance)
    except Exception as e:
        analysis['row_reduction_error'] = str(e)
    
    # Overall consistency determination
    consistency_methods = [
        analysis.get('consistent_rank_test', False),
        analysis.get('consistent_residual_test', False)
    ]
    
    analysis['overall_consistent'] = any(consistency_methods)
    
    # Solution classification
    if analysis['overall_consistent']:
        if rank_A == n:
            analysis['solution_type'] = 'unique'
            analysis['num_free_variables'] = 0
        else:
            analysis['solution_type'] = 'infinite'
            analysis['num_free_variables'] = n - rank_A
    else:
        analysis['solution_type'] = 'none'
        analysis['num_free_variables'] = None
    
    return analysis

def analyze_row_reduction_consistency(augmented_matrix, tolerance=1e-10):
    """
    Analyze consistency using row reduction
    
    Args:
        augmented_matrix: [A|b] matrix
        tolerance: Tolerance for zero detection
    
    Returns:
        Row reduction analysis results
    """
    rows, cols = augmented_matrix.shape
    
    # Perform Gaussian elimination
    for i in range(min(rows, cols - 1)):  # -1 because last column is b
        # Find pivot
        pivot_row = i + np.argmax(np.abs(augmented_matrix[i:rows, i]))
        
        # Swap rows if needed
        if pivot_row != i:
            augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]
        
        # Skip if pivot is effectively zero
        if abs(augmented_matrix[i, i]) < tolerance:
            continue
        
        # Eliminate below pivot
        for j in range(i + 1, rows):
            if abs(augmented_matrix[j, i]) > tolerance:
                factor = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j] -= factor * augmented_matrix[i]
    
    # Check for inconsistent rows (0 = non-zero)
    inconsistent_rows = []
    for i in range(rows):
        # Check if all coefficients are zero but RHS is non-zero
        coeffs_zero = np.allclose(augmented_matrix[i, :-1], 0, atol=tolerance)
        rhs_nonzero = abs(augmented_matrix[i, -1]) > tolerance
        
        if coeffs_zero and rhs_nonzero:
            inconsistent_rows.append(i)
    
    return {
        'reduced_matrix': augmented_matrix,
        'inconsistent_rows': inconsistent_rows,
        'consistent': len(inconsistent_rows) == 0
    }

def demonstrate_consistency_conditions():
    """Demonstrate various consistency conditions with examples"""
    
    print("="*60)
    print("CONSISTENCY CONDITIONS DEMONSTRATION")
    print("="*60)
    
    # Example 1: Consistent with unique solution
    print("\n1. CONSISTENT SYSTEM - UNIQUE SOLUTION:")
    A1 = np.array([[2, 3], [4, 1]], dtype=float)
    b1 = np.array([7, 10], dtype=float)
    
    analysis1 = comprehensive_consistency_analysis(A1, b1)
    print(f"System: 2x + 3y = 7, 4x + y = 10")
    print(f"Coefficient matrix rank: {analysis1['rank_A']}")
    print(f"Augmented matrix rank: {analysis1['rank_augmented']}")
    print(f"Consistent: {analysis1['overall_consistent']}")
    print(f"Solution type: {analysis1['solution_type']}")
    if 'lstsq_solution' in analysis1:
        print(f"Solution: {analysis1['lstsq_solution']}")
    
    # Example 2: Consistent with infinite solutions
    print("\n2. CONSISTENT SYSTEM - INFINITE SOLUTIONS:")
    A2 = np.array([[1, 2, 3], [2, 4, 6]], dtype=float)
    b2 = np.array([1, 2], dtype=float)
    
    analysis2 = comprehensive_consistency_analysis(A2, b2)
    print(f"System: x + 2y + 3z = 1, 2x + 4y + 6z = 2")
    print(f"Coefficient matrix rank: {analysis2['rank_A']}")
    print(f"Augmented matrix rank: {analysis2['rank_augmented']}")
    print(f"Consistent: {analysis2['overall_consistent']}")
    print(f"Solution type: {analysis2['solution_type']}")
    print(f"Free variables: {analysis2['num_free_variables']}")
    
    # Example 3: Inconsistent system
    print("\n3. INCONSISTENT SYSTEM:")
    A3 = np.array([[1, 2], [2, 4]], dtype=float)
    b3 = np.array([1, 3], dtype=float)  # Note: second equation is 2x + 4y = 3
    
    analysis3 = comprehensive_consistency_analysis(A3, b3)
    print(f"System: x + 2y = 1, 2x + 4y = 3")
    print(f"Coefficient matrix rank: {analysis3['rank_A']}")
    print(f"Augmented matrix rank: {analysis3['rank_augmented']}")
    print(f"Consistent: {analysis3['overall_consistent']}")
    print(f"Solution type: {analysis3['solution_type']}")
    print(f"Residual norm: {analysis3.get('residual_norm', 'N/A')}")
    
    # Show row reduction analysis
    if 'row_reduction' in analysis3:
        print(f"Inconsistent rows found: {len(analysis3['row_reduction']['inconsistent_rows'])}")

def special_cases_analysis():
    """Analyze special cases of consistency"""
    
    print(f"\n{'='*60}")
    print("SPECIAL CASES ANALYSIS")
    print(f"{'='*60}")
    
    # Case 1: Homogeneous system
    print("\n1. HOMOGENEOUS SYSTEM (Ax = 0):")
    A_homo = np.array([[1, 2, 3], [2, 4, 6], [1, 2, 3]], dtype=float)
    b_homo = np.zeros(3)
    
    analysis_homo = comprehensive_consistency_analysis(A_homo, b_homo)
    print(f"Matrix rank: {analysis_homo['rank_A']}")
    print(f"Always consistent: True (trivial solution x = 0)")
    print(f"Non-trivial solutions exist: {analysis_homo['rank_A'] < A_homo.shape[1]}")
    print(f"Null space dimension: {A_homo.shape[1] - analysis_homo['rank_A']}")
    
    # Case 2: Square system
    print("\n2. SQUARE SYSTEM:")
    A_square = np.array([[1, 2], [3, 4]], dtype=float)
    det_A = np.linalg.det(A_square)
    print(f"Determinant: {det_A:.6f}")
    print(f"Invertible: {abs(det_A) > 1e-10}")
    
    # Test with different right-hand sides
    for i, b_test in enumerate([[1, 2], [0, 0]]):
        analysis = comprehensive_consistency_analysis(A_square, b_test)
        print(f"  RHS {b_test}: Consistent = {analysis['overall_consistent']}, Type = {analysis['solution_type']}")
    
    # Case 3: Overdetermined system
    print("\n3. OVERDETERMINED SYSTEM (more equations than unknowns):")
    np.random.seed(42)
    A_over = np.random.randn(5, 3)  # 5 equations, 3 unknowns
    
    # Create consistent system
    x_true = np.array([1, 2, 3])
    b_consistent = A_over @ x_true
    
    # Create inconsistent system
    b_inconsistent = b_consistent + np.array([0, 0, 0, 0, 1])  # Add inconsistency
    
    analysis_consistent = comprehensive_consistency_analysis(A_over, b_consistent)
    analysis_inconsistent = comprehensive_consistency_analysis(A_over, b_inconsistent)
    
    print(f"  Consistent case: {analysis_consistent['overall_consistent']}")
    print(f"    Residual: {analysis_consistent.get('residual_norm', 'N/A'):.2e}")
    print(f"  Inconsistent case: {analysis_inconsistent['overall_consistent']}")
    print(f"    Residual: {analysis_inconsistent.get('residual_norm', 'N/A'):.2e}")
    
    # Case 4: Underdetermined system
    print("\n4. UNDERDETERMINED SYSTEM (fewer equations than unknowns):")
    A_under = np.random.randn(3, 5)  # 3 equations, 5 unknowns
    b_under = np.random.randn(3)
    
    analysis_under = comprehensive_consistency_analysis(A_under, b_under)
    print(f"  System shape: {A_under.shape}")
    print(f"  Consistent: {analysis_under['overall_consistent']}")
    print(f"  Solution type: {analysis_under['solution_type']}")
    print(f"  Free variables: {analysis_under.get('num_free_variables', 'N/A')}")

def practical_consistency_checking():
    """Practical considerations for consistency checking"""
    
    print(f"\n{'='*60}")
    print("PRACTICAL CONSISTENCY CHECKING")
    print(f"{'='*60}")
    
    # Numerical tolerance effects
    print("\n1. NUMERICAL TOLERANCE EFFECTS:")
    
    # Create nearly inconsistent system
    A_numeric = np.array([[1, 1], [1, 1 + 1e-12]], dtype=float)
    b_numeric = np.array([1, 1 + 1e-10], dtype=float)
    
    tolerances = [1e-15, 1e-12, 1e-10, 1e-8]
    
    for tol in tolerances:
        analysis = comprehensive_consistency_analysis(A_numeric, b_numeric, tolerance=tol)
        print(f"  Tolerance {tol:.0e}: Consistent = {analysis['overall_consistent']}")
        print(f"    Ranks: A={analysis['rank_A']}, [A|b]={analysis['rank_augmented']}")
    
    # Condition number effects
    print("\n2. CONDITION NUMBER EFFECTS:")
    
    # Well-conditioned system
    A_well = np.array([[1, 0], [0, 1]], dtype=float)
    cond_well = np.linalg.cond(A_well)
    
    # Ill-conditioned system
    A_ill = np.array([[1, 1], [1, 1.0001]], dtype=float)
    cond_ill = np.linalg.cond(A_ill)
    
    print(f"  Well-conditioned: condition number = {cond_well:.2e}")
    print(f"  Ill-conditioned: condition number = {cond_ill:.2e}")
    
    # Test with small perturbations
    b_test = np.array([1, 1])
    b_perturbed = b_test + 1e-10 * np.random.randn(2)
    
    for A, label in [(A_well, "Well-conditioned"), (A_ill, "Ill-conditioned")]:
        sol1 = np.linalg.solve(A, b_test)
        sol2 = np.linalg.solve(A, b_perturbed)
        error = np.linalg.norm(sol1 - sol2)
        print(f"    {label}: Solution sensitivity = {error:.2e}")

def machine_learning_consistency_examples():
    """Machine learning examples of consistency issues"""
    
    print(f"\n{'='*60}")
    print("MACHINE LEARNING CONSISTENCY EXAMPLES")
    print(f"{'='*60}")
    
    # Example 1: Linear regression consistency
    print("\n1. LINEAR REGRESSION NORMAL EQUATIONS:")
    np.random.seed(42)
    
    # Generate data
    n_samples, n_features = 50, 3
    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([1, 2, -1])
    y = X @ true_beta + 0.1 * np.random.randn(n_samples)
    
    # Normal equations: X^T X β = X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    
    analysis = comprehensive_consistency_analysis(XtX, Xty)
    print(f"  Design matrix X shape: {X.shape}")
    print(f"  Normal equations consistent: {analysis['overall_consistent']}")
    print(f"  Solution type: {analysis['solution_type']}")
    
    if analysis['overall_consistent']:
        beta_estimate = analysis['lstsq_solution']
        print(f"  True β: {true_beta}")
        print(f"  Estimated β: {beta_estimate}")
        print(f"  Estimation error: {np.linalg.norm(true_beta - beta_estimate):.4f}")
    
    # Example 2: Underdetermined case (more features than samples)
    print("\n2. UNDERDETERMINED CASE (p > n):")
    n_samples_small, n_features_large = 10, 20
    X_under = np.random.randn(n_samples_small, n_features_large)
    y_under = np.random.randn(n_samples_small)
    
    XtX_under = X_under.T @ X_under
    Xty_under = X_under.T @ y_under
    
    analysis_under = comprehensive_consistency_analysis(XtX_under, Xty_under)
    print(f"  Design matrix X shape: {X_under.shape}")
    print(f"  Normal equations consistent: {analysis_under['overall_consistent']}")
    print(f"  Solution type: {analysis_under['solution_type']}")
    print(f"  Free variables: {analysis_under.get('num_free_variables', 'N/A')}")
    print(f"  → Regularization needed for unique solution!")

if __name__ == "__main__":
    demonstrate_consistency_conditions()
    special_cases_analysis()
    practical_consistency_checking()
    machine_learning_consistency_examples()
    
    print(f"\n{'='*60}")
    print("CONSISTENCY SUMMARY")
    print(f"{'='*60}")
    
    print("\nConsistency Test: rank(A) = rank([A|b])")
    print("├─ If TRUE:")
    print("│  ├─ rank(A) = n → Unique solution")
    print("│  └─ rank(A) < n → Infinite solutions")
    print("└─ If FALSE:")
    print("   └─ No solution (inconsistent)")
    
    print("\nPractical Considerations:")
    print("• Use appropriate numerical tolerance")
    print("• Consider condition number for stability")
    print("• Apply regularization for underdetermined systems")
    print("• Use least squares for overdetermined systems")
    print("• Check consistency before solving")
```

**Advanced Consistency Concepts:**

**1. Fredholm Alternative:**
For matrix A, exactly one of the following holds:
- Either Ax = b has a solution for every b
- Or A^T y = 0 has a non-trivial solution

**2. Solvability Conditions:**
For system Ax = b to be consistent:
b must be orthogonal to all vectors in the left null space of A

**3. Perturbation Analysis:**
Small changes in A or b can affect consistency, especially for ill-conditioned systems

**4. Regularization for Consistency:**
- **Ridge regression**: Always consistent
- **Lasso**: Handles overdetermined systems
- **Tikhonov**: Ensures well-posed problems

**Practical Applications:**

**1. Engineering Design:**
- **Constraint satisfaction**: All design requirements must be consistent
- **Control systems**: Controllability and observability conditions
- **Structural analysis**: Static equilibrium equations

**2. Computer Graphics:**
- **3D reconstruction**: Multiple view consistency
- **Animation**: Constraint-based motion
- **Rendering**: Lighting equation consistency

**3. Economics:**
- **Linear programming**: Feasible region existence
- **Input-output models**: Economic balance equations
- **Portfolio optimization**: Return and risk constraints

**4. Signal Processing:**
- **System identification**: Model parameter estimation
- **Filter design**: Frequency response specifications
- **Reconstruction**: Sampling consistency conditions

**Best Practices:**
- Always test consistency before attempting to solve
- Use appropriate numerical tolerances for rank computations
- Consider regularization for ill-posed problems
- Understand the geometric meaning of your system
- Check condition numbers for numerical stability
- Use domain knowledge to validate solution reasonableness

---

## Question 20

**Explain the LU decomposition of a matrix.**

**Answer:** LU decomposition is a matrix factorization technique that decomposes a square matrix A into the product of a lower triangular matrix L and an upper triangular matrix U. This factorization is fundamental in numerical linear algebra for efficiently solving systems of equations, computing determinants, and matrix inversion.

**Mathematical Definition:**
For an n×n matrix A, LU decomposition finds matrices L and U such that:
```
A = LU
```
where:
- **L**: Lower triangular matrix (elements above diagonal are zero)
- **U**: Upper triangular matrix (elements below diagonal are zero)

**Standard Form:**
```
A = [a₁₁ a₁₂ a₁₃]   [1   0   0 ] [u₁₁ u₁₂ u₁₃]
    [a₂₁ a₂₂ a₂₃] = [l₂₁ 1   0 ] [0   u₂₂ u₂₃]
    [a₃₁ a₃₂ a₃₃]   [l₃₁ l₃₂ 1 ] [0   0   u₃₃]
```

**Types of LU Decomposition:**

**1. Doolittle Decomposition:**
- **L has unit diagonal** (lᵢᵢ = 1)
- **U has arbitrary diagonal**
- Most common form

**2. Crout Decomposition:**
- **U has unit diagonal** (uᵢᵢ = 1)
- **L has arbitrary diagonal**

**3. LU with Partial Pivoting (PLU):**
- **PA = LU** where P is permutation matrix
- **Improves numerical stability**
- **Industry standard**

**4. LU with Complete Pivoting (PLUQ):**
- **PAQ = LU** with both row and column permutations
- **Maximum stability but higher computational cost**

**Algorithm (Gaussian Elimination Method):**

**Step-by-Step Process:**
1. **Initialize**: L as identity matrix, U as copy of A
2. **Forward elimination**: For each column k:
   - Compute multipliers: lᵢₖ = uᵢₖ / uₖₖ for i > k
   - Update U: subtract lᵢₖ × (row k) from row i
   - Store multipliers in L

**Detailed Algorithm:**
```
For k = 1 to n-1:
    For i = k+1 to n:
        l[i,k] = u[i,k] / u[k,k]  # Compute multiplier
        For j = k to n:
            u[i,j] = u[i,j] - l[i,k] * u[k,j]  # Update row
```

**Existence and Uniqueness:**

**Existence Conditions:**
- **All leading principal minors are non-zero**
- **Equivalently**: Gaussian elimination proceeds without row exchanges
- **If condition fails**: Use partial pivoting (PLU)

**Uniqueness:**
- **With normalization** (unit diagonal in L or U): decomposition is unique
- **Without normalization**: Infinitely many solutions

**Computational Aspects:**

**Time Complexity:**
- **Standard LU**: O(n³) operations
- **Same as Gaussian elimination**
- **Once computed**: Solving Lz = b and Ux = z each takes O(n²)

**Space Complexity:**
- **In-place**: Can overwrite A with L and U
- **Memory efficient**: No additional storage needed

**Numerical Stability:**
- **Without pivoting**: Can be unstable for some matrices
- **With partial pivoting**: Generally stable
- **Growth factor**: Measures element magnitude increase

**Applications:**

**1. Solving Linear Systems:**
```
Ax = b → LUx = b
Step 1: Solve Lz = b (forward substitution)
Step 2: Solve Ux = z (backward substitution)
```

**2. Matrix Inversion:**
```
A⁻¹ = U⁻¹L⁻¹
```
Compute by solving AX = I column by column

**3. Determinant Calculation:**
```
det(A) = det(L) × det(U) = det(U) = u₁₁ × u₂₂ × ... × uₙₙ
```

**4. Matrix Powers:**
For A^k, use LU decomposition to solve iteratively

**5. Least Squares Problems:**
Use LU of normal equations X^T X

**Machine Learning Applications:**

**1. Linear Regression:**
- **Normal equations**: (X^T X)β = X^T y
- **LU decomposition**: Solve efficiently for β
- **Multiple right-hand sides**: Reuse factorization

**2. Neural Network Training:**
- **Hessian methods**: Second-order optimization
- **Newton's method**: Solve H × Δw = -∇L
- **Quasi-Newton**: Update LU factors incrementally

**3. Kalman Filtering:**
- **Covariance updates**: Matrix inversion via LU
- **State estimation**: Solve linear systems efficiently
- **Sequential processing**: Reuse factorizations

**4. Principal Component Analysis:**
- **Covariance matrix**: Eigenvalue problems
- **Gram matrix**: X^T X decomposition
- **Whitening transformation**: Matrix square root via LU

**5. Gaussian Processes:**
- **Kernel matrix inversion**: K⁻¹y for predictions
- **Log-likelihood**: log det(K) via LU
- **Hyperparameter optimization**: Gradient computation

**Implementation Examples:**

```python
import numpy as np
from scipy.linalg import lu, solve_triangular
import time

def lu_decomposition_manual(A):
    """
    Manual implementation of LU decomposition without pivoting
    
    Args:
        A: Square matrix to decompose
    
    Returns:
        L: Lower triangular matrix
        U: Upper triangular matrix
    """
    n = A.shape[0]
    A = A.astype(float)
    
    # Initialize L as identity and U as copy of A
    L = np.eye(n)
    U = A.copy()
    
    # Forward elimination
    for k in range(n - 1):
        # Check for zero pivot
        if abs(U[k, k]) < 1e-10:
            raise ValueError(f"Zero pivot encountered at position ({k}, {k})")
        
        # Compute multipliers and eliminate
        for i in range(k + 1, n):
            # Compute multiplier
            multiplier = U[i, k] / U[k, k]
            L[i, k] = multiplier
            
            # Update row i
            for j in range(k, n):
                U[i, j] -= multiplier * U[k, j]
    
    return L, U

def lu_with_partial_pivoting(A):
    """
    LU decomposition with partial pivoting
    
    Args:
        A: Square matrix to decompose
    
    Returns:
        P: Permutation matrix
        L: Lower triangular matrix  
        U: Upper triangular matrix
    """
    n = A.shape[0]
    A = A.astype(float)
    
    # Initialize
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)
    
    for k in range(n - 1):
        # Find pivot (largest element in column k, rows k to n-1)
        pivot_row = k + np.argmax(np.abs(U[k:n, k]))
        
        # Swap rows in U and P if needed
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            
            # Swap corresponding rows in L (only lower part)
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        # Check for zero pivot
        if abs(U[k, k]) < 1e-10:
            print(f"Warning: Very small pivot {U[k, k]} at position ({k}, {k})")
        
        # Eliminate below pivot
        for i in range(k + 1, n):
            if U[k, k] != 0:
                multiplier = U[i, k] / U[k, k]
                L[i, k] = multiplier
                U[i] -= multiplier * U[k]
    
    return P, L, U

def solve_using_lu(L, U, b, P=None):
    """
    Solve linear system using LU decomposition
    
    Args:
        L: Lower triangular matrix
        U: Upper triangular matrix
        b: Right-hand side vector
        P: Optional permutation matrix
    
    Returns:
        x: Solution vector
    """
    if P is not None:
        b = P @ b
    
    # Forward substitution: Lz = b
    z = solve_triangular(L, b, lower=True)
    
    # Backward substitution: Ux = z
    x = solve_triangular(U, z, lower=False)
    
    return x

def demonstrate_lu_decomposition():
    """Demonstrate LU decomposition with examples"""
    
    print("="*60)
    print("LU DECOMPOSITION DEMONSTRATION")
    print("="*60)
    
    # Example 1: Simple 3x3 matrix
    print("\n1. BASIC LU DECOMPOSITION:")
    A1 = np.array([[2, 1, 1],
                   [4, 3, 3], 
                   [8, 7, 9]], dtype=float)
    
    print(f"Original matrix A:")
    print(A1)
    
    try:
        L1, U1 = lu_decomposition_manual(A1)
        
        print(f"\nL (Lower triangular):")
        print(L1)
        print(f"\nU (Upper triangular):")
        print(U1)
        
        # Verify decomposition
        reconstruction = L1 @ U1
        print(f"\nVerification L@U:")
        print(reconstruction)
        print(f"Reconstruction error: {np.linalg.norm(A1 - reconstruction):.2e}")
        
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 2: LU with partial pivoting
    print(f"\n2. LU WITH PARTIAL PIVOTING:")
    A2 = np.array([[0, 1, 1],
                   [2, 1, 1],
                   [4, 3, 3]], dtype=float)  # First element is 0 (needs pivoting)
    
    print(f"Matrix A (needs pivoting):")
    print(A2)
    
    P2, L2, U2 = lu_with_partial_pivoting(A2)
    
    print(f"\nP (Permutation matrix):")
    print(P2)
    print(f"\nL (Lower triangular):")
    print(L2)
    print(f"\nU (Upper triangular):")
    print(U2)
    
    # Verify: PA = LU
    verification = P2 @ A2
    reconstruction = L2 @ U2
    print(f"\nVerification PA:")
    print(verification)
    print(f"LU:")
    print(reconstruction)
    print(f"Error ||PA - LU||: {np.linalg.norm(verification - reconstruction):.2e}")
    
    # Example 3: Solving linear system
    print(f"\n3. SOLVING LINEAR SYSTEM:")
    b = np.array([4, 8, 18])
    print(f"Solving Ax = b where b = {b}")
    
    # Solve using LU
    x_lu = solve_using_lu(L2, U2, b, P2)
    print(f"Solution using LU: x = {x_lu}")
    
    # Verify solution
    residual = A2 @ x_lu - b
    print(f"Residual ||Ax - b||: {np.linalg.norm(residual):.2e}")
    
    # Compare with direct solve
    x_direct = np.linalg.solve(A2, b)
    print(f"Solution using numpy: x = {x_direct}")
    print(f"Difference: {np.linalg.norm(x_lu - x_direct):.2e}")

def performance_comparison():
    """Compare LU decomposition performance"""
    
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    sizes = [50, 100, 200, 500]
    
    print(f"{'Size':<8} {'LU Time':<12} {'Solve Time':<12} {'Direct Time':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for n in sizes:
        # Generate random matrix and RHS
        np.random.seed(42)
        A = np.random.randn(n, n) + n * np.eye(n)  # Well-conditioned
        b = np.random.randn(n)
        
        # Time LU decomposition
        start_time = time.time()
        P, L, U = lu(A)
        lu_time = time.time() - start_time
        
        # Time solving with precomputed LU
        start_time = time.time()
        x_lu = solve_using_lu(L, U, b, P)
        solve_time = time.time() - start_time
        
        # Time direct solve
        start_time = time.time()
        x_direct = np.linalg.solve(A, b)
        direct_time = time.time() - start_time
        
        # Calculate speedup for multiple solves
        total_lu_time = lu_time + solve_time
        speedup = direct_time / total_lu_time if total_lu_time > 0 else 0
        
        print(f"{n:<8} {lu_time:<12.4f} {solve_time:<12.4f} {direct_time:<12.4f} {speedup:<10.2f}")

def advanced_lu_applications():
    """Demonstrate advanced applications of LU decomposition"""
    
    print(f"\n{'='*60}")
    print("ADVANCED APPLICATIONS")
    print(f"{'='*60}")
    
    # Application 1: Matrix determinant
    print("\n1. DETERMINANT CALCULATION:")
    A_det = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 10]], dtype=float)
    
    P, L, U = lu(A_det)
    
    # Determinant = product of U diagonal elements × sign of permutation
    det_from_lu = np.prod(np.diag(U)) * (-1)**(len(P) - np.trace(P))
    det_numpy = np.linalg.det(A_det)
    
    print(f"Matrix A:")
    print(A_det)
    print(f"Determinant from LU: {det_from_lu:.6f}")
    print(f"Determinant from NumPy: {det_numpy:.6f}")
    print(f"Difference: {abs(det_from_lu - det_numpy):.2e}")
    
    # Application 2: Matrix inversion
    print(f"\n2. MATRIX INVERSION:")
    A_inv = np.array([[4, 3], [3, 2]], dtype=float)
    
    P, L, U = lu(A_inv)
    
    # Solve AX = I column by column
    n = A_inv.shape[0]
    I = np.eye(n)
    A_inverse = np.zeros_like(A_inv)
    
    for i in range(n):
        A_inverse[:, i] = solve_using_lu(L, U, I[:, i], P)
    
    print(f"Original matrix:")
    print(A_inv)
    print(f"Inverse via LU:")
    print(A_inverse)
    print(f"Verification A @ A^(-1):")
    print(A_inv @ A_inverse)
    
    # Application 3: Multiple right-hand sides
    print(f"\n3. MULTIPLE RIGHT-HAND SIDES:")
    np.random.seed(42)
    A_multi = np.random.randn(4, 4) + 4 * np.eye(4)
    B_multi = np.random.randn(4, 3)  # 3 different RHS vectors
    
    # Single LU decomposition
    P, L, U = lu(A_multi)
    
    # Solve for all RHS
    X_multi = np.zeros_like(B_multi)
    for i in range(B_multi.shape[1]):
        X_multi[:, i] = solve_using_lu(L, U, B_multi[:, i], P)
    
    # Verify solutions
    residual_norm = np.linalg.norm(A_multi @ X_multi - B_multi)
    print(f"System size: {A_multi.shape}")
    print(f"Number of RHS vectors: {B_multi.shape[1]}")
    print(f"Residual norm: {residual_norm:.2e}")
    print("→ Efficient: One decomposition, multiple solves!")

def numerical_stability_analysis():
    """Analyze numerical stability of LU decomposition"""
    
    print(f"\n{'='*60}")
    print("NUMERICAL STABILITY ANALYSIS")
    print(f"{'='*60}")
    
    # Test with ill-conditioned matrix
    print("\n1. ILL-CONDITIONED MATRIX:")
    epsilon = 1e-12
    A_ill = np.array([[1, 1], [1, 1 + epsilon]], dtype=float)
    
    condition_number = np.linalg.cond(A_ill)
    print(f"Condition number: {condition_number:.2e}")
    
    try:
        P, L, U = lu(A_ill)
        print(f"LU decomposition successful")
        print(f"U matrix:")
        print(U)
        
        # Check growth factor
        max_original = np.max(np.abs(A_ill))
        max_U = np.max(np.abs(U))
        growth_factor = max_U / max_original
        print(f"Growth factor: {growth_factor:.2e}")
        
    except Exception as e:
        print(f"LU decomposition failed: {e}")
    
    # Compare with and without pivoting
    print(f"\n2. PIVOTING COMPARISON:")
    A_pivot_test = np.array([[1e-15, 1], [1, 1]], dtype=float)
    
    print(f"Test matrix (small pivot):")
    print(A_pivot_test)
    
    # Without pivoting (manual implementation)
    try:
        L_no_pivot, U_no_pivot = lu_decomposition_manual(A_pivot_test)
        growth_no_pivot = np.max(np.abs(U_no_pivot)) / np.max(np.abs(A_pivot_test))
        print(f"Without pivoting - Growth factor: {growth_no_pivot:.2e}")
    except:
        print("Without pivoting - Failed due to small pivot")
    
    # With pivoting
    P_pivot, L_pivot, U_pivot = lu_with_partial_pivoting(A_pivot_test)
    growth_pivot = np.max(np.abs(U_pivot)) / np.max(np.abs(A_pivot_test))
    print(f"With pivoting - Growth factor: {growth_pivot:.2e}")

def machine_learning_lu_examples():
    """Machine learning applications using LU decomposition"""
    
    print(f"\n{'='*60}")
    print("MACHINE LEARNING APPLICATIONS")
    print(f"{'='*60}")
    
    # Example 1: Linear regression normal equations
    print("\n1. LINEAR REGRESSION:")
    np.random.seed(42)
    
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_beta = np.random.randn(n_features)
    y = X @ true_beta + 0.1 * np.random.randn(n_samples)
    
    # Normal equations: (X^T X) β = X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    
    # Solve using LU decomposition
    P, L, U = lu(XtX)
    beta_estimate = solve_using_lu(L, U, Xty, P)
    
    print(f"True coefficients: {true_beta}")
    print(f"Estimated coefficients: {beta_estimate}")
    print(f"Estimation error: {np.linalg.norm(true_beta - beta_estimate):.4f}")
    
    # Example 2: Covariance matrix operations
    print(f"\n2. COVARIANCE MATRIX OPERATIONS:")
    # Generate correlated data
    n_dims = 4
    correlation_matrix = np.random.randn(n_dims, n_dims)
    correlation_matrix = correlation_matrix @ correlation_matrix.T  # Make positive definite
    
    # Add some data points
    n_points = 50
    data = np.random.multivariate_normal(np.zeros(n_dims), correlation_matrix, n_points)
    empirical_cov = np.cov(data.T)
    
    # Use LU for various operations
    P, L, U = lu(empirical_cov)
    
    # Log determinant (for multivariate Gaussian likelihood)
    log_det = np.sum(np.log(np.abs(np.diag(U))))
    log_det_numpy = np.linalg.slogdet(empirical_cov)[1]
    
    print(f"Log determinant via LU: {log_det:.4f}")
    print(f"Log determinant via NumPy: {log_det_numpy:.4f}")
    print(f"Difference: {abs(log_det - log_det_numpy):.2e}")

if __name__ == "__main__":
    demonstrate_lu_decomposition()
    performance_comparison()
    advanced_lu_applications()
    numerical_stability_analysis()
    machine_learning_lu_examples()
    
    print(f"\n{'='*60}")
    print("LU DECOMPOSITION SUMMARY")
    print(f"{'='*60}")
    
    print("\nKey Benefits:")
    print("• Efficient multiple solves: O(n³) decomposition + O(n²) per solve")
    print("• Matrix operations: determinant, inverse, powers")
    print("• Numerical stability: with partial pivoting")
    print("• Memory efficient: in-place computation possible")
    
    print("\nApplications:")
    print("• Linear systems: Ax = b")
    print("• Linear regression: normal equations")
    print("• Gaussian processes: kernel matrix operations")
    print("• Kalman filtering: covariance updates")
    print("• Optimization: Newton's method, quasi-Newton")
    
    print("\nBest Practices:")
    print("• Use partial pivoting for stability")
    print("• Check condition number for ill-conditioned matrices")
    print("• Consider iterative methods for very large sparse systems")
    print("• Reuse factorization for multiple right-hand sides")
    print("• Monitor growth factor for numerical issues")
```

**Theoretical Properties:**

**1. Existence Theorem:**
LU decomposition exists without pivoting if and only if all leading principal minors are non-zero.

**2. Uniqueness Theorem:**
If LU decomposition exists with unit diagonal constraint, it is unique.

**3. Stability Theorem:**
LU with partial pivoting is backward stable: computed solution solves a nearby problem.

**4. Growth Factor:**
Maximum element growth during elimination. For partial pivoting: growth ≤ 2^(n-1).

**Variations and Extensions:**

**1. Block LU Decomposition:**
For large matrices, partition into blocks and apply recursively.

**2. Sparse LU:**
Specialized algorithms to preserve sparsity structure.

**3. Incremental LU:**
Update factorization when matrix changes slightly.

**4. GPU Implementation:**
Parallel algorithms for high-performance computing.

**5. Iterative Refinement:**
Improve solution accuracy using LU factorization.

**Comparison with Other Methods:**

**Vs QR Decomposition:**
- **LU**: Faster for square systems
- **QR**: Better for overdetermined systems, more stable

**Vs Cholesky:**
- **LU**: General matrices
- **Cholesky**: Symmetric positive definite (faster)

**Vs SVD:**
- **LU**: Fast for solving systems
- **SVD**: Better for rank-deficient problems

**Best Practices:**
- Always use partial pivoting for numerical stability
- Check condition number before factorization
- Consider alternative methods for special matrix structures
- Reuse factorization when solving multiple systems
- Monitor growth factor for stability assessment
- Use block algorithms for cache efficiency on large matrices

---

## Question 21

**What are singular or ill-conditioned matrices?**

**Answer:** Singular and ill-conditioned matrices are fundamental concepts in numerical linear algebra that describe matrices with problematic numerical properties. Understanding these concepts is crucial for machine learning practitioners as they directly affect the stability and reliability of computational algorithms.

**Singular Matrices:**

**Definition:**
A square matrix A is **singular** (also called non-invertible or degenerate) if its determinant is zero and it does not have an inverse. Mathematically:
```
det(A) = 0  ⟺  A is singular
```

**Mathematical Characterization:**
A matrix A is singular if and only if:
- **Determinant**: det(A) = 0
- **Rank deficiency**: rank(A) < n (for n×n matrix)
- **Linear dependence**: Columns (or rows) are linearly dependent
- **Null space**: Non-trivial null space (Ax = 0 has non-zero solutions)
- **Zero eigenvalue**: At least one eigenvalue equals zero

**Examples of Singular Matrices:**

**1. Zero Matrix:**
```
A = [0 0]  → det(A) = 0
    [0 0]
```

**2. Linearly Dependent Columns:**
```
A = [1 2]  → det(A) = 1×4 - 2×2 = 0
    [2 4]
```

**3. Rank-Deficient Matrix:**
```
A = [1 2 3]  → rank(A) = 2 < 3
    [4 5 6]
    [7 8 9]
```

**Ill-Conditioned Matrices:**

**Definition:**
An ill-conditioned matrix is a non-singular matrix that is "close" to being singular, meaning small changes in the input can cause large changes in the output. This is quantified by the **condition number**.

**Condition Number:**
```
κ(A) = ||A|| × ||A⁻¹||
```

For the 2-norm (most common):
```
κ₂(A) = σₘₐₓ(A) / σₘᵢₙ(A)
```
where σₘₐₓ and σₘᵢₙ are the largest and smallest singular values.

**Condition Number Interpretation:**
- **κ(A) = 1**: Perfectly conditioned (orthogonal matrices)
- **κ(A) < 10**: Well-conditioned
- **10 ≤ κ(A) < 100**: Moderately conditioned
- **κ(A) ≥ 100**: Ill-conditioned
- **κ(A) → ∞**: Approaching singularity

**Sources of Ill-Conditioning:**

**1. Near-Linear Dependence:**
```
A = [1.000  1.001]  → κ(A) ≈ 4000
    [1.001  1.002]
```

**2. Wide Range of Scales:**
```
A = [1    0  ]  → κ(A) = 10⁶
    [0   10⁻⁶]
```

**3. Hilbert Matrices:**
```
H_ij = 1/(i+j-1)  → Notoriously ill-conditioned
```

**4. Vandermonde Matrices:**
```
V_ij = x_i^(j-1)  → Ill-conditioned for clustered points
```

**Effects of Ill-Conditioning:**

**1. Amplified Errors:**
Small input errors δx lead to large output errors:
```
||δy|| / ||y|| ≤ κ(A) × ||δx|| / ||x||
```

**2. Loss of Significant Digits:**
Approximately log₁₀(κ(A)) digits lost in computation.

**3. Numerical Instability:**
Algorithms may fail or produce unreliable results.

**Detection Methods:**

**1. Condition Number Estimation:**
```python
import numpy as np

# Direct computation
cond_num = np.linalg.cond(A)

# Using SVD
U, s, Vt = np.linalg.svd(A)
cond_num_svd = s[0] / s[-1]  # σ_max / σ_min
```

**2. Rank Estimation:**
```python
# Numerical rank
rank_numerical = np.linalg.matrix_rank(A, tol=1e-10)
rank_theoretical = min(A.shape)

if rank_numerical < rank_theoretical:
    print("Matrix is numerically singular")
```

**3. Determinant Analysis:**
```python
det_A = np.linalg.det(A)
if abs(det_A) < 1e-12:
    print("Matrix is near-singular")
```

**4. Eigenvalue Analysis:**
```python
eigenvals = np.linalg.eigvals(A)
min_eigenval = min(abs(eigenvals))
if min_eigenval < 1e-12:
    print("Matrix has near-zero eigenvalues")
```

**Machine Learning Implications:**

**1. Linear Regression:**

**Problem:** Normal equations X^T X can be ill-conditioned
```python
# Ill-conditioned normal equations
XtX = X.T @ X
cond_XtX = np.linalg.cond(XtX)

# If κ(X^T X) is large, use regularization
if cond_XtX > 1e12:
    # Add ridge regularization
    beta = np.linalg.solve(XtX + lambda_reg * np.eye(n), X.T @ y)
```

**Solution:** Use QR decomposition or SVD instead:
```python
# More stable: QR decomposition
Q, R = np.linalg.qr(X)
beta = np.linalg.solve(R, Q.T @ y)
```

**2. Principal Component Analysis:**

**Problem:** Covariance matrix can be ill-conditioned
```python
# Check covariance matrix conditioning
cov_matrix = np.cov(data.T)
cond_cov = np.linalg.cond(cov_matrix)

if cond_cov > 1e12:
    print("PCA may be unstable")
    # Use SVD directly on data matrix
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
```

**3. Neural Network Training:**

**Problem:** Weight matrices can become ill-conditioned
```python
# Monitor weight matrix condition numbers
for layer in model.layers:
    if hasattr(layer, 'weight'):
        cond_weight = np.linalg.cond(layer.weight.detach().numpy())
        if cond_weight > 1e6:
            print(f"Layer {layer} has ill-conditioned weights")
```

**Solutions:** Use proper initialization, gradient clipping, batch normalization

**4. Optimization Problems:**

**Problem:** Hessian matrix ill-conditioning affects convergence
```python
# Check Hessian conditioning
H = compute_hessian(loss_function, params)
cond_H = np.linalg.cond(H)

if cond_H > 1e8:
    # Use regularized Newton's method
    step = np.linalg.solve(H + damping * np.eye(len(H)), gradient)
```

**Remedies and Solutions:**

**1. Regularization:**
```python
# Ridge regularization
A_reg = A.T @ A + lambda_reg * np.eye(n)
```

**2. Singular Value Decomposition:**
```python
# Pseudo-inverse using SVD
U, s, Vt = np.linalg.svd(A)
s_inv = np.where(s > tolerance, 1/s, 0)
A_pinv = Vt.T @ np.diag(s_inv) @ U.T
```

**3. QR Decomposition with Pivoting:**
```python
# More stable than direct inversion
Q, R, P = scipy.linalg.qr(A, pivoting=True)
```

**4. Iterative Methods:**
```python
# For large sparse systems
from scipy.sparse.linalg import cg
x, info = cg(A, b, tol=1e-10)
```

**Implementation Examples:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd
from scipy.sparse.linalg import cg
import warnings

def analyze_matrix_conditioning(A, name="Matrix"):
    """
    Comprehensive analysis of matrix conditioning
    
    Args:
        A: Input matrix
        name: Name for display
    """
    print(f"\n{'='*50}")
    print(f"CONDITIONING ANALYSIS: {name}")
    print(f"{'='*50}")
    
    # Basic properties
    m, n = A.shape
    print(f"Shape: {m} × {n}")
    
    # Determinant (for square matrices)
    if m == n:
        det_A = np.linalg.det(A)
        print(f"Determinant: {det_A:.2e}")
        
        if abs(det_A) < 1e-12:
            print("→ Matrix is numerically singular!")
    
    # Rank analysis
    rank_A = np.linalg.matrix_rank(A)
    theoretical_rank = min(m, n)
    print(f"Rank: {rank_A} / {theoretical_rank}")
    
    if rank_A < theoretical_rank:
        print("→ Matrix is rank deficient!")
    
    # Condition number
    try:
        cond_A = np.linalg.cond(A)
        print(f"Condition number: {cond_A:.2e}")
        
        if cond_A < 10:
            status = "Well-conditioned"
        elif cond_A < 100:
            status = "Moderately conditioned" 
        elif cond_A < 1e12:
            status = "Ill-conditioned"
        else:
            status = "Nearly singular"
        
        print(f"Status: {status}")
        
        # Estimate digits lost
        digits_lost = np.log10(cond_A)
        print(f"Approximate digits lost: {digits_lost:.1f}")
        
    except np.linalg.LinAlgError:
        print("Condition number: Infinite (singular matrix)")
    
    # SVD analysis
    U, s, Vt = svd(A, full_matrices=False)
    print(f"Singular values: {s}")
    print(f"Singular value ratio: {s[0]/s[-1]:.2e}")
    
    # Eigenvalue analysis (for square matrices)
    if m == n:
        try:
            eigenvals = np.linalg.eigvals(A)
            min_eigenval = min(abs(eigenvals))
            print(f"Minimum |eigenvalue|: {min_eigenval:.2e}")
            
            if min_eigenval < 1e-12:
                print("→ Matrix has near-zero eigenvalues!")
                
        except np.linalg.LinAlgError:
            print("Eigenvalue computation failed")
    
    return cond_A if 'cond_A' in locals() else float('inf')

def demonstrate_singular_matrices():
    """Demonstrate different types of singular matrices"""
    
    print("SINGULAR MATRICES EXAMPLES")
    print("="*40)
    
    # Example 1: Linearly dependent columns
    print("\n1. LINEARLY DEPENDENT COLUMNS:")
    A1 = np.array([[1, 2, 3],
                   [2, 4, 6], 
                   [3, 6, 9]], dtype=float)
    
    print(f"Matrix A1:")
    print(A1)
    analyze_matrix_conditioning(A1, "Linearly Dependent")
    
    # Example 2: Rank-1 matrix
    print("\n2. RANK-1 MATRIX:")
    u = np.array([[1], [2], [3]])
    v = np.array([[1, 2, 1]])
    A2 = u @ v  # Rank-1 matrix
    
    print(f"Matrix A2 = u @ v^T:")
    print(A2)
    analyze_matrix_conditioning(A2, "Rank-1")
    
    # Example 3: Nearly singular
    print("\n3. NEARLY SINGULAR MATRIX:")
    epsilon = 1e-12
    A3 = np.array([[1, 1], 
                   [1, 1 + epsilon]], dtype=float)
    
    print(f"Matrix A3 (ε = {epsilon}):")
    print(A3)
    analyze_matrix_conditioning(A3, "Nearly Singular")

def demonstrate_ill_conditioned_matrices():
    """Demonstrate ill-conditioned matrices and their effects"""
    
    print(f"\n{'='*50}")
    print("ILL-CONDITIONED MATRICES")
    print(f"{'='*50}")
    
    # Hilbert matrix (notorious for ill-conditioning)
    print("\n1. HILBERT MATRIX:")
    n = 6
    H = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
    
    print(f"Hilbert matrix H_{n}×{n}:")
    print(H)
    cond_H = analyze_matrix_conditioning(H, f"Hilbert {n}×{n}")
    
    # Demonstrate error amplification
    print(f"\n2. ERROR AMPLIFICATION:")
    b = np.ones(n)
    x_true = np.linalg.solve(H, b)
    
    # Add small perturbation to b
    perturbation = 1e-10
    b_perturbed = b + perturbation * np.random.randn(n)
    x_perturbed = np.linalg.solve(H, b_perturbed)
    
    input_error = np.linalg.norm(b_perturbed - b) / np.linalg.norm(b)
    output_error = np.linalg.norm(x_perturbed - x_true) / np.linalg.norm(x_true)
    amplification = output_error / input_error
    
    print(f"Input relative error: {input_error:.2e}")
    print(f"Output relative error: {output_error:.2e}")
    print(f"Error amplification: {amplification:.2e}")
    print(f"Condition number: {cond_H:.2e}")
    print(f"Theoretical bound: {cond_H * input_error:.2e}")
    
    # Vandermonde matrix
    print(f"\n3. VANDERMONDE MATRIX:")
    x_points = np.linspace(0, 1, 8)  # Clustered points
    V = np.vander(x_points, increasing=True)
    
    print(f"Vandermonde matrix for points {x_points[:3]}...")
    print(V[:3, :3], "...")
    cond_V = analyze_matrix_conditioning(V, "Vandermonde")

def solution_strategies():
    """Demonstrate strategies for handling singular/ill-conditioned matrices"""
    
    print(f"\n{'='*50}")
    print("SOLUTION STRATEGIES")
    print(f"{'='*50}")
    
    # Create ill-conditioned problem
    np.random.seed(42)
    n = 8
    
    # Generate ill-conditioned matrix
    U, _, Vt = svd(np.random.randn(n, n))
    s = np.logspace(0, -12, n)  # Condition number ≈ 10^12
    A_ill = U @ np.diag(s) @ Vt
    b = np.random.randn(n)
    
    print(f"Problem: Ax = b with κ(A) ≈ {np.linalg.cond(A_ill):.2e}")
    
    # Strategy 1: Direct solve (may be unstable)
    print(f"\n1. DIRECT SOLVE:")
    try:
        x_direct = np.linalg.solve(A_ill, b)
        residual_direct = np.linalg.norm(A_ill @ x_direct - b)
        print(f"Residual: {residual_direct:.2e}")
    except np.linalg.LinAlgError:
        print("Direct solve failed")
    
    # Strategy 2: SVD pseudo-inverse
    print(f"\n2. SVD PSEUDO-INVERSE:")
    U, s, Vt = svd(A_ill)
    tolerance = 1e-10
    s_inv = np.where(s > tolerance, 1/s, 0)
    A_pinv = Vt.T @ np.diag(s_inv) @ U.T
    x_svd = A_pinv @ b
    residual_svd = np.linalg.norm(A_ill @ x_svd - b)
    print(f"Tolerance: {tolerance}")
    print(f"Effective rank: {np.sum(s > tolerance)}/{len(s)}")
    print(f"Residual: {residual_svd:.2e}")
    
    # Strategy 3: Regularization
    print(f"\n3. TIKHONOV REGULARIZATION:")
    lambda_reg = 1e-6
    A_reg = A_ill.T @ A_ill + lambda_reg * np.eye(n)
    x_reg = np.linalg.solve(A_reg, A_ill.T @ b)
    residual_reg = np.linalg.norm(A_ill @ x_reg - b)
    print(f"Regularization parameter: {lambda_reg}")
    print(f"Regularized condition: {np.linalg.cond(A_reg):.2e}")
    print(f"Residual: {residual_reg:.2e}")
    
    # Strategy 4: QR decomposition
    print(f"\n4. QR DECOMPOSITION:")
    try:
        Q, R = qr(A_ill)
        x_qr = np.linalg.solve(R, Q.T @ b)
        residual_qr = np.linalg.norm(A_ill @ x_qr - b)
        print(f"QR condition (R): {np.linalg.cond(R):.2e}")
        print(f"Residual: {residual_qr:.2e}")
    except np.linalg.LinAlgError:
        print("QR decomposition failed")
    
    # Compare solutions
    print(f"\n5. SOLUTION COMPARISON:")
    solutions = {
        'Direct': x_direct if 'x_direct' in locals() else None,
        'SVD': x_svd,
        'Regularized': x_reg,
        'QR': x_qr if 'x_qr' in locals() else None
    }
    
    for name, x in solutions.items():
        if x is not None:
            norm_x = np.linalg.norm(x)
            print(f"{name:<12}: ||x|| = {norm_x:.4f}")

def machine_learning_examples():
    """ML examples of singular/ill-conditioned matrix problems"""
    
    print(f"\n{'='*50}")
    print("MACHINE LEARNING APPLICATIONS")
    print(f"{'='*50}")
    
    # Example 1: Multicollinearity in regression
    print("\n1. MULTICOLLINEARITY IN REGRESSION:")
    
    np.random.seed(42)
    n_samples, n_features = 100, 5
    
    # Create design matrix with multicollinearity
    X = np.random.randn(n_samples, n_features-1)
    # Last column is nearly linear combination of first two
    X_multicol = np.column_stack([X, X[:, 0] + X[:, 1] + 0.01 * np.random.randn(n_samples)])
    
    # Generate response
    true_beta = np.array([1, -1, 0.5, 2, 0])  # Last coefficient should be near 0
    y = X_multicol @ true_beta + 0.1 * np.random.randn(n_samples)
    
    # Check conditioning
    XtX = X_multicol.T @ X_multicol
    cond_XtX = np.linalg.cond(XtX)
    print(f"Design matrix condition: {cond_XtX:.2e}")
    
    # Compare solutions
    # Normal equations (unstable)
    beta_normal = np.linalg.solve(XtX, X_multicol.T @ y)
    
    # Ridge regression (stable)
    lambda_ridge = 0.1
    beta_ridge = np.linalg.solve(XtX + lambda_ridge * np.eye(n_features), 
                                X_multicol.T @ y)
    
    print(f"True coefficients:   {true_beta}")
    print(f"Normal equations:    {np.round(beta_normal, 3)}")
    print(f"Ridge regression:    {np.round(beta_ridge, 3)}")
    
    error_normal = np.linalg.norm(true_beta - beta_normal)
    error_ridge = np.linalg.norm(true_beta - beta_ridge)
    print(f"Error (normal):      {error_normal:.4f}")
    print(f"Error (ridge):       {error_ridge:.4f}")
    
    # Example 2: PCA with ill-conditioned covariance
    print(f"\n2. PCA WITH ILL-CONDITIONED COVARIANCE:")
    
    # Generate data with some very small variance directions
    n_samples, n_dims = 200, 6
    
    # True covariance with wide range of eigenvalues
    true_eigenvals = np.logspace(2, -8, n_dims)  # 10^10 condition number
    true_eigenvecs = ortho_group.rvs(n_dims)
    true_cov = true_eigenvecs @ np.diag(true_eigenvals) @ true_eigenvecs.T
    
    # Generate data
    data = np.random.multivariate_normal(np.zeros(n_dims), true_cov, n_samples)
    
    # Empirical covariance
    emp_cov = np.cov(data.T)
    cond_cov = np.linalg.cond(emp_cov)
    
    print(f"True eigenvalues: {true_eigenvals}")
    print(f"Covariance condition: {cond_cov:.2e}")
    
    # Standard PCA (may be unstable)
    eigenvals_emp, eigenvecs_emp = np.linalg.eigh(emp_cov)
    eigenvals_emp = eigenvals_emp[::-1]  # Sort descending
    
    # SVD-based PCA (more stable)
    U, s, Vt = svd(data - np.mean(data, axis=0), full_matrices=False)
    eigenvals_svd = (s**2) / (n_samples - 1)
    
    print(f"Eigenvalues (cov):   {eigenvals_emp}")
    print(f"Eigenvalues (SVD):   {eigenvals_svd}")
    
    error_eigenvals = np.linalg.norm(eigenvals_emp - eigenvals_svd)
    print(f"Eigenvalue difference: {error_eigenvals:.2e}")

if __name__ == "__main__":
    demonstrate_singular_matrices()
    demonstrate_ill_conditioned_matrices()
    solution_strategies()
    machine_learning_examples()
    
    print(f"\n{'='*50}")
    print("SUMMARY: SINGULAR vs ILL-CONDITIONED")
    print(f"{'='*50}")
    
    print("\nSingular Matrices:")
    print("• Definition: det(A) = 0, not invertible")
    print("• Characteristics: rank(A) < n, has null space")
    print("• Detection: Zero determinant, zero eigenvalues")
    print("• Solution: Pseudo-inverse, regularization")
    
    print("\nIll-Conditioned Matrices:")
    print("• Definition: Large condition number κ(A)")
    print("• Characteristics: Small changes → large errors")
    print("• Detection: κ(A) >> 1, wide singular value range")
    print("• Solution: Regularization, stable algorithms")
    
    print("\nBest Practices:")
    print("• Monitor condition numbers in ML pipelines")
    print("• Use regularization for ill-conditioned problems")
    print("• Prefer QR/SVD over normal equations")
    print("• Check for multicollinearity in regression")
    print("• Use appropriate tolerances for rank determination")
    print("• Consider iterative methods for large sparse systems")
```

**Prevention and Best Practices:**

**1. Data Preprocessing:**
- **Standardization**: Scale features to similar ranges
- **Multicollinearity detection**: Remove highly correlated features
- **Dimensionality reduction**: Use PCA to remove redundant directions

**2. Algorithm Selection:**
- **Avoid normal equations** for ill-conditioned problems
- **Use QR decomposition** for stability
- **Apply regularization** (Ridge, Lasso) when appropriate
- **Consider iterative methods** for large sparse systems

**3. Numerical Monitoring:**
- **Check condition numbers** before computation
- **Monitor determinants** for singularity detection
- **Use appropriate tolerances** for rank determination
- **Implement error checking** in numerical routines

**4. Robust Alternatives:**
- **SVD**: Most robust for rank-deficient problems
- **Regularization**: Tikhonov, Ridge regression
- **Iterative solvers**: Conjugate gradient, GMRES
- **Preconditioning**: Improve conditioning for iterative methods

Understanding singular and ill-conditioned matrices is essential for developing robust machine learning algorithms that perform reliably across diverse datasets and problem domains.

---

## Question 22

**What is the Singular Value Decomposition (SVD) and its applications in machine learning?**

**Answer:** Singular Value Decomposition (SVD) is one of the most important and versatile matrix factorizations in linear algebra and machine learning. It provides a fundamental way to decompose any matrix into three component matrices that reveal the intrinsic geometric and algebraic structure of the linear transformation, making it invaluable for dimensionality reduction, data compression, noise reduction, and many other ML applications.

**Mathematical Definition:**

For any real matrix A ∈ ℝᵐˣⁿ, the SVD is:
```
A = UΣVᵀ
```
where:
- **U ∈ ℝᵐˣᵐ**: Left singular vectors (orthogonal matrix, UᵀU = I)
- **Σ ∈ ℝᵐˣⁿ**: Diagonal matrix of singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
- **V ∈ ℝⁿˣⁿ**: Right singular vectors (orthogonal matrix, VᵀV = I)

**Reduced/Thin SVD:**
For m > n (tall matrices), the economical form is:
```
A = ŨΣ̃Ṽᵀ
```
where Ũ ∈ ℝᵐˣⁿ, Σ̃ ∈ ℝⁿˣⁿ, Ṽ ∈ ℝⁿˣⁿ

**Geometric Interpretation:**

SVD decomposes any linear transformation into three conceptually simple operations:
1. **Vᵀ**: Rotation in input space
2. **Σ**: Scaling along principal axes
3. **U**: Rotation in output space

The singular values σᵢ represent the amount of "stretching" along each principal direction.

**Mathematical Properties:**

**1. Singular Values and Eigenvalues:**
- **AᵀA eigenvalues**: σᵢ² (with eigenvectors V)
- **AAᵀ eigenvalues**: σᵢ² (with eigenvectors U)
- **σᵢ**: Non-negative square roots of eigenvalues

**2. Rank and Matrix Properties:**
- **Rank**: rank(A) = number of non-zero singular values
- **Condition number**: κ(A) = σ₁/σᵣ (ratio of largest to smallest non-zero σ)
- **Frobenius norm**: ||A||²_F = Σσᵢ²
- **Spectral norm**: ||A||₂ = σ₁

**3. Fundamental Subspaces:**
- **Column space**: span(U) - spanned by left singular vectors
- **Row space**: span(V) - spanned by right singular vectors  
- **Null space**: span(V_{r+1:n}) - right singular vectors for zero σ
- **Left null space**: span(U_{r+1:m}) - left singular vectors for zero σ

**SVD Computation:**

**1. Direct Method via Eigendecomposition:**
```
AᵀA = VΣ²Vᵀ  →  V, σᵢ²
AAᵀ = UΣ²Uᵀ  →  U
```

**2. Bidiagonalization Approach:**
Most numerical libraries use a two-stage process:
- Reduce A to bidiagonal form using Householder reflections
- Apply iterative algorithms (QR, divide-and-conquer) to the bidiagonal matrix

**3. Randomized SVD:**
For large matrices, approximate SVD using random projections:
```
Ω = random matrix
Y = AΩ
Q = orth(Y)  # Orthonormalize
B = QᵀA
SVD(B) = ŨΣVᵀ
U = QŨ
```

**Applications in Machine Learning:**

**1. Principal Component Analysis (PCA):**

SVD is the most stable way to compute PCA:

```python
# Traditional PCA via covariance matrix
C = (X - μ).T @ (X - μ) / (n-1)
eigenvals, eigenvecs = eig(C)

# SVD-based PCA (more stable)
X_centered = X - np.mean(X, axis=0)
U, s, Vt = svd(X_centered, full_matrices=False)
principal_components = Vt.T
explained_variance = s**2 / (n-1)
```

**Benefits of SVD for PCA:**
- **Numerical stability**: No need to form XᵀX
- **Direct computation**: Gets principal components directly
- **Rank revelation**: Automatically handles rank deficiency

**2. Dimensionality Reduction:**

**Low-rank approximation** using truncated SVD:
```
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

This is the **best rank-k approximation** in both Frobenius and spectral norms (Eckart-Young theorem).

**3. Recommendation Systems:**

**Matrix Completion** for collaborative filtering:
```
R = UΣVᵀ  # User-item rating matrix
R_approx = U[:, :k] @ Σ[:k, :k] @ V[:, :k].T
```

**Advantages:**
- **Handles missing data**: SVD can work with incomplete matrices
- **Latent factors**: U and V represent user and item embeddings
- **Noise reduction**: Low-rank approximation filters noise

**4. Latent Semantic Analysis (LSA):**

**Document-term matrices** for text analysis:
```
A = UΣVᵀ  # Documents × Terms
U: Document representations in latent space
V: Term representations in latent space
Σ: Importance of each latent concept
```

**5. Image Compression:**

**Compress images** using low-rank approximation:
```python
# Each color channel
R_compressed = U_r[:, :k] @ np.diag(s_r[:k]) @ Vt_r[:k, :]
G_compressed = U_g[:, :k] @ np.diag(s_g[:k]) @ Vt_g[:k, :]
B_compressed = U_b[:, :k] @ np.diag(s_b[:k]) @ Vt_b[:k, :]
```

**Compression ratio**: (m×n) → k(m+n+1)

**6. Data Denoising:**

**Noise reduction** by truncating small singular values:
```python
# Remove noise by keeping only large singular values
threshold = 0.1 * s[0]  # Keep singular values > 10% of largest
k = np.sum(s > threshold)
A_denoised = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

**7. Feature Engineering:**

**Singular vectors as features**:
- **Left singular vectors**: New feature space for samples
- **Right singular vectors**: Feature transformations
- **Singular values**: Feature importance weights

**8. Solving Linear Systems:**

**Moore-Penrose pseudoinverse** via SVD:
```
A⁺ = VΣ⁺Uᵀ
where Σ⁺ᵢᵢ = 1/σᵢ if σᵢ > 0, else 0
```

**Applications:**
- **Least squares**: x = A⁺b
- **Overdetermined systems**: Minimum norm solution
- **Underdetermined systems**: Minimum norm solution

**Implementation Examples:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces, make_blobs
from sklearn.decomposition import PCA, TruncatedSVD
import seaborn as sns

def demonstrate_svd_basics():
    """Demonstrate basic SVD properties and computation"""
    
    print("="*60)
    print("SVD FUNDAMENTALS")
    print("="*60)
    
    # Create test matrix
    np.random.seed(42)
    A = np.array([[3, 2, 2],
                  [2, 3, -2]], dtype=float)
    
    print(f"Original matrix A:")
    print(A)
    print(f"Shape: {A.shape}")
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(A)
    
    print(f"\nSVD Components:")
    print(f"U (left singular vectors):")
    print(U)
    print(f"Shape: {U.shape}")
    
    print(f"\ns (singular values):")
    print(s)
    print(f"Shape: {s.shape}")
    
    print(f"\nVt (right singular vectors transposed):")
    print(Vt)
    print(f"Shape: {Vt.shape}")
    
    # Verify reconstruction
    Sigma = np.zeros_like(A)
    Sigma[:len(s), :len(s)] = np.diag(s)
    A_reconstructed = U @ Sigma @ Vt
    
    print(f"\nReconstruction A = UΣVᵀ:")
    print(A_reconstructed)
    
    reconstruction_error = np.linalg.norm(A - A_reconstructed)
    print(f"Reconstruction error: {reconstruction_error:.2e}")
    
    # Verify orthogonality
    print(f"\nOrthogonality checks:")
    print(f"UᵀU:")
    print(U.T @ U)
    print(f"VᵀV:")
    print(Vt @ Vt.T)
    
    # Matrix properties via SVD
    print(f"\nMatrix properties:")
    print(f"Rank: {np.sum(s > 1e-10)}")
    print(f"Condition number: {s[0]/s[-1]:.4f}")
    print(f"Frobenius norm: {np.sqrt(np.sum(s**2)):.4f}")
    print(f"Spectral norm: {s[0]:.4f}")

def svd_for_pca():
    """Demonstrate SVD-based PCA"""
    
    print(f"\n{'='*60}")
    print("SVD-BASED PCA")
    print(f"{'='*60}")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    
    # Create correlated 2D data
    angle = np.pi/4
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    
    # Generate data with different variances along axes
    data_std = np.random.randn(n_samples, 2) @ np.diag([3, 1])
    data = data_std @ rotation.T + np.array([2, 1])
    
    print(f"Data shape: {data.shape}")
    print(f"Data mean: {np.mean(data, axis=0)}")
    print(f"Data std: {np.std(data, axis=0)}")
    
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Method 1: Traditional PCA via covariance matrix
    cov_matrix = np.cov(data_centered.T)
    eigenvals_cov, eigenvecs_cov = np.linalg.eigh(cov_matrix)
    # Sort in descending order
    idx = np.argsort(eigenvals_cov)[::-1]
    eigenvals_cov = eigenvals_cov[idx]
    eigenvecs_cov = eigenvecs_cov[:, idx]
    
    # Method 2: SVD-based PCA
    U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
    eigenvals_svd = (s**2) / (n_samples - 1)
    eigenvecs_svd = Vt.T
    
    print(f"\nPCA Results Comparison:")
    print(f"Eigenvalues (covariance): {eigenvals_cov}")
    print(f"Eigenvalues (SVD):       {eigenvals_svd}")
    print(f"Difference: {np.linalg.norm(eigenvals_cov - eigenvals_svd):.2e}")
    
    print(f"\nEigenvectors (covariance):")
    print(eigenvecs_cov)
    print(f"Eigenvectors (SVD):")
    print(eigenvecs_svd)
    
    # Check if eigenvectors are the same (up to sign)
    eigenvec_diff = np.min([np.linalg.norm(eigenvecs_cov - eigenvecs_svd),
                           np.linalg.norm(eigenvecs_cov + eigenvecs_svd)])
    print(f"Eigenvector difference: {eigenvec_diff:.2e}")
    
    # Project data onto principal components
    data_projected_svd = data_centered @ eigenvecs_svd
    
    print(f"\nProjected data statistics:")
    print(f"PC1 variance: {np.var(data_projected_svd[:, 0]):.4f}")
    print(f"PC2 variance: {np.var(data_projected_svd[:, 1]):.4f}")
    print(f"Total variance preserved: {np.sum(eigenvals_svd):.4f}")
    
    # Explained variance ratio
    explained_variance_ratio = eigenvals_svd / np.sum(eigenvals_svd)
    print(f"Explained variance ratio: {explained_variance_ratio}")

def low_rank_approximation():
    """Demonstrate low-rank approximation using SVD"""
    
    print(f"\n{'='*60}")
    print("LOW-RANK APPROXIMATION")
    print(f"{'='*60}")
    
    # Create a low-rank matrix with noise
    np.random.seed(42)
    m, n = 50, 30
    rank_true = 5
    
    # Generate true low-rank matrix
    U_true = np.random.randn(m, rank_true)
    V_true = np.random.randn(n, rank_true)
    A_true = U_true @ V_true.T
    
    # Add noise
    noise_level = 0.1
    noise = noise_level * np.random.randn(m, n)
    A_noisy = A_true + noise
    
    print(f"Matrix dimensions: {m} × {n}")
    print(f"True rank: {rank_true}")
    print(f"Noise level: {noise_level}")
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(A_noisy, full_matrices=False)
    
    print(f"\nSingular values:")
    print(f"First 10: {s[:10]}")
    
    # Try different ranks for approximation
    ranks_to_try = [3, 5, 8, 10, 15]
    
    print(f"\n{'Rank':<6} {'Error':<12} {'Compression':<12}")
    print("-" * 35)
    
    for k in ranks_to_try:
        # Low-rank approximation
        A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        # Approximation error
        error = np.linalg.norm(A_noisy - A_k, 'fro')
        
        # Compression ratio
        original_size = m * n
        compressed_size = k * (m + n + 1)
        compression_ratio = compressed_size / original_size
        
        print(f"{k:<6} {error:<12.4f} {compression_ratio:<12.3f}")
    
    # Find optimal rank (elbow method)
    rank_errors = []
    for k in range(1, min(m, n)):
        A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        error = np.linalg.norm(A_noisy - A_k, 'fro')
        rank_errors.append(error)
    
    # Simple elbow detection (second derivative)
    second_deriv = np.diff(rank_errors, 2)
    optimal_rank = np.argmax(second_deriv) + 2  # +2 due to double differencing
    
    print(f"\nOptimal rank (elbow method): {optimal_rank}")
    print(f"True rank: {rank_true}")

def image_compression_demo():
    """Demonstrate image compression using SVD"""
    
    print(f"\n{'='*60}")
    print("IMAGE COMPRESSION WITH SVD")
    print(f"{'='*60}")
    
    # Load faces dataset
    try:
        faces = fetch_olivetti_faces()
        # Take one face image
        image = faces.images[0]  # 64x64 grayscale image
        
        print(f"Original image shape: {image.shape}")
        print(f"Original image size: {image.size} pixels")
        
        # Compute SVD
        U, s, Vt = np.linalg.svd(image, full_matrices=False)
        
        print(f"SVD shapes: U{U.shape}, s{s.shape}, Vt{Vt.shape}")
        
        # Compression analysis
        ranks = [1, 5, 10, 20, 30, 40, 50]
        
        print(f"\n{'Rank':<6} {'MSE':<12} {'Compression':<12} {'Size Ratio':<12}")
        print("-" * 50)
        
        for k in ranks:
            # Compressed image
            image_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            
            # Mean squared error
            mse = np.mean((image - image_k)**2)
            
            # Storage requirements
            original_storage = image.size
            compressed_storage = k * (image.shape[0] + image.shape[1] + 1)
            size_ratio = compressed_storage / original_storage
            compression_ratio = 1 / size_ratio
            
            print(f"{k:<6} {mse:<12.6f} {compression_ratio:<12.1f} {size_ratio:<12.3f}")
        
        print(f"\nSingular value decay:")
        print(f"σ₁ = {s[0]:.4f}")
        print(f"σ₁₀ = {s[9]:.4f}")
        print(f"σ₂₀ = {s[19]:.4f}")
        print(f"Ratio σ₁/σ₂₀ = {s[0]/s[19]:.2f}")
        
    except Exception as e:
        print(f"Could not load face data: {e}")
        print("Using synthetic image instead...")
        
        # Create synthetic image with structure
        x = np.linspace(-2, 2, 64)
        y = np.linspace(-2, 2, 64)
        X, Y = np.meshgrid(x, y)
        image = np.exp(-(X**2 + Y**2)) + 0.5 * np.exp(-((X-1)**2 + (Y-1)**2))
        
        # Add some noise
        image += 0.1 * np.random.randn(*image.shape)
        
        print(f"Synthetic image shape: {image.shape}")
        
        # Perform similar analysis...
        U, s, Vt = np.linalg.svd(image, full_matrices=False)
        
        # Show compression for a few ranks
        for k in [5, 10, 20]:
            image_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            mse = np.mean((image - image_k)**2)
            print(f"Rank {k}: MSE = {mse:.6f}")

def recommendation_system_demo():
    """Demonstrate collaborative filtering using SVD"""
    
    print(f"\n{'='*60}")
    print("COLLABORATIVE FILTERING WITH SVD")
    print(f"{'='*60}")
    
    # Create synthetic user-item rating matrix
    np.random.seed(42)
    n_users, n_items = 20, 15
    n_factors = 5  # True number of latent factors
    
    # Generate user and item factors
    user_factors = np.random.randn(n_users, n_factors)
    item_factors = np.random.randn(n_items, n_factors)
    
    # True rating matrix (full)
    ratings_true = user_factors @ item_factors.T
    
    # Normalize to typical rating scale (1-5)
    ratings_true = 3 + 2 * ratings_true / np.std(ratings_true)
    ratings_true = np.clip(ratings_true, 1, 5)
    
    # Create sparse observations (only some ratings observed)
    observation_rate = 0.3
    mask = np.random.random((n_users, n_items)) < observation_rate
    ratings_observed = np.where(mask, ratings_true, 0)
    
    print(f"Rating matrix shape: {n_users} users × {n_items} items")
    print(f"True factors: {n_factors}")
    print(f"Observation rate: {observation_rate:.1%}")
    print(f"Observed ratings: {np.sum(mask)} / {n_users * n_items}")
    
    # Simple SVD approach (treating missing as zeros - not optimal but illustrative)
    print(f"\nNaive SVD approach (treating missing as 0):")
    U, s, Vt = np.linalg.svd(ratings_observed, full_matrices=False)
    
    # Try different numbers of factors
    factors_to_try = [3, 5, 8, 10]
    
    print(f"\n{'Factors':<8} {'RMSE (observed)':<15} {'RMSE (missing)':<15}")
    print("-" * 45)
    
    for k in factors_to_try:
        # Reconstruct with k factors
        ratings_pred = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        # RMSE on observed ratings
        rmse_observed = np.sqrt(np.mean((ratings_observed[mask] - ratings_pred[mask])**2))
        
        # RMSE on missing ratings (true test)
        rmse_missing = np.sqrt(np.mean((ratings_true[~mask] - ratings_pred[~mask])**2))
        
        print(f"{k:<8} {rmse_observed:<15.4f} {rmse_missing:<15.4f}")
    
    print(f"\nNote: This is a simplified approach. Real collaborative filtering")
    print(f"uses matrix completion techniques that properly handle missing values.")

def noise_reduction_demo():
    """Demonstrate noise reduction using SVD"""
    
    print(f"\n{'='*60}")
    print("NOISE REDUCTION WITH SVD")
    print(f"{'='*60}")
    
    # Create clean signal (low-rank structure)
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 100)
    
    # Clean signal: combination of sinusoids
    signal_clean = (2 * np.sin(t) + 
                   1.5 * np.sin(2*t + np.pi/4) + 
                   0.8 * np.sin(0.5*t))
    
    # Create matrix by time-delaying (Hankel matrix)
    n_delays = 20
    signal_matrix_clean = np.array([signal_clean[i:i+len(t)-n_delays+1] 
                                   for i in range(n_delays)])
    
    # Add noise
    noise_level = 0.5
    noise = noise_level * np.random.randn(*signal_matrix_clean.shape)
    signal_matrix_noisy = signal_matrix_clean + noise
    
    print(f"Signal length: {len(t)}")
    print(f"Matrix shape: {signal_matrix_noisy.shape}")
    print(f"SNR: {20 * np.log10(np.std(signal_clean) / noise_level):.1f} dB")
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(signal_matrix_noisy, full_matrices=False)
    
    print(f"\nSingular values (first 10):")
    print(s[:10])
    
    # Noise reduction by truncating small singular values
    print(f"\n{'Rank':<6} {'Denoising SNR (dB)':<20}")
    print("-" * 30)
    
    for k in [2, 4, 6, 8, 10]:
        # Reconstruct with k components
        signal_matrix_denoised = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        # Extract denoised signal (first row)
        signal_denoised = signal_matrix_denoised[0, :]
        
        # Compute SNR improvement
        noise_remaining = signal_denoised - signal_clean[:len(signal_denoised)]
        snr_denoised = 20 * np.log10(np.std(signal_clean) / np.std(noise_remaining))
        
        print(f"{k:<6} {snr_denoised:<20.1f}")
    
    # Find optimal rank using noise floor estimation
    # Typically, noise adds equal power to all singular values
    # So we look for the "elbow" where singular values start decreasing rapidly
    
    # Simple threshold: keep singular values > threshold * max(singular_value)
    threshold = 0.1
    k_optimal = np.sum(s > threshold * s[0])
    
    print(f"\nOptimal rank (threshold method): {k_optimal}")
    print(f"Threshold: {threshold * s[0]:.4f}")

def advanced_svd_applications():
    """Demonstrate advanced SVD applications"""
    
    print(f"\n{'='*60}")
    print("ADVANCED SVD APPLICATIONS")
    print(f"{'='*60}")
    
    # Application 1: Pseudoinverse computation
    print("\n1. PSEUDOINVERSE COMPUTATION:")
    
    # Create overdetermined system
    np.random.seed(42)
    m, n = 8, 5
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    
    # Add some rank deficiency
    A[:, -1] = A[:, 0] + A[:, 1] + 0.01 * np.random.randn(m)
    
    print(f"System: {m} equations, {n} unknowns")
    print(f"Matrix rank: {np.linalg.matrix_rank(A)}")
    
    # Pseudoinverse via SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tolerance = 1e-10
    s_inv = np.where(s > tolerance, 1/s, 0)
    A_pinv = Vt.T @ np.diag(s_inv) @ U.T
    
    # Solve least squares
    x_svd = A_pinv @ b
    
    # Compare with numpy
    x_numpy = np.linalg.pinv(A) @ b
    
    print(f"Solution (SVD):   {x_svd}")
    print(f"Solution (NumPy): {x_numpy}")
    print(f"Difference: {np.linalg.norm(x_svd - x_numpy):.2e}")
    
    # Residual
    residual = np.linalg.norm(A @ x_svd - b)
    print(f"Residual: {residual:.6f}")
    
    # Application 2: Data compression analysis
    print(f"\n2. DATA COMPRESSION ANALYSIS:")
    
    # Generate structured data
    n_samples, n_features = 200, 50
    
    # Create data with intrinsic low-dimensional structure
    intrinsic_dim = 5
    latent_data = np.random.randn(n_samples, intrinsic_dim)
    
    # Random projection to high dimension
    projection_matrix = np.random.randn(intrinsic_dim, n_features)
    data_structured = latent_data @ projection_matrix
    
    # Add some noise
    data_noisy = data_structured + 0.1 * np.random.randn(n_samples, n_features)
    
    print(f"Data shape: {data_noisy.shape}")
    print(f"True intrinsic dimension: {intrinsic_dim}")
    
    # SVD analysis
    U, s, Vt = np.linalg.svd(data_noisy, full_matrices=False)
    
    # Compute cumulative explained variance
    explained_variance = s**2 / np.sum(s**2)
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"\nExplained variance by component:")
    for i in range(min(10, len(s))):
        print(f"PC{i+1}: {explained_variance[i]:.3f} (cumsum: {cumulative_variance[i]:.3f})")
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nComponents for 95% variance: {n_components_95}")
    print(f"Compression ratio: {n_components_95 / n_features:.3f}")

if __name__ == "__main__":
    demonstrate_svd_basics()
    svd_for_pca()
    low_rank_approximation()
    image_compression_demo()
    recommendation_system_demo()
    noise_reduction_demo()
    advanced_svd_applications()
    
    print(f"\n{'='*60}")
    print("SVD SUMMARY")
    print(f"{'='*60}")
    
    print("\nKey Properties:")
    print("• Universal: Every matrix has an SVD")
    print("• Optimal: Best low-rank approximations (Eckart-Young)")
    print("• Stable: Numerically robust computation")
    print("• Geometric: Clear interpretation via rotations and scaling")
    
    print("\nMachine Learning Applications:")
    print("• PCA: Most stable method for principal components")
    print("• Dimensionality Reduction: Optimal low-rank projections")
    print("• Recommendation Systems: Matrix factorization/completion")
    print("• Image/Signal Processing: Compression and denoising")
    print("• Feature Engineering: Latent factor discovery")
    print("• Pseudoinverse: Solving over/underdetermined systems")
    
    print("\nComputational Considerations:")
    print("• Full SVD: O(min(mn², m²n)) for m×n matrix")
    print("• Truncated SVD: O(k·min(m,n)²) for k components")
    print("• Randomized SVD: O(k²(m+n)) for approximate k-SVD")
    print("• Memory: Can use iterative methods for very large matrices")
    
    print("\nBest Practices:")
    print("• Use SVD instead of eigendecomposition for PCA")
    print("• Choose rank based on explained variance or cross-validation")
    print("• Consider randomized SVD for large matrices")
    print("• Monitor singular value decay for rank selection")
    print("• Use appropriate tolerance for pseudoinverse computation")
```

**Advanced SVD Variants:**

**1. Truncated SVD:**
Only compute the k largest singular values and vectors:
```
A ≈ U_k Σ_k V_k^T
```
Useful for large matrices where full SVD is computationally expensive.

**2. Randomized SVD:**
Uses random projections for approximate SVD:
- **Faster**: O(k²(m+n)) vs O(mn²) for full SVD  
- **Scalable**: Works with matrices that don't fit in memory
- **Accurate**: High probability bounds on approximation error

**3. Incremental SVD:**
Updates SVD when new data arrives:
- **Online learning**: Process streaming data
- **Memory efficient**: No need to store entire dataset
- **Applications**: Real-time recommendation systems

**Theoretical Foundations:**

**1. Eckart-Young Theorem:**
SVD provides the best rank-k approximation in both Frobenius and spectral norms.

**2. Schmidt Decomposition:**
SVD is the generalization of Schmidt decomposition from quantum mechanics.

**3. Fundamental Theorem of Linear Algebra:**
SVD reveals all four fundamental subspaces of a matrix.

**Best Practices:**
- Use SVD for numerical stability in PCA and least squares
- Choose rank based on explained variance ratio or cross-validation
- Consider computational complexity: use randomized methods for large matrices
- Monitor condition number (σ₁/σᵣ) for numerical stability
- Apply appropriate regularization for noisy data
- Use truncated SVD for dimensionality reduction in high-dimensional spaces

---

## Question 23

**Explain the concept of matrix factorization.**

**Answer:** Matrix factorization is a fundamental technique in linear algebra that decomposes a matrix into a product of two or more simpler matrices. This decomposition reveals underlying structure in data, enables efficient computation, and provides insights into the mathematical properties of linear transformations. Matrix factorization forms the backbone of many machine learning algorithms and numerical methods.

**General Concept:**

Matrix factorization seeks to express a matrix A as a product:
```
A = BC  or  A = BCD  or  A = B₁B₂...Bₖ
```
where B, C, D, etc., are matrices with specific desired properties such as:
- **Orthogonality**: Columns/rows are orthonormal
- **Triangular structure**: Upper or lower triangular  
- **Diagonal form**: Diagonal or block diagonal
- **Sparsity**: Many zero elements
- **Low rank**: Fewer columns/rows than original

**Why Matrix Factorization?**

**1. Computational Efficiency:**
- **Solve systems**: Ax = b becomes solving simpler systems
- **Matrix operations**: A⁻¹, det(A), Aᵏ computed via factors
- **Storage**: Often requires less memory

**2. Numerical Stability:**
- **Avoid problematic operations**: Direct inversion, computing AᵀA
- **Stable algorithms**: QR, SVD are backward stable
- **Conditioning**: Better control over numerical errors

**3. Structural Insights:**
- **Rank**: Reveals effective dimensionality
- **Null spaces**: Finds dependencies and solutions
- **Eigenstructure**: Understanding of matrix behavior

**4. Machine Learning Applications:**
- **Dimensionality reduction**: Principal components, latent factors
- **Collaborative filtering**: User-item interactions
- **Feature extraction**: Hidden representations

**Major Types of Matrix Factorizations:**

**1. LU Decomposition:**
```
A = LU  (or PA = LU)
```
- **L**: Lower triangular with unit diagonal
- **U**: Upper triangular
- **Applications**: Solving linear systems, matrix inversion
- **Use case**: General square systems

**2. QR Decomposition:**
```
A = QR
```
- **Q**: Orthogonal matrix (QᵀQ = I)
- **R**: Upper triangular matrix
- **Applications**: Least squares, eigenvalue computation
- **Use case**: Overdetermined systems, numerical stability

**3. Singular Value Decomposition (SVD):**
```
A = UΣVᵀ
```
- **U, V**: Orthogonal matrices
- **Σ**: Diagonal matrix of singular values
- **Applications**: PCA, dimensionality reduction, pseudoinverse
- **Use case**: Any matrix, optimal low-rank approximation

**4. Eigenvalue Decomposition:**
```
A = QΛQᵀ  (for symmetric A)
A = QΛQ⁻¹  (for general A)
```
- **Q**: Matrix of eigenvectors
- **Λ**: Diagonal matrix of eigenvalues
- **Applications**: PCA, spectral methods, dynamical systems
- **Use case**: Square matrices, especially symmetric

**5. Cholesky Decomposition:**
```
A = LLᵀ
```
- **L**: Lower triangular matrix
- **Applications**: Solving positive definite systems, simulation
- **Use case**: Symmetric positive definite matrices

**6. Schur Decomposition:**
```
A = QTQᵀ
```
- **Q**: Orthogonal matrix
- **T**: Upper triangular (quasi-triangular for real matrices)
- **Applications**: Eigenvalue computation, matrix functions
- **Use case**: Any square matrix

**Machine Learning Specific Factorizations:**

**1. Non-negative Matrix Factorization (NMF):**
```
A ≈ WH  (where A, W, H ≥ 0)
```
- **Constraint**: All elements non-negative
- **Applications**: Topic modeling, image processing, parts-based learning
- **Interpretation**: Additive parts decomposition

**2. Principal Component Analysis (PCA):**
```
X = UΣVᵀ  →  X ≈ UₖΣₖVₖᵀ
```
- **Goal**: Maximum variance projection
- **Applications**: Dimensionality reduction, visualization
- **Interpretation**: Principal directions of data variation

**3. Independent Component Analysis (ICA):**
```
X = AS
```
- **S**: Independent source signals
- **A**: Mixing matrix
- **Applications**: Blind source separation, feature extraction
- **Goal**: Statistical independence of components

**4. Matrix Completion:**
```
A ≈ UVᵀ  (with missing entries in A)
```
- **Goal**: Fill missing values assuming low rank
- **Applications**: Recommendation systems, image inpainting
- **Methods**: Nuclear norm minimization, alternating least squares

**5. Tensor Factorization:**
```
𝒯 ≈ ∑ᵣ λᵣ uᵣ ⊗ vᵣ ⊗ wᵣ  (CP decomposition)
𝒯 ≈ ∑ᵣ Gᵣ ×₁ Uᵣ ×₂ Vᵣ ×₃ Wᵣ  (Tucker decomposition)
```
- **Applications**: Multi-way data analysis, recommendation systems
- **Challenges**: Non-uniqueness, computational complexity

**Implementation Examples:**

```python
import numpy as np
from scipy.linalg import lu, qr, svd, eigh, cholesky, schur
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.datasets import fetch_olivetti_faces, make_blobs
import matplotlib.pyplot as plt
import time

def demonstrate_basic_factorizations():
    """Demonstrate basic matrix factorizations"""
    
    print("="*60)
    print("BASIC MATRIX FACTORIZATIONS")
    print("="*60)
    
    # Create test matrix
    np.random.seed(42)
    A = np.array([[4, 2, 1],
                  [2, 5, 3],
                  [1, 3, 6]], dtype=float)
    
    print(f"Original matrix A:")
    print(A)
    print(f"Properties: {A.shape}, symmetric: {np.allclose(A, A.T)}")
    
    # 1. LU Decomposition
    print(f"\n1. LU DECOMPOSITION:")
    P, L, U = lu(A)
    print(f"P (permutation):\n{P}")
    print(f"L (lower triangular):\n{L}")
    print(f"U (upper triangular):\n{U}")
    
    # Verify: PA = LU
    reconstruction_lu = L @ U
    error_lu = np.linalg.norm(P @ A - reconstruction_lu)
    print(f"Reconstruction error ||PA - LU||: {error_lu:.2e}")
    
    # 2. QR Decomposition
    print(f"\n2. QR DECOMPOSITION:")
    Q, R = qr(A)
    print(f"Q (orthogonal):\n{Q}")
    print(f"R (upper triangular):\n{R}")
    
    # Verify: A = QR
    reconstruction_qr = Q @ R
    error_qr = np.linalg.norm(A - reconstruction_qr)
    print(f"Reconstruction error ||A - QR||: {error_qr:.2e}")
    print(f"Orthogonality check ||Q^T Q - I||: {np.linalg.norm(Q.T @ Q - np.eye(3)):.2e}")
    
    # 3. Eigenvalue Decomposition (for symmetric matrix)
    print(f"\n3. EIGENVALUE DECOMPOSITION:")
    eigenvals, eigenvecs = eigh(A)  # eigh for symmetric matrices
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    # Verify: A = Q Λ Q^T
    Lambda = np.diag(eigenvals)
    reconstruction_eigen = eigenvecs @ Lambda @ eigenvecs.T
    error_eigen = np.linalg.norm(A - reconstruction_eigen)
    print(f"Reconstruction error ||A - QΛQ^T||: {error_eigen:.2e}")
    
    # 4. SVD
    print(f"\n4. SINGULAR VALUE DECOMPOSITION:")
    U_svd, s, Vt = svd(A)
    print(f"U (left singular vectors):\n{U_svd}")
    print(f"Singular values: {s}")
    print(f"V^T (right singular vectors):\n{Vt}")
    
    # Verify: A = U Σ V^T
    Sigma = np.diag(s)
    reconstruction_svd = U_svd @ Sigma @ Vt
    error_svd = np.linalg.norm(A - reconstruction_svd)
    print(f"Reconstruction error ||A - UΣV^T||: {error_svd:.2e}")
    
    # 5. Cholesky Decomposition (since A is positive definite)
    print(f"\n5. CHOLESKY DECOMPOSITION:")
    try:
        L_chol = cholesky(A, lower=True)
        print(f"L (lower triangular):\n{L_chol}")
        
        # Verify: A = L L^T
        reconstruction_chol = L_chol @ L_chol.T
        error_chol = np.linalg.norm(A - reconstruction_chol)
        print(f"Reconstruction error ||A - LL^T||: {error_chol:.2e}")
        
    except np.linalg.LinAlgError:
        print("Matrix is not positive definite - Cholesky decomposition failed")

def compare_computational_efficiency():
    """Compare computational efficiency of different factorizations"""
    
    print(f"\n{'='*60}")
    print("COMPUTATIONAL EFFICIENCY COMPARISON")
    print(f"{'='*60}")
    
    sizes = [100, 200, 500]
    
    print(f"{'Size':<8} {'LU':<10} {'QR':<10} {'SVD':<10} {'Eigen':<10} {'Cholesky':<10}")
    print("-" * 65)
    
    for n in sizes:
        # Create positive definite matrix for fair comparison
        np.random.seed(42)
        A_rand = np.random.randn(n, n)
        A = A_rand @ A_rand.T + n * np.eye(n)  # Ensure positive definite
        
        times = {}
        
        # LU decomposition
        start = time.time()
        P, L, U = lu(A)
        times['LU'] = time.time() - start
        
        # QR decomposition
        start = time.time()
        Q, R = qr(A)
        times['QR'] = time.time() - start
        
        # SVD
        start = time.time()
        U_svd, s, Vt = svd(A)
        times['SVD'] = time.time() - start
        
        # Eigenvalue decomposition
        start = time.time()
        eigenvals, eigenvecs = eigh(A)
        times['Eigen'] = time.time() - start
        
        # Cholesky decomposition
        start = time.time()
        L_chol = cholesky(A, lower=True)
        times['Cholesky'] = time.time() - start
        
        print(f"{n:<8} {times['LU']:<10.3f} {times['QR']:<10.3f} {times['SVD']:<10.3f} "
              f"{times['Eigen']:<10.3f} {times['Cholesky']:<10.3f}")

def matrix_factorization_for_ml():
    """Demonstrate matrix factorization applications in ML"""
    
    print(f"\n{'='*60}")
    print("MATRIX FACTORIZATION IN MACHINE LEARNING")
    print(f"{'='*60}")
    
    # Application 1: PCA for dimensionality reduction
    print("\n1. PCA FOR DIMENSIONALITY REDUCTION:")
    
    # Generate high-dimensional data with intrinsic low-dimensional structure
    np.random.seed(42)
    n_samples, n_features = 300, 50
    n_components_true = 5
    
    # Create latent factors
    latent_factors = np.random.randn(n_samples, n_components_true)
    # Random projection to high dimensions
    projection = np.random.randn(n_components_true, n_features)
    X = latent_factors @ projection + 0.1 * np.random.randn(n_samples, n_features)
    
    print(f"Data shape: {X.shape}")
    print(f"True intrinsic dimension: {n_components_true}")
    
    # PCA via SVD
    X_centered = X - np.mean(X, axis=0)
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    # Explained variance
    explained_variance_ratio = (s**2 / np.sum(s**2))
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"Explained variance by first 10 components:")
    for i in range(min(10, len(s))):
        print(f"PC{i+1}: {explained_variance_ratio[i]:.4f} (cumsum: {cumulative_variance[i]:.4f})")
    
    # Find components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Components for 95% variance: {n_components_95}")
    
    # Application 2: Non-negative Matrix Factorization
    print(f"\n2. NON-NEGATIVE MATRIX FACTORIZATION:")
    
    # Generate non-negative data (e.g., word counts, pixel intensities)
    np.random.seed(42)
    n_samples, n_features = 100, 20
    n_components = 5
    
    # True non-negative factors
    W_true = np.abs(np.random.randn(n_samples, n_components))
    H_true = np.abs(np.random.randn(n_components, n_features))
    X_nonneg = W_true @ H_true + 0.1 * np.abs(np.random.randn(n_samples, n_features))
    
    print(f"Non-negative data shape: {X_nonneg.shape}")
    print(f"Data range: [{X_nonneg.min():.3f}, {X_nonneg.max():.3f}]")
    
    # Apply NMF
    nmf = NMF(n_components=n_components, random_state=42, max_iter=200)
    W_nmf = nmf.fit_transform(X_nonneg)
    H_nmf = nmf.components_
    
    # Reconstruction
    X_reconstructed = W_nmf @ H_nmf
    reconstruction_error = np.linalg.norm(X_nonneg - X_reconstructed, 'fro')
    
    print(f"NMF reconstruction error: {reconstruction_error:.4f}")
    print(f"W shape: {W_nmf.shape}, H shape: {H_nmf.shape}")
    
    # Compare with SVD (which allows negative values)
    U_svd, s_svd, Vt_svd = svd(X_nonneg, full_matrices=False)
    X_svd_recon = U_svd[:, :n_components] @ np.diag(s_svd[:n_components]) @ Vt_svd[:n_components, :]
    svd_error = np.linalg.norm(X_nonneg - X_svd_recon, 'fro')
    
    print(f"SVD reconstruction error: {svd_error:.4f}")
    print(f"NMF enforces non-negativity, SVD optimizes reconstruction")

def collaborative_filtering_demo():
    """Demonstrate matrix factorization for collaborative filtering"""
    
    print(f"\n{'='*60}")
    print("COLLABORATIVE FILTERING WITH MATRIX FACTORIZATION")
    print(f"{'='*60}")
    
    # Create synthetic user-item rating matrix
    np.random.seed(42)
    n_users, n_items = 30, 20
    n_latent_factors = 8
    
    # Generate user and item latent factors
    user_factors = 2 * np.random.randn(n_users, n_latent_factors)
    item_factors = 2 * np.random.randn(n_latent_factors, n_items)
    
    # Generate ratings (with some user/item biases)
    user_bias = np.random.randn(n_users, 1)
    item_bias = np.random.randn(1, n_items)
    global_mean = 3.5
    
    ratings_full = (global_mean + user_bias + item_bias + 
                   user_factors @ item_factors + 
                   0.2 * np.random.randn(n_users, n_items))
    
    # Clip to valid rating range
    ratings_full = np.clip(ratings_full, 1, 5)
    
    # Create sparse observation mask (only 30% of ratings observed)
    observation_rate = 0.3
    mask = np.random.random((n_users, n_items)) < observation_rate
    ratings_observed = np.where(mask, ratings_full, 0)
    
    print(f"Rating matrix: {n_users} users × {n_items} items")
    print(f"Observation rate: {observation_rate:.1%}")
    print(f"Observed ratings: {np.sum(mask)} / {n_users * n_items}")
    
    # Simple matrix factorization approach (treating missing as zeros - simplified)
    print(f"\nMatrix factorization approaches:")
    
    # Method 1: SVD on observed ratings (naive approach)
    U, s, Vt = svd(ratings_observed, full_matrices=False)
    
    factors_to_try = [4, 8, 12, 16]
    
    print(f"\n{'Factors':<8} {'RMSE (observed)':<15} {'RMSE (missing)':<15}")
    print("-" * 45)
    
    for k in factors_to_try:
        # Reconstruct with k factors
        ratings_pred = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        # RMSE on observed ratings
        if np.sum(mask) > 0:
            rmse_observed = np.sqrt(np.mean((ratings_observed[mask] - ratings_pred[mask])**2))
        else:
            rmse_observed = float('inf')
        
        # RMSE on missing ratings (true test of generalization)
        rmse_missing = np.sqrt(np.mean((ratings_full[~mask] - ratings_pred[~mask])**2))
        
        print(f"{k:<8} {rmse_observed:<15.4f} {rmse_missing:<15.4f}")
    
    print(f"\nNote: Real collaborative filtering uses matrix completion")
    print(f"algorithms that properly handle missing values, not SVD on sparse matrices.")

def advanced_factorization_techniques():
    """Demonstrate advanced factorization techniques"""
    
    print(f"\n{'='*60}")
    print("ADVANCED FACTORIZATION TECHNIQUES")
    print(f"{'='*60}")
    
    # 1. Regularized Matrix Factorization
    print("\n1. REGULARIZED MATRIX FACTORIZATION:")
    
    def matrix_factorization_als(R, mask, k=10, lambda_reg=0.01, n_iter=50):
        """
        Alternating Least Squares for Matrix Factorization
        
        Args:
            R: Rating matrix
            mask: Observed entries mask
            k: Number of latent factors
            lambda_reg: Regularization parameter
            n_iter: Number of iterations
        """
        m, n = R.shape
        
        # Initialize factors
        U = np.random.randn(m, k) * 0.1
        V = np.random.randn(n, k) * 0.1
        
        errors = []
        
        for iteration in range(n_iter):
            # Update U (fix V)
            for i in range(m):
                observed_items = mask[i, :]
                if np.sum(observed_items) > 0:
                    V_i = V[observed_items, :]
                    R_i = R[i, observed_items]
                    
                    # Solve: (V_i^T V_i + λI) u_i = V_i^T r_i
                    A = V_i.T @ V_i + lambda_reg * np.eye(k)
                    b = V_i.T @ R_i
                    U[i, :] = np.linalg.solve(A, b)
            
            # Update V (fix U)
            for j in range(n):
                observed_users = mask[:, j]
                if np.sum(observed_users) > 0:
                    U_j = U[observed_users, :]
                    R_j = R[observed_users, j]
                    
                    # Solve: (U_j^T U_j + λI) v_j = U_j^T r_j
                    A = U_j.T @ U_j + lambda_reg * np.eye(k)
                    b = U_j.T @ R_j
                    V[j, :] = np.linalg.solve(A, b)
            
            # Compute error on observed entries
            R_pred = U @ V.T
            error = np.sqrt(np.mean((R[mask] - R_pred[mask])**2))
            errors.append(error)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: RMSE = {error:.4f}")
        
        return U, V, errors
    
    # Generate test data
    np.random.seed(42)
    m, n, k_true = 25, 20, 5
    U_true = np.random.randn(m, k_true)
    V_true = np.random.randn(n, k_true)
    R_true = U_true @ V_true.T + 0.1 * np.random.randn(m, n)
    
    # Create observation mask
    mask = np.random.random((m, n)) < 0.4
    R_observed = np.where(mask, R_true, 0)
    
    print(f"Matrix size: {m} × {n}")
    print(f"True rank: {k_true}")
    print(f"Observation rate: {np.mean(mask):.1%}")
    
    # Apply regularized matrix factorization
    U_est, V_est, training_errors = matrix_factorization_als(
        R_observed, mask, k=k_true, lambda_reg=0.01, n_iter=50
    )
    
    # Evaluate
    R_pred = U_est @ V_est.T
    rmse_observed = np.sqrt(np.mean((R_observed[mask] - R_pred[mask])**2))
    rmse_missing = np.sqrt(np.mean((R_true[~mask] - R_pred[~mask])**2))
    
    print(f"Final RMSE (observed): {rmse_observed:.4f}")
    print(f"Final RMSE (missing): {rmse_missing:.4f}")
    
    # 2. Independent Component Analysis
    print(f"\n2. INDEPENDENT COMPONENT ANALYSIS (ICA):")
    
    # Generate mixed signals
    np.random.seed(42)
    n_samples = 1000
    time = np.linspace(0, 8, n_samples)
    
    # Original source signals
    s1 = np.sin(2 * time)  # Sinusoidal
    s2 = np.sign(np.sin(3 * time))  # Square wave
    s3 = np.random.laplace(size=n_samples)  # Laplacian noise
    
    S = np.c_[s1, s2, s3]
    S += 0.1 * np.random.randn(n_samples, 3)  # Add some noise
    S = S.T  # Shape: (n_components, n_samples)
    
    # Create mixing matrix
    A = np.array([[1, 1, 1],
                  [0.5, 2, 1.0],
                  [1.5, 1.0, 2.0]])
    
    # Mixed signals
    X = A @ S
    
    print(f"Source signals shape: {S.shape}")
    print(f"Mixed signals shape: {X.shape}")
    print(f"Mixing matrix A:")
    print(A)
    
    # Apply ICA
    ica = FastICA(n_components=3, random_state=42, max_iter=200)
    S_estimated = ica.fit_transform(X.T).T  # ICA expects (n_samples, n_features)
    A_estimated = ica.mixing_
    
    print(f"Estimated mixing matrix:")
    print(A_estimated)
    
    # Measure reconstruction quality
    X_reconstructed = A_estimated @ S_estimated
    reconstruction_error = np.linalg.norm(X - X_reconstructed, 'fro') / np.linalg.norm(X, 'fro')
    print(f"Reconstruction error (relative): {reconstruction_error:.4f}")
    
    # Note: ICA solutions are unique up to permutation and scaling
    print(f"Note: ICA components may be permuted and scaled versions of originals")

def factorization_selection_guide():
    """Guide for selecting appropriate matrix factorization"""
    
    print(f"\n{'='*60}")
    print("MATRIX FACTORIZATION SELECTION GUIDE")
    print(f"{'='*60}")
    
    print("\nFactorization Selection Criteria:")
    print(f"{'='*40}")
    
    selection_guide = {
        "LU Decomposition": {
            "Use when": "Solving multiple linear systems Ax = b with same A",
            "Matrix type": "Square, non-singular",
            "Advantages": "Fast for multiple solves, exact factorization",
            "Disadvantages": "May be unstable without pivoting",
            "Applications": "Linear systems, matrix inversion, determinants"
        },
        
        "QR Decomposition": {
            "Use when": "Solving least squares problems, need numerical stability",
            "Matrix type": "Any matrix (typically m ≥ n)",
            "Advantages": "Numerically stable, handles rank deficiency",
            "Disadvantages": "More expensive than LU for square systems",
            "Applications": "Least squares, eigenvalue algorithms, orthogonalization"
        },
        
        "SVD": {
            "Use when": "Need best low-rank approximation, dealing with any matrix",
            "Matrix type": "Any matrix (rectangular, singular, etc.)",
            "Advantages": "Most general, optimal approximations, reveals all subspaces",
            "Disadvantages": "Most expensive computationally",
            "Applications": "PCA, dimensionality reduction, pseudoinverse, matrix completion"
        },
        
        "Eigendecomposition": {
            "Use when": "Analyzing matrix powers, spectral properties",
            "Matrix type": "Square matrices (especially symmetric)",
            "Advantages": "Direct access to eigenvalues/eigenvectors",
            "Disadvantages": "Limited to square matrices, may not exist for all matrices",
            "Applications": "PCA (via covariance), spectral clustering, dynamical systems"
        },
        
        "Cholesky": {
            "Use when": "Matrix is symmetric positive definite",
            "Matrix type": "Symmetric positive definite",
            "Advantages": "Most efficient for positive definite matrices",
            "Disadvantages": "Very restrictive matrix requirements",
            "Applications": "Solving normal equations, generating random samples"
        },
        
        "NMF": {
            "Use when": "Data and factors should be non-negative",
            "Matrix type": "Non-negative matrices",
            "Advantages": "Interpretable parts-based decomposition",
            "Disadvantages": "Restricted to non-negative data, non-convex optimization",
            "Applications": "Topic modeling, image processing, spectral unmixing"
        }
    }
    
    for method, properties in selection_guide.items():
        print(f"\n{method}:")
        print(f"  Use when: {properties['Use when']}")
        print(f"  Matrix type: {properties['Matrix type']}")
        print(f"  Advantages: {properties['Advantages']}")
        print(f"  Disadvantages: {properties['Disadvantages']}")
        print(f"  Applications: {properties['Applications']}")

if __name__ == "__main__":
    demonstrate_basic_factorizations()
    compare_computational_efficiency()
    matrix_factorization_for_ml()
    collaborative_filtering_demo()
    advanced_factorization_techniques()
    factorization_selection_guide()
    
    print(f"\n{'='*60}")
    print("MATRIX FACTORIZATION SUMMARY")
    print(f"{'='*60}")
    
    print("\nCore Principles:")
    print("• Decompose complex matrices into simpler, structured components")
    print("• Enable efficient computation and reveal mathematical structure")
    print("• Provide numerical stability and computational advantages")
    print("• Form foundation for many machine learning algorithms")
    
    print("\nSelection Strategy:")
    print("• Consider matrix properties: shape, symmetry, definiteness, sparsity")
    print("• Identify primary goal: solving systems, approximation, analysis")
    print("• Evaluate computational constraints: time, memory, accuracy")
    print("• Choose based on stability requirements and numerical considerations")
    
    print("\nMachine Learning Applications:")
    print("• Dimensionality Reduction: PCA via SVD, eigendecomposition")
    print("• Collaborative Filtering: Matrix completion via factorization")
    print("• Feature Learning: NMF for parts-based representations")
    print("• Signal Processing: ICA for source separation")
    print("• Optimization: Factorizations for efficient gradient computation")
    
    print("\nBest Practices:")
    print("• Use SVD for maximum generality and optimal approximations")
    print("• Choose QR for numerical stability in least squares")
    print("• Apply specialized factorizations when structure permits (Cholesky, etc.)")
    print("• Consider computational trade-offs: accuracy vs. speed vs. memory")
    print("• Validate factorization quality through reconstruction error")
    print("• Monitor numerical stability via condition numbers")
```

**Advanced Factorization Concepts:**

**1. Matrix Completion:**
Recover missing entries assuming low-rank structure:
```
minimize ||A||* subject to A_Ω = M_Ω
```
where ||·||* is nuclear norm and Ω is set of observed entries.

**2. Sparse + Low-rank Decomposition:**
```
M = L + S
```
where L is low-rank and S is sparse, used for robust PCA.

**3. Tensor Factorizations:**
Extend matrix factorization to higher-order tensors for multi-way data analysis.

**4. Online/Streaming Factorization:**
Update factorizations incrementally as new data arrives.

**Computational Considerations:**

**Time Complexity:**
- **LU**: O(n³) for n×n matrix
- **QR**: O(mn² - n³/3) for m×n matrix
- **SVD**: O(min(mn², m²n)) 
- **Eigendecomposition**: O(n³)
- **Cholesky**: O(n³/3)

**Space Complexity:**
- Most factorizations: O(mn) storage
- In-place variants available for some methods
- Sparse factorizations: Preserve sparsity structure

**Best Practices:**
- Choose factorization based on matrix structure and computational goals
- Use appropriate numerical libraries (LAPACK, BLAS) for efficiency
- Consider iterative methods for very large matrices
- Apply regularization for noisy or ill-conditioned problems
- Validate results through reconstruction error and stability analysis

---

## Question 24

**What is a linear transformation in linear algebra?**

**Answer:** A linear transformation is a function between two vector spaces that preserves the fundamental operations of vector addition and scalar multiplication. Linear transformations are central to linear algebra and form the mathematical foundation for understanding how matrices operate on vectors, making them essential for machine learning applications including neural networks, dimensionality reduction, and feature transformations.

**Mathematical Definition:**

A function T: V → W between vector spaces V and W is a **linear transformation** (or linear map) if it satisfies two properties for all vectors **u**, **v** ∈ V and all scalars c:

**1. Additivity (Superposition):**
```
T(u + v) = T(u) + T(v)
```

**2. Homogeneity (Scalar Multiplication):**
```
T(cu) = cT(u)
```

**Combined Linearity Property:**
These can be combined into a single condition:
```
T(au + bv) = aT(u) + bT(v)
```
for all scalars a, b and vectors u, v.

**Matrix Representation:**

Every linear transformation T: ℝⁿ → ℝᵐ can be represented by an m×n matrix A such that:
```
T(x) = Ax
```

The columns of A are the images of the standard basis vectors:
```
A = [T(e₁) | T(e₂) | ... | T(eₙ)]
```
where eᵢ are the standard basis vectors of ℝⁿ.

**Geometric Interpretation:**

Linear transformations preserve:
- **Lines through origin**: Remain lines through origin
- **Parallelism**: Parallel lines stay parallel
- **Ratios of distances**: Along any line through origin
- **Linear combinations**: T(c₁v₁ + c₂v₂) = c₁T(v₁) + c₂T(v₂)

**Common Types of Linear Transformations:**

**1. Identity Transformation:**
```
T(x) = x  ↔  A = I (identity matrix)
```
Every vector maps to itself.

**2. Scaling (Dilation):**
```
T(x) = cx  ↔  A = cI
```
Uniform scaling by factor c.

**3. Reflection:**
- **Across x-axis**: A = [1  0; 0 -1]
- **Across y-axis**: A = [-1 0; 0  1]
- **Across line y=x**: A = [0 1; 1 0]

**4. Rotation:**
```
A = [cos θ  -sin θ]
    [sin θ   cos θ]
```
Counterclockwise rotation by angle θ.

**5. Shearing:**
- **Horizontal shear**: A = [1 k; 0 1]
- **Vertical shear**: A = [1 0; k 1]

**6. Projection:**
- **Onto x-axis**: A = [1 0; 0 0]
- **Onto line through origin**: A = (vvᵀ)/(vᵀv) for direction v

**Properties of Linear Transformations:**

**1. Kernel (Null Space):**
```
ker(T) = {x ∈ V : T(x) = 0}
```
The set of vectors that map to the zero vector.

**2. Image (Range):**
```
im(T) = {T(x) : x ∈ V} = {Ax : x ∈ ℝⁿ}
```
The set of all possible outputs (column space of A).

**3. Rank:**
```
rank(T) = dim(im(T))
```
Dimension of the image space.

**4. Nullity:**
```
nullity(T) = dim(ker(T))
```
Dimension of the kernel.

**5. Rank-Nullity Theorem:**
```
rank(T) + nullity(T) = dim(V)
```
Fundamental relationship for any linear transformation.

**Composition of Linear Transformations:**

If T₁: U → V and T₂: V → W are linear transformations, their composition T₂ ∘ T₁: U → W is also linear:
```
(T₂ ∘ T₁)(x) = T₂(T₁(x))
```

In matrix form: If T₁ corresponds to matrix A and T₂ to matrix B, then T₂ ∘ T₁ corresponds to BA.

**Inverse Linear Transformations:**

A linear transformation T: V → W is **invertible** if there exists T⁻¹: W → V such that:
```
T⁻¹ ∘ T = I_V  and  T ∘ T⁻¹ = I_W
```

**Conditions for Invertibility:**
- T is bijective (one-to-one and onto)
- Matrix A is square and det(A) ≠ 0
- ker(T) = {0} (injective)
- im(T) = W (surjective)

**Machine Learning Applications:**

**1. Neural Network Layers:**
Each layer performs a linear transformation followed by a nonlinear activation:
```
z = Wx + b  (linear transformation)
a = σ(z)    (nonlinear activation)
```

**2. Principal Component Analysis (PCA):**
- **Data centering**: Translation (affine, not purely linear)
- **Projection**: Linear transformation onto principal components
```
Y = XW  where W contains principal component vectors
```

**3. Feature Transformations:**
- **Standardization**: Linear scaling and translation
- **Whitening**: Linear transformation to decorrelate features
- **Basis changes**: Representation in different coordinate systems

**4. Convolutional Neural Networks:**
- **Convolution**: Linear transformation with weight sharing
- **Pooling**: Often linear (average pooling) or piecewise linear (max pooling)

**5. Linear Regression:**
```
ŷ = Xβ
```
Linear transformation from feature space to prediction space.

**6. Dimensionality Reduction:**
- **Linear projections**: To lower-dimensional subspaces
- **Matrix factorizations**: Reveal linear structure in data

**Implementation Examples:**

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def demonstrate_linear_transformation_properties():
    """Demonstrate fundamental properties of linear transformations"""
    
    print("="*60)
    print("LINEAR TRANSFORMATION PROPERTIES")
    print("="*60)
    
    # Define a sample linear transformation matrix
    A = np.array([[2, 1],
                  [1, 3]], dtype=float)
    
    print(f"Transformation matrix A:")
    print(A)
    
    # Test vectors
    u = np.array([1, 2])
    v = np.array([3, 1])
    scalars = [2, -1.5]
    
    print(f"\nTest vectors: u = {u}, v = {v}")
    print(f"Scalars: a = {scalars[0]}, b = {scalars[1]}")
    
    # Property 1: Additivity T(u + v) = T(u) + T(v)
    print(f"\n1. ADDITIVITY TEST:")
    T_u_plus_v = A @ (u + v)
    T_u_plus_T_v = A @ u + A @ v
    print(f"T(u + v) = {T_u_plus_v}")
    print(f"T(u) + T(v) = {T_u_plus_T_v}")
    print(f"Difference: {np.linalg.norm(T_u_plus_v - T_u_plus_T_v):.2e}")
    
    # Property 2: Homogeneity T(cu) = cT(u)
    print(f"\n2. HOMOGENEITY TEST:")
    c = scalars[0]
    T_cu = A @ (c * u)
    c_T_u = c * (A @ u)
    print(f"T({c}u) = {T_cu}")
    print(f"{c}T(u) = {c_T_u}")
    print(f"Difference: {np.linalg.norm(T_cu - c_T_u):.2e}")
    
    # Property 3: General linearity T(au + bv) = aT(u) + bT(v)
    print(f"\n3. GENERAL LINEARITY TEST:")
    a, b = scalars
    linear_combination = a * u + b * v
    T_linear_comb = A @ linear_combination
    linear_T_comb = a * (A @ u) + b * (A @ v)
    print(f"T({a}u + {b}v) = {T_linear_comb}")
    print(f"{a}T(u) + {b}T(v) = {linear_T_comb}")
    print(f"Difference: {np.linalg.norm(T_linear_comb - linear_T_comb):.2e}")
    
    # Demonstrate what linear transformations preserve
    print(f"\n4. PRESERVATION PROPERTIES:")
    
    # Zero vector maps to zero
    zero_vec = np.zeros(2)
    T_zero = A @ zero_vec
    print(f"T(0) = {T_zero} (should be zero vector)")
    
    # Lines through origin remain lines through origin
    t_values = np.linspace(-2, 2, 100)
    direction = np.array([1, 0.5])
    original_line = np.array([t * direction for t in t_values])
    transformed_line = np.array([A @ point for point in original_line])
    
    print(f"Original line direction: {direction}")
    print(f"Transformed line direction: {A @ direction}")
    print(f"Lines through origin are preserved ✓")

def visualize_2d_transformations():
    """Visualize common 2D linear transformations"""
    
    print(f"\n{'='*60}")
    print("2D LINEAR TRANSFORMATIONS VISUALIZATION")
    print(f"{'='*60}")
    
    # Create unit square and unit circle for visualization
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])
    
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Define various transformations
    transformations = {
        "Identity": np.array([[1, 0], [0, 1]]),
        "Scaling": np.array([[2, 0], [0, 0.5]]),
        "Rotation 45°": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                  [np.sin(np.pi/4), np.cos(np.pi/4)]]),
        "Reflection (x-axis)": np.array([[1, 0], [0, -1]]),
        "Shear": np.array([[1, 0.5], [0, 1]]),
        "Projection (x-axis)": np.array([[1, 0], [0, 0]])
    }
    
    print(f"Demonstrating transformations on unit square and circle:")
    
    for name, matrix in transformations.items():
        print(f"\n{name}:")
        print(f"Matrix: \n{matrix}")
        
        # Transform shapes
        square_transformed = matrix @ square
        circle_transformed = matrix @ circle
        
        # Calculate some properties
        det = np.linalg.det(matrix)
        print(f"Determinant: {det:.3f}")
        
        if det != 0:
            print(f"Invertible: Yes")
        else:
            print(f"Invertible: No (singular transformation)")
        
        # Check if it preserves areas
        if abs(det - 1) < 1e-10:
            print(f"Area-preserving: Yes")
        else:
            print(f"Area scaling factor: {abs(det):.3f}")

def demonstrate_kernel_and_image():
    """Demonstrate kernel and image of linear transformations"""
    
    print(f"\n{'='*60}")
    print("KERNEL AND IMAGE ANALYSIS")
    print(f"{'='*60}")
    
    # Example 1: Full rank transformation
    print(f"1. FULL RANK TRANSFORMATION:")
    A1 = np.array([[2, 1],
                   [1, 1]], dtype=float)
    
    print(f"Matrix A1:")
    print(A1)
    print(f"Rank: {np.linalg.matrix_rank(A1)}")
    print(f"Determinant: {np.linalg.det(A1):.3f}")
    
    # Kernel: solve Ax = 0
    # For full rank 2x2 matrix, kernel is just {0}
    print(f"Kernel: {{0}} (only zero vector)")
    print(f"Nullity: 0")
    print(f"Image: ℝ² (entire 2D space)")
    print(f"Rank: 2")
    
    # Example 2: Rank-deficient transformation
    print(f"\n2. RANK-DEFICIENT TRANSFORMATION:")
    A2 = np.array([[1, 2, 3],
                   [2, 4, 6]], dtype=float)
    
    print(f"Matrix A2:")
    print(A2)
    print(f"Rank: {np.linalg.matrix_rank(A2)}")
    
    # Find kernel using SVD
    U, s, Vt = np.linalg.svd(A2)
    null_space = Vt[s < 1e-10, :].T
    
    print(f"Kernel basis vectors:")
    for i, vec in enumerate(null_space.T):
        print(f"  v{i+1} = {vec}")
    
    # Verify kernel vectors
    print(f"Verification Av = 0:")
    for i, vec in enumerate(null_space.T):
        result = A2 @ vec
        print(f"  A·v{i+1} = {result} (norm: {np.linalg.norm(result):.2e})")
    
    print(f"Nullity: {null_space.shape[1]}")
    print(f"Image dimension (rank): {np.linalg.matrix_rank(A2)}")
    
    # Rank-nullity theorem verification
    domain_dim = A2.shape[1]
    rank = np.linalg.matrix_rank(A2)
    nullity = null_space.shape[1]
    print(f"\nRank-Nullity Theorem verification:")
    print(f"Domain dimension: {domain_dim}")
    print(f"Rank + Nullity = {rank} + {nullity} = {rank + nullity}")
    print(f"Theorem satisfied: {rank + nullity == domain_dim}")

def linear_transformations_in_ml():
    """Demonstrate linear transformations in machine learning contexts"""
    
    print(f"\n{'='*60}")
    print("LINEAR TRANSFORMATIONS IN MACHINE LEARNING")
    print(f"{'='*60}")
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 300, 4
    
    # Create correlated data
    true_transform = np.array([[1, 0.8, 0.3, 0.1],
                               [0, 1, 0.6, 0.2],
                               [0, 0, 1, 0.4],
                               [0, 0, 0, 1]])
    
    latent_data = np.random.randn(n_samples, n_features)
    X = latent_data @ true_transform.T + 0.1 * np.random.randn(n_samples, n_features)
    
    print(f"Generated data shape: {X.shape}")
    print(f"Data correlation structure introduced by linear transformation")
    
    # Application 1: PCA as linear transformation
    print(f"\n1. PCA AS LINEAR TRANSFORMATION:")
    
    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Original data shape: {X_scaled.shape}")
    print(f"PCA transformed shape: {X_pca.shape}")
    print(f"PCA transformation matrix shape: {pca.components_.shape}")
    
    print(f"Principal components (transformation matrix rows):")
    for i, component in enumerate(pca.components_):
        print(f"  PC{i+1}: {component}")
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Verify linearity of PCA transformation
    print(f"\nVerifying PCA linearity:")
    test_vec1 = X_scaled[0]
    test_vec2 = X_scaled[1]
    
    # Transform individual vectors
    pca_vec1 = test_vec1 @ pca.components_.T
    pca_vec2 = test_vec2 @ pca.components_.T
    
    # Transform linear combination
    linear_comb = 0.3 * test_vec1 + 0.7 * test_vec2
    pca_linear_comb = linear_comb @ pca.components_.T
    
    # Linear combination of transformed vectors
    linear_comb_pca = 0.3 * pca_vec1 + 0.7 * pca_vec2
    
    difference = np.linalg.norm(pca_linear_comb - linear_comb_pca)
    print(f"Linearity verification error: {difference:.2e}")
    
    # Application 2: Feature transformation for neural networks
    print(f"\n2. NEURAL NETWORK LAYER AS LINEAR TRANSFORMATION:")
    
    # Simulate a neural network layer (without bias for pure linearity)
    input_dim, output_dim = 4, 3
    W = np.random.randn(output_dim, input_dim) * 0.1
    
    print(f"Neural network weight matrix W:")
    print(W)
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    
    # Transform a batch of data
    batch_size = 5
    X_batch = X_scaled[:batch_size]
    Z_batch = X_batch @ W.T  # Linear transformation
    
    print(f"Batch input shape: {X_batch.shape}")
    print(f"Batch output shape: {Z_batch.shape}")
    
    # Verify linearity
    print(f"\nVerifying neural layer linearity:")
    x1, x2 = X_batch[0], X_batch[1]
    z1, z2 = x1 @ W.T, x2 @ W.T
    
    # Linear combination
    alpha, beta = 0.4, 0.6
    x_comb = alpha * x1 + beta * x2
    z_comb_direct = x_comb @ W.T
    z_comb_linear = alpha * z1 + beta * z2
    
    linearity_error = np.linalg.norm(z_comb_direct - z_comb_linear)
    print(f"Linearity verification error: {linearity_error:.2e}")
    
    # Application 3: Data whitening transformation
    print(f"\n3. DATA WHITENING TRANSFORMATION:")
    
    # Compute whitening transformation
    cov_matrix = np.cov(X_scaled.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Whitening matrix: W = Λ^(-1/2) Q^T where Q is eigenvectors, Λ is eigenvalues
    whitening_matrix = np.diag(1.0 / np.sqrt(eigenvals + 1e-8)) @ eigenvecs.T
    
    # Apply whitening
    X_whitened = X_scaled @ whitening_matrix.T
    
    print(f"Original covariance matrix:")
    print(cov_matrix)
    print(f"\nWhitened data covariance:")
    print(np.cov(X_whitened.T))
    
    # Verify whitening (should be approximately identity)
    whitened_cov = np.cov(X_whitened.T)
    identity_error = np.linalg.norm(whitened_cov - np.eye(n_features))
    print(f"Whitening quality (distance from identity): {identity_error:.4f}")

def composition_and_inverse_transformations():
    """Demonstrate composition and inverse of linear transformations"""
    
    print(f"\n{'='*60}")
    print("COMPOSITION AND INVERSE TRANSFORMATIONS")
    print(f"{'='*60}")
    
    # Define two linear transformations
    A = np.array([[2, 1], [0, 1]], dtype=float)  # Shear + scale
    B = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                  [np.sin(np.pi/6), np.cos(np.pi/6)]])  # 30° rotation
    
    print(f"Transformation A (shear + scale):")
    print(A)
    print(f"Transformation B (30° rotation):")
    print(B)
    
    # Composition: B ∘ A (apply A first, then B)
    composition_BA = B @ A
    print(f"\nComposition B ∘ A:")
    print(composition_BA)
    
    # Different order: A ∘ B (apply B first, then A)
    composition_AB = A @ B
    print(f"Composition A ∘ B:")
    print(composition_AB)
    
    print(f"Note: Matrix multiplication is not commutative!")
    print(f"||BA - AB|| = {np.linalg.norm(composition_BA - composition_AB):.4f}")
    
    # Test on a vector
    test_vector = np.array([1, 2])
    
    # Method 1: Apply composition matrix
    result1 = composition_BA @ test_vector
    
    # Method 2: Apply transformations sequentially
    intermediate = A @ test_vector
    result2 = B @ intermediate
    
    print(f"\nTesting composition on vector {test_vector}:")
    print(f"Method 1 (composition matrix): {result1}")
    print(f"Method 2 (sequential application): {result2}")
    print(f"Difference: {np.linalg.norm(result1 - result2):.2e}")
    
    # Inverse transformations
    print(f"\n4. INVERSE TRANSFORMATIONS:")
    
    # Check if A is invertible
    det_A = np.linalg.det(A)
    print(f"det(A) = {det_A:.4f}")
    
    if abs(det_A) > 1e-10:
        A_inv = np.linalg.inv(A)
        print(f"A is invertible. A^(-1):")
        print(A_inv)
        
        # Verify A * A^(-1) = I
        identity_check = A @ A_inv
        print(f"A @ A^(-1):")
        print(identity_check)
        
        identity_error = np.linalg.norm(identity_check - np.eye(2))
        print(f"Error from identity: {identity_error:.2e}")
        
        # Test inverse on a vector
        original_vector = np.array([3, 4])
        transformed = A @ original_vector
        recovered = A_inv @ transformed
        
        print(f"\nInverse transformation test:")
        print(f"Original vector: {original_vector}")
        print(f"Transformed: {transformed}")
        print(f"Recovered: {recovered}")
        print(f"Recovery error: {np.linalg.norm(original_vector - recovered):.2e}")
    
    else:
        print(f"A is not invertible (singular matrix)")

def advanced_transformation_concepts():
    """Demonstrate advanced concepts related to linear transformations"""
    
    print(f"\n{'='*60}")
    print("ADVANCED TRANSFORMATION CONCEPTS")
    print(f"{'='*60}")
    
    # 1. Change of basis as linear transformation
    print(f"1. CHANGE OF BASIS:")
    
    # Standard basis
    e1, e2 = np.array([1, 0]), np.array([0, 1])
    
    # New basis (not orthogonal)
    v1, v2 = np.array([1, 1]), np.array([1, -1])
    
    # Change of basis matrix (columns are new basis vectors)
    P = np.column_stack([v1, v2])
    
    print(f"Standard basis: e1 = {e1}, e2 = {e2}")
    print(f"New basis: v1 = {v1}, v2 = {v2}")
    print(f"Change of basis matrix P:")
    print(P)
    
    # Express a vector in new basis
    x = np.array([3, 1])  # Vector in standard coordinates
    x_new_basis = np.linalg.solve(P, x)  # Coordinates in new basis
    
    print(f"\nVector x in standard basis: {x}")
    print(f"Vector x in new basis: {x_new_basis}")
    
    # Verify: P @ x_new_basis should equal x
    verification = P @ x_new_basis
    print(f"Verification P @ x_new_basis = {verification}")
    print(f"Error: {np.linalg.norm(x - verification):.2e}")
    
    # 2. Linear transformations and coordinate systems
    print(f"\n2. TRANSFORMATIONS IN DIFFERENT COORDINATE SYSTEMS:")
    
    # Linear transformation in standard basis
    T_standard = np.array([[2, 1], [1, 2]])
    
    print(f"Transformation T in standard basis:")
    print(T_standard)
    
    # Same transformation in new basis: P^(-1) T P
    P_inv = np.linalg.inv(P)
    T_new_basis = P_inv @ T_standard @ P
    
    print(f"Same transformation in new basis:")
    print(T_new_basis)
    
    # Test consistency
    x_standard = np.array([2, 3])
    
    # Method 1: Transform in standard basis, then change basis
    Tx_standard = T_standard @ x_standard
    Tx_new_basis_method1 = P_inv @ Tx_standard
    
    # Method 2: Change basis first, then transform
    x_new_basis = P_inv @ x_standard
    Tx_new_basis_method2 = T_new_basis @ x_new_basis
    
    print(f"\nConsistency check:")
    print(f"Method 1 result: {Tx_new_basis_method1}")
    print(f"Method 2 result: {Tx_new_basis_method2}")
    print(f"Difference: {np.linalg.norm(Tx_new_basis_method1 - Tx_new_basis_method2):.2e}")
    
    # 3. Eigenspaces as invariant subspaces
    print(f"\n3. EIGENSPACES AS INVARIANT SUBSPACES:")
    
    # Symmetric matrix with known eigenstructure
    A_symmetric = np.array([[3, 1], [1, 3]], dtype=float)
    
    eigenvals, eigenvecs = np.linalg.eigh(A_symmetric)
    
    print(f"Matrix A:")
    print(A_symmetric)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:")
    print(eigenvecs)
    
    # Test that eigenvectors are invariant under transformation
    for i, (λ, v) in enumerate(zip(eigenvals, eigenvecs.T)):
        Av = A_symmetric @ v
        λv = λ * v
        
        print(f"\nEigenvector {i+1}: v = {v}")
        print(f"A @ v = {Av}")
        print(f"λ * v = {λv}")
        print(f"Difference: {np.linalg.norm(Av - λv):.2e}")

if __name__ == "__main__":
    demonstrate_linear_transformation_properties()
    visualize_2d_transformations()
    demonstrate_kernel_and_image()
    linear_transformations_in_ml()
    composition_and_inverse_transformations()
    advanced_transformation_concepts()
    
    print(f"\n{'='*60}")
    print("LINEAR TRANSFORMATION SUMMARY")
    print(f"{'='*60}")
    
    print("\nFundamental Properties:")
    print("• Preserves vector addition: T(u + v) = T(u) + T(v)")
    print("• Preserves scalar multiplication: T(cu) = cT(u)")
    print("• Maps zero vector to zero vector: T(0) = 0")
    print("• Preserves linear combinations and geometric relationships")
    
    print("\nMatrix Representation:")
    print("• Every linear transformation T: ℝⁿ → ℝᵐ has matrix A")
    print("• Columns of A are images of standard basis vectors")
    print("• Composition corresponds to matrix multiplication")
    print("• Inverse transformation corresponds to matrix inverse")
    
    print("\nKey Concepts:")
    print("• Kernel (null space): vectors mapping to zero")
    print("• Image (range): all possible outputs")
    print("• Rank-nullity theorem: rank + nullity = domain dimension")
    print("• Invertibility: bijective ⟺ full rank ⟺ determinant ≠ 0")
    
    print("\nMachine Learning Applications:")
    print("• Neural networks: each layer is linear transformation + activation")
    print("• PCA: linear projection onto principal components")
    print("• Feature transformations: standardization, whitening, rotation")
    print("• Dimensionality reduction: projection to lower-dimensional spaces")
    print("• Data preprocessing: linear scaling and normalization")
    
    print("\nGeometric Interpretations:")
    print("• Scaling: uniform or non-uniform size changes")
    print("• Rotation: orientation changes preserving shape")
    print("• Reflection: mirroring across hyperplanes")
    print("• Shearing: angular deformation")
    print("• Projection: dimensional reduction with information loss")
```

**Advanced Topics:**

**1. Affine Transformations:**
Extension of linear transformations including translation:
```
T(x) = Ax + b
```
Common in computer graphics and data preprocessing.

**2. Isomorphisms:**
Bijective linear transformations that preserve vector space structure.

**3. Endomorphisms:**
Linear transformations from a space to itself: T: V → V.

**4. Spectral Theory:**
Study of eigenvalues and eigenvectors reveals transformation behavior.

**5. Functional Analysis:**
Extension to infinite-dimensional spaces with additional structure.

**Best Practices:**
- Always verify linearity properties when implementing custom transformations
- Use matrix representation for computational efficiency
- Understand geometric meaning for intuitive algorithm design
- Monitor numerical stability, especially for compositions and inverses
- Choose appropriate coordinate systems for problem simplification
- Leverage eigenstructure for understanding transformation behavior

---

## Question 25

**Describe the kernel and image of a linear transformation.**

**Answer:** The kernel and image are two fundamental subspaces associated with any linear transformation that completely characterize its behavior. These concepts are central to understanding the structure of linear transformations, their invertibility properties, and their applications in machine learning, including dimensionality reduction, feature extraction, and solving systems of equations.

**Mathematical Definitions:**

For a linear transformation T: V → W between vector spaces V and W:

**Kernel (Null Space):**
```
ker(T) = {v ∈ V : T(v) = 0_W}
```
The kernel is the set of all vectors in the domain V that map to the zero vector in the codomain W.

**Image (Range/Column Space):**
```
im(T) = {T(v) : v ∈ V} = {w ∈ W : w = T(v) for some v ∈ V}
```
The image is the set of all vectors in W that are actually reached by the transformation T.

**Matrix Perspective:**

For a linear transformation T represented by matrix A ∈ ℝᵐˣⁿ:

**Kernel (Null Space of A):**
```
ker(T) = null(A) = {x ∈ ℝⁿ : Ax = 0}
```

**Image (Column Space of A):**
```
im(T) = col(A) = span{a₁, a₂, ..., aₙ}
```
where aᵢ are the columns of A.

**Fundamental Properties:**

**1. Subspace Properties:**
- Both ker(T) and im(T) are vector subspaces
- ker(T) ⊆ V (subspace of domain)
- im(T) ⊆ W (subspace of codomain)

**2. Dimensions:**
- **Nullity**: nullity(T) = dim(ker(T))
- **Rank**: rank(T) = dim(im(T))

**3. Rank-Nullity Theorem:**
```
rank(T) + nullity(T) = dim(V)
```
This fundamental theorem relates the dimensions of kernel and image to the dimension of the domain.

**Geometric Interpretation:**

**Kernel:**
- **Dimension 0**: Only contains zero vector (injective transformation)
- **Dimension > 0**: Contains non-trivial solutions, transformation "collapses" some dimensions
- **Higher dimension**: More "information loss" in the transformation

**Image:**
- **Full dimension**: Transformation is surjective (onto)
- **Lower dimension**: Output is constrained to a subspace
- **Rank equals**: Effective dimensionality of transformation

**Computing Kernel and Image:**

**1. Finding the Kernel:**

**Method 1: Row Reduction**
Solve Ax = 0 using Gaussian elimination:
```
[A | 0] → [RREF | 0]
```
Free variables correspond to kernel basis vectors.

**Method 2: SVD**
For A = UΣVᵀ, kernel basis vectors are columns of V corresponding to zero singular values.

**Method 3: QR with Pivoting**
Use QR decomposition with column pivoting to find dependencies.

**2. Finding the Image:**

**Method 1: Column Space**
Image is the span of linearly independent columns of A.

**Method 2: Row Reduction**
Use RREF to identify pivot columns; corresponding original columns form image basis.

**Method 3: SVD**
Image basis vectors are columns of U corresponding to non-zero singular values.

**Relationship to Invertibility:**

**Injective (One-to-One):**
```
T is injective ⟺ ker(T) = {0} ⟺ nullity(T) = 0
```

**Surjective (Onto):**
```
T is surjective ⟺ im(T) = W ⟺ rank(T) = dim(W)
```

**Bijective (Invertible):**
```
T is bijective ⟺ T is injective and surjective
⟺ ker(T) = {0} and rank(T) = dim(W)
```

**Machine Learning Applications:**

**1. Principal Component Analysis (PCA):**
- **Image**: Subspace spanned by principal components
- **Kernel**: Directions with zero variance (for centered data)
- **Interpretation**: Image captures main data variation

**2. Linear Regression:**
- **Image**: Prediction space reachable by linear combinations of features
- **Kernel**: Feature combinations that don't affect predictions
- **Overfitting**: Large kernel indicates redundant features

**3. Neural Networks:**
- **Each layer**: Kernel represents "dead" neurons or feature combinations
- **Bottleneck layers**: Small image dimension for compression
- **Gradient flow**: Large kernel can cause vanishing gradients

**4. Dimensionality Reduction:**
- **Projection**: Image is the reduced-dimensional subspace
- **Information loss**: Kernel represents discarded information
- **Reconstruction error**: Related to kernel size

**5. Feature Selection:**
- **Redundant features**: Contribute to kernel of feature matrix
- **Feature importance**: Related to contribution to image
- **Multicollinearity**: Large kernel indicates feature dependencies

**Implementation Examples:**

```python
import numpy as np
from scipy.linalg import svd, qr, null_space
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression

def compute_kernel_and_image(A, tolerance=1e-10):
    """
    Compute kernel and image of matrix A using SVD
    
    Args:
        A: Input matrix
        tolerance: Threshold for zero singular values
    
    Returns:
        kernel_basis: Basis vectors for kernel
        image_basis: Basis vectors for image
        rank: Rank of matrix
        nullity: Nullity of matrix
    """
    print(f"Computing kernel and image for matrix A:")
    print(f"Shape: {A.shape}")
    print(A)
    
    # SVD decomposition
    U, s, Vt = svd(A, full_matrices=True)
    
    print(f"\nSingular values: {s}")
    
    # Find rank (number of non-zero singular values)
    rank = np.sum(s > tolerance)
    nullity = A.shape[1] - rank
    
    print(f"Rank: {rank}")
    print(f"Nullity: {nullity}")
    
    # Kernel basis (right singular vectors for zero singular values)
    if nullity > 0:
        kernel_basis = Vt[rank:, :].T
        print(f"\nKernel basis vectors:")
        for i, vec in enumerate(kernel_basis.T):
            print(f"  v{i+1} = {vec}")
            # Verify it's in kernel
            result = A @ vec
            print(f"  A @ v{i+1} = {result} (norm: {np.linalg.norm(result):.2e})")
    else:
        kernel_basis = np.zeros((A.shape[1], 0))
        print(f"\nKernel: {{0}} (only zero vector)")
    
    # Image basis (left singular vectors for non-zero singular values)
    image_basis = U[:, :rank]
    print(f"\nImage basis vectors:")
    for i, vec in enumerate(image_basis.T):
        print(f"  u{i+1} = {vec}")
    
    return kernel_basis, image_basis, rank, nullity

def demonstrate_rank_nullity_theorem():
    """Demonstrate the rank-nullity theorem"""
    
    print("="*60)
    print("RANK-NULLITY THEOREM DEMONSTRATION")
    print("="*60)
    
    # Test matrices with different rank properties
    test_matrices = {
        "Full Rank 3x3": np.array([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 10]], dtype=float),
        
        "Rank 2, 3x3": np.array([[1, 2, 3],
                                 [4, 5, 6],
                                 [5, 7, 9]], dtype=float),  # Row 3 = Row 1 + Row 2
        
        "Rank 1, 3x4": np.array([[1, 2, 3, 4],
                                 [2, 4, 6, 8],
                                 [3, 6, 9, 12]], dtype=float),  # All rows proportional
        
        "Overdetermined": np.random.randn(5, 3),
        
        "Underdetermined": np.random.randn(3, 5)
    }
    
    for name, A in test_matrices.items():
        print(f"\n{name}:")
        print(f"Matrix shape: {A.shape}")
        
        kernel_basis, image_basis, rank, nullity = compute_kernel_and_image(A)
        
        # Verify rank-nullity theorem
        domain_dim = A.shape[1]
        print(f"\nRank-Nullity Theorem:")
        print(f"  Domain dimension: {domain_dim}")
        print(f"  Rank: {rank}")
        print(f"  Nullity: {nullity}")
        print(f"  Rank + Nullity: {rank + nullity}")
        print(f"  Theorem satisfied: {rank + nullity == domain_dim}")
        
        print("-" * 40)

def kernel_image_in_linear_regression():
    """Demonstrate kernel and image concepts in linear regression"""
    
    print(f"\n{'='*60}")
    print("KERNEL AND IMAGE IN LINEAR REGRESSION")
    print(f"{'='*60}")
    
    # Generate regression data with multicollinearity
    np.random.seed(42)
    n_samples, n_features = 100, 5
    
    # Create design matrix with redundancy
    X_independent = np.random.randn(n_samples, 3)
    # Add two dependent features
    X_dependent = np.column_stack([
        X_independent[:, 0] + X_independent[:, 1],  # Feature 4 = Feature 1 + Feature 2
        2 * X_independent[:, 0] - X_independent[:, 2]  # Feature 5 = 2*Feature 1 - Feature 3
    ])
    
    X = np.column_stack([X_independent, X_dependent])
    
    # Generate target with only 3 true features
    true_beta = np.array([1, -1, 0.5, 0, 0])  # Last two should be zero due to redundancy
    y = X @ true_beta + 0.1 * np.random.randn(n_samples)
    
    print(f"Design matrix X shape: {X.shape}")
    print(f"True coefficients: {true_beta}")
    
    # Analyze X^T X (normal equations matrix)
    XtX = X.T @ X
    print(f"\nNormal equations matrix X^T X:")
    
    kernel_basis, image_basis, rank, nullity = compute_kernel_and_image(XtX)
    
    print(f"\nMulticollinearity analysis:")
    print(f"Expected rank (true features): 3")
    print(f"Actual rank: {rank}")
    print(f"Nullity (redundant feature combinations): {nullity}")
    
    if nullity > 0:
        print(f"\nFeature dependencies (kernel vectors):")
        for i, vec in enumerate(kernel_basis.T):
            print(f"  Dependency {i+1}: {vec}")
            # Interpret the dependency
            significant_indices = np.where(np.abs(vec) > 0.1)[0]
            print(f"    Involves features: {significant_indices}")
    
    # Solve using pseudoinverse (handles multicollinearity)
    beta_pinv = np.linalg.pinv(X) @ y
    
    print(f"\nRegression results:")
    print(f"True coefficients:      {true_beta}")
    print(f"Estimated coefficients: {beta_pinv}")
    
    # Compare with regularized solution (Ridge regression)
    lambda_reg = 0.01
    beta_ridge = np.linalg.solve(XtX + lambda_reg * np.eye(n_features), X.T @ y)
    print(f"Ridge coefficients:     {beta_ridge}")

def pca_kernel_image_analysis():
    """Analyze kernel and image in PCA context"""
    
    print(f"\n{'='*60}")
    print("KERNEL AND IMAGE IN PCA")
    print(f"{'='*60}")
    
    # Generate data with intrinsic low-dimensional structure
    np.random.seed(42)
    n_samples = 200
    
    # True 2D latent variables
    latent_dim = 2
    Z = np.random.randn(n_samples, latent_dim)
    
    # Project to higher dimensional space with noise
    observed_dim = 5
    projection_matrix = np.random.randn(latent_dim, observed_dim)
    noise_level = 0.1
    
    X = Z @ projection_matrix + noise_level * np.random.randn(n_samples, observed_dim)
    
    print(f"Generated data:")
    print(f"  True latent dimension: {latent_dim}")
    print(f"  Observed dimension: {observed_dim}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Noise level: {noise_level}")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    print(f"\nCovariance matrix analysis:")
    kernel_basis, image_basis, rank, nullity = compute_kernel_and_image(cov_matrix)
    
    # PCA via SVD
    U, s, Vt = svd(X_centered, full_matrices=False)
    explained_variance = s**2 / (n_samples - 1)
    cumulative_variance = np.cumsum(explained_variance) / np.sum(explained_variance)
    
    print(f"\nPCA results:")
    print(f"Explained variance: {explained_variance}")
    print(f"Cumulative variance: {cumulative_variance}")
    
    # Determine effective dimensionality
    variance_threshold = 0.95
    effective_dim = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"Effective dimension (95% variance): {effective_dim}")
    print(f"Close to true latent dimension: {effective_dim == latent_dim}")
    
    # Analyze PCA transformation matrix
    pca_transform = Vt[:effective_dim, :]  # First few principal components
    
    print(f"\nPCA transformation matrix (first {effective_dim} components):")
    print(pca_transform)
    
    # Image: subspace spanned by principal components
    print(f"\nPCA image analysis:")
    print(f"  Image dimension: {effective_dim}")
    print(f"  Image represents {cumulative_variance[effective_dim-1]:.1%} of variance")
    
    # Kernel: directions with minimal variance
    minor_components = Vt[effective_dim:, :]
    if minor_components.shape[0] > 0:
        print(f"\nMinor components (approximate kernel):")
        for i, component in enumerate(minor_components):
            variance_explained = explained_variance[effective_dim + i] / np.sum(explained_variance)
            print(f"  Component {effective_dim + i + 1}: variance = {variance_explained:.1%}")

def neural_network_layer_analysis():
    """Analyze kernel and image in neural network context"""
    
    print(f"\n{'='*60}")
    print("KERNEL AND IMAGE IN NEURAL NETWORKS")
    print(f"{'='*60}")
    
    # Simulate neural network layer
    np.random.seed(42)
    input_dim, hidden_dim = 8, 5
    
    # Weight matrix (with some structure)
    W = np.random.randn(hidden_dim, input_dim) * 0.1
    
    # Introduce some structure (make last row dependent on first two)
    W[-1, :] = 0.5 * W[0, :] + 0.3 * W[1, :]
    
    print(f"Neural network layer:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Weight matrix W shape: {W.shape}")
    
    print(f"\nWeight matrix W:")
    print(W)
    
    # Analyze weight matrix
    kernel_basis, image_basis, rank, nullity = compute_kernel_and_image(W)
    
    print(f"\nLayer analysis:")
    print(f"  Effective rank: {rank}")
    print(f"  Redundant neurons: {hidden_dim - rank}")
    
    if nullity > 0:
        print(f"\nInput combinations that produce zero output:")
        for i, vec in enumerate(kernel_basis.T):
            print(f"  Null vector {i+1}: {vec}")
            
            # Test on sample input
            test_input = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
            output = W @ test_input
            print(f"    Output norm: {np.linalg.norm(output):.2e}")
    
    # Generate sample inputs to test layer behavior
    n_samples = 100
    X_test = np.random.randn(n_samples, input_dim)
    Z_test = X_test @ W.T  # Linear transformation
    
    print(f"\nLayer output analysis:")
    print(f"  Input data shape: {X_test.shape}")
    print(f"  Output data shape: {Z_test.shape}")
    
    # Analyze output covariance to see effective dimensionality
    Z_centered = Z_test - np.mean(Z_test, axis=0)
    output_cov = np.cov(Z_centered.T)
    
    eigenvals, eigenvecs = np.linalg.eigh(output_cov)
    eigenvals = eigenvals[::-1]  # Sort descending
    
    print(f"  Output eigenvalues: {eigenvals}")
    
    # Effective output dimension
    eigenval_threshold = 0.01 * eigenvals[0]
    effective_output_dim = np.sum(eigenvals > eigenval_threshold)
    
    print(f"  Effective output dimension: {effective_output_dim}")
    print(f"  Dimension reduction: {input_dim} → {effective_output_dim}")

def advanced_kernel_image_applications():
    """Demonstrate advanced applications of kernel and image concepts"""
    
    print(f"\n{'='*60}")
    print("ADVANCED APPLICATIONS")
    print(f"{'='*60}")
    
    # Application 1: Matrix completion using kernel/image structure
    print("1. MATRIX COMPLETION:")
    
    # Create low-rank matrix with missing entries
    np.random.seed(42)
    m, n, r = 20, 15, 3  # m×n matrix with rank r
    
    U_true = np.random.randn(m, r)
    V_true = np.random.randn(r, n)
    M_complete = U_true @ V_true
    
    # Create observation mask (60% observed)
    observation_rate = 0.6
    mask = np.random.random((m, n)) < observation_rate
    M_observed = np.where(mask, M_complete, 0)
    
    print(f"Matrix completion setup:")
    print(f"  Matrix size: {m}×{n}")
    print(f"  True rank: {r}")
    print(f"  Observation rate: {observation_rate:.1%}")
    print(f"  Observed entries: {np.sum(mask)}")
    
    # Simple completion: SVD of observed matrix
    U_obs, s_obs, Vt_obs = svd(M_observed, full_matrices=False)
    
    # Truncate to estimated rank
    M_completed = U_obs[:, :r] @ np.diag(s_obs[:r]) @ Vt_obs[:r, :]
    
    # Evaluate completion quality
    completion_error = np.linalg.norm(M_complete[~mask] - M_completed[~mask])
    total_error = np.linalg.norm(M_complete - M_completed)
    
    print(f"  Completion error (missing entries): {completion_error:.4f}")
    print(f"  Total reconstruction error: {total_error:.4f}")
    
    # Application 2: Feature selection using kernel analysis
    print(f"\n2. FEATURE SELECTION VIA KERNEL ANALYSIS:")
    
    # Generate features with dependencies
    n_samples, n_features = 150, 8
    X_independent = np.random.randn(n_samples, 4)
    
    # Add dependent features
    X_dependent = np.column_stack([
        X_independent[:, 0] + 0.1 * np.random.randn(n_samples),  # Almost identical to feature 1
        X_independent[:, 1] + X_independent[:, 2],  # Linear combination
        0.5 * X_independent[:, 0] - 0.3 * X_independent[:, 3],  # Another combination
        np.random.randn(n_samples)  # Independent feature
    ])
    
    X_all = np.column_stack([X_independent, X_dependent])
    
    print(f"Feature dependency analysis:")
    print(f"  Total features: {n_features}")
    print(f"  True independent features: 5 (4 original + 1 new)")
    
    # Analyze feature correlation matrix
    feature_corr = np.corrcoef(X_all.T)
    
    kernel_basis, image_basis, rank, nullity = compute_kernel_and_image(feature_corr)
    
    print(f"  Correlation matrix rank: {rank}")
    print(f"  Redundant feature combinations: {nullity}")
    
    if nullity > 0:
        print(f"\nRedundant feature patterns:")
        for i, pattern in enumerate(kernel_basis.T):
            significant_features = np.where(np.abs(pattern) > 0.1)[0]
            print(f"  Pattern {i+1}: Features {significant_features}")
            print(f"    Coefficients: {pattern[significant_features]}")

def computational_methods_comparison():
    """Compare different methods for computing kernel and image"""
    
    print(f"\n{'='*60}")
    print("COMPUTATIONAL METHODS COMPARISON")
    print(f"{'='*60}")
    
    # Create test matrix
    np.random.seed(42)
    A = np.array([[1, 2, 3, 4],
                  [2, 4, 6, 8],
                  [1, 1, 1, 1],
                  [3, 5, 7, 9]], dtype=float)
    
    print(f"Test matrix A:")
    print(A)
    print(f"Shape: {A.shape}")
    
    # Method 1: SVD
    print(f"\n1. SVD METHOD:")
    U, s, Vt = svd(A, full_matrices=True)
    tolerance = 1e-10
    
    rank_svd = np.sum(s > tolerance)
    kernel_svd = Vt[rank_svd:, :].T if rank_svd < A.shape[1] else np.zeros((A.shape[1], 0))
    image_svd = U[:, :rank_svd]
    
    print(f"  Singular values: {s}")
    print(f"  Rank: {rank_svd}")
    print(f"  Kernel dimension: {kernel_svd.shape[1]}")
    
    # Method 2: QR with pivoting
    print(f"\n2. QR WITH PIVOTING:")
    Q, R, P = qr(A.T, pivoting=True)  # QR of A^T to get column dependencies
    
    # Find rank from R diagonal
    rank_qr = np.sum(np.abs(np.diag(R)) > tolerance)
    
    print(f"  R diagonal: {np.diag(R)}")
    print(f"  Rank: {rank_qr}")
    print(f"  Permutation: {P}")
    
    # Method 3: Gaussian elimination (RREF)
    print(f"\n3. ROW REDUCTION METHOD:")
    from scipy.linalg import solve
    
    # For demonstration, manually analyze row reduction
    A_rref = A.copy()
    print(f"  Starting with A")
    
    # Simple rank estimation via matrix rank function
    rank_rref = np.linalg.matrix_rank(A_rref, tol=tolerance)
    print(f"  Rank (via matrix_rank): {rank_rref}")
    
    # Method 4: Using scipy null_space
    print(f"\n4. SCIPY NULL_SPACE:")
    try:
        null_basis = null_space(A)
        print(f"  Null space dimension: {null_basis.shape[1]}")
        print(f"  Kernel basis vectors:")
        for i, vec in enumerate(null_basis.T):
            print(f"    v{i+1} = {vec}")
            verification = A @ vec
            print(f"    ||A @ v{i+1}|| = {np.linalg.norm(verification):.2e}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Compare results
    print(f"\n5. COMPARISON:")
    print(f"  SVD rank: {rank_svd}")
    print(f"  QR rank: {rank_qr}")
    print(f"  RREF rank: {rank_rref}")
    print(f"  All methods agree: {rank_svd == rank_qr == rank_rref}")

if __name__ == "__main__":
    # Demonstrate basic computation
    A_example = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=float)
    
    print("BASIC KERNEL AND IMAGE COMPUTATION")
    print("="*50)
    compute_kernel_and_image(A_example)
    
    # Run all demonstrations
    demonstrate_rank_nullity_theorem()
    kernel_image_in_linear_regression()
    pca_kernel_image_analysis()
    neural_network_layer_analysis()
    advanced_kernel_image_applications()
    computational_methods_comparison()
    
    print(f"\n{'='*60}")
    print("KERNEL AND IMAGE SUMMARY")
    print(f"{'='*60}")
    
    print("\nDefinitions:")
    print("• Kernel: All vectors mapping to zero vector")
    print("• Image: All possible outputs of the transformation")
    print("• Both are vector subspaces with well-defined dimensions")
    
    print("\nFundamental Theorem:")
    print("• Rank-Nullity: rank(T) + nullity(T) = dim(domain)")
    print("• Connects transformation properties to subspace dimensions")
    
    print("\nComputational Methods:")
    print("• SVD: Most numerically stable, reveals both kernel and image")
    print("• QR with pivoting: Good for rank determination")
    print("• Row reduction: Theoretical understanding, less stable")
    print("• Specialized algorithms: For large sparse matrices")
    
    print("\nML Applications:")
    print("• Feature selection: Kernel reveals redundant feature combinations")
    print("• Dimensionality reduction: Image determines effective output dimension")
    print("• Neural networks: Layer analysis via kernel/image structure")
    print("• Regression: Multicollinearity detection through kernel analysis")
    print("• Matrix completion: Low-rank structure via kernel/image")
    
    print("\nPractical Implications:")
    print("• Large kernel → information loss, redundancy")
    print("• Small image → limited output diversity")
    print("• Rank deficiency → non-unique solutions")
    print("• Full rank → invertible transformation (for square matrices)")
```

**Advanced Theoretical Connections:**

**1. Four Fundamental Subspaces:**
For matrix A ∈ ℝᵐˣⁿ:
- **Column space**: im(A) ⊆ ℝᵐ
- **Null space**: ker(A) ⊆ ℝⁿ  
- **Row space**: im(Aᵀ) ⊆ ℝⁿ
- **Left null space**: ker(Aᵀ) ⊆ ℝᵐ

**2. Orthogonal Complements:**
- ker(A) ⊥ im(Aᵀ) (orthogonal complement in ℝⁿ)
- ker(Aᵀ) ⊥ im(A) (orthogonal complement in ℝᵐ)

**3. Projection Operators:**
- Projection onto im(A): P = A(AᵀA)⁺Aᵀ
- Projection onto ker(A): I - (AᵀA)⁺Aᵀ

**Numerical Considerations:**
- Use SVD for maximum numerical stability
- Choose appropriate tolerance for rank determination
- Consider iterative methods for very large matrices
- Monitor condition number for stability assessment
- Apply regularization when kernel is large (ill-conditioned problems)

---

## Question 26

**How does change of basis affect matrix representation of linear transformations?**

**Answer:** Change of basis is a fundamental concept in linear algebra that demonstrates how the matrix representation of a linear transformation depends on the choice of coordinate system. Understanding this relationship is crucial for applications in machine learning, computer graphics, signal processing, and optimization, where different coordinate systems can reveal different aspects of the same underlying transformation.

**Fundamental Concept:**

A linear transformation T: V → W can be represented by different matrices depending on the bases chosen for the vector spaces V and W. The transformation itself remains the same, but its matrix representation changes with the coordinate system.

**Mathematical Framework:**

**Setup:**
- Vector space V with bases B = {v₁, v₂, ..., vₙ} and B' = {v'₁, v'₂, ..., v'ₙ}
- Vector space W with bases C = {w₁, w₂, ..., wₘ} and C' = {w'₁, w'₂, ..., w'ₘ}
- Linear transformation T: V → W
- Matrix representations [T]ᴄ_ᴮ in (B,C) bases and [T]ᴄ'_ᴮ' in (B',C') bases

**Change of Basis Matrices:**
- **Input basis change**: P = [I]ᴮ_ᴮ' (from B' to B coordinates)
- **Output basis change**: Q = [I]ᴄ'_ᴄ (from C to C' coordinates)

**Transformation Relationship:**
```
[T]ᴄ'_ᴮ' = Q⁻¹[T]ᴄ_ᴮP
```

This is the fundamental change of basis formula for linear transformations.

**Special Case - Same Space (Similarity Transformation):**

For transformations T: V → V (endomorphisms), with basis change P:
```
[T]ᴮ' = P⁻¹[T]ᴮP
```

This creates **similar matrices** - different representations of the same transformation.

**Step-by-Step Change of Basis Process:**

**1. Coordinate Conversion:**
```
v ∈ V → [v]ᴮ → P[v]ᴮ = [v]ᴮ' → T application → [T(v)]ᴄ' → Q[T(v)]ᴄ' = [T(v)]ᴄ
```

**2. Matrix Transformation:**
```
[T]ᴄ_ᴮ → Q⁻¹[T]ᴄ_ᴮP = [T]ᴄ'_ᴮ'
```

**Properties Preserved Under Change of Basis:**

**Invariant Properties:**
- **Rank**: rank([T]ᴮ) = rank([T]ᴮ') for any bases
- **Determinant**: det([T]ᴮ) = det([T]ᴮ') (for square matrices)
- **Trace**: tr([T]ᴮ) = tr([T]ᴮ') (sum of diagonal elements)
- **Eigenvalues**: Eigenvalues remain the same
- **Characteristic polynomial**: Coefficients unchanged
- **Nullity**: Dimension of kernel remains constant

**Changed Properties:**
- **Matrix entries**: Individual elements change
- **Eigenvectors**: Coordinate representations change
- **Condition number**: Can improve or worsen
- **Sparsity pattern**: Structure may change dramatically

**Geometric Interpretation:**

**2D Example:**
Consider rotation by θ in different coordinate systems:

**Standard basis**: 
```
R = [cos θ  -sin θ]
    [sin θ   cos θ]
```

**Diagonal basis** (if available):
In eigenvector basis, rotation becomes:
```
R' = [e^(iθ)    0   ]  (complex form)
     [0      e^(-iθ)]
```

**Applications in Machine Learning:**

**1. Principal Component Analysis (PCA):**
- **Original basis**: Data in feature space
- **PCA basis**: Eigenvectors of covariance matrix
- **Transformation effect**: Diagonalizes covariance, decorrelates features
- **Matrix change**: Covariance becomes diagonal in PCA coordinates

**2. Whitening Transformations:**
- **Goal**: Transform data to have identity covariance
- **Basis change**: From original to whitened coordinates
- **Matrix effect**: Σ → I (covariance matrix becomes identity)

**3. Neural Network Layer Analysis:**
- **Input/output spaces**: Different layer representations
- **Basis change**: Layer weights define coordinate transformation
- **Optimization**: Finding optimal representation at each layer

**4. Optimization Landscapes:**
- **Coordinate system**: Choice affects conditioning
- **Preconditioning**: Change basis to improve convergence
- **Second-order methods**: Natural gradient uses information geometry

**Practical Implementation Examples:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.decomposition import PCA

def demonstrate_change_of_basis_2d():
    """Demonstrate change of basis for 2D linear transformation"""
    
    print("="*60)
    print("2D CHANGE OF BASIS DEMONSTRATION")
    print("="*60)
    
    # Define a 2D linear transformation (shear + scaling)
    T_standard = np.array([[2, 1],
                          [0, 1]], dtype=float)
    
    print("Original transformation matrix T (standard basis):")
    print(T_standard)
    
    # Define a new basis (rotated and scaled)
    # New basis vectors as columns
    v1_new = np.array([1, 1])  # First new basis vector
    v2_new = np.array([1, -1]) # Second new basis vector
    
    # Change of basis matrix P (columns are new basis vectors expressed in old basis)
    P = np.column_stack([v1_new, v2_new])
    P_inv = np.linalg.inv(P)
    
    print(f"\nChange of basis matrix P:")
    print(P)
    print(f"P inverse:")
    print(P_inv)
    
    # Transform T to new basis
    T_new_basis = P_inv @ T_standard @ P
    
    print(f"\nTransformation matrix T in new basis:")
    print(T_new_basis)
    
    # Verify invariant properties
    print(f"\nInvariant properties verification:")
    print(f"Original determinant: {np.linalg.det(T_standard):.6f}")
    print(f"New basis determinant: {np.linalg.det(T_new_basis):.6f}")
    print(f"Original trace: {np.trace(T_standard):.6f}")
    print(f"New basis trace: {np.trace(T_new_basis):.6f}")
    
    # Eigenvalues
    eig_orig, _ = np.linalg.eig(T_standard)
    eig_new, _ = np.linalg.eig(T_new_basis)
    
    print(f"Original eigenvalues: {eig_orig}")
    print(f"New basis eigenvalues: {eig_new}")
    print(f"Eigenvalues match: {np.allclose(np.sort(eig_orig), np.sort(eig_new))}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Test vectors
    test_vectors = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]]).T
    
    # Plot 1: Original basis
    ax = axes[0]
    # Original vectors
    ax.quiver(0, 0, test_vectors[0], test_vectors[1], 
             angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7,
             label='Original vectors')
    
    # Transformed vectors in original basis
    transformed_vectors = T_standard @ test_vectors
    ax.quiver(0, 0, transformed_vectors[0], transformed_vectors[1],
             angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7,
             label='Transformed vectors')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title('Standard Basis')
    ax.legend()
    
    # Plot 2: New basis vectors
    ax = axes[1]
    # Show new basis vectors
    ax.quiver(0, 0, P[0], P[1], angles='xy', scale_units='xy', scale=1,
             color='green', width=0.005, label='New basis vectors')
    
    # Standard basis for reference
    ax.quiver(0, 0, [1, 0], [0, 1], angles='xy', scale_units='xy', scale=1,
             color='gray', alpha=0.5, label='Standard basis')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title('Basis Comparison')
    ax.legend()
    
    # Plot 3: Transformation in new basis
    ax = axes[2]
    # Express test vectors in new basis
    test_vectors_new_coords = P_inv @ test_vectors
    
    ax.quiver(0, 0, test_vectors_new_coords[0], test_vectors_new_coords[1],
             angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7,
             label='Vectors (new coords)')
    
    # Transform in new basis
    transformed_new_coords = T_new_basis @ test_vectors_new_coords
    ax.quiver(0, 0, transformed_new_coords[0], transformed_new_coords[1],
             angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7,
             label='Transformed (new coords)')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title('New Basis Coordinates')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('change_of_basis_2d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return T_standard, T_new_basis, P

def pca_as_change_of_basis():
    """Demonstrate PCA as change of basis"""
    
    print(f"\n{'='*60}")
    print("PCA AS CHANGE OF BASIS")
    print(f"{'='*60}")
    
    # Generate correlated 2D data
    np.random.seed(42)
    n_samples = 200
    
    # Create correlated data
    angle = np.pi / 6  # 30 degrees
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    
    scaling = np.array([[3, 0],
                       [0, 1]])
    
    # Generate data in principal component space
    data_pc_space = np.random.randn(n_samples, 2)
    
    # Transform to correlated space
    transform_matrix = rotation @ scaling
    data_original = data_pc_space @ transform_matrix.T
    
    print(f"Generated {n_samples} correlated data points")
    print(f"Data shape: {data_original.shape}")
    
    # Compute covariance matrix
    data_centered = data_original - np.mean(data_original, axis=0)
    cov_matrix = np.cov(data_centered.T)
    
    print(f"\nCovariance matrix (original basis):")
    print(cov_matrix)
    
    # PCA via eigendecomposition
    eigenvalues, eigenvectors = eigh(cov_matrix)
    
    # Sort by eigenvalue magnitude (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nPCA eigenvalues: {eigenvalues}")
    print(f"PCA eigenvectors:")
    print(eigenvectors)
    
    # Change of basis matrix (PCA transformation)
    P_pca = eigenvectors.T  # Rows are principal components
    
    print(f"\nPCA transformation matrix (change of basis):")
    print(P_pca)
    
    # Transform data to PCA coordinates
    data_pca_coords = data_centered @ eigenvectors
    
    # Verify covariance in PCA space
    cov_pca_space = np.cov(data_pca_coords.T)
    
    print(f"\nCovariance matrix (PCA basis):")
    print(cov_pca_space)
    print(f"Is diagonal: {np.allclose(cov_pca_space, np.diag(np.diag(cov_pca_space)), atol=1e-10)}")
    
    # Compare with sklearn PCA
    pca_sklearn = PCA()
    data_sklearn_pca = pca_sklearn.fit_transform(data_centered)
    
    print(f"\nComparison with sklearn:")
    print(f"Eigenvalues match: {np.allclose(eigenvalues, pca_sklearn.explained_variance_)}")
    print(f"Transformed data match: {np.allclose(np.abs(data_pca_coords), np.abs(data_sklearn_pca))}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(data_original[:, 0], data_original[:, 1], alpha=0.6, s=20)
    
    # Plot covariance ellipse
    from matplotlib.patches import Ellipse
    mean = np.mean(data_original, axis=0)
    
    # Eigenvalues and eigenvectors for ellipse
    vals, vecs = eigh(cov_matrix)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    # Confidence ellipse (2 standard deviations)
    width, height = 2 * np.sqrt(vals)
    angle_deg = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    ellipse = Ellipse(mean, width, height, angle=angle_deg, 
                     facecolor='none', edgecolor='red', linewidth=2)
    plt.gca().add_patch(ellipse)
    
    # Plot principal components
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        plt.arrow(mean[0], mean[1], vec[0]*np.sqrt(val), vec[1]*np.sqrt(val),
                 head_width=0.2, head_length=0.2, fc=f'C{i+1}', ec=f'C{i+1}',
                 linewidth=3, label=f'PC{i+1}')
    
    plt.xlabel('X1 (original)')
    plt.ylabel('X2 (original)')
    plt.title('Original Data with Principal Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # PCA coordinates
    plt.subplot(1, 3, 2)
    plt.scatter(data_pca_coords[:, 0], data_pca_coords[:, 1], alpha=0.6, s=20)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Data in PCA Coordinates')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Variance explanation
    plt.subplot(1, 3, 3)
    variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(variance_ratio)
    
    plt.bar(range(len(variance_ratio)), variance_ratio, alpha=0.7, 
            label='Individual')
    plt.plot(range(len(cumulative_variance)), cumulative_variance, 'ro-',
            label='Cumulative')
    
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('Variance Explanation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_change_of_basis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return P_pca, cov_matrix, cov_pca_space

def optimization_preconditioning():
    """Demonstrate preconditioning as change of basis for optimization"""
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION PRECONDITIONING VIA CHANGE OF BASIS")
    print(f"{'='*60}")
    
    # Define ill-conditioned quadratic function: f(x) = (1/2)x^T A x
    # where A has very different eigenvalues
    
    A_original = np.array([[100, 0],
                          [0, 1]], dtype=float)
    
    print(f"Original objective matrix A:")
    print(A_original)
    
    # Condition number
    cond_original = np.linalg.cond(A_original)
    print(f"Condition number: {cond_original:.2f}")
    
    # Eigendecomposition for preconditioning
    eigenvals, eigenvecs = eigh(A_original)
    
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:")
    print(eigenvecs)
    
    # Preconditioning matrix (square root of inverse)
    # P = A^(-1/2) transforms A to identity
    P = eigenvecs @ np.diag(1/np.sqrt(eigenvals)) @ eigenvecs.T
    
    print(f"\nPreconditioning matrix P:")
    print(P)
    
    # Transformed matrix should be identity
    A_preconditioned = P.T @ A_original @ P
    
    print(f"\nPreconditioned matrix P^T A P:")
    print(A_preconditioned)
    print(f"Is identity: {np.allclose(A_preconditioned, np.eye(2), atol=1e-10)}")
    
    # Simulate gradient descent optimization
    def quadratic_function(x, A):
        return 0.5 * x.T @ A @ x
    
    def gradient(x, A):
        return A @ x
    
    # Starting point
    x0 = np.array([10, 10], dtype=float)
    learning_rate = 0.01
    max_iterations = 100
    
    # Original space optimization
    x_orig = x0.copy()
    trajectory_orig = [x_orig.copy()]
    
    for i in range(max_iterations):
        grad = gradient(x_orig, A_original)
        x_orig = x_orig - learning_rate * grad
        trajectory_orig.append(x_orig.copy())
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    print(f"\nOriginal space optimization:")
    print(f"Iterations to convergence: {len(trajectory_orig)}")
    print(f"Final point: {trajectory_orig[-1]}")
    
    # Preconditioned space optimization
    x_precond = P.T @ x0  # Transform initial point
    trajectory_precond_space = [x_precond.copy()]
    
    for i in range(max_iterations):
        grad = gradient(x_precond, A_preconditioned)
        x_precond = x_precond - learning_rate * grad
        trajectory_precond_space.append(x_precond.copy())
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    # Transform back to original space
    trajectory_precond = [P @ x for x in trajectory_precond_space]
    
    print(f"\nPreconditioned space optimization:")
    print(f"Iterations to convergence: {len(trajectory_precond_space)}")
    print(f"Final point (original coords): {trajectory_precond[-1]}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Create contour plot
    x_range = np.linspace(-12, 12, 100)
    y_range = np.linspace(-12, 12, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Original space
    plt.subplot(1, 3, 1)
    Z_orig = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i,j], Y[i,j]])
            Z_orig[i,j] = quadratic_function(point, A_original)
    
    contours = plt.contour(X, Y, Z_orig, levels=20, alpha=0.6)
    plt.clabel(contours, inline=True, fontsize=8)
    
    # Plot optimization trajectory
    traj_orig = np.array(trajectory_orig)
    plt.plot(traj_orig[:, 0], traj_orig[:, 1], 'ro-', markersize=4, alpha=0.7,
            label=f'Original ({len(trajectory_orig)} steps)')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Original Space Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Preconditioned space
    plt.subplot(1, 3, 2)
    Z_precond = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i,j], Y[i,j]])
            Z_precond[i,j] = quadratic_function(point, A_preconditioned)
    
    contours = plt.contour(X, Y, Z_precond, levels=20, alpha=0.6)
    plt.clabel(contours, inline=True, fontsize=8)
    
    # Plot optimization trajectory in preconditioned space
    traj_precond_space = np.array(trajectory_precond_space)
    plt.plot(traj_precond_space[:, 0], traj_precond_space[:, 1], 'bo-', 
            markersize=4, alpha=0.7,
            label=f'Preconditioned ({len(trajectory_precond_space)} steps)')
    
    plt.xlabel('y1 (preconditioned)')
    plt.ylabel('y2 (preconditioned)')
    plt.title('Preconditioned Space Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparison in original space
    plt.subplot(1, 3, 3)
    contours = plt.contour(X, Y, Z_orig, levels=20, alpha=0.6)
    
    # Both trajectories in original space
    plt.plot(traj_orig[:, 0], traj_orig[:, 1], 'ro-', markersize=4, alpha=0.7,
            label=f'Original ({len(trajectory_orig)} steps)')
    
    traj_precond_orig = np.array(trajectory_precond)
    plt.plot(traj_precond_orig[:, 0], traj_precond_orig[:, 1], 'bo-', 
            markersize=4, alpha=0.7,
            label=f'Preconditioned ({len(trajectory_precond)} steps)')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Optimization Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_preconditioning.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return A_original, A_preconditioned, P

def neural_network_layer_basis_change():
    """Demonstrate change of basis in neural network layers"""
    
    print(f"\n{'='*60}")
    print("NEURAL NETWORK LAYERS AS BASIS CHANGES")
    print(f"{'='*60}")
    
    # Simulate a simple neural network layer
    # Input dimension: 4, Hidden dimension: 3
    np.random.seed(42)
    
    input_dim = 4
    hidden_dim = 3
    
    # Weight matrix (linear transformation)
    W = np.random.randn(hidden_dim, input_dim) * 0.5
    b = np.random.randn(hidden_dim) * 0.1
    
    print(f"Neural network layer:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Weight matrix W:")
    print(W)
    print(f"  Bias vector b: {b}")
    
    # Analyze the weight matrix as a change of basis
    U, s, Vt = np.linalg.svd(W, full_matrices=True)
    
    print(f"\nSVD analysis of weight matrix:")
    print(f"  Singular values: {s}")
    print(f"  Rank: {np.sum(s > 1e-10)}")
    
    # The weight matrix transforms from input space to hidden space
    print(f"\nLinear transformation analysis:")
    print(f"  Input space dimension: {input_dim}")
    print(f"  Output space dimension: {hidden_dim}")
    print(f"  Effective rank: {np.sum(s > 1e-10)}")
    
    # Generate sample input data
    n_samples = 100
    X_input = np.random.randn(n_samples, input_dim)
    
    # Forward pass (linear part only)
    Z_hidden = X_input @ W.T + b
    
    print(f"\nData transformation:")
    print(f"  Input data shape: {X_input.shape}")
    print(f"  Hidden representation shape: {Z_hidden.shape}")
    
    # Analyze input and output spaces
    print(f"\nInput space analysis:")
    input_cov = np.cov(X_input.T)
    input_eigs, _ = np.linalg.eigh(input_cov)
    print(f"  Input covariance eigenvalues: {np.sort(input_eigs)[::-1]}")
    
    print(f"\nHidden space analysis:")
    hidden_cov = np.cov(Z_hidden.T)
    hidden_eigs, _ = np.linalg.eigh(hidden_cov)
    print(f"  Hidden covariance eigenvalues: {np.sort(hidden_eigs)[::-1]}")
    
    # Effective dimensionality
    input_effective_dim = np.sum(input_eigs > 0.1 * np.max(input_eigs))
    hidden_effective_dim = np.sum(hidden_eigs > 0.1 * np.max(hidden_eigs))
    
    print(f"\nEffective dimensionality:")
    print(f"  Input effective dimension: {input_effective_dim}")
    print(f"  Hidden effective dimension: {hidden_effective_dim}")
    
    # Information preservation analysis
    # Compute how much input variance is preserved
    total_input_var = np.trace(input_cov)
    total_hidden_var = np.trace(hidden_cov)
    
    print(f"\nInformation preservation:")
    print(f"  Total input variance: {total_input_var:.4f}")
    print(f"  Total hidden variance: {total_hidden_var:.4f}")
    print(f"  Variance ratio: {total_hidden_var/total_input_var:.4f}")
    
    return W, X_input, Z_hidden

def advanced_similarity_transformations():
    """Demonstrate advanced similarity transformation concepts"""
    
    print(f"\n{'='*60}")
    print("ADVANCED SIMILARITY TRANSFORMATIONS")
    print(f"{'='*60}")
    
    # Example: Diagonalization via similarity transformation
    # Create a matrix that can be diagonalized
    np.random.seed(42)
    
    # Create a symmetric matrix (guaranteed to be diagonalizable)
    A_sym = np.random.randn(3, 3)
    A = A_sym + A_sym.T  # Make symmetric
    
    print(f"Original matrix A (symmetric):")
    print(A)
    
    # Eigendecomposition
    eigenvals, eigenvecs = eigh(A)
    
    print(f"\nEigenvalues: {eigenvals}")
    print(f"Eigenvectors (columns):")
    print(eigenvecs)
    
    # Diagonalization: A = P D P^(-1)
    P = eigenvecs
    D = np.diag(eigenvals)
    P_inv = P.T  # For orthogonal matrix, inverse = transpose
    
    # Verify diagonalization
    A_reconstructed = P @ D @ P_inv
    
    print(f"\nDiagonalization verification:")
    print(f"P @ D @ P^(-1):")
    print(A_reconstructed)
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")
    
    # Different basis transformations
    bases = {
        "Standard": np.eye(3),
        "Rotated": np.array([[1, 1, 0],
                           [0, 1, 1],
                           [1, 0, 1]]) / np.sqrt(2),
        "Scaled": np.array([[2, 0, 0],
                          [0, 3, 0],
                          [0, 0, 0.5]])
    }
    
    print(f"\nMatrix representations in different bases:")
    
    for name, basis in bases.items():
        if name != "Standard":
            # Normalize basis vectors
            basis_norm = basis / np.linalg.norm(basis, axis=0)
            try:
                basis_inv = np.linalg.inv(basis_norm)
                A_new_basis = basis_inv @ A @ basis_norm
                
                print(f"\n{name} basis:")
                print(f"  Basis matrix:")
                print(basis_norm)
                print(f"  Matrix representation:")
                print(A_new_basis)
                print(f"  Condition number: {np.linalg.cond(A_new_basis):.2f}")
                
                # Verify invariants
                det_original = np.linalg.det(A)
                det_new = np.linalg.det(A_new_basis)
                trace_original = np.trace(A)
                trace_new = np.trace(A_new_basis)
                
                print(f"  Determinant preservation: {abs(det_original - det_new) < 1e-10}")
                print(f"  Trace preservation: {abs(trace_original - trace_new) < 1e-10}")
                
            except np.linalg.LinAlgError:
                print(f"\n{name} basis: Singular matrix, cannot invert")
    
    # Jordan normal form example (for non-diagonalizable matrix)
    print(f"\nJordan normal form example:")
    
    # Create a matrix with repeated eigenvalue but not enough eigenvectors
    J_example = np.array([[2, 1, 0],
                         [0, 2, 1],
                         [0, 0, 2]], dtype=float)
    
    print(f"Matrix with Jordan structure:")
    print(J_example)
    
    eigs_J, vecs_J = np.linalg.eig(J_example)
    print(f"Eigenvalues: {eigs_J}")
    print(f"Rank of (A - λI): {np.linalg.matrix_rank(J_example - 2*np.eye(3))}")
    print(f"Geometric multiplicity: {3 - np.linalg.matrix_rank(J_example - 2*np.eye(3))}")
    print(f"Algebraic multiplicity: 3")
    print(f"Diagonalizable: {np.linalg.matrix_rank(J_example - 2*np.eye(3)) == 0}")

if __name__ == "__main__":
    # Run all demonstrations
    T_std, T_new, P = demonstrate_change_of_basis_2d()
    print("\n" + "="*80 + "\n")
    
    P_pca, cov_orig, cov_pca = pca_as_change_of_basis()
    print("\n" + "="*80 + "\n")
    
    A_orig, A_precond, P_opt = optimization_preconditioning()
    print("\n" + "="*80 + "\n")
    
    W_nn, X_in, Z_hid = neural_network_layer_basis_change()
    print("\n" + "="*80 + "\n")
    
    advanced_similarity_transformations()
    
    print(f"\n{'='*60}")
    print("CHANGE OF BASIS SUMMARY")
    print(f"{'='*60}")
    
    print("\nFundamental Formula:")
    print("• For transformation T: V → W")
    print("• [T]_C'_B' = Q^(-1) [T]_C_B P")
    print("• P: input basis change, Q: output basis change")
    
    print("\nSimilarity Transformations (same space):")
    print("• [T]_B' = P^(-1) [T]_B P")
    print("• Preserves eigenvalues, determinant, trace, rank")
    print("• Changes eigenvectors, condition number, matrix entries")
    
    print("\nKey Applications:")
    print("• PCA: Change to principal component basis")
    print("• Diagonalization: Change to eigenvector basis")
    print("• Optimization: Preconditioning for better convergence")
    print("• Neural networks: Layer transformations between representations")
    print("• Graphics: Coordinate transformations and projections")
    
    print("\nComputational Benefits:")
    print("• Simplified calculations in appropriate basis")
    print("• Improved numerical stability")
    print("• Revealing geometric structure")
    print("• Optimizing algorithm performance")
    
    print("\nPractical Guidelines:")
    print("• Choose basis to reveal problem structure")
    print("• Consider computational efficiency")
    print("• Verify invariant properties for validation")
    print("• Understand geometric meaning of transformation")
```

**Theoretical Extensions:**

**1. Change of Basis for Multilinear Maps:**
For bilinear forms B: V × V → ℝ:
```
[B]_B' = P^T [B]_B P
```

**2. Tensor Transformations:**
For tensors of rank (m,n):
```
T'_ij = Σ_kl P^(-1)_ik Q_jl T_kl
```

**3. Functional Analysis:**
- Change of basis in infinite-dimensional spaces
- Fourier transforms as basis changes
- Wavelet bases for signal processing

**4. Differential Geometry:**
- Coordinate charts and transformations
- Metric tensors under coordinate changes
- Connection forms and curvature

**Advanced Applications:**
- **Computer Graphics**: Coordinate transformations, camera projections
- **Signal Processing**: Fourier, wavelet, and other transform bases
- **Quantum Mechanics**: Representation theory, measurement bases
- **Machine Learning**: Feature transformations, manifold learning
- **Control Theory**: State-space representations, canonical forms

---

## Question 27

**Describe the role of linear algebra in neural network computations.**

**Answer:** Linear algebra forms the mathematical foundation of neural networks, providing the computational framework for all major operations including forward propagation, backpropagation, weight updates, and optimization. Understanding these linear algebraic concepts is essential for neural network design, implementation, optimization, and theoretical analysis.

**Fundamental Neural Network Components:**

**1. Weight Matrices and Bias Vectors:**
- **Dense layers**: W ∈ ℝᵐˣⁿ transforms n-dimensional input to m-dimensional output
- **Bias terms**: b ∈ ℝᵐ provides translation/shift
- **Linear transformation**: z = Wx + b

**2. Activation Functions:**
- Applied element-wise: a = σ(z) where σ is activation function
- Common choices: ReLU, sigmoid, tanh, softmax

**3. Layer Composition:**
- Deep networks: f(x) = σₖ(Wₖσₖ₋₁(Wₖ₋₁...σ₁(W₁x + b₁)...))
- Each layer performs affine transformation followed by nonlinearity

**Forward Propagation:**

**Mathematical Framework:**
For layer l with input aˡ⁻¹ and output aˡ:
```
zˡ = Wˡaˡ⁻¹ + bˡ  (linear transformation)
aˡ = σ(zˡ)         (activation function)
```

**Batch Processing:**
For batch of m samples X ∈ ℝⁿˣᵐ:
```
Z = WX + b̃        (broadcasting bias)
A = σ(Z)
```

**Vectorization Benefits:**
- **Efficiency**: Single matrix operation vs loops
- **Parallelization**: GPU-optimized BLAS operations
- **Memory locality**: Better cache performance

**Backpropagation and Gradients:**

**Chain Rule in Matrix Form:**
For loss L and parameters W:
```
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
```

**Gradient Computation:**
- **Output layer**: δᴸ = ∇ₐL ⊙ σ'(zᴸ)
- **Hidden layers**: δˡ = ((Wˡ⁺¹)ᵀδˡ⁺¹) ⊙ σ'(zˡ)
- **Weight gradients**: ∂L/∂Wˡ = δˡ(aˡ⁻¹)ᵀ
- **Bias gradients**: ∂L/∂bˡ = δˡ

**Key Matrix Operations:**
- **Matrix multiplication**: Forward and backward passes
- **Hadamard product** (⊙): Element-wise multiplication for activation derivatives
- **Outer product**: Computing weight gradients
- **Transpose**: Propagating errors backward

**Optimization and Linear Algebra:**

**Gradient Descent:**
```
W ← W - η∇_W L  (parameter update)
```

**Advanced Optimizers:**
- **Momentum**: Vₜ = βVₜ₋₁ + (1-β)∇L, W ← W - ηVₜ
- **Adam**: Combines momentum and adaptive learning rates using first/second moments
- **Natural gradients**: Use Fisher information matrix for geometry-aware updates

**Second-Order Methods:**
- **Newton's method**: W ← W - H⁻¹∇L (H is Hessian)
- **Quasi-Newton**: Approximate Hessian with lower-rank updates
- **L-BFGS**: Limited-memory Broyden-Fletcher-Goldfarb-Shanno

**Specialized Neural Network Architectures:**

**1. Convolutional Neural Networks (CNNs):**
- **Convolution as matrix multiplication**: Toeplitz matrices
- **Im2col transformation**: Convert convolution to GEMM
- **Pooling operations**: Subsampling via matrix indexing

**2. Recurrent Neural Networks (RNNs):**
- **State evolution**: hₜ = σ(Wₕₕhₜ₋₁ + Wₓₕxₜ + bₕ)
- **LSTM/GRU**: Gating mechanisms using element-wise operations
- **Gradient flow**: Managing vanishing/exploding gradients through eigenvalue analysis

**3. Attention Mechanisms:**
- **Scaled dot-product**: Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
- **Multi-head attention**: Parallel projections and concatenation
- **Transformer architecture**: Self-attention via linear projections

**Normalization Techniques:**

**Batch Normalization:**
```
x̂ = (x - μ)/σ  (standardization)
y = γx̂ + β    (scale and shift)
```

**Layer Normalization:**
- Normalize across features rather than batch
- Important for RNNs and Transformers

**Group/Instance Normalization:**
- Variations for computer vision applications

**Weight Initialization and Linear Algebra:**

**Xavier/Glorot Initialization:**
```
W ~ N(0, 2/(nᵢₙ + nₒᵤₜ))
```

**He Initialization:**
```
W ~ N(0, 2/nᵢₙ)  (for ReLU networks)
```

**Orthogonal Initialization:**
- Initialize weights as orthogonal matrices
- Preserves gradient norms during backpropagation

**Matrix Factorization in Neural Networks:**

**Low-Rank Approximations:**
- Replace W ∈ ℝᵐˣⁿ with UV where U ∈ ℝᵐˣʳ, V ∈ ℝʳˣⁿ
- Reduces parameters from mn to r(m+n)

**Tensor Decompositions:**
- Tucker decomposition for convolutional layers
- CP decomposition for fully connected layers

**Implementation Examples:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, qr
import torch
import torch.nn as nn
import torch.optim as optim

class LinearAlgebraNN:
    """Neural network implementation highlighting linear algebra operations"""
    
    def __init__(self, layer_sizes):
        """
        Initialize neural network with given layer sizes
        
        Args:
            layer_sizes: List of integers [input_size, hidden1, hidden2, ..., output_size]
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            # Xavier initialization
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            
            W = np.random.uniform(-limit, limit, (fan_out, fan_in))
            b = np.zeros((fan_out, 1))
            
            self.weights.append(W)
            self.biases.append(b)
        
        print(f"Initialized neural network with architecture: {layer_sizes}")
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            print(f"  Layer {i+1}: W shape {W.shape}, b shape {b.shape}")
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def softmax(self, z):
        """Softmax activation (numerically stable)"""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward_propagation(self, X, verbose=False):
        """
        Forward propagation through the network
        
        Args:
            X: Input data (features × samples)
            verbose: Print intermediate shapes
        
        Returns:
            activations: List of activations for each layer
            z_values: List of pre-activation values
        """
        activations = [X]
        z_values = []
        
        current_activation = X
        
        if verbose:
            print(f"Forward propagation:")
            print(f"  Input shape: {X.shape}")
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation: z = Wx + b
            z = W @ current_activation + b
            z_values.append(z)
            
            if verbose:
                print(f"  Layer {i+1}:")
                print(f"    Weight matrix: {W.shape}")
                print(f"    Input: {current_activation.shape}")
                print(f"    z = Wx + b: {z.shape}")
            
            # Apply activation function
            if i == self.num_layers - 1:  # Output layer
                activation = self.softmax(z)
            else:  # Hidden layers
                activation = self.relu(z)
            
            activations.append(activation)
            current_activation = activation
            
            if verbose:
                print(f"    Activation: {activation.shape}")
        
        return activations, z_values
    
    def backward_propagation(self, X, y, activations, z_values, verbose=False):
        """
        Backward propagation to compute gradients
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            activations: Forward pass activations
            z_values: Forward pass pre-activations
            verbose: Print intermediate computations
        
        Returns:
            weight_gradients: List of weight gradients
            bias_gradients: List of bias gradients
        """
        m = X.shape[1]  # Number of samples
        
        weight_gradients = []
        bias_gradients = []
        
        if verbose:
            print(f"Backward propagation:")
        
        # Output layer error
        delta = activations[-1] - y  # For softmax + cross-entropy
        
        # Backpropagate through each layer
        for i in reversed(range(self.num_layers)):
            # Compute gradients for current layer
            dW = (1/m) * delta @ activations[i].T
            db = (1/m) * np.sum(delta, axis=1, keepdims=True)
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            if verbose:
                print(f"  Layer {i+1}:")
                print(f"    Delta shape: {delta.shape}")
                print(f"    Weight gradient: {dW.shape}")
                print(f"    Bias gradient: {db.shape}")
            
            # Propagate error to previous layer (if not input layer)
            if i > 0:
                # delta^(l-1) = (W^l)^T * delta^l ⊙ σ'(z^(l-1))
                delta = (self.weights[i].T @ delta) * self.relu_derivative(z_values[i-1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients, learning_rate):
        """Update parameters using gradient descent"""
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
    
    def compute_loss(self, activations, y):
        """Compute cross-entropy loss"""
        predictions = activations[-1]
        m = y.shape[1]
        
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        loss = -(1/m) * np.sum(y * np.log(predictions))
        return loss

def demonstrate_matrix_operations():
    """Demonstrate key matrix operations in neural networks"""
    
    print("="*60)
    print("NEURAL NETWORK MATRIX OPERATIONS")
    print("="*60)
    
    # Network architecture
    input_size, hidden_size, output_size = 4, 3, 2
    batch_size = 5
    
    # Initialize random weights and data
    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size) * 0.5
    b1 = np.random.randn(hidden_size, 1) * 0.1
    W2 = np.random.randn(output_size, hidden_size) * 0.5
    b2 = np.random.randn(output_size, 1) * 0.1
    
    X = np.random.randn(input_size, batch_size)
    
    print(f"Network architecture: {input_size} -> {hidden_size} -> {output_size}")
    print(f"Batch size: {batch_size}")
    print(f"\nWeight matrices:")
    print(f"  W1 shape: {W1.shape}")
    print(f"  W2 shape: {W2.shape}")
    print(f"  Input X shape: {X.shape}")
    
    # Forward pass operations
    print(f"\n1. FORWARD PASS MATRIX OPERATIONS:")
    
    # Layer 1
    print(f"\nLayer 1 computation:")
    print(f"  z1 = W1 @ X + b1")
    print(f"  {W1.shape} @ {X.shape} + {b1.shape}")
    
    z1 = W1 @ X + b1  # Broadcasting bias
    a1 = np.maximum(0, z1)  # ReLU activation
    
    print(f"  Result z1 shape: {z1.shape}")
    print(f"  Activation a1 shape: {a1.shape}")
    
    # Layer 2
    print(f"\nLayer 2 computation:")
    print(f"  z2 = W2 @ a1 + b2")
    print(f"  {W2.shape} @ {a1.shape} + {b2.shape}")
    
    z2 = W2 @ a1 + b2
    
    # Softmax (numerically stable)
    exp_z2 = np.exp(z2 - np.max(z2, axis=0, keepdims=True))
    a2 = exp_z2 / np.sum(exp_z2, axis=0, keepdims=True)
    
    print(f"  Result z2 shape: {z2.shape}")
    print(f"  Final output a2 shape: {a2.shape}")
    
    # Analyze computational complexity
    print(f"\n2. COMPUTATIONAL COMPLEXITY:")
    flops_layer1 = hidden_size * input_size * batch_size  # Matrix multiplication
    flops_layer2 = output_size * hidden_size * batch_size
    total_flops = flops_layer1 + flops_layer2
    
    print(f"  Layer 1 FLOPs: {flops_layer1}")
    print(f"  Layer 2 FLOPs: {flops_layer2}")
    print(f"  Total forward pass FLOPs: {total_flops}")
    
    # Memory usage
    memory_weights = (W1.size + W2.size) * 8  # 8 bytes per float64
    memory_activations = (X.size + z1.size + a1.size + z2.size + a2.size) * 8
    
    print(f"  Weight memory: {memory_weights} bytes")
    print(f"  Activation memory: {memory_activations} bytes")
    
    return W1, W2, b1, b2, X, a1, a2, z1, z2

def analyze_gradient_flow():
    """Analyze gradient flow using linear algebra"""
    
    print(f"\n{'='*60}")
    print("GRADIENT FLOW ANALYSIS")
    print(f"{'='*60}")
    
    # Create a deeper network to analyze gradient behavior
    layer_sizes = [10, 8, 6, 4, 2]
    nn = LinearAlgebraNN(layer_sizes)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(10, 50)  # 50 samples
    y = np.random.randint(0, 2, (2, 50))  # Binary classification, one-hot
    
    # Forward and backward pass
    activations, z_values = nn.forward_propagation(X)
    weight_gradients, bias_gradients = nn.backward_propagation(X, y, activations, z_values)
    
    print(f"Gradient analysis:")
    
    # Analyze gradient magnitudes
    for i, (dW, db) in enumerate(zip(weight_gradients, bias_gradients)):
        grad_norm_W = np.linalg.norm(dW)
        grad_norm_b = np.linalg.norm(db)
        
        print(f"  Layer {i+1}:")
        print(f"    Weight gradient norm: {grad_norm_W:.6f}")
        print(f"    Bias gradient norm: {grad_norm_b:.6f}")
        
        # Analyze weight matrix properties
        W = nn.weights[i]
        eigenvals = np.linalg.eigvals(W @ W.T)
        spectral_norm = np.sqrt(np.max(eigenvals))
        
        print(f"    Weight spectral norm: {spectral_norm:.6f}")
        print(f"    Weight condition number: {np.linalg.cond(W):.6f}")
    
    # Simulate gradient explosion/vanishing
    print(f"\nGradient flow simulation:")
    
    # Start with unit gradient
    gradient = np.ones((2, 1))
    
    print(f"  Initial gradient norm: {np.linalg.norm(gradient):.6f}")
    
    # Propagate backward through network
    for i in reversed(range(len(nn.weights))):
        W = nn.weights[i]
        
        # Simulate backward pass: gradient = W^T @ gradient
        gradient = W.T @ gradient
        grad_norm = np.linalg.norm(gradient)
        
        print(f"  After layer {i+1}: gradient norm = {grad_norm:.6f}")
        
        if grad_norm > 1000:
            print("    -> Gradient explosion detected!")
        elif grad_norm < 0.001:
            print("    -> Gradient vanishing detected!")

def matrix_factorization_compression():
    """Demonstrate neural network compression via matrix factorization"""
    
    print(f"\n{'='*60}")
    print("NEURAL NETWORK COMPRESSION VIA MATRIX FACTORIZATION")
    print(f"{'='*60}")
    
    # Create a large weight matrix
    m, n = 1000, 500  # Large fully connected layer
    np.random.seed(42)
    
    # Original weight matrix
    W_original = np.random.randn(m, n) * 0.1
    
    print(f"Original weight matrix:")
    print(f"  Shape: {W_original.shape}")
    print(f"  Parameters: {W_original.size}")
    print(f"  Memory (float32): {W_original.size * 4} bytes")
    
    # SVD factorization
    U, s, Vt = svd(W_original, full_matrices=False)
    
    print(f"\nSVD analysis:")
    print(f"  Singular values shape: {s.shape}")
    print(f"  Largest singular values: {s[:10]}")
    
    # Analyze rank and compression potential
    total_variance = np.sum(s**2)
    cumulative_variance = np.cumsum(s**2) / total_variance
    
    # Find rank for different compression ratios
    compression_ratios = [0.9, 0.95, 0.99]
    
    for ratio in compression_ratios:
        rank = np.argmax(cumulative_variance >= ratio) + 1
        
        # Low-rank approximation
        W_approx = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        
        # Compression statistics
        original_params = m * n
        compressed_params = rank * (m + n)
        compression_factor = original_params / compressed_params
        
        # Approximation error
        error = np.linalg.norm(W_original - W_approx, 'fro') / np.linalg.norm(W_original, 'fro')
        
        print(f"\n{ratio:.0%} variance preservation:")
        print(f"  Rank: {rank}")
        print(f"  Original parameters: {original_params}")
        print(f"  Compressed parameters: {compressed_params}")
        print(f"  Compression factor: {compression_factor:.2f}x")
        print(f"  Relative error: {error:.4f}")
    
    # Demonstrate factorized computation
    rank = 50  # Choose a moderate rank
    U_compressed = U[:, :rank]
    S_compressed = np.diag(s[:rank])
    Vt_compressed = Vt[:rank, :]
    
    print(f"\nFactorized computation example (rank {rank}):")
    print(f"  U shape: {U_compressed.shape}")
    print(f"  S shape: {S_compressed.shape}")
    print(f"  Vt shape: {Vt_compressed.shape}")
    
    # Test computation
    x = np.random.randn(n, 10)  # Batch of inputs
    
    # Original computation
    y_original = W_original @ x
    
    # Factorized computation: y = U * S * Vt * x
    y_factorized = U_compressed @ (S_compressed @ (Vt_compressed @ x))
    
    # Error in computation
    computation_error = np.linalg.norm(y_original - y_factorized) / np.linalg.norm(y_original)
    print(f"  Computation error: {computation_error:.6f}")

def attention_mechanism_linear_algebra():
    """Demonstrate attention mechanism using linear algebra"""
    
    print(f"\n{'='*60}")
    print("ATTENTION MECHANISM LINEAR ALGEBRA")
    print(f"{'='*60}")
    
    # Attention parameters
    seq_length = 8  # Sequence length
    d_model = 64    # Model dimension
    d_k = 16        # Key/Query dimension
    d_v = 16        # Value dimension
    n_heads = 4     # Number of attention heads
    
    np.random.seed(42)
    
    # Input sequence (batch_size=1 for simplicity)
    X = np.random.randn(seq_length, d_model)
    
    print(f"Attention mechanism setup:")
    print(f"  Sequence length: {seq_length}")
    print(f"  Model dimension: {d_model}")
    print(f"  Key/Query dimension: {d_k}")
    print(f"  Value dimension: {d_v}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Input shape: {X.shape}")
    
    # Linear projection matrices for single head
    W_q = np.random.randn(d_model, d_k) / np.sqrt(d_k)
    W_k = np.random.randn(d_model, d_k) / np.sqrt(d_k)
    W_v = np.random.randn(d_model, d_v) / np.sqrt(d_v)
    
    print(f"\nLinear projection matrices:")
    print(f"  W_q shape: {W_q.shape}")
    print(f"  W_k shape: {W_k.shape}")
    print(f"  W_v shape: {W_v.shape}")
    
    # Compute Q, K, V
    Q = X @ W_q  # Queries
    K = X @ W_k  # Keys
    V = X @ W_v  # Values
    
    print(f"\nQ, K, V computation:")
    print(f"  Q = X @ W_q: {X.shape} @ {W_q.shape} = {Q.shape}")
    print(f"  K = X @ W_k: {X.shape} @ {W_k.shape} = {K.shape}")
    print(f"  V = X @ W_v: {X.shape} @ {W_v.shape} = {V.shape}")
    
    # Attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    print(f"\nAttention scores:")
    print(f"  Scores = Q @ K^T / sqrt(d_k)")
    print(f"  {Q.shape} @ {K.T.shape} = {scores.shape}")
    print(f"  Scaling factor: 1/sqrt({d_k}) = {1/np.sqrt(d_k):.4f}")
    
    # Softmax attention weights
    attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
    
    print(f"\nAttention weights:")
    print(f"  Shape: {attention_weights.shape}")
    print(f"  Row sums (should be 1): {np.sum(attention_weights, axis=1)}")
    
    # Attention output
    output = attention_weights @ V
    
    print(f"\nAttention output:")
    print(f"  Output = Attention_weights @ V")
    print(f"  {attention_weights.shape} @ {V.shape} = {output.shape}")
    
    # Analyze attention pattern
    print(f"\nAttention pattern analysis:")
    
    # Self-attention strength (diagonal elements)
    self_attention = np.diag(attention_weights)
    print(f"  Self-attention strengths: {self_attention}")
    print(f"  Average self-attention: {np.mean(self_attention):.4f}")
    
    # Attention entropy (measure of focus)
    attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=1)
    print(f"  Attention entropy: {attention_entropy}")
    print(f"  Average entropy: {np.mean(attention_entropy):.4f}")
    
    # Multi-head attention
    print(f"\nMulti-head attention:")
    
    # Reshape for multi-head
    d_k_per_head = d_k // n_heads
    d_v_per_head = d_v // n_heads
    
    print(f"  Dimensions per head: d_k={d_k_per_head}, d_v={d_v_per_head}")
    
    # For demonstration, use same Q, K, V but reshape
    Q_multihead = Q.reshape(seq_length, n_heads, d_k_per_head)
    K_multihead = K.reshape(seq_length, n_heads, d_k_per_head)
    V_multihead = V.reshape(seq_length, n_heads, d_v_per_head)
    
    print(f"  Multi-head Q shape: {Q_multihead.shape}")
    print(f"  Multi-head K shape: {K_multihead.shape}")
    print(f"  Multi-head V shape: {V_multihead.shape}")
    
    # Parallel attention computation for all heads
    multihead_outputs = []
    
    for head in range(n_heads):
        Q_h = Q_multihead[:, head, :]
        K_h = K_multihead[:, head, :]
        V_h = V_multihead[:, head, :]
        
        # Attention for this head
        scores_h = Q_h @ K_h.T / np.sqrt(d_k_per_head)
        weights_h = np.exp(scores_h - np.max(scores_h, axis=1, keepdims=True))
        weights_h = weights_h / np.sum(weights_h, axis=1, keepdims=True)
        output_h = weights_h @ V_h
        
        multihead_outputs.append(output_h)
    
    # Concatenate heads
    concatenated_output = np.concatenate(multihead_outputs, axis=1)
    
    print(f"  Concatenated output shape: {concatenated_output.shape}")
    
    # Final linear projection
    W_o = np.random.randn(d_v, d_model) / np.sqrt(d_model)
    final_output = concatenated_output @ W_o
    
    print(f"  Final output shape: {final_output.shape}")
    
    return Q, K, V, attention_weights, output

def neural_network_training_demo():
    """Complete training demonstration with linear algebra focus"""
    
    print(f"\n{'='*60}")
    print("NEURAL NETWORK TRAINING DEMONSTRATION")
    print(f"{'='*60}")
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    # Generate data with some structure
    X = np.random.randn(n_features, n_samples)
    
    # Create class-dependent features
    true_W = np.random.randn(n_classes, n_features) * 0.5
    logits = true_W @ X + np.random.randn(n_classes, n_samples) * 0.1
    
    # Convert to one-hot labels
    y_labels = np.argmax(logits, axis=0)
    y = np.zeros((n_classes, n_samples))
    y[y_labels, np.arange(n_samples)] = 1
    
    print(f"Dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # Create neural network
    network_architecture = [n_features, 15, 10, n_classes]
    nn = LinearAlgebraNN(network_architecture)
    
    # Training parameters
    learning_rate = 0.01
    epochs = 100
    
    # Training loop
    losses = []
    accuracies = []
    
    print(f"\nTraining neural network:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    
    for epoch in range(epochs):
        # Forward pass
        activations, z_values = nn.forward_propagation(X)
        
        # Compute loss
        loss = nn.compute_loss(activations, y)
        losses.append(loss)
        
        # Compute accuracy
        predictions = np.argmax(activations[-1], axis=0)
        accuracy = np.mean(predictions == y_labels)
        accuracies.append(accuracy)
        
        # Backward pass
        weight_gradients, bias_gradients = nn.backward_propagation(X, y, activations, z_values)
        
        # Update parameters
        nn.update_parameters(weight_gradients, bias_gradients, learning_rate)
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")
    
    # Final performance
    final_activations, _ = nn.forward_propagation(X)
    final_predictions = np.argmax(final_activations[-1], axis=0)
    final_accuracy = np.mean(final_predictions == y_labels)
    
    print(f"\nFinal performance:")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Final accuracy: {final_accuracy:.4f}")
    
    # Analyze learned weights
    print(f"\nLearned weight analysis:")
    for i, W in enumerate(nn.weights):
        weight_norm = np.linalg.norm(W)
        weight_std = np.std(W)
        
        print(f"  Layer {i+1}:")
        print(f"    Weight matrix norm: {weight_norm:.4f}")
        print(f"    Weight standard deviation: {weight_std:.4f}")
        print(f"    Weight range: [{np.min(W):.4f}, {np.max(W):.4f}]")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neural_network_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return nn, losses, accuracies

if __name__ == "__main__":
    # Run all demonstrations
    W1, W2, b1, b2, X, a1, a2, z1, z2 = demonstrate_matrix_operations()
    print("\n" + "="*80 + "\n")
    
    analyze_gradient_flow()
    print("\n" + "="*80 + "\n")
    
    matrix_factorization_compression()
    print("\n" + "="*80 + "\n")
    
    Q, K, V, attn_weights, attn_output = attention_mechanism_linear_algebra()
    print("\n" + "="*80 + "\n")
    
    trained_nn, loss_history, acc_history = neural_network_training_demo()
    
    print(f"\n{'='*60}")
    print("LINEAR ALGEBRA IN NEURAL NETWORKS SUMMARY")
    print(f"{'='*60}")
    
    print("\nCore Operations:")
    print("• Matrix multiplication: Forward propagation (Wx + b)")
    print("• Element-wise operations: Activation functions, Hadamard products")
    print("• Transpose operations: Backpropagation error propagation")
    print("• Outer products: Computing weight gradients")
    
    print("\nKey Matrices:")
    print("• Weight matrices: Linear transformations between layers")
    print("• Jacobians: Gradients for backpropagation")
    print("• Hessians: Second-order optimization information")
    print("• Covariance matrices: Batch normalization, natural gradients")
    
    print("\nOptimization Aspects:")
    print("• Gradient descent: Vector operations in parameter space")
    print("• Momentum: Exponential moving averages")
    print("• Adam: First/second moment estimation")
    print("• Natural gradients: Fisher information matrix geometry")
    
    print("\nAdvanced Techniques:")
    print("• Matrix factorization: Network compression")
    print("• Attention mechanisms: Scaled dot-product operations")
    print("• Batch processing: Vectorized operations for efficiency")
    print("• Regularization: L1/L2 norms, spectral normalization")
    
    print("\nComputational Benefits:")
    print("• Vectorization: SIMD/GPU parallelization")
    print("• Memory efficiency: Contiguous memory access")
    print("• Numerical stability: Proper initialization and normalization")
    print("• Gradient flow: Eigenvalue analysis for training dynamics")
```

**Advanced Linear Algebra Concepts in Deep Learning:**

**1. Tensor Operations:**
- **Convolution**: Tensor contraction with learned kernels
- **Attention**: Multi-dimensional tensor manipulations
- **Broadcasting**: Efficient element-wise operations across different tensor shapes

**2. Optimization Geometry:**
- **Loss landscapes**: Understanding through eigenvalue analysis
- **Natural gradients**: Riemannian geometry for parameter updates
- **Information geometry**: Fisher information matrix for optimal learning

**3. Regularization Techniques:**
- **Spectral normalization**: Controlling largest singular value
- **Orthogonal regularization**: Encouraging orthogonal weight matrices
- **Nuclear norm**: Promoting low-rank solutions

**4. Advanced Architectures:**
- **Residual connections**: Identity mappings and gradient flow
- **Normalization layers**: Statistical transformations
- **Graph neural networks**: Adjacency matrix operations

**Practical Implementation Guidelines:**
- Use efficient BLAS libraries (cuBLAS, MKL) for matrix operations
- Implement proper numerical stability measures
- Consider memory layout for optimal cache performance
- Leverage automatic differentiation frameworks for gradient computation

---

## Question 28

**Explain how the SVD is used in recommendation systems.**

**Answer:** Singular Value Decomposition (SVD) is a fundamental technique in recommendation systems that enables collaborative filtering by identifying latent factors that capture user preferences and item characteristics. SVD transforms high-dimensional, sparse user-item interaction data into lower-dimensional representations that reveal hidden patterns, making it possible to predict user preferences for unseen items and handle the sparsity challenge inherent in recommendation systems.

**Mathematical Foundation:**

**User-Item Rating Matrix:**
Consider rating matrix R ∈ ℝᵐˣⁿ where:
- m = number of users
- n = number of items  
- R[i,j] = rating given by user i to item j (many entries missing)

**SVD Decomposition:**
```
R ≈ UΣVᵀ
```
Where:
- U ∈ ℝᵐˣᵏ: User factor matrix (user latent features)
- Σ ∈ ℝᵏˣᵏ: Diagonal matrix of singular values (importance weights)
- V ∈ ℝⁿˣᵏ: Item factor matrix (item latent features)
- k: Number of latent factors (k << min(m,n))

**Interpretation of Factors:**

**User Matrix U:**
- Each row uᵢ represents user i in latent factor space
- Captures user preferences across different dimensions (genres, styles, etc.)
- Example dimensions: comedy preference, action preference, romance preference

**Item Matrix V:**
- Each row vⱼ represents item j in latent factor space
- Captures item characteristics in same latent dimensions
- Example: how much comedy/action/romance content an item has

**Singular Values Σ:**
- Represent importance of each latent factor
- Larger values indicate more significant patterns
- Enable dimensionality reduction by keeping top-k factors

**Recommendation Generation:**

**Rating Prediction:**
For user i and item j:
```
r̂ᵢⱼ = uᵢᵀvⱼ = Σₗ₌₁ᵏ uᵢₗσₗvⱼₗ
```

**Top-N Recommendations:**
1. Compute predicted ratings for all unrated items
2. Sort by predicted rating
3. Return top-N highest-rated items

**Handling Sparsity:**

**Traditional SVD Problem:**
- Direct SVD requires complete matrix
- Rating matrices are typically 90%+ sparse
- Missing values cannot be ignored

**Solutions:**

**1. Matrix Completion Approaches:**
- **Mean filling**: Replace missing values with user/item means
- **Matrix factorization**: Iterative optimization
- **Weighted SVD**: Give zero weight to missing entries

**2. Regularized Matrix Factorization:**
Minimize:
```
L = Σᵢⱼ∈Ω (rᵢⱼ - uᵢᵀvⱼ)² + λ(||U||²F + ||V||²F)
```
Where Ω is the set of observed ratings.

**Implementation Variants:**

**1. Truncated SVD:**
- Use only top-k singular values/vectors
- Reduces noise and computational complexity
- Captures most important latent factors

**2. Incremental SVD:**
- Update decomposition as new ratings arrive
- Avoid recomputing entire SVD
- Suitable for online systems

**3. Probabilistic Matrix Factorization (PMF):**
- Bayesian approach to matrix factorization
- Models uncertainty in predictions
- Incorporates prior knowledge

**Advanced SVD Techniques:**

**1. SVD++:**
Incorporates implicit feedback:
```
r̂ᵢⱼ = μ + bᵢ + bⱼ + (pᵢ + |I(i)|⁻⁰·⁵ Σₖ∈I(i) yₖ)ᵀqⱼ
```
Where I(i) is the set of items rated by user i.

**2. Temporal SVD:**
- Accounts for time-evolving preferences
- Models concept drift in user behavior
- Time-dependent factors

**3. Non-negative Matrix Factorization (NMF):**
- Constrains factors to be non-negative
- More interpretable factors
- Natural for rating data

**Implementation Examples:**

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class SVDRecommendationSystem:
    """SVD-based recommendation system implementation"""
    
    def __init__(self, n_factors=50, regularization=0.01):
        """
        Initialize SVD recommendation system
        
        Args:
            n_factors: Number of latent factors
            regularization: Regularization parameter for matrix factorization
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.user_factors = None
        self.item_factors = None
        self.singular_values = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        
    def create_rating_matrix(self, ratings_df):
        """
        Create user-item rating matrix from ratings dataframe
        
        Args:
            ratings_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        
        Returns:
            rating_matrix: Sparse matrix of ratings
            user_encoder: Mapping from user_id to matrix index
            item_encoder: Mapping from item_id to matrix index
        """
        # Create mappings
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()
        
        user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        item_encoder = {item: idx for idx, item in enumerate(unique_items)}
        
        # Create rating matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        rating_matrix = np.zeros((n_users, n_items))
        
        for _, row in ratings_df.iterrows():
            user_idx = user_encoder[row['user_id']]
            item_idx = item_encoder[row['item_id']]
            rating_matrix[user_idx, item_idx] = row['rating']
        
        return rating_matrix, user_encoder, item_encoder
    
    def fit_basic_svd(self, rating_matrix, verbose=True):
        """
        Fit SVD model using basic approach with mean filling
        
        Args:
            rating_matrix: User-item rating matrix
            verbose: Print fitting information
        """
        if verbose:
            print("Fitting basic SVD model...")
            print(f"Rating matrix shape: {rating_matrix.shape}")
            print(f"Sparsity: {np.count_nonzero(rating_matrix) / rating_matrix.size:.1%}")
        
        # Handle missing values by filling with user means
        filled_matrix = rating_matrix.copy()
        
        # Calculate global mean
        nonzero_mask = rating_matrix > 0
        self.global_mean = np.mean(rating_matrix[nonzero_mask])
        
        # Fill missing values with user means (fallback to global mean)
        for i in range(rating_matrix.shape[0]):
            user_ratings = rating_matrix[i, rating_matrix[i] > 0]
            if len(user_ratings) > 0:
                user_mean = np.mean(user_ratings)
            else:
                user_mean = self.global_mean
            
            filled_matrix[i, rating_matrix[i] == 0] = user_mean
        
        # Perform SVD
        U, s, Vt = np.linalg.svd(filled_matrix, full_matrices=False)
        
        # Keep only top n_factors
        self.user_factors = U[:, :self.n_factors]
        self.singular_values = s[:self.n_factors]
        self.item_factors = Vt[:self.n_factors, :].T
        
        if verbose:
            print(f"User factors shape: {self.user_factors.shape}")
            print(f"Item factors shape: {self.item_factors.shape}")
            print(f"Singular values: {self.singular_values[:10]}")
            
            # Explained variance
            total_variance = np.sum(s**2)
            explained_variance = np.sum(self.singular_values**2) / total_variance
            print(f"Explained variance: {explained_variance:.1%}")
    
    def fit_matrix_factorization(self, rating_matrix, max_iterations=100, learning_rate=0.01, verbose=True):
        """
        Fit SVD model using matrix factorization with gradient descent
        
        Args:
            rating_matrix: User-item rating matrix
            max_iterations: Maximum number of iterations
            learning_rate: Learning rate for gradient descent
            verbose: Print training information
        """
        if verbose:
            print("Fitting matrix factorization model...")
        
        n_users, n_items = rating_matrix.shape
        
        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Initialize biases
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        # Calculate global mean
        nonzero_mask = rating_matrix > 0
        self.global_mean = np.mean(rating_matrix[nonzero_mask])
        
        # Get observed ratings
        observed_indices = np.where(rating_matrix > 0)
        observed_ratings = rating_matrix[observed_indices]
        
        losses = []
        
        for iteration in range(max_iterations):
            total_loss = 0
            
            # Shuffle training data
            perm = np.random.permutation(len(observed_ratings))
            
            for idx in perm:
                i, j = observed_indices[0][idx], observed_indices[1][idx]
                rating = rating_matrix[i, j]
                
                # Predict rating
                prediction = (self.global_mean + 
                            self.user_bias[i] + 
                            self.item_bias[j] + 
                            np.dot(self.user_factors[i], self.item_factors[j]))
                
                # Calculate error
                error = rating - prediction
                total_loss += error**2
                
                # Update parameters using gradient descent
                # Gradients
                user_factor_grad = -2 * error * self.item_factors[j] + 2 * self.regularization * self.user_factors[i]
                item_factor_grad = -2 * error * self.user_factors[i] + 2 * self.regularization * self.item_factors[j]
                user_bias_grad = -2 * error + 2 * self.regularization * self.user_bias[i]
                item_bias_grad = -2 * error + 2 * self.regularization * self.item_bias[j]
                
                # Updates
                self.user_factors[i] -= learning_rate * user_factor_grad
                self.item_factors[j] -= learning_rate * item_factor_grad
                self.user_bias[i] -= learning_rate * user_bias_grad
                self.item_bias[j] -= learning_rate * item_bias_grad
            
            # Add regularization to loss
            total_loss += self.regularization * (
                np.sum(self.user_factors**2) + 
                np.sum(self.item_factors**2) + 
                np.sum(self.user_bias**2) + 
                np.sum(self.item_bias**2)
            )
            
            losses.append(total_loss)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Loss = {total_loss:.6f}")
        
        return losses
    
    def predict(self, user_indices, item_indices, method='basic'):
        """
        Predict ratings for given user-item pairs
        
        Args:
            user_indices: Array of user indices
            item_indices: Array of item indices
            method: 'basic' or 'matrix_factorization'
        
        Returns:
            predictions: Predicted ratings
        """
        if method == 'basic':
            predictions = np.sum(
                self.user_factors[user_indices] * 
                (self.singular_values * self.item_factors[item_indices].T).T, 
                axis=1
            )
        else:  # matrix_factorization
            predictions = (
                self.global_mean +
                self.user_bias[user_indices] +
                self.item_bias[item_indices] +
                np.sum(self.user_factors[user_indices] * self.item_factors[item_indices], axis=1)
            )
        
        return predictions
    
    def recommend_items(self, user_idx, rating_matrix, n_recommendations=10, method='basic'):
        """
        Recommend top-N items for a user
        
        Args:
            user_idx: User index
            rating_matrix: Original rating matrix
            n_recommendations: Number of recommendations
            method: Prediction method
        
        Returns:
            recommendations: List of (item_idx, predicted_rating) tuples
        """
        # Find items not rated by the user
        unrated_items = np.where(rating_matrix[user_idx] == 0)[0]
        
        if len(unrated_items) == 0:
            return []
        
        # Predict ratings for unrated items
        user_indices = np.full(len(unrated_items), user_idx)
        predictions = self.predict(user_indices, unrated_items, method)
        
        # Sort by predicted rating
        sorted_indices = np.argsort(predictions)[::-1]
        
        # Return top-N recommendations
        recommendations = [
            (unrated_items[idx], predictions[idx]) 
            for idx in sorted_indices[:n_recommendations]
        ]
        
        return recommendations

def generate_synthetic_ratings():
    """Generate synthetic movie rating dataset"""
    
    np.random.seed(42)
    
    # Dataset parameters
    n_users = 1000
    n_items = 500
    n_ratings = 50000  # Sparse: ~10% density
    
    # Generate user and item factors for ground truth
    n_true_factors = 10
    true_user_factors = np.random.normal(0, 1, (n_users, n_true_factors))
    true_item_factors = np.random.normal(0, 1, (n_items, n_true_factors))
    
    # Generate ratings based on latent factors
    ratings_data = []
    
    for _ in range(n_ratings):
        user_id = np.random.randint(0, n_users)
        item_id = np.random.randint(0, n_items)
        
        # Generate rating based on user-item interaction
        true_rating = np.dot(true_user_factors[user_id], true_item_factors[item_id])
        
        # Add noise and clip to rating scale
        noisy_rating = true_rating + np.random.normal(0, 0.5)
        rating = np.clip(noisy_rating, 1, 5)  # 1-5 rating scale
        
        ratings_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })
    
    # Remove duplicates (keep last rating)
    ratings_df = pd.DataFrame(ratings_data)
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    
    print(f"Generated synthetic ratings dataset:")
    print(f"  Users: {n_users}")
    print(f"  Items: {n_items}")
    print(f"  Ratings: {len(ratings_df)}")
    print(f"  Density: {len(ratings_df) / (n_users * n_items):.1%}")
    print(f"  Rating range: [{ratings_df['rating'].min():.1f}, {ratings_df['rating'].max():.1f}]")
    
    return ratings_df

def evaluate_recommendation_system():
    """Comprehensive evaluation of SVD recommendation system"""
    
    print("="*60)
    print("SVD RECOMMENDATION SYSTEM EVALUATION")
    print("="*60)
    
    # Generate data
    ratings_df = generate_synthetic_ratings()
    
    # Create recommendation system
    svd_system = SVDRecommendationSystem(n_factors=50, regularization=0.01)
    
    # Create rating matrix
    rating_matrix, user_encoder, item_encoder = svd_system.create_rating_matrix(ratings_df)
    
    print(f"\nRating matrix analysis:")
    print(f"  Shape: {rating_matrix.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(rating_matrix)}")
    print(f"  Sparsity: {1 - np.count_nonzero(rating_matrix) / rating_matrix.size:.1%}")
    
    # Split data for evaluation
    train_mask = np.random.random(rating_matrix.shape) < 0.8
    test_mask = (rating_matrix > 0) & (~train_mask)
    
    train_matrix = rating_matrix * train_mask
    test_ratings = rating_matrix[test_mask]
    test_indices = np.where(test_mask)
    
    print(f"\nTrain/test split:")
    print(f"  Train ratings: {np.count_nonzero(train_matrix)}")
    print(f"  Test ratings: {len(test_ratings)}")
    
    # Method 1: Basic SVD
    print(f"\n1. BASIC SVD EVALUATION:")
    
    svd_basic = SVDRecommendationSystem(n_factors=50)
    svd_basic.fit_basic_svd(train_matrix)
    
    # Predict test ratings
    predictions_basic = svd_basic.predict(test_indices[0], test_indices[1], method='basic')
    rmse_basic = np.sqrt(mean_squared_error(test_ratings, predictions_basic))
    
    print(f"  RMSE: {rmse_basic:.4f}")
    
    # Method 2: Matrix Factorization
    print(f"\n2. MATRIX FACTORIZATION EVALUATION:")
    
    svd_mf = SVDRecommendationSystem(n_factors=50, regularization=0.01)
    losses = svd_mf.fit_matrix_factorization(train_matrix, max_iterations=50, learning_rate=0.01)
    
    # Predict test ratings
    predictions_mf = svd_mf.predict(test_indices[0], test_indices[1], method='matrix_factorization')
    rmse_mf = np.sqrt(mean_squared_error(test_ratings, predictions_mf))
    
    print(f"  RMSE: {rmse_mf:.4f}")
    
    # Analyze factor interpretation
    print(f"\n3. FACTOR ANALYSIS:")
    
    # User factor analysis
    user_factors = svd_mf.user_factors
    print(f"  User factors shape: {user_factors.shape}")
    print(f"  User factor norms: mean={np.mean(np.linalg.norm(user_factors, axis=1)):.4f}")
    
    # Item factor analysis  
    item_factors = svd_mf.item_factors
    print(f"  Item factors shape: {item_factors.shape}")
    print(f"  Item factor norms: mean={np.mean(np.linalg.norm(item_factors, axis=1)):.4f}")
    
    # Factor correlation
    factor_corr = np.corrcoef(user_factors.T)
    print(f"  User factor correlation: mean abs correlation = {np.mean(np.abs(factor_corr - np.eye(factor_corr.shape[0]))):.4f}")
    
    # Recommendation examples
    print(f"\n4. RECOMMENDATION EXAMPLES:")
    
    # Pick a random user with sufficient ratings
    user_rating_counts = np.sum(rating_matrix > 0, axis=1)
    active_users = np.where(user_rating_counts >= 20)[0]
    
    if len(active_users) > 0:
        example_user = active_users[0]
        user_ratings = rating_matrix[example_user]
        rated_items = np.where(user_ratings > 0)[0]
        
        print(f"  Example user {example_user}:")
        print(f"    Rated {len(rated_items)} items")
        print(f"    Average rating: {np.mean(user_ratings[rated_items]):.2f}")
        
        # Get recommendations
        recommendations = svd_mf.recommend_items(
            example_user, rating_matrix, n_recommendations=10, method='matrix_factorization'
        )
        
        print(f"    Top 5 recommendations:")
        for i, (item_idx, pred_rating) in enumerate(recommendations[:5]):
            print(f"      {i+1}. Item {item_idx}: predicted rating {pred_rating:.2f}")
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Training loss
    plt.subplot(2, 3, 1)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Matrix Factorization Training Loss')
    plt.grid(True, alpha=0.3)
    
    # Prediction scatter plot
    plt.subplot(2, 3, 2)
    plt.scatter(test_ratings, predictions_mf, alpha=0.5, s=1)
    plt.plot([1, 5], [1, 5], 'r--', alpha=0.8)
    plt.xlabel('True Rating')
    plt.ylabel('Predicted Rating')
    plt.title('True vs Predicted Ratings')
    plt.grid(True, alpha=0.3)
    
    # Rating distribution
    plt.subplot(2, 3, 3)
    plt.hist(test_ratings, bins=20, alpha=0.7, label='True', density=True)
    plt.hist(predictions_mf, bins=20, alpha=0.7, label='Predicted', density=True)
    plt.xlabel('Rating')
    plt.ylabel('Density')
    plt.title('Rating Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # User factor heatmap
    plt.subplot(2, 3, 4)
    sample_users = np.random.choice(user_factors.shape[0], min(50, user_factors.shape[0]), replace=False)
    sns.heatmap(user_factors[sample_users, :10], cmap='coolwarm', center=0, cbar=True)
    plt.xlabel('Latent Factors')
    plt.ylabel('Users (sample)')
    plt.title('User Latent Factors')
    
    # Item factor heatmap
    plt.subplot(2, 3, 5)
    sample_items = np.random.choice(item_factors.shape[0], min(50, item_factors.shape[0]), replace=False)
    sns.heatmap(item_factors[sample_items, :10], cmap='coolwarm', center=0, cbar=True)
    plt.xlabel('Latent Factors')
    plt.ylabel('Items (sample)')
    plt.title('Item Latent Factors')
    
    # Singular values
    plt.subplot(2, 3, 6)
    if hasattr(svd_basic, 'singular_values'):
        plt.plot(svd_basic.singular_values[:20], 'o-')
        plt.xlabel('Factor Index')
        plt.ylabel('Singular Value')
        plt.title('Singular Values (Basic SVD)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svd_recommendation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return svd_mf, rating_matrix, rmse_basic, rmse_mf

def cold_start_analysis():
    """Analyze cold start problem and potential solutions"""
    
    print(f"\n{'='*60}")
    print("COLD START PROBLEM ANALYSIS")
    print(f"{'='*60}")
    
    # Generate data with varying user activity levels
    np.random.seed(42)
    
    n_users = 500
    n_items = 200
    
    # Create users with different activity levels
    # 20% heavy users, 50% medium users, 30% light users
    user_types = np.random.choice(['heavy', 'medium', 'light'], n_users, p=[0.2, 0.5, 0.3])
    
    ratings_data = []
    
    for user_id in range(n_users):
        user_type = user_types[user_id]
        
        # Determine number of ratings based on user type
        if user_type == 'heavy':
            n_user_ratings = np.random.poisson(50)
        elif user_type == 'medium':
            n_user_ratings = np.random.poisson(15)
        else:  # light
            n_user_ratings = np.random.poisson(3)
        
        # Generate ratings for this user
        for _ in range(n_user_ratings):
            item_id = np.random.randint(0, n_items)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
            
            ratings_data.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'user_type': user_type
            })
    
    ratings_df = pd.DataFrame(ratings_data)
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    
    print(f"Cold start analysis dataset:")
    print(f"  Users: {n_users}")
    print(f"  Items: {n_items}")
    print(f"  Ratings: {len(ratings_df)}")
    
    # Analyze user activity distribution
    user_activity = ratings_df.groupby('user_id').size()
    
    print(f"\nUser activity distribution:")
    for user_type in ['heavy', 'medium', 'light']:
        type_users = ratings_df[ratings_df['user_type'] == user_type]['user_id'].unique()
        type_activity = user_activity[type_users]
        print(f"  {user_type.capitalize()} users: mean={type_activity.mean():.1f}, std={type_activity.std():.1f}")
    
    # Create recommendation system
    svd_system = SVDRecommendationSystem(n_factors=20, regularization=0.05)
    rating_matrix, user_encoder, item_encoder = svd_system.create_rating_matrix(ratings_df)
    
    # Identify cold start users (users with very few ratings)
    cold_start_threshold = 5
    user_rating_counts = np.sum(rating_matrix > 0, axis=1)
    cold_start_users = np.where(user_rating_counts <= cold_start_threshold)[0]
    warm_users = np.where(user_rating_counts > cold_start_threshold)[0]
    
    print(f"\nCold start analysis:")
    print(f"  Cold start users ({cold_start_threshold} ratings or fewer): {len(cold_start_users)}")
    print(f"  Warm users (more than {cold_start_threshold} ratings): {len(warm_users)}")
    
    # Train model
    svd_system.fit_matrix_factorization(rating_matrix, max_iterations=30, verbose=False)
    
    # Evaluate performance for different user types
    print(f"\nRecommendation performance by user type:")
    
    for user_group, group_name in [(warm_users, 'Warm'), (cold_start_users, 'Cold Start')]:
        if len(user_group) == 0:
            continue
        
        # Sample some users from this group
        sample_users = user_group[:min(100, len(user_group))]
        
        # Get recommendations for these users
        group_performance = []
        
        for user_idx in sample_users:
            user_ratings = rating_matrix[user_idx]
            rated_items = np.where(user_ratings > 0)[0]
            
            if len(rated_items) < 2:  # Need at least 2 ratings for meaningful evaluation
                continue
            
            # Leave-one-out evaluation
            test_item = rated_items[-1]  # Use last rated item as test
            true_rating = user_ratings[test_item]
            
            # Temporarily remove test rating
            temp_matrix = rating_matrix.copy()
            temp_matrix[user_idx, test_item] = 0
            
            # Predict rating
            predicted_rating = svd_system.predict([user_idx], [test_item], method='matrix_factorization')[0]
            
            group_performance.append(abs(true_rating - predicted_rating))
        
        if group_performance:
            mae = np.mean(group_performance)
            print(f"  {group_name} users: MAE = {mae:.4f} (n={len(group_performance)})")
    
    # Content-based fallback for cold start
    print(f"\nContent-based fallback strategy:")
    
    # Simulate item features (e.g., genre, director, year)
    n_features = 10
    item_features = np.random.randn(n_items, n_features)
    
    # For cold start users, recommend popular items in preferred categories
    # Based on their few ratings, find preferred item features
    
    cold_start_example = cold_start_users[0] if len(cold_start_users) > 0 else None
    
    if cold_start_example is not None:
        user_ratings = rating_matrix[cold_start_example]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) > 0:
            # Compute user feature preferences
            user_features = np.zeros(n_features)
            for item_idx in rated_items:
                rating = user_ratings[item_idx]
                user_features += (rating - 3) * item_features[item_idx]  # Center around 3
            
            user_features /= len(rated_items)
            
            # Recommend items similar to user preferences
            unrated_items = np.where(user_ratings == 0)[0]
            item_scores = item_features[unrated_items] @ user_features
            
            top_items = unrated_items[np.argsort(item_scores)[::-1][:5]]
            
            print(f"  Cold start user {cold_start_example}:")
            print(f"    Rated {len(rated_items)} items")
            print(f"    Content-based recommendations: {top_items}")

def advanced_svd_techniques():
    """Demonstrate advanced SVD techniques for recommendations"""
    
    print(f"\n{'='*60}")
    print("ADVANCED SVD TECHNIQUES")
    print(f"{'='*60}")
    
    # 1. SVD with temporal dynamics
    print("1. TEMPORAL SVD:")
    
    # Simulate temporal rating data
    np.random.seed(42)
    n_users, n_items = 200, 100
    n_time_steps = 10
    
    # Generate time-evolving user preferences
    base_user_factors = np.random.normal(0, 1, (n_users, 5))
    item_factors = np.random.normal(0, 1, (n_items, 5))
    
    temporal_ratings = []
    
    for t in range(n_time_steps):
        # User preferences drift over time
        time_drift = 0.1 * t * np.random.normal(0, 0.1, (n_users, 5))
        current_user_factors = base_user_factors + time_drift
        
        # Generate ratings for this time step
        for _ in range(500):  # 500 ratings per time step
            user_id = np.random.randint(0, n_users)
            item_id = np.random.randint(0, n_items)
            
            base_rating = np.dot(current_user_factors[user_id], item_factors[item_id])
            rating = np.clip(base_rating + np.random.normal(0, 0.5), 1, 5)
            
            temporal_ratings.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': t
            })
    
    temporal_df = pd.DataFrame(temporal_ratings)
    
    print(f"  Generated temporal data: {len(temporal_df)} ratings over {n_time_steps} time steps")
    
    # Analyze temporal patterns
    temporal_stats = temporal_df.groupby('timestamp')['rating'].agg(['mean', 'std', 'count'])
    print(f"  Rating evolution:")
    for t in range(n_time_steps):
        stats = temporal_stats.loc[t]
        print(f"    Time {t}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, count={stats['count']}")
    
    # 2. Matrix factorization with side information
    print(f"\n2. MATRIX FACTORIZATION WITH SIDE INFORMATION:")
    
    # Simulate user demographics and item features
    user_demographics = np.random.randn(n_users, 3)  # age, gender, location
    item_features = np.random.randn(n_items, 4)      # genre, year, budget, duration
    
    print(f"  User demographics shape: {user_demographics.shape}")
    print(f"  Item features shape: {item_features.shape}")
    
    # Enhanced matrix factorization that incorporates side information
    # R ≈ (U + X_u W_u)(V + X_v W_v)^T
    # where X_u, X_v are side information matrices
    
    class EnhancedSVD:
        def __init__(self, n_factors=10, n_demographic_factors=2, n_content_factors=2):
            self.n_factors = n_factors
            self.n_demographic_factors = n_demographic_factors
            self.n_content_factors = n_content_factors
        
        def fit(self, rating_matrix, user_features, item_features, verbose=True):
            n_users, n_items = rating_matrix.shape
            
            # Initialize factors
            self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
            self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
            
            # Side information projection matrices
            self.user_feature_weights = np.random.normal(0, 0.1, (user_features.shape[1], self.n_demographic_factors))
            self.item_feature_weights = np.random.normal(0, 0.1, (item_features.shape[1], self.n_content_factors))
            
            if verbose:
                print(f"    Initialized enhanced SVD with {self.n_factors} main factors")
                print(f"    Demographic factors: {self.n_demographic_factors}")
                print(f"    Content factors: {self.n_content_factors}")
        
        def predict(self, user_indices, item_indices, user_features, item_features):
            # Main collaborative filtering component
            cf_prediction = np.sum(
                self.user_factors[user_indices] * self.item_factors[item_indices], 
                axis=1
            )
            
            # Demographic component
            demo_factors = user_features[user_indices] @ self.user_feature_weights
            demo_prediction = np.sum(demo_factors * self.item_factors[item_indices, :self.n_demographic_factors], axis=1)
            
            # Content component  
            content_factors = item_features[item_indices] @ self.item_feature_weights
            content_prediction = np.sum(self.user_factors[user_indices, :self.n_content_factors] * content_factors, axis=1)
            
            return cf_prediction + 0.1 * demo_prediction + 0.1 * content_prediction
    
    enhanced_svd = EnhancedSVD()
    
    # Create sample rating matrix
    sample_ratings = temporal_df[temporal_df['timestamp'] < 5]  # Use first half of data
    rating_matrix, user_enc, item_enc = SVDRecommendationSystem(n_factors=10).create_rating_matrix(sample_ratings)
    
    enhanced_svd.fit(rating_matrix, user_demographics, item_features)
    
    # 3. Non-negative matrix factorization
    print(f"\n3. NON-NEGATIVE MATRIX FACTORIZATION:")
    
    from sklearn.decomposition import NMF
    
    # NMF requires non-negative data
    # Shift ratings to be non-negative
    rating_matrix_positive = rating_matrix.copy()
    rating_matrix_positive[rating_matrix_positive > 0] -= 1  # Shift 1-5 scale to 0-4
    
    # Apply NMF
    nmf = NMF(n_components=20, init='random', random_state=42, max_iter=100)
    
    # Only fit on observed ratings (approximate approach)
    filled_matrix = rating_matrix_positive.copy()
    filled_matrix[filled_matrix == 0] = np.mean(filled_matrix[filled_matrix > 0])
    
    user_factors_nmf = nmf.fit_transform(filled_matrix)
    item_factors_nmf = nmf.components_.T
    
    print(f"  NMF reconstruction error: {nmf.reconstruction_err_:.4f}")
    print(f"  User factors shape: {user_factors_nmf.shape}")
    print(f"  Item factors shape: {item_factors_nmf.shape}")
    
    # Analyze non-negative factors
    print(f"  User factors - all non-negative: {np.all(user_factors_nmf >= 0)}")
    print(f"  Item factors - all non-negative: {np.all(item_factors_nmf >= 0)}")
    
    # Interpret factors (find most characteristic items for each factor)
    print(f"  Factor interpretation (top items per factor):")
    for factor_idx in range(min(5, item_factors_nmf.shape[1])):
        top_items = np.argsort(item_factors_nmf[:, factor_idx])[::-1][:3]
        factor_strength = item_factors_nmf[top_items, factor_idx]
        print(f"    Factor {factor_idx}: items {top_items} (strengths: {factor_strength})")

if __name__ == "__main__":
    # Run comprehensive evaluation
    svd_model, rating_data, rmse_basic, rmse_mf = evaluate_recommendation_system()
    print("\n" + "="*80 + "\n")
    
    # Cold start analysis
    cold_start_analysis()
    print("\n" + "="*80 + "\n")
    
    # Advanced techniques
    advanced_svd_techniques()
    
    print(f"\n{'='*60}")
    print("SVD RECOMMENDATION SYSTEMS SUMMARY")
    print(f"{'='*60}")
    
    print("\nCore SVD Approach:")
    print("• Decompose rating matrix: R ≈ UΣV^T")
    print("• U: User latent factors, V: Item latent factors")
    print("• Predict ratings: r̂_ij = u_i^T v_j")
    print("• Handle sparsity through low-rank approximation")
    
    print("\nKey Advantages:")
    print("• Dimensionality reduction handles sparsity")
    print("• Captures latent user preferences and item characteristics")
    print("• Efficient computation for large-scale systems")
    print("• Mathematical foundation enables extensions")
    
    print("\nChallenges and Solutions:")
    print("• Sparsity: Matrix factorization with regularization")
    print("• Cold start: Content-based features, demographic info")
    print("• Scalability: Incremental SVD, parallel computation")
    print("• Temporal dynamics: Time-aware factorization")
    
    print("\nAdvanced Techniques:")
    print("• SVD++: Incorporating implicit feedback")
    print("• Temporal SVD: Time-evolving preferences")
    print("• Side information: User/item features integration")
    print("• Non-negative factorization: Interpretable factors")
    
    print("\nPractical Implementation:")
    print("• Regularization for overfitting prevention")
    print("• Bias terms for baseline effects")
    print("• Cross-validation for hyperparameter tuning")
    print("• Evaluation metrics: RMSE, MAE, precision@K, recall@K")
```

**Practical Considerations:**

**1. Scalability:**
- **Distributed SVD**: Compute decomposition across multiple machines
- **Online learning**: Update factors incrementally with new ratings
- **Approximate methods**: Randomized SVD for faster computation

**2. Evaluation Metrics:**
- **RMSE/MAE**: Rating prediction accuracy
- **Precision@K/Recall@K**: Top-N recommendation quality
- **Diversity metrics**: Avoiding filter bubbles
- **Coverage**: Fraction of items recommended

**3. Production Systems:**
- **Real-time inference**: Pre-compute user/item embeddings
- **A/B testing**: Compare different factorization approaches
- **Business metrics**: Click-through rates, conversion rates
- **Fairness**: Ensuring equitable recommendations across user groups

**Advanced Research Directions:**
- **Deep learning extensions**: Neural collaborative filtering
- **Graph-based methods**: Social network integration
- **Multi-armed bandits**: Exploration-exploitation trade-offs
- **Causal inference**: Understanding recommendation impact

---

## Question 29

**Explain how you would preprocess data to be used in linear algebra computations.**

**Answer:** Data preprocessing for linear algebra computations is crucial for ensuring numerical stability, computational efficiency, and meaningful results. Proper preprocessing transforms raw data into a format suitable for matrix operations while preserving essential information and preventing common pitfalls like numerical instability, scaling issues, and convergence problems.

**Core Preprocessing Steps:**

**1. Data Cleaning and Validation:**

**Missing Value Handling:**
- **Detection**: Identify NaN, null, or sentinel values
- **Strategies**: 
  - Mean/median imputation for numerical features
  - Mode imputation for categorical features
  - Forward/backward fill for time series
  - Matrix completion techniques (iterative imputation)
  - Deletion of samples/features with excessive missing values

**Outlier Detection and Treatment:**
- **Statistical methods**: Z-score, IQR-based detection
- **Robust methods**: Modified Z-score, isolation forest
- **Treatment options**: Winsorization, transformation, removal

**Data Type Conversion:**
- Ensure numerical data is in appropriate floating-point format
- Handle categorical variables through encoding
- Convert sparse representations when beneficial

**2. Scaling and Normalization:**

**Why Scaling Matters:**
- Different feature scales can dominate computations
- Gradient-based algorithms may converge poorly
- Distance-based methods become biased
- Matrix conditioning can deteriorate

**Standardization (Z-score Normalization):**
```
x_scaled = (x - μ) / σ
```
- Mean μ = 0, standard deviation σ = 1
- Preserves distribution shape
- Handles outliers poorly

**Min-Max Normalization:**
```
x_scaled = (x - x_min) / (x_max - x_min)
```
- Maps to [0, 1] range
- Preserves relationships
- Sensitive to outliers

**Robust Scaling:**
```
x_scaled = (x - median) / IQR
```
- Uses median and interquartile range
- Less sensitive to outliers
- Better for skewed distributions

**Unit Vector Scaling:**
```
x_scaled = x / ||x||₂
```
- Normalizes to unit length
- Preserves direction
- Useful for text data, sparse features

**3. Dimensionality and Feature Engineering:**

**Dimensionality Reduction:**
- **PCA**: Remove correlated features, reduce noise
- **Feature selection**: Remove irrelevant or redundant features
- **Matrix rank analysis**: Identify linear dependencies

**Feature Engineering:**
- **Polynomial features**: Create interaction terms
- **Basis functions**: Transform to more suitable space
- **Domain-specific transforms**: Log, sqrt, Box-Cox transformations

**4. Matrix Structure Optimization:**

**Memory Layout:**
- **Row-major vs column-major**: Optimize for access patterns
- **Dense vs sparse**: Choose appropriate representation
- **Data alignment**: Ensure proper memory alignment for SIMD operations

**Numerical Conditioning:**
- **Condition number analysis**: Check matrix invertibility
- **Regularization**: Add small values to diagonal for stability
- **Pivoting strategies**: Reorder for numerical stability

**Categorical Data Handling:**

**One-Hot Encoding:**
```
Original: [cat, dog, bird]
Encoded: [[1,0,0], [0,1,0], [0,0,1]]
```
- Creates binary indicator variables
- Increases dimensionality
- Suitable for linear models

**Label Encoding:**
```
Original: [cat, dog, bird]
Encoded: [0, 1, 2]
```
- Maps to integers
- Compact representation
- May introduce artificial ordering

**Target Encoding:**
- Replace categories with target statistics
- Useful for high-cardinality features
- Requires cross-validation to prevent overfitting

**Advanced Preprocessing Techniques:**

**1. Whitening Transformations:**
```
x_whitened = Σ^(-1/2)(x - μ)
```
- Decorrelates features
- Unit variance in all directions
- Useful for optimization algorithms

**2. Power Transformations:**
- **Box-Cox**: Stabilize variance, normalize distribution
- **Yeo-Johnson**: Handles negative values
- **Quantile transforms**: Map to uniform/normal distribution

**3. Temporal Preprocessing:**
- **Detrending**: Remove systematic trends
- **Seasonal adjustment**: Account for periodic patterns
- **Stationarity**: Ensure statistical properties don't change over time

**Implementation Examples:**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

class LinearAlgebraPreprocessor:
    """Comprehensive preprocessing pipeline for linear algebra computations"""
    
    def __init__(self, scaling_method='standard', handle_outliers=True, 
                 dimensionality_reduction=None, categorical_encoding='onehot'):
        """
        Initialize preprocessor with specified methods
        
        Args:
            scaling_method: 'standard', 'minmax', 'robust', 'unit'
            handle_outliers: Whether to detect and handle outliers
            dimensionality_reduction: None, 'pca', or 'select_k_best'
            categorical_encoding: 'onehot', 'label', 'target'
        """
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.dimensionality_reduction = dimensionality_reduction
        self.categorical_encoding = categorical_encoding
        
        # Initialize components
        self.scaler = None
        self.imputer = None
        self.outlier_detector = None
        self.dim_reducer = None
        self.categorical_encoder = None
        
        # Store preprocessing parameters
        self.preprocessing_params = {}
        
    def detect_data_types(self, X):
        """Detect and categorize data types"""
        
        if isinstance(X, pd.DataFrame):
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            # Assume all numerical for numpy arrays
            numerical_cols = list(range(X.shape[1]))
            categorical_cols = []
        
        return numerical_cols, categorical_cols
    
    def handle_missing_values(self, X, strategy='mean', verbose=True):
        """Handle missing values in the dataset"""
        
        if verbose:
            print("Handling missing values...")
        
        if isinstance(X, pd.DataFrame):
            missing_info = X.isnull().sum()
            if missing_info.sum() > 0:
                print(f"  Found missing values in {(missing_info > 0).sum()} columns")
                print(f"  Total missing values: {missing_info.sum()}")
        
        # Choose imputation strategy
        if strategy == 'iterative':
            self.imputer = IterativeImputer(random_state=42, max_iter=10)
        else:
            self.imputer = SimpleImputer(strategy=strategy)
        
        # Fit and transform
        if isinstance(X, pd.DataFrame):
            numerical_cols, _ = self.detect_data_types(X)
            if numerical_cols:
                X_numerical = X[numerical_cols]
                X_imputed_numerical = self.imputer.fit_transform(X_numerical)
                
                # Create result dataframe
                X_result = X.copy()
                X_result[numerical_cols] = X_imputed_numerical
                
                return X_result
        else:
            return self.imputer.fit_transform(X)
        
        return X
    
    def detect_outliers(self, X, method='iqr', threshold=1.5):
        """Detect outliers using various methods"""
        
        outliers = np.zeros(X.shape[0], dtype=bool)
        
        if method == 'iqr':
            for col in range(X.shape[1]):
                Q1 = np.percentile(X[:, col], 25)
                Q3 = np.percentile(X[:, col], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_outliers = (X[:, col] < lower_bound) | (X[:, col] > upper_bound)
                outliers |= col_outliers
        
        elif method == 'zscore':
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            outliers = np.any(z_scores > threshold, axis=1)
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(X) == -1
        
        return outliers
    
    def apply_scaling(self, X, verbose=True):
        """Apply specified scaling method"""
        
        if verbose:
            print(f"Applying {self.scaling_method} scaling...")
        
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'unit':
            from sklearn.preprocessing import normalize
            # Unit scaling doesn't need fitting
            X_scaled = normalize(X, norm='l2', axis=1)
            return X_scaled
        else:
            print(f"Unknown scaling method: {self.scaling_method}")
            return X
        
        X_scaled = self.scaler.fit_transform(X)
        
        if verbose:
            print(f"  Original data range: [{np.min(X):.4f}, {np.max(X):.4f}]")
            print(f"  Scaled data range: [{np.min(X_scaled):.4f}, {np.max(X_scaled):.4f}]")
            print(f"  Scaled data mean: {np.mean(X_scaled):.4f}")
            print(f"  Scaled data std: {np.std(X_scaled):.4f}")
        
        return X_scaled
    
    def encode_categorical(self, X, y=None, verbose=True):
        """Encode categorical variables"""
        
        if isinstance(X, pd.DataFrame):
            numerical_cols, categorical_cols = self.detect_data_types(X)
            
            if not categorical_cols:
                return X
            
            if verbose:
                print(f"Encoding {len(categorical_cols)} categorical columns...")
            
            X_result = X.copy()
            
            if self.categorical_encoding == 'onehot':
                self.categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                
                for col in categorical_cols:
                    encoded = self.categorical_encoder.fit_transform(X[[col]])
                    encoded_df = pd.DataFrame(
                        encoded, 
                        columns=[f"{col}_{cat}" for cat in self.categorical_encoder.categories_[0]],
                        index=X.index
                    )
                    
                    # Remove original column and add encoded columns
                    X_result = X_result.drop(columns=[col])
                    X_result = pd.concat([X_result, encoded_df], axis=1)
            
            elif self.categorical_encoding == 'label':
                for col in categorical_cols:
                    le = LabelEncoder()
                    X_result[col] = le.fit_transform(X[col].astype(str))
            
            return X_result
        
        return X
    
    def apply_dimensionality_reduction(self, X, y=None, n_components=None, verbose=True):
        """Apply dimensionality reduction techniques"""
        
        if self.dimensionality_reduction is None:
            return X
        
        if verbose:
            print(f"Applying {self.dimensionality_reduction} dimensionality reduction...")
        
        if self.dimensionality_reduction == 'pca':
            if n_components is None:
                # Use enough components to explain 95% of variance
                n_components = min(X.shape[0], X.shape[1])
            
            self.dim_reducer = PCA(n_components=n_components)
            X_reduced = self.dim_reducer.fit_transform(X)
            
            if verbose:
                explained_var_ratio = self.dim_reducer.explained_variance_ratio_
                cumulative_var = np.cumsum(explained_var_ratio)
                
                print(f"  Original dimensions: {X.shape[1]}")
                print(f"  Reduced dimensions: {X_reduced.shape[1]}")
                print(f"  Explained variance ratio: {explained_var_ratio[:5]}")
                print(f"  Cumulative variance (first 10): {cumulative_var[:10]}")
                
                # Find components for 95% variance
                n_95 = np.argmax(cumulative_var >= 0.95) + 1
                print(f"  Components for 95% variance: {n_95}")
        
        elif self.dimensionality_reduction == 'select_k_best':
            if n_components is None:
                n_components = min(50, X.shape[1])  # Default to 50 features
            
            if y is not None:
                self.dim_reducer = SelectKBest(f_classif, k=n_components)
                X_reduced = self.dim_reducer.fit_transform(X, y)
                
                if verbose:
                    scores = self.dim_reducer.scores_
                    selected_features = self.dim_reducer.get_support()
                    print(f"  Original dimensions: {X.shape[1]}")
                    print(f"  Selected dimensions: {X_reduced.shape[1]}")
                    print(f"  Average score of selected features: {np.mean(scores[selected_features]):.4f}")
            else:
                print("  Warning: SelectKBest requires target variable y")
                return X
        
        return X_reduced
    
    def fit_transform(self, X, y=None, verbose=True):
        """Complete preprocessing pipeline"""
        
        if verbose:
            print("="*60)
            print("LINEAR ALGEBRA DATA PREPROCESSING PIPELINE")
            print("="*60)
            print(f"Input data shape: {X.shape}")
            if isinstance(X, pd.DataFrame):
                print(f"Data types: {X.dtypes.value_counts().to_dict()}")
        
        # Step 1: Handle missing values
        X_processed = self.handle_missing_values(X, verbose=verbose)
        
        # Step 2: Encode categorical variables
        X_processed = self.encode_categorical(X_processed, y, verbose=verbose)
        
        # Convert to numpy array for numerical operations
        if isinstance(X_processed, pd.DataFrame):
            X_processed = X_processed.select_dtypes(include=[np.number]).values
        
        # Step 3: Handle outliers
        if self.handle_outliers and verbose:
            outliers = self.detect_outliers(X_processed)
            print(f"Detected {np.sum(outliers)} outliers ({np.mean(outliers):.1%} of data)")
            # For demonstration, we'll keep outliers but flag them
        
        # Step 4: Apply scaling
        X_processed = self.apply_scaling(X_processed, verbose=verbose)
        
        # Step 5: Dimensionality reduction
        X_processed = self.apply_dimensionality_reduction(X_processed, y, verbose=verbose)
        
        # Step 6: Final validation
        if verbose:
            print(f"\nFinal preprocessing results:")
            print(f"  Output shape: {X_processed.shape}")
            print(f"  Data type: {X_processed.dtype}")
            print(f"  Memory usage: {X_processed.nbytes / 1024:.1f} KB")
            print(f"  No NaN values: {not np.isnan(X_processed).any()}")
            print(f"  No infinite values: {not np.isinf(X_processed).any()}")
            
            # Matrix properties
            if X_processed.shape[1] <= X_processed.shape[0]:
                try:
                    cond_num = np.linalg.cond(X_processed.T @ X_processed)
                    print(f"  Condition number of X^T X: {cond_num:.2e}")
                except:
                    print(f"  Could not compute condition number")
        
        return X_processed
    
    def transform(self, X):
        """Transform new data using fitted preprocessors"""
        
        # Apply same transformations in same order
        X_processed = X.copy()
        
        # Handle missing values
        if self.imputer is not None:
            X_processed = self.imputer.transform(X_processed)
        
        # Apply scaling
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
        
        # Apply dimensionality reduction
        if self.dim_reducer is not None:
            X_processed = self.dim_reducer.transform(X_processed)
        
        return X_processed

def demonstrate_preprocessing_effects():
    """Demonstrate effects of different preprocessing choices"""
    
    print("PREPROCESSING EFFECTS DEMONSTRATION")
    print("=" * 50)
    
    # Generate synthetic dataset with various challenges
    np.random.seed(42)
    n_samples = 1000
    
    # Feature 1: Normal distribution
    feature1 = np.random.normal(100, 15, n_samples)
    
    # Feature 2: Exponential distribution (skewed)
    feature2 = np.random.exponential(2, n_samples)
    
    # Feature 3: Uniform distribution with large scale
    feature3 = np.random.uniform(1000, 5000, n_samples)
    
    # Feature 4: Binary feature
    feature4 = np.random.binomial(1, 0.3, n_samples)
    
    # Add some missing values and outliers
    feature1[np.random.choice(n_samples, 50, replace=False)] = np.nan
    feature2[np.random.choice(n_samples, 20, replace=False)] *= 10  # Outliers
    
    # Create dataset
    X = np.column_stack([feature1, feature2, feature3, feature4])
    
    # Create target variable
    y = (0.5 * feature1 + 2 * feature2 + 0.001 * feature3 + 10 * feature4 + 
         np.random.normal(0, 10, n_samples))
    
    print(f"Original dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Missing values: {np.isnan(X).sum()}")
    print(f"  Feature ranges:")
    for i in range(X.shape[1]):
        valid_data = X[~np.isnan(X[:, i]), i]
        print(f"    Feature {i+1}: [{np.min(valid_data):.2f}, {np.max(valid_data):.2f}]")
    
    # Test different preprocessing approaches
    preprocessing_methods = [
        ('No Preprocessing', {'scaling_method': None}),
        ('Standard Scaling', {'scaling_method': 'standard'}),
        ('Min-Max Scaling', {'scaling_method': 'minmax'}),
        ('Robust Scaling', {'scaling_method': 'robust'}),
        ('Standard + PCA', {'scaling_method': 'standard', 'dimensionality_reduction': 'pca'})
    ]
    
    results = {}
    
    for method_name, params in preprocessing_methods:
        print(f"\n{method_name}:")
        print("-" * 30)
        
        if params['scaling_method'] is None:
            # Handle missing values only
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_processed = imputer.fit_transform(X)
        else:
            preprocessor = LinearAlgebraPreprocessor(**params)
            X_processed = preprocessor.fit_transform(X, y, verbose=False)
        
        # Analyze results
        print(f"  Processed shape: {X_processed.shape}")
        print(f"  Mean: {np.mean(X_processed, axis=0)}")
        print(f"  Std: {np.std(X_processed, axis=0)}")
        
        # Compute condition number if possible
        try:
            if X_processed.shape[1] <= X_processed.shape[0]:
                cond_num = np.linalg.cond(X_processed.T @ X_processed)
                print(f"  Condition number: {cond_num:.2e}")
        except:
            print(f"  Could not compute condition number")
        
        results[method_name] = X_processed
    
    return X, y, results

def matrix_conditioning_analysis():
    """Analyze how preprocessing affects matrix conditioning"""
    
    print(f"\nMATRIX CONDITIONING ANALYSIS")
    print("=" * 40)
    
    # Create ill-conditioned matrix
    np.random.seed(42)
    n = 100
    
    # Generate data with high correlation
    base_feature = np.random.randn(n)
    
    X_ill_conditioned = np.column_stack([
        base_feature,
        base_feature + 0.01 * np.random.randn(n),  # Highly correlated
        base_feature * 2 + 0.02 * np.random.randn(n),  # Linear combination
        np.random.randn(n) * 1000,  # Large scale
        np.random.randn(n) * 0.001  # Small scale
    ])
    
    print(f"Original matrix:")
    print(f"  Shape: {X_ill_conditioned.shape}")
    
    # Compute condition number
    XtX_original = X_ill_conditioned.T @ X_ill_conditioned
    cond_original = np.linalg.cond(XtX_original)
    print(f"  Condition number: {cond_original:.2e}")
    
    # Apply different preprocessing
    preprocessing_effects = {}
    
    # Standard scaling
    scaler = StandardScaler()
    X_standard = scaler.fit_transform(X_ill_conditioned)
    XtX_standard = X_standard.T @ X_standard
    cond_standard = np.linalg.cond(XtX_standard)
    preprocessing_effects['Standard Scaling'] = cond_standard
    
    # PCA
    pca = PCA(n_components=4)  # Remove one component
    X_pca = pca.fit_transform(X_standard)
    XtX_pca = X_pca.T @ X_pca
    cond_pca = np.linalg.cond(XtX_pca)
    preprocessing_effects['PCA (4 components)'] = cond_pca
    
    # Regularization (add small value to diagonal)
    regularization = 1e-6
    XtX_regularized = XtX_standard + regularization * np.eye(XtX_standard.shape[0])
    cond_regularized = np.linalg.cond(XtX_regularized)
    preprocessing_effects['Regularization'] = cond_regularized
    
    print(f"\nConditioning effects:")
    print(f"  Original: {cond_original:.2e}")
    for method, cond_num in preprocessing_effects.items():
        improvement = cond_original / cond_num
        print(f"  {method}: {cond_num:.2e} (improvement: {improvement:.1f}x)")

def sparse_data_preprocessing():
    """Demonstrate preprocessing for sparse data"""
    
    print(f"\nSPARSE DATA PREPROCESSING")
    print("=" * 35)
    
    # Create sparse data (e.g., text features, user-item interactions)
    from scipy.sparse import random as sparse_random
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Simulate sparse matrix (e.g., TF-IDF features)
    n_samples, n_features = 1000, 5000
    density = 0.01  # 1% non-zero values
    
    X_sparse = sparse_random(n_samples, n_features, density=density, random_state=42)
    X_dense = X_sparse.toarray()
    
    print(f"Sparse data characteristics:")
    print(f"  Shape: {X_sparse.shape}")
    print(f"  Density: {X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.1%}")
    print(f"  Memory (sparse): {X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes} bytes")
    print(f"  Memory (dense): {X_dense.nbytes} bytes")
    print(f"  Memory ratio: {X_dense.nbytes / (X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes):.1f}x")
    
    # Preprocessing considerations for sparse data
    print(f"\nSparse data preprocessing considerations:")
    print(f"  1. Scaling can destroy sparsity")
    print(f"  2. Mean centering makes all values non-zero")
    print(f"  3. Use appropriate sparse-aware algorithms")
    print(f"  4. Consider L2 normalization (preserves sparsity)")
    
    # Demonstrate L2 normalization (preserves sparsity)
    from sklearn.preprocessing import normalize
    X_normalized = normalize(X_sparse, norm='l2', axis=1)
    
    print(f"  After L2 normalization:")
    print(f"    Still sparse: {hasattr(X_normalized, 'nnz')}")
    print(f"    Density preserved: {X_normalized.nnz / (X_normalized.shape[0] * X_normalized.shape[1]):.1%}")

def create_preprocessing_visualization():
    """Create visualizations showing preprocessing effects"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    
    # Create data with different distributions
    normal_data = np.random.normal(100, 20, n_samples)
    exponential_data = np.random.exponential(5, n_samples)
    uniform_data = np.random.uniform(0, 1000, n_samples)
    
    # Add outliers
    outlier_indices = np.random.choice(n_samples, 20, replace=False)
    exponential_data[outlier_indices] *= 5
    
    data = np.column_stack([normal_data, exponential_data, uniform_data])
    
    # Apply different scaling methods
    scalers = {
        'Original': None,
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'Robust': RobustScaler()
    }
    
    fig, axes = plt.subplots(len(scalers), 3, figsize=(15, 12))
    
    for i, (scaler_name, scaler) in enumerate(scalers.items()):
        if scaler is None:
            scaled_data = data
        else:
            scaled_data = scaler.fit_transform(data)
        
        for j in range(3):
            ax = axes[i, j]
            
            # Plot histogram
            ax.hist(scaled_data[:, j], bins=30, alpha=0.7, density=True)
            ax.set_title(f'{scaler_name} - Feature {j+1}')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(scaled_data[:, j])
            std_val = np.std(scaled_data[:, j])
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.8, label=f'±1σ')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.8)
            
            if i == 0:  # Only show legend for first row
                ax.legend()
    
    plt.tight_layout()
    plt.savefig('preprocessing_effects.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run demonstrations
    X_original, y, preprocessing_results = demonstrate_preprocessing_effects()
    matrix_conditioning_analysis()
    sparse_data_preprocessing()
    create_preprocessing_visualization()
    
    print(f"\n{'='*60}")
    print("DATA PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    
    print("\nEssential Steps:")
    print("• Data cleaning: Handle missing values, outliers, data types")
    print("• Scaling: Standardize feature ranges for stable computation")
    print("• Encoding: Transform categorical variables to numerical")
    print("• Dimensionality: Reduce features to prevent overfitting")
    
    print("\nScaling Methods:")
    print("• Standard: Mean=0, Std=1 (assumes normal distribution)")
    print("• Min-Max: Scale to [0,1] range (preserves relationships)")
    print("• Robust: Use median/IQR (less sensitive to outliers)")
    print("• Unit: Normalize to unit length (for directional data)")
    
    print("\nNumerical Considerations:")
    print("• Condition number: Lower is better for stability")
    print("• Memory layout: Row vs column major for cache efficiency")
    print("• Sparse representation: Preserve sparsity when possible")
    print("• Data types: Use appropriate precision (float32 vs float64)")
    
    print("\nCommon Pitfalls:")
    print("• Scaling destroys sparsity in sparse matrices")
    print("• Data leakage: Using future information in time series")
    print("• Inconsistent preprocessing between train/test sets")
    print("• Ignoring domain-specific requirements")
    
    print("\nBest Practices:")
    print("• Validate preprocessing pipeline on holdout data")
    print("• Monitor numerical stability (condition numbers)")
    print("• Document preprocessing steps for reproducibility")
    print("• Consider computational efficiency for large datasets")
```

**Domain-Specific Considerations:**

**1. Time Series Data:**
- **Temporal ordering**: Preserve chronological sequence
- **Stationarity**: Remove trends and seasonal patterns
- **Lagged features**: Create autoregressive terms
- **Missing values**: Forward fill or interpolation

**2. Text Data:**
- **Tokenization**: Convert text to numerical vectors
- **TF-IDF**: Weight terms by frequency and rarity
- **Dimensionality**: Use techniques like LSA or word embeddings
- **Sparsity**: Maintain sparse representations

**3. Image Data:**
- **Normalization**: Scale pixel values to [0,1] or [-1,1]
- **Augmentation**: Geometric transformations for robustness
- **Channel ordering**: RGB vs BGR considerations
- **Memory efficiency**: Batch processing for large images

**4. Graph Data:**
- **Adjacency matrices**: Handle different graph sizes
- **Node features**: Normalize across nodes
- **Spectral features**: Use graph Laplacian eigenvalues
- **Sparse representation**: Efficient storage for large graphs

**Computational Efficiency Tips:**
- Use appropriate data types (float32 vs float64)
- Leverage vectorized operations (NumPy, BLAS)
- Consider memory vs computation trade-offs
- Profile preprocessing pipelines for bottlenecks
- Use parallel processing for independent operations

---

## Question 30

**Describe ways to find the rank of a matrix effectively.**

**Answer:** The rank of a matrix is one of the most fundamental concepts in linear algebra, representing the dimension of the column space (or equivalently, the row space) of the matrix. Finding the rank effectively is crucial for understanding linear systems, matrix invertibility, dimensionality reduction, and many machine learning applications. Various methods exist, each with different computational complexities and numerical considerations.

**Mathematical Definition:**

The **rank** of an m×n matrix A is:
- rank(A) = dimension of column space = dimension of row space
- rank(A) = number of linearly independent columns = number of linearly independent rows
- rank(A) = number of non-zero singular values
- rank(A) ≤ min(m, n)

**Properties:**
- rank(A) = rank(Aᵀ) (rank is preserved under transpose)
- rank(AB) ≤ min(rank(A), rank(B))
- rank(A + B) ≤ rank(A) + rank(B)
- For invertible matrices P, Q: rank(PAQ) = rank(A)

**Methods for Computing Matrix Rank:**

**1. Row Echelon Form (Gaussian Elimination):**

**Algorithm:**
1. Use elementary row operations to transform matrix to row echelon form
2. Count number of non-zero rows
3. This equals the rank

**Advantages:**
- Conceptually simple and exact for rational entries
- Provides insight into linear dependencies
- Can be done by hand for small matrices

**Disadvantages:**
- O(mn·min(m,n)) complexity
- Numerically unstable due to round-off errors
- Sensitive to pivot selection

**Implementation:**
```python
def gaussian_elimination_rank(A, tolerance=1e-10):
    """Compute rank using Gaussian elimination with partial pivoting"""
    
    A = A.astype(float)  # Work with floating point
    m, n = A.shape
    rank = 0
    
    for col in range(min(m, n)):
        # Find pivot row
        pivot_row = np.argmax(np.abs(A[rank:m, col])) + rank
        
        # Check if pivot is non-zero
        if abs(A[pivot_row, col]) < tolerance:
            continue  # Skip this column
        
        # Swap rows if needed
        if pivot_row != rank:
            A[[rank, pivot_row]] = A[[pivot_row, rank]]
        
        # Eliminate below pivot
        for row in range(rank + 1, m):
            factor = A[row, col] / A[rank, col]
            A[row, col:] -= factor * A[rank, col:]
        
        rank += 1
    
    return rank
```

**2. Singular Value Decomposition (SVD):**

**Method:**
- Compute A = UΣVᵀ
- Count non-zero singular values in Σ
- Use numerical tolerance for "effectively zero" values

**Mathematical Foundation:**
```
rank(A) = number of σᵢ > tolerance
where σ₁ ≥ σ₂ ≥ ... ≥ σₘᵢₙ ≥ 0
```

**Advantages:**
- Numerically most stable method
- Provides singular values for condition number analysis
- Robust to round-off errors
- Industry standard for numerical rank computation

**Disadvantages:**
- O(mn²) complexity for m ≥ n, O(m²n) for m < n
- More computationally expensive than QR
- Requires choosing appropriate tolerance

**3. QR Decomposition with Column Pivoting:**

**Method:**
- Compute AΠ = QR where Π is permutation matrix
- Rank equals number of diagonal elements in R above tolerance
- Column pivoting ensures largest elements appear first on diagonal

**Algorithm:**
```
AP = QR where P is permutation matrix
rank(A) = number of |Rᵢᵢ| > tolerance
```

**Advantages:**
- O(mn²) complexity
- Faster than SVD
- Naturally reveals column dependencies
- Good numerical stability with pivoting

**Disadvantages:**
- Less numerically stable than SVD
- May miss some rank deficiencies without pivoting

**4. LU Decomposition with Pivoting:**

**Method:**
- Compute PA = LU with partial or complete pivoting
- Count non-zero diagonal elements in U
- Use tolerance for numerical considerations

**Advantages:**
- O(mn²) complexity
- Can reuse factorization for multiple rank queries
- Provides factorization useful for other computations

**Disadvantages:**
- Less stable than QR or SVD for rank computation
- Sensitive to pivot strategy

**5. Eigenvalue Decomposition (Square Matrices Only):**

**Method:**
- For square matrix A, compute eigenvalues λᵢ
- For symmetric matrices: rank(A) = number of non-zero eigenvalues
- For general matrices: use A*A or AA* (which are symmetric positive semidefinite)

**Comprehensive Implementation:**

```python
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from time import time

class MatrixRankComputer:
    """Comprehensive matrix rank computation with multiple methods"""
    
    def __init__(self, tolerance=1e-10):
        """Initialize with numerical tolerance for zero detection"""
        self.tolerance = tolerance
        self.timing_results = {}
        
    def rank_gaussian_elimination(self, A):
        """Rank via Gaussian elimination with partial pivoting"""
        start_time = time()
        
        A_work = A.astype(float).copy()
        m, n = A_work.shape
        rank = 0
        
        for col in range(min(m, n)):
            # Find best pivot
            pivot_candidates = A_work[rank:m, col]
            if len(pivot_candidates) == 0:
                break
                
            best_pivot_idx = np.argmax(np.abs(pivot_candidates))
            pivot_row = best_pivot_idx + rank
            
            # Check if pivot is significant
            if abs(A_work[pivot_row, col]) < self.tolerance:
                continue
            
            # Swap rows
            if pivot_row != rank:
                A_work[[rank, pivot_row]] = A_work[[pivot_row, rank]]
            
            # Eliminate
            pivot = A_work[rank, col]
            for row in range(rank + 1, m):
                if abs(A_work[row, col]) > self.tolerance:
                    factor = A_work[row, col] / pivot
                    A_work[row, col:] -= factor * A_work[rank, col:]
            
            rank += 1
        
        self.timing_results['gaussian'] = time() - start_time
        return rank
    
    def rank_svd(self, A):
        """Rank via Singular Value Decomposition"""
        start_time = time()
        
        try:
            # Compute SVD
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            
            # Count significant singular values
            rank = np.sum(s > self.tolerance)
            
            self.timing_results['svd'] = time() - start_time
            return rank, s  # Return singular values for analysis
            
        except np.linalg.LinAlgError:
            self.timing_results['svd'] = time() - start_time
            return 0, np.array([])
    
    def rank_qr_pivoting(self, A):
        """Rank via QR decomposition with column pivoting"""
        start_time = time()
        
        try:
            # QR with column pivoting
            Q, R, P = la.qr(A, pivoting=True)
            
            # Count significant diagonal elements
            diag_R = np.abs(np.diag(R))
            rank = np.sum(diag_R > self.tolerance)
            
            self.timing_results['qr'] = time() - start_time
            return rank, diag_R
            
        except la.LinAlgError:
            self.timing_results['qr'] = time() - start_time
            return 0, np.array([])
    
    def rank_lu(self, A):
        """Rank via LU decomposition with pivoting"""
        start_time = time()
        
        try:
            # LU with partial pivoting
            P, L, U = la.lu(A)
            
            # Count significant diagonal elements in U
            diag_U = np.abs(np.diag(U))
            rank = np.sum(diag_U > self.tolerance)
            
            self.timing_results['lu'] = time() - start_time
            return rank, diag_U
            
        except la.LinAlgError:
            self.timing_results['lu'] = time() - start_time
            return 0, np.array([])
    
    def rank_eigenvalue(self, A):
        """Rank via eigenvalue decomposition (square matrices)"""
        start_time = time()
        
        if A.shape[0] != A.shape[1]:
            # For non-square matrices, use A.T @ A
            ATA = A.T @ A
            eigenvals = np.linalg.eigvals(ATA)
        else:
            # Check if symmetric
            if np.allclose(A, A.T):
                eigenvals = np.linalg.eigvals(A)
            else:
                # Use A.T @ A for general matrices
                ATA = A.T @ A
                eigenvals = np.linalg.eigvals(ATA)
        
        # Count positive eigenvalues
        rank = np.sum(np.real(eigenvals) > self.tolerance)
        
        self.timing_results['eigenvalue'] = time() - start_time
        return rank, eigenvals
    
    def rank_numpy_builtin(self, A):
        """Use NumPy's built-in rank function for comparison"""
        start_time = time()
        
        rank = np.linalg.matrix_rank(A, tol=self.tolerance)
        
        self.timing_results['numpy'] = time() - start_time
        return rank
    
    def comprehensive_rank_analysis(self, A, verbose=True):
        """Perform comprehensive rank analysis using all methods"""
        
        if verbose:
            print(f"COMPREHENSIVE RANK ANALYSIS")
            print(f"Matrix shape: {A.shape}")
            print(f"Tolerance: {self.tolerance}")
            print("=" * 50)
        
        results = {}
        
        # Method 1: Gaussian Elimination
        try:
            rank_gauss = self.rank_gaussian_elimination(A)
            results['Gaussian Elimination'] = rank_gauss
            if verbose:
                print(f"Gaussian Elimination: {rank_gauss}")
        except Exception as e:
            if verbose:
                print(f"Gaussian Elimination: Failed ({e})")
        
        # Method 2: SVD
        try:
            rank_svd, singular_values = self.rank_svd(A)
            results['SVD'] = rank_svd
            if verbose:
                print(f"SVD: {rank_svd}")
                if len(singular_values) > 0:
                    print(f"  Singular values: {singular_values[:5]}")  # Show first 5
                    if len(singular_values) > 5:
                        print(f"  (showing first 5 of {len(singular_values)})")
        except Exception as e:
            if verbose:
                print(f"SVD: Failed ({e})")
        
        # Method 3: QR with pivoting
        try:
            rank_qr, diag_R = self.rank_qr_pivoting(A)
            results['QR Pivoting'] = rank_qr
            if verbose:
                print(f"QR with Pivoting: {rank_qr}")
                if len(diag_R) > 0:
                    print(f"  |R_ii|: {diag_R[:5]}")
        except Exception as e:
            if verbose:
                print(f"QR with Pivoting: Failed ({e})")
        
        # Method 4: LU decomposition
        try:
            rank_lu, diag_U = self.rank_lu(A)
            results['LU'] = rank_lu
            if verbose:
                print(f"LU Decomposition: {rank_lu}")
                if len(diag_U) > 0:
                    print(f"  |U_ii|: {diag_U[:5]}")
        except Exception as e:
            if verbose:
                print(f"LU Decomposition: Failed ({e})")
        
        # Method 5: Eigenvalue method
        try:
            rank_eigen, eigenvals = self.rank_eigenvalue(A)
            results['Eigenvalue'] = rank_eigen
            if verbose:
                print(f"Eigenvalue Method: {rank_eigen}")
                if len(eigenvals) > 0:
                    print(f"  Eigenvalues: {np.real(eigenvals[:5])}")
        except Exception as e:
            if verbose:
                print(f"Eigenvalue Method: Failed ({e})")
        
        # Method 6: NumPy built-in
        try:
            rank_numpy = self.rank_numpy_builtin(A)
            results['NumPy Built-in'] = rank_numpy
            if verbose:
                print(f"NumPy Built-in: {rank_numpy}")
        except Exception as e:
            if verbose:
                print(f"NumPy Built-in: Failed ({e})")
        
        # Timing comparison
        if verbose:
            print(f"\nTiming Comparison:")
            for method, time_taken in self.timing_results.items():
                print(f"  {method}: {time_taken:.6f} seconds")
        
        # Consistency check
        if verbose:
            ranks = list(results.values())
            if len(set(ranks)) == 1:
                print(f"\n✓ All methods agree: rank = {ranks[0]}")
            else:
                print(f"\n⚠ Methods disagree: {results}")
                print("This may indicate numerical issues or method limitations")
        
        return results

def create_test_matrices():
    """Create test matrices with known ranks for validation"""
    
    test_matrices = {}
    
    # Full rank matrix
    np.random.seed(42)
    A_full = np.random.randn(5, 5)
    test_matrices['Full Rank 5x5'] = (A_full, 5)
    
    # Rank deficient square matrix
    A_rank3 = np.random.randn(5, 3) @ np.random.randn(3, 5)
    test_matrices['Rank 3 (5x5)'] = (A_rank3, 3)
    
    # Rectangular matrices
    A_rect1 = np.random.randn(3, 5)  # Should have rank 3
    test_matrices['Rectangular 3x5'] = (A_rect1, 3)
    
    A_rect2 = np.random.randn(5, 3)  # Should have rank 3
    test_matrices['Rectangular 5x3'] = (A_rect2, 3)
    
    # Very low rank matrix
    u = np.random.randn(4, 1)
    v = np.random.randn(1, 6)
    A_rank1 = u @ v
    test_matrices['Rank 1 (4x6)'] = (A_rank1, 1)
    
    # Nearly singular matrix
    A_near_singular = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9 + 1e-12]  # Almost rank 2
    ])
    test_matrices['Nearly Singular'] = (A_near_singular, 2)
    
    # Zero matrix
    A_zero = np.zeros((3, 4))
    test_matrices['Zero Matrix'] = (A_zero, 0)
    
    return test_matrices

def numerical_stability_analysis():
    """Analyze numerical stability of different rank computation methods"""
    
    print(f"\nNUMERICAL STABILITY ANALYSIS")
    print("=" * 40)
    
    # Create ill-conditioned matrix
    n = 10
    condition_numbers = [1e2, 1e6, 1e10, 1e14]
    
    computer = MatrixRankComputer()
    
    for cond_num in condition_numbers:
        print(f"\nCondition number ≈ {cond_num:.0e}:")
        
        # Create matrix with specified condition number
        U, _ = np.linalg.qr(np.random.randn(n, n))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        s = np.logspace(0, -np.log10(cond_num), n)  # Geometric decay
        A = U @ np.diag(s) @ V.T
        
        actual_cond = np.linalg.cond(A)
        print(f"  Actual condition number: {actual_cond:.2e}")
        
        # Test different tolerances
        tolerances = [1e-10, 1e-12, 1e-14, 1e-16]
        
        for tol in tolerances:
            computer.tolerance = tol
            rank_svd, _ = computer.rank_svd(A)
            rank_qr, _ = computer.rank_qr_pivoting(A)
            
            print(f"    Tolerance {tol:.0e}: SVD={rank_svd}, QR={rank_qr}")

def performance_comparison():
    """Compare performance of different methods"""
    
    print(f"\nPERFORMANCE COMPARISON")
    print("=" * 30)
    
    sizes = [50, 100, 200, 500]
    computer = MatrixRankComputer()
    
    results = {method: [] for method in ['gaussian', 'svd', 'qr', 'lu', 'numpy']}
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        
        # Create random full-rank matrix
        np.random.seed(42)
        A = np.random.randn(size, size)
        
        # Run each method
        computer.rank_gaussian_elimination(A)
        computer.rank_svd(A)
        computer.rank_qr_pivoting(A)
        computer.rank_lu(A)
        computer.rank_numpy_builtin(A)
        
        # Store timing results
        for method in results.keys():
            if method in computer.timing_results:
                results[method].append(computer.timing_results[method])
                print(f"  {method}: {computer.timing_results[method]:.4f}s")
            else:
                results[method].append(np.nan)
    
    return sizes, results

def demonstrate_rank_applications():
    """Demonstrate applications of matrix rank"""
    
    print(f"\nRANK APPLICATIONS IN MACHINE LEARNING")
    print("=" * 45)
    
    # 1. Linear system solvability
    print("1. Linear System Solvability:")
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([1, 2, 3])
    
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
    
    print(f"   System: Ax = b")
    print(f"   rank(A) = {rank_A}")
    print(f"   rank([A|b]) = {rank_Ab}")
    
    if rank_A == rank_Ab:
        if rank_A == A.shape[1]:
            print(f"   → Unique solution exists")
        else:
            print(f"   → Infinite solutions exist")
    else:
        print(f"   → No solution exists")
    
    # 2. PCA and dimensionality reduction
    print(f"\n2. PCA and Effective Dimensionality:")
    
    # Create data with intrinsic low-dimensional structure
    np.random.seed(42)
    true_dim = 3
    ambient_dim = 10
    n_samples = 100
    
    # Generate low-rank data
    W_true = np.random.randn(ambient_dim, true_dim)
    Z = np.random.randn(n_samples, true_dim)
    X = Z @ W_true.T + 0.01 * np.random.randn(n_samples, ambient_dim)  # Add noise
    
    # Compute rank
    rank_X = np.linalg.matrix_rank(X, tol=0.1)  # Use larger tolerance for noisy data
    
    print(f"   Data shape: {X.shape}")
    print(f"   True intrinsic dimension: {true_dim}")
    print(f"   Estimated rank: {rank_X}")
    
    # SVD analysis
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Compute explained variance
    explained_var_ratio = s**2 / np.sum(s**2)
    cumulative_var = np.cumsum(explained_var_ratio)
    
    print(f"   Singular values: {s[:5]}")
    print(f"   Explained variance (first 5): {explained_var_ratio[:5]}")
    
    # Find effective rank for 95% variance
    effective_rank = np.argmax(cumulative_var >= 0.95) + 1
    print(f"   Effective rank (95% variance): {effective_rank}")

if __name__ == "__main__":
    # Initialize rank computer
    computer = MatrixRankComputer(tolerance=1e-10)
    
    print("MATRIX RANK COMPUTATION METHODS")
    print("=" * 50)
    
    # Test on various matrices
    test_matrices = create_test_matrices()
    
    for name, (matrix, expected_rank) in test_matrices.items():
        print(f"\nTesting: {name}")
        print(f"Expected rank: {expected_rank}")
        print("-" * 30)
        
        results = computer.comprehensive_rank_analysis(matrix, verbose=False)
        
        # Check consistency
        ranks = list(results.values())
        consistent = len(set(ranks)) == 1
        
        print(f"Results: {results}")
        print(f"Consistent: {'Yes' if consistent else 'No'}")
        
        if consistent and ranks[0] == expected_rank:
            print(f"✓ Correct rank computed")
        else:
            print(f"⚠ Issues detected")
    
    # Numerical stability analysis
    numerical_stability_analysis()
    
    # Performance comparison
    sizes, timing_results = performance_comparison()
    
    # Applications
    demonstrate_rank_applications()
    
    print(f"\n{'='*50}")
    print("RANK COMPUTATION SUMMARY")
    print(f"{'='*50}")
    
    print("\nMethods Comparison:")
    print("• Gaussian Elimination: O(mn²), exact for rational, unstable for floating point")
    print("• SVD: O(mn²), most numerically stable, expensive but reliable")
    print("• QR with pivoting: O(mn²), good balance of speed and stability")
    print("• LU decomposition: O(mn²), fast but less stable for rank")
    print("• Eigenvalue method: O(n³) for square matrices, uses A^T A")
    
    print("\nNumerical Considerations:")
    print("• Choose tolerance based on machine precision and condition number")
    print("• SVD is gold standard for numerical rank computation")
    print("• QR with pivoting is good compromise for speed vs accuracy")
    print("• Always validate with multiple methods for critical applications")
    
    print("\nApplications:")
    print("• Linear system solvability: rank(A) vs rank([A|b])")
    print("• PCA: Effective dimensionality from singular value decay")
    print("• Matrix completion: Exploiting low-rank structure")
    print("• Feature selection: Identifying linear dependencies")
```

**Advanced Considerations:**

**1. Numerical Tolerance Selection:**
```
tolerance = max(m, n) × machine_epsilon × ||A||
```
Where machine_epsilon ≈ 2.22 × 10⁻¹⁶ for double precision.

**2. Condition Number Impact:**
For matrices with condition number κ:
- Expect to lose log₁₀(κ) decimal digits of accuracy
- Use SVD for κ > 10¹²
- Consider regularization for very ill-conditioned matrices

**3. Specialized Cases:**
- **Sparse matrices**: Use iterative methods or sparse-aware factorizations
- **Structured matrices**: Exploit structure (Toeplitz, circulant, etc.)
- **Streaming data**: Randomized algorithms for large-scale problems

**Best Practices:**
- Use SVD for critical numerical accuracy
- Use QR with pivoting for good speed-accuracy balance  
- Validate results with multiple methods
- Consider problem context when choosing tolerance
- Profile for performance-critical applications

---

## Question 31

**Explain how you would use linear algebra to clean and preprocess a dataset.**

**Answer:** Linear algebra provides powerful mathematical frameworks for dataset cleaning and preprocessing through matrix operations, transformations, and decompositions. These techniques can handle missing values, outliers, feature scaling, dimensionality reduction, and structural data issues efficiently and systematically. Understanding linear algebra applications in data preprocessing is essential for building robust machine learning pipelines.

**Core Linear Algebra Concepts for Data Preprocessing:**

**1. Matrix Representation of Data:**
```
X ∈ ℝⁿˣᵖ where:
- n = number of samples (rows)
- p = number of features (columns)
- Xᵢⱼ = value of feature j for sample i
```

**2. Vector Operations for Feature Processing:**
- **Centering**: x_centered = x - μ (where μ is mean vector)
- **Scaling**: x_scaled = (x - μ) / σ (element-wise division)
- **Normalization**: x_norm = x / ||x||₂ (unit length vectors)

**3. Matrix Operations for Batch Processing:**
- Vectorized operations process entire dataset simultaneously
- Broadcasting enables efficient computation across samples
- Matrix multiplications implement linear transformations

**Major Preprocessing Applications:**

**1. Missing Value Imputation:**

**Mean/Median Imputation (Matrix Completion):**
```python
# Replace missing values with column means
X_filled = X.copy()
for j in range(p):
    missing_mask = np.isnan(X[:, j])
    if np.any(missing_mask):
        X_filled[missing_mask, j] = np.nanmean(X[:, j])
```

**Matrix Factorization Imputation:**
Use low-rank matrix approximation to fill missing values:
```
X ≈ UV^T where U ∈ ℝⁿˣʳ, V ∈ ℝᵖˣʳ, r << min(n,p)
```

**Iterative Imputation (Matrix-based):**
- Initialize missing values with means
- Fit regression model using complete features
- Update missing values iteratively
- Converge to stable solution

**2. Outlier Detection and Treatment:**

**Mahalanobis Distance:**
```
d²(x) = (x - μ)^T Σ⁻¹ (x - μ)
```
Where Σ is the covariance matrix.

**Principal Component Analysis (PCA) for Outlier Detection:**
- Project data to principal components
- Identify samples with large reconstruction errors
- Remove or downweight outliers

**Robust Covariance Estimation:**
- Use robust estimators less sensitive to outliers
- Minimum Covariance Determinant (MCD)
- Robust PCA using iterative methods

**3. Feature Scaling and Normalization:**

**Standardization (Z-score):**
```
X_standardized = (X - μ) / σ
```
Matrix form: X_std = X @ D⁻¹ - μ₁ᵀ @ D⁻¹
where D = diag(σ₁, σ₂, ..., σₚ)

**Min-Max Scaling:**
```
X_minmax = (X - X_min) / (X_max - X_min)
```

**Unit Vector Scaling:**
```
X_unit = X / ||X||₂ (row-wise normalization)
```

**4. Dimensionality Reduction:**

**Principal Component Analysis (PCA):**
```
X_centered = X - μ₁ᵀ
C = (1/(n-1)) X_centered^T X_centered
C = VΛV^T (eigendecomposition)
X_pca = X_centered V[:,:k] (first k components)
```

**Singular Value Decomposition (SVD):**
```
X = UΣV^T
X_reduced = U[:,:k] Σ[:k,:k] (reduced representation)
```

**5. Feature Engineering through Linear Transformations:**

**Polynomial Features:**
Create interaction terms and higher-order features through Kronecker products and outer products.

**Basis Transformations:**
Transform features to different basis (Fourier, wavelet, etc.) using matrix multiplications.

**Comprehensive Implementation:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import IterativeImputer
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

class LinearAlgebraDataPreprocessor:
    """Comprehensive data preprocessing using linear algebra techniques"""
    
    def __init__(self, missing_strategy='iterative', outlier_method='mahalanobis',
                 scaling_method='standard', dim_reduction=None):
        """
        Initialize preprocessor with linear algebra-based methods
        
        Args:
            missing_strategy: 'mean', 'median', 'iterative', 'matrix_factorization'
            outlier_method: 'mahalanobis', 'pca_reconstruction', 'robust_covariance'
            scaling_method: 'standard', 'minmax', 'robust', 'unit_vector'
            dim_reduction: None, 'pca', 'svd', 'ica'
        """
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.scaling_method = scaling_method
        self.dim_reduction = dim_reduction
        
        # Store fitted parameters
        self.fitted_params = {}
        self.preprocessing_stats = {}
        
    def handle_missing_values(self, X, verbose=True):
        """Handle missing values using various linear algebra approaches"""
        
        if verbose:
            print("MISSING VALUE IMPUTATION")
            print("=" * 30)
            print(f"Strategy: {self.missing_strategy}")
        
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        missing_mask = np.isnan(X)
        missing_count = np.sum(missing_mask)
        
        if missing_count == 0:
            if verbose:
                print("No missing values found")
            return X
        
        if verbose:
            print(f"Missing values: {missing_count} ({missing_count/(n_samples*n_features)*100:.1f}%)")
            print(f"Features with missing values: {np.sum(np.any(missing_mask, axis=0))}")
        
        if self.missing_strategy == 'mean':
            # Simple mean imputation
            X_imputed = X.copy()
            for j in range(n_features):
                if np.any(missing_mask[:, j]):
                    mean_val = np.nanmean(X[:, j])
                    X_imputed[missing_mask[:, j], j] = mean_val
            
            if verbose:
                print("Applied column-wise mean imputation")
        
        elif self.missing_strategy == 'median':
            # Median imputation (more robust to outliers)
            X_imputed = X.copy()
            for j in range(n_features):
                if np.any(missing_mask[:, j]):
                    median_val = np.nanmedian(X[:, j])
                    X_imputed[missing_mask[:, j], j] = median_val
            
            if verbose:
                print("Applied column-wise median imputation")
        
        elif self.missing_strategy == 'iterative':
            # Iterative imputation using regression
            imputer = IterativeImputer(max_iter=10, random_state=42)
            X_imputed = imputer.fit_transform(X)
            
            self.fitted_params['imputer'] = imputer
            
            if verbose:
                print("Applied iterative imputation using regression")
        
        elif self.missing_strategy == 'matrix_factorization':
            # Low-rank matrix completion
            X_imputed = self._matrix_completion_svd(X, missing_mask, rank=min(10, min(n_samples, n_features)//2))
            
            if verbose:
                print("Applied matrix factorization imputation")
        
        # Store statistics
        self.preprocessing_stats['missing_imputed'] = missing_count
        
        return X_imputed
    
    def _matrix_completion_svd(self, X, missing_mask, rank=5, max_iter=100, tol=1e-4):
        """Matrix completion using SVD (simplified version)"""
        
        X_filled = X.copy()
        
        # Initialize missing values with column means
        for j in range(X.shape[1]):
            if np.any(missing_mask[:, j]):
                X_filled[missing_mask[:, j], j] = np.nanmean(X[:, j])
        
        # Iterative SVD imputation
        for iteration in range(max_iter):
            X_old = X_filled.copy()
            
            # SVD decomposition
            try:
                U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
                
                # Keep only top rank components
                if rank < len(s):
                    U = U[:, :rank]
                    s = s[:rank]
                    Vt = Vt[:rank, :]
                
                # Reconstruct matrix
                X_reconstructed = U @ np.diag(s) @ Vt
                
                # Update only missing values
                X_filled[missing_mask] = X_reconstructed[missing_mask]
                
                # Check convergence
                diff = np.linalg.norm(X_filled - X_old, 'fro')
                if diff < tol:
                    break
                    
            except np.linalg.LinAlgError:
                print("SVD failed, using simple mean imputation")
                break
        
        return X_filled
    
    def detect_outliers(self, X, contamination=0.1, verbose=True):
        """Detect outliers using linear algebra methods"""
        
        if verbose:
            print(f"\nOUTLIER DETECTION")
            print("=" * 20)
            print(f"Method: {self.outlier_method}")
        
        n_samples = X.shape[0]
        outlier_mask = np.zeros(n_samples, dtype=bool)
        
        if self.outlier_method == 'mahalanobis':
            # Mahalanobis distance
            mean = np.mean(X, axis=0)
            cov = np.cov(X.T)
            
            # Handle singular covariance matrix
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse for singular matrices
                cov_inv = np.linalg.pinv(cov)
            
            # Compute Mahalanobis distances
            diff = X - mean
            mahal_dist = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))
            
            # Use threshold based on chi-square distribution
            threshold = np.percentile(mahal_dist, (1 - contamination) * 100)
            outlier_mask = mahal_dist > threshold
            
            if verbose:
                print(f"Mahalanobis distance threshold: {threshold:.3f}")
                print(f"Mean distance: {np.mean(mahal_dist):.3f}")
        
        elif self.outlier_method == 'pca_reconstruction':
            # PCA reconstruction error
            pca = PCA(n_components=min(10, X.shape[1]))
            X_pca = pca.fit_transform(X)
            X_reconstructed = pca.inverse_transform(X_pca)
            
            # Reconstruction error
            reconstruction_errors = np.linalg.norm(X - X_reconstructed, axis=1)
            threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)
            outlier_mask = reconstruction_errors > threshold
            
            if verbose:
                print(f"PCA reconstruction error threshold: {threshold:.3f}")
                print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        elif self.outlier_method == 'robust_covariance':
            # Robust covariance estimation
            robust_cov = EllipticEnvelope(contamination=contamination, random_state=42)
            outlier_predictions = robust_cov.fit_predict(X)
            outlier_mask = outlier_predictions == -1
        
        if verbose:
            print(f"Detected {np.sum(outlier_mask)} outliers ({np.mean(outlier_mask)*100:.1f}%)")
        
        self.preprocessing_stats['outliers_detected'] = np.sum(outlier_mask)
        return outlier_mask
    
    def apply_scaling(self, X, verbose=True):
        """Apply scaling using linear algebra operations"""
        
        if verbose:
            print(f"\nFEATURE SCALING")
            print("=" * 20)
            print(f"Method: {self.scaling_method}")
        
        if self.scaling_method == 'standard':
            # Z-score standardization: (X - μ) / σ
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0, ddof=1)
            
            # Handle zero variance features
            std[std == 0] = 1
            
            X_scaled = (X - mean) / std
            
            self.fitted_params['scaler_mean'] = mean
            self.fitted_params['scaler_std'] = std
            
            if verbose:
                print(f"Standardized to mean=0, std=1")
                print(f"Original range: [{np.min(X):.3f}, {np.max(X):.3f}]")
                print(f"Scaled range: [{np.min(X_scaled):.3f}, {np.max(X_scaled):.3f}]")
        
        elif self.scaling_method == 'minmax':
            # Min-max scaling: (X - min) / (max - min)
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            
            # Handle constant features
            range_vals = X_max - X_min
            range_vals[range_vals == 0] = 1
            
            X_scaled = (X - X_min) / range_vals
            
            self.fitted_params['scaler_min'] = X_min
            self.fitted_params['scaler_range'] = range_vals
        
        elif self.scaling_method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            iqr = q75 - q25
            
            # Handle zero IQR
            iqr[iqr == 0] = 1
            
            X_scaled = (X - median) / iqr
            
            self.fitted_params['scaler_median'] = median
            self.fitted_params['scaler_iqr'] = iqr
        
        elif self.scaling_method == 'unit_vector':
            # Unit vector scaling (L2 normalization)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Handle zero vectors
            X_scaled = X / norms
        
        return X_scaled
    
    def apply_dimensionality_reduction(self, X, n_components=None, verbose=True):
        """Apply dimensionality reduction using linear algebra"""
        
        if self.dim_reduction is None:
            return X
        
        if verbose:
            print(f"\nDIMENSIONALITY REDUCTION")
            print("=" * 25)
            print(f"Method: {self.dim_reduction}")
        
        if n_components is None:
            n_components = min(50, X.shape[1] // 2)
        
        if self.dim_reduction == 'pca':
            # Principal Component Analysis
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X)
            
            self.fitted_params['pca'] = pca
            
            if verbose:
                print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
                print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.3f}")
                print(f"Individual ratios: {pca.explained_variance_ratio_[:5]}")
        
        elif self.dim_reduction == 'svd':
            # Truncated SVD
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = svd.fit_transform(X)
            
            self.fitted_params['svd'] = svd
            
            if verbose:
                print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
                print(f"Explained variance ratio: {np.sum(svd.explained_variance_ratio_):.3f}")
        
        return X_reduced
    
    def remove_linear_dependencies(self, X, tolerance=1e-10, verbose=True):
        """Remove linearly dependent features using matrix rank analysis"""
        
        if verbose:
            print(f"\nLINEAR DEPENDENCY REMOVAL")
            print("=" * 30)
        
        # Compute QR decomposition with column pivoting
        Q, R, P = linalg.qr(X.T, pivoting=True)
        
        # Find rank by counting significant diagonal elements
        rank = np.sum(np.abs(np.diag(R)) > tolerance)
        
        if rank < X.shape[1]:
            # Select linearly independent columns
            independent_cols = P[:rank]
            X_independent = X[:, independent_cols]
            
            self.fitted_params['independent_features'] = independent_cols
            
            if verbose:
                print(f"Removed {X.shape[1] - rank} linearly dependent features")
                print(f"Kept {rank} independent features")
                print(f"Feature indices kept: {independent_cols[:10]}...")  # Show first 10
            
            return X_independent
        else:
            if verbose:
                print("No linear dependencies found")
            return X
    
    def detect_feature_correlations(self, X, threshold=0.95, verbose=True):
        """Detect and handle highly correlated features"""
        
        if verbose:
            print(f"\nCORRELATION ANALYSIS")
            print("=" * 25)
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(corr_matrix[i, j]) > threshold:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        if verbose:
            print(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > {threshold})")
            if high_corr_pairs:
                print("Top correlations:")
                for i, j, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    print(f"  Features {i}-{j}: r = {corr:.3f}")
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for i, j, corr in high_corr_pairs:
            if i not in features_to_remove and j not in features_to_remove:
                # Remove the feature with higher index (arbitrary choice)
                features_to_remove.add(max(i, j))
        
        if features_to_remove:
            features_to_keep = [i for i in range(n_features) if i not in features_to_remove]
            X_decorr = X[:, features_to_keep]
            
            self.fitted_params['decorrelated_features'] = features_to_keep
            
            if verbose:
                print(f"Removed {len(features_to_remove)} correlated features")
            
            return X_decorr
        else:
            return X
    
    def fit_transform(self, X, remove_outliers=True, verbose=True):
        """Complete preprocessing pipeline"""
        
        if verbose:
            print("LINEAR ALGEBRA DATA PREPROCESSING PIPELINE")
            print("=" * 50)
            print(f"Input shape: {X.shape}")
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        X = X.astype(float)
        
        # Step 1: Handle missing values
        X_processed = self.handle_missing_values(X, verbose=verbose)
        
        # Step 2: Detect outliers
        outlier_mask = self.detect_outliers(X_processed, verbose=verbose)
        
        if remove_outliers and np.any(outlier_mask):
            X_processed = X_processed[~outlier_mask]
            if verbose:
                print(f"Removed {np.sum(outlier_mask)} outlier samples")
        
        # Step 3: Remove linear dependencies
        X_processed = self.remove_linear_dependencies(X_processed, verbose=verbose)
        
        # Step 4: Handle feature correlations
        X_processed = self.detect_feature_correlations(X_processed, verbose=verbose)
        
        # Step 5: Apply scaling
        X_processed = self.apply_scaling(X_processed, verbose=verbose)
        
        # Step 6: Dimensionality reduction
        X_processed = self.apply_dimensionality_reduction(X_processed, verbose=verbose)
        
        # Final statistics
        if verbose:
            print(f"\nFINAL RESULTS:")
            print(f"  Output shape: {X_processed.shape}")
            print(f"  Data type: {X_processed.dtype}")
            print(f"  Memory usage: {X_processed.nbytes / 1024:.1f} KB")
            print(f"  No missing values: {not np.isnan(X_processed).any()}")
            print(f"  No infinite values: {not np.isinf(X_processed).any()}")
            
            # Matrix condition number
            if X_processed.shape[1] <= X_processed.shape[0]:
                try:
                    cond_num = np.linalg.cond(X_processed.T @ X_processed)
                    print(f"  Condition number: {cond_num:.2e}")
                except:
                    pass
        
        return X_processed

def demonstrate_preprocessing_pipeline():
    """Demonstrate comprehensive preprocessing with synthetic data"""
    
    print("PREPROCESSING DEMONSTRATION")
    print("=" * 40)
    
    # Create synthetic dataset with various issues
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Base data
    X_base = np.random.randn(n_samples, n_features)
    
    # Add some structure and problems
    # 1. Add linear dependencies
    X_base[:, 5] = 2 * X_base[:, 0] + 0.1 * np.random.randn(n_samples)  # Almost linearly dependent
    X_base[:, 10] = X_base[:, 1] + X_base[:, 2]  # Exactly linearly dependent
    
    # 2. Add high correlations
    X_base[:, 15] = 0.98 * X_base[:, 3] + 0.02 * np.random.randn(n_samples)  # High correlation
    
    # 3. Add different scales
    X_base[:, 8] *= 1000  # Large scale
    X_base[:, 12] *= 0.001  # Small scale
    
    # 4. Add missing values
    missing_indices = np.random.choice(n_samples * n_features, 
                                     size=int(0.05 * n_samples * n_features), 
                                     replace=False)
    X_flat = X_base.flatten()
    X_flat[missing_indices] = np.nan
    X_with_missing = X_flat.reshape(n_samples, n_features)
    
    # 5. Add outliers
    outlier_indices = np.random.choice(n_samples, size=50, replace=False)
    X_with_missing[outlier_indices] *= 5  # Scale outliers
    
    print(f"Created synthetic dataset with issues:")
    print(f"  Shape: {X_with_missing.shape}")
    print(f"  Missing values: {np.sum(np.isnan(X_with_missing))}")
    print(f"  Value range: [{np.nanmin(X_with_missing):.1f}, {np.nanmax(X_with_missing):.1f}]")
    
    # Apply different preprocessing strategies
    strategies = [
        ('Conservative', {
            'missing_strategy': 'mean',
            'outlier_method': 'mahalanobis',
            'scaling_method': 'robust',
            'dim_reduction': None
        }),
        ('Aggressive', {
            'missing_strategy': 'iterative',
            'outlier_method': 'pca_reconstruction',
            'scaling_method': 'standard',
            'dim_reduction': 'pca'
        }),
        ('Matrix-based', {
            'missing_strategy': 'matrix_factorization',
            'outlier_method': 'robust_covariance',
            'scaling_method': 'unit_vector',
            'dim_reduction': 'svd'
        })
    ]
    
    results = {}
    
    for strategy_name, params in strategies:
        print(f"\n{'='*50}")
        print(f"STRATEGY: {strategy_name.upper()}")
        print(f"{'='*50}")
        
        preprocessor = LinearAlgebraDataPreprocessor(**params)
        X_processed = preprocessor.fit_transform(X_with_missing, verbose=True)
        
        results[strategy_name] = {
            'processed_data': X_processed,
            'stats': preprocessor.preprocessing_stats,
            'params': preprocessor.fitted_params
        }
    
    return X_with_missing, results

def create_preprocessing_visualizations(X_original, results):
    """Create visualizations showing preprocessing effects"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Missing value pattern
    ax = axes[0, 0]
    missing_pattern = np.isnan(X_original)
    ax.imshow(missing_pattern[:100, :], cmap='Reds', aspect='auto')
    ax.set_title('Missing Value Pattern\n(First 100 samples)')
    ax.set_xlabel('Features')
    ax.set_ylabel('Samples')
    
    # 2. Feature correlations
    ax = axes[0, 1]
    # Handle missing values for correlation
    X_temp = X_original.copy()
    for j in range(X_temp.shape[1]):
        X_temp[np.isnan(X_temp[:, j]), j] = np.nanmean(X_temp[:, j])
    
    corr_matrix = np.corrcoef(X_temp.T)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=ax)
    
    # 3. Feature scales (before preprocessing)
    ax = axes[0, 2]
    feature_stds = []
    for j in range(X_temp.shape[1]):
        feature_stds.append(np.std(X_temp[:, j]))
    
    ax.bar(range(len(feature_stds)), feature_stds)
    ax.set_title('Feature Standard Deviations\n(Before Preprocessing)')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Standard Deviation')
    ax.set_yscale('log')
    
    # 4. Compare processed data distributions
    ax = axes[1, 0]
    for i, (strategy, result) in enumerate(results.items()):
        X_processed = result['processed_data']
        # Plot first feature distribution
        ax.hist(X_processed[:, 0], alpha=0.6, label=strategy, bins=30)
    ax.set_title('First Feature Distribution\n(After Preprocessing)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # 5. Dimensionality comparison
    ax = axes[1, 1]
    strategies = list(results.keys())
    dimensions = [results[s]['processed_data'].shape[1] for s in strategies]
    original_dim = X_original.shape[1]
    
    bars = ax.bar(['Original'] + strategies, [original_dim] + dimensions)
    bars[0].set_color('red')  # Highlight original
    ax.set_title('Dimensionality Comparison')
    ax.set_ylabel('Number of Features')
    
    # 6. Processing statistics
    ax = axes[1, 2]
    stats_to_plot = ['missing_imputed', 'outliers_detected']
    x_pos = np.arange(len(strategies))
    
    for i, stat in enumerate(stats_to_plot):
        values = [results[s]['stats'].get(stat, 0) for s in strategies]
        ax.bar(x_pos + i*0.35, values, 0.35, label=stat.replace('_', ' ').title())
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Count')
    ax.set_title('Preprocessing Statistics')
    ax.set_xticks(x_pos + 0.175)
    ax.set_xticklabels(strategies)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('linear_algebra_preprocessing.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run demonstration
    X_original, preprocessing_results = demonstrate_preprocessing_pipeline()
    
    # Create visualizations
    create_preprocessing_visualizations(X_original, preprocessing_results)
    
    print(f"\n{'='*60}")
    print("LINEAR ALGEBRA PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    
    print("\nKey Linear Algebra Applications:")
    print("• Matrix operations for vectorized computation")
    print("• SVD/PCA for dimensionality reduction and missing value imputation")
    print("• QR decomposition for linear dependency detection")
    print("• Eigenvalue decomposition for outlier detection (Mahalanobis distance)")
    print("• Matrix norms for scaling and normalization")
    
    print("\nComputational Advantages:")
    print("• Vectorized operations are much faster than loops")
    print("• BLAS/LAPACK optimizations for matrix operations")
    print("• Parallel processing capabilities")
    print("• Memory-efficient operations")
    
    print("\nBest Practices:")
    print("• Choose appropriate numerical tolerances")
    print("• Handle singular/ill-conditioned matrices")
    print("• Validate preprocessing steps with multiple methods")
    print("• Monitor condition numbers for numerical stability")
    print("• Use robust methods for outlier-prone data")
    
    print("\nCommon Pitfalls:")
    print("• Data leakage: Don't use test set statistics for preprocessing")
    print("• Scaling after splitting: Fit on training, transform on test")
    print("• Ignoring numerical precision and overflow issues")
    print("• Over-preprocessing: Removing too much information")
```

**Advanced Linear Algebra Techniques:**

**1. Robust Principal Component Analysis (RPCA):**
Decomposes matrix as: X = L + S + N
- L: Low-rank component (main structure)
- S: Sparse component (outliers/anomalies)  
- N: Noise component

**2. Matrix Completion with Nuclear Norm:**
Minimize: ||X||* subject to constraints
Where ||X||* is the nuclear norm (sum of singular values)

**3. Independent Component Analysis (ICA):**
Find linear transformation: S = WX
Where S has independent components (non-Gaussian)

**4. Canonical Correlation Analysis (CCA):**
Find linear combinations maximizing correlation between datasets

**Key Insights:**
- Linear algebra provides the mathematical foundation for systematic data preprocessing
- Matrix decompositions reveal data structure and enable principled cleaning approaches
- Vectorized operations ensure computational efficiency for large datasets
- Understanding the mathematical properties helps choose appropriate methods and parameters

---

## Question 32

**Describe a scenario where linear algebra could be used to improve model accuracy.**

**Answer:** Linear algebra provides powerful tools that can significantly improve model accuracy through feature engineering, regularization, optimization, and architectural improvements. One compelling scenario is using linear algebra techniques to enhance the performance of a recommendation system for an e-commerce platform, where accuracy improvements directly translate to better user experience and increased revenue.

**Scenario: E-commerce Recommendation System Enhancement**

**Initial Challenge:**
A large e-commerce platform has a recommendation system with suboptimal performance:
- **Current accuracy**: 65% precision@10 for product recommendations
- **User engagement**: Low click-through rates (2.1%)
- **Business impact**: Poor conversion rates and customer satisfaction
- **Technical issues**: Cold start problem, sparse user-item interactions, scalability concerns

**Linear Algebra Solutions for Accuracy Improvement:**

**1. Matrix Factorization Enhancement:**

**Problem**: Basic collaborative filtering yields poor recommendations due to sparsity.

**Linear Algebra Solution**: Advanced matrix factorization with regularization and bias terms.

**Mathematical Framework:**
```
R ≈ U V^T + μ + b_u + b_i
```
Where:
- R: User-item rating matrix
- U ∈ ℝ^(m×k): User latent factors
- V ∈ ℝ^(n×k): Item latent factors  
- μ: Global bias
- b_u: User bias vector
- b_i: Item bias vector

**Optimization with Regularization:**
```
min_{U,V,b} ||R - UV^T - μ1^T - b_u1^T - 1b_i^T||_F^2 + λ(||U||_F^2 + ||V||_F^2 + ||b_u||^2 + ||b_i||^2)
```

**Implementation:**

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

class AdvancedMatrixFactorization:
    """Advanced matrix factorization for recommendation systems"""
    
    def __init__(self, n_factors=50, learning_rate=0.01, reg_lambda=0.01, 
                 n_epochs=100, use_bias=True, use_temporal=False):
        """
        Initialize advanced matrix factorization model
        
        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            reg_lambda: Regularization parameter
            n_epochs: Number of training epochs
            use_bias: Whether to include bias terms
            use_temporal: Whether to include temporal dynamics
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        self.use_bias = use_bias
        self.use_temporal = use_temporal
        
        # Model parameters
        self.U = None  # User factors
        self.V = None  # Item factors
        self.mu = None  # Global mean
        self.b_u = None  # User biases
        self.b_i = None  # Item biases
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def initialize_parameters(self, n_users, n_items):
        """Initialize model parameters"""
        
        # Xavier initialization for factors
        self.U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        if self.use_bias:
            self.b_u = np.zeros(n_users)
            self.b_i = np.zeros(n_items)
        
        self.mu = 0.0
    
    def predict_rating(self, user_id, item_id, timestamp=None):
        """Predict rating for user-item pair"""
        
        prediction = self.mu + np.dot(self.U[user_id], self.V[item_id])
        
        if self.use_bias:
            prediction += self.b_u[user_id] + self.b_i[item_id]
        
        # Add temporal dynamics if enabled
        if self.use_temporal and timestamp is not None:
            # Simplified temporal factor
            temporal_factor = 1.0 + 0.01 * np.sin(timestamp / 86400)  # Daily pattern
            prediction *= temporal_factor
        
        return prediction
    
    def compute_loss(self, rating_matrix, mask_matrix):
        """Compute total loss including regularization"""
        
        # Reconstruction loss
        predictions = self.U @ self.V.T
        if self.use_bias:
            predictions += self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
        
        mse_loss = np.sum(mask_matrix * (rating_matrix - predictions)**2)
        
        # Regularization loss
        reg_loss = self.reg_lambda * (
            np.sum(self.U**2) + np.sum(self.V**2)
        )
        
        if self.use_bias:
            reg_loss += self.reg_lambda * (np.sum(self.b_u**2) + np.sum(self.b_i**2))
        
        return mse_loss + reg_loss
    
    def fit(self, rating_matrix, validation_data=None, verbose=True):
        """Train the matrix factorization model"""
        
        if verbose:
            print("Training Advanced Matrix Factorization")
            print("=" * 45)
        
        # Convert to dense if sparse
        if sp.issparse(rating_matrix):
            rating_matrix = rating_matrix.toarray()
        
        n_users, n_items = rating_matrix.shape
        
        # Create mask for observed ratings
        mask_matrix = (rating_matrix > 0).astype(float)
        
        # Compute global mean
        self.mu = np.sum(rating_matrix) / np.sum(mask_matrix)
        
        # Initialize parameters
        self.initialize_parameters(n_users, n_items)
        
        # Get list of observed ratings for SGD
        user_ids, item_ids = np.where(mask_matrix)
        ratings = rating_matrix[user_ids, item_ids]
        
        if verbose:
            print(f"Dataset statistics:")
            print(f"  Users: {n_users}")
            print(f"  Items: {n_items}")
            print(f"  Ratings: {len(ratings)}")
            print(f"  Sparsity: {1 - len(ratings)/(n_users*n_items):.3f}")
            print(f"  Global mean: {self.mu:.3f}")
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle training data
            indices = np.random.permutation(len(ratings))
            epoch_loss = 0
            
            for idx in indices:
                u = user_ids[idx]
                i = item_ids[idx]
                r_ui = ratings[idx]
                
                # Predict rating
                prediction = self.predict_rating(u, i)
                error = r_ui - prediction
                
                # Update parameters using gradient descent
                # User factors
                u_factors = self.U[u].copy()
                self.U[u] += self.learning_rate * (
                    error * self.V[i] - self.reg_lambda * self.U[u]
                )
                
                # Item factors
                self.V[i] += self.learning_rate * (
                    error * u_factors - self.reg_lambda * self.V[i]
                )
                
                # Biases
                if self.use_bias:
                    self.b_u[u] += self.learning_rate * (
                        error - self.reg_lambda * self.b_u[u]
                    )
                    self.b_i[i] += self.learning_rate * (
                        error - self.reg_lambda * self.b_i[i]
                    )
                
                epoch_loss += error**2
            
            # Compute full loss
            total_loss = self.compute_loss(rating_matrix, mask_matrix)
            self.train_losses.append(total_loss)
            
            # Validation loss
            if validation_data is not None:
                val_loss = self.evaluate(validation_data['matrix'], validation_data['mask'])
                self.val_losses.append(val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                if validation_data is not None:
                    print(f"Epoch {epoch+1:3d}: Train Loss = {total_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1:3d}: Train Loss = {total_loss:.4f}")
        
        if verbose:
            print("Training completed!")
    
    def evaluate(self, test_matrix, test_mask):
        """Evaluate model on test data"""
        
        predictions = self.U @ self.V.T
        if self.use_bias:
            predictions += self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
        
        # Compute RMSE on test set
        test_predictions = predictions[test_mask > 0]
        test_actuals = test_matrix[test_mask > 0]
        
        rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        return rmse
    
    def recommend_items(self, user_id, n_recommendations=10, exclude_rated=True):
        """Generate top-N recommendations for a user"""
        
        if user_id >= self.U.shape[0]:
            return []
        
        # Predict all items for user
        user_predictions = self.U[user_id] @ self.V.T
        
        if self.use_bias:
            user_predictions += self.mu + self.b_u[user_id] + self.b_i
        
        # Get top items
        if exclude_rated:
            # This would need the original rating matrix to exclude rated items
            # For simplicity, we'll return top predictions
            pass
        
        top_items = np.argsort(user_predictions)[::-1][:n_recommendations]
        top_scores = user_predictions[top_items]
        
        return list(zip(top_items, top_scores))

def create_synthetic_recommendation_data():
    """Create synthetic recommendation dataset"""
    
    np.random.seed(42)
    
    n_users = 1000
    n_items = 500
    n_factors_true = 10
    sparsity = 0.95  # 95% sparse
    
    # Generate true user and item factors
    U_true = np.random.normal(0, 1, (n_users, n_factors_true))
    V_true = np.random.normal(0, 1, (n_items, n_factors_true))
    
    # Generate ratings with noise
    R_true = U_true @ V_true.T
    
    # Add user and item biases
    user_bias = np.random.normal(0, 0.5, n_users)
    item_bias = np.random.normal(0, 0.5, n_items)
    global_mean = 3.5
    
    R_true += global_mean + user_bias[:, np.newaxis] + item_bias[np.newaxis, :]
    
    # Add noise
    R_true += np.random.normal(0, 0.1, (n_users, n_items))
    
    # Create sparse observation mask
    mask = np.random.random((n_users, n_items)) > sparsity
    R_observed = R_true * mask
    
    # Scale ratings to 1-5 range
    R_observed = np.clip(R_observed, 1, 5)
    
    return R_observed, mask, R_true

def demonstrate_accuracy_improvement():
    """Demonstrate accuracy improvement using linear algebra"""
    
    print("LINEAR ALGEBRA FOR MODEL ACCURACY IMPROVEMENT")
    print("=" * 55)
    
    # Create synthetic data
    R_observed, mask, R_true = create_synthetic_recommendation_data()
    
    print(f"Dataset created:")
    print(f"  Users: {R_observed.shape[0]}")
    print(f"  Items: {R_observed.shape[1]}")
    print(f"  Observed ratings: {np.sum(mask)}")
    print(f"  Sparsity: {1 - np.sum(mask)/mask.size:.3f}")
    
    # Split into train/test
    train_mask = mask.copy()
    test_indices = np.random.choice(np.sum(mask), size=int(0.2 * np.sum(mask)), replace=False)
    
    # Convert to coordinate format for easier manipulation
    user_ids, item_ids = np.where(mask)
    test_users = user_ids[test_indices]
    test_items = item_ids[test_indices]
    
    # Remove test ratings from training set
    for u, i in zip(test_users, test_items):
        train_mask[u, i] = False
    
    # Create test mask
    test_mask = np.zeros_like(mask, dtype=bool)
    for u, i in zip(test_users, test_items):
        test_mask[u, i] = True
    
    print(f"Train/test split:")
    print(f"  Training ratings: {np.sum(train_mask)}")
    print(f"  Test ratings: {np.sum(test_mask)}")
    
    # Baseline: Simple mean prediction
    global_mean = np.sum(R_observed * train_mask) / np.sum(train_mask)
    baseline_predictions = np.full_like(R_observed, global_mean)
    baseline_rmse = np.sqrt(mean_squared_error(
        R_observed[test_mask], baseline_predictions[test_mask]
    ))
    
    print(f"\nBaseline Results:")
    print(f"  Global mean: {global_mean:.3f}")
    print(f"  RMSE: {baseline_rmse:.4f}")
    
    # Method 1: Basic SVD
    print(f"\nMethod 1: Basic SVD")
    print("-" * 25)
    
    # Fill missing values with global mean for SVD
    R_filled = R_observed.copy()
    R_filled[~train_mask] = global_mean
    
    # Perform SVD
    k = 20  # Number of factors
    U, s, Vt = svds(R_filled, k=k)
    
    # Reconstruct matrix
    R_svd = U @ np.diag(s) @ Vt
    
    svd_rmse = np.sqrt(mean_squared_error(R_observed[test_mask], R_svd[test_mask]))
    print(f"  SVD RMSE: {svd_rmse:.4f}")
    print(f"  Improvement: {(baseline_rmse - svd_rmse)/baseline_rmse*100:.1f}%")
    
    # Method 2: Advanced Matrix Factorization
    print(f"\nMethod 2: Advanced Matrix Factorization")
    print("-" * 40)
    
    # Prepare validation data
    val_data = {
        'matrix': R_observed,
        'mask': test_mask
    }
    
    # Train advanced model
    advanced_mf = AdvancedMatrixFactorization(
        n_factors=50,
        learning_rate=0.01,
        reg_lambda=0.001,
        n_epochs=100,
        use_bias=True
    )
    
    R_train = R_observed * train_mask.astype(float)
    advanced_mf.fit(R_train, validation_data=val_data, verbose=False)
    
    # Evaluate
    advanced_rmse = advanced_mf.evaluate(R_observed, test_mask)
    print(f"  Advanced MF RMSE: {advanced_rmse:.4f}")
    print(f"  Improvement over baseline: {(baseline_rmse - advanced_rmse)/baseline_rmse*100:.1f}%")
    print(f"  Improvement over SVD: {(svd_rmse - advanced_rmse)/svd_rmse*100:.1f}%")
    
    return {
        'baseline_rmse': baseline_rmse,
        'svd_rmse': svd_rmse,
        'advanced_rmse': advanced_rmse,
        'model': advanced_mf
    }

def demonstrate_feature_engineering_improvement():
    """Show how linear algebra improves feature engineering"""
    
    print(f"\nFEATURE ENGINEERING WITH LINEAR ALGEBRA")
    print("=" * 45)
    
    # Create high-dimensional feature data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # Generate data with latent structure
    n_latent = 5
    W_true = np.random.randn(n_features, n_latent)
    Z = np.random.randn(n_samples, n_latent)
    X = Z @ W_true.T + 0.1 * np.random.randn(n_samples, n_features)
    
    # Create target with complex interactions
    y = (np.sum(Z[:, :3], axis=1) + 
         0.5 * np.prod(Z[:, :2], axis=1) + 
         np.random.randn(n_samples) * 0.1)
    
    print(f"Original data: {X.shape}")
    print(f"True latent dimensions: {n_latent}")
    
    # Baseline: Linear regression on raw features
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    baseline_score = r2_score(y_test, baseline_model.predict(X_test))
    
    print(f"Baseline R²: {baseline_score:.4f}")
    
    # Improvement 1: PCA feature extraction
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_latent)
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test = pca.transform(X_test)
    
    pca_model = LinearRegression()
    pca_model.fit(X_pca_train, y_train)
    pca_score = r2_score(y_test, pca_model.predict(X_pca_test))
    
    print(f"PCA R²: {pca_score:.4f}")
    print(f"Improvement: {(pca_score - baseline_score)/baseline_score*100:.1f}%")
    
    # Improvement 2: Feature interactions via tensor products
    def create_interaction_features(X, degree=2):
        """Create polynomial interaction features"""
        from itertools import combinations_with_replacement
        
        n_samples, n_features = X.shape
        interaction_features = []
        
        # Add original features
        interaction_features.append(X)
        
        # Add degree-2 interactions
        if degree >= 2:
            interactions = []
            for i, j in combinations_with_replacement(range(n_features), 2):
                interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
            
            if interactions:
                interaction_features.append(np.hstack(interactions))
        
        return np.hstack(interaction_features)
    
    # Apply to PCA features only (to keep dimensionality manageable)
    X_pca_interact_train = create_interaction_features(X_pca_train, degree=2)
    X_pca_interact_test = create_interaction_features(X_pca_test, degree=2)
    
    interact_model = LinearRegression()
    interact_model.fit(X_pca_interact_train, y_train)
    interact_score = r2_score(y_test, interact_model.predict(X_pca_interact_test))
    
    print(f"PCA + Interactions R²: {interact_score:.4f}")
    print(f"Total improvement: {(interact_score - baseline_score)/baseline_score*100:.1f}%")

def visualize_accuracy_improvements(results):
    """Create visualization of accuracy improvements"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. RMSE comparison
    ax = axes[0]
    methods = ['Baseline\n(Global Mean)', 'SVD', 'Advanced MF']
    rmse_values = [results['baseline_rmse'], results['svd_rmse'], results['advanced_rmse']]
    
    bars = ax.bar(methods, rmse_values, color=['red', 'orange', 'green'])
    ax.set_ylabel('RMSE')
    ax.set_title('Recommendation Accuracy Improvement')
    
    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Training progress
    ax = axes[1]
    model = results['model']
    epochs = range(1, len(model.train_losses) + 1)
    
    ax.plot(epochs, model.train_losses, label='Training Loss', color='blue')
    if model.val_losses:
        ax.plot(epochs, model.val_losses, label='Validation Loss', color='red')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('accuracy_improvement_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    print("SCENARIO: E-COMMERCE RECOMMENDATION SYSTEM IMPROVEMENT")
    print("=" * 65)
    
    # Main demonstration
    results = demonstrate_accuracy_improvement()
    
    # Feature engineering demonstration
    demonstrate_feature_engineering_improvement()
    
    # Visualizations
    visualize_accuracy_improvements(results)
    
    print(f"\n{'='*65}")
    print("ACCURACY IMPROVEMENT SUMMARY")
    print(f"{'='*65}")
    
    # Calculate improvements
    baseline_rmse = results['baseline_rmse']
    advanced_rmse = results['advanced_rmse']
    improvement = (baseline_rmse - advanced_rmse) / baseline_rmse * 100
    
    print(f"\nKey Results:")
    print(f"• Baseline RMSE: {baseline_rmse:.4f}")
    print(f"• Advanced Matrix Factorization RMSE: {advanced_rmse:.4f}")
    print(f"• Total improvement: {improvement:.1f}%")
    
    print(f"\nLinear Algebra Techniques Used:")
    print("• Matrix factorization (SVD, NMF)")
    print("• Regularized optimization (Ridge, Lasso)")
    print("• Principal Component Analysis (PCA)")
    print("• Bias correction through matrix operations")
    print("• Feature interactions via tensor products")
    print("• Gradient-based optimization")
    
    print(f"\nBusiness Impact:")
    print("• Improved recommendation accuracy → Higher click-through rates")
    print("• Better user experience → Increased customer satisfaction")
    print("• More relevant suggestions → Higher conversion rates")
    print("• Reduced computational complexity → Lower operational costs")
    
    print(f"\nTechnical Benefits:")
    print("• Handles sparse data effectively")
    print("• Scalable to large datasets")
    print("• Incorporates multiple signal types")
    print("• Provides interpretable latent factors")
    print("• Enables online learning and updates")
```

**Additional Accuracy Improvement Scenarios:**

**2. Computer Vision - Image Classification Enhancement:**
- **Technique**: Convolutional kernels as learnable linear transformations
- **Improvement**: Data augmentation through affine transformations
- **Result**: 15-25% accuracy boost through geometric invariance

**3. Natural Language Processing - Embedding Optimization:**
- **Technique**: Linear transformations in embedding spaces
- **Improvement**: Cross-lingual alignment using Procrustes analysis
- **Result**: 20-30% improvement in multilingual tasks

**4. Time Series Forecasting - State Space Models:**
- **Technique**: Kalman filtering with linear algebra
- **Improvement**: Optimal state estimation under noise
- **Result**: 10-40% reduction in forecasting error

**Key Principles for Accuracy Improvement:**
1. **Dimensionality Reduction**: Remove noise, keep signal
2. **Regularization**: Prevent overfitting through constraint optimization
3. **Feature Engineering**: Create meaningful representations via linear transformations
4. **Optimization**: Use gradient-based methods for parameter learning
5. **Ensemble Methods**: Combine models through weighted linear combinations

---

## Question 33

**What are sparse matrices and how are they efficiently represented and used?**

**Answer:** Sparse matrices are matrices where most elements are zero, making them memory-inefficient to store in dense format. They arise naturally in many machine learning applications including text processing, graph analysis, collaborative filtering, and scientific computing. Efficient representation and operations on sparse matrices are crucial for scalability and performance in large-scale applications.

**Definition and Characteristics:**

A matrix is considered **sparse** when:
- **Sparsity ratio**: (number of zeros) / (total elements) > 0.95
- **Density**: (number of non-zeros) / (total elements) < 0.05
- **Storage consideration**: Storing only non-zero elements saves significant memory

**Mathematical Properties:**
- **Addition**: A + B remains sparse if both A and B are sparse
- **Multiplication**: A × B may become dense even if A and B are sparse
- **Transpose**: Sparsity pattern changes but remains sparse
- **Factorization**: May produce dense factors requiring special algorithms

**Efficient Storage Formats:**

**1. Coordinate List (COO) Format:**
Stores triplets (row, column, value) for each non-zero element.

```
Structure: (row_indices, col_indices, data)
Memory: O(nnz) where nnz = number of non-zeros
```

**Advantages**: Simple to construct and understand
**Disadvantages**: Inefficient for arithmetic operations

**2. Compressed Sparse Row (CSR) Format:**
Compresses row information using indirection.

```
Structure: (data, indices, indptr)
- data[i]: value of i-th non-zero element
- indices[i]: column index of i-th non-zero element  
- indptr[i]: index in data where row i starts
```

**3. Compressed Sparse Column (CSC) Format:**
Similar to CSR but compresses column information.

**4. Dictionary of Keys (DOK) Format:**
Uses dictionary mapping (row, col) → value.

**5. Block Sparse Row (BSR) Format:**
Optimized for matrices with dense sub-blocks.

**Comprehensive Implementation:**

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, eigsh
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

class SparseMatrixAnalyzer:
    """Comprehensive sparse matrix analysis and operations"""
    
    def __init__(self):
        """Initialize sparse matrix analyzer"""
        self.timing_results = {}
        
    def demonstrate_storage_formats(self):
        """Demonstrate different sparse storage formats"""
        
        print("SPARSE MATRIX STORAGE FORMATS")
        print("=" * 40)
        
        # Create example sparse matrix
        np.random.seed(42)
        n_rows, n_cols = 1000, 800
        density = 0.01  # 1% non-zero elements
        
        # Generate sparse matrix
        nnz = int(n_rows * n_cols * density)
        rows = np.random.randint(0, n_rows, nnz)
        cols = np.random.randint(0, n_cols, nnz)
        data = np.random.randn(nnz)
        
        print(f"Example matrix: {n_rows}×{n_cols}, density={density:.1%}")
        print(f"Non-zero elements: {nnz}")
        print(f"Memory if dense: {n_rows * n_cols * 8 / 1024**2:.1f} MB")
        
        # Create matrices in different formats
        formats = {}
        
        # COO format
        coo_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
        formats['COO'] = coo_matrix
        
        # CSR format
        csr_matrix = coo_matrix.tocsr()
        formats['CSR'] = csr_matrix
        
        # CSC format  
        csc_matrix = coo_matrix.tocsc()
        formats['CSC'] = csc_matrix
        
        # DOK format
        dok_matrix = coo_matrix.todok()
        formats['DOK'] = dok_matrix
        
        # Analyze storage efficiency
        print(f"\nStorage Format Comparison:")
        print(f"{'Format':<6} {'Memory (KB)':<12} {'Efficiency':<12} {'Best For':<30}")
        print("-" * 70)
        
        format_info = {
            'COO': {'memory': coo_matrix.data.nbytes + coo_matrix.row.nbytes + coo_matrix.col.nbytes,
                   'best_for': 'Construction, conversion'},
            'CSR': {'memory': csr_matrix.data.nbytes + csr_matrix.indices.nbytes + csr_matrix.indptr.nbytes,
                   'best_for': 'Row operations, matrix-vector mult'},
            'CSC': {'memory': csc_matrix.data.nbytes + csc_matrix.indices.nbytes + csc_matrix.indptr.nbytes,
                   'best_for': 'Column operations, decomposition'},
            'DOK': {'memory': len(dok_matrix.keys()) * (3 * 8),  # Approximate
                   'best_for': 'Incremental construction'}
        }
        
        for fmt_name, matrix in formats.items():
            memory_kb = format_info[fmt_name]['memory'] / 1024
            efficiency = memory_kb / (n_rows * n_cols * 8 / 1024) * 100
            best_for = format_info[fmt_name]['best_for']
            
            print(f"{fmt_name:<6} {memory_kb:<12.1f} {efficiency:<12.1f}% {best_for:<30}")
        
        return formats
    
    def benchmark_operations(self, formats):
        """Benchmark operations across different formats"""
        
        print(f"\nOPERATION PERFORMANCE COMPARISON")
        print("=" * 40)
        
        n_tests = 5
        operations = ['matrix_vector', 'matrix_matrix', 'transpose', 'element_access']
        
        # Create test vectors and matrices
        n_rows, n_cols = formats['CSR'].shape
        vector = np.random.randn(n_cols)
        small_matrix = np.random.randn(n_cols, 50)
        
        benchmark_results = {}
        
        for fmt_name, matrix in formats.items():
            if fmt_name not in benchmark_results:
                benchmark_results[fmt_name] = {}
            
            print(f"\nBenchmarking {fmt_name} format:")
            
            # 1. Matrix-vector multiplication
            if hasattr(matrix, 'dot'):
                times = []
                for _ in range(n_tests):
                    start_time = time()
                    result = matrix.dot(vector)
                    times.append(time() - start_time)
                
                avg_time = np.mean(times)
                benchmark_results[fmt_name]['matrix_vector'] = avg_time
                print(f"  Matrix-vector: {avg_time:.4f}s")
            
            # 2. Matrix-matrix multiplication
            if hasattr(matrix, 'dot') and fmt_name != 'DOK':  # DOK is too slow for this
                times = []
                for _ in range(n_tests):
                    start_time = time()
                    result = matrix.dot(small_matrix)
                    times.append(time() - start_time)
                
                avg_time = np.mean(times)
                benchmark_results[fmt_name]['matrix_matrix'] = avg_time
                print(f"  Matrix-matrix: {avg_time:.4f}s")
            
            # 3. Transpose
            times = []
            for _ in range(n_tests):
                start_time = time()
                result = matrix.T
                times.append(time() - start_time)
            
            avg_time = np.mean(times)
            benchmark_results[fmt_name]['transpose'] = avg_time
            print(f"  Transpose: {avg_time:.4f}s")
            
            # 4. Element access (for formats that support it efficiently)
            if fmt_name in ['CSR', 'CSC', 'DOK']:
                times = []
                test_indices = [(100, 100), (500, 300), (800, 600)]
                
                for _ in range(n_tests):
                    start_time = time()
                    for i, j in test_indices:
                        val = matrix[i, j]
                    times.append(time() - start_time)
                
                avg_time = np.mean(times)
                benchmark_results[fmt_name]['element_access'] = avg_time
                print(f"  Element access: {avg_time:.4f}s")
        
        return benchmark_results
    
    def demonstrate_sparse_algorithms(self):
        """Demonstrate algorithms optimized for sparse matrices"""
        
        print(f"\nSPARSE-OPTIMIZED ALGORITHMS")
        print("=" * 35)
        
        # Create sparse test matrix
        np.random.seed(42)
        n = 1000
        density = 0.005
        
        # Create symmetric positive definite sparse matrix
        A_dense = np.random.randn(n, n) * 0.1
        A_dense = A_dense + A_dense.T + n * np.eye(n)  # Make SPD
        
        # Sparsify
        mask = np.random.random((n, n)) < density
        mask = mask | mask.T  # Keep symmetry
        np.fill_diagonal(mask, True)  # Keep diagonal
        
        A_sparse = sp.csr_matrix(A_dense * mask)
        
        print(f"Test matrix: {n}×{n}, density={A_sparse.nnz/(n*n):.1%}")
        print(f"Non-zeros: {A_sparse.nnz}")
        
        # 1. Sparse linear system solving
        print(f"\n1. Linear System Solving:")
        b = np.random.randn(n)
        
        start_time = time()
        x_sparse = spsolve(A_sparse, b)
        sparse_solve_time = time() - start_time
        
        print(f"   Sparse solver time: {sparse_solve_time:.4f}s")
        print(f"   Solution norm: {np.linalg.norm(x_sparse):.4f}")
        print(f"   Residual norm: {np.linalg.norm(A_sparse @ x_sparse - b):.2e}")
        
        # 2. Eigenvalue computation
        print(f"\n2. Eigenvalue Computation:")
        k = 5  # Number of eigenvalues
        
        start_time = time()
        eigenvals, eigenvecs = eigsh(A_sparse, k=k, which='LM')
        sparse_eigen_time = time() - start_time
        
        print(f"   Sparse eigenvalue time: {sparse_eigen_time:.4f}s")
        print(f"   Largest eigenvalues: {eigenvals}")
        
        return A_sparse, x_sparse, eigenvals, eigenvecs
    
    def text_analysis_example(self):
        """Real-world example: Text analysis with sparse matrices"""
        
        print(f"\nREAL-WORLD EXAMPLE: TEXT ANALYSIS")
        print("=" * 40)
        
        # Load text dataset
        print("Loading 20 newsgroups dataset...")
        categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
        newsgroups = fetch_20newsgroups(subset='train', categories=categories, 
                                       remove=('headers', 'footers', 'quotes'))
        
        print(f"Documents: {len(newsgroups.data)}")
        print(f"Categories: {len(categories)}")
        
        # Create TF-IDF matrix (naturally sparse)
        print("\nCreating TF-IDF matrix...")
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english',
                                   min_df=2, max_df=0.95)
        
        start_time = time()
        X_tfidf = vectorizer.fit_transform(newsgroups.data)
        vectorization_time = time() - start_time
        
        print(f"TF-IDF matrix shape: {X_tfidf.shape}")
        print(f"Sparsity: {1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.3f}")
        print(f"Non-zero elements: {X_tfidf.nnz}")
        print(f"Vectorization time: {vectorization_time:.2f}s")
        
        # Memory comparison
        dense_memory = X_tfidf.shape[0] * X_tfidf.shape[1] * 8 / 1024**2
        sparse_memory = (X_tfidf.data.nbytes + X_tfidf.indices.nbytes + 
                        X_tfidf.indptr.nbytes) / 1024**2
        
        print(f"\nMemory usage:")
        print(f"  Dense representation: {dense_memory:.1f} MB")
        print(f"  Sparse representation: {sparse_memory:.1f} MB")
        print(f"  Memory savings: {(1 - sparse_memory/dense_memory)*100:.1f}%")
        
        # Demonstrate sparse operations
        print(f"\nSparse matrix operations:")
        
        # Document similarity (cosine similarity via dot product)
        start_time = time()
        doc_similarities = X_tfidf @ X_tfidf.T
        similarity_time = time() - start_time
        
        print(f"  Document similarity computation: {similarity_time:.4f}s")
        print(f"  Similarity matrix shape: {doc_similarities.shape}")
        print(f"  Similarity matrix sparsity: {1 - doc_similarities.nnz / (doc_similarities.shape[0] * doc_similarities.shape[1]):.3f}")
        
        # Feature selection via variance
        feature_variances = np.array(X_tfidf.multiply(X_tfidf).mean(axis=0)).flatten()
        top_features_idx = np.argsort(feature_variances)[-20:]
        feature_names = vectorizer.get_feature_names_out()
        top_features = [feature_names[i] for i in top_features_idx]
        
        print(f"  Top 10 features by variance: {top_features[-10:]}")
        
        return X_tfidf, newsgroups, vectorizer
    
    def graph_analysis_example(self):
        """Demonstrate sparse matrices in graph analysis"""
        
        print(f"\nGRAPH ANALYSIS WITH SPARSE MATRICES")
        print("=" * 40)
        
        # Create random graph
        np.random.seed(42)
        n_nodes = 1000
        edge_prob = 0.01  # 1% edge probability
        
        # Generate adjacency matrix
        adj_matrix_dense = (np.random.random((n_nodes, n_nodes)) < edge_prob).astype(float)
        
        # Make symmetric (undirected graph)
        adj_matrix_dense = (adj_matrix_dense + adj_matrix_dense.T) > 0
        np.fill_diagonal(adj_matrix_dense, 0)  # No self-loops
        
        # Convert to sparse
        adj_matrix = sp.csr_matrix(adj_matrix_dense.astype(float))
        
        print(f"Graph with {n_nodes} nodes")
        print(f"Edges: {adj_matrix.nnz // 2}")  # Divide by 2 for undirected
        print(f"Density: {adj_matrix.nnz / (n_nodes * n_nodes):.4f}")
        
        # Graph algorithms using sparse operations
        print(f"\nGraph algorithms:")
        
        # 1. Degree centrality
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        max_degree_node = np.argmax(degrees)
        
        print(f"  Max degree: {degrees[max_degree_node]} (node {max_degree_node})")
        print(f"  Average degree: {np.mean(degrees):.2f}")
        
        # 2. PageRank (power iteration with sparse matrix)
        def sparse_pagerank(adj_matrix, damping=0.85, max_iter=100, tol=1e-6):
            """Compute PageRank using sparse matrix operations"""
            
            n_nodes = adj_matrix.shape[0]
            
            # Normalize adjacency matrix (transition matrix)
            out_degrees = np.array(adj_matrix.sum(axis=1)).flatten()
            out_degrees[out_degrees == 0] = 1  # Avoid division by zero
            
            # Create transition matrix
            D_inv = sp.diags(1.0 / out_degrees)
            transition_matrix = adj_matrix.T @ D_inv
            
            # Initialize PageRank vector
            pagerank = np.ones(n_nodes) / n_nodes
            
            for iteration in range(max_iter):
                old_pagerank = pagerank.copy()
                
                # PageRank update: PR = (1-d)/N + d * M * PR
                pagerank = ((1 - damping) / n_nodes + 
                           damping * transition_matrix @ pagerank)
                
                # Check convergence
                if np.linalg.norm(pagerank - old_pagerank) < tol:
                    break
            
            return pagerank
        
        start_time = time()
        pagerank_scores = sparse_pagerank(adj_matrix)
        pagerank_time = time() - start_time
        
        top_pagerank_nodes = np.argsort(pagerank_scores)[-5:]
        
        print(f"  PageRank computation time: {pagerank_time:.4f}s")
        print(f"  Top 5 PageRank nodes: {top_pagerank_nodes}")
        print(f"  Their PageRank scores: {pagerank_scores[top_pagerank_nodes]}")
        
        return adj_matrix, pagerank_scores
    
    def sparse_machine_learning_example(self):
        """Demonstrate sparse matrices in machine learning"""
        
        print(f"\nSPARSE MATRICES IN MACHINE LEARNING")
        print("=" * 40)
        
        # Create sparse feature matrix (e.g., one-hot encoded categorical features)
        np.random.seed(42)
        n_samples = 5000
        n_categories = [100, 50, 200, 75]  # Different categorical features
        
        # Generate sparse one-hot encoded features
        sparse_features = []
        
        for i, n_cat in enumerate(n_categories):
            # Random categorical assignments
            categories = np.random.randint(0, n_cat, n_samples)
            
            # One-hot encoding
            one_hot = sp.csr_matrix((np.ones(n_samples), 
                                   (np.arange(n_samples), categories)),
                                  shape=(n_samples, n_cat))
            sparse_features.append(one_hot)
        
        # Combine all features
        X_sparse = sp.hstack(sparse_features)
        
        # Generate target
        # Create some meaningful relationships with sparse features
        true_weights = np.random.randn(X_sparse.shape[1]) * 0.1
        y = X_sparse @ true_weights + np.random.randn(n_samples) * 0.1
        
        print(f"Feature matrix shape: {X_sparse.shape}")
        print(f"Sparsity: {1 - X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.3f}")
        print(f"Memory usage: {(X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes) / 1024**2:.1f} MB")
        
        # Train sparse linear regression
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sparse, y, test_size=0.2, random_state=42
        )
        
        # Ridge regression (handles sparse matrices efficiently)
        print(f"\nTraining Ridge regression...")
        start_time = time()
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        
        training_time = time() - start_time
        
        # Predict
        train_score = ridge.score(X_train, y_train)
        test_score = ridge.score(X_test, y_test)
        
        print(f"  Training time: {training_time:.4f}s")
        print(f"  Training R²: {train_score:.4f}")
        print(f"  Test R²: {test_score:.4f}")
        
        # Feature importance analysis
        feature_importance = np.abs(ridge.coef_)
        top_features = np.argsort(feature_importance)[-10:]
        
        print(f"  Top 10 feature weights: {feature_importance[top_features]}")
        
        return X_sparse, y, ridge

def create_sparse_visualizations():
    """Create visualizations for sparse matrix concepts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sparsity pattern visualization
    ax = axes[0, 0]
    np.random.seed(42)
    n = 50
    density = 0.1
    
    sparse_matrix = sp.random(n, n, density=density, format='coo')
    
    ax.spy(sparse_matrix, markersize=2)
    ax.set_title(f'Sparse Matrix Pattern\n({n}×{n}, density={density:.1%})')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    
    # 2. Memory usage comparison
    ax = axes[0, 1]
    matrix_sizes = [100, 500, 1000, 2000, 5000]
    densities = [0.01, 0.05, 0.1]
    
    for density in densities:
        sparse_memory = []
        dense_memory = []
        
        for size in matrix_sizes:
            # Estimate memory usage
            nnz = int(size * size * density)
            
            # Sparse (CSR): data + indices + indptr
            sparse_mem = nnz * 8 + nnz * 4 + (size + 1) * 4  # bytes
            
            # Dense
            dense_mem = size * size * 8  # bytes
            
            sparse_memory.append(sparse_mem / 1024**2)  # MB
            dense_memory.append(dense_mem / 1024**2)  # MB
        
        ax.plot(matrix_sizes, sparse_memory, 'o-', label=f'Sparse (density={density:.0%})')
        
    # Dense line
    ax.plot(matrix_sizes, dense_memory, 's-', color='red', linewidth=2, label='Dense')
    
    ax.set_xlabel('Matrix Size (n×n)')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage: Sparse vs Dense')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Operation performance comparison
    ax = axes[1, 0]
    operations = ['Matrix-Vector', 'Matrix-Matrix', 'Transpose', 'Element Access']
    sparse_times = [0.002, 0.15, 0.001, 0.0001]  # Example times
    dense_times = [0.01, 0.8, 0.05, 0.00001]  # Example times
    
    x = np.arange(len(operations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sparse_times, width, label='Sparse', alpha=0.8)
    bars2 = ax.bar(x + width/2, dense_times, width, label='Dense', alpha=0.8)
    
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Operation Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(operations, rotation=45)
    ax.legend()
    ax.set_yscale('log')
    
    # 4. Sparsity in different domains
    ax = axes[1, 1]
    domains = ['Text (TF-IDF)', 'Social Networks', 'Recommender\nSystems', 'Scientific\nComputing']
    typical_sparsity = [0.99, 0.95, 0.999, 0.9]
    
    bars = ax.bar(domains, typical_sparsity, color=['skyblue', 'lightgreen', 'coral', 'plum'])
    ax.set_ylabel('Typical Sparsity')
    ax.set_title('Sparsity Across Domains')
    ax.set_ylim(0.8, 1.0)
    
    # Add percentage labels
    for bar, sparsity in zip(bars, typical_sparsity):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{sparsity:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sparse_matrix_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    analyzer = SparseMatrixAnalyzer()
    
    print("SPARSE MATRICES: REPRESENTATION AND USAGE")
    print("=" * 50)
    
    # 1. Storage format demonstration
    formats = analyzer.demonstrate_storage_formats()
    
    # 2. Performance benchmarking
    benchmark_results = analyzer.benchmark_operations(formats)
    
    # 3. Sparse algorithms
    sparse_demo_results = analyzer.demonstrate_sparse_algorithms()
    
    # 4. Real-world examples
    text_results = analyzer.text_analysis_example()
    graph_results = analyzer.graph_analysis_example()
    ml_results = analyzer.sparse_machine_learning_example()
    
    # 5. Visualizations
    create_sparse_visualizations()
    
    print(f"\n{'='*50}")
    print("SPARSE MATRIX SUMMARY")
    print(f"{'='*50}")
    
    print("\nKey Storage Formats:")
    print("• COO (Coordinate): Simple triplets, good for construction")
    print("• CSR (Compressed Sparse Row): Efficient row operations")
    print("• CSC (Compressed Sparse Column): Efficient column operations") 
    print("• DOK (Dictionary of Keys): Efficient incremental construction")
    print("• BSR (Block Sparse Row): Optimized for block-structured matrices")
    
    print("\nComputational Advantages:")
    print("• Memory: 10-1000x reduction in memory usage")
    print("• Speed: Faster operations when properly optimized")
    print("• Scalability: Enables processing of massive datasets")
    print("• Numerical: Better numerical stability in some algorithms")
    
    print("\nCommon Applications:")
    print("• Text processing: TF-IDF matrices, word embeddings")
    print("• Graph analysis: Adjacency matrices, Laplacian matrices")
    print("• Recommender systems: User-item interaction matrices")
    print("• Scientific computing: Finite element methods, PDEs")
    print("• Machine learning: Categorical features, high-dimensional data")
    
    print("\nBest Practices:")
    print("• Choose format based on primary operations needed")
    print("• Use specialized sparse algorithms when available")
    print("• Monitor fill-in during matrix factorizations")
    print("• Consider hybrid dense-sparse approaches for partially sparse data")
    print("• Profile memory and compute performance for your specific use case")
    
    print("\nLimitations:")
    print("• Matrix multiplication can destroy sparsity")
    print("• Some algorithms don't have efficient sparse versions")
    print("• Overhead of indirection can hurt performance on small matrices")
    print("• Format conversion costs can be significant")
```

**Advanced Sparse Matrix Techniques:**

**1. Sparse Matrix Factorizations:**
- **LU with pivoting**: Minimize fill-in through careful pivot selection
- **Cholesky**: Specialized for symmetric positive definite matrices
- **QR**: Orthogonal factorization preserving sparsity patterns
- **SVD**: Truncated SVD for large sparse matrices using iterative methods

**2. Preconditioning for Iterative Solvers:**
- **Incomplete factorizations**: ILU, IC preconditioners
- **Multigrid methods**: Hierarchical approaches for structured sparsity
- **Algebraic multigrid**: For unstructured sparse matrices

**3. Graph-Based Ordering:**
- **Cuthill-McKee**: Reduce bandwidth of sparse matrices
- **Nested dissection**: Minimize fill-in during factorization
- **Metis partitioning**: For parallel sparse computations

**Key Insights:**
- Choose storage format based on access patterns and operations
- Sparsity enables processing of problems otherwise intractable
- Many linear algebra algorithms have specialized sparse versions
- Memory access patterns are crucial for performance optimization

---

## Question 34

**Explain how tensor operations are vital in algorithms working with higher-dimensional data.**

**Answer:** Tensor operations are fundamental to modern machine learning and data science, providing the mathematical framework for efficiently processing higher-dimensional data. Tensors generalize scalars (0-dimensional), vectors (1-dimensional), and matrices (2-dimensional) to arbitrary dimensions, enabling representation and manipulation of complex data structures like images, videos, neural network parameters, and multidimensional time series.

**Mathematical Foundation:**

**Tensor Definition:**
A tensor is a multidimensional array with a consistent data type:
- **Rank/Order**: Number of dimensions (axes)
- **Shape**: Size along each dimension
- **Elements**: Accessed via multiple indices T[i₁, i₂, ..., iₙ]

**Tensor Hierarchy:**
```
Rank 0: Scalar                    → 5
Rank 1: Vector                    → [1, 2, 3]
Rank 2: Matrix                    → [[1, 2], [3, 4]]
Rank 3: 3D Tensor                 → [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
Rank n: n-dimensional tensor
```

**Key Tensor Operations:**

**1. Element-wise Operations:**
```
Addition:     C = A + B  (broadcast compatible)
Multiplication: C = A ⊙ B  (Hadamard product)
Functions:    C = f(A)   (apply f element-wise)
```

**2. Tensor Contraction (Generalized Matrix Multiplication):**
```
C[i,k] = Σⱼ A[i,j] × B[j,k]  (matrix multiplication)
C[i,k,m] = Σⱼ A[i,j,m] × B[j,k]  (tensor contraction)
```

**3. Tensor Reduction:**
```
Sum:     s = Σᵢ₁,ᵢ₂,...,ᵢₙ T[i₁, i₂, ..., iₙ]
Mean:    μ = (1/N) × Σ T
Max:     m = max(T)
```

**4. Tensor Reshape and Transpose:**
```
Reshape:   T(2,3,4) → T(6,4) or T(24,)
Transpose: T[i,j,k] → T[k,i,j] (permute axes)
```

**Comprehensive Implementation:**

```python
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from time import time
import pandas as pd

class TensorOperationsAnalyzer:
    """Comprehensive analysis of tensor operations in higher-dimensional data"""
    
    def __init__(self):
        """Initialize tensor operations analyzer"""
        self.timing_results = {}
        
    def demonstrate_tensor_basics(self):
        """Demonstrate basic tensor concepts and operations"""
        
        print("TENSOR FUNDAMENTALS")
        print("=" * 25)
        
        # Create tensors of different ranks
        print("Tensor Examples:")
        
        # Rank 0: Scalar
        scalar = np.array(42)
        print(f"Rank 0 (Scalar): {scalar}, shape: {scalar.shape}")
        
        # Rank 1: Vector
        vector = np.array([1, 2, 3, 4])
        print(f"Rank 1 (Vector): {vector}, shape: {vector.shape}")
        
        # Rank 2: Matrix
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        print(f"Rank 2 (Matrix): \n{matrix}, shape: {matrix.shape}")
        
        # Rank 3: 3D Tensor
        tensor_3d = np.random.randn(2, 3, 4)
        print(f"Rank 3 (3D Tensor): shape {tensor_3d.shape}")
        print(f"Sample slice [0,:,:]: \n{tensor_3d[0,:,:]}")
        
        # Rank 4: 4D Tensor (common in deep learning)
        tensor_4d = np.random.randn(2, 3, 4, 5)  # batch, height, width, channels
        print(f"Rank 4 (4D Tensor): shape {tensor_4d.shape}")
        
        # Basic operations
        print(f"\nBasic Tensor Operations:")
        
        # Element-wise operations
        A = np.random.randn(2, 3, 4)
        B = np.random.randn(2, 3, 4)
        
        C_add = A + B
        C_mult = A * B
        C_func = np.tanh(A)
        
        print(f"Element-wise addition: {A.shape} + {B.shape} = {C_add.shape}")
        print(f"Element-wise multiplication: {A.shape} * {B.shape} = {C_mult.shape}")
        print(f"Element-wise function: tanh({A.shape}) = {C_func.shape}")
        
        # Broadcasting
        vector_broadcast = np.array([1, 2, 4])  # shape (3,)
        C_broadcast = A + vector_broadcast  # Broadcasts to (2, 3, 4)
        print(f"Broadcasting: {A.shape} + {vector_broadcast.shape} = {C_broadcast.shape}")
        
        return A, B, tensor_3d, tensor_4d
    
    def demonstrate_tensor_contractions(self):
        """Demonstrate tensor contraction operations"""
        
        print(f"\nTENSOR CONTRACTIONS")
        print("=" * 25)
        
        # Matrix multiplication as tensor contraction
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 5)
        C_matmul = A @ B
        
        print(f"Matrix multiplication: ({A.shape}) @ ({B.shape}) = {C_matmul.shape}")
        
        # 3D tensor contractions
        T1 = np.random.randn(2, 3, 4)
        T2 = np.random.randn(4, 5)
        
        # Contract along last axis of T1 and first axis of T2
        T3 = np.tensordot(T1, T2, axes=([2], [0]))
        print(f"3D contraction: ({T1.shape}) contract ({T2.shape}) = {T3.shape}")
        
        # Multiple axis contraction
        T4 = np.random.randn(2, 3, 4, 5)
        T5 = np.random.randn(3, 4, 6)
        
        T6 = np.tensordot(T4, T5, axes=([1, 2], [0, 1]))
        print(f"Multi-axis contraction: ({T4.shape}) contract ({T5.shape}) = {T6.shape}")
        
        # Einstein summation notation
        print(f"\nEinstein Summation Examples:")
        
        # Matrix multiplication
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 5)
        C_einsum = np.einsum('ij,jk->ik', A, B)
        print(f"Matrix mult (einsum): 'ij,jk->ik' {A.shape}, {B.shape} = {C_einsum.shape}")
        
        # Batch matrix multiplication
        batch_A = np.random.randn(10, 3, 4)
        batch_B = np.random.randn(10, 4, 5)
        batch_C = np.einsum('bij,bjk->bik', batch_A, batch_B)
        print(f"Batch matmul: 'bij,bjk->bik' {batch_A.shape}, {batch_B.shape} = {batch_C.shape}")
        
        # Tensor trace
        T = np.random.randn(3, 4, 3)
        trace = np.einsum('iji->j', T)
        print(f"Tensor trace: 'iji->j' {T.shape} = {trace.shape}")
        
        return T1, T2, T3, batch_A, batch_B, batch_C
    
    def image_processing_example(self):
        """Demonstrate tensor operations in image processing"""
        
        print(f"\nIMAGE PROCESSING WITH TENSORS")
        print("=" * 35)
        
        # Create synthetic image batch
        batch_size = 32
        height, width = 64, 64
        channels = 3
        
        # Image tensor: (batch, height, width, channels)
        images = np.random.randint(0, 256, (batch_size, height, width, channels), dtype=np.uint8)
        
        print(f"Image batch tensor: {images.shape}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image dimensions: {height}×{width}")
        print(f"  Channels: {channels}")
        print(f"  Data type: {images.dtype}")
        print(f"  Memory usage: {images.nbytes / 1024**2:.1f} MB")
        
        # Convert to float for processing
        images_float = images.astype(np.float32) / 255.0
        
        # Tensor operations for image processing
        print(f"\nImage Processing Operations:")
        
        # 1. Channel-wise mean (global average pooling)
        channel_means = np.mean(images_float, axis=(1, 2))  # Average over H, W
        print(f"Channel means shape: {channel_means.shape}")
        
        # 2. Batch normalization (simplified)
        batch_mean = np.mean(images_float, axis=0, keepdims=True)  # Mean over batch
        batch_std = np.std(images_float, axis=0, keepdims=True)   # Std over batch
        normalized_images = (images_float - batch_mean) / (batch_std + 1e-8)
        
        print(f"Batch normalized images shape: {normalized_images.shape}")
        print(f"New mean (should be ~0): {np.mean(normalized_images):.6f}")
        print(f"New std (should be ~1): {np.std(normalized_images):.6f}")
        
        # 3. Convolution operation (simplified)
        def simple_conv2d(images, kernel):
            """Simplified 2D convolution using tensor operations"""
            
            batch, h, w, c_in = images.shape
            kh, kw, c_in_k, c_out = kernel.shape
            
            assert c_in == c_in_k, "Input channels must match"
            
            # Output dimensions
            out_h = h - kh + 1
            out_w = w - kw + 1
            
            # Initialize output
            output = np.zeros((batch, out_h, out_w, c_out))
            
            # Convolution loop (not optimized, for demonstration)
            for b in range(batch):
                for oh in range(out_h):
                    for ow in range(out_w):
                        # Extract patch
                        patch = images[b, oh:oh+kh, ow:ow+kw, :]  # (kh, kw, c_in)
                        
                        # Apply kernels using einsum
                        output[b, oh, ow, :] = np.einsum('hwi,hwio->o', patch, kernel)
            
            return output
        
        # Create random convolution kernels
        kernel_size = 3
        input_channels = 3
        output_channels = 16
        
        conv_kernel = np.random.randn(kernel_size, kernel_size, input_channels, output_channels) * 0.1
        
        print(f"\nConvolution example:")
        print(f"  Input: {images_float.shape}")
        print(f"  Kernel: {conv_kernel.shape}")
        
        # Apply convolution to small batch for demonstration
        start_time = time()
        conv_output = simple_conv2d(images_float[:4], conv_kernel)  # Only 4 images
        conv_time = time() - start_time
        
        print(f"  Output: {conv_output.shape}")
        print(f"  Computation time: {conv_time:.4f}s")
        
        return images_float, conv_kernel, conv_output
    
    def neural_network_tensors(self):
        """Demonstrate tensor operations in neural networks"""
        
        print(f"\nNEURAL NETWORK TENSOR OPERATIONS")
        print("=" * 40)
        
        # Simulate a simple neural network forward pass
        batch_size = 128
        input_dim = 784  # 28x28 flattened images
        hidden_dim = 256
        output_dim = 10
        
        # Input data
        X = np.random.randn(batch_size, input_dim)
        
        # Network parameters (weights and biases)
        W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        b1 = np.zeros((1, hidden_dim))
        W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        b2 = np.zeros((1, output_dim))
        
        print(f"Network architecture:")
        print(f"  Input: {X.shape}")
        print(f"  W1: {W1.shape}, b1: {b1.shape}")
        print(f"  W2: {W2.shape}, b2: {b2.shape}")
        
        # Forward pass using tensor operations
        def relu(x):
            return np.maximum(0, x)
        
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        print(f"\nForward pass:")
        
        # Layer 1: Linear transformation + activation
        start_time = time()
        Z1 = X @ W1 + b1  # Broadcasting adds bias
        A1 = relu(Z1)
        layer1_time = time() - start_time
        
        print(f"  Layer 1: {X.shape} @ {W1.shape} + {b1.shape} = {Z1.shape}")
        print(f"  After ReLU: {A1.shape}")
        print(f"  Computation time: {layer1_time:.4f}s")
        
        # Layer 2: Linear transformation + softmax
        start_time = time()
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)
        layer2_time = time() - start_time
        
        print(f"  Layer 2: {A1.shape} @ {W2.shape} + {b2.shape} = {Z2.shape}")
        print(f"  After Softmax: {A2.shape}")
        print(f"  Computation time: {layer2_time:.4f}s")
        
        # Demonstrate backpropagation (gradient computation)
        print(f"\nBackpropagation (gradient computation):")
        
        # Dummy target (one-hot encoded)
        y_true = np.eye(output_dim)[np.random.randint(0, output_dim, batch_size)]
        
        # Loss computation (cross-entropy)
        epsilon = 1e-15
        loss = -np.mean(np.sum(y_true * np.log(A2 + epsilon), axis=1))
        print(f"  Cross-entropy loss: {loss:.4f}")
        
        # Gradient computation using tensor operations
        dA2 = (A2 - y_true) / batch_size  # Gradient w.r.t. output
        
        # Gradients for layer 2
        dW2 = A1.T @ dA2
        db2 = np.sum(dA2, axis=0, keepdims=True)
        dA1 = dA2 @ W2.T
        
        # Gradients for layer 1 (through ReLU)
        dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        print(f"  Gradient shapes:")
        print(f"    dW2: {dW2.shape}, db2: {db2.shape}")
        print(f"    dW1: {dW1.shape}, db1: {db1.shape}")
        
        return X, A1, A2, dW1, dW2
    
    def tensor_decomposition_example(self):
        """Demonstrate tensor decomposition methods"""
        
        print(f"\nTENSOR DECOMPOSITION")
        print("=" * 25)
        
        # Create a 3D tensor with known low-rank structure
        np.random.seed(42)
        rank = 5
        I, J, K = 50, 40, 30
        
        # Generate rank-r tensor using CP decomposition structure
        A = np.random.randn(I, rank)
        B = np.random.randn(J, rank)
        C = np.random.randn(K, rank)
        
        # Construct tensor using outer products
        X = np.zeros((I, J, K))
        for r in range(rank):
            X += np.outer(A[:, r], np.outer(B[:, r], C[:, r])).reshape(I, J, K)
        
        # Add noise
        noise_level = 0.1
        X += noise_level * np.random.randn(I, J, K)
        
        print(f"Original tensor shape: {X.shape}")
        print(f"True rank: {rank}")
        print(f"Noise level: {noise_level}")
        
        # Tensor matricization (unfolding)
        print(f"\nTensor matricization:")
        
        # Mode-1 matricization
        X_1 = X.reshape(I, J*K)
        print(f"  Mode-1: {X.shape} → {X_1.shape}")
        
        # Mode-2 matricization
        X_2 = np.transpose(X, (1, 0, 2)).reshape(J, I*K)
        print(f"  Mode-2: {X.shape} → {X_2.shape}")
        
        # Mode-3 matricization
        X_3 = np.transpose(X, (2, 0, 1)).reshape(K, I*J)
        print(f"  Mode-3: {X.shape} → {X_3.shape}")
        
        # SVD of matricized tensors
        print(f"\nSVD analysis of matricizations:")
        
        U1, s1, V1 = np.linalg.svd(X_1, full_matrices=False)
        U2, s2, V2 = np.linalg.svd(X_2, full_matrices=False)
        U3, s3, V3 = np.linalg.svd(X_3, full_matrices=False)
        
        print(f"  Mode-1 singular values (top 10): {s1[:10]}")
        print(f"  Mode-2 singular values (top 10): {s2[:10]}")
        print(f"  Mode-3 singular values (top 10): {s3[:10]}")
        
        # Estimate rank from singular value decay
        def estimate_rank(singular_values, threshold=0.01):
            normalized_sv = singular_values / singular_values[0]
            return np.sum(normalized_sv > threshold)
        
        estimated_ranks = [
            estimate_rank(s1),
            estimate_rank(s2),
            estimate_rank(s3)
        ]
        
        print(f"  Estimated ranks: {estimated_ranks}")
        print(f"  True rank: {rank}")
        
        return X, X_1, X_2, X_3, s1, s2, s3
    
    def time_series_tensor_analysis(self):
        """Demonstrate tensor operations in time series analysis"""
        
        print(f"\nTIME SERIES TENSOR ANALYSIS")
        print("=" * 35)
        
        # Create multivariate time series data
        np.random.seed(42)
        n_series = 10  # Number of time series
        n_timepoints = 1000  # Length of each series
        n_features = 5  # Number of features per timepoint
        
        # Generate synthetic time series with seasonal patterns
        t = np.linspace(0, 4*np.pi, n_timepoints)
        
        time_series_tensor = np.zeros((n_series, n_timepoints, n_features))
        
        for s in range(n_series):
            for f in range(n_features):
                # Base trend
                trend = 0.1 * t
                
                # Seasonal component
                seasonal_freq = np.random.uniform(0.5, 2.0)
                seasonal = np.sin(seasonal_freq * t + np.random.uniform(0, 2*np.pi))
                
                # Noise
                noise = 0.2 * np.random.randn(n_timepoints)
                
                time_series_tensor[s, :, f] = trend + seasonal + noise
        
        print(f"Time series tensor shape: {time_series_tensor.shape}")
        print(f"  Number of series: {n_series}")
        print(f"  Time points: {n_timepoints}")
        print(f"  Features per timepoint: {n_features}")
        
        # Tensor operations for time series analysis
        print(f"\nTime series tensor operations:")
        
        # 1. Cross-series correlation
        series_correlations = np.zeros((n_series, n_series))
        for i in range(n_series):
            for j in range(n_series):
                # Compute correlation between multivariate series
                series_i = time_series_tensor[i, :, :].flatten()
                series_j = time_series_tensor[j, :, :].flatten()
                series_correlations[i, j] = np.corrcoef(series_i, series_j)[0, 1]
        
        print(f"  Cross-series correlation matrix: {series_correlations.shape}")
        print(f"  Average correlation: {np.mean(series_correlations[np.triu_indices(n_series, k=1)]):.3f}")
        
        # 2. Feature correlation across time
        feature_correlations = np.zeros((n_features, n_features))
        for f1 in range(n_features):
            for f2 in range(n_features):
                # Aggregate across series and time
                feat1_data = time_series_tensor[:, :, f1].flatten()
                feat2_data = time_series_tensor[:, :, f2].flatten()
                feature_correlations[f1, f2] = np.corrcoef(feat1_data, feat2_data)[0, 1]
        
        print(f"  Feature correlation matrix: {feature_correlations.shape}")
        
        # 3. Temporal patterns using tensor slicing
        # Compute moving averages using tensor operations
        window_size = 50
        moving_averages = np.zeros((n_series, n_timepoints - window_size + 1, n_features))
        
        for i in range(n_timepoints - window_size + 1):
            moving_averages[:, i, :] = np.mean(
                time_series_tensor[:, i:i+window_size, :], axis=1
            )
        
        print(f"  Moving averages tensor: {moving_averages.shape}")
        
        # 4. Principal tensor analysis (simplified PCA on flattened data)
        flattened_data = time_series_tensor.reshape(n_series, -1)  # (series, time*features)
        
        # Compute covariance and eigendecomposition
        cov_matrix = np.cov(flattened_data)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        print(f"  Principal component eigenvalues (top 5): {eigenvals[:5]}")
        print(f"  Explained variance ratio: {eigenvals[:3] / np.sum(eigenvals)}")
        
        return time_series_tensor, series_correlations, feature_correlations, moving_averages

def create_tensor_visualizations():
    """Create visualizations for tensor operations"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Tensor shapes and operations
    ax1 = plt.subplot(3, 3, 1)
    
    # Visualize tensor dimensions
    tensor_info = [
        ('Scalar', 0, [1]),
        ('Vector', 1, [4]),
        ('Matrix', 2, [3, 4]),
        ('3D Tensor', 3, [2, 3, 4]),
        ('4D Tensor', 4, [2, 3, 4, 5])
    ]
    
    names = [info[0] for info in tensor_info]
    ranks = [info[1] for info in tensor_info]
    
    bars = ax1.bar(names, ranks, color=['red', 'orange', 'yellow', 'green', 'blue'])
    ax1.set_ylabel('Tensor Rank')
    ax1.set_title('Tensor Hierarchy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add shape annotations
    for bar, info in zip(bars, tensor_info):
        height = bar.get_height()
        shape_str = '×'.join(map(str, info[2]))
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'({shape_str})', ha='center', va='bottom', fontsize=8)
    
    # 2. Memory usage comparison
    ax2 = plt.subplot(3, 3, 2)
    
    sizes = [10, 50, 100, 200]
    dimensions = [2, 3, 4, 5]
    
    memory_usage = []
    for dim in dimensions:
        memory_dim = []
        for size in sizes:
            total_elements = size ** dim
            memory_mb = total_elements * 8 / 1024**2  # 8 bytes per float64
            memory_dim.append(memory_mb)
        memory_usage.append(memory_dim)
    
    for i, dim in enumerate(dimensions):
        ax2.plot(sizes, memory_usage[i], 'o-', label=f'{dim}D Tensor')
    
    ax2.set_xlabel('Size per Dimension')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage vs Tensor Dimensions')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Operation complexity
    ax3 = plt.subplot(3, 3, 3)
    
    operations = ['Element-wise', 'Matrix Mult', 'Tensor Contract', 'SVD', 'Eigendecomp']
    complexities = ['O(n)', 'O(n³)', 'O(n⁴)', 'O(n³)', 'O(n³)']
    complexity_values = [1, 3, 4, 3, 3]  # Exponents for visualization
    
    bars = ax3.bar(operations, complexity_values, 
                   color=['green', 'orange', 'red', 'purple', 'brown'])
    ax3.set_ylabel('Complexity Exponent')
    ax3.set_title('Computational Complexity')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add complexity labels
    for bar, complexity in zip(bars, complexities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                complexity, ha='center', va='bottom', fontsize=9)
    
    # 4. 3D tensor visualization
    ax4 = plt.subplot(3, 3, 4, projection='3d')
    
    # Create a simple 3D tensor visualization
    x = np.arange(4)
    y = np.arange(3)
    z = np.arange(2)
    
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Flatten for scatter plot
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    
    # Color by sum of coordinates
    colors = X_flat + Y_flat + Z_flat
    
    scatter = ax4.scatter(X_flat, Y_flat, Z_flat, c=colors, cmap='viridis', s=100)
    ax4.set_xlabel('X (dim 0)')
    ax4.set_ylabel('Y (dim 1)')
    ax4.set_zlabel('Z (dim 2)')
    ax4.set_title('3D Tensor Structure\n(4×3×2)')
    
    # 5. Convolution operation visualization
    ax5 = plt.subplot(3, 3, 5)
    
    # Simulate convolution
    input_size = 8
    kernel_size = 3
    output_size = input_size - kernel_size + 1
    
    # Create input and kernel
    input_img = np.random.rand(input_size, input_size)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # Edge detection
    
    im = ax5.imshow(input_img, cmap='gray')
    ax5.set_title(f'Input Image ({input_size}×{input_size})')
    ax5.set_xticks([])
    ax5.set_yticks([])
    
    # 6. Kernel visualization
    ax6 = plt.subplot(3, 3, 6)
    im_kernel = ax6.imshow(kernel, cmap='RdBu_r')
    ax6.set_title(f'Convolution Kernel ({kernel_size}×{kernel_size})')
    plt.colorbar(im_kernel, ax=ax6)
    
    # Add kernel values as text
    for i in range(kernel_size):
        for j in range(kernel_size):
            ax6.text(j, i, f'{kernel[i, j]}', ha='center', va='center',
                    color='white' if abs(kernel[i, j]) > 4 else 'black')
    
    # 7. Time series tensor
    ax7 = plt.subplot(3, 3, 7)
    
    # Generate sample time series
    t = np.linspace(0, 4*np.pi, 100)
    n_series = 5
    
    for i in range(n_series):
        freq = 0.5 + 0.3 * i
        series = np.sin(freq * t) + 0.1 * np.random.randn(len(t))
        ax7.plot(t, series + i, label=f'Series {i+1}', alpha=0.8)
    
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Series (offset)')
    ax7.set_title('Multivariate Time Series\n(Tensor: series × time × features)')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Tensor decomposition illustration
    ax8 = plt.subplot(3, 3, 8)
    
    # Show CP decomposition concept
    rank = 3
    I, J, K = 6, 5, 4
    
    # Create factor matrices
    A_factor = np.random.rand(I, rank)
    B_factor = np.random.rand(J, rank)
    C_factor = np.random.rand(K, rank)
    
    # Show first factor
    im_factor = ax8.imshow(A_factor, cmap='viridis', aspect='auto')
    ax8.set_title(f'Factor Matrix A\n({I}×{rank})')
    ax8.set_xlabel('Rank')
    ax8.set_ylabel('Mode 1')
    plt.colorbar(im_factor, ax=ax8)
    
    # 9. Performance comparison
    ax9 = plt.subplot(3, 3, 9)
    
    # Simulated performance data
    frameworks = ['NumPy', 'TensorFlow', 'PyTorch', 'JAX']
    operations = ['Element-wise', 'MatMul', 'Convolution']
    
    # Performance data (higher is better)
    performance_data = np.array([
        [1.0, 2.5, 3.2, 3.8],  # Element-wise
        [1.0, 4.2, 4.0, 4.5],  # MatMul
        [0.5, 5.0, 4.8, 5.2]   # Convolution
    ])
    
    x = np.arange(len(frameworks))
    width = 0.25
    
    for i, op in enumerate(operations):
        ax9.bar(x + i*width, performance_data[i], width, 
               label=op, alpha=0.8)
    
    ax9.set_xlabel('Framework')
    ax9.set_ylabel('Relative Performance')
    ax9.set_title('Tensor Operations Performance')
    ax9.set_xticks(x + width)
    ax9.set_xticklabels(frameworks)
    ax9.legend()
    
    plt.tight_layout()
    plt.savefig('tensor_operations_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    analyzer = TensorOperationsAnalyzer()
    
    print("TENSOR OPERATIONS IN HIGHER-DIMENSIONAL DATA")
    print("=" * 55)
    
    # 1. Basic tensor concepts
    basic_results = analyzer.demonstrate_tensor_basics()
    
    # 2. Tensor contractions
    contraction_results = analyzer.demonstrate_tensor_contractions()
    
    # 3. Image processing example
    image_results = analyzer.image_processing_example()
    
    # 4. Neural network operations
    nn_results = analyzer.neural_network_tensors()
    
    # 5. Tensor decomposition
    decomposition_results = analyzer.tensor_decomposition_example()
    
    # 6. Time series analysis
    timeseries_results = analyzer.time_series_tensor_analysis()
    
    # 7. Visualizations
    create_tensor_visualizations()
    
    print(f"\n{'='*55}")
    print("TENSOR OPERATIONS SUMMARY")
    print(f"{'='*55}")
    
    print("\nKey Tensor Operations:")
    print("• Element-wise: Broadcasting, arithmetic, function application")
    print("• Contraction: Generalized matrix multiplication, inner products")
    print("• Reduction: Sum, mean, max along specified axes")
    print("• Reshaping: Tensor flattening, dimension permutation")
    print("• Slicing: Extracting subtensors, temporal windows")
    
    print("\nCritical Applications:")
    print("• Deep Learning: Forward/backward propagation, convolutions")
    print("• Computer Vision: Image/video processing, feature extraction")
    print("• NLP: Attention mechanisms, transformer architectures")
    print("• Time Series: Multivariate analysis, temporal patterns")
    print("• Scientific Computing: Multidimensional simulations")
    
    print("\nComputational Advantages:")
    print("• Vectorization: Parallel execution of operations")
    print("• Memory Efficiency: Optimal data layout and access patterns")
    print("• GPU Acceleration: Natural fit for parallel architectures")
    print("• Automatic Differentiation: Efficient gradient computation")
    print("• Optimized Libraries: BLAS, cuDNN, specialized kernels")
    
    print("\nFramework Capabilities:")
    print("• NumPy: CPU-based, fundamental tensor operations")
    print("• TensorFlow: Production ML, static/dynamic graphs")
    print("• PyTorch: Research-friendly, dynamic computation graphs")
    print("• JAX: Functional programming, JIT compilation, auto-diff")
    
    print("\nBest Practices:")
    print("• Choose appropriate tensor shapes for memory efficiency")
    print("• Use broadcasting to avoid explicit loops")
    print("• Profile memory usage and computational bottlenecks")
    print("• Leverage framework-specific optimizations")
    print("• Consider sparse representations for high-dimensional sparse data")
```

**Advanced Tensor Concepts:**

**1. Tensor Networks:**
- **Representation**: Efficient factorization of high-order tensors
- **Applications**: Quantum computing, compressed sensing
- **Examples**: Matrix Product States (MPS), Tensor Ring decomposition

**2. Automatic Differentiation:**
- **Forward mode**: Computes derivatives alongside function evaluation
- **Reverse mode**: Backpropagation for efficient gradient computation
- **Higher-order**: Computing Hessians and higher derivatives

**3. Tensor Compression:**
- **CP decomposition**: Sum of rank-1 tensors
- **Tucker decomposition**: Higher-order SVD
- **Tensor Train**: Chain of low-rank matrices

**4. Distributed Tensor Operations:**
- **Data parallelism**: Distribute samples across devices
- **Model parallelism**: Distribute tensor operations across devices
- **Pipeline parallelism**: Temporal distribution of computation

**Key Insights:**
- Tensors provide natural representation for multidimensional data
- Efficient tensor operations are crucial for scalable machine learning
- Modern frameworks optimize tensor computations for parallel hardware
- Understanding tensor operations is essential for deep learning architecture design

---

## Question 35

**What is the role of linear algebra in time series analysis?**

**Answer:** Linear algebra forms the mathematical backbone of modern time series analysis, providing the fundamental framework for representing, transforming, and analyzing temporal data. From basic autoregressive models to advanced state-space methods and machine learning approaches, linear algebra enables efficient computation, pattern recognition, and forecasting in time series. Understanding these linear algebraic foundations is essential for developing robust time series models and algorithms.

**Core Linear Algebra Concepts in Time Series:**

**1. Vector and Matrix Representation of Time Series:**

**Time Series as Vectors:**
```
x = [x₁, x₂, ..., xₜ]ᵀ ∈ ℝᵀ
```
Where each element represents a value at a specific time point.

**Multivariate Time Series as Matrices:**
```
X = [x₁, x₂, ..., xₜ] ∈ ℝⁿˣᵀ
```
Where n is the number of variables and T is the number of time points.

**Embedding Matrices (Hankel Matrices):**
```
H = [x₁   x₂   ...  xₜ₋ₘ₊₁]
    [x₂   x₃   ...  xₜ₋ₘ₊₂]
    [⋮    ⋮    ⋱   ⋮     ]
    [xₘ   xₘ₊₁ ...  xₜ    ]
```
Used in singular spectrum analysis and trajectory matrix methods.

**2. Linear Time Series Models:**

**Autoregressive (AR) Models:**
```
xₜ = φ₁xₜ₋₁ + φ₂xₜ₋₂ + ... + φₚxₜ₋ₚ + εₜ
```
Matrix form: **x** = **Φx**₋₁ + **ε** where Φ is the coefficient matrix.

**Vector Autoregression (VAR):**
```
Xₜ = Φ₁Xₜ₋₁ + Φ₂Xₜ₋₂ + ... + ΦₚXₜ₋ₚ + εₜ
```
Where Xₜ ∈ ℝⁿ is a vector of variables and Φᵢ ∈ ℝⁿˣⁿ are coefficient matrices.

**State-Space Models:**
```
State equation:     xₜ = Axₜ₋₁ + Buₜ + wₜ
Observation equation: yₜ = Cxₜ + Duₜ + vₜ
```

**Major Linear Algebra Applications:**

**1. Spectral Analysis and Frequency Domain:**

**Discrete Fourier Transform (DFT) as Matrix Multiplication:**
```
Y = Fx where F is the DFT matrix
F_jk = e^(-2πijk/N) / √N
```

**Power Spectral Density through Covariance:**
```
Γ(τ) = E[xₜxₜ₋τ] (autocovariance function)
S(ω) = Σ Γ(τ)e^(-iωτ) (spectral density)
```

**2. Singular Value Decomposition (SVD) in Time Series:**

**Singular Spectrum Analysis (SSA):**
- Decompose trajectory matrix: H = UΣVᵀ
- Extract trends, seasonality, and noise components
- Reconstruct time series from selected eigentriples

**Principal Component Analysis:**
- Identify dominant patterns in multivariate time series
- Dimensionality reduction for high-dimensional temporal data

**3. Kalman Filtering and State Estimation:**

**Prediction Step:**
```
x̂ₜ|ₜ₋₁ = Ax̂ₜ₋₁|ₜ₋₁ + Buₜ
Pₜ|ₜ₋₁ = APₜ₋₁|ₜ₋₁Aᵀ + Q
```

**Update Step:**
```
Kₜ = Pₜ|ₜ₋₁Cᵀ(CPₜ|ₜ₋₁Cᵀ + R)⁻¹
x̂ₜ|ₜ = x̂ₜ|ₜ₋₁ + Kₜ(yₜ - Cx̂ₜ|ₜ₋₁)
```

**Comprehensive Implementation:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import hankel, svd, solve, inv
from scipy.signal import periodogram, welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesLinearAlgebra:
    """Comprehensive linear algebra tools for time series analysis"""
    
    def __init__(self):
        """Initialize time series linear algebra analyzer"""
        self.models = {}
        self.results = {}
        
    def generate_synthetic_data(self, n_samples=1000, n_series=5):
        """Generate synthetic multivariate time series data"""
        
        np.random.seed(42)
        t = np.linspace(0, 4*np.pi, n_samples)
        
        # Create multivariate time series with different patterns
        ts_data = np.zeros((n_samples, n_series))
        
        for i in range(n_series):
            # Trend component
            trend = 0.02 * i * t
            
            # Seasonal components
            seasonal1 = 2 * np.sin(2 * t + i * np.pi/4)
            seasonal2 = 1.5 * np.cos(4 * t + i * np.pi/6)
            
            # AR(2) component
            ar_coeff = [0.7 - 0.1*i, 0.2 + 0.05*i]
            ar_noise = np.random.normal(0, 0.5, n_samples)
            ar_component = np.zeros(n_samples)
            
            for j in range(2, n_samples):
                ar_component[j] = (ar_coeff[0] * ar_component[j-1] + 
                                 ar_coeff[1] * ar_component[j-2] + 
                                 ar_noise[j])
            
            # Combine components
            ts_data[:, i] = trend + seasonal1 + seasonal2 + ar_component + np.random.normal(0, 0.2, n_samples)
        
        return ts_data, t
    
    def demonstrate_matrix_representations(self, ts_data):
        """Demonstrate various matrix representations of time series"""
        
        print("MATRIX REPRESENTATIONS OF TIME SERIES")
        print("=" * 45)
        
        n_samples, n_series = ts_data.shape
        print(f"Time series data shape: {ts_data.shape}")
        
        # 1. Basic matrix representation
        print(f"\n1. Basic Matrix Representation:")
        print(f"   X ∈ ℝ^{n_samples}×{n_series} (time × variables)")
        print(f"   Each column represents one time series")
        print(f"   Each row represents values at one time point")
        
        # 2. Lagged embedding (Hankel matrix)
        print(f"\n2. Lagged Embedding (Hankel Matrix):")
        embedding_dim = 20
        
        # Create Hankel matrix for first time series
        ts_single = ts_data[:, 0]
        hankel_matrix = hankel(ts_single[:embedding_dim], 
                              ts_single[embedding_dim-1:])
        
        print(f"   Hankel matrix shape: {hankel_matrix.shape}")
        print(f"   Embedding dimension: {embedding_dim}")
        print(f"   Used for SSA and nonlinear dynamics analysis")
        
        # 3. Difference matrix
        print(f"\n3. Difference Matrix:")
        diff_matrix = np.diff(ts_data, axis=0)
        print(f"   First differences shape: {diff_matrix.shape}")
        print(f"   Removes trends, focuses on changes")
        
        # 4. Covariance matrix
        print(f"\n4. Covariance Matrix:")
        cov_matrix = np.cov(ts_data.T)
        print(f"   Covariance matrix shape: {cov_matrix.shape}")
        print(f"   Captures linear relationships between series")
        
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        print(f"   Eigenvalues: {eigenvals}")
        print(f"   Condition number: {np.max(eigenvals)/np.min(eigenvals):.2f}")
        
        return {
            'hankel_matrix': hankel_matrix,
            'diff_matrix': diff_matrix,
            'cov_matrix': cov_matrix,
            'eigenvals': eigenvals,
            'eigenvecs': eigenvecs
        }
    
    def singular_value_decomposition_analysis(self, ts_data):
        """Demonstrate SVD applications in time series analysis"""
        
        print("\n\nSVD IN TIME SERIES ANALYSIS")
        print("=" * 35)
        
        # Center the data
        ts_centered = ts_data - np.mean(ts_data, axis=0)
        
        # Perform SVD
        U, s, Vt = svd(ts_centered, full_matrices=False)
        
        print(f"SVD decomposition:")
        print(f"  U (left singular vectors): {U.shape}")
        print(f"  s (singular values): {s.shape}")
        print(f"  Vt (right singular vectors): {Vt.shape}")
        
        # Analyze singular values
        cumsum_variance = np.cumsum(s**2) / np.sum(s**2)
        
        print(f"\nVariance explained:")
        for i in range(min(5, len(s))):
            print(f"  Component {i+1}: {s[i]**2/np.sum(s**2):.3f} ({cumsum_variance[i]:.3f} cumulative)")
        
        # Reconstruct with first k components
        k = 3
        ts_reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        reconstruction_error = np.linalg.norm(ts_centered - ts_reconstructed, 'fro')
        print(f"\nReconstruction with {k} components:")
        print(f"  Reconstruction error: {reconstruction_error:.4f}")
        print(f"  Variance preserved: {cumsum_variance[k-1]:.3f}")
        
        return {
            'U': U,
            's': s,
            'Vt': Vt,
            'reconstructed': ts_reconstructed,
            'cumsum_variance': cumsum_variance
        }
    
    def vector_autoregression_analysis(self, ts_data):
        """Demonstrate Vector Autoregression using linear algebra"""
        
        print("\n\nVECTOR AUTOREGRESSION (VAR) MODEL")
        print("=" * 40)
        
        # Prepare data for VAR
        p = 3  # lag order
        n_samples, n_series = ts_data.shape
        
        # Create lagged data matrix
        Y = ts_data[p:, :]  # dependent variables
        X = np.ones((n_samples - p, 1))  # intercept
        
        # Add lagged variables
        for lag in range(1, p + 1):
            X = np.hstack([X, ts_data[p-lag:-lag, :]])
        
        print(f"VAR({p}) model setup:")
        print(f"  Y (dependent): {Y.shape}")
        print(f"  X (predictors): {X.shape}")
        print(f"  Number of parameters per equation: {X.shape[1]}")
        
        # Estimate coefficients using OLS: B = (X'X)^(-1)X'Y
        XtX = X.T @ X
        XtY = X.T @ Y
        
        # Check for numerical stability
        cond_number = np.linalg.cond(XtX)
        print(f"  Design matrix condition number: {cond_number:.2e}")
        
        if cond_number > 1e12:
            print("  Warning: Ill-conditioned design matrix, using regularization")
            # Add ridge regularization
            ridge_lambda = 0.01
            B = solve(XtX + ridge_lambda * np.eye(XtX.shape[0]), XtY)
        else:
            B = solve(XtX, XtY)
        
        print(f"  Coefficient matrix B: {B.shape}")
        
        # Make predictions
        Y_pred = X @ B
        
        # Calculate residuals and statistics
        residuals = Y - Y_pred
        mse = np.mean(residuals**2, axis=0)
        r_squared = 1 - np.var(residuals, axis=0) / np.var(Y, axis=0)
        
        print(f"\nModel performance:")
        for i in range(n_series):
            print(f"  Series {i+1}: R² = {r_squared[i]:.3f}, MSE = {mse[i]:.4f}")
        
        # Granger causality test (simplified)
        print(f"\nGranger causality analysis:")
        print("  Testing if series i Granger-causes series j")
        
        causality_matrix = np.zeros((n_series, n_series))
        
        for i in range(n_series):
            for j in range(n_series):
                if i != j:
                    # Restricted model (without series i)
                    X_restricted = np.ones((n_samples - p, 1))
                    for lag in range(1, p + 1):
                        X_restricted = np.hstack([X_restricted, 
                                                np.delete(ts_data[p-lag:-lag, :], i, axis=1)])
                    
                    # Full model residuals for series j
                    res_full = residuals[:, j]
                    
                    # Restricted model estimation
                    try:
                        B_restricted = solve(X_restricted.T @ X_restricted, X_restricted.T @ Y[:, j])
                        Y_pred_restricted = X_restricted @ B_restricted
                        res_restricted = Y[:, j] - Y_pred_restricted
                        
                        # F-statistic for Granger causality
                        rss_restricted = np.sum(res_restricted**2)
                        rss_full = np.sum(res_full**2)
                        
                        f_stat = ((rss_restricted - rss_full) / p) / (rss_full / (len(res_full) - X.shape[1]))
                        causality_matrix[i, j] = f_stat
                        
                    except np.linalg.LinAlgError:
                        causality_matrix[i, j] = 0
        
        print(f"  Causality F-statistics matrix:")
        for i in range(n_series):
            print(f"    {causality_matrix[i, :]}")
        
        return {
            'B': B,
            'Y_pred': Y_pred,
            'residuals': residuals,
            'r_squared': r_squared,
            'causality_matrix': causality_matrix
        }
    
    def kalman_filter_example(self, ts_data):
        """Demonstrate Kalman filtering for time series"""
        
        print("\n\nKALMAN FILTERING")
        print("=" * 20)
        
        # Use first time series for univariate Kalman filter
        y = ts_data[:, 0]
        n = len(y)
        
        # State-space model: local level model
        # State equation: x_t = x_{t-1} + w_t, w_t ~ N(0, Q)
        # Observation equation: y_t = x_t + v_t, v_t ~ N(0, R)
        
        # Model parameters
        Q = 0.1  # process noise variance
        R = 1.0  # observation noise variance
        
        # Initialize
        x_pred = np.zeros(n)  # predicted state
        x_filt = np.zeros(n)  # filtered state
        P_pred = np.zeros(n)  # predicted covariance
        P_filt = np.zeros(n)  # filtered covariance
        
        # Initial conditions
        x_filt[0] = y[0]
        P_filt[0] = 1.0
        
        print(f"Local level model:")
        print(f"  Process noise variance (Q): {Q}")
        print(f"  Observation noise variance (R): {R}")
        print(f"  Time series length: {n}")
        
        # Kalman filter recursions
        for t in range(1, n):
            # Prediction step
            x_pred[t] = x_filt[t-1]  # A = 1 for random walk
            P_pred[t] = P_filt[t-1] + Q
            
            # Update step
            K = P_pred[t] / (P_pred[t] + R)  # Kalman gain
            x_filt[t] = x_pred[t] + K * (y[t] - x_pred[t])
            P_filt[t] = (1 - K) * P_pred[t]
        
        # Calculate innovation sequence
        innovations = y[1:] - x_pred[1:]
        
        print(f"\nKalman filter results:")
        print(f"  Mean absolute prediction error: {np.mean(np.abs(innovations)):.4f}")
        print(f"  Innovation variance: {np.var(innovations):.4f}")
        print(f"  Final state estimate: {x_filt[-1]:.4f}")
        print(f"  Final covariance: {P_filt[-1]:.4f}")
        
        return {
            'x_filt': x_filt,
            'x_pred': x_pred,
            'P_filt': P_filt,
            'innovations': innovations
        }
    
    def spectral_analysis(self, ts_data, t):
        """Demonstrate spectral analysis using linear algebra"""
        
        print("\n\nSPECTRAL ANALYSIS")
        print("=" * 20)
        
        # Use first time series
        x = ts_data[:, 0]
        n = len(x)
        
        # 1. Discrete Fourier Transform
        print(f"Discrete Fourier Transform:")
        
        # Manual DFT matrix construction (for educational purposes)
        # For efficiency, use FFT in practice
        frequencies = np.fft.fftfreq(n, d=t[1]-t[0])
        X_fft = np.fft.fft(x)
        
        # Power spectral density
        psd = np.abs(X_fft)**2 / n
        
        print(f"  Time series length: {n}")
        print(f"  Frequency resolution: {frequencies[1] - frequencies[0]:.4f}")
        print(f"  Nyquist frequency: {np.max(frequencies):.4f}")
        
        # Find dominant frequencies
        dominant_indices = np.argsort(psd[:n//2])[-5:]  # Top 5 frequencies
        dominant_freqs = frequencies[dominant_indices]
        dominant_powers = psd[dominant_indices]
        
        print(f"  Dominant frequencies:")
        for i, (freq, power) in enumerate(zip(dominant_freqs, dominant_powers)):
            print(f"    {i+1}: f = {freq:.4f}, power = {power:.2f}")
        
        # 2. Autocovariance function
        print(f"\nAutocovariance Analysis:")
        
        # Compute autocovariance using convolution
        x_centered = x - np.mean(x)
        autocov = np.correlate(x_centered, x_centered, mode='full')
        autocov = autocov[autocov.size // 2:] / n
        
        # Theoretical spectral density via autocorrelation
        max_lag = min(50, len(autocov) - 1)
        lags = np.arange(max_lag + 1)
        
        print(f"  Autocovariance at lag 0: {autocov[0]:.4f}")
        print(f"  Autocovariance at lag 1: {autocov[1]:.4f}")
        print(f"  Autocovariance at lag 5: {autocov[5]:.4f}")
        
        return {
            'frequencies': frequencies,
            'psd': psd,
            'dominant_freqs': dominant_freqs,
            'autocov': autocov[:max_lag+1],
            'lags': lags
        }

def create_comprehensive_visualizations(analyzer, ts_data, t, results):
    """Create comprehensive visualizations for time series linear algebra"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. Original time series
    ax = axes[0, 0]
    for i in range(ts_data.shape[1]):
        ax.plot(t, ts_data[:, i], label=f'Series {i+1}', alpha=0.8)
    ax.set_title('Original Multivariate Time Series')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. SVD decomposition
    ax = axes[0, 1]
    svd_results = results['svd']
    ax.plot(svd_results['s']**2 / np.sum(svd_results['s']**2), 'o-')
    ax.set_title('SVD: Explained Variance by Component')
    ax.set_xlabel('Component')
    ax.set_ylabel('Variance Explained')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. SVD reconstruction
    ax = axes[0, 2]
    ax.plot(t, ts_data[:, 0], label='Original', alpha=0.7)
    ax.plot(t, svd_results['reconstructed'][:, 0] + np.mean(ts_data[:, 0]), 
            label='Reconstructed (3 components)', linestyle='--')
    ax.set_title('SVD Reconstruction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Covariance matrix
    ax = axes[1, 0]
    matrix_results = results['matrix']
    im = ax.imshow(matrix_results['cov_matrix'], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Cross-Series Covariance Matrix')
    ax.set_xlabel('Series')
    ax.set_ylabel('Series')
    plt.colorbar(im, ax=ax)
    
    # 5. VAR residuals
    ax = axes[1, 1]
    var_results = results['var']
    residuals = var_results['residuals']
    ax.plot(residuals[:, 0], label='Series 1', alpha=0.7)
    ax.plot(residuals[:, 1], label='Series 2', alpha=0.7)
    ax.set_title('VAR Model Residuals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Granger causality matrix
    ax = axes[1, 2]
    causality = var_results['causality_matrix']
    im = ax.imshow(causality, cmap='Reds')
    ax.set_title('Granger Causality F-Statistics')
    ax.set_xlabel('Caused Series')
    ax.set_ylabel('Causing Series')
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(causality.shape[0]):
        for j in range(causality.shape[1]):
            if i != j:
                ax.text(j, i, f'{causality[i, j]:.1f}', 
                       ha='center', va='center', color='white' if causality[i, j] > np.max(causality)/2 else 'black')
    
    # 7. Kalman filter results
    ax = axes[2, 0]
    kalman_results = results['kalman']
    ax.plot(t, ts_data[:, 0], label='Observed', alpha=0.7)
    ax.plot(t, kalman_results['x_filt'], label='Filtered', linestyle='--')
    ax.fill_between(t, 
                    kalman_results['x_filt'] - 2*np.sqrt(kalman_results['P_filt']),
                    kalman_results['x_filt'] + 2*np.sqrt(kalman_results['P_filt']),
                    alpha=0.3, label='95% Confidence')
    ax.set_title('Kalman Filter State Estimation')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Power spectral density
    ax = axes[2, 1]
    spectral_results = results['spectral']
    freqs = spectral_results['frequencies']
    psd = spectral_results['psd']
    
    # Plot only positive frequencies
    n_half = len(freqs) // 2
    ax.loglog(freqs[1:n_half], psd[1:n_half])
    ax.set_title('Power Spectral Density')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.grid(True, alpha=0.3)
    
    # 9. Autocovariance function
    ax = axes[2, 2]
    autocov = spectral_results['autocov']
    lags = spectral_results['lags']
    ax.plot(lags, autocov, 'o-')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Autocovariance Function')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocovariance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_linear_algebra.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    analyzer = TimeSeriesLinearAlgebra()
    
    print("LINEAR ALGEBRA IN TIME SERIES ANALYSIS")
    print("=" * 50)
    
    # Generate synthetic data
    ts_data, t = analyzer.generate_synthetic_data()
    
    # Demonstrate various linear algebra applications
    matrix_results = analyzer.demonstrate_matrix_representations(ts_data)
    svd_results = analyzer.singular_value_decomposition_analysis(ts_data)
    var_results = analyzer.vector_autoregression_analysis(ts_data)
    kalman_results = analyzer.kalman_filter_example(ts_data)
    spectral_results = analyzer.spectral_analysis(ts_data, t)
    
    # Collect all results
    results = {
        'matrix': matrix_results,
        'svd': svd_results,
        'var': var_results,
        'kalman': kalman_results,
        'spectral': spectral_results
    }
    
    # Create visualizations
    create_comprehensive_visualizations(analyzer, ts_data, t, results)
    
    print(f"\n{'='*50}")
    print("LINEAR ALGEBRA IN TIME SERIES SUMMARY")
    print(f"{'='*50}")
    
    print("\nCore Applications:")
    print("• Matrix Representation: Organizing temporal data efficiently")
    print("• SVD/PCA: Dimensionality reduction and noise filtering")
    print("• State-Space Models: Kalman filtering for optimal estimation")
    print("• Vector Autoregression: Modeling multivariate dependencies")
    print("• Spectral Analysis: Frequency domain analysis via DFT")
    print("• Cointegration: Long-run equilibrium relationships")
    
    print("\nKey Mathematical Tools:")
    print("• Eigendecomposition: Stability analysis, spectral methods")
    print("• Matrix factorizations: LU, QR, Cholesky for efficient computation")
    print("• Least squares: Parameter estimation in linear models")
    print("• Hankel matrices: Embedding for nonlinear dynamics")
    print("• Toeplitz matrices: Stationary process representations")
    
    print("\nComputational Benefits:")
    print("• Vectorization: Efficient batch processing of time series")
    print("• Parallel computation: Matrix operations leverage BLAS libraries")
    print("• Numerical stability: Well-conditioned formulations")
    print("• Memory efficiency: Sparse representations when applicable")
    print("• Scalability: Algorithms scale with modern linear algebra libraries")
    
    print("\nAdvanced Techniques:")
    print("• Dynamic Factor Models: Common trends in high-dimensional data")
    print("• Regime-Switching Models: Time-varying parameters via filtering")
    print("• Long Memory Models: Fractional integration and cointegration")
    print("• Multiresolution Analysis: Wavelets and filter banks")
    print("• Tensor Methods: Higher-order patterns in panel data")
    
    print("\nPractical Considerations:")
    print("• Choose model order using information criteria")
    print("• Check stationarity and cointegration properties")
    print("• Validate models using out-of-sample forecasting")
    print("• Handle missing data through EM algorithm or Kalman smoothing")
    print("• Regularization for high-dimensional time series")

```

**Advanced Linear Algebra Techniques in Time Series:**

**1. Cointegration Analysis:**
Uses eigendecomposition to find long-run equilibrium relationships:
```
Johansen test: Π = αβ' where β contains cointegrating vectors
```

**2. Dynamic Factor Models:**
Extract common factors using principal components:
```
xₜ = Λfₜ + εₜ where fₜ are latent factors
```

**3. Wavelets and Multiresolution:**
Matrix-based filter banks for time-frequency analysis.

**4. Tensor Methods:**
Higher-order decompositions for panel time series data.

**Key Insights:**
- Linear algebra provides the computational foundation for time series analysis
- Matrix decompositions reveal hidden structures and patterns
- Efficient algorithms enable real-time processing of high-dimensional temporal data
- Understanding the mathematical foundations leads to better model selection and interpretation

---

