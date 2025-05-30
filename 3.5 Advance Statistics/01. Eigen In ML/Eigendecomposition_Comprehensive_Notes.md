# Eigendecomposition in Machine Learning: Comprehensive Notes

## 1. Core Concepts

### 1.1 Introduction to Eigendecomposition

Eigendecomposition is a fundamental concept in linear algebra that plays a crucial role in many machine learning algorithms. At its core, eigendecomposition breaks down a square matrix into its constituent parts, revealing the underlying structure and behavior of the linear transformation represented by the matrix.

The eigendecomposition equation is:

**A = PDP⁻¹**

Where:
- A is the original matrix
- P is a matrix whose columns are the eigenvectors of A
- D is a diagonal matrix containing the eigenvalues of A
- P⁻¹ is the inverse of P

### 1.2 Eigenvectors and Eigenvalues

**Definition**: If A is a square matrix and v is a non-zero vector such that Av = λv for some scalar λ, then v is an eigenvector of A with eigenvalue λ.

**Intuitive understanding**: 
- Eigenvectors are special vectors that, when transformed by a matrix, only change in scale (stretched or compressed), not in direction.
- Eigenvalues are the scaling factors by which eigenvectors are stretched or compressed during the transformation.

```
          ↗
         /|
        / |
       /  |
→     /   ↑
      |  /
      | /
      |/
      →
```

In a 2D transformation represented by matrix A = [3 1; 0 2]:
1. The vector (1,0) is an eigenvector with eigenvalue 3
2. The vector (-1,1) is an eigenvector with eigenvalue 2

### 1.3 Computing Eigenvectors and Eigenvalues

To find eigenvectors and eigenvalues:

1. **Set up the characteristic equation**: 
   - Start with Av = λv
   - Rearrange to (A - λI)v = 0
   - For non-trivial solutions, det(A - λI) = 0

2. **Find eigenvalues**:
   - Calculate det(A - λI) = 0
   - Solve the resulting characteristic polynomial for λ

3. **Find eigenvectors**:
   - For each eigenvalue λ, solve (A - λI)v = 0
   - Find the null space of (A - λI)

### 1.4 Properties of Eigenvalues and Eigenvectors

1. **Eigenvalue types**:
   - Positive eigenvalues indicate stretching
   - Negative eigenvalues indicate flipping and stretching/compressing
   - Zero eigenvalues indicate collapsing dimensions
   - Complex eigenvalues indicate rotation and scaling

2. **Eigenbasis**:
   - If a matrix has n linearly independent eigenvectors, they form an eigenbasis
   - Using an eigenbasis, the matrix can be diagonalized: P⁻¹AP = D
   - Diagonal matrices simplify many operations (multiplication, powers, etc.)

3. **Trace and determinant properties**:
   - Sum of eigenvalues = trace of matrix
   - Product of eigenvalues = determinant of matrix

## 2. Applications

### 2.1 Principal Component Analysis (PCA)

PCA is one of the most common applications of eigendecomposition in machine learning:

1. **Process**:
   - Calculate the covariance matrix of centered data
   - Compute eigenvectors and eigenvalues of the covariance matrix
   - Eigenvectors represent principal components (directions of maximum variance)
   - Eigenvalues indicate the amount of variance along each principal component
   - Sort eigenvectors by decreasing eigenvalues
   - Project data onto top k eigenvectors to reduce dimensionality

2. **Benefits**:
   - Reduces dimensionality while preserving maximum variance
   - Uncorrelated features in the new space
   - Helps visualize high-dimensional data
   - Addresses the curse of dimensionality
   - Speeds up subsequent machine learning algorithms

```
          Original data                 After PCA
          o   o                            |
         o o o  o                          |
        o  o  o   o                        |
       o   o   o                           |    o o o o o o
      o    o    o                          |  o o o o o o o
     o     o     o          →              |    o o o o o
    o      o      o                        |
   o       o       o                       |
                                         --+--
                                           |
```

### 2.2 Spectral Clustering

Spectral clustering uses eigendecomposition for effective clustering in complex data:

1. **Process**:
   - Create a similarity graph between data points
   - Compute the Laplacian matrix of the graph
   - Find eigenvectors corresponding to the smallest eigenvalues of the Laplacian
   - Use these eigenvectors as features for clustering algorithms like K-means

2. **Advantages**:
   - Can identify non-convex clusters
   - Works well when clusters have complicated shapes
   - Number of near-zero eigenvalues indicates the number of clusters

### 2.3 Facial Recognition and Image Processing

Eigendecomposition enables efficient image analysis and face recognition:

1. **Eigenfaces**:
   - Images are represented as vectors
   - The covariance matrix of these vectors is computed
   - Eigenvectors of this matrix (eigenfaces) form a basis to represent faces
   - New faces are classified by projecting onto this eigenface space

2. **Image compression**:
   - Using eigendecomposition to identify important dimensions
   - Discarding components with small eigenvalues reduces storage needs

### 2.4 Recommender Systems

Eigendecomposition helps uncover hidden patterns in user-item interactions:

1. **Matrix factorization**:
   - The user-item interaction matrix is decomposed
   - Latent factors (related to eigenvectors) represent underlying preferences
   - Used in collaborative filtering algorithms

### 2.5 Natural Language Processing

Eigendecomposition supports semantic understanding of text:

1. **Latent Semantic Analysis (LSA)**:
   - Create a term-document matrix
   - Apply Singular Value Decomposition (related to eigendecomposition)
   - The resulting dimensions represent "topics" or semantic concepts

### 2.6 Network Analysis

Eigendecomposition helps identify important nodes in networks:

1. **Eigenvector centrality**:
   - Each node's importance is proportional to the importance of connected nodes
   - The principal eigenvector of the adjacency matrix gives centrality scores
   - Used in algorithms like PageRank for web search

### 2.7 Dynamic Systems and Stability Analysis

Eigendecomposition helps analyze stability in machine learning models:

1. **System behavior**:
   - Eigenvalues determine whether a system converges, diverges, or oscillates
   - Negative eigenvalues indicate stability
   - Positive eigenvalues indicate instability

## 3. Advanced Topics

### 3.1 Singular Value Decomposition (SVD)

SVD is a generalization of eigendecomposition that works for any matrix, not just square matrices.

1. **Definition**:
   For any m×n matrix A, the SVD is:
   
   **A = UΣV^T**
   
   Where:
   - U is an m×m orthogonal matrix whose columns are the left singular vectors
   - Σ is an m×n diagonal matrix containing the singular values
   - V^T is the transpose of an n×n orthogonal matrix V whose columns are the right singular vectors

2. **Relationship with eigendecomposition**:
   - The singular values of A are the square roots of the eigenvalues of A^TA (or AA^T)
   - The right singular vectors are the eigenvectors of A^TA
   - The left singular vectors are the eigenvectors of AA^T

3. **Types of SVD**:
   - **Full SVD**: U and V are full square matrices
   - **Reduced/Thin SVD**: Only keeps the columns in U and V corresponding to non-zero singular values
   - **Truncated SVD**: Keeps only the k largest singular values and their corresponding vectors

4. **Relationship between SVD and PCA**:
   - PCA can be implemented using SVD
   - When the data is centered, the right singular vectors from SVD are the same as the eigenvectors from PCA
   - The singular values are related to the eigenvalues by √λ

### 3.2 Eigenvalue Decomposition vs. SVD

| Feature | Eigenvalue Decomposition | Singular Value Decomposition |
|---------|-------------------------|------------------------------|
| Applicable matrices | Square matrices only | Any matrix (rectangular or square) |
| Decomposition | A = PDP⁻¹ | A = UΣV^T |
| Components | Eigenvectors and eigenvalues | Left/right singular vectors and singular values |
| Orthogonality | Eigenvectors may not be orthogonal | U and V are orthogonal matrices |
| Complex values | May have complex eigenvalues | Singular values are always real and non-negative |

### 3.3 Power Iteration Method

Power iteration is an efficient algorithm for computing the dominant eigenvector and eigenvalue:

1. **Algorithm**:
   - Start with a random vector v₀
   - Repeatedly compute vₖ₊₁ = Avₖ/||Avₖ|| until convergence
   - The resulting vector approximates the eigenvector with largest eigenvalue
   - The Rayleigh quotient v^TAv/v^Tv approximates the largest eigenvalue

2. **Shifted power iteration**:
   - To find other eigenvalues, we can use shifts: (A - σI)
   - This allows finding eigenvalues near a specific value σ

### 3.4 QR Algorithm

The QR algorithm is a method for computing all eigenvalues of a matrix:

1. **Algorithm**:
   - Start with A₀ = A
   - For each iteration k:
     - Factorize Aₖ = QₖRₖ (QR decomposition)
     - Compute Aₖ₊₁ = RₖQₖ
   - As k increases, Aₖ converges to a triangular form with eigenvalues on diagonal

2. **Variants**:
   - Shifted QR algorithm for faster convergence
   - Implicit QR with Householder transformations for numerical stability

## 4. Implementation Focus

### 4.1 Eigendecomposition with NumPy

Basic eigendecomposition in Python:

```python
import numpy as np

# Create a matrix
A = np.array([[3, 1], [0, 2]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# Verify Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_v = eigenvalues[i]
    
    print(f"A·v{i} =", np.dot(A, v))
    print(f"λ{i}·v{i} =", lambda_v * v)
```

### 4.2 Implementing PCA from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

def pca_from_scratch(X, n_components=2):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    
    # Project data onto principal components
    pca_result = np.dot(X_centered, eigenvectors)
    
    return pca_result, eigenvectors, eigenvalues

# Example usage with iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca_result, eigenvectors, eigenvalues = pca_from_scratch(X, n_components=2)

# Plot results
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
target_names = iris.target_names

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(pca_result[y == i, 0], pca_result[y == i, 1], color=color, alpha=0.8, lw=2, label=target_name)

plt.title('PCA of Iris dataset')
plt.xlabel(f'Principal Component 1 (Explained variance: {eigenvalues[0]:.2f})')
plt.ylabel(f'Principal Component 2 (Explained variance: {eigenvalues[1]:.2f})')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

### 4.3 SVD Implementation and Application

```python
import numpy as np
import matplotlib.pyplot as plt

# SVD for dimensionality reduction
def truncated_svd(X, n_components=2):
    # Compute SVD
    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Keep only top n_components
    U_reduced = U[:, :n_components]
    sigma_reduced = sigma[:n_components]
    Vt_reduced = Vt[:n_components, :]
    
    # Project data
    X_reduced = np.dot(U_reduced, np.diag(sigma_reduced))
    
    # Calculate explained variance
    explained_variance = (sigma_reduced ** 2) / (len(X) - 1)
    total_var = (sigma ** 2).sum() / (len(X) - 1)
    explained_variance_ratio = explained_variance / total_var.sum()
    
    return X_reduced, explained_variance_ratio

# Example usage with iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Center the data
X_centered = X - np.mean(X, axis=0)

# Apply truncated SVD
svd_result, explained_variance_ratio = truncated_svd(X_centered, n_components=2)

# Plot results
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
target_names = iris.target_names

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(svd_result[y == i, 0], svd_result[y == i, 1], color=color, alpha=0.8, lw=2, label=target_name)

plt.title('SVD of Iris dataset')
plt.xlabel(f'Component 1 (Explained variance: {explained_variance_ratio[0]:.2f})')
plt.ylabel(f'Component 2 (Explained variance: {explained_variance_ratio[1]:.2f})')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

### 4.4 Image Compression with SVD

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Load sample image
china = load_sample_image("china.jpg")
china_gray = np.mean(china, axis=2).astype(np.float64)  # Convert to grayscale

# Perform SVD
U, sigma, Vt = np.linalg.svd(china_gray, full_matrices=False)

# Function to reconstruct image with k singular values
def reconstruct_image(U, sigma, Vt, k):
    return np.dot(U[:, :k], np.dot(np.diag(sigma[:k]), Vt[:k, :]))

# Reconstruct with different numbers of components
components = [5, 20, 50, 200]
reconstructed_images = [reconstruct_image(U, sigma, Vt, k) for k in components]

# Calculate compression ratio and error
def compute_metrics(original, compressed, k, total_singular_values):
    # Compression ratio
    original_size = original.shape[0] * original.shape[1]
    compressed_size = k * (original.shape[0] + original.shape[1] + 1)
    compression_ratio = original_size / compressed_size
    
    # Reconstruction error
    error = np.linalg.norm(original - compressed, 'fro') / np.linalg.norm(original, 'fro')
    
    # Explained variance
    explained_variance = np.sum(sigma[:k]**2) / np.sum(sigma**2)
    
    return compression_ratio, error, explained_variance

# Plot original and reconstructed images
plt.figure(figsize=(15, 12))
plt.subplot(2, 3, 1)
plt.imshow(china_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

for i, (k, img) in enumerate(zip(components, reconstructed_images), 2):
    compression_ratio, error, explained_var = compute_metrics(china_gray, img, k, len(sigma))
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(f'k={k}, Compression Ratio: {compression_ratio:.1f}x\nError: {error:.2%}, Variance: {explained_var:.2%}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Plot singular values distribution
plt.figure(figsize=(10, 5))
plt.semilogy(sigma)
plt.title('Singular Values Distribution')
plt.xlabel('Index')
plt.ylabel('Singular Value (log scale)')
plt.grid(True)
plt.show()
```

### 4.5 Stability Issues and Solutions

1. **Numerical stability challenges**:
   - Rounding errors in floating-point calculations
   - Ill-conditioned matrices (high condition number)
   - Nearly identical eigenvalues

2. **Solutions**:
   - Use robust libraries like NumPy/SciPy that implement stable algorithms
   - Apply preprocessing: scaling, centering, regularization
   - Use QR decomposition with shifts for stability in eigenvalue computation
   - Employ iterative methods with convergence checks
   - Consider singular value decomposition instead of eigendecomposition

```python
import numpy as np

# Example: Dealing with ill-conditioned matrices
def robust_eigendecomposition(A, epsilon=1e-10):
    # Add small regularization to improve conditioning
    A_reg = A + epsilon * np.eye(A.shape[0])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A_reg)
    
    return eigenvalues, eigenvectors

# Example: Checking stability by reconstructing the original matrix
def check_decomposition_stability(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Reconstruct A = PDP^(-1)
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    
    A_reconstructed = P @ D @ P_inv
    
    # Compute reconstruction error
    error = np.linalg.norm(A - A_reconstructed) / np.linalg.norm(A)
    print(f"Reconstruction error: {error:.2e}")
    
    return error < 1e-10
```

## References and Further Reading

1. "Linear Algebra Done Right" by Sheldon Axler
2. "Introduction to Linear Algebra" by Gilbert Strang
3. "The Matrix Cookbook" by Kaare Brandt Petersen and Michael Syskind Pedersen
4. "Pattern Recognition and Machine Learning" by Christopher Bishop
5. "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
6. "Numerical Linear Algebra" by Trefethen and Bau
