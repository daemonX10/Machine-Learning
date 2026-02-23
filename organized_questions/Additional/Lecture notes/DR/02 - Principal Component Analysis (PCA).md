# Principal Component Analysis (PCA)

## 1. Overview
- First introduced by **Karl Pearson in 1901**.
- A **linear**, **global**, **deterministic** dimensionality reduction technique.
- Projects data into a new coordinate system whose axes are **principal components** (PCs).
- PCs are **uncorrelated** directions that capture maximum variance.

## 2. Core Intuition
- **Central idea**: reduce dimensions by finding principal components — uncorrelated axes that retain the most information.
- **Variance ≈ Information**: axes with high variance preserve more data structure; zero variance means all points collapse to the same value.
- **Example**: City size vs. cost of living — correlated features → one PC captures ~90% of information.

### Variance Formula
$$\text{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

## 3. Principal Components Explained
- PCs are **linear combinations** of original features that capture more information than individual features.
- They form an **orthogonal basis** → no correlation between PCs.
- For an M-dimensional dataset → M principal components (keep only top ones).
- **Scree plot**: shows explained variance per PC → pick PCs that cover sufficient variance.

### Eigenfaces Example
- PCs of face images = **eigenfaces** (eigenvectors of the face image covariance matrix).
- Any face can be reconstructed as a **linear combination** of eigenfaces (with some information loss if fewer PCs are used).

## 4. Mathematical Framework

### Step 1: Center the Data
- Subtract the mean of each feature so data is at the origin.
- Optionally: scale to unit variance (ensures equal contribution from each variable).

### Step 2: Compute the Covariance Matrix
$$\text{Cov}(X_i, X_j) = \frac{1}{n} \sum_{k=1}^{n} (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)$$

- **Diagonal entries**: variance of each variable.
- **Off-diagonal entries**: covariance between variables (joint direction of movement).
- Covariance matrix is **symmetric**.
- Difference from correlation: correlation = normalized covariance (measures intensity + direction; covariance only measures direction).

### Step 3: Eigenvalue Decomposition
- **Spectral Theorem**: any symmetric matrix (like covariance matrix) has **real eigenvectors and eigenvalues**.
- **Eigenvectors** of the covariance matrix = **principal components** (directions of max variance).
- **Eigenvalues** = explained variance of each PC.

#### Eigenvector Equation
$$A\mathbf{v} = \lambda \mathbf{v}$$
- $A$: covariance matrix
- $\mathbf{v}$: eigenvector (direction unchanged under transformation)
- $\lambda$: eigenvalue (scalar stretch factor)

### Step 4: Project Data onto PCs
$$Y = X \cdot V$$
- $X$: centered data matrix
- $V$: matrix of selected eigenvectors
- $Y$: transformed data in PC space

## 5. Key Equivalences
- **Minimizing squared distances** to principal axis = **Maximizing variance** along that axis (proved via Pythagorean theorem).
- Both optimization perspectives arrive at the same solution.

## 6. SVD vs. Eigendecomposition

| Property | Eigendecomposition | SVD |
|----------|--------------------|-----|
| Complexity | $O(n^3)$ | $O(n^2 m)$ — faster |
| Matrix type | Square matrices only | Any matrix |
| Requires covariance matrix | Yes | No (applied directly to data) |
| Results | Same as SVD | Same as eigendecomposition |
| Used in practice | For Kernel PCA | Default in sklearn |

- Most libraries (sklearn) use **SVD** for efficiency.
- **Kernel PCA** (nonlinear extension) still requires eigendecomposition.

## 7. Properties Summary Table

| Property | Value |
|----------|-------|
| **Scope** | Global |
| **Linearity** | Linear (Kernel PCA is nonlinear) |
| **Method type** | Projection-based (decomposition) |
| **Purpose** | Data analysis, visualization |
| **Deterministic** | Yes |
| **Complexity** | $O(n^3)$ eigendecomp / $O(n^2 m)$ SVD |
| **Key hyperparameter** | Number of principal components (use scree plot) |
| **Key idea** | Decompose covariance matrix into eigenvectors & eigenvalues |
| **Other applications** | Denoising, compression, correlation analysis |

## 8. Practical Notes (sklearn)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

# Access results
pca.explained_variance_ratio_  # eigenvalues (proportion)
pca.components_                # eigenvectors (PCs)
```
- Uses SVD internally (per documentation).
- Can extract explained variance ratios and components.
