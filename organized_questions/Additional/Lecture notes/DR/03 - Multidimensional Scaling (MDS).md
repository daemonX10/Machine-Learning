# Multidimensional Scaling (MDS)

## 1. Overview
- Roots in the **1950s** (Torgerson's publication); a **class of techniques**, not a single method.
- MDS starts from a **distance/dissimilarity matrix** (not raw feature data) — works even without access to original features.
- Goal: map data into a lower-dimensional space while **preserving distances** between observations.
- Name origin: "scaling" distances into a multi-dimensional space.

## 2. Historical Context
- Originates from **psychometrics** — understanding human judgment of similarity.
- Example: ranking similarity of emotions (happy, sad, surprised) → visualize emotional space from pairwise distances alone.
- Modern neural network embeddings are a direct evolution of this idea (Shepard, 1951).

## 3. Three Variants of MDS

| Variant | Published By | Distance Type | Method | Notes |
|---------|-------------|---------------|--------|-------|
| **Classical MDS** (PCoA) | Torgerson & Gower | Euclidean only | Eigen decomposition | Equivalent to PCA on distance matrix |
| **Metric MDS** | Shepard & Kruskal | Any metric distance | Iterative (SMACOF) | Most widely used; sklearn default |
| **Non-metric MDS** | Kruskal | Ordinal / non-metric | Iterative (SMACOF + isotonic regression) | Preserves rank order, not exact distances |

### Notation
- $\delta_{ij}$: original distances
- $d_{ij}$: distances in mapped low-dimensional space
- $\hat{d}_{ij}$: disparities (used in non-metric MDS)

## 4. Classical MDS (Principal Coordinate Analysis)

### Algorithm Steps
1. **Construct Gram Matrix** $S$ from distance matrix:
   $$S = -\frac{1}{2} C D^2 C$$
   - $D^2$: element-wise squared distances
   - $C$: centering matrix (removes row/column averages for translational invariance)
   
2. **Eigendecomposition** of the Gram matrix:
   - Gram matrix is symmetric & positive definite → can be diagonalized.
   - Eigenvectors give projection directions.

3. **Project** using eigenvectors × √(eigenvalues):
   - Keep top eigenvectors (largest eigenvalues) for dimensionality reduction.
   - First coordinate = largest variation.

### Key Constraints
- Only works with **Euclidean distances** (requires inner product reformulation).
- Solution is **not unique** — rotation/reflection invariant.

## 5. Metric MDS (Least-Squares MDS)

### Distance Metrics
A **metric distance** must satisfy:
1. **Positivity**: $d(i,j) \geq 0$, $d(i,i) = 0$
2. **Symmetry**: $d(i,j) = d(j,i)$
3. **Triangle inequality**: $d(i,j) \leq d(i,k) + d(k,j)$

Common metrics: Euclidean, Manhattan, Cosine, Chebyshev, Mahalanobis.

### Algorithm Steps
1. **Initialize** low-dimensional points (randomly or advanced strategy).
2. **Calculate** Euclidean distance matrix for the low-dimensional map.
3. **Compute stress** (loss function) comparing the two distance matrices.
4. **Optimize** using SMACOF to minimize stress.

### Stress Function (Kruskal)
**Raw stress:**
$$\sigma_r = \sum_{i<j} (\delta_{ij} - d_{ij})^2$$

**Normalized stress** (scale-invariant, between 0 and 1):
$$\sigma = \sqrt{\frac{\sum_{i<j} (\delta_{ij} - d_{ij})^2}{\sum_{i<j} d_{ij}^2}}$$

| Stress Value | Interpretation |
|-------------|----------------|
| < 5% | Excellent |
| 5–10% | Good |
| 10–20% | Fair |
| > 20% | Poor |

### Stress Variants
- **Weighted stress**: weights $w_{ij}$ per distance (handles missing data by setting $w=0$).
- **Sammon stress**: gives more weight to small distances.

### SMACOF Algorithm
- **S**caling by **MA**jorizing a **CO**mplicated **F**unction.
- Handles the **non-convex** stress objective by constructing a **convex surrogate function**.
- Guarantees **monotone convergence**.
- This is what **sklearn uses** under the hood.

### Shepard Diagram
- Scatter plot: actual distances vs. low-dimensional distances.
- Ideal fit: all points on the diagonal.

### Choosing Dimensions
- Run MDS with different dimensions → plot stress vs. dimension.
- Use **elbow criterion** (like PCA's scree plot).

## 6. Non-Metric MDS

### When to Use
- Distances violate the **triangle inequality** (non-metric).
- **Ordinal data**: only rank-order of distances is available (not exact values).

### Key Concept: Monotonic Function
- Maps original distances to **disparities** $\hat{d}_{ij}$ preserving order.
- Fitted using **isotonic regression** (pool adjacent violators algorithm).
- Optimization then compares $d_{ij}$ vs. $\hat{d}_{ij}$ instead of $d_{ij}$ vs. $\delta_{ij}$.

### Algorithm
1. Fit monotonic function to distances → get disparities.
2. Use SMACOF to optimize stress between low-dim distances and disparities.
3. Both monotonic function parameters and point coordinates are learned jointly.

## 7. Properties Summary Table

| Property | Classical MDS | Metric MDS | Non-metric MDS |
|----------|--------------|------------|----------------|
| **Scope** | Global | Global | Global |
| **Linearity** | Linear | Nonlinear | Nonlinear |
| **Method** | Projection (eigendecomp) | Manifold learning | Manifold learning |
| **Deterministic** | Yes | No (random init) | No (random init) |
| **Complexity** | $O(n^3)$ | $O(n^2)$ | $O(n^2)$ |
| **Key hyperparams** | Dimensionality | Metric, dimensions, iterations, ε | Same + monotonic function |
| **Distance type** | Euclidean only | Any metric | Non-metric / ordinal |

## 8. Practical Notes (sklearn)
```python
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

# Compute distance matrix
dist_matrix = pairwise_distances(X, metric='manhattan')

# Metric MDS with precomputed distances
mds = MDS(n_components=2, metric=True, dissimilarity='precomputed')
X_mds = mds.fit_transform(dist_matrix)

# Access stress
print(mds.stress_)  # raw stress value
```
- Pass `dissimilarity='precomputed'` for pre-computed distance matrices.
- Manhattan distance often works better in high dimensions than Euclidean.
- Run multiple times with different initializations (non-convex objective).
