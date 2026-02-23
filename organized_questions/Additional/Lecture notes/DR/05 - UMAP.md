# Uniform Manifold Approximation and Projection (UMAP)

## 1. Overview
- Published **2018** by McInnes, Healy & Melville (Tutte Institute, Canada).
- Claims to preserve **both local and global structure** (sits between t-SNE and MDS).
- Computational complexity: ≈ $O(n^{1.14})$ — **near-linear**, faster than t-SNE.
- Mathematically grounded in **topological data analysis (TDA)**.

## 2. UMAP vs. t-SNE: Clearing Up Myths
- UMAP is often claimed to be "superior" to t-SNE, but:
  - Kobak & Linderman (2021) showed differences mainly arise from **initialization** strategy, not algorithmic superiority.
  - No definitive evidence UMAP preserves global structure better per se.
- **Clear advantage**: UMAP is significantly **faster** for large datasets.

## 3. Topological Foundations

### 3.1 Topological Data Analysis (TDA)
- Interested in the **shape** of data in high-dimensional space.
- Extracts topological features: **connections, loops, holes** (field of homology).

### 3.2 Simplices & Simplicial Complexes
- **Simplex**: generalization of triangles to arbitrary dimensions.
  - 0-simplex: point; 1-simplex: edge; 2-simplex: triangle; 3-simplex: tetrahedron.
- **Simplicial complex**: structure formed by gluing simplices along faces → approximates data geometry.

### 3.3 Filtration (Vietoris-Rips Complex)
- Place balls of radius ε around each point.
- Connect points whose balls overlap → forms a graph.
- Vary ε → evolving structure reveals manifold topology.
- UMAP uses the **Vietoris-Rips complex** (0- and 1-simplices = nodes and edges).

### 3.4 Persistent Homology
- Tracks topological features (holes, loops) across different radii.
- Features that **persist** across many radii are most meaningful.

## 4. UMAP Algorithm — Step 1: Construct Topological Representation

### 4.1 The Radius Problem
- No single radius works for all data → UMAP uses **number of neighbors K** as hyperparameter.

### 4.2 Uniform Manifold Assumption (the "U" in UMAP)
- If data were **uniformly distributed** on the manifold → same radius captures same number of neighbors everywhere.
- **Nerve Theorem**: under uniform distribution, a simplicial complex of the cover correctly approximates the manifold.
- **Geodesic distance** ≈ $1/r$ (average distance between uniform points).

### 4.3 Handling Non-Uniform Data
- Real data is **not uniform** → different densities in different regions.
- UMAP's solution: assign each point an **individual distance function** that forces data to appear locally uniform.
  - Sparse regions **shrink**; dense regions **enlarge**.
  - Balls stretch to the **K nearest neighbor** of each point.
  - Choosing K (neighbors) is more robust than choosing a radius.

### 4.4 Local Distance Function
$$d_i(x_i, x_j) = \exp\left(\frac{-\max(0, \|x_i - x_j\| - \rho_i)}{\sigma_i}\right)$$

Where:
- $\rho_i$ = distance to nearest neighbor (**local connectivity constraint**)
  - Ensures decay starts **beyond** the first neighbor → no isolated points.
  - First neighbor always gets weight = 1.
- $\sigma_i$ = normalization factor controlling decay intensity.
  - Chosen so total probability mass ≈ $\log_2(K)$.

### 4.5 Benefits of Local Connectivity Constraint
- No isolated points in the graph.
- Stretches cramped distance distributions apart → better separation.

### 4.6 Merging Local Views → Global Graph
- Each point has its own weighted K-nearest-neighbor graph.
- Edge weights are **asymmetric** ($d_i(x_j) \neq d_j(x_i)$).
- Merge via **fuzzy simplicial sets** (fuzzy topology):
$$W_{ij} = a_{ij} + a_{ji} - a_{ij} \cdot a_{ji}$$
- This combines as: "probability that at least one edge exists."
- Provides **mathematical guarantee** of appropriate topological representation.

## 5. UMAP Algorithm — Step 2: Optimize Low-Dimensional Layout

### 5.1 Process
1. Initialize low-D points (via **spectral embedding** of the adjacency matrix).
2. Construct K-NN graph in low-D space with same procedure.
3. Compare high-D graph $H$ and low-D graph $L$ via their **adjacency matrices**.

### 5.2 Loss Function: Cross-Entropy
$$C = \sum_{(i,j)} \left[ w_{ij}^H \log\frac{w_{ij}^H}{w_{ij}^L} + (1-w_{ij}^H) \log\frac{1-w_{ij}^H}{1-w_{ij}^L} \right]$$

- First term: points with high edge weight should be **close** in low-D.
- Second term: points with low edge weight should be **far apart** in low-D.
- Considers both **existing and non-existing edges** (unlike t-SNE's KL divergence).

### 5.3 Optimization
- **Stochastic gradient descent** on point coordinates.
- Equivalent to a **force-directed graph layout** (attractive + repulsive forces).

## 6. Implementation Speedups
1. **Nearest Neighbor Descent**: near-linear approximate K-NN (neighbors-of-neighbors).
2. **Negative Sampling**: skip computing all non-edges; sample a subset.
3. **Spectral Embedding** initialization (not random) → better starting position.
4. **Simplified loss**: some terms removed for efficiency.

## 7. UMAP vs. t-SNE Comparison

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| **Distance function** | Gaussian → t-distribution | Exponential decay with local connectivity |
| **Local resolution** | Perplexity (via σ binary search) | Number of neighbors K (more intuitive) |
| **Global structure** | Poor (KL divergence ignores it) | Better (cross-entropy on all edges) |
| **Loss function** | KL divergence | Cross-entropy |
| **Complexity** | $O(n^2)$ or $O(n \log n)$ (Barnes-Hut) | $O(n^{1.14})$ |
| **Initialization** | Random or PCA | Spectral embedding |
| **Deterministic** | No | No |

- Both use **exponential decay** and **σ** normalization.
- UMAP adds the **local connectivity constraint** (beneficial for global structure).
- Some literature argues UMAP is a **special case of t-SNE**.

## 8. Properties Summary Table

| Property | Value |
|----------|-------|
| **Scope** | Local (with some global preservation) |
| **Linearity** | Nonlinear (manifold learning) |
| **Purpose** | Visualization, data analysis |
| **Deterministic** | No (stochastic approximations) |
| **Complexity** | $O(n^{1.14})$ (empirical) |
| **Key hyperparams** | `n_neighbors` (K), `min_dist` |
| **Key idea** | Fuzzy simplicial complex → K-NN graph → graph layout optimization |
| **Applications** | Bioinformatics, genomics, general high-D data |

## 9. Practical Notes
```python
import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,     # K nearest neighbors (locality vs. globality)
    min_dist=0.1,       # how tightly points cluster
    metric='euclidean'  # distance metric for high-D space
)
X_umap = reducer.fit_transform(X)
```

### Hyperparameter Guidelines
| Parameter | Effect |
|-----------|--------|
| **n_neighbors ↑** | Larger balls → more global structure |
| **n_neighbors ↓** | Smaller balls → finer local detail |
| **min_dist ↑** | Looser clusters, more spread |
| **min_dist ↓** | Tighter, denser clusters |

- Not in sklearn — separate `umap-learn` package.
- Excellent documentation with interactive examples.
- Google's interactive playground available for experimenting with parameters.

## 10. Full Series Comparison Table

| Property | PCA | Classical MDS | Metric MDS | Non-metric MDS | t-SNE | UMAP |
|----------|-----|--------------|------------|----------------|-------|------|
| Scope | Global | Global | Global | Global | Local | Local-Global |
| Linear? | Yes | Yes | No | No | No | No |
| Deterministic | Yes | Yes | No | No | No | No |
| Complexity | $O(n^2m)$ | $O(n^3)$ | $O(n^2)$ | $O(n^2)$ | $O(n \log n)$* | $O(n^{1.14})$ |
| Key params | # PCs | Dim | Metric, iter | + monotonic fn | Perplexity | n_neighbors, min_dist |

*Barnes-Hut approximation
