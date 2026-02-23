# t-distributed Stochastic Neighbor Embedding (t-SNE)

## 1. Overview
- One of the most commonly used nonlinear dimensionality reduction techniques.
- A **local, nonlinear manifold learning** technique for **visualization**.
- Produces better low-dimensional embeddings than PCA/MDS because it focuses on **local neighborhoods** instead of preserving global Euclidean distances.

## 2. Why Not PCA/MDS for Visualization?
- PCA/MDS try to preserve **global Euclidean distances** in low dimensions.
- Euclidean distance is unreliable in high dimensions (**curse of dimensionality**).
- Fewer dimensions → less space → distances converge to zero → one big cluster.
- **Geodesic distance** (shortest path along the manifold surface) is better for unrolling manifolds like the Swiss Roll, but requires neighbor-to-neighbor paths.

## 3. Historical Development

| Year | Contribution | Authors |
|------|-------------|---------|
| 2002 | **SNE** (Stochastic Neighbor Embedding) | Hinton & Roweis |
| 2008 | **t-SNE** (improvements to SNE) | van der Maaten & Hinton |
| 2013 | **Barnes-Hut t-SNE** (scalability) | van der Maaten |

## 4. SNE Algorithm

### Step 1: Compute Pairwise Distances
- Calculate Euclidean distances between all points in high-D space → distance matrix.
- Even though global distances aren't preserved, we need them to identify neighbors.

### Step 2: Convert Distances to Conditional Probabilities
For each point $x_i$, define similarity to neighbor $x_j$:
$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

- Gaussian distribution centered at each point.
- Closer points → higher probability; farther points → very low probability.
- This **relaxes** the distance-preservation problem by making far-away points irrelevant.

### Step 3: Perplexity & Sigma Selection
**Perplexity** = hyperparameter controlling how many neighbors to include.

$$\text{Perp}(P_i) = 2^{H(P_i)}$$

where $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$ is the **Shannon entropy**.

- High perplexity → flat Gaussian → many neighbors → more global view.
- Low perplexity → peaked Gaussian → few neighbors → more local view.
- Each point gets its **own σ** via **binary search** to match the specified perplexity.
- Ensures each point has roughly the **same effective number of neighbors**.

### Step 4: Low-Dimensional Probabilities
- Same Gaussian approach applied to low-D points $y_i$:
$$q_{j|i} = \frac{\exp(-\|y_i - y_j\|^2)}{\sum_{k \neq i} \exp(-\|y_i - y_k\|^2)}$$

### Step 5: Minimize KL Divergence
$$C = \sum_i KL(P_i \| Q_i) = \sum_i \sum_j p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$$

- Compares high-D distribution ($P$) with low-D distribution ($Q$) for each point.
- Penalizes more when close neighbors in $P$ are placed far apart in $Q$.
- Optimized via **gradient descent**.
- Gradient = **spring forces** between points (attractive/repulsive = N-body simulation).

## 5. t-SNE Improvements over SNE

### 5.1 The Crowding Problem
- When intrinsic dimensionality > target dimensions, low-D space is too cramped.
- Small residual attractive forces cause all points to **clump together**.
- SNE can't form clear clusters.

### 5.2 Solution: Student's t-Distribution (the "t" in t-SNE)
- Replace Gaussian with **Student's t-distribution (1 degree of freedom)** in low-D space:
$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

- **Heavy tails** → assigns higher similarity to moderate-distance points.
- Allows points to be further apart while maintaining reasonable probability.
- **Benefits**:
  - No binary search needed for σ (same distribution for all points).
  - No exponential computation → more efficient.

### 5.3 Symmetric Probabilities
- Original SNE: $p_{j|i} \neq p_{i|j}$ (asymmetric, problematic for outliers).
- t-SNE uses **joint probabilities**: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$
- **Benefits**: outliers contribute to loss function; simpler gradient computation.

### 5.4 Early Exaggeration
- Problem: with random initialization, repulsive forces can trap separated points.
- Solution: **multiply attractive forces** by a factor >1 for the first few iterations.
- Allows separated points to overcome repulsive barriers → then return to normal forces.
- Produces more **densely packed**, well-separated clusters.

## 6. Barnes-Hut t-SNE (Scalability)

| Method | Complexity | Scale |
|--------|-----------|-------|
| Exact t-SNE | $O(n^2)$ | Small datasets |
| Barnes-Hut t-SNE | $O(n \log n)$ | Millions of points |

### How It Works
- Borrowed from **N-body simulation** in physics (Barnes & Hut).
- Divide space into **cubic cells** → summarize clusters by their **centers**.
- Build a **quad-tree** structure to efficiently compute repulsive forces.
- Approximate repulsive gradient using cell centers instead of all individual points.

## 7. Quality Assessment
- Use **Shepard diagrams**: scatter plot of high-D distances vs. low-D distances.
- Points near diagonal = good preservation.

## 8. Important Limitations
- **Not useful for clustering** — doesn't guarantee meaningful within-cluster or between-cluster distances.
- **Cannot apply to new data** (no parametric mapping learned).
- **Stochastic** — different initializations → different results (non-convex objective).
- Cluster **distances between groups are meaningless**.

## 9. Properties Summary Table

| Property | Value |
|----------|-------|
| **Scope** | Local (neighborhood-based) |
| **Linearity** | Nonlinear (manifold learning) |
| **Purpose** | Visualization only |
| **Deterministic** | No (non-convex, random init) |
| **Complexity** | $O(n^2)$ exact; $O(n \log n)$ Barnes-Hut |
| **Key hyperparams** | Perplexity, early exaggeration, learning rate |
| **Key idea** | Neighborhood probabilities + KL divergence |
| **Can apply to new data?** | No (iterative, no projection function) |

## 10. Practical Notes (sklearn)
```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,           # ~number of neighbors
    early_exaggeration=12.0, # attractive force multiplier
    learning_rate='auto',
    init='pca',              # PCA initialization (more stable)
    method='barnes_hut',     # or 'exact'
    n_jobs=4                 # parallelize binary search
)
X_tsne = tsne.fit_transform(X)

# Access final KL divergence
print(tsne.kl_divergence_)
```

### Hyperparameter Guidelines
| Parameter | Effect |
|-----------|--------|
| **Perplexity ↑** | More neighbors, more global view |
| **Perplexity ↓** | Fewer neighbors, tighter local clusters |
| **Early exaggeration ↑** | Stronger initial clustering force |
| **Learning rate** | Too low = slow; too high = instability |
