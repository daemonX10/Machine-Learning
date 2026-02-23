# Dimensionality Reduction: Introduction & Manifold Learning

## 1. Why Dimensionality Reduction?
- High-dimensional data is common: images (pixel space), word embeddings, molecular data, etc.
- Human perception is limited to **3 dimensions** — multi-dimensional data must be projected to lower dimensions for visualization and comprehension.
- Also used to **improve machine learning algorithms** by reducing feature count.

## 2. Taxonomy of Techniques

| Category | Type | Examples |
|----------|------|----------|
| **Linear** | Global | PCA, Metric MDS |
| **Nonlinear (Manifold Learning)** | Global | Non-metric MDS |
| **Nonlinear (Manifold Learning)** | Local | t-SNE |
| **Nonlinear (Manifold Learning)** | Local–Global | UMAP |
| Neural network-based | — | Autoencoders |

## 3. Mathematical Formulation
- Start with **N samples** in **M-dimensional** space (original/ambient space).
- Define a distance metric $D_M$ in the original space.
- Goal: find a mapping to a **lower-dimensional space** (e.g., 2D) with metric $D_2$, such that **distance ratios are preserved**.
- Mapping always has **inherent error** since degrees of freedom are reduced.
- We typically care about the **topological structure** (relationships between points) more than absolute values.

## 4. Curse of Dimensionality
- Coined by **Bellman (1961)**.
- As dimensionality increases:
  - Distances become **more uniform** (distance concentration) — especially with Euclidean metric.
  - Absolute distances grow larger — data spreads to the **shell/edges** of the data space.
  - ML algorithms struggle to **separate data** with too many dimensions.
- **More features ≠ always better**.

## 5. Blessing of Non-Uniformity
- The opposite effect: data often has a **lower intrinsic dimensionality** than the ambient space.
- Real-world data is **not uniformly distributed** — it concentrates in specific regions.
- **Example — face images**: thousands of pixels, but describable with very few attributes (hair, lip shape, etc.) → intrinsically low-dimensional.

## 6. Manifolds
- A **manifold** is a topological space that **locally behaves like Euclidean space**.
- Originated from mathematician **Riemann**.
- Types:
  - 1D manifolds: lines, circles
  - 2D manifolds: spheres, planes, the surface of Earth
- **Swiss Roll Dataset**: benchmark for dimensionality reduction — a 3D curved manifold that should be "unrolled" into a 2D plane.

## 7. Manifold Hypothesis
> Many high-dimensional datasets lie on **low-dimensional latent manifolds**.
- The manifold is **latent** — we don't know its shape a priori.
- Manifold learning techniques **approximate** the underlying low-dimensional manifold.
- Data can often be separated more effectively on the correct manifold → improves ML performance.

## 8. Real-World Applications
| Domain | Application |
|--------|-------------|
| **Computational Biology** | Gene expression analysis, genetic interaction clustering (UMAP) |
| **Anomaly Detection** | Unsupervised anomaly detection in heating systems (time series) |
| **Finance** | Dow Jones index analysis — clusters reveal market crashes, pandemics |

## 9. Key Takeaways
1. Techniques can be **linear vs. nonlinear** and **global vs. local**.
2. Goal: transform high-dimensional data to low-dimensional space while **preserving data structure**.
3. Data may lie on **low-dimensional manifolds** that we can try to learn.
