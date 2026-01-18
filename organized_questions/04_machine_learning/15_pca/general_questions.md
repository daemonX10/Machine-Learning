# Pca Interview Questions - General Questions

## Question 1

**How is PCA used for dimensionality reduction?**

### Answer

**Definition:**
PCA reduces dimensions by transforming correlated features into a smaller set of uncorrelated principal components, keeping those that capture maximum variance.

**Process:**
1. Standardize data
2. Compute covariance matrix
3. Find eigenvectors (directions) and eigenvalues (variance)
4. Sort by eigenvalue, select top k
5. Project: $X_{new} = X \cdot V_k$

**Example:**
100 features → PCA → Keep 10 components capturing 95% variance

**Practical Benefits:**
- Faster model training
- Reduced storage
- Combat overfitting
- Enable visualization (reduce to 2D/3D)

---

## Question 2

**Why is PCA considered an unsupervised technique?**

### Answer

**Definition:**
PCA is unsupervised because it uses only input features (X) without any target labels (y). It finds structure based purely on variance within the data.

**Key Points:**
- No labels used in computation
- Objective: maximize variance (data-internal goal)
- Doesn't optimize for prediction
- Contrast: LDA uses labels to maximize class separation

**Implication:**
PCA can be used as preprocessing for both supervised and unsupervised downstream tasks.

---

## Question 3

**Derive the PCA from the optimization perspective, i.e., minimization of reconstruction error.**

### Answer

**Objective:**
Find k-dimensional subspace minimizing reconstruction error.

**Setup:**
- $V_k$: Orthonormal basis (p × k)
- Projection: $\hat{x} = V_k V_k^T x$
- Error: $||x - \hat{x}||^2$

**Derivation:**

$$J = \sum_i ||x_i - V_k V_k^T x_i||^2$$

Expanding:
$$= \sum_i (x_i^T x_i - x_i^T V_k V_k^T x_i)$$

First term is constant. Minimizing J = Maximizing:
$$\sum_i x_i^T V_k V_k^T x_i = Tr(V_k^T X^T X V_k)$$

**Result:**
Maximize $Tr(V_k^T C V_k)$ subject to $V_k^T V_k = I$

**Solution:** Columns of $V_k$ = top k eigenvectors of C

---

## Question 4

**How do you determine the number of principal components to use?**

### Answer

**Methods:**

| Method | Approach |
|--------|----------|
| **Cumulative Variance** | Keep k components for 90-95% variance |
| **Scree Plot** | Find elbow where curve flattens |
| **Kaiser Rule** | Keep components with eigenvalue > 1 |
| **Cross-validation** | Tune k for best downstream model performance |

**Recommended Approach:**
1. Plot cumulative variance
2. Check scree plot for elbow
3. If for ML model, cross-validate k

**Python:**
```python
cumsum = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cumsum >= 0.95) + 1
```

---

## Question 5

**Provide examples of how PCA can be used in image processing.**

### Answer

**1. Facial Recognition (Eigenfaces):**
- Flatten face images to vectors
- PCA finds principal "eigenfaces"
- Represent any face as combination of eigenfaces
- Match new faces in low-dimensional space

**2. Image Compression:**
- Divide image into blocks
- PCA on block collection
- Store only top component scores
- Reconstruct with minimal quality loss

**3. Hyperspectral Imaging:**
- Hundreds of spectral bands (highly correlated)
- PCA reduces to few components
- Enables efficient classification

**4. Handwriting Recognition:**
- Extract principal components of digit variations
- Use as features for classification

---

## Question 6

**Can PCA be applied to categorical data? Why or why not?**

### Answer

**Short Answer:**
Standard PCA is NOT appropriate for categorical data.

**Why Not:**
- PCA assumes continuous, Gaussian-like data
- Covariance/correlation meaningless for categories
- Distance metrics don't apply to categories
- One-hot encoding creates sparse, high-dimensional data

**Alternatives:**

| Method | Description |
|--------|-------------|
| **MCA (Multiple Correspondence Analysis)** | PCA equivalent for categorical data |
| **FAMD** | Mixed data (categorical + numerical) |
| **Categorical PCA (CATPCA)** | Optimal scaling for categories |

**If You Must Use PCA:**
- One-hot encode categories
- But results may be misleading
- Better to use MCA

---

## Question 7

**How can PCA be parallelized to handle very large datasets?**

### Answer

**Approaches:**

**1. Incremental PCA:**
- Process mini-batches sequentially
- Update components iteratively
- Memory efficient

**2. Randomized SVD:**
- Use random projections
- Approximate top k components
- Much faster for large matrices

**3. Distributed PCA:**
- Split data across nodes
- Compute local covariances
- Aggregate and decompose

**4. Parallel SVD Libraries:**
- Use GPU-accelerated libraries (cuML, RAPIDS)
- Distributed computing (Spark MLlib)

**sklearn Options:**
```python
# Randomized (faster)
PCA(n_components=k, svd_solver='randomized')

# Incremental (memory efficient)
IncrementalPCA(n_components=k, batch_size=1000)
```

---

## Question 8

**Compare the use of PCA to select features with other feature selection methods.**

### Answer

**Key Distinction:**
PCA is feature EXTRACTION (creates new features), not feature SELECTION (picks original features).

**Comparison:**

| Aspect | PCA | Feature Selection |
|--------|-----|-------------------|
| Output | New synthetic features | Subset of original features |
| Interpretability | Less interpretable | Original feature meaning preserved |
| Supervision | Unsupervised | Can be supervised |
| Information | Combines all features | Discards features |

**Feature Selection Methods:**
- **Filter**: Correlation, mutual information, chi-square
- **Wrapper**: RFE, forward/backward selection
- **Embedded**: L1 regularization, tree importance

**When to Use PCA:**
- Multicollinearity present
- Want decorrelated features
- Interpretability less important

**When to Use Feature Selection:**
- Need interpretable features
- Domain knowledge suggests specific features matter
- Regulatory requirements for explainability

---

## Question 9

**What recent advancements have been made concerning PCA for big data?**

### Answer

**Key Advancements:**

**1. Randomized PCA:**
- Probabilistic approximation
- O(npk) instead of O(np²)
- Near-exact results

**2. Streaming/Online PCA:**
- Processes data incrementally
- Handles concept drift
- Memory independent of data size

**3. GPU Acceleration:**
- RAPIDS cuML library
- Orders of magnitude faster
- Handles larger datasets

**4. Distributed Implementations:**
- Spark MLlib PCA
- Handles TB-scale data
- Parallel covariance computation

**5. Sparse PCA:**
- Produces sparse loadings
- More interpretable components
- Useful for high-dimensional data

**6. Robust PCA:**
- Handles outliers
- Separates low-rank structure from sparse noise

---
