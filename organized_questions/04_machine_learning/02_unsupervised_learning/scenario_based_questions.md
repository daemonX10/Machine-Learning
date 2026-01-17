# Unsupervised Learning - Scenario-Based Questions

## Question 1: Can you discuss the differences between hard and soft clustering?

### Definition

| | Hard Clustering | Soft Clustering |
|---|----------------|-----------------|
| **Assignment** | Each point belongs to exactly one cluster | Each point has probability of belonging to each cluster |
| **Output** | Single label per point | Vector of probabilities per point |
| **Example** | K-means, DBSCAN | GMM, Fuzzy C-Means |

### Example Output

**Hard (K-means)**:
- Point A → Cluster 1

**Soft (GMM)**:
- Point A → 70% Cluster 1, 30% Cluster 2

### When to Use Each

| Use Hard When | Use Soft When |
|---------------|---------------|
| Clear, actionable segmentation needed | Clusters naturally overlap |
| Each customer gets one marketing campaign | Need to represent uncertainty |
| Simple interpretation required | Mixed membership is realistic (e.g., genres) |

### Analogy
- **Hard**: Mail goes into one mailbox only
- **Soft**: Song is "70% rock, 20% pop, 10% jazz"

---

## Question 2: Discuss the concepts of support, confidence, and lift in association rule learning.

### For Rule: {A} → {B}

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Support** | P(A ∩ B) | How frequent is itemset? |
| **Confidence** | P(A ∩ B) / P(A) | Given A, how likely is B? |
| **Lift** | Support(A,B) / (Support(A) × Support(B)) | Is relationship real or coincidence? |

### Example: {Diapers} → {Beer}

- **Support = 0.1**: 10% of transactions have both
- **Confidence = 0.8**: 80% of diaper buyers also buy beer
- **Lift = 2.0**: Diaper buyers are 2× more likely to buy beer than average customer

### Lift Interpretation

| Lift | Meaning |
|------|---------|
| **> 1** | Positive correlation (useful rule) |
| **= 1** | Independent (no relationship) |
| **< 1** | Negative correlation (substitute products) |

### Why Confidence Alone Isn't Enough
High confidence {X} → {Bread} might just mean bread is very popular (appears in 90% of transactions), not that X drives bread purchases. Lift corrects for this.

---

## Question 3: Discuss the Expectation-Maximization (EM) algorithm and its application in clustering.

### Definition
EM is an iterative algorithm for finding maximum likelihood estimates when there are latent (hidden) variables.

### Application: Training GMMs

**The Problem**: We don't know which Gaussian generated each point (latent variable)

### The Two Steps

**E-Step (Expectation)**:
- Given current parameters, calculate probability each point belongs to each Gaussian
- "Soft assignment" based on current beliefs

**M-Step (Maximization)**:
- Given soft assignments, update parameters to maximize likelihood
- New μ = weighted average of points
- New Σ = weighted covariance
- New π = average responsibility

### The Loop
```
Initialize parameters randomly
Repeat:
    E-step: Calculate responsibilities (who generated what?)
    M-step: Update parameters (what are the Gaussians?)
Until convergence
```

### Analogy
- **E-step**: "Based on my theory, what's probability this evidence belongs to Suspect A vs B?"
- **M-step**: "Update my theory about suspects based on these probabilities"

---

## Question 4: Discuss how you could evaluate the performance of a clustering algorithm.

### Without Ground Truth (Real-World Scenario)

**Internal Validation Indices**

| Index | Measures | Better |
|-------|----------|--------|
| **Silhouette** | Cohesion vs separation | Higher (max 1) |
| **Davies-Bouldin** | Cluster similarity | Lower |
| **Calinski-Harabasz** | Variance ratio | Higher |

### With Ground Truth (Benchmarking)

**External Validation Indices**

| Index | Description | Perfect Score |
|-------|-------------|---------------|
| **Adjusted Rand Index** | Agreement corrected for chance | 1.0 |
| **Normalized Mutual Information** | Mutual information normalized | 1.0 |
| **V-measure** | Harmonic mean of homogeneity & completeness | 1.0 |

### Practical Strategy

1. **Quantitative**: Use silhouette score to guide K selection
2. **Qualitative**: Analyze cluster characteristics
   - Plot feature distributions per cluster
   - Use domain knowledge to validate meaningfulness
3. **Business Validation**: Do clusters make sense? Are they actionable?

**Key Point**: High silhouette score ≠ useful clusters. Always validate with domain expertise.

---

## Question 5: Discuss how unsupervised learning can be used in image segmentation.

### Core Idea
Treat each pixel as a data point → Cluster pixels → Segments = Clusters

### Feature Representation

| Feature | Description |
|---------|-------------|
| **Color** | (R, G, B) per pixel |
| **Spatial** | (x, y) coordinates |
| **Combined** | (R, G, B, x, y) for color + location |

### Algorithm Choice

**K-means**:
- Choose K (number of segments)
- Cluster by color/position
- Replace pixels with centroid color
- Good for color quantization

**DBSCAN / Mean-Shift**:
- No K needed
- Better for irregular shapes
- Can identify varying density regions

### Example: Background Removal
1. Represent pixels by RGB
2. K-means with K=2
3. One cluster = foreground, one = background
4. Create mask to separate

### Limitations
- Simple approach; deep learning (U-Net) is state-of-the-art for complex segmentation
- Works well for color-based separation, less for semantic understanding

---

## Question 6: Discuss the use of self-organizing maps in unsupervised learning.

### Definition
Self-Organizing Map (SOM) is a neural network that projects high-dimensional data onto a 2D grid while preserving topological relationships.

### Architecture
- **Input Layer**: Receives high-D vectors
- **Output Layer**: 2D grid of neurons, each with weight vector same dimension as input

### Training (Competitive Learning)

1. **Competition**: For each input, find Best Matching Unit (BMU) - neuron with most similar weights
2. **Cooperation**: Update BMU and its neighbors on the grid
3. **Adaptation**: Move weights closer to input; neighbors move less

### Result
- Neurons close on map have similar weight vectors
- Preserves topology: similar inputs activate nearby neurons

### Use Cases

| Application | How |
|-------------|-----|
| **Visualization** | Project high-D data to 2D map |
| **Clustering** | Neurons form natural clusters |
| **Feature Extraction** | BMU position as new feature |

### Comparison to t-SNE
- SOM: Discrete grid, faster, good for large data
- t-SNE: Continuous space, better local structure preservation

---

## Question 7: Propose an unsupervised learning strategy to segment customers for targeted marketing.

### Strategy Pipeline

**Step 1: Feature Engineering (360° Customer View)**

| Category | Features |
|----------|----------|
| **RFM** | Recency, Frequency, Monetary value |
| **Demographic** | Age, location, tenure |
| **Behavioral** | Session length, categories viewed, discount usage |

**Step 2: Preprocessing**
```python
# Handle missing values
X = X.fillna(X.median())

# Scale features
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# Optional: PCA if many features
```

**Step 3: Clustering**
```python
from sklearn.cluster import KMeans

# Find optimal K
for k in range(2, 11):
    score = silhouette_score(X_scaled, KMeans(n_clusters=k).fit_predict(X_scaled))
    
# Train final model
kmeans = KMeans(n_clusters=optimal_k)
labels = kmeans.fit_predict(X_scaled)
```

**Step 4: Cluster Profiling**
- Calculate mean features per cluster
- Create personas:

| Cluster | Profile | Action |
|---------|---------|--------|
| 1 | Champions: High RFM | Loyalty rewards |
| 2 | At-Risk: High past value, low recency | Re-engagement campaign |
| 3 | Bargain Hunters: High discount affinity | Sale notifications |

**Step 5: Deploy & Monitor**
- Assign new customers to clusters
- Track conversion rates per segment
- Retrain periodically

---

## Question 8: How would you use clustering to inform feature creation in a supervised learning task?

### The Approach
Use cluster labels as a new feature for supervised model.

### Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Fit clustering on training data only
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train_scaled)

# 3. Create cluster feature for both sets
X_train['cluster_id'] = kmeans.predict(X_train_scaled)
X_test['cluster_id'] = kmeans.predict(X_test_scaled)

# 4. One-hot encode (avoid false ordinal relationship)
encoder = OneHotEncoder(sparse=False)
cluster_features_train = encoder.fit_transform(X_train[['cluster_id']])
cluster_features_test = encoder.transform(X_test[['cluster_id']])

# 5. Concatenate with original features
X_train_augmented = np.hstack([X_train_scaled, cluster_features_train])
```

### Why It Works
- Cluster ID captures complex non-linear interactions
- Acts as high-level summary of feature relationships
- Especially helpful for linear models (adds non-linearity)

### Alternative: Distance Features
Create features for distance to each centroid:
```python
distances = kmeans.transform(X_scaled)  # (n_samples, k)
```

---

## Question 9: Discuss a framework for detecting communities in social networks via unsupervised learning.

### Framework

**Step 1: Graph Representation**
- Nodes = Users
- Edges = Connections (friends, follows, interactions)
- Store as adjacency matrix or list

**Step 2: Choose Algorithm**

**Option A: Louvain Method (Recommended)**
- Maximizes modularity (within-community density vs between)
- Fast, scalable to millions of nodes
- Automatically determines number of communities

**Option B: Spectral Clustering**
- Compute Laplacian eigenvectors
- Project nodes to low-D space
- Run K-means on projections
- Requires specifying K

```python
import networkx as nx
from community import community_louvain  # pip install python-louvain

# Create graph
G = nx.Graph()
G.add_edges_from(edges)

# Detect communities
partition = community_louvain.best_partition(G)

# Analyze
n_communities = len(set(partition.values()))
modularity = community_louvain.modularity(partition, G)
print(f"Communities: {n_communities}, Modularity: {modularity:.3f}")
```

**Step 3: Evaluation**
- **Modularity Score**: Higher = better community structure
- **Analyze characteristics**: Common attributes within communities

**Step 4: Interpretation**
- Label communities by dominant attributes
- "Python Developers in London", "Marketing Professionals"

---

## Question 10: Discuss the challenges of interpretability in unsupervised learning models.

### Core Challenge
No ground truth → No way to verify if discovered patterns are "correct"

### Specific Challenges

| Challenge | Description |
|-----------|-------------|
| **Cluster Labeling** | Clusters are just "0, 1, 2..." - no inherent meaning |
| **Ambiguity** | Different algorithms/parameters → different results |
| **Black-box Features** | PCA components, autoencoder latent space have no intuitive meaning |
| **Validation Difficulty** | High silhouette ≠ business value |

### Strategies to Improve Interpretability

**1. Start Simple**
- Use K-means before GMM
- Centroid-based clusters easier to explain

**2. Feature Importance for Clusters**
```python
# Train decision tree to predict cluster labels
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X, cluster_labels)
# Inspect feature importances and rules
```

**3. Extensive Visualization**
- Box plots of features per cluster
- Parallel coordinate plots
- t-SNE/UMAP colored by cluster

**4. Domain Expert Validation**
- Have experts review cluster characteristics
- Ask: "Do these segments make business sense?"

**5. Cluster Profiling Report**
```python
# For each cluster, show:
# - Size (% of data)
# - Mean/median of key features
# - Distinguishing characteristics vs other clusters
```

### Key Takeaway
Interpretability requires **human-in-the-loop** analysis. Quantitative metrics are necessary but not sufficient - domain knowledge is essential for validation.
