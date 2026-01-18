# K Nearest Neighbors Interview Questions - Scenario_Based Questions

## Question 1

**Discuss the impact of imbalanced datasets on the K-NN algorithm.**

### Answer

**Impact:**
Imbalanced datasets severely bias K-NN toward the majority class. Since K-NN uses majority voting among K neighbors, and majority class samples are more densely distributed, they dominate the neighborhood of most query points. Result: model predicts majority class almost always, especially for minority class samples.

**Why This Happens:**
- Majority class has higher density in feature space
- More majority samples available to be "nearest"
- Minority class gets outvoted even when query is actually minority

**Example:**
- 95% Class A, 5% Class B
- K=5: Neighbors likely 5 Class A → predicts A even for true B samples

**Solutions (in order of preference):**

**1. Weighted K-NN (First Try)**
```python
# Closer minority neighbors get high weight
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
```
- Why: Close minority neighbor can outvote distant majority

**2. SMOTE (Synthetic Oversampling)**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```
- Creates synthetic minority samples

**3. Undersampling Majority**
- Randomly remove majority samples
- Risk: Losing important information

**4. Stratified K Selection**
- Use stratified CV to find optimal K
- Smaller K may help detect local minority patterns

**Evaluation:**
- Never use accuracy alone (misleading)
- Use F1-score, Precision, Recall, AUC-ROC
- Focus on minority class recall

---

## Question 2

**How would you explain the concept of locality-sensitive hashing and its relation to K-NN?**

### Answer

**What is LSH:**
Locality-Sensitive Hashing (LSH) is a technique for approximate nearest neighbor search. It uses special hash functions where similar items have high probability of same hash, while dissimilar items have low probability. Instead of searching entire dataset, you only search within matching hash buckets.

**Core Principle:**
$$P(h(x) = h(y) | x \text{ close to } y) >> P(h(x) = h(y) | x \text{ far from } y)$$

**Relation to K-NN:**

| Aspect | Exact K-NN | K-NN with LSH |
|--------|------------|---------------|
| Search | All N points | Only bucket candidates |
| Complexity | O(N) | Sub-linear (nearly O(1)) |
| Accuracy | Exact neighbors | Approximate (may miss some) |
| Scalability | Poor for large N | Excellent |

**How LSH Works:**

**Building Phase:**
1. Create multiple hash tables with LSH functions
2. Hash all training points into buckets
3. Similar points likely share buckets

**Search Phase:**
1. Hash query point
2. Retrieve candidates from matching buckets
3. Brute-force search only among candidates
4. Return K nearest from candidates

**Trade-off:**
- More hash tables → Higher recall, slower
- Fewer hash tables → Lower recall, faster
- Tunable based on accuracy requirements

**When to Use:**
- Millions+ of data points
- Real-time latency requirements
- Acceptable to miss some true neighbors

**Example Applications:**
- Large-scale image retrieval
- Document near-duplicate detection
- Recommendation candidate generation

**Key Insight:**
LSH makes K-NN practical for massive datasets by sacrificing perfect accuracy for dramatic speed improvement.

---

## Question 3

**Discuss how missing values in the dataset affect K-NN and how you would handle them.**

### Answer

**Impact of Missing Values:**
K-NN cannot compute distances when feature values are missing. Distance calculation requires all feature values. Missing values cause: (1) complete failure if not handled, (2) incorrect distances if imputed poorly, (3) biased neighbor selection.

**Why K-NN is Particularly Affected:**
- Distance metrics need all dimensions
- Cannot simply ignore missing features (changes distance scale)
- Every prediction requires complete feature vectors

**Handling Strategies (Best to Worst):**

**1. KNNImputer (Recommended)**
- Uses K-NN to impute missing values
- Finds K similar complete samples
- Imputes with mean of their values
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```
**Why Best:** Uses local data structure, preserves relationships

**2. Mean/Median Imputation**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_train)
```
**Limitation:** Ignores relationships between features

**3. Delete Rows (if very few missing)**
```python
df_clean = df.dropna()
```
**Limitation:** Loses data, may introduce bias

**4. Delete Features (if many missing)**
- Remove feature if >50% missing
- Only if feature not critical

**Pipeline Order:**
1. Impute missing values (fit on train only)
2. Scale features
3. Train K-NN

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
```

**Key Point:** Always impute before scaling, and fit imputer only on training data.

---

## Question 4

**Discuss how bootstrap aggregating (bagging) can improve the performance of K-NN.**

### Answer

**Why K-NN Benefits from Bagging:**
K-NN has high variance, especially with small K—predictions are sensitive to specific training samples and noise. Bagging reduces variance by training multiple K-NN models on different bootstrap samples and averaging their predictions.

**How Bagging Works with K-NN:**

**Process:**
1. Create N bootstrap samples (random sampling with replacement)
2. Train separate K-NN on each sample
3. For prediction: get vote from each K-NN
4. Final prediction: majority vote (classification) or average (regression)

**Why It Helps:**
- Single K-NN sensitive to noisy neighbors
- Bagged K-NN: noisy sample affects only some models
- Averaging smooths out noise effects
- More stable, robust predictions

**Implementation:**
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Single K-NN (high variance)
knn_single = KNeighborsClassifier(n_neighbors=5)

# Bagged K-NN (reduced variance)
bagged_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=5),
    n_estimators=10,      # Number of K-NN models
    max_samples=0.8,      # 80% of samples per model
    bootstrap=True,       # Sample with replacement
    n_jobs=-1
)

bagged_knn.fit(X_train_scaled, y_train)
predictions = bagged_knn.predict(X_test_scaled)
```

**Random Subspace (Feature Bagging):**
Even more effective for K-NN in high dimensions:
```python
bagged_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=5),
    n_estimators=10,
    max_features=0.7,     # 70% of features per model
    bootstrap_features=True
)
```
**Why:** Each K-NN works in lower-dimensional subspace where distances are more meaningful.

**Benefits:**
- Reduced variance → more stable predictions
- Better handling of noise and outliers
- Mitigates curse of dimensionality (with feature bagging)

**Trade-off:**
- Increased computation (N models instead of 1)
- Slightly reduced interpretability

---

## Question 5

**How would you apply the K-NN algorithm in a recommendation system?**

### Answer

**Approach: Memory-Based Collaborative Filtering**

K-NN is the foundation of collaborative filtering recommendation systems. Two approaches: User-based (find similar users, recommend what they liked) and Item-based (find similar items to what user liked).

**User-Based Collaborative Filtering:**

**Logic:**
1. Represent each user as vector of item ratings
2. Find K most similar users to target user
3. Recommend items those users rated highly

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-Item rating matrix (rows=users, cols=items)
# 0 means not rated
ratings = np.array([
    [5, 3, 0, 1, 4],  # User 0
    [4, 0, 0, 1, 5],  # User 1
    [1, 1, 0, 5, 2],  # User 2
    [0, 0, 5, 4, 0],  # User 3 (target - wants recommendation)
])

def recommend_for_user(user_id, ratings, k=2):
    # Step 1: Calculate user similarity (cosine)
    user_sim = cosine_similarity(ratings)
    
    # Step 2: Find K most similar users (exclude self)
    similar_users = np.argsort(user_sim[user_id])[-k-1:-1][::-1]
    
    # Step 3: Find items user hasn't rated
    unrated_items = np.where(ratings[user_id] == 0)[0]
    
    # Step 4: Predict rating for unrated items
    predictions = {}
    for item in unrated_items:
        # Weighted average of similar users' ratings
        weights = user_sim[user_id, similar_users]
        item_ratings = ratings[similar_users, item]
        
        # Only consider users who rated this item
        mask = item_ratings > 0
        if mask.sum() > 0:
            pred = np.dot(weights[mask], item_ratings[mask]) / weights[mask].sum()
            predictions[item] = pred
    
    # Return top recommendations
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)

recommendations = recommend_for_user(3, ratings, k=2)
print(f"Recommended items: {recommendations}")
```

**Item-Based Collaborative Filtering:**
- Find items similar to what user already liked
- More stable (item similarity changes less than user similarity)
- Used by Amazon

**Advantages of K-NN for Recommendations:**
- **Interpretable**: "Recommended because users like you also liked this"
- **No model training**: Easy to add new users/items
- **Simple**: Easy to understand and implement

**Challenges:**
- **Sparsity**: Many users rate few items
- **Cold start**: New users have no ratings
- **Scalability**: O(N) for finding neighbors

**Modern Alternative:**
For production, use Matrix Factorization or Deep Learning, but K-NN remains useful for explainability.

---

## Question 6

**Discuss the role of approximate nearest neighbor search in scaling K-NN for big data.**

### Answer

**The Scaling Problem:**
Exact K-NN has O(N×d) prediction complexity—infeasible for millions/billions of points. For 1 billion points, even millisecond-per-distance means hours per query. Approximate Nearest Neighbor (ANN) trades small accuracy loss for massive speedup.

**Why ANN is Essential:**

| Dataset Size | Exact K-NN | ANN (HNSW) |
|--------------|------------|------------|
| 1 Million | ~1 second | ~1 ms |
| 100 Million | ~100 seconds | ~5 ms |
| 1 Billion | Infeasible | ~10 ms |

**Key ANN Algorithms:**

**1. Locality-Sensitive Hashing (LSH)**
- Hash similar items to same bucket
- Search only within buckets

**2. HNSW (Hierarchical Navigable Small World)**
- Graph-based: nodes connected to similar nodes
- Multi-layer hierarchy for fast navigation
- State-of-the-art for recall vs speed

**3. IVF (Inverted File Index)**
- Cluster data into partitions
- Search only nearest partitions

**4. Product Quantization (PQ)**
- Compress vectors for memory efficiency
- Approximate distance from compressed codes

**Trade-off: Recall vs Speed**
```
Recall = (True neighbors found) / (K neighbors requested)
```
- Higher recall → slower, more accurate
- Lower recall → faster, may miss some neighbors

**Implementation with FAISS:**
```python
import faiss
import numpy as np

# Data: 1 million 128-dim vectors
n_data = 1_000_000
d = 128
data = np.random.random((n_data, d)).astype('float32')

# Build HNSW index (state-of-the-art)
index = faiss.IndexHNSWFlat(d, 32)  # 32 connections per node
index.add(data)

# Fast search
k = 10
query = np.random.random((1, d)).astype('float32')
distances, indices = index.search(query, k)  # ~1 ms for 1M points!
```

**Production Systems Using ANN:**
- Spotify (Annoy): Music recommendations
- Meta (FAISS): Image/video search
- Pinecone, Milvus: Vector databases
- OpenAI (FAISS): Embedding retrieval for RAG

**Key Insight:**
For big data K-NN, the question isn't "exact vs approximate" but "which ANN method and what recall level." Modern ANN achieves 95%+ recall with orders of magnitude speedup.

---
