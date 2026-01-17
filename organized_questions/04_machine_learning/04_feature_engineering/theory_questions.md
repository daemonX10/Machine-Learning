# Feature Engineering Interview Questions - Theory Questions

## Question 1: What is feature engineering and why is it important in machine learning?

### Answer

**Definition:**
Feature engineering is the process of using domain knowledge to extract, create, and transform features from raw data that make machine learning algorithms work more effectively. It bridges the gap between raw data and the input required by ML models.

**Why It's Important:**

| Aspect | Impact |
|--------|--------|
| **Model Performance** | Well-engineered features can improve accuracy by 10-50% |
| **Algorithm Flexibility** | Allows simpler models to capture complex patterns |
| **Interpretability** | Creates meaningful features for business insights |
| **Training Efficiency** | Reduces computational requirements |
| **Generalization** | Helps models perform well on unseen data |

**Key Components:**
1. **Feature Extraction** - Deriving features from raw data (e.g., extracting edges from images)
2. **Feature Transformation** - Modifying existing features (scaling, normalization)
3. **Feature Construction** - Creating new features from existing ones (ratios, interactions)
4. **Feature Selection** - Choosing the most relevant features

**Example:**
```python
# Raw data: transaction_date, transaction_amount
# Engineered features:
df['day_of_week'] = df['transaction_date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])
df['amount_log'] = np.log1p(df['transaction_amount'])
df['amount_per_day'] = df.groupby('user_id')['transaction_amount'].transform('mean')
```

**Real-World Impact:**
- In Kaggle competitions, feature engineering often separates top performers
- Andrew Ng: "Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering."

---

## Question 2: What is the difference between feature selection and feature extraction?

### Answer

| Aspect | Feature Selection | Feature Extraction |
|--------|------------------|-------------------|
| **Definition** | Selecting a subset of original features | Creating new features from original ones |
| **Output** | Subset of existing features | New transformed features |
| **Interpretability** | Preserves original meaning | Often loses interpretability |
| **Dimensionality** | Reduces by removing features | Reduces by combining features |
| **Information** | May lose some information | Aims to preserve information in fewer dimensions |

**Feature Selection Methods:**

| Category | Methods | Characteristics |
|----------|---------|-----------------|
| **Filter Methods** | Correlation, Chi-squared, ANOVA | Fast, model-agnostic |
| **Wrapper Methods** | RFE, Forward/Backward Selection | Model-specific, expensive |
| **Embedded Methods** | Lasso, Tree-based importance | Built into training |

**Feature Extraction Methods:**

| Method | Type | Use Case |
|--------|------|----------|
| **PCA** | Linear | Reducing correlated features |
| **LDA** | Linear | Classification with class separation |
| **t-SNE** | Non-linear | Visualization |
| **Autoencoders** | Non-linear | Deep representation learning |

**When to Use What:**
- **Feature Selection**: When interpretability is crucial (medical, financial)
- **Feature Extraction**: When prediction accuracy is priority over interpretability

---

## Question 3: What are the common challenges in feature engineering?

### Answer

**1. Missing Data**
- **Challenge:** Deciding between imputation methods
- **Solutions:** Mean/median imputation, KNN imputation, creating "is_missing" indicators

**2. High Cardinality Categorical Variables**
- **Challenge:** Too many unique values (e.g., zip codes)
- **Solutions:** Target encoding, frequency encoding, embedding layers

**3. Feature Scaling Issues**
- **Challenge:** Features on different scales affect gradient-based algorithms
- **Solutions:** Standardization, normalization, robust scaling

**4. Multicollinearity**
- **Challenge:** Highly correlated features cause unstable coefficients
- **Solutions:** VIF analysis, PCA, removing redundant features

**5. Curse of Dimensionality**
- **Challenge:** Too many features relative to samples
- **Solutions:** Feature selection, regularization, dimensionality reduction

**6. Data Leakage**
- **Challenge:** Information from future/target leaking into features
- **Solutions:** Proper train-test splits, careful feature engineering

**7. Domain Knowledge Gap**
- **Challenge:** Creating meaningful features without expertise
- **Solutions:** Collaboration with domain experts, exploratory analysis

**8. Non-Linear Relationships**
- **Challenge:** Linear features miss complex patterns
- **Solutions:** Polynomial features, binning, interaction terms

---

## Question 4: What is the difference between normalization and standardization?

### Answer

| Aspect | Normalization (Min-Max) | Standardization (Z-score) |
|--------|------------------------|---------------------------|
| **Formula** | $(x - x_{min}) / (x_{max} - x_{min})$ | $(x - \mu) / \sigma$ |
| **Output Range** | [0, 1] | No bounded range |
| **Mean** | Depends on distribution | 0 |
| **Std Dev** | Depends on distribution | 1 |
| **Outlier Sensitivity** | High | Moderate |

**When to Use Normalization:**
- Neural networks with sigmoid/tanh activations
- Image pixel values
- When bounded range is required
- K-nearest neighbors, algorithms using distance metrics

**When to Use Standardization:**
- Linear regression, logistic regression
- SVM
- PCA
- When data is approximately Gaussian
- Algorithms assuming zero-centered data

**Code Example:**
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Normalization
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)

# Standardization
standardizer = StandardScaler()
X_standardized = standardizer.fit_transform(X)
```

**Important Note:** Always fit scalers on training data only, then transform both train and test sets to prevent data leakage.

---

## Question 5: How does feature scaling affect gradient descent optimization?

### Answer

**The Problem Without Scaling:**
- Features with larger scales dominate the gradient
- Optimization takes longer, zigzagging path
- May not converge or converge to suboptimal solution

**Visual Intuition:**

| Without Scaling | With Scaling |
|-----------------|--------------|
| Elongated, elliptical contours | Circular contours |
| Zigzag path to minimum | Direct path to minimum |
| Many iterations needed | Fewer iterations |
| Learning rate hard to tune | Easier to tune |

**Mathematical Explanation:**
- Gradient: $\nabla J = X^T(X\theta - y)$
- Large feature values → Large gradients → Overshooting
- Small feature values → Small gradients → Slow progress

**Impact on Different Algorithms:**

| Algorithm | Scaling Impact |
|-----------|---------------|
| **Gradient Descent** | Critical |
| **Neural Networks** | Critical |
| **SVM** | High |
| **KNN** | High (distance-based) |
| **Decision Trees** | Not affected |
| **Random Forest** | Not affected |

**Best Practice:**
```python
# Always scale for gradient-based optimization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler
```

---

## Question 6: Explain one-hot encoding and its advantages and disadvantages.

### Answer

**Definition:**
One-hot encoding converts categorical variables into binary vectors where each category becomes a separate column with values 0 or 1.

**Example:**
```
Color    →  Color_Red  Color_Blue  Color_Green
Red          1           0           0
Blue         0           1           0
Green        0           0           1
```

**Advantages:**

| Advantage | Explanation |
|-----------|-------------|
| **No Ordinal Assumption** | Doesn't impose order on categories |
| **Works with Most Algorithms** | Compatible with linear models, neural networks |
| **Simple Implementation** | Easy to understand and implement |
| **Captures All Information** | Each category is explicitly represented |

**Disadvantages:**

| Disadvantage | Impact |
|--------------|--------|
| **High Dimensionality** | K categories → K-1 or K new columns |
| **Sparse Matrices** | Most values are 0, memory inefficient |
| **Multicollinearity** | All columns sum to 1 (use drop_first=True) |
| **Cannot Handle New Categories** | Unknown categories at test time fail |

**Implementation:**
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Pandas
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)

# Scikit-learn
encoder = OneHotEncoder(sparse=False, drop='first')
encoded = encoder.fit_transform(df[['color']])
```

**Alternatives for High Cardinality:**
- Target encoding
- Frequency encoding
- Embedding layers (deep learning)
- Feature hashing

---

## Question 7: What are the benefits of dimensionality reduction in machine learning?

### Answer

**Key Benefits:**

| Benefit | Description |
|---------|-------------|
| **Curse of Dimensionality** | Reduces sparsity in high-dimensional space |
| **Computational Efficiency** | Faster training and inference |
| **Reduced Overfitting** | Fewer features = simpler model |
| **Noise Reduction** | Removes uninformative dimensions |
| **Visualization** | Enables plotting in 2D/3D |
| **Storage** | Less memory for data storage |

**Mathematical Perspective:**
- In high dimensions, data becomes sparse
- Distance metrics become less meaningful
- Volume of hypersphere → 0 relative to hypercube

**Common Techniques:**

| Technique | Type | Best For |
|-----------|------|----------|
| **PCA** | Linear | Correlated features, preprocessing |
| **LDA** | Linear | Classification tasks |
| **t-SNE** | Non-linear | Visualization |
| **UMAP** | Non-linear | Clustering, visualization |
| **Autoencoders** | Non-linear | Complex patterns, deep learning |

**Trade-offs:**
- Reduced interpretability (especially with extraction)
- Information loss if too aggressive
- Computational cost of the reduction itself

---

## Question 8: What are filter methods in feature selection?

### Answer

**Definition:**
Filter methods select features based on their intrinsic statistical properties and relationship with the target, independent of any learning algorithm.

**Characteristics:**
- Fast and computationally efficient
- Model-agnostic
- Univariate (evaluate features independently)
- Preprocessing step before training

**Common Filter Methods:**

| Method | Feature Type | Target Type | Use Case |
|--------|--------------|-------------|----------|
| **Correlation** | Numerical | Numerical | Linear relationships |
| **Chi-Squared** | Categorical | Categorical | Independence test |
| **ANOVA F-test** | Numerical | Categorical | Classification |
| **Mutual Information** | Any | Any | Non-linear relationships |
| **Variance Threshold** | Any | N/A | Removing constant features |

**Implementation:**
```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected_mi = selector_mi.fit_transform(X, y)
```

**Advantages:**
- Very fast, scales well
- Good for initial screening
- No risk of overfitting selection

**Disadvantages:**
- Ignores feature interactions
- May select redundant features
- Univariate nature is limiting

---

## Question 9: What are wrapper methods in feature selection?

### Answer

**Definition:**
Wrapper methods use a specific machine learning model to evaluate different feature subsets, treating feature selection as a search problem.

**How They Work:**
1. Generate candidate feature subset
2. Train model on subset
3. Evaluate performance (cross-validation)
4. Search for better subsets
5. Select best-performing subset

**Common Wrapper Methods:**

| Method | Strategy | Process |
|--------|----------|---------|
| **Forward Selection** | Bottom-up | Start empty, add best feature iteratively |
| **Backward Elimination** | Top-down | Start full, remove worst feature iteratively |
| **RFE** | Recursive | Remove least important features recursively |
| **Exhaustive Search** | Brute force | Test all combinations (impractical for large p) |

**Implementation:**
```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
estimator = RandomForestClassifier(n_estimators=100)
rfe = RFE(estimator, n_features_to_select=10, step=1)
X_rfe = rfe.fit_transform(X, y)

# RFE with Cross-Validation (finds optimal number)
rfecv = RFECV(estimator, step=1, cv=5, scoring='accuracy')
X_rfecv = rfecv.fit_transform(X, y)
print(f"Optimal features: {rfecv.n_features_}")
```

**Advantages:**
- Considers feature interactions
- Tailored to specific model
- Often better performance than filter methods

**Disadvantages:**
- Computationally expensive
- Risk of overfitting the selection
- Specific to chosen model

---

## Question 10: What are embedded methods in feature selection?

### Answer

**Definition:**
Embedded methods perform feature selection as part of the model training process. The selection is an integral, "embedded" part of the algorithm.

**Key Embedded Methods:**

| Method | Mechanism | Feature Selection |
|--------|-----------|-------------------|
| **Lasso (L1)** | L1 penalty | Shrinks coefficients to exactly 0 |
| **Ridge (L2)** | L2 penalty | Shrinks but doesn't eliminate |
| **Elastic Net** | L1 + L2 | Combines both approaches |
| **Tree-based** | Split importance | Ranks by impurity decrease |

**L1 Regularization (Lasso):**
- Loss = MSE + α × Σ|βᵢ|
- Diamond-shaped constraint region
- Solutions lie at corners (sparse)

**Implementation:**
```python
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

# Lasso for feature selection
lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected_features = np.where(lasso.coef_ != 0)[0]

# Tree-based importance
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
top_features = np.argsort(importances)[::-1][:10]
```

**Comparison of Selection Methods:**

| Aspect | Filter | Wrapper | Embedded |
|--------|--------|---------|----------|
| **Speed** | Fast | Slow | Moderate |
| **Interactions** | No | Yes | Yes |
| **Overfitting Risk** | Low | High | Moderate |
| **Model Dependency** | None | High | High |

---

## Question 11: How do you use correlation analysis for feature selection?

### Answer

**Two Types of Correlation Analysis:**

**1. Feature-Target Correlation:**
- Identifies features with strong relationship to target
- Higher correlation → more predictive power

**2. Feature-Feature Correlation:**
- Identifies redundant features (multicollinearity)
- Highly correlated features provide similar information

**Implementation:**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Feature-Target Correlation
correlations = df.corrwith(df['target']).abs().sort_values(ascending=False)
top_features = correlations[1:11].index  # Top 10, excluding target

# Feature-Feature Correlation Matrix
corr_matrix = df.corr()

# Visualization
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Remove highly correlated features
def remove_correlated(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)
```

**Limitations:**
- Only captures linear relationships
- Sensitive to outliers
- Doesn't account for feature interactions

**Best Practice:**
Use correlation analysis as initial screening, follow up with more sophisticated methods.

---

## Question 12: What is Recursive Feature Elimination (RFE)?

### Answer

**Definition:**
RFE is a wrapper method that recursively removes features and builds a model on remaining features. It ranks features by importance and eliminates the least important.

**Algorithm:**
1. Train model on all features
2. Rank features by importance (coefficients or feature_importances_)
3. Remove least important feature(s)
4. Repeat until desired number of features

**Implementation:**
```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier

# Basic RFE
estimator = SVR(kernel='linear')
rfe = RFE(estimator, n_features_to_select=5, step=1)
rfe.fit(X, y)

# Get selected features
selected_mask = rfe.support_
feature_ranking = rfe.ranking_

# RFECV - automatically finds optimal number
rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=100),
    step=1,
    cv=5,
    scoring='accuracy',
    min_features_to_select=1
)
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")

# Plot performance vs number of features
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.xlabel('Number of Features')
plt.ylabel('CV Score')
```

**Advantages:**
- Considers feature interactions
- Works with any estimator with feature_importances_ or coef_

**Disadvantages:**
- Computationally expensive
- Greedy algorithm (local optima)
- Results depend on base estimator

---

## Question 13: How does regularization lead to implicit feature selection?

### Answer

**L1 Regularization (Lasso) - True Feature Selection:**

**Mathematical Explanation:**
- Loss = $\frac{1}{n}\sum(y_i - \hat{y}_i)^2 + \alpha\sum|\beta_j|$
- L1 penalty has diamond-shaped constraint region
- Optimal solution often lies at corners (coefficient = 0)

**Geometric Intuition:**
- Elliptical loss contours meet diamond constraint at corners
- Corners lie on axes → sparse solutions

**L2 Regularization (Ridge) - Feature Shrinkage:**
- Loss = $\frac{1}{n}\sum(y_i - \hat{y}_i)^2 + \alpha\sum\beta_j^2$
- Circular constraint region
- Contours meet circle tangentially → coefficients shrink but rarely reach 0

**Comparison:**

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| **Sparsity** | Yes (exact zeros) | No (small but non-zero) |
| **Feature Selection** | Automatic | No |
| **Correlated Features** | Picks one arbitrarily | Distributes weight |
| **Stability** | Less stable | More stable |

**Code Example:**
```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Lasso - automatic feature selection
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
selected = np.where(lasso.coef_ != 0)[0]
print(f"Selected {len(selected)} features")

# Elastic Net - combines L1 and L2
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

---

## Question 14: How is PCA used in feature engineering?

### Answer

**Definition:**
PCA (Principal Component Analysis) transforms features into a new coordinate system where dimensions are orthogonal and ordered by variance explained.

**How It Works:**
1. Standardize the data
2. Compute covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Sort by eigenvalue magnitude
5. Project data onto top k eigenvectors

**Mathematical Formulation:**
- Maximize variance: $\max_w w^T\Sigma w$ subject to $||w|| = 1$
- Solution: eigenvectors of covariance matrix $\Sigma$

**Implementation:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize first (critical!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} features")
print(f"Explained variance: {pca.explained_variance_ratio_.cumsum()}")

# Visualize variance explained
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
```

**Use Cases:**
- Removing multicollinearity
- Visualization (reduce to 2-3 components)
- Noise reduction
- Preprocessing for other algorithms

**Limitations:**
- Assumes linear relationships
- Loss of interpretability
- Sensitive to scaling

---

## Question 15: How are autoencoders used for feature extraction?

### Answer

**Definition:**
Autoencoders are neural networks that learn compressed representations by encoding input to a lower-dimensional latent space and reconstructing the original input.

**Architecture:**
```
Input → Encoder → Latent Space (Bottleneck) → Decoder → Output
[n]      [...]        [k << n]            [...]      [n]
```

**How They Extract Features:**
1. Train autoencoder to minimize reconstruction loss
2. Use encoder output (latent representation) as features
3. Discard decoder

**Advantages over PCA:**
- Captures non-linear relationships
- Can handle complex data types (images, text)
- More flexible architecture

**Implementation:**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Define Autoencoder
input_dim = X.shape[1]
encoding_dim = 32

# Encoder
inputs = layers.Input(shape=(input_dim,))
encoded = layers.Dense(128, activation='relu')(inputs)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

# Models
autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32)

# Extract features
X_encoded = encoder.predict(X)
```

---

## Question 16: What is t-SNE and when should you use it?

### Answer

**Definition:**
t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique primarily used for visualization of high-dimensional data.

**How It Works:**
1. Compute pairwise similarities in high-dimensional space (Gaussian)
2. Compute pairwise similarities in low-dimensional space (t-distribution)
3. Minimize KL divergence between distributions
4. Heavy tails of t-distribution prevent crowding

**Key Parameter - Perplexity:**
- Related to number of effective neighbors
- Typical range: 5-50
- Higher = more global structure

**Implementation:**
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE Visualization')
```

**When to Use:**
- Visualizing high-dimensional data
- Exploring cluster structure
- Verifying embeddings quality

**When NOT to Use:**
- Feature extraction for downstream ML (not deterministic)
- Preserving global structure (focus is on local)
- Large datasets (O(n²) complexity)

**Comparison with UMAP:**
| Aspect | t-SNE | UMAP |
|--------|-------|------|
| Speed | Slower | Faster |
| Global Structure | Weaker | Better preserved |
| Scalability | Limited | Better |
| Determinism | No | Optional |

---

## Question 17: What is LDA for feature extraction?

### Answer

**Definition:**
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that finds linear combinations of features that best separate classes.

**Objective:**
Maximize between-class variance while minimizing within-class variance:
$$J(w) = \frac{w^T S_B w}{w^T S_W w}$$

Where:
- $S_B$ = Between-class scatter matrix
- $S_W$ = Within-class scatter matrix

**Key Difference from PCA:**

| Aspect | PCA | LDA |
|--------|-----|-----|
| **Supervision** | Unsupervised | Supervised |
| **Objective** | Maximize variance | Maximize class separation |
| **Max Components** | min(n, p) | min(p, C-1) |
| **Use Case** | General reduction | Classification preprocessing |

**Implementation:**
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Explained variance ratio
print(f"Explained variance: {lda.explained_variance_ratio_}")

# Visualize
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
plt.xlabel('LD1')
plt.ylabel('LD2')
```

**Assumptions:**
- Features are normally distributed
- Classes have identical covariance matrices
- Features are statistically independent

**Best Used When:**
- Classification task with multiple classes
- Want to visualize class separation
- Preprocessing before classifier

---

## Question 18: How do you handle feature engineering for imbalanced datasets?

### Answer

**Challenges:**
- Features may be biased toward majority class
- Feature importance skewed
- Scaling affected by class distribution

**Strategies:**

**1. SMOTE (Synthetic Minority Over-sampling):**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**2. Class-Weighted Feature Selection:**
```python
from sklearn.ensemble import RandomForestClassifier

# Use class weights in model for feature importance
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

**3. Stratified Sampling in Cross-Validation:**
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

cv = StratifiedKFold(n_splits=5)
rfecv = RFECV(estimator, cv=cv, scoring='f1')
```

**4. Cost-Sensitive Feature Engineering:**
- Create features that help identify minority class
- Focus on distinguishing characteristics

**5. Anomaly Detection Features:**
- Treat minority class as anomalies
- Engineer features capturing "deviation from normal"

**Best Practices:**
- Always evaluate with appropriate metrics (F1, AUC-PR)
- Apply resampling after feature selection
- Use stratification in all cross-validation

---

## Question 19: What is the curse of dimensionality and how does it affect feature engineering?

### Answer

**Definition:**
The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces, where distance metrics become meaningless and data becomes sparse.

**Key Effects:**

| Effect | Description |
|--------|-------------|
| **Sparsity** | Data points spread thin in high dimensions |
| **Distance Meaninglessness** | All points become equidistant |
| **Volume Concentration** | Most volume is at corners of hypercube |
| **Sample Requirements** | Exponential increase in samples needed |

**Mathematical Insight:**
- In d dimensions, need $n^d$ samples for uniform coverage
- Distance to nearest neighbor → Distance to farthest neighbor as d → ∞

**Impact on Feature Engineering:**

**1. Model Performance:**
- Overfitting becomes likely
- Generalization suffers

**2. Distance-Based Algorithms:**
- KNN becomes unreliable
- Clustering quality degrades

**3. Feature Selection Importance:**
- Critical to reduce dimensions
- Focus on most informative features

**Solutions:**
```python
# 1. Feature Selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=50)

# 2. Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

# 3. Regularization
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)

# 4. Manifold Learning (if data lies on lower-dim manifold)
from sklearn.manifold import Isomap
isomap = Isomap(n_components=10)
```

---

## Question 20: What are polynomial features and when should you use them?

### Answer

**Definition:**
Polynomial features create new features by taking polynomial combinations of existing features, capturing non-linear relationships.

**Example (degree=2):**
```
[a, b] → [1, a, b, a², ab, b²]
```

**Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X)

print(f"Original features: {X.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")
print(f"Feature names: {poly.get_feature_names_out()}")
```

**When to Use:**
- Non-linear relationships exist but linear model needed
- Before linear regression to capture curvature
- When interaction effects are important

**When NOT to Use:**
- Already using non-linear models (trees, neural nets)
- High-dimensional data (explosion of features)
- Risk of overfitting is high

**Feature Count Formula:**
- For p features and degree d:
$$C(p+d, d) = \frac{(p+d)!}{p! \cdot d!}$$

**Best Practices:**
- Start with low degree (2-3)
- Combine with regularization (Ridge, Lasso)
- Use interaction_only=True for just interactions

---

## Question 21: How do you engineer features from text data?

### Answer

**Common Techniques:**

| Technique | Description | Output |
|-----------|-------------|--------|
| **Bag of Words** | Word frequency counts | Sparse matrix |
| **TF-IDF** | Term frequency × inverse doc frequency | Sparse matrix |
| **Word Embeddings** | Dense vector representations | Dense vectors |
| **BERT/Transformers** | Contextual embeddings | Dense vectors |

**Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    stop_words='english'
)
X_tfidf = tfidf.fit_transform(texts)

# Count Vectorizer (Bag of Words)
count = CountVectorizer(max_features=5000)
X_count = count.fit_transform(texts)

# Pre-trained Word Embeddings
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Sentence embedding (average of word vectors)
def get_sentence_embedding(sentence, model):
    words = sentence.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)
```

**Feature Selection for Text:**
```python
from sklearn.feature_selection import SelectKBest, chi2

# Select top features using chi-squared
selector = SelectKBest(chi2, k=1000)
X_selected = selector.fit_transform(X_tfidf, y)
```

**Modern Approach (Transformers):**
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Get embeddings
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
```

---

## Question 22: How does deep learning perform automatic feature engineering?

### Answer

**Concept:**
Deep neural networks automatically learn hierarchical feature representations from raw data during training, eliminating much of the need for manual feature engineering.

**How It Works:**
1. **Early layers**: Learn low-level features (edges, textures)
2. **Middle layers**: Combine into mid-level patterns
3. **Later layers**: Form high-level semantic concepts
4. **Final layers**: Task-specific representations

**Example - CNN for Images:**
```
Raw Pixels → Edges → Textures → Parts → Objects → Classification
```

**Advantages:**
- No manual feature engineering needed
- Learns task-specific features
- Discovers patterns humans might miss
- Transfers to new tasks (transfer learning)

**Limitations:**
- Requires large amounts of data
- Computationally expensive
- Less interpretable
- May not learn domain-specific features

**Transfer Learning:**
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Use pre-trained features
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False
```

**When Traditional FE Still Matters:**
- Tabular data with domain knowledge
- Small datasets
- Interpretability requirements
- Incorporating business rules

---

## Question 23: How do you engineer features for recommendation systems?

### Answer

**Three Categories of Features:**

**1. User Features:**
```python
user_features = {
    'user_age': 28,
    'user_gender': 'M',
    'user_location': 'NYC',
    'avg_rating_given': 3.8,
    'num_ratings': 150,
    'favorite_genres': ['action', 'comedy'],
    'user_embedding': [0.1, 0.3, ...]  # From collaborative filtering
}
```

**2. Item Features:**
```python
item_features = {
    'item_category': 'electronics',
    'item_price': 299.99,
    'item_avg_rating': 4.2,
    'num_reviews': 1500,
    'description_embedding': [0.2, 0.4, ...],  # From NLP
    'item_popularity': 0.85
}
```

**3. Interaction Features (Most Powerful):**
```python
interaction_features = {
    'user_item_similarity': 0.72,
    'user_category_affinity': 0.8,
    'days_since_last_interaction': 5,
    'num_previous_purchases': 3,
    'context_time_of_day': 'evening',
    'context_device': 'mobile'
}
```

**Feature Engineering Pipeline:**
```python
# Collaborative filtering embeddings
from sklearn.decomposition import TruncatedSVD

# Create user-item matrix
user_item_matrix = create_user_item_matrix(interactions)

# Get embeddings
svd = TruncatedSVD(n_components=50)
user_embeddings = svd.fit_transform(user_item_matrix)
item_embeddings = svd.components_.T

# Interaction features
def create_interaction_features(user_id, item_id, df):
    features = {}
    features['user_item_sim'] = cosine_similarity(
        user_embeddings[user_id], item_embeddings[item_id]
    )
    features['user_avg_rating_for_category'] = df[
        (df['user_id'] == user_id) & 
        (df['category'] == item_category)
    ]['rating'].mean()
    return features
```

---

## Question 24: Describe feature engineering for customer churn prediction.

### Answer

**Feature Categories:**

**1. Static/Demographic Features:**
```python
static_features = [
    'age', 'gender', 'location', 'acquisition_channel',
    'subscription_tier', 'payment_method', 'signup_date'
]
```

**2. Behavioral Features (Time-Windowed):**
```python
def create_behavioral_features(df, window_days):
    features = {
        'login_frequency_30d': count_logins(df, 30),
        'login_frequency_90d': count_logins(df, 90),
        'time_since_last_login': days_since_last_login(df),
        'feature_usage_counts': count_feature_usage(df),
        'session_duration_avg': avg_session_duration(df),
        'pages_viewed_per_session': avg_pages_per_session(df)
    }
    return features
```

**3. Trend Features (Critical for Churn):**
```python
def create_trend_features(df):
    return {
        'usage_trend_30d_vs_90d': (
            usage_30d - usage_90d_avg
        ) / usage_90d_avg,
        'activity_slope': calculate_activity_slope(df),
        'engagement_decay_rate': calculate_decay_rate(df)
    }
```

**4. Support/Billing Features:**
```python
support_features = {
    'num_support_tickets_90d': count_tickets(df, 90),
    'avg_satisfaction_score': avg_csat(df),
    'num_failed_payments': count_failed_payments(df),
    'days_since_last_complaint': days_since_complaint(df)
}
```

**Complete Pipeline:**
```python
def engineer_churn_features(df, snapshot_date):
    features = {}
    
    # Static
    features.update(get_static_features(df))
    
    # Behavioral (multiple windows)
    for window in [7, 30, 90]:
        features.update(create_behavioral_features(df, window))
    
    # Trends
    features.update(create_trend_features(df))
    
    # Support
    features.update(get_support_features(df))
    
    # Preprocessing
    features_df = pd.DataFrame([features])
    features_df = pd.get_dummies(features_df, columns=categorical_cols)
    features_df = StandardScaler().fit_transform(features_df)
    
    return features_df
```

---

## Question 25: What is feature selection vs feature extraction? (Detailed)

### Answer

**Comprehensive Comparison:**

| Dimension | Feature Selection | Feature Extraction |
|-----------|------------------|-------------------|
| **Definition** | Select subset of original features | Create new features from original |
| **Output** | Original features (subset) | Transformed features |
| **Dimensionality** | d' ≤ d (subset) | d' < d (projection) |
| **Interpretability** | Preserved | Usually lost |
| **Information** | Discards some features | Combines all features |
| **Reversibility** | Original features recoverable | Original features not directly recoverable |

**Feature Selection Categories:**

```python
# 1. Filter Methods
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=10)

# 2. Wrapper Methods
from sklearn.feature_selection import RFE
rfe = RFE(estimator, n_features_to_select=10)

# 3. Embedded Methods
from sklearn.linear_model import LassoCV
lasso = LassoCV()  # Features with non-zero coefficients
```

**Feature Extraction Methods:**

```python
# 1. Linear Methods
from sklearn.decomposition import PCA
pca = PCA(n_components=10)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)

# 2. Non-linear Methods
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)

# Autoencoder (neural network)
encoder = build_encoder_model(input_dim, latent_dim)
```

**Decision Framework:**

| Scenario | Recommendation |
|----------|---------------|
| Need interpretability | Feature Selection |
| Regulatory requirements | Feature Selection |
| Complex non-linear patterns | Feature Extraction |
| Very high dimensionality | Feature Extraction (then Selection) |
| Mixed data types | Feature Selection |
| Image/Text/Audio | Feature Extraction (deep learning) |

---

## Question 26: What are the main categories of feature selection methods?

### Answer

**Three Main Categories:**

| Category | Approach | Speed | Interaction Awareness |
|----------|----------|-------|----------------------|
| **Filter** | Statistical measures | Fast | No |
| **Wrapper** | Model-based search | Slow | Yes |
| **Embedded** | Built into training | Moderate | Yes |

**1. Filter Methods:**
```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif
)

# ANOVA F-test (numerical features, categorical target)
selector = SelectKBest(f_classif, k=10)

# Chi-squared (categorical features, categorical target)
selector = SelectKBest(chi2, k=10)

# Mutual Information (any feature type)
selector = SelectKBest(mutual_info_classif, k=10)

# Correlation-based
correlations = df.corrwith(df['target']).abs()
top_features = correlations.nlargest(10).index
```

**2. Wrapper Methods:**
```python
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector

# Recursive Feature Elimination
rfe = RFE(estimator, n_features_to_select=10)

# Forward Selection
sfs = SequentialFeatureSelector(estimator, direction='forward', n_features_to_select=10)

# Backward Selection
sbs = SequentialFeatureSelector(estimator, direction='backward', n_features_to_select=10)
```

**3. Embedded Methods:**
```python
# L1 Regularization
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected = X.columns[lasso.coef_ != 0]

# Tree-based
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(10).index
```

---

## Question 27: How do filter methods work and their pros/cons?

### Answer

**Process:**
1. Score each feature independently using statistical metric
2. Rank features by score
3. Select top k features or apply threshold

**Statistical Tests by Data Type:**

| Feature Type | Target Type | Test |
|--------------|-------------|------|
| Numerical | Numerical | Pearson Correlation |
| Numerical | Categorical | ANOVA F-test |
| Categorical | Categorical | Chi-Squared, Mutual Info |
| Any | Any | Mutual Information |

**Implementation:**
```python
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr

# Select K best
selector = SelectKBest(score_func=f_classif, k=10)

# Select top percentile
selector = SelectPercentile(score_func=f_classif, percentile=20)

# Select by false positive rate
selector = SelectFpr(score_func=f_classif, alpha=0.05)

# Fit and transform
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
```

**Advantages:**
- ⚡ Very fast (O(n×p) complexity)
- 🔄 Model-agnostic
- ✅ No overfitting risk in selection
- 📊 Good for initial screening

**Disadvantages:**
- ❌ Ignores feature interactions
- ❌ May select redundant features
- ❌ Univariate - evaluates features independently
- ❌ May miss features useful in combination

---

## Question 28: When to choose wrapper methods over others?

### Answer

**Choose Wrapper Methods When:**

| Scenario | Reason |
|----------|--------|
| **Performance is priority** | They optimize for actual model performance |
| **Feature interactions matter** | They evaluate features together |
| **Dataset is small-medium** | Computational cost is acceptable |
| **Using simple model** | KNN, linear models benefit most |

**Comparison Decision Tree:**
```
Is dataset very large (>100k features)?
├── Yes → Use Filter methods first, then Embedded
└── No
    └── Is interpretability critical?
        ├── Yes → Use Filter + manual selection
        └── No
            └── Is computational budget limited?
                ├── Yes → Use Embedded methods
                └── No → Use Wrapper methods
```

**Wrapper Method Selection:**
```python
# Forward Selection - start empty, add features
# Good when you expect few features to be important
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs = SFS(estimator, k_features=10, forward=True)

# Backward Elimination - start full, remove features  
# Good when you expect most features are important
sbs = SFS(estimator, k_features=10, forward=False)

# RFE - recursive elimination
# Good general-purpose choice
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator, cv=5, scoring='accuracy')
```

**Cost vs Benefit:**
- Filter: O(n×p) - Best for p > 10,000
- Embedded: O(model training) - Best for p < 10,000
- Wrapper: O(p² × model training) - Best for p < 1,000

---

## Question 29: How do embedded methods integrate with training?

### Answer

**Integration Mechanisms:**

**1. Regularization-Based (Lasso):**
```python
# Loss function includes feature selection penalty
# L = MSE + α * Σ|β_i|

from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50))
lasso.fit(X, y)

# Features selected during training
selected_features = X.columns[lasso.coef_ != 0]
print(f"Selected {len(selected_features)} features")
```

**2. Tree-Based Methods:**
```python
# Feature importance calculated during tree construction
# Based on impurity decrease at each split

from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X, y)
importances = gb.feature_importances_

# LightGBM
lgb_model = lgb.LGBMClassifier(importance_type='gain')
lgb_model.fit(X, y)
importances = lgb_model.feature_importances_

# Select top features
top_k = 20
top_features = X.columns[np.argsort(importances)[-top_k:]]
```

**Benefits of Embedded Methods:**
- Single training pass (efficient)
- Considers feature interactions
- Inherent regularization
- Less prone to overfitting than wrappers

**Comparison of Embedded Methods:**

| Method | Sparsity | Interaction | Nonlinear |
|--------|----------|-------------|-----------|
| Lasso | Yes | No | No |
| Elastic Net | Yes | No | No |
| Random Forest | No | Yes | Yes |
| XGBoost | No | Yes | Yes |

---

## Question 30: How to handle feature selection for high-dimensional datasets?

### Answer

**Multi-Stage Funnel Approach:**

```
Stage 1: Variance Threshold (10,000 → 5,000 features)
    ↓
Stage 2: Univariate Filter (5,000 → 1,000 features)
    ↓
Stage 3: Embedded Method (1,000 → 100 features)
    ↓
Stage 4: Wrapper Method (100 → optimal features)
```

**Implementation:**
```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# Stage 1: Remove low variance features
var_threshold = VarianceThreshold(threshold=0.01)
X_var = var_threshold.fit_transform(X)

# Stage 2: Univariate filter
selector = SelectKBest(mutual_info_classif, k=1000)
X_filtered = selector.fit_transform(X_var, y)

# Stage 3: Lasso for sparse selection
lasso = LassoCV(cv=5, max_iter=10000)
lasso.fit(X_filtered, y)
important_idx = np.where(lasso.coef_ != 0)[0]
X_lasso = X_filtered[:, important_idx]

# Stage 4 (Optional): Fine-tuning with RFE
if X_lasso.shape[1] < 100:
    rfecv = RFECV(RandomForestClassifier(n_estimators=100), cv=5)
    X_final = rfecv.fit_transform(X_lasso, y)
```

**Best Practices for High Dimensions:**
1. Start with fast filter methods
2. Use sparse methods (Lasso, Elastic Net)
3. Consider dimensionality reduction (PCA) before selection
4. Use stratified cross-validation
5. Monitor for overfitting in selection process

---

## Question 31: Univariate vs multivariate feature selection?

### Answer

**Comparison:**

| Aspect | Univariate | Multivariate |
|--------|------------|--------------|
| **Evaluation** | Each feature independently | Features together |
| **Interactions** | Ignored | Captured |
| **Speed** | Fast | Slower |
| **Redundancy** | May select redundant | Handles redundancy |
| **Methods** | Filter methods | Wrapper, Embedded |

**Univariate Example:**
```python
# Evaluates each feature's correlation with target independently
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
# Problem: May select Feature_A and Feature_B even if they're redundant
```

**Multivariate Example:**
```python
# Evaluates features in context of other features
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe = RFE(RandomForestClassifier(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
# Advantage: Won't select redundant features
```

**Analogy:**
- **Univariate**: Picking basketball team by individual scoring average
- **Multivariate**: Picking team by how well players work together

**When to Use Each:**

| Use Univariate | Use Multivariate |
|----------------|------------------|
| Initial screening | Final selection |
| Very high dimensions | Moderate dimensions |
| Fast baseline | Optimal performance |
| No interaction expected | Complex interactions |

---

## Question 32: How do correlation-based methods work and limitations?

### Answer

**Process:**

**Step 1: Feature-Target Correlation**
```python
# Calculate correlation with target
correlations = df.drop('target', axis=1).corrwith(df['target'])
top_features = correlations.abs().nlargest(20).index
```

**Step 2: Feature-Feature Correlation (Remove Redundancy)**
```python
def remove_correlated_features(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)
```

**Complete Workflow:**
```python
import pandas as pd
import numpy as np

# 1. Feature-target correlation
target_corr = df.corrwith(df['target']).abs()
high_corr_features = target_corr[target_corr > 0.3].index

# 2. Feature-feature correlation matrix
subset_df = df[high_corr_features]
corr_matrix = subset_df.corr()

# 3. Remove redundant features
selected_features = []
for col in high_corr_features:
    if col == 'target':
        continue
    is_redundant = False
    for selected in selected_features:
        if abs(corr_matrix.loc[col, selected]) > 0.8:
            is_redundant = True
            break
    if not is_redundant:
        selected_features.append(col)
```

**Limitations:**
- ❌ Only captures **linear** relationships
- ❌ Ignores feature interactions
- ❌ Sensitive to outliers
- ❌ Pearson requires numerical data
- ❌ May miss non-linear but predictive features

**Better Alternatives:**
- Spearman correlation (monotonic relationships)
- Mutual information (any relationship)
- Distance correlation (captures non-linear)

---

## Question 33: How to handle redundant vs irrelevant features?

### Answer

**Definitions:**

| Type | Definition | Example |
|------|------------|---------|
| **Irrelevant** | No relationship with target | "favorite_color" for predicting income |
| **Redundant** | Highly correlated with other features | "area_sqft" and "area_sqm" |

**Handling Irrelevant Features:**
```python
# Goal: Remove features with no predictive power

# Method 1: Filter by statistical test
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k='all')
selector.fit(X, y)
scores = selector.scores_
irrelevant = X.columns[scores < threshold]

# Method 2: Embedded method (Lasso)
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X, y)
irrelevant = X.columns[lasso.coef_ == 0]
```

**Handling Redundant Features:**
```python
# Goal: Keep one representative from each correlated group

# Method 1: Correlation matrix analysis
def remove_redundant(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        if correlated:
            # Keep feature with higher target correlation
            to_drop.add(col)  # or choose based on domain knowledge
    
    return df.drop(columns=to_drop)

# Method 2: VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data['feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) 
                       for i in range(df.shape[1])]
    return vif_data

# Remove features with VIF > 10
```

**Complete Workflow:**
```
1. First, remove irrelevant features (feature-target analysis)
2. Then, remove redundant features (feature-feature analysis)
3. Validate with cross-validation
```

---

