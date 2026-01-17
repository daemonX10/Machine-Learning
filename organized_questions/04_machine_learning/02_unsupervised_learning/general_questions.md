# Unsupervised Learning - General Questions

## Question 1: Name the main types of problems addressed by unsupervised learning.

### The Four Main Problem Types

| Problem Type | Goal | Key Question | Algorithms |
|-------------|------|--------------|------------|
| **Clustering** | Group similar data points | "What natural groupings exist?" | K-means, DBSCAN, Hierarchical, GMM |
| **Dimensionality Reduction** | Reduce features while preserving info | "Can we represent this more simply?" | PCA, t-SNE, Autoencoders |
| **Association Rule Mining** | Find co-occurrence patterns | "What items occur together?" | Apriori, FP-Growth |
| **Anomaly Detection** | Identify rare, unusual points | "Does this look unusual?" | Isolation Forest, LOF, Autoencoders |

### Examples

- **Clustering**: Customer segmentation, document grouping
- **Dimensionality Reduction**: Visualization, feature engineering
- **Association Rules**: Market basket analysis, web usage mining
- **Anomaly Detection**: Fraud detection, network intrusion

---

## Question 2: How can association rule learning be applied in a market-basket analysis?

### The Process

1. **Data**: Transaction database (items bought together)
2. **Algorithm**: Apriori or FP-Growth
3. **Output**: Rules like {Diapers} → {Beer}

### Business Applications

| Application | Insight | Action |
|-------------|---------|--------|
| **Store Layout** | Diapers → Beer association | Place beer near diapers |
| **Cross-selling** | Steak → Red Wine | Recommend wine at checkout |
| **Product Bundling** | Printer, Ink, Paper | Create "Home Office Kit" |
| **Loss-Leader Strategy** | Milk appears with high-margin items | Discount milk to draw customers |

### Metrics Used
- **Support**: How frequent is the itemset?
- **Confidence**: How reliable is the rule?
- **Lift**: Is the relationship real or coincidence?

---

## Question 3: How can you determine the optimal number of clusters for a dataset?

### Three Main Methods

**1. Elbow Method**
- Plot WCSS vs K (K = 1, 2, 3...)
- Find "elbow" where decrease slows
- Limitation: Elbow can be ambiguous

**2. Silhouette Score**
- Plot silhouette score vs K
- Choose K with **highest** score
- More reliable than Elbow

**3. Gap Statistic**
- Compare observed WCSS to null reference
- Choose K maximizing the "gap"
- More rigorous, but complex

### Best Practice
Combine methods: Use Elbow for range, Silhouette to pick best K. Also consider domain knowledge (how many segments can marketing handle?).

---

## Question 4: What challenges do you face when clustering high-dimensional data?

### The Curse of Dimensionality

| Challenge | Impact |
|-----------|--------|
| **Meaningless Distances** | All points become roughly equidistant |
| **Data Sparsity** | Points spread out, hard to find dense regions |
| **Irrelevant Features** | Noise masks true cluster structure |
| **Computational Cost** | Time/memory increase exponentially |

### Solutions

1. **Dimensionality Reduction**
   - PCA before clustering
   - Autoencoders for non-linear compression

2. **Feature Selection**
   - Remove low-variance features
   - Use domain knowledge

3. **Specialized Algorithms**
   - Subspace clustering (CLIQUE)
   - Algorithms designed for high-D data

---

## Question 5: What preprocessing steps are suggested before performing unsupervised learning?

### Essential Preprocessing Pipeline

**1. Feature Scaling** (Critical for distance-based algorithms)
- Standardization: mean=0, std=1 (most common)
- Min-Max: scale to [0,1]

**2. Handle Missing Values**
- Impute: median (numerical), mode (categorical)
- Advanced: KNN imputation, iterative imputation

**3. Handle Categorical Variables**
- One-hot encoding for nominal features

**4. Dimensionality Reduction**
- PCA after scaling
- Remove low-variance or highly correlated features

**5. Handle Outliers**
- Detect and decide: remove, cap, or use robust algorithm (DBSCAN)

### Typical Pipeline for K-means
```
Missing Values → One-Hot Encode → Scale → PCA → K-means
```

---

## Question 6: How do you handle missing values in an unsupervised learning context?

### Strategies by Complexity

**1. Deletion**
- Remove rows with missing values (if < 1-2%)
- Remove columns with > 60% missing

**2. Simple Imputation**
| Feature Type | Method |
|-------------|--------|
| Numerical | Median (robust to outliers) |
| Categorical | Mode (most frequent) |

**3. Advanced Imputation**

**KNN Imputation**:
- Find k similar points based on other features
- Impute from neighbors' values
- Good for unsupervised (uses local structure)

**Iterative Imputation**:
- Treat each feature as target, predict from others
- Iterate until convergence
- Most accurate but slowest

### Recommended Workflow
1. Analyze missingness pattern
2. Start with median/mode baseline
3. Try KNN or iterative for better results
4. Always scale data after imputation

---

## Question 7: How can unsupervised learning be applied to anomaly detection?

### Core Idea
Learn what "normal" looks like → Flag points that don't fit

### Approaches

**1. Clustering-Based**
- **DBSCAN**: Points labeled "noise" are anomalies
- **K-means**: Points far from their centroid are suspicious

**2. Density-Based**
- **Local Outlier Factor (LOF)**: Compares local density to neighbors
- Low density relative to neighbors = outlier

**3. Reconstruction-Based (Autoencoders)**
- Train on normal data only
- High reconstruction error = anomaly (never seen before)

**4. Isolation-Based**
- **Isolation Forest**: Anomalies are easier to isolate
- Fewer random splits needed = more anomalous

### Why Unsupervised?
No labeled anomaly examples needed - learns "normal" from data structure.

---

## Question 8: How do unsupervised learning techniques contribute to the field of natural language processing (NLP)?

### Key Contributions

**1. Word Embeddings (Word2Vec, GloVe)**
- Learn vector representations from context
- Capture semantic relationships: king - man + woman ≈ queen

**2. Topic Modeling (LDA)**
- Discover topics in document collections
- Each document = mixture of topics

**3. Pre-trained Language Models (BERT, GPT)**

| Task | Description |
|------|-------------|
| **Masked LM (BERT)** | Predict masked words from context |
| **Causal LM (GPT)** | Predict next word |

**Impact**: 
- Train on massive unlabeled text
- Fine-tune on small labeled data
- State-of-the-art on all NLP tasks

This is the **foundation of modern NLP** - unsupervised pre-training revolutionized the field.

---

## Question 9: Design an approach to group similar documents using unsupervised learning.

### Pipeline

**Step 1: Text Preprocessing**
```python
text = text.lower()                    # Lowercase
text = remove_punctuation(text)        # Clean
tokens = tokenize(text)                # Split into words
tokens = remove_stopwords(tokens)      # Remove "the", "a", etc.
tokens = lemmatize(tokens)             # "running" → "run"
```

**Step 2: Vectorization (TF-IDF)**
- Captures word importance (frequent in doc, rare overall)
- Output: Document-term matrix

**Step 3: Dimensionality Reduction (Optional)**
- Apply LSA (SVD on TF-IDF) or PCA
- Reduce to ~100-300 dimensions

**Step 4: Clustering**
- K-means with Elbow/Silhouette for optimal K
- Or DBSCAN if K unknown

**Step 5: Interpretation**
- Analyze top TF-IDF words per cluster
- Assign topic labels: "Sports", "Politics", "Tech"

---

## Question 10: How has unsupervised learning been used in the field of reinforcement learning?

### Two Major Applications

**1. Representation Learning**
- **Problem**: High-dimensional states (e.g., raw pixels)
- **Solution**: Use autoencoders or contrastive learning to compress state space
- **Result**: RL agent learns policy on simpler representation

**Example**: CURL (Contrastive Unsupervised Representations for RL)

**2. Intrinsic Motivation for Exploration**
- **Problem**: Sparse rewards → agent never finds signal
- **Solution**: Create "curiosity" reward from unsupervised learning

**Methods**:
- **Prediction Error**: Reward for surprising (hard-to-predict) states
- **State Novelty**: Reward for visiting low-density regions
- **Skill Discovery**: Learn diverse skills unsupervised

### Impact
Makes RL more scalable and sample-efficient for complex environments.

---

## Question 11: How can you use unsupervised learning for cross-lingual or multilingual text analysis?

### Core Approach: Multilingual Embeddings
Create shared vector space where words/sentences from different languages with similar meanings are close together.

### Methods

**1. Multilingual Word Embeddings**
- Train separate embeddings per language
- Learn rotation matrix to align spaces
- "cat" (English) ≈ "gato" (Spanish) ≈ "chat" (French)

**2. Multilingual Transformers (State-of-the-Art)**
- **mBERT**, **XLM-R**: Pre-trained on 100+ languages
- Self-supervised (Masked LM) on combined multilingual corpus
- Automatically learns shared representation space

### Key Application: Zero-Shot Cross-Lingual Transfer

1. Fine-tune mBERT on English sentiment data
2. Model works on German, Spanish, French sentiment
3. **No labeled data needed in target languages**

### Other Applications
- Cross-lingual information retrieval
- Machine translation
- Multilingual document clustering
