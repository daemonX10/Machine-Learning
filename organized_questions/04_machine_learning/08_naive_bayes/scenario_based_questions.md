# Naive Bayes Interview Questions - Scenario-Based Questions

## Question 1

**How would you handle an imbalanced dataset when using a Naive Bayes classifier?**

**Answer:**

When dealing with imbalanced data (e.g., 95% normal, 5% fraud), standard NB will be biased toward the majority class due to prior probabilities. The solution involves multiple strategies: adjusting class priors, resampling the data, adjusting the decision threshold, or using appropriate evaluation metrics.

**Scenario Analysis:**

Suppose you have:
- 9,500 legitimate transactions (95%)
- 500 fraudulent transactions (5%)

**Problem:** NB learns P(fraud) = 0.05, biasing predictions toward legitimate.

**Solution Strategies:**

**Strategy 1: Adjust Class Priors**
```python
from sklearn.naive_bayes import MultinomialNB

# Option A: Use uniform priors
nb = MultinomialNB(fit_prior=False)  # Treats P(fraud) = P(legitimate) = 0.5

# Option B: Manually set priors
nb = MultinomialNB(class_prior=[0.5, 0.5])  # Balance the priors
```

**Strategy 2: Resample the Data**
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Oversample minority (fraud)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Or undersample majority
rus = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = rus.fit_resample(X, y)
```

**Strategy 3: Adjust Decision Threshold**
```python
# Get probabilities instead of hard predictions
probs = nb.predict_proba(X_test)[:, 1]  # P(fraud)

# Lower threshold to catch more fraud (increase recall)
threshold = 0.3  # Instead of default 0.5
predictions = (probs > threshold).astype(int)
```

**Strategy 4: Use Appropriate Metrics**
```python
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

# Don't use accuracy! Use these instead:
print(f"F1 Score: {f1_score(y_test, y_pred)}")
print(f"ROC-AUC: {roc_auc_score(y_test, probs)}")

# Find optimal threshold using precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
```

**Recommended Approach for This Scenario:**
1. Start with balanced priors (`fit_prior=False`)
2. Use SMOTE if you have enough minority samples
3. Tune threshold based on business requirements (precision vs recall trade-off)
4. Evaluate with F1-score and ROC-AUC, not accuracy

---

## Question 2

**Discuss the impact of feature scaling on Naive Bayes classifiers.**

**Answer:**

Feature scaling (normalization, standardization) generally has **no impact on Naive Bayes** classification results, unlike many other algorithms. This is because NB computes probabilities based on feature distributions independently - scaling doesn't change the relative likelihood calculations.

**Scenario Analysis:**

Consider features with different scales:
- Feature 1: Age (20-80)
- Feature 2: Income (20,000-200,000)

**Why Scaling Doesn't Matter for NB:**

**For Gaussian NB:**
```
Original:     P(income=50000 | class) uses μ=60000, σ=20000
Standardized: P(z=-0.5 | class) uses μ=0, σ=1

Both give the same relative likelihood ranking because:
- The Gaussian PDF shape is preserved
- Only the parameters change, not the relative probabilities
```

**Mathematical Proof:**

For Gaussian NB, standardizing feature x to z = (x-μ)/σ:

Original: $P(x|c) = \frac{1}{\sigma_c\sqrt{2\pi}} e^{-\frac{(x-\mu_c)^2}{2\sigma_c^2}}$

Standardized: $P(z|c) = \frac{1}{\sqrt{2\pi}} e^{-\frac{(z-\mu_c')^2}{2}}$

The likelihood ratio between classes remains the same!

**When Scaling Might Matter:**

| Variant | Impact of Scaling |
|---------|------------------|
| Gaussian NB | None - ranking unchanged |
| Multinomial NB | N/A - expects counts, not scaled values |
| Bernoulli NB | N/A - expects binary values |

**Practical Verification:**
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Without scaling
gnb = GaussianNB()
gnb.fit(X_train, y_train)
acc_original = accuracy_score(y_test, gnb.predict(X_test))

# With scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_scaled = GaussianNB()
gnb_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, gnb_scaled.predict(X_test_scaled))

print(f"Original: {acc_original:.4f}")
print(f"Scaled: {acc_scaled:.4f}")
# Results will be identical!
```

**Key Takeaway:**
Unlike SVM, KNN, or Neural Networks, you don't need to scale features for Naive Bayes. Focus on other preprocessing (handling missing values, smoothing) instead.

---

## Question 3

**How can overfitting occur in Naive Bayes, and how would you prevent it?**

**Answer:**

While Naive Bayes is generally resistant to overfitting due to its strong simplicity bias, it can still overfit in specific scenarios: (1) when training data is very small, (2) when vocabulary is huge relative to data (text), (3) when smoothing is too low.

**Scenario Analysis:**

**When NB Overfits:**

1. **Rare Features Dominate:**
   - A word appears once in training, only in spam
   - Without smoothing: P(word|spam) = 1.0, P(word|ham) = 0.0
   - Model becomes overly confident on rare patterns

2. **Very Small Training Set:**
   - With only 100 samples and 10,000 features
   - Estimated probabilities are unreliable

3. **Insufficient Smoothing:**
   - Low alpha makes model memorize training patterns

**Prevention Strategies:**

**Strategy 1: Increase Smoothing (alpha)**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# Find optimal smoothing parameter
param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print(f"Best alpha: {grid_search.best_params_['alpha']}")
```

**Strategy 2: Feature Selection (Reduce Vocabulary)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# Limit vocabulary size
vectorizer = TfidfVectorizer(
    max_features=5000,  # Cap vocabulary
    min_df=5,           # Ignore rare words
    max_df=0.95         # Ignore too common words
)

# Or use chi-square selection
selector = SelectKBest(chi2, k=1000)
X_selected = selector.fit_transform(X_tfidf, y)
```

**Strategy 3: Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

# Use CV to detect overfitting
train_score = nb.score(X_train, y_train)
cv_scores = cross_val_score(nb, X_train, y_train, cv=5)

print(f"Train: {train_score:.4f}")
print(f"CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Large gap indicates overfitting
if train_score - cv_scores.mean() > 0.1:
    print("Warning: Potential overfitting detected")
```

**Strategy 4: Use Prior Regularization**
```python
# For imbalanced data, uniform priors prevent overfitting to majority
nb = MultinomialNB(fit_prior=False)
```

**Best Practices:**
- Start with alpha=1.0 (Laplace smoothing)
- Use min_df to remove rare features
- Validate with cross-validation
- Prefer simpler feature sets over large vocabularies

---

## Question 4

**Discuss how the Naive Bayes classifier can be applied to recommendation systems.**

**Answer:**

Naive Bayes can power content-based and collaborative filtering recommendations by predicting whether a user will like an item based on features (content-based) or similar users' preferences (collaborative). It's fast, handles sparse data well, and provides probability scores for ranking recommendations.

**Scenario: Movie Recommendation System**

**Approach 1: Content-Based Filtering**
```
Goal: Predict if User A will like Movie X based on movie features

Features of Movie X: [Action=1, Comedy=0, Year=2020, Rating=8.5]
User A's preference: Learned P(Like | feature) from their history

P(Like | Movie X) = P(Like) × ∏ P(feature | Like)
```

**Implementation:**
```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# Movie features (one-hot encoded genres)
# Columns: Action, Comedy, Drama, Sci-Fi, Romance
movie_features = np.array([
    [1, 0, 0, 1, 0],  # Movie 1: Action, Sci-Fi
    [0, 1, 1, 0, 0],  # Movie 2: Comedy, Drama
    [1, 0, 0, 0, 0],  # Movie 3: Action
    [0, 0, 1, 0, 1],  # Movie 4: Drama, Romance
])

# User ratings (1=liked, 0=not liked)
user_ratings = np.array([1, 0, 1, 0])

# Train NB on user's preferences
nb = MultinomialNB()
nb.fit(movie_features, user_ratings)

# Recommend: predict for new movies
new_movies = np.array([
    [1, 0, 0, 1, 0],  # Action, Sci-Fi - should recommend
    [0, 1, 0, 0, 1],  # Comedy, Romance - probably not
])

# Get like probabilities
probs = nb.predict_proba(new_movies)[:, 1]  # P(Like)
print("Recommendation scores:", probs)
```

**Approach 2: Collaborative Filtering (User-Based)**
```python
def collaborative_nb(user_item_matrix, target_user, item):
    """
    Predict if target_user will like item based on similar users.
    
    Features: Other items the user rated
    Target: Rating for the specific item
    """
    # Users who rated this item
    users_rated = user_item_matrix[:, item] != 0
    
    if users_rated.sum() < 5:
        return 0.5  # Not enough data
    
    # Features: ratings on other items
    X = user_item_matrix[users_rated][:, np.arange(user_item_matrix.shape[1]) != item]
    y = (user_item_matrix[users_rated, item] > 3).astype(int)  # Liked if rating > 3
    
    nb = MultinomialNB()
    nb.fit(X, y)
    
    # Predict for target user
    target_features = user_item_matrix[target_user:target_user+1, np.arange(user_item_matrix.shape[1]) != item]
    prob_like = nb.predict_proba(target_features)[0, 1]
    
    return prob_like
```

**Advantages of NB for Recommendations:**
- Fast training and prediction (real-time recommendations)
- Handles sparse user-item matrices
- Provides probability scores for ranking
- Works with limited user history

**Limitations:**
- Independence assumption may not capture complex preferences
- Doesn't model user-item interactions well
- Modern systems prefer deep learning or matrix factorization

---

## Question 5

**How would you use Naive Bayes to build an email categorization system (e.g., important, social, promotions)?**

**Answer:**

Build a multi-class text classifier that categorizes emails into folders like Gmail's Primary, Social, Promotions, Updates. Use Multinomial NB with TF-IDF features extracted from email subject, body, and metadata. Train on user-labeled emails to learn category-specific word patterns.

**System Design:**

```
Email → Preprocessing → Feature Extraction → NB Classifier → Category
                              ↓
              [Subject, Body, Sender, Headers]
                              ↓
              [TF-IDF, Metadata Features]
```

**Implementation:**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from email text and metadata."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, emails):
        # Combine subject and body
        texts = [f"{email['subject']} {email['body']}" for email in emails]
        return texts

class MetadataFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract metadata features (sender domain, etc.)."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, emails):
        features = []
        for email in emails:
            sender_domain = email['sender'].split('@')[-1] if '@' in email['sender'] else 'unknown'
            features.append(f"domain_{sender_domain}")
        return features


def build_email_classifier():
    """
    Build email categorization pipeline.
    
    Categories:
    - 0: Primary (important)
    - 1: Social (social networks)
    - 2: Promotions (marketing)
    - 3: Updates (notifications, receipts)
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('extract', EmailFeatureExtractor()),
                ('tfidf', TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    stop_words='english'
                ))
            ])),
            ('metadata', Pipeline([
                ('extract', MetadataFeatureExtractor()),
                ('tfidf', TfidfVectorizer())  # Vectorize domain features
            ]))
        ])),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    return pipeline


# Example usage
if __name__ == "__main__":
    # Sample training data
    emails = [
        {'subject': 'Meeting tomorrow', 'body': 'Please join the project meeting', 'sender': 'boss@company.com'},
        {'subject': 'John liked your post', 'body': 'See what John commented', 'sender': 'notify@facebook.com'},
        {'subject': '50% OFF Sale!', 'body': 'Limited time offer on all items', 'sender': 'deals@shop.com'},
        {'subject': 'Your order shipped', 'body': 'Track your package here', 'sender': 'orders@amazon.com'},
        {'subject': 'Quarterly report', 'body': 'Attached is the Q3 report', 'sender': 'cfo@company.com'},
        {'subject': 'New follower', 'body': 'Jane started following you', 'sender': 'notify@twitter.com'},
    ]
    labels = [0, 1, 2, 3, 0, 1]  # Primary, Social, Promotion, Update
    
    # Build and train
    classifier = build_email_classifier()
    classifier.fit(emails, labels)
    
    # Classify new email
    new_email = [{'subject': 'Flash Sale Today!', 'body': 'Buy now and save big', 'sender': 'promo@store.com'}]
    prediction = classifier.predict(new_email)
    
    categories = ['Primary', 'Social', 'Promotions', 'Updates']
    print(f"Category: {categories[prediction[0]]}")
```

**Key Features to Include:**
- Subject line keywords
- Body text patterns
- Sender domain (facebook.com → Social)
- Presence of unsubscribe links (Promotions)
- Keywords: "sale", "offer", "liked", "commented"

**Continuous Improvement:**
- Collect user corrections (moved emails)
- Retrain periodically with new labels
- A/B test against rule-based system

---

## Question 6

**Propose a strategy for using Naive Bayes in a real-time bidding system for online advertising.**

**Answer:**

In Real-Time Bidding (RTB), the ad system must decide within milliseconds whether to bid on an ad impression and how much to bid. Naive Bayes predicts click probability P(click | user, context) extremely fast, making it ideal for this latency-critical application.

**Scenario: Ad Click Prediction**

```
User visits webpage → Ad Request → NB Predicts P(click) → Bid = P(click) × Value
                                          ↓
              Features: [User demographics, Page context, Ad features, Time]
```

**System Architecture:**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Ad Request │ →  │ Feature     │ →  │ NB Model    │
│  (10-100ms) │    │ Extraction  │    │ Inference   │
└─────────────┘    └─────────────┘    └─────────────┘
                                             ↓
                                      P(click) × CPC
                                             ↓
                                      ┌─────────────┐
                                      │ Bid/No-Bid  │
                                      │ Decision    │
                                      └─────────────┘
```

**Implementation Strategy:**

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import time

class RTBClickPredictor:
    """
    Real-time bidding click prediction using Naive Bayes.
    
    Requirements:
    - Prediction latency < 5ms
    - Handle 100K+ requests/second
    - Update model periodically
    """
    
    def __init__(self):
        self.model = MultinomialNB(alpha=0.01)
        self.feature_map = {}  # Feature name to index
        self.n_features = 0
    
    def prepare_features(self, request):
        """
        Extract features from bid request.
        Must be fast - avoid complex processing.
        """
        features = np.zeros(self.n_features)
        
        # User features (pre-computed, stored in cookie/user_id lookup)
        user_segment = request.get('user_segment', 'unknown')
        if f'user_{user_segment}' in self.feature_map:
            features[self.feature_map[f'user_{user_segment}']] = 1
        
        # Context features
        device = request.get('device', 'desktop')
        features[self.feature_map.get(f'device_{device}', 0)] = 1
        
        hour = request.get('hour', 12)
        features[self.feature_map.get(f'hour_{hour}', 0)] = 1
        
        # Page category
        page_cat = request.get('page_category', 'other')
        features[self.feature_map.get(f'page_{page_cat}', 0)] = 1
        
        return features.reshape(1, -1)
    
    def predict_click_probability(self, request):
        """
        Predict P(click) for given ad request.
        Must complete in < 5ms.
        """
        start = time.time()
        
        features = self.prepare_features(request)
        prob_click = self.model.predict_proba(features)[0, 1]
        
        latency = (time.time() - start) * 1000
        if latency > 5:
            print(f"Warning: Latency exceeded {latency:.2f}ms")
        
        return prob_click
    
    def calculate_bid(self, request, cpc_value=1.0):
        """
        Calculate bid amount.
        
        Bid = P(click) × Expected value per click
        """
        p_click = self.predict_click_probability(request)
        
        # Apply bid floor and ceiling
        bid = p_click * cpc_value
        bid = max(0.01, min(bid, cpc_value))  # Floor: $0.01, Ceiling: CPC
        
        return bid, p_click
    
    def batch_train(self, impressions, clicks):
        """
        Periodic model update (offline).
        Train on collected impression data.
        """
        self.model.fit(impressions, clicks)


# Example bid request
request = {
    'user_segment': 'tech_enthusiast',
    'device': 'mobile',
    'hour': 14,
    'page_category': 'news',
    'ad_id': 'ad_12345'
}

# Latency test
predictor = RTBClickPredictor()
# ... (after training)
# bid, prob = predictor.calculate_bid(request, cpc_value=0.50)
```

**Why NB is Ideal for RTB:**

| Requirement | NB Advantage |
|-------------|--------------|
| < 5ms latency | O(d) prediction, no matrix ops |
| High throughput | Simple multiplication, vectorized |
| Sparse features | Works natively with sparse matrices |
| Online learning | partial_fit for incremental updates |
| Interpretability | Feature contributions visible |

**Production Considerations:**
1. Pre-compute feature vectors where possible
2. Use sparse matrices for memory efficiency
3. Periodic batch retraining (hourly/daily)
4. A/B test against more complex models (XGBoost, Deep Learning)
5. Calibrate probabilities for accurate bidding

---

## Question 7

**Discuss improvements over standard Naive Bayes for dealing with highly correlated features.**

**Answer:**

Standard NB assumes feature independence, which fails with correlated features (e.g., "machine" and "learning" often appear together). This causes NB to double-count evidence. Solutions include: Tree-Augmented NB (TAN), feature clustering, PCA preprocessing, or Complement NB.

**Problem Demonstration:**

```
Features: "machine", "learning" (highly correlated in ML documents)

Standard NB treats them independently:
P(ML_topic | "machine learning") ∝ P("machine"|ML) × P("learning"|ML)

But "machine" and "learning" co-occur, so evidence is counted twice.
This leads to overconfident predictions.
```

**Solution 1: Tree-Augmented Naive Bayes (TAN)**

TAN allows one dependency per feature, forming a tree structure:

```
        Class
       /  |  \
      /   |   \
   F1 ←── F2 ──→ F3
   
F2 depends on both Class and F1 (one parent feature allowed)
```

```python
# Simplified TAN concept (not full implementation)
# Model: P(F1|C) × P(F2|C, F1) × P(F3|C, F2)

def tan_likelihood(features, c, dependencies):
    """
    Compute TAN likelihood with feature dependencies.
    """
    prob = 1.0
    for i, f in enumerate(features):
        if dependencies[i] is None:
            # No parent feature, standard NB
            prob *= P(f | c)
        else:
            # Has parent feature
            parent_idx = dependencies[i]
            prob *= P(f | c, features[parent_idx])
    return prob
```

**Solution 2: Feature Clustering**

Group correlated features, use one representative per group:

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr

def cluster_correlated_features(X, threshold=0.7):
    """
    Cluster features based on correlation, keep one per cluster.
    """
    # Compute correlation matrix
    n_features = X.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            corr, _ = spearmanr(X[:, i], X[:, j])
            corr_matrix[i, j] = abs(corr)
    
    # Cluster features
    distance = 1 - corr_matrix
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-threshold,
        affinity='precomputed',
        linkage='average'
    )
    clustering.fit(distance)
    
    # Select representative from each cluster (highest variance)
    selected_features = []
    for cluster_id in np.unique(clustering.labels_):
        cluster_features = np.where(clustering.labels_ == cluster_id)[0]
        variances = X[:, cluster_features].var(axis=0)
        best_feature = cluster_features[np.argmax(variances)]
        selected_features.append(best_feature)
    
    return selected_features
```

**Solution 3: PCA Preprocessing**

Transform features to uncorrelated principal components:

```python
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

# PCA creates uncorrelated features
pipeline = Pipeline([
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('nb', GaussianNB())
])

# Now NB independence assumption holds better
pipeline.fit(X_train, y_train)
```

**Solution 4: Complement Naive Bayes**

Uses complement class to compute weights, reducing correlation impact:

```python
from sklearn.naive_bayes import ComplementNB

# ComplementNB is more robust to feature correlations
# Especially good for imbalanced text classification
cnb = ComplementNB(alpha=1.0, norm=True)
cnb.fit(X_train, y_train)
```

**Comparison:**

| Method | Pros | Cons |
|--------|------|------|
| Standard NB | Fast, simple | Overconfident with correlations |
| TAN | Models some dependencies | More complex, slower |
| Feature Clustering | Keeps NB simplicity | May lose information |
| PCA + NB | Guarantees independence | Loses interpretability |
| Complement NB | Robust, easy to use | Still assumes some independence |

**Recommendation:**
- Try Complement NB first (easy drop-in replacement)
- Use feature selection/clustering for highly redundant features
- Consider PCA if interpretability isn't crucial

---
