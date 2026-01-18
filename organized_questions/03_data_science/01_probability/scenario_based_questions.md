# Probability Interview Questions - Scenario_Based Questions

## Question 1

**How would you use probability to design a recommendation system model?**

---

### 1. Scenario Understanding
- User-item rating matrix R with many missing entries
- Goal: Predict missing ratings to recommend items

### 2. Approach: Probabilistic Matrix Factorization (PMF)

| Component | Probabilistic Model |
|-----------|---------------------|
| **Rating** | $R_{ij} \sim \mathcal{N}(U_i^T V_j, \sigma^2)$ |
| **User Prior** | $U_i \sim \mathcal{N}(0, \sigma_U^2 I)$ |
| **Item Prior** | $V_j \sim \mathcal{N}(0, \sigma_V^2 I)$ |

- $U_i$: Latent user preference vector
- $V_j$: Latent item feature vector
- Priors act as regularization

### 3. Algorithm Steps (Byheart)
```
1. Initialize U, V randomly
2. Optimize (SGD or ALS):
   - Maximize likelihood of observed ratings
   - Regularize via Gaussian priors
3. Predict: rating_ij = dot(U_i, V_j)
4. Recommend: Sort items by predicted rating
```

### 4. Why Probabilistic?

| Benefit | Explanation |
|---------|-------------|
| **Sparsity** | Priors help with few observations |
| **Uncertainty** | Can estimate confidence in predictions |
| **Cold Start** | New users/items start at prior mean |
| **Regularization** | Built-in via Gaussian priors |

### 5. Interview Tip
Mention extensions: Bayesian PMF (full posterior), Neural Collaborative Filtering (combines with deep learning)

---

## Question 2

**Propose a method for predicting customer churn using probabilistic models.**

---

### 1. Scenario Understanding
- Predict P(Churn = 1 | Customer Features)
- Need actionable probability score
- Must balance precision vs recall

### 2. Recommended Approach: Logistic Regression

**Why Logistic Regression?**
- Outputs calibrated probability P(churn)
- Interpretable coefficients (odds ratios)
- Simple, fast, reliable baseline

### 3. Feature Engineering

| Category | Features |
|----------|----------|
| **Usage** | Login frequency, session duration |
| **Support** | Ticket count, resolution time |
| **Billing** | Plan type, payment failures |
| **Tenure** | Account age, engagement trend |

### 4. Model Output
$$P(\text{churn}) = \sigma(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)$$

Score: 0 to 1 (e.g., 0.85 = 85% churn risk)

### 5. Decision Logic
```python
# Cost analysis: C_FN >> C_FP → lower threshold
threshold = 0.3  # More conservative: catch more churners

high_risk = customers[churn_prob > threshold]
# Trigger intervention: discount, outreach, etc.
```

### 6. Advanced Alternative: Survival Analysis
- Cox Proportional Hazards model
- Outputs: S(t) = P(customer survives to time t)
- Handles censoring (customers who haven't churned yet)

### 7. Interview Tip
Emphasize: threshold depends on business costs (intervention cost vs customer lifetime value)

---

## Question 3

**Discuss how you would use probabilities to detect anomalies in transaction data.**

---

### 1. Scenario Understanding
- Goal: Flag unusual transactions as potential fraud
- Challenge: Anomalies are rare, normal behavior varies

### 2. Approach: Density-Based Anomaly Detection

**Core Idea**: Model "normal" → flag low-probability events

### 3. Feature Selection

| Feature | Why |
|---------|-----|
| Transaction amount | Unusual amounts |
| Time of day | Odd hours |
| Frequency | Burst activity |
| Location deviation | Geographic anomaly |
| Merchant category | Unusual spending pattern |

### 4. Model Normal Behavior

**Option A: Multivariate Gaussian**
$$p(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

**Option B: Gaussian Mixture Model**
For multi-modal patterns (weekday vs weekend)

### 5. Algorithm Steps
```
1. Fit μ, Σ on NORMAL transactions only
2. For new transaction x:
   - Compute p(x) using fitted model
3. If p(x) < threshold → ANOMALY
4. Threshold = 1st percentile of training densities
```

### 6. Code (Writable)
```python
from scipy.stats import multivariate_normal
import numpy as np

# Fit on normal transactions
mu = np.mean(normal_transactions, axis=0)
sigma = np.cov(normal_transactions.T)

# Score new transactions
scores = multivariate_normal.pdf(new_transactions, mean=mu, cov=sigma)

# Flag bottom 1%
threshold = np.percentile(scores, 1)
anomalies = new_transactions[scores < threshold]
```

### 7. Interview Tips
- Log-transform skewed features (amounts)
- Retrain periodically (concept drift)
- Alternative: Autoencoders (high reconstruction error = anomaly)

---

