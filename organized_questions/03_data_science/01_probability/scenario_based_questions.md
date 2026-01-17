# Probability Interview Questions - Scenario_Based Questions

## Question 1

**How would you use probability to design a recommendation system model?**

**Answer:**

### Approach: Probabilistic Matrix Factorization (PMF)

**The Setup**
- User-item rating matrix R with many missing entries
- Goal: Predict missing ratings to recommend items

### Probabilistic Model

**1. Rating Generation (Likelihood)**
$$R_{ij} \sim \mathcal{N}(U_i^T V_j, \sigma^2)$$

Where:
- $U_i$: Latent user preference vector
- $V_j$: Latent item feature vector
- $U_i^T V_j$: Predicted rating (dot product)

**2. Priors (Regularization)**
$$U_i \sim \mathcal{N}(0, \sigma_U^2 I)$$
$$V_j \sim \mathcal{N}(0, \sigma_V^2 I)$$

Prevents overfitting, handles sparse data.

**3. Objective: Maximum A Posteriori (MAP)**
$$\arg\max_{U,V} P(U, V | R) \propto P(R | U, V) \cdot P(U) \cdot P(V)$$

### Implementation Logic
```
1. Initialize U, V randomly
2. Optimize (SGD or ALS):
   - Maximize likelihood of observed ratings
   - Regularization from priors
3. Predict: rating_ij = dot(U_i, V_j)
4. Recommend: Top items by predicted rating
```

### Why Probabilistic?

| Benefit | Explanation |
|---------|-------------|
| **Handles Sparsity** | Priors help with few observations |
| **Uncertainty** | Can estimate confidence in predictions |
| **Cold Start** | New users/items start at prior mean |
| **Regularization** | Built-in via Gaussian priors |

### Practical Considerations
- Implicit feedback: Use Bernoulli likelihood instead of Gaussian
- Variational inference for full Bayesian treatment
- Matrix factorization + neural = Neural Collaborative Filtering

---

## Question 2

**Propose a method for predicting customer churn using probabilistic models.**

**Answer:**

### Recommended Approach: Logistic Regression

**Why Logistic Regression?**
- Outputs probability P(churn | features)
- Interpretable coefficients
- Well-calibrated with proper tuning

### Model Framework

**1. Define Problem**
- Target: P(Churn = 1 | Customer Features)
- Time window: e.g., churn in next 30 days

**2. Feature Engineering**
| Category | Features |
|----------|----------|
| Usage | Login frequency, session duration |
| Support | Ticket count, resolution time |
| Billing | Plan type, payment failures |
| Tenure | Account age, engagement trend |

**3. Model Output**
$$P(\text{churn}) = \sigma(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)$$

Output: Churn score between 0 and 1 (e.g., 0.85 = 85% churn risk)

### From Probability to Decision

**Cost-Benefit Analysis**
- C_FP: Cost of false positive (discount to happy customer)
- C_FN: Cost of false negative (losing churning customer)

**Threshold Selection**
```python
# If C_FN >> C_FP, lower threshold to catch more churners
threshold = 0.3  # Flag if P(churn) > 0.3
high_risk = customers[churn_prob > threshold]
```

### Advanced: Survival Analysis

For time-to-churn prediction:
- Cox Proportional Hazards model
- Outputs survival curve S(t) = P(hasn't churned by time t)
- Handles censored data (customers who haven't churned yet)

### Key Actions
1. **Rank customers** by churn probability
2. **Prioritize intervention** for high-risk customers
3. **Measure** model calibration (predicted vs actual rates)
4. **Track** coefficient signs for business insights

---

## Question 3

**Discuss how you would use probabilities to detect anomalies in transaction data.**

**Answer:**

### Approach: Density-Based Anomaly Detection

**Core Idea**: Model "normal" transactions probabilistically; flag low-probability transactions as anomalies.

### Step-by-Step Method

**1. Feature Selection**
- Transaction amount
- Time of day
- Frequency (transactions per user recently)
- Location deviation
- Merchant category

**2. Model Normal Behavior**

Train on historical normal transactions:

**Option A: Multivariate Gaussian**
$$p(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

Learn μ (mean vector) and Σ (covariance matrix) from normal data.

**Option B: Gaussian Mixture Model**
For multi-modal normal behavior (e.g., weekday vs weekend patterns).

**3. Anomaly Detection**
```python
def is_anomaly(x, threshold):
    """
    Flag transaction as anomaly if probability is too low.
    """
    prob_density = multivariate_normal.pdf(x, mean=mu, cov=sigma)
    return prob_density < threshold
```

**4. Setting Threshold (ε)**

| Method | Approach |
|--------|----------|
| Percentile | ε = 1st percentile of training densities |
| Labeled data | Optimize F1-score on validation set |
| Business rule | Accept X% false positive rate |

### Implementation Outline
```python
from scipy.stats import multivariate_normal
import numpy as np

# 1. Fit on normal transactions
mu = np.mean(normal_transactions, axis=0)
sigma = np.cov(normal_transactions.T)

# 2. Score new transactions
scores = multivariate_normal.pdf(new_transactions, mean=mu, cov=sigma)

# 3. Flag anomalies
threshold = np.percentile(scores, 1)  # Bottom 1%
anomalies = new_transactions[scores < threshold]
```

### Practical Considerations
- **Feature normalization**: Log-transform skewed features (amounts)
- **Concept drift**: Retrain periodically as normal behavior evolves
- **Interpretability**: Which features contributed to low probability?

### Alternative: Autoencoder
- Reconstruction error ∝ 1/probability
- High error = anomaly (data model hasn't seen)

---

