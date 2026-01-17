# Probability Interview Questions - Coding Questions

## Question 1

**Write a Python function that calculates the probability of rolling a sum of 'S' on two dice.**

**Answer:**

### Approach
1. Total outcomes = 6 × 6 = 36
2. Count favorable outcomes where die1 + die2 = S
3. P(sum = S) = favorable / total

### Code
```python
def probability_of_sum(S):
    """
    Calculate probability of rolling sum S on two fair dice.
    
    Args:
        S: Target sum (valid: 2 to 12)
    Returns:
        Probability as float
    """
    # Validate input
    if not isinstance(S, int) or not 2 <= S <= 12:
        return 0.0
    
    total_outcomes = 36  # 6 * 6
    favorable = 0
    
    # Count combinations that sum to S
    for die1 in range(1, 7):
        for die2 in range(1, 7):
            if die1 + die2 == S:
                favorable += 1
    
    return favorable / total_outcomes

# Test
print(f"P(sum=7) = {probability_of_sum(7):.4f}")   # 6/36 = 0.1667
print(f"P(sum=2) = {probability_of_sum(2):.4f}")   # 1/36 = 0.0278
print(f"P(sum=12) = {probability_of_sum(12):.4f}") # 1/36 = 0.0278
```

### Output
```
P(sum=7) = 0.1667
P(sum=2) = 0.0278
P(sum=12) = 0.0278
```

### Quick Reference
| Sum | Ways | Probability |
|-----|------|-------------|
| 2 | 1 | 1/36 |
| 7 | 6 | 6/36 |
| 12 | 1 | 1/36 |

---

## Question 2

**Implement a function that simulates a biased coin flip n times and estimates the probability of heads.**

**Answer:**

### Approach
1. Simulate n flips using biased probability
2. Count heads (represented as 1)
3. Estimate P(heads) = count / n

### Code
```python
import numpy as np

def simulate_biased_coin(n_flips, prob_heads=0.7):
    """
    Simulate biased coin flips and estimate P(heads).
    
    Args:
        n_flips: Number of coin flips
        prob_heads: True probability of heads (bias)
    Returns:
        Estimated probability of heads
    """
    if not 0 <= prob_heads <= 1:
        raise ValueError("prob_heads must be between 0 and 1")
    if n_flips <= 0:
        return 0.0
    
    # Simulate flips: 1 = Heads, 0 = Tails
    flips = np.random.choice(
        [1, 0], 
        size=n_flips, 
        p=[prob_heads, 1 - prob_heads]
    )
    
    # Estimate probability
    num_heads = np.sum(flips)
    estimated_prob = num_heads / n_flips
    
    return estimated_prob

# Test with different sample sizes
true_prob = 0.7

print(f"True P(heads): {true_prob}")
print(f"Estimated (n=10):    {simulate_biased_coin(10, true_prob):.4f}")
print(f"Estimated (n=100):   {simulate_biased_coin(100, true_prob):.4f}")
print(f"Estimated (n=10000): {simulate_biased_coin(10000, true_prob):.4f}")
```

### Output
```
True P(heads): 0.7
Estimated (n=10):    0.6000  # High variance
Estimated (n=100):   0.7200
Estimated (n=10000): 0.6987  # Close to true (LLN)
```

### Key Insight
Law of Large Numbers: As n increases, estimate → true probability.

---

## Question 3

**Code a Gaussian Naive Bayes classifier from scratch using Python.**

**Answer:**

### Approach
1. **fit()**: Learn priors P(C) and Gaussian parameters (μ, σ²) per feature per class
2. **predict()**: Use Bayes' theorem with "naive" independence assumption

### Code
```python
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        """Learn class priors and feature statistics."""
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = len(X_c) / len(X)
    
    def _gaussian_pdf(self, x, mean, var):
        """Calculate Gaussian probability density."""
        eps = 1e-9  # Avoid division by zero
        coef = 1 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        return coef * exponent
    
    def predict(self, X):
        """Predict class for each sample."""
        predictions = []
        
        for x in X:
            posteriors = {}
            for c in self.classes:
                # Log prior
                log_prior = np.log(self.priors[c])
                
                # Sum of log likelihoods (naive assumption)
                log_likelihood = np.sum(
                    np.log(self._gaussian_pdf(x, self.mean[c], self.var[c]))
                )
                
                posteriors[c] = log_prior + log_likelihood
            
            # Predict class with highest posterior
            predictions.append(max(posteriors, key=posteriors.get))
        
        return np.array(predictions)

# Test
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100, n_features=2, 
                           n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### Key Points
- Use **log probabilities** to avoid numerical underflow
- **Naive assumption**: Features independent given class

---

## Question 4

**Simulate the Law of Large Numbers using Python: verify that as the number of coin tosses increases, the average of the results becomes closer to the expected value.**

**Answer:**

### Approach
1. Simulate many coin flips (H=1, T=0)
2. Calculate running average after each flip
3. Visualize convergence to E[X] = 0.5

### Code
```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_lln(n_trials):
    """
    Demonstrate Law of Large Numbers with coin flips.
    E[X] = 0.5 for fair coin where H=1, T=0
    """
    # Expected value for fair coin
    expected_value = 0.5
    
    # Simulate coin flips: 1=Heads, 0=Tails
    flips = np.random.randint(0, 2, size=n_trials)
    
    # Calculate running average
    # cumsum: [f1, f1+f2, f1+f2+f3, ...]
    # divide by: [1, 2, 3, ...]
    running_avg = np.cumsum(flips) / np.arange(1, n_trials + 1)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(running_avg, label='Sample Average')
    plt.axhline(y=expected_value, color='r', linestyle='--', 
                label=f'Expected Value ({expected_value})')
    plt.xlabel('Number of Flips')
    plt.ylabel('Average (Proportion of Heads)')
    plt.title('Law of Large Numbers: Convergence to E[X]')
    plt.xscale('log')  # Log scale shows convergence better
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"After {n_trials} flips:")
    print(f"  Sample average: {running_avg[-1]:.4f}")
    print(f"  Expected value: {expected_value}")

# Run simulation
simulate_lln(100000)
```

### Output
```
After 100000 flips:
  Sample average: 0.5003
  Expected value: 0.5
```

### Observation
- Initially: Running average fluctuates wildly
- As n increases: Average stabilizes and converges to 0.5
- This IS the Law of Large Numbers

---

## Question 5

**Write a Python script that estimates the mean and variance of a dataset and plots the corresponding Gaussian distribution.**

**Answer:**

### Approach
1. Estimate μ (mean) and σ² (variance) from data
2. Plot histogram of data (normalized)
3. Overlay fitted Gaussian PDF

### Code
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def analyze_and_plot(data):
    """
    Estimate Gaussian parameters and visualize fit.
    
    Args:
        data: 1D array of numerical values
    """
    # 1. Estimate parameters (MLE)
    mu = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    
    print(f"Estimated Mean (μ): {mu:.2f}")
    print(f"Estimated Variance (σ²): {var:.2f}")
    print(f"Estimated Std Dev (σ): {std:.2f}")
    
    # 2. Plot histogram (normalized to density)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, 
             color='green', label='Data Histogram')
    
    # 3. Plot fitted Gaussian PDF
    x = np.linspace(min(data), max(data), 100)
    pdf = norm.pdf(x, mu, std)
    plt.plot(x, pdf, 'k', linewidth=2, label='Fitted Gaussian')
    
    plt.title(f'Data vs Fitted Gaussian (μ={mu:.2f}, σ={std:.2f})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate sample data
np.random.seed(42)
sample_data = np.random.normal(loc=50, scale=10, size=1000)

# Analyze
analyze_and_plot(sample_data)
```

### Output
```
Estimated Mean (μ): 50.12
Estimated Variance (σ²): 99.45
Estimated Std Dev (σ): 9.97
```

### Key Points
- `density=True` normalizes histogram so area = 1 (comparable to PDF)
- `norm.pdf(x, μ, σ)` gives Gaussian density at x
- Good fit: Gaussian curve follows histogram shape

---

