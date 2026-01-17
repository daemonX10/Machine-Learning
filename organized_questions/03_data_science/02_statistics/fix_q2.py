import re

file_path = r'c:\Users\damod\OneDrive\Desktop\Machine-Learning\organized_questions\03_data_science\02_statistics\theory_questions.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Using smart quotes to match the file
old_q2 = '''Question 2

**Explain what a "distribution" is in statistics, and give examples of common distributions.**

**Answer:** _[To be filled]_'''

new_q2 = '''Question 2

**Explain what a "distribution" is in statistics, and give examples of common distributions.**

**Answer:**

### Definition
A **distribution** describes how values of a variable are spread or arranged. It shows the frequency or probability of each possible value.

### Key Distributions

| Distribution | Type | Use Case |
|-------------|------|----------|
| Normal | Continuous | Heights, test scores, measurement errors |
| Binomial | Discrete | Success/failure trials (clicks, conversions) |
| Poisson | Discrete | Count events (website visits, defects) |
| Exponential | Continuous | Time between events (customer arrivals) |
| Uniform | Both | Random sampling, simulations |

### Python Examples
```python
import numpy as np
import scipy.stats as stats

# Normal distribution
normal_data = np.random.normal(mean=0, scale=1, size=1000)

# Binomial: 10 trials, 50% success probability  
binom_data = np.random.binomial(n=10, p=0.5, size=1000)

# Poisson: average 5 events
poisson_data = np.random.poisson(lam=5, size=1000)
```

### Interview Tip
Know common distributions and parameters: Normal(μ, σ²), Binomial(n, p), Poisson(λ).'''

if old_q2 in content:
    content = content.replace(old_q2, new_q2)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Q2 replaced successfully')
else:
    print('Q2 not found with exact match')
