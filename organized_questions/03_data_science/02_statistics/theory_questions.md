

1. What is the difference between descriptive and inferential statistics?

**Answer:**

**Definition:**
Descriptive statistics summarizes and describes the characteristics of a collected dataset (what does my data look like?), while inferential statistics uses sample data to make generalizations and predictions about a larger population (what can I conclude about the population?).

**Core Concepts:**
- **Descriptive Statistics:**
  - Summarizes known data - no uncertainty involved
  - Measures: Mean, Median, Mode (central tendency); Variance, SD, Range, IQR (dispersion)
  - Visualizations: Histograms, Box plots, Bar charts
  - Scope: Conclusions apply only to the collected data

- **Inferential Statistics:**
  - Makes predictions about population from sample
  - Techniques: Hypothesis testing (t-test, ANOVA, Chi-squared), Confidence intervals, Regression
  - Always involves uncertainty quantified by p-values, confidence levels
  - Scope: Conclusions generalized to entire population

**Practical Relevance:**
- Descriptive: "Average age of 1,000 surveyed customers is 35 years"
- Inferential: "We are 95% confident the average age of ALL customers is between 33-37 years"

**Summary Table:**
| Feature | Descriptive | Inferential |
|---------|-------------|-------------|
| Goal | Describe sample | Generalize to population |
| Output | Summary stats, charts | p-values, confidence intervals |
| Uncertainty | Not applicable | Quantified |

---

2. Define and distinguish between population and sample in statistics.
**Answer:**

**Definition:**
A **population** is the entire set of individuals, items, or data points that you want to study and draw conclusions about. A **sample** is a subset of the population, selected to represent it, from which data is actually collected for analysis.

**Core Concepts:**
- **Population:**
  - Complete set of all elements of interest
  - Denoted by N (size), μ (mean), σ (standard deviation)
  - Usually impractical or impossible to measure entirely
  - Example: All voters in a country

- **Sample:**
  - Subset drawn from population
  - Denoted by n (size), x̄ (mean), s (standard deviation)
  - Must be representative to make valid inferences
  - Example: 1,000 randomly selected voters for a poll

**Why Sampling:**
- Cost-effective and time-efficient
- Population may be infinite or inaccessible
- Destructive testing scenarios (quality control)

**Key Principle:** The goal of sampling is to collect data from a representative subset so that we can make valid inferences about the entire population using inferential statistics.

**Sampling Methods:**
- Random Sampling, Stratified Sampling, Cluster Sampling, Systematic Sampling

---
3. Explain what a “distribution” is in statistics, and give examples of common distributions.
**Answer:**

**Definition:**
A distribution is a mathematical function that describes the probability of all possible values a random variable can take. It tells us which outcomes are more likely and which are less likely, and can be visualized as a graph showing values on the x-axis and their probabilities on the y-axis.

**Core Concepts:**
- Distributions are either **Discrete** (countable outcomes) or **Continuous** (uncountable outcomes)
- Characterized by parameters (mean, variance, rate, etc.)
- Total probability always sums/integrates to 1

**Common Discrete Distributions:**
| Distribution | Use Case | Example |
|--------------|----------|---------|
| Bernoulli | Single trial, 2 outcomes | One coin flip |
| Binomial | # successes in n trials | Heads in 10 coin flips |
| Poisson | Count of events in fixed interval | Customers per hour |

**Common Continuous Distributions:**
| Distribution | Use Case | Example |
|--------------|----------|---------|
| Normal (Gaussian) | Symmetric bell curve | Heights, test scores |
| Uniform | All outcomes equally likely | Random number [0,1] |
| Exponential | Time between events | Time until next customer |
| Log-Normal | Right-skewed positive values | Income, stock prices |

**Practical Relevance:**
- Model data generation process
- Hypothesis testing uses theoretical distributions (t, Chi-squared, F)
- Simulation and Monte Carlo methods

---
4. What is the Central Limit Theorem and why is it important in statistics?

**Answer:**

**Definition:**
The Central Limit Theorem (CLT) states that for a sufficiently large sample size (n > 30), the sampling distribution of the sample mean will be approximately normally distributed, regardless of the original population's distribution.

**Mathematical Formulation:**
If we draw samples of size n from a population with mean μ and standard deviation σ:
- Mean of sampling distribution = μ
- Standard Error = σ / √n
- Distribution shape → Normal (as n increases)

**Core Concepts:**
- Works regardless of population shape (skewed, uniform, etc.)
- Larger n → Better approximation to normal
- Standard Error decreases as sample size increases

**Why It's Important:**
1. **Foundation for Hypothesis Testing:** Justifies using t-tests and Z-tests even when population isn't normal
2. **Confidence Intervals:** Formula for CI is derived from CLT
3. **Generalization:** Bridges sample statistics to population parameters

**Intuition:**
Even if individual customer spending is highly skewed, the average spending of 100 randomly selected customers will follow a normal distribution. This lets us use normal distribution properties for inference.

**Practical Relevance:**
- Enables statistical inference on any data type with sufficient sample size
- Critical for A/B testing, polling, quality control

---

5. Describe what a p-value is and what it signifies about the statistical significance of a result.
**Answer:**

**Definition:**
A p-value is the probability of obtaining results at least as extreme as the observed results, assuming the null hypothesis (H₀) is true. It quantifies how "surprising" the data is under the assumption of no effect.

**Core Concepts:**
- **Small p-value (≤ 0.05):** Data is unlikely under H₀ → Reject H₀ → Statistically significant
- **Large p-value (> 0.05):** Data is plausible under H₀ → Fail to reject H₀ → Not significant

**Critical Misinterpretation:**
- p-value is **NOT** the probability that H₀ is true
- p-value is **NOT** the probability of the result being due to chance

**Decision Rule:**
```
If p-value ≤ α (significance level): Reject H₀
If p-value > α: Fail to reject H₀
```

**Example (A/B Test):**
- H₀: Conversion rates of A and B are equal
- Observed: B has 2% higher conversion
- p-value = 0.03
- Interpretation: Only 3% chance of seeing this difference if A and B were truly equal
- Conclusion: Since 0.03 ≤ 0.05, reject H₀ → Significant difference

**Interview Tip:** Always clarify that low p-value means evidence against H₀, not proof of H₁.

---
6. What does the term “statistical power” refer to?
**Answer:**

**Definition:**
Statistical power is the probability of correctly rejecting the null hypothesis when it is actually false. In other words, it's the probability of detecting a real effect when one exists. Power = 1 - β, where β is the probability of Type II error.

**Mathematical Formulation:**
$$\text{Power} = 1 - \beta = P(\text{Reject } H_0 \mid H_0 \text{ is false})$$

**Core Concepts:**
- Power ranges from 0 to 1 (often expressed as percentage)
- Typical target: **Power ≥ 0.80** (80%)
- Higher power → Lower chance of missing a true effect

**Factors Affecting Power:**
| Factor | Effect on Power |
|--------|-----------------|
| ↑ Sample size (n) | ↑ Power |
| ↑ Effect size | ↑ Power |
| ↑ Significance level (α) | ↑ Power |
| ↓ Variance in data | ↑ Power |

**Practical Relevance:**
- **Sample Size Planning:** Before running an experiment, calculate required sample size for desired power
- **Interpreting Non-Significant Results:** Low power study may fail to detect real effects
- **A/B Testing:** Underpowered tests lead to inconclusive results

**Interview Tip:** A non-significant result in a low-power study doesn't mean there's no effect—it means the study wasn't sensitive enough to detect it.

---
7. Explain the concepts of Type I and Type II errors in hypothesis testing.

**Answer:**

**Definition:**
Type I and Type II errors are the two kinds of mistakes that can be made when making a decision about a hypothesis based on sample data.

**Decision Matrix:**
|  | H₀ is True (Reality) | H₀ is False (Reality) |
|--|---------------------|----------------------|
| **Reject H₀** | Type I Error (α) - False Positive | Correct Decision (Power = 1-β) |
| **Fail to Reject H₀** | Correct Decision | Type II Error (β) - False Negative |

**Type I Error (False Positive):**
- Rejecting a true null hypothesis
- Probability = α (significance level)
- **Analogy:** Convicting an innocent person
- **Medical:** Diagnosing disease in healthy person

**Type II Error (False Negative):**
- Failing to reject a false null hypothesis
- Probability = β
- **Analogy:** Acquitting a guilty person
- **Medical:** Missing disease in sick person

**The Trade-off:**
- Decreasing α (stricter threshold) → Increases β
- You cannot minimize both simultaneously
- Choice depends on relative costs of each error type

**When to prioritize:**
- **Minimize Type I:** When false positives are costly (approving ineffective drugs)
- **Minimize Type II:** When missing an effect is dangerous (cancer screening)

**Interview Tip:** Always consider the business/real-world cost of each error type when setting α.

---

8. What is the significance level in a hypothesis test and how is it chosen?

**Answer:**

**Definition:**
The significance level (α) is the threshold probability set before conducting a hypothesis test. It represents the maximum acceptable probability of making a Type I error (rejecting a true null hypothesis). It defines what we consider "statistically significant."

**Core Concepts:**
- α = P(Reject H₀ | H₀ is true) = P(Type I Error)
- Common values: **0.05** (standard), 0.01 (strict), 0.10 (lenient)
- Set **before** collecting data, not after

**Decision Rule:**
- If p-value ≤ α → Reject H₀ (statistically significant)
- If p-value > α → Fail to reject H₀

**How to Choose α:**

| Context | Recommended α | Reasoning |
|---------|---------------|-----------|
| Standard research | 0.05 | Convention, balanced |
| High-stakes medical | 0.01 | False positive costly |
| Exploratory analysis | 0.10 | Don't want to miss effects |
| Multiple comparisons | 0.05/n (Bonferroni) | Control family-wise error |

**Practical Guidelines:**
1. **Convention:** Use 0.05 as default unless there's a reason not to
2. **Cost of Errors:** Lower α when Type I error is more costly than Type II
3. **Field Standards:** Follow domain conventions (physics uses 5σ ≈ 0.0000003)

**Interview Tip:** Explain that α is not about proving anything—it's about controlling the long-run false positive rate.

---

9. Define confidence interval and its importance in statistics.

**Answer:**

**Definition:**
A confidence interval (CI) is a range of values that, with a certain level of confidence (e.g., 95%), is likely to contain the true population parameter. It quantifies the uncertainty around a point estimate.

**Mathematical Formulation:**
$$CI = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

For 95% CI with known σ: $\bar{x} \pm 1.96 \cdot SE$

**Core Concepts:**
- **95% CI Interpretation:** If we repeated the experiment many times, 95% of the calculated intervals would contain the true parameter
- **NOT:** "95% probability the true value is in this interval"
- Width depends on: sample size, variability, confidence level

**Factors Affecting CI Width:**
| Factor | Effect on Width |
|--------|-----------------|
| ↑ Sample size | Narrower CI |
| ↑ Confidence level | Wider CI |
| ↑ Variability (σ) | Wider CI |

**Importance:**
1. **Quantifies Uncertainty:** Shows precision of estimate
2. **Practical Interpretation:** More informative than just p-value
3. **Hypothesis Testing:** If CI doesn't include null value, result is significant

**Example:**
"Mean customer spending: $50 (95% CI: $45-$55)"
→ We're 95% confident the true population mean is between $45 and $55

**Interview Tip:** Emphasize the frequentist interpretation—it's about the procedure's long-run success rate, not probability about this specific interval.

---

10. What is a null hypothesis and an alternative hypothesis?
**Answer:**

**Definition:**
The null hypothesis (H₀) and alternative hypothesis (H₁) are two competing, mutually exclusive statements about a population parameter. Hypothesis testing uses sample data to decide which statement is better supported.

**Null Hypothesis (H₀):**
- Statement of "no effect," "no difference," or "status quo"
- Always contains equality (=, ≤, ≥)
- Assumed true until evidence suggests otherwise
- We never "prove" H₀, only reject or fail to reject it

**Alternative Hypothesis (H₁ or Hₐ):**
- Statement that contradicts H₀
- What the researcher wants to find evidence for
- Always contains inequality (≠, <, >)

**Types of Alternative Hypotheses:**
| Type | H₁ | Use Case |
|------|-----|----------|
| Two-tailed | μ ≠ μ₀ | Testing for any difference |
| Left-tailed | μ < μ₀ | Testing if value is less |
| Right-tailed | μ > μ₀ | Testing if value is greater |

**Examples:**
| Scenario | H₀ | H₁ |
|----------|-----|-----|
| A/B Test | μ_A = μ_B | μ_A ≠ μ_B |
| Drug efficacy | Drug has no effect | Drug has effect |
| Correlation | ρ = 0 | ρ ≠ 0 |

**Process:**
Collect data → Calculate test statistic → Get p-value → Compare to α → Decision (Reject/Fail to reject H₀)

**Interview Tip:** The burden of proof is on the alternative hypothesis. We need strong evidence to reject the default assumption (H₀).

---
11. What is Bayes’ Theorem, and how is it used in statistics?
**Answer:**

**Definition:**
Bayes' Theorem is a mathematical formula that describes how to update the probability of a hypothesis based on new evidence. It is the foundation of Bayesian statistics, which treats probability as a degree of belief.

**Mathematical Formulation:**
$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$

**Components:**
| Term | Name | Meaning |
|------|------|---------|
| P(H\|E) | Posterior | Updated probability of H given evidence E |
| P(E\|H) | Likelihood | Probability of observing E if H is true |
| P(H) | Prior | Initial belief about H before seeing evidence |
| P(E) | Evidence | Total probability of observing E |

**Bayesian Inference Workflow:**
1. **Define Prior:** P(θ) — Initial belief about parameter
2. **Define Likelihood:** P(data|θ) — How data depends on parameter
3. **Compute Posterior:** P(θ|data) ∝ P(data|θ) × P(θ)
4. **Make Inferences:** Use posterior for estimates, credible intervals

**Use in Statistics:**
- **Bayesian A/B Testing:** Get P(B > A | data) directly
- **Spam Filtering:** Update spam probability with each word
- **Medical Diagnosis:** P(Disease | Test+) with disease prevalence as prior

**vs. Frequentist:**
| Frequentist | Bayesian |
|-------------|----------|
| Fixed parameter | Parameter is random variable |
| p-values | Posterior probabilities |
| Confidence interval | Credible interval |

**Interview Tip:** Bayes allows incorporating prior knowledge and provides intuitive probability statements about parameters.

---
12. Describe the difference between discrete and continuous probability distributions.

**Answer:**

**Definition:**
Discrete distributions describe random variables that can only take countable values (integers), while continuous distributions describe variables that can take any value within a range (infinite precision).

**Key Differences:**

| Feature | Discrete | Continuous |
|---------|----------|------------|
| Values | Countable (0, 1, 2, ...) | Uncountable (any real number in range) |
| Function | PMF (Probability Mass Function) | PDF (Probability Density Function) |
| P(X = x) | Can be > 0 | Always = 0 |
| Probability calc | Sum: Σ P(x) | Integral: ∫ f(x) dx |
| Visualization | Bar chart | Smooth curve |

**Discrete Distribution:**
- **PMF:** P(X = x) gives direct probability
- **Examples:** Binomial, Poisson, Bernoulli
- **Use:** Number of heads, customer counts, defects

**Continuous Distribution:**
- **PDF:** f(x) is density, not probability
- **Probability:** P(a < X < b) = ∫ₐᵇ f(x)dx = Area under curve
- **Examples:** Normal, Exponential, Uniform
- **Use:** Heights, weights, time, temperature

**Key Insight:**
For continuous variables, P(X = exactly 1.5) = 0 because there are infinite possible values. We only compute probability over intervals.

**Python Example:**
```python
from scipy import stats
# Discrete: P(X = 3) for Binomial(n=10, p=0.5)
print(stats.binom.pmf(3, n=10, p=0.5))  # 0.117

# Continuous: P(0 < X < 1) for Standard Normal
print(stats.norm.cdf(1) - stats.norm.cdf(0))  # 0.341
```

---

13. Explain the properties of a Normal distribution.

**Answer:**

**Definition:**
The Normal (Gaussian) distribution is a continuous probability distribution defined by its mean (μ) and standard deviation (σ), forming the classic symmetric "bell curve."

**Mathematical Formulation:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**Key Properties:**

1. **Bell-Shaped & Symmetric:** Perfectly symmetric around mean
2. **Mean = Median = Mode:** All central tendency measures are equal
3. **Defined by μ and σ:** 
   - μ → location (center)
   - σ → spread (width)
4. **Asymptotic Tails:** Extends to ±∞, never touches x-axis

**Empirical Rule (68-95-99.7):**
| Range | % of Data |
|-------|-----------|
| μ ± 1σ | 68% |
| μ ± 2σ | 95% |
| μ ± 3σ | 99.7% |

**Standard Normal Distribution:**
- Special case: μ = 0, σ = 1
- Z-score transformation: $Z = \frac{X - \mu}{\sigma}$
- Enables use of standard Z-tables

**Why It's Important:**
- **CLT:** Sample means follow normal distribution
- **Many natural phenomena:** Heights, IQ, measurement errors
- **Foundation for:** Z-tests, t-tests, regression assumptions

**Python Example:**
```python
from scipy import stats
# P(X < 1.96) for standard normal
print(stats.norm.cdf(1.96))  # 0.975
```

---

14. What is the Law of Large Numbers, and how does it relate to statistics?

**Answer:**

**Definition:**
The Law of Large Numbers (LLN) states that as the sample size increases, the sample mean converges to the true population mean (expected value). Larger samples provide more accurate estimates.

**Mathematical Formulation:**
$$\lim_{n \to \infty} \bar{X}_n = \mu$$

As n → ∞, sample mean $\bar{X}_n$ → population mean μ

**Core Concepts:**
- **Weak LLN:** Sample mean converges in probability
- **Strong LLN:** Sample mean converges almost surely
- Applies to any distribution with finite mean

**Relation to Statistics:**

1. **Justifies Sampling:** Guarantees that samples can reliably estimate population parameters
2. **Estimation Foundation:** Sample statistics are consistent estimators (converge to true values)
3. **Monte Carlo Methods:** Basis for simulation-based estimation
4. **Connects Probability to Frequency:** Theoretical probability ≈ observed frequency as trials increase

**Intuition (Coin Flip Example):**
| Flips | Observed Heads % |
|-------|------------------|
| 10 | 40% |
| 100 | 48% |
| 1,000 | 50.2% |
| 10,000 | 49.98% |

As flips increase, proportion converges to true probability (0.5)

**LLN vs CLT:**
| LLN | CLT |
|-----|-----|
| Sample mean → population mean | Sample mean follows normal distribution |
| About convergence | About distribution shape |

**Interview Tip:** LLN explains WHY sampling works; CLT explains HOW to do inference with samples.

---

15. What is the role of the Binomial distribution in statistics?

**Answer:**

**Definition:**
The Binomial distribution models the number of successes (k) in a fixed number of independent trials (n), where each trial has the same probability of success (p).

**Mathematical Formulation:**
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**Parameters:**
- n = number of trials
- p = probability of success per trial
- Mean = np, Variance = np(1-p)

**Conditions for Binomial:**
1. Fixed number of trials (n)
2. Each trial has only 2 outcomes (success/failure)
3. Trials are independent
4. Probability p is constant across trials

**Role in Statistics:**

1. **Modeling Binary Data:** Clinical trials, quality control, surveys
2. **Hypothesis Testing for Proportions:** A/B testing, polling
3. **Foundation for Logistic Regression:** Log-loss derived from Binomial likelihood
4. **Confidence Intervals for Proportions:** Election polls, market surveys

**Approximations:**
| Condition | Approximation |
|-----------|---------------|
| Large n, p not extreme | Normal(μ=np, σ²=np(1-p)) |
| Large n, small p | Poisson(λ=np) |

**Python Example:**
```python
from scipy import stats
# P(exactly 7 heads in 10 coin flips)
prob = stats.binom.pmf(k=7, n=10, p=0.5)
print(f"P(X=7): {prob:.4f}")  # 0.1172
```

**Practical Use:** "If website conversion rate is 5%, what's the probability of getting 10+ conversions from 100 visitors?"

---

16. Explain the difference between joint, marginal, and conditional probability.

**Answer:**

**Definition:**
Three interconnected probability concepts describing relationships between multiple events/variables.

**Joint Probability: P(A, B) or P(A ∩ B)**
- Probability that both events A and B occur simultaneously
- Question: "What's the probability of sunny weather AND going to picnic?"
- Found in the cells of a contingency table

**Marginal Probability: P(A)**
- Probability of a single event, regardless of other events
- Calculated by summing over all outcomes of other variables
- P(A) = Σ P(A, B) for all values of B
- Found in the margins (totals) of a contingency table

**Conditional Probability: P(A|B)**
- Probability of A occurring, given that B has occurred
- Formula: $P(A|B) = \frac{P(A, B)}{P(B)}$
- Question: "Given it's sunny, what's the probability of going to picnic?"

**Example (Contingency Table):**
|  | Picnic=Yes | Picnic=No | Marginal |
|--|------------|-----------|----------|
| Sunny | 0.4 | 0.1 | **P(Sunny)=0.5** |
| Rainy | 0.1 | 0.4 | P(Rainy)=0.5 |
| Marginal | **P(Yes)=0.5** | P(No)=0.5 | 1.0 |

- **Joint:** P(Sunny, Yes) = 0.4
- **Marginal:** P(Sunny) = 0.5
- **Conditional:** P(Yes|Sunny) = 0.4/0.5 = 0.8

**Key Relationship:**
$$P(A, B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

This is the multiplication rule and foundation of Bayes' Theorem.

---

17. How does the Poisson distribution differ from the Normal distribution?

**Answer:**

**Definition:**
Poisson models count of events in fixed interval; Normal models continuous measurements around a central value.

**Key Differences:**

| Feature | Poisson | Normal |
|---------|---------|--------|
| Variable Type | Discrete (0, 1, 2, ...) | Continuous (any real number) |
| Shape | Right-skewed (symmetric when λ large) | Symmetric bell curve |
| Parameters | λ (rate) only | μ (mean) and σ (std dev) |
| Mean & Variance | Both = λ | Mean = μ, Variance = σ² |
| Range | [0, ∞) integers only | (-∞, +∞) |
| Use Case | Counting rare events | Measurements, errors |

**When to Use:**
- **Poisson:** Email arrivals per hour, defects per unit, accidents per day
- **Normal:** Heights, test scores, measurement errors

**Mathematical Formulations:**
$$\text{Poisson: } P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
$$\text{Normal: } f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**Relationship:**
When λ > 20, Poisson ≈ Normal(μ=λ, σ²=λ)

**Python Example:**
```python
from scipy import stats
# Poisson: P(X=5) with λ=3
print(stats.poisson.pmf(5, mu=3))  # 0.1008

# Normal: P(X < 5) with μ=3, σ=1.73
print(stats.norm.cdf(5, loc=3, scale=1.73))  # 0.876
```

---

18. What is a cumulative distribution function (CDF)?

**Answer:**

**Definition:**
The Cumulative Distribution Function (CDF) gives the probability that a random variable X takes a value less than or equal to x. It represents accumulated probability up to point x.

**Mathematical Formulation:**
$$F(x) = P(X \leq x)$$

- Discrete: $F(x) = \sum_{t \leq x} P(X = t)$
- Continuous: $F(x) = \int_{-\infty}^{x} f(t) \, dt$

**Properties:**
1. F(x) ranges from 0 to 1
2. Non-decreasing function
3. F(-∞) = 0, F(+∞) = 1
4. Right-continuous

**Key Uses:**

1. **Calculating Probabilities:**
   - P(X > a) = 1 - F(a)
   - P(a < X ≤ b) = F(b) - F(a)

2. **Percentiles/Quantiles:**
   - Median: value x where F(x) = 0.5
   - Inverse CDF (quantile function): F⁻¹(p) gives value at percentile p

3. **Goodness-of-Fit Tests:** Kolmogorov-Smirnov compares empirical CDF to theoretical

**Python Example:**
```python
from scipy import stats

# Standard Normal CDF
print(stats.norm.cdf(1.96))  # 0.975 → P(Z ≤ 1.96)
print(stats.norm.cdf(1) - stats.norm.cdf(-1))  # 0.68 → P(-1 < Z < 1)

# Inverse CDF (Quantile) - find 95th percentile
print(stats.norm.ppf(0.95))  # 1.645
```

**Visualization:** CDF is an S-shaped curve (for normal), starting at 0 and approaching 1.

---

19. Describe the use cases of the Exponential distribution and Uniform distribution.

**Answer:**

**Uniform Distribution:**
- **Definition:** All outcomes in range [a, b] are equally likely
- **PDF:** f(x) = 1/(b-a) for a ≤ x ≤ b
- **Mean:** (a+b)/2, **Variance:** (b-a)²/12

**Uniform Use Cases:**
| Use Case | Example |
|----------|---------|
| Random Number Generation | U(0,1) is foundation for all random sampling |
| Monte Carlo Simulation | Inverse transform sampling |
| Uninformative Prior | Bayesian analysis when no prior knowledge |
| Rounding Errors | Error in rounding is often uniform |

**Exponential Distribution:**
- **Definition:** Models time between events in a Poisson process
- **PDF:** f(x) = λe^(-λx) for x ≥ 0
- **Mean:** 1/λ, **Variance:** 1/λ²
- **Key Property:** Memoryless—P(X > s+t | X > s) = P(X > t)

**Exponential Use Cases:**
| Use Case | Example |
|----------|---------|
| Survival/Reliability Analysis | Time until device failure |
| Queuing Theory | Time between customer arrivals |
| Radioactive Decay | Time until particle decay |
| Service Time | Duration of phone calls |

**Key Distinction:**
- **Uniform:** "When will it happen?" → "Any time is equally likely"
- **Exponential:** "When will it happen?" → "Most likely soon, decreasing probability later"

**Python Example:**
```python
from scipy import stats
# Uniform: P(0.3 < X < 0.7) for U(0,1)
print(stats.uniform.cdf(0.7) - stats.uniform.cdf(0.3))  # 0.4

# Exponential: P(X < 2) with λ=0.5 (mean=2)
print(stats.expon.cdf(2, scale=2))  # 0.632
```

---

20. How is Covariance different from Correlation?

**Answer:**

**Definition:**
Both measure the linear relationship between two variables, but correlation is the standardized (scaled) version of covariance.

**Mathematical Formulations:**
$$\text{Covariance: } Cov(X,Y) = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{n-1}$$

$$\text{Correlation (Pearson): } r = \frac{Cov(X,Y)}{\sigma_X \cdot \sigma_Y}$$

**Key Differences:**

| Feature | Covariance | Correlation |
|---------|------------|-------------|
| Range | (-∞, +∞) | [-1, +1] |
| Units | Product of X and Y units | Unitless (standardized) |
| Interpretability | Hard to interpret magnitude | Easy to interpret strength |
| Scale-dependent | Yes | No |
| Comparability | Cannot compare across datasets | Can compare across datasets |

**Interpretation of Correlation:**
| r Value | Interpretation |
|---------|----------------|
| +1 | Perfect positive linear relationship |
| +0.7 to +0.9 | Strong positive |
| +0.4 to +0.6 | Moderate positive |
| 0 | No linear relationship |
| -1 | Perfect negative linear relationship |

**When to Use:**
- **Covariance:** Portfolio variance calculation, theoretical derivations
- **Correlation:** Comparing relationships, feature selection, reporting

**Python Example:**
```python
import numpy as np
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

print(f"Covariance: {np.cov(x, y)[0,1]:.2f}")  # 1.5
print(f"Correlation: {np.corrcoef(x, y)[0,1]:.2f}")  # 0.77
```

**Interview Tip:** Correlation normalizes covariance, making it interpretable and comparable across different variable scales.

---

21. What are measures of central tendency, and why are they important?

**Answer:**

**Definition:**
Measures of central tendency are statistics that describe the center or typical value of a dataset—the point around which data tends to cluster.

**Three Main Measures:**

| Measure | Definition | Formula | Best For |
|---------|------------|---------|----------|
| Mean | Arithmetic average | Σx/n | Symmetric data, no outliers |
| Median | Middle value (sorted) | 50th percentile | Skewed data, with outliers |
| Mode | Most frequent value | Value with max frequency | Categorical data |

**Why They're Important:**

1. **Data Summarization:** Condense entire dataset into single representative value
2. **Comparison:** Compare groups easily (e.g., mean income of cities)
3. **Foundation for Inference:** Hypothesis tests often compare means
4. **Skewness Detection:** Relationship reveals distribution shape

**Relationship in Different Distributions:**
| Distribution | Relationship |
|--------------|--------------|
| Symmetric | Mean = Median = Mode |
| Right-skewed | Mean > Median > Mode |
| Left-skewed | Mean < Median < Mode |

**When to Use Which:**
- **Mean:** Normal distribution, interval/ratio data
- **Median:** Skewed data (income, house prices), ordinal data
- **Mode:** Categorical data, finding most common category

**Python Example:**
```python
import numpy as np
from scipy import stats

data = [1, 2, 2, 3, 4, 5, 100]  # With outlier

print(f"Mean: {np.mean(data):.1f}")    # 16.7 (affected by outlier)
print(f"Median: {np.median(data):.1f}")  # 3.0 (robust)
print(f"Mode: {stats.mode(data)[0]}")   # 2
```

---

22. Explain measures of dispersion: Range, Interquartile Range (IQR), Variance, and Standard Deviation.

**Answer:**

**Definition:**
Measures of dispersion describe how spread out or scattered the data points are around the center. They complement central tendency measures.

**Summary Table:**

| Measure | Formula | Robust to Outliers | Units |
|---------|---------|-------------------|-------|
| Range | Max - Min | No | Same as data |
| IQR | Q3 - Q1 | Yes | Same as data |
| Variance | Σ(xᵢ - μ)²/n | No | Squared units |
| Std Dev | √Variance | No | Same as data |

**1. Range:**
- Simplest measure: Max value - Min value
- **Con:** Extremely sensitive to outliers
- **Use:** Quick rough estimate of spread

**2. Interquartile Range (IQR):**
- Range of middle 50% of data: Q3 - Q1
- **Pro:** Robust to outliers (ignores extreme 25% on each end)
- **Use:** Box plots, outlier detection (1.5×IQR rule)

**3. Variance (σ²):**
- Average of squared deviations from mean
- **Con:** Units are squared, hard to interpret
- **Use:** Theoretical calculations, ANOVA

**4. Standard Deviation (σ):**
- Square root of variance
- **Pro:** Same units as data, interpretable
- **Use:** Most common measure; Empirical rule (68-95-99.7)

**Python Example:**
```python
import numpy as np
data = [10, 20, 30, 40, 50, 100]

print(f"Range: {np.max(data) - np.min(data)}")  # 90
print(f"IQR: {np.percentile(data,75) - np.percentile(data,25)}")  # 27.5
print(f"Variance: {np.var(data, ddof=1):.1f}")  # 1016.7
print(f"Std Dev: {np.std(data, ddof=1):.1f}")   # 31.9
```

**Interview Tip:** Use IQR for skewed data/outliers; use SD for normal distributions.

---

23. What is the difference between mean and median, and when would you use each?

**Answer:**

**Definition:**
- **Mean:** Arithmetic average = Σx/n
- **Median:** Middle value when data is sorted (50th percentile)

**Key Differences:**

| Feature | Mean | Median |
|---------|------|--------|
| Calculation | Sum / Count | Middle value |
| Outlier Sensitivity | High (pulled toward outliers) | Low (robust) |
| Uses All Data Points | Yes | No (only position matters) |
| Best Distribution | Symmetric | Skewed |

**Example - Salary Data:**
```
Salaries: [50k, 55k, 60k, 65k, 70k, 1M]
Mean: $216,667 (misleading—pulled by CEO salary)
Median: $62,500 (better represents typical employee)
```

**When to Use Mean:**
- Symmetric distributions (Normal)
- No significant outliers
- When all values should contribute equally
- Statistical modeling (regression, t-tests)

**When to Use Median:**
- Skewed distributions
- Presence of outliers
- Income, house prices, response times
- Ordinal data

**Relationship with Distribution Shape:**
| Distribution | Relationship |
|--------------|--------------|
| Symmetric | Mean ≈ Median |
| Right-skewed | Mean > Median |
| Left-skewed | Mean < Median |

**Python Example:**
```python
import numpy as np
data = [50, 55, 60, 65, 70, 1000]
print(f"Mean: {np.mean(data):.0f}")    # 217
print(f"Median: {np.median(data):.0f}")  # 62
# Median better represents "typical" value
```

**Interview Tip:** If asked which to report, ask about the data distribution and presence of outliers.

---

24. How would you describe skewness and kurtosis in a dataset?

**Answer:**

**Definition:**
Skewness and kurtosis are measures of the shape of a distribution, describing asymmetry and tail heaviness respectively.

**Skewness (Asymmetry):**
Measures the degree of asymmetry of a distribution around its mean.

| Skewness | Value | Description | Example |
|----------|-------|-------------|---------|
| Right (Positive) | > 0 | Long tail on right; Mean > Median | Income data |
| Symmetric | ≈ 0 | Balanced tails | Normal distribution |
| Left (Negative) | < 0 | Long tail on left; Mean < Median | Age at retirement |

**Formula:**
$$\text{Skewness} = \frac{1}{n} \sum \left(\frac{x_i - \bar{x}}{s}\right)^3$$

**Kurtosis (Tail Heaviness):**
Measures the "tailedness" of a distribution—how much data is in the tails vs center.

| Type | Excess Kurtosis | Description |
|------|-----------------|-------------|
| Mesokurtic | = 0 | Normal distribution (baseline) |
| Leptokurtic | > 0 | Heavy tails, sharp peak (more outliers) |
| Platykurtic | < 0 | Light tails, flat peak (fewer outliers) |

**Formula:**
$$\text{Kurtosis} = \frac{1}{n} \sum \left(\frac{x_i - \bar{x}}{s}\right)^4 - 3$$

**Practical Relevance:**
- **ML:** Many algorithms assume normality; high skewness/kurtosis violates this
- **Risk Assessment:** High kurtosis = more extreme events (fat tails)
- **Transformation:** Log transform can reduce right skewness

**Python Example:**
```python
from scipy import stats
data = [1, 2, 2, 3, 3, 3, 4, 4, 5, 100]

print(f"Skewness: {stats.skew(data):.2f}")    # Positive (right-skewed)
print(f"Kurtosis: {stats.kurtosis(data):.2f}")  # High (heavy tail due to outlier)
```

---

25. What is the five-number summary in descriptive statistics?

**Answer:**

**Definition:**
The five-number summary consists of five statistics that describe the distribution of a dataset: Minimum, Q1, Median, Q3, and Maximum. It's robust to outliers and forms the basis for box plots.

**The Five Numbers:**

| Statistic | Description | Percentile |
|-----------|-------------|------------|
| Minimum | Smallest value | 0th |
| Q1 (First Quartile) | 25% of data below this | 25th |
| Median (Q2) | Middle value | 50th |
| Q3 (Third Quartile) | 75% of data below this | 75th |
| Maximum | Largest value | 100th |

**What It Tells Us:**
- **Center:** Median
- **Spread:** IQR = Q3 - Q1 (middle 50%)
- **Range:** Max - Min
- **Skewness:** Compare distances (Q2-Q1) vs (Q3-Q2)

**Box Plot Visualization:**
```
     Min   Q1    Median   Q3    Max
      |----[========|========]-----|
           |<-- IQR -->|
```

**Outlier Detection Rule:**
- Lower fence: Q1 - 1.5 × IQR
- Upper fence: Q3 + 1.5 × IQR
- Points outside fences are potential outliers

**Python Example:**
```python
import numpy as np
data = [1, 2, 5, 6, 8, 9, 12, 15, 18, 19, 22, 50]

print(f"Min: {np.min(data)}")              # 1
print(f"Q1: {np.percentile(data, 25)}")    # 5.75
print(f"Median: {np.median(data)}")        # 10.5
print(f"Q3: {np.percentile(data, 75)}")    # 18.25
print(f"Max: {np.max(data)}")              # 50
```

**Interview Tip:** Five-number summary is preferred over mean/SD for skewed data or when outliers are present.

---

26. Explain the steps in conducting a hypothesis test.

**Answer:**

**Definition:**
A hypothesis test is a formal procedure to determine whether sample evidence is strong enough to reject a claim about a population parameter.

**Steps to Conduct Hypothesis Test:**

**Step 1: State the Hypotheses**
- H₀ (Null): Statement of no effect/difference (contains =)
- H₁ (Alternative): What you want to prove (contains ≠, <, or >)

**Step 2: Set Significance Level (α)**
- Choose threshold before seeing data
- Common: α = 0.05, 0.01, or 0.10
- This is your tolerance for Type I error

**Step 3: Choose Appropriate Test**
| Scenario | Test |
|----------|------|
| Compare 1 sample mean to value | One-sample t-test |
| Compare 2 independent means | Independent t-test |
| Compare 2 paired means | Paired t-test |
| Compare 3+ group means | ANOVA |
| Compare proportions | Z-test for proportions |
| Categorical relationship | Chi-squared test |

**Step 4: Calculate Test Statistic**
- Compute from sample data
- General form: (Observed - Expected) / Standard Error

**Step 5: Find p-value**
- Probability of getting result as extreme, assuming H₀ true
- Use statistical tables or software

**Step 6: Make Decision**
```
If p-value ≤ α: Reject H₀ → Statistically significant
If p-value > α: Fail to reject H₀ → Not significant
```

**Step 7: State Conclusion in Context**
- Translate statistical decision to real-world meaning

**Algorithm to Remember:**
```
1. H₀ and H₁
2. Set α
3. Select test
4. Calculate statistic
5. Get p-value
6. Compare p to α
7. Conclude
```

---

27. Describe how a t-test is performed and when it is appropriate to use.

**Answer:**

**Definition:**
A t-test determines whether there is a statistically significant difference between the means of groups. It uses the t-distribution and is appropriate when sample size is small or population variance is unknown.

**When to Use:**
- Comparing means of exactly 2 groups
- Continuous dependent variable
- Sample size small (n < 30) or population σ unknown
- Data approximately normal (or n > 30 by CLT)

**Types of t-tests:**

| Type | Use Case | Example |
|------|----------|---------|
| One-sample | Compare sample mean to known value | "Is class average ≠ 75?" |
| Independent | Compare 2 unrelated groups | "Control vs Treatment group" |
| Paired | Compare same subjects twice | "Before vs After treatment" |

**Formula (Independent t-test):**
$$t = \frac{\bar{x}_1 - \bar{x}_2}{SE} = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

**Steps to Perform:**
1. State H₀: μ₁ = μ₂ and H₁: μ₁ ≠ μ₂
2. Calculate t-statistic from data
3. Determine degrees of freedom (df = n₁ + n₂ - 2)
4. Find p-value from t-distribution
5. Compare p-value to α (usually 0.05)

**Python Example:**
```python
from scipy import stats

group_a = [85, 90, 78, 92, 88]
group_b = [75, 80, 72, 85, 78]

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
# If p < 0.05, reject H₀ → significant difference
```

**Assumptions:**
- Independence of observations
- Normality (or large n)
- Equal variances (use Welch's t-test if violated)

---


---

28. What is ANOVA (analysis of variance), and when is it used?

**Answer:**

**Definition:** ANOVA (Analysis of Variance) is a statistical method that compares means across 3+ groups by analyzing variance. It determines if at least one group mean differs significantly from others.

**Core Concepts:**
- Tests Hâ‚€: Î¼â‚ = Î¼â‚‚ = Î¼â‚ƒ = ... = Î¼â‚–
- Compares between-group variance to within-group variance
- F-statistic = (Between-group variance) / (Within-group variance)
- One-way ANOVA: one factor; Two-way ANOVA: two factors

**Mathematical Formula:**
$$F = \frac{MS_{between}}{MS_{within}} = \frac{SS_{between}/(k-1)}{SS_{within}/(N-k)}$$

Where:
- SSbetween = Î£náµ¢(xÌ„áµ¢ - xÌ„)Â²
- SSwithin = Î£Î£(xáµ¢â±¼ - xÌ„áµ¢)Â²

**When to Use:**
- Comparing means of 3+ independent groups
- Testing treatment effects across multiple conditions
- Analyzing experimental designs

**Python Example:**
```python
from scipy import stats
group1 = [85, 90, 88, 92, 87]
group2 = [78, 82, 80, 85, 79]
group3 = [92, 95, 91, 94, 93]

f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"F = {f_stat:.3f}, p = {p_value:.4f}")
```

**Interview Tip:** Always mention post-hoc tests (Tukey HSD) after significant ANOVA to identify which groups differ.

---

29. Explain the concepts of effect size and Cohen's d.

**Answer:**

**Definition:** Effect size quantifies the magnitude of a difference or relationship, independent of sample size. Cohen's d specifically measures the standardized mean difference between two groups.

**Core Concepts:**
- Effect size tells you "how big" not just "is it significant"
- p-value tells significance; effect size tells practical importance
- Large samples can make tiny effects significant

**Cohen's d Formula:**
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

Where pooled SD:
$$s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$$

**Interpretation Guidelines (Cohen's conventions):**
| d value | Effect Size |
|---------|-------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

**Python Example:**
```python
import numpy as np

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

d = cohens_d([85,90,88], [78,82,80])
print(f"Cohen's d = {d:.2f}")
```

**ML Relevance:** Report effect size alongside p-values in A/B tests to show practical significance.

---

30. How do you perform a Chi-squared test, and what does it tell you?

**Answer:**

**Definition:** Chi-squared (Ï‡Â²) test is a non-parametric test that compares observed frequencies with expected frequencies to determine if there's a significant association between categorical variables.

**Core Concepts:**
- Tests independence between two categorical variables
- Compares observed vs expected frequencies under Hâ‚€
- Requires expected counts â‰¥ 5 in each cell

**Types:**
1. **Test of Independence**: Are two variables related?
2. **Goodness-of-Fit**: Does data follow expected distribution?

**Formula:**
$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

Where: O = observed frequency, E = expected frequency
Expected = (row total Ã— column total) / grand total

**Steps:**
1. Create contingency table
2. Calculate expected frequencies
3. Compute Ï‡Â² statistic
4. Compare to chi-squared distribution with df = (r-1)(c-1)

**Python Example:**
```python
from scipy.stats import chi2_contingency
import numpy as np

# Observed: rows=gender, cols=preference
observed = [[30, 10],   # Male: A, B
            [20, 40]]   # Female: A, B

chi2, p_value, dof, expected = chi2_contingency(observed)
print(f"Ï‡Â² = {chi2:.2f}, p = {p_value:.4f}")
```

**Interview Tip:** Mention Yates' correction for 2Ã—2 tables or Fisher's exact test for small samples.

---

31. What is a nonparametric statistical test, and why might you use one?

**Answer:**

**Definition:** Nonparametric tests make no assumptions about the underlying population distribution. They work with ordinal data, ranks, or when parametric assumptions are violated.

**Core Concepts:**
- Don't assume normal distribution
- Use ranks instead of raw values
- Less powerful but more robust
- Suitable for ordinal data and small samples

**When to Use:**
- Data violates normality assumption
- Ordinal or ranked data (Likert scales)
- Presence of outliers
- Small sample sizes
- Unknown distribution

**Common Nonparametric Tests:**

| Parametric | Nonparametric Equivalent |
|------------|-------------------------|
| One-sample t-test | Wilcoxon signed-rank |
| Two-sample t-test | Mann-Whitney U |
| Paired t-test | Wilcoxon signed-rank |
| One-way ANOVA | Kruskal-Wallis |
| Pearson correlation | Spearman correlation |

**Python Example:**
```python
from scipy import stats

# Mann-Whitney U (alternative to t-test)
group1 = [5, 7, 8, 6, 9]
group2 = [3, 4, 5, 2, 6]

u_stat, p_value = stats.mannwhitneyu(group1, group2)
print(f"U = {u_stat:.2f}, p = {p_value:.4f}")
```

**Trade-off:** Less statistical power than parametric tests when assumptions are met.

---

32. What is linear regression, and when is it used?

**Answer:**

**Definition:** Linear regression models the linear relationship between a dependent variable (Y) and one or more independent variables (X). It finds the best-fitting line that minimizes the sum of squared residuals.

**Core Concepts:**
- Simple linear regression: one predictor
- Multiple linear regression: multiple predictors
- Minimizes Sum of Squared Errors (SSE)
- Assumes linear relationship between X and Y

**Mathematical Formula:**
$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \epsilon$$

**OLS Coefficients:**
$$\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$$
$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

**When to Use:**
- Predicting continuous outcome
- Understanding variable relationships
- Quantifying impact of features

**Python Example:**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 4.0, 5.8, 8.2, 10.1])

model = LinearRegression()
model.fit(X, y)
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")
```

**Interview Tip:** Always mention LINE assumptions (Linearity, Independence, Normality, Equal variance).

---

33. How do you interpret R-squared and adjusted R-squared in the context of a regression model?

**Answer:**

**Definition:** RÂ² (coefficient of determination) measures the proportion of variance in Y explained by X. Adjusted RÂ² penalizes for adding unnecessary predictors.

**Core Concepts:**
- RÂ² ranges from 0 to 1 (0% to 100% variance explained)
- RÂ² always increases when adding predictors
- Adjusted RÂ² only increases if predictor improves model

**Formulas:**
$$R^2 = 1 - \frac{SS_{residual}}{SS_{total}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

$$Adj. R^2 = 1 - \frac{(1-R^2)(n-1)}{n-k-1}$$

Where: n = sample size, k = number of predictors

**Interpretation:**
| RÂ² Value | Interpretation |
|----------|----------------|
| 0.0 - 0.3 | Weak |
| 0.3 - 0.6 | Moderate |
| 0.6 - 0.9 | Strong |
| > 0.9 | Very strong |

**Example:**
- RÂ² = 0.75 â†’ Model explains 75% of variance in Y
- If adding a variable increases RÂ² but decreases Adj RÂ² â†’ overfitting

**Python Example:**
```python
from sklearn.metrics import r2_score

y_true = [3, 5, 2.5, 7]
y_pred = [2.8, 5.2, 2.3, 7.1]

r2 = r2_score(y_true, y_pred)
print(f"RÂ² = {r2:.3f}")
```

**Interview Tip:** Always use Adjusted RÂ² when comparing models with different numbers of predictors.

---

34. Explain the assumptions underlying linear regression.

**Answer:**

**Definition:** Linear regression requires several assumptions (LINE) to produce valid, unbiased estimates and reliable inference.

**The LINE Assumptions:**

**L - Linearity**
- Relationship between X and Y is linear
- Check: Residual vs fitted plot (no pattern)

**I - Independence**
- Residuals are independent of each other
- Especially important in time series (no autocorrelation)
- Check: Durbin-Watson test

**N - Normality**
- Residuals are normally distributed
- Less critical with large samples (CLT)
- Check: Q-Q plot, Shapiro-Wilk test

**E - Equal variance (Homoscedasticity)**
- Constant variance of residuals across all X values
- Check: Residual vs fitted plot (no funnel shape)
- Test: Breusch-Pagan test

**Additional Assumptions:**
- No multicollinearity among predictors
- No significant outliers/influential points

**Diagnostic Code:**
```python
import statsmodels.api as sm
from scipy import stats

# Fit model
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
residuals = model.resid

# Normality test
print(stats.shapiro(residuals))

# Homoscedasticity
print(sm.stats.diagnostic.het_breuschpagan(residuals, X))
```

**Interview Tip:** Know remedies - log transform for non-linearity, GLS for heteroscedasticity.

---

35. What is multicollinearity, and why is it a problem in regression analyses?

**Answer:**

**Definition:** Multicollinearity occurs when independent variables are highly correlated with each other, making it difficult to isolate individual variable effects.

**Core Concepts:**
- High correlation between predictors
- Inflates standard errors of coefficients
- Makes coefficients unstable and unreliable
- Model predicts well but individual coefficients meaningless

**Problems Caused:**
1. Unstable coefficient estimates
2. Large standard errors
3. Unreliable p-values
4. Difficulty interpreting variable importance
5. Coefficients may flip signs

**Detection Methods:**

**VIF (Variance Inflation Factor):**
$$VIF_j = \frac{1}{1-R_j^2}$$

| VIF Value | Interpretation |
|-----------|----------------|
| 1 | No correlation |
| 1-5 | Moderate |
| >5 | High (problematic) |
| >10 | Severe |

**Python Example:**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])]
print(vif_data)
```

**Remedies:**
- Remove highly correlated variables
- Use PCA to create uncorrelated components
- Ridge regression (L2 regularization)
- Collect more data

---

36. Explain the difference between correlation and causation.

**Answer:**

**Definition:** Correlation measures the statistical association between two variables. Causation means one variable directly affects another. Correlation does NOT imply causation.

**Core Concepts:**

**Correlation:**
- Two variables move together
- Can be positive, negative, or zero
- Symmetric: Corr(X,Y) = Corr(Y,X)
- Doesn't indicate direction of influence

**Causation:**
- X causes Y (X â†’ Y)
- Requires: temporal precedence, covariation, no confounders
- Asymmetric: X causes Y â‰  Y causes X
- Established through experiments, not observation

**Why Correlation â‰  Causation:**

1. **Confounding variable**: Third variable affects both
   - Ice cream sales â†” Drowning deaths (confounder: hot weather)

2. **Reverse causation**: Y might cause X
   - Exercise â†” Health (which causes which?)

3. **Spurious correlation**: Pure coincidence
   - Nicolas Cage films â†” Pool drownings

**Establishing Causation:**
- Randomized Controlled Trials (RCTs)
- Natural experiments
- Instrumental variables
- Difference-in-differences

**Python Example:**
```python
import numpy as np
# High correlation doesn't mean causation
np.random.seed(42)
confounder = np.random.randn(100)  # Hot weather
ice_cream = confounder + np.random.randn(100)*0.5
drowning = confounder + np.random.randn(100)*0.5

print(f"Correlation: {np.corrcoef(ice_cream, drowning)[0,1]:.2f}")
# High correlation, but ice cream doesn't cause drowning!
```

**Interview Tip:** Always mention confounders and experimental design when discussing causation.

---

37. How can you detect and remedy heteroscedasticity in a regression model?

**Answer:**

**Definition:** Heteroscedasticity means non-constant variance of residuals across predicted values. The spread of residuals changes with X or Å¶ (funnel-shaped pattern).

**Core Concepts:**
- Violates "E" in LINE assumptions
- Doesn't bias coefficients
- But: invalidates standard errors and hypothesis tests
- Common in economic/financial data

**Detection Methods:**

**1. Visual: Residual Plot**
```python
import matplotlib.pyplot as plt
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
# Look for funnel shape
```

**2. Breusch-Pagan Test**
```python
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
# p < 0.05 â†’ heteroscedasticity present
```

**3. White Test**
```python
from statsmodels.stats.diagnostic import het_white
white_stat, white_pvalue, _, _ = het_white(residuals, X)
```

**Remedies:**

| Method | Description |
|--------|-------------|
| Log transformation | Transform Y or X |
| WLS | Weighted Least Squares |
| Robust SE | Heteroscedasticity-consistent SE |
| GLS | Generalized Least Squares |

**Code for Robust Standard Errors:**
```python
import statsmodels.api as sm
model = sm.OLS(y, X).fit(cov_type='HC3')  # Robust SEs
print(model.summary())
```

**Interview Tip:** Robust standard errors are the most practical solution in real-world applications.

---

38. What is logistic regression, and how does it differ from linear regression?

**Answer:**

**Definition:** Logistic regression predicts the probability of a binary outcome (0/1) by modeling the log-odds as a linear function of predictors.

**Core Concepts:**
- Output: probability between 0 and 1
- Uses sigmoid function to constrain output
- Optimized using Maximum Likelihood, not OLS
- Classification, not regression (despite name)

**Key Differences:**

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|---------------------|
| Output | Continuous | Probability (0-1) |
| Target | Numeric | Binary/Categorical |
| Function | Linear | Sigmoid |
| Loss | MSE | Log-loss |
| Optimization | OLS | MLE |

**Mathematical Formula:**
$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_nX_n)}}$$

**Log-odds (logit):**
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1X_1 + ...$$

**Interpretation:** Coefficient Î²â‚ = change in log-odds for 1-unit increase in Xâ‚

**Python Example:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]
# Predict classes
predictions = model.predict(X_test)
```

**Interview Tip:** Know how to interpret odds ratios: exp(Î²) = odds ratio.

---

39. What is a time series, and what makes it different from other types of data?

**Answer:**

**Definition:** A time series is a sequence of data points collected at successive, equally spaced time intervals. The ordering and temporal structure matter.

**Core Concepts:**
- Data has natural temporal ordering
- Observations are typically dependent (autocorrelated)
- Can exhibit trend, seasonality, cycles
- Past values help predict future values

**Unique Characteristics:**
| Feature | Time Series | Cross-sectional |
|---------|-------------|-----------------|
| Ordering | Matters | Doesn't matter |
| Independence | Dependent | Independent |
| Splits | Chronological | Random |
| Patterns | Trend, seasonality | N/A |

**Components:**
1. **Trend**: Long-term increase/decrease
2. **Seasonality**: Regular periodic patterns
3. **Cycles**: Irregular fluctuations
4. **Noise**: Random variation

**Decomposition:**
$$Y_t = T_t + S_t + C_t + \epsilon_t$$ (additive)
$$Y_t = T_t \times S_t \times C_t \times \epsilon_t$$ (multiplicative)

**Python Example:**
```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
result = seasonal_decompose(df['value'], model='additive', period=12)
result.plot()
```

**ML Relevance:** Never use random train-test split for time series - always use chronological split.

---

40. Explain autocorrelation and partial autocorrelation in the context of time series.

**Answer:**

**Definition:** Autocorrelation measures correlation of a time series with its lagged values. Partial autocorrelation measures correlation at lag k after removing effects of intermediate lags.

**Core Concepts:**

**ACF (Autocorrelation Function):**
- Correlation between Yâ‚œ and Yâ‚œâ‚‹â‚–
- Includes indirect effects through intermediate lags
- Used to identify MA(q) order

**PACF (Partial Autocorrelation Function):**
- Direct correlation between Yâ‚œ and Yâ‚œâ‚‹â‚–
- Removes effect of intermediate lags
- Used to identify AR(p) order

**Formula:**
$$ACF(k) = \frac{Cov(Y_t, Y_{t-k})}{Var(Y_t)} = \frac{\sum(Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum(Y_t - \bar{Y})^2}$$

**Model Identification:**
| Pattern | ACF | PACF |
|---------|-----|------|
| AR(p) | Tails off | Cuts off at lag p |
| MA(q) | Cuts off at lag q | Tails off |
| ARMA(p,q) | Tails off | Tails off |

**Python Example:**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series, ax=axes[0], lags=20)
plot_pacf(series, ax=axes[1], lags=20)
plt.show()
```

**Interview Tip:** Blue shaded region = 95% confidence interval. Significant lags exceed this band.

---

41. What is stationarity in a time series, and why is it important?

**Answer:**

**Definition:** A stationary time series has statistical properties (mean, variance, autocorrelation) that remain constant over time. It doesn't depend on when you observe it.

**Core Concepts:**

**Strict Stationarity:**
- Joint distribution is time-invariant
- Too strict for practical use

**Weak (Covariance) Stationarity:**
1. Constant mean: E[Yâ‚œ] = Î¼
2. Constant variance: Var(Yâ‚œ) = ÏƒÂ²
3. Covariance depends only on lag: Cov(Yâ‚œ, Yâ‚œâ‚‹â‚–) = Î³(k)

**Why Important:**
- Most forecasting models assume stationarity
- Non-stationary data can give spurious results
- Easier to model and predict
- Statistical properties are meaningful

**Testing for Stationarity:**

**ADF (Augmented Dickey-Fuller) Test:**
- Hâ‚€: Series has unit root (non-stationary)
- Hâ‚: Series is stationary
- p < 0.05 â†’ Reject Hâ‚€ â†’ Stationary

**Python Example:**
```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")

if result[1] < 0.05:
    print("Series is stationary")
else:
    print("Series is non-stationary")
```

**Interview Tip:** Common non-stationary patterns: trends, seasonality, changing variance.

---

42. Describe some methods to make a non-stationary time series stationary.

**Answer:**

**Definition:** Transforming a non-stationary series to remove trends, seasonality, or changing variance to achieve constant statistical properties.

**Methods:**

**1. Differencing**
- Subtract previous value: Yâ‚œ' = Yâ‚œ - Yâ‚œâ‚‹â‚
- Most common method
- d=1: first difference, d=2: second difference

```python
# First difference
df['diff'] = df['value'].diff()

# Seasonal difference
df['seasonal_diff'] = df['value'].diff(12)  # for monthly data
```

**2. Log Transformation**
- Stabilizes changing variance
- Often combined with differencing

```python
df['log'] = np.log(df['value'])
df['log_diff'] = df['log'].diff()
```

**3. Detrending**
- Remove trend component
- Subtract fitted trend line

```python
from scipy import signal
detrended = signal.detrend(df['value'])
```

**4. Seasonal Decomposition**
- Remove both trend and seasonality

```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['value'], period=12)
stationary = result.resid
```

**Summary Table:**
| Problem | Solution |
|---------|----------|
| Trend | Differencing |
| Changing variance | Log transform |
| Seasonality | Seasonal differencing |
| All combined | Log + Seasonal diff |

**Interview Tip:** After transformation, verify stationarity with ADF test.

---

43. What is ARIMA, and how is it used for forecasting time series data?

**Answer:**

**Definition:** ARIMA (AutoRegressive Integrated Moving Average) is a forecasting model combining autoregression, differencing, and moving average components. Written as ARIMA(p, d, q).

**Core Concepts:**

**Components:**
- **AR(p)**: Autoregressive - uses past values
- **I(d)**: Integrated - differencing order
- **MA(q)**: Moving Average - uses past errors

**Parameters:**
| Parameter | Meaning | Identified By |
|-----------|---------|---------------|
| p | AR lags | PACF cutoff |
| d | Differencing order | ADF test |
| q | MA lags | ACF cutoff |

**Mathematical Formula:**
$$Y_t = c + \phi_1Y_{t-1} + ... + \phi_pY_{t-p} + \theta_1\epsilon_{t-1} + ... + \theta_q\epsilon_{t-q} + \epsilon_t$$

**Box-Jenkins Methodology:**
1. **Identification**: Plot ACF/PACF, determine p, d, q
2. **Estimation**: Fit model parameters
3. **Diagnostic**: Check residuals (should be white noise)
4. **Forecasting**: Generate predictions

**Python Example:**
```python
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Manual ARIMA
model = ARIMA(series, order=(1, 1, 1))
fitted = model.fit()
forecast = fitted.forecast(steps=10)

# Auto ARIMA
auto_model = pm.auto_arima(series, seasonal=False, trace=True)
```

**Seasonal ARIMA:** SARIMA(p,d,q)(P,D,Q,s) for seasonal data.

**Interview Tip:** Know AIC/BIC for model selection - lower is better.

---

44. What is the purpose of dimensionality reduction in data analysis?

**Answer:**

**Definition:** Dimensionality reduction reduces the number of features while preserving important information. It transforms high-dimensional data into lower dimensions.

**Core Concepts:**
- Reduces computational complexity
- Helps visualization (project to 2D/3D)
- Removes noise and redundancy
- Addresses curse of dimensionality

**Benefits:**
| Benefit | Explanation |
|---------|-------------|
| Speed | Faster training |
| Storage | Less memory |
| Visualization | Can plot 2D/3D |
| Overfitting | Reduces model complexity |
| Multicollinearity | Creates uncorrelated features |

**Types:**

**1. Feature Selection:**
- Keep subset of original features
- Methods: Filter, Wrapper, Embedded

**2. Feature Extraction:**
- Create new features from combinations
- Methods: PCA, t-SNE, UMAP, Autoencoders

**When to Use:**
- High-dimensional data (images, text)
- Correlated features
- Visualization needs
- Before clustering

**Python Example:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)
print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} features")
```

**Interview Tip:** Trade-off: information loss vs computational gain.

---

45. Explain Principal Component Analysis (PCA) and its applications.

**Answer:**

**Definition:** PCA is an unsupervised technique that transforms data into orthogonal (uncorrelated) components ordered by variance explained. First PC captures maximum variance.

**Core Concepts:**
- Linear transformation
- Finds directions of maximum variance
- Components are uncorrelated
- Dimensionality reduction while preserving variance

**Steps:**
1. Standardize data (mean=0, std=1)
2. Compute covariance matrix
3. Calculate eigenvectors and eigenvalues
4. Sort by eigenvalues (descending)
5. Select top k eigenvectors
6. Project data onto new axes

**Mathematical Foundation:**
- Covariance matrix: C = Xáµ€X / (n-1)
- Solve: Cv = Î»v (eigendecomposition)
- PCâ‚ = direction of maximum variance

**Applications:**
- Image compression
- Noise reduction
- Visualization (reduce to 2D/3D)
- Feature extraction before ML
- Genetic data analysis

**Python Example:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Total variance: {sum(pca.explained_variance_ratio_):.2%}")
```

**Interview Tip:** Always standardize data before PCA. PCA is sensitive to scale.

---

46. How does Factor Analysis differ from PCA?

**Answer:**

**Definition:** Factor Analysis (FA) identifies underlying latent factors that explain observed correlations. PCA finds directions of maximum variance. FA models error, PCA doesn't.

**Core Differences:**

| Aspect | PCA | Factor Analysis |
|--------|-----|-----------------|
| Goal | Maximize variance | Explain correlations |
| Model | No error term | Includes unique error |
| Latent variables | Components | Factors |
| Interpretation | Mathematical | Meaningful constructs |
| Assumptions | None | Factors exist |
| Use case | Dimensionality reduction | Understanding structure |

**Mathematical Models:**

**PCA:**
$$X = TW^T$$ (just transformation)

**Factor Analysis:**
$$X = Lf + \epsilon$$

Where:
- L = factor loadings
- f = latent factors
- Îµ = unique error (specific to each variable)

**When to Use:**

**Use PCA when:**
- Need dimension reduction
- Don't care about interpretability
- Preprocessing for ML

**Use FA when:**
- Want to identify latent constructs
- Building psychological scales
- Exploratory research
- Variables are indicators of underlying factor

**Python Example:**
```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=3, random_state=42)
X_fa = fa.fit_transform(X)
print("Factor loadings:\n", fa.components_)
```

**Interview Tip:** FA is used in psychology (IQ tests), PCA in general ML.

---

47. What is the curse of dimensionality?

**Answer:**

**Definition:** The curse of dimensionality refers to problems that arise when analyzing data in high-dimensional spaces: data becomes sparse, distances become meaningless, and more data is needed.

**Core Concepts:**
- Data becomes sparse as dimensions increase
- Volume increases exponentially
- Distance-based algorithms fail
- Need exponentially more data

**Problems:**

**1. Sparsity**
- Points spread out in high dimensions
- Nearest neighbors become far
- k-NN and clustering fail

**2. Distance Concentration**
- All distances become similar
- Max distance â‰ˆ Min distance
- Similarity measures meaningless

**3. Sample Requirements**
- Need exponentially more data
- 10 points per dimension â†’ 10áµˆ total points

**4. Overfitting**
- More features than samples
- Model memorizes noise

**Mathematical Insight:**
For unit hypercube, 99% of volume is within thin shell near boundary:
$$\text{As } d \to \infty: \frac{V_{inner}}{V_{total}} \to 0$$

**Solutions:**
| Approach | Method |
|----------|--------|
| Dimensionality reduction | PCA, t-SNE, UMAP |
| Feature selection | Remove irrelevant features |
| Regularization | L1/L2 penalties |
| Domain knowledge | Select meaningful features |

**Python Example:**
```python
import numpy as np

# Distance concentration demo
for d in [2, 10, 100, 1000]:
    points = np.random.rand(100, d)
    dists = np.linalg.norm(points - points[0], axis=1)
    ratio = dists.max() / dists.min()
    print(f"d={d}: max/min ratio = {ratio:.2f}")
# Ratio approaches 1 as d increases
```

---

48. What is Singular Value Decomposition (SVD), and how is it used in Machine Learning?

**Answer:**

**Definition:** SVD decomposes any matrix into three matrices: U (left singular vectors), Î£ (singular values), and Váµ€ (right singular vectors). Fundamental for dimensionality reduction.

**Mathematical Formula:**
$$A = U\Sigma V^T$$

Where (for mÃ—n matrix A):
- U: mÃ—m orthogonal matrix (left singular vectors)
- Î£: mÃ—n diagonal matrix (singular values)
- Váµ€: nÃ—n orthogonal matrix (right singular vectors)

**Core Concepts:**
- Works on any matrix (unlike eigendecomposition)
- Singular values = square root of eigenvalues of Aáµ€A
- Truncated SVD â†’ dimensionality reduction
- Numerically stable

**ML Applications:**

| Application | How SVD is Used |
|-------------|-----------------|
| PCA | SVD on centered data |
| LSA | Topic modeling in text |
| Recommender systems | Matrix factorization |
| Image compression | Keep top k singular values |
| Pseudoinverse | Solving linear systems |

**Python Example:**
```python
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Manual SVD
A = np.random.rand(100, 50)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Truncated SVD (keep top k)
k = 10
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Scikit-learn
svd = TruncatedSVD(n_components=10)
X_reduced = svd.fit_transform(A)
```

**Image Compression Example:**
```python
# Keep top k singular values for compression
k = 50
compressed = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
```

**Interview Tip:** PCA is SVD applied to centered data.

---

49. What is A/B testing, and why is it an important tool in statistics?

**Answer:**

**Definition:** A/B testing is a randomized controlled experiment comparing two versions (A and B) to determine which performs better on a specific metric.

**Core Concepts:**
- Control group (A): Current version
- Treatment group (B): New version
- Random assignment eliminates confounders
- Statistical test determines significance

**Why Important:**
- Data-driven decision making
- Establishes causal effect
- Reduces bias and guesswork
- Quantifies improvement

**Process:**
1. Define hypothesis and metric
2. Calculate sample size
3. Randomly assign users
4. Run experiment
5. Analyze with statistical test
6. Make decision

**Common Metrics:**
- Conversion rate
- Click-through rate (CTR)
- Revenue per user
- Time on page
- Bounce rate

**Statistical Framework:**
- Hâ‚€: Î¼A = Î¼B (no difference)
- Hâ‚: Î¼A â‰  Î¼B (B is different)
- Use t-test (continuous) or chi-squared (proportions)

**Python Example:**
```python
from scipy import stats

# Conversion rates
conversions_A, total_A = 120, 1000
conversions_B, total_B = 150, 1000

# Chi-squared test
from scipy.stats import chi2_contingency
observed = [[conversions_A, total_A - conversions_A],
            [conversions_B, total_B - conversions_B]]
chi2, p_value, _, _ = chi2_contingency(observed)
print(f"p-value: {p_value:.4f}")
```

**Interview Tip:** Always define success metric and minimum detectable effect BEFORE running test.

---

50. How do you design an A/B test and determine the sample size required?

**Answer:**

**Definition:** Proper A/B test design requires defining hypotheses, calculating required sample size based on statistical power, and ensuring random assignment.

**Design Steps:**

**1. Define Hypotheses**
- Hâ‚€: No difference between A and B
- Hâ‚: There is a difference
- One-tailed vs two-tailed test

**2. Choose Parameters**
| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| Î± (significance) | 0.05 | False positive rate |
| Î² (Type II error) | 0.20 | False negative rate |
| Power (1-Î²) | 0.80 | Detect effect if exists |
| MDE | Varies | Minimum Detectable Effect |

**3. Sample Size Formula (for proportions):**

$$n = \frac{2(Z_{\alpha/2} + Z_{\beta})^2 \cdot p(1-p)}{\delta^2}$$

Where:
- p = baseline conversion rate
- Î´ = minimum detectable effect (MDE)
- ZÎ±/2 = 1.96 for 95% confidence
- ZÎ² = 0.84 for 80% power

**Python Sample Size Calculator:**
```python
from statsmodels.stats.power import zt_ind_solve_power
import numpy as np

# Parameters
baseline_rate = 0.10  # 10% current conversion
mde = 0.02           # Want to detect 2% increase
alpha = 0.05
power = 0.80

# Effect size (Cohen's h for proportions)
effect_size = 2 * (np.arcsin(np.sqrt(baseline_rate + mde)) - 
                   np.arcsin(np.sqrt(baseline_rate)))

# Sample size per group
n = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, 
                        power=power, alternative='two-sided')
print(f"Sample size per group: {int(np.ceil(n))}")
```

**Interview Tip:** Under-powered tests waste resources; over-powered tests delay decisions.

---

51. What are control and treatment groups in the context of an experiment?

**Answer:**

**Definition:** Control group receives no treatment (baseline), treatment group receives the intervention being tested. Comparison between them reveals causal effect.

**Core Concepts:**

**Control Group:**
- Receives current/standard version (A)
- Serves as baseline for comparison
- Also called "holdout" group

**Treatment Group:**
- Receives new version/intervention (B)
- The group we're testing
- Can have multiple treatments (A/B/C testing)

**Key Requirements:**

| Requirement | Purpose |
|-------------|---------|
| Random assignment | Eliminate selection bias |
| Similar size | Statistical power |
| Same time period | Control external factors |
| Isolation | Prevent contamination |

**Example Scenarios:**
```
Website A/B test:
- Control: Original checkout page
- Treatment: New checkout design

Drug trial:
- Control: Placebo
- Treatment: New medication
```

**Why Both Needed:**
- Without control: Can't distinguish treatment effect from external factors
- Hawthorne effect: People behave differently when observed
- Regression to mean: Extreme values naturally move toward average

**Python Simulation:**
```python
import numpy as np

# Randomly assign users
n_users = 10000
np.random.seed(42)
groups = np.random.choice(['control', 'treatment'], size=n_users)

control_users = groups == 'control'
treatment_users = groups == 'treatment'

print(f"Control: {control_users.sum()}")
print(f"Treatment: {treatment_users.sum()}")
```

**Interview Tip:** Always ensure groups are comparable before experiment starts.

---

52. Explain how you would use hypothesis testing to analyze the results of an A/B test.

**Answer:**

**Definition:** After running an A/B test, hypothesis testing determines if the observed difference is statistically significant or due to random chance.

**Step-by-Step Process:**

**1. State Hypotheses**
- Hâ‚€: pA = pB (no difference)
- Hâ‚: pA â‰  pB (two-tailed) or pA < pB (one-tailed)

**2. Choose Significance Level**
- Î± = 0.05 (standard)
- Î± = 0.01 (stricter)

**3. Select Appropriate Test**

| Metric Type | Test |
|-------------|------|
| Proportions (conversion) | Chi-squared, Z-test |
| Means (revenue) | t-test |
| Non-normal | Mann-Whitney U |

**4. Calculate Test Statistic**

**For proportions:**
```python
from scipy.stats import chi2_contingency

# Data
control = {'converted': 150, 'not_converted': 850}
treatment = {'converted': 180, 'not_converted': 820}

observed = [[150, 850], [180, 820]]
chi2, p_value, dof, expected = chi2_contingency(observed)
```

**For means:**
```python
from scipy import stats
t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
```

**5. Make Decision**
```python
alpha = 0.05
if p_value < alpha:
    print("Reject H0: Significant difference")
else:
    print("Fail to reject H0: No significant difference")
```

**6. Report Results**
```
Results:
- Control conversion: 15.0%
- Treatment conversion: 18.0%
- Absolute lift: +3.0%
- Relative lift: +20%
- p-value: 0.012
- Conclusion: Treatment significantly outperforms control
```

**Interview Tip:** Always report confidence intervals alongside p-values.

---

53. How can you avoid biases when conducting experiments and A/B tests?

**Answer:**

**Definition:** Biases are systematic errors that distort results. Proper experimental design and execution minimize these biases.

**Common Biases and Solutions:**

**1. Selection Bias**
- Problem: Non-random group assignment
- Solution: True randomization, stratified sampling

**2. Survivorship Bias**
- Problem: Only analyzing users who completed
- Solution: Intent-to-treat analysis

**3. Novelty/Primacy Effect**
- Problem: Users react differently to new things
- Solution: Run test long enough for effect to stabilize

**4. Sample Ratio Mismatch (SRM)**
- Problem: Actual split â‰  intended split
- Solution: Check ratio before analyzing results

**5. Peeking/Early Stopping**
- Problem: Stopping when results look significant
- Solution: Pre-define stopping rules, use sequential testing

**6. Multiple Testing**
- Problem: Testing many metrics inflates false positives
- Solution: Bonferroni correction, focus on primary metric

**Best Practices Checklist:**

| Practice | Implementation |
|----------|----------------|
| Pre-registration | Document hypothesis before test |
| Power analysis | Calculate sample size upfront |
| Randomization check | Verify groups are comparable |
| Run full duration | Don't stop early |
| Primary metric | One main metric, others secondary |
| Segment carefully | Pre-specify segments |

**Python Check for SRM:**
```python
from scipy.stats import chisquare

observed = [4950, 5050]  # Actual split
expected = [5000, 5000]  # Expected 50-50

chi2, p = chisquare(observed, expected)
if p < 0.01:
    print("Warning: Sample Ratio Mismatch detected!")
```

**Interview Tip:** Pre-analysis plan (PAP) is gold standard for avoiding bias.

---

54. What defines Bayesian statistics, and how does it differ from frequentist statistics?

**Answer:**

**Definition:** Bayesian statistics treats probability as degree of belief and updates beliefs with data using Bayes' theorem. Frequentist treats probability as long-run frequency.

**Core Differences:**

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Probability | Long-run frequency | Degree of belief |
| Parameters | Fixed but unknown | Random variables |
| Prior knowledge | Not used | Incorporated |
| Result | Point estimate + CI | Posterior distribution |
| Interpretation | p(data\|hypothesis) | p(hypothesis\|data) |

**Bayes' Theorem:**
$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

- **Prior P(Î¸)**: Belief before data
- **Likelihood P(D|Î¸)**: Probability of data given parameter
- **Posterior P(Î¸|D)**: Updated belief after data
- **Evidence P(D)**: Normalizing constant

**Key Concepts:**

**Frequentist:**
- 95% CI: If we repeated experiment many times, 95% of CIs would contain true value
- p-value: Probability of seeing this extreme data if Hâ‚€ true

**Bayesian:**
- 95% Credible Interval: 95% probability parameter is in interval
- Posterior probability: Direct probability of hypothesis

**Python Example:**
```python
import pymc as pm
import numpy as np

# Bayesian coin flip
data = [1, 1, 0, 1, 1, 1, 0, 1]  # 1=heads

with pm.Model() as model:
    # Prior: Beta(1,1) = Uniform
    theta = pm.Beta('theta', alpha=1, beta=1)
    # Likelihood
    y = pm.Bernoulli('y', p=theta, observed=data)
    # Sample posterior
    trace = pm.sample(1000)

pm.plot_posterior(trace)
```

**Interview Tip:** Bayesian is intuitive for decision making; Frequentist for hypothesis testing.

---

55. Explain what a prior, likelihood, and posterior are in Bayesian inference.

**Answer:**

**Definition:** In Bayesian inference, the prior represents initial beliefs, likelihood is the data model, and posterior is the updated belief after observing data.

**Components:**

**1. Prior P(Î¸)**
- Belief about parameter BEFORE seeing data
- Based on domain knowledge or previous studies
- Can be informative or non-informative

**Types of Priors:**
| Type | Description | Example |
|------|-------------|---------|
| Uninformative | No strong belief | Uniform, Jeffreys |
| Weakly informative | Regularizes | Normal(0, 10) |
| Informative | Strong prior knowledge | Based on literature |

**2. Likelihood P(D|Î¸)**
- Probability of observed data given parameter value
- Same as frequentist likelihood
- Comes from statistical model (Normal, Binomial, etc.)

**3. Posterior P(Î¸|D)**
- Updated belief AFTER seeing data
- Combines prior and likelihood
- Used for inference and prediction

**Formula:**
$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

$$P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$$

**Intuitive Example:**
```
Estimating coin fairness:
- Prior: Believe coin is fair â†’ Beta(10, 10)
- Data: Observe 7 heads out of 10 flips
- Likelihood: Binomial(7 | n=10, Î¸)
- Posterior: Beta(10+7, 10+3) = Beta(17, 13)
- Posterior mean: 17/(17+13) = 0.567
```

**Python Example:**
```python
import numpy as np
from scipy import stats

# Prior: Beta(2, 2)
alpha_prior, beta_prior = 2, 2

# Data: 8 successes out of 10
successes, trials = 8, 10

# Posterior: Beta(alpha + successes, beta + failures)
alpha_post = alpha_prior + successes
beta_post = beta_prior + (trials - successes)

posterior = stats.beta(alpha_post, beta_post)
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"95% Credible Interval: {posterior.interval(0.95)}")
```

---

56. Describe a scenario where applying Bayesian statistics would be advantageous.

**Answer:**

**Definition:** Bayesian methods excel when incorporating prior knowledge, handling small samples, updating beliefs incrementally, or needing probability statements about parameters.

**Scenario 1: Medical Diagnosis with Rare Disease**

```
Problem: Test for rare disease (prevalence 0.1%)
- Test sensitivity: 99%
- Test specificity: 95%
- Patient tests positive. What's probability they have disease?

Frequentist approach: Can't directly answer this
Bayesian approach: Use Bayes' theorem
```

```python
# Prior: P(Disease) = 0.001
# Likelihood: P(Positive|Disease) = 0.99
# P(Positive|No Disease) = 0.05

prior_disease = 0.001
sensitivity = 0.99
false_positive = 0.05

# P(Positive)
p_positive = sensitivity * prior_disease + false_positive * (1 - prior_disease)

# Posterior: P(Disease|Positive)
posterior = (sensitivity * prior_disease) / p_positive
print(f"P(Disease|Positive) = {posterior:.2%}")  # ~1.9%
```

**Scenario 2: A/B Testing with Limited Data**

- Traditional A/B test needs large sample
- Bayesian can incorporate prior from previous tests
- Get results faster with uncertainty quantification

**Scenario 3: Sequential Decision Making**

```
Clinical trial monitoring:
- Update belief after each patient
- Stop early if strong evidence of harm/benefit
- No multiple testing correction needed
```

**Scenario 4: Spam Filter**

- Start with prior probabilities for spam words
- Update as user marks emails
- Personalized and adaptive

**When Bayesian is Better:**
| Situation | Why Bayesian |
|-----------|--------------|
| Small samples | Prior adds information |
| Prior knowledge exists | Incorporate formally |
| Sequential data | Update continuously |
| Decision making | Direct probability statements |
| Hierarchical data | Natural framework |

---

57. What is Markov Chain Monte Carlo (MCMC), and where is it used in statistics?

**Answer:**

**Definition:** MCMC is a family of algorithms that sample from probability distributions by constructing a Markov chain whose stationary distribution is the target distribution.

**Why Needed:**
- Posterior distributions often can't be computed analytically
- MCMC generates samples from complex distributions
- Enables Bayesian inference for complex models

**Core Concepts:**

**Markov Chain:**
- Sequence where next state depends only on current state
- Has stationary distribution it converges to

**Monte Carlo:**
- Use random sampling to estimate quantities
- Law of large numbers ensures accuracy

**Popular Algorithms:**

| Algorithm | Description |
|-----------|-------------|
| Metropolis-Hastings | Accept/reject proposals |
| Gibbs Sampling | Sample each variable conditionally |
| Hamiltonian MC | Uses gradients for efficiency |
| NUTS | No-U-Turn Sampler (PyMC default) |

**Metropolis-Hastings Steps:**
1. Start at initial value Î¸â‚€
2. Propose new value Î¸* from proposal distribution
3. Calculate acceptance probability Î± = min(1, p(Î¸*)/p(Î¸))
4. Accept with probability Î±, else stay
5. Repeat many times

**Python Example:**
```python
import numpy as np

def metropolis_hastings(target, n_samples, initial):
    samples = [initial]
    current = initial
    
    for _ in range(n_samples):
        # Propose
        proposed = current + np.random.normal(0, 1)
        # Accept/reject
        if np.random.random() < target(proposed) / target(current):
            current = proposed
        samples.append(current)
    
    return np.array(samples)

# Sample from N(0,1)
target = lambda x: np.exp(-0.5 * x**2)
samples = metropolis_hastings(target, 10000, 0)
```

**Applications:**
- Bayesian inference
- Computational physics
- Machine learning (training BNNs)
- Finance (option pricing)

**Interview Tip:** Know about burn-in (discard initial samples) and thinning (reduce autocorrelation).

---

58. How would you update a Bayesian model with new data?

**Answer:**

**Definition:** Bayesian updating uses yesterday's posterior as today's prior. When new data arrives, apply Bayes' theorem using the current posterior as the new prior.

**Core Concept:**
$$\text{New Posterior} = \frac{\text{Likelihood of New Data} \times \text{Old Posterior}}{\text{Evidence}}$$

**Sequential Bayesian Updating:**

```
Initial: Prior P(Î¸)
After Data 1: Posteriorâ‚ = P(Î¸|Dâ‚)
After Data 2: Posteriorâ‚‚ = P(Î¸|Dâ‚,Dâ‚‚) using Posteriorâ‚ as prior
After Data n: Posteriorâ‚™ = P(Î¸|Dâ‚,...,Dâ‚™)
```

**Key Property:**
- Order of data doesn't matter (for i.i.d. data)
- Batch update = Sequential update (same result)
- Posterior concentrates as more data arrives

**Python Example - Coin Flip:**
```python
import numpy as np
from scipy import stats

def bayesian_update(prior_alpha, prior_beta, successes, failures):
    """Update Beta distribution with new observations"""
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + failures
    return post_alpha, post_beta

# Initial prior: Beta(1, 1) = Uniform
alpha, beta = 1, 1

# Observe data in batches
data_batches = [(7, 3), (5, 5), (8, 2)]  # (heads, tails)

for i, (heads, tails) in enumerate(data_batches):
    alpha, beta = bayesian_update(alpha, beta, heads, tails)
    posterior = stats.beta(alpha, beta)
    print(f"Batch {i+1}: Î±={alpha}, Î²={beta}")
    print(f"  Mean: {posterior.mean():.3f}")
    print(f"  95% CI: {posterior.interval(0.95)}")
```

**Online Learning Example:**
```python
# Streaming data update
class BayesianEstimator:
    def __init__(self, prior_alpha=1, prior_beta=1):
        self.alpha = prior_alpha
        self.beta = prior_beta
    
    def update(self, observation):
        """Update with single observation (0 or 1)"""
        if observation == 1:
            self.alpha += 1
        else:
            self.beta += 1
    
    def get_estimate(self):
        return self.alpha / (self.alpha + self.beta)

estimator = BayesianEstimator()
for obs in [1, 1, 0, 1, 1, 0, 1, 1]:
    estimator.update(obs)
    print(f"After {obs}: estimate = {estimator.get_estimate():.3f}")
```

**Practical Considerations:**
- Conjugate priors make updating easy (closed-form)
- Non-conjugate: Use MCMC with new data
- Can "forget" old data by down-weighting prior

**Interview Tip:** Bayesian online learning naturally handles streaming data.

---

