# Model Evaluation Interview Questions - Theory Questions

## Question 1

**What is model evaluation in the context of machine learning?**

**Answer:**

### Definition
Model evaluation is the process of assessing how well a trained model performs on unseen data and comparing different models to select the best one.

### Purpose
- Estimate real-world performance
- Compare alternative models
- Detect overfitting/underfitting
- Guide model improvement

### Interview Tip
Always evaluate on data the model hasn't seen during training.

---

## Question 2

**Explain the difference between training, validation, and test datasets.**

**Answer:**

### Roles

| Dataset | Purpose | Used For |
|---------|---------|----------|
| Training | Fit model parameters | Learning weights |
| Validation | Tune hyperparameters | Model selection |
| Test | Final evaluation | Performance reporting |

### Typical Split
60-70% train, 15-20% validation, 15-20% test

### Interview Tip
Never use test data for any decision-making during development.

---

## Question 3

**What is cross-validation, and why is it used?**

**Answer:**

### Definition
Cross-validation rotates training and validation sets to use all data for both purposes.

### K-Fold Process
1. Split data into K equal parts
2. Train on K-1 folds, validate on 1
3. Repeat K times
4. Average the metrics

### Benefits
- More reliable performance estimates
- Better use of limited data
- Reduces variance of estimates

### Interview Tip
Use stratified K-fold for classification to maintain class balance.

---

## Question 4

**Explain the concept of the ROC curve and AUC.**

**Answer:**

### ROC Curve
Plot of True Positive Rate vs False Positive Rate at all classification thresholds.

### AUC (Area Under Curve)
- 0.5 = Random classifier
- 0.7-0.8 = Fair
- 0.8-0.9 = Good
- 0.9+ = Excellent

### Interpretation
AUC = probability that model ranks a random positive higher than a random negative.

### Interview Tip
ROC/AUC is threshold-independent, useful for comparing models.

---

## Question 5

**What is meant by ‘overfitting’ and ‘underfitting’ in machine learning models?**

**Answer:**

### Overfitting
- Model learns noise in training data
- High training accuracy, low test accuracy
- Too complex model

### Underfitting
- Model fails to capture patterns
- Poor on both training and test
- Too simple model

### Solutions
- Overfitting: Regularization, more data, simpler model
- Underfitting: More features, complex model, longer training

### Interview Tip
Use validation curves to diagnose which problem you have.

---

## Question 6

**What is the difference between explained variance and R-squared?**

**Answer:**

### R-squared
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

### Explained Variance
$$EV = 1 - \frac{Var(y - \hat{y})}{Var(y)}$$

### Key Difference
They differ when predictions are biased.

### Interview Tip
Use adjusted R² when comparing models with different feature counts.

---

## Question 7

**Explain the use of the Mean Squared Error (MSE) in regression models.**

**Answer:**

### Formula
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Properties
- Always non-negative
- Penalizes large errors quadratically
- Units are squared

### Related Metrics
RMSE = √MSE (interpretable units)

### Interview Tip
Use RMSE for interpretation, MAE when outliers are concern.

---

## Question 8

**What is the distinction between macro-average and micro-average in classification metrics?**

**Answer:**

### Comparison

| Macro | Micro |
|-------|-------|
| Average per class | Aggregate all samples |
| Equal class weight | Weight by frequency |

### Interview Tip
Use macro when all classes equally important.

---

## Question 9

**What is the Brier score, and when would you use it?**

**Answer:**

### Definition
Mean squared error for probability predictions:
$$Brier = \frac{1}{n}\sum(p_i - o_i)^2$$

### Interpretation
Lower is better (0 = perfect, 1 = worst)

### Interview Tip
Use for evaluating probability calibration.

---

## Question 10

**Describe how you would use bootstrapping in model evaluation.**

**Answer:**

### Method
1. Sample with replacement from data
2. Train model on sample
3. Evaluate on out-of-bag samples
4. Repeat many times (100-1000)
5. Compute confidence intervals

### Interview Tip
Bootstrapping gives confidence intervals for metrics.

---

## Question 11

**What are the trade-offs between the different model evaluation metrics?**

**Answer:**

### Classification Trade-offs

| Metric | Pros | Cons |
|--------|------|------|
| Accuracy | Simple | Misleading for imbalanced |
| Precision | FP focus | Ignores FN |
| Recall | FN focus | Ignores FP |
| F1 | Balances P & R | Ignores TN |
| AUC | Threshold-independent | Less interpretable |

### Interview Tip
Choose metric based on business cost of errors.

---

## Question 12

**Explain the concept of p-value in the context of model evaluation.**

**Answer:**

### Definition
Probability of observing results as extreme, assuming null hypothesis is true.

### In Model Evaluation
- Compare models statistically
- McNemar's test for classifiers
- p < 0.05: Typically significant

### Interview Tip
Statistical significance ≠ practical significance.

---

## Question 13

**What is a receiver operating characteristic (ROC) curve, and what does it tell us?**

**Answer:**

### Components
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Each point = different threshold

### Interpretation
- Diagonal = random classifier
- Upper left = perfect classifier

### Interview Tip
Use PR curves for highly imbalanced datasets.

---

## Question 14

**Describe model selection criteria based on AIC and BIC.**

**Answer:**

### Formulas
- AIC = 2k - 2ln(L)
- BIC = k·ln(n) - 2ln(L)

### Comparison
- AIC: Less penalty, favors complex models
- BIC: More penalty, favors simpler models

### Interview Tip
Lower AIC/BIC is better.

---

## Question 15

**What is the Elbow Method, and how is it used to evaluate models?**

**Answer:**

### Application
Determining optimal number of clusters in K-Means.

### Process
1. Plot k vs inertia
2. Find the "elbow" where improvement slows
3. Choose k at the elbow point

### Interview Tip
Combine with silhouette score for reliability.

---

## Question 16

**What is the best way to evaluate a recommendation system?**

**Answer:**

### Offline Metrics
- Precision@K, Recall@K
- NDCG (rank-aware)
- MAP (Mean Average Precision)

### Online Metrics
- Click-through rate
- Conversion rate

### Interview Tip
A/B testing is crucial for recommendation systems.

---

## Question 17

**Describe a method for evaluating the performance of a clustering algorithm.**

**Answer:**

### Internal Metrics (No Labels)
- Silhouette Score: Cohesion vs separation
- Davies-Bouldin Index: Lower is better
- Calinski-Harabasz: Higher is better

### Interview Tip
Use silhouette score for unsupervised evaluation.

---

## Question 18

**Describe how you would set up an A/B test to evaluate changes in a machine learning model.**

**Answer:**

### Steps
1. Define hypothesis and success metric
2. Calculate sample size needed
3. Randomize users into control/treatment
4. Deploy old model (A) and new model (B)
5. Collect data for sufficient duration
6. Analyze with statistical tests

### Interview Tip
Ensure statistical power before running test.

---

## Question 19

**How does transfer learning affect the way we evaluate models?**

**Answer:**

### Considerations
- Evaluate on target domain specifically
- Compare with domain-specific baseline
- Check for negative transfer

### Interview Tip
Pre-trained features may not transfer to very different domains.

---

## Question 20

**What are ensemble learning models, and how do their evaluation strategies differ?**

**Answer:**

### Types
- Bagging (Random Forest)
- Boosting (XGBoost)
- Stacking

### Evaluation Differences
- Out-of-bag (OOB) error for bagging
- Validation during boosting rounds
- Nested CV for stacking

### Interview Tip
OOB error is a free validation estimate for Random Forest.

---

## Question 21

**Explain adversarial validation and where it might be used.**

**Answer:**

### Definition
Train a classifier to distinguish train from test data.

### Process
1. Label train data as 0, test as 1
2. Train binary classifier
3. If AUC ≈ 0.5 → distributions similar
4. If AUC >> 0.5 → distribution shift exists

### Interview Tip
High adversarial AUC indicates potential problem.

---

## Question 22

**What is the concept of ‘model drift’, and how do you measure it?**

**Answer:**

### Types
- **Concept drift**: Relationship between X and y changes
- **Data drift**: Input distribution changes

### Detection Methods
- Monitor prediction distributions
- Track performance metrics over time
- Statistical tests (KS test, PSI)

### Interview Tip
Set up monitoring and alerting for production models.

---

