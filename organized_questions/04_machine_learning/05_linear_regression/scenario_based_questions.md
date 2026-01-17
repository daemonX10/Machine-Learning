# Linear Regression Interview Questions - Scenario-Based Questions

## Question 1: Can you discuss the use of spline functions in regression?

### Answer

**Definition:**
Spline regression fits piecewise polynomials connected at "knots" to model complex non-linear relationships while maintaining smoothness. It's more flexible than global polynomial regression and avoids edge instability.

**Core Concepts:**
- **Knots:** Points where polynomial pieces connect
- **Piecewise polynomials:** Different polynomial in each interval
- **Smoothness constraints:** Function and derivatives are continuous at knots
- **Cubic splines:** Most common (degree 3), ensures smooth curves

**When to Use Splines vs Polynomial:**

| Situation | Use |
|-----------|-----|
| Smooth non-linear relationship | Splines |
| Local flexibility needed | Splines |
| Simple curve, few data points | Polynomial |
| Edge stability important | Splines |

**Intuition:**
Think of a flexible ruler that bends smoothly through data points. Unlike a single polynomial that wiggles wildly, splines bend locally without affecting distant regions.

**Practical Relevance:**
- Dose-response curves in medicine
- Age-income relationships (non-linear but smooth)
- Any relationship that changes pattern at certain thresholds

**Interview Tip:**
Mention that splines avoid Runge's phenomenon (oscillation at edges) that plagues high-degree polynomials.

---

## Question 2: Discuss how linear regression can be used for sales forecasting

### Answer

**Definition:**
Linear regression predicts future sales by modeling the relationship between sales (target) and predictors like time, price, marketing spend, and seasonality.

**Approach Framework:**

**Step 1: Feature Engineering**
- **Time features:** trend, month, day_of_week, is_holiday
- **Lag features:** sales_last_week, sales_last_month
- **External:** price, promotion_flag, competitor_price, weather

**Step 2: Model Structure**
$$Sales = \beta_0 + \beta_1 \cdot trend + \beta_2 \cdot price + \beta_3 \cdot promo + \beta_4 \cdot season + \epsilon$$

**Step 3: Handle Non-linearities**
- Log-transform sales (if skewed)
- Add polynomial terms for diminishing returns
- Use adstock for marketing (carryover effect)

**Key Considerations:**

| Challenge | Solution |
|-----------|----------|
| Seasonality | Add month/quarter dummies |
| Trend | Include time index |
| Autocorrelation | Add lag features |
| Diminishing returns | Log transform ad spend |

**Practical Application:**
1. Collect 2-3 years of weekly/daily sales data
2. Create features (time, external factors)
3. Split chronologically (no random split for time series)
4. Fit model, check residuals for patterns
5. Generate forecasts with confidence intervals

**Business Value:**
- Inventory planning
- Budget allocation
- Staffing decisions
- Goal setting

---

## Question 3: How would you approach building a linear regression model to predict customer churn?

### Answer

**Important Clarification:**
Customer churn is typically a **binary classification problem** (churn: yes/no), so **logistic regression** is more appropriate than linear regression. However, you can use linear regression for related continuous targets.

**Scenario A: If interviewer insists on Linear Regression**
- Predict **churn probability score** (0 to 1) or **days until churn**
- Caveat: predictions may fall outside [0,1] range

**Scenario B: Correct Approach (Logistic Regression)**

**Step 1: Define Target**
- Binary: churned (1) vs retained (0)
- Use logistic regression for probability output

**Step 2: Feature Engineering**
| Category | Features |
|----------|----------|
| Behavioral | login_frequency, feature_usage, support_tickets |
| Engagement | days_since_last_login, session_duration |
| Financial | contract_length, payment_delays, plan_type |
| Demographics | tenure, age, location |

**Step 3: Model Building**
```
Churn_Probability = σ(β₀ + β₁·recency + β₂·frequency + β₃·complaints + ...)
```
Where σ is sigmoid function

**Step 4: Interpretation**
- Positive coefficient → increases churn probability
- Use odds ratios for business interpretation

**Interview Tip:**
If asked about "linear regression for churn," politely clarify that logistic regression is more appropriate for binary outcomes, then explain why (bounded probabilities, proper classification metrics).

---

## Question 4: Propose a framework for using regression to evaluate promotional impact on sales

### Answer

**Definition:**
Marketing Mix Modeling (MMM) uses regression to quantify the ROI of different promotional channels and optimize budget allocation.

**Framework:**

**Step 1: Data Collection (Weekly/Daily)**
- Target: Sales volume or revenue
- Predictors:
  - TV_spend, Radio_spend, Digital_spend
  - Price, competitor_price
  - Seasonality (month dummies, holidays)
  - Distribution/availability

**Step 2: Feature Transformations**

| Transformation | Purpose |
|----------------|---------|
| **Log(Sales)** | Handle skewness |
| **Log(Ad_spend)** | Model diminishing returns |
| **Adstock** | Capture carryover effect |

**Adstock Formula:** $Adstock_t = Spend_t + \lambda \cdot Adstock_{t-1}$
- λ = decay rate (how quickly effect fades)

**Step 3: Model**
$$\log(Sales) = \beta_0 + \beta_1\log(TV_{adstock}) + \beta_2\log(Radio_{adstock}) + Controls + \epsilon$$

**Step 4: Interpretation**
- Coefficients = elasticities
- β₁ = 0.1 means 10% increase in TV → 1% increase in sales
- Calculate ROI: (Sales Lift × Margin) / Ad Spend

**Step 5: Optimization**
- Use coefficients to simulate different budget allocations
- Find mix that maximizes total sales given fixed budget

**Key Considerations:**
- Control for baseline sales (what happens without promotion)
- Account for seasonality to avoid false attribution
- Check for multicollinearity between channels

---

## Question 5: Discuss recent advances in optimization algorithms for linear regression

### Answer

**Definition:**
Beyond classical OLS and gradient descent, recent advances focus on scalability, sparsity, and valid inference in high-dimensional settings.

**Key Advances:**

**1. Stochastic Gradient Descent (SGD) Variants**
- **Adam, AdaGrad:** Adaptive learning rates
- **Mini-batch SGD:** Balance between batch and stochastic
- Use case: Very large datasets that don't fit in memory

**2. Coordinate Descent**
- Optimizes one coordinate (coefficient) at a time
- Very efficient for Lasso/Elastic Net
- Used in scikit-learn's Lasso implementation

**3. Proximal Gradient Methods**
- Handle non-smooth penalties (L1)
- ISTA, FISTA (Fast Iterative Shrinkage-Thresholding)

**4. Adaptive Regularization**
- **Adaptive Lasso:** Different penalties for different coefficients
- Better variable selection consistency

**5. Debiased/Double Machine Learning**
- Valid confidence intervals after Lasso selection
- Combines ML for nuisance parameters with classical inference
- Important for causal inference

**6. Distributed Optimization**
- Split data across machines
- Algorithms: ADMM, distributed SGD
- Use case: Big data environments

**Practical Relevance:**

| Scenario | Recommended Approach |
|----------|---------------------|
| Large n, small p | Normal equation or SGD |
| Large p, small n | Coordinate descent (Lasso) |
| Very large n and p | Distributed SGD |
| Causal inference | Double ML |

---

## Question 6: Discuss the role of linear regression in AI for personalized medicine

### Answer

**Definition:**
Linear regression helps model patient-specific treatment responses by relating outcomes to patient characteristics, enabling personalized dosing and treatment selection.

**Applications:**

**1. Dose-Response Modeling**
$$Response = \beta_0 + \beta_1 \cdot Dose + \beta_2 \cdot Age + \beta_3 \cdot Weight + \beta_4 \cdot (Dose \times Genetics)$$

- Interaction terms capture personalized effects
- Different patients may need different doses

**2. Treatment Effect Estimation**
- Estimate how treatment effect varies by patient subgroup
- Heterogeneous treatment effects (HTE)

**3. Biomarker Discovery**
- Use Lasso to identify which genes/proteins predict drug response
- High-dimensional feature selection

**4. Risk Prediction**
- Predict patient outcomes (survival time, readmission risk)
- Combine with clinical decision support

**Key Considerations:**

| Challenge | Approach |
|-----------|----------|
| High-dimensional genomics | Regularized regression (Lasso) |
| Interpretability for doctors | Keep model simple, use coefficients |
| Causal inference | Careful confounding control |
| Small samples | Regularization, Bayesian methods |

**Why Linear Regression (vs Black-box ML):**
- **Interpretability:** Doctors need to understand why
- **Regulatory:** FDA prefers explainable models
- **Trust:** Coefficients can be validated clinically

**Limitations:**
- May miss complex interactions (consider tree-based or neural networks for discovery, then validate with interpretable models)

---

## Question 7: How would you explain linear regression to a non-technical stakeholder?

### Answer

**Simple Explanation:**

"Linear regression finds the best straight line that describes the relationship between things we know (inputs) and something we want to predict (output)."

**Analogy:**
"Imagine plotting house prices against square footage on a graph. Linear regression draws the best-fit line through those points. Once we have that line, given any square footage, we can predict the price by reading off the line."

**Key Points to Communicate:**

**1. What it does:**
- Finds patterns in historical data
- Uses those patterns to make predictions

**2. The output:**
- A formula: *Price = Base + (Rate × Size)*
- Each factor has a "weight" showing its importance

**3. Business value:**
- "For every additional 100 sq ft, price increases by $X"
- Quantifies relationships for decision-making

**4. Limitations (honest communication):**
- Works best when relationships are roughly linear
- Past patterns may not hold in future
- Correlation ≠ causation

**Visual Aid:**
Draw a scatter plot with a line through it. Show:
- Points = actual data
- Line = model's prediction
- Gap between point and line = error

**Avoid:**
- Mathematical formulas
- Technical jargon (coefficients, residuals)
- Overcomplicating

**Key Message:**
"It's a tool that helps us understand how different factors influence an outcome, and lets us make data-driven predictions."

---

## Question 8: How would you use A/B testing to validate a linear regression model in production?

### Answer

**Definition:**
A/B testing validates that the regression model's recommendations actually improve real-world outcomes compared to the current approach.

**Framework:**

**Step 1: Define the Hypothesis**
- H₀: Model recommendations perform same as current method
- H₁: Model recommendations improve the metric

**Step 2: Design the Test**

| Group | Treatment |
|-------|-----------|
| Control (A) | Current approach (no model) |
| Treatment (B) | Use model predictions |

**Step 3: Sample Size Calculation**
- Determine minimum detectable effect
- Calculate required sample size for statistical power (typically 80%)
- Ensure random assignment

**Step 4: Implementation Example**
*Scenario: Model predicts optimal discount percentage*
- Control: Fixed 10% discount to everyone
- Treatment: Model-predicted personalized discount

**Step 5: Run the Test**
- Random 50/50 split of users
- Run for sufficient duration (account for seasonality)
- Track primary metric (e.g., conversion, revenue)

**Step 6: Analyze Results**
- Compare mean outcome between groups
- Calculate confidence interval
- Check statistical significance (p < 0.05)

**Key Considerations:**

| Issue | Solution |
|-------|----------|
| Novelty effect | Run test long enough |
| Seasonality | Run full business cycle |
| Simpson's paradox | Check subgroup results |
| Multiple testing | Adjust p-values |

**Success Criteria:**
- Statistically significant improvement
- Practically meaningful lift (business impact)
- No negative secondary effects

**After Validation:**
- Gradual rollout (10% → 50% → 100%)
- Monitor for drift over time
