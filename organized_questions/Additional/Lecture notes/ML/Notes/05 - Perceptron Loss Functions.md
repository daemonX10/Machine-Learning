# Perceptron Loss Functions

## 1. Problems with the Perceptron Trick

| Problem | Description |
|---------|-------------|
| **No guarantee of best line** | Different random seeds → different lines; no way to compare which is better |
| **Cannot quantify performance** | No metric to measure how good the current line is |
| **Convergence issues** | Possible (though unlikely) to never converge due to random point selection |

> The Perceptron Trick is a *jugaad* (hack) — works ~98% of the time but is not principled.

### Why We Need Loss Functions
A **loss function** provides a **number** that quantifies how bad the model is performing → allows optimization toward the **minimum loss** → guarantees finding the best parameters.

---

## 2. What is a Loss Function?

> A loss function is a mathematical function of the model parameters ($w_1, w_2, b$) that measures how poorly the model is performing.

$$L = f(w_1, w_2, b)$$

- Each set of $(w_1, w_2, b)$ defines a **different line**
- The loss function assigns a **score** to each line
- **Goal**: find $(w_1, w_2, b)$ that **minimizes** $L$

$$\arg\min_{w_1, w_2, b} L(w_1, w_2, b)$$

### Common Loss Functions in ML

| Algorithm | Loss Function |
|-----------|--------------|
| Linear Regression | Mean Squared Error (MSE) |
| Logistic Regression | Log Loss (Binary Cross-Entropy) |
| SVM | Hinge Loss |
| Perceptron | Perceptron Loss (modified Hinge Loss) |

---

## 3. Building Intuition for Loss Functions

### Attempt 1: Count Misclassified Points

$$L = \text{Number of misclassified points}$$

- **Problem**: Treats all misclassifications equally
- A point far from the line is a bigger error than one close to it

### Attempt 2: Sum of Distances of Misclassified Points

$$L = \sum_{\text{misclassified}} d_i$$

where $d_i$ is the perpendicular distance from point $i$ to the line.

- **Better**: Accounts for magnitude of error
- **Problem**: Computing perpendicular distance is complex

### Attempt 3: Use Dot Product as Proxy for Distance

Instead of actual distance, substitute the point's coordinates into the line equation:

For point $(x_1, x_2)$ and line $w_1 x_1 + w_2 x_2 + b = 0$:

$$\text{proxy} = |w_1 x_1 + w_2 x_2 + b|$$

This is **directly proportional** to the actual distance and much simpler to compute (just a dot product).

---

## 4. Perceptron Loss Function (Hinge Loss Variant)

### Formula

$$L(w_1, w_2, b) = \frac{1}{n} \sum_{i=1}^{n} \max\left(0,\; -y_i \cdot f(\mathbf{x}_i)\right)$$

Where:
- $n$ = number of data points
- $y_i \in \{+1, -1\}$ = true label
- $f(\mathbf{x}_i) = w_1 x_{i1} + w_2 x_{i2} + b$ = raw score (dot product)

> Note: Labels must be $\{+1, -1\}$, not $\{1, 0\}$ for this formula.

### Breakdown of $\max(0, -y_i \cdot f(\mathbf{x}_i))$

| Scenario | $y_i$ | $f(\mathbf{x}_i)$ | $-y_i \cdot f(\mathbf{x}_i)$ | $\max(0, \ldots)$ | Meaning |
|----------|--------|------|-------------|---------|---------|
| Correct: positive point in positive region | +1 | > 0 | negative | **0** | No loss |
| Correct: negative point in negative region | -1 | < 0 | negative | **0** | No loss |
| Wrong: positive point in negative region | +1 | < 0 | positive | **positive** | Loss! |
| Wrong: negative point in positive region | -1 | > 0 | positive | **positive** | Loss! |

### Key Insight
- **Correctly classified points** contribute **zero** to the loss
- **Misclassified points** contribute a **positive value** proportional to how far they are from the line
- The further a misclassified point from the line → the larger its contribution → **magnitude matters**

---

## 5. Optimization with Gradient Descent

To minimize $L(w_1, w_2, b)$, use **Gradient Descent**:

### Update Rules

$$w_1^{\text{new}} = w_1^{\text{old}} + \eta \cdot \frac{\partial L}{\partial w_1}$$

$$w_2^{\text{new}} = w_2^{\text{old}} + \eta \cdot \frac{\partial L}{\partial w_2}$$

$$b^{\text{new}} = b^{\text{old}} + \eta \cdot \frac{\partial L}{\partial b}$$

> **Note**: Plus sign because the derivative of the max function already includes the negative sign.

### Partial Derivatives

Applying chain rule to $L = \frac{1}{n}\sum \max(0, -y_i \cdot f(\mathbf{x}_i))$:

For each point $i$, if $-y_i \cdot f(\mathbf{x}_i) > 0$ (misclassified):

$$\frac{\partial L}{\partial w_1} = -y_i \cdot x_{i1}$$

$$\frac{\partial L}{\partial w_2} = -y_i \cdot x_{i2}$$

$$\frac{\partial L}{\partial b} = -y_i$$

If correctly classified: derivative = 0.

---

## 6. Python Implementation with Loss Function

```python
import numpy as np

def perceptron_loss(X, y, lr=0.1, epochs=1000):
    # y must be +1 / -1
    w1, w2, b = 1.0, 1.0, 1.0
    
    for _ in range(epochs):
        for i in range(X.shape[0]):
            z = w1 * X[i, 0] + w2 * X[i, 1] + b
            
            if -y[i] * z > 0:  # misclassified
                w1 = w1 + lr * y[i] * X[i, 0]
                w2 = w2 + lr * y[i] * X[i, 1]
                b  = b  + lr * y[i]
    
    return w1, w2, b
```

---

## 7. Perceptron as a Flexible Mathematical Model

The perceptron's design allows swapping the **activation function** and **loss function** to produce different algorithms:

### Configuration Table

| Activation Function | Loss Function | Equivalent Algorithm | Output Type |
|---------------------|---------------|---------------------|-------------|
| **Step** | **Hinge Loss** | **Perceptron** | Binary: $\{0, 1\}$ or $\{+1, -1\}$ |
| **Sigmoid** | **Binary Cross-Entropy** | **Logistic Regression** | Probability: $[0, 1]$ |
| **Softmax** | **Categorical Cross-Entropy** | **Softmax Regression** | Multi-class probabilities |
| **Linear** (identity) | **MSE** | **Linear Regression** | Continuous number |

---

### Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- Maps any real number to $(0, 1)$
- Used when you want **probability** output

### Binary Cross-Entropy Loss

$$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]$$

### Softmax Function

$$P(y = k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

- Generalizes sigmoid to **multiple classes**
- Output is a probability distribution over $K$ classes

### Categorical Cross-Entropy Loss

$$L = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)$$

### Mean Squared Error (for Regression)

$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

---

## 8. Summary Diagram

```
                    PERCEPTRON MODEL
                   ┌─────────────────┐
   Inputs ──► Σ ──►│ Activation Func │──► Output
  (x, w, b)       │   (swappable)   │
                   └────────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  Loss Function │ ← (also swappable)
                    │  (swappable)   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │   Optimizer    │
                    │ (Grad Descent) │
                    └────────────────┘
```

---

## 9. Key Takeaways

1. **Perceptron Trick** is a heuristic — cannot guarantee best solution or quantify quality
2. **Loss functions** solve this by providing a measurable, minimizable objective
3. **Perceptron Loss** = $\frac{1}{n}\sum \max(0, -y_i \cdot f(\mathbf{x}_i))$ — only misclassified points contribute
4. Use **Gradient Descent** with partial derivatives to minimize the loss
5. The perceptron is a **highly flexible model** — by swapping activation + loss:
   - Step + Hinge → **Perceptron**
   - Sigmoid + Log Loss → **Logistic Regression**
   - Softmax + Categorical CE → **Multi-class Classification**
   - Linear + MSE → **Linear Regression**
6. This flexibility is what makes the perceptron the **building block** of all neural networks
