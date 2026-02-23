# Lecture 21 — Data Scaling in Neural Networks

## 1. The Problem: Unscaled Features

When input features have **very different scales** (e.g., Age: 0–100, Salary: 10,000–1,000,000), neural network training becomes:

- **Very slow** — takes many epochs to converge
- **Unstable** — loss oscillates without converging
- **May not converge at all**

### Why This Happens

During backpropagation, the gradient update focuses disproportionately on the weight connected to the **larger-scale feature** ($w_2$ for salary) and **ignores** the weight connected to the smaller-scale feature ($w_1$ for age).

### Geometric Intuition (Cost Function Contours)

| Scenario | Contour Shape | Training Behavior |
|---|---|---|
| **Unnormalized** inputs (different scales) | Elongated ellipse (non-symmetric) | Gradient oscillates before reaching minimum |
| **Normalized** inputs (same scale) | Circular / symmetric contours | Gradient descends smoothly and quickly to minimum |

> With unnormalized data, the cost function landscape is stretched in one direction, forcing gradient descent to take a zig-zag path. Normalized data produces symmetric contours → direct path to the optimum.

---

## 2. Two Scaling Methods

### Standardization (Z-score Normalization)

$$X_{\text{std}} = \frac{X - \mu}{\sigma}$$

- Transforms data to **mean = 0**, **standard deviation = 1**
- Output range: approximately $[-3, +3]$ (depends on distribution)
- Brings data into a **unit circle**

### Min-Max Normalization

$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

- Transforms data to range **[0, 1]**
- Brings data into a **unit box**

### When to Use Which?

| Condition | Use |
|---|---|
| Min and Max values are **known** (e.g., CGPA: 0–10) | **Normalization** |
| Min/Max **unknown** or data is **normally distributed** | **Standardization** |
| Features already on the **same scale** | No scaling needed |

> **Practical rule:** Always scale your data. In practice, scaling **never hurts** and often helps significantly.

---

## 3. Implementation in Keras / Scikit-Learn

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Only transform, don't fit!
```

After scaling, values lie approximately in $[-1, +1]$ range.

### Key Points

- Apply `fit_transform` on **training data** only
- Apply `transform` (not `fit_transform`) on **test data** — to avoid data leakage
- The model architecture, compilation, and training remain **exactly the same**

---

## 4. Results: Before vs After Scaling

| Metric | Without Scaling | With Scaling |
|---|---|---|
| Validation accuracy | Stuck around 60%, oscillating | Climbs steadily to ~90% |
| Training stability | Loss fluctuates wildly (40–60 range) | Loss decreases smoothly |
| Convergence | May not converge even after 100 epochs | Converges within few epochs |

---

## 5. Summary

1. **Different feature scales → elongated loss contours → slow/unstable training**
2. **Solution:** Scale all features to the same range via **Standardization** or **Normalization**
3. **Standardize** when distribution is unknown; **Normalize** when min/max are known
4. Always scale inputs before training a neural network — it's a **mandatory preprocessing step**
5. Use `StandardScaler` or `MinMaxScaler` from `sklearn.preprocessing`
