# Lecture 25 — Regularization in Deep Learning (L1, L2, Weight Decay)

## 1. Why Do Neural Networks Overfit?

The root cause is **model complexity**:

- More neurons → more connections → more capacity to memorize
- Each neuron can draw a **line** (decision boundary component)
- 1 neuron → 1 line; 10 neurons → 10 lines; 1000 neurons → 1000 lines
- With enough neurons, the network creates **extremely complex decision boundaries** that fit every tiny pattern (noise) in the training data

| # Neurons | Decision Boundary | Behavior |
|---|---|---|
| 1 | Straight line | Underfits |
| 10 | Slightly curved | Reasonable |
| 50 | Complex curve | Starting to overfit |
| 250 | Very complex | Overfitting |
| 1000 | Extremely jagged | Severe overfitting |

---

## 2. Approaches to Reduce Overfitting

### Category A: Increase Data
| Technique | Description |
|---|---|
| **Collect more data** | More diversity → better generalization |
| **Data Augmentation** | Create synthetic data from existing (flip, rotate, crop images) |

### Category B: Reduce Model Complexity
| Technique | Description |
|---|---|
| **Dropout** | Randomly disable neurons during training |
| **Early Stopping** | Stop training when validation loss starts increasing |
| **Regularization** | Add penalty term to loss function to shrink weights |

---

## 3. Regularization — Core Idea

Regularization adds a **penalty term** to the cost function that discourages large weights:

$$J_{\text{new}} = J_{\text{original}} + \underbrace{\lambda \cdot \text{Penalty}}_{\text{Regularization term}}$$

By **shrinking weights toward zero**, the effective number of active neurons is reduced → simpler model → less overfitting.

---

## 4. L2 Regularization (Ridge / Weight Decay)

### Formula

$$J = J_{\text{original}} + \frac{\lambda}{2n} \sum_{i=1}^{n} w_i^2$$

Or equivalently (layer-wise for neural networks):

$$J = J_{\text{original}} + \frac{\lambda}{2n} \sum_{l=1}^{L} \| W^{[l]} \|_F^2$$

where $\| W^{[l]} \|_F^2 = \sum_i \sum_j (w_{ij}^{[l]})^2$ is the **Frobenius norm** (sum of squares of all weights in layer $l$).

### Key Points
- $\lambda$ = **regularization strength** (hyperparameter)
  - $\lambda = 0$ → no regularization (original loss)
  - $\lambda$ too large → underfitting (weights → 0, model too simple)
- $n$ = number of training examples
- **Only weights are penalized, NOT biases** ($b$ terms are excluded)
- L2 is the **most commonly used** regularization in deep learning

### How L2 Reduces Weights (Mathematical Proof)

Standard weight update (gradient descent):

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial J}{\partial w}$$

With L2 regularization, the loss becomes $J + \frac{\lambda}{2} w^2$, so the derivative includes an extra term:

$$\frac{\partial}{\partial w}\left(J + \frac{\lambda}{2} w^2\right) = \frac{\partial J}{\partial w} + \lambda w$$

Substituting into the update rule:

$$w_{\text{new}} = w_{\text{old}} - \alpha \left(\frac{\partial J}{\partial w} + \lambda \cdot w_{\text{old}}\right)$$

Rearranging:

$$\boxed{w_{\text{new}} = w_{\text{old}} \cdot \underbrace{(1 - \alpha \lambda)}_{\text{shrinkage factor}} - \alpha \cdot \frac{\partial J}{\partial w}}$$

Since $\alpha > 0$ and $\lambda > 0$, the factor $(1 - \alpha\lambda) < 1$, so **weights shrink every update**.

> This is why L2 regularization is also called **Weight Decay** — the weight is multiplied by a factor slightly less than 1 at each step.

---

## 5. L1 Regularization (Lasso)

### Formula

$$J = J_{\text{original}} + \frac{\lambda}{2n} \sum_{i=1}^{n} |w_i|$$

### Key Difference from L2

| Property | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Penalty term | $\sum |w_i|$ | $\sum w_i^2$ |
| Effect on weights | Drives many weights **exactly to zero** | Drives weights **close to zero** (but not exactly) |
| Result | **Sparse model** (feature selection) | **Small but non-zero weights** |
| Use in DL | Less common | **More common** |

> **L1 produces sparse models** — effectively eliminates neurons by zeroing their weights. L2 keeps all neurons but with small weights.

---

## 6. Keras Implementation

### L2 Regularization

```python
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_dim=2,
          kernel_regularizer=l2(0.01)),    # λ = 0.01
    Dense(128, activation='relu',
          kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])
```

### L1 Regularization

```python
from tensorflow.keras.regularizers import l1

model = Sequential([
    Dense(128, activation='relu', input_dim=2,
          kernel_regularizer=l1(0.001)),   # λ = 0.001
    Dense(128, activation='relu',
          kernel_regularizer=l1(0.001)),
    Dense(1, activation='sigmoid')
])
```

### L1 + L2 (Elastic Net)

```python
from tensorflow.keras.regularizers import l1_l2

model = Sequential([
    Dense(128, activation='relu', input_dim=2,
          kernel_regularizer=l1_l2(l1=0.001, l2=0.01)),
    Dense(128, activation='relu',
          kernel_regularizer=l1_l2(l1=0.001, l2=0.01)),
    Dense(1, activation='sigmoid')
])
```

**Key parameter:** `kernel_regularizer` — applies penalty to **weights only** (not biases), as expected.

---

## 7. Experimental Results

### Weight Distribution: Without vs With L2 Regularization

| Metric | Without Regularization | With L2 ($\lambda = 0.01$) |
|---|---|---|
| Max weight | ~1.75 | ~0.35 |
| Min weight | ~-2.78 | ~-0.50 |
| Weight range | Wide ($[-3, +2]$) | Narrow ($[-0.5, +0.5]$) |
| Distribution | Spread out | **Concentrated near zero** |
| Decision boundary | Jagged, complex | **Smooth, clean** |
| Overfitting | Severe | **Minimal** |

### Decision Boundary Comparison

- **Without regularization:** Jagged, complex curves trying to fit every training point
- **With L2 regularization:** Smooth boundary that captures the **overall pattern**

---

## 8. L2 vs L1 in Practice (Deep Learning)

| Aspect | L2 Regularization | L1 Regularization |
|---|---|---|
| Weights | Small but non-zero | Many become exactly zero |
| Model type | Dense (all neurons active) | Sparse (many neurons eliminated) |
| DL performance | **Generally better** | Less commonly used |
| Recommendation | **Preferred in deep learning** | Use when you want feature selection |

> **Practical recommendation:** In deep learning, L2 regularization generally gives better results than L1.

---

## 9. Tuning $\lambda$

| $\lambda$ Value | Effect |
|---|---|
| **Too small** (→ 0) | Regularization has no effect → overfitting persists |
| **Just right** | Weights are appropriately shrunk → good generalization |
| **Too large** | Weights shrink too much → model becomes too simple → **underfitting** |

> Typical starting values: $\lambda \in [0.001, 0.01, 0.1]$ — tune via cross-validation.

---

## 10. Summary

1. **Overfitting** in ANNs happens because complex architectures memorize noise
2. **Regularization** adds a penalty term to the loss function to **shrink weights**
3. **L2 (Ridge):** Penalizes $\sum w_i^2$ → weights become small but non-zero → **most used in DL**
4. **L1 (Lasso):** Penalizes $\sum |w_i|$ → weights become exactly zero → sparse models
5. L2 regularization ≈ **Weight Decay**: each update multiplies weight by $(1 - \alpha\lambda) < 1$
6. In Keras: use `kernel_regularizer=l2(λ)` in `Dense` layers
7. **Only weights are regularized, not biases**
8. Tune $\lambda$ carefully: too small → overfitting, too large → underfitting
