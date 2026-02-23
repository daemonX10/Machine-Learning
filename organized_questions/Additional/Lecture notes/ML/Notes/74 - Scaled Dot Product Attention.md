# Scaled Dot Product Attention

## Overview

- **Core Question:** Why do we divide by $\sqrt{d_k}$ in the attention formula?
- **One-word answer:** The **nature of the dot product** — high-dimensional vectors produce high-variance dot products, which destabilize softmax and training

---

## Self Attention Recap

Given Q, K, V matrices, self attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}(Q \cdot K^T) \cdot V$$

### The Difference from the Original Paper

The "Attention Is All You Need" paper uses a slightly different formula:

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V}$$

The only addition: **scaling by** $\frac{1}{\sqrt{d_k}}$ before softmax.

This is why it's called **Scaled Dot Product Attention**.

---

## What Is $d_k$?

$d_k$ = **dimension of the key vectors**

| If key vectors are... | Then $d_k$ = |
|----------------------|---------------|
| 3-dimensional | 3 |
| 64-dimensional | 64 |
| 512-dimensional | 512 |

The dimension of key vectors is determined by the embedding dimension and the weight matrix $W_K$:

$$\text{Embedding } (1 \times d) \times W_K (d \times d_k) = \text{Key vector } (1 \times d_k)$$

> In the original Transformer: $d_k = d_q = d_v = 64$ (per head)

---

## Why Scale? — The Dot Product's Nature

### Key Observation

When you compute $Q \cdot K^T$, behind the scenes you're performing **multiple vector-vector dot products**. Each element of the resulting matrix is a dot product of two vectors.

**Critical property of dot products:**

| Vector Dimension | Variance of Dot Products |
|-----------------|-------------------------|
| Low (e.g., 3) | **Low** variance |
| High (e.g., 512) | **High** variance |

> As vector dimensionality increases, the **variance of their dot products increases linearly**.

### Empirical Demonstration

| Experiment | Vector Dim | Dot Product Variance |
|-----------|-----------|---------------------|
| 1000 random vector pairs | 3D | Low (spread: ~±4) |
| 1000 random vector pairs | 100D | Medium (spread: ~±10) |
| 1000 random vector pairs | 1000D | High (spread: ~±30) |

---

## Why High Variance Is a Problem

### Step 1: High variance → Extreme values in the score matrix

With 512-dimensional vectors, the $Q \cdot K^T$ matrix contains numbers with **huge spread** — some very large, some very small.

### Step 2: Softmax amplifies extremes

Softmax uses exponentials ($e^x$), so it has a polarizing nature:

| Input to softmax | Output behavior |
|-----------------|-----------------|
| Numbers close together (low variance) | Balanced probabilities |
| Numbers far apart (high variance) | **Extreme probabilities** (~0.99 vs ~0.01) |

```python
# Example: low variance input
softmax([2, 3])  # → [0.27, 0.73]  ← balanced

# Example: high variance input
softmax([1, 10]) # → [0.0001, 0.9999]  ← extreme
```

### Step 3: Extreme probabilities → Vanishing gradients

During backpropagation:
- Training **focuses entirely on the large-probability entries**
- Small-probability entries are **ignored** (their gradients vanish)
- Parameters associated with ignored entries **stop updating**
- Result: **unstable, poor training**

### Analogy: Classroom with Tall and Short Kids

- **High variance** = kids have very different heights
- Teacher asks: "Raise your hand" → only tall kids are visible → teacher only answers their doubts
- Short kids are **ignored** → overall class learning suffers
- **Low variance** = kids are similar height → teacher sees everyone → better learning for all

---

## The Solution: Scale to Reduce Variance

### Simple idea: Divide all values by a constant to shrink variance

Given numbers: 10, 20, 30, 40, 50, 60, 70 → Variance = 400

Divide by 10: 1, 2, 3, 4, 5, 6, 7 → Variance = **4** (reduced!)

### But what should the scaling factor be?

---

## Mathematical Derivation of the Scaling Factor

### Setup

Consider dot products of vectors. If vectors are $d$-dimensional:

| Dimension | Expected Variance of Dot Products |
|-----------|----------------------------------|
| 1 | $\text{Var}(X)$ |
| 2 | $2 \cdot \text{Var}(X)$ |
| 3 | $3 \cdot \text{Var}(X)$ |
| $d$ | $d \cdot \text{Var}(X)$ |

> **Variance scales linearly with dimension:** $\text{Var}(\text{dot product}) = d \cdot \text{Var}(X)$

### Goal

Keep variance **constant** regardless of dimension — always $\text{Var}(X)$.

### Using the Variance Scaling Property

If $X$ is a random variable with $\text{Var}(X)$, and $Y = cX$, then:

$$\text{Var}(Y) = c^2 \cdot \text{Var}(X)$$

### Applying the Fix

For $d$-dimensional vectors, variance = $d \cdot \text{Var}(X)$.

Divide every value by $\sqrt{d}$:

$$\text{Var}\left(\frac{\text{dot product}}{\sqrt{d}}\right) = \frac{1}{(\sqrt{d})^2} \cdot d \cdot \text{Var}(X) = \frac{1}{d} \cdot d \cdot \text{Var}(X) = \text{Var}(X)$$

| Dimension | Before Scaling | After Dividing by $\sqrt{d}$ |
|-----------|---------------|------------------------------|
| 1 | $\text{Var}(X)$ | $\text{Var}(X)$ |
| 2 | $2 \cdot \text{Var}(X)$ | $\text{Var}(X)$ ✓ |
| 3 | $3 \cdot \text{Var}(X)$ | $\text{Var}(X)$ ✓ |
| $d$ | $d \cdot \text{Var}(X)$ | $\text{Var}(X)$ ✓ |

> Dividing by $\sqrt{d_k}$ **normalizes the variance** to be independent of vector dimension.

---

## Final Scaled Dot Product Attention Formula

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V}$$

### Process

1. Compute $Q \cdot K^T$ → similarity/score matrix
2. **Scale** by dividing every element by $\sqrt{d_k}$ → reduces variance
3. Apply **softmax** → balanced probabilities (no extreme values)
4. Multiply by $V$ → contextual embeddings

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What is $d_k$?** | Dimension of the key vectors |
| **Why scale?** | High-dimensional dot products have high variance → softmax produces extreme probabilities → vanishing gradients |
| **Why $\sqrt{d_k}$ specifically?** | Variance of dot products grows linearly with $d$; dividing by $\sqrt{d}$ cancels this growth exactly |
| **Mathematical property** | $\text{Var}(cX) = c^2 \cdot \text{Var}(X)$; choosing $c = 1/\sqrt{d_k}$ normalizes variance |
| **Effect** | Stable softmax outputs → stable gradients → better training |
| **Final formula** | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$ |
