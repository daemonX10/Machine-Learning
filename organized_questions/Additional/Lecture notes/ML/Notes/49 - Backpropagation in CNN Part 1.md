# Backpropagation in CNN — Part 1

## Overview

- **Topic:** How backpropagation works on a CNN architecture
- **Difficulty:** Conceptually challenging — rarely covered in textbooks
- **Three-part series:**
  1. **Part 1 (this video):** Overall approach + backprop through the ANN (fully connected) portion
  2. **Part 2:** Backprop through convolution, max pooling, and flatten layers
  3. **Part 3:** Full backprop on a more complex CNN architecture

---

## Simple CNN Architecture for Demonstration

### Architecture

```
[6×6×1]  INPUT (grayscale image)
    │
    ▼  Conv: 1 filter, 3×3, stride 1, no padding
[4×4]   Z1 (pre-activation feature map)
    │
    ▼  ReLU
[4×4]   A1 (activated feature map)
    │
    ▼  Max Pooling: 2×2, stride 2
[2×2]   P1 (pooled feature map)
    │
    ▼  Flatten
[4]     F (flattened vector)
    │
    ▼  Single neuron (sigmoid)
[1]     A2 = ŷ (prediction)
    │
    ▼
    L  (Binary Cross-Entropy Loss)
```

### Shape Flow

| Step | Operation | Output Shape |
|------|-----------|-------------|
| Input | Image $X$ | $6 \times 6$ |
| Filter $W_1$ | — | $3 \times 3$ |
| Convolution | $X * W_1 + b_1$ | $4 \times 4$ |
| ReLU | $\max(0, Z_1)$ | $4 \times 4$ |
| Max Pooling | $2 \times 2$, stride 2 | $2 \times 2$ |
| Flatten | Reshape to 1D | $4 \times 1$ |
| Output neuron | $\sigma(W_2 \cdot F + b_2)$ | $1 \times 1$ |

---

## Trainable Parameters

There are only **two locations** with trainable parameters:

### 1. Convolution Filter ($W_1$, $b_1$)

$$W_1 \in \mathbb{R}^{3 \times 3} \quad (9 \text{ weights})$$
$$b_1 \in \mathbb{R}^{1 \times 1} \quad (1 \text{ bias})$$

### 2. Fully Connected Neuron ($W_2$, $b_2$)

$$W_2 \in \mathbb{R}^{1 \times 4} \quad (4 \text{ weights, one per flattened input})$$
$$b_2 \in \mathbb{R}^{1 \times 1} \quad (1 \text{ bias})$$

### Total

| Parameter | Shape | Count |
|-----------|-------|-------|
| $W_1$ | $3 \times 3$ | 9 |
| $b_1$ | $1 \times 1$ | 1 |
| $W_2$ | $1 \times 4$ | 4 |
| $b_2$ | $1 \times 1$ | 1 |
| **Total** | — | **15** |

---

## Computational Graph (Logical Diagram)

$$X \xrightarrow{\text{Conv}(W_1, b_1)} Z_1 \xrightarrow{\text{ReLU}} A_1 \xrightarrow{\text{MaxPool}} P_1 \xrightarrow{\text{Flatten}} F \xrightarrow{W_2 \cdot F + b_2} Z_2 \xrightarrow{\sigma} A_2 \xrightarrow{} L$$

Where:

| Symbol | Meaning |
|--------|---------|
| $X$ | Input image ($6 \times 6$) |
| $W_1, b_1$ | Convolution filter weights and bias |
| $Z_1$ | Pre-activation feature map = $X * W_1 + b_1$ |
| $A_1$ | Post-activation = $\text{ReLU}(Z_1)$ |
| $P_1$ | Pooled output = $\text{MaxPool}(A_1)$ |
| $F$ | Flattened vector = $\text{Flatten}(P_1)$ |
| $W_2, b_2$ | FC layer weights and bias |
| $Z_2$ | $W_2 \cdot F + b_2$ |
| $A_2$ | $\sigma(Z_2)$ = prediction $\hat{y}$ |
| $L$ | Loss |

---

## Forward Propagation Equations

$$Z_1 = X * W_1 + b_1$$

$$A_1 = \text{ReLU}(Z_1)$$

$$P_1 = \text{MaxPool}(A_1)$$

$$F = \text{Flatten}(P_1)$$

$$Z_2 = W_2 \cdot F + b_2$$

$$A_2 = \sigma(Z_2)$$

---

## Loss Function

For **binary classification** (e.g., cat vs dog), using Binary Cross-Entropy:

### Single Image

$$L = -\left[ y \log(A_2) + (1 - y) \log(1 - A_2) \right]$$

### Batch of $n$ Images

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(A_2^{(i)}) + (1 - y_i) \log(1 - A_2^{(i)}) \right]$$

> For **multi-class** classification, use Softmax + Categorical Cross-Entropy instead.

---

## Backpropagation Goal

Find optimal values of all 15 parameters that **minimize the loss**:

$$\min_{W_1, b_1, W_2, b_2} L$$

Using gradient descent, we need these **four derivatives**:

$$\frac{\partial L}{\partial W_2}, \quad \frac{\partial L}{\partial b_2}, \quad \frac{\partial L}{\partial W_1}, \quad \frac{\partial L}{\partial b_1}$$

> **Key:** $W_1$ is a **matrix** ($3 \times 3$), so $\frac{\partial L}{\partial W_1}$ is also a $3 \times 3$ matrix — each entry tells how much to adjust the corresponding weight.

---

## Two-Part Strategy

Split the architecture into two logical parts for easier analysis:

| Part | Covers | Derivatives |
|------|--------|------------|
| **Part A: ANN portion** | Flatten → FC → Output | $\frac{\partial L}{\partial W_2}$, $\frac{\partial L}{\partial b_2}$ |
| **Part B: CNN portion** | Conv → ReLU → MaxPool → Flatten | $\frac{\partial L}{\partial W_1}$, $\frac{\partial L}{\partial b_1}$ |

This video covers **Part A**. Part B requires understanding backprop through convolution, max pooling, and flatten (covered in Part 2).

---

## Part A: Backprop Through the ANN Portion

### Derivative 1: $\frac{\partial L}{\partial W_2}$

Using the chain rule along the path $W_2 \to Z_2 \to A_2 \to L$:

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial W_2}$$

### Derivative 2: $\frac{\partial L}{\partial b_2}$

Using the path $b_2 \to Z_2 \to A_2 \to L$:

$$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial b_2}$$

> The first two terms ($\frac{\partial L}{\partial A_2}$ and $\frac{\partial A_2}{\partial Z_2}$) are **common** to both derivatives.

---

### Computing Each Term

#### Term 1: $\frac{\partial L}{\partial A_2}$

Starting from the loss (single image, $A_2$ is a scalar):

$$L = -\left[ y \log(A_2) + (1-y) \log(1 - A_2) \right]$$

$$\frac{\partial L}{\partial A_2} = -\left[ \frac{y}{A_2} - \frac{1-y}{1-A_2} \right]$$

$$= \frac{-y(1-A_2) + (1-y)A_2}{A_2(1-A_2)}$$

$$\boxed{\frac{\partial L}{\partial A_2} = \frac{A_2 - y}{A_2(1-A_2)}}$$

#### Term 2: $\frac{\partial A_2}{\partial Z_2}$

Since $A_2 = \sigma(Z_2)$ and the derivative of sigmoid is $\sigma(z)(1 - \sigma(z))$:

$$\boxed{\frac{\partial A_2}{\partial Z_2} = A_2(1 - A_2)}$$

#### Combined: $\frac{\partial L}{\partial Z_2}$

$$\frac{\partial L}{\partial Z_2} = \frac{\partial L}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} = \frac{A_2 - y}{A_2(1 - A_2)} \cdot A_2(1 - A_2)$$

$$\boxed{\frac{\partial L}{\partial Z_2} = A_2 - y}$$

> This clean result ($A_2 - y$ = prediction minus truth) is identical to logistic regression — it's a well-known property of sigmoid + BCE loss.

---

### Term 3a: $\frac{\partial Z_2}{\partial W_2}$

Since $Z_2 = W_2 \cdot F + b_2$, differentiating w.r.t. $W_2$:

$$\frac{\partial Z_2}{\partial W_2} = F$$

### Term 3b: $\frac{\partial Z_2}{\partial b_2}$

$$\frac{\partial Z_2}{\partial b_2} = 1$$

---

### Final Results

#### $\frac{\partial L}{\partial W_2}$

$$\frac{\partial L}{\partial W_2} = (A_2 - y) \cdot F$$

**Shape check:**
- $W_2$ has shape $1 \times 4$, so $\frac{\partial L}{\partial W_2}$ must be $1 \times 4$
- $(A_2 - y)$ is a scalar ($1 \times 1$)
- $F$ has shape $4 \times 1$, so $F^T$ has shape $1 \times 4$

$$\boxed{\frac{\partial L}{\partial W_2} = F^T \cdot (A_2 - y) \quad \in \mathbb{R}^{1 \times 4}}$$

#### $\frac{\partial L}{\partial b_2}$

$$\boxed{\frac{\partial L}{\partial b_2} = A_2 - y \quad \in \mathbb{R}^{1 \times 1}}$$

---

## Derivatives for $W_1$ and $b_1$ (Preview)

For the CNN portion, using the chain rule through the full path:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial F} \cdot \frac{\partial F}{\partial P_1} \cdot \frac{\partial P_1}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial W_1}$$

This requires three new concepts:

| Derivative | Requires Understanding |
|-----------|----------------------|
| $\frac{\partial Z_1}{\partial W_1}$ | **Backprop through Convolution** |
| $\frac{\partial P_1}{\partial A_1}$ | **Backprop through Max Pooling** |
| $\frac{\partial F}{\partial P_1}$ | **Backprop through Flatten** |

> These are covered in **Part 2**.

---

## Mini-Batch Extension

When processing a batch of $m$ images simultaneously, shapes change:

| Quantity | Single Image | Batch of $m$ |
|----------|-------------|-------------|
| $F$ | $4 \times 1$ | $4 \times m$ |
| $A_2$ | $1 \times 1$ | $1 \times m$ |
| $y$ | $1 \times 1$ | $1 \times m$ |
| $A_2 - y$ | $1 \times 1$ | $1 \times m$ |

For batch gradient:

$$\frac{\partial L}{\partial W_2} = \frac{1}{m} F^T \cdot (A_2 - Y) \quad \in \mathbb{R}^{1 \times 4}$$

> The shape of the derivative **always matches** the shape of the parameter being updated.

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Architecture** | 6×6 input → Conv(3×3) → ReLU → MaxPool(2×2) → Flatten → FC(1) → Sigmoid |
| **Total parameters** | 15 (9 + 1 in filter, 4 + 1 in FC) |
| **Loss** | Binary Cross-Entropy |
| **Derivatives computed** | $\frac{\partial L}{\partial W_2} = F^T(A_2 - y)$ and $\frac{\partial L}{\partial b_2} = A_2 - y$ |
| **Key insight** | CNN backprop splits into ANN part (standard) + CNN part (requires special treatment of conv, pooling, flatten) |
| **Next (Part 2)** | Backprop through convolution, max pooling, and flatten operations |
