# Backpropagation Through Time (BPTT)

## Overview

- Backpropagation in RNNs is called **Backpropagation Through Time (BPTT)**
- The RNN is **unfolded across time steps**, creating a computational graph similar to a deep feedforward network
- Gradients are computed by traversing this unfolded graph **backward through time**

---

## Setup: Toy Example

### Dataset

Three reviews, each with 3 words, binary sentiment labels:

| Review | Words | Label |
|--------|-------|-------|
| $X_1$ | cat sat mat | 1 (positive) |
| $X_2$ | rat rat mat | 0 (negative) |
| $X_3$ | cat cat rat | 0 (negative) |

### Vocabulary & One-Hot Encoding

Vocabulary = {cat, mat, rat} → 3 unique words

| Word | One-Hot Vector |
|------|---------------|
| cat | $[1, 0, 0]$ |
| mat | $[0, 1, 0]$ |
| rat | $[0, 0, 1]$ |

---

## RNN Architecture

- **Input dimension:** 3 (one-hot vectors)
- **Hidden units:** 3
- **Output units:** 1 (binary classification)
- **Biases:** Ignored for simplicity

### Weight Matrices

| Weight | Connects | Shape |
|--------|----------|-------|
| $W_{in}$ | Input → Hidden | $3 \times 3$ |
| $W_h$ | Hidden → Hidden (recurrent) | $3 \times 3$ |
| $W_{out}$ | Hidden → Output | $3 \times 1$ |

### Architecture Diagram (Unfolded for 3 time steps)

```
x₁₁ ─→ [W_in] ─→ ⊕ ─→ tanh ─→ o₁
                    ↑
                   h₀ (zeros)
                    
x₁₂ ─→ [W_in] ─→ ⊕ ─→ tanh ─→ o₂
                    ↑
                   o₁ ─→ [W_h]

x₁₃ ─→ [W_in] ─→ ⊕ ─→ tanh ─→ o₃ ─→ [W_out] ─→ σ ─→ ŷ
                    ↑
                   o₂ ─→ [W_h]
```

---

## Forward Propagation Equations

For review $X_i$ with words $x_{i1}, x_{i2}, x_{i3}$:

### Hidden State at Each Time Step

$$o_1 = \tanh(x_{i1} \cdot W_{in} + \mathbf{0} \cdot W_h)$$

$$o_2 = \tanh(x_{i2} \cdot W_{in} + o_1 \cdot W_h)$$

$$o_3 = \tanh(x_{i3} \cdot W_{in} + o_2 \cdot W_h)$$

> General form: $o_t = \tanh(x_{it} \cdot W_{in} + o_{t-1} \cdot W_h)$

### Output & Prediction

$$\hat{y} = \sigma(o_3 \cdot W_{out})$$

Where $\sigma$ is the sigmoid function (binary classification).

### Loss Function (Binary Cross-Entropy)

$$L = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

---

## Backpropagation: Gradient Descent Update Rules

To minimize the loss, update all three weight matrices:

$$W_{out} \leftarrow W_{out} - \eta \cdot \frac{\partial L}{\partial W_{out}}$$

$$W_{in} \leftarrow W_{in} - \eta \cdot \frac{\partial L}{\partial W_{in}}$$

$$W_h \leftarrow W_h - \eta \cdot \frac{\partial L}{\partial W_h}$$

The task is to compute these **three partial derivatives**.

---

## Derivative 1: $\frac{\partial L}{\partial W_{out}}$ (Easiest)

$W_{out}$ only appears in the **final output computation**, so the chain is simple:

$$\frac{\partial L}{\partial W_{out}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_{out}}$$

This is straightforward — identical to backpropagation in a standard feedforward network.

---

## Derivative 2: $\frac{\partial L}{\partial W_{in}}$ (Complex)

$W_{in}$ appears at **every time step** during the unfolded computation. Changing $W_{in}$ affects $o_1$, which affects $o_2$, which affects $o_3$, and ultimately affects $L$.

### Multiple Paths from $L$ to $W_{in}$

There are **three separate paths** through which $W_{in}$ influences $L$:

| Path | Route | Dependency |
|------|-------|-----------|
| **Direct (via $o_3$)** | $L \to \hat{y} \to o_3 \to W_{in}$ | Short-term |
| **Via $o_2$** | $L \to \hat{y} \to o_3 \to o_2 \to W_{in}$ | Medium-term |
| **Via $o_1$** | $L \to \hat{y} \to o_3 \to o_2 \to o_1 \to W_{in}$ | Long-term |

### Full Gradient (Sum Over All Paths)

$$\frac{\partial L}{\partial W_{in}} = \sum_{t=1}^{T} \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_T} \cdot \left( \prod_{k=t+1}^{T} \frac{\partial o_k}{\partial o_{k-1}} \right) \cdot \frac{\partial o_t}{\partial W_{in}}$$

#### Expanded for 3 Time Steps

$$\frac{\partial L}{\partial W_{in}} = \underbrace{\frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_3} \cdot \frac{\partial o_3}{\partial W_{in}}}_{\text{short-term (via } o_3 \text{)}}$$

$$+ \underbrace{\frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_3} \cdot \frac{\partial o_3}{\partial o_2} \cdot \frac{\partial o_2}{\partial W_{in}}}_{\text{medium-term (via } o_2 \text{)}}$$

$$+ \underbrace{\frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_3} \cdot \frac{\partial o_3}{\partial o_2} \cdot \frac{\partial o_2}{\partial o_1} \cdot \frac{\partial o_1}{\partial W_{in}}}_{\text{long-term (via } o_1 \text{)}}$$

#### Compact Summation Form

$$\boxed{\frac{\partial L}{\partial W_{in}} = \sum_{t=1}^{T} \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_T} \cdot \left( \prod_{j=t+1}^{T} \frac{\partial o_j}{\partial o_{j-1}} \right) \cdot \frac{\partial o_t}{\partial W_{in}}}$$

Where $T$ = total number of time steps.

---

## Derivative 3: $\frac{\partial L}{\partial W_h}$ (Same Complexity as $W_{in}$)

$W_h$ also appears at **every time step** (in the recurrent connection), creating the same multi-path situation.

### Multiple Paths from $L$ to $W_h$

| Path | Route |
|------|-------|
| **Via $o_3$** | $L \to \hat{y} \to o_3 \to W_h$ |
| **Via $o_2$** | $L \to \hat{y} \to o_3 \to o_2 \to W_h$ |
| **Via $o_1$** | $L \to \hat{y} \to o_3 \to o_2 \to o_1 \to W_h$ |

### Full Gradient

$$\boxed{\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_T} \cdot \left( \prod_{j=t+1}^{T} \frac{\partial o_j}{\partial o_{j-1}} \right) \cdot \frac{\partial o_t}{\partial W_h}}$$

> Note: The structure is identical to $\frac{\partial L}{\partial W_{in}}$ — only the last factor changes from $\frac{\partial o_t}{\partial W_{in}}$ to $\frac{\partial o_t}{\partial W_h}$.

---

## The Product Term: Key to Understanding RNN Issues

The critical term in both gradients is:

$$\prod_{j=t+1}^{T} \frac{\partial o_j}{\partial o_{j-1}}$$

Since $o_t = \tanh(x_t W_{in} + o_{t-1} W_h)$:

$$\frac{\partial o_t}{\partial o_{t-1}} = \tanh'(\cdot) \cdot W_h$$

The product becomes:

$$\prod_{j=t+1}^{T} \tanh'(\cdot) \cdot W_h$$

| Condition | Result | Problem |
|-----------|--------|---------|
| $\|\tanh'(\cdot) \cdot W_h\| < 1$ | Product $\to 0$ as $T$ grows | **Vanishing gradients** |
| $\|\tanh'(\cdot) \cdot W_h\| > 1$ | Product $\to \infty$ as $T$ grows | **Exploding gradients** |

---

## Training Procedure (Full BPTT Loop)

1. **Initialize** $W_{in}$, $W_h$, $W_{out}$ with random values
2. **For each review** (training example):
   a. **Forward pass:** Feed words one by one, compute $o_1, o_2, \ldots, o_T$, then $\hat{y}$
   b. **Compute loss** $L$
   c. **Backward pass:** Compute $\frac{\partial L}{\partial W_{out}}$, $\frac{\partial L}{\partial W_{in}}$, $\frac{\partial L}{\partial W_h}$
   d. **Update weights** using gradient descent
3. **Repeat** across all reviews for multiple epochs until convergence

---

## Why BPTT Is Different from Standard Backpropagation

| Aspect | Standard Backprop (ANN) | BPTT (RNN) |
|--------|------------------------|------------|
| **Computational graph** | Fixed depth | Unfolded across time steps (variable depth) |
| **Weight sharing** | Each layer has unique weights | Same $W_{in}$, $W_h$ shared across all time steps |
| **Gradient computation** | Single path per weight | **Multiple paths** — sum over all time steps |
| **Gradient formula** | Simple chain rule | Chain rule with **product over time steps** |
| **Key challenge** | Depth of network | Length of sequence (vanishing/exploding gradients) |

---

## Summary

| Derivative | Difficulty | Reason |
|-----------|-----------|--------|
| $\frac{\partial L}{\partial W_{out}}$ | Easy | Only appears at the output; single-path chain rule |
| $\frac{\partial L}{\partial W_{in}}$ | Hard | Appears at every time step; multi-path summation |
| $\frac{\partial L}{\partial W_h}$ | Hard | Appears at every time step; multi-path summation |

$$\text{Key Formula: } \frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_T} \cdot \left( \prod_{j=t+1}^{T} \frac{\partial o_j}{\partial o_{j-1}} \right) \cdot \frac{\partial o_t}{\partial W}$$
