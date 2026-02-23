# AdaGrad (Adaptive Gradient)

## Overview

- **Full Form:** Adaptive Gradient
- **Core Idea:** Use **different learning rates** for different parameters instead of a single fixed learning rate
- **Key Innovation:** Automatically adapts the learning rate per-parameter based on the history of gradients

---

## When to Use AdaGrad

| Scenario | Why AdaGrad Helps |
|----------|-------------------|
| **Features with very different scales** (e.g., CGPA: 0–10 vs Salary: 0–200000) | Normalizes the effective update for each parameter |
| **Sparse data** (columns with mostly zeros, e.g., one-hot encoded features like "from_IIT") | Gives larger updates to rarely-updated parameters |

> **Note:** If features have different scales, you can also just normalize the data. AdaGrad is **most useful** for sparse features.

---

## The Problem: Elongated Contour Plots

When data has sparse features:

1. The loss contour plot becomes **elongated** (stretched ellipses) instead of circular
2. Standard optimizers (BGD, Momentum) waste time **zig-zagging** — moving fast along one axis while barely moving along the other
3. The **sparse feature's weight** gets very small gradients (many zeros in the data → derivative is often zero)
4. The **non-sparse feature's weight** gets normal/large gradients

**Result:** The resultant movement is dominated by one direction → slow, inefficient convergence.

### Why Sparse Features Cause Small Gradients

For a single neuron with inputs $x$ (sparse) and a constant $1$ (bias):

$$\frac{\partial L}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \cdot x_i$$

If $x_i = 0$ for most rows → most terms are zero → **small aggregate gradient** → **small weight update**

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \cdot 1$$

Bias input is always 1 → **normal gradient** → **normal update**

---

## AdaGrad Solution: Per-Parameter Learning Rate

**Key Insight:**

| Gradient Size | Learning Rate Adjustment | Effect |
|--------------|--------------------------|--------|
| Large gradient | Make learning rate **smaller** | Reduce overshooting |
| Small gradient | Keep learning rate **larger** | Boost sluggish parameters |

This keeps updates across all parameters at **comparable magnitudes**.

---

## Mathematical Formulation

### Standard Gradient Descent (for reference):

$$w_{t+1} = w_t - \eta \cdot g_t$$

### AdaGrad Update Rule:

$$\boxed{w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t}$$

Where:

$$v_t = v_{t-1} + g_t^2$$

| Symbol | Meaning |
|--------|---------|
| $w_t$ | Weight at step $t$ |
| $\eta$ | Initial (global) learning rate |
| $g_t$ | Gradient at step $t$ |
| $v_t$ | Cumulative sum of squared gradients up to step $t$ |
| $\epsilon$ | Small constant (~$10^{-8}$) to prevent division by zero |

### How It Works

1. At each step, compute the gradient $g_t$
2. Square it and **accumulate** into $v_t$ (running sum of all past squared gradients)
3. Divide the learning rate $\eta$ by $\sqrt{v_t}$
4. Large past gradients → large $v_t$ → **smaller effective learning rate**
5. Small past gradients → small $v_t$ → **larger effective learning rate**

---

## Advantage

- Handles **sparse data** and **features with different scales** efficiently
- Automatically adapts learning rate per parameter — no manual tuning needed
- Performs well on **convex optimization** problems (e.g., linear regression)

---

## Disadvantage (Critical)

> **AdaGrad cannot converge to the global minimum in complex problems.**

### Why?

- $v_t$ is a **monotonically increasing** sum of squared gradients
- As training progresses, $v_t$ grows very large
- $\frac{\eta}{\sqrt{v_t}}$ becomes **vanishingly small**
- Updates effectively stop → algorithm **stalls before reaching the minimum**

$$v_t = g_1^2 + g_2^2 + \cdots + g_t^2 \quad \xrightarrow{t \to \infty} \quad \text{very large} \quad \Rightarrow \quad \frac{\eta}{\sqrt{v_t}} \approx 0$$

### Consequence

- Works for **simple convex problems** (linear regression)
- **Not suitable for complex neural networks** (non-convex optimization)

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it does** | Adapts learning rate per-parameter using accumulated squared gradients |
| **Best for** | Sparse data, features with different scales, convex problems |
| **Main formula** | $w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t$ |
| **Advantage** | Automatic per-parameter learning rate adaptation |
| **Disadvantage** | Learning rate decays too aggressively → fails to converge in deep networks |
| **Successors** | RMSProp (fixes the decay problem), Adam (combines momentum + adaptive LR) |
