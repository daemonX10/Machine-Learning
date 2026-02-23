# Adam Optimizer (Adaptive Moment Estimation)

## Overview

- **Full Form:** Adaptive Moment Estimation
- **Core Idea:** Combines **Momentum** (first moment) + **RMSProp** (second moment) into a single optimizer
- **Status:** Currently the **most powerful and widely used** optimization technique in deep learning

---

## Prerequisites: Evolution of Optimizers

| Optimizer | Key Idea | Limitation |
|-----------|----------|------------|
| **Batch Gradient Descent** | Basic gradient update | Slow convergence |
| **Momentum** | Uses past gradients for velocity → faster descent | Oscillations around minimum |
| **NAG** (Nesterov Accelerated Gradient) | Look-ahead gradient → dampens oscillations | Still struggles with sparse data |
| **AdaGrad** | Per-parameter adaptive learning rate | Learning rate decays to ~0, stalls |
| **RMSProp** | EWMA of squared gradients → controlled LR decay | Lacks momentum component |
| **Adam** | **Momentum + RMSProp combined** | Current best general-purpose optimizer |

### Two Key Innovations Combined

```
Adam = Momentum (velocity / first moment)
     + RMSProp (adaptive learning rate / second moment)
```

---

## Mathematical Formulation

### Step 1: Compute First Moment (Momentum Term)

$$\boxed{m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t}$$

- Exponentially weighted moving average of **gradients**
- Provides the **momentum** / velocity effect

### Step 2: Compute Second Moment (RMSProp Term)

$$\boxed{v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2}$$

- Exponentially weighted moving average of **squared gradients**
- Controls the **per-parameter learning rate**

### Step 3: Bias Correction

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

> **Why bias correction?** Both $m_0 = 0$ and $v_0 = 0$, so early estimates are biased toward zero. Dividing by $(1 - \beta^t)$ corrects this, especially in the first few steps.

### Step 4: Update Weights

$$\boxed{w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t}$$

---

## Hyperparameters

| Parameter | Meaning | Default Value |
|-----------|---------|---------------|
| $\eta$ | Learning rate | 0.001 |
| $\beta_1$ | Decay rate for first moment (momentum) | **0.9** |
| $\beta_2$ | Decay rate for second moment (RMSProp) | **0.999** |
| $\epsilon$ | Small constant to prevent division by zero | $10^{-8}$ |

> These defaults work well in practice. Adam **automatically handles** learning rate adaptation, so extensive LR tuning is generally not needed.

---

## Breaking Down the Formula

| Component | Source | Purpose |
|-----------|--------|---------|
| $m_t$ (first moment) | **Momentum** | Accelerates convergence, smooths gradient direction |
| $v_t$ (second moment) | **RMSProp / AdaGrad** | Adapts learning rate per-parameter |
| Bias correction ($\hat{m}_t$, $\hat{v}_t$) | **Adam-specific** | Corrects initialization bias |
| $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$ | **RMSProp** | Effective per-parameter learning rate |
| $\hat{m}_t$ in place of raw $g_t$ | **Momentum** | Uses smoothed gradient instead of raw gradient |

---

## Algorithm Summary

```
Initialize: m_0 = 0, v_0 = 0, t = 0

For each iteration:
    t = t + 1
    g_t = ∇L(w_t)                                    # Compute gradient
    m_t = β₁ · m_{t-1} + (1 - β₁) · g_t             # Update first moment
    v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²            # Update second moment
    m̂_t = m_t / (1 - β₁ᵗ)                            # Bias-corrected first moment
    v̂_t = v_t / (1 - β₂ᵗ)                            # Bias-corrected second moment
    w_{t+1} = w_t - η / (√v̂_t + ε) · m̂_t            # Update weights
```

---

## Visual Behavior

On sparse/elongated contour plots:

| Optimizer | Behavior |
|-----------|----------|
| **BGD** | Zig-zags, slow convergence |
| **Momentum** | Fast but overshoots, oscillates |
| **NAG** | Dampened oscillations vs Momentum |
| **AdaGrad** | Goes straight but stalls before minimum |
| **RMSProp** | Converges nicely |
| **Adam** | Converges **fastest** — combines straight path (adaptive LR) + speed (momentum) |

---

## Practical Recommendations

| Situation | Recommended Optimizer |
|-----------|-----------------------|
| **Default / Starting point** | **Adam** — works well in most cases |
| Adam not giving good results | Try **RMSProp** |
| RMSProp not working either | Try **Momentum** or **SGD with scheduling** |
| Simple convex problem | Any optimizer works; SGD is sufficient |

> **Rule of thumb:** Start with Adam. If not satisfied, experiment via hyperparameter tuning across optimizers.

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it does** | Combines momentum + adaptive learning rate per parameter |
| **Key formula** | $w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$ |
| **First moment** | $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ |
| **Second moment** | $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ |
| **Default $\beta_1, \beta_2$** | 0.9, 0.999 |
| **Bias correction** | $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$ |
| **Advantage** | Best general-purpose optimizer; fast convergence; handles sparse + non-convex |
| **When to use** | Default choice for ANN, CNN, RNN, and most deep learning tasks |
