# Lecture 27: Optimizers in Deep Learning — Overview

## 1. Role of Optimizers

An **optimizer** is the algorithm used to find the optimal values of weights and biases in a neural network that **minimize the loss function**.

```
Goal: Find W*, b* = argmin L(W, b)
      where L = loss function
```

### Training Process
1. Initialize weights randomly
2. Forward pass → compute predictions → compute loss
3. **Optimizer** updates weights iteratively to reduce loss
4. Repeat until convergence

---

## 2. Gradient Descent — The Base Optimizer

### Update Rule

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

Where:
- $\eta$ = learning rate (step size)
- $\frac{\partial L}{\partial w}$ = gradient of loss w.r.t. weight

### Three Variants

| Variant | Data per Update | Updates per Epoch | Characteristics |
|---------|----------------|-------------------|-----------------|
| **Batch GD** | Entire dataset | 1 | Smooth convergence, slow |
| **Stochastic GD (SGD)** | 1 sample | $N$ (dataset size) | Noisy, fast updates, can escape local minima |
| **Mini-batch GD** | Batch of $B$ samples | $N/B$ | Balance of both, most commonly used |

---

## 3. Problems with Vanilla Gradient Descent

### Problem 1: Choosing the Right Learning Rate

| $\eta$ too small | $\eta$ too large |
|-----------------|-----------------|
| Painfully slow convergence | Overshooting, divergence |
| May never reach minimum | Unstable training |

> Finding the optimal $\eta$ for a given dataset is difficult and often requires trial-and-error.

### Problem 2: Learning Rate Scheduling Limitations

- **Learning rate scheduling** adjusts $\eta$ during training according to a predefined schedule
- **Problem**: Schedule must be defined **before** training → it's not adaptive to the data
- What works for one dataset may not work for another

### Problem 3: Same Learning Rate for All Parameters

$$w_1^{new} = w_1^{old} - \eta \cdot \frac{\partial L}{\partial w_1}$$
$$w_2^{new} = w_2^{old} - \eta \cdot \frac{\partial L}{\partial w_2}$$

- Same $\eta$ is used for **all** parameters
- Different dimensions of the loss surface may require different step sizes
- Cannot assign per-parameter learning rates in vanilla GD

### Problem 4: Local Minima

- Neural network loss surfaces are **non-convex** → multiple minima exist
- GD can get **stuck in local minima** and return sub-optimal solutions
- SGD has slightly better chance of escaping (due to noisy updates)

### Problem 5: Saddle Points

- A **saddle point** is where the slope is zero but it's not a minimum
  - One direction curves upward, another curves downward
- Gradient ≈ 0 at saddle points → updates become negligible
- Training **stalls** in these flat regions

---

## 4. Summary of Challenges

| # | Problem | Consequence |
|---|---------|-------------|
| 1 | Right learning rate selection | Slow or unstable training |
| 2 | Static learning rate schedule | Not adaptive to data |
| 3 | Single learning rate for all params | Cannot optimize per-dimension |
| 4 | Local minima | Sub-optimal solution |
| 5 | Saddle points | Training stalls |

---

## 5. Advanced Optimizers (Roadmap)

These optimizers address the above problems by modifying the gradient descent update rule:

| Optimizer | Key Idea |
|-----------|----------|
| **SGD with Momentum** | Accumulates past gradients for faster convergence |
| **NAG (Nesterov Accelerated Gradient)** | Look-ahead gradient for reduced oscillation |
| **AdaGrad** | Per-parameter adaptive learning rates |
| **RMSProp** | Fixes AdaGrad's diminishing learning rate |
| **Adam** | Combines Momentum + RMSProp (most popular) |

> **Prerequisite:** Understanding **Exponentially Weighted Moving Average (EWMA)** is essential before studying these optimizers.

---

## 6. Key Takeaways

1. Gradient descent is the foundation — all advanced optimizers are modifications of it
2. The 5 core problems (LR selection, scheduling, per-param LR, local minima, saddle points) motivate advanced optimizers
3. Advanced optimizers are not entirely new algorithms — they are **incremental improvements** over GD
4. Understanding EWMA is a prerequisite for Momentum, NAG, RMSProp, and Adam
