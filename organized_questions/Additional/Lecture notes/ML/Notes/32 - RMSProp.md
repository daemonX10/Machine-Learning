# RMSProp (Root Mean Square Propagation)

## Overview

- **Full Form:** Root Mean Square Propagation
- **Core Idea:** An **improvement over AdaGrad** — uses exponentially weighted moving average (EWMA) of squared gradients instead of a cumulative sum
- **Solves:** AdaGrad's problem of the learning rate decaying to near-zero

---

## Recap: AdaGrad's Problem

In AdaGrad:

$$v_t = v_{t-1} + g_t^2 = \sum_{i=1}^{t} g_i^2$$

- $v_t$ is a **running sum** of all past squared gradients
- It **only grows** — never shrinks
- After many epochs: $v_t \to$ very large → $\frac{\eta}{\sqrt{v_t}} \to$ very small
- **Result:** Updates become negligible, algorithm stalls before reaching the minimum

---

## RMSProp Solution: Exponentially Weighted Moving Average

Instead of summing **all** past squared gradients equally, RMSProp uses EWMA to give **more weight to recent gradients** and **decay older ones**.

### Key Change (Only Difference from AdaGrad)

| | AdaGrad | RMSProp |
|--|---------|---------|
| $v_t$ formula | $v_t = v_{t-1} + g_t^2$ | $v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2$ |
| Past gradients | All equally weighted | Exponentially decaying weights |
| $v_t$ behavior | Monotonically increasing | Bounded / controlled |

---

## Mathematical Formulation

### Update Rules:

$$\boxed{v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2}$$

$$\boxed{w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t}$$

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| $\beta$ | Decay rate (controls how much history to keep) | 0.9 – 0.95 |
| $\eta$ | Learning rate | — |
| $g_t$ | Gradient at step $t$ | — |
| $v_t$ | Exponentially weighted average of squared gradients | — |
| $\epsilon$ | Small constant to prevent division by zero | ~$10^{-8}$ |

---

## Why EWMA Fixes the Problem

### Mathematical Proof (Expansion)

Starting with $v_0 = 0$ and $\beta = 0.95$:

$$v_1 = 0.95 \times 0 + 0.05 \times g_1^2 = 0.05 \cdot g_1^2$$

$$v_2 = 0.95 \times v_1 + 0.05 \times g_2^2 = 0.95 \times 0.05 \cdot g_1^2 + 0.05 \cdot g_2^2$$

$$v_3 = 0.95^2 \times 0.05 \cdot g_1^2 + 0.95 \times 0.05 \cdot g_2^2 + 0.05 \cdot g_3^2$$

**Observation:**
- The coefficient on $g_1^2$ is $0.05 \times 0.95^2 \approx 0.045$ — shrinking rapidly
- The most recent gradient $g_3^2$ has the **largest weight** (0.05)
- Older gradients are effectively **"forgotten"**

### Consequence:

- $v_t$ stays **bounded** — doesn't blow up
- Learning rate $\frac{\eta}{\sqrt{v_t}}$ remains **meaningful**
- Updates continue throughout training → **convergence is achieved**

---

## AdaGrad vs RMSProp Behavior

| Property | AdaGrad | RMSProp |
|----------|---------|---------|
| Convex problems (e.g., linear regression) | Works well, converges | Works well, converges |
| Non-convex problems (neural networks) | **Stalls** before minimum | **Converges** to minimum |
| Learning rate over time | Decays to ~0 | Remains effective |
| $v_t$ growth | Unbounded (monotonic) | Bounded (EWMA) |

> On convex problems, AdaGrad and RMSProp perform nearly **identically** (same trajectory). The difference becomes apparent on **complex neural networks** with non-convex loss surfaces.

---

## Advantages

- Fixes AdaGrad's aggressive learning rate decay
- Works on **both convex and non-convex** optimization problems
- Practically **no significant disadvantages**
- Before Adam, RMSProp was the **most widely used** optimizer for neural networks
- Still competes with Adam — if Adam doesn't give good results, RMSProp is the next choice

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it does** | Adaptive learning rate using EWMA of squared gradients |
| **Improvement over** | AdaGrad (prevents learning rate from vanishing) |
| **Main formula** | $v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2$ |
| **Key hyperparameter** | $\beta \approx 0.9$ to $0.95$ |
| **Advantage** | Converges even on non-convex problems |
| **Disadvantage** | None significant — largely superseded by Adam in practice |
| **When to use** | When Adam doesn't work well; strong alternative optimizer |
