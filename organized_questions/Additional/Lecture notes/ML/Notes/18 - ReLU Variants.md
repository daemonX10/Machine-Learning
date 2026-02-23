# Lecture 18: ReLU Variants — Leaky ReLU, Parametric ReLU, ELU, SELU

## 1. The Dying ReLU Problem

### What Is It?

When a neuron's weighted sum $z = \sum w_i x_i + b$ becomes **negative**, ReLU outputs **0**. If this persists, the neuron becomes **permanently dead** — it stops contributing to learning.

- Output is always 0 regardless of input
- Weights never update (gradient is 0)
- The neuron is effectively removed from the network

### Why It's a Problem

- If > 50% of neurons die → network cannot capture data patterns effectively
- **Worst case:** 100% dead neurons → network learns nothing

### Why Dead Neurons Cannot Recover

Once $z < 0$:
- ReLU derivative = 0
- Weight update rule: $W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}$
- Since $\frac{\partial L}{\partial W}$ includes the ReLU derivative (which is 0), the entire gradient = 0
- $W_{\text{new}} = W_{\text{old}}$ → weights freeze → neuron stays dead **permanently**

### Mathematical Proof

For a neuron with activation $a_1 = \text{ReLU}(z_1)$ where $z_1 = w_1 x_1 + w_2 x_2 + b_1$:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_1} \cdot \underbrace{\frac{\partial a_1}{\partial z_1}}_{= 0 \text{ when } z_1 < 0} \cdot \frac{\partial z_1}{\partial w_1}$$

If $z_1 < 0$ → entire chain = 0 → both $w_1$ and $w_2$ stop updating.

### Two Root Causes

| Cause | Mechanism |
|-------|-----------|
| **High learning rate** | Large update overshoots → weights become very negative → $z$ becomes negative in next cycle |
| **Large negative bias** | If bias is highly negative, it dominates → $z < 0$ regardless of input |

### Solutions

1. **Use a lower learning rate**
2. **Initialize bias with a small positive value** (e.g., 0.01)
3. **Use ReLU variants** instead of standard ReLU

---

## 2. Classification of ReLU Variants

```
ReLU Variants
├── Linear Variants (linear transform on negative side)
│   ├── Leaky ReLU
│   └── Parametric ReLU (PReLU)
└── Non-linear Variants (non-linear transform on negative side)
    ├── ELU (Exponential Linear Unit)
    └── SELU (Scaled Exponential Linear Unit)
```

---

## 3. Leaky ReLU

### Formula

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ 0.01x & \text{if } x < 0 \end{cases}$$

### Derivative

$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0.01 & \text{if } x < 0 \end{cases}$$

The key difference: **derivative is 0.01 (not 0)** for negative inputs → small gradient always flows → no dying ReLU.

### Advantages

| Advantage | Detail |
|-----------|--------|
| **Non-saturating** | Unbounded in both directions |
| **Computationally cheap** | No exponentials |
| **No dying ReLU** | Derivative is never zero → always updates |
| **Close to zero-centered** | Negative values are small but non-zero |

### Disadvantage

- The slope 0.01 is **fixed/arbitrary** — why 0.01 and not something else?

---

## 4. Parametric ReLU (PReLU)

### Formula

$$\text{PReLU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases}$$

Where $\alpha$ is a **learnable/trainable parameter** — optimized during training just like weights.

### Key Difference from Leaky ReLU

| | Leaky ReLU | PReLU |
|--|-----------|-------|
| Negative slope | Fixed at 0.01 | Learned from data |
| Parameters | None | $\alpha$ is trainable |
| Flexibility | Less | More — adapts to the dataset |

### Advantages

- All benefits of Leaky ReLU
- **Data-dependent slope** → sometimes better performance

---

## 5. ELU (Exponential Linear Unit)

### Formula

$$\text{ELU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha(e^x - 1) & \text{if } x < 0 \end{cases}$$

Where $\alpha$ is a **hyperparameter** (typically 0.1 to 0.3).

### Derivative

$$\text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \text{ELU}(x) + \alpha & \text{if } x \leq 0 \end{cases}$$

### Graph Behavior

- Positive side: identical to ReLU
- Negative side: smoothly approaches $-\alpha$ as $x \to -\infty$
- **No discontinuity** at $x = 0$ — the function is **always continuous and differentiable**

### Advantages

| Advantage | Detail |
|-----------|--------|
| **Zero-centered** ✓ | Outputs can be negative → mean closer to 0 → **faster convergence** |
| **Better generalization** | Experimentally shown to outperform ReLU on many datasets |
| **No dying ReLU** | Non-zero gradient for negative inputs |
| **Continuous & differentiable** | Smooth everywhere (unlike ReLU's kink at 0) |

### Disadvantage

| Disadvantage | Detail |
|--------------|--------|
| **Computationally expensive** | Requires computing $e^x$ for negative values |

Despite the computational cost, the faster convergence typically **outweighs** the per-computation overhead.

---

## 6. SELU (Scaled Exponential Linear Unit)

### Formula

$$\text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

Where:
- $\lambda \approx 1.0507$
- $\alpha \approx 1.6733$
- Both are **fixed constants** (not trainable, derived mathematically)

### Derivative

$$\text{SELU}'(x) = \lambda \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x & \text{if } x \leq 0 \end{cases}$$

### Key Property: Self-Normalizing

> SELU is **self-normalizing** — activations automatically converge to **zero mean and unit variance** during training.

This means:
- **No need for Batch Normalization** — the activation function itself handles normalization
- Faster convergence
- Better generalization

### Advantages

| Advantage | Detail |
|-----------|--------|
| **Self-normalizing** | Activations auto-normalize → no need for Batch Norm |
| **Faster convergence** | Due to implicit normalization |
| **Good generalization** | Strong test-set performance |
| **No dying ReLU** | Non-zero negative output |

### Disadvantage

- **Relatively new** — not as extensively researched/adopted as ReLU
- May become more widely adopted in the future

---

## 7. Comparison Table

| Property | ReLU | Leaky ReLU | PReLU | ELU | SELU |
|----------|------|------------|-------|-----|------|
| **Formula (x < 0)** | $0$ | $0.01x$ | $\alpha x$ | $\alpha(e^x - 1)$ | $\lambda\alpha(e^x - 1)$ |
| **Dying ReLU** | Yes | No | No | No | No |
| **Zero-centered** | No | ≈ Yes | ≈ Yes | Yes | Yes |
| **Differentiable everywhere** | No | No | No | Yes | Yes |
| **Computationally cheap** | ✓✓ | ✓✓ | ✓✓ | ✓ | ✓ |
| **Self-normalizing** | No | No | No | No | **Yes** |
| **Trainable params** | None | None | $\alpha$ | None | None |
| **Saturating** | No (positive) | No | No | No | No |

---

## 8. Practical Recommendations

```
Default choice: ReLU (with Batch Normalization)
      │
      ├── If dying ReLU is a problem → Leaky ReLU or ELU
      │
      ├── Want data-adaptive slope → PReLU
      │
      ├── Want best convergence without Batch Norm → SELU
      │
      └── Binary classification output → Sigmoid
          Multi-class output → Softmax
          Regression output → Linear
```
