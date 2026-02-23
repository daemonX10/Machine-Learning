# Lecture 17: Activation Functions — Sigmoid, Tanh, and ReLU

## 1. What is an Activation Function?

An activation function is a **mathematical gate** between the input and output of a neuron.

For a neuron receiving inputs $x_1, x_2$ with weights $w_1, w_2$ and bias $b$:

$$z = w_1 x_1 + w_2 x_2 + b$$
$$a = g(z) \quad \text{where } g \text{ is the activation function}$$

The activation function decides:
- **Whether** a neuron is activated
- **How much** it is activated

---

## 2. Why Are Activation Functions Needed?

**Without activation functions, a neural network can only learn linear relationships** — it degenerates into linear/logistic regression regardless of depth.

### Mathematical Proof

For a 2-layer network with linear activation ($g(z) = z$):

$$a_1 = W_1 \cdot X + b_1$$
$$\hat{y} = W_2 \cdot a_1 + b_2 = W_2(W_1 X + b_1) + b_2 = W'X + b'$$

The output is a **degree-1 polynomial** of input → **linear relationship** → equivalent to linear regression.

With **non-linear** activation (e.g., sigmoid), the composition creates non-linear mappings, enabling the network to capture complex patterns.

> **Universal Approximation Theorem:** A neural network with non-linear activation functions and sufficient hidden units can approximate any continuous function.

---

## 3. Properties of an Ideal Activation Function

| # | Property | Why It Matters |
|---|----------|----------------|
| 1 | **Non-linear** | Enables capturing non-linear patterns in data |
| 2 | **Differentiable** | Required for gradient descent / backpropagation to compute derivatives |
| 3 | **Computationally inexpensive** | Faster training; derivatives should be simple to calculate |
| 4 | **Zero-centered** | Outputs have mean ≈ 0 → normalized data → faster convergence |
| 5 | **Non-saturating** | Doesn't squash inputs to a bounded range → avoids vanishing gradient |

---

## 4. Sigmoid Activation Function

### Formula & Graph

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- **Output range:** $(0, 1)$
- $\sigma(0) = 0.5$
- Large positive $x$ → output ≈ 1
- Large negative $x$ → output ≈ 0

### Derivative

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

- **Maximum value:** $0.25$ at $x = 0$
- Outside $[-3, 3]$: derivative ≈ 0

### Advantages

| Advantage | Detail |
|-----------|--------|
| **Probabilistic interpretation** | Output ∈ (0,1) → can be treated as probability; used in **output layer for binary classification** |
| **Non-linear** | Captures non-linear patterns |
| **Smooth & differentiable** | Easy to differentiate everywhere |

### Disadvantages

| Disadvantage | Detail |
|--------------|--------|
| **Saturating** → Vanishing gradient | For large $\|x\|$, derivative → 0; gradients vanish during backpropagation |
| **Not zero-centered** | All outputs are positive → all gradients have the same sign → constrained weight updates → **slow, zig-zag convergence** |
| **Computationally expensive** | Requires computing exponentials |

### Current Usage

- **Only** used in the **output layer** for **binary classification**
- **Never** used in hidden layers today

---

## 5. Tanh (Hyperbolic Tangent) Activation Function

### Formula & Graph

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- **Output range:** $(-1, 1)$
- $\tanh(0) = 0$
- Shape is similar to sigmoid but shifted and scaled

### Derivative

$$\tanh'(x) = 1 - \tanh^2(x)$$

- Maximum value: $1.0$ at $x = 0$ (higher than sigmoid's 0.25)

### Advantages

| Advantage | Detail |
|-----------|--------|
| **Non-linear** | Captures non-linear patterns |
| **Differentiable** | Smooth derivative everywhere |
| **Zero-centered** ✓ | Outputs range from -1 to 1 → mean ≈ 0 → **faster convergence** than sigmoid |

### Disadvantages

| Disadvantage | Detail |
|--------------|--------|
| **Still saturating** | Squashes to $(-1, 1)$ → vanishing gradient problem persists |
| **Computationally expensive** | Exponentials in the formula |

### Sigmoid vs Tanh

| Property | Sigmoid | Tanh |
|----------|---------|------|
| Output range | $(0, 1)$ | $(-1, 1)$ |
| Zero-centered | ✗ | ✓ |
| Max derivative | 0.25 | 1.0 |
| Vanishing gradient | Yes | Yes (less severe) |
| Convergence speed | Slower | Faster |

**Key improvement:** Tanh solves the **non-zero-centered** problem of sigmoid, but the **vanishing gradient** problem remains.

---

## 6. ReLU (Rectified Linear Unit) Activation Function

### Formula & Graph

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} 0 & \text{if } x \leq 0 \\ x & \text{if } x > 0 \end{cases}$$

### Derivative

$$\text{ReLU}'(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x > 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

In practice, derivative at $x = 0$ is set to 0 or 1 by convention.

### Advantages

| Advantage | Detail |
|-----------|--------|
| **Non-linear** | $\max(0, x)$ is non-linear; combinations of ReLUs can approximate any function |
| **Non-saturating** (positive side) | No upper bound → no vanishing gradient in positive region |
| **Computationally cheap** | No exponentials; just a threshold comparison |
| **Fast convergence** | ~6× faster convergence than sigmoid/tanh |

### Disadvantages

| Disadvantage | Detail |
|--------------|--------|
| **Not differentiable at 0** | Handled pragmatically in code |
| **Not zero-centered** | All outputs ≥ 0; solved using **Batch Normalization** |
| **Dying ReLU problem** | Neurons with negative input permanently output 0 → stop learning |

### Current Usage

- **Default choice** for hidden layers in most neural networks today
- Combined with **Batch Normalization** to handle the zero-centered issue

---

## 7. Summary: When to Use What

| Activation | Where to Use | Key Limitation |
|------------|-------------|----------------|
| **Sigmoid** | Output layer (binary classification only) | Vanishing gradient, not zero-centered |
| **Tanh** | Rarely in hidden layers; replaced by ReLU | Vanishing gradient |
| **ReLU** | Default for all hidden layers | Dying ReLU |
| **Softmax** | Output layer (multi-class classification) | — |
| **Linear** | Output layer (regression) | — |
