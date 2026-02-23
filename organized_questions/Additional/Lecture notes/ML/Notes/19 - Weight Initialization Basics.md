# Lecture 19: Weight Initialization Techniques — What NOT to Do

## 1. Why Weight Initialization Matters

The **first step** in training a neural network is initializing all parameters (weights and biases) to some starting values. This choice critically affects:

| Problem | Consequence |
|---------|-------------|
| **Vanishing gradients** | Gradients shrink to ~0; training stalls |
| **Exploding gradients** | Gradients blow up; training becomes unstable |
| **Slow convergence** | Model takes excessively long to reach a good solution |

> Historically (~2012), deep learning research stalled because of vanishing gradients. Two culprits were identified: **(1) sigmoid activation** and **(2) wrong weight initialization**.

---

## 2. Setup for Discussion

- **Task:** Classification with 2 inputs, hidden layer(s) with multiple neurons, 1 output
- **Test across:** ReLU, Tanh, Sigmoid activations
- **Goal:** Understand why certain initialization strategies fail

---

## 3. Bad Strategy #1: Zero Initialization

### Setting all weights = 0

#### Case A: ReLU Activation

$$z = w_1 x_1 + w_2 x_2 + b = 0 \cdot x_1 + 0 \cdot x_2 + 0 = 0$$
$$a = \text{ReLU}(0) = 0$$

- All activations = 0
- ReLU derivative at $z = 0$ is 0
- Gradient = 0 → **no weight update** → **no training ever**
- Weights remain at 0 forever

#### Case B: Tanh Activation

$$z = 0 \quad \Rightarrow \quad a = \tanh(0) = 0$$
$$\tanh'(0) = 1 - \tanh^2(0) = 1 - 0 = 1$$

But $a = 0$ → when this feeds into the next layer with zero weights → still zero. Derivative eventually produces zero updates. **No training.**

#### Case C: Sigmoid Activation

$$z = 0 \quad \Rightarrow \quad a = \sigma(0) = 0.5$$

Activations are **not zero** (they're 0.5), but all neurons produce the **same activation** (0.5).

> **This is the critical insight for the next strategy...**

### Result: Zero init → Training fails for all activations

---

## 4. Bad Strategy #2: Constant (Non-Zero) Initialization

### Setting all weights = same constant (e.g., 0.5)

#### The Symmetry Problem

When all weights are identical:

1. All neurons in a layer compute the **same weighted sum** $z$
2. All produce the **same activation** $a$
3. During backpropagation, all weight gradients are **identical**
4. All weights receive the **same update**
5. After update, all weights are **still identical** (just a different value)

This continues **forever** — the weights are "tied" together.

#### Mathematical Proof (Sigmoid case)

For two neurons with identical weights receiving inputs $x_1, x_2$:

$$z_{11} = w_{11} x_1 + w_{12} x_2 + b_1$$
$$z_{12} = w_{11} x_1 + w_{12} x_2 + b_1 \quad \text{(same weights → same z)}$$

Since $z_{11} = z_{12}$:
$$a_{11} = \sigma(z_{11}) = \sigma(z_{12}) = a_{12}$$

Gradients:

$$\frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_{11}} \cdot \sigma'(z_{11}) \cdot x_1$$

$$\frac{\partial L}{\partial w_{12}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_{12}} \cdot \sigma'(z_{12}) \cdot x_1$$

Since $a_{11} = a_{12}$ and $z_{11} = z_{12}$, these gradients are **equal** → weights update identically → **symmetry is never broken**.

#### Consequence

| No. of neurons in hidden layer | Effective neurons |
|-------------------------------|-------------------|
| 2 | 1 |
| 10 | 1 |
| 128 | 1 |
| Any $n$ | **1** |

> All neurons behave as a **single neuron** → the network degenerates into a **linear model** (like logistic regression) → **cannot capture non-linearity**.

#### Code Verification

```python
# After training with constant initialization
weights = model.get_weights()[0]
# All weights from input_1 to each neuron → SAME
# All weights from input_2 to each neuron → SAME
# Network acts as single perceptron
```

### Result: Constant init → Model becomes linear, no non-linearity captured

---

## 5. Bad Strategy #3: Small Random Values

### Setting weights = random × 0.01

```python
weights = np.random.randn(fan_in, fan_out) * 0.01
```

Weights are in range ≈ $[-0.03, 0.03]$

#### What Happens (with Tanh)

1. **Inputs** are normalized (≈ mean 0, std 1)
2. **Weighted sum**: $z = \sum w_i x_i$ → very small (product of small × normal)
3. **Tanh(small number) ≈ small number** (near-linear region around 0)
4. All activations cluster **very close to 0**

As this propagates through layers:

| Layer | Activation distribution |
|-------|------------------------|
| Layer 1 | Narrow peak around 0 |
| Layer 2 | Even narrower peak around 0 |
| Layer 3 | Almost all values ≈ 0 |

5. **Gradients also become very small** → vanishing gradient problem
6. With many layers: **training stalls completely**

#### With Sigmoid

- $\sigma(0) = 0.5$, so activations cluster around 0.5 (not exactly 0)
- Slightly better than tanh, but still suffers vanishing gradients in deep networks

#### With ReLU

- Small positive values pass through; negative → 0
- Not zero everywhere, so vanishing gradient is **less severe**
- But convergence is **extremely slow** (thousands of epochs needed)

### Result: Small random init → Vanishing gradient (tanh/sigmoid) or very slow convergence (ReLU)

---

## 6. Bad Strategy #4: Large Random Values

### Setting weights = random × 1.0 (range ≈ [-3, 3])

```python
weights = np.random.randn(fan_in, fan_out) * 1.0
```

#### What Happens (with Tanh/Sigmoid)

1. With 500 inputs: $z = \sum_{i=1}^{500} w_i x_i$ → value in range $[-250, 250]$
2. **Tanh($\pm 250$) ≈ $\pm 1$** → complete saturation
3. All activations are either -1 or +1 (tanh) / 0 or 1 (sigmoid)

**Saturation** → derivatives ≈ 0 → vanishing gradient → **no training**

#### What Happens (with ReLU)

1. ReLU is not saturating in positive direction → output can be 250
2. Large activations → **large gradients** → **large weight updates**
3. Weight updates overshoot → unstable training
4. **Exploding gradient problem**

```
Gradient descent with large gradients:
    ↗ overshoot
  ↙      ↗ overshoot again
↗          ↙ never converges
```

### Result: Large random init → Saturation/vanishing gradient (tanh/sigmoid) or exploding gradient (ReLU)

---

## 7. Summary: What NOT to Do

| Strategy | Problem | Mechanism |
|----------|---------|-----------|
| **All zeros** | No training | Gradients = 0 (ReLU/tanh); symmetry (sigmoid) |
| **All same constant** | Linear model | Symmetry → all neurons identical → single effective neuron |
| **Small random** | Vanishing gradient / slow convergence | Activations collapse to ~0 across layers |
| **Large random** | Vanishing gradient (sigmoid/tanh) or exploding gradient (ReLU) | Saturation or excessive activations |

### The Key Insight

> We need **random** initialization (to break symmetry), but with a **carefully chosen variance** — not too small, not too large. The variance should **depend on the network architecture** (number of inputs/outputs per layer).

This leads to the **Xavier/Glorot** and **He** initialization techniques → covered in the next lecture.
