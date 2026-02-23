# Lecture 16: Vanishing and Exploding Gradient Problem

## 1. What is the Vanishing Gradient Problem?

A fundamental problem in training **deep neural networks** using **gradient-based methods** (backpropagation + gradient descent).

### Core Mathematical Principle

When you multiply many numbers less than 1 together, the product becomes extremely small:

$$0.5 \times 0.3 \times 0.2 \times 0.4 = 0.012$$

As you keep multiplying more small numbers, the result approaches **zero**.

### How It Manifests

During backpropagation, weight updates depend on **partial derivatives**:

$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}$$

In a deep network, $\frac{\partial L}{\partial W}$ is a **chain of multiplied derivatives**. If each derivative component is small (between 0 and 1), the final gradient becomes vanishingly small.

**Example:**
$$W_{\text{new}} = 0.5 - 0.01 \times 0.00001 = 0.4999999$$

The weight barely changes → **no learning happens** → loss stays constant.

### Key Conditions

| Condition | Detail |
|-----------|--------|
| **Network depth** | Occurs in **deep** networks (8-10+ layers), not shallow ones |
| **Activation function** | Primarily with **sigmoid** or **tanh** (saturating functions) |
| **Effect** | Initial layers learn extremely slowly or stop learning entirely |

### Why Sigmoid/Tanh Cause This

These are **squashing functions** — they compress any input range into a small output range:
- Sigmoid: $(0, 1)$
- Tanh: $(-1, 1)$

Their derivatives are always small:
- Sigmoid derivative max = $0.25$ (at $x = 0$)
- Outside $[-3, 3]$, derivative $\approx 0$

When you multiply many such small derivatives during backpropagation → gradient vanishes.

---

## 2. How to Detect Vanishing Gradient

| Method | How |
|--------|-----|
| **Monitor loss** | If loss plateaus and stops decreasing after initial epochs |
| **Plot weight changes** | Track a specific weight across epochs; if it remains constant → vanishing gradient |
| **Percentage change** | Compute `(W_new - W_old) / W_old × 100`; if ≈ 0% → problem exists |

### Code Detection Pattern

```python
# Store weights before training
old_weights = model.get_weights()[0].copy()

# Train for some epochs
model.fit(X_train, y_train, epochs=100)

# Compare weights after training
new_weights = model.get_weights()[0]

# Calculate gradient and percentage change
gradient = (old_weights - new_weights) / learning_rate
pct_change = ((new_weights - old_weights) / old_weights) * 100
# If pct_change ≈ 0 → vanishing gradient problem
```

---

## 3. Five Solutions to Vanishing Gradient

### Solution 1: Reduce Network Complexity

- Fewer layers → fewer derivative multiplications → gradient doesn't shrink as much
- **Limitation:** Reduces model's ability to capture complex non-linear patterns

### Solution 2: Use Non-Saturating Activation Functions (e.g., ReLU)

$$\text{ReLU}(x) = \max(0, x)$$

- Derivative is either **0** or **1**
- Multiplying many 1s together still gives 1 → gradient doesn't vanish
- **Caveat:** Has its own **Dying ReLU** problem (derivative = 0 for negative inputs)
- Variants: Leaky ReLU, ELU, SELU address the dying ReLU issue

### Solution 3: Proper Weight Initialization

- Techniques like **Xavier/Glorot** (for sigmoid/tanh) and **He** (for ReLU)
- Ensures activations and gradients maintain reasonable variance across layers

### Solution 4: Batch Normalization

- A layer added between hidden layers that **normalizes activations**
- Prevents internal covariate shift
- Keeps gradients in a healthy range throughout training

### Solution 5: Residual Networks (Skip Connections)

- **ResNets** use skip/shortcut connections that allow gradients to flow directly through the network
- Gradient can bypass layers via the identity connection → doesn't shrink

---

## 4. Exploding Gradient Problem

The **opposite** of vanishing gradients — more common in **Recurrent Neural Networks (RNNs)**.

### Principle

If derivative components are **greater than 1**, multiplying them produces an extremely large number:

$$2.5 \times 3.0 \times 1.8 = 13.5 \quad \text{(keeps growing with more terms)}$$

### Effect

$$W_{\text{new}} = 1.0 - 0.01 \times (-9999) = 100.99$$

- Weights become extremely large
- Model makes **random, unstable predictions**
- Loss does not converge — it may even increase or oscillate wildly

### Solution

- **Gradient Clipping**: Cap the gradient at a maximum threshold value during backpropagation

---

## 5. Summary Table

| Problem | Cause | Detection | Solutions |
|---------|-------|-----------|-----------|
| **Vanishing Gradient** | Derivatives < 1 multiplied across many layers | Loss plateau, weights unchanged | ReLU, proper init, batch norm, residual nets, reduce depth |
| **Exploding Gradient** | Derivatives > 1 multiplied across many layers | Loss diverges, weights blow up | Gradient clipping |
