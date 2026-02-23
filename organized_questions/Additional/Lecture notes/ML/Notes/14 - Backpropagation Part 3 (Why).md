# Lecture 14 — Backpropagation Part 3: The Why

## 1. Purpose

This lecture explains the **intuition** behind why backpropagation works — addressing all the "why" questions from Parts 1 & 2.

---

## 2. Concept 1 — Loss is a Function of All Parameters

### Key Insight

The loss function $L$ is a **function of all trainable parameters** (weights and biases):

$$L = f(w_{11}, w_{12}, w_{21}, w_{22}, w_{2_{11}}, w_{2_{12}}, b_{11}, b_{12}, b_{21})$$

**Why?** Because $\hat{y}$ is computed from parameters, and $L$ depends on $\hat{y}$. Expanding $\hat{y}$ in terms of all weights and biases makes this explicit:

$$\hat{y} = w_{2_{11}} \cdot \sigma(w_{11} \cdot x_1 + w_{12} \cdot x_2 + b_{11}) + w_{2_{12}} \cdot \sigma(w_{21} \cdot x_1 + w_{22} \cdot x_2 + b_{12}) + b_{21}$$

Everything except the weights/biases ($x_1, x_2, y$) comes from the **data** and is **constant**.

> **Analogy:** The neural network is a box with 9 **knobs**. Turn any knob → loss changes. Your goal: find the knob settings that **minimize** loss.

---

## 3. Concept 2 — What is a Gradient?

### Derivative vs Gradient

| Scenario | Term | Notation |
|---|---|---|
| Function of **one** variable: $y = f(x)$ | **Derivative** | $\frac{dy}{dx}$ |
| Function of **multiple** variables: $z = f(x, y)$ | **Gradient** (partial derivatives) | $\frac{\partial z}{\partial x},\; \frac{\partial z}{\partial y}$ |

### Example

For $f(x, y) = x^2 + y^2$:

$$\frac{\partial f}{\partial x} = 2x, \quad \frac{\partial f}{\partial y} = 2y$$

Our loss function depends on **9 parameters** → we compute **9 partial derivatives** (= the gradient).

---

## 4. Concept 3 — What Does a Derivative Mean?

### Rate of Change

$$\frac{dy}{dx} = \text{rate of change of } y \text{ with respect to } x$$

It tells us:
1. **Magnitude** — *how much* $y$ changes per unit change in $x$
2. **Sign** — *direction* of change (positive = same direction, negative = opposite)

### Derivative at a Point

For $y = x^2 + 2x$:

$$\frac{dy}{dx} = 2x + 1$$

At $x = 5$: $\frac{dy}{dx}\bigg|_{x=5} = 11$

**Meaning:** At $x = 5$, increasing $x$ by 1 unit causes $y$ to increase by ~11 units.

### Applied to Backpropagation

When we compute $\frac{\partial L}{\partial w_{11}}$, we're asking:

> "If I change $w_{11}$ by a small amount, how much does the loss change — and in which direction?"

---

## 5. Concept 4 — Finding the Minimum

### Single Variable

To minimize $f(x) = x^2$:

$$\frac{df}{dx} = 2x = 0 \implies x = 0 \quad \text{(minimum)}$$

### Multiple Variables

To minimize $f(x, y) = x^2 + y^2$:

$$\frac{\partial f}{\partial x} = 2x = 0 \implies x = 0$$
$$\frac{\partial f}{\partial y} = 2y = 0 \implies y = 0$$

Minimum at $(x, y) = (0, 0)$.

### Applied to Neural Networks

Our loss depends on 9 parameters → we need to find **9 values** such that:

$$\frac{\partial L}{\partial w_i} = 0 \quad \forall\; i$$

This is what gradient descent iteratively approximates.

---

## 6. The Weight Update Rule — Why It Works

### The Formula

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

### Why the Negative Sign?

**Case 1:** Derivative is **positive** ($\frac{\partial L}{\partial w} > 0$)
- Increasing $w$ → loss increases
- To **reduce** loss → **decrease** $w$
- $w_{new} = w_{old} - (\text{positive quantity})$ → $w$ decreases ✓

**Case 2:** Derivative is **negative** ($\frac{\partial L}{\partial w} < 0$)
- Increasing $w$ → loss decreases
- To **reduce** loss → **increase** $w$
- $w_{new} = w_{old} - (\text{negative quantity}) = w_{old} + |\cdot|$ → $w$ increases ✓

> The **negative gradient** automatically handles both cases — always moves in the direction that **reduces loss**.

### Graphical Intuition

On the loss curve $L$ vs $w$:

```
    L
    |    *
    |   / \
    |  /   \     ← positive slope → move left (decrease w)
    | /     \
    |/   *   \   ← minimum
    |     \   \
    |      \  ← negative slope → move right (increase w)
    +--------→ w
```

Starting from any point, following the **negative slope** always moves toward the minimum.

---

## 7. Why Learning Rate Matters

### Step Size = Slope × Learning Rate

Without $\eta$, the update step equals the slope magnitude, which can cause:
- **Overshooting** — jumping past the minimum
- **Oscillation** — bouncing back and forth across the minimum

### Learning Rate Effects

| $\eta$ Value | Behavior |
|---|---|
| Too large (e.g., 1.0) | Oscillation, divergence — may never converge |
| Too small (e.g., 0.0001) | Very slow convergence — takes many epochs |
| Just right (e.g., 0.01–0.1) | Smooth, efficient convergence |

### Example

Starting at $w = -5$, slope = $-5$:

- **Without** $\eta$: $w_{new} = -5 - (-5) = 0$ → jumps to 0 immediately
- Next step from 0: slope could push to +5 → oscillation!
- **With** $\eta = 0.1$: $w_{new} = -5 - 0.1 \times (-5) = -4.5$ → small smooth step

> $\eta$ is a **hyperparameter** — must be tuned carefully. Google's TensorFlow Playground demonstrates this interactively.

---

## 8. Convergence — When to Stop

### Definition

**Convergence** occurs when:

$$w_{new} \approx w_{old}$$

Which means:

$$\eta \cdot \frac{\partial L}{\partial w} \approx 0 \implies \frac{\partial L}{\partial w} \approx 0$$

i.e., the slope is near zero → you're at/near the minimum.

### Practical Approach

Instead of checking convergence explicitly, practitioners set a fixed number of epochs (e.g., 100, 1000) and trust that convergence happens within that budget.

```python
for epoch in range(1000):  # fixed budget
    # ... forward prop, loss, backprop, update ...
```

---

## 9. Visualization Tools

An interactive website (by Grant Sanderson / 3Blue1Brown style) animates the full backpropagation process:
- Forward propagation step-by-step
- Loss calculation
- Derivative computation with chain rule
- Backward propagation of gradients
- Weight updates

> **Project idea:** Build an interactive backpropagation visualizer using web technologies.

---

## 10. Summary

| Concept | Key Understanding |
|---|---|
| Loss as a function | $L = f(\text{all weights and biases})$ — a multi-dimensional surface |
| Gradient | Vector of all partial derivatives — points toward steepest ascent |
| Negative gradient | Points toward steepest **descent** → minimizes loss |
| Learning rate $\eta$ | Controls step size; too big = overshoot, too small = slow |
| Convergence | Slope ≈ 0 → at minimum → stop updating |
| Chain rule | Enables computing gradients through multiple layers |
