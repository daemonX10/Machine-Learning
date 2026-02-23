# Lecture 07 - Multi-Layer Perceptron (MLP) Intuition

## The Problem Recap

Perceptron can only create **linear decision boundaries**. For data like this:

```
  ● ● ○ ○
  ● ○ ○ ●
  ○ ○ ● ●
  ○ ● ● ○
```

You need a **non-linear** (curved) boundary — which a single perceptron cannot provide.

**Goal:** Build an algorithm using **multiple perceptrons** that can capture any non-linear relationship.

---

## Key Clarification: Sigmoid, Not Step

In this lecture, the perceptron uses:
- **Activation function:** Sigmoid (not step function)
- **Loss function:** Log loss (not hinge loss)
- Behaves like **logistic regression** — outputs probability $\in [0, 1]$

### Sigmoid Perceptron Behavior

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w_1 x_1 + w_2 x_2 + b$$

- Output is a **probability** (not 0/1)
- The decision boundary line divides the space into two regions
- **On the line:** $P(\text{yes}) = 0.5$
- **Far from line (positive side):** $P(\text{yes}) \to 1$
- **Far from line (negative side):** $P(\text{yes}) \to 0$

→ Creates a **gradient** of probabilities across the feature space.

---

## Building the MLP: First Principles

### Step 1 — Train Two Perceptrons on Same Data

Each perceptron learns a **different decision boundary**:

- **Perceptron 1:** Line equation $2x_1 + x_2 + 6 = 0$ → gives $P_1$ for each point
- **Perceptron 2:** Line equation $5x_1 + 4x_2 + 3 = 0$ → gives $P_2$ for each point

### Step 2 — Combine Outputs (Superimposition)

For any point, add both probabilities:

$$z_{\text{new}} = P_1 + P_2$$

**Problem:** Sum can exceed 1 (e.g., $0.7 + 0.8 = 1.5$), which is not a valid probability.

### Step 3 — Apply Sigmoid (Smoothing)

Pass the sum through sigmoid to get a valid probability:

$$P_{\text{combined}} = \sigma(P_1 + P_2) = \frac{1}{1 + e^{-(P_1 + P_2)}}$$

This is the **superimposition + smoothing** that creates a non-linear boundary.

### Step 4 — Add Weights for Flexibility (Weighted Combination)

Instead of direct addition, use **weighted combination**:

$$z = w_1 \cdot P_1 + w_2 \cdot P_2 + b$$

$$\hat{y} = \sigma(z)$$

This allows one perceptron to be **more dominant** than another, giving control over the shape of the decision boundary.

> **Example:** $w_1 = 10, w_2 = 5, b = 3$ → Perceptron 1 has **double** the influence.

### Result: This IS a Perceptron Itself!

The combination unit is itself a perceptron:
- **Inputs:** outputs of Perceptron 1 and 2
- **Weights:** $w_1, w_2$
- **Bias:** $b$
- **Activation:** Sigmoid

→ **Three perceptrons combined = Multi-Layer Perceptron**

---

## MLP Architecture

```
Input Layer          Hidden Layer         Output Layer
(Layer 0)            (Layer 1)            (Layer 2)

  x₁ ──────────┐
                ├──→ [Perceptron 1] ──┐
  x₂ ──────────┤                     ├──→ [Perceptron 3] ──→ ŷ
                ├──→ [Perceptron 2] ──┘
                │
```

| Component | Description |
|:---|:---|
| **Input Layer** | Raw features ($x_1, x_2$) |
| **Hidden Layer** | Multiple perceptrons that each learn a different linear boundary |
| **Output Layer** | Combines hidden layer outputs via weighted sum + sigmoid |

---

## Four Ways to Modify Neural Network Architecture

### 1. Add Nodes to Hidden Layer
- More perceptrons in hidden layer → more linear boundaries to combine
- Captures **more complex non-linearity**
- Example: 3 hidden nodes instead of 2 → linear combination of 3 boundaries

### 2. Add Nodes to Input Layer
- More input features → more dimensions in feature space
- In 2D: decision boundary is a **line**
- In 3D: decision boundary is a **plane**
- In nD: decision boundary is a **hyperplane**

### 3. Add Nodes to Output Layer
- For **multi-class classification**
- Example: Dog vs Cat vs Human → 3 output nodes, one per class
- Each output node gives probability for its class
- Highest probability wins

### 4. Add More Hidden Layers (Depth)
- Each layer captures **increasingly complex** relationships
- First layer: simple linear boundaries
- Deeper layers: combinations of combinations → highly non-linear boundaries
- This is what makes **deep** learning "deep"

---

## Universal Function Approximation

> Neural networks are **universal function approximators**: given enough nodes, layers, and training time, they can approximate **any** mathematical function / capture **any** non-linear relationship.

**Caveats:**
- More complexity → more training time
- More parameters → risk of overfitting
- Practical constraints exist (compute, data)

---

## TensorFlow Playground Demo

| Scenario | Architecture | Result |
|:---|:---|:---|
| Simple circular data | 2 hidden nodes | Converges quickly ✅ |
| Spiral data | 2 hidden nodes | Struggles |
| Spiral data | 4 hidden nodes | Better but slow |
| Spiral data | 4 nodes + ReLU activation | Converges well ✅ |

**Key observations:**
- Can visualize each hidden node's decision boundary
- Layer-by-layer: boundaries get progressively more complex
- **ReLU** activation often works better than sigmoid for hidden layers

---

## Summary

$$\text{MLP} = \text{Linear Combination of Multiple Perceptrons} + \text{Sigmoid Smoothing}$$

1. Each hidden node learns a **linear boundary**
2. Outputs are **weighted and summed**
3. Sigmoid **squashes** into valid probability
4. Result: a **non-linear decision boundary**
