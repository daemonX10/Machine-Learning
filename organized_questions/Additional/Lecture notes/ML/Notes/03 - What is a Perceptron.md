# What is a Perceptron

## 1. Definition

A **Perceptron** is:
- A **supervised learning algorithm** for binary classification
- A **mathematical model / function** inspired by a biological neuron
- The **fundamental building block** of Artificial Neural Networks

> You must understand the single Perceptron before studying Multi-Layer Perceptrons (MLPs).

---

## 2. Perceptron Architecture

```
  x₁ ──w₁──┐
             ├──► Σ (summation) ──► f(z) (activation) ──► ŷ (output)
  x₂ ──w₂──┘                           │
   1 ──b────┘                     (Step Function)
```

### Components

| Component | Symbol | Description |
|-----------|--------|-------------|
| **Inputs** | $x_1, x_2, \ldots, x_n$ | Feature values from the data |
| **Weights** | $w_1, w_2, \ldots, w_n$ | Strength of connection; learned during training |
| **Bias** | $b$ | Shifts the decision boundary; also learned |
| **Summation** | $z$ | Weighted sum (dot product) |
| **Activation Function** | $f(z)$ | Maps $z$ to output range |
| **Output** | $\hat{y}$ | Prediction (0 or 1 for step function) |

### Summation (Dot Product)

$$z = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b = \mathbf{w} \cdot \mathbf{x} + b$$

### Step Function (Activation)

$$\hat{y} = f(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

---

## 3. How Perceptron Works

### Two Stages (like any ML algorithm):

#### Stage 1: Training
- Use labeled data to find optimal values of $w_1, w_2, \ldots, w_n$ and $b$
- The **entire goal** of training is to find these parameters

#### Stage 2: Prediction
1. Plug new input values into the summation formula
2. Compute $z = w_1 x_1 + w_2 x_2 + b$
3. Apply activation function: if $z \geq 0 \Rightarrow 1$ (positive class), else $0$

### Example
Given trained parameters: $w_1 = 1, w_2 = 2, b = 3$

For a new student with IQ = 100, CGPA = 5.1:

$$z = (1)(100) + (2)(5.1) + 3 = 100 + 10.2 + 3 = 113.2$$

Since $z = 113.2 \geq 0$, prediction = **1** (placement will happen).

---

## 4. Interpretation of Weights

Weights indicate **feature importance**:

| Feature | Weight | Interpretation |
|---------|--------|---------------|
| IQ ($x_1$) | $w_1 = 2$ | Less important |
| CGPA ($x_2$) | $w_2 = 4$ | **More important** (2× the weight of IQ) |

> Higher absolute weight = more influential feature in the decision.

---

## 5. Perceptron vs Biological Neuron

```
 Biological Neuron                Perceptron
 ─────────────────                ──────────
 Dendrites (input)        ↔      Inputs (x₁, x₂)
 Nucleus (processing)     ↔      Summation + Activation
 Axon (output)            ↔      Output (ŷ)
```

| Aspect | Biological Neuron | Perceptron |
|--------|-------------------|------------|
| **Complexity** | Extremely complex structure | Very simple mathematical function |
| **Internal Processing** | Unknown electrochemical reactions | Simple dot product + step function |
| **Neuroplasticity** | Connections change over time (thicken, thin, form, break) | Connections (weights) are **fixed** after training |
| **Relationship** | — | **Weakly inspired** by neuron, not a copy |

> Perceptron is **inspired** by the neuron but is **not** a faithful replica. The biological neuron is vastly more complex.

---

## 6. Geometric Intuition

### Key Insight: Perceptron = A Line (in 2D)

The decision boundary $z = 0$ gives:

$$w_1 x_1 + w_2 x_2 + b = 0$$

This is the **equation of a line** ($ax + by + c = 0$)!

### How Classification Works

$$\text{If } w_1 x_1 + w_2 x_2 + b \geq 0 \Rightarrow \text{Class 1 (positive region)}$$
$$\text{If } w_1 x_1 + w_2 x_2 + b < 0 \Rightarrow \text{Class 0 (negative region)}$$

The line divides the feature space into **two regions** → **binary classification**.

### Higher Dimensions
| Dimensions | Decision Boundary | Equation Form |
|-----------|-------------------|---------------|
| 2D | **Line** | $w_1x_1 + w_2x_2 + b = 0$ |
| 3D | **Plane** | $w_1x_1 + w_2x_2 + w_3x_3 + b = 0$ |
| nD ($n > 3$) | **Hyperplane** | $\sum_{i=1}^{n} w_i x_i + b = 0$ |

### Limitation: Linear Separability
- Perceptron can **only** classify **linearly separable** data
- If data cannot be separated by a single line/plane → perceptron **fails**
- This is the biggest limitation and the reason **multi-layer perceptrons** were needed

---

## 7. Practical Code Example (sklearn)

```python
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

# Data: 100 students with CGPA, Resume Score, Placement (0/1)
X = df[['cgpa', 'resume_score']].values
y = df['placed'].values

# Train
p = Perceptron()
p.fit(X, y)

# Extract parameters
print("Weights (w1, w2):", p.coef_)       # e.g., [40.xx, -36.xx]
print("Bias (b):", p.intercept_)          # e.g., [-25]

# Visualize decision boundary
plot_decision_regions(X, y, clf=p)
```

---

## 8. Key Takeaways

1. Perceptron = simplest neural network unit = **linear binary classifier**
2. Core operation: $z = \mathbf{w} \cdot \mathbf{x} + b$ followed by step function
3. Training finds $w$ and $b$; prediction uses them
4. Geometrically, perceptron draws a **line** (2D) / **plane** (3D) / **hyperplane** (nD)
5. **Cannot handle non-linear data** — this limitation led to multi-layer perceptrons
6. Weakly inspired by biological neurons, but vastly simpler
