# Lecture 10 - Forward Propagation

## What is Forward Propagation?

> **Forward propagation** is the process of passing input data **through the network layer by layer** to produce a prediction (output). It is how a neural network makes predictions.

In training, two phases alternate:
1. **Forward propagation** — input flows forward through layers to produce output
2. **Backpropagation** — error flows backward to update weights

---

## Reference Architecture

```
Layer 0 (Input)    Layer 1 (Hidden)    Layer 2 (Hidden)    Layer 3 (Output)
  4 nodes             3 nodes             2 nodes             1 node

  CGPA ─────────┐
                ├──→ ○ ────────────┐
  IQ ───────────┤                  ├──→ ○ ────────────→ ○ ──→ ŷ
                ├──→ ○ ────────────┤
  10th Marks ───┤                  ├──→ ○
                ├──→ ○ ────────────┘
  12th Marks ───┘
```

### Trainable Parameters Count

| Connection | Weights | Biases | Subtotal |
|:---|:---:|:---:|:---:|
| Layer 0 → Layer 1 | $4 \times 3 = 12$ | $3$ | 15 |
| Layer 1 → Layer 2 | $3 \times 2 = 6$ | $2$ | 8 |
| Layer 2 → Layer 3 | $2 \times 1 = 2$ | $1$ | 3 |
| **Total** | **20** | **6** | **26** |

---

## Layer-by-Layer Computation

### Core Formula (Per Node)

For any node $j$ in layer $l$:

$$z_j^l = \sum_i w_{ij}^l \cdot a_i^{l-1} + b_j^l$$

$$a_j^l = \sigma(z_j^l)$$

where $\sigma$ is the sigmoid activation function.

### Matrix Form (Per Layer)

$$\mathbf{z}^l = (\mathbf{W}^l)^T \cdot \mathbf{a}^{l-1} + \mathbf{b}^l$$

$$\mathbf{a}^l = \sigma(\mathbf{z}^l)$$

---

## Step-by-Step Example

### Step 1: Input Layer → Hidden Layer 1

**Weight matrix** $\mathbf{W}^1$ (4 × 3):

$$\mathbf{W}^1 = \begin{bmatrix} w_{11}^1 & w_{12}^1 & w_{13}^1 \\ w_{21}^1 & w_{22}^1 & w_{23}^1 \\ w_{31}^1 & w_{32}^1 & w_{33}^1 \\ w_{41}^1 & w_{42}^1 & w_{43}^1 \end{bmatrix}$$

- **Rows** → correspond to input nodes (source)
- **Columns** → correspond to hidden nodes (destination)

**Input vector** $\mathbf{a}^0$ (4 × 1):

$$\mathbf{a}^0 = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} = \begin{bmatrix} \text{CGPA} \\ \text{IQ} \\ \text{10th Marks} \\ \text{12th Marks} \end{bmatrix}$$

**Computation:**

$$\mathbf{z}^1 = (\mathbf{W}^1)^T \cdot \mathbf{a}^0 + \mathbf{b}^1 \quad \Rightarrow \quad (3 \times 4) \cdot (4 \times 1) + (3 \times 1) = (3 \times 1)$$

$$\mathbf{a}^1 = \sigma(\mathbf{z}^1) \quad \Rightarrow \quad (3 \times 1)$$

**Result:** Three outputs $[a_1^1, a_2^1, a_3^1]^T$ — activations of hidden layer 1.

### Step 2: Hidden Layer 1 → Hidden Layer 2

**Weight matrix** $\mathbf{W}^2$ (3 × 2):

$$\mathbf{W}^2 = \begin{bmatrix} w_{11}^2 & w_{12}^2 \\ w_{21}^2 & w_{22}^2 \\ w_{31}^2 & w_{32}^2 \end{bmatrix}$$

**Computation:**

$$\mathbf{z}^2 = (\mathbf{W}^2)^T \cdot \mathbf{a}^1 + \mathbf{b}^2 \quad \Rightarrow \quad (2 \times 3) \cdot (3 \times 1) + (2 \times 1) = (2 \times 1)$$

$$\mathbf{a}^2 = \sigma(\mathbf{z}^2) \quad \Rightarrow \quad (2 \times 1)$$

**Result:** Two outputs $[a_1^2, a_2^2]^T$ — activations of hidden layer 2.

### Step 3: Hidden Layer 2 → Output Layer

**Weight matrix** $\mathbf{W}^3$ (2 × 1):

$$\mathbf{W}^3 = \begin{bmatrix} w_{11}^3 \\ w_{21}^3 \end{bmatrix}$$

**Computation:**

$$z^3 = (\mathbf{W}^3)^T \cdot \mathbf{a}^2 + b^3 \quad \Rightarrow \quad (1 \times 2) \cdot (2 \times 1) + (1 \times 1) = (1 \times 1)$$

$$\hat{y} = a^3 = \sigma(z^3) \quad \Rightarrow \quad \text{scalar}$$

**Result:** Single probability value — the model's prediction.

---

## General Forward Propagation Formula

### Layer-by-Layer (Iterative)

$$\mathbf{a}^l = \sigma\left((\mathbf{W}^l)^T \cdot \mathbf{a}^{l-1} + \mathbf{b}^l\right) \quad \text{for } l = 1, 2, \dots, L$$

### Activation Notation

| Symbol | Meaning |
|:---|:---|
| $\mathbf{a}^0$ | Input features (layer 0 activation) |
| $\mathbf{a}^1$ | Hidden layer 1 output |
| $\mathbf{a}^2$ | Hidden layer 2 output |
| $\mathbf{a}^L$ | Final output ($\hat{y}$) |

### Nested Expression (Full)

The entire network computes a **nested composition of functions**:

$$\hat{y} = \sigma\Bigg(\mathbf{W}^{3T} \cdot \sigma\Big(\mathbf{W}^{2T} \cdot \sigma\big(\mathbf{W}^{1T} \cdot \mathbf{a}^0 + \mathbf{b}^1\big) + \mathbf{b}^2\Big) + \mathbf{b}^3\Bigg)$$

Expanding step by step:

$$\mathbf{a}^1 = \sigma(\mathbf{W}^{1T} \cdot \mathbf{a}^0 + \mathbf{b}^1)$$

$$\mathbf{a}^2 = \sigma(\mathbf{W}^{2T} \cdot \mathbf{a}^1 + \mathbf{b}^2)$$

$$\hat{y} = \sigma(\mathbf{W}^{3T} \cdot \mathbf{a}^2 + \mathbf{b}^3)$$

---

## Why Linear Algebra Makes This Efficient

| Without Matrix Operations | With Matrix Operations |
|:---|:---|
| Compute each node individually | Single matrix multiply per layer |
| Many nested loops | Vectorized operations |
| Scales poorly | Scales to any architecture |

> No matter how complex the architecture, forward propagation reduces to **repeated matrix multiplications followed by element-wise activation functions**.

---

## Dimension Tracking Cheat Sheet

| Layer Transition | $\mathbf{W}^l$ shape | $(\mathbf{W}^l)^T$ shape | $\mathbf{a}^{l-1}$ shape | $\mathbf{b}^l$ shape | $\mathbf{a}^l$ shape |
|:---|:---:|:---:|:---:|:---:|:---:|
| 0 → 1 | $(4 \times 3)$ | $(3 \times 4)$ | $(4 \times 1)$ | $(3 \times 1)$ | $(3 \times 1)$ |
| 1 → 2 | $(3 \times 2)$ | $(2 \times 3)$ | $(3 \times 1)$ | $(2 \times 1)$ | $(2 \times 1)$ |
| 2 → 3 | $(2 \times 1)$ | $(1 \times 2)$ | $(2 \times 1)$ | $(1 \times 1)$ | $(1 \times 1)$ |

**General rule:** If layer $l-1$ has $m$ nodes and layer $l$ has $n$ nodes:
- $\mathbf{W}^l$ is $(m \times n)$
- $(\mathbf{W}^l)^T$ is $(n \times m)$
- Output $\mathbf{a}^l$ is $(n \times 1)$

---

## Summary

1. Forward propagation = **layer-by-layer matrix multiplications + activation**
2. At each layer: $\mathbf{a}^l = \sigma((\mathbf{W}^l)^T \cdot \mathbf{a}^{l-1} + \mathbf{b}^l)$
3. The full network is a **nested function composition**
4. **Linear algebra** handles all complexity — just dot products and transposes
5. Initial weights start **random**; training (backpropagation) optimizes them
