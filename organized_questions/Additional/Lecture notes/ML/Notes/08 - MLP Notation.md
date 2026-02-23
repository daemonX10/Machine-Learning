# Lecture 08 - MLP Notation

## Why Notation Matters

When studying **backpropagation**, confusion arises from not knowing proper notation for weights, biases, and outputs. Learning notation first eliminates this confusion.

---

## Reference Architecture

```
Layer 0 (Input)     Layer 1 (Hidden)     Layer 2 (Hidden)     Layer 3 (Output)
  4 nodes              3 nodes              2 nodes              1 node

   ○ ─────────────┐
   ○ ─────────────┼──→ ○ ─────────────┐
   ○ ─────────────┼──→ ○ ─────────────┼──→ ○ ──────────→ ○ ──→ ŷ
   ○ ─────────────┼──→ ○ ─────────────┘
                  │
```

- **Input layer (Layer 0):** 4 nodes (4 features)
- **Hidden layer 1 (Layer 1):** 3 nodes
- **Hidden layer 2 (Layer 2):** 2 nodes
- **Output layer (Layer 3):** 1 node

---

## Counting Trainable Parameters

For any architecture, count: **Total = Weights + Biases**

| Connection | Weights | Biases | Total |
|:---|:---:|:---:|:---:|
| Layer 0 → Layer 1 | $4 \times 3 = 12$ | $3$ | **15** |
| Layer 1 → Layer 2 | $3 \times 2 = 6$ | $2$ | **8** |
| Layer 2 → Layer 3 | $2 \times 1 = 2$ | $1$ | **3** |
| **Grand Total** | **20** | **6** | **26** |

### Formula

$$\text{Parameters between layers } l-1 \text{ and } l = (n_{l-1} \times n_l) + n_l$$

where $n_l$ = number of nodes in layer $l$.

---

## Bias Notation

$$b_j^l$$

| Symbol | Meaning |
|:---|:---|
| $l$ (superscript) | **Layer number** |
| $j$ (subscript) | **Node number** within that layer |

### Examples

| Bias | Layer | Node |
|:---|:---:|:---:|
| $b_1^1$ | Layer 1 | Node 1 |
| $b_2^1$ | Layer 1 | Node 2 |
| $b_3^1$ | Layer 1 | Node 3 |
| $b_1^2$ | Layer 2 | Node 1 |
| $b_2^2$ | Layer 2 | Node 2 |
| $b_1^3$ | Layer 3 | Node 1 |

---

## Output (Activation) Notation

$$a_j^l$$

Same notation as bias — identifies **which node** in **which layer**.

| Symbol | Meaning |
|:---|:---|
| $l$ (superscript) | **Layer number** |
| $j$ (subscript) | **Node number** within that layer |

### Examples

| Output | Meaning |
|:---|:---|
| $a_1^1$ | Output of Layer 1, Node 1 |
| $a_2^1$ | Output of Layer 1, Node 2 |
| $a_3^1$ | Output of Layer 1, Node 3 |
| $a_1^2$ | Output of Layer 2, Node 1 |
| $a_2^2$ | Output of Layer 2, Node 2 |
| $a_1^3$ | Output of Layer 3, Node 1 (final output) |

---

## Weight Notation

$$w_{ij}^k$$

Weights need **three identifiers** — the connection goes from one layer to the next:

| Symbol | Meaning |
|:---|:---|
| $k$ (superscript) | **Layer the weight is entering** (destination layer) |
| $i$ (first subscript) | **Node number in the previous layer** (source node) |
| $j$ (second subscript) | **Node number in the current layer** (destination node) |

### Reading a Weight

> $w_{ij}^k$ = "Weight entering **layer $k$**, coming from **node $i$** of the previous layer, going to **node $j$** of layer $k$"

### Examples

| Weight | Entering Layer | From Node | To Node | Description |
|:---|:---:|:---:|:---:|:---|
| $w_{11}^1$ | 1 | Input node 1 | Hidden1 node 1 | Input 1 → H1 Node 1 |
| $w_{12}^1$ | 1 | Input node 1 | Hidden1 node 2 | Input 1 → H1 Node 2 |
| $w_{13}^1$ | 1 | Input node 1 | Hidden1 node 3 | Input 1 → H1 Node 3 |
| $w_{41}^1$ | 1 | Input node 4 | Hidden1 node 1 | Input 4 → H1 Node 1 |
| $w_{21}^2$ | 2 | H1 node 2 | H2 node 1 | H1 Node 2 → H2 Node 1 |
| $w_{22}^2$ | 2 | H1 node 2 | H2 node 2 | H1 Node 2 → H2 Node 2 |
| $w_{11}^3$ | 3 | H2 node 1 | Output node 1 | H2 Node 1 → Output |
| $w_{21}^3$ | 3 | H2 node 2 | Output node 1 | H2 Node 2 → Output |

---

## Visual Summary

```
Layer 0          Layer 1          Layer 2          Layer 3
(Input)          (Hidden 1)       (Hidden 2)       (Output)

 x₁ ───w₁₁¹───→ [b₁¹] a₁¹ ──w₁₁²──→ [b₁²] a₁² ──w₁₁³──→ [b₁³] a₁³ → ŷ
     ╲  w₁₂¹  ╱                ╲  w₁₂²  ╱                ╱
 x₂ ──╳──────→ [b₂¹] a₂¹ ──────╳──────→ [b₂²] a₂² ──w₂₁³
     ╱  w₂₁¹  ╲                ╱
 x₃ ───w₃₁¹───→ [b₃¹] a₃¹
     ╲
 x₄ ───w₄₁¹───→
```

---

## Practice Exercise

> Draw a more complex neural network (e.g., 5 inputs, 4 hidden nodes, 3 hidden nodes, 2 outputs). Try writing the notation for every weight, bias, and output. This is the best preparation for understanding backpropagation.

---

## Key Takeaway

| Element | Notation | Identifiers |
|:---|:---:|:---|
| Bias | $b_j^l$ | Layer $l$, Node $j$ |
| Output | $a_j^l$ | Layer $l$, Node $j$ |
| Weight | $w_{ij}^k$ | Entering layer $k$, from node $i$, to node $j$ |
