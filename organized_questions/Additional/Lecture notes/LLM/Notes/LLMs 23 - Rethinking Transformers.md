# Lecture 23: Rethinking Transformers — Residual Stream & Circuit Perspective

## 1. Recap: Masked Self-Attention

In decoder-only models, token $i$ can only attend to tokens $j \leq i$.

- Compute attention scores: $\text{score}(i, j) = \frac{q_i \cdot k_j}{\sqrt{d_k}}$
- Mask upper-triangular entries to $-\infty$ → softmax yields 0 for future tokens
- Weighted sum over value vectors

---

## 2. Residual Stream Perspective

Instead of viewing Transformer as sequential blocks, view it as a **stream** where **all operations are additive**:

```
x_i (embedding)
  + attn_head_1^(1)     ← Layer 1, Head 1
  + attn_head_2^(1)     ← Layer 1, Head 2
  + ...
  + FFN^(1)             ← Layer 1 MLP
  + attn_head_1^(2)     ← Layer 2, Head 1
  + ...
  + FFN^(L)             ← Layer L MLP
  → W_U                 ← Unembedding layer → logits
```

> **Key idea**: The embedding enters a "pipeline" and each component (attention head, FFN) **adds** its contribution. The final representation is the sum of all additive contributions.

### Notation

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| $x_i^{(l)}$ | Representation of token $i$ at layer $l$ | $1 \times d$ |
| $X^{(l)}$ | All token representations at layer $l$ | $n \times d$ |
| $X_{\leq i}^{(l)}$ | Representations of tokens 1 to $i$ at layer $l$ | $i \times d$ |
| $d$ | Model/residual stream dimension | scalar |
| $d_h$ | Head dimension ($d_h \neq d$ in general) | scalar |
| $d_f$ | FFN intermediate dimension | scalar |
| $W_U$ | Unembedding matrix | $d \times |V|$ |

---

## 3. Attention Block — Detailed Decomposition

### 3.1 Attention Weight Computation (Single Head $h$, Layer $l$, Token $i$)

$$a_i^{(l,h)} = \text{softmax}\!\left( x_i^{(l-1)} \cdot W_Q^{(l,h)} \cdot \left(X_{\leq i}^{(l-1)} \cdot W_K^{(l,h)}\right)^T \right)$$

**Dimensions**:

| Term | Dimension |
|------|-----------|
| $x_i^{(l-1)}$ | $1 \times d$ |
| $W_Q^{(l,h)}$ | $d \times d_h$ |
| $X_{\leq i}^{(l-1)}$ | $i \times d$ |
| $W_K^{(l,h)}$ | $d \times d_h$ |
| $a_i^{(l,h)}$ (attention weights) | $1 \times i$ |

### 3.2 Query-Key Circuit

Combining $W_Q$ and $W_K$:

$$W_{QK}^{(l,h)} = W_Q^{(l,h)} \cdot \left(W_K^{(l,h)}\right)^T$$

- Dimension: $d \times d$
- Determines **which tokens attend to which** (attention pattern)

### 3.3 Attention Output — Value + Output Projection

For each pair $(i, j)$ where $j \leq i$:

$$\text{attn}^{(l,h)}(x^{(l-1)}) = \sum_{j \leq i} a_{i,j}^{(l,h)} \cdot x_j^{(l-1)} \cdot W_V^{(l,h)} \cdot W_O^{(l,h)}$$

**Dimensions**:

| Term | Dimension |
|------|-----------|
| $a_{i,j}^{(l,h)}$ | scalar |
| $x_j^{(l-1)}$ | $1 \times d$ |
| $W_V^{(l,h)}$ | $d \times d_h$ |
| $W_O^{(l,h)}$ | $d_h \times d$ |
| Output | $1 \times d$ ✓ (compatible with residual stream) |

### 3.4 Output-Value (OV) Circuit

$$W_{OV}^{(l,h)} = W_V^{(l,h)} \cdot W_O^{(l,h)}$$

- Dimension: $d \times d$
- Responsible for **writing information to the residual stream**
- Determines **what content is moved** between positions

### 3.5 Multi-Head Aggregation

Attention heads produce $d$-dimensional vectors (not $d_h$) thanks to $W_O$ projection. They are **summed** (not concatenated):

$$\text{attn}^{(l)}(X^{(l-1)}) = \sum_{h=1}^{H} \text{attn}^{(l,h)}(X^{(l-1)})$$

### 3.6 Intermediate Representation

$$x_i^{mid,(l)} = x_i^{(l-1)} + \text{attn}^{(l)}(X_{\leq i}^{(l-1)})$$

---

## 4. Feed-Forward Network (FFN) — Key-Value Memory View

### 4.1 Standard FFN Computation

$$\text{FFN}^{(l)}(x_i^{mid}) = g\!\left(x_i^{mid} \cdot W_{in}^{(l)}\right) \cdot W_{out}^{(l)}$$

| Matrix | Dimension | Role |
|--------|-----------|------|
| $W_{in}^{(l)}$ | $d \times d_f$ | **Key matrix** — pattern detector |
| $W_{out}^{(l)}$ | $d_f \times d$ | **Value matrix** — stores retrieved values |
| $g(\cdot)$ | nonlinearity | ReLU / GELU / SiLU |

### 4.2 Key-Value Memory Interpretation

1. **Key retrieval**: $x_i^{mid} \cdot W_{in}$ → depending on input, extracts a pattern (key match)
2. **Nonlinearity**: $g(\cdot)$ gates neuron activations → produces $1 \times d_f$ vector
3. **Value retrieval**: Multiply with $W_{out}$ → up/down-weights rows of $W_{out}$, producing output of dimension $d$

> FFN acts as a **key-value memory**: $W_{in}$ stores keys (patterns), $W_{out}$ stores values (associated outputs).

### 4.3 Neuron-Level Decomposition

Each FFN output decomposes as a **sum over individual neurons**:

$$\text{FFN}^{(l)}(x_i^{mid}) = \sum_{u=1}^{d_f} \underbrace{g\!\left(x_i^{mid} \cdot W_{in,u}^{(l)}\right)}_{n_u^{(l)} \text{ (neuron activation, scalar)}} \cdot W_{out,u}^{(l)}$$

where:
- $W_{in,u}^{(l)}$ is the $u$-th column of $W_{in}$ (dimension $d \times 1$)
- $W_{out,u}^{(l)}$ is the $u$-th row of $W_{out}$ (dimension $1 \times d$)
- $n_u^{(l)}$ is a scalar — the activation of neuron $u$
- $\mathbf{n}^{(l)} = [n_1^{(l)}, \ldots, n_{d_f}^{(l)}]$ is the full neuron activation vector

---

## 5. Final Output Decomposition

Since all operations are additive:

$$x_i^{(L)} = x_i^{(0)} + \sum_{l=1}^{L}\left[\sum_{h=1}^{H} \text{attn}^{(l,h)}(\cdot) + \text{FFN}^{(l)}(\cdot)\right]$$

The final logits:

$$f(x) = x_i^{(L)} \cdot W_U = \underbrace{x_i^{(0)} \cdot W_U}_{\text{direct path}} + \sum_{l=1}^{L}\left[\underbrace{\sum_{h=1}^{H} \text{attn}^{(l,h)} \cdot W_U}_{\text{attention contribution}} + \underbrace{\text{FFN}^{(l)} \cdot W_U}_{\text{FFN contribution}}\right]$$

> Because of linearity of addition, we can **separately analyze** each component's contribution to the final output.

---

## 6. Circuit Analysis — Two-Layer Attention-Only Example

Consider: 2 layers, no FFN, single head per layer, OV circuits only.

$$f(x) = x \cdot W_U + x \cdot W_{OV}^{(1)} \cdot W_U + x \cdot W_{OV}^{(2)} \cdot W_U + x \cdot W_{OV}^{(1)} \cdot W_{OV}^{(2)} \cdot W_U$$

### Four Effective Circuits

| Circuit | Path | Name |
|---------|------|------|
| $x \cdot W_U$ | Input → Unembed | **Direct Path** |
| $x \cdot W_{OV}^{(1)} \cdot W_U$ | Input → OV₁ → Unembed | **Full OV Circuit** (layer 1) |
| $x \cdot W_{OV}^{(2)} \cdot W_U$ | Input → OV₂ → Unembed | **Full OV Circuit** (layer 2) |
| $x \cdot W_{OV}^{(1)} \cdot W_{OV}^{(2)} \cdot W_U$ | Input → OV₁ → OV₂ → Unembed | **Virtual Attention Head** (V-composition) |

> **Virtual Attention Head**: $W_{OV}^{(1)} \cdot W_{OV}^{(2)}$ can be treated as a single effective attention head that spans two layers.

---

## 7. Summary of Key Concepts

| Concept | Definition |
|---------|------------|
| **Residual Stream** | The running sum that each component adds to; dimension $d$ throughout |
| **QK Circuit** ($W_{QK}$) | $W_Q \cdot W_K^T$ — determines attention patterns (who attends to whom) |
| **OV Circuit** ($W_{OV}$) | $W_V \cdot W_O$ — determines what information is written to the stream |
| **FFN as Key-Value Memory** | $W_{in}$ = keys (pattern detection), $W_{out}$ = values (stored information) |
| **Neuron Activation** | Scalar gate $n_u$ controlling how much each value row contributes |
| **Direct Path** | $x \cdot W_U$ — input directly to output, bypassing all layers |
| **Full OV Circuit** | Single attention layer's contribution to final output |
| **Virtual Attention Head** | Composition of OV circuits across layers (V-composition) |

> This decomposition is the **foundation for mechanistic interpretability** — understanding what each component does by analyzing circuits individually.
