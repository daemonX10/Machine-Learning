# Lecture 04 — Transformer Architecture and Training

## 1. Motivation: Removing Recurrence

In attention over RNNs, decoder states can access all encoder states — but **recurrent connections** still prevent parallel processing.

**Key insight**: if we remove recurrence and use only attention, we can process all tokens in parallel.

**Problem introduced**: without recurrence, **position information is lost** — $\{x_1, x_2, x_3\}$ and $\{x_3, x_2, x_1\}$ become indistinguishable. This is solved by **positional encoding**.

---

## 2. Self-Attention

### Generating Q, K, V

From each hidden state $h_i$, generate three vectors via learned projection matrices:

$$Q_i = W_Q \cdot h_i, \quad K_i = W_K \cdot h_i, \quad V_i = W_V \cdot h_i$$

- $W_Q, W_K, W_V$ are **shared across all positions** within a layer
- Each layer has its **own** set of these matrices

### Computing Attention

For query $Q_i$ attending to all keys:

$$\text{score}_{i,j} = Q_i^T \cdot K_j \quad \forall j$$

$$\alpha_{i,j} = \text{softmax}_j(\text{score}_{i,j})$$

$$A_i = \sum_j \alpha_{i,j} \cdot V_j$$

**Key difference from cross-attention**: in self-attention, Q, K, V all come from the **same** set of hidden states.

### Matrix Form

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Why Q, K, V? (Information Retrieval Analogy)

| Component | IR Analogy |
|---|---|
| **Query** | Search query |
| **Key** | Document metadata/keywords — compared to query for relevance |
| **Value** | Actual document content — retrieved based on key relevance ($V = W_V \cdot h$ extracts a "relevant part") |

---

## 3. Self-Attention is a Linear Operation

The attention output can be rewritten:

$$A_i = \sum_j \alpha_{i,j} \cdot V_j = W_V \sum_j \alpha_{i,j} \cdot h_j$$

Since $W_V$ is constant and $\alpha$ (from softmax) is fixed given inputs, the remaining operation is **linear**.

> Self-attention alone does **not** add nonlinearity — this motivates the **feed-forward network**.

---

## 4. Multi-Head Attention

Instead of a single set of $(W_Q, W_K, W_V)$, use $K$ independent sets ("heads"):

$$\text{head}_k = \text{Attention}(Q^{(k)}, K^{(k)}, V^{(k)})$$

$$\text{MultiHead} = \text{Concat}(\text{head}_1, \ldots, \text{head}_K) \cdot W_O$$

| Aspect | Detail |
|---|---|
| Each head dimension | $d$ (same as hidden state) |
| Concatenated dimension | $K \times d$ |
| **Output projection** $W_O$ | Projects $K \times d$ back to $d$ |
| Analogy | Like multiple CNN filters — each head captures different aspects |

---

## 5. Position-wise Feed-Forward Network (FFN)

Applied **independently** to each position (parameters shared across positions):

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

| Parameter | Shape | Role |
|---|---|---|
| $W_1$ | $d \times 2d$ | Up-projection |
| $W_2$ | $2d \times d$ | Down-projection |

- Adds the **nonlinearity** that self-attention lacks
- Parameters are **shared across tokens** within a layer but **differ across layers**

### Each Layer Contains

1. (Multi-head) self-attention
2. Position-wise FFN

---

## 6. Masked Self-Attention (Causal Masking)

For **autoregressive** (next-token prediction) tasks, token $i$ must **not** attend to tokens $j > i$.

### Implementation

The $QK^T$ matrix:

$$\begin{bmatrix} q_1k_1 & q_1k_2 & q_1k_3 & \cdots \\ q_2k_1 & q_2k_2 & q_2k_3 & \cdots \\ q_3k_1 & q_3k_2 & q_3k_3 & \cdots \\ \vdots & & & \ddots \end{bmatrix}$$

**Mask**: set upper-triangular entries to $-\infty$ before softmax:

$$\text{Masked score} = QK^T + M, \quad M_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After softmax: $e^{-\infty} = 0$ → future tokens contribute zero weight.

> During **training**, all tokens are available but masking **simulates** the autoregressive constraint of test time.

---

## 7. Cross-Attention (Encoder-Decoder Interaction)

Connects the encoder and decoder towers in the Transformer:

| Source | Provides |
|---|---|
| **Decoder** (after masked self-attention) | Queries ($Q$) |
| **Encoder** (final layer output) | Keys ($K$) and Values ($V$) |

- Cross-attention exists at **every decoder layer**
- All decoder layers attend to the **same** final encoder output (not layer-to-layer)
- Adds its own set of parameter matrices $(W_Q, W_K, W_V)$ per layer

---

## 8. Full Transformer Architecture

### Encoder Block (repeated $N$ times)
```
Input Embeddings + Positional Encoding
    → Multi-Head Self-Attention
        → Add & LayerNorm (residual connection)
    → Position-wise FFN
        → Add & LayerNorm (residual connection)
```

### Decoder Block (repeated $N$ times)
```
Output Embeddings + Positional Encoding
    → Masked Multi-Head Self-Attention
        → Add & LayerNorm
    → Multi-Head Cross-Attention (Q from decoder, K/V from encoder)
        → Add & LayerNorm
    → Position-wise FFN
        → Add & LayerNorm
```

Typical values: $N = 6, 12, 24$ layers.

### Key Components Summary

| Component | Purpose |
|---|---|
| Self-Attention | Token-to-token interaction (parallel) |
| Masked Self-Attention | Causal / autoregressive constraint |
| Cross-Attention | Encoder-decoder communication |
| FFN | Adds nonlinearity |
| Residual Connections | Mitigates vanishing gradients |
| Layer Normalization | Training stability |
| Positional Encoding | Injects position information |

---

## 9. Residual Connections & Layer Normalization

### Residual Connections

After each sub-module (attention or FFN):
$$\text{output} = \text{LayerNorm}(x + \text{SubModule}(x))$$

- Helps gradient flow through deep networks (same principle as ResNets)

### Layer Norm vs Batch Norm

| Aspect | Batch Norm | Layer Norm |
|---|---|---|
| Normalizes across | All instances in a batch | All features of a single instance |
| Problem with text | Sequences have variable lengths → batch statistics are unreliable | Independent of sequence length |
| Used in Transformers | ✗ | ✓ |

---

## 10. Positional Encoding (Sinusoidal)

Since self-attention has no recurrence, position must be explicitly encoded.

### Sinusoidal Formula

$$PE_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

$$PE_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

- $\text{pos}$: token position in the sequence
- $i$: dimension index in the encoding vector
- Even dimensions use **sine**, odd dimensions use **cosine**
- The $10000$ base is empirically chosen

| Property | Detail |
|---|---|
| No learnable parameters | Fixed function of position |
| Captures periodicity | Sine/cosine encode relative positions |
| Added to embeddings | $\text{input} = e_w + PE_{\text{pos}}$ |

### Alternatives

| Method | Type |
|---|---|
| Sinusoidal | Fixed, non-learnable |
| Learned positional embeddings | Trainable vectors per position |
| **RoPE** (Rotary Position Embedding) | State-of-the-art; relative position via rotation matrices |

---

## 11. Key Takeaways

- **Self-attention** removes recurrence → enables full parallelism over tokens
- Q, K, V are **linear projections** of hidden states; shared across positions but differ across layers
- **Multi-head attention** = multiple parallel attention heads → richer representations
- **FFN** adds essential nonlinearity after the linear self-attention operation
- **Causal masking** uses $-\infty$ in upper triangle to prevent attending to future tokens
- **Cross-attention** bridges encoder and decoder with Q from decoder, K/V from encoder
- **Layer Norm** preferred over Batch Norm for variable-length sequences
- **Sinusoidal positional encoding** injects position info; RoPE is the modern alternative
