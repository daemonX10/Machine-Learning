# Lecture 6.1 — Transformer: Self-Attention & Multi-Head Attention

---

## 1. Recap: Vanilla Attention Limitations

Three problems with RNN + Attention:

| # | Problem | Status after Vanilla Attention |
|---|---|---|
| 1 | Bottleneck problem | **Solved** — direct decoder→encoder connections |
| 2 | Vanishing gradient | **Partially solved** — shortcut gradient paths |
| 3 | Sequential processing | **Not solved** — still need $h_{t-1}$ before computing $h_t$ |

**Key question:** If attention already gives every decoder state access to all encoder states, **why do we still need recurrent connections?**

---

## 2. Query, Key, Value — Information Retrieval Analogy

| IR Concept | Attention Equivalent |
|---|---|
| **User query** | Decoder hidden state |
| **Document metadata / keywords** | Keys (encoder hidden states) |
| **Document content** | Values (encoder hidden states) |
| **Similarity matching** | Dot product between query and keys |
| **Retrieve relevant part** | Weighted sum of values using attention weights |

In vanilla attention: **keys = values** (both are encoder hidden states $h_i$)

---

## 3. From Attention to Self-Attention

### Motivation

- In RNN attention, recurrent links create a **sequential bottleneck**
- If attention lets every state access all other states → **remove recurrent connections entirely**
- Without recurrence, all positions can be processed **in parallel** (GPU-friendly)

### What is Lost

- **Positional/temporal information** — without recurrence, position is unknown
- Shuffling tokens produces the same output (position invariant)
- Solution: **Positional Encoding** (discussed below)

---

## 4. Self-Attention Mechanism

### Step 1 — Generate Q, K, V from Hidden States

For each hidden state $h_t$, generate three vectors using **learnable parameter matrices**:

$$q_t = W_Q \cdot h_t \quad \text{(Query)}$$

$$k_t = W_K \cdot h_t \quad \text{(Key)}$$

$$v_t = W_V \cdot h_t \quad \text{(Value)}$$

- $W_Q, W_K, W_V$ are **shared across all tokens** within a layer
- $W_Q, W_K, W_V$ are **unique per layer** (not shared across layers)

### Step 2 — Compute Attention Scores

For query at position $l$ attending to key at position $t$:

$$e_{l,t} = q_l \cdot k_t$$

### Step 3 — Attention Distribution (Softmax)

$$\alpha_{l,t} = \frac{\exp(e_{l,t})}{\sum_{t'} \exp(e_{l,t'})}$$

### Step 4 — Attention Vector (Weighted Sum of Values)

$$a_l = \sum_{t} \alpha_{l,t} \cdot v_t$$

### Key Difference from Vanilla Attention

| Vanilla Attention | Self-Attention |
|---|---|
| Query from decoder, K/V from encoder | **Q, K, V all from the same sequence** |
| Operates across two RNNs | Operates **within a single sequence** |
| K = V = encoder hidden state | K, V generated via separate learned projections |

---

## 5. Self-Attention as a Layer

- The entire Q-K-V → dot product → softmax → weighted sum process forms a **self-attention layer**
- Self-attention layers can be **stacked** repeatedly
- Each layer produces new attention vectors → fed as input to the next layer
- Each layer has its **own** $W_Q, W_K, W_V$

---

## 6. Self-Attention is Linear

The attention output can be written as:

$$a_l = \sum_t \alpha_{l,t} \cdot W_V \cdot h_t = W_V \sum_t \alpha_{l,t} \cdot h_t$$

- The weights $\alpha_{l,t}$ come from a **nonlinear process** (softmax of dot products)
- But the final combination is **linear in the hidden states**
- Linear models have limited representational power → need to add nonlinearity

---

## 7. Positional Encoding

### Problem

Self-attention is **position-invariant** — "the man eats cheese" and "cheese eats the man" produce identical representations.

### Solution

Add a **positional encoding vector** $p_t$ to each embedding $e_t$:

$$\text{input}_t = e_t + p_t$$

Properties of positional encodings:
- Each position gets a **unique vector**
- Distance and ordering properties are preserved
- $p_t$ is the same dimension as $e_t$
- Addition is element-wise

> Detailed methods (sinusoidal, learned, RoPE, etc.) covered in a separate lecture.

---

## 8. Multi-Head Self-Attention

### Motivation (CNN Analogy)

In CNN: multiple **kernels/filters** on the same image patch → each captures a different aspect (edges, textures, shapes).

In attention: multiple **attention heads** → each captures a different aspect of the token:
- One head may capture **syntactic role** (subject/object)
- Another may capture **POS tag** (noun/verb)
- Another may capture **named entity type**

> We don't explicitly assign roles to heads — the model learns what each head captures.

### Mechanism

For $H$ attention heads, maintain $H$ sets of parameter matrices:

$$\text{Head}^{(h)}: \quad W_Q^{(h)}, \; W_K^{(h)}, \; W_V^{(h)} \quad \text{for } h = 1, \dots, H$$

Each head independently computes:

$$q_t^{(h)} = W_Q^{(h)} h_t, \quad k_t^{(h)} = W_K^{(h)} h_t, \quad v_t^{(h)} = W_V^{(h)} h_t$$

Each head produces its own attention vector $a_t^{(h)}$ of dimension $d$.

### Combining Heads

1. **Concatenate** all head outputs: $[a_t^{(1)}; a_t^{(2)}; \dots; a_t^{(H)}] \in \mathbb{R}^{Hd}$
2. **Project** back to $d$ dimensions via output matrix $W_O \in \mathbb{R}^{Hd \times d}$:

$$\text{MultiHead}(t) = W_O \cdot [a_t^{(1)}; \dots; a_t^{(H)}]$$

In the original Transformer paper: $H = 8$ (base) or $H = 16$ (large).

---

## 9. Feed-Forward Network (FFN)

### Why Needed

Self-attention is **linear** → FFN adds **nonlinearity** for richer representations.

### Architecture

Position-wise (applied independently at each token position):

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

- $W_1 \in \mathbb{R}^{d \times d_{ff}}$ — projects $d$ → $d_{ff}$ (up-projection)
- $W_2 \in \mathbb{R}^{d_{ff} \times d}$ — projects $d_{ff}$ → $d$ (down-projection)
- Typically $d_{ff} = 4d$ (e.g., $d = 512, d_{ff} = 2048$)
- ReLU (or GELU) as activation

### Parameters

- $W_1, W_2$ are **shared across tokens** (position-wise) but **unique per layer**
- FFN is considered the **internal memory** of the Transformer

### Why Up-then-Down Projection?

- Project to **higher dimension first** to capture richer representations
- Then compress back to $d$
- Down→Up would lose information at the bottleneck

### Conceptual Roles

| Component | Role |
|---|---|
| **Self-Attention** | Memory fetching / message passing between tokens |
| **Feed-Forward Network** | Processing unit / computation on fetched information |

---

## 10. Masked Self-Attention (Decoder)

### Problem

In the decoder, future tokens shouldn't be visible during training or inference.

### Encoder vs Decoder Attention

| | Encoder | Decoder |
|---|---|---|
| **Type** | Unmasked self-attention | **Masked** self-attention |
| **Access** | Each token sees all others | Each token sees only **previous tokens + itself** |

### How Masking Works

The attention score matrix $QK^T$ has shape $n \times n$:

$$\begin{bmatrix} q_1 k_1 & q_1 k_2 & q_1 k_3 & \cdots \\ q_2 k_1 & q_2 k_2 & q_2 k_3 & \cdots \\ q_3 k_1 & q_3 k_2 & q_3 k_3 & \cdots \\ \vdots & & & \ddots \end{bmatrix}$$

Upper-triangular entries (where token attends to future) must be blocked.

**Masking procedure:**
1. Create an upper-triangular mask filled with $-\infty$
2. Add mask to the score matrix: upper-triangle entries become $-\infty$
3. Apply softmax: $e^{-\infty} = 0$ → future positions get **zero attention weight**
4. Lower triangle + diagonal retain valid attention scores

**During inference:** Only previous tokens exist anyway, so masking is naturally enforced.

---

## 11. Cross-Attention (Decoder ↔ Encoder)

### Purpose

Connect the decoder to the encoder — the link between the two stacks.

### Mechanism

| Source | Generates |
|---|---|
| **Encoder** final-layer outputs $h_1, \dots, h_n$ | Keys $k_i$ and Values $v_i$ (via $W_K^{cross}$, $W_V^{cross}$) |
| **Decoder** masked self-attention outputs $m_1, \dots, m_T$ | Queries $q_t$ (via $W_Q^{cross}$) |

- Each decoder query attends to **all encoder keys**
- Produces attention vectors the same way as self-attention
- **No masking needed** — decoder queries attend to encoder (not to other decoder states)

---

## 12. Complete Transformer Block Structure

### Encoder Block (×N)

```
Input Embeddings + Positional Encoding
    → Multi-Head Self-Attention (unmasked)
    → Add & Norm
    → Feed-Forward Network
    → Add & Norm
```

### Decoder Block (×N)

```
Output Embeddings + Positional Encoding
    → Multi-Head Masked Self-Attention
    → Add & Norm
    → Multi-Head Cross-Attention (Q from decoder, K/V from encoder)
    → Add & Norm
    → Feed-Forward Network
    → Add & Norm
```

**Transformer Base:** $N = 6$ layers | **Transformer Large:** $N = 12$ layers

### Remaining Topics (Next Lecture)

- Positional encoding details
- Add & Norm (residual connection + layer normalization)
- Residual connections between sub-layers

---

## 13. Complexity

| Operation | Complexity |
|---|---|
| Self-attention (per layer) | $O(n^2 \cdot d)$ where $n$ = sequence length, $d$ = dimension |
| FFN (per layer) | $O(n \cdot d \cdot d_{ff})$ |

The $O(n^2)$ self-attention cost is the key limitation for very long sequences.

---

## 14. Summary Table

| Concept | Purpose |
|---|---|
| **Self-Attention** | Parallel token interaction; replaces recurrence |
| **Positional Encoding** | Injects position information lost by removing recurrence |
| **Multi-Head Attention** | Captures multiple aspects/relationships per token |
| **FFN** | Adds nonlinearity; per-position processing |
| **Masked Self-Attention** | Prevents decoder from seeing future tokens |
| **Cross-Attention** | Links decoder to encoder; replaces the bottleneck connection |
