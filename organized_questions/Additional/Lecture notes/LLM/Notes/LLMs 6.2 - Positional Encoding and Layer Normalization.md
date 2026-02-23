# LLMs 6.2 — Positional Encoding and Layer Normalization

## 1. Why Positional Encoding?

- Transformers process all tokens **in parallel** → no inherent notion of token order
- Position embeddings are **vectors of the same dimension** as token embeddings
- Combined via **element-wise addition** (not concatenation — fewer parameters)
- Added **only once** at the input of the first encoder/decoder layer, not at every layer

---

## 2. Naive Approaches and Their Problems

### Approach 1 — Integer Encoding
| Position | Embedding |
|----------|-----------|
| 0 | [0,0,0,0] |
| 1 | [1,1,1,1] |
| 2 | [2,2,2,2] |

**Problems:**
- Max sequence length must be predefined — impossible with variable-length inputs
- Model can't distinguish "position 3 as last token" from "position 3 in a 30-token sequence"

### Approach 2 — Normalized Integers
Divide each position by the sequence length → last position always `[1,1,1,1]`.

**Problem:** Same position gets different vectors in sequences of different lengths (e.g., position 1 in a 3-token vs 30-token sequence).

---

## 3. Sinusoidal Positional Encoding (Vaswani et al.)

### Motivation from Binary Encoding
Binary representations of positions show **periodicity**: LSB alternates every 1 step, next bit every 2 steps, etc. → captured naturally by sinusoidal functions.

### Formula

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

| Symbol | Meaning |
|--------|---------|
| $pos$ | Token position in the sequence |
| $i$ | Index within the embedding vector |
| $d_{\text{model}}$ | Model/embedding dimension (e.g., 512) |

### How Frequency Varies
- Small $i$ → **high frequency** → rapidly varying signal → differentiates **nearby** positions
- Large $i$ → **low frequency** → slowly varying signal → differentiates **distant** positions

### Key Properties
1. **Independent of sequence length** — same position always gets the same vector
2. **Nearby positions** differ mainly in the high-frequency (early) dimensions
3. **Distant positions** differ in both high-frequency and low-frequency dimensions
4. **Dot-product similarity** between nearby position vectors is high; decreases with distance

### Even/Odd Interleaving
- Even indices ($2i$) use $\sin$
- Odd indices ($2i+1$) use $\cos$ (same frequency)
- Mixing sin and cos further helps differentiate positions

---

## 4. Learned (Trainable) Positional Embeddings

- Treat each position's embedding as a **learnable parameter**
- A lookup table $\{P_1, P_2, \dots, P_{\max}\}$ shared across all sequences
- Updated via backpropagation during training

---

## 5. Three Desired Properties of Positional Embeddings

| Property | Definition |
|----------|------------|
| **Monotonicity** | Proximity of position vectors decreases as absolute distance increases: $\text{sim}(PE_x, PE_{x+n}) > \text{sim}(PE_x, PE_{x+m})$ when $n < m$ |
| **Translation Invariance** | Relative distance encoding is position-independent: $\text{sim}(PE_x, PE_{x+m}) = \text{sim}(PE_y, PE_{y+m})$ for any $x, y$ |
| **Symmetry** | $\text{sim}(PE_x, PE_y) = \text{sim}(PE_y, PE_x)$ |

---

## 6. Absolute vs. Relative Positional Encoding

### Absolute Positional Encoding
- Encode each position independently: either learned or sinusoidal
- Queries, keys, values all receive: $f(x_m + p_m)$

### Relative Positional Encoding
- Encode the **relative distance** $m - n$ between token pairs
- Only the relationship between positions matters, not their absolute values

### Key Papers on Relative Position Encoding

| Paper | Approach |
|-------|----------|
| **Shaw et al.** | Add relative position embedding $p_{r}$ (where $r = \text{clip}(m-n)$) to **keys and values only**; queries have no position info |
| **Transformer-XL** | Expand $q_m^T k_n$ into 4 terms; replace key position $p_n$ with $p_{m-n}$; replace query position $p_m$ with learnable biases $u, v$ |
| **T5** | Remove all position-dependent terms from $q^T k$; learn a scalar bias $b_{m-n}$ per relative distance |
| **DeBERTa** | Further modification of the Transformer-XL decomposition |

---

## 7. Rotary Positional Encoding (RoPE)

**Paper:** RoFormer (2023) — used in **LLaMA, PaLM, GPT-Neo, GPT-J**

### Core Idea
- Encode **absolute position** via a **rotation matrix**
- When computing self-attention ($q^T k$), the **relative position** $m - n$ is **automatically preserved**
- Position change → vectors **rotate**; relative angle stays constant

### Mathematical Formulation

**Goal:** Find functions $f_q, f_k, g$ such that:

$$\langle f_q(x_m, m),\; f_k(x_n, n) \rangle = g(x_m, x_n, m - n)$$

**Solution using complex exponentials:**

$$f_q(x_m, m) = (W_q x_m)\, e^{im\theta}$$
$$f_k(x_n, n) = (W_k x_n)\, e^{in\theta}$$

**Dot product:**

$$\langle f_q, f_k \rangle = \text{Re}\!\left[(W_q x_m)(W_k x_n)^* \cdot e^{i(m-n)\theta}\right]$$

→ Result depends on **relative position** $m - n$ only.

### Rotation Matrix (2D case)

$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

### Extension to $d$-Dimensional Vectors
- Block-diagonal matrix with $d/2$ rotation blocks
- Each block rotates a **pair of dimensions** with angle $\theta_i$:

$$\theta_i = 10000^{-2i/d}$$

### Key Properties
| Property | Description |
|----------|-------------|
| **Multiplicative** | Position info is multiplied (not added) to embeddings |
| **Relative position** | Naturally preserved through rotation matrix product |
| **No learnable parameters** | $\theta$ is deterministic — nothing to learn for position encoding |
| **Computationally efficient** | Sparse rotation matrix; separable cos/sin operations |

### Results (RoFormer)
- Machine translation (De→En): small BLEU improvement over vanilla Transformer
- BERT-style tasks: **71→86, 85→87** accuracy gains by just changing positional encoding
- **Faster convergence** than standard BERT

---

## 8. Residual Connections

$$\text{output} = F(x) + x$$

- Skip connection from input to output of each sub-layer
- **Addresses vanishing gradient**: gradient flows directly through the shortcut
- Applied **around each sub-layer**: self-attention, feed-forward network

---

## 9. Layer Normalization

### Purpose
- Stabilize and speed up training by keeping layer inputs in a consistent distribution
- Prevent gradient explosion/vanishing
- Accelerate convergence
- Help prevent overfitting (complementary to dropout)

### Batch Normalization vs. Layer Normalization

| Aspect | Batch Norm | Layer Norm |
|--------|-----------|------------|
| **Operates over** | Same feature across all samples in a batch | All features within a single sample |
| **Depends on batch size** | Yes — small batches → poor statistics | No — independent of batch |
| **Variable-length sequences** | Problematic (different lengths in batch) | No issue |
| **Train vs. inference** | Different processing (running stats) | Same processing |

### Layer Normalization Formula

For a vector $h = [h_1, h_2, \dots, h_d]$:

$$\mu = \frac{1}{d}\sum_{i=1}^d h_i, \qquad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (h_i - \mu)^2}$$

$$\hat{h}_i = \frac{h_i - \mu}{\sigma}$$

Optional learnable scale ($\gamma$) and shift ($\beta$):

$$y_i = \gamma\,\hat{h}_i + \beta$$

### Add & Norm in Transformer
- **Add** = residual connection
- **Norm** = layer normalization
- Applied after every sub-layer (self-attention, cross-attention, feed-forward)
- Reduces **covariate shift** between layers

---

## 10. Transformer Training Overview

- After the final decoder block: **Linear layer → Softmax → token prediction**
- Loss: **cross-entropy** at every decoder position (computed in parallel)
- Losses summed and backpropagated end-to-end through all layers
- All weights updated: attention, FFN, Add&Norm, input embeddings, position embeddings (if learnable)

### Encoder-Decoder Cross-Attention
- Every decoder layer attends to the **last encoder layer's** output (not layer-by-layer matching)
- Query from decoder; keys and values from the final encoder layer

---

## 11. Complete Transformer Block Summary

### Encoder Block
```
Input + Positional Encoding
    → Multi-Head Self-Attention → Add & Norm
    → Feed-Forward Network      → Add & Norm
```

### Decoder Block
```
Input + Positional Encoding
    → Masked Multi-Head Self-Attention → Add & Norm
    → Cross-Attention (Q: decoder, K/V: encoder) → Add & Norm
    → Feed-Forward Network → Add & Norm
```
