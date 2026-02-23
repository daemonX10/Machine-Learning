# Transformer Decoder Architecture (Training Perspective)

## Overview

- The decoder generates the output sequence (e.g., Hindi translation)
- During **training**: decoder is **non-autoregressive** — all output tokens processed in parallel
- During **inference**: decoder is **autoregressive** — generates one token at a time
- This document covers the **training-time** architecture

---

## High-Level Structure

### Transformer = Encoder + Decoder

```
Transformer
├── Encoder (6 identical blocks)
│   └── Each block: Self-Attention → Feed Forward
└── Decoder (6 identical blocks)
    └── Each block: Masked Self-Attention → Cross Attention → Feed Forward
```

### Single Encoder Block (2 sub-layers)

1. Multi-Head Self-Attention
2. Feed-Forward Neural Network

### Single Decoder Block (3 sub-layers)

1. **Masked** Multi-Head Self-Attention
2. **Cross Attention** (Encoder-Decoder Attention)
3. Feed-Forward Neural Network

> All 6 decoder blocks are **architecturally identical** (same structure, different learned parameters).

---

## Three-Part Walkthrough

The decoder explanation is divided into:

1. **Input Preparation** — preparing the target sentence for the decoder
2. **Decoder Block Processing** — what happens inside a single decoder block
3. **Output Block** — converting final vectors into word predictions

---

## Part 1: Input Preparation

**Example:** Translating "We are friends" → "हम दोस्त हैं"

The input block performs **4 operations** on the target (Hindi) sentence:

### Step 1: Right Shifting

Add a `<START>` token at the beginning of the target sentence:

$$\text{"हम दोस्त हैं"} \xrightarrow{\text{shift right}} \text{"<START> हम दोस्त हैं"}$$

> The start token signals the decoder to begin generation.

### Step 2: Tokenization

Break the shifted sentence into individual tokens:

$$\text{"<START> हम दोस्त हैं"} \rightarrow [\text{<START>}, \text{हम}, \text{दोस्त}, \text{हैं}]$$

### Step 3: Embedding

Each token is converted to a **512-dimensional vector**:

$$e_1 \leftarrow \text{<START>}, \quad e_2 \leftarrow \text{हम}, \quad e_3 \leftarrow \text{दोस्त}, \quad e_4 \leftarrow \text{हैं}$$

All vectors: $e_i \in \mathbb{R}^{512}$

### Step 4: Positional Encoding

Add position information (since transformers have no inherent sequence order):

$$x_i = e_i + \text{PE}(i)$$

**Final input vectors:** $x_1, x_2, x_3, x_4 \in \mathbb{R}^{512}$

These are fed into the **first decoder block**.

---

## Part 2: Inside a Single Decoder Block

### Sub-Layer 1: Masked Multi-Head Self-Attention

Input: $x_1, x_2, x_3, x_4$

For each input vector, produces a **contextual embedding** $z_i$, but with **masking**:

| Output | Can attend to | Cannot attend to |
|--------|--------------|-----------------|
| $z_1$ (for `<START>`) | `<START>` only | हम, दोस्त, हैं |
| $z_2$ (for हम) | `<START>`, हम | दोस्त, हैं |
| $z_3$ (for दोस्त) | `<START>`, हम, दोस्त | हैं |
| $z_4$ (for हैं) | `<START>`, हम, दोस्त, हैं | — |

> Masking ensures each position can only attend to **current and previous** positions (causal masking). Future tokens are set to $-\infty$ before softmax → zero attention weight.

**Then: Add & Norm**

$$z_i' = z_i + x_i \quad \text{(residual connection)}$$

$$z_i^{\text{norm}} = \text{LayerNorm}(z_i')$$

---

### Sub-Layer 2: Cross Attention (Encoder-Decoder Attention)

This is the **most important** block — it connects encoder and decoder.

**Two inputs:**
- From previous sub-layer (decoder): $z_1^{\text{norm}}, z_2^{\text{norm}}, z_3^{\text{norm}}, z_4^{\text{norm}}$ → **Query vectors**
- From encoder output: encoded English word vectors → **Key and Value vectors**

$$Q = Z^{\text{norm}}_{\text{decoder}} \cdot W_Q$$

$$K = H_{\text{encoder}} \cdot W_K, \quad V = H_{\text{encoder}} \cdot W_V$$

Cross attention computes: for each Hindi token, **which English tokens are most relevant?**

**Output:** Contextual embeddings $z_{c1}, z_{c2}, z_{c3}, z_{c4}$ — one per output token.

**Then: Add & Norm**

$$z_{ci}' = z_{ci} + z_i^{\text{norm}} \quad \text{(residual connection)}$$

$$z_{ci}^{\text{norm}} = \text{LayerNorm}(z_{ci}')$$

---

### Sub-Layer 3: Feed-Forward Neural Network

**Architecture:**

| Layer | Neurons | Activation |
|-------|---------|------------|
| Hidden Layer 1 | 2048 | ReLU |
| Output Layer | 512 | Linear |

**Parameters:**

$$W_1 \in \mathbb{R}^{512 \times 2048}, \quad b_1 \in \mathbb{R}^{2048}$$

$$W_2 \in \mathbb{R}^{2048 \times 512}, \quad b_2 \in \mathbb{R}^{512}$$

**Computation (batch of 4 vectors):**

$$\text{FFN}(Z) = \text{ReLU}(Z \cdot W_1 + b_1) \cdot W_2 + b_2$$

Shape flow:

$$(4 \times 512) \xrightarrow{W_1} (4 \times 2048) \xrightarrow{\text{ReLU}} (4 \times 2048) \xrightarrow{W_2} (4 \times 512)$$

> Purpose: Capture **non-linear** transformations. Input and output dimensions remain 512.

**Output:** $y_1, y_2, y_3, y_4 \in \mathbb{R}^{512}$

**Then: Add & Norm**

$$y_i' = y_i + z_{ci}^{\text{norm}} \quad \text{(residual connection)}$$

$$y_i^{\text{norm}} = \text{LayerNorm}(y_i')$$

---

### Stacking 6 Decoder Blocks

The output of decoder block 1 ($y_1^{\text{norm}}, y_2^{\text{norm}}, y_3^{\text{norm}}, y_4^{\text{norm}}$) feeds into decoder block 2, which performs the **same 3 sub-layers** with different parameters.

This repeats through all 6 blocks. The final output:

$$y_{f1}^{\text{norm}}, y_{f2}^{\text{norm}}, y_{f3}^{\text{norm}}, y_{f4}^{\text{norm}} \in \mathbb{R}^{512}$$

where $f$ = final (from 6th decoder block).

---

## Part 3: Output Block (Linear + Softmax)

### Architecture

A single dense layer followed by softmax:

| Component | Detail |
|-----------|--------|
| Input dimension | 512 |
| Output dimension | $V$ (Hindi vocabulary size) |
| Weights | $W_3 \in \mathbb{R}^{512 \times V}$, biases $b_3 \in \mathbb{R}^{V}$ |
| Activation | Softmax |

> $V$ = total number of unique Hindi words in the training dataset (e.g., 10,000). Each neuron corresponds to one vocabulary word.

### Computation

For each token's final vector $y_{fi}^{\text{norm}}$:

$$\text{logits} = y_{fi}^{\text{norm}} \cdot W_3 + b_3 \quad \in \mathbb{R}^{V}$$

$$P(\text{word}_j) = \text{softmax}(\text{logits})_j = \frac{e^{\text{logit}_j}}{\sum_{k=1}^{V} e^{\text{logit}_k}}$$

### Prediction

Select the word with the **highest probability**:

| Input Token | Predicted Output |
|-------------|-----------------|
| `<START>` | हम (highest P) |
| हम | दोस्त (highest P) |
| दोस्त | हैं (highest P) |
| हैं | `<END>` (highest P) |

> During training, the predicted outputs are compared with ground-truth tokens using **cross-entropy loss**, and gradients flow back through the entire architecture.

---

## Complete Data Flow Summary

```
Target sentence: "हम दोस्त हैं"
    ↓ Right Shift
"<START> हम दोस्त हैं"
    ↓ Tokenize
[<START>, हम, दोस्त, हैं]
    ↓ Embedding (512-d)
[e1, e2, e3, e4]
    ↓ + Positional Encoding
[x1, x2, x3, x4]
    ↓ Decoder Block 1
        ├─ Masked Self-Attention → Add & Norm
        ├─ Cross Attention (with encoder output) → Add & Norm
        └─ Feed-Forward → Add & Norm
    ↓ Decoder Blocks 2–6 (same operations, different params)
[yf1_norm, yf2_norm, yf3_norm, yf4_norm]
    ↓ Linear Layer (512 → V)
    ↓ Softmax
[P(word) for each position]
    ↓ argmax
[हम, दोस्त, हैं, <END>]
```

---

## Key Points

| Aspect | Detail |
|--------|--------|
| **Decoder blocks** | 6 identical blocks, each with 3 sub-layers |
| **Sub-layers** | Masked Self-Attention → Cross Attention → FFN |
| **Residual connections** | After every sub-layer |
| **Layer normalization** | After every residual addition |
| **Training mode** | Non-autoregressive (all tokens processed in parallel) |
| **FFN architecture** | 512 → 2048 (ReLU) → 512 (Linear) |
| **Output layer** | Linear ($512 \times V$) + Softmax → probability over vocabulary |
