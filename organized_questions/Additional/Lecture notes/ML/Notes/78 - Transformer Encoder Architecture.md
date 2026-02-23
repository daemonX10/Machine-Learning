# Transformer Encoder Architecture

## Architecture Overview

The Transformer consists of two main parts:

```
[ENCODER] → [DECODER]
```

- **Encoder:** 6 identical encoder blocks stacked on top of each other
- **Decoder:** 6 identical decoder blocks stacked on top of each other
- The number 6 is **empirically determined** (not a magic number — varies across implementations)

> All 6 encoder blocks have **identical architecture** but **different parameter values** (weights/biases are learned independently during backpropagation).

---

## Simplified Representation

```
Transformer
├── Encoder (N=6 blocks)
│   ├── Encoder Block 1
│   ├── Encoder Block 2
│   ├── ...
│   └── Encoder Block 6
└── Decoder (N=6 blocks)
    ├── Decoder Block 1
    ├── ...
    └── Decoder Block 6
```

Each encoder block consists of **two sub-components:**

1. **Multi-Head Attention** (Self-Attention)
2. **Feed-Forward Neural Network**

Both sub-components are followed by **Add & Norm** (Residual Connection + Layer Normalization).

---

## Data Flow Through a Single Encoder Block

```
Input → [Multi-Head Attention] → [Add & Norm] → [Feed-Forward NN] → [Add & Norm] → Output
           ↑___(residual)___↗                       ↑___(residual)___↗
```

---

## Input Processing (Before Encoder)

Using example sentence: **"How are you"**

### Step 1: Tokenization

Split sentence into individual tokens (word-level):

```
"How are you" → ["How", "are", "you"]
```

### Step 2: Text Vectorization (Embedding)

Each word is converted to a **512-dimensional vector** via an embedding layer:

| Token | Vector Dimension |
|-------|-----------------|
| "How" | 512 |
| "are" | 512 |
| "you" | 512 |

### Step 3: Positional Encoding

Embeddings carry no positional information (which word comes first). Positional encoding fixes this:

$$x_i = \text{embedding}_i + \text{positional\_encoding}_i$$

For each position $i$, a **512-dimensional positional vector** is generated and **added element-wise** to the embedding vector.

| Token | Embedding (512-d) | + Positional Encoding (512-d) | = Input Vector (512-d) |
|-------|-------------------|-------------------------------|------------------------|
| "How" | $e_1$ | $p_1$ | $x_1 = e_1 + p_1$ |
| "are" | $e_2$ | $p_2$ | $x_2 = e_2 + p_2$ |
| "you" | $e_3$ | $p_3$ | $x_3 = e_3 + p_3$ |

---

## Inside the Encoder Block: Part 1 — Multi-Head Attention + Add & Norm

### Multi-Head Attention

Input: $x_1, x_2, x_3$ (each 512-d)

Multi-head attention generates **contextually-aware embeddings**:

$$x_i \text{ (512-d)} \xrightarrow{\text{Multi-Head Attention}} z_i \text{ (512-d)}$$

| Input | Output | Dimension |
|-------|--------|-----------|
| $x_1$ (How) | $z_1$ (contextual) | 512 |
| $x_2$ (are) | $z_2$ (contextual) | 512 |
| $x_3$ (you) | $z_3$ (contextual) | 512 |

**Why Multi-Head Attention?**

- Regular embeddings are **not context-aware** (e.g., "bank" in "river bank" vs "money bank" has the same embedding)
- Self-attention makes embeddings **consider surrounding words**
- Multi-head = multiple self-attention modules → **diverse contextual representations**

### Residual Connection (Add)

The **original input** $x_i$ bypasses the multi-head attention and is added to its output:

$$z_i' = z_i + x_i$$

| Component | Value | Dimension |
|-----------|-------|-----------|
| Multi-Head Attention output | $z_i$ | 512 |
| Original input (skip connection) | $x_i$ | 512 |
| **Sum** | $z_i' = z_i + x_i$ | **512** |

### Layer Normalization (Norm)

Applied to each vector independently:

$$z_i^{\text{norm}} = \text{LayerNorm}(z_i')$$

For each 512-d vector:
1. Compute **mean** $\mu$ and **standard deviation** $\sigma$ across the 512 values
2. Normalize: $\hat{z} = \frac{z - \mu}{\sigma}$
3. Scale and shift: $z^{\text{norm}} = \gamma \cdot \hat{z} + \beta$ (learned parameters)

**Why normalize?** Stabilizes training — self-attention outputs can have unbounded ranges; normalization keeps values in a consistent range.

---

## Inside the Encoder Block: Part 2 — Feed-Forward Network + Add & Norm

### Feed-Forward Neural Network Architecture

A **two-layer fully connected network** applied independently to each position:

```
Input (512) → [Linear + ReLU] → Hidden (2048) → [Linear] → Output (512)
```

| Layer | Input Dim | Output Dim | Activation | Weights | Biases |
|-------|-----------|------------|------------|---------|--------|
| Layer 1 | 512 | 2048 | **ReLU** | $W_1$ (512 × 2048) | $b_1$ (2048) |
| Layer 2 | 2048 | 512 | **Linear** | $W_2$ (2048 × 512) | $b_2$ (512) |

### Mathematical Formulation

$$\text{FFN}(x) = \max(0, \, xW_1 + b_1) \cdot W_2 + b_2$$

Or step-by-step:

$$\text{hidden} = \text{ReLU}(Z^{\text{norm}} \cdot W_1 + b_1)$$

$$\text{output} = \text{hidden} \cdot W_2 + b_2$$

Where $Z^{\text{norm}}$ is the matrix of normalized vectors (shape: $3 \times 512$).

### Dimension Flow

```
Input: 3 × 512
  → Layer 1: (3 × 512) × (512 × 2048) + bias = 3 × 2048  (dimension expanded)
  → ReLU applied
  → Layer 2: (3 × 2048) × (2048 × 512) + bias = 3 × 512  (dimension restored)
```

**Why expand then compress?** The expansion to 2048 followed by ReLU activation introduces **non-linearity** — self-attention is entirely linear, so the FFN captures non-linear patterns in the data.

### Residual Connection + Layer Normalization (Again)

Same as before:

$$y_i' = y_i + z_i^{\text{norm}}$$

$$y_i^{\text{norm}} = \text{LayerNorm}(y_i')$$

Where $y_i$ is the FFN output and $z_i^{\text{norm}}$ is the FFN input (bypassed via residual connection).

---

## Complete Single Encoder Block Flow

```
x₁, x₂, x₃  (512-d each)
       │
       ▼
┌──────────────────┐
│ Multi-Head        │◄── x (residual connection)
│ Attention         │
└──────┬───────────┘
       │ z = MHA(x)
       ▼
   z' = z + x         ← Add (residual)
       │
       ▼
   z_norm = LN(z')    ← Layer Normalization
       │
       ▼
┌──────────────────┐
│ Feed-Forward NN   │◄── z_norm (residual connection)
│ (512→2048→512)    │
└──────┬───────────┘
       │ y = FFN(z_norm)
       ▼
   y' = y + z_norm     ← Add (residual)
       │
       ▼
   y_norm = LN(y')    ← Layer Normalization
       │
       ▼
y₁_norm, y₂_norm, y₃_norm  (512-d each) → Next Encoder Block
```

---

## Stacking Encoder Blocks

**Output of Block $k$** → **Input of Block $k+1$**

```
Input (3 × 512) → Block 1 → Block 2 → Block 3 → Block 4 → Block 5 → Block 6 → Output (3 × 512)
```

- **Dimension is preserved throughout**: every intermediate output is always $3 \times 512$
- Each block has its **own unique weights** (not shared across blocks)
- Final output of Block 6 → sent to the **Decoder**

---

## Key Questions Answered

### Q1: Why Use Residual Connections?

The paper doesn't explicitly state why. Two speculated reasons:

| Reason | Explanation |
|--------|-------------|
| **Stable Training** | In deep networks, gradients can vanish. Residual connections provide an **alternate gradient path** that bypasses transformations, preventing gradient shrinkage. (Same idea as ResNet in CNNs) |
| **Feature Preservation** | If a transformation (e.g., Multi-Head Attention) performs poorly, the original features are still **passed forward** via the skip connection. Acts as a safety net. |

> Empirical evidence: Removing residual connections causes **significant performance degradation**.

### Q2: Why Use the Feed-Forward Network After Multi-Head Attention?

**Primary answer:** To introduce **non-linearity**.

- Self-attention operations are all **linear** (dot products, weighted sums)
- The FFN with **ReLU activation** captures non-linear complexity in the data
- FFN accounts for **~2/3 of all Transformer parameters**

> Research paper insight: "Transformer Feed-Forward Layers Are Key-Value Memories" suggests FFN layers act as **key-value memory stores** — each key correlates with textual patterns, and each value induces a distribution over the output vocabulary.

### Q3: Why 6 Encoder Blocks (Not Just 1)?

- Human language is inherently **complex**
- A single encoder block has limited **representation power**
- Stacking multiple blocks creates a **deeper model** that can capture increasingly abstract patterns
- 6 is an **empirically optimal** number from the original paper (varies in other architectures)
- This is the fundamental principle of **deep learning**: more layers → richer data representations

---

## Summary

| Component | Purpose | Input/Output Dim |
|-----------|---------|-----------------|
| Tokenization | Split text into tokens | Text → tokens |
| Embedding | Convert tokens to vectors | tokens → 512-d vectors |
| Positional Encoding | Add position information | 512-d → 512-d |
| Multi-Head Attention | Generate context-aware embeddings | 512-d → 512-d |
| Add & Norm (1st) | Residual connection + Layer Normalization | 512-d → 512-d |
| Feed-Forward NN | Add non-linearity (512→2048→512) | 512-d → 512-d |
| Add & Norm (2nd) | Residual connection + Layer Normalization | 512-d → 512-d |
| **6× stacked blocks** | Deeper representations of language | 512-d → 512-d |
