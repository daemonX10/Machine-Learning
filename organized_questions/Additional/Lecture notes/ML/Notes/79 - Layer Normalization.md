# Layer Normalization in Transformers

## Why Normalization in Deep Learning?

Normalization = transforming data to have specific statistical properties (e.g., mean=0, std=1).

### Where Is Normalization Applied?

1. **Input data** — normalize features before feeding to the network
2. **Hidden layer activations** — normalize outputs of intermediate layers

### Benefits of Normalization

| Benefit | Description |
|---------|-------------|
| **Training Stability** | Prevents extreme values that cause exploding gradients |
| **Faster Convergence** | Consistent gradient magnitudes → model converges more quickly |
| **Reduced Internal Covariate Shift** | Keeps hidden layer input distributions stable across training steps |
| **Regularization Effect** | Some forms (e.g., Batch Norm) add slight regularization |

---

## Internal Covariate Shift

### Covariate Shift

- Training data has one distribution, but test/prediction data has a **different distribution**
- Example: Trained flower classifier on red roses, tested on yellow/white roses

### Internal Covariate Shift

- The **hidden layer outputs (activations)** change their distribution over time during training
- Why? Because the weights feeding into a hidden layer **keep updating** via backpropagation
- The next layer effectively sees a **shifting input distribution** every iteration

> Normalization mitigates this by keeping activations in a stable range.

---

## Batch Normalization (Quick Review)

### Setup

- Data: $n$ rows, features $f_1, f_2, \ldots$
- Neural network with hidden layer having 3 nodes
- Pre-activations: $z_1, z_2, z_3$ for each node
- Batch size = 5 (5 rows processed together)

### How It Works

Normalize **across the batch** (column-wise):

For node $j$ with pre-activations $z_j^{(1)}, z_j^{(2)}, \ldots, z_j^{(B)}$ across batch of size $B$:

$$\mu_j = \frac{1}{B} \sum_{i=1}^{B} z_j^{(i)}$$

$$\sigma_j = \sqrt{\frac{1}{B} \sum_{i=1}^{B} (z_j^{(i)} - \mu_j)^2}$$

$$\hat{z}_j^{(i)} = \frac{z_j^{(i)} - \mu_j}{\sigma_j}$$

$$\tilde{z}_j^{(i)} = \gamma_j \cdot \hat{z}_j^{(i)} + \beta_j$$

| Symbol | Meaning |
|--------|---------|
| $\mu_j$ | Mean of node $j$'s activations across the batch |
| $\sigma_j$ | Std dev of node $j$'s activations across the batch |
| $\gamma_j$ | Learnable scale parameter (initialized to 1) |
| $\beta_j$ | Learnable shift parameter (initialized to 0) |

**Direction of normalization:** Down the column (across samples in the batch).

Each node has its **own** $\gamma$ and $\beta$ parameters.

---

## Why Batch Normalization Fails for Sequential Data (Transformers)

### The Padding Problem

When processing sentences of **different lengths** in a batch:

1. Sentences must be **padded** to equal length (with zero vectors)
2. Shorter sentences get many **padding zeros**

**Example:**
- Batch of 32 sentences
- Longest sentence: 100 words
- Average sentence: ~30 words
- Result: **Most sentences have ~70 rows of padding zeros**

### Why This Breaks Batch Norm

When computing $\mu_j$ (column-wise mean):

```
Column of activations:
[6.5, 2.41, 78.3, 0, 0, 0, 0, 0, ...]  ← many padding zeros!
```

- The **padding zeros are artificial** (not real data)
- Including them in mean/std calculation → **biased statistics**
- The computed $\mu$ and $\sigma$ are **not representative** of the actual data
- This distorts normalization → degrades model performance

> **Key insight:** Batch normalization computes statistics across the batch dimension. When sentences have vastly different lengths with lots of padding, these cross-batch statistics become meaningless.

---

## Layer Normalization

### Core Difference from Batch Norm

| Property | Batch Normalization | Layer Normalization |
|----------|-------------------|-------------------|
| **Normalize across** | Batch (column-wise) | Features (row-wise) |
| **Direction** | ↓ Down the column | → Across the row |
| **Depends on batch** | Yes | **No** |
| **Padding impact** | Severely affected | **Not affected** |

### How It Works

For a single data point $i$ with feature values $z_1^{(i)}, z_2^{(i)}, \ldots, z_D^{(i)}$:

$$\mu^{(i)} = \frac{1}{D} \sum_{j=1}^{D} z_j^{(i)}$$

$$\sigma^{(i)} = \sqrt{\frac{1}{D} \sum_{j=1}^{D} (z_j^{(i)} - \mu^{(i)})^2}$$

$$\hat{z}_j^{(i)} = \frac{z_j^{(i)} - \mu^{(i)}}{\sigma^{(i)}}$$

$$\tilde{z}_j^{(i)} = \gamma_j \cdot \hat{z}_j^{(i)} + \beta_j$$

| Symbol | Meaning |
|--------|---------|
| $\mu^{(i)}$ | Mean across all features **for sample $i$** |
| $\sigma^{(i)}$ | Std dev across all features **for sample $i$** |
| $\gamma_j$ | Learnable scale per feature $j$ (initialized to 1) |
| $\beta_j$ | Learnable shift per feature $j$ (initialized to 0) |

**Direction of normalization:** Across the row (across features for one sample).

### Why It Solves the Padding Problem

For a **real word's** embedding (e.g., "How"):
- $\mu$ and $\sigma$ computed from its **own 3 (or 512) embedding dimensions**
- No other sentences' data is involved
- Padding zeros in other rows have **zero impact** on this computation

For a **padding row** (all zeros):
- $\mu = 0$, $\sigma = 0$
- After normalization → still $0$
- Padding **only affects itself**, never contaminates real data statistics

> **This is the key advantage:** Each token's normalization is independent of other tokens and other sentences in the batch.

---

## Layer Normalization in Transformers

### Where It's Applied

In each encoder/decoder block, Layer Norm is applied **twice**:

```
x → Multi-Head Attention → Add → LayerNorm → FFN → Add → LayerNorm → output
```

### Concrete Example (Self-Attention Output)

After self-attention produces contextual embeddings for sentences in a batch:

| | Dim 1 | Dim 2 | Dim 3 |
|---|-------|-------|-------|
| **S1: "hi"** | 6.5 | 2.41 | 78.3 |
| **S1: "nitesh"** | 99.2 | 33.4 | 1.2 |
| **S1: pad** | 0 | 0 | 0 |
| **S1: pad** | 0 | 0 | 0 |
| **S2: "how"** | 15.3 | 44.1 | 7.8 |
| **S2: "are"** | 22.6 | 8.9 | 55.1 |
| **S2: "you"** | 3.4 | 67.2 | 11.5 |
| **S2: "today"** | 41.8 | 5.6 | 90.3 |

**Layer Norm computation for "hi":**
- $\mu_1 = \frac{6.5 + 2.41 + 78.3}{3}$, $\sigma_1 = ...$
- Uses **only** dim 1, dim 2, dim 3 of "hi" — no other rows involved

**Layer Norm for padding row:**
- $\mu = 0$, $\sigma = 0$ → output stays $0$
- **Does not pollute** any other row's statistics

**If Batch Norm were used instead:**
- Column 1 mean: $\frac{6.5 + 99.2 + 0 + 0 + 15.3 + 22.6 + 3.4 + 41.8}{8}$ ← zeros pulled the mean down artificially!

---

## Batch Norm vs Layer Norm — Visual Comparison

```
Data Matrix (rows = samples, columns = features):

Batch Norm:          Layer Norm:
Normalize ↓          Normalize →
┌─────────┐          ┌─────────┐
│ → → → → │          │ ↓       │
│ → → → → │          │ ↓       │
│ → → → → │          │ ↓       │
│ → → → → │          │ ↓       │
└─────────┘          └─────────┘
Each column gets      Each row gets
its own μ, σ         its own μ, σ
```

---

## Summary

| Aspect | Batch Normalization | Layer Normalization |
|--------|-------------------|-------------------|
| **Normalizes across** | Batch (samples) | Features (within one sample) |
| **μ, σ computed from** | All samples in batch for one feature | All features for one sample |
| **Learnable params** | $\gamma_j, \beta_j$ per feature | $\gamma_j, \beta_j$ per feature |
| **Affected by padding** | **Yes** — zeros corrupt statistics | **No** — each sample normalized independently |
| **Used in Transformers** | ✗ | **✓** |
| **Works for sequential data** | Poorly (variable-length sequences + padding) | **Well** |
| **Depends on batch size** | Yes | No |

> **Bottom line:** Transformers use **Layer Normalization** because it normalizes each token's embedding independently, avoiding the padding-zero corruption problem that plagues Batch Normalization on sequential data.
