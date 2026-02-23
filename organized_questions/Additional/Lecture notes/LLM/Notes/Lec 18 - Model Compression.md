# Lecture 18 — Model Compression: Pruning & Knowledge Distillation

---

## 1. Motivation

Large models (trillion+ parameters) contain many redundant parameters. Model compression reduces size while preserving performance.

### Lottery Ticket Hypothesis (Frankle & Carlin, 2019)

> A dense network contains a sparse **subnetwork** (the "winning ticket") that, when trained in isolation from scratch, matches the full network's performance.

**Implication**: Most parameters are redundant — pruning can find the winning ticket.

---

## 2. Pruning Framework

```
┌───────────┐     ┌─────────┐     ┌─────────────┐     ┌──────────────────┐
│ Original W │ ──→ │ Pruning │ ──→ │  W' (pruned) │ ──→ │ Recovery Fine-   │
│            │     │  Step   │     │              │     │ Tuning           │
└───────────┘     └────┬────┘     └──────┬───────┘     └──────────────────┘
                       │                  │
                 Calibration         Evaluate on
                   Data              calibration data
```

| Data Type | Purpose |
|-----------|---------|
| **Calibration data** | Evaluate pruning quality; guide pruning decisions |
| **Recovery fine-tuning data** | Restore lost pre-trained knowledge after pruning |

### Accuracy vs Efficiency Trade-off
- Pruning ↑ → Accuracy ↓, Speed ↑
- Recovery fine-tuning partially restores accuracy
- Pruning + fine-tuning > pruning + retraining from scratch (cascading effect preserves pre-trained knowledge)

---

## 3. Unstructured Pruning

Individual elements removed from weight matrices. Matrix dimensions unchanged (zeros inserted).

### 3.1 Magnitude Pruning

Remove parameters with the **smallest absolute values**:

$$\text{mask}_{ij} = \begin{cases} 1 & \text{if } |W_{ij}| > \tau \\ 0 & \text{otherwise} \end{cases}$$

Simple but effective baseline.

### 3.2 Wanda (Weights and Activations)

Normalize weight magnitudes by **input activation magnitudes** before pruning:

$$S_{ij} = |W_{ij}| \cdot \|X_j\|$$

**Procedure**:
1. For each row, compute input-normalized scores $S_{ij}$
2. Set bottom $x\%$ per row to zero

**Advantage**: More stable than pure magnitude pruning; accounts for how much each weight actually affects the output.

### 3.3 SparseGPT

Model pruning as a **regression problem**:

$$\min_{M, \hat{W}} \| W X - (M \odot \hat{W}) X \|_2^2$$

where:
- $W$: original weight matrix
- $M$: binary mask (learnable)
- $\hat{W}$: proxy weight matrix (learnable)
- $\odot$: element-wise multiplication

| Property | Detail |
|----------|--------|
| Objective | NP-hard; use second-order Hessian approximation |
| Flexibility | Works for both structured and unstructured pruning |
| Disadvantage | Computationally expensive |

---

## 4. Structured Pruning

Remove entire rows, columns, heads, or layers. Matrix dimensions actually shrink → hardware-friendly speedup.

### 4.1 Attention Head Pruning (2019)

Learn a mask vector $\mathbf{m} \in \{0, 1\}^H$ over attention heads:

$$\text{output} = \sum_{h=1}^{H} m_h \cdot \text{Attn}_h(Q, K, V)$$

Similarly applicable to FFN layer paths (columns of $W_1$, corresponding rows of $W_2$).

### 4.2 SliceGPT

Apply **orthogonal transformation** before pruning to preserve information:

$$XW = (XQ)(Q^T W)$$

where $QQ^T = I$.

**Steps**:
1. Compute Gram matrix $C = X^T X$
2. Eigen-decompose $C$ to get orthogonal $Q$
3. Transform: $X' = XQ$, $W' = Q^T W$
4. Prune rows/columns in the transformed space

**Advantage**: Orthogonal transformation preserves the output $XW = X'W'$.

### 4.3 Layer Collapse

Compare outputs of adjacent layers:

$$\text{sim}(h_l, h_{l+1}) > \tau \implies \text{merge layers } l \text{ and } l+1$$

If two layers produce nearly identical outputs, one is redundant and can be removed.

---

## 5. PruneNet (ICLR) — Calibration-Free Pruning

### Problem with Existing Methods
- SliceGPT/SparseGPT require calibration data
- The orthogonal transformation $Q$ adds parameters; if compression ratio $R$ is below a threshold, **more parameters are added than removed**:

$$R < \frac{d_{\text{hidden}}}{d_{\text{hidden}} + d_{\text{intermediate}}}$$

### Core Theory: Poincaré Separation Theorem

For $W \in \mathbb{R}^{n \times d}$ and a submatrix $W' \in \mathbb{R}^{m \times d}$ ($m \leq n$):

> The singular values of $W'$ are **interlaced** within the singular values of $W$.

**Goal**: Choose $W'$ such that the **distribution of singular values** of $W'$ is as close as possible to that of $W$.

### Architecture

```
┌────────────────┐
│ FFN1 (up-proj)  │  ← W ∈ ℝ^{n×d}
│  weight matrix  │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Policy Network  │  ← MLP: input = W, output = importance vector ∈ ℝ^n
│    (MLP)        │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Select top-k%   │  ← Based on compression ratio R
│ rows by score   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Pruned FFN1     │  ← W' ∈ ℝ^{m×d}, drop corresponding columns from FFN2
└────────────────┘
```

### Reward Function

Measure **Kolmogorov-Smirnov (KS) distance** between singular value distributions:

$$r = -D_{KS}\!\left(\sigma(W), \sigma(W')\right)$$

where $\sigma(W)$ = sorted singular values of $W$.

Train the policy network using **PPO** to maximize this reward.

### Key Advantages

| Property | PruneNet | SliceGPT / SparseGPT |
|----------|----------|----------------------|
| Calibration data | **Not needed** | Required |
| Compression ratio flexibility | Adjust $R$ without retraining | Must retrain for new $R$ |
| Architecture dependency | Agnostic | Model-specific |
| Additional parameters | None (policy is separate) | $Q$ matrix added |

### Results

| Metric | Value |
|--------|-------|
| Throughput increase | 73% |
| Zero-shot performance retention | 80% |
| LLaMA-2 compression time | ~15 minutes |

---

## 6. Knowledge Distillation

### Concept
Transfer knowledge from a large **teacher** model to a small **student** model.

```
┌──────────────┐         ┌──────────────┐
│ Teacher Model │         │ Student Model │
│  (large, T)   │         │  (small, S)   │
└──────┬───────┘         └──────┬───────┘
       │                        │
       ▼                        ▼
   P_T(y|x)                P_S(y|x)
       │                        │
       └────── KL Divergence ───┘
```

### Loss Function

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{task}} + (1 - \alpha) \cdot T^2 \cdot D_{KL}\!\left(P_T^{(T)} \| P_S^{(T)}\right)$$

where:
- $\mathcal{L}_{\text{task}}$: standard cross-entropy with ground truth
- $P_T^{(T)}, P_S^{(T)}$: softened distributions using temperature $T$
- $T^2$: scaling factor for gradient magnitude balance

### Soft Labels (Temperature Scaling)

$$P_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Higher temperature → softer distribution → more information about teacher's learned similarities.

---

## 7. Summary

| Method | Type | Key Idea |
|--------|------|----------|
| **Magnitude Pruning** | Unstructured | Remove smallest weights |
| **Wanda** | Unstructured | Input-normalized magnitude |
| **SparseGPT** | Unstructured/Structured | Regression-based mask learning |
| **Head Pruning** | Structured | Learn attention head mask |
| **SliceGPT** | Structured | Orthogonal transform + prune |
| **Layer Collapse** | Structured | Merge similar adjacent layers |
| **PruneNet** | Structured | Policy-learned row importance; calibration-free |
| **Knowledge Distillation** | — | Teacher→Student via soft labels |

### Comparison: Pruning Approaches

| | Unstructured | Structured |
|---|---|---|
| **Granularity** | Individual elements | Rows, columns, heads, layers |
| **Matrix size** | Unchanged (sparse) | Reduced |
| **Hardware speedup** | Needs sparse hardware | Direct speedup |
| **Methods** | Magnitude, Wanda, SparseGPT | Head pruning, SliceGPT, PruneNet |
