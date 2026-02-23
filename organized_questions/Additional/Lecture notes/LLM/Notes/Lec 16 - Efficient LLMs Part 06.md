# Lecture 16 — Efficient LLMs Part 06: Flash Attention Details, MoE & Neural Scaling Laws

---

## 1. Language Modeling Fundamentals

### What We Model
A language model assigns probability to sequences:

$$P_\theta(X) = \prod_{t=1}^{T} P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

- **Decoder-only**: Autoregressive factorization (left-to-right); used for generation
- **Encoder-only**: Bidirectional attention; cannot generate left-to-right (future tokens used in conditioning)
- **Encoder-decoder**: Encoder encodes input bidirectionally; decoder generates autoregressively with cross-attention

### Training Forward Pass
- Input: sequence $(x_1, \ldots, x_T)$ → Output: logits of shape $T \times V$
- Labels: input shifted left by one position
- Loss: sum of cross-entropy at each position

### Why Decoder-Only Models Prevailed
Hypothesis: the causal constraint (no future tokens) acts as a **regularizer** during large-scale training.

---

## 2. Blockwise Attention — Detailed Derivation

### Target Expression
For query $Q_i$ attending over all keys $K_1, \ldots, K_T$:

$$O_i = \frac{\sum_{j=1}^{T} \exp(Q_i \cdot K_j^T) \cdot V_j}{\sum_{j=1}^{T} \exp(Q_i \cdot K_j^T)}$$

### Splitting into Blocks
Partition keys/values into blocks $B_1, B_2$. For each block compute:

- $L_k = \sum_{j \in B_k} \exp(Q_i \cdot K_j^T)$ — local normalization sum
- $O_k = \frac{\sum_{j \in B_k} \exp(Q_i \cdot K_j^T) \cdot V_j}{L_k}$ — local weighted output

### Combining Blocks

$$O = \frac{L_1 \cdot O_1 + L_2 \cdot O_2}{L_1 + L_2}$$

This generalizes to $n$ blocks:

$$O = \frac{\sum_{k=1}^{n} L_k \cdot O_k}{\sum_{k=1}^{n} L_k}$$

---

## 3. Flash Attention — Online Algorithm

### Sequential Block Processing
When SRAM is too small to hold all KV blocks simultaneously:

```
Initialize: L = 0, O = 0 (in HBM)

For each KV block j:
    Load K_j, V_j into SRAM
    Compute L_j = Σ exp(Q · K_j^T)     (local sum)
    Compute O_j = softmax_local(Q, K_j) · V_j  (local output)
    
    # Online update:
    O ← (L · O + L_j · O_j) / (L + L_j)
    L ← L + L_j
    
    Write updated O, L back to HBM
```

### Safe Softmax
Subtract max value before exponentiation for numerical stability:

$$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$$

### Tiling
Efficient partitioning of Q, K, V matrices into blocks that fit in SRAM. Flash Attention tiles along both Q and KV dimensions.

---

## 4. Flash Decoding

### Problem
During **decoding** (1 query, $T$ cached KV pairs), the standard flash attention parallelizes over queries — but there's only **one query**.

### Solution: Parallelize Over KV Blocks

1. Load all KV blocks across GPU threads **simultaneously**
2. Each thread computes partial $(O_k, L_k)$ for its KV block
3. **Reduce** in HBM: combine all partial results using online softmax

$$O = \frac{\sum_k L_k \cdot O_k}{\sum_k L_k}$$

### Comparison

| | Flash Attention | Flash Decoding |
|---|---|---|
| Parallelism | Over queries | Over KV blocks |
| Best for | Prefill (many queries) | Decode (1 query, many KVs) |

---

## 5. Speculative Decoding

### Problem
Standard autoregressive decoding: 1 forward pass per token → slow.

### Idea
Use a **small draft model** to guess $k$ future tokens. Then verify all $k$ tokens in a **single forward pass** of the large model.

| Step | Action |
|------|--------|
| 1 | Draft model generates $k$ candidate tokens |
| 2 | Concatenate candidates with current sequence |
| 3 | Large model does **one** forward pass over all $k+1$ positions |
| 4 | Accept tokens that match large model's distribution |
| 5 | Reject and resample from first mismatch onward |

**Speedup**: Up to $k\times$ if draft model predictions are accurate.

---

## 6. Neural Scaling Laws (Kaplan et al., 2020)

LLM loss follows a **power law** with respect to model size, data, and compute:

$$L(N) = \alpha \cdot N^{-\beta}$$

| Variable | What it Scales |
|----------|---------------|
| $N$ | Number of parameters |
| $D$ | Dataset size (tokens) |
| $C$ | Compute budget (FLOPs) |

**Key insights**:
- Larger models are more sample-efficient
- Performance improves predictably with scale
- Optimal model size depends on compute budget

---

## 7. Mixture of Experts (MoE)

### Architecture
Replace the **FFN layer** with multiple parallel **expert FFNs** + a gating network:

$$\text{MoE}(x) = \sum_{i=1}^{N} g(x)_i \cdot \text{FFN}_i(x)$$

where $g(x) = \text{softmax}(W_G \cdot x)$ routes tokens to experts.

### Top-K Routing
Only top-$k$ experts activated per token (typically $k = 2$):

$$\text{MoE}(x) = \sum_{i \in \text{top-}k} g(x)_i \cdot \text{FFN}_i(x)$$

### Parameter vs Compute Analysis

| Component | Params per Layer |
|-----------|-----------------|
| Self-Attention (QKV + O) | $4H^2$ |
| Single FFN (up + down) | $8H^2$ |
| MoE with $E$ experts | $8H^2 \times E$ |

With 10 experts and top-2 routing: **10× capacity, same compute** as dense model (only 2 experts active per token).

### Timeline

| Year | Model/Paper | Contribution |
|------|------------|--------------|
| 1991 | Hinton et al. | Original MoE concept |
| 2021 | Switch Transformer | Simplified top-1 routing for transformers |
| 2024 | Mixtral | Open-source MoE LLM |
| 2024 | GPT-4 | Rumored MoE architecture |
| 2025 | DeepSeek | Open-source MoE with expert parallelism |

### Self-Attention Is Shared
Only FFN layers are replaced with MoE. Self-attention remains shared across all tokens.

---

## 8. Expert Collapse & Prevention

### Why Experts Don't Collapse
If experts had no nonlinearity (linear experts), the loss surface would be convex → experts converge to the same solution. The **nonlinearity** in FFN experts creates a non-convex loss → experts diverge to different specializations.

### Router Collapse Problem
Even with non-linear experts, the **router** may collapse → sending all tokens to the same expert.

### Load Balancing Loss
Auxiliary loss to encourage uniform expert utilization:

$$\mathcal{L}_{\text{balance}} = N \cdot \sum_{i=1}^{N} f_i \cdot p_i$$

where:
- $f_i = \frac{\text{tokens routed to expert } i}{\text{total tokens}}$ (fraction, non-differentiable)
- $p_i = \frac{1}{T}\sum_{t} \text{softmax}(W_G \cdot x_t)_i$ (average routing probability, differentiable)

**Why this works**: Minimizing $\sum f_i \cdot p_i$ subject to $\sum f_i = 1$, $\sum p_i = 1$ pushes toward uniform allocation: $f_i = p_i = \frac{1}{N}$.

---

## 9. Expert Parallelism — The 5th Dimension

### Concept
Distribute experts across GPUs. Each GPU holds a subset of experts.

### Communication Pattern: All-to-All

| Step | Action |
|------|--------|
| 1 | Router determines which expert each token goes to |
| 2 | **All-to-all dispatch**: tokens sent to GPUs holding their target experts |
| 3 | Each GPU processes tokens through its local experts |
| 4 | **All-to-all combine**: results sent back to source GPUs |

### 5D Parallelism

| Dimension | What is Split |
|-----------|--------------|
| Data | Batch |
| Tensor | Hidden dimensions / heads |
| Context | Sequence length |
| Pipeline | Layers |
| Expert | FFN experts across GPUs |

---

## 10. Comprehensive Recap — GPU Memory Contents

| Component | Memory Usage |
|-----------|-------------|
| **Weights** | Model parameters |
| **Gradients** | Same size as weights |
| **Optimizer States** | 2× weights (Adam: $m$ and $v$) |
| **Activations** | Intermediate values for backward pass |

### Techniques Summary

| Technique | What it Reduces |
|-----------|----------------|
| ZeRO Stage 1 | Shard optimizer states |
| ZeRO Stage 2 | + Shard gradients |
| ZeRO Stage 3 | + Shard weights |
| Gradient Accumulation | Activation memory (smaller micro-batch) |
| Gradient Checkpointing | Activation memory (recompute in backward) |
| Tensor Parallelism | Split within a layer (intra-node) |
| Sequence/Context Parallelism | Split sequence dimension |
| Pipeline Parallelism | Split layers across nodes |
| Expert Parallelism | Split experts across GPUs |
| Flash Attention | Attention memory $O(T^2) \to O(T)$ |
| Fused Kernels | Reduce HBM ↔ SRAM transfers |
