# Lecture 13 - Efficient LLMs (Part 03)

## Overview
This lecture covers tensor parallelism for multi-head attention, sequence parallelism for layer norm/dropout, and context parallelism (ring attention) for handling long sequences.

---

## 1. Recap: Memory Components

### Static Memory (Input-Independent)
- Per layer: $12H^2 + 13H$ parameters
- Total: $L \times (12H^2 + 13H) + V \times H + 2H$
- Mixed precision total: $16N$ bytes (params + grads + optimizer)

### Dynamic Memory (Activations)
- Linear layers: $34 \times B \times S \times H$ (per layer)
- Attention: $5 \times B \times n_h \times S^2$ (per layer)
- Attention is $O(S^2)$ — the main bottleneck

### Memory Reduction Toolkit (So Far)
| Technique | What it Reduces |
|-----------|----------------|
| Activation recomputation | Activation memory (up to 70%) |
| Gradient accumulation | Effective batch size without memory increase |
| ZeRO 1/2/3 | Optimizer, gradient, parameter memory across GPUs |

**Remaining problem**: Activations for a single sequence can still exceed GPU memory.

---

## 2. Tensor Parallelism — MLP Block

### Column-Wise Split (First Linear Layer)

Given $U = X^T W$ where $W \in \mathbb{R}^{H \times 4H}$:

Split $W$ by **columns** across $N$ GPUs:
$$W = [W_1 | W_2 | ... | W_N]$$

- GPU $i$ computes: $X^T W_i$ → produces $\frac{4H}{N}$ neurons
- Same input $X$ on all GPUs
- **No communication needed** at this stage

### Row-Wise Split (Second Linear Layer)

Given output $O = U^T M$ where $M \in \mathbb{R}^{4H \times H}$:

Split $M$ by **rows** to match the column split:
$$M = \begin{bmatrix} M_1 \\ M_2 \\ \vdots \\ M_N \end{bmatrix}$$

Each GPU computes a **partial output**:
$$O = U_1^T M_1 + U_2^T M_2 + ... + U_N^T M_N$$

→ Requires **all-reduce** to sum partial results.

### What's Achieved
- Parameters split across GPUs (each GPU holds $\frac{1}{N}$)
- Activations split along **hidden dimension** ($H$ axis)
- Gradients stay local to each GPU

---

## 3. Tensor Parallelism — Multi-Head Attention

### Natural Parallelism
Each attention head has independent parameters → assign heads to different GPUs.

| GPU | Assigned Heads | Computation |
|-----|---------------|-------------|
| GPU 1 | Heads 1 to $\frac{n_h}{N}$ | QKV + attention for assigned heads |
| GPU 2 | Heads $\frac{n_h}{N}+1$ to $\frac{2n_h}{N}$ | QKV + attention for assigned heads |
| ... | ... | ... |

### Final Projection Layer
After attention, concatenated head outputs pass through a linear layer $B$:
- Split $B$ **row-wise** (matching the head split)
- Each GPU processes its portion: $Z_i = Y_i \cdot B_i$
- **All-reduce** to get final output: $Z = Z_1 + Z_2 + ... + Z_N$

---

## 4. Redundant Operations in Tensor Parallelism

### Problem
Between tensor-parallel regions, some operations run identically on all GPUs:

| Operation | Issue |
|-----------|-------|
| **Layer Norm** | Same operation, same input on all TP GPUs → wasted compute |
| **Dropout** | Same operation, same input → must also sync dropout masks |

These operate **independently per token** (no cross-token interaction).

---

## 5. Sequence Parallelism

### Idea
For token-independent operations (layer norm, dropout, MLP), split along the **sequence dimension** instead of replicating.

| Parallelism Type | Split Dimension | Operations |
|-----------------|-----------------|------------|
| Tensor Parallel | Hidden dimension ($H$) | Attention, MLP linear layers |
| Sequence Parallel | Sequence dimension ($S$) | Layer Norm, Dropout |

### Workflow

```
Sequence Parallel Region:
  - X split into [X₁, X₂] across GPUs (by sequence dim)
  - Apply Layer Norm independently → [X̃₁, X̃₂]

Transition to Tensor Parallel (ALL-GATHER):
  - Gather full sequence [X̃₁, X̃₂] on all GPUs
  - (Attention needs all tokens)

Tensor Parallel Region:
  - Multi-head attention (heads split across GPUs)
  - Row-linear split on output projection
  - Each GPU has partial result for all tokens

Transition back to Sequence Parallel (REDUCE-SCATTER):
  - Reduce partial outputs (sum across GPUs)
  - Scatter by sequence dimension → each GPU gets different tokens

Sequence Parallel Region:
  - Apply Dropout on local token subset
  - Apply Layer Norm on local token subset

(Repeat for MLP block with same pattern)
```

### Communication Pattern

| Transition | Operation | What Happens |
|------------|-----------|-------------|
| SP → TP | **All-Gather** | Gather all tokens before attention |
| TP → SP | **Reduce-Scatter** | Sum partial hidden dims + scatter by sequence |

**Note**: These are **exposed** (blocking) communications — cannot overlap with computation.

---

## 6. Tensor Parallelism Scaling

### Throughput Impact
- Within a single node (8 GPUs): Fast NVLink interconnect → acceptable overhead
- Across nodes: Slower interconnect → **significant throughput drop**

### Typical Configuration
- TP degree = number of GPUs within one node (usually 8)
- DP across nodes

---

## 7. Context Parallelism (Long Sequences)

### Motivation
Even with tensor + sequence parallelism, **attention** requires all tokens:
- Attention is $O(S^2)$ — memory grows quadratically with sequence length
- For very long sequences, even one GPU can't hold all attention activations

### Idea
Split the **entire input sequence** across GPUs:
- GPU 1: tokens 1–T/N
- GPU 2: tokens T/N+1–2T/N
- etc.

| Operation | Issue with Context Parallelism? |
|-----------|-------------------------------|
| Layer Norm | ✅ No issue (token-independent) |
| MLP | ✅ No issue (token-independent) |
| Dropout | ✅ No issue (token-independent) |
| **Attention** | ❌ Each token needs KV from ALL other tokens |

### Naive Solution
All-gather all tokens before attention → defeats the purpose (blocking, full memory).

### Ring Attention

Each GPU starts by computing attention using its **local KV pairs**, then receives KV blocks from neighboring GPUs in a ring pattern:

```
Round 0: GPU_i computes attention using local K_i, V_i
Round 1: GPU_i receives K_{i+1}, V_{i+1} from neighbor
         → extends attention computation
Round 2: GPU_i receives K_{i+2}, V_{i+2}
         → extends further
...
After N-1 rounds: GPU_i has attended to all tokens
```

### Maintaining Partial Results
Each GPU maintains running sums:
- Numerator: $\sum_j e^{K_j^T Q_i} \cdot V_j$
- Denominator: $\sum_j e^{K_j^T Q_i}$

Final output = numerator / denominator (online softmax).

### Load Balancing Problem with Causal Masking

**Naive allocation** (contiguous chunks):
| GPU | Tokens | Rounds Active |
|-----|--------|---------------|
| GPU 1 | 1–5 | 1 round (idle after) |
| GPU 2 | 6–10 | 2 rounds |
| GPU 3 | 11–15 | 3 rounds |
| GPU 4 | 16–20 | 4 rounds (overloaded) |

Early tokens need fewer keys → GPUs with early tokens finish early and sit idle.

**Zigzag/Striped allocation** (interleaved):
| GPU | Tokens |
|-----|--------|
| GPU 1 | 1, 8, 9, 16 |
| GPU 2 | 2, 7, 10, 15 |
| GPU 3 | 3, 6, 11, 14 |
| GPU 4 | 4, 5, 12, 13 |

Each GPU has a mix of early and late tokens → **all GPUs stay busy** through all communication rounds.

### Computation-Communication Timeline

```
GPU_i:
  [Compute attn Q_i on local K_i,V_i]
  [Fetch K_{i+1},V_{i+1}] ← overlapped with computation
  [Compute attn Q_i on K_{i+1},V_{i+1}]
  [Fetch K_{i+2},V_{i+2}] ← overlapped with computation
  ...
```

Communication (fetching next KV block) overlaps with computation (current attention).

---

## 8. Summary: All Parallelism Strategies

| Strategy | What's Parallelized | Dimension Split | Communication |
|----------|-------------------|-----------------|---------------|
| **Data Parallel** | Micro-batches | Batch dim | All-reduce gradients |
| **ZeRO 1/2/3** | Optimizer/grad/params | Sharded across GPUs | Reduce-scatter + all-gather |
| **Tensor Parallel** | Attention heads + MLP neurons | Hidden dim | All-reduce after each block |
| **Sequence Parallel** | LayerNorm, Dropout | Sequence dim | All-gather / reduce-scatter at transitions |
| **Context Parallel** | Full sequence (inc. attention) | Sequence dim | Ring exchange of KV pairs |

### Activation Tensor: $B \times S \times H$

| Parallelism | Splits Along |
|-------------|-------------|
| Data Parallel | $B$ (batch) |
| Tensor Parallel | $H$ (hidden) |
| Sequence Parallel | $S$ (sequence) — for non-attention ops |
| Context Parallel | $S$ (sequence) — including attention via ring |

---

## 9. Next: Pipeline Parallelism
Splits model **layer-wise** across GPUs (model parallelism in the depth dimension).
