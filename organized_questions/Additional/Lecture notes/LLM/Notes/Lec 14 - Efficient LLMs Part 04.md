# Lecture 14 — Efficient LLMs Part 04: Pipeline Parallelism, GPU Kernels & Flash Attention

---

## 1. Pipeline Parallelism

### Core Idea
Split transformer layers across multiple GPUs sequentially. Each GPU holds a subset of layers and processes micro-batches in turn.

### The Bubble Problem
When $p$ GPUs process a batch, idle time (bubble) arises because each GPU must wait for the previous one.

$$\text{Bubble Ratio (naive)} = \frac{p - 1}{p}$$

With $p = 4$ GPUs, **75%** of GPU time is wasted (bubble).

### Micro-batching
Split the batch into $m$ micro-batches to fill pipeline stages more efficiently:

$$\text{Bubble Ratio} = \frac{p - 1}{m + p - 1}$$

As $m \to \infty$, the bubble ratio → 0. Trade-off: larger $m$ requires storing more activations.

### 1F1B (One Forward One Backward) Schedule

| Phase | Description |
|-------|-------------|
| **Warm-up** | Forward passes fill the pipeline |
| **Steady state** | Alternate 1 Forward → 1 Backward per micro-batch |
| **Cool-down** | Drain remaining backward passes |

**Key advantage**: Backward pass frees activation memory early → allows increasing $m$ → reduces bubble further.

### Interleaved Stages
Each GPU holds **non-contiguous** layers (zigzag pattern):

| GPU | Layers (non-interleaved) | Layers (interleaved, $v=2$) |
|-----|--------------------------|------------------------------|
| GPU 0 | 1–4 | 1–2 and 9–10 |
| GPU 1 | 5–8 | 3–4 and 11–12 |
| GPU 2 | 9–12 | 5–6 and 13–14 |
| GPU 3 | 13–16 | 7–8 and 15–16 |

Bubble ratio reduces by factor $v$ (number of stages per GPU):

$$\text{Bubble Ratio (interleaved)} = \frac{p - 1}{v \cdot m + p - 1}$$

**Trade-off**: More inter-GPU communication due to non-contiguous layer placement.

---

## 2. 4D Parallelism (LLaMA 3 Example)

All four parallelism dimensions combined:

| Dimension | What is Split | Scope |
|-----------|--------------|-------|
| **Data Parallelism** | Batch | Across replicas |
| **Context Parallelism** | Sequence (tokens) | Ring attention across GPUs |
| **Tensor Parallelism** | Hidden dimensions / attention heads | Intra-node (fast NVLink) |
| **Pipeline Parallelism** | Layers | Inter-node (slower network) |

**Design rule**: Tensor parallelism within a node (high bandwidth), pipeline parallelism across nodes (tolerates latency).

---

## 3. GPU Kernels

### What is a Kernel?
A **kernel** = smallest unit of code that runs on a GPU core. PyTorch operations (matmul, softmax, layer norm) are each separate kernel calls.

### Kernel Launch Overhead
Each kernel call requires:
1. Load data from **HBM** (main GPU memory) → **SRAM** (on-chip cache)
2. Compute on SRAM
3. Write result back to HBM

Multiple sequential kernel calls = multiple round-trips to HBM.

---

## 4. HBM vs SRAM

| Property | HBM (High Bandwidth Memory) | SRAM (Static RAM) |
|----------|----------------------------|-------------------|
| **Capacity** | 40–80 GB | ~20 MB |
| **Speed** | ~2 TB/s | ~19 TB/s |
| **Role** | Main GPU memory | On-chip cache |

**Bottleneck**: The HBM ↔ SRAM data transfer, not the compute itself. Most operations are **memory-bound**, not compute-bound.

---

## 5. Fused Kernels

### Problem
Sequential operations (e.g., matmul → add bias → GELU → dropout) each require separate HBM round-trips.

### Solution: Kernel Fusion
Combine multiple operations into **one kernel** that keeps intermediate data in SRAM:

```
# Without fusion: 4 HBM round-trips
y = dropout(gelu(x @ W + b))

# With fusion: 1 HBM round-trip (fused into single kernel)
```

### Tools for Fusion

| Tool | Description |
|------|-------------|
| `torch.compile` | Decorator for automatic kernel fusion |
| **Liger Kernels** | Library of fused kernels for transformer ops (RMSNorm, SwiGLU, CrossEntropy, RoPE) |
| **Triton** | Framework for writing custom GPU kernels in Python |

```python
@torch.compile
def fused_op(x, W, b):
    return F.dropout(F.gelu(x @ W + b))
```

---

## 6. Flash Attention

### Naive Attention — Memory Flow

$$O = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Step-by-step with HBM transfers:

| Step | Computation | HBM Access |
|------|------------|------------|
| 1 | $S = QK^T$ | Read Q, K → Write S |
| 2 | $P = \text{softmax}(S)$ | Read S → Write P |
| 3 | $O = PV$ | Read P, V → Write O |

**Problem**: Full $T \times T$ attention matrix $S$ materialized in HBM → $O(T^2)$ memory.

### Flash Attention Algorithm

**Key insight**: Process attention in blocks without materializing the full $S$ matrix.

#### Algorithm (outer loop over KV blocks):

```
For each KV block j:
    Load K_j, V_j into SRAM
    For each query Q_i:
        Compute local scores: s_ij = Q_i · K_j^T
        Compute local exponents: a_ij = exp(s_ij)
        Compute local sum: L_j = sum(a_ij)
        Compute local output: O_j = sum(a_ij * V_j) / L_j
        
        # Online update (combine with running result):
        L_new = L_old + L_j
        O_new = (L_old * O_old + L_j * O_j) / L_new
        L_old = L_new
        O_old = O_new
    Write final O, L back to HBM
```

### Online Softmax Combination Formula

For two blocks with partial outputs $(O_1, L_1)$ and $(O_2, L_2)$:

$$O = \frac{L_1 \cdot O_1 + L_2 \cdot O_2}{L_1 + L_2}$$

where $L_i = \sum_k \exp(Q \cdot K_k^T)$ over keys in block $i$.

### Benefits

| Metric | Naive Attention | Flash Attention |
|--------|----------------|-----------------|
| **Memory** | $O(T^2)$ | $O(T)$ |
| **HBM accesses** | $O(T^2)$ | $O(T^2 / \text{SRAM size})$ |
| **Materializes S?** | Yes (full $T \times T$) | No |
| **Wall-clock time** | Baseline | 2–4× faster |

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| Pipeline Parallelism | Split layers across GPUs; bubble reduced via micro-batching, 1F1B, interleaving |
| 4D Parallelism | Data + Context + Tensor + Pipeline; tensor within node, pipeline across nodes |
| GPU Kernels | Minimize HBM ↔ SRAM transfers via kernel fusion |
| Flash Attention | Block-wise online softmax avoids materializing $T \times T$ matrix |
