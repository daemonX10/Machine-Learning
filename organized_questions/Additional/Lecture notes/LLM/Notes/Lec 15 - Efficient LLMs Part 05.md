# Lecture 15 — Efficient LLMs Part 05: LLM Inference, KV Caching & Paged Attention

---

## 1. Training vs Inference

### Training (Forward Pass)

- Input: full sequence $X = (x_1, x_2, \ldots, x_T)$
- Output: logits of shape $T \times V$ (tokens × vocabulary)
- All token positions computed **in parallel** via self-attention
- **Teacher forcing**: ground truth tokens used at every position
- Loss: cross-entropy at each position, sum over all positions

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

### Inference (Autoregressive Generation)

- Generate **one token per forward pass**
- Each new token appended to input → another forward pass
- Only the **last position's logit** matters (next token prediction)
- Multiple forward passes required for full generation

### Decoding Strategies

| Strategy | Description |
|----------|-------------|
| **Greedy (argmax)** | Pick highest probability token |
| **Sampling** | Sample from the distribution |
| **Nucleus (top-p)** | Sample from smallest set whose cumulative probability ≥ $p$ |
| **Top-k** | Sample from top $k$ tokens |

---

## 2. KV Caching

### Problem
Naive inference recomputes K, V for **all previous tokens** at every generation step → redundant computation.

### Solution
Cache previously computed K, V vectors. At each new step:

1. Compute Q, K, V only for the **new token**
2. Append new K, V to the cache
3. Attend over **all cached K, V** using the new Q

### Two Phases of Inference

| Phase | Description | Parallelism |
|-------|-------------|-------------|
| **Prefill** | Process entire input prompt; populate KV cache | All tokens in parallel |
| **Decode** | Generate one token at a time; append to cache | Sequential (1 token/step) |

### KV Cache Memory Formula

$$\text{KV Cache Size} = 2 \times \text{precision} \times n_{\text{layers}} \times d_{\text{hidden}} \times T \times B$$

where:
- Factor 2: one for K, one for V
- precision: bytes per parameter (2 for FP16, 4 for FP32)
- $n_{\text{layers}}$: number of transformer layers
- $d_{\text{hidden}}$: hidden dimension (= $n_{\text{heads}} \times d_{\text{head}}$)
- $T$: sequence length
- $B$: batch size

**Example**: 13B parameter model, FP16, 40 layers, $d=5120$, 10 sequences, 2048 tokens:

$$2 \times 2 \times 40 \times 5120 \times 2048 \times 10 \approx 17 \text{ GB}$$

---

## 3. Memory Fragmentation Problem

### Types of Fragmentation

| Type | Description |
|------|-------------|
| **Internal** | Pre-allocated block larger than needed; unused space wasted |
| **External** | Free memory exists but in non-contiguous chunks; can't allocate large block |

Naive KV cache management pre-allocates for max sequence length → **20–30% memory waste**.

---

## 4. Paged Attention (vLLM)

### Inspiration
OS virtual memory paging: logical pages → physical frames via page table.

### Key Idea
Divide KV cache into **fixed-size blocks** (pages). Map logical KV positions to physical GPU memory blocks on demand.

### Block Table

Each sequence maintains a **block table** mapping logical block index → physical block ID:

| Logical Block | Physical Block |
|---------------|----------------|
| 0 | 7 |
| 1 | 1 |
| 2 | 3 |

### Benefits

| Metric | Naive Allocation | Paged Attention |
|--------|-----------------|-----------------|
| Internal fragmentation | High (pre-allocate max length) | ≤ 1 block per sequence |
| External fragmentation | Possible | None (non-contiguous OK) |
| Memory waste | 20–30% | < 4% |
| Allocation | Contiguous required | On-demand, any free block |

### Paged Attention Computation

KV cache is fragmented across non-contiguous blocks. Compute attention per block and combine:

For query $Q$ and KV blocks $B_1, B_2, \ldots$:

1. For each block $B_j$: compute partial output $O_j$ and normalization sum $L_j$
2. Combine using **online softmax aggregation** (same as flash attention):

$$O = \frac{\sum_j L_j \cdot O_j}{\sum_j L_j}$$

---

## 5. Copy-on-Write for Shared Prefixes

### Use Case
Multiple output sequences generated from the **same prompt** (e.g., beam search, parallel sampling).

### Mechanism

1. All sequences initially **share** the same physical KV blocks (reference counted)
2. When a sequence needs to write a new KV entry to a shared block:
   - **Copy** the block to a new physical location
   - Decrement reference count on original
   - Write new entry to the copy
3. If ref count = 1, write in-place (no copy needed)

### Benefit
Avoids duplicating prompt KV cache across parallel generations.

---

## 6. vLLM Architecture

```
┌─────────────┐
│   Frontend   │  ← CLI / API
└──────┬──────┘
       │
┌──────▼──────┐
│  Scheduler   │  ← Manages request queue, preemption
└──────┬──────┘
       │
┌──────▼──────────────┐
│  Block Manager       │  ← Logical ↔ physical block mapping
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Distributed Model   │  ← Tensor parallelism (intra-node)
│  Executor            │     Pipeline parallelism (inter-node)
└─────────────────────┘
```

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| Training vs Inference | Training: all positions in parallel; Inference: one token per step |
| KV Caching | Cache K,V to avoid redundant recomputation; two phases (prefill + decode) |
| KV Cache Memory | $2 \times \text{prec} \times L \times d \times T \times B$; can be very large |
| Paged Attention | OS-inspired block-level KV management; < 4% waste |
| Copy-on-Write | Share KV blocks across sequences from same prompt |
| Paged Attention Computation | Block-wise attention + online softmax aggregation |
