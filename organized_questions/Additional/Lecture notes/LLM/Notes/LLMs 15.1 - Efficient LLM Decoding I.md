# Lecture 15.1 — Efficient LLM Decoding I

## Why Efficient Inference (Not Training)?

| Aspect | Training | Inference |
|--------|----------|-----------|
| Sequence processing | Entire sequence processed **in parallel** (single forward pass) | Tokens generated **sequentially** (one forward pass per token) |
| Forward passes | 1 forward pass for the whole sequence | $N$ forward passes for $N$ tokens |
| Bottleneck | Compute-bound | Memory-bound & latency-bound |

**Key insight:** Generating $N$ tokens requires $N$ forward passes — infeasible at production scale.

---

## 1. KV Caching

### The Problem

At inference, each new token requires re-processing the entire sequence through the attention mechanism. Most of the computation (keys, values, attention weights for prior tokens) is **redundant**.

### The Solution — Cache Keys & Values

Split inference into two phases:

| Phase | Description |
|-------|-------------|
| **Prefill** | Process entire prompt in parallel, compute all K/V vectors, populate KV cache |
| **Decode** | Generate one token per step; only compute Q for the new token, read K/V from cache, append new K/V to cache |

**Process:**
1. Forward pass the full prompt → get probability distribution from last position → pick next token
2. For each subsequent token:
   - Compute $Q_{\text{new}}$ from the new token embedding
   - Read all cached $K, V$ vectors
   - Compute attention: $\text{Attn} = \text{softmax}\!\left(\frac{Q_{\text{new}} \cdot K^T}{\sqrt{d}}\right) \cdot V$
   - Append new $K_{\text{new}}, V_{\text{new}}$ to cache
   - Pass output through classifier → next token
3. Repeat until EOS or max tokens

### KV Cache Memory Usage

$$\text{KV Cache Size} = 2 \times \text{precision} \times n_{\text{layers}} \times d_{\text{hidden}} \times L_{\text{seq}} \times B$$

where:
- $2$ = two matrices (K and V)
- precision = bytes per element (e.g., 2 for FP16)
- $n_{\text{layers}}$ = number of attention layers
- $d_{\text{hidden}}$ = hidden dimension
- $L_{\text{seq}}$ = sequence length
- $B$ = batch size

**Example (OPT-13B on A100 40GB):**

| Component | Value |
|-----------|-------|
| Precision | FP16 (2 bytes) |
| Layers | 40 |
| Hidden dim | 5120 |
| Max seq length | 2048 |
| Model params | 26 GB (~65% of GPU) |
| Available for KV cache | ~12 GB (~30%) |
| Max batch size | **8** (with 2048-length sequences) |

---

## 2. KV Cache Memory Fragmentation

### Three Types of Fragmentation

| Type | Cause |
|------|-------|
| **Internal fragmentation** | Over-allocation based on max tokens (reserved slots never used) |
| **External fragmentation** | Memory allocator leaves unusable gaps between allocations |
| **Reserved slots** | Slots pre-allocated for future tokens not yet generated |

Standard allocation requires **contiguous memory** for tensor operations → static chunk pre-allocation → significant waste.

---

## 3. Paged Attention (vLLM)

### Inspiration: OS Virtual Memory & Paging

| OS Concept | vLLM Analogy |
|------------|--------------|
| Process | Input prompt / request |
| Virtual memory | Logical KV blocks (contiguous per request) |
| Physical memory | Physical KV blocks (non-contiguous on GPU) |
| Page table | Block table (maps logical → physical blocks) |

### How It Works

1. **Physical KV blocks**: GPU memory divided into fixed-size blocks (e.g., block size = 16)
2. **Logical KV blocks**: Each request gets a contiguous logical view
3. **Block table**: Maps logical block indices → physical block indices; tracks fill count per block
4. **On-demand allocation**: New physical blocks allocated only when current block fills up

**Fragmentation reduction:**
- Internal fragmentation bounded by block size (typically 16–32 tokens)
- **No external fragmentation** (blocks allocated independently)
- Reserved slots per sequence ≤ block size
- Average wasted space: **< 4%** of KV cache

### Parallel Sampling with Shared Blocks

For scenarios requiring multiple completions from the same prompt (beam search, rejection sampling, reward model training):

- Shared prompts share physical KV blocks via **reference counting**
- **Copy-on-write**: When a shared block needs modification:
  1. Check reference count
  2. If ref count > 1: copy block to new location, decrement ref count, modify copy
  3. If ref count = 1: modify in place
- Only the **last shared block** needs copying — all preceding blocks remain shared

### vLLM Throughput Improvement

| Comparison | Throughput Gain |
|------------|----------------|
| vs. HuggingFace `model.generate()` | **24×** higher throughput |
| vs. Text Generation Inference (TGI) | **3.5×** higher throughput |
| Batch size improvement | 8 → **40** batches (5× increase) |

### vLLM System Architecture

```
┌─────────┐
│ Frontend │ ← HTTP API
└────┬────┘
     │
┌────▼─────────────────┐
│  Centralized Engine   │ ← Manages block tables
├──────────────────────┤
│     Scheduler         │ ← Schedules requests
├──────────────────────┤
│ Distributed Executor  │ ← Multiple GPU workers
│  (Megatron + Ray)     │
└───────────────────────┘
```

- **Megatron**: Model parallelism across GPUs
- **Ray**: Distributed computing framework

---

## 4. Paged Attention Kernel

### Problem: Non-Contiguous Memory for Attention

KV blocks are scattered across GPU memory → standard matrix multiplication (`Q · K^T`) requires contiguous tensors.

### Solution: Block-wise Softmax Decomposition

Softmax can be decomposed over blocks and then combined:

$$\text{softmax}([A_1, A_2]) = \alpha \cdot \text{softmax}(A_1) + \beta \cdot \text{softmax}(A_2)$$

where $\alpha, \beta$ are normalizing constants derived from partial sums:

$$\alpha = \frac{\sum_{i \in \text{block}_1} e^{q \cdot k_i}}{\sum_{i \in \text{all}} e^{q \cdot k_i}}, \quad \beta = \frac{\sum_{i \in \text{block}_2} e^{q \cdot k_i}}{\sum_{i \in \text{all}} e^{q \cdot k_i}}$$

**Same decomposition trick as Flash Attention** — compute attention per block, store partial numerator and denominator, aggregate.

---

## Summary: Optimization Roadmap

| Optimization | Problem Addressed | Status (after this lecture) |
|-------------|-------------------|----------------------------|
| KV Caching | Redundant computation | ✅ |
| Paged Attention (vLLM) | Inefficient GPU memory management | ✅ |
| Flash Decoding | Speeding up attention computation | Next lecture |
| Speculative Decoding | Sequential token generation | Next lecture |
