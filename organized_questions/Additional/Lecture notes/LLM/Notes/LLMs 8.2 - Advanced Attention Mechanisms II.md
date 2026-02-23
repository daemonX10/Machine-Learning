# LLMs 8.2 — Advanced Attention Mechanisms II: Flash Attention

## 1. Recap of Previous Optimizations

| Method | What It Does | Limitation |
|--------|-------------|------------|
| KV Caching | Avoids recomputing previous keys/values | Doesn't reduce single-head computation |
| Sliding Window | Limits context to window $W$ | Quality loss for long-range tasks |
| MQA | Shares K, V across heads | Quality degradation + training instability |
| GQA | Groups queries, shares K, V within groups | Still no optimization at single-head level |

**Remaining problem:** At the single-head level, dot product + softmax + value weighting is still $O(n^2)$.

---

## 2. GPU Memory Hierarchy

### Two Key Memory Types

| Memory | Name | Speed | Size |
|--------|------|-------|------|
| **HBM** | High Bandwidth Memory (VRAM) | ~1.5 TB/s | Large (40–80 GB) |
| **SRAM** | Static RAM (on-chip cache) | **Much faster** than HBM | Very small |

- GPUs advertise HBM capacity (e.g., "A100 80GB")
- SRAM is rarely discussed but is **critical** for Flash Attention
- All traditional attention computation happens in HBM

### The I/O Bottleneck
- Computation in SRAM is **super fast**
- Reading/writing to HBM is **slow**
- The bottleneck is **not** the matrix math — it's the **data transfer** between HBM and SRAM

---

## 3. Standard Attention: I/O Analysis

### Step-by-step HBM ↔ SRAM Transfers

| Step | Operation | HBM Read | SRAM Compute | HBM Write |
|------|-----------|----------|--------------|-----------|
| 1 | Similarity: $S = QK^T$ | Read $Q, K$ | Compute dot product | Write $S$ |
| 2 | Masking (causal) | Read $S$ | Apply mask → $S'$ | Write $S'$ |
| 3 | Softmax | Read $S'$ | Compute softmax → $A$ | Write $A$ |
| 4 | Output: $Z = AV$ | Read $A, V$ | Compute weighted sum | Write $Z$ |

**Total:** Multiple unnecessary round-trips between HBM and SRAM.

**Key insight:** The actual matrix multiplication is fast; the **I/O (read/write to HBM)** is the bottleneck.

---

## 4. Flash Attention — Core Idea

**Paper:** Dao et al., Stanford, NeurIPS 2022

### Philosophy
> Use SRAM **more**, HBM **less**. Implement the same algorithm but eliminate unnecessary intermediate HBM read/writes.

### Fused Kernel

Instead of separate read-compute-write for each step:

```
Standard:  Read Q,K → Compute S → Write S → Read S → Mask → Write S' → Read S' → Softmax → Write A → Read A,V → Output → Write Z

Flash:     Read Q,K → Compute S → Mask → Softmax → Read V → Output → Write Z
```

**All intermediate steps happen in SRAM** without writing back to HBM.

| Aspect | Standard | Flash Attention |
|--------|----------|----------------|
| HBM reads | Many (per step) | **2** (initial Q,K and V) |
| HBM writes | Many (per step) | **1** (final output) |
| Intermediate storage | In HBM | **In SRAM only** |

This fusion of operations is called a **fused kernel** (GPU kernel = GPU operation).

---

## 5. Tiling — Processing One K,V at a Time

### Problem with Softmax
The softmax denominator requires the **sum over all** dot products:

$$\text{softmax}(s_i) = \frac{e^{s_i}}{\sum_{j=1}^{n} e^{s_j}}$$

When processing one key at a time, you don't yet have all $s_j$ values → denominator is incomplete.

### Tiling Approach
- **Outer loop** over keys (and corresponding values)
- For each key $k_j$: compare with **all queries** simultaneously
- Accumulate results incrementally

### Example (1D vectors)

Given: $q = 1$, keys $= [1, 2, 3]$, values $= [2, 4, 8]$

| Full computation | Tiled (one key at a time) |
|-----------------|--------------------------|
| $QK^T = [1, 2, 3]$ | Process $k_1=1$, then $k_2=2$, then $k_3=3$ |
| Denominator: $e^1 + e^2 + e^3$ | After step 1: denom = $e^1$ (incomplete!) |

---

## 6. Summary Statistics — Fixing the Denominator

### The Denominator Problem
At step $t$, you only have partial sums:
- Step 1: denominator = $e^{s_1}$ (missing $e^{s_2}, e^{s_3}$)
- Step 2: denominator = $e^{s_1} + e^{s_2}$ (missing $e^{s_3}$)

### Solution: Running Correction

At each step, maintain:
- $D$ = running denominator
- $N$ = running numerator (contextual embedding × denominator to recover numerator)

**Update rules at each new key $k_b$:**

$$D \leftarrow D + D_b$$

$$N \leftarrow N_{\text{prev}} + N_b$$

Where recovering $N_{\text{prev}}$ from previous quotient:

$$N_{\text{prev}} = Z_{\text{prev}} \times D_{\text{prev}}$$

### Step-by-step Example

**Step 1** (process $k_1$):
$$D_1 = e^1, \quad Z_1 = \frac{e^1 \cdot 2}{e^1} = 2$$

**Step 2** (process $k_2$):
$$D_2 = e^1 + e^2$$
$$N = Z_1 \cdot D_1 + e^2 \cdot 4 = e^1 \cdot 2 + e^2 \cdot 4$$
$$Z_2 = \frac{N}{D_2}$$

**Step 3** (process $k_3$):
$$D_3 = e^1 + e^2 + e^3$$
$$N = Z_2 \cdot D_2 + e^3 \cdot 8 = e^1 \cdot 2 + e^2 \cdot 4 + e^3 \cdot 8$$
$$Z_3 = \frac{N}{D_3}$$

→ **Final result matches** standard full-attention computation exactly.

### Pseudocode

```
D = 0        # Running denominator
Z = 0        # Running output (contextual embedding)

for each (k_b, v_b):
    s_b = q · k_b                    # Similarity
    D_b = exp(s_b)                   # Current denominator contribution
    N_prev = Z * D                   # Recover previous numerator
    N_b = D_b * v_b                  # Current numerator contribution
    D = D + D_b                      # Update denominator
    Z = (N_prev + N_b) / D           # Update output
```

---

## 7. Complexity of Flash Attention

| Metric | Standard Attention | Flash Attention |
|--------|--------------------|-----------------|
| **Time** | $O(n^2 d)$ | **$O(nd)$** — linear! |
| **Algorithm** | Exact same computation | **No approximation** |
| **Accuracy** | Baseline | **Identical** to standard |

### Performance Results (GPT-2)
- Dramatically reduced inference time
- The **bottleneck was I/O, not computation**: masking, softmax, dropout were slow because of HBM read/writes
- Matrix multiplication itself was already fast
- Outperformed **NVIDIA MLPerf** best-known implementations

---

## 8. Requirements and Limitations

### Hardware Requirements

| Requirement | Detail |
|-------------|--------|
| **CUDA** | NVIDIA GPUs only (kernel written in CUDA) |
| **SRAM** | GPU must have fast on-chip static RAM |
| **Tensor Cores** | Dedicated matrix multiplication cores (Ampere, Volta, Turing+) |

### Portability Issues

| If your GPU is... | You need to... |
|-------------------|----------------|
| AMD | Rewrite kernel in **ROCm** |
| Intel | Rewrite kernel in **SYCL** |
| Older NVIDIA (no SRAM/Tensor Cores) | Cannot use Flash Attention |

### Custom Attention
If you design a **new attention mechanism**, you must **rewrite the entire kernel in CUDA** — no plug-and-play.

---

## 9. Flash Attention v2
- Released after v1 with further implementation improvements
- Better kernel fusion and parallelism
- Even faster inference and training

---

## 10. Key Takeaways

| Principle | Description |
|-----------|-------------|
| **Minimize HBM I/O** | Reduce unnecessary reads/writes to GPU VRAM |
| **Maximize SRAM usage** | Use fast on-chip memory for intermediate computations |
| **Fused Kernel** | Combine multiple operations into a single GPU kernel |
| **Tiling** | Process one Key-Value pair at a time |
| **Summary Statistics** | Incrementally correct softmax denominator across tiles |
| **No approximation** | Exact same result as standard attention |
| **Linear time** | $O(nd)$ instead of $O(n^2d)$ |
