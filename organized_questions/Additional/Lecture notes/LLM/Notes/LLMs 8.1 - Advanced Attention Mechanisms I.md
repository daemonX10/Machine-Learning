# LLMs 8.1 — Advanced Attention Mechanisms I

## 1. Self-Attention Recap

### Full Self-Attention
For each token, compute:
1. **Query** ($Q$), **Key** ($K$), **Value** ($V$) projections
2. Similarity: $S = QK^T$
3. Percentage/weight: $A = \text{softmax}(S)$
4. Contextual embedding: $Z = AV$

**Key insight:** Keys are "representatives" — a token doesn't need to be nearby in embedding space to influence another; its key just needs high similarity with the query.

### Causal Attention (Decoder / GPT-style)
- Mask future tokens: set upcoming positions to $-\infty$ before softmax → become 0
- Each token only attends to itself and previous tokens

---

## 2. Complexity Analysis of Full Self-Attention

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| $QK^T$ (dot product) | $O(n^2 d)$ | — |
| Softmax | $O(n^2)$ | — |
| Score × $V$ | $O(n^2 d)$ | — |
| **Total time** | $O(n^2 d)$ | — |
| Store $Q, K, V$ | — | $3 \times n \times d$ |
| Store score matrix | — | $n^2$ |
| Store output | — | $n \times d$ |
| **Total space** | — | $O(n^2 + nd)$ |

**Exact byte size:** $(3nd + n^2) \times \text{sizeof(float)}$

> All complexities are **per head, per layer**. Multiply by $H$ (heads) and $L$ (layers) for full model.

---

## 3. KV Caching

### Problem
In causal attention, when computing attention for token $t$:
- Row $t$ of the score matrix recomputes all previous rows' dot products
- Only the **last row** is actually needed (we only predict the next token)
- Previous rows have already been computed in earlier steps

### Solution: Cache Keys and Values

| Aspect | Without KV Cache | With KV Cache |
|--------|-----------------|---------------|
| Query | Stack all queries ($n \times d$) | Only current query ($1 \times d$) |
| Keys | Recompute all | Fetch previous from cache, append new |
| Score matrix | Full $n \times n$ | Only last row ($1 \times n$) |
| Values | Recompute all | Fetch previous from cache |
| Output | Full $n \times d$ | Only current token's embedding |

### Complexity with KV Cache

| Metric | Full Attention | With KV Cache |
|--------|---------------|---------------|
| **Time** | $O(n^2 d)$ | $O(nd)$ |
| **Space (bytes)** | $(3nd + n^2) \times F$ | $2nd \times F$ |

- $n^2$ term eliminated (no full score matrix)
- Factor of 3 → 2 (queries not piled up)

---

## 4. Sliding Window Attention (Longformer, 2020)

**Paper:** Allen AI, 2020

### Hypothesis
For many NLP tasks, you don't need attention over the **entire** sequence — a fixed **local context window** ($W$) suffices.

### Mechanism
- Instead of attending to all $n$ tokens, each token attends to only $W$ neighbors
- For causal LMs: attend to $W$ **previous** tokens only (no future)
- Non-causal: attend to $W/2$ on each side

### Complexity Reduction

| Metric | Full Attention | Sliding Window |
|--------|---------------|----------------|
| **Time** | $O(n^2 d)$ | $O(nWd)$ |
| **KV Cache size** | $2nd \times F$ | $2Wd \times F$ |
| **Space** | $O(n^2 + nd)$ | $O(nW + nd)$ |

### With KV Caching
- Cache stores only the last $W$ keys/values (not all $n$)
- Older entries evicted as new tokens arrive

### Limitations
- **Fails for tasks requiring long-range context:**
  - Multi-turn dialog (history from earlier turns)
  - Long document understanding
  - Mathematical/physics reasoning requiring extensive context
  - Descriptive explanations spanning many tokens

---

## 5. Multi-Head Attention (MHA) — Recap

### Why Multiple Heads?
One attention weight (one percentage) per token pair captures only **one type of association**. Multiple heads capture different linguistic relationships:
- Lexical co-occurrence
- Syntactic roles (noun-verb, modifier-head)
- Semantic similarity
- Coreference
- etc.

### Cost
Each head computes $QK^T$ and $AV$ independently → multiply complexity by $H$ (number of heads).

---

## 6. Multi-Query Attention (MQA)

**Paper:** Shazeer, Google, 2019

### Key Idea
- **Queries** differ across heads (each head views tokens from a different perspective)
- **Keys and Values are shared** across all heads (the "influencer's personality" doesn't change)

### Analogy
The entities being influenced (queries) change their perspective in each head, but the influencers (K, V) remain constant.

### KV Cache Reduction
$$\text{KV cache reduction factor} = H$$

Instead of $H$ copies of K and V matrices, store only **one** shared copy.

### Problem
- **Assumption violation**: The influencer's representation *should* change across contexts
- Results in **quality degradation** and **training instability**

---

## 7. Up-training (Converting MHA → MQA)

**Paper:** Georgia Tech + Google Research, 2022 (ICLR 2023)

### Problem with Training MQA from Scratch
Random initialization of shared K, V → training instability, poor performance.

### Solution: Up-training

**Step 1:** Take a pre-trained MHA model → **mean-pool** all head-specific K matrices into one K, and all V matrices into one V.

**Step 2:** Continue pre-training (further pre-training) from that checkpoint.

### Process

```
MHA Training: 0 ────────── K' (stop) ──── K (finish)
                                ↓
                         Mean-pool K, V
                                ↓
                         Resume training K' → K
```

### Results
- Proportion $\alpha$ = 0% (no step 2): poor performance
- $\alpha$ = 10%+: **huge jump** in performance and training stability
- But **never reaches MHA-level performance** — the shared K/V assumption is still wrong

---

## 8. Grouped Query Attention (GQA)

**Paper:** Google Research, EMNLP 2023

### Key Idea
- Group queries into $G$ groups
- Each group shares one K and one V
- Trade-off between MHA ($G = H$, no sharing) and MQA ($G = 1$, full sharing)

### Spectrum

```
MHA  ←──────────────────────→  MQA
G = H                          G = 1
(separate K,V per head)        (one shared K,V)
         └──── GQA (1 < G < H) ────┘
```

### Results

| Metric | MHA | GQA | MQA |
|--------|-----|-----|-----|
| **Quality** | Best | **Near-best** (sometimes better than MHA) | Degraded |
| **Inference time** | Slowest | Comparable to MQA | Fastest |
| **KV cache** | $H \times$ | $G \times$ | $1 \times$ |

- GQA achieved **comparable or better** quality than MHA on summarization (CNN, ArXiv, MultiNews), translation (WMT), QA (TriviaQA)
- Significant inference speedup

---

## 9. Summary: What's Optimized Where

| Method | Optimizes | Level |
|--------|-----------|-------|
| **KV Caching** | Avoids recomputation of previous K, V | Single head |
| **Sliding Window** | Reduces context window ($n → W$) | Single head |
| **MQA** | Shares K, V across all heads | Multi-head |
| **GQA** | Shares K, V within groups | Multi-head |

### What's Still Missing
- MQA/GQA reduce **across-head** overhead but do **NOT** optimize at the single-head level
- The dot product + softmax computation within each head remains $O(n^2)$ or $O(nW)$
- **Flash Attention** (next lecture) addresses this → achieves **O(n)** time with **no approximation**
