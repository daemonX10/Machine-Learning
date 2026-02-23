# Lecture 15.2 — Efficient LLM Decoding II

## Recap: What We've Covered

| Optimization | What it does |
|-------------|--------------|
| KV Caching | Avoids redundant computation across decoding steps |
| Paged Attention (vLLM) | Efficient GPU memory management for KV cache via paging |

**Remaining goals:** Speed up attention computation → Flash Decoding; Break sequential generation → Speculative Decoding.

---

## 1. Flash Attention Recap

### GPU Memory Hierarchy

```
CPU/DRAM ←→ HBM (e.g., 80 GB) ←→ SRAM (fast, small)
               GPU main memory        On-chip cache
```

### The Bottleneck

Naive attention requires multiple HBM ↔ SRAM transfers:

1. Load Q, K → SRAM → compute $S = QK^T$ → write $S$ back to HBM
2. Load $S$ → SRAM → apply mask → write back
3. Load masked $S$ → SRAM → compute softmax → write back
4. Load softmax output + V → compute $O = \text{softmax}(S) \cdot V$ → write back

Each operation = separate **kernel** (matrix multiply, mask, softmax).

### Flash Attention Solution

**Fused kernel**: Load data to SRAM once → perform all operations → write result back once.

**Key technique — Tiling**: Compute softmax over blocks and combine:

$$\text{softmax}([A_1, A_2]) = \alpha \cdot \text{softmax}(A_1) + \beta \cdot \text{softmax}(A_2)$$

**Parallelization in Flash Attention (training):** across queries and batch size.

---

## 2. Flash Decoding

### Problem

During decoding: **single query**, but potentially thousands of key-value pairs (long context). Flash Attention parallelizes across queries — useless when there's only one query per step.

### Solution

Split the KV sequence into $S$ splits → apply Flash Attention to each split in parallel → combine results.

```
     Q (single query)
     │
     ├──→ Split 1 (K₁,V₁) → Flash Attn → partial result₁
     ├──→ Split 2 (K₂,V₂) → Flash Attn → partial result₂
     ├──→ Split 3 (K₃,V₃) → Flash Attn → partial result₃
     ├──→ Split 4 (K₄,V₄) → Flash Attn → partial result₄
     └──→ Split 5 (K₅,V₅) → Flash Attn → partial result₅
                                    │
                                    ▼
                              Combine (tiling)
                                    │
                                    ▼
                              Final output
```

**Key difference from Flash Attention:**
- Flash Attention: parallelizes across **queries** and batch
- Flash Decoding: parallelizes across **KV splits** (sequence dimension)

### Performance

| Method | Speedup |
|--------|---------|
| Flash Decoding vs. standard PyTorch | Up to **8×** |
| Flash Decoding vs. Flash Attention V2 | Significant improvement for long sequences |

---

## 3. Speculative Decoding

### Core Idea: Guess and Verify

Instead of generating one token per forward pass, **guess multiple tokens** and **verify them in a single forward pass**.

**Lower bound guarantee:** At least 1 token per iteration (same as standard decoding) — never worse.

### Algorithm

1. **Guess**: Generate a draft completion of $K$ tokens
2. **Verify**: Forward pass the entire guessed sequence through the target model (parallel, like training)
3. **Accept/Reject**: Compare each guessed token against the target model's distribution
   - If target agrees → accept
   - If target disagrees → reject, but **replace with model's choice** → still get one new token
4. **Repeat** with updated prefix

### Example

```
Prompt: "The cat sat"
Guess:  "The cat sat on the chair"

Verify with target model:
  Position 4: Target says "on" ✓ (matches guess) → ACCEPT
  Position 5: Target says "a"  ✗ (guess was "the") → REJECT, use "a"

Result: Generated 2 tokens ("on", "a") in one forward pass
New prompt: "The cat sat on a" → make new guess → repeat
```

### Method A: Draft Model (Original Speculative Decoding)

Use a **smaller model** from the same family as the draft model:
- e.g., 1B model drafts for a 70B target model
- Same tokenizer required
- Small model is fast (memory-bound, fewer parameters)

**Rejection Sampling** (for sampling-based decoding):

Given draft distribution $p(x)$ and target distribution $q(x)$:
- Accept token $x$ with probability $\min\!\left(1, \frac{q(x)}{p(x)}\right)$
- If rejected, sample from adjusted distribution: $\frac{(q(x) - p(x))^+}{\sum_{x'} (q(x') - p(x'))^+}$

**Speedup:** Up to **2–3×** depending on draft model quality and acceptance rate.

### Method B: Medusa (Multiple LM Heads)

**Idea:** Instead of a separate draft model, train **additional language model heads** on the frozen target model backbone.

| Head | Predicts |
|------|----------|
| Head 0 (original) | Next token ($t+1$) |
| Head 1 | Token $t+2$ |
| Head 2 | Token $t+3$ |
| Head $k$ | Token $t+k+1$ |

**Training:**
- Freeze backbone, train heads via PEFT
- Training time: ~5–6 hours depending on model size
- Top-1 accuracy per head: ~60%
- Top-5 accuracy per head: ~80%

**Candidate Generation:**
1. Take top-$k$ predictions from each head
2. Form **Cartesian product** → multiple candidate sequences
3. Organize as a **tree** of candidates

**Tree Attention:**
- Process all candidate sequences in **one forward pass**
- Use a custom **attention mask** ensuring each path in the tree only attends to its own ancestors
- Each node in the tree represents a potential partial completion

**Tree Pruning:**
- Total nodes = $O(k^{\text{depth}})$ — exponential
- Use per-head accuracy statistics to prune unlikely branches
- Pruned tree >> random dense subtree in acceptance rate

**Acceptance Criteria:**
- Dynamic threshold based on:
  1. Model probability of the token
  2. **Entropy** at that position (low entropy → higher bar; high entropy → lower bar)
- Select the **longest subsequence** where all tokens satisfy the threshold

**Speedup:** ~**2×** in tokens/second.

### Method C: Prompt Lookup Decoding

For **content-grounded tasks** (summarization, RAG, etc.):

1. Take a prefix from the current generation
2. Search within the **input prompt** for matching n-grams
3. Use the continuation as the draft
4. Verify with the target model

**Speedup:** ~**2.4×** — especially effective when output heavily overlaps with input.

### Method D: Lookahead Decoding (Brief)

Generate candidate n-grams **without** a draft model, extra heads, or prompt search:

1. Start with random tokens appended to the current sequence
2. Forward pass produces distributions → bigrams from (random token, model's predicted next token)
3. Cache these candidate n-grams
4. Verify using tree attention in subsequent iterations
5. Verification step also produces new candidate n-grams → simultaneous generation and verification

---

## 4. Continuous Batching

### Problem

In a server setting, multiple requests arrive with different lengths. If batched together, shorter sequences finish early → GPU resources wasted.

### Solution

**Continuous batching**: As soon as a sequence finishes, immediately insert a new request into the batch.

```
Time →
S1: ████████████████████████████
S2: ██████████████████████
S3: ████████████ → S5: ██████████████████
S4: ████████████████████████████████ 
                   S3 ends, S5 starts (after prefill pause)
```

- Requires a brief pause for the **prefill step** of the new sequence
- Implemented in frameworks like **Orca**

---

## Summary: Complete Optimization Landscape

| Technique | Problem Solved | Approach |
|-----------|---------------|----------|
| KV Caching | Redundant computation | Cache K, V from previous steps |
| Paged Attention (vLLM) | Memory fragmentation | OS-inspired paging for KV cache |
| Flash Decoding | Slow attention with single query | Parallelize across KV splits |
| Speculative Decoding (Draft Model) | Sequential generation | Small model guesses, large model verifies |
| Medusa | Sequential generation (no draft model) | Multiple LM heads + tree attention |
| Prompt Lookup Decoding | Sequential generation (grounded tasks) | Search prompt for draft tokens |
| Lookahead Decoding | Sequential generation (no extra model) | Generate n-gram candidates from random init |
| Continuous Batching | Wasted GPU on finished sequences | Dynamically replace completed sequences |
