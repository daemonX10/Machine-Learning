# Lecture 10.2 — Mixture of Experts (MoE) – Part II

---

## 1. Recap: MoE Fundamentals

### Dense vs. Sparse MoE

| Aspect | Dense (Single Expert) | MoE Layer |
|--------|----------------------|-----------|
| Model size | $M$ | $\approx n \cdot M$ |
| Compute (top-2) | $C$ | $2C + \text{routing}$ |
| Compute (top-1) | $C$ | $C + \text{routing}$ |

- Routing cost is negligible (one linear layer + softmax)
- In Transformers: only the FFN is replicated; attention layers are **shared**
- So with 4 experts replacing FFN: model size ≈ $4M$ (not $4M$ exactly, since attention params unchanged)

### Total Expert & Path Count
- If 4 experts per layer × 32 layers → **128 total experts**
- With top-1 routing: each token has $4^{32}$ possible paths through the network
- Router parameters are **different** in each layer

---

## 2. Switch Transformer (Google Brain)

### Three Goals
1. **Reduce complexity** of MoE
2. **Reduce communication cost** across devices
3. **Address training instability**

### Core Design: Top-1 (Greedy) Routing
- Routes each token to **exactly one** expert (unlike prior top-2 approaches)
- First paper to challenge the belief that ≥ 2 experts needed for learning contrast
- This single choice reduces both complexity and communication cost

### Distributed Architecture — Expert Parallelism

| Component | Strategy |
|-----------|----------|
| Expert FFN layers | **Sharded** — one expert per device |
| Attention layers | **Replicated** across all devices |
| Communication | All-to-all between devices |

- With $e$ experts, use $e$ devices
- Not all layers are MoE: Switch Transformer uses MoE in **alternate layers**

### Results: Scaling with Matched FLOPs

| Configuration | Params | Test Loss |
|---------------|--------|-----------|
| Base T5 (1 expert) | 223M | ~6.0 |
| 256 experts | 14.7B | ~4.8 |

- FLOPs per token are **matched** (each token passes through only 1 expert)
- Loss reduces monotonically with more experts at constant compute

### Sample Efficiency
- Sparse models are more **sample efficient** than dense baselines
- 128 experts outperform 16 experts, but with **diminishing returns**

### Training Time Speedup
- Switch-Base 128 achieves same perplexity as T5-Base in **~7× less wall-clock time**

### Comparison with Dense Model Parallelism
- Fair comparison: T5-Large (770M) split across $e$ devices via tensor parallelism
- Sample efficiency: roughly comparable between MoE and dense
- **Clock time**: MoE still achieves **2.5× speedup** over dense model parallelism

---

## 3. Model Parallelism Techniques

### Pipeline Parallelism
- Shard model **layer-by-layer** across GPUs
- Layer 1 on GPU 1, Layer 2 on GPU 2, etc.

### Tensor Parallelism

#### Column-wise Splitting
- Split weight matrix $B$ by **columns** across GPUs
- Input $A$ is replicated on all devices
- Each GPU computes a subset of columns of $C = A \times B$
- **No reduction needed** — results are disjoint columns

#### Row-wise Splitting
- Split weight matrix $W$ by **rows** across GPUs
- Input must also be split accordingly
- Each GPU computes a partial sum
- **All-reduce (addition)** needed to get final output

$$C_{11} = \underbrace{a_{11}w_{11} + a_{12}w_{21}}_{\text{GPU 0}} + \underbrace{a_{13}w_{31} + a_{14}w_{41}}_{\text{GPU 1}}$$

---

## 4. Training Stability Improvements

### 4.1 Differentiable Load Balancing Loss

**Problem**: Router may collapse — sending all tokens to one expert.

**Notation per batch** ($T$ tokens, $N$ experts):

| Term | Definition |
|------|-----------|
| $f_i$ | Fraction of tokens actually routed to expert $i$ |
| $P_i$ | Mean routing probability for expert $i$ (averaged over batch) |

$$f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbf{1}[\text{argmax}(G(x)) = i]$$

$$P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p(i \mid x)$$

**Auxiliary loss**:

$$\mathcal{L}_{\text{balance}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

**Intuition**: If $f_1$ is high (too many tokens to expert 1), minimizing the loss forces $P_1$ down and shifts probability mass to underused experts. **Equilibrium**: $f_i = \frac{1}{N}$ and $P_i = \frac{1}{N}$ for all $i$ (uniform distribution).

> **Note**: Per-token routing should be **skewed** (confident), but the **aggregate** distribution across the batch should be **uniform**.

### 4.2 Selective Precision

| Component | Precision |
|-----------|-----------|
| Experts | BFloat16 |
| Router | **FP32** |

- Router involves softmax (exponentiation) — sensitive to numerical errors
- Training **diverged** in full BF16; converged with selective precision
- Speed impact negligible (router is tiny)

### 4.3 Reduced Initialization Scale

- Standard init: $\mathcal{N}(0, \sqrt{1/d})$
- Switch Transformer: scale reduced to $\mathcal{N}(0, 0.1 \cdot \sqrt{1/d})$
- Empirically reduces variance across training runs and improves stability

### 4.4 Higher Expert Dropout (for Fine-tuning)

| Layer | Dropout |
|-------|---------|
| All other layers | 0.1 |
| Expert layers | **0.4** |

- MoE increases model capacity → more prone to overfitting during fine-tuning
- Increased expert dropout provides necessary regularization

---

## 5. Capacity Factor & Token Dropping

### The Static Graph Problem (TensorFlow/TPU)
- Computation graph is pre-compiled → batch dimensions must be fixed
- Can't dynamically handle uneven routing

### Expert Capacity

$$\text{Expert Capacity} = \frac{T}{N} \times \text{Capacity Factor}$$

| Capacity Factor | Expert Capacity (6 tokens, 3 experts) | Effect |
|-----------------|---------------------------------------|--------|
| 1.0 | 2 | Exact uniform — tokens may be **dropped** |
| 1.5 | 3 | Buffer added — fewer drops, more **wasted slots** |

### Token Dropping
- If an expert is full, overflow tokens are **dropped** (skipped)
- "No Token Left Behind" (2-stage routing to next-best expert) **did not help empirically**
- **Hypothesis**: Token dropping acts as implicit regularization (like dropout)

### Best Configuration
- Capacity factor **1.25** achieved best perplexity and fastest time-to-quality

---

## 6. Mixtral (8×7B)

### Architecture

| Hyperparameter | Value |
|----------------|-------|
| Transformer layers | 32 |
| Experts per layer | 8 (in **every** layer, not alternating) |
| Routing | **Top-2** (not top-1 like Switch) |
| Total params | **47B** |
| Active params per token | **13B** |
| Base dense model | Mistral 7B |

### Router Design

$$G(x) = \text{Softmax}(\text{TopK}(W \cdot x, k=2))$$

- Simple linear layer + top-K + softmax (no noise)

### SwiGLU Activation (in Expert FFN)

$$\text{SwiGLU}(x) = (\sigma(\beta x) \odot x) \cdot W + b$$

- **Swish**: $\sigma(\beta x) \cdot x$ — gating mechanism decides how much of $x$ to pass
- **GLU**: Gated Linear Unit — input itself gates the transformation
- If sigmoid → 1: pass input unchanged; if → 0: apply full transformation
- Empirically outperforms standard FFN

### Analysis: FFN Stores Knowledge, Attention Stores Reasoning

| Task Type | Gap (Dense → Sparse) |
|-----------|---------------------|
| **Knowledge-intensive** (e.g., trivia, facts) | **Large gap** — MoE much better |
| **Reasoning/comprehension** | Small gap |

- Supports hypothesis that FFN layers = domain knowledge; attention = reasoning

### Routing Interpretability
- **Layer 0**: Routing near-uniform across experts (no specialization)
- **Middle layers**: Clear specialization — e.g., math tokens cluster to Expert 3
- **Consecutive tokens**: Strong tendency to route to the **same expert** (contiguity)
- Digits consistently routed to the same expert across examples

### Expert Importance (Ablation)
- Dropping Expert 3 → accuracy drops from 65% to **0%** on MMLU
- Other experts contribute much less individually
- (Viral meme: "7 experts idle, Expert 3 does all the work")

---

## 7. Key Takeaways

1. **Switch Transformer**: Top-1 routing reduces complexity, communication, and enables massive scaling with matched FLOPs
2. **Load balancing loss** prevents router collapse by encouraging uniform token distribution across experts
3. **Selective precision** (FP32 router), **reduced init scale**, and **expert dropout** all improve training stability
4. **Capacity factor** controls the trade-off between token drops and wasted computation; 1.25 works best
5. **Mixtral 8×7B**: 47B params, 13B active; top-2 routing; commercially friendly; outperforms GPT-3.5 Turbo
6. Expert specialization emerges naturally — knowledge-intensive tasks benefit most from MoE
