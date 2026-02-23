# Lecture 20: Long Context LLMs — Challenges & Solutions

---

## 1. Why Long Context is Desirable

| Use Case | Problem |
|---|---|
| Book summarization | Entire book exceeds typical 2K–4K context window |
| Video inference | 10K frames × 100 tokens/frame ≈ 1M tokens |
| Large codebases | Company codebases have millions of tokens; finding bugs requires full context |
| RAG limitations | Retriever may miss relevant documents; large context eliminates retrieval dependency |
| In-context learning | Feed all task-specific input-output pairs directly → no fine-tuning needed |

**Key insight**: With a sufficiently large context window, RAG's retriever accuracy becomes irrelevant — feed all documents directly.

---

## 2. Core Challenge: Quadratic Self-Attention

For a sequence of $n$ tokens:

$$\text{Attention cost} = O(n^2)$$

- **Memory**: $O(n^2)$ for the full attention matrix
- **Compute**: $O(n^2)$ dot products between $n$ query and $n$ key vectors
- For $n = 10^6$ tokens → $\approx 10^{12}$ operations

**Flash Attention**: Reduces **memory** to $O(n)$ by block-wise computation, but **compute remains** $O(n^2)$.

---

## 3. Approach 1: LongNet — Dilated Attention

### 3.1 Segment-Level Attention

Divide $n$ tokens into $\frac{n}{w}$ segments of length $w$. Each segment computes attention independently:

$$\text{Cost} = \frac{n}{w} \times w^2 = O(nw)$$

If $w$ is constant → **linear** in $n$.

### 3.2 Dilated (Sparsified) Attention

Within each segment, attend to every $r$-th token (sparsification factor $r$):

$$\text{Cost} = \frac{n}{w} \times \left(\frac{w}{r}\right)^2 = O\!\left(\frac{nw}{r^2}\right)$$

### 3.3 Multi-Scale Mixing

**Problem**: Isolated segments cannot exchange information.

**Solution**: Use multiple attention matrices with different $(w, r)$ combinations:

| Attention Matrix | Segment Length $w$ | Dilation Rate $r$ | Focus |
|---|---|---|---|
| 1st | 4 | 1 (none) | **Local** context |
| 2nd | 8 | 2 | Medium range |
| 3rd | 16 | 4 | **Global** context |

Compute output $O_i$ from each matrix, combine using softmax denominators as weights:

$$O = \frac{\sum_i S_i \cdot O_i}{\sum_i S_i}$$

where $S_i$ = denominator of softmax from the $i$-th attention computation.

### 3.4 Results

| Context Length | Vanilla Attention Time | LongNet (Dilated) Time |
|---|---|---|
| 8K | ~300 | ~100 |
| 64K | ~6000 (20× increase) | ~100 (≈ constant) |

---

## 4. Approach 2: ALiBi — Attention with Linear Biases

### 4.1 The Position Embedding Problem

Models trained on context length $L$ fail when given $L+1$ tokens at inference:

| Position Encoding | Behavior Beyond Training Length |
|---|---|
| Sinusoidal | Perplexity **explodes** immediately at $L+1$ |
| Rotary (RoPE) | Perplexity increases immediately |
| T5 Bias | Slight generalization, then increases |

### 4.2 Core Idea: Remove Position Embeddings Entirely

Instead of adding position embeddings to token embeddings, inject position information **into the attention matrix**:

$$\text{Attention}(q_i, k_j) = q_i^\top k_j + m \cdot (i - j)$$

where:
- $(i - j)$ = distance between tokens (non-positive for causal attention)
- $m$ = scalar slope controlling attention decay rate

### 4.3 Multi-Head Slopes

Different heads get different slopes for local vs. global attention:

$$m_h = \frac{1}{2^{8h/n}}$$

where $n$ = number of attention heads, $h$ = head index (0-indexed).

| Head | Slope $m$ | Effective Attention Span |
|---|---|---|
| Head 0 (large $m$) | $\frac{1}{2}$ | ~2 tokens (**local**) |
| Head $n$ (small $m$) | $\frac{1}{2^8}$ | ~256 tokens (**global**) |

### 4.4 Properties

- **No new position embeddings** → no unseen signals at inference → **perplexity never increases**
- Tokens far apart may get zero attention (limitation)
- Perplexity plateaus but doesn't blow up beyond training length

---

## 5. Approach 3: Position Interpolation (PI) for RoPE

### 5.1 Rotary Position Embeddings (RoPE) Recap

Given a $d$-dimensional vector $\mathbf{x}$, treat as $d/2$ two-dimensional sub-vectors. Rotate each pair by angle dependent on position $m$:

$$f(\mathbf{x}, m) = \begin{pmatrix} x_0 + ix_1 \\ x_2 + ix_3 \\ \vdots \end{pmatrix} \odot \begin{pmatrix} e^{im\theta_0} \\ e^{im\theta_1} \\ \vdots \end{pmatrix}$$

Attention between positions $m$ and $n$:

$$\text{Attention} \propto \text{Re}\left[\sum_j \bar{q}_j k_j \cdot e^{i(m-n)\theta_j}\right]$$

→ Attention depends on **relative position** $(m - n)$.

### 5.2 Interpolation Method

To extend from training length $L$ to inference length $L'$:

$$f'(\mathbf{x}, m) = f\!\left(\mathbf{x},\; \frac{mL}{L'}\right)$$

- Divide angles by factor $\frac{L'}{L}$
- Position indices become fractional → all values remain within the trained range $[0, L]$

### 5.3 Theoretical Guarantee

Interpolated attention scores are bounded:

$$|\text{Attention}| \leq \frac{d}{2} \cdot \max_j |a_j|$$

→ Attention scores **don't change drastically** after interpolation.

### 5.4 Fine-Tuning Requirement

Even though attention scores are bounded, **small fine-tuning is still needed** because the model's notion of "distance 1" now maps to "distance 2" (or more).

### 5.5 Results

After interpolation + fine-tuning:
- Perplexity **decreases** as context increases (expected behavior)
- No perplexity blow-up beyond training length

**NTK-aware interpolations**: Non-linear angle divisions (further improvements, not covered in detail).

---

## 6. Lost in the Middle

### 6.1 Experimental Setup

- Given: question + 1 gold document + $k$ distractor documents
- Vary: position of gold document among distractors
- Measure: accuracy of LLM answer

### 6.2 Key Finding: U-Shaped Performance Curve

| Gold Document Position | Model Performance |
|---|---|
| **Beginning** (position 1) | **Best** |
| **Middle** (position 5, 10, 15) | **Worst** |
| **End** (last position) | Moderate (better than middle) |

This pattern is consistent across:
- GPT-3.5, Claude, and other LLMs
- 10, 20, 30 documents
- Causal and bidirectional models
- Pre-trained and instruction-tuned models

### 6.3 Implications

- **More retrieval ≠ better performance**: Retriever recall improves with more documents, but LLM accuracy plateaus or drops
- Bidirectional models (Flan-T5, Flan-UL2) show slightly less drop for small document counts, but same trend at scale
- Models effectively **ignore information in the middle** of the context window

---

## 7. Summary Table

| Approach | What It Solves | Limitation |
|---|---|---|
| **LongNet** (Dilated Attention) | Reduces attention to $O(n)$ | Still requires training on long context |
| **ALiBi** | Train short, infer long (no perplexity blow-up) | Far tokens get zero attention |
| **Position Interpolation** | Extend RoPE to longer contexts | Requires small fine-tuning |
| **Lost in the Middle** | Identifies attention bias in position | No solution — documents problem |
