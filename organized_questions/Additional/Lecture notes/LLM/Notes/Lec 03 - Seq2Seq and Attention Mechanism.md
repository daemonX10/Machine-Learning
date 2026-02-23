# Lecture 03 — Seq2Seq Models & Attention Mechanism

## 1. Sequence-to-Sequence Problems

A seq2seq problem maps an **input sequence** to an **output sequence**.

| Type | Input | Output | Example |
|---|---|---|---|
| Many-to-One | Sequence | Single entity | Text → Image |
| One-to-Many | Single entity | Sequence | Image → Caption |
| Many-to-Many | Sequence | Sequence | Machine translation, Summarization |

---

## 2. Encoder-Decoder RNN Architecture

### Architecture

Two separate RNNs (can be LSTM/GRU):

1. **Encoder RNN**: processes source sequence token-by-token, produces hidden states $h_1^e, h_2^e, \ldots, h_T^e$
2. **Decoder RNN**: takes the **final encoder hidden state** $h_T^e$ as initial input, generates target sequence

### Teacher Forcing

| Mode | Input to Decoder at Step $t$ |
|---|---|
| **Teacher forcing** (training) | Ground-truth token $y_{t-1}$ |
| **Autoregressive** (inference) | Model's own prediction $\hat{y}_{t-1}$ |

### Training

- **Loss**: cross-entropy at each decoder position between predicted and actual token
- **Total loss**: $J = \sum_{t} J_t$
- Back-propagated end-to-end via **BPTT** — both encoder and decoder parameters are updated
- Encoder parameters must be updated to learn **alignment** between source and target

### Parameters

| Component | Parameters |
|---|---|
| Encoder | $W_e$ (input-to-hidden projection) |
| Decoder | $W_d$ (hidden-to-hidden), $U$ (unembedding layer) |

---

## 3. The Bottleneck Problem

The final encoder hidden state $h_T^e$ must compress the **entire input sequence** into a single fixed-size vector.

- For long inputs, early information gets **forgotten**
- This single vector becomes a **bottleneck** — too much responsibility on one representation

---

## 4. Attention Mechanism

**Core idea**: give every decoder state **direct access** to all encoder states (not just the last one).

### Step-by-Step Computation

For decoder state $d_t$ attending to encoder states $h_1^e, h_2^e, \ldots, h_T^e$:

**Step 1 — Attention Scores** (dot product):
$$\text{score}(d_t, h_j^e) = d_t^T \cdot h_j^e \quad \forall j \in \{1, \ldots, T\}$$

**Step 2 — Attention Distribution** (softmax):
$$\alpha_{t,j} = \frac{\exp(\text{score}(d_t, h_j^e))}{\sum_{k=1}^{T} \exp(\text{score}(d_t, h_k^e))}$$

**Step 3 — Attention Output** (weighted sum):
$$\text{context}_t = \sum_{j=1}^{T} \alpha_{t,j} \cdot h_j^e$$

**Step 4 — Combine & Predict**:
$$\hat{y}_t = \text{softmax}\left(U \cdot [d_t \;;\; \text{context}_t]\right)$$

> The unembedding matrix $U$ grows in input size due to concatenation of $d_t$ and $\text{context}_t$.

### Terminology Summary

| Term | Definition |
|---|---|
| **Attention scores** | Raw dot products between decoder state and encoder states |
| **Attention distribution** | Softmax-normalized scores (sums to 1) |
| **Attention output** | Weighted sum of encoder states using attention distribution |

### Key Properties

- **No extra parameters** in vanilla attention — only $U$ grows as a byproduct
- Attention output is always a **single vector** regardless of input length
- Weights are **dependent on the decoder state** — different decoder positions attend differently

---

## 5. Benefits of Attention

| Benefit | Explanation |
|---|---|
| **Solves bottleneck** | Every decoder state has direct access to all encoder states |
| **Alignment** | Attention distribution reveals which source tokens map to which target tokens |
| **Interpretability** | Attention weights can be visualized as alignment matrices |
| **Reduces vanishing gradient** | Gradient can flow directly from decoder to encoder via attention shortcut |

---

## 6. Attention on a Single RNN (Non-Seq2Seq)

Attention is not limited to encoder-decoder setups. For a single RNN:

- Each hidden state can attend to **previous** hidden states (backward dependency)
- **Cannot** attend to future states — temporal constraint
- Still produces a context-aware summary vector at each position

---

## 7. Query, Key, Value Framework

| Concept | Role | Analogy (Information Retrieval) |
|---|---|---|
| **Query** ($Q$) | Decoder state — what we're looking for | User's search query |
| **Key** ($K$) | Encoder states — metadata to match against | Document snippets/keywords |
| **Value** ($V$) | Content to retrieve | Actual document content |

- In **vanilla attention**: $K = V$ (both are encoder hidden states)
- In **self-attention** (next lecture): $K \neq V$ — separate projections via learned matrices

**Attention formula**:
1. Score: $\text{score} = Q \cdot K^T$
2. Distribution: $\alpha = \text{softmax}(\text{score})$
3. Output: $\text{Attention} = \alpha \cdot V$

---

## 8. Attention Variants

| Variant | Formula | Parameters |
|---|---|---|
| **Dot-product** | $Q^T K$ | None |
| **Scaled dot-product** | $\frac{Q^T K}{\sqrt{d_k}}$ | None (used in Transformer) |
| **Bilinear (General)** | $Q^T W K$ | $W$ matrix |
| **Additive (Concat)** | $W_2^T \tanh(W_1 [Q; K])$ | $W_1, W_2$ matrices |

> More parameters → more expressive but needs more data.

---

## 9. Remaining Problem

Attention solves the **bottleneck** problem but does **not** solve the **parallelization** problem.

- RNNs still process tokens sequentially due to recurrent dependencies
- **Self-attention** (next lecture) removes recurrence → enables parallel processing

---

## 10. Key Takeaways

- Encoder-decoder RNN is the standard seq2seq architecture
- The **bottleneck problem** arises from compressing entire input into one vector
- **Attention** = weighted access to all encoder states from any decoder state
- Vanilla attention adds **no extra parameters** — it's essentially a similarity-based shortcut
- Attention provides alignment, interpretability, and gradient flow benefits
- The Q/K/V abstraction generalizes attention to self-attention (Transformer)
