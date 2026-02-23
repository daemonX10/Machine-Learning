# Lecture 16.2 — Retrieval-based Language Models II

## Recap: Three Paradigms

| Paradigm | When retrieval is used | Analogy |
|----------|----------------------|---------|
| **Output Interpolation** | After generating the answer | Solve, then verify in the book |
| **Intermediate Fusion** | During generation (on demand) | Get stuck, search book, continue |
| **Input Augmentation (RAG)** | Before generation | Search book first, then solve |

---

## 1. Output Interpolation: kNN-LM

### Core Idea

Combine the LM's parametric distribution with a non-parametric distribution from a nearest-neighbor search over an external corpus.

### How the Index Is Built

For every position in the corpus, create an entry:

$$(\text{key}, \text{value}) = (\text{context prefix}, \text{next token})$$

**Example:** For sentence "Obama was born in Hawaii and graduated..."

| Key (prefix) | Value (next token) |
|-------------|-------------------|
| "Obama" | "was" |
| "Obama was" | "born" |
| "Obama was born in" | "Hawaii" |

- Index size = total number of tokens in the corpus
- Keys are stored as dense embeddings

### Inference

1. Given input context, compute its embedding
2. Search index for top-$K$ nearest contexts
3. Convert distances to probability distribution (softmax over top-$K$)
4. Aggregate: if same token appears multiple times, sum probabilities
5. **Interpolate** with LM distribution:

$$P(y|x) = \lambda \cdot P_{\text{kNN}}(y|x) + (1 - \lambda) \cdot P_{\text{LM}}(y|x)$$

where $\lambda$ is a hyperparameter controlling reliance on external knowledge.

### Results

- Using 3B token corpus: perplexity **much better** than training an LM on the same 3B tokens
- Even a small corpus (< 1B) outperforms LMs trained on 3B tokens
- Performance scales with both corpus size and number of retrieved contexts ($K$)

### Cross-Domain Transfer

- LM trained on Wiki + external knowledge from Books corpus → competitive with LM trained on Books data
- Optimal $\lambda$ adapts: out-of-domain → higher $\lambda$ (rely more on retrieval); in-domain → lower $\lambda$

### Limitations

| Limitation | Description |
|-----------|-------------|
| Retriever called per token | Extremely frequent retrieval — slow |
| Single-token retrieval | Returns only one next token per lookup |
| No downstream fine-tuning | Index stores (prefix → next token); unclear how to organize for QA tasks |
| Only perplexity evaluated | Not tested on downstream tasks |

---

## 2. Intermediate Fusion: RETRO (Retrieval-Enhanced Transformer)

### Key Improvements Over kNN-LM

| Feature | kNN-LM | RETRO |
|---------|--------|-------|
| Unit of retrieval | Single tokens | **Chunks** (e.g., 64 tokens) |
| Index size | # tokens in corpus | # tokens / chunk_size (100× smaller) |
| Retrieval output | Next token only | Full continuation |
| Retrieval frequency | Every token | Every $L$ tokens |
| Architecture change | None (output interpolation) | **Cross-attention layers** added |

### Architecture

1. **Split input** into fixed-size chunks: $X_1, X_2, X_3, ...$
2. **Retrieve** $K$ chunks for each $X_i$ using a frozen retriever
3. **Encode** retrieved chunks → $E_i \in \mathbb{R}^{K \times R \times D}$
   - $K$ = number of retrieved chunks
   - $R$ = tokens per chunk
   - $D$ = embedding dimension
4. **Chunked Cross-Attention (CCA):**
   - For last token of $X_1$ + all tokens of $X_2$: attend on $E_1$
   - For last token of $X_2$ + all tokens of $X_3$: attend on $E_2$
   - Why: $E_1$ available only after processing $X_1$ completely

```
Standard Transformer Block:
  Self-Attention → Add & Norm → Cross-Attention (on E_i) → Add & Norm → FFN → Add & Norm
```

### Results

- Indexed **1.7 trillion tokens** (feasible due to chunk-level indexing)
- Perplexity: 16 → **4** (massive reduction)

### Fine-Tuning for Downstream Tasks

**Format:** Pad input so the last token of the final chunk aligns with the chunk boundary:

```
[Question: ... | Answer: ] ← padded to chunk boundary
```

Retrieval happens at chunk boundaries → retrieved info informs answer generation.

**QA results:** Comparable to existing methods; however, performance plateaus beyond 20 retrieved passages.

### Limitation: Fixed Retrieval Frequency

RETRO triggers retrieval every $L$ tokens — this is **not adaptive**.

---

## 3. On-Demand Retrieval

### ToolFormer

- LM trained to generate **special tokens** that trigger external tool calls (retrieval, calculator, etc.)
- Model learns **when** to retrieve autonomously

### FLARE (Active Retrieval Augmented Generation)

1. Retrieve with initial query → generate response
2. If generation is **confident** (high probability) → accept
3. If **uncertain** (low probability) → go back to retriever with current context → retrieve more → continue
4. Repeat per sentence

### Retriever Interleaved Generation (RIG)

- Trains LM to generate structured queries for a public **Knowledge Graph** (Data Commons, 250B+ data points)
- LM generates KG queries mid-generation; results are fed back

---

## 4. Input Augmentation: RAG and REALM

### RAG Pipeline Overview

```
Documents → Index (offline)
                   ↑
Query → Retriever → Top-K passages → [Query + Passages] → Generator/LLM → Answer
```

**Advantages:**
- Any off-the-shelf retriever (even Google Search)
- Any off-the-shelf LLM (fine-tuned for reading comprehension)
- Tools: LangChain, LlamaIndex

---

### REALM (Retrieval-Augmented Language Model Pre-training)

**Focus:** Masked Language Models (BERT-style).

#### Joint Training Objective

$$P(y|x) = \sum_{z \in \text{top-}K} P_\eta(z|x) \cdot P_\theta(y|x, z)$$

where:
- $P_\eta(z|x)$ = retriever probability (parameterized by $\eta$)
- $P_\theta(y|x,z)$ = generator probability (parameterized by $\theta$)
- Sum is approximated over top-$K$ documents only (full corpus sum infeasible)

#### Retriever Scoring

$$P_\eta(z|x) \propto \exp\!\big(\text{sim}(E_{\eta_q}(x), \, E_{\eta_d}(z))\big)$$

- $E_{\eta_q}$: query encoder
- $E_{\eta_d}$: document encoder
- Similarity normalized via softmax over top-$K$

#### Generator (Masked LM)

Pass $[x; z]$ through BERT → use embedding at masked position → predict masked token.

#### Fine-Tuning for QA

**Assumption:** Answer is a **span** within retrieved documents.

For each candidate span $(s, e)$:

$$\text{score}(s, e) = \text{MLP}\big(\text{concat}(h_s, h_e)\big)$$

where $h_s, h_e$ are embeddings of the start and end tokens. Normalize scores to get span probabilities.

#### Cold Start Problem

If both retriever and generator are randomly initialized:
- Retriever returns junk → Generator learns to ignore retrieval → Retriever never improves

**Solution — Warm starting:**
1. Pre-train retriever using **Inverse Cloze Task (ICT):** remove a sentence from a passage → that sentence = query, remaining passage = document → train to increase similarity
2. Pre-train LM separately
3. Then jointly train

#### Index Update

- Document encoder $\eta_d$ frozen during training; index is static
- Index re-built every $T$ steps with updated document encoder
- Query encoder $\eta_q$ updates continuously

**Results:** 40% exact match on Natural Questions.

---

### RAG (Retrieval-Augmented Generation)

**Focus:** Generative (decoder-only) models — applicable to any task.

#### Two Variants

##### RAG-Sequence

Generate entire answer conditioned on each retrieved document separately, then marginalize:

$$P(y|x) = \sum_{z \in \text{top-}K} P_\eta(z|x) \cdot \prod_{t=1}^{|y|} P_\theta(y_t | x, z, y_{<t})$$

- For each document $z_i$: generate full sequence $y$
- Weight each completion by retrieval probability $P(z_i|x)$
- At inference: generate one complete sequence per document → pick best

##### RAG-Token

Marginalize at **each token** step:

$$P(y_t | x, y_{<t}) = \sum_{z \in \text{top-}K} P_\eta(z|x) \cdot P_\theta(y_t | x, z, y_{<t})$$

- All documents contribute to generating **every token**
- Weighted average of $K$ distributions (weights = retrieval probabilities)
- At inference: generate one token at a time using combined distribution

**Results:** 44% exact match on Natural Questions (RAG-Sequence).

---

## 5. Training Paradigms Comparison

| Paradigm | What's trained | Pros | Cons |
|----------|---------------|------|------|
| **Independent** | Retriever & LM trained separately | Works with off-the-shelf models; each improved independently | LM not trained to leverage retrieval |
| **Sequential** | One frozen, other fine-tuned (e.g., RETRO) | Simpler than joint training | Frozen component may be suboptimal |
| **Joint (End-to-End)** | Both retriever & LM (REALM, RAG) | Retriever-aware generation | Cold start problem; expensive index updates; train-test discrepancy |

---

## 6. Open Challenges

| Challenge | Description |
|-----------|-------------|
| **Lost in the middle** | LMs attend to start/end of context, ignoring middle — hurts when retrieved info is mid-context |
| **Persistent hallucination** | RAG does not guarantee faithful answers |
| **Retriever failures** | Wrong documents retrieved → wrong answers |
| **Reasoning failures** | Correct documents retrieved but model fails to reason over them |
| **Domain adaptation** | Fine-tuning LM on domain corpus may degrade general capabilities |
| **Faithfulness quantification** | Need post-hoc methods to verify if output is faithful to retrieved passages |

---

## Summary Table

| Method | Paradigm | Retrieval Unit | Retrieval Freq | Architecture Change | Trainable Components |
|--------|----------|---------------|----------------|--------------------|--------------------|
| kNN-LM | Output interpolation | Token | Every token | None | None (inference only) |
| RETRO | Intermediate fusion | Chunk | Every $L$ tokens | Cross-attention layers | LM (retriever frozen) |
| ToolFormer | Intermediate fusion | Variable | On demand (special tokens) | None (learned tokens) | LM |
| FLARE | Intermediate fusion | Variable | On demand (low confidence) | None | LM |
| REALM | Input augmentation | Passage | Once per query | Span extraction MLP | Retriever + LM (jointly) |
| RAG | Input augmentation | Passage | Once per query | None | Retriever + LM (jointly) |
