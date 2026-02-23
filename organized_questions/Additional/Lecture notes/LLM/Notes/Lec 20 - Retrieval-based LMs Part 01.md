# Lecture 20 — Retrieval-based Language Models (Part 01)

---

## 1. Motivation: Why Retrieval?

### Problems with Parametric LLMs

| Problem | Description |
|---------|-------------|
| **Hallucination** | Model generates plausible but factually incorrect text |
| **Knowledge cutoff** | Training data has a fixed timestamp; can't answer about recent events |
| **Verifiability** | No way to trace/verify where the information came from |
| **Domain knowledge** | Enterprise/proprietary data not in pre-training corpus |

### Parametric vs Semi-Parametric LMs

| Type | Analogy | Knowledge Source |
|------|---------|-----------------|
| **Parametric** (closed-book) | Exam without reference | All knowledge encoded in model weights |
| **Semi-parametric** (open-book) | Exam with reference book | Weights + external retrieved documents |

---

## 2. Three Approaches to Using Retrieved Information

| Approach | Where Retrieved Info is Used | Description |
|----------|------------------------------|-------------|
| **Output interpolation** | Final output layer | Interpolate LM output distribution with retrieval-based distribution (e.g., kNN-LM) |
| **Intermediate fusion** | Hidden layers | Inject retrieved info into intermediate transformer layers |
| **Input augmentation (RAG)** | Input context | Prepend retrieved passages to the input prompt — **most popular** |

---

## 3. Retrieval Pipeline

```
Documents → Chunking → Representation → Indexing → [query] → Matching → Top-K passages
```

### 3.1 Document Chunking

Split documents into manageable passages/chunks before indexing. Chunk size affects retrieval granularity.

### 3.2 Representation

Two paradigms: **Sparse** (lexical) and **Dense** (semantic).

### 3.3 Indexing

Store representations in a searchable data structure for efficient retrieval.

### 3.4 Matching

Given a query, find the top-K most similar documents.

---

## 4. Sparse Retrieval

### 4.1 Bag-of-Words Representation

Represent documents as vectors where each dimension corresponds to a vocabulary term.

### 4.2 Term Frequency (TF)

$$\text{TF}(t, d) = \frac{\text{count}(t, d)}{|d|}$$

Raw frequency of term $t$ in document $d$, normalized by document length.

### 4.3 TF-IDF

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

$$\text{IDF}(t) = \log \frac{N}{|\{d : t \in d\}|}$$

- Terms appearing in many documents get **low** IDF (less discriminative).
- Rare terms get **high** IDF (more informative).

### 4.4 BM25 (Best Match 25)

The gold standard for sparse retrieval:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{TF}(t,d) \cdot (k_1 + 1)}{\text{TF}(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

| Parameter | Role | Typical Value |
|-----------|------|---------------|
| $k_1$ | TF saturation — controls diminishing returns of repeated terms | 1.2–2.0 |
| $b$ | Document length normalization ($b=0$: no normalization, $b=1$: full) | 0.75 |
| $\text{avgdl}$ | Average document length in corpus | — |

**Key property**: TF saturation — unlike raw TF, additional occurrences of a term yield diminishing score increases.

### 4.5 Inverted Index (for Efficient Search)

- **Data structure**: Maps each word → list of document IDs containing that word.
- At query time: look up each query term → get candidate document sets → compute BM25 only on the intersection/union.
- Avoids computing similarity against all documents.

```
"NLP"    → [doc1, doc10, doc23]
"what"   → [doc1, doc5, doc8, doc10]
"is"     → [doc1, doc3, doc5, doc8, doc10, ...]
```

### 4.6 Limitations of Sparse Retrieval

| Limitation | Explanation |
|------------|-------------|
| **Lexical matching only** | Can't match synonyms (e.g., "car" vs "automobile") |
| **No semantic understanding** | Can't disambiguate polysemous words (e.g., "bank" → river vs financial) |
| **Vocabulary mismatch** | Relevant documents using different terminology won't be retrieved |

---

## 5. Dense Retrieval

### 5.1 Core Idea

Replace sparse BoW vectors with **dense embeddings** from neural encoders (e.g., BERT):
- Document: $\mathbf{e}_d = \text{Encoder}_D(d) \in \mathbb{R}^h$
- Query: $\mathbf{e}_q = \text{Encoder}_Q(q) \in \mathbb{R}^h$

Similarity via dot product: $\text{sim}(q, d) = \mathbf{e}_q \cdot \mathbf{e}_d$

### 5.2 Bi-Encoder Architecture

| Component | Purpose |
|-----------|---------|
| **Query Encoder** $E_Q$ | Embeds query into dense vector |
| **Document Encoder** $E_D$ | Embeds document into dense vector |

- Can use **same** or **separate** encoder models.
- If same model: distinguish via special prefix tokens (e.g., `[Q]` vs `[D]`).
- Using separate models is preferred — query and document have fundamentally different semantics (question ≠ answer).

### 5.3 Why Not Use BERT Out-of-the-Box?

BERT is pre-trained for MLM/NSP, **not** for making question embeddings similar to answer embeddings. The query encoder and document encoder must be **fine-tuned** with a retrieval objective.

### 5.4 Training Objective

Maximize probability of selecting the gold document $d^*$ from a pool:

$$P(d^* | q) = \frac{\exp(\mathbf{e}_q \cdot \mathbf{e}_{d^*})}{\sum_{j=1}^{|D|} \exp(\mathbf{e}_q \cdot \mathbf{e}_{d_j})}$$

**Problem**: Computing denominator over all $|D|$ (millions) documents is intractable.

### 5.5 Negative Sampling

Approximate the denominator with a small set of $n$ negatives ($n \ll |D|$):

$$\mathcal{L} = -\log \frac{\exp(s^+)}{\exp(s^+) + \sum_{i=1}^{n} \exp(s_i^-)}$$

| Strategy | Method | Quality |
|----------|--------|---------|
| **Random / In-batch** | Use other documents in the mini-batch as negatives | Easy negatives — model learns coarse distinctions |
| **Hard negatives (BM25)** | Use top-K BM25 results (excluding gold) as negatives | Hard negatives — forces fine-grained discrimination |

**Hard negatives** are critical: force the model to distinguish between semantically similar but irrelevant documents vs the gold document.

### 5.6 Unsupervised Dense Retrieval (Contriever)

When no query–document pairs are available:
- Take two **random spans** from the same passage.
- Treat them as positive pairs.
- Train encoder so spans from the same document get similar embeddings.

---

## 6. Search Problem for Dense Embeddings

Dense embeddings cannot use inverted indices. Need **Approximate Nearest Neighbor (ANN)** search — covered in Part 02.

---

## 7. Summary

| Method | Representation | Matching | Strengths | Weaknesses |
|--------|----------------|----------|-----------|------------|
| **BM25** | Sparse (TF-IDF-like) | Inverted index | Fast, strong baseline, no training needed | Lexical only, no semantics |
| **Dense Retrieval (DPR)** | Dense (neural embeddings) | ANN search | Semantic matching, handles synonyms | Requires training, slower indexing |
| **Contriever** | Dense | ANN search | No supervised data needed | May underperform supervised methods |
