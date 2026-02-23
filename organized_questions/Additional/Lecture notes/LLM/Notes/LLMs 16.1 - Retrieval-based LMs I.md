# Lecture 16.1 — Retrieval-based Language Models I

## Motivation: Why Retrieval?

### Problems with Parametric LLMs

| Problem | Description |
|---------|-------------|
| **Hallucination** | Models fabricate facts, citations, URLs that don't exist |
| **Knowledge cutoff** | Training data has a fixed date; no knowledge of recent events |
| **Learning failures** | Models don't achieve zero perplexity; not all knowledge is memorized |
| **Verification difficulty** | Even self-verification can produce incorrect outputs |

**Real-world consequences:** Lawyers citing fake cases, chatbots creating binding contracts, fabricated academic citations.

### Closed-Book vs. Open-Book Analogy

| Paradigm | Analogy | Description |
|----------|---------|-------------|
| **Parametric LM** | Closed-book exam | All knowledge stored in parameters; must answer from memory |
| **Retrieval-based LM** | Open-book exam | Can access external knowledge at test time |

---

## Three Paradigms for Using Retrieved Information

| Paradigm | Analogy | When retrieval happens | Example works |
|----------|---------|----------------------|---------------|
| **Output Interpolation** | Solve first, then verify in the book | After generation | kNN-LM |
| **Intermediate Fusion** | Solve, get stuck, search book, continue | During generation | RETRO, ToolFormer |
| **Input Augmentation** | Read question, search book, then solve | Before generation | RAG, REALM |

---

## Retrieval Methods

### 1. Sparse Retrieval (Lexical / Token-Based)

**Representation:** Bag-of-words vector of dimension $|V|$ (vocabulary size). Most entries are zero → **sparse**.

#### Term Frequency (TF)

$$\text{TF}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total tokens in } d}$$

**Problem:** Stop words (the, is, a) dominate similarity scores.

#### TF-IDF

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \underbrace{\log \frac{N}{|\{d' : t \in d'\}|}}_{\text{IDF}(t)}$$

- $N$ = total documents in corpus
- IDF upweights rare terms, downweights frequent terms

#### BM25 (Best Match 25)

Improvements over TF-IDF:

1. **Document length normalization**: Penalizes long documents that contain most vocabulary terms

$$\text{BM25} \propto \text{IDF}(t) \cdot \frac{\text{TF}(t,d) \cdot (k_1 + 1)}{\text{TF}(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{d_{\text{avg}}}\right)}$$

2. **Term frequency saturation** ($k_1$): Bounds the impact of repeated rare words; beyond a threshold, additional occurrences are not counted

**Efficient search:** Inverted index maps each term → list of document IDs containing it. Software: **Lucene**.

#### Limitations of Sparse Retrieval

- No semantic understanding (polysemy, synonyms missed)
- Out-of-vocabulary words not handled
- Exact lexical match required

---

### 2. Dense Retrieval

**Representation:** Encode queries and documents as dense vectors ($d$-dimensional, e.g., 1024).

#### Why Not Use Off-the-Shelf Embeddings?

Queries and documents lie in **different semantic spaces** — a question-asking space vs. an answer-containing space. Generic models (BERT, RoBERTa) aren't trained to align these.

#### Training Dense Retrievers

**Training data:** (query, positive document) pairs.
**Objective:** Contrastive learning — bring query and positive document embeddings close, push negatives apart.

**Loss (contrastive / InfoNCE):**

$$\mathcal{L} = -\log \frac{e^{\text{sim}(q, d^+)}}{e^{\text{sim}(q, d^+)} + \sum_{d^- \in \mathcal{N}} e^{\text{sim}(q, d^-)}}$$

#### Negative Sampling Strategies

| Method | Strategy |
|--------|----------|
| **Static negatives** | Fixed set per query — poor generalization |
| **In-batch negatives** | Other documents in the mini-batch serve as negatives |
| **DPR (Dense Passage Retrieval)** | Hard negatives from BM25 top results (lexically similar but irrelevant) |
| **Contriever** | Unsupervised: two chunks from the same passage treated as positives |

#### Encoder Architectures

- **Bi-encoder (dual encoder):** Separate query encoder + document encoder (or shared model with special prefix tokens)
- Documents embedded **offline** → stored in index
- Query embedded **online** → search index

---

### 3. Approximate Nearest Neighbor (ANN) Search

Computing similarity with all documents is $O(N)$ — infeasible for millions of documents.

#### Hierarchical Tree-Based (Clustering)

1. Cluster all document embeddings (e.g., 3 clusters)
2. Query → find nearest cluster center → search within that cluster
3. Recurse: sub-cluster → $O(\log N)$ search
4. Software: **Annoy**, **FAISS**

#### Locality-Sensitive Hashing (LSH)

1. Initialize $H$ random hyperplane classifiers
2. Each partitions the space into two halves → $H$-bit hash per document
3. Query → compute hash → retrieve documents with same/similar hash
4. Effectively creates an inverted index over hash buckets

**Note:** Both methods are **approximate** — may miss relevant documents near cluster boundaries.

---

### 4. Cross-Encoder Reranking

| Architecture | Pros | Cons |
|-------------|------|------|
| **Bi-encoder** | Documents pre-computed offline; fast search | Single embedding per document loses token-level detail |
| **Cross-encoder** | Query + document encoded jointly; full token interaction | Cannot pre-compute; must run model per (query, doc) pair |

**Typical pipeline:** Bi-encoder retrieves top-100 → Cross-encoder reranks.

---

### 5. ColBERT (Contextualized Late Interaction)

A **hybrid** between bi-encoder and cross-encoder:

1. **Offline:** Store **per-token embeddings** for each document (higher storage cost)
2. **Online:** Compute per-token query embeddings
3. **Scoring — MaxSim:**

$$\text{Score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \, q_i \cdot d_j$$

For each query token, find the most similar document token (MaxSim), then sum across all query tokens. Preserves token-level granularity without joint encoding.

---

### 6. HyDE (Hypothetical Document Embeddings)

**Idea:** Use an LLM to generate a hypothetical answer document, then embed it with a standard document embedder.

```
Query → LLM (e.g., GPT-3) → Hypothetical document → Document embedder → Search index
```

- Bridges the query–document space gap without training a specialized retriever
- Can also serve as **query expansion** for BM25 (adds synonyms, related terms)

---

### 7. Differentiable Search Index (DSI)

**Radical approach:** Train a seq2seq model to **directly map queries to document IDs**.

```
Encoder: query → representation
Decoder: representation → document ID
```

**Two training tasks:**
1. **Indexing:** document content → document ID
2. **Retrieval:** query → document ID

**Document ID representations:**

| Type | Description | Performance |
|------|-------------|-------------|
| Atomic | One token per doc ID; softmax over $|\text{corpus}|$ | Baseline |
| Naive string | "doc-id-5" decoded token by token | Moderate |
| Semantic string | Hierarchical cluster-based IDs (e.g., "2-3-3") | **Best** (~2× hit@1 vs. dual encoder) |

**Semantic IDs:** Cluster document embeddings hierarchically → each document gets an ID encoding its cluster path (level1-level2-leaf).

**Limitation:** Preliminary; tested on up to ~300K documents.

---

### 8. Table of Contents Search

**Idea:** Provide the document's **table of contents** to an LLM, ask it to identify the relevant section.

```
Input: [ToC of book] + [Question]
Output: Chapter/subsection title containing the answer
```

- Exploits LLMs' ability to match questions to section headings
- Can be zero-shot or fine-tuned
- Preliminary results show 17–19 point gains over BM25/DSI on some domains

---

## Summary

| Retrieval Method | Type | Key Feature |
|-----------------|------|-------------|
| BM25 | Sparse | TF-IDF with length norm + saturation |
| DPR / Contriever | Dense (bi-encoder) | Learned embeddings with contrastive loss |
| Cross-encoder | Dense (joint) | Full token interaction; expensive |
| ColBERT | Dense (late interaction) | Per-token MaxSim; offline doc embeddings |
| HyDE | Hybrid | LLM generates hypothetical doc for embedding |
| DSI | Generative | Seq2seq model outputs doc IDs directly |
| ToC Search | Prompting | LLM searches table of contents |
