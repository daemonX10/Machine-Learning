# Lecture 21 — Retrieval-based Language Models (Part 02)

---

## 1. Approximate Nearest Neighbor (ANN) Search

Dense retrieval yields $h$-dimensional vectors. Computing dot product with all $|D|$ documents is $O(|D| \cdot h)$ — infeasible for millions of documents. ANN trades **exact** results for **speed**.

### 1.1 Hierarchical Clustering (IVF)

1. **Offline**: Cluster all document embeddings into $K$ clusters (e.g., K-means).
2. **Online**: Given query, find nearest cluster centroid(s) → search only within those clusters.

$$\text{Complexity} \approx O(\log K + |C_{\text{nearest}}|)$$

- Multi-level clustering → tree structure → $O(\log n)$ search.
- Used in **FAISS** (Facebook AI Similarity Search).

### 1.2 Locality Sensitive Hashing (LSH)

1. Generate $m$ **random hyperplanes** in the embedding space.
2. Each hyperplane assigns a bit: point is on positive (+1) or negative (0) side.
3. Each vector gets an $m$-bit **hash code**.
4. Vectors with the **same hash** are likely near each other.

| Property | Detail |
|----------|--------|
| Hash length $m$ | More bits → fewer false positives, more false negatives |
| Multiple hash tables | Use several independent hash functions → improve recall |
| Complexity | $O(1)$ lookup per hash table (approximate) |

### 1.3 Graph-based Methods (HNSW)

Hierarchical Navigable Small World graphs — build a proximity graph; navigate via greedy search. State-of-art ANN speed-accuracy tradeoff.

---

## 2. Scoring: Bi-Encoder vs Cross-Encoder

### 2.1 Bi-Encoder (Dual Encoder)

```
Query  →  Encoder_Q  →  e_q  ─┐
                                ├─→  dot(e_q, e_d)  →  score
Doc    →  Encoder_D  →  e_d  ─┘
```

| Advantage | Limitation |
|-----------|-----------|
| Documents encoded **offline** | Single vector per document — information bottleneck |
| Sub-linear search via ANN index | No cross-attention between query and document |
| Scales to millions of documents | Lower accuracy than cross-encoder |

### 2.2 Cross-Encoder

```
[CLS] Query [SEP] Document [SEP]  →  Encoder  →  CLS score
```

| Advantage | Limitation |
|-----------|-----------|
| Full cross-attention between query and document tokens | **Cannot** pre-compute document embeddings offline |
| Higher accuracy — richer interaction modeling | Must run encoder for **every** (query, doc) pair at inference |
| | $O(|D|)$ per query — only usable for **reranking** a small candidate set |

### 2.3 When to Use Which

| Stage | Method | Input |
|-------|--------|-------|
| **Retrieval** (stage 1) | Bi-encoder + ANN | All documents |
| **Reranking** (stage 2) | Cross-encoder | Top-K from stage 1 (e.g., K=100) |

---

## 3. ColBERT — Late Interaction

### 3.1 Architecture

ColBERT keeps **token-level embeddings** for both query and document (instead of a single vector):

- Query tokens: $\{q_1, q_2, \ldots, q_m\}$, each $\in \mathbb{R}^h$
- Document tokens: $\{d_1, d_2, \ldots, d_n\}$, each $\in \mathbb{R}^h$

### 3.2 MaxSim Scoring

$$\text{score}(q, d) = \sum_{i=1}^{m} \max_{j=1}^{n} (q_i \cdot d_j)$$

For each query token, find the **maximum similarity** across all document tokens, then sum.

### 3.3 Key Advantage

| Property | Detail |
|----------|--------|
| **Offline indexing** | Document token embeddings can be pre-computed and stored |
| **Richer representation** | Each document stores $n \times h$ values (vs $1 \times h$ for bi-encoder) |
| **No information bottleneck** | Token-level embeddings preserve fine-grained information |
| **ANN compatible** | Can index all token embeddings and use ANN for initial retrieval |

### 3.4 Storage Cost

Much higher than bi-encoder: storing $n$ embeddings per document (e.g., 100 tokens × 128 dims = 12,800 floats per document vs 128 for bi-encoder).

---

## 4. HyDE — Hypothetical Document Embeddings

1. Given query $q$, ask an LLM to generate a **hypothetical answer**.
2. Embed the hypothetical answer (not the query).
3. Search the index using the hypothetical answer embedding.

**Intuition**: The hypothetical answer is lexically/semantically closer to the actual document than the question is.

---

## 5. Differentiable Search Index (DSI)

Train an encoder-decoder model to directly map queries → document IDs:

| Phase | Input | Output |
|-------|-------|--------|
| **Indexing** | Document text | Document ID |
| **Querying** | Query text | Document ID |

- Document IDs can be **atomic** (1M-way softmax) or **structured** (hierarchical: book → chapter → section).
- Structured IDs enable **progressive narrowing** during decoding.

---

## 6. RAG — Retrieval-Augmented Generation

The **input augmentation** approach:

```
User Query + Top-K Retrieved Passages  →  LLM  →  Response
```

- LLMs excel at reading comprehension: if relevant info is in context, they can usually extract & synthesize it.
- **Tools**: LangChain, LlamaIndex for pipeline setup; ChromaDB, FAISS for vector indices; Lucene for BM25.

---

## 7. Graph RAG (HippoRAG)

### Problem
Naive RAG fails for **multi-hop queries** — answer requires connecting information from multiple passages.

### Solution: Knowledge Graph-enhanced Retrieval

**Offline (Indexing)**:
1. Extract **triples** (entity, relation, entity) from passages via Open IE / LLM prompting.
2. Build a knowledge graph; connect synonym nodes via embedding similarity > threshold.
3. Store: node embeddings, knowledge graph, node-to-passage mapping.

**Online (Query)**:
1. Extract named entities from query.
2. Search node embeddings via ANN to find matching KG nodes.
3. Run **Personalized PageRank** (PPR) from matched nodes → get subgraph with node probabilities.
4. Map nodes → passages using stored mapping → score passages.

**Result**: Significantly improves recall@2 and recall@5 on multi-hop benchmarks (e.g., MuSiQue).

---

## 8. Open Challenges

| Challenge | Description |
|-----------|-------------|
| **Lost in the Middle** | LLMs attend to beginning/end of context; information in the middle gets neglected. Need permutation-invariant inference strategies. |
| **Retriever failure** | Retriever may not find relevant documents at all |
| **Reasoning failure** | Even with gold passages in context, LLMs can still fail to reason correctly |
| **Hallucination with context** | LLM may ignore retrieved passages and hallucinate |
| **Hallucination quantification** | Detecting whether the answer comes from retrieved passages or model's parametric memory |

### RAFT — Domain-Aware RAG

Ingest domain knowledge into model weights during fine-tuning, so the model is **aware** of the corpus even without retrieval. Hybrid approach combining parametric + semi-parametric knowledge.

---

## 9. Summary Table

| Method | Type | Indexing | Search | Strength |
|--------|------|----------|--------|----------|
| **BM25** | Sparse | Inverted index | Exact lexical | Fast, strong baseline |
| **Bi-Encoder (DPR)** | Dense | ANN (FAISS) | Semantic | Handles synonyms |
| **Cross-Encoder** | Dense | N/A | Rerank only | Highest accuracy |
| **ColBERT** | Dense (token-level) | ANN on tokens | MaxSim | Rich representation + offline indexing |
| **HyDE** | Dense | Standard | Query expansion via LLM | Better query representation |
| **DSI** | Generative | In model weights | Decoder generates doc ID | End-to-end neural |
| **HippoRAG** | Graph + Dense | KG + ANN | PPR on KG | Multi-hop queries |

### Key Libraries & Tools

| Tool | Purpose |
|------|---------|
| **LangChain** / **LlamaIndex** | RAG pipeline orchestration |
| **FAISS** | Dense vector ANN search |
| **ChromaDB** | Vector database |
| **Lucene** | BM25 / sparse retrieval |
| **Neo4j** | Knowledge graph storage & algorithms |
