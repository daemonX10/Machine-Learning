# Lecture 4.1 — Word Representation & Word2Vec

---

## 1. The Word Representation Problem

**Goal**: Represent words as numerical vectors that capture their meaning.

Computers don't understand characters — they need **numerical representations**. The challenge is designing vectors that:
- Capture semantic similarity (similar words → similar vectors)
- Have fixed dimensionality (regardless of word length)
- Handle polysemy (multiple meanings)
- Scale with vocabulary growth

---

## 2. One-Hot Encoding

### Method
- Vocabulary of size $|V|$ (e.g., 1 million)
- Each word gets a vector of size $|V|$
- Position corresponding to word's ID = 1, all others = 0

### Problems

| Problem | Description |
|---------|-------------|
| **Orthogonality** | All vectors are orthogonal: $\vec{v}_{\text{motel}} \cdot \vec{v}_{\text{hotel}} = 0$ even though semantically similar |
| **High dimensionality** | Vector size = vocabulary size (millions) |
| **No similarity** | Cannot measure semantic relatedness between words |
| **Recomputation** | Adding new words requires rebuilding all vectors |

---

## 3. WordNet-Based Approaches

**WordNet** — a manually constructed lexical ontology with:
- **Synsets** (synonym sets): each word *sense* is represented by a set of synonyms
- **Gloss**: textual description of a synset
- **Relations**: synonym, antonym, hypernym, hyponym, meronym, holonym

### Example: "bass"
Multiple synsets: musical range, fish, singing voice, bass guitar — each sense has a separate synset and gloss.

### Problems with Ontology-Based Approaches

| Problem | Description |
|---------|-------------|
| Missing meanings | Not all senses/contexts captured |
| New words | Adding "nifty", "ninja" requires extensive manual study |
| Subjectivity | Annotator disagreement on senses |
| Limited coverage | Mostly nouns, verbs, adjectives, adverbs — not all POS |
| Human labor | Enormous manual effort to maintain |
| Static | Difficult to keep up-to-date |

---

## 4. Count-Based Approaches

### 4.1 Term-Context Matrix (Co-occurrence Matrix)

Based on **distributional semantics**:
> *"You shall know a word by the company it keeps."* — J.R. Firth (1957)

- **Rows** = words (terms), **Columns** = words (context)
- **Entry** $(i,j)$ = number of times word $i$ and word $j$ co-occur within a sliding window
- This is a **square matrix** (unlike bigram table, context window can be larger than 1)
- Each **row** = term vector; each **column** = context vector for a word

### Measuring Similarity

If two words co-occur with similar context words → they are semantically similar.

| | computer | data | pinch | sugar |
|---|---------|------|-------|-------|
| digital | ✓ | ✓ | ✗ | ✗ |
| information | ✓ | ✓ | ✗ | ✗ |
| mango | ✗ | ✗ | ✓ | ✓ |
| pineapple | ✗ | ✗ | ✓ | ✓ |

→ "digital" ≈ "information" and "mango" ≈ "pineapple"

### Problems with Raw Counts

1. Matrix size grows with vocabulary
2. Vectors are very high-dimensional
3. Matrix is extremely **sparse**
4. **Stop words** dominate the counts (articles, prepositions have highest counts)
5. Word ordering is lost (bag-of-words)

---

### 4.2 TF-IDF

Improves on raw counts by weighting terms based on importance.

#### Term Frequency (TF)

$$\text{TF}(t, d) = \log_{10}(\text{count}(t, d) + 1)$$

- Raw count of term $t$ in document $d$, log-dampened
- Adding 1 prevents $\log(0)$

#### Document Frequency (DF)

$$\text{DF}(t) = \text{number of documents containing term } t$$

| Metric | Definition | Scope |
|--------|-----------|-------|
| **Term Frequency** | Count of $t$ in document $d$ | Per-document |
| **Collection Frequency** | Total count of $t$ across all documents | Corpus-wide |
| **Document Frequency** | Number of documents containing $t$ | Corpus-wide |

#### Inverse Document Frequency (IDF)

$$\text{IDF}(t) = \log_{10}\left(\frac{N}{\text{DF}(t)}\right)$$

- $N$ = total number of documents
- Words in **all** documents → IDF ≈ 0 (unimportant)
- Words in **few** documents → high IDF (distinctive)

**Example**:
| Word | DF | IDF |
|------|-----|-----|
| Romeo | 1 | 1.57 (distinctive) |
| good | all docs | ≈ 0 (common) |
| sweet | all docs | ≈ 0 (common) |

#### TF-IDF Score

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

- High TF-IDF → word is **frequent in this document** but **rare across corpus**
- Addresses the stop-word problem of raw counts

### Remaining Problems (Even After TF-IDF)

- Matrix grows with vocabulary
- High-dimensional, sparse vectors
- SVD can reduce dimensions but is $O(n^2)$ and must be recomputed for new words
- **Does not consider word ordering** (bag-of-words limitation)

---

## 5. Word2Vec — Prediction-Based Approach ⭐

### 5.1 Overview

Proposed by **Mikolov et al. (Google, 2013)**.

A **dense, low-dimensional** vector representation (e.g., 100–500 dimensions) learned via a **self-supervised prediction task**.

| Property | Count-Based | Word2Vec |
|----------|------------|---------|
| Matrix construction | One-time scan | Iterative training |
| Vector type | Sparse, high-dim | Dense, low-dim |
| Statistics preservation | Exact | Approximate |
| New word handling | Recompute matrix | Re-run classifier |
| Word ordering | Ignored | Partially captured via context window |

### 5.2 Two Architectures

| Model | Input → Output | Description |
|-------|----------------|-------------|
| **CBOW** (Continuous Bag of Words) | Context words → Center word | Predict center word from surrounding context |
| **Skip-gram** | Center word → Context words | Predict surrounding context from center word |

> **Skip-gram** is more popular and widely used.

---

## 6. Skip-gram with Negative Sampling (SGNS) — Detailed

### 6.1 Setup

- **Target word** $t$ (center word)
- **Context words** $c$ (words within sliding window of size $k$)
- Slide window one word at a time through the corpus

### 6.2 Positive and Negative Pairs

For each window position:

**Positive pairs** (co-occurring):
$(w_t, w_{t-3}), (w_t, w_{t-2}), (w_t, w_{t-1}), (w_t, w_{t+1}), (w_t, w_{t+2}), (w_t, w_{t+3})$

**Negative pairs** (sampled randomly):
- Words **outside** the context window
- Sample $K$ times more negative than positive (typically $K = 5\text{–}20$)

### 6.3 Classifier: Logistic Regression

The model learns to classify: *Is this pair a real context pair or not?*

**Probability of a positive pair** $(w, c)$:

$$P(+ \mid w, c) = \sigma(\vec{c} \cdot \vec{w}) = \frac{1}{1 + e^{-\vec{c} \cdot \vec{w}}}$$

**Probability of a negative pair**:

$$P(- \mid w, c) = 1 - \sigma(\vec{c} \cdot \vec{w}) = \sigma(-\vec{c} \cdot \vec{w})$$

Using the sigmoid property: $1 - \sigma(x) = \sigma(-x)$

### 6.4 Objective Function

For one target word with positive context $c^+$ and $K$ negative contexts $c^-_1, \dots, c^-_K$:

$$\mathcal{L} = \log \sigma(\vec{c}^+ \cdot \vec{w}) + \sum_{k=1}^{K} \log \sigma(-\vec{c}^-_k \cdot \vec{w})$$

**Maximize** this log-likelihood:
- Maximize $P(+ \mid w, c^+)$ — positive pairs should be classified as positive
- Maximize $P(- \mid w, c^-)$ — negative pairs should be classified as negative

### 6.5 Two Embedding Matrices

Each word has **two vectors**:
- **Target embedding** $\vec{w}$ (when word acts as center word)
- **Context embedding** $\vec{c}$ (when word acts as context word)

**Embedding matrix size**: $2|V| \times d$
- $|V|$ = vocabulary size
- $d$ = embedding dimension (hyperparameter, typically 100–300)

**Final word vector**: Sum (or average) of target and context embeddings:
$$\vec{v}_{\text{word}} = \vec{w}_{\text{word}} + \vec{c}_{\text{word}}$$

### 6.6 Gradient Update Rules

Derivatives of $\mathcal{L}$:

$$\frac{\partial \mathcal{L}}{\partial \vec{c}^+} = (\sigma(\vec{c}^+ \cdot \vec{w}) - 1) \cdot \vec{w}$$

$$\frac{\partial \mathcal{L}}{\partial \vec{c}^-} = \sigma(\vec{c}^- \cdot \vec{w}) \cdot \vec{w}$$

$$\frac{\partial \mathcal{L}}{\partial \vec{w}} = (\sigma(\vec{c}^+ \cdot \vec{w}) - 1) \cdot \vec{c}^+ + \sum_{k=1}^{K} \sigma(\vec{c}^-_k \cdot \vec{w}) \cdot \vec{c}^-_k$$

**Update rules** (gradient ascent with learning rate $\eta$):

$$\vec{c}^+ \leftarrow \vec{c}^+ + \eta \frac{\partial \mathcal{L}}{\partial \vec{c}^+}$$

$$\vec{c}^- \leftarrow \vec{c}^- + \eta \frac{\partial \mathcal{L}}{\partial \vec{c}^-}$$

$$\vec{w} \leftarrow \vec{w} + \eta \frac{\partial \mathcal{L}}{\partial \vec{w}}$$

### 6.7 Geometric Interpretation

During training:
- **Positive context** vectors move **closer** to the target vector
- **Negative context** vectors move **farther** from the target vector
- After convergence: semantically similar words cluster together in the vector space

---

## 7. Negative Sampling Strategy

### Modified Unigram Distribution

Sample negatives not from raw unigram distribution (stop words dominate), but from:

$$P_\alpha(w) = \frac{C(w)^\alpha}{\sum_{w'} C(w')^\alpha}$$

Where $\alpha = 0.75$ (as used in the original paper).

**Effect**:

| Word | Raw $P$ | Modified $P_{\alpha=0.75}$ |
|------|---------|---------------------------|
| A (frequent) | 0.99 | 0.97 |
| B (rare) | 0.01 | 0.03 |

→ Rare words get **slightly higher** sampling probability; frequent words get **slightly lower**.

---

## 8. Subsampling Frequent Words

**Problem**: Common words like "the" create redundant training pairs ("the quick", "quick the" are essentially the same pair).

**Solution**: Remove frequent words with probability:

$$P(\text{discard } w) = 1 - \sqrt{\frac{t}{f(w)}}$$

Where $f(w)$ is the word's frequency and $t$ is a threshold.

| Frequency $f(w)$ | Discard probability |
|-------------------|-------------------|
| ≤ 0.0026 (0.26%) | 0 (always keep) |
| 0.01 | 0.5 (50% discard) |
| 1.0 (only word) | 0.67 (67% discard) |

→ Very frequent words are often removed from training; rare words are always kept.

---

## 9. Word Analogy Test

Word2Vec produces vectors that support **linear algebraic relationships**:

$$\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$$

Other examples:
- Paris − France + Italy ≈ Rome
- bigger − big + small ≈ smaller

---

## 10. Word2Vec Limitations

| Limitation | Description |
|-----------|-------------|
| **Static embeddings** | One vector per word — cannot handle polysemy ("bank" gets one vector for all senses) |
| **OOV words** | No embedding for words not seen during training |
| **No subword info** | Cannot leverage morphological structure |

---

## 11. FastText — Subword Embeddings

Proposed by **Facebook** to address the OOV problem.

### Method
- Decompose words into **character n-grams** (with boundary markers `<` and `>`)
- Example: "where" with $n=3$: `<wh`, `whe`, `her`, `ere`, `re>`, `<where>`
- Each n-gram gets its own embedding
- Word embedding = **sum of all its n-gram embeddings**
- Uses the same Skip-gram training as Word2Vec

### Handling Unknown Words

For a new word "nifty" not in training data:
1. Decompose into character n-grams: `<ni`, `nif`, `ift`, `fty`, `ty>`, `<nifty>`
2. Look up embeddings of known n-grams (many may have been seen)
3. Sum them → approximate embedding for "nifty"

---

## 12. Training Summary: Word2Vec Skip-gram

```
1. Initialize embedding matrix (2|V| × d) randomly
2. For each position in corpus:
   a. Define sliding window (context size k)
   b. Curate positive pairs: (target, context_word)  
   c. Sample K × negative pairs: (target, random_word)
   d. Run logistic regression:
      - Maximize P(+ | positive pairs)
      - Maximize P(- | negative pairs)
   e. Update embeddings via gradient ascent
   f. Slide window forward by 1
3. Repeat until convergence
4. Final embedding = w_target + c_context for each word
```

---

## 13. Count-Based vs. Prediction-Based: Comparison

| Feature | Count-Based (TF-IDF, Co-occurrence) | Prediction-Based (Word2Vec) |
|---------|--------------------------------------|----------------------------|
| Construction | Single pass through corpus | Iterative training (multiple epochs) |
| Vector type | Sparse, high-dimensional | Dense, low-dimensional |
| Statistics | Exact corpus statistics | Learned approximation |
| Scalability | Matrix grows with vocab | Fixed embedding size |
| New words | Recompute entire matrix | Re-train or use subword (FastText) |
| Synonymy capture | Indirect (via shared context) | Direct (vectors are close) |
| Word order | Ignored | Partially captured via window |

---

## 14. Key Formulas Quick Reference

| Concept | Formula |
|---------|---------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ |
| Skip-gram probability | $P(+\mid w,c) = \sigma(\vec{c}\cdot\vec{w})$ |
| Negative probability | $P(-\mid w,c) = \sigma(-\vec{c}\cdot\vec{w})$ |
| SGNS objective | $\mathcal{L} = \log\sigma(\vec{c}^+\cdot\vec{w}) + \sum_k \log\sigma(-\vec{c}^-_k\cdot\vec{w})$ |
| Negative sampling dist. | $P_\alpha(w) = \frac{C(w)^{0.75}}{\sum C(w')^{0.75}}$ |
| TF | $\text{TF}(t,d) = \log(\text{count}(t,d)+1)$ |
| IDF | $\text{IDF}(t) = \log\frac{N}{\text{DF}(t)}$ |
| TF-IDF | $\text{TF-IDF} = \text{TF} \times \text{IDF}$ |
| Word analogy | $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$ |
