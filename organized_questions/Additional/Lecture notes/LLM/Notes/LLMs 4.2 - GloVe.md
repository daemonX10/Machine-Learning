# Lecture 4.2 — GloVe (Global Vectors for Word Representation)

---

## 1. Recap: Count-Based vs Prediction-Based Methods

### Count-Based Methods (TF-IDF, PMI, PPMI)

| Aspect | Detail |
|---|---|
| **How it works** | Build a word × word co-occurrence matrix, scan corpus once |
| **Advantages** | Fast training; captures co-occurrence statistics accurately |
| **Disadvantages** | Sparse, high-dimensional matrix; captures only word similarity (not dense embeddings); gives disproportionate importance to highly frequent co-occurrences (e.g., stop-word pairs) |

### Prediction-Based Methods (Word2Vec)

| Aspect | Detail |
|---|---|
| **How it works** | Slide a context window, run logistic regression for every window |
| **Advantages** | Produces dense embeddings; improved downstream task performance; captures complex patterns (analogy tests like $\text{King} - \text{Man} + \text{Woman} = \text{Queen}$) |
| **Disadvantages** | Scales with corpus size (more data → more computation even if vocabulary is fixed); doesn't leverage global co-occurrence statistics |

---

## 2. GloVe — Combining Best of Both Worlds

- **Proposed:** 2014 by Pennington, Socher & Manning (Stanford)
- **Core idea:** Use the **co-occurrence matrix** (count-based) but learn **dense embeddings** (prediction-based)

---

## 3. Intuition: Co-occurrence Ratios as Discriminators

Consider target words **ice** and **steam** with context (pivot) words:

| Context word $k$ | $P(k \mid \text{ice})$ | $P(k \mid \text{steam})$ | Ratio $\frac{P(k \mid \text{ice})}{P(k \mid \text{steam})}$ |
|---|---|---|---|
| **solid** | High | Low | **Large** (≫ 1) |
| **gas** | Low | High | **Small** (≪ 1) |
| **water** | High | High | ≈ 1 |
| **fashion** | Low | Low | ≈ 1 |

**Key insight:** The ratio differentiates:
- **Relevant context** for ice (ratio ≫ 1) vs steam (ratio ≪ 1)
- **Irrelevant context** — either unrelated (fashion) or shared (water) → ratio ≈ 1

---

## 4. Deriving the GloVe Objective

### Step 1 — Model dot product as log-probability

For target word $i$ and context word $j$ with embeddings $w_i$ and $w_j$:

$$w_i \cdot w_j = \log P(j \mid i) = \log \frac{X_{ij}}{X_i}$$

where:
- $X_{ij}$ = co-occurrence count of words $i$ and $j$
- $X_i = \sum_k X_{ik}$ = total count for word $i$

### Step 2 — Symmetrize

When $i$ is target and $j$ is context:
$$w_i \cdot w_j = \log X_{ij} - \log X_i$$

When $j$ is target and $i$ is context:
$$w_j \cdot w_i = \log X_{ij} - \log X_j$$

Adding the two equations:
$$w_i \cdot w_j = \log X_{ij} - \frac{1}{2}\log X_i - \frac{1}{2}\log X_j$$

### Step 3 — Absorb word-specific terms into bias

The terms $-\frac{1}{2}\log X_i$ and $-\frac{1}{2}\log X_j$ depend only on individual words, so model them as bias terms $b_i$ and $b_j$:

$$w_i \cdot w_j + b_i + b_j \approx \log X_{ij}$$

### Step 4 — Squared loss objective

$$J = \sum_{i,j} f(X_{ij}) \left( w_i \cdot w_j + b_i + b_j - \log X_{ij} \right)^2$$

---

## 5. The Weighting Function $f(X_{ij})$

Addresses the problem of disproportionate weight to frequent co-occurrences.

### Required Properties

1. $f(0) = 0$ and $f$ is continuous — finite at zero
2. **Non-decreasing** — never decreases with increasing co-occurrence
3. **Relatively small for large $X_{ij}$** — doesn't over-weight very frequent pairs

### Chosen Function

$$f(x) = \begin{cases} \left(\frac{x}{x_{\max}}\right)^\alpha & \text{if } x < x_{\max} \\ 1 & \text{if } x \geq x_{\max} \end{cases}$$

- $x_{\max}$ is a cutoff (typically 100)
- $\alpha$ is typically 0.75
- After $x_{\max}$: weight is capped at 1 (no extra weight for very frequent pairs)
- Before $x_{\max}$: approximately linear growth

---

## 6. Training

- Minimize $J$ via gradient descent
- Take derivatives w.r.t. $w_i$ and $w_j$
- Two sets of embeddings learned: target ($w_i$) and context ($w_j$) — same as Word2Vec

---

## 7. GloVe Properties

| Property | Detail |
|---|---|
| **Static embedding** | One fixed vector per word (no context-dependence) |
| **Fast training** | Leverages pre-computed co-occurrence matrix |
| **Scalable** | Handles huge corpora efficiently |
| **Small corpus** | Works well even with limited data (unlike Word2Vec) |
| **Combines both paradigms** | $\log X_{ij}$ → count-based; $w_i \cdot w_j$ → prediction-based |

---

## 8. Applications of Word Embeddings

### Analogy Tests
$$\vec{\text{Beijing}} - \vec{\text{China}} + \vec{\text{India}} \approx \vec{\text{Delhi}}$$

### Bias Detection
- "Father : Computer Programmer :: Mother : ?" → **Homemaker**
- "Man : Doctor :: Woman : ?" → **Nurse**
- Word embeddings trained on biased data inherit and amplify biases

### Diachronic (Historical) Word Meaning Shifts
- Train separate embeddings per decade (e.g., on NYT articles 1950–2010)
- Track how a word's nearest neighbors change over time
- Example: **"broadcast"** — agriculture (spreading seeds) → radio/media

---

## 9. Key Takeaways

- GloVe bridges count-based and prediction-based methods via a weighted least-squares objective on log co-occurrence counts
- The weighting function $f$ prevents stop-word pairs from dominating
- GloVe embeddings serve as inputs to downstream neural models (including Transformers)
