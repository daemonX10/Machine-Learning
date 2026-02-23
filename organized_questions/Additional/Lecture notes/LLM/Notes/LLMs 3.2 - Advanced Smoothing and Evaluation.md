# Lecture 3.2 — Advanced Smoothing & Evaluation of Language Models

---

## 1. Recap: Add-K Smoothing Issues

- Adding +1 to all bigram counts introduces bias: **high-count bigrams are drastically reduced**
- Example: 600 → 300, 500 → 100 — "steals from the rich, distributes to the poor"
- Need more sophisticated smoothing that preserves original count distribution

---

## 2. Good-Turing Smoothing

### 2.1 Core Intuition

> **Use the count of things seen once to estimate the count of things never seen.**

### 2.2 Frequency of Frequencies

Define $N_c$ = **frequency of frequency** $c$ = number of items that appear exactly $c$ times.

**Example corpus**: "Rohan I am I am Rohan I like to play"

| Token | Count |
|-------|-------|
| I | 3 |
| Rohan | 2 |
| am | 2 |
| like | 1 |
| to | 1 |
| play | 1 |

- $N_1 = 3$ (like, to, play — each appears once)
- $N_2 = 2$ (Rohan, am — each appears twice)
- $N_3 = 1$ (I — appears three times)

### 2.3 Bird Watching Example

Observed 18 birds: 10 flamingos, 2 kingfishers, 1 Indian roller, 1 woodpecker, 1 peacock, etc.

| Estimate | P(next = woodpecker) | P(next = new species) |
|----------|---------------------|----------------------|
| **MLE** | $\frac{1}{18}$ | $\frac{0}{18} = 0$ |
| **Good-Turing** | $\frac{2/3}{18} = \frac{1}{27}$ | $\frac{N_1}{N} = \frac{3}{18}$ |

### 2.4 Good-Turing Formula

**For unseen events** (count = 0):

$$P_{\text{GT}}(\text{unseen}) = \frac{N_1}{N}$$

**Modified count** for items with count $c$:

$$c^* = (c + 1) \cdot \frac{N_{c+1}}{N_c}$$

| Original Count $c$ | Modified Count $c^*$ | Change |
|-------|--------|--------|
| 0 | 0.0270 | Now non-zero ✓ |
| 1 | 0.446 | Slight decrease |
| 2 | 1.26 | Slight decrease |
| 3 | 2.24 | Slight decrease |
| 4 | 3.24 | Slight decrease |
| 5 | 4.22 | Slight decrease |

### 2.5 Key Observation

> After Good-Turing, the modified count ≈ **original count − 0.75**

This observation motivates **Absolute Discounting**.

---

## 3. Absolute Discounting Interpolation

Since Good-Turing roughly subtracts a constant (~0.75), simplify:

$$P_{\text{abs}}(W_i \mid W_{i-1}) = \frac{\max(C(W_{i-1}, W_i) - D, \ 0)}{C(W_{i-1})} + \lambda(W_{i-1}) \cdot P(W_i)$$

Where:
- $D$ = discount factor (≈ 0.75)
- $\lambda(W_{i-1})$ = interpolation weight (function of context, tuned on validation set)
- $P(W_i)$ = unigram probability of $W_i$

**Structure**: Discounted bigram + interpolated unigram backup.

### Problem with Unigram Probability

- Stop words ("the", "a", "is") have very high unigram counts
- Domain-specific artifacts: if corpus is about the US, "Angeles" has artificially high unigram probability
- Solution → **Kneser-Ney Smoothing**

---

## 4. Kneser-Ney Smoothing ⭐

### 4.1 Core Idea: Continuation Probability

Replace unigram probability $P(W_i)$ with **continuation probability** $P_{\text{cont}}(W_i)$:

> How likely is $W_i$ to appear as a **novel continuation** — i.e., how many distinct contexts does it complete?

**Example**:
- "Angeles" has high unigram count (frequent in US corpus) but only continues **one** bigram: "Los Angeles"
- "coffee" appears in many contexts: "hot coffee", "cold coffee", "morning coffee", "breakfast coffee" → high continuation count

### 4.2 Continuation Probability Formula

$$P_{\text{cont}}(W_i) = \frac{|\{W_{i-1} : C(W_{i-1}, W_i) > 0\}|}{|\{(W_a, W_b) : C(W_a, W_b) > 0\}|}$$

- **Numerator**: Number of unique bigrams that $W_i$ completes
- **Denominator**: Total number of unique bigrams in the corpus

### 4.3 Full Kneser-Ney Formula

$$P_{\text{KN}}(W_i \mid W_{i-1}) = \frac{\max(C(W_{i-1}, W_i) - D, \ 0)}{C(W_{i-1})} + \lambda(W_{i-1}) \cdot P_{\text{cont}}(W_i)$$

### 4.4 Lambda Calculation

$\lambda(W_{i-1})$ accounts for the total probability mass discounted from all non-zero bigrams:

$$\lambda(W_{i-1}) = \frac{D}{C(W_{i-1})} \cdot |\{W_i : C(W_{i-1}, W_i) > 0\}|$$

- $D$ = discount factor
- Numerator: $D \times$ (number of non-zero bigram types starting with $W_{i-1}$)
- Denominator: unigram count of $W_{i-1}$ (= sum of row in bigram table)

---

## 5. Evaluation of Language Models

### 5.1 Two Approaches

| Type | Method | Pros | Cons |
|------|--------|------|------|
| **Extrinsic** | Apply LM to downstream tasks (MT, ASR, etc.), measure task metric (BLEU, WER) | Real-world relevance | Slow; depends on task quality; which task to choose? |
| **Intrinsic** | Measure LM quality directly using **perplexity** | Fast, task-independent | May not reflect downstream performance |

### 5.2 Extrinsic Evaluation Problems
1. Quality of the downstream model affects evaluation
2. Unclear which downstream task to choose (typically use 10–12 tasks)
3. Downstream tasks are time-consuming (neural models may take days)

---

## 6. Perplexity ⭐

### 6.1 Intuition

> Perplexity measures **how surprised** the model is by the test data, or equivalently, **how many words** the model is choosing between at each step.

- More context → fewer candidates → lower perplexity → **better model**
- "I always order pizza with cheese and ___" — few candidates (low perplexity)
- "I wrote a ___" — many candidates (high perplexity)

### 6.2 Definition

$$PP(W_1, \dots, W_N) = P(W_1, W_2, \dots, W_N)^{-\frac{1}{N}}$$

This is the **inverse geometric mean** of the token probabilities.

For a **bigram model**:

$$PP = \left[\prod_{i=1}^{N} P(W_i \mid W_{i-1})\right]^{-\frac{1}{N}}$$

For a **unigram model**:

$$PP = \left[\prod_{i=1}^{N} P(W_i)\right]^{-\frac{1}{N}}$$

> **Lower perplexity = better language model** (more certain about next token)

---

## 7. Connection: Perplexity ↔ Entropy ↔ Cross-Entropy

### 7.1 Shannon Entropy

$$H(X) = -\sum_{i} P(X_i) \log_2 P(X_i)$$

- Measures **average bits** needed to encode outcomes
- Maximum when all outcomes are equally likely (uniform distribution)

### 7.2 Entropy Rate (Per-Word Entropy)

For a sequence of length $N$:

$$H_{\text{rate}} = \frac{1}{N} H(W_1, \dots, W_N) = -\frac{1}{N} \sum P(W_1 \dots W_N) \log_2 P(W_1 \dots W_N)$$

Generalized via limit:

$$H(L) = \lim_{N \to \infty} -\frac{1}{N} \sum P(W_1 \dots W_N) \log_2 P(W_1 \dots W_N)$$

### 7.3 Shannon-McMillan-Breiman Theorem

If the stochastic process is **stationary and ergodic** (regular):

$$H(L) = \lim_{N \to \infty} -\frac{1}{N} \log_2 P(W_1, W_2, \dots, W_N)$$

> We can estimate entropy from a **single sufficiently long sequence** instead of summing over all possible sequences.

### 7.4 Cross-Entropy

We don't know the true language distribution $P_L$. We approximate with model $P_M$.

$$H(L, M) = \lim_{N \to \infty} -\frac{1}{N} \log_2 P_M(W_1, W_2, \dots, W_N)$$

**Lower bound property**:

$$H(L) \leq H(L, M)$$

**Proof sketch**:

$$H(L) = H(L, M) - D_{KL}(L \| M)$$

Since KL divergence $D_{KL} \geq 0$, we get $H(L) \leq H(L, M)$.

> As the model $M$ gets closer to the true language $L$, cross-entropy approaches true entropy.

### 7.5 Perplexity = 2^(Cross-Entropy)

$$\boxed{PP = 2^{H(L,M)}}$$

**Derivation**:

$$\text{If } H = -\frac{1}{N} \log_2 P(W_1 \dots W_N)$$

$$\text{Then } 2^H = 2^{-\frac{1}{N} \log_2 P(W_1 \dots W_N)} = P(W_1 \dots W_N)^{-\frac{1}{N}} = PP$$

### 7.6 Intuitive Interpretation

| Concept | Meaning |
|---------|---------|
| **Entropy** $H$ | Average number of **bits** to encode next token |
| **Perplexity** $2^H$ | Number of **equally likely words** the model considers at each step |

- Entropy = 3 bits → Perplexity = $2^3 = 8$ (model considers 8 options)
- Entropy = 1 bit → Perplexity = $2^1 = 2$ (model considers 2 options)

> **Why perplexity over entropy?** "This model considers 8 possible next words" is more interpretable than "this model has 3 bits of entropy."

---

## 8. Limitations of Statistical N-gram Models

| Problem | Description |
|---------|-------------|
| **Fixed window** | N must be chosen beforehand; unclear optimal size |
| **Matrix explosion** | As vocabulary grows, N-gram matrix becomes enormous |
| **Sparsity** | Even with smoothing, many entries remain zero or near-zero |
| **OOV handling** | Weak handling of out-of-vocabulary words |
| **Computational cost** | Building bigram/trigram tables over trillion-token corpora is expensive |
| **No long-range dependencies** | Cannot capture relationships beyond the window |

→ This motivates **word embeddings** and **neural language models**.

---

## 9. Smoothing Techniques Summary

| Method | Approach | Key Feature |
|--------|----------|-------------|
| **Add-K (Laplace)** | Add $k$ to all counts | Simple but distorts high counts |
| **Good-Turing** | Use $N_{c+1}/N_c$ to adjust counts | Preserves distribution better |
| **Absolute Discounting** | Subtract fixed $D$ from all non-zero counts | Motivated by Good-Turing observation |
| **Kneser-Ney** | Absolute discounting + continuation probability | Best classical method; handles stop words |
| **Backoff** | Fall back to lower-order N-gram if count = 0 | Simple fallback strategy |
| **Interpolation** | Weighted sum of different N-gram orders | $\lambda$-weighted combination |

---

## 10. Key Formulas Quick Reference

| Formula | Expression |
|---------|-----------|
| Good-Turing count | $c^* = (c+1) \cdot \frac{N_{c+1}}{N_c}$ |
| Continuation probability | $P_{\text{cont}}(W) = \frac{\lvert\{W' : C(W', W) > 0\}\rvert}{\lvert\{(a,b) : C(a,b) > 0\}\rvert}$ |
| Kneser-Ney | $P_{\text{KN}} = \frac{\max(C - D, 0)}{C(W_{i-1})} + \lambda \cdot P_{\text{cont}}$ |
| Perplexity | $PP = P(W_1 \dots W_N)^{-1/N}$ |
| Perplexity-Entropy link | $PP = 2^{H(L,M)}$ |
| Cross-entropy | $H(L,M) = \lim_{N \to \infty} -\frac{1}{N} \log_2 P_M(W_1 \dots W_N)$ |
| Entropy lower bound | $H(L) \leq H(L, M)$ |
