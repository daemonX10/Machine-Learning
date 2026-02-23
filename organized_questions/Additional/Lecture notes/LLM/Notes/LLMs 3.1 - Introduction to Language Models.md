# Lecture 3.1 — Introduction to Language Models

---

## 1. Origins: Shannon's Character Guessing Game (1950s)

- Claude Shannon played a **character guessing game** with his wife
- Given a sequence of characters, predict the next character (27 options: 26 letters + space)
- On average, she guessed correctly in **3–4 attempts**
- This relates to the **entropy of English**: $H = -\sum P(x) \log_2 P(x)$
- If all 27 characters are equally likely: $H = \log_2 27 \approx 4.75$ bits
- Connection to **perplexity** (covered in Lecture 3.2)

---

## 2. Language Model Definition

> A **language model** is a probability distribution over a sequence of tokens.

$$P(W_1, W_2, \dots, W_n)$$

Using the **chain rule**:

$$P(W_1, \dots, W_n) = P(W_1) \cdot P(W_2 \mid W_1) \cdot P(W_3 \mid W_1, W_2) \cdots P(W_n \mid W_1, \dots, W_{n-1})$$

Each term is a **conditional probability** given the **context** (all preceding tokens).

---

## 3. Applications of Language Models

| Application | How LM Helps |
|-------------|-------------|
| **Autocomplete** | Predict most likely next word (Google Search) |
| **Speech Recognition** | "I bought fresh mangoes from the market" vs. "I bot frsh man goes from da mar kit" — LM ranks correct transcription higher |
| **Machine Translation** | "heavy rainfall" vs. "big rainfall" — LM prefers idiomatic usage |
| **Spell Checking** | Rank corrections by probability |
| **Text Generation** | ChatGPT, text completion |

> Language models capture not just grammar, but also **world knowledge**.

---

## 4. N-gram Language Models

### 4.1 The Sparsity Problem

Computing $P(W_n \mid W_1, \dots, W_{n-1})$ requires counting the full context in the corpus. For long contexts, the count is almost certainly **zero**.

### 4.2 Markov Assumption

Restrict the context to only the **previous $k$ words**:

| Order | Name | Assumption | Formula |
|-------|------|-----------|---------|
| 0 | **Unigram** | Words are independent | $P(W_i)$ |
| 1 | **Bigram** | Depends on previous 1 word | $P(W_i \mid W_{i-1})$ |
| 2 | **Trigram** | Depends on previous 2 words | $P(W_i \mid W_{i-2}, W_{i-1})$ |
| $n-1$ | **N-gram** | Depends on previous $n-1$ words | $P(W_i \mid W_{i-n+1}, \dots, W_{i-1})$ |

> **N-gram model** = **(N−1)-order Markov assumption**

### 4.3 The Trade-off

| Direction | More Context (higher N) | Less Context (lower N) |
|-----------|------------------------|----------------------|
| Grammar quality | ✓ Better | ✗ Meaningless |
| Probability estimation | ✗ Often zero | ✓ Non-zero |

**Practical choice**: Trigram or 4-gram models are a good balance.

---

## 5. Bigram Model — Detailed

### 5.1 Maximum Likelihood Estimation (MLE)

$$P(W_i \mid W_{i-1}) = \frac{C(W_{i-1}, W_i)}{C(W_{i-1})}$$

Where:
- $C(W_{i-1}, W_i)$ = bigram count (how many times the pair appears)
- $C(W_{i-1})$ = unigram count of the preceding word

### 5.2 Bigram Count Table

- **Rows**: vocabulary words (as $W_{i-1}$)
- **Columns**: vocabulary words (as $W_i$)
- **Entry**: count of bigram $(W_{i-1}, W_i)$
- Matrix is **asymmetric** (order matters: "I want" ≠ "want I")

> **Key property**: Row sum for word $w$ = Column sum for word $w$
> (because every word appears as both $W_{i-1}$ and $W_i$ in the corpus, accounting for `<s>` and `</s>` tokens)

### 5.3 Bigram Probability Table

Divide each bigram count by the unigram count of the row word:

$$P(W_i \mid W_{i-1}) = \frac{C(W_{i-1}, W_i)}{C(W_{i-1})}$$

### 5.4 Sparsity Issue

In a real corpus (e.g., Berkeley Restaurant Project):
- **99.95%** of bigram matrix entries are **zero**

Types of zeros:

| Type | Description | Example |
|------|-------------|---------|
| **Structural zero** | Grammatically impossible | "eat want" (two verbs) |
| **Contingent zero** | Valid but absent from corpus | "enjoyed the festival" |

The contingent zeros are the problem — valid phrases incorrectly assigned zero probability.

---

## 6. Long-Range Dependency Problem

> "The project, which he had been working on for months, was finally **approved** by the committee."

- "approved" relates to "project" — but they are 10+ words apart
- A bigram/trigram model cannot capture this dependency
- Solution: RNNs (LSTM, GRU) and later Transformers

---

## 7. Out-of-Vocabulary (OOV) Words

**Problem**: Test set words not in training vocabulary → no entry in bigram table.

**Solution**: 
1. Sort all vocabulary words by frequency
2. Set a frequency threshold (e.g., 5)
3. Words below threshold → grouped into a special `<UNK>` token
4. Build bigram table with lexicon entries + `<UNK>`
5. At test time, unknown words use `<UNK>` probabilities

> **Lexicon** = subset of vocabulary with frequency above threshold  
> **Vocabulary − Lexicon** → acts as proxy for `<UNK>`

---

## 8. Smoothing — Add-One (Laplace) Smoothing

### Problem
Zero counts → zero probability → entire sequence probability becomes zero.

### Add-One Smoothing

Add 1 to **every** bigram count:

$$P_{\text{add-1}}(W_i \mid W_{i-1}) = \frac{C(W_{i-1}, W_i) + 1}{C(W_{i-1}) + |V|}$$

Where $|V|$ = vocabulary size (to ensure probabilities sum to 1).

### Generalization: Add-K Smoothing

$$P_{\text{add-k}}(W_i \mid W_{i-1}) = \frac{C(W_{i-1}, W_i) + k}{C(W_{i-1}) + k \cdot |V|}$$

### Reconstituted (Adjusted) Count

$$C^*(W_{i-1}, W_i) = \frac{(C(W_{i-1}, W_i) + 1) \cdot C(W_{i-1})}{C(W_{i-1}) + |V|}$$

### Problem with Add-One Smoothing

| Original Count | Adjusted Count | Change |
|---------------|---------------|--------|
| 827 | 527 | **−36%** |
| 608 | 238 | **−61%** |
| 686 | 430 | **−37%** |
| 5 | 3.8 | −24% |
| 0 | >0 | ✓ Fixed |

> "Steals from the rich and distributes to the poor" — high-count bigrams are **drastically reduced** while zero-counts get only tiny values.

### Improvement: Replace Uniform Prior with Unigram Prior

Instead of adding uniform $\frac{1}{|V|}$, add unigram probability $P(W_i)$:

$$P_{\text{smooth}}(W_i \mid W_{i-1}) = \frac{C(W_{i-1}, W_i) + m \cdot P(W_i)}{C(W_{i-1}) + m}$$

- $m$ is a hyperparameter tuned on validation data

---

## 9. Backoff and Interpolation

### Backoff
If trigram count is zero → fall back to bigram → fall back to unigram.

### Interpolation
Linear combination of different N-gram probabilities:

$$\hat{P}(W_n \mid W_{n-2}, W_{n-1}) = \lambda_1 P_{\text{tri}}(W_n \mid W_{n-2}, W_{n-1}) + \lambda_2 P_{\text{bi}}(W_n \mid W_{n-1}) + \lambda_3 P_{\text{uni}}(W_n)$$

**Constraints**: $\lambda_1 + \lambda_2 + \lambda_3 = 1$

> $\hat{P}$ is an estimate, not a true probability.

**Context-dependent interpolation**: $\lambda$ values become functions of the context rather than fixed constants.

---

## 10. Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Chain rule | $P(W_1 \dots W_n) = \prod_{i=1}^{n} P(W_i \mid W_1 \dots W_{i-1})$ |
| Bigram MLE | $P(W_i \mid W_{i-1}) = \frac{C(W_{i-1}, W_i)}{C(W_{i-1})}$ |
| Add-k smoothing | $\frac{C(W_{i-1}, W_i) + k}{C(W_{i-1}) + k|V|}$ |
| Interpolation | $\hat{P} = \lambda_1 P_{\text{tri}} + \lambda_2 P_{\text{bi}} + \lambda_3 P_{\text{uni}}$ |
