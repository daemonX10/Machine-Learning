# What is Self Attention

## The Core NLP Challenge: Words → Numbers

Any NLP application (sentiment analysis, NER, machine translation) requires converting **words into numerical representations** that machines can process.

### Evolution of Text Vectorization

| Technique | Approach | Limitation |
|-----------|----------|------------|
| **One-Hot Encoding** | Binary vector per word (1 at word's index, 0 elsewhere) | High dimensionality, no semantic meaning |
| **Bag of Words** | Count of each word in vocabulary | Loses word order, no semantics |
| **TF-IDF** | Weighted word frequency | Still no semantic understanding |
| **Word Embeddings** | Dense $n$-dimensional vector capturing semantic meaning | **Static** — same vector regardless of context |

---

## Word Embeddings: Strengths

### How They Are Created

1. Take a large corpus (e.g., all Wikipedia articles)
2. Train a neural network to understand word contexts
3. Each word is mapped to an $n$-dimensional vector ($n$ = 64, 256, 512, etc.)

### Key Property: Semantic Similarity

- **Similar words → similar vectors** (close in high-dimensional space)
- **Dissimilar words → different vectors** (far apart)

**Example** (hypothetical 5D vectors):

| Word | Vector | Notes |
|------|--------|-------|
| King | [6, 2, 1, 0, 9] | High royalty, high human |
| Queen | [5, 1, 1, 0, 8] | Similar to King |
| Cricketer | [0, 9, 1, 0, 2] | High athletics, low royalty |

### What Each Dimension Captures

Each dimension encodes some **aspect** of meaning (learned automatically, not human-interpretable):
- Dimension 1 might capture "royalty"
- Dimension 2 might capture "athleticism"
- Dimension 3 might capture "is human"

---

## The Problem: Static Embeddings

Word embeddings are trained **once** and reused everywhere — they are **static**.

### Why This Is a Problem

The embedding captures the **average meaning** across all training data, not the meaning in a specific context.

**Example: The word "Apple"**

If the training data has:
- 9,000 sentences about Apple as a **fruit**
- 1,000 sentences about Apple as a **tech company**

The resulting embedding will be **heavily biased toward the fruit meaning**:

| Dimension | Taste | Technology |
|-----------|-------|------------|
| Value | ~9.0 (high) | ~0.3 (low) |

### Failure Case

Given the sentence: *"Apple launched a new phone while I was eating an orange"*

- "Apple" here clearly means the **tech company**
- But the static embedding says it's mostly a **fruit** (value ≈ [9.0, 0.3])
- The model gets the wrong representation

> Static embeddings use the **same vector** for a word regardless of how it's used in the current sentence.

---

## The Solution: Contextual Embeddings

**What we need:** Embeddings that **dynamically change** based on the surrounding words in the current sentence.

| Static Embedding | Contextual Embedding |
|--|--|
| Same vector for "bank" everywhere | Different vector for "money bank" vs "river bank" |
| Trained once, used as-is | Adapts based on surrounding words |
| Captures average meaning | Captures context-specific meaning |

### Ideal Behavior

For *"Apple launched a new phone while I was eating an orange"*:
- See "launched", "phone" in context → **increase** the technology component
- **Decrease** the taste/fruit component
- Smart enough to **not be confused by "orange"** (which is a fruit but used in a different clause)

---

## Self Attention: The Mechanism

**Self attention is a mechanism that converts static word embeddings into smart contextual embeddings.**

### How It Works (High Level)

```
Input: Static embeddings for each word in sentence
       [e₁, e₂, e₃, ..., eₙ]
            ↓
    ┌───────────────────┐
    │  Self Attention    │
    │  Block             │
    │  (calculations)    │
    └───────────────────┘
            ↓
Output: Contextual embeddings for each word
       [y₁, y₂, y₃, ..., yₙ]
```

- For **every** input static embedding → one output contextual embedding
- The output embeddings **understand context** — they know how each word is being used in the current sentence
- These contextual embeddings are then fed into downstream architectures like Transformers

---

## One-Line Summary

> **Self attention is a mechanism that takes static embeddings as input and generates contextual embeddings that are aware of how each word is used in its surrounding context — making them far more useful for any NLP application.**

---

## What's Next

The next step is understanding **what happens inside** the self-attention block:
- **Query, Key, Value vectors** — what they are
- How self-attention uses them to create contextual embeddings
- The mathematical operations involved
