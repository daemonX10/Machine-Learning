# Self Attention — Detailed Mechanism

## Recap: The Goal

- **Static embeddings** don't capture context ("bank" is the same in "money bank" and "river bank")
- **Self attention** converts static embeddings → **dynamic contextual embeddings**
- This video covers the **exact calculations** inside the self-attention block

---

## Step 1: First Principles — Simple Self Attention (No Learning)

### Core Idea

Represent each word as a **weighted sum of all words' embeddings** in the sentence:

$$\text{new\_embedding}(\text{bank}) = \alpha_1 \cdot e(\text{money}) + \alpha_2 \cdot e(\text{bank}) + \alpha_3 \cdot e(\text{grows})$$

Where $\alpha_i$ are **similarity scores** between the target word and every word in the sentence.

### Why This Helps

| Sentence 1 | Sentence 2 |
|------------|------------|
| **money** bank grows | **river** bank flows |
| bank = $\alpha_1 \cdot e(\text{money}) + \alpha_2 \cdot e(\text{bank}) + \alpha_3 \cdot e(\text{grows})$ | bank = $\beta_1 \cdot e(\text{river}) + \beta_2 \cdot e(\text{bank}) + \beta_3 \cdot e(\text{flows})$ |
| Context pulled toward finance | Context pulled toward geography |

Even though "bank" is the same word, the contextual embedding is **different** because the surrounding words are different.

### Computing Similarity: Dot Product

Two vectors' dot product measures their similarity:
- **High dot product** → similar vectors
- **Low dot product** → dissimilar vectors

For a sentence with words $w_1, w_2, \ldots, w_n$:

$$s_{ij} = e(w_i)^T \cdot e(w_j) \quad \text{(raw similarity score)}$$

### Normalization: Softmax

Raw scores can be any value (positive, negative, large, small). Apply **softmax** to normalize:

$$w_{ij} = \frac{\exp(s_{ij})}{\sum_{k=1}^{n} \exp(s_{ik})}$$

Properties after softmax:
- All weights are **positive**
- All weights **sum to 1** (interpretable as probabilities)

### Final Contextual Embedding

$$y_i = \sum_{j=1}^{n} w_{ij} \cdot e(w_j)$$

This is the **weighted sum** of all word embeddings, where weights represent how relevant each word is.

---

## Step 2: Matrix Form (Parallelization)

### The Three Operations

All of the above can be done in **one parallel computation** using matrix operations:

**Given:** Embedding matrix $X$ of shape $(n \times d)$ where $n$ = number of words, $d$ = embedding dimension

$$S = X \cdot X^T \quad \text{(similarity matrix, shape } n \times n\text{)}$$

$$W = \text{softmax}(S) \quad \text{(weight matrix, shape } n \times n\text{)}$$

$$Y = W \cdot X \quad \text{(contextual embeddings, shape } n \times d\text{)}$$

### Why Parallelization Matters

| Property | Sequential (RNNs) | Parallel (Self Attention) |
|----------|-------------------|--------------------------|
| Processing | Word-by-word | All words simultaneously |
| Hardware utilization | Poor GPU usage | Full GPU parallelism |
| Speed | Slow for long sequences | Fast regardless of length |

> **Trade-off:** Parallelization loses **sequence order** information (position of words). This is addressed separately by positional encodings.

---

## Step 3: The Problem — No Learnable Parameters

### What's Missing

The simple self-attention has **three operations**, none with trainable parameters:

| Step | Operation | Learnable? |
|------|-----------|-----------|
| 1 | Dot product $X \cdot X^T$ | ❌ No |
| 2 | Softmax | ❌ No |
| 3 | Weighted sum $W \cdot X$ | ❌ No |

### Consequence: General vs Task-Specific Embeddings

Without learnable parameters, the model produces **general contextual embeddings** that don't adapt to the task.

**Example: "Piece of cake"**

| Task: Translation | General Embedding | Task-Specific (Ideal) |
|---|---|---|
| English → Hindi | "केक का टुकड़ा" (literal: piece of cake) | "बहुत आसान काम" (idiomatic: very easy task) |

The general embedding doesn't understand that "piece of cake" is an **idiom** in the training data's context. A task-specific embedding trained on translation data **would** learn this.

> **We need learnable parameters** so the model can learn from data and generate **task-specific** contextual embeddings.

---

## Step 4: Query, Key, Value — Three Roles

### Observation

In the simple model, each embedding vector plays **three different roles**:

1. **Query (Green):** Asks "how similar am I to every other word?" — used to compute similarity scores
2. **Key (Pink):** Responds to queries — provides information for similarity computation
3. **Value (Blue):** Contributes to the final weighted sum

### The Problem with One Vector for Three Roles

Using the **same** vector for all three roles is like a person using their **entire autobiography** for:
- Their dating profile (should be curated highlights)
- Their search preferences (should be specific criteria)
- Their in-person presentation (should show best qualities)

Each role requires **different, optimized information** from the same source.

### Solution: Separate Vectors via Linear Transformation

For each word embedding $e_i$, create three separate vectors:

$$q_i = W_q \cdot e_i \quad \text{(query vector)}$$

$$k_i = W_k \cdot e_i \quad \text{(key vector)}$$

$$v_i = W_v \cdot e_i \quad \text{(value vector)}$$

Where $W_q$, $W_k$, $W_v$ are **learnable weight matrices** (same for all words in the sentence).

### Why This Is Better (Separation of Concerns)

| Vector | Role | Optimized For |
|--------|------|---------------|
| $q_i$ | Query | Asking the right questions (what context do I need?) |
| $k_i$ | Key | Answering queries (what information do I provide?) |
| $v_i$ | Value | Contributing to the final representation |

Each vector is a **different linear transformation** of the same embedding, specialized for its purpose.

### The Weight Matrices

| Matrix | Shape | Role | How It's Learned |
|--------|-------|------|-----------------|
| $W_q$ | $(d \times d_q)$ | Generates query vectors | Backpropagation |
| $W_k$ | $(d \times d_k)$ | Generates key vectors | Backpropagation |
| $W_v$ | $(d \times d_v)$ | Generates value vectors | Backpropagation |

- Initialized with **random values**
- Updated via **backpropagation** during training
- **Same matrices** are used for every word in the sentence (shared weights)
- After training, they learn to extract the best Q, K, V for the given task

---

## Step 5: Full Self-Attention Mechanism

### Per-Word Computation

For target word $i$ in a sentence of $n$ words:

**1. Compute similarity scores:**

$$s_{ij} = q_i^T \cdot k_j \quad \forall j \in \{1, \ldots, n\}$$

**2. Normalize with softmax:**

$$w_{ij} = \text{softmax}(s_{ij}) = \frac{\exp(s_{ij})}{\sum_{k=1}^{n} \exp(s_{ik})}$$

**3. Weighted sum of value vectors:**

$$y_i = \sum_{j=1}^{n} w_{ij} \cdot v_j$$

### Matrix Form (Parallel for All Words)

Given embedding matrix $X$ of shape $(n \times d)$:

**Generate Q, K, V matrices:**

$$Q = X \cdot W_q \qquad K = X \cdot W_k \qquad V = X \cdot W_v$$

**Compute attention:**

$$S = Q \cdot K^T \quad \text{(similarity scores, } n \times n\text{)}$$

$$W = \text{softmax}(S) \quad \text{(attention weights, } n \times n\text{)}$$

$$Y = W \cdot V \quad \text{(contextual embeddings, } n \times d_v\text{)}$$

### Complete Self-Attention Formula

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(Q \cdot K^T\right) \cdot V}$$

> Note: The scaled version (used in Transformers) divides by $\sqrt{d_k}$:
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$

---

## Summary: Building Self-Attention from First Principles

| Step | What We Did | Key Insight |
|------|-------------|-------------|
| 1 | Represent each word as weighted sum of all words | Context = combination of surrounding words |
| 2 | Use dot product for similarity | Simple, effective similarity measure |
| 3 | Normalize with softmax | Weights sum to 1, all positive |
| 4 | Parallelize via matrix operations | GPU-friendly, handles any sentence length |
| 5 | Add learnable W_q, W_k, W_v | Task-specific embeddings via backpropagation |
| 6 | Separate Q, K, V roles | Better embeddings through separation of concerns |

### Properties of Self-Attention

| Property | Detail |
|----------|--------|
| **Input** | Static word embeddings (from Word2Vec, GloVe, etc.) |
| **Output** | Task-specific contextual embeddings |
| **Learnable params** | Three weight matrices $W_q$, $W_k$, $W_v$ |
| **Parallelizable** | Yes — all words processed simultaneously |
| **Limitation** | No inherent sense of word order (needs positional encoding) |
| **Significance** | Core mechanism of the Transformer architecture |
