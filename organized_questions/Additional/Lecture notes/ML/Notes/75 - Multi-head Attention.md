# Multi-Head Attention

## Overview

- **Core Idea:** Run **multiple self attention mechanisms in parallel** on the same input to capture **multiple perspectives/interpretations**
- **Each parallel self attention** = one "head"
- Multiple heads = **Multi-Head Attention**

---

## Self Attention Recap

Self attention takes a sentence, generates Q, K, V vectors per word, and produces **contextual embeddings**:

1. Input: word embeddings (static)
2. Linear projection → Q, K, V vectors via $W_Q$, $W_K$, $W_V$
3. Compute similarity scores → softmax → weighted sum
4. Output: contextual embeddings (dynamic, context-aware)

**Limitation:** A single self attention module produces only **one similarity table** — capturing only **one perspective** of the relationships between words.

---

## The Problem with Single Self Attention

### Example: Ambiguous Sentence

> "The man saw the astronomer with a telescope"

**Two valid interpretations:**

| Interpretation | Key Relationship | Meaning |
|---------------|-----------------|---------|
| 1 | man ↔ telescope (high similarity) | The man used a telescope to see the astronomer |
| 2 | astronomer ↔ telescope (high similarity) | The man saw an astronomer who had a telescope |

A single self attention head can only learn **one** of these similarity patterns — it produces one set of attention weights.

### General Problem

- Natural language often has **multiple valid interpretations**
- Document summarization requires understanding **multiple perspectives**
- Single self attention = single perspective = **information loss**

---

## Multi-Head Attention: The Solution

### Core Idea

Instead of one set of weight matrices $(W_Q, W_K, W_V)$, use **multiple sets**:

| Single Self Attention | Multi-Head Attention (h heads) |
|----------------------|-------------------------------|
| 1 set: $W_Q, W_K, W_V$ | $h$ sets: $W_Q^1, W_K^1, W_V^1, \ldots, W_Q^h, W_K^h, W_V^h$ |
| 1 set of Q, K, V vectors per word | $h$ sets of Q, K, V vectors per word |
| 1 contextual embedding per word | $h$ contextual embeddings per word |

Each set of weight matrices learns **different attention patterns** → different perspectives.

---

## Architecture Step by Step

### Step 1: Linear Projection (Multiple Sets)

For each head $i$ (from 1 to $h$):

$$Q_i = E \cdot W_Q^i, \quad K_i = E \cdot W_K^i, \quad V_i = E \cdot W_V^i$$

Where $E$ is the embedding matrix (all words stacked).

### Step 2: Parallel Self Attention

Each head independently computes:

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

### Step 3: Concatenate All Heads

$$Z_{\text{concat}} = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)$$

### Step 4: Final Linear Projection

$$\boxed{\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O}$$

Where $W_O$ is a learned weight matrix that:
- Combines/balances all perspectives
- Projects back to the original embedding dimension

---

## Dimensions in the Original Transformer

| Component | Shape | Value in Paper |
|-----------|-------|----------------|
| Embedding dimension ($d_{\text{model}}$) | — | 512 |
| Number of heads ($h$) | — | 8 |
| Dimension per head ($d_k = d_v$) | $d_{\text{model}} / h$ | $512 / 8 = 64$ |
| Each $W_Q^i, W_K^i, W_V^i$ | $d_{\text{model}} \times d_k$ | $512 \times 64$ |
| Each Q, K, V per head | $n \times d_k$ | $n \times 64$ |
| Concatenated output | $n \times (h \cdot d_k)$ | $n \times 512$ |
| $W_O$ | $(h \cdot d_k) \times d_{\text{model}}$ | $512 \times 512$ |
| Final output | $n \times d_{\text{model}}$ | $n \times 512$ |

> $n$ = number of words/tokens in the sentence

---

## Why Reduce Dimension Per Head?

### Computational Efficiency

Instead of running 8 self attention heads at the full 512 dimension (8× the compute), each head operates on a **reduced** 64-dimensional subspace:

$$d_k = \frac{d_{\text{model}}}{h} = \frac{512}{8} = 64$$

**Result:**

| Approach | Total Computation |
|----------|------------------|
| 1 self attention head at 512D | Baseline |
| 8 self attention heads at 64D each | **Same** as baseline |

> You get **8 different perspectives** for the **same computational cost** as a single full-dimensional self attention.

---

## Visualization: Capturing Different Perspectives

Using the sentence "The man saw the astronomer with a telescope":

| Head | What It Captures | Key Similarity |
|------|-----------------|----------------|
| Head 0 (Layer 0) | Man used the telescope | man ↔ telescope (strong) |
| Head 1 (Layer 0) | Astronomer had the telescope | astronomer ↔ telescope (strong), man ↔ astronomer (strong) |

Different heads learn to attend to **different relationships** in the same sentence.

---

## Role of $W_O$ (Output Projection)

The output projection matrix $W_O$ serves to:

1. **Combine** the multiple perspectives from all heads
2. **Learn the importance** of each perspective
3. Create a **balanced mixture** of perspectives
4. Project back to the original dimension ($d_{\text{model}}$)

> $W_O$ is a learned parameter matrix — trained via backpropagation to find the optimal balance of perspectives.

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Why multi-head?** | Single self attention captures only one perspective; multiple heads capture multiple perspectives |
| **What is a head?** | One independent self attention module with its own $W_Q, W_K, W_V$ |
| **How many heads?** | 8 in the original Transformer paper |
| **Dimension per head** | $d_k = d_{\text{model}} / h$ (512/8 = 64) |
| **Computation cost** | Same as single full-dimension self attention (due to dimension reduction per head) |
| **Final formula** | $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O$ |
| **Output shape** | Same as input shape ($n \times d_{\text{model}}$) |
| **$W_O$ role** | Balances and mixes perspectives, projects to original dimension |
