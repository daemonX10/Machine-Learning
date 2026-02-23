# Self Attention vs Luong Attention

## Overview

- **Core Question:** Why is self attention called "self"? And why is it called "attention" at all?
- **Key Insight:** Self attention performs the **same mathematical operations** as Luong/Bahdanau attention, but on a **single sequence** instead of between two different sequences

---

## Recap: Luong Attention in Encoder-Decoder

### The Problem Attention Solved

In a basic encoder-decoder (seq2seq) for translation:

1. Encoder processes input sentence word-by-word, maintaining hidden states $h_0, h_1, \ldots, h_n$
2. The final hidden state $h_n$ becomes the **context vector** — a summary of the entire input
3. Decoder uses this single context vector to generate output word-by-word

**Problem:** For sentences > 30 words, a single context vector **cannot capture** all the information → translation quality degrades.

### Attention Mechanism Solution

Instead of one context vector for the entire sentence, generate a **different context vector for each decoder time step**:

$$c_i = \sum_j \alpha_{ij} \cdot h_j$$

Where:
- $c_i$ = context vector for decoder time step $i$
- $h_j$ = encoder hidden state at position $j$
- $\alpha_{ij}$ = attention weight (how much input word $j$ matters for output word $i$)

### How Attention Weights Are Computed (Luong)

**Three key equations:**

| Step | Formula | Description |
|------|---------|-------------|
| 1. Alignment scores | $e_{ij} = s_i \cdot h_j$ | Dot product of decoder hidden state $s_i$ with encoder hidden state $h_j$ |
| 2. Attention weights | $\alpha_{ij} = \text{softmax}(e_{ij})$ | Normalize scores to get probabilities |
| 3. Context vector | $c_i = \sum_j \alpha_{ij} \cdot h_j$ | Weighted sum of encoder hidden states |

---

## Comparing Self Attention with Luong Attention

### Self Attention Setup

Given a sentence "Turn off the lights":

1. Generate embedding vectors for each word
2. Create three new vectors per word: **Query** ($q$), **Key** ($k$), **Value** ($v$)
3. Compute contextual embeddings for every word

### Computing Contextual Embedding (e.g., for "Turn")

$$y_{\text{turn}} = w_{11} \cdot v_{\text{turn}} + w_{12} \cdot v_{\text{off}} + w_{13} \cdot v_{\text{the}} + w_{14} \cdot v_{\text{lights}}$$

Where weights come from:

$$w_{1j} = \text{softmax}(s_{1j})$$

And similarity scores:

$$s_{1j} = q_{\text{turn}} \cdot k_j \quad \text{(dot product of query with each key)}$$

### Side-by-Side Comparison

| Aspect | Luong Attention | Self Attention |
|--------|----------------|----------------|
| **Query** | Decoder hidden state $s_i$ | Query vector $q_i$ |
| **Key** | Encoder hidden states $h_j$ | Key vectors $k_j$ |
| **Value** | Encoder hidden states $h_j$ | Value vectors $v_j$ |
| **Alignment/Similarity** | $e_{ij} = s_i \cdot h_j$ | $s_{ij} = q_i \cdot k_j$ |
| **Normalization** | $\alpha_{ij} = \text{softmax}(e_{ij})$ | $w_{ij} = \text{softmax}(s_{ij})$ |
| **Output** | $c_i = \sum_j \alpha_{ij} \cdot h_j$ | $y_i = \sum_j w_{ij} \cdot v_j$ |

> The **same three operations** happen in both: compute similarity → normalize with softmax → weighted sum

---

## Why "Attention"?

Self attention is called an attention mechanism because **the mathematical formulation is identical** to Luong/Bahdanau attention:

1. Compute alignment/similarity scores (dot product)
2. Normalize with softmax to get weights
3. Compute weighted sum to get output

Despite looking architecturally different (no encoder/decoder), the core math is the same — the same three equations are reused in a different setting.

---

## Why "Self"?

| Attention Type | Operates Between | Type |
|---------------|-----------------|------|
| Luong / Bahdanau | Two **different** sequences (e.g., English → Hindi) | **Inter-sequence** attention |
| Self Attention | The **same** sequence with itself | **Intra-sequence** attention |

In Luong attention, you compute alignment between an input sequence and an output sequence — **two different sequences**.

In self attention, the query sequence and the key sequence are the **same sentence**. You compute how each word in a sentence relates to every other word **in that same sentence**.

> **Self attention = attention applied within a single sequence (to itself)**
> Hence the name "self."

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Why "attention"?** | Uses the same three-step math as Luong/Bahdanau: similarity → softmax → weighted sum |
| **Why "self"?** | Attention is computed within the same sequence (intra-sequence), not between two different sequences (inter-sequence) |
| **Luong/Bahdanau** | Inter-sequence: encoder states ↔ decoder states |
| **Self Attention** | Intra-sequence: same sentence's Q, K, V vectors interact with each other |
| **Key formula** | $y_i = \sum_j \text{softmax}(q_i \cdot k_j) \cdot v_j$ |
