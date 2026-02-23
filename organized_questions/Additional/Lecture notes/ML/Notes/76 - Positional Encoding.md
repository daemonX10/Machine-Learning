# Positional Encoding in Transformers

## Overview

- **Core Problem:** Self attention processes all words **in parallel** — it has no mechanism to understand **word order**
- **Solution:** Positional encoding injects position information into the embeddings before feeding them to the self attention block

---

## Why Positional Encoding Is Needed

### Self Attention's Two Benefits

| Benefit | Description |
|---------|-------------|
| **Contextual embeddings** | Dynamic representations that change based on context |
| **Parallel computation** | All words processed simultaneously (unlike sequential RNNs) |

### The Parallelism Trade-off

Because all words are fed simultaneously, self attention **cannot distinguish word order**:

> For self attention: "Nitish killed lion" = "Lion killed Nitish"

Both sentences have the same three words — self attention sees no difference.

### Contrast with RNNs

RNNs are inherently sequential:
- Time step 1: "Nitish" → Time step 2: "killed" → Time step 3: "lion"
- Order is **naturally captured** through the sequential processing
- But RNNs are **slow** (can't parallelize) and struggle with long sequences

---

## Approach 1: Simple Counting (Naive)

Assign integer positions: word 1 → 1, word 2 → 2, ..., word $n$ → $n$

Append the position number as an extra dimension to each embedding.

### Problems with Simple Counting

| Problem | Explanation |
|---------|-------------|
| **Unbounded values** | For a 100,000-word document, position = 100,000. Large numbers cause **unstable gradients** in neural networks (vanishing/exploding) |
| **Normalization doesn't work** | Dividing by total words makes position values inconsistent across sentences of different lengths. Position 2 = 1.0 in a 2-word sentence but 0.5 in a 4-word sentence |
| **Discrete values** | Integer positions (1, 2, 3, 4) provide no smooth transitions. Neural networks prefer **continuous** values |
| **No relative position** | Cannot capture **distance** between positions. The model can't easily learn that position 3 is "2 steps after" position 1 |

---

## Approach 2: Single Sine Curve

Use $y = \sin(\text{pos})$ to encode positions.

### What It Solves

| Previously | Now (with sine) |
|-----------|-----------------|
| Unbounded (1 to 100,000) | **Bounded** (−1 to 1) ✓ |
| Discrete (1, 2, 3, ...) | **Continuous** (smooth curve) ✓ |
| No relative position info | **Periodic** → can capture relative positions ✓ |

### Remaining Problem

Sine is **periodic** — values **repeat** after one full cycle.

> Two different positions (e.g., position 3 and position 35) could get the **same encoded value** → model can't distinguish them.

---

## Approach 3: Sine + Cosine Pair

Use **two** trigonometric functions for each position:

$$y_1 = \sin(\text{pos}), \quad y_2 = \cos(\text{pos})$$

Now each position is a **2D vector** instead of a scalar → reduces collision probability.

### But Still Not Enough

For very long sequences, two values can still coincide. So we **extend** the idea:

---

## Approach 4: Multiple Sine-Cosine Pairs (Final Solution)

Add more pairs with **decreasing frequencies**:

| Pair | Sine Component | Cosine Component |
|------|---------------|-----------------|
| 1 | $\sin(\text{pos})$ | $\cos(\text{pos})$ |
| 2 | $\sin(\text{pos}/2)$ | $\cos(\text{pos}/2)$ |
| 3 | $\sin(\text{pos}/3)$ | $\cos(\text{pos}/3)$ |
| ... | ... | ... |

Each additional pair:
- Adds 2 more dimensions to the positional encoding vector
- Uses a **lower frequency** → repeats more slowly
- Makes collisions between different positions **exponentially less likely**

---

## The Actual Formula (from "Attention Is All You Need")

$$\boxed{PE_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)}$$

$$\boxed{PE_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)}$$

### Variable Definitions

| Variable | Meaning |
|----------|---------|
| $\text{pos}$ | Position of the word in the sentence (0-indexed) |
| $i$ | Dimension index pair: $i = 0, 1, 2, \ldots, d_{\text{model}}/2 - 1$ |
| $d_{\text{model}}$ | Embedding dimension (512 in original paper) |
| $2i$ | Even dimensions → use **sine** |
| $2i+1$ | Odd dimensions → use **cosine** |

### Key Properties of the Formula

- As $i$ increases, $10000^{2i/d_{\text{model}}}$ grows → the **frequency decreases**
- Lower dimensions ($i$ small) → **high frequency** (changes rapidly across positions)
- Higher dimensions ($i$ large) → **low frequency** (changes slowly across positions)

---

## How to Compute Positional Encoding: Worked Example

**Setup:** Sentence = "river bank", embedding dimension = 6, so $d_{\text{model}} = 6$

$i$ ranges from 0 to $d_{\text{model}}/2 - 1 = 2$

### For "river" (pos = 0):

| Dimension | Formula | Value |
|-----------|---------|-------|
| 0 ($2i$, $i=0$) | $\sin(0 / 10000^{0/6})$ | $\sin(0) = 0$ |
| 1 ($2i+1$, $i=0$) | $\cos(0 / 10000^{0/6})$ | $\cos(0) = 1$ |
| 2 ($2i$, $i=1$) | $\sin(0 / 10000^{2/6})$ | $\sin(0) = 0$ |
| 3 ($2i+1$, $i=1$) | $\cos(0 / 10000^{2/6})$ | $\cos(0) = 1$ |
| 4 ($2i$, $i=2$) | $\sin(0 / 10000^{4/6})$ | $\sin(0) = 0$ |
| 5 ($2i+1$, $i=2$) | $\cos(0 / 10000^{4/6})$ | $\cos(0) = 1$ |

**PE("river")** = $[0, 1, 0, 1, 0, 1]$

### For "bank" (pos = 1):

| Dimension | Formula | Value |
|-----------|---------|-------|
| 0 ($2i$, $i=0$) | $\sin(1 / 10000^{0/6})$ | $\sin(1) \approx 0.841$ |
| 1 ($2i+1$, $i=0$) | $\cos(1 / 10000^{0/6})$ | $\cos(1) \approx 0.540$ |
| 2 ($2i$, $i=1$) | $\sin(1 / 10000^{2/6})$ | $\sin(1/21.54) \approx 0.046$ |
| 3 ($2i+1$, $i=1$) | $\cos(1 / 10000^{2/6})$ | $\cos(1/21.54) \approx 0.999$ |
| 4 ($2i$, $i=2$) | $\sin(1 / 10000^{4/6})$ | $\sin(1/464.16) \approx 0.002$ |
| 5 ($2i+1$, $i=2$) | $\cos(1 / 10000^{4/6})$ | $\cos(1/464.16) \approx 1.000$ |

> Notice how lower dimensions change rapidly (0.841 vs 0) while higher dimensions barely change (0.002 vs 0) — this is the **varying frequency** effect.

---

## How Positional Encoding Is Added

Positional encoding is **added** (not concatenated) to the embedding:

$$\text{Input to Self Attention} = \text{Embedding} + \text{Positional Encoding}$$

### Why Addition, Not Concatenation?

| Method | Result | Problem |
|--------|--------|---------|
| **Concatenation** | Dimension doubles ($d_{\text{model}} \to 2 \cdot d_{\text{model}}$) | Doubles parameters → doubles training time |
| **Addition** | Dimension stays the same ($d_{\text{model}}$) | No extra cost ✓ |

> The positional encoding vector must have the **same dimension** as the embedding vector for element-wise addition.

---

## Heatmap Visualization

For 50 words with 128-dimensional embeddings:

- **Each row** = one word's positional encoding vector (128 values)
- **Lower dimensions** (left side) → high frequency → values change rapidly across words
- **Higher dimensions** (right side) → low frequency → values change slowly across words

### Key Observations

1. **First word (pos=0):** Alternating pattern of 0 and 1 (since $\sin(0)=0$, $\cos(0)=1$)
2. **Low-index dimensions:** Show high variation (many color changes across rows)
3. **High-index dimensions:** Show minimal variation (mostly uniform color) for small sentence lengths
4. For longer sentences, even higher dimensions start varying

---

## Analogy: Binary Encoding in Continuous Domain

| Binary Encoding | Positional Encoding |
|----------------|-------------------|
| Least significant bit flips every number | Lowest dimension (highest frequency) changes every position |
| Next bit flips every 2 numbers | Next dimension changes every ~2 positions |
| Higher bits flip less frequently | Higher dimensions change very slowly |
| **Discrete** (0 or 1) | **Continuous** (−1 to 1) |

> Positional encoding = **binary encoding in the domain of continuous numbers**, using sine/cosine curves instead of 0/1 bits.

---

## Relative Position Capture (Key Property)

The sine-cosine formulation has an **important mathematical property**:

For any fixed offset $\delta$, there exists a **linear transformation** (matrix $M$) such that:

$$M \cdot PE(\text{pos}) = PE(\text{pos} + \delta)$$

This means:
- The relationship between $PE(10)$ and $PE(20)$ (distance = 10)
- Is the **same transformation** as between $PE(30)$ and $PE(40)$ (distance = 10)
- The model can learn this matrix $M$ and thus understand **relative distances**

> This works because both **sine and cosine** components are needed — the linear transformation matrix contains both $\sin$ and $\cos$ terms (from the angle addition identities).

This is why sine-cosine **pairs** are used, not just sine alone.

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Why needed?** | Self attention has no notion of word order (parallel processing) |
| **What it does** | Encodes word position into a vector, added to the embedding |
| **Formula** | $PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$, $PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$ |
| **Dimension** | Same as embedding dimension ($d_{\text{model}}$) |
| **How added** | Element-wise addition (not concatenation) to avoid doubling parameters |
| **Key design** | Varying frequencies: low dimensions = high freq, high dimensions = low freq |
| **Analogy** | Binary encoding in continuous domain |
| **Critical property** | Relative positions captured via linear transformations ($M \cdot PE(pos) = PE(pos + \delta)$) |
| **Why sine + cosine?** | Both needed for the linear transformation property (angle addition identities) |
