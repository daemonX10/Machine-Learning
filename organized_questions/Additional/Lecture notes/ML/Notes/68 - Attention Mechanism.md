# Attention Mechanism (Bahdanau Attention)

## Overview

- **Purpose:** Solve the **information bottleneck** in vanilla Encoder-Decoder by allowing the decoder to **focus on relevant parts** of the input at each time step
- **Key Innovation:** Replace the static context vector with a **dynamic, time-step-specific** context vector
- **Based on:** Paper by Bahdanau et al. — *"Neural Machine Translation by Jointly Learning to Align and Translate"*

---

## Problem with Vanilla Encoder-Decoder

### The Bottleneck

In the standard Seq2Seq architecture:

1. The **encoder** compresses the entire input sentence into a **single fixed-size context vector**
2. The **decoder** must generate the entire output from this one vector
3. For **long sentences** (>25-30 words), the context vector cannot carry enough information → **translation quality degrades**

### Two Specific Issues

| Issue | Description |
|-------|-------------|
| **Encoder overload** | Too much pressure on a small vector to summarize a long, complex sentence |
| **Static representation** | The same context vector is used at every decoder time step, regardless of which part of the input is relevant |

### Empirical Evidence

- BLEU score **drops sharply** for sentences >30 words in non-attention models
- Attention-based models maintain stable performance even for longer sequences

### Human Analogy

When translating a long text, humans don't memorize the entire sentence and then translate. Instead, they **focus on a relevant portion** (attention span) and translate on-the-go, shifting focus as they progress.

---

## Core Idea of Attention

At each decoder time step $i$:

1. **Look at all encoder hidden states** $h_1, h_2, \ldots, h_T$
2. **Compute a relevance score** (attention weight) for each encoder state
3. Create a **weighted sum** of encoder states → this is the **dynamic context vector** $c_i$
4. Feed $c_i$ to the decoder along with the previous output and previous decoder state

> The decoder can now **attend to different parts** of the input at different time steps.

---

## Notation Setup

### Encoder Side

| Symbol | Meaning |
|--------|---------|
| $h_j$ | Encoder hidden state at time step $j$ (where $j = 1, \ldots, T$) |

### Decoder Side

| Symbol | Meaning |
|--------|---------|
| $s_i$ | Decoder hidden state at time step $i$ |
| $y_{i-1}$ | Decoder input at time step $i$ (previous output token, via teacher forcing during training) |
| $c_i$ | Context vector (attention input) for decoder time step $i$ |

---

## Inputs to the Decoder at Each Time Step

| Architecture | Inputs at decoder time step $i$ |
|-------------|-------------------------------|
| **Vanilla Encoder-Decoder** | $y_{i-1}$, $s_{i-1}$ |
| **Attention-based** | $y_{i-1}$, $s_{i-1}$, **$c_i$** |

The attention mechanism adds **one extra input**: the dynamic context vector $c_i$.

---

## Computing the Context Vector $c_i$

### Step 1: Compute Alignment Scores (Energy)

For decoder time step $i$, compute a score for each encoder hidden state $h_j$:

$$e_{ij} = a(s_{i-1}, h_j)$$

Where $a(\cdot)$ is an **alignment function** (a small feedforward neural network).

**Why does $e_{ij}$ depend on $s_{i-1}$?**
- $s_{i-1}$ encodes "what has been translated so far"
- Given the translation context so far, the model decides which encoder state is most relevant **next**

### Step 2: Normalize with Softmax → Attention Weights

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}$$

This ensures:
- $\alpha_{ij} \geq 0$
- $\sum_{j=1}^{T} \alpha_{ij} = 1$ (weights form a probability distribution over encoder states)

### Step 3: Weighted Sum → Context Vector

$$c_i = \sum_{j=1}^{T} \alpha_{ij} \cdot h_j$$

| Property | Detail |
|----------|--------|
| Dimension of $c_i$ | Same as dimension of $h_j$ (encoder hidden state size) |
| $c_i$ is | A **weighted combination** of all encoder hidden states |
| Dominant $\alpha_{ij}$ | The encoder step $j$ with highest weight contributes most to $c_i$ |

---

## The Alignment Model (Neural Network)

The function $a(s_{i-1}, h_j)$ is implemented as a **small feedforward neural network**:

```
   s_{i-1}  ──┐
               ├──→ [Neural Network] ──→ e_{ij} (scalar)
   h_j      ──┘
```

- **Inputs:** $s_{i-1}$ (decoder's previous hidden state) and $h_j$ (encoder's hidden state at position $j$)
- **Output:** $e_{ij}$ — a scalar alignment/similarity score
- The NN has its **own trainable weights and biases**
- It is trained **jointly** with the entire encoder-decoder via backpropagation (no separate training)

### Why Use a Neural Network?

- ANNs are **universal function approximators** — they can learn any alignment function from data
- No need to manually design a similarity metric
- The alignment model learns what "relevance" means for the specific task

---

## Total Number of Attention Weights

For decoder with $I$ time steps and encoder with $J$ time steps:

$$\text{Total } \alpha \text{ values} = I \times J$$

**Example:** 4 encoder steps × 4 decoder steps = 16 attention weights

These can be arranged in an **attention matrix** and visualized as a heatmap.

---

## Full Architecture (One Decoder Step)

For decoder time step $i = 2$ (printing "बंद"):

```
                    Alignment NN
                    ┌─────────┐
   s₁, h₁  ──────→ │   NN    │ ──→ e₂₁ ──┐
   s₁, h₂  ──────→ │   NN    │ ──→ e₂₂ ──┤
   s₁, h₃  ──────→ │   NN    │ ──→ e₂₃ ──┤── Softmax → α₂₁, α₂₂, α₂₃, α₂₄
   s₁, h₄  ──────→ │   NN    │ ──→ e₂₄ ──┘
                    └─────────┘

   c₂ = α₂₁·h₁ + α₂₂·h₂ + α₂₃·h₃ + α₂₄·h₄    (weighted sum)

   Decoder LSTM inputs: c₂, s₁, y₁ → produces s₂ and output "बंद"
```

---

## Complete Equations Summary

### Alignment Score

$$e_{ij} = a(s_{i-1}, h_j) \quad \text{(feedforward NN)}$$

### Attention Weights

$$\alpha_{ij} = \text{softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}$$

### Context Vector

$$c_i = \sum_{j=1}^{T} \alpha_{ij} \cdot h_j$$

### Decoder Hidden State Update

$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

Where $f$ is the LSTM/GRU function of the decoder.

---

## Attention Visualization

The attention weights $\alpha_{ij}$ can be plotted as a **heatmap**:

- **X-axis:** Source language words (encoder)
- **Y-axis:** Target language words (decoder)
- **Cell intensity:** $\alpha_{ij}$ value (higher = more attention)

### Example (English → French)

| | european | economic | area |
|---|---|---|---|
| **européenne** | **0.9** | 0.05 | 0.05 |
| **économique** | 0.05 | **0.85** | 0.1 |
| **zone** | 0.1 | 0.05 | **0.85** |

The diagonal pattern shows the model correctly **aligning** corresponding words across languages.

---

## Key Benefit: Performance on Long Sentences

| Model | BLEU Score (short sentences) | BLEU Score (long sentences >30 words) |
|-------|------------------------------|---------------------------------------|
| Vanilla Encoder-Decoder | Good | **Drops sharply** |
| Attention-based | Good | **Remains stable** |

---

## Original Paper Details (Bahdanau et al.)

| Detail | Value |
|--------|-------|
| Encoder | **Bidirectional LSTM** (captures both forward and backward context) |
| Decoder | Standard LSTM (unidirectional) |
| Alignment model | Feedforward neural network with learnable weights |
| Task | English → French translation |
| Key result | Stable BLEU scores even for sentences >30 words |

> The encoder uses **BiLSTM** so that each $h_j$ encodes context from both directions. The rest of the architecture remains the same.

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Problem solved** | Static context vector bottleneck in Seq2Seq models |
| **Core mechanism** | Dynamic context vector $c_i$ via weighted sum of encoder hidden states |
| **Attention weights** | $\alpha_{ij}$ — learned via a small NN, softmax-normalized |
| **Key equation** | $c_i = \sum_j \alpha_{ij} \cdot h_j$ |
| **Alignment function** | $e_{ij} = a(s_{i-1}, h_j)$ — feedforward neural network |
| **Training** | Joint end-to-end training (alignment NN trains alongside encoder-decoder) |
| **Benefit** | Handles long sequences; decoder focuses on relevant input parts per step |
| **Visualization** | Attention weights form an interpretable heatmap |
| **Next step** | **Cross-Attention** → **Self-Attention** → **Transformers** |
