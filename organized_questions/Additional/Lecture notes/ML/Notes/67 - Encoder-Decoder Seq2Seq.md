# Encoder-Decoder (Sequence-to-Sequence) Architecture

## Overview

- **Purpose:** Handle **sequence-to-sequence (Seq2Seq)** problems where both input and output are variable-length sequences
- **Key Application:** Machine Translation (e.g., English → Hindi)
- **Based on:** Paper by Ilya Sutskever et al. — *"Sequence to Sequence Learning with Neural Networks"*
- **Foundation for:** Attention Mechanism → Transformers → LLMs

---

## The Seq2Seq Challenge

Previous architectures handled:

| Stage | Data Type | Architecture |
|-------|-----------|-------------|
| 1 | Tabular data | ANN |
| 2 | Image data (2D grid) | CNN |
| 3 | Sequential data (input) | RNN / LSTM / GRU |
| **4** | **Seq-to-Seq** (variable input → variable output) | **Encoder-Decoder** |

### Three Core Challenges

1. **Variable-length input** — Input sentences can be 2 to 200+ words
2. **Variable-length output** — Output sentences can be any length
3. **Input length ≠ Output length** — "Nice to meet you" (4 words) → "आपसे मिलकर अच्छा लगा" (6 words)

---

## High-Level Architecture

```
┌──────────────┐     Context      ┌──────────────┐
│              │     Vector        │              │
│   ENCODER    │ ───────────────→ │   DECODER    │
│   (LSTM)     │   (hT, cT)       │   (LSTM)     │
│              │                   │              │
└──────────────┘                   └──────────────┘
      ↑                                   ↓
  Input Sequence                    Output Sequence
 (source language)                (target language)
```

**Three components:**

| Component | Role |
|-----------|------|
| **Encoder** | Reads the input sequence and compresses it into a fixed-size **context vector** |
| **Context Vector** | Summary of the entire input sequence (final $h_T$ and $c_T$ from encoder LSTM) |
| **Decoder** | Takes the context vector and generates the output sequence token by token |

---

## Encoder

- An **LSTM cell** unfolded over the input sequence length
- Processes input **word by word** (token by token)
- The final hidden state $h_T$ and cell state $c_T$ become the **context vector**

$$\text{Context Vector} = (h_T, c_T) \text{ — final states of encoder LSTM}$$

```
   [Nice]   [to]   [meet]   [you]
     ↓       ↓       ↓       ↓
   LSTM  →  LSTM  →  LSTM  →  LSTM  →  (hT, cT)
   t=1      t=2      t=3      t=4      Context Vector
```

> Can also use GRU instead of LSTM. Simple RNN is generally avoided (vanishing gradient).

---

## Decoder

- Also an **LSTM cell**, but a **separate** one from the encoder
- **Initial state** = context vector from the encoder (sets $h_0 = h_T^{\text{enc}}$, $c_0 = c_T^{\text{enc}}$)
- At each time step:
  - Takes the **previous output token** as input
  - Passes through LSTM → Dense layer → **Softmax** over target vocabulary
  - Outputs the token with **highest probability**

### Decoding Process

```
Context Vector → LSTM → Softmax → "आपसे"
                   ↓
   "आपसे" →     LSTM → Softmax → "मिलकर"
                   ↓
   "मिलकर" →    LSTM → Softmax → "अच्छा"
                   ↓
   "अच्छा" →    LSTM → Softmax → "लगा"
                   ↓
   "लगा" →       LSTM → Softmax → <END>   ← STOP
```

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<START>` | Fed as first input to decoder — signals "begin generating" |
| `<END>` | When decoder outputs this, generation **stops** |

### Softmax Output Layer

- Number of nodes = **target vocabulary size**
- Each node outputs a probability for one word
- The word with **highest probability** is selected as the output

---

## Training Process

### 1. Data Preparation

**Parallel dataset** (supervised learning):

| English (Source) | Hindi (Target) |
|-----------------|----------------|
| Think about it | सोच लो |
| Come in | अंदर आ जाओ |

### 2. Tokenization & Encoding

- Tokenize both source and target sentences
- Build separate vocabularies for source and target languages
- Add `<START>` and `<END>` tokens to target vocabulary
- One-hot encode all tokens (or use embeddings)

### 3. Forward Propagation

1. Feed input sentence to encoder token by token
2. Encoder produces context vector $(h_T, c_T)$
3. Pass context vector to decoder as initial state
4. Feed `<START>` token to decoder
5. Decoder produces output at each time step via softmax
6. Continue until `<END>` token is predicted

### 4. Teacher Forcing

During training, at each decoder time step:

- **Instead of** feeding the model's own (possibly wrong) prediction as the next input
- **Feed the correct target token** from the training data

| Without Teacher Forcing | With Teacher Forcing |
|------------------------|----------------------|
| Model predicted "लो" (wrong) → feeds "लो" to next step | Correct answer is "सोच" → feeds "सोच" to next step |
| Slow, unstable training | **Faster convergence** |

> Teacher forcing is used **only during training**, not during inference.

### 5. Loss Calculation

Since the decoder picks one word from the vocabulary at each step → **multi-class classification**:

$$\mathcal{L} = -\sum_{i=1}^{V} y_i^{\text{true}} \cdot \log(\hat{y}_i^{\text{pred}})$$

This is **Categorical Cross-Entropy**, computed at each time step, then summed or averaged across all time steps.

### 6. Backpropagation & Weight Update

1. Compute gradients of loss w.r.t. all trainable parameters (encoder LSTM, decoder LSTM, dense layer, softmax)
2. Update parameters using an optimizer (SGD, Adam, etc.) with a learning rate

---

## Inference (Prediction)

After training (all weights are frozen):

1. Feed input sentence to encoder → get context vector
2. Feed `<START>` to decoder + context vector
3. Decoder outputs first word → feed it back as input to next step
4. Repeat until `<END>` is generated

> **No teacher forcing** during inference — the model feeds its own predictions back.

---

## Improvements Over Basic Architecture

### 1. Embedding Layers

**Problem:** One-hot vectors are high-dimensional and sparse (vocabulary can be 100K+ words)

**Solution:** Add an **embedding layer** before both encoder and decoder LSTMs

| Property | One-Hot Encoding | Embeddings |
|----------|-----------------|------------|
| Dimensionality | = Vocabulary size (e.g., 100K) | Low (e.g., 32, 64, 256) |
| Sparsity | Very sparse (all zeros except one) | Dense |
| Semantic meaning | None | Captures word context |

**Options for embeddings:**
- Pre-trained (Word2Vec, GloVe)
- Trained jointly with the model (embedding layer in Keras)

### 2. Deep LSTM (Stacked Layers)

Replace single-layer LSTM with **multi-layer (deep) LSTM** in both encoder and decoder.

**Benefits:**

| Benefit | Explanation |
|---------|-------------|
| Better long-term dependency handling | Multiple context vectors (one per layer) carry more summary capacity |
| Hierarchical representation | Lower layers → word-level; Middle → sentence-level; Upper → paragraph-level |
| Increased model capacity | More parameters → captures finer-grained variations |

> The original Sutskever paper used **4-layer deep LSTMs** with 1000 units per layer.

### 3. Reversed Input Sequence

Feed the input to the encoder in **reverse order**:

- Instead of: Think → About → It
- Feed: It → About → Think

**Why it helps:**
- Reduces the **distance** between corresponding input-output word pairs
- The first input word and first output word become closer in the computation graph
- Gradients propagate more effectively for initial word pairs
- Works best for language pairs where **initial words carry more context** (e.g., English → French)

> This trick **does not always help** — it depends on the language pair.

---

## Original Paper Details (Sutskever et al.)

| Detail | Value |
|--------|-------|
| Task | English → French translation |
| Dataset | ~12 million sentences (304M English words, 348M French words) |
| English vocabulary | 160,000 words |
| French vocabulary | 80,000 words |
| Out-of-vocabulary token | `<UNK>` |
| Embedding dimension | 1,000 |
| LSTM layers | **4** (deep LSTM) |
| Units per LSTM layer | 1,000 |
| Input reversal | Yes (reversed input sequences) |
| Special token | `<EOS>` (end of sentence) |
| Output layer | Softmax |
| Performance metric | BLEU score = **34.8** (exceeded state-of-the-art baseline at the time) |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it solves** | Variable-length input → variable-length output (Seq2Seq) |
| **Encoder** | LSTM that compresses input into a context vector |
| **Context vector** | Final $(h_T, c_T)$ — fixed-size summary of input |
| **Decoder** | LSTM that generates output token-by-token using context vector |
| **Training** | Teacher forcing + categorical cross-entropy loss + backpropagation |
| **Improvements** | Embeddings, deep (stacked) LSTMs, reversed input |
| **Limitation** | Entire input compressed into one fixed-size vector → struggles with long sequences (>25-30 words) |
| **Next step** | **Attention Mechanism** (solves the fixed context vector bottleneck) |
