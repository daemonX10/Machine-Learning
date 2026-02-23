# Bidirectional RNN (BiRNN, BiLSTM, BiGRU)

## Overview

- **Core Idea:** Process the input sequence in **both directions** — left-to-right (forward) and right-to-left (backward) — simultaneously
- **Purpose:** Capture context from **both past and future** inputs at every time step
- **Applies to:** Simple RNN, LSTM (BiLSTM), and GRU (BiGRU)

---

## Motivation: Why Unidirectional RNN Fails

In a standard (unidirectional) RNN, information flows **only left to right**:

$$\text{Output at time } t \quad \text{depends on} \quad x_1, x_2, \ldots, x_t$$

**Problem:** There are tasks where **future inputs affect past outputs**.

### Example: Named Entity Recognition (NER)

| Sentence | Entity | Type |
|----------|--------|------|
| I love **Amazon**. It's a great **website**. | Amazon | Organization (ORG) |
| I love **Amazon**. It's a beautiful **river**. | Amazon | Location (LOC) |

- At the word "Amazon", the unidirectional RNN has only seen "I love" — **not enough** to disambiguate
- The word after Amazon ("website" vs "river") determines the entity type
- A unidirectional RNN **cannot look ahead** → fails to classify correctly

### Other Tasks Where Future Context Matters

- **Parts-of-Speech (POS) Tagging**
- **Machine Translation**
- **Sentiment Analysis**
- **Time Series Forecasting**

---

## Architecture

A Bidirectional RNN uses **two separate RNNs**:

| Component | Direction | Processes |
|-----------|-----------|-----------|
| **Forward RNN** (→) | Left to Right | $x_1 \to x_2 \to x_3 \to x_4$ |
| **Backward RNN** (←) | Right to Left | $x_4 \to x_3 \to x_2 \to x_1$ |

At each time step, the outputs of both RNNs are **concatenated** to produce the final output.

```
Forward RNN:   [x₁] → [x₂] → [x₃] → [x₄]
                 ↓       ↓       ↓       ↓
               h₁→     h₂→     h₃→     h₄→
                 ↘       ↘       ↘       ↘
                  ŷ₁      ŷ₂      ŷ₃      ŷ₄   (concat)
                 ↗       ↗       ↗       ↗
               h₁←     h₂←     h₃←     h₄←
                 ↑       ↑       ↑       ↑
Backward RNN:  [x₁] ← [x₂] ← [x₃] ← [x₄]
```

At any time step $t$, the output $\hat{y}_t$ has access to information from **all** input tokens (both past and future).

---

## Mathematical Formulation

### Forward Hidden State

$$\overrightarrow{h_t} = \tanh\left(W_{\text{fwd}} \cdot \overrightarrow{h_{t-1}} + U \cdot x_t + b_{\text{fwd}}\right)$$

### Backward Hidden State

$$\overleftarrow{h_t} = \tanh\left(W_{\text{bwd}} \cdot \overleftarrow{h_{t+1}} + U \cdot x_t + b_{\text{bwd}}\right)$$

> Note: In the backward RNN, the recurrence goes from $t+1$ to $t$ (future affects current).

### Output at Time Step $t$

$$\hat{y}_t = \sigma\left(V \cdot \left[\overrightarrow{h_t} \; ; \; \overleftarrow{h_t}\right] + b_y\right)$$

Where $[\;\cdot\;;\;\cdot\;]$ denotes **concatenation** of the forward and backward hidden states.

| Symbol | Meaning |
|--------|---------|
| $\overrightarrow{h_t}$ | Forward hidden state at time $t$ |
| $\overleftarrow{h_t}$ | Backward hidden state at time $t$ |
| $W_{\text{fwd}}, W_{\text{bwd}}$ | Recurrent weights (separate for each direction) |
| $V$ | Output weight matrix |
| $\sigma$ | Output activation (sigmoid, softmax, etc.) |

---

## Keras Implementation

### Simple BiRNN

```python
from keras.layers import SimpleRNN, Bidirectional, Embedding, Dense
from keras.models import Sequential

model = Sequential([
    Embedding(input_dim=10000, output_dim=32),
    Bidirectional(SimpleRNN(5)),       # wraps SimpleRNN in bidirectional
    Dense(1, activation='sigmoid')
])
```

### BiLSTM

```python
from keras.layers import LSTM, Bidirectional

model = Sequential([
    Embedding(input_dim=10000, output_dim=32),
    Bidirectional(LSTM(5)),
    Dense(1, activation='sigmoid')
])
```

### BiGRU

```python
from keras.layers import GRU, Bidirectional

model = Sequential([
    Embedding(input_dim=10000, output_dim=32),
    Bidirectional(GRU(5)),
    Dense(1, activation='sigmoid')
])
```

> **Key:** The `Bidirectional()` wrapper works with **any** RNN cell type.

### Parameter Count

| Model | Units | Parameters |
|-------|-------|------------|
| SimpleRNN(5) | 5 | 190 |
| Bidirectional(SimpleRNN(5)) | 5×2 | **380** (exactly 2×) |

Parameters **double** because two separate RNNs are used.

---

## Applications

| Application | Why BiRNN Helps |
|-------------|-----------------|
| **Named Entity Recognition (NER)** | Current word's entity type depends on surrounding context |
| **POS Tagging** | A word's part of speech can depend on words that follow |
| **Machine Translation** | Full sentence context improves translation quality |
| **Sentiment Analysis** | Bidirectional context often outperforms unidirectional |
| **Time Series Forecasting** | Viewing data forward and backward helps forecasting |

---

## Drawbacks

### 1. Increased Computational Cost

- **Weights and biases double** → more parameters to train
- Longer training time
- Higher risk of **overfitting** (requires dropout, regularization)

### 2. Latency in Real-Time Applications

- BiRNN needs the **entire input sequence** before processing
- **Cannot be used** for real-time / streaming tasks (e.g., live speech recognition) where input arrives incrementally
- Causes **latency issues** — must wait for the full input before producing any output

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it does** | Processes sequences in both forward and backward directions |
| **Architecture** | Two separate RNNs (forward + backward), outputs concatenated |
| **Key equation** | $\hat{y}_t = \sigma(V \cdot [\overrightarrow{h_t} ; \overleftarrow{h_t}] + b)$ |
| **Applies to** | SimpleRNN, LSTM (→ BiLSTM), GRU (→ BiGRU) |
| **Most used variant** | **BiLSTM** |
| **Best for** | Tasks requiring both past and future context |
| **Drawback** | 2× parameters; cannot handle streaming/real-time input |
