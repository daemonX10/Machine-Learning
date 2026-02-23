# Deep (Stacked) RNNs

## Overview

- **Also called:** Stacked RNNs, Deep LSTMs, Deep GRUs
- **Core Idea:** Stack multiple RNN layers **vertically** (on top of each other) to increase the network's representational power
- **Analogy:** Just as adding hidden layers to an ANN makes it a Deep Neural Network, stacking RNN cells makes it a Deep RNN

---

## Motivation

### ANN Analogy (Spiral Dataset Example)

| Setup | Result |
|-------|--------|
| 1 hidden layer, 4 neurons | Poor fit — most regions mislabeled |
| 1 hidden layer, 6 neurons | Slight improvement |
| 2 hidden layers, 6 neurons each | Significant improvement |

**Lesson:** Adding more layers (depth) increases the network's ability to capture complex patterns in data. The same principle applies to RNNs.

---

## Architecture

### Single-Layer RNN (Baseline)

```
Input:   [x₁]  →  [x₂]  →  [x₃]  →  [x₄]  →  [x₅]
            ↓        ↓        ↓        ↓        ↓
Layer 1: [RNN] → [RNN] → [RNN] → [RNN] → [RNN] → output
```

### Deep RNN (2 Layers)

```
Input:   [x₁]  →  [x₂]  →  [x₃]  →  [x₄]  →  [x₅]
            ↓        ↓        ↓        ↓        ↓
Layer 1: [RNN] → [RNN] → [RNN] → [RNN] → [RNN]
            ↓        ↓        ↓        ↓        ↓
Layer 2: [RNN] → [RNN] → [RNN] → [RNN] → [RNN] → output
```

**Two axes in the unfolded architecture:**

| Axis | Symbol | Meaning |
|------|--------|---------|
| **Time axis** (horizontal) | $t$ | Time step (1, 2, …, T) |
| **Depth axis** (vertical) | $l$ | Layer number (1, 2, …, L) |

---

## Information Flow

Each RNN cell at layer $l$, time step $t$ receives **two inputs**:

1. **From the left:** Hidden state from the same layer at the previous time step → $h_{t-1}^{(l)}$
2. **From below:** Output from the previous layer at the same time step → $h_t^{(l-1)}$

And produces **two outputs**:

1. **To the right:** Hidden state to the next time step → $h_{t+1}^{(l)}$
2. **Upward:** Input to the next layer → fed into $h_t^{(l+1)}$

---

## Notation

A cell at time step $t$, layer $l$ is denoted:

$$h_t^{(l)}$$

### General Equation

$$h_t^{(l)} = \tanh\left(W^{(l)} \cdot h_{t-1}^{(l)} + U^{(l)} \cdot h_t^{(l-1)} + b^{(l)}\right)$$

| Symbol | Meaning |
|--------|---------|
| $h_t^{(l)}$ | Hidden state at time $t$, layer $l$ |
| $W^{(l)}$ | Recurrent weight matrix for layer $l$ (time direction) |
| $U^{(l)}$ | Input weight matrix for layer $l$ (depth direction) |
| $b^{(l)}$ | Bias for layer $l$ |
| $h_t^{(0)}$ | The raw input $x_t$ (layer 0 = input layer) |

---

## Parameter Count Example

**Setup:** Embedding(10000, 32) → SimpleRNN(5) → SimpleRNN(5) → Dense(1)

| Layer | Connections | Parameters |
|-------|-------------|------------|
| Embedding | 10000 × 32 | 320,000 |
| SimpleRNN Layer 1 | (32×5) + (5×5) + 5 | 190 |
| SimpleRNN Layer 2 | (5×5) + (5×5) + 5 | 55 |
| Dense | 5×1 + 1 | 6 |
| **Total** | | **320,251** |

**Layer 1:** Input dim = 32 (from embedding), units = 5
- Input weights: $32 \times 5 = 160$
- Recurrent weights: $5 \times 5 = 25$
- Biases: $5$
- Total: $190$

**Layer 2:** Input dim = 5 (from Layer 1 output), units = 5
- Input weights: $5 \times 5 = 25$
- Recurrent weights: $5 \times 5 = 25$
- Biases: $5$
- Total: $55$

---

## `return_sequences` Parameter

When stacking RNN layers, **all layers except the last** must have `return_sequences=True`:

```python
model = Sequential([
    Embedding(10000, 32),
    SimpleRNN(5, return_sequences=True),   # outputs at EVERY time step → feeds Layer 2
    SimpleRNN(5),                           # outputs only at LAST time step
    Dense(1, activation='sigmoid')
])
```

| Setting | Behavior |
|---------|----------|
| `return_sequences=True` | Output hidden state at **every** time step (needed to feed the next RNN layer) |
| `return_sequences=False` | Output hidden state at **only the last** time step (default) |

> If `return_sequences=False` on a non-final RNN layer, the next RNN layer receives no sequence input → **error**.

---

## Why Use Deep RNNs

### 1. Hierarchical Representation Learning

| Layer Level | What It Captures | Example (Sentiment Analysis) |
|-------------|------------------|------------------------------|
| **Lower layers** | Primitive, word-level features | "love", "hate", "terrible" |
| **Middle layers** | Sentence-level semantics | "audio is bad" |
| **Upper layers** | Overall document-level meaning | "audio is bad but display is great, overall I am happy" → Positive |

Deep RNNs capture the **hierarchy** inherent in language: words → phrases → sentences → paragraphs.

### 2. Customization for Advanced Architectures

- Deep RNNs are building blocks of advanced architectures like **Encoder-Decoder** with **Attention Mechanism**
- Google Translate used **Deep LSTMs** (stacked LSTM layers) in its architecture

### 3. Increased Model Capacity

- More parameters → can capture **fine-grained variations** in data
- Better generalization (with sufficient data)
- Better handling of input variability

---

## When to Use Deep RNNs

| Condition | Recommendation |
|-----------|---------------|
| Complex tasks (speech recognition, machine translation) | ✅ Use Deep RNNs |
| Large datasets available | ✅ Needed to prevent overfitting |
| Sufficient computational resources | ✅ Training is expensive |
| Single-layer RNN gives unsatisfactory results | ✅ Try adding depth |

---

## Keras Implementation

### Deep SimpleRNN

```python
model = Sequential([
    Embedding(10000, 32),
    SimpleRNN(5, return_sequences=True),
    SimpleRNN(5),
    Dense(1, activation='sigmoid')
])
```

### Deep LSTM (Most Common)

```python
model = Sequential([
    Embedding(10000, 32),
    LSTM(5, return_sequences=True),
    LSTM(5),
    Dense(1, activation='sigmoid')
])
```

### Deep GRU

```python
model = Sequential([
    Embedding(10000, 32),
    GRU(5, return_sequences=True),
    GRU(5),
    Dense(1, activation='sigmoid')
])
```

> **In practice,** Deep LSTMs and Deep GRUs are preferred over Deep SimpleRNNs due to vanishing gradient issues.

---

## Disadvantages

### 1. Overfitting Risk

- More parameters means higher chance of overfitting
- Requires careful use of **dropout**, **regularization**, **learning rate tuning**, and **weight initialization**

### 2. Increased Training Time

- More layers = more parameters = more backpropagation computation
- Gradients can flow along **any path** in the 2D grid (both time and depth axes), making gradient computation more complex

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it does** | Stacks multiple RNN layers vertically for deeper feature extraction |
| **Key equation** | $h_t^{(l)} = \tanh(W^{(l)} h_{t-1}^{(l)} + U^{(l)} h_t^{(l-1)} + b^{(l)})$ |
| **Critical setting** | `return_sequences=True` on all RNN layers except the last |
| **Best variants** | Deep LSTM, Deep GRU (not Deep SimpleRNN) |
| **Best for** | Complex sequential tasks, large datasets, hierarchical data |
| **Drawback** | Overfitting risk, longer training time |
