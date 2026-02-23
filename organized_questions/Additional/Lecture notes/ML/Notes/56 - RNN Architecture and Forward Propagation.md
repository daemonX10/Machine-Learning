# RNN Architecture and Forward Propagation

## Overview

- **Topic:** How a simple RNN is structured and how it makes predictions (forward propagation)
- **Key Concept:** RNNs process inputs **one time step at a time**, feeding the hidden state output back as input for the next step
- **Core Mechanism:** Unfolding through time — the same recurrent layer is reused at every time step with shared weights

---

## Input Data Format

### Shape Convention

RNN inputs follow a specific shape:

$$\text{Input Shape} = (\text{batch\_size}, \text{time\_steps}, \text{input\_features})$$

| Dimension | Meaning | Example (Sentiment Analysis) |
|-----------|---------|------------------------------|
| `batch_size` | Number of samples processed together | Number of movie reviews |
| `time_steps` | Number of sequential steps (max words in longest sentence) | Longest review = 4 words → 4 |
| `input_features` | Size of each input vector | Vocabulary size = 5 → each word is a 5D vector |

### Example — Sentiment Analysis

Given vocabulary: {movie, was, good, bad, not} → 5 words

**One-hot encoding** of each word:

| Word | Vector |
|------|--------|
| movie | [1, 0, 0, 0, 0] |
| was | [0, 1, 0, 0, 0] |
| good | [0, 0, 1, 0, 0] |
| bad | [0, 0, 0, 1, 0] |
| not | [0, 0, 0, 0, 1] |

**Review 1:** "movie was good" → shape = (3, 5) → 3 time steps, 5 features

### Notation

| Symbol | Meaning |
|--------|---------|
| $X_1$ | First review (all words) |
| $X_{11}$ | First review, first word |
| $X_{12}$ | First review, second word |
| $X_{it}$ | $i$-th review, $t$-th word |

---

## RNN vs ANN — Two Key Differences

| Aspect | ANN | RNN |
|--------|-----|-----|
| **Input feeding** | All inputs fed **simultaneously** | Inputs fed **one at a time** (per time step) |
| **Information flow** | Feed-forward only (input → output) | Has a **feedback loop** — hidden layer output feeds back as input |

> The feedback connection from hidden layer to itself is what makes an RNN "recurrent."

---

## RNN Architecture

### Structure

```
Input Layer (input_features nodes)
    ↓ [Weights: W_i]
Hidden Layer (n nodes) ←──── Feedback [Weights: W_h]
    ↓ [Weights: W_o]              ↑
Output Layer                      │
                            Hidden State (h_t)
```

### Three Weight Matrices

| Weight | Shape | Connects |
|--------|-------|----------|
| $W_i$ | (input_features × hidden_units) | Input → Hidden |
| $W_h$ | (hidden_units × hidden_units) | Hidden → Hidden (recurrent connection) |
| $W_o$ | (hidden_units × output_units) | Hidden → Output |

Plus biases: $b_h$ (for hidden layer) and $b_o$ (for output layer).

### Parameter Count Example

For input_features = 5, hidden_units = 3, output_units = 1:

| Component | Count |
|-----------|-------|
| $W_i$ | $5 \times 3 = 15$ |
| $W_h$ | $3 \times 3 = 9$ |
| $W_o$ | $3 \times 1 = 3$ |
| $b_h$ | $3$ |
| $b_o$ | $1$ |
| **Total** | **31 parameters** |

---

## Forward Propagation — Unfolding Through Time

The RNN processes one word at a time, reusing the same network at each step.

### Step-by-Step Process

#### Time Step $t = 1$ (First word)

1. Input: $X_{i1}$ (first word vector)
2. No previous hidden state → initialize $h_0 = \vec{0}$ (zero vector)
3. Compute hidden state:

$$h_1 = \tanh(W_i \cdot X_{i1} + W_h \cdot h_0 + b_h)$$

#### Time Step $t = 2$ (Second word)

1. Input: $X_{i2}$ (second word vector)
2. Previous hidden state: $h_1$ (output from step 1)
3. Compute hidden state:

$$h_2 = \tanh(W_i \cdot X_{i2} + W_h \cdot h_1 + b_h)$$

#### Time Step $t = T$ (Last word)

$$h_T = \tanh(W_i \cdot X_{iT} + W_h \cdot h_{T-1} + b_h)$$

#### Final Output (only at last time step):

$$\hat{y} = \sigma(W_o \cdot h_T + b_o)$$

Where $\sigma$ is the output activation function (sigmoid for binary, softmax for multi-class, linear for regression).

---

## General RNN Equations

### Hidden State Update (at each time step $t$):

$$\boxed{h_t = \tanh(W_i \cdot X_t + W_h \cdot h_{t-1} + b_h)}$$

### Output (at final time step or each step, depending on architecture):

$$\boxed{\hat{y} = g(W_o \cdot h_T + b_o)}$$

Where:

| Symbol | Meaning |
|--------|---------|
| $X_t$ | Input at time step $t$ |
| $h_t$ | Hidden state at time step $t$ |
| $h_{t-1}$ | Hidden state from previous time step |
| $h_0$ | Initial hidden state (typically zero vector) |
| $W_i$ | Input-to-hidden weight matrix |
| $W_h$ | Hidden-to-hidden weight matrix (recurrent weights) |
| $W_o$ | Hidden-to-output weight matrix |
| $b_h$ | Hidden layer bias |
| $b_o$ | Output layer bias |
| $\tanh$ | Default activation for hidden state |
| $g(\cdot)$ | Output activation — sigmoid / softmax / linear |

---

## Visual Summary — Unfolded RNN

```
    X_{i1}        X_{i2}        X_{i3}        X_{iT}
      ↓             ↓             ↓             ↓
h_0 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → ... → [RNN] → h_T → [Output] → ŷ
       ↑              ↑              ↑                ↑
      W_i, W_h       W_i, W_h      W_i, W_h         W_i, W_h
      (shared)       (shared)      (shared)          (shared)
```

### Key Observations

1. **Same weights reused** at every time step → **parameter sharing / weight sharing**
2. **Two inputs** at each step (except $t=1$): current word $X_t$ + previous hidden state $h_{t-1}$
3. At $t=1$: only one input ($X_{i1}$) + zero-initialized hidden state ($h_0 = \vec{0}$)
4. **Sequence information is preserved**: final hidden state $h_T$ contains information from **all** previous time steps
5. Output is typically computed only at the **last time step** (for many-to-one tasks like sentiment analysis)

---

## How RNNs Capture Sequence Information

When computing the final output:
- $h_T$ depends on $h_{T-1}$ and $X_T$
- $h_{T-1}$ depends on $h_{T-2}$ and $X_{T-1}$
- ... and so on back to $h_1$

> **The final hidden state is a compressed representation of the entire input sequence.** A simple RNN can effectively retain information from approximately the last 10 time steps.

---

## Keras Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential([
    SimpleRNN(3, input_shape=(4, 5), activation='tanh'),  # 3 hidden units, 4 time steps, 5 features
    Dense(1, activation='sigmoid')                         # Binary output
])

model.summary()
# Total params: 31 (matches our manual calculation)
```

### `SimpleRNN` Parameters

| Parameter | Meaning |
|-----------|---------|
| `units` | Number of hidden units (neurons in recurrent layer) |
| `input_shape` | `(time_steps, input_features)` |
| `activation` | Default is `'tanh'` |
| `return_sequences` | `False` (default) — return only last $h_T$; `True` — return $h_t$ at every step |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Input format** | `(batch_size, time_steps, input_features)` |
| **Key mechanism** | Hidden state feedback — $h_{t-1}$ feeds into computation of $h_t$ |
| **Weight sharing** | Same $W_i$, $W_h$, $W_o$ used at every time step |
| **Hidden state formula** | $h_t = \tanh(W_i \cdot X_t + W_h \cdot h_{t-1} + b_h)$ |
| **Output formula** | $\hat{y} = g(W_o \cdot h_T + b_o)$ |
| **Default hidden activation** | $\tanh$ |
| **Initial hidden state** | $h_0 = \vec{0}$ (zero vector) |
| **Sequence memory** | Final $h_T$ encodes information from all previous time steps |
| **Next topic** | Backpropagation Through Time (BPTT) |
