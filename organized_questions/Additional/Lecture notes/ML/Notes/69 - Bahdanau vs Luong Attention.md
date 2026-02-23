# Bahdanau vs Luong Attention

## Background: Encoder-Decoder for NMT

### Architecture Recap

- **Task:** Neural Machine Translation (NMT) — translate one language to another using deep learning
- **Architecture:** Encoder-Decoder with LSTM (can be single, bidirectional, or stacked)

**Encoder:**
- Processes input sentence word-by-word: turn → off → the → lights
- Generates a **context vector** — a numerical summary/representation of the input sentence

**Decoder:**
- Takes the context vector as input
- Generates output sentence step-by-step: लाइट → बंद → करो → \<END\>

### The Bottleneck Problem

For sentences with **>30 words** (paragraphs, documents):

- The encoder struggles to compress the **entire input** into a single fixed-size context vector
- Information loss increases with sentence length
- Translation quality degrades significantly

> This single context vector bottleneck is what attention mechanisms were designed to solve.

---

## Attention Mechanism: Core Idea

Instead of passing **one** context vector from the last encoder hidden state:

| Without Attention | With Attention |
|---|---|
| Single context vector $h_T$ passed to decoder | All encoder hidden states $h_1, h_2, \ldots, h_T$ remain available |
| Decoder has limited information | Decoder dynamically selects relevant hidden states at each time step |
| Fixed representation | Dynamic context vector $c_i$ per decoder step |

### How Attention Works

At each decoder time step $i$, compute a **context vector** $c_i$:

$$c_i = \sum_{j=1}^{T} \alpha_{ij} \cdot h_j$$

Where:
- $h_j$ = encoder hidden state at position $j$
- $\alpha_{ij}$ = **alignment score** (attention weight) — how much decoder step $i$ should attend to encoder step $j$
- All $\alpha_{ij}$ sum to 1 (via softmax)

**Total alignment scores needed:** (number of output words) × (number of input words)

### What Do Alignment Scores Represent?

$\alpha_{ij}$ tells us: given what has been translated so far, how relevant is encoder hidden state $h_j$ for producing the current decoder output?

### What Do Alignment Scores Depend On?

$$\alpha_{ij} = f(h_j, s_{i-1}) \quad \text{(Bahdanau)} \qquad \text{or} \qquad \alpha_{ij} = f(h_j, s_i) \quad \text{(Luong)}$$

Two things:
1. **Encoder hidden state** $h_j$ — the information about input word $j$
2. **Decoder hidden state** — context of what has been generated so far

> The key difference between Bahdanau and Luong is **which decoder state** is used and **how** the alignment function $f$ is computed.

---

## Bahdanau Attention (Additive Attention)

### Key Characteristics

| Property | Value |
|----------|-------|
| **Also called** | Additive Attention |
| **Alignment function** | Feed-forward neural network |
| **Decoder state used** | Previous decoder state $s_{i-1}$ |
| **Context vector position** | Used as **input** to the decoder LSTM |

### Alignment Score Computation

**Step 1:** Compute energy scores using a neural network:

$$e_{ij} = v^T \cdot \tanh\left(W \cdot [s_{i-1} ; h_j] + b\right)$$

Where:
- $[s_{i-1} ; h_j]$ = concatenation of previous decoder state and encoder hidden state
- $W$ = weight matrix of hidden layer (learnable)
- $v$ = weight vector of output layer (learnable)
- $b$ = bias (optional)

**Step 2:** Normalize using softmax to get alignment weights:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}$$

**Step 3:** Compute context vector:

$$c_i = \sum_{j=1}^{T} \alpha_{ij} \cdot h_j$$

### Architecture Details

The neural network (called the **alignment model**) is:
- A single hidden layer feed-forward network
- **Input:** concatenation of $s_{i-1}$ and $h_j$ → dimension $2d$ (if hidden states are $d$-dimensional)
- **Hidden layer:** $n$ units with $\tanh$ activation
- **Output layer:** single unit producing scalar $e_{ij}$

### Batch Processing

For a sentence with $T$ input words:
1. Concatenate $s_{i-1}$ with each $h_j$ → matrix of shape $(T \times 2d)$
2. Multiply with weights $(2d \times n)$ → $(T \times n)$
3. Apply $\tanh$ + bias → $(T \times n)$
4. Multiply with output weights $(n \times 1)$ → $(T \times 1)$ — these are the $e_{ij}$ scores
5. Apply softmax → $\alpha_{ij}$ values

### Weight Sharing

The alignment network weights are **shared across all decoder time steps** (time-distributed fully connected network):
- Same weights used at time step 1, 2, 3, …
- Weights update only during backpropagation (after full decoder output)
- What changes at each time step is the **decoder hidden state** $s_{i-1}$

### Summary Equations (Bahdanau)

$$c_i = \sum_j \alpha_{ij} h_j$$

$$\alpha_{ij} = \text{softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

$$e_{ij} = v^T \cdot \tanh\left(W \cdot [s_{i-1} ; h_j] + b\right)$$

---

## Luong Attention (Multiplicative Attention)

### Key Characteristics

| Property | Value |
|----------|-------|
| **Also called** | Multiplicative Attention |
| **Alignment function** | Dot product |
| **Decoder state used** | Current decoder state $s_i$ |
| **Context vector position** | Concatenated with decoder output **after** the LSTM |

### Alignment Score Computation

$$e_{ij} = s_i^T \cdot h_j$$

Simply the **dot product** between the current decoder hidden state and the encoder hidden state. No neural network, no extra parameters.

Then apply softmax as usual:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}$$

### Architecture Flow (Luong)

1. Decoder LSTM computes current hidden state $s_i$ (without context vector as input)
2. Compute dot product of $s_i$ with all encoder hidden states $h_j$
3. Apply softmax → alignment weights $\alpha_{ij}$
4. Compute context vector: $c_i = \sum_j \alpha_{ij} h_j$
5. **Concatenate** $c_i$ with $s_i$ → new combined state $\tilde{s}_i$
6. Pass $\tilde{s}_i$ through a feed-forward layer + softmax → output word

### Why Dot Product Works

- If two vectors are **similar** → dot product is **large**
- If two vectors are **dissimilar** → dot product is **small**
- This naturally captures relevance/similarity without extra parameters

---

## Comparison: Bahdanau vs Luong

| Aspect | Bahdanau (Additive) | Luong (Multiplicative) |
|--------|--------------------|-----------------------|
| **Alignment function** | Feed-forward neural network | Dot product |
| **Decoder state used** | Previous: $s_{i-1}$ | Current: $s_i$ |
| **Context vector usage** | Input to decoder LSTM | Concatenated with LSTM output |
| **Extra parameters** | Yes (alignment model weights) | No |
| **Speed** | Slower (more parameters) | Faster |
| **Information freshness** | Uses older state | Uses most recent state |
| **Performance** | Good | Empirically better (more dynamic) |

### Why Current State ($s_i$) Is Better

- $s_i$ contains **more updated information** than $s_{i-1}$
- Includes the most recent translation context
- Allows more **dynamic** adjustment of attention weights
- Empirically proven to give **better translation quality**

### Why Dot Product Is Better

- Eliminates extra parameters → **faster training**
- Especially beneficial for long sentences
- Dot product is a natural similarity measure for vectors
- Simpler function, comparable or better results

---

## Importance for Transformers

Both attention types directly inspire **self-attention** in Transformers:
- Self-attention builds on these alignment score concepts
- Understanding Bahdanau and Luong attention provides the foundation for understanding the Transformer architecture
