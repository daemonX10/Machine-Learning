# Lecture 02 — Introduction to Language Models

## 1. Language Model — Formal Definition

A **language model** assigns a probability to a sequence of tokens $w_1, w_2, \ldots, w_n$:

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})$$

Each factor is a **conditional probability** — the probability of the next token given all previous tokens.

> $P(w_1)$ can be written as $P(w_1 \mid \langle \text{START} \rangle)$ where $\langle \text{START} \rangle$ is a sentence-begin marker.

---

## 2. Statistical Language Models (Count-Based)

Conditional probabilities estimated by counting co-occurrences in a corpus:

$$P(\text{going} \mid \text{I am}) = \frac{\text{count}(\text{I am going})}{\text{count}(\text{I am})}$$

### Problem: Long Contexts → Zero Counts

If the context $w_1, \ldots, w_{n-1}$ is long, the exact sequence may never appear in the corpus → numerator or denominator becomes 0 → entire sequence probability collapses to 0.

### Solution: Markov (N-gram) Assumption

Limit context to a **fixed window** of $n-1$ previous tokens:

$$P(w_t \mid w_1, \ldots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

| N-gram | Context Length | Markov Order | Example |
|---|---|---|---|
| Unigram ($n=1$) | 0 | 0th order | $P(w_i)$ |
| Bigram ($n=2$) | 1 | 1st order | $P(w_i \mid w_{i-1})$ |
| Trigram ($n=3$) | 2 | 2nd order | $P(w_i \mid w_{i-2}, w_{i-1})$ |

> **An n-gram LM is an $(n-1)$-order Markov process.**

### Limitation

N-gram models **cannot capture long-range dependencies**. Example:
> "The project which he had been working on for months was finally approved by the ___."

Predicting the blank requires looking back to "project" — impossible with small $n$.

---

## 3. Evaluating Language Models

### Extrinsic Evaluation (Task-Based)

- Apply LMs to a downstream task (translation, summarization, etc.)
- Compare accuracy: if $A_2 > A_1$, then $LM_2$ is better
- **Problem**: results depend on task choice — no universal ground-truth task

### Intrinsic Evaluation — Perplexity

$$\text{PPL}(w_1, \ldots, w_n) = P(w_1, \ldots, w_n)^{-1/n} = \left(\prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})\right)^{-1/n}$$

| Property | Meaning |
|---|---|
| Higher probability → | **Lower** perplexity |
| Lower perplexity → | **Better** language model |
| Goal | **Minimize** perplexity |

- Perplexity is closely related to **entropy** of the distribution
- Task-independent metric — intrinsic to the model

---

## 4. Neural Language Models

### 4.1 Fixed-Window Neural LM (CNN-style)

- Fix a context window of size $k$ (e.g., last 4 words)
- Concatenate embeddings of $k$ words → feed through linear layers $W$ and $U$
- $U$ is the **unembedding layer** — projects hidden state to vocabulary-sized distribution

$$h = \sigma(W \cdot [e_{t-k}; \ldots; e_{t-1}] + b_1)$$
$$P(w_t) = \text{softmax}(U \cdot h + b_2)$$

**Problems:**
1. Must pre-decide window size $k$
2. Changing $k$ changes parameter matrix dimensions
3. Still cannot capture long-range dependencies

### 4.2 Recurrent Neural Network (RNN)

For each token $x_t$, compute a hidden state:

$$h_t = \sigma(W_e \cdot e_t + W_h \cdot h_{t-1} + b)$$

- **$W_e$**: input-to-hidden projection (shared across all positions)
- **$W_h$**: hidden-to-hidden projection (shared across all positions)
- **Parameter sharing** → independent of sequence length
- Final hidden state → unembedding layer $U$ → softmax → next word probability

| Advantage | Detail |
|---|---|
| No fixed context window | Theoretically captures entire history |
| Shared parameters | $W_e$, $W_h$, $U$ — same across all time steps |

**Problems:**
1. **Sequential processing** — cannot parallelize (each $h_t$ depends on $h_{t-1}$)
2. **Vanishing gradient** — during BPTT, gradients shrink exponentially over long sequences

### 4.3 Training RNNs — Backpropagation Through Time (BPTT)

- Loss at each position: **cross-entropy** between predicted and actual token
- Total loss: $J = \sum_{t} J_t$
- Gradient of $J$ w.r.t. $h_t$ involves chain rule through all previous hidden states
- $\frac{\partial h_t}{\partial h_{t-k}}$ produces a **matrix** (Jacobian) — its magnitude shrinks/explodes over long chains

---

## 5. LSTM (Long Short-Term Memory)

Addresses vanishing gradients by introducing a **cell state** $c_t$ (protected memory) and **three gates**:

| Gate | Controls | Formula |
|---|---|---|
| **Forget** $f_t$ | What to retain from previous cell state | $f_t = \sigma(W_f \cdot x_t + U_f \cdot h_{t-1} + b_f)$ |
| **Input** $i_t$ | What to write from cell content to cell state | $i_t = \sigma(W_i \cdot x_t + U_i \cdot h_{t-1} + b_i)$ |
| **Output** $o_t$ | What to read from cell state to hidden state | $o_t = \sigma(W_o \cdot x_t + U_o \cdot h_{t-1} + b_o)$ |

**Cell content** (candidate):
$$\tilde{c}_t = \tanh(W_c \cdot x_t + U_c \cdot h_{t-1} + b_c)$$

**Cell state update:**
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Hidden state:**
$$h_t = o_t \odot \tanh(c_t)$$

> Cell state is **not directly accessible** by input $x_t$ — only through the gating mechanism via cell content.

---

## 6. GRU (Gated Recurrent Unit)

Simplified LSTM with **2 gates** (update, reset) and **no separate cell state**:

| Aspect | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Memory | Separate cell state $c_t$ | Hidden state acts as memory |
| Parameters | More | Fewer |
| Performance | Similar in practice | Similar in practice |

> **Rule of thumb**: Start with GRU; switch to LSTM if you have abundant data.

---

## 7. Summary & Road Ahead

| Model | Long-range Deps | Parallelizable | Key Issue |
|---|---|---|---|
| N-gram | ✗ | ✓ | Fixed context, zero counts |
| Fixed-window NN | ✗ | ✓ | Fixed window size |
| RNN | Partially | ✗ | Vanishing gradient, sequential |
| LSTM/GRU | Better (gated) | ✗ | Still sequential |
| **Transformer** | ✓ | ✓ | *(Next lecture)* |

### Key Takeaways
- Language models predict next tokens via conditional probabilities
- **Perplexity** (lower is better) is the standard intrinsic evaluation metric
- RNNs overcome fixed context but suffer from sequential processing and vanishing gradients
- LSTM/GRU use gating to protect long-term memory
- The **bottleneck problem** in seq2seq and lack of parallelism motivate **attention** and **Transformers**
