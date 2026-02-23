# Lecture 5.1 — Neural Language Models: RNNs

---

## 1. From Statistical to Neural Language Models

| Aspect | Statistical LM | Neural LM |
|---|---|---|
| **Task** | Predict next word | Predict next word |
| **Method** | Sample from n-gram distribution (unigram, bigram, trigram) | Neural network over embeddings |
| **Problems** | Can't handle unknown words; sparse bigram matrix | Dense vectors; handles unknown words via embeddings |

---

## 2. CNN-Based Language Model

### Architecture

1. **Fix a window size** $k$ (e.g., 4 previous words)
2. Retrieve embeddings $x_1, x_2, \dots, x_k$ (from Word2Vec/GloVe/one-hot)
3. **Flatten** into a single vector: $[x_1; x_2; \dots; x_k] \in \mathbb{R}^{1 \times kd}$
4. Multiply by kernel/filter matrix $W \in \mathbb{R}^{kd \times h}$ → hidden state $\in \mathbb{R}^{h}$
5. Apply **linear projection** $U \in \mathbb{R}^{h \times |V|}$ → logits over vocabulary
6. Apply **softmax** → probability distribution → sample next word

### Problems with CNN Approach

| # | Problem |
|---|---|
| 1 | **Fixed window size** — must choose $k$ as a hyperparameter |
| 2 | **$W$ grows with window size** — larger $k$ → larger weight matrix |
| 3 | **No message passing** — each embedding $e_i$ only affects one column of $W$; no cross-token interaction during the multiplication |

### The Message-Passing Problem (Subtle)

When multiplying $W \cdot [e_1, e_2, e_3, e_4]^T$:
- $e_1$ affects only column 1 of the output
- $e_2$ affects only column 2, etc.
- Result: learning **4 independent functions**, not a joint representation
- No information flows between token positions

---

## 3. Recurrent Neural Network (RNN)

### Architecture

At each time step $t$:

$$h_t = \sigma(W_h \cdot h_{t-1} + W_e \cdot x_t + b)$$

where:
- $h_t$ = hidden state at time $t$
- $h_{t-1}$ = previous hidden state (initialized as $h_0 = \mathbf{0}$)
- $x_t$ = input embedding at time $t$
- $W_h$ = hidden-to-hidden weight matrix
- $W_e$ = input-to-hidden weight matrix
- $\sigma$ = nonlinearity (sigmoid, tanh)

### Key Properties

| Property | Detail |
|---|---|
| **Weight sharing** | Same $W_h$ and $W_e$ across all time steps |
| **Size independence** | $\dim(W_h) = d_h \times d_h$, $\dim(W_e) = d_e \times d_h$ — independent of sequence length |
| **Theoretically infinite context** | $h_t$ carries info from all previous steps $0 \dots t-1$ |
| **Fair message passing** | $W_e$ is applied to every embedding equally |

### Next-Word Prediction with RNN

1. Embedding lookup: one-hot → pre-trained embedding $e_t$
2. Compute hidden states: $h_1, h_2, \dots, h_T$
3. Final hidden state $h_T$ passed through linear layer $U \in \mathbb{R}^{d_h \times |V|}$
4. Softmax → distribution → sample next word

### Advantages over CNN

- Processes arbitrary-length context
- Can capture long-distance dependencies (ideally)
- Model size doesn't grow with context length
- Symmetric weight application (fair message passing)

### Disadvantages

- **Slow** — sequential computation; must compute $h_1, h_2, \dots$ in order
- **Vanishing gradient** — effect of early words diminishes with distance

---

## 4. Training RNNs

### Loss Function

At each time step $t$, predict the next word and compute cross-entropy loss:

$$J_t(\theta) = -\log \hat{y}_{t}[c]$$

where $c$ is the index of the correct next word (the only position where one-hot $y_t = 1$).

**Total loss:**

$$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} J_t(\theta)$$

### Teacher Forcing

| Paradigm | Input to next step |
|---|---|
| **Normal** | Model's own predicted output |
| **Teacher Forcing** | Ground-truth token (even if model predicted wrong) |

Teacher forcing accelerates training convergence.

### Parameters to Learn

- $W_h$ — hidden-to-hidden
- $W_e$ — input-to-hidden
- $U$ — hidden-to-output (linear projection)

---

## 5. Backpropagation Through Time (BPTT)

### Derivation Using Multivariable Chain Rule

For a function $f(x, y)$ where $x = x(p)$ and $y = y(p)$:

$$\frac{df}{dp} = \frac{\partial f}{\partial x}\frac{dx}{dp} + \frac{\partial f}{\partial y}\frac{dy}{dp}$$

### Applying to RNN

The gradient of loss w.r.t. shared weight $W$:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial h_T} \sum_{k=0}^{T} \frac{\partial h_T}{\partial h_k} \cdot \frac{\partial h_k}{\partial W}$$

**Dependency tree:** $h_T$ depends on $W$ directly AND through $h_{T-1}$, which depends on $W$ through $h_{T-2}$, and so on recursively.

Each position's loss affects $W$, so $W$ is updated multiple times per sequence.

---

## 6. Vanishing & Exploding Gradient Problem

### Root Cause

The term $\frac{\partial h_T}{\partial h_k}$ involves a product of Jacobians:

$$\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

Each $\frac{\partial h_t}{\partial h_{t-1}}$ is a **matrix** (derivative of vector w.r.t. vector).

| Condition | Result |
|---|---|
| Largest eigenvalue of Jacobian $< 1$ | **Vanishing gradient** — early hidden states barely affect the loss |
| Largest eigenvalue of Jacobian $> 1$ | **Exploding gradient** — gradients grow uncontrollably |

### Solutions

| Problem | Solution |
|---|---|
| **Exploding gradient** | **Gradient clipping** — cap gradient norm at a threshold |
| **Vanishing gradient** | **Truncated BPTT** — limit how far back gradients propagate; or use **LSTM/GRU** (next lecture) |

---

## 7. Key Takeaways

- RNNs address CNN's fixed-window and message-passing problems via recurrent computation
- Weight matrices are shared and size-independent of context length
- BPTT enables end-to-end training but suffers from vanishing/exploding gradients
- Teacher forcing feeds ground-truth tokens during training for faster convergence
