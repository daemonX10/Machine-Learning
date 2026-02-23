# Lecture 5.2 — Neural Language Models: LSTM and GRU

---

## 1. RNN Applications in NLP (Quick Overview)

### POS Tagging (Sequence Labeling)

- Input: token sequence → RNN hidden states $h_1, h_2, \dots, h_T$
- Each $h_t$ → linear layer → softmax over 46 POS tags
- Sample from distribution (classification task, not generation)

### Sentence Classification (e.g., Sentiment Analysis)

Two approaches:
1. **Aggregate all hidden states** (e.g., mean/max pooling) → single vector → classify
2. **Use only final hidden state** $h_T$ (assumes it encodes full sequence) → classify

---

## 2. The Vanishing Gradient Problem — Illustrated

> "When she tried to print her tickets ... after installing the toner into the printer, she finally printed her ___"

- Correct answer: **tickets** (appears ~7th position, prediction at end)
- Due to vanishing gradient, RNN forgets distant context
- Gradient of loss w.r.t. early hidden states involves product of many Jacobians → shrinks to near-zero

---

## 3. LSTM (Long Short-Term Memory)

### Core Idea

Vanilla RNN has **one state** (hidden state $h_t$). LSTM maintains **two states**:

| State | Symbol | Role |
|---|---|---|
| **Hidden state** | $h_t$ | Acts as **output** |
| **Cell state** | $c_t$ | Acts as **memory** (long-term information storage) |

Three **gates** (vectors with values in $[0, 1]$) control read/write to the cell state.

---

### Gate Definitions

All gates are functions of the **previous hidden state** $h_{t-1}$ and **current input** $x_t$:

$$f_t = \sigma(W_f \cdot x_t + U_f \cdot h_{t-1} + b_f) \quad \text{(Forget Gate)}$$

$$i_t = \sigma(W_i \cdot x_t + U_i \cdot h_{t-1} + b_i) \quad \text{(Input Gate)}$$

$$o_t = \sigma(W_o \cdot x_t + U_o \cdot h_{t-1} + b_o) \quad \text{(Output Gate)}$$

- $\sigma$ = sigmoid → outputs in $[0, 1]$
- Each gate has its **own learnable parameters** $(W, U, b)$

---

### Gate Functions

| Gate | Controls |
|---|---|
| **Forget gate** $f_t$ | How much of **previous cell state** $c_{t-1}$ to keep |
| **Input gate** $i_t$ | How much of **new content** $\tilde{c}_t$ to write to cell |
| **Output gate** $o_t$ | How much of **cell state** $c_t$ to read into hidden state $h_t$ |

---

### Cell Content and State Update

**New candidate content:**

$$\tilde{c}_t = \tanh(W_c \cdot x_t + U_c \cdot h_{t-1} + b_c)$$

**Cell state update** (element-wise operations):

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

- $f_t \odot c_{t-1}$: selectively retain previous memory
- $i_t \odot \tilde{c}_t$: selectively write new content

**Hidden state update:**

$$h_t = o_t \odot \tanh(c_t)$$

---

### Why LSTM Mitigates Vanishing Gradient

- If $f_t = \mathbf{1}$ (all ones) and $i_t = \mathbf{0}$: cell state is **copied unchanged** → information preserved indefinitely
- If $f_t = \mathbf{0}$ and $i_t = \mathbf{1}$: cell state is **completely replaced**
- Gates **dynamically control** memory retention vs. updating

> **Note:** LSTM **reduces** vanishing gradient but does **not guarantee** elimination.

### Parameter Count

LSTM has significantly more parameters than vanilla RNN:
- **4 sets** of weight matrices: $(W_f, U_f)$, $(W_i, U_i)$, $(W_o, U_o)$, $(W_c, U_c)$
- → 8 weight matrices + 4 bias vectors vs. vanilla RNN's 2 weight matrices + 1 bias
- Requires **large amounts of data** to train effectively; otherwise overfitting risk

---

## 4. Alternative Approaches to Vanishing Gradient

### Residual (Skip) Connection

Instead of $F(x) \to F'(F(x))$, pass:

$$\text{output} = F(x) + x$$

- Input $x$ is added to the layer's output
- Ensures the network "remembers" the original input at every layer
- Used extensively in Transformers

### Dense Connection (DenseNet)

- Every layer is connected to **all subsequent layers** (not just the next one)
- $h_1 \to h_2, h_3, h_4, \dots$; $h_2 \to h_3, h_4, \dots$
- Ensures information is never lost but raises the question of how to combine connections

### Highway Connection (Highway Networks)

- Combination of dense connections + gating mechanism
- Gate controls how much to remember from distant states vs. current state

---

## 5. GRU (Gated Recurrent Unit)

**Proposed:** 2014 — simplification of LSTM

### Key Differences from LSTM

| Feature | LSTM | GRU |
|---|---|---|
| States | Hidden state + Cell state | **Hidden state only** |
| Gates | 3 (forget, input, output) | **2 (update, reset)** |
| Parameters | 8 weight matrices | **6 weight matrices** |

---

### Gate Definitions

$$z_t = \sigma(W_z \cdot x_t + U_z \cdot h_{t-1}) \quad \text{(Update Gate)}$$

$$r_t = \sigma(W_r \cdot x_t + U_r \cdot h_{t-1}) \quad \text{(Reset Gate)}$$

| Gate | Controls |
|---|---|
| **Update gate** $z_t$ | Balance between previous hidden state and new content |
| **Reset gate** $r_t$ | How much of previous hidden state to use when computing new content |

---

### GRU Equations

**Candidate hidden content:**

$$\tilde{h}_t = \tanh(W_h \cdot x_t + U_h \cdot (r_t \odot h_{t-1}))$$

- $r_t$ controls how much of $h_{t-1}$ influences the new content

**Hidden state update:**

$$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$

- If $z_t \approx 1$: preserve previous state (ignore new content)
- If $z_t \approx 0$: fully replace with new content
- **Single gate** $z_t$ balances old vs. new (complementary weighting)

---

## 6. LSTM vs GRU — Comparison

| Criterion | LSTM | GRU |
|---|---|---|
| **Parameters** | More (8 weight matrices) | Fewer (6 weight matrices) |
| **Separate memory** | Yes (cell state) | No (only hidden state) |
| **Performance** | Sometimes better | Sometimes better |
| **Practical guideline** | Start with LSTM | Switch to GRU if need efficiency |
| **Data requirement** | Needs more data | Works with less data |

> **No clear winner** — task-dependent. Rule of thumb: start LSTM, try GRU for efficiency.

---

## 7. Bidirectional RNN (BiRNN)

- **Forward RNN:** processes sequence left → right → produces $\overrightarrow{h_t}$
- **Backward RNN:** processes sequence right → left → produces $\overleftarrow{h_t}$
- **Final hidden state:** $h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$ (concatenation)
- **Use case:** When the entire sequence is available (classification, tagging) — NOT for generation

---

## 8. Multi-Layer (Stacked/Deep) RNN

- Stack multiple RNN layers vertically
- Input to layer $l+1$ = hidden state output from layer $l$
- **Difference from BiRNN:** BiRNN has two directions at the same layer; Multi-layer stacks layers where each layer's input is the previous layer's hidden state (not the raw input)

---

## 9. Remaining Problem

Even with LSTM/GRU:
- **Cannot process the sequence in parallel** — must wait for $h_{t-1}$ to compute $h_t$
- This sequential bottleneck is addressed by the **Transformer** architecture (later lectures)
