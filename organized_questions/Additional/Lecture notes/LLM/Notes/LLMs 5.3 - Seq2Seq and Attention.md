# Lecture 5.3 — Seq2Seq and Attention

---

## 1. Sequence-to-Sequence (Seq2Seq) Problems

### Types

| Type | Example |
|---|---|
| **Many-to-Many** | Machine translation, summarization, dialogue |
| **Many-to-One** | Sentiment classification |
| **One-to-Many** | Image captioning (image → text sequence) |

---

## 2. Seq2Seq Architecture (Encoder-Decoder)

### Structure

Two separate RNNs:
- **Encoder RNN** — processes input (source) sequence, produces final hidden state
- **Decoder RNN** — generates output (target) sequence, initialized with encoder's final hidden state

### Flow

1. Encoder scans input sentence token by token → hidden states $h_1, h_2, \dots, h_n$
2. Final encoder hidden state $h_n$ is passed as the initial hidden state of the decoder
3. Decoder generates output tokens one at a time, starting with a `<START>` token
4. At each step, decoder produces a distribution over vocabulary via linear layer + softmax
5. Decoding stops when `<EOS>` (end-of-sentence) token is generated

### Training (End-to-End)

- **Teacher forcing:** Feed the ground-truth token (not the model's prediction) as input at each decoder step
- Loss at each decoder time step: cross-entropy between predicted distribution and one-hot ground truth
- **Total loss:**

$$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} J_t(\theta)$$

- Backpropagate through **both** decoder and encoder (BPTT) — updates all weights end-to-end

### Conditional Language Model

Seq2Seq is a **conditional language model** — generation is conditioned on the input $X$:

$$P(Y \mid X) = \prod_{t=1}^{T} P(y_t \mid y_1, y_2, \dots, y_{t-1}, X)$$

### Parameters

| Component | Parameters |
|---|---|
| **Encoder** | $W_h$ (hidden-to-hidden), $W_e$ (input-to-hidden) |
| **Decoder** | $W_h'$ (hidden-to-hidden), $W_e'$ (input-to-hidden), $U$ (hidden-to-output) |

- Encoder has **no** output projection $U$ (not predicting at encoder stage)
- If source/target languages differ → different embedding matrices
- If same language (e.g., summarization) → embeddings can be shared

---

## 3. Decoding Strategies

### Greedy Decoding

- At each step, choose the token with **maximum probability**
- **Problem:** Irreversible errors — if a wrong token is chosen, no backtracking possible

### Exhaustive Search

- Generate all $|V|$ tokens at each step, explore all branches
- **Complexity:** $O(|V|^T)$ — computationally infeasible

### Beam Search

**Core idea:** Keep track of top-$K$ hypotheses at each step (instead of 1 or $|V|$)

**Algorithm:**
1. At step 1, generate top-$K$ tokens
2. For each of the $K$ tokens, generate $K$ next tokens → $K^2$ candidates
3. Keep only the top-$K$ candidates by cumulative log-probability
4. Repeat until `<EOS>` is generated
5. Choose the hypothesis with the highest score

**Score for a hypothesis:**

$$\text{score}(y_1, \dots, y_T) = \sum_{t=1}^{T} \log P(y_t \mid y_1, \dots, y_{t-1}, X)$$

**Length normalization:** Longer hypotheses accumulate more negative log-probs → penalized. Fix by normalizing:

$$\text{normalized score} = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t \mid y_1, \dots, y_{t-1}, X)$$

**Complexity:** $O(K^T)$ but $K \ll |V|$ (typically $K = 5\text{–}10$)

---

## 4. Three Persistent Problems with RNN-based Seq2Seq

| # | Problem | Description |
|---|---|---|
| 1 | **Sequential processing** | Cannot access tokens in parallel; must compute $h_t$ after $h_{t-1}$ |
| 2 | **Linear interaction distance** | All consecutive hidden-state pairs treated equally; no distinction based on distance |
| 3 | **Bottleneck problem** | Encoder's final hidden state must compress the **entire** input → information loss |

### The Bottleneck Problem

- The decoder only has access to the **last encoder hidden state**
- This single vector must encode the entire source sequence
- For long sequences, early information is lost due to vanishing gradient and over-smoothing
- Example: In translating, the decoder generating "pie" (which corresponds to the first input word) has no direct access to the first encoder hidden state

---

## 5. Attention Mechanism

**Proposed:** 2014–2016 (Bahdanau et al.)

### Core Idea

On each decoder step, establish **direct connections** from the decoder to **all** encoder states — no bottleneck.

### Terminology

| Term | Corresponds to |
|---|---|
| **Query** $q$ | Current decoder hidden state $s_t$ |
| **Value** $v$ | Encoder hidden states $h_1, h_2, \dots, h_n$ |

---

### Attention Computation (Step by Step)

Given decoder state $s_t$ (query) and encoder states $h_1, \dots, h_n$ (values):

**Step 1 — Attention Scores** (dot product):

$$e_{t,i} = s_t \cdot h_i \quad \text{for } i = 1, \dots, n$$

**Step 2 — Attention Distribution** (softmax):

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{n} \exp(e_{t,j})}$$

**Step 3 — Attention Vector** (weighted sum of encoder states):

$$a_t = \sum_{i=1}^{n} \alpha_{t,i} \cdot h_i$$

**Step 4 — Concatenate and predict:**

$$[s_t; a_t] \xrightarrow{\text{linear + softmax}} P(y_t)$$

Instead of feeding only $s_t$ to the output layer, concatenate $s_t$ with $a_t$.

---

### Key Properties

| Property | Detail |
|---|---|
| **No bottleneck** | Every decoder state has direct access to all encoder states |
| **Reduces vanishing gradient** | Shortcuts provide alternative gradient paths (like skip connections) |
| **No additional parameters** | Vanilla dot-product attention adds zero learnable parameters |
| **Interpretability** | Attention weights show which encoder tokens are relevant for each decoder token → alignment visualization |

### Attention as Alignment

- In statistical MT: separate alignment model needed to reorder tokens
- With attention: alignment is **learned for free** from the attention distribution
- Attention heatmap: dark = high weight → shows which source token aligns to which target token

### Attention as Summary

- The attention vector $a_t$ is a **weighted summary** of all encoder states
- Regardless of encoder sequence length, produces a **single fixed-size vector**
- Can be viewed as "squashing" the encoder sequence into a compact representation

---

## 6. Attention Variants

| Variant | Formula | Notes |
|---|---|---|
| **Dot product** | $e = q \cdot k$ | Simplest; no extra parameters |
| **Bilinear (Luong)** | $e = q^T W k$ | Learnable weight matrix $W$ |
| **Additive (Bahdanau)** | $e = v^T \tanh(W_1 q + W_2 k)$ | Concatenate, linear, nonlinearity, linear |
| **Scaled dot product** | $e = \frac{q \cdot k}{\sqrt{d_k}}$ | Used in Transformers; normalizes by key dimension $d_k$ |

The scaled dot product prevents large dot products from pushing softmax into saturation regions.

---

## 7. Attention Beyond Seq2Seq

- Attention is a **generic deep learning concept** — not restricted to encoder-decoder
- Can be applied to a **single RNN**: each hidden state attends to all previous states
- General formulation: given a query and a set of values, compute importance weights and return a weighted summary

---

## 8. Key Takeaways

- Seq2Seq uses two RNNs (encoder + decoder) trained end-to-end
- Beam search balances greedy efficiency with exhaustive correctness
- The bottleneck problem motivates attention: direct decoder→encoder connections
- Attention provides shortcuts, interpretability, and free alignment with zero extra parameters
- Scaled dot-product attention (used in Transformers) is a normalized variant
