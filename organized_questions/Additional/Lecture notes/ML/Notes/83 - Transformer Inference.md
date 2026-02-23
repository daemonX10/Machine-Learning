# Transformer Inference

## Overview

- **Training vs Inference:** The encoder behaves **identically** in both stages; the **decoder** behaves differently
- **Training:** Decoder is **non-autoregressive** — all target tokens fed simultaneously
- **Inference:** Decoder is **autoregressive** — generates one token per time step, feeds it back as input for the next step

| Stage | Encoder | Decoder |
|-------|---------|---------|
| Training | Standard processing | Non-autoregressive (parallel) |
| Inference | Standard processing (same as training) | **Autoregressive** (sequential, step-by-step) |

---

## Setup

- Task: English → Hindi machine translation
- Model: Fully trained transformer (weights and biases are fixed)
- Query: "We are friends" → predict Hindi translation

---

## Step 1: Encoder Processing (Same as Training)

The input English sentence passes through the encoder exactly as during training:

```
"We are friends"
  → Tokenize: [We, are, friends]
  → Embedding (512-d per token)
  → + Positional Encoding
  → Multi-Head Self-Attention → Add & Norm
  → Feed-Forward → Add & Norm
  → Repeat through 6 encoder blocks
  → Output: 3 context vectors (one per token, each 512-d)
```

Each context vector captures how its word relates to all other words in the sentence. These encoder outputs are **reused at every decoder time step** in the cross attention layer.

---

## Step 2: Decoder — Autoregressive Generation

### Time Step 1: Generate First Word

**Input:** `<SOS>` (Start of Sentence token)

#### 2.1 Embedding + Positional Encoding

$$x_1 = \text{Embed}(\text{<SOS>}) + \text{PE}(0) \quad \in \mathbb{R}^{512}$$

#### 2.2 Masked Self-Attention

Only one token exists, so self-attention is trivial:

- Compute $Q_1 = x_1 \cdot W_Q$, $K_1 = x_1 \cdot W_K$, $V_1 = x_1 \cdot W_V$
- $\text{score} = Q_1 \cdot K_1^T$ → single scalar
- Scale by $\frac{1}{\sqrt{d_k}}$, apply softmax → attention weight = 1.0
- Output: $z_1 = \text{weight} \cdot V_1$

**Add & Norm:** $z_1^{\text{norm}} = \text{LayerNorm}(z_1 + x_1)$

#### 2.3 Cross Attention

- **Query:** from decoder → $q_{\text{SOS}} = z_1^{\text{norm}} \cdot W_Q$
- **Keys:** from encoder → $k_{\text{we}}, k_{\text{are}}, k_{\text{friends}}$
- **Values:** from encoder → $v_{\text{we}}, v_{\text{are}}, v_{\text{friends}}$

Compute attention scores:

$$w_1 = q_{\text{SOS}} \cdot k_{\text{we}}^T, \quad w_2 = q_{\text{SOS}} \cdot k_{\text{are}}^T, \quad w_3 = q_{\text{SOS}} \cdot k_{\text{friends}}^T$$

Scale, softmax, then weighted sum:

$$z_{c1} = w_1' \cdot v_{\text{we}} + w_2' \cdot v_{\text{are}} + w_3' \cdot v_{\text{friends}}$$

**Add & Norm:** $z_{c1}^{\text{norm}} = \text{LayerNorm}(z_{c1} + z_1^{\text{norm}})$

#### 2.4 Feed-Forward Network

$$y_1 = \text{FFN}(z_{c1}^{\text{norm}}) = \text{ReLU}(z_{c1}^{\text{norm}} \cdot W_1 + b_1) \cdot W_2 + b_2$$

**Add & Norm:** $y_1^{\text{norm}} = \text{LayerNorm}(y_1 + z_{c1}^{\text{norm}})$

#### 2.5 Repeat Through 6 Decoder Blocks

$$y_1^{\text{norm}} \xrightarrow{\text{Block 2}} \cdots \xrightarrow{\text{Block 6}} y_{f1}^{\text{norm}} \in \mathbb{R}^{512}$$

#### 2.6 Linear + Softmax → Output

$$\text{logits} = y_{f1}^{\text{norm}} \cdot W_3 + b_3 \quad \in \mathbb{R}^{V}$$

$$P(\text{word}_j) = \text{softmax}(\text{logits})_j$$

Pick the word with highest probability → **"हम"**

---

### Time Step 2: Generate Second Word

**Input:** `[<SOS>, हम]` (previous output appended)

#### Embedding + Positional Encoding

$$x_1 = \text{Embed}(\text{<SOS>}) + \text{PE}(0), \quad x_2 = \text{Embed}(\text{हम}) + \text{PE}(1)$$

#### Masked Self-Attention (with 2 tokens)

Compute Q, K, V for both tokens. Attention score matrix:

|  | `<SOS>` | हम |
|---|---|---|
| **`<SOS>`** | score | **masked (0)** |
| **हम** | score | score |

> **Masking still applies during inference.** Even though both tokens are known (one was predicted by us), masking is necessary to match training behavior. Removing it would cause a **distribution shift** and degrade prediction quality.

- $z_1$: uses only `<SOS>` context
- $z_2$: uses both `<SOS>` and हम context

**Add & Norm** → **Cross Attention** (2 queries × 3 keys) → **Add & Norm** → **FFN** → **Add & Norm**

#### Through 6 Decoder Blocks → Two final vectors

$$y_{f1}^{\text{norm}}, \quad y_{f2}^{\text{norm}}$$

#### Output: Use **Only the Last Vector**

- $y_{f1}^{\text{norm}}$ corresponds to `<SOS>` → already used in time step 1 → **ignored**
- $y_{f2}^{\text{norm}}$ corresponds to हम → **sent to Linear + Softmax**

$$\text{Output} = \arg\max \text{softmax}(y_{f2}^{\text{norm}} \cdot W_3 + b_3) \rightarrow \text{"दोस्त"}$$

> **Critical point:** At every time step, all vectors are carried through the decoder, but only the **last position's vector** is used for prediction.

---

### Time Step 3: Generate Third Word

**Input:** `[<SOS>, हम, दोस्त]` → 3 tokens

- Masked self-attention: 3×3 matrix with upper triangle masked
- Cross attention: 3 queries × 3 keys
- FFN processes 3 vectors
- Through 6 decoder blocks → 3 final vectors
- Only $y_{f3}^{\text{norm}}$ (दोस्त's vector) sent to Linear + Softmax
- **Output: "हैं"**

---

### Time Step 4: Generate Fourth Word

**Input:** `[<SOS>, हम, दोस्त, हैं]` → 4 tokens

- Same pipeline with 4 vectors
- Only $y_{f4}^{\text{norm}}$ (हैं's vector) sent to output layer
- **Output: `<EOS>`** (End of Sentence)

> When `<EOS>` is predicted, inference **stops**.

**Final translation: "हम दोस्त हैं"**

---

## Masking During Inference

A common misconception is that masking is unnecessary during inference since all tokens are "known" (either given or previously predicted).

**Why masking is still required:**

1. The model was **trained with masking** — the learned weights expect masked attention patterns
2. Removing masking introduces a **train-test distribution mismatch**
3. This mismatch degrades prediction quality

| Scenario | Masking Applied? |
|----------|-----------------|
| Training | ✅ Yes (causal mask) |
| Inference | ✅ Yes (same causal mask) |

---

## Token Growth Across Time Steps

| Time Step | Input Tokens | Vectors Through Decoder | Vector Used for Prediction |
|-----------|-------------|------------------------|---------------------------|
| 1 | `<SOS>` | 1 | 1st (only one) |
| 2 | `<SOS>`, हम | 2 | 2nd (last) |
| 3 | `<SOS>`, हम, दोस्त | 3 | 3rd (last) |
| 4 | `<SOS>`, हम, दोस्त, हैं | 4 | 4th (last) |

---

## Complete Inference Flow Diagram

```
Encoder (runs once):
  "We are friends" → [ctx_we, ctx_are, ctx_friends]

Decoder Time Step 1:
  Input: [<SOS>]
  → Masked Self-Attn → Cross Attn (with encoder) → FFN → ×6 blocks
  → Linear + Softmax → "हम"

Decoder Time Step 2:
  Input: [<SOS>, हम]
  → Same pipeline → use last vector → "दोस्त"

Decoder Time Step 3:
  Input: [<SOS>, हम, दोस्त]
  → Same pipeline → use last vector → "हैं"

Decoder Time Step 4:
  Input: [<SOS>, हम, दोस्त, हैं]
  → Same pipeline → use last vector → <EOS> → STOP
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Encoder at inference** | Identical to training; runs once |
| **Decoder at inference** | Autoregressive — one token per time step |
| **Input grows** | Each step appends the previously predicted token |
| **Masking** | Still applied during inference (matches training) |
| **Output selection** | Only the **last token's** vector goes to Linear + Softmax |
| **Stopping criterion** | Generation stops when `<EOS>` token is predicted |
| **Encoder output reuse** | Same encoder context vectors used at every decoder time step |
