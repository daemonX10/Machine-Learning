# Masked Self-Attention (Transformer Decoder)

## Key Statement

> **"The Transformer Decoder is autoregressive at inference time, and non-autoregressive at training time."**

Understanding this sentence is the key to understanding masked self-attention.

---

## Autoregressive Models

### Definition

In the context of deep learning, **autoregressive models** generate data points in a sequence by **conditioning each new output on previously generated outputs**.

$$P(y_t \mid y_1, y_2, \ldots, y_{t-1})$$

### Example: LSTM-Based Encoder-Decoder (Machine Translation)

```
English: "Nice to meet you"

Encoder processes: Nice → to → meet → you → [context vector]

Decoder generates (autoregressive):
  Step 1: <start> + context       → "आपसे"
  Step 2: "आपसे" + context        → "मिलकर"
  Step 3: "मिलकर" + context       → "अच्छा"
  Step 4: "अच्छा" + context       → "लगा"
  Step 5: "लगा" + context         → <end>
```

At each step, the **previous output** becomes part of the **next input**.

### Why Must Seq2Seq Models Be Autoregressive?

- Sequential data is **inherently ordered** — future words depend on past words
- You cannot generate the entire output in one shot
- The next word **depends on** what was already generated

---

## The Training Problem: Sequential = Slow

### Autoregressive Training (Naive Approach)

If the decoder is autoregressive during **training**:

```
Training example: "How are you" → "आप कैसे हैं"

Step 1: <start> + encoder_output → predict "आप"  (wrong: "तुम")
Step 2: "आप" + encoder_output   → predict "कैसे" (teacher forcing)
Step 3: "कैसे" + encoder_output → predict "हैं"  (wrong: "थे")
Step 4: "हैं" + encoder_output  → predict <end>
```

**Note:** Teacher forcing ensures we always feed the **correct previous word** from the dataset (not the model's prediction).

### The Bottleneck

| Scenario | Consequence |
|----------|------------|
| Output sentence has 3 words | Decoder runs **4 times** sequentially |
| Output paragraph has 300 words | Decoder runs **301 times** sequentially |
| Dataset has 100,000 rows | × 100,000 examples → **extremely slow** |

Every internal decoder operation (attention, FFN, normalization) must execute once per time step, sequentially.

---

## The Key Insight: Teacher Forcing Eliminates Dependency

In autoregressive training with teacher forcing:

| Time Step | Input Fed to Decoder | Source of Input |
|-----------|---------------------|-----------------|
| 1 | `<start>` | Known (fixed token) |
| 2 | "आप" | From **dataset** (not from step 1's output!) |
| 3 | "कैसे" | From **dataset** (not from step 2's output!) |
| 4 | "हैं" | From **dataset** (not from step 3's output!) |

> Because of teacher forcing, **the input at each step doesn't depend on the previous step's output** — it always comes from the dataset, which is available from the start.

**Therefore:** All steps can be executed **in parallel** — no need for sequential processing!

This makes the Transformer decoder **non-autoregressive at training time**.

---

## The Data Leakage Problem

### Parallel Processing Exposes Future Tokens

When we send all words simultaneously through self-attention:

$$\text{contextual}(\text{"आप"}) = 0.8 \cdot v_{\text{आप}} + 0.1 \cdot v_{\text{कैसे}} + 0.1 \cdot v_{\text{हैं}}$$

**Problem:** When computing "आप"'s representation, self-attention uses "कैसे" and "हैं" — but those words **don't exist yet** at the time "आप" is generated!

$$\text{contextual}(\text{"कैसे"}) = 0.15 \cdot v_{\text{आप}} + 0.75 \cdot v_{\text{कैसे}} + 0.10 \cdot v_{\text{हैं}}$$

**Problem:** "कैसे" uses "हैं" which comes **after** it.

$$\text{contextual}(\text{"हैं"}) = 0.10 \cdot v_{\text{आप}} + 0.20 \cdot v_{\text{कैसे}} + 0.70 \cdot v_{\text{हैं}}$$

**No problem here:** "हैं" is the last word — all previous words are available.

### Why This Is Unfair

| During Training | During Inference |
|----------------|-----------------|
| Current token can "see" future tokens | Current token has **no future tokens** available |
| Model learns with extra (leaked) info | Model must predict **without** that info |

This is **data leakage** — the model has information during training that it won't have during prediction, leading to poor generalization.

---

## The Dilemma

| Approach | Speed | Data Leakage |
|----------|-------|-------------|
| **Autoregressive** (sequential) | ✗ Very slow | ✓ No leakage |
| **Non-autoregressive** (parallel) | ✓ Very fast | ✗ Data leakage! |

**Goal:** Get the speed of parallel processing **without** data leakage.

**Solution:** **Masked Self-Attention**

---

## How Self-Attention Works (Review)

For sentence "आप कैसे हैं":

### Step 1: Compute Embeddings → Q, K, V Vectors

For each word, compute query ($q$), key ($k$), and value ($v$) vectors:

$$q_i = x_i \cdot W_Q, \quad k_i = x_i \cdot W_K, \quad v_i = x_i \cdot W_V$$

Stack into matrices: $Q$, $K$, $V$

### Step 2: Compute Attention Scores

$$\text{scores} = Q \cdot K^T$$

### Step 3: Scale

$$\text{scaled\_scores} = \frac{Q \cdot K^T}{\sqrt{d_k}}$$

### Step 4: Softmax

$$\text{weights} = \text{softmax}(\text{scaled\_scores})$$

### Step 5: Compute Contextual Embeddings

$$\text{output} = \text{weights} \cdot V$$

The weight matrix (after softmax) looks like:

$$W = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{bmatrix}$$

Contextual embedding for "आप":
$$z_{\text{आप}} = w_{11} \cdot v_{\text{आप}} + w_{12} \cdot v_{\text{कैसे}} + w_{13} \cdot v_{\text{हैं}}$$

---

## The Masking Solution

### Which Weights Cause Data Leakage?

In the weight matrix:

$$W = \begin{bmatrix} w_{11} & \color{red}{w_{12}} & \color{red}{w_{13}} \\ w_{21} & w_{22} & \color{red}{w_{23}} \\ w_{31} & w_{32} & w_{33} \end{bmatrix}$$

- $\color{red}{w_{12}}$: "आप" attending to "कैसे" (future) ← **leak**
- $\color{red}{w_{13}}$: "आप" attending to "हैं" (future) ← **leak**
- $\color{red}{w_{23}}$: "कैसे" attending to "हैं" (future) ← **leak**

**The upper-right triangle** above the diagonal represents attention to future tokens.

### Goal: Zero Out the Upper Triangle

If $w_{12} = w_{13} = w_{23} = 0$:

$$z_{\text{आप}} = w_{11} \cdot v_{\text{आप}} + 0 + 0 = w_{11} \cdot v_{\text{आप}}$$

$$z_{\text{कैसे}} = w_{21} \cdot v_{\text{आप}} + w_{22} \cdot v_{\text{कैसे}} + 0$$

$$z_{\text{हैं}} = w_{31} \cdot v_{\text{आप}} + w_{32} \cdot v_{\text{कैसे}} + w_{33} \cdot v_{\text{हैं}}$$

Now each token only attends to **itself and previous tokens** — no future leakage!

---

## Masking: Step-by-Step

### Step 1: Compute Scaled Attention Scores (as usual)

$$S = \frac{Q \cdot K^T}{\sqrt{d_k}} = \begin{bmatrix} s_{11} & s_{12} & s_{13} \\ s_{21} & s_{22} & s_{23} \\ s_{31} & s_{32} & s_{33} \end{bmatrix}$$

### Step 2: Create Mask Matrix

Same shape as $S$, with $0$ where attention is allowed and $-\infty$ where it must be blocked:

$$M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}$$

### Step 3: Add Mask to Scores

$$S_{\text{masked}} = S + M = \begin{bmatrix} s_{11} & -\infty & -\infty \\ s_{21} & s_{22} & -\infty \\ s_{31} & s_{32} & s_{33} \end{bmatrix}$$

### Step 4: Apply Softmax

Since $\text{softmax}(-\infty) = 0$:

$$W = \text{softmax}(S_{\text{masked}}) = \begin{bmatrix} 1.0 & 0 & 0 \\ w_{21}' & w_{22}' & 0 \\ w_{31}' & w_{32}' & w_{33}' \end{bmatrix}$$

### Step 5: Compute Contextual Embeddings (No Leakage!)

$$z_{\text{आप}} = 1.0 \cdot v_{\text{आप}}$$

$$z_{\text{कैसे}} = w_{21}' \cdot v_{\text{आप}} + w_{22}' \cdot v_{\text{कैसे}}$$

$$z_{\text{हैं}} = w_{31}' \cdot v_{\text{आप}} + w_{32}' \cdot v_{\text{कैसे}} + w_{33}' \cdot v_{\text{हैं}}$$

---

## Complete Masked Self-Attention Formula

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

Where $M$ is the **causal mask** (upper-triangular matrix of $-\infty$).

---

## Summary: Best of Both Worlds

| Property | Without Masking | With Masking |
|----------|----------------|-------------|
| **Processing** | Parallel ✓ | Parallel ✓ |
| **Data Leakage** | Yes ✗ | No ✓ |
| **Future tokens visible** | Yes ✗ | No ✓ |
| **Training speed** | Fast ✓ | Fast ✓ |
| **Inference behavior matches** | No ✗ | Yes ✓ |

---

## Decoder Behavior Summary

| Phase | Mode | Processing | Why |
|-------|------|-----------|-----|
| **Training** | Non-autoregressive | **Parallel** (all tokens at once) | Teacher forcing provides all inputs; masking prevents data leakage |
| **Inference** | Autoregressive | **Sequential** (one token at a time) | Future tokens genuinely don't exist yet |

---

## Key Takeaways

1. **Autoregressive** = each output depends on previous outputs (sequential generation)
2. **Transformers at inference** = autoregressive (must generate one token at a time)
3. **Transformers at training** = non-autoregressive (all tokens processed in parallel)
4. **The mask** = upper-triangular matrix of $-\infty$ added before softmax
5. **$\text{softmax}(-\infty) = 0$** ← the mathematical trick that makes masking work
6. **Result:** Parallel training speed + no data leakage from future tokens
7. **Location in architecture:** First block of the Transformer Decoder = "Masked Multi-Head Attention"
