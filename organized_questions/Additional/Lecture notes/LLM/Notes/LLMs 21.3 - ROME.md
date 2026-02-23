# Lecture 21.3: ROME — Locating and Editing Factual Associations in GPT

---

## 1. Overview

- **Paper**: "Locating and Editing Factual Associations in GPT" (MIT)
- **Category**: Local Modification
- **Key idea**: First **locate** which MLP layer stores a fact via causal tracing, then **edit** only that layer's parameters

---

## 2. Residual Stream View of Transformers

Each token $i$ has a **residual stream** — a pipeline that accumulates information across layers:

$$h_i^{(l)} = h_i^{(l-1)} + a_i^{(l)} + m_i^{(l)}$$

where:
- $h_i^{(l)}$ = hidden state at layer $l$, token $i$
- $a_i^{(l)}$ = attention output (depends on **all tokens**: $h_1^{(l-1)}, h_2^{(l-1)}, \ldots$)
- $m_i^{(l)}$ = MLP output (**independent** per token — point-wise)

### Key Observations

| Component | Cross-token? | Description |
|---|---|---|
| Attention | ✅ Yes | Connects residual streams across tokens |
| MLP | ❌ No | Operates independently per stream |

---

## 3. MLP as Key-Value Memory

Each MLP has two layers:

$$m_i^{(l)} = W_{\text{proj}} \cdot \sigma\!\left(W_{\text{fc}} \cdot \text{input}\right)$$

| Component | Role | Interpretation |
|---|---|---|
| $W_{\text{fc}}$ (up-projection) | Input → hidden | Produces **key** $k$ |
| $\sigma(\cdot)$ | Non-linearity | — |
| $W_{\text{proj}}$ (down-projection) | Hidden → output | Maps key to **value** $v$ |

$$k = \sigma(W_{\text{fc}} \cdot \text{input}), \quad v = W_{\text{proj}} \cdot k$$

> The MLP acts as an **associative memory**: given key $k$, it retrieves value $v$.

---

## 4. Causal Tracing

### 4.1 Three Runs of the Model

**Input**: "The Space Needle is in downtown ___" → Expected output: "Seattle"

| Run | Setup | Output Probability |
|---|---|---|
| **Clean** | Normal input | $P_R$ (high for "Seattle") |
| **Corrupted** | Add Gaussian noise to **subject embeddings** only ("The Space Needle") | $P_R^*$ (low for "Seattle") |
| **Corrupted + Restore** | Corrupted input, but **replace one hidden state** $(h_i^{(l)})$ with its clean version | $P_R^{**}$ |

> Only subject tokens are corrupted — not relation/object tokens.

### 4.2 Measuring Importance

**Total Effect**: $P_R - P_R^*$ (clean vs. corrupted)

**Indirect Effect** (key metric):

$$\text{IE}(i, l) = P_R^{**}(i, l) - P_R^*$$

Higher IE → restoring that hidden state **recovers** more of the correct output → that location is **important** for the fact.

### 4.3 Results: Two Important Regions

Plotting IE across all (layer, token) positions reveals:

| Region | Location | Significance |
|---|---|---|
| **Early Site** | Middle layers (10–30), **last token of subject** | MLP knowledge storage |
| **Late Site** | Last few layers, last token | Output generation (obvious/expected) |

### 4.4 Disentangling MLP vs. Attention

To isolate each component's contribution, use **severing**:

| Setup | What's Clean | MLP | Attention | Result |
|---|---|---|---|---|
| State only | $h^{(l-1)}$ restored | Inherits from clean $h$ | Inherits from clean $h$ | Baseline (purple bars) |
| Sever MLP | $h^{(l-1)}$ restored | **Replaced with corrupted** | Inherits from clean $h$ | Green bars — **major drop** at early layers |
| Sever Attention | $h^{(l-1)}$ restored | Inherits from clean $h$ | **Replaced with corrupted** | Red bars — same as baseline |

**Conclusion**: At early/middle layers, **MLP is the critical component** for factual knowledge. Attention plays a larger role only at late layers.

---

## 5. ROME: The Editing Algorithm

### 5.1 Formulation as Constrained Least Squares

Given the target MLP layer with weight matrix $W$:

$$\min_{\hat{W}} \|WK - V\|$$

**Subject to**: $\hat{W} k^* = v^*$

where:
- $K$ = matrix of all existing keys
- $V$ = matrix of all existing values
- $(k^*, v^*)$ = new key-value pair to inject

### 5.2 Closed-Form Solution

$$\hat{W} = W + \Lambda (C^{-1} k^*)^\top$$

where:
- $C = KK^\top$ (covariance of keys — symmetric matrix)
- $\Lambda$ = Lagrange multiplier:

$$\Lambda = \frac{v^* - W k^*}{(C^{-1} k^*)^\top k^*}$$

> No iterative optimization needed — direct computation.

---

## 6. Computing $k^*$ and $v^*$

### 6.1 Key $k^*$: Average Subject Representation

1. Sample $n$ diverse prompts with the **same subject** (e.g., "The Space Needle")
2. Run each through the transformer up to the target layer
3. Extract MLP's intermediate output (after $W_{\text{fc}}$) at the **last subject token**
4. Average across all $n$ prompts:

$$k^* = \frac{1}{n} \sum_{j=1}^{n} k_{j+s}^{(l)}(x_j)$$

where $j+s$ = last token position of the subject in prompt $x_j$.

### 6.2 Value $v^*$: Optimization via Fine-Tuning

Optimize $v^*$ to produce desired output $o^*$:

$$v^* = \arg\min_v \left[-\log P(o^* \mid v) + \lambda \cdot D_{KL}(F_\theta \| F_{\theta'})\right]$$

Two components:
1. **Likelihood**: maximize probability of desired output $o^*$
2. **KL regularization**: prevent the updated model from deviating too far (similar to DPO)

---

## 7. Results

| Method | Success (Reliability) | Generalization | Localization | Fluency |
|---|---|---|---|---|
| Full Fine-tuning | Baseline | Low | Low | — |
| Knowledge Editor (KE) | Good | Moderate | Good | — |
| MEND | — | — | — | — |
| **ROME** | **Best** | **Best** | **Best** | Consistent |

Tested on GPT-2 XL and GPT-J with metrics combining reliability, generalization, localization, fluency, and consistency.

---

## 8. ROME Pipeline Summary

```
1. LOCATE: Causal tracing → identify critical MLP layer (early site, last subject token)
          ↓
2. COMPUTE k*: Average MLP hidden state across N prompts with same subject
          ↓
3. COMPUTE v*: Fine-tune value vector to produce desired output o* (with KL constraint)
          ↓
4. UPDATE: Closed-form rank-1 update: Ŵ = W + Λ(C⁻¹k*)ᵀ
          ↓
5. RESULT: Single MLP layer modified → new fact injected
```
