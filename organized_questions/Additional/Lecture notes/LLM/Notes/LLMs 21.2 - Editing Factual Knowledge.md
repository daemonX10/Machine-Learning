# Lecture 21.2: Knowledge Editor — Editing Factual Knowledge (Global Optimization)

---

## 1. Overview

- **Year**: 2021 (one of the first knowledge editing papers)
- **Category**: Global Optimization
- **Core idea**: Train a **hyper-network** $G_\phi$ to learn parameter updates $\Delta\theta$ that get added to the original LLM parameters $\theta$

$$\theta' = \theta + \Delta\theta$$

---

## 2. Architecture

### 2.1 Two Networks

| Component | Symbol | Role |
|---|---|---|
| Original LLM | $F_\theta$ | Pre-trained model with parameters $\theta$ |
| Hyper-network (Knowledge Editor) | $G_\phi$ | Learns $\Delta\theta$; parameters $\phi$ |

### 2.2 Hyper-Network Architecture

**Input**: $(x, y, a)$ where $x$ = input prompt, $y$ = current output, $a$ = desired updated object

```
(x, y, a) → Embedding → BiLSTM → FFN → h (single vector)
                                          ↓
                    ┌──────────┬──────────┼──────────┬──────────┐
                    ↓          ↓          ↓          ↓          ↓
                  α(FFN)    β(FFN)     γ(FFN)     δ(FFN)     η(FFN)
                    ↓          ↓          ↓          ↓          ↓
                 softmax    softmax      ·          ·       sigmoid
                    ↓          ↓          ↓          ↓          ↓
                  α̂ ⊗ γ    β̂ ⊗ δ       ↓          ↓          η
                 (matrix)  (matrix)      ↓          ↓       (scalar)
```

**Components**:
- **α, β** → passed through softmax → probability distributions
- **γ, δ** → raw vectors
- **α̂ ⊗ γ** and **β̂ ⊗ δ** → outer products → matrices (same shape as gradient)
- **η** → passed through sigmoid → scaling factor ∈ (0, 1)

### 2.3 Gated Gradient Update

The update $\Delta W$ is computed as:

$$\Delta W = \eta \cdot \left(\hat{\alpha} \odot \nabla_W \mathcal{L} + \hat{\beta}\right)$$

where:
- $\nabla_W \mathcal{L}$ = gradient from the original LLM's loss
- $\hat{\alpha}$ = learned gate (element-wise product with gradient → **selective gating**)
- $\hat{\beta}$ = learned bias term
- $\eta$ = learned scaling factor
- Analogous to LSTM/GRU gating mechanisms

---

## 3. Training Objective

### 3.1 Loss Function

$$\min_\phi \sum_{t \in P_X} \mathcal{L}\!\left(F_{\theta + \Delta\theta}(t),\; y_{\text{alt}}\right)$$

**Subject to**:

$$C(\theta, \theta') \leq M$$

where:
- $P_X$ = original + paraphrased queries
- $y_{\text{alt}}$ = desired updated output
- $C$ = constraint function, $M$ = slack hyperparameter

### 3.2 Constraint Functions Tested

| Constraint | Formula | Result |
|---|---|---|
| **KL Divergence** | $D_{KL}(F_\theta \| F_{\theta'}) \leq M$ | **Better** performance |
| **$L_p$ Norm** | $\|\theta - \theta'\|_p \leq M$ | Worse — model deviates too much |

KL divergence preserves output distribution similarity → preferred.

---

## 4. Training Variants

| Variant | Description |
|---|---|
| **Simple KE** | Train only on $D_X$ |
| **KE + PX** | Train on $D_X \cup P_X$ (paraphrases) |
| **KE + Loop** | Iterate updates on $D_X$ until 100% success |
| **KE + PX + Loop** | Combine paraphrases + iterative updates |

---

## 5. Evaluation Metrics

| Metric | Measures | Corresponds To |
|---|---|---|
| **Success Rate** | % of $D_X$ correctly updated | Reliability |
| **Retain Accuracy** | % of $O_X$ unchanged | Localization |
| **Equivalence Accuracy** | % of $P_X$ correctly updated | Generalization |

---

## 6. Results

| Method | Success Rate | Retain Accuracy | Equiv. Accuracy |
|---|---|---|---|
| Full Fine-tuning (1st layer) | 100% | 86% | — |
| Full Fine-tuning (all layers) | 100% | 86% | — |
| KE + $L_2$ Norm | 99% | Low (model deviates) | — |
| **KE + KL + PX** | **98%** | **97–98%** | Improved |

**Key insight**: Full fine-tuning achieves 100% success but **destroys** 14% of pre-trained knowledge (retain = 86%). KE preserves 97–98%.

---

## 7. Error Analysis (Logit Plots)

For each output, measure logit = $\log \frac{p}{1-p}$ for both original and updated models.

### Expected Behavior

| Set | Original Logit | Updated Logit | Pattern |
|---|---|---|---|
| $D_X$ (targeted) | High for old answer | High for new answer | **Flipped** |
| $O_X$ (non-targeted) | Value $v$ | Value $v$ | **On diagonal** (unchanged) |

### Observed Results

| Method | $D_X$ (green dots) | $O_X$ (blue/orange dots) |
|---|---|---|
| **Full Fine-tuning** | ✅ Flipped correctly, no errors | ❌ Orange dots off-diagonal (pre-trained knowledge damaged) |
| **KE + $L_2$** | ✅ Flipped | ❌ Blue dots widely scattered (model too different) |
| **KE + KL + PX** | ✅ Flipped (2% red cross errors) | ✅ Blue dots follow diagonal (knowledge preserved) |
