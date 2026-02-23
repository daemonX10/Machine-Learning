# Lecture 11 — Scaling Laws

---

## 1. Emergent Abilities in LLMs

### What is Emergence?
- **Not** the complete absence of a capability in smaller models
- Rather, a **sudden, abrupt jump** in performance beyond a certain scale threshold
- Observed in the range of **10B–100B parameters** for most tasks

### Evidence from GPT-3 Paper
- LLMs are **few-shot learners**: performance improves as more demonstration examples are given (0-shot → 1-shot → few-shot)
- This improvement only manifests clearly in **larger models** — sharp jump from 13B → 175B parameters
- Harder tasks (e.g., 4-digit addition) require even larger models to exhibit the spike

### Emergence Across Dimensions

| Scaling Factor | Spike Observed? |
|---------------|----------------|
| Model size (parameters) | Yes — between 10B and 100B |
| Training compute (FLOPs) | Yes — after sufficient training |
| Training steps | Yes — with enough iterations |

### Emergence Score (BIG-Bench)

For a given model across scale points $i$:

$$\text{Emergence Score} = \text{sign}\left(\arg\max_i s_i - \arg\min_i s_i\right) \cdot \sqrt{\left(\frac{\max s_i - \min s_i}{\text{median}(|s_i - s_{i-1}|)}\right)^2}$$

- Positive sign → performance improves with scale (expected)
- Normalized by median step-to-step difference

### Emergence in Context
- **Emergent ability** usually discussed in the context of **in-context learning (ICL)**
- ICL: model performs tasks it was never explicitly trained on — by leveraging latent concept associations from pre-training
- The unseen task must be within the "neighborhood" of known task classes

---

## 2. Three Scaling Factors

| Factor | Symbol | Description |
|--------|--------|-------------|
| Model size | $N$ | Number of non-embedding parameters |
| Training data | $D$ | Total number of tokens |
| Compute | $C$ | Training FLOPs |

### Relationship

$$C \approx 6 \cdot N \cdot D$$

Where the factor 6 accounts for forward + backward pass operations.

### PetaFLOP-days (PF-days)

$$\text{PF-days} = C \times \frac{1}{10^{15} \times 24 \times 3600}$$

### Critical Constraint
When experimenting with one factor, the other two **must not be a bottleneck** — they should be supplied as needed.

---

## 3. Kaplan Scaling Laws (OpenAI, 2020)

### Parameter Counts

| Component | Parameters |
|-----------|-----------|
| Embedding | $(\text{vocab} + \text{context}) \times d_{\text{model}}$ |
| Attention | $3 \times d_{\text{model}}^2$ per layer (Q, K, V) |
| Projection | $d_{\text{model}}^2$ per layer |
| Feed-forward | $2 \times d_{\text{model}} \times d_{\text{ff}}$ per layer |

### The Three Power Laws

**1. Model Size** (fix data & compute, scale $N$):

$$L(N) = \frac{N_c}{N^{\alpha_N}}$$

- $N_c \approx 8.8 \times 10^{13}$
- $\alpha_N \approx 0.076$

**2. Training Data** (fix model & compute, scale $D$):

$$L(D) = \frac{D_c}{D^{\alpha_D}}$$

- $D_c \approx 5.4 \times 10^{13}$
- $\alpha_D \approx 0.095$
- $D$ = total tokens; $D_c$ depends on **tokenization method**
- Kaplan excluded the embedding matrix from $N$

**3. Compute** (scale $C$):

$$L(C) = \frac{C_c}{C^{\alpha_C}}$$

### Power Law Property
- On a **log-log plot**, power laws appear as **straight lines**
- If $L = a \cdot x^{-k}$, then $\log L = -k \log x + \log a$

### The $C_{\min}$ Version (More Reliable)

$$C_{\min} = C \cdot \frac{B_{\text{crit}}}{B}$$

Where **critical batch size**:

$$B_{\text{crit}}(L) = \frac{2 \times 10^8}{L^{1/\alpha_B}} \text{ tokens}$$

- If batch size $B > B_{\text{crit}}$ → diminishing returns (loss can increase)
- $C_{\min}$ scales more reliably than raw $C$

$$L(C_{\min}) = \frac{C_c^{\min}}{C_{\min}^{\alpha_{C_{\min}}}}$$

### Empirical Validation
- Models closely follow the predicted straight line on log-log plots
- Largest scale tested: **~1B parameters** (extrapolated to 100B+ by assumption)
- This is **empirical curve fitting** — not a mathematical guarantee

---

## 4. Bottleneck Experiments

### Data Bottleneck (Fix $D$, Scale $N$)
- Loss **saturates** — larger models don't help
- Larger models **overfit quickly** (memorize training data)
- "Universality of overfitting": regardless of model, insufficient data → same overfitting trap

### Model Bottleneck (Fix $N$, Scale $D$)
- Small models **never show emergence** regardless of how much data you feed
- Need at least ~0.5B–1B+ parameters before power law behavior appears

### Compute Bottleneck (Fix $N$, Scale Compute)
- With insufficient compute, loss may even **increase** before eventually decreasing
- Must have minimum compute threshold before power law kicks in

> **Takeaway**: You must scale **both** parameters and data simultaneously. Scaling only one factor does not yield the power law.

---

## 5. Joint Scaling Law (Kaplan)

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}$$

### Kaplan's Data–Parameter Relationship

$$D \propto N^{0.74}$$

> **Rule of thumb**: If you increase model size **8×**, you need to increase data by **~5×**

### Universality of Training

Joint law also needed for **parameters vs. compute/steps**:

$$L(N, S) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_S}} + \frac{S_c}{S_{\min}}\right]^{\alpha_S}$$

Where $S_{\min}$ = minimum steps to achieve a given loss.

---

## 6. What Doesn't Matter Much

| Factor | Effect on Scaling |
|--------|-------------------|
| **Number of attention heads** | Weak association |
| **Aspect ratio** (depth vs. width) | Not well-modeled by power law |
| **FFN dimension** ($d_{\text{ff}} / d_{\text{model}}$) | Smooth, no sharp emergence |
| **Embedding vs. non-embedding params** | Both show scaling; minimal difference |
| **Dataset composition** | Same power law regardless of data type |

---

## 7. Chinchilla Scaling Law (DeepMind, 2022)

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

Where $E$ = irreducible loss (noise floor / lower bound).

### Fitted Constants

| Constant | Value |
|----------|-------|
| $A$ | 406.4 |
| $B$ | 410.7 |
| $\alpha$ | 0.34 |
| $\beta$ | 0.28 |
| $E$ | 1.69 |

### Key Difference from Kaplan

| | Kaplan | Chinchilla |
|---|--------|-----------|
| Data–Parameter ratio | 8× params → 5× data | **Equal scaling** |
| Embedding matrix | Excluded | **Included** ($N_T = N_E$) |
| Best fit when | — | $\alpha \approx \beta$ |

> **Chinchilla rule**: Scale parameters and data **equally** for optimal results.

### Epoch AI Revision (2024)

| Constant | Chinchilla | Revised |
|----------|-----------|---------|
| $\alpha$ | 0.34 | 0.3478 |
| $\beta$ | 0.28 | 0.3658 |
| $E$ | 1.69 | **1.82** (scaling factor for data is higher) |

---

## 8. Are Emergent Abilities Real? (Stanford, NeurIPS 2023)

### Challenging Finding
- Out of 39 metrics, **only 4 (~10%)** showed emergent spikes
- 90% of metrics showed **no emergence** at all

### The "Harsh Metric" Hypothesis

| Metric Type | % of Emergent Cases | Property |
|-------------|-------------------|----------|
| **Multiple Choice Grade (MCG)** | 76% | Binary — exact match or zero |
| **Exact String Match** | 16% | Binary — character-level exact match |
| Others (BLEU, etc.) | 8% | — |

### When Metrics Are Changed to Continuous:

**Exact string match → Edit distance (Levenshtein)**:
- Rewards partial matches
- Result: emergence **disappears** → near-linear scaling

**MCG → Brier Score**:

$$\text{Brier Score} = \frac{1}{N}\sum(p_i - o_i)^2$$

- Rewards intermediate probabilities instead of all-or-nothing
- Result: emergence **disappears** → near-linear scaling

### The S-Curve Explanation
- Power law on loss: $L \propto N^{-\alpha}$
- Convert to per-token probability: $p = e^{-L}$
- Accuracy for $k$-token output: $\text{Acc} = p^k$ → produces **S-curve**
- The "spike" is just the steep part of an S-curve — not true emergence

### Implication

| Metric Type | Behavior |
|-------------|----------|
| **Hard/discontinuous** (MCG, exact match) | Shows apparent "emergence" |
| **Soft/continuous** (Brier score, edit distance) | **Near-linear scaling** — predictable |

> **Conclusion**: "Emergence" may largely be an **artifact of metric choice**, not an intrinsic property of scale. With soft metrics, performance is predictable without power laws.

---

## 9. Practical Value of Scaling Laws

| Use Case | How Scaling Laws Help |
|----------|----------------------|
| **Budgeting compute** | Predict return on investment for training |
| **Model vs. data trade-off** | Given fixed compute, how to allocate between $N$ and $D$ |
| **Predicting capabilities** | Forecast when abilities emerge at certain scales |
| **Safety/regulation** | Anticipate undesirable abilities before they manifest |
| **Small model research** | Understand whether desirable abilities can emerge earlier |

---

## 10. Key Takeaways

1. **Three factors** drive scaling: model size ($N$), data ($D$), and compute ($C$); related by $C \approx 6ND$
2. **Kaplan laws**: Power law relationship between each factor and loss; must scale $N$ and $D$ jointly ($D \propto N^{0.74}$)
3. **Chinchilla**: Scale $N$ and $D$ **equally**; includes embedding parameters; adds irreducible loss term $E$
4. Model shape (depth, width, heads) has **weak** effect compared to total parameter count
5. **Stanford challenge**: Most "emergence" is an artifact of **discontinuous metrics**; with continuous metrics, scaling is near-linear and predictable
