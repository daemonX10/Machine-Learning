# Lecture 21.4: GRACE — Lifelong Model Editing with Discrete Key-Value Adaptors

---

## 1. Overview

- **Category**: External Memorization
- **Core idea**: Attach an external **codebook** (key-value memory) to a transformer layer. New knowledge is stored in the codebook without modifying original model parameters.
- Supports **lifelong (streaming) editing** — continuously add new knowledge over time

---

## 2. Key Concepts

### 2.1 Knowledge as Spheres in Vector Space

Each stored knowledge entry has:

| Component | Symbol | Description |
|---|---|---|
| Key | $k$ | Embedding representation of the knowledge |
| Value | $v$ | Output representation to propagate to next layer |
| Radius | $\epsilon$ | Span around the key capturing paraphrases |

- In vector space: each knowledge = a **sphere** centered at $k$ with radius $\epsilon$
- The radius captures the **paraphrase neighborhood** of the knowledge

### 2.2 GRACE Adapter

Attached to layer $l$ of the transformer. Two components:

| Component | Purpose |
|---|---|
| **Codebook** $C$ | Stores all edited knowledge as $(k_i, v_i, \epsilon_i)$ triplets |
| **Deferral Mechanism** | If/else logic to decide how to handle new knowledge |

---

## 3. Algorithm

### Input

- $x_t$ = input prompt (e.g., "The president of the US is ___")
- $y_t$ = desired output (e.g., "Joe Biden")
- $\epsilon_{\text{init}}$ = initial radius for new entries
- $D(\cdot, \cdot)$ = distance function (cosine or Euclidean)

### Step 1: Extract Key

$$h^{(l-1)} = F_{l-1}(x_t)$$

Output of layer $l-1$ serves as the **key** for the new knowledge.

### Step 2: Find Nearest Entry in Codebook

$$d_{\text{mean}} = \min_i D(h^{(l-1)}, k_i)$$

$$k_{\text{nearest}} = \arg\min_{k_i \in C} D(h^{(l-1)}, k_i)$$

### Step 3: Deferral Mechanism (Three Cases)

---

#### Case 1: No Overlap → **New Entry**

**Condition**: $d_{\text{mean}} > \epsilon_i + \epsilon_{\text{init}}$  OR  codebook is empty

The new knowledge doesn't fall within any existing entry's span.

**Action**:
1. Initialize $v_{\text{init}}$ randomly
2. Pass $v_{\text{init}}$ through layers $l+1, \ldots, L$
3. Compute loss against $y_t$, backpropagate
4. Learn $v_{\text{new}}$ through fine-tuning (~100 iterations)
5. **Append** $(k = h^{(l-1)},\; v = v_{\text{new}},\; \epsilon = \epsilon_{\text{init}})$ to codebook

```
   ●──ε_init──●  (new entry, no overlap with existing)
                        ●──ε_i──●  (existing entry)
```

---

#### Case 2: Overlap + Same Output → **Expand Radius**

**Condition**: $d_{\text{mean}} \leq \epsilon_i + \epsilon_{\text{init}}$ AND model already outputs $y_t$

The new knowledge overlaps with existing entry AND they agree on the output.

**Action**: Expand the existing entry's radius:

$$\epsilon_i' = \epsilon_i + \epsilon_{\text{init}}$$

No new entry needed — just widen the existing entry's coverage.

```
Before:  ●──ε_i──●      After:  ●────ε_i + ε_init────●
              ●──ε_init──●              (merged)
```

---

#### Case 3: Overlap + Different Output → **Split**

**Condition**: $d_{\text{mean}} \leq \epsilon_i + \epsilon_{\text{init}}$ AND model does NOT output $y_t$

The spheres overlap but represent **different knowledge** (conflicting outputs).

**Action**:
1. **Shrink** existing entry's radius: $\epsilon_i' = \frac{d_{\text{mean}}}{2}$
2. **Add** new entry with radius: $\epsilon_{\text{new}} = \frac{d_{\text{mean}}}{2}$
3. Learn $v_{\text{new}}$ via fine-tuning (same as Case 1)
4. Append $(k = h^{(l-1)},\; v = v_{\text{new}},\; \epsilon = \frac{d_{\text{mean}}}{2})$

```
Before:  ●───ε_i───●
              ●───ε_init───●   (overlap, different outputs)

After:   ●──d/2──|──d/2──●    (separated)
```

---

## 4. Inference with GRACE

When a query comes in at inference:

1. Compute $h^{(l-1)}$ from input
2. Check if $h^{(l-1)}$ falls within any codebook entry's sphere: $D(h^{(l-1)}, k_i) \leq \epsilon_i$
3. **If yes**: return stored $v_i$ → pass to next layer (updated knowledge)
4. **If no**: proceed with original model computation (pre-trained knowledge preserved)

---

## 5. Properties

| Property | Mechanism |
|---|---|
| **Pre-trained knowledge preserved** | Original parameters never modified |
| **Lifelong editing** | New entries appended to codebook over time |
| **Localization** | Only affected queries routed through codebook |
| **Layer selection** | Can attach to any layer (middle/last); combinable with ROME's causal tracing to identify optimal layer |

---

## 6. Results

### Evaluation Metrics

| Metric | Symbol | Measures |
|---|---|---|
| Test Retention | TR | Success rate of **new** edits (reliability) |
| Edit Retention | ER | Retention of **old** edits after new ones (lifelong stability) |

### Performance Comparison (T5 and BART, ~110M params)

| Method | TR | ER | Average |
|---|---|---|---|
| Full Fine-tuning | Low | Low | Low |
| ROME | Good | Moderate | Moderate |
| **GRACE** | **Best** | **Best** | **Best** |

Tested on:
- Zero-shot Relation Extraction dataset
- Legal dataset

---

## 7. Criticisms & Limitations

| Criticism | Response |
|---|---|
| **Extra parameters** (codebook grows) | Codebook size is negligible compared to billion-parameter LLMs |
| **Which layer to attach?** | Add to all layers, or use ROME-style causal tracing to identify optimal layer |
| **Codebook search at inference** | Linear scan; can be optimized with approximate nearest neighbor methods |

---

## 8. Summary

```
GRACE = Codebook (K, V, ε triplets) + Deferral Mechanism (3 if-else cases)

New knowledge → Extract key from layer l-1
             → Compare with codebook entries
             → Case 1: No overlap      → Add new entry, learn V via fine-tuning
             → Case 2: Overlap + agree → Expand existing radius
             → Case 3: Overlap + clash → Shrink existing, add new (halved radius)

Inference: Query → falls in sphere? → Yes: return stored V
                                    → No:  use original model
```

> **Recommended reading**: Survey paper on Knowledge Editing (covers 50+ methods across all three categories).
