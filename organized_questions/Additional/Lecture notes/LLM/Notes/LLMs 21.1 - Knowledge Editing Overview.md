# Lecture 21.1: Knowledge Editing — Overview & Motivation

---

## 1. Motivation: Why Knowledge Editing?

LLMs become **outdated** over time. Example:
- Query: "Did Messi win the FIFA World Cup?"
- Outdated model (trained ≤ 2021): "No" ← **Incorrect** (Argentina won in 2022)

### Why Not Just Fine-Tune?

| Problem | Details |
|---|---|
| **Cost** | LLaMA training: 21 days, ~2800 GPUs, >$2.4M USD, 1000 tons CO₂ |
| **Catastrophic Forgetting** | Unconstrained fine-tuning on new facts → model forgets pre-trained knowledge |
| **Frequency** | Cannot fine-tune every time a fact changes |

**Alternative**: Treat the LLM as a **knowledge base** and directly **edit** specific facts.

---

## 2. Knowledge Representation

Facts stored as **triplets** $(s, r, o)$:
- $s$ = Subject, $r$ = Relation, $o$ = Object
- Example: (Space Needle, location, Seattle)

**Goal**: Update $(s, r, o) \rightarrow (s, r, o^*)$
- Example: (US, president, Barack Obama) → (US, president, Joe Biden)

> Whether LLMs internally maintain knowledge graph-like structures is **not confirmed**.

---

## 3. Three Evaluation Criteria

### 3.1 Reliability

$$\forall\; t \in D_X: \quad f_{\theta'}(t) = y_{\text{alt}}$$

- $D_X$ = set of **targeted queries** to update
- $\theta'$ = updated parameters
- The edited model must return the **updated output** $y_{\text{alt}}$ for all targeted queries

### 3.2 Generalization

$$\forall\; t \in P_X: \quad f_{\theta'}(t) = y_{\text{alt}}$$

- $P_X$ = **paraphrased versions** of $D_X$ ($P_X \supseteq D_X$)
- Example: "The president of the US is ___" ↔ "Who is the president of the US?"
- Model must return updated output for **all paraphrases**

### 3.3 Localization

$$\forall\; t \in O_X: \quad f_{\theta'}(t) = y_{\text{old}}$$

- $O_X$ = **non-targeted queries** (everything not in $D_X$ or $P_X$)
- Edits must be **localized** — unrelated knowledge remains unchanged
- Example: Updating US president → "President of Russia is ___" must still return "Putin"

---

## 4. Three Categories of Knowledge Editing Methods

### 4.1 Global Optimization

- **Update**: Entire parameter space $W \rightarrow W + \Delta W$
- Learn $\Delta W$ with constraints
- **Example method**: Knowledge Editor (KE)

### 4.2 External Memorization

- **Add** external caching memory alongside LLM parameters
- Updated knowledge stored in external memory
- When a query matches updated knowledge → retrieve from cache
- **Example method**: GRACE

### 4.3 Local Modification

- **Locate** the specific neuron/parameter responsible for the knowledge
- **Update only that parameter**
- Most targeted but most complex
- **Example method**: ROME

### Comparison

| Category | What Changes | Where | How |
|---|---|---|---|
| Global Optimization | All parameters | Internal | $W + \Delta W$ via constrained optimization |
| External Memory | New parameters added | External | Caching/codebook lookup |
| Local Modification | Specific parameter | Internal | Locate neuron → edit in-place |

---

## 5. Methods Covered in This Lecture Series

| Method | Category | Paper |
|---|---|---|
| **Knowledge Editor** | Global Optimization | Lec 21.2 |
| **ROME** | Local Modification | Lec 21.3 |
| **GRACE** | External Memory | Lec 21.4 |
