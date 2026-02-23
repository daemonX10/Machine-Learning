# Lecture 38: Interpretability of LLMs

## 1. Motivation

| Reason | Description |
|---|---|
| **Safety** | Understand failure modes before deployment |
| **Accountability** | Explain decisions in regulated domains (healthcare, finance) |
| **Debugging** | Diagnose unexpected model behavior |
| **Model editing** | Targeted fixes without full retraining |
| **Regulation** | EU AI Act and similar frameworks require explainability |

### Interpretability vs. Explainability

| Term | Meaning |
|---|---|
| **Interpretability** | Understanding the model's **internal mechanisms** — how it arrives at outputs |
| **Explainability** | Providing **post-hoc human-readable justifications** — may not reflect true internals |

---

## 2. Local Explanation Methods

Explain **individual predictions** — why did the model produce this specific output for this specific input?

### 2.1 Feature Attribution

Assign an importance score to each input feature (token).

#### LIME (Local Interpretable Model-agnostic Explanations)

1. Perturb input by removing/masking tokens
2. Get model predictions on perturbed inputs
3. Fit a **local linear model** (interpretable surrogate) on the perturbation-prediction pairs
4. Linear coefficients = feature importances

$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

- $f$: original model
- $g$: interpretable surrogate (linear model)
- $\pi_x$: proximity kernel weighting samples near $x$
- $\Omega(g)$: complexity penalty (encourages sparsity)

#### SHAP (SHapley Additive exPlanations)

Based on **Shapley values** from cooperative game theory:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

- $\phi_i$: contribution of feature $i$
- Sum over all coalitions $S$ not containing $i$
- Measures marginal contribution of $i$ across all possible subsets

**Properties:** Local accuracy, missingness, consistency — the only method satisfying all three axioms.

### 2.2 Attention-Based Explanations

Use attention weights as a proxy for importance:
- Visualize which tokens the model attends to
- **Caveat:** attention ≠ explanation. Attention is a distribution over values for computation, not necessarily a faithful indicator of importance.

### 2.3 Example-Based Methods

- **Adversarial examples:** find minimal input perturbations that change the output
- Reveal decision boundaries and model sensitivity

### 2.4 Self-Explanation

Prompt the model to explain its own reasoning (e.g., chain-of-thought). Not guaranteed to reflect true internal computation.

---

## 3. Global Explanation Methods

Understand **what the model has learned overall** — its internal representations and computational structure.

### 3.1 Linear Probing

**Core idea:** Train a simple **linear classifier** on intermediate hidden states to predict metadata labels.

```
Input → [Frozen LLM] → hidden states at layer l → [Trainable Linear Classifier] → label (e.g., sentiment, POS tag)
```

| If probe succeeds | If probe fails |
|---|---|
| The information is linearly encoded at layer $l$ | The information is either not present or encoded non-linearly |

**Applications:** Detect what information is available at each layer — syntactic features in early layers, semantic features in later layers.

---

### 3.2 Mechanistic Interpretability

#### The Residual Stream View

A transformer can be viewed as a **residual stream** where each layer adds information:

$$x_l = x_{l-1} + \text{Attn}_l(x_{l-1}) + \text{MLP}_l(x_{l-1} + \text{Attn}_l(x_{l-1}))$$

- The residual stream carries information from all previous layers
- Each attention head and MLP block **reads from** and **writes to** this stream

#### Circuits

A **circuit** = a subgraph of the model's computation that implements a specific behavior.

Example: An **induction head** circuit:
1. Head A (earlier layer): detects pattern `[A][B]` and copies `B` to residual stream
2. Head B (later layer): attends to previous `[B]` → predicts the token that followed `[B]` before

#### The Linear Representation Hypothesis

Features (concepts) are represented as **directions** in activation space:

$$\text{concept} \approx \text{direction } \vec{v} \in \mathbb{R}^d$$

The presence/strength of a concept in a hidden state $h$ is measured by $h \cdot \vec{v}$.

#### Superposition Hypothesis

Models represent **more features than dimensions** by encoding features in overlapping (superposition) directions:

$$\text{number of features} \gg \text{dimension of hidden state}$$

This leads to:

| Type | Description |
|---|---|
| **Polysemantic neurons** | A single neuron activates for multiple unrelated concepts |
| **Monosemantic neurons** | A single neuron activates for exactly one concept (rare) |

Superposition enables a model to track many more concepts than its hidden dimension would naively allow, at the cost of some interference.

---

### 3.3 Sparse Autoencoders (SAEs)

**Goal:** Disentangle the superposed features into a higher-dimensional space where each dimension represents one clean feature.

#### Architecture

$$h \xrightarrow{\text{Encode}} z = \text{ReLU}(W_e h + b_e) \xrightarrow{\text{Decode}} \hat{h} = W_d z + b_d$$

| Property | Value |
|---|---|
| Input dim | $d$ (model hidden size) |
| Latent dim | $D \gg d$ (e.g., 10-100×) |
| Sparsity | L1 penalty on $z$ |

$$\mathcal{L} = \|h - \hat{h}\|^2 + \lambda \|z\|_1$$

- **Reconstruction loss** ensures the SAE preserves the information in $h$
- **L1 sparsity** ensures each input activates only a few features → monosemantic features

#### SAE Variants

| Variant | Description |
|---|---|
| **Per-layer SAE** | One SAE per transformer layer |
| **Shared SAE** | Single SAE applied across multiple layers |
| **Sparse Crosscoders** | SAEs that span multiple layers, capturing cross-layer features |
| **Transcoders** | Replace MLP blocks entirely — input is pre-MLP activation, output approximates MLP output. Directly reveals what the MLP computes |

#### Anthropic's Results

- Trained SAEs on Claude and identified millions of monosemantic features
- Features correspond to interpretable concepts: cities, programming languages, emotional states, safety-relevant patterns
- Features are **causally effective** — clamping them changes model behavior

---

### 3.4 Logit Lens

**Idea:** At each intermediate layer, multiply the hidden state by the **unembedding matrix** to peek at what tokens the model is "thinking about."

$$\text{logits}_l = W_{\text{unembed}} \cdot h_l$$

Then look at the **top-ranked tokens** or track how a specific token's rank changes across layers.

| Layer | Typical Observation |
|---|---|
| Early layers | Top tokens are noisy / related to surface patterns |
| Middle layers | Semantic candidates emerge |
| Final layers | Correct next-token prediction stabilizes |

**Tuned Lens:** Add a learned linear transformation per layer for better calibration:

$$\text{logits}_l = W_{\text{unembed}} \cdot (\text{Affine}_l(h_l))$$

---

### 3.5 Activation Patching

**Goal:** Determine the **causal role** of specific activations — moves beyond correlation to causation.

#### Method

1. Run model on **clean** input → get activations
2. Run model on **corrupted** input (e.g., change a key word) → get (different) activations
3. **Patch:** replace one specific activation in the corrupted run with its clean counterpart
4. If patching **restores** the correct output → that activation is **causally responsible**

```
Clean:     "The Eiffel Tower is in" → "Paris"     (activations: a_clean)
Corrupted: "The Colosseum is in"    → "Rome"      (activations: a_corrupted)

Patch a_clean[layer l, position p] into corrupted run:
  If output becomes "Paris" → activation at (l, p) causally encodes "Eiffel Tower → Paris"
```

### Activation Patching vs. Attention Analysis

| Method | Measures |
|---|---|
| Attention weights | Correlation — where does the model look? |
| Activation patching | **Causation** — what computations actually matter? |

---

### 3.6 Activation Steering

**Goal:** Control model behavior at inference time by adding a **steering vector** to its activations.

#### Computing the Steering Vector

1. Collect $n$ examples expressing concept $C^+$ (e.g., "be very polite")
2. Collect $n$ examples expressing concept $C^-$ (e.g., "be rude")
3. Run both through the model, extract activations at a chosen layer $l$
4. Compute:

$$\vec{v}_{\text{steer}} = \frac{1}{n}\sum_i h_l^{(C^+_i)} - \frac{1}{n}\sum_i h_l^{(C^-_i)}$$

#### Applying the Steering Vector

At inference, add the scaled vector to the target layer's activations:

$$h_l' = h_l + \alpha \cdot \vec{v}_{\text{steer}}$$

- $\alpha > 0$: steer toward $C^+$
- $\alpha < 0$: steer toward $C^-$
- $|\alpha|$: controls strength

**Applications:**
- Make model more/less formal
- Increase/decrease refusal behavior
- Steer toward specific topics or styles
- Control truthfulness, helpfulness, harmlessness

---

## 4. Summary & Comparison

| Method | Scope | Type | Key Insight |
|---|---|---|---|
| LIME | Local | Post-hoc | Linear surrogate around input |
| SHAP | Local | Post-hoc | Game-theoretic fair attribution |
| Attention | Local | Intrinsic | Not a faithful explanation |
| Linear Probing | Global | Diagnostic | What info exists at each layer |
| Circuits | Global | Mechanistic | Subgraphs implementing behaviors |
| SAEs | Global | Mechanistic | Disentangle superposed features |
| Logit Lens | Global | Diagnostic | Token predictions per layer |
| Activation Patching | Global | Causal | Which activations matter |
| Activation Steering | Global | Interventional | Control behavior via vectors |

### Timeline of Key Developments

| Year | Development |
|---|---|
| 2013 | Saliency maps for neural networks |
| 2016 | LIME |
| 2017 | SHAP, attention visualization |
| 2020 | Probing classifiers widespread |
| 2021 | Logit lens, early circuits work |
| 2022 | Induction heads, mechanistic interpretability formalized |
| 2023 | Sparse autoencoders (Anthropic), activation patching, transcoders |
| 2024-25 | Large-scale SAE dictionaries, activation steering, crosscoders |
