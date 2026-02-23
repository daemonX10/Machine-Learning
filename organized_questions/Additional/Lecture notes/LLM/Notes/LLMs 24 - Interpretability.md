# Lecture 24: Interpretability — Demystifying Black-Box Language Models

## 1. Why Interpretability?

- LLMs are **black-box systems** — we don't understand their internal computations
- Needed for: **safety**, **bias detection**, **trustworthiness**, real-world deployment
- NLP interpretability gained traction ~2016 (studying RNNs/LSTMs), then Transformers arrived in 2017 creating a boom
- First BlackboxNLP Workshop: 2018 (EMNLP); ACL later added an interpretability track

---

## 2. Taxonomy of Interpretability Techniques

```
Interpretability
├── Behavior Localization
│   ├── Input Attribution (role of input tokens → output)
│   └── Model Component Attribution
│       ├── Logit Attribution
│       ├── Causal Interventions (activation/attribution patching)
│       └── Circuit Analysis
└── Information Decoding
    ├── Probing
    ├── Decoding in Vocabulary Space (logit lens, patchscopes)
    └── Dictionary Learning (sparse autoencoders)
```

### Earlier Techniques (Pre-2020)
- **Distributional Semantics**: Word2Vec analogies (king − man + woman = queen)
- **Attention Maps**: Visualize which tokens receive most attention
- **Neuron Analysis**: Analyze individual neuron contributions
- **Probing**: Train classifier on hidden states

---

## 3. Probing

### Basic Idea

Train a classifier $g$ on **frozen** model hidden states $h^{(l)}$ to predict a linguistic property $z$:

$$g: h^{(l)} \rightarrow z$$

- If classifier succeeds → layer $l$ **encodes** that property
- Example: Use Part-of-Speech tagging as probe task on BERT

### Information-Theoretic View

Probing measures **mutual information** $I(H; Z)$:
- $H$ = random variable for hidden representations
- $Z$ = random variable for linguistic property
- High $I(H; Z)$ → hidden states carry information about $Z$

### Simple vs. Complex Probes — The Dilemma

| Probe | POS Accuracy | Control Task Accuracy | Selectivity |
|-------|-------------|----------------------|-------------|
| 2-layer MLP | ~85% | ~35% | **High** ✓ |
| 1000-unit MLP | ~92% | ~86% | **Low** ✗ |

- Complex probes may achieve high accuracy via **memorization**, not because the model encodes the information
- If accuracy is high for **both** probe task and control task → probe is **memorizing**

### Control Tasks (Solution)

1. Define probe task (e.g., POS tagging with 40 tags)
2. Define **control task**: randomly assign labels (0–39) to tokens — no structure
3. **Selectivity** = Probe accuracy − Control accuracy
4. A good probe: **high selectivity** (high on real task, low on control)

> Ideally: Probe accuracy ≫ Control task accuracy

---

## 4. Mechanistic Interpretability (MI)

### What is MI?

> *"Attempting to reverse engineer the computations of a Transformer"* — Anthropic (2022)

> *"Reverse engineering the algorithms implemented by neural networks into human-understandable mechanisms"* — ICML 2024 MI Workshop

### Four Definitions (Position Paper, EMNLP)

| Scope | Definition |
|-------|-----------|
| **Narrow Technical** | Finding causal mechanisms — what caused a specific output |
| **Broader Technical** | Any research using model internals (activations, weights) |
| **Cultural** | Work by the self-identified MI community |
| **Broadest** | Any AI interpretability research |

### Origin: From Vision to NLP

- **Chris Olah** (OpenAI → Anthropic) coined the term studying CNNs (ResNet)
- Paper: *"Zoom In: An Introduction to Circuits"*
- Key concept: **Circuits** = computational subgraphs of neural networks

### Polysemantic Neurons

**Problem**: A single neuron encodes **multiple concepts** (e.g., car-detecting neuron responds to windows, car body, and wheels differently)

→ Cannot say "this neuron = this feature" — makes interpretation hard

### Three Claims (Zoom In Paper)

1. **Features** are fundamental units of neural networks (like cells in biology)
2. **Circuits** = features connected by weights
3. **Universality** — analogous features/circuits form across models and tasks (ambitious, criticized)

---

## 5. Finding Circuits — Three Steps

### Step 1: Isolate a Task

Define a behavior, create a dataset, define a metric:

| Task | Description | Metric |
|------|-------------|--------|
| **IOI** (Indirect Object Identification) | "John and Mary went to the store. John gave a drink to ___" → Mary | Logit difference between IO and Subject |
| **Docstring** | Predict next parameter from function | Match accuracy |
| **Induction** | Predict token that followed same context before | Match accuracy |

### Step 2: Define Scope (Granularity)

Choose nodes for the computation graph:
- Attention heads
- MLP layers
- Residual stream states
- Or finer: Q, K, V activations separately

### Step 3: Activation Patching

**Goal**: Determine if component $c$ is important for the task.

**Procedure**:

1. **Clean run**: Forward pass with clean input → cache activation $a$ at component $c$
2. **Corrupt run**: Forward pass with corrupted input (change one token) → get activation $a'$
3. **Patched run**: Run corrupt input, but **replace** $a'$ with clean $a$ at component $c$
4. **Measure**: Change in logit difference (or KL divergence)

```
Clean:   "The Eiffel Tower is in ___"  → Paris (logit for Paris high)
Corrupt: "The Colosseum is in ___"     → Rome (logit for Rome high)
Patched: Corrupt input + clean activation at component c
         → If logit(Paris) ↑ and logit(Rome) ↓ → component c is important
```

**Decision**: If logit change > threshold → component is **part of the circuit**; else prune it.

### ACDC Algorithm (Automatic Circuit Discovery)

1. Represent model as computational graph $G$
2. Sort edges in **reverse topological order** (output → input)
3. For each edge: remove it, compute KL divergence $D_{KL}(P_G \| P_{G \setminus e})$
4. If KL change < threshold → prune edge
5. Result: minimal subgraph (circuit) for the task

**Cost**: $O(n)$ forward passes for $n$ components (expensive!)

---

## 6. Attribution Patching — Linear Approximation

**Goal**: Approximate activation patching without $n$ forward passes.

**Setup**: Model $M$, clean input $I$, metric $P$ (logit difference), activation $a$

$$P(I, a) = f_a(I, a)$$

**Taylor expansion** of the logit-difference function at $a = a_R$ (corrupt activation):

$$f_a(a_C) - f_a(a_R) \approx \frac{\partial f_a}{\partial a}\bigg|_{a=a_R} \cdot (a_C - a_R)$$

**Assumptions**:
- Changing a single activation results in a **small perturbation** (valid for large models like GPT-2)
- Higher-order terms are negligible

**Computational Cost**:

| Method | Forward Passes | Backward Passes |
|--------|---------------|-----------------|
| Activation Patching | $2 + n$ | 0 |
| **Attribution Patching** | **2** | **1** |

**Limitations**:
- Only works when changing a **single token**
- Best at fine granularity (individual attention heads)
- Approximation degrades at coarser levels (full layers, residual streams)

---

## 7. Applications of Circuit Analysis

### 7.1 Induction Heads — Mechanism for In-Context Learning

**Induction head** = a two-head circuit:

| Head | Role |
|------|------|
| **Previous-token head** (Layer 1) | Finds where the current token appeared before in context |
| **Induction head** (Layer 2) | Copies the token that **followed** the previous occurrence |

**Example**: Input: `A B ... A _`
- Head 1: Finds previous `A` → attends to its position
- Head 2: Copies what came after → outputs `B`

**Evidence**:
- 1-layer models: No induction heads → poor in-context learning
- 2+ layer models: Induction heads emerge → in-context learning score ($L_{50} - L_{500}$) drops sharply
- Emergence of induction heads correlates with sudden improvement in ICL

**In-Context Learning Score**: $L_{50} - L_{500}$
- $L_{50}$: loss at 50th token (limited context seen)
- $L_{500}$: loss at 500th token (full context utilized)
- Lower difference → better ICL

### 7.2 Mechanistic Understanding of CoT (IIT Delhi)

For fictional ontology reasoning tasks, three subtasks identified:
1. **Decision** — Choose which entity to focus on
2. **Copying** — Search context for entity, copy related info (like induction heads)
3. **Induction** — Reason from two statements about entities

**Finding**: Functional split in model — first half moves information in residual stream, second half computes answers.

---

## 8. Logit Lens — Decoding in Vocabulary Space

**Idea**: Project intermediate hidden state through the unembedding matrix to see what tokens are "forming":

$$\text{logits}^{(l)} = x_i^{(l)} \cdot W_U$$

Applying softmax gives a distribution over vocabulary at each layer → observe how the output **gradually forms** across layers.

**Applications**:
- Track token predictions at each layer
- Applied to **Vision-Language Models**: see how each image patch is categorized at each layer

---

## 9. Patchscopes — Generalized Probing Framework

**Generalizes** both probing and logit lens using **two models**:

$$M \xrightarrow{\text{extract } a^{(l)}} f(\cdot) \xrightarrow{\text{patch into}} M' \text{ at layer } l'$$

| Special Case | $M$ vs $M'$ | $f$ | Patch Layer $l'$ | Result |
|-------------|-------------|-----|-----------------|--------|
| **Logit Lens** | $M = M'$ | Identity | $l' = L$ (last) | Project hidden state to vocab |
| **Probing** | $M = M'$ | Classifier | $l' = L$ | Predict linguistic property |
| **Cross-Model** | $M \neq M'$ | Adapter | Any | Test if $M$ encodes entity info usable by $M'$ |

**Cross-Model Example**:
- Model $M$ input: "The largest river of India is..."
- Extract activation for "India" at layer $l$
- Model $M'$ input: "The largest city of X is..."
- Patch $M$'s activation into $M'$ at position of X
- If $M'$ outputs correct city → $M$ encodes entity information at layer $l$

---

## 10. Dictionary Learning — Sparse Autoencoders (SAEs)

### Problem: Superposition

- Model embeds **more features** than dimensions: features $\gg d$ (e.g., 1000 features in 512-dim space)
- Features cannot align with basis vectors → **polysemantic neurons**

### Linear Representation Hypothesis

> Interpretable features are represented as **linear directions** in latent space; features activate when embeddings align with these directions.

### Sparse Autoencoders

**Architecture**: For activation $x$ (dimension $d$):

$$f(x) = g(x \cdot W_{\text{enc}}) \quad \text{(dimension } d_f \gg d\text{)}$$
$$x' = f(x) \cdot W_{\text{dec}} \quad \text{(reconstruct back to } d\text{)}$$

**Training Loss**:

$$\mathcal{L} = \underbrace{\|x - x'\|^2}_{\text{reconstruction loss}} + \lambda \underbrace{\sum_i |f_i(x)|}_{\text{sparsity loss (L1)}}$$

- **Unsupervised** — no labels needed
- L1 penalty drives most $f_i$ to zero → sparse, interpretable features
- Example: 512-dim activation → 131,072 sparse features (256× expansion)

### Results (Claude 3 Sonnet — Anthropic, May 2024)

- Extracted millions of features from Claude 3 Sonnet
- **Identified specific features** for: gender bias, hateful content, various concepts
- **Controllability**: Manually increase/decrease feature activation → controls specific behaviors in output
- Demonstrated bias reduction by suppressing identified bias features

---

## 11. Summary Table

| Technique | Type | What it reveals | Cost |
|-----------|------|----------------|------|
| **Probing** | Information Decoding | What information layers encode | Low (train small classifier) |
| **Activation Patching** | Causal Intervention | Which components are causally important | High ($n$ forward passes) |
| **Attribution Patching** | Causal Intervention (approx.) | Same as above, approximate | Low (2 forward + 1 backward) |
| **Logit Lens** | Vocabulary Decoding | Token predictions at each layer | Very low (matrix multiply) |
| **Patchscopes** | Generalized Probing | Cross-model information transfer | Medium |
| **Sparse Autoencoders** | Dictionary Learning | Disentangled, interpretable features | Very high (train SAE) |
