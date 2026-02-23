# Lecture 12.1 — Pre-training of Causal LMs & In-Context Learning

---

## 1. Three Steps to Build ChatGPT

| Step | Phase | Data | What It Does |
|------|-------|------|-------------|
| 1 | **Pre-training** | Billions of tokens of unstructured web text | Injects reasoning & world knowledge |
| 2 | **Instruction Tuning** | Thousands of (instruction, output) pairs | Makes model follow human instructions |
| 3 | **RLHF** | Human preference feedback / reward model | Aligns outputs with human preferences (format, safety) |

> **Pre-training is the most critical step** — all reasoning power comes from here. Instruction tuning and RLHF cannot fix a badly pre-trained model.

---

## 2. Generative Model for Sentences

### Why Not BERT?
- BERT (masked LM) does **not assign probability to sentences**
- No straightforward way to **generate** text from BERT
- Useful for embeddings & classification, not generation

### Autoregressive Factorization

Any joint probability over a token sequence can be decomposed as:

$$P(t_1, t_2, \ldots, t_n) = P(t_1) \cdot P(t_2 \mid t_1) \cdot P(t_3 \mid t_1, t_2) \cdots P(t_n \mid t_1, \ldots, t_{n-1})$$

$$= \prod_{j=1}^{n} P(t_j \mid t_1, \ldots, t_{j-1})$$

- This decomposition is **universally true** (chain rule of probability)
- An **autoregressive model** is one that can efficiently compute each $P(t_j \mid t_{<j})$
- Each $P(t_j \mid \cdot)$ is a distribution over the **vocabulary** (~50K–100K tokens)

### Why Autoregressive?
- **Easy to sample**: Generate one token at a time from multinomial distribution
- **Tractable for text**: Sequences are ~1K–100K tokens (unlike images with millions of pixels)
- **Natural fit** for language: left-to-right generation

### Architecture Requirements

For the $j$-th prediction $P_j$:
- $P_1$: depends only on `<BOS>` (a learned parameter)
- $P_2$: depends on $t_1$ only
- $P_j$: depends on $t_1, \ldots, t_{j-1}$ only
- Must **not** see future tokens $t_{j+1}, \ldots, t_n$

---

## 3. Encoder Architectures for Autoregressive Models

### RNN (Satisfies Criteria, But Slow)
- $h_j = f(h_{j-1}, t_j)$ — each hidden state depends only on past
- **Problem**: Sequential computation — $O(M)$ time for $M$ tokens
- Infeasible for context lengths of 4K–100K+ tokens

### Transformer with Full Attention (BERT-style) — Invalid
- Every token attends to every other token
- $P_j$ can see future tokens → **violates** autoregressive constraint

### Transformer with **Causal Attention Mask** — Correct

| | Full Attention (Encoder) | Causal Attention (Decoder) |
|---|---|---|
| Token $j$ attends to | All tokens | Only tokens $\leq j$ |
| Use case | BERT, classification | GPT, text generation |

**Causal mask** (lower-triangular):

$$\text{Mask}[i][j] = \begin{cases} 1 & \text{if } j \leq i \\ 0 & \text{otherwise} \end{cases}$$

- The only difference between Transformer "encoder" and "decoder" is this mask
- Enables **parallel computation** of all positions (unlike RNN)

---

## 4. Maximum Likelihood Training

Given $n$ i.i.d. observations $S_1, \ldots, S_n$ (sentences):

$$\theta^* = \arg\max_\theta \sum_{i=1}^{n} \log P_\theta(S_i)$$

For each sentence $S = (t_1, \ldots, t_m)$:

$$\log P_\theta(S) = \sum_{j=1}^{m} \log P_\theta(t_j \mid t_1, \ldots, t_{j-1})$$

> **Note**: Tokens within a sentence are **not independent** — only the sentences themselves are i.i.d. observations.

---

## 5. Pre-training in Practice

### Data Curation Pipeline

| Step | Description |
|------|-------------|
| Web scraping | Common Crawl or similar; extract content from HTML |
| Deduplication | Remove duplicate documents |
| Remove non-informative docs | Filter logs, error messages, repetitive noise |
| Remove outlier docs | Documents with broken/unusual token distributions |
| Quality classification | (Optional) Train classifier to score document quality |

> **Why filter?** Non-informative/repetitive content can cause the model to "forget" language modeling and generate noise.

### Model Architecture Decisions

| Decision | Typical Choices |
|----------|----------------|
| Tokenizer | BPE; vocab size 50K (monolingual) or 100K (multilingual) |
| Layers | Varies (12–96+) |
| Token/hidden dimension | Usually equal ($d_{\text{model}} = d_{\text{hidden}}$) |
| Attention heads | Varies per layer count |
| Max sequence length | Training: 4,096; Inference: can differ (e.g., with RoPE) |
| Optimizer | Adam, AdaGrad, Adafactor (never plain SGD) |

### Tokenization for Pre-training

1. Tokenize each document → list of token IDs + `<EOT>` (end of text)
2. **Concatenate all documents** into one massive stream
3. **No padding** — randomly sample contiguous windows of fixed length (e.g., 4,096 tokens)
4. Each window is treated as one training sequence

> All sequences during pre-training have **exactly the same length** — no padding needed.

---

## 6. Pre-trained Model Behavior

### What They Can Do
- **Mimic training data patterns**: If the corpus contains translation examples, the model learns translation-like behavior
- Complete text in the style of the training data (poems, facts, code)

### What They Cannot Do
- **Follow arbitrary instructions**: Giving "write a poem about X" may produce meta-commentary instead of a poem
- **Reason**: Training data is not in instruction-output format — it's raw web text (question papers, articles, logs)
- The model learns to **continue** patterns, not **respond** to commands

### Two Solutions

| Approach | When Applied |
|----------|-------------|
| **In-Context Learning** | Inference time (no weight updates) |
| **Instruction Tuning** | Fine-tuning phase (updates weights) |

Both are **emergent properties** — they only appear in sufficiently large models trained with enough data and compute.

---

## 7. In-Context Learning (ICL)

### Definition
Provide demonstration examples in the prompt → model infers the task and solves a new query.

**Example** (sentiment analysis):
```
Review: "Delicious food" → Sentiment: Positive
Review: "The food is awful" → Sentiment: Negative
Review: "Good meal" → Sentiment: ?
```

The model outputs "Positive" without any parameter updates.

### Improving ICL During Pre-training
- **Cluster similar documents** together in the training corpus
- During pre-training, sample sequences **across related documents**
- This mirrors the in-context format used during inference

### Improving ICL During Inference

#### Example Selection via Embedding Similarity
1. Embed the query $x$ using a sentence encoder (e.g., RoBERTa)
2. Embed all candidate examples
3. Select top-$k$ examples by **cosine similarity** to $x$
4. Feed selected examples + query to the LLM

**Challenge**: Top-$k$ by similarity may yield **non-diverse** examples (all very similar)

#### Contrastive Learning of the Similarity Metric

**Score** each candidate example $(x_i, y_i)$:

$$\text{score}(x_i, y_i) = \log P(y \mid x, x_i, y_i)$$

Where $y$ is the ground-truth output for query $x$.

**Contrastive training**:
- Example with **highest score** → embedding should be **close** to $x$
- Example with **lowest score** → embedding should be **far** from $x$
- Use **triplet loss** or similar contrastive objective

---

## 8. Why Does In-Context Learning Work?

### Theory: Transformers Learn In-Context by Gradient Descent

**Setup**: Linear Transformer (no softmax in attention):

$$\text{Attention}(Q, K, V) = \frac{QK^\top}{\sqrt{d}} \cdot V$$

**Key result**: For linear attention Transformers trained on sequences $(x_1, y_1, \ldots, x_n, y_n, x_{n+1})$ to predict $y_{n+1}$:

> There exist stationary points of the training objective such that the prediction at layer $k$ equals $k$ steps of gradient descent on the in-context examples.

- Layer 1 → 1 step of GD
- Layer 2 → 2 steps of GD
- ...

**Implication**: The Transformer is never explicitly trained to do gradient descent, but GD **emerges naturally** in its layer computations.

### Prompt-Dependent Behavior

| Prompt | Input: "Grapes are ___" |
|--------|------------------------|
| "Apples are red, bananas are yellow" | **purple** (color completion) |
| "Lemons are sour, cranberries are bitter" | **sweet** (taste completion) |

The in-context examples steer the model's implicit "gradient descent" toward different learned functions.

---

## 9. Key Takeaways

1. **Causal attention mask** in Transformers enables autoregressive language modeling (left-to-right generation)
2. Pre-training uses **MLE** on massive concatenated text with no padding
3. Pre-trained models are powerful but **don't follow instructions** — they continue patterns
4. **In-context learning** is an emergent property: provide demos in the prompt → model infers and solves the task
5. ICL can be improved by **clustering training data** (pre-training) or **selecting similar examples** (inference)
6. ICL implicitly performs **gradient descent** in the Transformer's layers
