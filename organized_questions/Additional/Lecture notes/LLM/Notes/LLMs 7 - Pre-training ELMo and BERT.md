# LLMs 7 — Pre-training Strategies: ELMo & BERT

## 1. Motivation: Contextual Word Embeddings

> "The complete meaning of a word is always contextual, and no study of meaning apart from a complete context can be taken seriously." — J.R. Firth

### Problem with Static Embeddings (Word2Vec, GloVe)
- Same word → **same vector** regardless of context
- "I **record** the **record**" — two completely different meanings, one embedding
- "A **bat** flew out of the cave" vs. "He hit the ball with a **bat**" — same vector for "bat"

### Solution: Contextual Embeddings
- Embedding of a word **changes depending on its surrounding context**
- Same word in different sentences → different vectors

---

## 2. ELMo (Embeddings from Language Models)

**Paper:** Peters et al., 2018 — "Deep Contextualized Word Representations"

### Architecture Overview
- **Not Transformer-based** — uses stacked **bi-directional LSTMs**
- Two-layer approach:
  1. **Bottom layer**: Language model (pre-trained, frozen)
  2. **Top layer**: Task-specific (fine-tuned)

### Pre-training Phase (Language Model Layer)

1. Initialize a word embedding dictionary (randomly)
2. Train an RNN with **language model objective** (next word prediction)
3. After training, each token's **hidden state** becomes its **contextual embedding**
4. Dictionary entries updated with learned hidden states

**Why contextual?** Hidden state at position $t$ is a function of all previous inputs → encoding context.

### Bidirectional Architecture
- **Forward LM**: Left-to-right LSTM (predicts next word)
- **Backward LM**: Right-to-left LSTM (predicts previous word)
- Both are **stacked 2-layer LSTMs**

### Parameter Sharing

| Parameter | Shared? |
|-----------|---------|
| $\Theta_x$ (input embeddings) | Yes — shared between forward and backward |
| $\Theta_s$ (softmax layer) | Yes — shared between forward and backward |
| $W_h$ (hidden-to-hidden weights) | No — separate for forward and backward |

### Combining Representations

For token $k$ at layer $j$:

$$ELMo_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task} \cdot [\overrightarrow{h}_{k,j};\, \overleftarrow{h}_{k,j}]$$

| Symbol | Meaning |
|--------|---------|
| $\overrightarrow{h}_{k,j}$ | Forward hidden state of token $k$ at layer $j$ |
| $\overleftarrow{h}_{k,j}$ | Backward hidden state of token $k$ at layer $j$ |
| $;$ | Concatenation |
| $s_j^{task}$ | **Layer-specific** learnable weight (learned per task) |
| $\gamma^{task}$ | **Task-specific** scalar scaling factor |

- Only **top-layer** representations are primarily used (most affected by backpropagation)
- Task-specific parameters ($s_j$, $\gamma$) allow slight task-dependent tuning without updating hidden states

### Inference Pipeline
1. Feed sentence into frozen language model layers
2. Obtain contextual hidden states (contextualized by surrounding words)
3. Pass to task-specific layer for prediction

### ELMo Details

| Aspect | Detail |
|--------|--------|
| **Tokenization** | Character-level (2048-char CNN + linear projection) |
| **Purpose of char-level** | Handles unknown/OOV words |
| **Model size** | ~94M parameters |
| **Can be used with Transformers** | Yes — as input embeddings instead of Word2Vec |

### Results
Consistent improvements across multiple tasks:

| Task | Dataset | Improvement |
|------|---------|-------------|
| Question Answering | SQuAD | Significant |
| Natural Language Inference | SNLI | Significant |
| Sentiment Analysis | SST-5 | Significant |
| Named Entity Recognition | CoNLL | Significant |
| Coreference Resolution | — | Significant |

---

## 3. From Pre-trained Word Vectors to Pre-trained Models

### Pre-trained Word Vectors (Word2Vec, GloVe, ELMo)
- Only the **embedding layer** is pre-trained
- Task-specific layers trained from scratch

### Pre-trained Models (BERT, GPT)
- **All parameters** are pre-trained (attention, feed-forward, etc.)
- Fine-tuned on downstream tasks

---

## 4. Three Types of Pre-training Architectures

| Type | Architecture | Models |
|------|-------------|--------|
| **Decoder-only** | Only decoder blocks; no cross-attention | GPT series |
| **Encoder-only** | Only encoder blocks; unmasked self-attention | BERT, ALBERT, RoBERTa |
| **Encoder-Decoder** | Full Transformer | T5, BART |

---

## 5. BERT (Bidirectional Encoder Representations from Transformers)

**Paper:** Devlin et al., Google, 2018

### Architecture
- **Encoder-only** Transformer
- **No masking** in self-attention — every token attends to all others (bidirectional)

### Pre-training Objectives

#### Objective 1: Masked Language Model (MLM)

- Randomly select **15%** of tokens for prediction
- Of the selected 15%:

| Action | Percentage | Purpose |
|--------|-----------|---------|
| Replace with `[MASK]` | 80% | Primary learning signal |
| Replace with random token | 10% | Corruption-based regularization |
| Keep unchanged | 10% | Teach model to predict actual tokens too |

**Why 15%?** Higher masking (e.g., 50%) destroys too much context. 15% is empirically optimal.

**Loss**: Computed **only at masked positions**, not at all tokens.

#### Objective 2: Next Sentence Prediction (NSP)

- Input: Sentence A + Sentence B
- Binary classification: does B follow A?
- **Positive samples**: Actual consecutive sentences
- **Negative samples**: Randomly paired sentences

### Special Tokens

| Token | Purpose |
|-------|---------|
| `[CLS]` | Prepended to input; used for classification tasks |
| `[SEP]` | Separates the two input sentences |

### Why `[CLS]` for Classification?
1. Attends to all tokens via self-attention → aggregates full sequence information
2. **Not biased** toward any specific word (unlike actual words that have self-attention bias)
3. Reduces parameter count: only needs $W \in \mathbb{R}^{d \times C}$ vs. concatenating all tokens

### Input Representation

Three embeddings summed:
1. **Token embedding** (from Word2Vec/random init)
2. **Segment embedding** (A or B — identifies which sentence)
3. **Positional embedding**

### Training Details

| Detail | Value |
|--------|-------|
| **Corpus** | 2.5B words (Wikipedia) + 800M words (BookCorpus) |
| **BERT-Base** | 12 layers, 768 hidden, 12 heads, 110M params |
| **BERT-Large** | 24 layers, 1024 hidden, 16 heads, 340M params |

### Training Loss
Two losses backpropagated jointly:
1. **MLM loss** at masked positions (cross-entropy)
2. **NSP loss** at `[CLS]` position (binary cross-entropy)

---

## 6. Fine-tuning BERT for Downstream Tasks

### Example: Extractive Question Answering (SQuAD)

**Input format:** `[CLS] Question [SEP] Passage [SEP]`

**Two classification heads** added on top of every token:
- $P_B(t_i)$: probability that token $i$ is the **start** of the answer span
- $P_E(t_i)$: probability that token $i$ is the **end** of the answer span

**Answer extraction:**

$$\text{answer} = \arg\max_{i \leq j} P_B(t_i) \cdot P_E(t_j)$$

- Constraint: $j \geq i$ (end must come after start)
- Only the classification heads + encoder weights are fine-tuned

### BERT Performance
State-of-the-art (2018–2020) across:

| Benchmark | Task Type |
|-----------|-----------|
| SQuAD | Question Answering |
| MNLI | Natural Language Inference |
| CoLA | Linguistic Acceptability |
| SST-2 | Sentiment Analysis |
| STS-B | Semantic Similarity |
| MRPC | Paraphrase Detection |

- BERT-Large significantly outperformed GPT-1 and all prior models

---

## 7. BERT Follow-ups

| Model | Key Contribution |
|-------|-----------------|
| **RoBERTa** | BERT is undertrained; more data + longer training + remove NSP |
| **DistilBERT** | Knowledge distillation → smaller, faster BERT |
| **TinyBERT** | Further compression via distillation |
| **ALBERT** | Parameter sharing across layers; factorized embeddings |
| **DeBERTa** | Disentangled attention (separate content and position) |
