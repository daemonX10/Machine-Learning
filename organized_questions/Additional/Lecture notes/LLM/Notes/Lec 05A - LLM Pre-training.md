# Lecture 05A — LLM Pre-training

## 1. From Static Embeddings to Pre-trained Models

| Aspect | Static Embeddings (Word2Vec, GloVe) | Pre-trained Models (BERT, GPT) |
|--------|--------------------------------------|-------------------------------|
| What is pre-trained | Only word/token embeddings | Entire model (all parameters) |
| Context dependency | Context-**independent** (same vector regardless of context) | Context-**dependent** (representation changes with context) |
| Polysemy handling | Single vector per word (can't distinguish "bank" = river vs. financial) | Different representations based on surrounding words |
| Analogy | Dictionary lookup | Contextual understanding |

- **ELMo (2018)**: First method to introduce context-dependent embeddings using deep RNN layers + pre-trained models

---

## 2. Transformer Architecture Variants

| Architecture | Attention Type | Use Cases | Examples |
|-------------|---------------|-----------|----------|
| **Encoder-only** | Non-masked (bidirectional) self-attention | Sentiment analysis, NER, NLI (access to all tokens simultaneously) | BERT |
| **Decoder-only** | Masked (causal) self-attention | Next-token prediction, dialogue, text generation | GPT series |
| **Encoder-Decoder** | Encoder: bidirectional; Decoder: causal + cross-attention | Seq2Seq tasks — MT, summarization | BART, T5 |

> Over time, the community converged on **decoder-only** models for most tasks due to simplicity and scalability.

---

## 3. BERT — Encoder-Only Pre-training

### 3.1 Masked Language Modeling (MLM)

Pre-training uses **self-supervised learning** — no labelled data needed.

**Masking strategy** — select **15%** of token positions:

| Portion | Treatment | Purpose |
|---------|-----------|---------|
| **80%** of 15% | Replace with `[MASK]` token | Core prediction task |
| **10%** of 15% | Replace with a **random** token | Force robustness; harder prediction |
| **10%** of 15% | **Keep original** token (but still compute loss) | Prevent model from forgetting actual tokens |

- Loss is computed **only** at the 15% selected positions using **cross-entropy** between predicted and actual tokens
- The remaining 85% positions: no loss is computed

> **Key insight**: Masking is done at the **position** level, not the word level. Different instances of the same word may or may not be masked.

### 3.2 Next Sentence Prediction (NSP)

| Component | Details |
|-----------|---------|
| **Positive pairs** | $(S_i, S_{i+1})$ — consecutive sentences from corpus |
| **Negative pairs** | $(S_i, S_j)$ where $j \neq i+1$ — random non-consecutive |
| **Ratio** | 50% positive, 50% negative |
| **Task** | Binary classification: does $S_j$ follow $S_i$? |

**Input format**: `[CLS] S_i [SEP] S_j`

**Purpose**:
- MLM captures **intra-sentence** token relationships
- NSP captures **inter-sentence** relationships

### 3.3 Joint Training

Both objectives are trained **simultaneously** in a single forward pass:

```
Input:  [CLS] I am going to [MASK] [SEP] [MASK] is far away
                                    ↓                ↓
                              MLM Loss          MLM Loss
         ↓
    CLS → FFN → Yes/No (NSP Loss)
```

- MLM loss: cross-entropy at masked positions
- NSP loss: binary classification from `[CLS]` token output
- Both losses are **backpropagated end-to-end**

### 3.4 Why the `[CLS]` Token?

| Alternative | Problem |
|-------------|---------|
| Concatenate all token vectors | Large vector ($512 \times T \times 2$ parameters); dummy tokens like `[MASK]`, `[SEP]` contaminate representation |
| Use an existing token (e.g., "I") | **Self-bias** — query of "I" attends most to key of "I", so output is dominated by that word |
| Max-pooling | Selects one specific vector; loses holistic sequence information |

`[CLS]` is a **neutral token** — no semantic bias, but via self-attention it has attended to all other tokens.

### 3.5 BERT Input Representation

The input embedding is the **sum** of three embeddings:

$$E_{\text{input}} = E_{\text{token}} + E_{\text{position}} + E_{\text{segment}}$$

| Embedding | Source | Purpose |
|-----------|--------|---------|
| Token embedding | Word2Vec / random init | Semantic representation |
| Positional embedding | Sinusoidal / RoPE | Token position in full sequence |
| Segment embedding | Learned | Distinguish sentence A vs. sentence B |

### 3.6 BERT Configurations

| Config | Layers | Hidden Dim | Attention Heads |
|--------|--------|-----------|-----------------|
| BERT-Base | 12 | 768 | 12 |
| BERT-Large | 24 | 1024 | 16 |

- **Data**: 1.25B words (Wikipedia + BookCorpus)
- **Optimizer**: Adam

### 3.7 Fine-tuning

After pre-training → task-specific fine-tuning with labelled data:
- Question Answering (SQuAD — extractive QA)
- Named Entity Recognition
- NLI (Natural Language Inference)
- Architecture is adjusted per task (task-specific heads added)

---

## 4. BART — Encoder-Decoder Pre-training

**BART** = **B**idirectional and **A**uto-**R**egressive **T**ransformer

### Input Corruption Techniques (Denoising Objectives)

| Technique | Input Example | Output | Difficulty |
|-----------|--------------|--------|------------|
| **Token masking** | `I [MASK] going [MASK] school` | `I am going to school` | Baseline |
| **Token deletion** | `I going school` | `I am going to school` | Harder (unknown length) |
| **Sentence permutation** | `going I school am to` | `I am going to school` | Hard |
| **Token rotation** | `school I am going to` | `I am going to school` | Moderate |
| **Span infilling** | `I [MASK] to school` (mask covers "am going") | `I am going to school` | Moderate (span length varies) |

- Encoder receives corrupted input; decoder predicts original
- Loss: conditional log-likelihood at decoder level

---

## 5. T5 — Text-to-Text Transfer Transformer

### Core Idea
**Every NLP task** can be framed as a text-to-text problem — including classification and regression.

| Task | Input | Output |
|------|-------|--------|
| Translation | `translate English to German: That is good` | `Das ist gut` |
| CoLA (acceptability) | `cola sentence: The course is jumping well` | `not acceptable` |
| STS-B (similarity) | `stsb sentence1: ... sentence2: ...` | `3.8` (as text) |
| Summarization | `summarize: <document>` | `<summary>` |

### Pre-training: Span Corruption

- Uses **span masking** (same as BART's span infilling)
- Different from BERT: each masked span gets a **unique sentinel token** (`<X>`, `<Y>`, ...)
- Output contains only sentinels + their predicted spans (not full reconstruction)

```
Input:  Thank you <X> me to your party last <Y> week
Output: <X> for inviting <Y> [end]
```

### T5 Configuration

| Aspect | Detail |
|--------|--------|
| Architecture | Encoder-decoder, 12 layers (base) |
| Pre-training objective | Span corruption (denoising) |
| Data | C4 (Colossal Clean Crawled Corpus) — web text, stories, scripts |
| Benchmarks | GLUE, SuperGLUE, SQuAD, CNN/DailyMail, MT (En→De, En→Fr, En→Ro) |
| Multilingual variant | mC4 (multilingual C4) |

---

## 6. Decoder-Only Pre-training (GPT-style)

- **Architecture**: Decoder layers only (causal/masked self-attention)
- **Objective**: Simple **next-token prediction** (autoregressive)
- **No input corruption** needed — just feed text and predict next token

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})$$

- Sum conditional log-likelihoods across all output tokens and maximize

---

## 7. Summary: Pre-training Paradigm

```
Pre-training (self-supervised, unlabelled data)
        ↓
Fine-tuning (task-specific, labelled data)
        ↓
Deployment
```

| Model | Type | Pre-training Objective |
|-------|------|----------------------|
| BERT | Encoder-only | MLM + NSP |
| BART | Encoder-decoder | Denoising (multiple corruption types) |
| T5 | Encoder-decoder | Span corruption with sentinels |
| GPT | Decoder-only | Next-token prediction (autoregressive) |
