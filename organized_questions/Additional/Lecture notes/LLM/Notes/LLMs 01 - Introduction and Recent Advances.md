# Lecture 01 — Introduction and Recent Advances in LLMs

---

## 1. What is a Language Model?

A **language model** is a **probability distribution over a sequence of tokens**.

- Tokens can be: characters, words, subwords, numbers, images
- The model assigns a probability $P(X_1, X_2, \dots, X_L)$ to any token sequence
- A good LM assigns **high probability** to valid sequences and **low probability** to invalid ones

### Chain Rule Decomposition

$$P(X_1, X_2, \dots, X_L) = P(X_1) \cdot P(X_2 \mid X_1) \cdot P(X_3 \mid X_1, X_2) \cdots P(X_L \mid X_1, \dots, X_{L-1})$$

Each conditional $P(X_i \mid X_1, \dots, X_{i-1})$ predicts the next token given the **context** (all preceding tokens).

### Auto-Regressive Language Model

- Input: context tokens → Neural network → Output: probability distribution over vocabulary
- Sample next token from the distribution
- Feed it back as context and repeat

---

## 2. What Makes a Language Model "Large"?

The word **"large"** refers to two things:

| Aspect | Description |
|--------|-------------|
| **Model size** | Number of parameters (weights) in the model |
| **Corpus size** | Number of tokens the model is trained on |

> There is **no fixed threshold** for "large", but conventionally:
> - ≥ 7B parameters → **Large** Language Model
> - < 7B parameters → **Small** Language Model

---

## 3. Evolution of Model Sizes

| Year | Model | Parameters | Notable Feature |
|------|-------|-----------|-----------------|
| 2018 | ELMo | ~94M | Contextualized embeddings |
| 2018 | GPT-1 | 117M | Decoder-only, 512 token context |
| 2018 | BERT | 110M–340M | Encoder-only, Masked LM |
| 2019 | GPT-2 | 1.5B | 1024 token context, zero-shot capability |
| 2019 | T5 | ~1B | Text-to-text transfer |
| 2019 | RoBERTa | ~355M | Better pre-training of BERT |
| 2020 | GPT-3 | 175B | In-context learning, few-shot prompting |
| 2022 | PaLM | 540B | Conversational, by Google |
| 2022 | ChatGPT | — | RLHF-based conversational model |
| 2023 | GPT-4 | ~1.76T (rumored) | Multimodal |
| 2023 | LLaMA 1/2 | 7B–65B | Open-source by Meta |
| 2023 | Mistral 7B | 7B | Small but powerful reasoning |
| 2024 | LLaMA 3 | 8B–70B | Open source by Meta |
| 2024 | GPT-4o | — | Mini version introduced |

> Corpus sizes grew from ~1B tokens (ELMo, 2018) to **trillions** of tokens in modern LLMs.
> Model sizes increased **~5000×** over 4–5 years.

---

## 4. Position of LLMs in AI Landscape

```
AI (rule-based, planning, supervised/unsupervised)
  └── Machine Learning (learns rules from data)
        └── Deep Learning (multi-layer neural networks)
              └── Large Language Models (generative text models)
                    └── Large Multimodal Models (text + image + video)
```

---

## 5. Historical Timeline

| Era | Milestone |
|-----|-----------|
| 1960s | **ELIZA** — first rule-based chatbot (MIT) |
| 1960s | Perceptron introduced |
| 1980s | Multi-layer perceptron + backpropagation |
| 1995–96 | LSTM proposed |
| ~2000 | NVIDIA introduces GPU |
| 2011 | IBM Watson wins Jeopardy |
| 2013 | Facebook AI Research (FAIR) lab |
| 2015 | Google TPU introduced |
| 2016 | SQuAD dataset (Stanford) |
| **2017** | **Transformer** — "Attention Is All You Need" (Google Brain) |

### Post-Transformer Era

| Year | Event |
|------|-------|
| 2017 | Transformer paper by Vaswani et al. |
| 2018 | BERT (Google) — encoder-only, MLM |
| 2018 | GPT-1 (OpenAI) — decoder-only |
| 2019 | GPT-2, T5, RoBERTa, XLM |
| 2020 | GPT-3 — 175B, in-context learning |
| 2021–22 | Megatron-Turing (Microsoft + NVIDIA), Chinchilla, Gopher, Codex |
| 2022 | PaLM (Google), OPT (Meta, open-source) |
| Nov 2022 | **ChatGPT launched** |
| 2023 | Bard (Google), LLaMA (Meta), Claude (Anthropic), GPT-4 |
| 2023 | Mistral AI — small models for reasoning |
| 2023 | Grok (xAI, Elon Musk) |
| Dec 2023 | Google unifies models under **Gemini** |

---

## 6. Key Concepts: Encoder vs Decoder

| Architecture | Focus | Examples |
|-------------|-------|---------|
| **Encoder-only** | Understanding / classification | BERT, RoBERTa |
| **Decoder-only** | Generation | GPT series |
| **Encoder-Decoder** | Seq2seq tasks (translation, summarization) | T5, BART |

---

## 7. Why Study LLMs?

### Emerging Properties
- **In-context learning** — performing new tasks with few examples, no fine-tuning
- **Chain-of-thought reasoning** — step-by-step problem solving
- **Zero-shot / few-shot** generalization

### Failure Modes (equally important)
- **Hallucination** — generating factually incorrect content
- **Bias** — gender, race, cultural biases from training data
- **Toxicity** — offensive outputs from unfiltered web data
- **Security vulnerabilities** — adversarial attacks, prompt injection

---

## 8. Course Modules Overview

| Module | Topics |
|--------|--------|
| **Basics** | NLP intro, statistical LMs, word embeddings, neural LMs (CNN, RNN, Seq2Seq, Attention) |
| **Architecture** | Transformer, encoder/decoder variants, advanced attention (MQA, GQA, sliding window, flash attention), RoPE, Mixture of Experts |
| **Learnability** | Scaling laws, emergent properties, prompting, instruction fine-tuning, alignment (DPO, PPO, KTO), knowledge distillation, PEFT (LoRA, QLoRA) |
| **User Acceptability** | RAG, multilingual models, vision-language models, reasoning, long-context (code), model editing |
| **Ethics & Safety** | Bias, toxicity, hallucination, interpretability (mechanistic & non-mechanistic) |
| **State Space Models** | Mamba, integrating RNN/HMM-style models with Transformer architecture |

---

## 9. Key Takeaway

> *"Never believe in any hypothesis until you experience it through experiments."*

- Hundreds of papers emerge daily with contradictory claims
- Always validate claims empirically before accepting them
