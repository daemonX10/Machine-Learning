# Lecture 01 — Course Introduction (Advances in Large Language Models)

## 1. What is a Language Model?

A **language model (LM)** is a **probability distribution over a sequence of tokens**.

Given a sequence $x_1, x_2, \ldots, x_L$, the LM assigns a probability $P(x_1, x_2, \ldots, x_L)$.

Using the **chain rule of probability**:

$$P(x_1, x_2, \ldots, x_L) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdots P(x_L \mid x_1, \ldots, x_{L-1})$$

- **Statistical LM**: conditional probabilities estimated from corpus counts:
$$P(\text{rains} \mid \text{monsoon}) = \frac{\text{count}(\text{monsoon rains})}{\text{count}(\text{monsoon})}$$

- **Neural LM**: a neural network (feedforward, RNN, etc.) estimates conditional probabilities directly.

---

## 2. AI / ML / DL / LLM / GenAI Hierarchy

| Layer | Description |
|---|---|
| **AI** | Broadest category — includes rule-based systems, search, etc. |
| **Machine Learning** | Subset of AI — learning from data |
| **Deep Learning** | Subset of ML — neural networks with multiple layers |
| **LLM** | Subset of DL — large-scale text models (Transformers) |
| **Generative AI** | Cuts across DL & LLM — includes vision models (diffusion), VLMs, etc. |

> GenAI ⊂ Deep Learning, but GenAI ∩ LLM ≠ ∅ — not all GenAI is LLM (e.g., image diffusion models).

---

## 3. Evolution of LLMs — Timeline

| Year | Milestone | Organization |
|---|---|---|
| 2015 | Google TPU developed; OpenAI founded | Google, OpenAI |
| 2017 | **Transformer** introduced ("Attention Is All You Need") | Google Brain |
| 2018 | **BERT** (encoder-only) | Google |
| 2018 | **GPT-1** (decoder-only) | OpenAI |
| 2019 | GPT-2; **RoBERTa** (re-trained BERT); **XLM** (cross-lingual); **T5** (encoder-decoder) | OpenAI, Meta, Google |
| 2020 | **GPT-3** (175B params) — OpenAI stops open-sourcing | OpenAI |
| 2022 | **PaLM** (540B); **OPT** (open-source suite); **ChatGPT** (Nov 30) | Google, Meta, OpenAI |
| 2023 | **LLaMA**, **GPT-4**, **Claude**, **Mistral-7B**, **Gemini**, **Grok** | Meta, OpenAI, Anthropic, Mistral, Google, X (Twitter) |
| 2024 | **OpenAI o1** — large reasoning models (LRMs); thinking before responding | OpenAI |
| 2025 | **DeepSeek R1** — open-source reasoning model competing with closed-source | DeepSeek |

### Key Architectural Types

| Type | Example | Description |
|---|---|---|
| Encoder-only | BERT | Masked language modeling; good for classification/NLU |
| Decoder-only | GPT series | Autoregressive; next-token prediction; dominant paradigm |
| Encoder-Decoder | T5 | Full seq2seq; text-to-text framework |

---

## 4. From LLMs to Large Reasoning Models (LRMs)

- Standard LLMs struggled with complex **reasoning benchmarks**
- 2024–2025: paradigm shift to **thinking models** that reason before responding
- Key reasoning models: OpenAI o1/o3, DeepSeek R1, Gemini 2.5 Pro, Phi-4, Qwen, Claude 3.7 Sonnet

---

## 5. Course Syllabus (5 Modules)

| Module | Topics |
|---|---|
| **1. Foundations** | Transformer, pre-training strategies, encoder/decoder models, alignment (PPO, DPO, GRPO) |
| **2. Efficiency** | Mixture of Experts (MoE), RoPE, efficient attention, KV caching, PEFT, model compression (quantization, pruning, distillation), efficient inference, test-time scaling |
| **3. Augmentation & Reasoning** | RAG, LLM agents (function calling, MCP, ACP, A2A), LRMs, training to reason via RL |
| **4. Alternative Paradigms** | State space models (Mamba), diffusion-based LMs, hybrid models, multimodal models (VLMs, audio-visual LMs) |
| **5. Theory & Ethics** | Physics of language models, interpretability, fairness & ethics |

---

## 6. Key Takeaways

- **Language model** = probability distribution over token sequences, factored via chain rule
- The field evolved from statistical n-gram models → neural LMs → Transformers → LLMs → LRMs
- **Decoder-only** models (GPT family) emerged as the dominant architecture
- Current frontier: **reasoning models** that think before answering
- Open-source (Meta LLaMA, DeepSeek) vs closed-source (OpenAI, Google) is a major axis of competition
