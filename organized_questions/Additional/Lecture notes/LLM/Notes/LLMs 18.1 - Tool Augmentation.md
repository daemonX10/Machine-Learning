# LLMs 18.1 — Tool Augmentation

## Overview

This lecture introduces how LLMs can be augmented with external tools (calculators, search engines, APIs) to overcome inherent limitations. Part 1 of a 3-part series: **Tool Augmentation → Function Calling → Agentic Workflows**.

---

## LLM Limitations Addressed by Tools

| Limitation | Tool Solution |
|---|---|
| Outdated knowledge (training cutoff) | Web search, retrieval systems |
| Math/arithmetic errors | Calculator, Python interpreter |
| Factual hallucination | QA systems, Wikipedia search |
| Language translation | Translation APIs |
| Temporal reasoning | Calendar tools |

---

## 1. GSM-8K (Grade School Math 8K)

**Dataset**: 8,800 grade school math QA pairs

### Approach: Calculator Annotations
- Annotated solutions with **calculator tool calls** at arithmetic steps
- Format: interleave natural language reasoning with `<<operation>>` calculator annotations

### Results with GPT-3 175B

| Method | GSM-8K Accuracy |
|---|---|
| Direct (no tools) | < 30% |
| Fine-tuned only | ~35% |
| **Verifier approach** (sample 100 solutions, score with classifier head) | ~55% |

**Verifier approach**: Generate 100 candidate solutions → train a classifier (verifier) on top of GPT-3 to score solutions → pick highest-scored answer.

---

## 2. TALM (Tool Augmented Language Models)

**Key Innovation**: General framework for augmenting LMs with arbitrary tools using special tokens.

### Tool Call Format
```
Input → [Tool Name] → [Parameters] → [RUN] → [Result] → Output
```

### Self-Play Data Augmentation
1. Start with small seed set of annotated examples (only $D \ll T$ data points annotated)
2. Fine-tune model on seed set
3. Run model on remaining data; where correct → add to training set
4. Repeat iteratively

### Tools Used
| Tool | Task |
|---|---|
| Web Search | Factoid QA (Natural Questions) |
| Calculator | Math QA |

### Results
- **NQ (factoid QA)**: Only ~150 demonstrations needed (task is simple)
- **Math QA**: ~2,000 examples needed; accuracy jumped from 20% → 50%

### Limitations
- Cannot generalize to different task setups (e.g., chitchat vs. QA)
- Requires task-specific fine-tuning

---

## 3. PAL (Program-Aided Language Models)

**Key Idea**: Replace calculator with **Python interpreter** — LLMs already know Python, so no custom syntax needed.

### Approach
- Create few-shot in-context examples where reasoning is expressed as **Python code**
- Natural language explanation as comments (`#`), executable logic as Python statements
- During inference: LLM generates Python program → execute → return result

### Example
```python
# Roger started with 5 tennis balls
tennis_balls = 5
# He bought 2 cans of 3 tennis balls each
bought = 2 * 3
# Total tennis balls
answer = tennis_balls + bought
print(answer)  # 11
```

### Advantages over Calculator
- No custom syntax or extensive annotations needed (just few-shot examples)
- Handles logical reasoning, not just arithmetic
- Can solve counting, set operations, symbolic problems

### GSM-8K vs GSM-Hard Results

| Method | GSM-8K | GSM-Hard |
|---|---|---|
| Direct (Codex) | 20% | Much lower |
| Chain-of-Thought (Codex) | 65% | Large drop |
| **PAL (Codex)** | **70%** | **Moderate drop** |

**GSM-Hard**: Replace all numbers in GSM-8K with large-magnitude numbers (e.g., 3 → 32,768) to test whether models actually compute or just memorize.

- CoT accuracy drops drastically on GSM-Hard → evidence of memorization
- PAL maintains more robust performance → actual computation via Python

### Limitations
- Still requires task-specific in-context prompts
- Each new task type needs new prompt examples

---

## 4. Toolformer (2023)

**Key Innovation**: Teach LLMs to use **multiple tools in a generalized setting** with minimal human annotation.

### Tools Available

| Tool | Purpose |
|---|---|
| QA System | LLM-based factoid question answering |
| Wikipedia Search | Knowledge retrieval |
| Calculator | Arithmetic operations |
| Calendar | Temporal queries |
| Translation | Language translation |

### Self-Supervised Data Generation Pipeline

**Goal**: Automatically create training data with tool annotations without human intervention.

#### Steps:
1. **Start with raw text corpus** (e.g., CCNet)
2. **Create per-tool prompts** with few-shot examples showing tool call annotations
3. **Sample multiple candidate annotations** for each text passage
4. **Execute tool calls** to get results
5. **Filter candidates** using negative log-likelihood comparison:

$$\Delta = \min\left(\text{NLL}(\text{prefix only}),\ \text{NLL}(\text{prefix + question})\right) - \text{NLL}(\text{prefix + question + answer})$$

- If $\Delta$ is large (NLL drops significantly with tool result) → **good candidate** (tool is helpful)
- If $\Delta$ is small or negative → **bad candidate** (tool not needed or wrong)

6. **Fine-tune model** on filtered annotated corpus

### Results (GPT-J 6B with Toolformer vs GPT-3 175B)

| Task | Toolformer (6B) vs GPT-3 (175B) |
|---|---|
| Temporal | Much better |
| Multilingual | Much better |
| Math | Much better |
| QA | Slightly worse (GPT-3 memorized Wikipedia) |

**Key Insight**: A 6B model with tools can outperform a 175B model without tools on many tasks.

### Contributions
1. Generalized tool use across arbitrary text (not task-specific)
2. Automatic training data creation (no human annotation)
3. Only 5-6 examples per tool needed

---

## 5. WebGPT (OpenAI, 2022)

**Goal**: Train LLMs to use web search like humans — search, browse, quote, synthesize.

### Training Data Collection
1. Built a **custom UI** for human annotators
2. Humans answered questions by:
   - Entering search queries
   - Clicking on results
   - Scrolling through pages
   - **Quoting** relevant text to a scratch pad
   - Stopping search when enough evidence gathered
3. All actions were logged as training data

### Action Space
| Action | Description |
|---|---|
| Search | Enter a search query |
| Click | Open a search result |
| Scroll Up/Down | Navigate within a page |
| Quote | Save text to scratch pad |
| End Search | Move to answer generation |

### Training Pipeline
- **Supervised learning** on human action sequences
- **Rejection sampling** + **RLHF** for quality improvement
- Final answer generated from collected quotes + citations

### Relation to Modern Systems
- This is the foundation behind **ChatGPT's web browsing**
- Inspired **Perplexity AI**: aggregate search results into coherent NL answers with citations

---

## Evolution Summary

| Approach | Annotations Needed | Generalization | Tool Flexibility |
|---|---|---|---|
| GSM-8K | Full dataset | Task-specific | Calculator only |
| TALM | Seed set + self-play | Limited | Any (but task-specific) |
| PAL | Few-shot in-context | Per-task prompts | Python interpreter |
| **Toolformer** | **~5-6 per tool** | **Any context** | **Multiple tools** |
| WebGPT | Human demonstrations | Web search | Search engine |

**Key Progression**: From heavy annotation → self-supervised tool use → generalized multi-tool augmentation.
