# Lecture 05B — LLM Instruction Tuning

## 1. Why Instruction Tuning?

Pre-trained LLMs predict the **next token** — they don't follow instructions.

| Input | Pre-trained Model Output | Desired Output |
|-------|--------------------------|----------------|
| "What is the capital of France?" | "What is the capital of Germany?" (generates more questions) | "Paris" |

**Training Pipeline**:
```
Pre-training (next-word prediction)
    → Instruction Fine-tuning (teach instruction-following)
        → RLHF Alignment (teach safety, quality)
```

---

## 2. GPT Timeline

| Model | Year | Parameters |
|-------|------|-----------|
| GPT-1 | 2018 | 117M |
| GPT-2 | 2019 | 1.5B |
| GPT-3 | 2020 | 175B |
| InstructGPT | 2022 | 175B (instruction-tuned GPT-3) |
| ChatGPT | 2022 | Based on GPT-3.5 |

---

## 3. Instruction Fine-tuning Loss

### 3.1 Encoder-Decoder Models

```
Instruction → Encoder
Response    → Decoder
Loss computed only on decoder (response) tokens
```

### 3.2 Decoder-Only Models

Instruction and response are concatenated into a single input:

```
[Instruction tokens | Response tokens]
```

- **Loss is computed only on response tokens** (instruction tokens are "masked" from loss)

$$\mathcal{L} = -\sum_{t \in \text{response}} \log P(y_t \mid y_{<t}, x)$$

where $x$ = instruction tokens, $y_t$ = response tokens.

### 3.3 Weighted Instruction Tuning (WIT)

Standard practice: instruction loss weight = 0 (loss only on response). WIT shows this is **suboptimal**.

$$\mathcal{L}_{\text{WIT}} = \lambda_p \cdot \mathcal{L}_{\text{instruction}} + \lambda_r \cdot \mathcal{L}_{\text{response}}$$

| Parameter | Range | Role |
|-----------|-------|------|
| $\lambda_p$ (instruction weight) | 0 – 0.6 | Non-zero weight on instruction tokens |
| $\lambda_r$ (response weight) | 0.6 – 1.0 | Primary weight on response tokens |

**Benefits**:
- The model becomes **less sensitive to prompt perturbation** (paraphrased instructions still produce similar outputs)
- Zeroing instruction loss entirely causes model to pay less attention to instruction semantics

---

## 4. Instruction Data Curation Methods

### 4.1 FLAN (Fine-tuned LAnguage Net) — 2021

- **First major instruction-tuning dataset** (Google)
- Took existing NLP task datasets (NLI, QA, summarization) and converted them to instruction format
- Hand-crafted **10 templates per task** to generate diverse instruction phrasing

```
Template: "Does {hypothesis} follow from {premise}? Answer yes or no."
                ↓ applied to NLI data ↓
Instruction: "Does 'The dog is sleeping' follow from 'An animal is resting'? Answer yes or no."
Response: "yes"
```

- Resulting model: **FLAN-T5** — 11B tokens, ~1800 tasks

### 4.2 Super-Natural Instructions

- Hired **NLP practitioners** (not just crowd workers)
- More complex and detailed instructions than FLAN
- Included **positive and negative examples** per task instruction

### 4.3 Self-Instruct (Synthetic Data) — 2022

Automated instruction generation using LLM itself:

```
Step 1: Start with 175 seed tasks (1 instruction + 1 instance each)
Step 2: Sample 8 instructions from pool → prompt GPT-3 → generate new instruction
Step 3: Filter by similarity (ROUGE-L < 0.7 to existing pool)
Step 4: Classification vs Non-classification routing:
        - Classification: output-first prompting (balanced classes)
        - Non-classification: input-first prompting
Step 5: Add to pool, repeat
```

- Self-Instruct data used to train **Alpaca** (Stanford, fine-tuned LLaMA-7B)

### 4.4 Evol-Instruct (WizardLM)

GPT used to evolve instructions along two axes:

| Evolution Type | Direction | Method |
|---------------|-----------|--------|
| **In-depth** | Make harder | Add constraints, increase reasoning steps, add edge cases |
| **In-breadth** | Make diverse | Generate related but different instructions |

```
Original: "Write a Python function to sort a list."
In-depth: "Write a Python function to sort a list of dictionaries 
           by a nested key, handling missing keys gracefully, 
           with O(n log n) time complexity guarantee."
In-breadth: "Write a Python function to find the median of a list."
```

### 4.5 Orca — Knowledge Distillation with Chain-of-Thought

Problem with naive distillation: student model only sees teacher's **final answer** — doesn't learn reasoning.

**Solution**: Add a system prompt to GPT-4 requesting step-by-step reasoning:

```
System: "You are a helpful assistant. Think step-by-step 
         and explain your reasoning before giving the final answer."
User:   "Is 17 a prime number?"
GPT-4:  "To check if 17 is prime, I need to test divisibility 
         by all integers from 2 to √17 ≈ 4.12. 
         17 ÷ 2 = 8.5, 17 ÷ 3 = 5.67, 17 ÷ 4 = 4.25. 
         None divide evenly. Therefore, 17 is prime."
```

- Student learns the **reasoning process**, not just the answer
- Uses **GPT-4** (not GPT-3.5) as teacher for higher quality chain-of-thought

### 4.6 Instruction Back-Translation

For domains with abundant **unlabeled text** but no instruction pairs:

```
Step 1: Train a backward model: text → instruction
Step 2: Apply to unlabeled text corpus → generate instructions
Step 3: (instruction, text) pairs become training data
```

---

## 5. Key Findings

- Even **6%** of data as instruction fine-tuning can boost model accuracy from 54% → 70%
- LLMs are "**quick learners**" for instruction-following — small data yields large gains
- Quality of instruction data matters more than quantity
- Mix of real + synthetic data improves generalization

---

## 6. Summary: Data Curation Landscape

| Method | Data Source | Key Innovation |
|--------|------------|----------------|
| FLAN | Human templates on existing NLP datasets | Template-based task conversion |
| Super-Natural Instructions | NLP practitioners | Detailed instructions + pos/neg examples |
| Self-Instruct | GPT-3 generated | Automated pool expansion from 175 seeds |
| Evol-Instruct | GPT evolved | In-depth + in-breadth evolution |
| Orca | GPT-4 (with CoT system prompt) | Knowledge distillation with reasoning |
| Back-Translation | Unlabeled text → generated instructions | Reverse instruction generation |
