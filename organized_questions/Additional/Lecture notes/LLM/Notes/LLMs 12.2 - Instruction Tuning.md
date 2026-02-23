# Lecture 12.2 — Instruction Tuning

---

## 1. Why Instruction Tuning?

### The Gap Between Pre-training and Human Expectations

| Pre-trained Model Behavior | What Humans Want |
|---------------------------|-----------------|
| Continues input patterns (may output more questions instead of answers) | Follow instructions → produce answers |
| No control over output format | Verbose, concise, list-format, etc. |
| May generate harmful/irrelevant content | Helpful, safe, task-focused outputs |

### Goals of Instruction Tuning
1. Make the model **follow natural language instructions**
2. Enforce **meta-behaviors** (helpfulness, verbosity control, safety)
3. Bridge the gap between raw pre-trained model and human-usable assistant

### Critical Constraints
- Instruction tuning data must be **diverse** (many tasks, not just one)
  - Single-task data (e.g., only summarization) → catastrophic forgetting of pre-trained capabilities
  - Better: **thousands of tasks**, even if only ~10 examples per task
- Pre-training remains the **most important phase** — instruction tuning only shapes the format/behavior, not the core reasoning

---

## 2. Training Loss Function

### Conditional Log-Likelihood Maximization

Given instruction tokens $\mathbf{x} = (x_1, \ldots, x_m)$ and output tokens $\mathbf{y} = (y_1, \ldots, y_k)$:

$$\mathcal{L}(\theta) = -\sum_{j=1}^{k} \log P_\theta(y_j \mid x_1, \ldots, x_m, y_1, \ldots, y_{j-1})$$

### Decoder-Only Model
- Concatenate instruction + output: $[x_1, \ldots, x_m, y_1, \ldots, y_k, \texttt{<EOT>}]$
- Feed entire sequence through the causal decoder
- **Teacher forcing**: ground-truth output tokens fed as input at each step
- Loss computed **only on output tokens** (not instruction tokens)
- `<EOT>` token trained so the model learns **when to stop**

### Encoder-Decoder Model
- Encoder receives: $[x_1, \ldots, x_m]$
- Decoder receives: $[\texttt{<BOS>}, y_1, \ldots, y_{k-1}]$ (teacher-forced)
- Loss is the same conditional log-likelihood
- Architecture differs but objective is identical

---

## 3. Data Sources for Instruction Tuning

### Overview

| Source | Quality | Diversity | Cost | Example |
|--------|---------|-----------|------|---------|
| Existing NLP datasets → templates | Good | Limited to NLP tasks | Low (template creation) | FLAN |
| NLP practitioners contribute tasks | High | Moderate | Medium | Super-Natural Instructions |
| Synthetic (LLM-generated) | Surprisingly good | Very high | Low | Self-Instruct, Evol-Instruct |

---

## 4. Human-Generated Data

### Approach 1: FLAN — Template-Based Conversion

Convert existing NLP datasets into instruction format using **human-written templates**.

**Example** (Natural Language Inference):

| Template | Instruction |
|----------|-------------|
| Template 1 | "Based on the paragraph above, can we conclude that [hypothesis]?" |
| Template 2 | "Read the following and determine if the hypothesis can be inferred from the premise." |

- Multiple templates per task → increases diversity
- Human effort = template creation only; data is already available
- This is how the original **FLAN** dataset was created

### Approach 2: Super-Natural Instructions

- NLP practitioners submit tasks via **GitHub issues**
- Each submission includes:
  - **Task definition** (instruction)
  - **Positive examples** (correct input-output pairs)
  - **Negative examples** (incorrect outputs)
- Encourages modifications of existing NLP tasks
- Positive/negative examples preserve **in-context learning** behavior during instruction tuning (supports 0-shot, 1-shot, few-shot variations)

---

## 5. Synthetic Data Generation

### Why Synthetic?
- Human data sources (FLAN, Super-Natural Instructions) have limited task diversity
- Mechanical Turk produces low-quality data
- LLMs can generate **more diverse** and surprisingly **higher quality** instruction data
- Cheap and scalable

> **Caveat**: LLM-generated data can be incorrect — "better quality on average" ≠ always correct

### 5.1 Self-Instruct

**Key insight**: A pre-trained LM (not instruction-tuned) can generate its own instruction tuning data.

#### Pipeline

```
Seed Tasks (175 human-written)
    ↓
[1] Instruction Generation
    - Sample 8 instructions from pool
    - Pre-trained LM generates new instructions
    - Filter by similarity (discard if too similar to existing)
    ↓
[2] Classification Identification
    - Use ICL to determine if task is classification or not
    - Why? Classification tasks → model tends to generate
      skewed label distributions (e.g., mostly positive sentiment)
    ↓
[3] Instance Generation
    - Input-first: Give instruction → generate input, then output
    - Output-first (for classification): Give label first → 
      generate input matching that label
    - Ensures balanced class distribution
    ↓
[4] Filtering & Add to Pool
    - Quality filter
    - Add new (instruction, example) pairs back to task pool
    - Iterate
```

#### Results

| Model | ROUGE-L (Super-NI test) | Human Eval |
|-------|------------------------|------------|
| GPT-3 (pre-trained, no instruct) | Very low | Poor |
| GPT-3 + Self-Instruct | Close to InstructGPT | Competitive |
| InstructGPT | 40.8 | Best |
| GPT-3 + Self-Instruct + Super-NI data | 49.5 | Marginal improvement |

> **Surprising result**: Self-Instruct on a pre-trained model nearly matches InstructGPT (which used human feedback)

### 5.2 Evol-Instruct (WizardLM)

**Problem**: Self-Instruct produces simple instructions → model can't handle complex reasoning.

**Solution**: Use an LLM to **evolve** instructions into harder versions.

#### In-Depth Evolution
Prompt the LLM:
> "I want you to act as a prompt rewriter. Create a more complex version of this prompt that GPT-4 cannot easily handle. [Constraint: add more constraints / increase reasoning steps / ...]"

Replace the red constraint with:
- "Add more constraints to the input"
- "Increase the number of reasoning steps"
- "Make the problem require multi-step reasoning"

#### In-Breadth Evolution
> "Create a brand new prompt from the same domain. Keep same difficulty level but different scope."

- Not making it harder — expanding **coverage** of the instruction set

#### Instruction Eliminator
- Generated evolved instructions are tested: if all responses are "I don't know" / non-informative → **discard**

### 5.3 Orca-Style Generation (Distillation++)

**Problem with standard distillation**: Teacher model (GPT-4) often generates **very short** responses → insufficient information for student model to learn from.

**Solution**: Force the teacher to generate **detailed reasoning**.

Add **system-level instructions**:
- "Let's think step by step"
- "Explain the solution as if explaining to a 5-year-old"
- "Be more informative in your response"
- "Provide a detailed answer explaining why your answer is correct"

**Example**:

| Setup | Teacher Output |
|-------|---------------|
| Standard distillation | "Answer: B" |
| Orca-style | "Based on the given options and context, the refrigerator door is closed because... [detailed reasoning]... Therefore, the answer is B." |

> The detailed output gives the student model **much more signal** to learn the teacher's reasoning process.

### 5.4 Instruction Back-Translation

**Idea**: Invert the problem — instead of generating instructions → examples, generate instructions **from** examples.

#### Pipeline

```
[1] Train Back-Translator
    - Use small set of human (instruction, response) pairs
    - Train LLaMA to: given response → generate instruction
    
[2] Apply to Unlabeled Web Text
    - Feed web documents to back-translator
    - Model infers: "What instruction would produce this text?"
    - Output: (generated instruction, original web text) pairs
    
[3] Quality Filtering
    - Score and remove low-quality pairs
    
[4] Forward Training
    - Use filtered (instruction, output) pairs
    - Standard instruction tuning on the language model
```

- Leverages **unlimited unlabeled web text** as a source of outputs
- Self-improvement technique: back-translate → filter → train

---

## 6. Notable Instruction-Tuned Models

| Model | Base | Instruction Data | Method |
|-------|------|-------------------|--------|
| **FLAN-T5** | T5-11B | FLAN dataset | Template-based NLP tasks |
| **Alpaca 7B** | LLaMA 7B | Synthetic (Self-Instruct) | Self-Instruct pipeline |
| **WizardLM 7B** | LLaMA 7B | Synthetic (Evol-Instruct) | Evolution of instructions |
| **Mistral 7B OpenOrca** | Mistral 7B | Orca-style generations | Detailed teacher outputs |

---

## 7. Properties of Instruction-Tuned Models

### Quick Learners (Few-Shot Adaptation)
- Instruction-tuned models adapt to new tasks with **very few examples**
- Example: TK-Instruct achieves ROUGE = 70 with only **6% training samples**
- Baseline T5 (pre-trained only) needs **100% samples** for ROUGE = 70.99

| Model | Training Samples | ROUGE |
|-------|-----------------|-------|
| T5 (pre-trained) | 100% | 70.99 |
| TK-Instruct (instruction-tuned) | **6%** | ~70 |

### Superficial Alignment Hypothesis

> **Claim**: Pre-training incorporates all reasoning into the model. Instruction tuning merely teaches the model **which sub-distribution of output formats** to use when interacting with humans.

**Implication**: If alignment is just about format, a **very small** number of high-quality examples should suffice.

### LIMA: Less Is More for Alignment

- **1,000 carefully curated examples** from Stack Exchange, WikiHow, Reddit (heavily filtered)
- Instruction-tuned on this tiny dataset
- Results: competitive human preference scores — close to models tuned on much larger datasets
- GPT-4 still better, but the gap is surprisingly small
- Supports the superficial alignment hypothesis

---

## 8. Key Takeaways

1. Instruction tuning bridges the gap between pre-trained models and human-usable assistants via **conditional log-likelihood** on (instruction, output) pairs
2. Data diversity is critical — thousands of diverse tasks matter more than many examples per task
3. **Synthetic data** (Self-Instruct, Evol-Instruct, Orca-style, Back-Translation) can match or exceed human-generated instruction data
4. Self-Instruct showed a pre-trained model can bootstrap its own instruction-following ability
5. Instruction-tuned models are **quick learners** — few examples suffice for new task adaptation
6. The **Superficial Alignment Hypothesis** suggests instruction tuning only teaches output format, not reasoning — supported by LIMA's success with just 1,000 examples
