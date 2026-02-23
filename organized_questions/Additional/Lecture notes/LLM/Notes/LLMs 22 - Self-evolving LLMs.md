# Lecture 22: Self-Evolving LLMs

## 1. Evolution of LLMs — Three Stages

| Year | Stage | Key Model | Description |
|------|-------|-----------|-------------|
| 2018-19 | **Pre-training** | BERT, GPT-2 | Self-supervised learning on massive corpora |
| 2020 | **Supervised Fine-Tuning (SFT)** | T5 | Cast every task (even classification) as generation; trained on annotated data |
| 2022 | **Human Alignment** | InstructGPT | RLHF — align model outputs to human preferences |

> Every state-of-the-art model goes through **all three stages** in order. All three stages are **data-hungry** (e.g., Llama-3 uses ~15 trillion tokens).

---

## 2. Motivation: AlphaGo and Self-Play

AlphaGo's training had two phases:

1. **Imitation Learning** — Trained to mimic human moves (analogous to pre-training / SFT / RLHF)
2. **Self-Play** — Two copies play against each other:
   - Generate new trajectories
   - Identify good moves
   - Get reward signal from environment (win/lose)
   - Use RL to update the model

> **Key insight**: Self-play made the critical difference — the model stopped relying on external data and improved by playing against itself.

---

## 3. Self-Evolution Paradigm for LLMs

```
┌───────────────────────────┐
│   Experience Acquisition  │ ← Generate synthetic data (trajectories)
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│   Experience Refinement   │ ← Filter good vs bad samples
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│      Model Update         │ ← SFT or RL training
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│       Evaluation          │ ← Assess & iterate
└───────────────────────────┘
```

### Challenges for LLMs (vs. AlphaGo)

| Challenge | AlphaGo | LLMs |
|-----------|---------|------|
| Data generation | Self-play trajectories | How to construct synthetic input-output pairs? |
| Quality signal | Deterministic win/lose | No unambiguous environment feedback |
| Reward | Clear terminal reward | Hard to assess partial/full answer quality |

---

## 4. Class 1: Weight-Updating Methods

### 4.1 STaR — Self-Taught Reasoner

**Premise**: Chain-of-Thought (CoT) fine-tuning produces better models than direct answer fine-tuning.

**Given**: Pre-trained LLM $M$, annotated dataset $D = \{(x_i, y_i)\}$, few-shot CoT examples $P$

**Algorithm**:

1. For each $(x_i, y_i) \in D$, prompt $M$ with $P \oplus x_i$
2. Model generates $(\hat{r}_i, \hat{y}_i)$ — reasoning + predicted answer
3. **Filter**: If $\hat{y}_i = y_i$, add $(x_i, \hat{r}_i, y_i)$ to $D'$; else discard
4. Fine-tune $M$ on $D'$ to get improved model $M_2$

**Rationalization Improvement** (reduces rejections):

- When $\hat{y}_i \neq y_i$, provide $y_i$ as a **hint** in the prompt
- Modified prompt $P'$ asks: "Given the answer is $y_i$, generate the reasoning"
- Still filters — model must also produce correct $\hat{y}_i$ even with hints

**Results** (Arithmetic Tasks): Rationalization accelerates learning across all difficulty levels (1–5 digit addition)

> **Limitation**: Still depends on annotated $(x, y)$ pairs — only generates reasoning $r$.

---

### 4.2 Self-Instruct

**Goal**: Generate instruction-tuning data synthetically from **175 seed tasks** alone.

**Algorithm**:

```
1. Initialize task pool with 175 seed tasks (instruction + example)
2. Loop:
   a. Use ICL to generate NEW instructions from sampled pool tasks
   b. Classify: is it a generation or classification task?
   c. For generation tasks:
      - Prompt: meta-instruction + ICL examples → generate (input, output)
   d. For classification tasks (output-first approach):
      - Prompt: give label first → generate input adhering to that label
      - Fixes class imbalance (e.g., only positive sentiment)
   e. Filter duplicates using heuristics
   f. Add to task pool
3. Fine-tune base model on synthesized data
```

**Results**:

| Model | ROUGE-L on SuperNI (119 tasks) |
|-------|-------------------------------|
| Vanilla GPT-3 | 6.8 |
| Self-Instruct GPT-3 | **39.9** |
| InstructGPT (with real data) | 40.8 |

- From 175 seeds → ~52K instructions → ~82K (instruction, input, output) triples
- Nearly matches InstructGPT without real instruction-tuning data

> **Limitation**: Model is instruction-following but not aligned to human preferences.

---

### 4.3 Self-Align

**Goal**: Add human alignment without human annotators.

**Given**: Base LLaMA + 175 self-instruct seeds + **20 red-teaming topics** + **16 alignment principles**

**Three Steps**:

1. **Topic-Guided Red-Teaming Self-Instruct** — Generate potentially harmful questions using 20 adversarial topics (195 total seed tasks)
2. **Principle-Driven Self-Alignment** — Replace generated outputs with new ones:
   - Prompt includes: 16 explicit rules + 5 expert ICL examples following those rules
   - Model generates **internal thought** (static CoT enforcing rules) + output
   - Example: "Who is US president in 2025?" → "My knowledge ends in 2021, I cannot answer"
3. **Principle Engraving** — SFT on the principle-aligned generated data

> **Key idea**: Align to **principles/rules** instead of human preference data — the 16 principles *mimic* human alignment.

---

## 5. Improving Experience Acquisition (Inference-Time)

### 5.1 Self-Refine

An inference-time improvement method (**no weight updates**):

```
Input x
  → Generate y₀ (Generator hat)
  → Feedback fb₀ (Examiner hat)
  → Refine: y₀ + fb₀ → y₁ (Refiner hat)
  → Feedback fb₁
  → Refine: y₁ + fb₁ → y₂
  → ... iterate
```

- LLM wears **three hats**: generator, examiner, refiner
- Local process — improves answer for a **specific input** without updating weights

---

### 5.2 Tree of Thought (ToT)

**Problem with CoT**: Decisions are token-level, no global planning, no backtracking.

| Method | Game of 24 Accuracy |
|--------|-------------------|
| Standard I/O Prompt | 7.3% |
| Chain of Thought | 4.0% |
| CoT + Self-Consistency (100 samples) | 9.0% |
| **Tree of Thought (b=1)** | **45%** |
| **Tree of Thought (b=5)** | **74%** |

**Algorithm**:

1. **Decompose** output into discrete thoughts/steps
2. **Generate** multiple candidate thoughts at each step
3. **Evaluate** each thought (LLM as examiner) → {sure, likely, impossible}
4. **Search** through thought tree using BFS/DFS
   - Prune "impossible" nodes
   - Backtrack when needed

```
        Input
       / | \
     T₁  T₂  T₃     ← Level 1 thoughts
    /|    |    
  T₄ T₅  T₆         ← Level 2 thoughts
  |       |
  T₇     T₈          ← Level 3 thoughts
  |
 Output
```

---

### 5.3 MCTS + Self-Refine

- Replaces BFS with **Monte Carlo Tree Search** for better exploration-exploitation
- Combines Tree of Thought + Self-Refine
- **Result**: LLaMA-3 8B achieves performance comparable to GPT-4

---

## 6. Combined Approach: Imagination, Searching, and Criticizing

Paper: *"Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"*

- **Data acquisition**: Uses MCTS (inference-time search)
- **Model update**: Uses SFT on acquired data
- Combines both paradigms — better data generation + weight updates

---

## 7. Summary Taxonomy

| Category | Weight Update? | Methods | Data Source |
|----------|---------------|---------|-------------|
| **Synthetic Data + Training** | ✅ Yes | STaR, Self-Instruct, Self-Align | Self-generated, filtered |
| **Inference-Time Improvement** | ❌ No | Self-Refine, ToT, MCTS+Self-Refine | LLM wears multiple hats |
| **Combined** | ✅ Yes | Imagination+Search+Criticize | MCTS for acquisition + SFT |

> **Core principle**: The model improves itself by wearing multiple hats (generator, evaluator, refiner) — reducing dependence on external data.
