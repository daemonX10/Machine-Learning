# Lecture 27: LLM Reasoning Part 03 — Test-Time Scaling

## 1. Background: Pre-Training Scaling Laws

### Kaplan Scaling Law
$$L = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}$$

- $N$ = number of parameters, $D$ = dataset size
- More parameters OR more data → lower loss

### Chinchilla Scaling Law
- For a **fixed compute budget**, optimal tradeoff between parameters and data
- Showed earlier models were undertrained

### Problems With Pre-Training Scaling
| Problem | Details |
|---------|---------|
| Limited compute/data | Diminishing returns at scale |
| Environmental impact | 10% improvement needs **3× energy** |
| Performance plateau | Marginal gains at extreme scale |

---

## 2. Test-Time Scaling: Overview

> Spend more compute at **inference** (with fixed model parameters) to improve performance.

### Analogy: Thinking Fast and Slow
- **Fast** (System 1): Standard autoregressive generation
- **Slow** (System 2): Extended reasoning at test time

### Taxonomy

| Category | Method | Type |
|----------|--------|------|
| **Parallel** | Best-of-N, Beam Search, DVTS | Generate multiple solutions |
| **Sequential** | Self-Refine, ReAct, S1/Budget Forcing | Iterate/think deeper |
| **Hybrid** | Tree of Thoughts, Graph of Thoughts | Both parallel + sequential |
| **Internal** | RL-based (O1, DeepSeek R1) | Model learns to think internally |

---

## 3. Parallel Scaling Methods

### Best-of-N Sampling

Generate $n$ solutions, select the best one.

**Probability of at least one correct**:
$$P(\text{success}) = 1 - (1-p)^n$$

where $p$ = probability of single correct solution.

### Self-Consistency (Majority Voting)
- Set temperature = 0.7
- Sample 21 reasoning paths
- **Majority vote** on final answers
- Outperforms greedy decoding and sample-and-rank

### Process Reward Model (PRM)
- Unlike **Outcome Reward Model** (ORM) which only evaluates final answer
- PRM evaluates **each intermediate step**
- Checks coherence, robustness, not just correctness

### Diverse Verifier Tree Search (DVTS)
- Uses PRM to guide tree search
- Evaluates partial solutions at each step

### First-Finish Search (FFS)
**Heuristic**: Among parallel traces, the one that finishes **first** (shortest length) is most likely correct.

**Intuition**: An oracle model would be precise and concise. Verbose traces indicate uncertainty.

**Empirical finding**: Correct responses tend to have **shorter length** than incorrect ones.

**Advantage**: Dramatically reduces inference compute — stop as soon as any branch finishes.

---

## 4. Sequential Scaling Methods

### Self-Refine (2023)
1. Model $M$ generates output for input
2. **Same model** acts as critic → produces feedback
3. Model **refines** its output based on feedback
4. Repeat until feedback is positive or max iterations reached

**Why it works**: Non-reasoning models (pre-2024) don't optimize for quality during generation. Feedback loop forces explicit quality assessment.

### S1 / Budget Forcing (Stanford, 2025)

- Only **1000 training instances** with special thinking tokens
- Simple **SFT** (no RL)
- Model learns to use **"Wait"** token to pause and rethink

**Example**: "How many R's in Raspberry?"
1. Model counts → gets 2
2. Encounters **"Wait"** token → rethinks
3. Recounts carefully → gets **3** (correct)

**Budget Forcing**: Set a maximum thinking budget; model learns where to allocate thinking.

### Thought Switching Penalty

| Parameter | Description |
|-----------|-------------|
| **Penalty Strength** | Amount of penalty added to logit of switching token |
| **Penalty Duration** | Number of tokens the penalty persists |

- **Acceptable**: Switch direction when confidence is low
- **Not acceptable**: Switch direction when confidence was high

Prevents the model from abandoning a correct reasoning path due to distribution artifacts.

---

## 5. Hybrid Scaling: Tree of Thoughts (ToT)

### Structure
1. Define **intermediate thought units** (e.g., paragraphs, steps)
2. From each state, generate $K$ candidate thoughts (parallel)
3. **State evaluator** (separate LLM/PRM) assesses whether branch leads to correct answer
4. Apply BFS/DFS strategy with pruning based on scores
5. Backtrack if branch scores decline

**Major innovation**: The state evaluation step — determining branch viability mid-solution.

---

## 6. Internal Scaling: O1 Model

### Architecture
Built on **GPT-4** with:

| Component | Description |
|-----------|-------------|
| Reasoning-focused pre-training | Math, science, code data with traces |
| SFT on QA reasoning data | Explicit thought sequences |
| RL with reasoning rewards | Reinforces good reasoning patterns |
| Internal reflection loops | Model self-assesses its steps |
| Adaptive halting | Dynamically decides when to stop thinking (easy → short, hard → long) |

---

## 7. Evaluating Reasoning Models

### Benchmarks

| Benchmark | Type |
|-----------|------|
| AIME / Math 500 | Olympiad math |
| GPQA | MCQ (physics, chemistry, math) |
| SWE-bench | Code generation |

### Four Evaluation Dimensions

#### 1. Performance
- Accuracy, Pass@K

#### 2. Controllability
Does the model stay within resource constraints?
$$C = \frac{\text{traces within } [L_{\min}, L_{\max}]}{\text{total traces}}$$

Also: mean deviation from target length, RMSE.

#### 3. Scalability
Does accuracy improve proportionally with more compute?
$$S = \frac{\Delta \text{accuracy}}{\Delta \text{length}}$$

Computed for all pairs of reasoning traces.

#### 4. Efficiency

| Metric | Definition |
|--------|------------|
| **Total compute** | Total tokens across all generated traces |
| **Sequential compute** | Tokens in the specific correct branch only |
| **Underthinking score** | See below |

### Underthinking Score

For **incorrect** traces only:

$$U_i = 1 - \frac{\hat{T}_i}{T_i}$$

- $T_i$ = total tokens in trace $i$
- $\hat{T}_i$ = tokens until first correct step

| Scenario | $U_i$ |
|----------|-------|
| First steps correct, then deviates | **High** (good start, model knows how to reason) |
| No correct steps at all | **0** (model can't reason about this) |
| Correct steps appear late | Low |

---

## 8. When to Use Which Method

### By Compute Availability × Task Difficulty

| | Easy Task | Hard Task |
|--|-----------|-----------|
| **Low Compute** | FFS (first-finish search) | Budget Forcing (sequential) |
| **High Compute** | Any method works | Majority Voting (parallel) |

### By Model Strength × Task Difficulty

| | Easy Task | Hard Task |
|--|-----------|-----------|
| **Weak Model** | FFS | Budget Forcing |
| **Strong Model** | Any method | Majority Voting |

**Key insight**: Parallel scaling for easy questions, sequential scaling for hard questions.
