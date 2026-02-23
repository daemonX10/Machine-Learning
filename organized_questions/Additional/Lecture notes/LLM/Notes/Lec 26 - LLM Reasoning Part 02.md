# Lecture 26: LLM Reasoning Part 02 — Advanced Techniques

## 1. GRPO — Group Relative Policy Optimization

### Core Idea
Online RL method that uses **all** generated samples (not just chosen/rejected pair like DPO).

### Algorithm

1. Sample a **group** of outputs $\{(z_i, y_i)\}_{i=1}^{G}$ for each input $x$
2. Compute reward $r_i$ for each output
3. Compute **normalized advantage**:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

4. Update policy with **importance-weighted clipped objective** (similar to PPO):

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} \hat{A}_i, \; \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i\right)\right]$$

### Advantages Over DPO
- Uses **all** samples, not just best/worst
- No need for reference model
- Better sample efficiency

---

## 2. DeepSeek R1-Zero

### Setup
- **No SFT** — GRPO applied directly to powerful base model (DeepSeek V3)
- Only **verifiable rewards** used

### Reward Types

| Reward | Domain | Description |
|--------|--------|-------------|
| **Accuracy reward** | Math | Exact match with ground truth |
| **Unit test reward** | Code | Pass/fail on test cases |
| **Format reward** | All | Correct use of `<think>` / `<answer>` tags |

### Emergent Reasoning Behaviors
Without explicit training, the model spontaneously develops:

| Behavior | Description |
|----------|-------------|
| **Verification** | Model checks its own intermediate steps |
| **Backtracking** | "Wait, that doesn't seem right. Let me go back..." |
| **Sub-goal setting** | Breaks problem into smaller parts |

### Training Dynamics

- **AIME benchmark**: 15% → 70% accuracy over ~8000 training steps
- **Response length increases** during RL training (model learns to think longer)
- Base model matters: **Qwen shows 62% verification** behavior vs **LLaMA only 10%**

---

## 3. Distillation for Weaker Models

### Why Distillation?
> "Always prefer distillation over RL" for weak models.

### Procedure
1. Generate solutions from a **powerful model** (e.g., DeepSeek R1)
2. **Verify** correctness (math: exact match, code: unit tests)
3. **SFT** weaker model on verified correct examples

### Results
- Distillation consistently outperforms RL on weaker models
- RL on weak models often fails — the model lacks the initial capability to reason

---

## 4. Teaching Backtracking via Linearization

### Goal
Teach the model to **restart and try a different approach** when it detects an error.

### Step 1: Generate Solution Tree
- Prompt model to generate step-by-step solution (with step delimiters like `\\`)
- Verify answer; if wrong, identify the faulty step
- **Restart from the step before the mistake** → generate alternative continuation
- Result: a **tree** of correct and incorrect paths

### Step 2: Linearize the Tree for SFT

1. **Rewrite each step** as a naturally flowing thought:
   - Prompt: *"Given a partially thought-out solution and the current step, rewrite the step so it smoothly continues the previous thoughts."*

2. **Add backtracking transition**:
   - Prompt: *"This solution is incorrect and needs to backtrack to step X. Continue by acknowledging the mistake and returning to that step."*
   - Model generates: *"Wait, this doesn't seem right. Let me go back to where I calculated the volume..."*

3. **Continue with correct path** — rewrite remaining steps naturally

Result: A **linearized** sequence that looks like natural reasoning with backtracking.

### Step 3: SFT + RL

| Stage | Math 500 | AMC | AIME |
|-------|----------|-----|------|
| Direct SFT (correct path only) | 65 | 45 | 16.7 |
| Linearized SFT (with backtracking) | **higher** | **higher** | ~same |
| Direct SFT + RL | improved | improved | improved |
| Linearized SFT + RL | **best** | **best** | **best** |

- After RL, backtracking behavior **increases significantly** (1 → many on Math 500)

---

## 5. MCTS — Monte Carlo Tree Search for Reasoning

### Why MCTS?
Random sampling produces solutions that may not intersect. MCTS **systematically explores high-reward solutions**.

### Tree Structure
- **Nodes** = partial solutions (all steps so far)
- **Edges/Actions** = individual reasoning steps
- Since environment is deterministic ($s_{t+1} = s_t \oplus a_t$), nodes can represent actions directly

### Four Phases of MCTS

#### 1. Selection (UCB)
Choose which node to expand using **Upper Confidence Bound**:

$$\text{UCB}(s, a) = V(s') + c \cdot \pi(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s')}$$

- $V(s')$ = value of child node (exploitation)
- $\frac{\sqrt{N(s)}}{1 + N(s')}$ = exploration bonus (favor under-explored nodes)
- $\pi(a|s)$ = model's probability of generating this step

#### 2. Expansion
From selected node, **sample $k$ possible next steps** using the LM.

#### 3. Evaluation (Node Reward)
Evaluate a node by **running multiple completions** from it:

$$r(s) = \frac{\text{number of correct completions}}{\text{total completions}}$$

- Expensive but produces high-quality node estimates

#### 4. Backpropagation
Update node values **bottom-up**:

$$V(s) = \frac{r(s) + \sum_{s' \in \text{children}(s)} V(s') \cdot N(s')}{1 + \sum_{s' \in \text{children}(s)} N(s')}$$

where $N(s)$ = number of times node $s$ has been selected.

### MCTS → Training Pipeline

```
MCTS Tree → Find correct + incorrect paths → Linearize → SFT → RL (GRPO)
```

### Key Paper: ASTRO
- Uses MCTS to generate reasoning trees
- Linearizes with backtracking for SFT
- Then applies GRPO on top
