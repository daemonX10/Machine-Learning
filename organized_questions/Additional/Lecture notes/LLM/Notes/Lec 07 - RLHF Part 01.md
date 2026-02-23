# Lecture 07 — RLHF Part 1: Reward Models & Preferences

## 1. Why RLHF After Instruction Tuning?

Instruction tuning tells the model **what to do** but never **what not to do**.

| Approach | Teaches | Limitation |
|----------|---------|------------|
| Instruction tuning | Generate helpful responses | Doesn't penalize harmful / low-quality outputs |
| RLHF | Maximize reward (quality) while penalizing bad behavior | Requires preference data + reward model |

---

## 2. Alignment Taxonomy

### By Objective

| Category | Method | Description |
|----------|--------|-------------|
| **Reward Maximization** | PPO, REINFORCE, GRPO | Maximize a reward signal |
| **Contrastive** | DPO, KTO | Learn from good vs. bad pairs directly |
| **Distribution Matching** | GDC | Match target distribution |

### By Training Paradigm

| Type | Description | Example |
|------|-------------|---------|
| **Online** | Model generates its own samples, gets feedback | PPO |
| **Offline** | Uses pre-collected preference data | DPO |
| **Hybrid** | Mix of both | Online DPO |

---

## 3. RL Basics for LLMs

### Core Components

| RL Concept | LLM Analogy |
|-----------|-------------|
| **Agent** | The language model |
| **State** $s_t$ | Prompt + all tokens generated so far |
| **Action** $a_t$ | Next token to generate |
| **Environment** | Append token to sequence |
| **Reward** $r$ | Scalar quality score (given at end of full generation) |
| **Policy** $\pi_\theta$ | Probability distribution over vocabulary at each step |

### Concrete Examples

| Scenario | State | Action | Reward |
|----------|-------|--------|--------|
| Chess | Board position | Move | Win/loss |
| Self-driving car | Sensors, position, speed | Steering angle | Safety + comfort + progress |
| LLM (token-level) | Prompt + tokens so far | Next token | Reward at final token |
| LLM (response-level) | Prompt | Full response | Human preference score |
| Code LLM | Prompt + code so far | Code tokens | Unit test pass/fail |

### Policy Definition

$$\pi_\theta(a_t \mid s_t) = P_\theta(\text{token}_t \mid \text{prompt}, \text{token}_1, \ldots, \text{token}_{t-1})$$

- The policy is the model's **softmax output distribution** at each timestep
- Parameters $\theta$ are all model weights

---

## 4. Reward Models

### 4.1 Verifiable Rewards (Preferred When Available)

| Domain | Verification Method |
|--------|-------------------|
| **Math** | Exact numerical match with ground truth |
| **Code** | Unit test execution (pass/fail) |
| **Factual QA** | String matching against known answers |
| **Formal proofs** | Automated proof checkers |
| **Chemistry** | Molecular simulation validity |
| **Protein structure** | Structural prediction accuracy |

**Advantages**: No noisy human preferences; unbiased; scalable.
Used by **GRPO, DeepSeek-R1** for math and code tasks.

### 4.2 Learned Reward Models

Needed when no automatic verification is possible (essays, creative writing, open-ended QA).

**Architecture**:

```
Input: [prompt, response] → LLM backbone
                               ↓
                    Pre-logit embeddings (hidden states)
                               ↓
                    Pooling (average or last-token)
                               ↓
                    Linear layer → scalar reward r(x, y)
```

- The backbone can share weights with the policy model or be separate
- Typically initialized from a pre-trained LLM

---

## 5. Bradley-Terry Preference Model

### 5.1 Core Formulation

Given two responses $y_i$ and $y_j$ to prompt $x$, the probability that $y_i$ is preferred:

$$P(y_i \succ y_j) = \frac{p_i}{p_i + p_j}$$

where $p_i = e^{\theta_i}$ (positive parameterization).

For learned reward models:

$$P(y_i \succ y_j \mid x) = \frac{e^{r_\theta(x, y_i)}}{e^{r_\theta(x, y_i)} + e^{r_\theta(x, y_j)}} = \sigma\big(r_\theta(x, y_i) - r_\theta(x, y_j)\big)$$

where $\sigma$ is the sigmoid function.

### 5.2 MLE Training Loss

$$\mathcal{L}(\theta) = -\mathbb{E}_{(x, y^+, y^-) \sim \mathcal{D}} \Big[\log \sigma\big(r_\theta(x, y^+) - r_\theta(x, y^-)\big)\Big]$$

- $y^+$ = preferred (chosen) response
- $y^-$ = rejected response
- Maximizes the **reward gap** between preferred and rejected responses

> **Intuition**: The loss is just binary cross-entropy — it pushes $r(x, y^+) \gg r(x, y^-)$.

---

## 6. Preference Data Sources

### 6.1 Human Preferences

| Dataset | Description |
|---------|-------------|
| **OpenAI Summarization** | Human comparisons of TL;DR summaries |
| **Anthropic HH-RLHF** | Helpful + harmless preference pairs |
| Custom | Generate from instruction-tuning prompts + model outputs + human ranking |

### 6.2 LLM-Generated Preferences

| Dataset | Method |
|---------|--------|
| **UltraFeedback** | Multiple model outputs per prompt; GPT-4 selects preferred |
| **RLAIF (Anthropic)** | AI provides preferences instead of humans |

---

## 7. Constitutional AI (Anthropic)

Uses a set of **16 principles** (constitution) to judge model outputs:

```
Principles include:
- Be ethical and responsible
- Be polite and helpful  
- Don't produce harmful content
- Respect privacy
- ...
```

**Process**:
1. Generate response
2. Present response + relevant constitutional principle to judge model
3. Judge provides preference based on principle adherence
4. Use preferences to train reward model

---

## 8. HelpSteer (NVIDIA)

Addresses **length bias** in reward models (longer responses get higher scores regardless of quality).

**Multi-attribute scoring**:

| Attribute | What It Measures |
|-----------|-----------------|
| Helpfulness | Does it answer the question? |
| Correctness | Is the information accurate? |
| Coherence | Is it logically structured? |
| Complexity | Appropriate detail level? |
| Verbosity | Right length (penalizes unnecessary length)? |

- Each response gets a **score vector** $[h, c_1, c_2, c_3, v]$
- Score vector is converted to a **string representation** for training
- Verbosity attribute explicitly counters length bias

---

## 9. LLM-as-Judge

### 9.1 Judging Modalities

| Mode | Input | Output |
|------|-------|--------|
| **Pairwise comparison** | Prompt + Response A + Response B | "A is better" / "B is better" |
| **Single-answer grading** | Prompt + Response | Score (1-5) |
| **Reference-guided** | Prompt + Response + Gold answer | Score (1-5) |

### 9.2 Pros and Cons

| Pros | Cons |
|------|------|
| No human preference data needed | Less accurate than trained reward models |
| Can leverage reasoning capability | Overconfident in judgments |
| Scalable | Biased toward longer responses |

### 9.3 Training an LLM-as-Judge

**Training objective**:
- Maximize $P(\text{"yes"} \mid \text{correct response})$
- Maximize $P(\text{"no"} \mid \text{wrong response})$

**Inference**: Use $\log P(\text{"yes"})$ as the reward score.

### 9.4 Chain-of-Thought Verification

```
Prompt: "Let's verify step by step."
Model: [generates verification reasoning]
       → "Step 1 checks out because..."
       → "Step 2 has an error because..."
       → Final verdict: "yes" / "no"
```

- Score = $P(\text{"yes"})$ after the full CoT verification
- More reliable than direct yes/no judgment

---

## 10. Summary: Reward Model Landscape

```
Reward Signal
├── Verifiable (math, code, factual QA)
│   └── Exact match / unit test → scalar reward
└── Learned (open-ended, creative, safety)
    ├── Bradley-Terry preference model
    │   └── Trained on human/LLM preference pairs
    ├── Multi-attribute (HelpSteer)
    │   └── Score vector → string → regression
    └── LLM-as-Judge
        └── Prompted / trained judge → P("yes") as reward
```
