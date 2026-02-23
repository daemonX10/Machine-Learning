# Lecture 25: LLM Reasoning Part 01 — Eliciting Thought

## What is Reasoning in LLMs?

Generate **intermediate tokens** $z$ (thought/reasoning) before the final response $y$:

$$P(y|x) = \pi_\theta(z|x) \cdot \pi_\theta(y|z, x)$$

- $x$ = input/prompt
- $z$ = thought/chain-of-thought
- $y$ = final response

---

## Eliciting Reasoning via Prompting

### Few-Shot Chain-of-Thought (CoT)
- Provide exemplars with step-by-step reasoning in the prompt
- Model mimics the reasoning pattern

### Zero-Shot CoT
- Simply append **"Let's think step by step"** to the prompt
- No exemplars needed; surprisingly effective

---

## Three Training Approaches to Reasoning

| Approach | Method | Algorithm | Key Idea |
|----------|--------|-----------|----------|
| **SFT-based** | Self-Taught Reasoner | STaR | Train on self-generated correct reasoning |
| **DPO-based** | Thought Preference Optimization | TPO | Preference learning on thought+response pairs |
| **RL-based** | Group Relative Policy Optimization | GRPO | Online RL with verifiable rewards |

---

## 1. STaR — Self-Taught Reasoner (SFT-based)

### Algorithm

1. Given dataset of $(x, y^*)$ pairs (question, ground-truth answer)
2. For each $x$, generate thought $z$ and response $y$ using current model $\pi_{\theta_t}$
3. **Filter**: keep only samples where generated $y$ matches ground truth $y^*$
4. **Rationalization**: for failed samples, provide ground-truth answer as a **hint**, regenerate thought
5. **SFT** on all correct (thought, response) pairs
6. **Iterate**: repeat from step 2 with updated model

### Key Features
- Verifiable reward (exact match with ground truth)
- Rationalization prevents losing hard examples
- Iterative self-improvement loop

### Limitations
- Requires verifiable answers (math, code)
- Cannot handle open-ended/subjective tasks

---

## 2. TPO — Thought Preference Optimization (DPO-based)

### Motivation
- Works for **non-verifiable tasks** (creative writing, dialogue, etc.)
- Uses a **judge model** instead of ground-truth matching
- Separates thought from response for evaluation

### Algorithm

1. Given input $x$, prepend instruction: *"Before responding, generate a thought"*
2. Model generates multiple $(z_i, y_i)$ pairs (thought + response)
3. **Extract response only** → feed to judge/reward model
4. Judge scores each response → rank to get:
   - **Chosen**: $(x, z^+, y^+)$ — highest reward
   - **Rejected**: $(x, z^-, y^-)$ — lowest reward
5. Train using **DPO loss**:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \cdot \left[r(x, z^+, y^+) - r(x, z^-, y^-)\right]\right)$$

where implicit DPO reward:

$$r(x, z, y) = \log \frac{\pi_{\theta_{t+1}}(z, y | x)}{\pi_{\theta_t}(z, y | x)}$$

6. **Iterate**: regenerate samples with updated model, repeat

### Length Control
Judge models tend to prefer longer outputs → response length grows with iterations.

**Solution**: Length-normalized reward:

$$\hat{r}_i = \text{normalize}(r_i) - \rho \cdot \text{normalize}(\text{len}_i)$$

- $\rho \in [0, 0.04]$ — small weight to avoid model collapsing to zero-length outputs

### Results (on AlpacaEval & Arena Hard)
- Performance improves for ~3 iterations then saturates
- Initially worse than SFT baseline (model hasn't seen thoughts before), then overtakes
- Gains vary by category: large on research/analysis, smaller on creative writing

### Why Performance Saturates

| Issue | Explanation |
|-------|-------------|
| **Reward hacking** | Model learns to exploit reward model biases |
| **Data overfitting** | Same prompts used every iteration |
| **DPO waste** | Only uses chosen/rejected, discards other samples |

### Addressing Limitations

| Problem | Solution |
|---------|----------|
| Reward models are imperfect | Use **verifiable rewards** (exact match) |
| Data doesn't get harder | **Curriculum learning** |
| DPO uses only 2 samples | Use **GRPO** (rewards all samples) |

---

## When CoT Hurts Performance

- **Text classification**: CoT can **decrease** accuracy
- CoT is most beneficial for **multi-step reasoning** tasks (math, logic, code)
- For simple pattern matching tasks, direct prediction is better
