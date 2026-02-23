# Lecture 10 - Reinforcement Learning from Human Feedback (Part 04)

## Overview
This lecture derives DPO (Direct Preference Optimization) from the RLHF objective, eliminating the need for a separate reward model. Covers the DPO objective, role of beta, reward hacking, and online DPO.

---

## 1. From On-Policy RL to Contrastive Learning

### Standard RLHF Pipeline
```
Preferences → Train Reward Model → Train Policy (PPO/GRPO)
```
- Reward model introduces bias (trained on limited data)
- Two-stage pipeline is complex (actor-critic, importance weights, clipping)

### DPO Question
> Can we skip reward model training and learn the policy **directly** from preference pairs?

### Contrastive Learning Approach
Given a preferred response $y_w$ and rejected response $y_l$:
- **Increase** probability of $y_w$
- **Decrease** probability of $y_l$
- While staying close to the reference policy (KL constraint)

---

## 2. DPO Derivation

### Step 1: Optimal Policy from RLHF Objective

The RLHF objective:

$$\pi^* = \arg\max_\pi \mathbb{E}_{y \sim \pi}[r^*(x, y)] - \beta \cdot D_{KL}(\pi \| \pi_{ref})$$

### Step 2: Lagrangian Formulation

Since $\pi$ must be a valid probability distribution ($\sum_y \pi(y|x) = 1$):

$$\mathcal{L}(\pi, \lambda) = \sum_y \pi(y|x) \cdot r^*(x,y) - \beta \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \lambda\left(\sum_y \pi(y|x) - 1\right)$$

### Step 3: Differentiating and Setting to Zero

$$\frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r^*(x,y) - \beta\left(1 + \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}\right) + \lambda = 0$$

### Step 4: Express Optimal Reward in Terms of Optimal Policy

$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + (\beta - \lambda)$$

### Step 5: Express Optimal Policy in Terms of Optimal Reward

$$\pi^*(y|x) \propto \pi_{ref}(y|x) \cdot \exp\left(\frac{r^*(x,y)}{\beta}\right)$$

**Interpretation**: Optimal policy assigns high probability to $y$ only if:
1. $\pi_{ref}$ already assigns high probability (KL constraint)
2. Reward $r^*(x, y)$ is high

---

## 3. The DPO Objective

### Plugging Reward into BT Model

Substitute the reward expression into the Bradley-Terry preference model loss:

$$\mathcal{L}_{BT} = -\log \sigma(r^*(x, y_w) - r^*(x, y_l))$$

The constant $(\beta - \lambda)$ cancels between preferred and rejected:

### Final DPO Loss

$$\boxed{\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]}$$

### Implicit Reward

$$\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

This is the **implicit reward** — no separate reward model needed.

---

## 4. DPO Implementation

```
For each (prompt, y_preferred, y_rejected):
  1. Pass y_w through π_θ → get log π_θ(y_w|x)
  2. Pass y_w through π_ref → get log π_ref(y_w|x)
  3. Compute implicit reward_w = β · (log π_θ(y_w) - log π_ref(y_w))
  4. Pass y_l through π_θ → get log π_θ(y_l|x)
  5. Pass y_l through π_ref → get log π_ref(y_l|x)
  6. Compute implicit reward_l = β · (log π_θ(y_l) - log π_ref(y_l))
  7. Loss = -log σ(reward_w - reward_l)
  8. Backprop through π_θ only (π_ref is frozen)
```

**Simplicity**: As easy to implement as SFT — no sampling, no variance issues, no gradient explosion concerns.

---

## 5. Where is the KL Divergence?

DPO has **no explicit KL term**, yet stays close to the reference policy.

### How KL is Implicitly Enforced

The implicit reward is a **ratio** $\frac{\pi_\theta(y)}{\pi_{ref}(y)}$, not an absolute probability.

| Scenario | $\pi_{ref}(y)$ | After DPO $\pi_\theta(y)$ | Ratio |
|----------|----------------|---------------------------|-------|
| Reference already confident | 0.8 | 0.88 | 1.1 |
| Reference assigns low prob | 0.1 | 0.11 | 1.1 |

**Key insight**: Even with the same ratio improvement, the absolute probability change is small when $\pi_{ref}$ assigns low probability. The policy scales **relative to** the reference, not absolutely.

---

## 6. Role of Beta (β)

### In RLHF
- Higher β → more KL regularization → policy stays closer to reference

### In DPO (Non-Intuitive)

The derivative of $\log \sigma(\beta x)$ w.r.t. $x$:

| $\beta$ range | Effect |
|---------------|--------|
| Increasing (before saturation) | Model tries to **increase** implicit reward gap |
| Beyond saturation point | Gradient **decreases** → policy becomes **less contrasting** |

### Practical Guidelines
- Start with $\beta \approx 0.1$–$0.5$
- Monitor KL divergence and accuracy
- If KL increases too fast → increase β
- If accuracy saturates → decrease β

---

## 7. Reward Hacking

### In Policy Gradient Methods
- Reward model has biases (e.g., prefers longer outputs, specific tokens)
- Policy exploits these biases → generates repetitive/meaningless outputs

**Solutions**:
- **Ensemble** multiple reward models → average reduces individual biases
- **Diversity constraints** on outputs

### In DPO
- No reward model, but **preference pairs** can be hacked
- Model may latch onto superficial features (grammar, word choice) instead of the actual reason for preference
- Cannot understand *why* a response was preferred

**Solutions**:
- Use more diverse preference pairs
- Add supervised loss on the positive response

---

## 8. Probability Mass Leakage in DPO

### The Problem
DPO only controls probability of $y_w$ and $y_l$ — says nothing about other outputs.

| Step | $\pi_\theta(y_w)$ | $\pi_\theta(y_l)$ | $\pi_\theta(y_{other})$ |
|------|-------|-------|---------|
| Initial | 0.5 | 0.5 | 0.0 |
| After training | 0.6 | 0.1 | **0.3** (uncontrolled!) |

The probability mass taken from $y_l$ can leak to **arbitrary unseen outputs**, including meaningless ones.

### When It's Worse
- Reference policy has high entropy (flat distribution)
- Training runs for many steps

---

## 9. Online DPO

### Solution to Probability Leakage
Instead of fixed preference pairs, create them **on the fly**:

```
Repeat:
  1. Sample outputs from current π_θ
  2. Use reward model / LLM-as-judge to rank them
  3. Create preference pairs (y_w, y_l)
  4. Train with DPO objective
  5. Go to step 1
```

### Why It Works
- If a bad output (e.g., "the the the...") gains probability, it will be **sampled**
- The judge will label it as rejected → model learns to suppress it
- Creates a **self-correcting feedback loop**

### Performance
- Online DPO **significantly outperforms** offline DPO in win-rate benchmarks

---

## 10. Summary: DPO vs RLHF

| Aspect | RLHF (PPO/GRPO) | DPO |
|--------|-----------------|-----|
| Reward model | Required (explicit) | Not needed (implicit) |
| Training complexity | High (sampling + training loop) | Low (similar to SFT) |
| Variance issues | Yes (sampling-based) | No (fixed pairs) |
| KL enforcement | Explicit KL term | Implicit via log ratio |
| Reward hacking | Via reward model bias | Via preference pair bias |
| Online variant | Inherent (always sampling) | Online DPO (recommended) |

---

## Key Resources
1. Original DPO paper (Rafailov et al.)
2. DPO explanation blog
3. Online DPO paper
4. HuggingFace DPO Trainer
5. "Build DPO from Scratch" tutorial
