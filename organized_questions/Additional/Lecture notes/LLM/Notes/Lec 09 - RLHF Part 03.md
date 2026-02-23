# Lecture 09 - Reinforcement Learning from Human Feedback (Part 03)

## Overview
This lecture covers the progression from actor-critic to PPO and GRPO: making RL training sample-efficient via importance sampling, stabilizing training with clipping (PPO), and removing the critic entirely (GRPO).

---

## 1. Recap: Vanilla Policy Gradient & Actor-Critic

### Vanilla Policy Gradient (REINFORCE)
- Directly optimizes: reward − baseline (average reward)
- **Pros**: Simple, unbiased
- **Cons**: Very high variance → needs many samples for stable gradients

### Actor-Critic
- Adds a **trainable baseline** (value function) to reduce variance
- Value function $V(s_t)$ predicts expected future reward from state $s_t$

---

## 2. Value Function in LLMs

### Architecture
- Same policy LLM processes input + generated tokens
- An **extra linear layer** added on top of the pre-logit layer
- Predicts value $V(s_t)$ at each time step

### Training the Value Function
- Minimize squared error between predicted value and actual future reward:

$$\mathcal{L}_V = \left(V_\phi(s_t) - R_t\right)^2$$

- Discount factor $\gamma = 1$ in language models (no discounting)

---

## 3. Advantage Function

### Definition

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

| Term | Meaning |
|------|---------|
| $Q(s_t, a_t)$ | Expected future reward **after taking action** $a_t$ at state $s_t$ |
| $V(s_t)$ | Expected future reward at state $s_t$ **before any action** |
| $A(s_t, a_t)$ | How much **better** token $a_t$ is compared to the average token |

### Estimating the Q-Function

#### Full Monte Carlo Estimate
- Use the actual sampled trajectory's future rewards
- $\gamma = 1 \Rightarrow Q = \sum \text{future rewards}$
- **Low bias, high variance** (depends on a single sample)

#### Temporal Difference (TD) Estimate
- $\hat{Q}(s_t, a_t) = r_t + V(s_{t+1})$
- Uses reward of current action + value of next state
- **Low variance, higher bias**

### Generalized Advantage Estimation (GAE)

The TD residual at each step:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE is a weighted combination of multi-step advantages:

$$\hat{A}_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

| Parameter | Typical Value | Role |
|-----------|--------------|------|
| $\gamma$ | 1.0 (for LLMs) | Discount factor |
| $\lambda$ | 0.95 | Bias-variance tradeoff |

- $\lambda = 0$: one-step TD (lowest variance, highest bias)
- $\lambda = 1$: full Monte Carlo (lowest bias, highest variance)

---

## 4. The Data Efficiency Problem

### On-Policy Training Loop (Wasteful)
```
Repeat:
  1. Sample prompts → Generate outputs (rollout)
  2. Compute advantage
  3. Forward pass → log probs → multiply by advantage → backprop
  4. Update weights
  5. DISCARD samples (they're from the old policy now)
```

**Problems**:
- Every weight update invalidates previous samples (they're "off-policy")
- Must regenerate samples after every update — very expensive
- Must communicate updated weights back to the sampler (slow)

---

## 5. Importance Sampling

### Core Idea
Use samples from an **old policy** $\pi_{\theta_{old}}$ to estimate gradients for the **current policy** $\pi_\theta$.

### Mathematical Foundation

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

where $\frac{p(x)}{q(x)}$ are **importance weights**.

### Per-Step Importance Weights for LLMs

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

> **Notation warning**: $r_t(\theta)$ denotes importance weights (not reward) when $\theta$ is present.

### Gradient with Importance Weights

$$\nabla_\theta J = \sum_{t=0}^{T} r_t(\theta) \cdot A_t \cdot \nabla_\theta \log \pi_\theta(a_t | s_t)$$

### Practical Implementation
1. Pass output through **old policy** → get $\log \pi_{\theta_{old}}$
2. Pass output through **current policy** → get $\log \pi_\theta$
3. Compute ratio: $r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{old}})$
4. Multiply ratio × advantage × log prob → sum → backprop

### Benefits of Importance Sampling
- Generate a **large rollout** (e.g., 1000 prompts × 16 outputs = 16,000 pairs)
- Treat as a fixed dataset → shuffle → create mini-batches
- Do **multiple epochs** of training without regenerating
- Only update the sampler after completing all training rounds

---

## 6. Problem: Importance Weights Can Blow Up

As training proceeds, $\pi_\theta$ diverges from $\pi_{\theta_{old}}$:
- Ratios $r_t(\theta)$ can become very large (→ gradient explosion)
- Ratios can become very small (→ no learning signal)

---

## 7. TRPO (Trust Region Policy Optimization)

- Constrains the policy to stay within a **trust region** around $\pi_{\theta_{old}}$
- Adds KL regularizer at every update
- Provides **very stable updates**
- **Problem**: Requires second-order derivatives → infeasible for large LLMs

---

## 8. PPO (Proximal Policy Optimization)

### Clipped Objective

$$L^{CLIP} = \mathbb{E}_t \left[\min\left(r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

### How Clipping Works

| Advantage | Clipping Applied | Reason |
|-----------|-----------------|--------|
| $A_t > 0$ (good sample) | Clip from **above** at $1+\epsilon$ | Prevent probability from increasing too much |
| $A_t < 0$ (bad sample) | Clip from **below** at $1-\epsilon$ | Prevent probability from decreasing too much |

### Intuition
- **Positive advantage**: Increase token probability, but don't overshoot
- **Negative advantage**: Decrease probability, but don't suppress completely
- Keeps updates small → allows **many updates per rollout**

### PPO Training Pipeline
```
1. Large rollout: prompts → outputs (e.g., 16,000 pairs)
2. Compute per-token advantage (GAE)
3. Compute log π_θ_old per token
4. Shuffle data → create mini-batches
5. For each mini-batch, for multiple epochs:
   a. Compute log π_θ (current policy)
   b. Compute importance ratio r_t(θ)
   c. CLIP the ratio
   d. Compute objective → backprop → update
6. Once done, regenerate samples and repeat
```

**Properties**: Sample-efficient, stable, no second-order derivatives needed

---

## 9. GRPO (Group Relative Policy Optimization)

### Motivation
- Value function adds **bias** (imperfect predictions)
- Training an additional value network is expensive
- For tasks like math/code, **verifiable rewards** exist → value function unnecessary

### Key Idea
Replace the learned value function with a **group-relative advantage**:

$$A_i = \frac{R_i - \text{mean}(R_1, ..., R_G)}{\text{std}(R_1, ..., R_G)}$$

where $R_i$ are rewards for different outputs of the **same prompt**.

### Comparison with PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Advantage source | Learned value function (critic) | Normalized group reward |
| Extra network | Yes (value function) | No |
| Best for | High diversity tasks, games | Math, code, verifiable rewards |
| Bias | Value function bias | No critic bias |

### Dr. GRPO Variant
- Removes the standard deviation division
- Reason: dividing by std doesn't discriminate between highly differentiated vs. similar responses

### When to Use

| Use GRPO | Use PPO |
|----------|---------|
| Verifiable rewards (math, code) | Subjective tasks |
| Limited response diversity | High diversity in possible outcomes |
| Want to avoid value function bias | Games, robotics |

---

## 10. Complete RLHF Journey Summary

```
Policy Gradient → high variance
    ↓ add trainable baseline
Actor-Critic → wasteful (sample once, train once)
    ↓ add importance sampling
Off-Policy Actor-Critic → importance weights blow up
    ↓ add clipping / trust region
PPO → needs extra value network
    ↓ remove critic, use group normalization
GRPO → simple, stable, efficient for verifiable rewards
```

---

## Key Resources
- OpenAI Spinning Up tutorials (Actor-Critic, PPO implementations)
- DeepSeek-Math paper (introduces GRPO)
- PPO paper (Schulman et al.)
- TRL framework (HuggingFace) for PPO/GRPO training
