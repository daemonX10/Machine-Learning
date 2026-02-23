# Lecture 08 — RLHF Part 2: Policy Optimization (REINFORCE, Actor-Critic, PPO)

## 1. Best-of-N Policy (Rejection Sampling)

The simplest alignment strategy — **no training** required.

```
1. Given prompt x, sample N responses from reference policy π_ref
2. Score each response with reward model r(x, y)
3. Return the response with the highest reward
```

| Pros | Cons |
|------|------|
| No gradient updates needed | Expensive at inference (N forward passes) |
| Simple to implement | No iterative improvement of the policy |
| Works with any reward model | Prompt-independent: knowledge doesn't transfer across prompts |

---

## 2. Reward Maximization Objective

### 2.1 Naive Objective

$$\max_\theta \; \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \big[ r(x, y) \big]$$

**Problem**: Pure reward maximization leads to **reward hacking** — the model generates nonsensical outputs that exploit reward model weaknesses to get high scores.

### 2.2 KL-Regularized Objective

Add a penalty to keep the trained policy close to the reference (pre-trained) policy:

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ r(x, y) - \beta \cdot \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

| Component | Role |
|-----------|------|
| $r(x, y)$ | Reward signal (maximize quality) |
| $\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ | KL divergence penalty (stay close to reference) |
| $\beta$ | KL penalty weight (hyperparameter) |

**KL Divergence Recap**:

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{\pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

- $\beta$ small → more reward optimization, risk of reward hacking
- $\beta$ large → stays close to reference, less improvement

**Regularized reward** (combined signal):

$$\tilde{r}(x, y) = r(x, y) - \beta \cdot \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

---

## 3. The Optimization Challenge

### Why Standard Backprop Fails

The objective involves an **expectation over samples from the policy**:

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta} [\tilde{r}(x, y)] = \sum_y \pi_\theta(y|x) \cdot \tilde{r}(x, y)$$

- **Sampling is non-differentiable** — can't backpropagate through the discrete sampling operation
- Need a way to estimate $\nabla_\theta J(\theta)$ without differentiating through sampling

---

## 4. REINFORCE (Vanilla Policy Gradient)

### 4.1 Log-Derivative Trick

Key identity:

$$\nabla_\theta \pi_\theta(y|x) = \pi_\theta(y|x) \cdot \nabla_\theta \log \pi_\theta(y|x)$$

This converts a gradient of a probability into a product, enabling Monte Carlo estimation.

### 4.2 Policy Gradient Derivation

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_y \pi_\theta(y|x) \cdot \tilde{r}(x,y)$$

$$= \sum_y \tilde{r}(x,y) \cdot \nabla_\theta \pi_\theta(y|x)$$

$$= \sum_y \tilde{r}(x,y) \cdot \pi_\theta(y|x) \cdot \nabla_\theta \log \pi_\theta(y|x)$$

$$= \mathbb{E}_{y \sim \pi_\theta} \left[ \tilde{r}(x,y) \cdot \nabla_\theta \log \pi_\theta(y|x) \right]$$

Now we can estimate this with **Monte Carlo sampling**:

### 4.3 REINFORCE Gradient Estimate

$$\hat{g} = \frac{1}{N} \sum_{i=1}^{N} \tilde{r}(x, y_i) \cdot \nabla_\theta \log \pi_\theta(y_i | x)$$

where $y_1, \ldots, y_N$ are sampled from $\pi_\theta$.

### 4.4 Token-Level Expansion

For autoregressive LLMs, $\log \pi_\theta(y|x)$ decomposes into token-level log-probs:

$$\log \pi_\theta(y|x) = \sum_{j=1}^{m} \log \pi_\theta(a_j | s_j)$$

where $a_j$ is the $j$-th token (action) and $s_j$ is the state (prompt + tokens before $j$).

Therefore:

$$\nabla_\theta \log \pi_\theta(y|x) = \sum_{j=1}^{m} \nabla_\theta \log \pi_\theta(a_j | s_j)$$

### 4.5 REINFORCE Algorithm

```
Repeat:
  1. Sample N responses y_1, ..., y_N from π_θ(·|x)
  2. Compute reward r̃(x, y_i) for each sample
  3. Compute gradient estimate:
     ĝ = (1/N) Σᵢ r̃(x, yᵢ) · ∇_θ log π_θ(yᵢ|x)
  4. Update: θ ← θ + α · ĝ
```

### 4.6 High Variance Problem

- Different samples produce **wildly different** gradient estimates
- Noisy gradients → unstable training
- Solution: **baseline subtraction**

---

## 5. Baseline Subtraction

Subtract a **baseline** $b(x)$ from the reward to reduce variance without changing the expected gradient:

$$\hat{g} = \frac{1}{N} \sum_{i=1}^{N} \big(\tilde{r}(x, y_i) - b(x)\big) \cdot \nabla_\theta \log \pi_\theta(y_i | x)$$

> **Mathematically**: $\mathbb{E}[b(x) \cdot \nabla_\theta \log \pi_\theta(y|x)] = 0$ (the baseline term vanishes in expectation), so the gradient remains unbiased.

### Common Baselines

| Baseline | Formula | Connection |
|----------|---------|------------|
| **Per-prompt mean** | $b(x) = \frac{1}{N} \sum_{i=1}^N \tilde{r}(x, y_i)$ | This is essentially **GRPO** |
| **Batch-wide mean** | $b = \frac{1}{|\mathcal{B}|} \sum_{(x,y) \in \mathcal{B}} \tilde{r}(x, y)$ | Standard REINFORCE baseline |
| **Learned value function** | $b(x) = V_\phi(x)$ | Actor-Critic methods |

After subtraction, the "advantage-like" signal:
- Positive $(\tilde{r} - b > 0)$: response was **better than average** → increase probability
- Negative $(\tilde{r} - b < 0)$: response was **worse than average** → decrease probability

---

## 6. Credit Assignment and Actor-Critic

### 6.1 The Credit Assignment Problem

REINFORCE assigns the **same reward to every token** in a response. This is problematic:

```
Prompt: "Where is the Taj Mahal?"

Response 1: "Taj Mahal is in Agra"     → reward = +1
Response 2: "Taj Mahal is in Paris"    → reward = -1

Token "Taj" gets gradient push:
  - From response 1: +1 × ∇log P("Taj")   → increase
  - From response 2: -1 × ∇log P("Taj")   → decrease
  
But "Taj" is correct in BOTH — only "Agra" vs "Paris" matters!
```

Need per-token credit: which tokens actually contributed to the reward?

### 6.2 Q-Function (Action-Value Function)

$$Q^\pi(s_t, a_t) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k \cdot r_{t+k} \;\middle|\; s_t, a_t, \pi\right]$$

- Expected cumulative reward starting from state $s_t$, taking action $a_t$, then following $\pi$
- For LLMs: $\gamma = 1$ (no discounting); reward only at end → $Q(s_t, a_t) \approx$ reward-to-go

### 6.3 Value Function (State-Value Function)

$$V^\pi(s_t) = \mathbb{E}_{a_t \sim \pi}\big[Q^\pi(s_t, a_t)\big]$$

- Expected reward from state $s_t$ **before** choosing an action
- Average over all possible next tokens weighted by policy probability

### 6.4 Advantage Function

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

| Advantage | Interpretation |
|-----------|---------------|
| $A > 0$ | This token was **better** than the average token at this position |
| $A = 0$ | This token was **average** |
| $A < 0$ | This token was **worse** than the average token at this position |

### 6.5 Policy Gradient with Advantage

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=1}^{T} A^\pi(s_t, a_t) \cdot \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

- Replaces the single response-level reward with **per-token advantage**
- Each token gets its own learning signal

---

## 7. Critic Network

### 7.1 Estimating Q and V

| Function | Estimation Method |
|----------|-------------------|
| $Q(s_t, a_t)$ | Single-sample estimate: actual reward-to-go from that trajectory |
| $V(s_t)$ | **Trained neural network** (the critic) |

### 7.2 Critic Architecture

```
Policy model (shared backbone)
    ↓
Pre-logit hidden states (one per token position)
    ↓
Additional linear layer (hidden_dim → 1)
    ↓
Scalar value V_φ(s_t) per position
```

- Critic shares the backbone with the policy (actor) model
- Only the value head (linear layer) is separate

### 7.3 Critic Training Loss

$$\mathcal{L}_{\text{critic}} = \frac{1}{T} \sum_{t=1}^{T} \big(V_\phi(s_t) - \text{reward-to-go}_t\big)^2$$

- Simple **MSE** between predicted value and actual reward-to-go
- Reward-to-go at position $t$ = cumulative reward from position $t$ onward

### 7.4 Actor-Critic Training Loop

```
Repeat:
  1. Actor (policy) generates response
  2. Get reward from reward model
  3. Compute reward-to-go for each token position
  4. Critic predicts V(s_t) for each position
  5. Compute advantage: A(s_t, a_t) = reward-to-go_t - V(s_t)
  6. Update actor: θ ← θ + α_actor · Σ_t A(s_t, a_t) · ∇_θ log π_θ(a_t|s_t)
  7. Update critic: φ ← φ - α_critic · ∇_φ Σ_t (V_φ(s_t) - reward-to-go_t)²
```

> **Adversarial analogy**: The actor tries to generate responses with reward *higher* than the critic's prediction. The critic tries to accurately predict future rewards. They push each other to improve.

---

## 8. Algorithm Hierarchy

```
Policy Gradient Methods
├── REINFORCE (no critic, response-level reward)
│   ├── + Baseline subtraction (still no critic)
│   │   └── Per-prompt mean baseline ≈ GRPO
│   └── High variance (every token gets same reward)
│
└── Actor-Critic (critic network for per-token advantage)
    ├── A2C (Advantage Actor-Critic)
    ├── A3C (Asynchronous A2C)
    └── PPO (Proximal Policy Optimization) ← most used for LLM RLHF
```

---

## 9. Summary Table

| Method | Reward Signal | Per-Token Credit? | Variance | Complexity |
|--------|--------------|-------------------|----------|------------|
| Best-of-N | Reward model | No (no training) | N/A | Low (inference only) |
| REINFORCE | Reward model | No (same reward for all tokens) | High | Low |
| REINFORCE + baseline | Reward model − baseline | No | Medium | Low |
| Actor-Critic | Advantage = Q − V | **Yes** | Low | High (need critic) |
| PPO | Clipped advantage | **Yes** | Low | High |
