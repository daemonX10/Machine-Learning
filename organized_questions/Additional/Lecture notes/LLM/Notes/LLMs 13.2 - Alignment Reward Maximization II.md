# LLMs 13.2 — Alignment: Reward Maximization II

## 1. Recap — Regularized Reward Maximization

### Objective

$$\underset{\pi_\theta}{\max} \; \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \Big[ R(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \Big]$$

- **First term:** Maximize expected reward of outputs sampled from the policy
- **Second term:** Minimize KL divergence from reference policy
- $\beta$ controls the trade-off

### Regularized Reward

$$R^s(x, y) = R(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

**Intuition:** Outputs should have high reward **and** reasonable probability under the reference model.

---

## 2. The Problem: Sampling Blocks Backpropagation

### Why Can't We Just Backpropagate?

```
Input → Policy π_θ → [SAMPLING] → Output y → Reward R(x,y)
                        ↑
              Non-differentiable!
```

- Sampling **discrete tokens** is a non-differentiable operation
- Gradient cannot flow from reward back through the sampling step to the policy parameters
- Need an alternative: **REINFORCE algorithm**

---

## 3. REINFORCE — The Log Derivative Trick

### Step 1: Write Gradient of the Objective

$$\nabla_\theta \, \mathbb{E}_{y \sim \pi_\theta} [R^s(x, y)] = \sum_{y \in \mathcal{Y}} \nabla_\theta \, \pi_\theta(y \mid x) \cdot R^s(x, y)$$

This exact summation is **intractable** (infinite output space).

### Step 2: Log Derivative Trick

Key identity:

$$\nabla_\theta \, \pi_\theta = \pi_\theta \cdot \nabla_\theta \log \pi_\theta$$

Apply this substitution:

$$\nabla_\theta \, \mathbb{E}[R^s] = \sum_{y \in \mathcal{Y}} \pi_\theta(y \mid x) \cdot \nabla_\theta \log \pi_\theta(y \mid x) \cdot R^s(x, y)$$

### Step 3: Recognize as Expectation

$$= \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \Big[ R^s(x, y) \cdot \nabla_\theta \log \pi_\theta(y \mid x) \Big]$$

Now we can **approximate using samples** (Monte Carlo):

$$\approx \frac{1}{N} \sum_{i=1}^{N} R^s(x, y_i) \cdot \nabla_\theta \log \pi_\theta(y_i \mid x), \quad y_i \sim \pi_\theta$$

---

## 4. Token-Level Expansion

### Decompose Sequence Probability

For output $y = (a_1, a_2, \ldots, a_T)$:

$$\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(a_t \mid s_t)$$

where $s_t$ = prompt + tokens generated so far.

### REINFORCE at Token Level

$$\sum_{t=1}^{T} R^s(x, y) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

**Key insight:** Every token gets the **same reward** — the reward of the complete output.

### Implementation

- Feed input + generated output to the LLM
- Multiply each token's log probability by the total reward
- Sum and backpropagate

> **REINFORCE = Supervised fine-tuning on generated outputs, weighted by reward.**
> Without the reward term $R$, this is exactly standard fine-tuning.

---

## 5. Variance Problem and the Q-Function

### The Problem

- Token "Taj" always gets the reward of the **complete sequence**
- If sequence = "Taj Mahal is in Agra" → positive reward → increase P("Taj")
- If sequence = "Taj Mahal is in Paris" → negative reward → decrease P("Taj")
- But "Taj" itself has negligible impact on reward — it's "Agra" vs. "Paris" that matters
- **Result:** Huge variance in training, unstable updates

### Solution: Q-Function

Replace the total reward with the **Q-function** at each token:

$$Q(s_t, a_t) = \text{Average discounted cumulative reward over all likely sequences starting with } a_t \text{ at state } s_t$$

$$Q(s_t, a_t) = \mathbb{E}_{\substack{a_{t+1}, a_{t+2}, \ldots \\ \sim \pi_\theta}} \Big[ \sum_{k=0}^{T-t} \gamma^k \cdot R_{t+k} \Big]$$

where $\gamma \in [0.9, 0.99]$ is the **discount factor** (future rewards weighted less).

### Value Function

Defined only for a state (averaging over all possible actions):

$$V(s_t) = \mathbb{E}_{a_t, a_{t+1}, \ldots \sim \pi_\theta} \Big[ \sum_{k=0}^{T-t} \gamma^k \cdot R_{t+k} \Big]$$

### Relationship

$$Q(s_t, a_t) = R(s_t, a_t) + \gamma \cdot V(s_{t+1})$$

where $s_{t+1} = s_t \oplus a_t$.

---

## 6. Advantage Function

### Definition

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

**Intuition:**
- $Q(s_t, a_t)$: expected reward if I take action $a_t$ then follow policy
- $V(s_t)$: expected reward if I take an **average** action then follow policy
- $A(s_t, a_t)$: how much **better** is action $a_t$ compared to the average action

### Why Advantage Reduces Variance

- Subtracting $V(s_t)$ removes the baseline (average contribution)
- Smaller numbers multiplied by gradients → lower variance
- Only the **relative advantage** of the specific action matters

### Final REINFORCE with Advantage

$$\nabla_\theta J = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^{T} A(s_t, a_t) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

---

## 7. Learning the Value Function

### Architecture

- Add a **linear layer** on top of the decoder model: $\text{embedding} \to \mathbb{R}^1$
- Same linear layer applied at every token position
- Outputs a scalar value $V_\phi(s_t)$ at each time step

### Training

1. Sample output $y = (a_0, \ldots, a_T)$ from policy
2. Compute **reward-to-go** at each token:

$$\hat{R}_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots$$

3. Minimize MSE:

$$\min_\phi \sum_t \big(V_\phi(s_t) - \hat{R}_t\big)^2$$

---

## 8. Vanilla Policy Gradient Algorithm

```
Repeat until convergence:
  1. Sample batch of prompts
  2. For each prompt, generate one or more outputs
  3. For each output:
     a. Compute reward at each token
     b. Compute cumulative discounted reward (reward-to-go)
     c. Compute value and advantage function at each token
  4. Apply gradient updates using REINFORCE with advantage
  5. Apply gradient updates to train value function (MSE loss)
```

### Problem: On-Policy Requirement

- Every gradient update requires **new samples** from the current policy
- Slows training drastically
- Samples come from a constantly changing distribution → unstable

---

## 9. Importance Sampling — Off-Policy Training

### Goal

Sample from a **fixed policy** $\pi_{\theta_k}$ instead of the current $\pi_\theta$.

### Derivation

$$\mathbb{E}_{y \sim \pi_\theta}[f(y)] = \sum_y \pi_\theta(y \mid x) \cdot f(y) = \sum_y \pi_{\theta_k}(y \mid x) \cdot \frac{\pi_\theta(y \mid x)}{\pi_{\theta_k}(y \mid x)} \cdot f(y)$$

$$= \mathbb{E}_{y \sim \pi_{\theta_k}} \left[ \frac{\pi_\theta(y \mid x)}{\pi_{\theta_k}(y \mid x)} \cdot f(y) \right]$$

The ratio $\frac{\pi_\theta(y \mid x)}{\pi_{\theta_k}(y \mid x)}$ is the **importance weight**.

### Implementation

Feed generated tokens to both:
- **Fixed policy** $\pi_{\theta_k}$ (frozen)
- **Trainable policy** $\pi_\theta$ (updated)

Multiply gradients by importance weights and advantage, then backpropagate.

---

## 10. PPO (Proximal Policy Optimization)

### Problem with Importance Weights

As $\pi_\theta$ diverges from $\pi_{\theta_k}$, importance weights become very large → unstable training.

### PPO Clipping

PPO bounds the importance ratio to $[1 - \epsilon, 1 + \epsilon]$:

$$L^{\text{PPO}}(\theta) = \mathbb{E} \left[ \min \left( \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)} \cdot A_t, \; \text{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_k}}, 1-\epsilon, 1+\epsilon\right) \cdot A_t \right) \right]$$

### How Clipping Works

| Advantage | Behavior |
|---|---|
| **Positive** $A > 0$ | Ratio increases (probability increasing). Clipped at $1 + \epsilon$ → stops when policy deviates too much in the positive direction |
| **Negative** $A < 0$ | Ratio decreases (probability decreasing). Clipped at $1 - \epsilon$ → stops when policy deviates too much in the negative direction |

**When ratio hits the clip boundary**, the objective becomes a constant (no gradient) → training stops for that token.

### PPO Algorithm

```
Repeat until convergence:
  1. Sample batch of prompts
  2. Sample outputs from FIXED policy π_θk
  3. Compute rewards and advantages
  4. Apply several gradient steps of PPO-clip
  5. Apply gradient updates to value function
  6. Update fixed policy: θk ← θ
  7. Go to step 2
```

---

## 11. Summary — Evolution of Policy Gradient

| Method | Token Weight | Key Improvement |
|---|---|---|
| REINFORCE | Total reward $R(x,y)$ | Enables gradient estimation via sampling |
| + Q-Function | $Q(s_t, a_t)$ | Reduces variance (average future reward) |
| + Advantage | $A(s_t, a_t) = Q - V$ | Further variance reduction |
| + Importance Sampling | + importance weight $\frac{\pi_\theta}{\pi_{\theta_k}}$ | Off-policy training (sample once, update multiple times) |
| **PPO** | + clipped importance weight | Prevents large destructive updates |

## 12. Key Takeaways

1. **Log derivative trick** converts intractable gradient into a sampleable expectation
2. **Advantage function** replaces raw reward to reduce variance at intermediate tokens
3. **Importance sampling** allows sampling from a fixed policy instead of continuously re-sampling
4. **PPO clips importance weights** to prevent large gradient updates when policy deviates too far
5. PPO alternates between generating samples (from fixed policy) and performing clipped gradient updates
