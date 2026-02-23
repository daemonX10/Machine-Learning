# LLMs 13.3 — Alignment: Contrastive Learning (DPO)

## 1. Recap — PPO Pipeline

```
Step 1: Collect human/AI preferences → (x, y⁺, y⁻)
Step 2: Train reward model (Bradley-Terry log-likelihood)
Step 3: Train policy using reward model (PPO)
        - Requires: reward model, value function, importance weight clipping
```

### PPO Objectives

**Reward model training:**

$$\max_\phi \sum \log \sigma\!\Big(R_\phi(x, y^+) - R_\phi(x, y^-)\Big)$$

**Policy optimization:**

$$\max_\theta \; \mathbb{E}_{y \sim \pi_\theta}\!\left[R(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

---

## 2. DPO — Core Idea

**Question:** Can we skip the reward model and train the policy **directly** on preferences?

| PPO Pipeline | DPO Pipeline |
|---|---|
| Preference data → Reward model → Policy training | Preference data → **Direct** policy training |

DPO eliminates the intermediate step of learning a reward model.

---

## 3. Derivation of DPO

### Step 1: Optimal Policy in Terms of Optimal Reward

Start from the regularized reward maximization objective. Using Lagrangian optimization:

$$\max_{\pi} \; \mathbb{E}_{y \sim \pi}\!\left[R^*(x,y) - \beta \, D_{\text{KL}}(\pi \| \pi_{\text{ref}})\right]$$

Differentiating w.r.t. $\pi(y|x)$ and setting to zero:

$$\pi^*(y \mid x) = \pi_{\text{ref}}(y \mid x) \cdot \frac{\exp\!\big(R^*(x,y) / \beta\big)}{Z(x)}$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\!\big(R^*(x,y)/\beta\big)$ is the normalization constant.

### Step 2: Express Reward in Terms of Policy

Rearranging:

$$R^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

### Step 3: Key DPO Assumption

**DPO assumes** that even during training (non-optimal), the reward function has this form:

$$R_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

This parameterizes the reward function **entirely in terms of the policy**.

### Step 4: Substitute into Bradley-Terry Loss

For a preference pair $(x, y^+, y^-)$:

$$\mathcal{L}_{\text{DPO}}(\theta) = \log \sigma\!\Big(R_\theta(x, y^+) - R_\theta(x, y^-)\Big)$$

Substituting the reward parameterization:

$$R_\theta(x, y^+) - R_\theta(x, y^-) = \beta \left[\log \frac{\pi_\theta(y^+ \mid x)}{\pi_{\text{ref}}(y^+ \mid x)} - \log \frac{\pi_\theta(y^- \mid x)}{\pi_{\text{ref}}(y^- \mid x)}\right]$$

> The $\beta \log Z(x)$ terms **cancel out** (they depend only on $x$, not $y$).

### Final DPO Objective

$$\boxed{\max_\theta \sum_{(x, y^+, y^-)} \log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y^+ \mid x)}{\pi_{\text{ref}}(y^+ \mid x)} - \log \frac{\pi_\theta(y^- \mid x)}{\pi_{\text{ref}}(y^- \mid x)}\right]\right)}$$

---

## 4. DPO as Softmax Classification

The DPO loss can be rewritten as:

$$\mathcal{L}_{\text{DPO}} = -\log \frac{\exp\!\big(\beta \log \frac{\pi_\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)}\big)}{\exp\!\big(\beta \log \frac{\pi_\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)}\big) + \exp\!\big(\beta \log \frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}\big)}$$

**Interpretation:** This is a **softmax/cross-entropy loss** where:
- **Logits** = $\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ (log-ratio of probabilities)
- Two classes: preferred ($y^+$) and rejected ($y^-$)
- Increase logit of $y^+$, decrease logit of $y^-$

### Differences from Standard Classification

1. **Logits** are computed from policy/reference probability ratios
2. **Output space** is intractably large — loss only considers $y^+$ and $y^-$ (not all possible outputs)

---

## 5. Role of $\pi_{\text{ref}}$ in DPO

### Practical: Length Normalization

- Longer sequences have lower absolute probabilities
- The ratio $\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ normalizes for length
- Without $\pi_{\text{ref}}$, model would bias toward very short or very long outputs

### Theoretical: Implicit KL Constraint

- If model already assigns high probability (e.g., 0.8) → needs large increase (to 0.9) for same ratio change
- If model assigns low probability (e.g., 0.1) → small increase (to 0.11) achieves same ratio
- **Effect:** Model doesn't drastically change probabilities relative to reference

### Hyperparameter $\beta$

- **High $\beta$** → model aggressively separates positive from negative (like high learning rate)
- **Low $\beta$** → conservative updates
- Typical range: **0.003 to 0.3** (depends on output lengths)

---

## 6. PPO vs. DPO Comparison

| Aspect | PPO | DPO |
|---|---|---|
| **Reward model** | Required (separate training) | Not needed |
| **Value function** | Required | Not needed |
| **Implementation** | Complex (reward model + value function + clipping) | Simple (softmax-like loss) |
| **Output coverage** | Considers full output distribution | Only considers $y^+$ and $y^-$ |
| **Main risk** | Length bias from reward model | Out-of-distribution bias |
| **Convergence** | Harder to implement correctly | Easier but can be unstable |

---

## 7. DPO's Out-of-Distribution (OOD) Bias Problem

### The Failure Mode

Initially: $\pi_\theta(y^+) = 0.5$, $\pi_\theta(y^-) = 0.5$, $\pi_\theta(y_{\text{nonsense}}) = 0$

During DPO training, as $\pi_\theta(y^-)$ decreases:
- Some probability should flow to $y^+$ ✓
- But some probability **leaks** to $y_{\text{nonsense}}$ ✗

| Step | $\pi(y^+)$ | $\pi(y^-)$ | $\pi(y_{\text{nonsense}})$ |
|---|---|---|---|
| Initial | 0.5 | 0.5 | 0.0 |
| After training | 0.6 | 0.1 | **0.3** |

### Why This Happens

- DPO loss only considers $y^+$ and $y^-$
- It has no mechanism to prevent probability flow to outputs **not in the loss**
- Model can make $\pi_\theta(y^-) \to 0$ (log → $-\infty$, exp → 0) to trivially maximize the objective
- The "freed" probability mass goes to arbitrary outputs

### Symptoms

- Nonsensical outputs emerge after extended training
- Repetitive text / degenerate outputs

---

## 8. Online DPO — Fixing OOD Bias

### Algorithm

```
Repeat:
  1. Generate outputs y₁, y₂ from current policy π_θ
  2. Score using reward model (or human): determine y⁺, y⁻
  3. Apply DPO update on (x, y⁺, y⁻)
  4. Go to step 1
```

### Why It Works

- If a nonsensical output's probability increases → it gets **sampled**
- Reward model gives it **negative reward** → it becomes $y^-$ in DPO
- Policy learns to suppress it immediately

> The reward model acts as an **adversary** to the DPO objective — any bad output the policy produces gets immediately suppressed.

### Key Requirement

- **Resample frequently** so bad outputs are caught early

### Empirical Results

Online DPO **significantly outperforms** offline DPO on:
- TL;DR (summarization)
- Helpfulness
- Harmlessness (Anthropic benchmarks)

---

## 9. PPO vs. DPO Known Biases

| Method | Known Bias |
|---|---|
| **PPO** | **Length bias** — reward models tend to prefer longer outputs → PPO produces verbose outputs |
| **DPO** | **OOD bias** — probability mass leaks to nonsensical outputs not in training pairs |

---

## 10. Key Takeaways

1. **DPO eliminates** the need for a separate reward model and value function
2. DPO derives from expressing the **reward function as a function of the policy**, then substituting into Bradley-Terry loss
3. The DPO loss is equivalent to **softmax classification** with log-ratio logits
4. $\pi_{\text{ref}}$ serves as both **length normalizer** and **implicit KL constraint**
5. DPO suffers from **OOD bias** — probability leaks to outputs outside the training pairs
6. **Online DPO** fixes this by resampling from the policy and using a reward model to catch bad outputs
7. Neither PPO nor DPO is strictly better — each has its own failure modes
