# Lecture 10.1 — Mixture of Experts (MoE) – Part I

---

## 1. Motivation: Why Increase Model Size?

### The Pre-training & Fine-tuning Paradigm
- Originated in CV (VGG/ResNet on ImageNet → fine-tune classifier head)
- Transformers brought this to NLP: large models pre-trained by big labs, then fine-tuned with parameter-efficient techniques (LoRA, adapters, etc.)

### Model Size Evolution

| Model | Parameters | Training Cost (A100-days) |
|-------|-----------|--------------------------|
| VGGNet | 144M | — |
| RoBERTa | 355M | 410 |
| GPT-2 | 1.5B | — |
| GPT-3 | 175B | 60,800 |
| Hypothetical 1.6T | 1.6T | ~9 months on 2,800 GPUs |

### Neural Scaling Laws (Preview)
- **Key insight**: Test loss decreases monotonically as model parameters increase (given sufficient data & compute)
- Similar curves exist for dataset size and compute
- **Given fixed compute**: training a **larger model for fewer steps** is more efficient than a smaller model for more steps
- This drives the race to increase model size — but brute-force scaling is infeasible

> **Solution**: Mixture of Experts — many parameters, but only a **subset active** at any time

---

## 2. Mixture of Experts — Core Concept

### Original MoE (1991, Hinton's Group)

**Architecture**: Replace one model with $n$ expert models + a **gating network**

$$y = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)$$

Where:
- $E_i(x)$ = output of expert $i$
- $G(x)_i$ = gating probability for expert $i$ (softmax over all experts)

In the original (dense) MoE, **all experts are active** → goal was to increase model **capacity**, not reduce compute.

### Chronology of MoE

| Year | Milestone |
|------|-----------|
| 1991 | MoE introduced (Hinton) — experts could be NNs, SVMs, Gaussian mixtures |
| 2013 | MoE used as a **layer** inside deep networks |
| 2021 | **Switch Transformer** (Google Brain) — MoE layer in Transformer LLMs |
| 2023–24 | **Mixtral** (8×7B) — open-weight, commercially friendly; surpasses GPT-3.5 Turbo, Claude, Gemini Pro, LLaMA-2 70B |

---

## 3. MoE as a Layer in Transformers

### Dense vs. Sparse Transformer Block

| Component | Dense Block | MoE Block |
|-----------|------------|-----------|
| Attention | Standard multi-head attention | Same (shared) |
| FFN | Single feed-forward network | $n$ expert FFNs + router |
| Residual + LayerNorm | Standard | Same |

### Token-Level Routing
- Each **token embedding** is independently routed by the router (not the entire sequence)
- Example: Token "more" → Expert 2 (prob 0.65); Token "parameters" → Expert 1 (prob 0.8)
- In each layer, different tokens can go to different experts
- Assignment can differ **across layers** for the same token

### Why Training is Faster (Conditional Activation)
- ~80% of Transformer parameters are in FFN layers; attention ≈ 20%
- With 8 experts: model size ≈ 8× larger, but only **1 expert active per token**
- **Compute per token = same as dense model** (plus negligible router cost)
- This is called **conditional/sparse activation**

---

## 4. Backpropagation in MoE

### Dense MoE (All Experts Active)

Output for token $x$:

$$y = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)$$

Gradients via chain rule:

$$\frac{\partial \mathcal{L}}{\partial E_i(x)} = \frac{\partial \mathcal{L}}{\partial y} \cdot G(x)_i$$

$$\frac{\partial \mathcal{L}}{\partial G(x)_i} = E_i(x)^\top \cdot \frac{\partial \mathcal{L}}{\partial y}$$

### Sparse MoE (Switch Transformer — Top-1 Routing)

Select expert $i^* = \arg\max_i G(x)_i$:

$$y = G(x)_{i^*} \cdot E_{i^*}(x)$$

- All gradients for experts $i \neq i^*$ are **zero** → speeds up backpropagation

### Noisy Top-K Gating (Shazeer et al.)

Logits with learnable noise:

$$H(x)_i = W_g \cdot x + \text{SoftPlus}(W_{\text{noise}} \cdot x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

- $W_g$: learnable gating weights
- $W_{\text{noise}}$: learnable noise scale (passed through **SoftPlus** to ensure positive std dev)
- **Top-K operator**: keep top $k$ logits, set rest to $-\infty$
- Softmax over filtered logits → sparse routing probabilities
- Noise improves training stability and learnability

---

## 5. Why Don't Experts Collapse?

### Synthetic Experiment Setup
- 4 clusters, each linearly separable
- Ideal: router assigns each cluster to one expert; expert learns the boundary

### Key Finding: Nonlinearity is Essential

| Expert Type | Outcome |
|-------------|---------|
| **Linear experts** | Loss is convex → all experts converge to **same** minima (collapse) |
| **Nonlinear experts** | Non-convex loss → experts **diverge** to different regions in parameter space |

### Two-Stage Learning Dynamics

| Stage | What Happens | Entropy |
|-------|-------------|---------|
| **1. Exploration** | Router is untrained, assigns near-uniform probabilities; experts start diverging | High |
| **2. Exploitation** | Experts are now distinct; router learns to route correctly | Decreasing |

- **Entropy** of routing distribution quantifies randomness:
  - High entropy = uniform routing (random)
  - Low entropy = confident routing to specific expert

---

## 6. Pros and Cons of MoE

### Advantages
- Scale model parameters efficiently without proportional compute increase
- Efficient pre-training via sparse/conditional computation
- Faster inference (same compute as dense equivalent)

### Challenges

| Issue | Description |
|-------|-------------|
| **Unstable training** | Random initialization can cause training divergence (loss spikes) |
| **Router collapse** | Router learns to send all tokens to one expert → 7 of 8 experts sit idle |
| **High memory requirement** | All experts must reside in GPU memory even if only one is active at a time — compute is efficient, but memory footprint remains large |

---

## 7. Key Takeaways

1. MoE enables scaling model size while keeping per-token compute constant
2. Replace the FFN layer in Transformers with multiple expert FFNs + a learned router
3. Routing happens at the **token level**, independently per layer
4. **Nonlinearity in experts** prevents collapse; router learns in two phases (exploration → exploitation)
5. Main challenges: training instability, router collapse, high GPU memory — addressed in Part II
