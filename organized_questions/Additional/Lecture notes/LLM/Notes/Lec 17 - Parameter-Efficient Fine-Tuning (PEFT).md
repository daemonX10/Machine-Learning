# Lecture 17 — Parameter-Efficient Fine-Tuning (PEFT)

---

## 1. Motivation

### Full Fine-Tuning Problems
- GPT-3: 175B params, ~1200 MWh power, 355 GPU-years, 700k liters of water
- Fine-tuning all parameters: expensive, non-reusable, high carbon footprint
- Separate fine-tuned copy needed for every task

### PEFT Goal
Fine-tune a **small subset** of parameters per task. The core model stays frozen.

**Benefits**: reduced compute, modular (plug-and-play), reusable, personalizable.

---

## 2. PEFT Taxonomy

| Category | Approach | Model Size Change |
|----------|----------|-------------------|
| **Additive** | Add new parameters/modules | Slight increase |
| **Selective** | Choose existing parameters to fine-tune | No change |
| **Reparameterization** | Re-express weight updates in low-rank form | No change (at inference) |

---

## 3. Additive Methods

### 3.1 Adapters

Insert small **bottleneck modules** between transformer layers:

```
Input (d) → Down-projection (d → r) → Nonlinearity → Up-projection (r → d) → + Residual → Output (d)
```

where $r \ll d$.

| Property | Detail |
|----------|--------|
| Structure | Bottleneck (down → nonlinearity → up + skip) |
| Placement | After attention and/or FFN blocks |
| Parameters | Only adapter weights are trainable |
| Result | **3.6% additional params → 99.6% of full fine-tuning performance** |

**Why down→up (not up→down)?** Up→down would increase the parameter count (larger intermediate dimension).

**Residual connection** is critical to preserve the original representation.

### 3.2 Prefix Tuning

Prepend **learnable vectors** to the K and V at **every layer**:

| Layer | Input |
|-------|-------|
| Layer 1 | $[\text{prefix}_1 ; \text{hidden}_1]$ |
| Layer 2 | $[\text{prefix}_2 ; \text{hidden}_2]$ |
| ... | ... |

- Prefix vectors affect K, V computation at that layer
- Each task has its own set of prefix vectors
- Only prefix parameters are updated during training

### 3.3 Prompt Tuning

Simpler than prefix tuning: prepend learnable tokens **only at the input layer**:

$$\text{Input} = [\text{soft\_prompt}; x_1, x_2, \ldots, x_T]$$

| Property | Prefix Tuning | Prompt Tuning |
|----------|---------------|---------------|
| Where added | Every layer's K, V | Input only |
| Parameters | More | Fewer (< 0.1%) |
| Effectiveness | Better | Slightly weaker with less data |

**Multi-task batching**: different prompt tokens prepended per task within the same batch.

---

## 4. Selective Methods

### Structured vs Unstructured

| Type | What is Selected | Hardware Friendly? |
|------|------------------|-------------------|
| **Structured** | Entire blocks (FFN layer, attention weights) | Yes |
| **Unstructured** | Individual parameters from matrices | Less so |

### 4.1 BitFit (2022)

Fine-tune **only bias terms** across the model.

$$Q = W_Q x + \mathbf{b_Q}, \quad K = W_K x + \mathbf{b_K}, \quad V = W_V x + \mathbf{b_V}$$

Only $\mathbf{b_Q}, \mathbf{b_K}, \mathbf{b_V}$, and FFN biases are updated.

**Empirical finding**: Query bias and second FFN layer bias are the most important.

**Limitation**: Not applicable to bias-free architectures (e.g., LLaMA uses RMSNorm without bias).

### 4.2 Fish Mask (Unstructured)

Use **Fisher Information** to rank parameter importance:

$$F_\theta = \mathbb{E}\left[\nabla_\theta \log p(y|x;\theta) \cdot \nabla_\theta \log p(y|x;\theta)^T\right]$$

**Approximation**: Diagonal of Fisher → one importance score per parameter.

**Procedure**:
1. Compute Fisher information per parameter (Monte Carlo approximation)
2. Rank all parameters by Fisher score
3. Select top-$k\%$ parameters for fine-tuning

### 4.3 PuFFy (Magnitude-Based)

**Opposite** of magnitude pruning: select parameters with the **lowest magnitude**.

**Intuition**: Low magnitude parameters are **undertrained** → most room for improvement.

| Method | Selects | Rationale |
|--------|---------|-----------|
| Fish Mask | Highest Fisher info | Most impactful parameters |
| PuFFy | Lowest magnitude | Most undertrained parameters |

**Advantage**: No computation needed beyond sorting by magnitude.

### 4.4 ID3 (Incremental Data-Driven Selection)

**Key idea**: Select parameters **incrementally** over $T$ iterations, not all at once.

Budget $K$ total parameters, select $K/T$ per iteration:

| Iteration | Action |
|-----------|--------|
| 1 | Select $K/T$ parameters using importance criterion |
| 2 | Fine-tune selected params; select additional $K/T$ |
| ... | Continue; each selection considers cascading effects |
| $T$ | All $K$ parameters selected and fine-tuned |

**Advantage over Fish Mask**: Captures **cascading effects** — parameters selected later account for the impact of earlier fine-tuning.

Can be applied **on top of LoRA** for further efficiency.

---

## 5. Reparameterization Methods

### 5.1 LoRA (Low-Rank Adaptation)

**Core idea**: The weight update $\Delta W$ is low-rank.

$$W' = W_0 + \Delta W = W_0 + BA$$

where:
- $W_0 \in \mathbb{R}^{d \times d}$: frozen pre-trained weight
- $A \in \mathbb{R}^{d \times r}$: trainable (initialized from $\mathcal{N}(0, \sigma^2)$)
- $B \in \mathbb{R}^{r \times d}$: trainable (initialized to **zero**)
- $r \ll d$: rank (typically 4, 8, 16)

**Parameter savings**: $2dr$ vs $d^2$ → for $d = 4096, r = 8$: 65K vs 16M.

**Initialization**: $B = 0$ ensures $\Delta W = 0$ at start (no disruption to pre-trained model).

**Inference**: Merge $W' = W_0 + BA$ → no additional latency.

**Plug-and-play**: Different $BA$ matrices for different tasks; swap at inference time.

### 5.2 AdaLoRA (Adaptive LoRA)

**Idea**: Not all layers need the same rank. Decompose $\Delta W$ using SVD-like structure:

$$\Delta W = P \Lambda Q$$

where:
- $P, Q$: orthogonal matrices (constrained during training)
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_r)$: singular values as **importance scores**

**Adaptive rank selection**:
1. Extract triplets $(P_i, \lambda_i, Q_i)$
2. Threshold on $|\lambda_i|$ — small singular values indicate unimportant directions
3. Different layers get different effective ranks

### 5.3 DoRA (Weight-Decomposed Low-Rank Adaptation)

**Observation**: In full fine-tuning, weight magnitude $\|W\|$ and direction $\frac{W}{\|W\|}$ are **uncorrelated**. In LoRA, they become correlated → suboptimal.

**Solution**: Decompose weight into magnitude and direction, train separately:

$$W' = M \cdot \frac{W_0 + BA}{\|W_0 + BA\|}$$

where:
- $M \in \mathbb{R}^d$: learnable **magnitude vector** (per-column norms)
- $\frac{W_0 + BA}{\|W_0 + BA\|}$: normalized direction (LoRA-updated)

**Benefits**: Better generalizability, reduced overfitting, closer to full fine-tuning behavior.

### 5.4 MonteLoRA (Bayesian LoRA)

**Idea**: Make LoRA Bayesian by sampling the $A$ matrix from a **mixture of Gaussians**.

$$A \sim \sum_k \pi_k \cdot \mathcal{N}(\mu_k, \Sigma_k)$$

- Variance $\Sigma$ sampled from **Wishart distribution**
- Mixing weights $\pi$ from **Dirichlet distribution**

| Property | LoRA | MonteLoRA |
|----------|------|-----------|
| A matrix | Deterministic | Stochastic (mixture of Gaussians) |
| Parameters | $2dr$ | Fewer (distribution params only) |
| Hyperparameter sensitivity | Higher | Lower (more stable) |
| Theoretical guarantee | — | Unbiased estimator of LoRA output |

### 5.5 QLoRA (Quantized LoRA)

Quantize the frozen base model to 4-bit, apply LoRA adaptors in full precision. Enables fine-tuning 65B models on a single 48GB GPU.

---

## 6. Summary Table

| Method | Category | Key Idea | Params Updated |
|--------|----------|----------|----------------|
| **Adapters** | Additive | Bottleneck modules between layers | ~3.6% |
| **Prefix Tuning** | Additive | Learnable K,V prefixes at every layer | < 1% |
| **Prompt Tuning** | Additive | Learnable input tokens | < 0.1% |
| **BitFit** | Selective (structured) | Fine-tune bias terms only | < 0.1% |
| **Fish Mask** | Selective (unstructured) | Select by Fisher information | Top-k% |
| **PuFFy** | Selective (unstructured) | Select by lowest magnitude | Bottom-k% |
| **ID3** | Selective (unstructured) | Incremental selection over iterations | Budget $K$ |
| **LoRA** | Reparameterization | Low-rank $\Delta W = BA$ | $2dr$ per matrix |
| **AdaLoRA** | Reparameterization | SVD-like with adaptive rank | Layer-dependent |
| **DoRA** | Reparameterization | Separate magnitude & direction | $M$ + LoRA |
| **MonteLoRA** | Reparameterization | Bayesian sampling of $A$ | Distribution params |
| **QLoRA** | Reparameterization | 4-bit base + FP16 LoRA | Same as LoRA |
