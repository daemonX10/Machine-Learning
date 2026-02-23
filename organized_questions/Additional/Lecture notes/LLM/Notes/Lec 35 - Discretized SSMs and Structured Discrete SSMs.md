# Lecture 35: Discretized SSMs & Structured Discrete SSMs (S4, Mamba)

## 1. Continuous State Space Models (SSMs)

### State-Space ODE Formulation

A continuous-time SSM is defined by:

$$\dot{x}(t) = Ax(t) + Bu(t)$$
$$y(t) = Cx(t) + Du(t)$$

| Symbol | Dimension | Role |
|---|---|---|
| $x(t)$ | $\mathbb{R}^N$ | Hidden state (memory) |
| $u(t)$ | $\mathbb{R}^1$ | Input signal |
| $y(t)$ | $\mathbb{R}^1$ | Output |
| $A$ | $\mathbb{R}^{N \times N}$ | State transition (dynamics) |
| $B$ | $\mathbb{R}^{N \times 1}$ | Input projection |
| $C$ | $\mathbb{R}^{1 \times N}$ | Output projection |
| $D$ | $\mathbb{R}^{1 \times 1}$ | Skip connection |

### Eigenvalue Interpretation

For eigenvalues $\lambda = \sigma + j\omega$ of $A$:

| Component | Meaning |
|---|---|
| $\sigma < 0$ (real part) | Decay → stable, forgetting |
| $\sigma > 0$ (real part) | Growth → unstable, exploding |
| $\omega$ (imaginary part) | Oscillation frequency |

**Stability requirement:** all eigenvalues must have $\text{Re}(\lambda) < 0$.

---

## 2. Discretization (Zero-Order Hold)

To process discrete token sequences, discretize the continuous SSM with step size $\Delta$:

$$\bar{A} = e^{A\Delta}$$

$$\bar{B} = A^{-1}(e^{A\Delta} - I) \cdot B$$

The discrete recurrence becomes:

$$x_{k} = \bar{A} x_{k-1} + \bar{B} u_k$$
$$y_k = C x_k + D u_k$$

- $\Delta$ controls the **temporal resolution** — how finely the continuous signal is sampled
- Larger $\Delta$ → coarser sampling, faster forgetting
- Smaller $\Delta$ → finer sampling, longer memory

---

## 3. From Sequential to Convolutional View

### Unrolling the Recurrence

$$x_k = \bar{A}^k x_0 + \sum_{i=0}^{k-1} \bar{A}^{k-1-i} \bar{B} u_i$$

The output can be written as a **convolution**:

$$y_k = \sum_{i=0}^{k} K_i \cdot u_{k-i}$$

where the **kernel** is:

$$K_i = C \bar{A}^{i} \bar{B}$$

### Analogy with Self-Attention

| Property | Self-Attention | SSM Kernel |
|---|---|---|
| Weight pattern | Data-dependent (QK similarity) | Fixed (determined by $A, B, C$) |
| Computation | $O(T^2)$ | $O(T)$ sequential, $O(T \log T)$ via FFT |
| Memory of past | Explicit (attend to all past tokens) | Implicit (compressed in state $x$) |

---

## 4. FFT Optimization

Since the output is a convolution $y = K * u$, use FFT:

$$y = \text{IFFT}(\text{FFT}(K) \odot \text{FFT}(u))$$

**Complexity:** $O(T^2) \rightarrow O(T \log T)$

### Frequency Domain Interpretation

| Frequency | Linguistic Analogy |
|---|---|
| Low (slow rhythms) | Topics, discourse structure |
| Medium | Phrases, noun/verb phrases |
| High (fast rhythms) | Syntax, connectors, function words |

---

## 5. The Stability Problem

When computing $\bar{A} = e^{A\Delta}$, the eigenvalues of $\bar{A}$ must satisfy $|\bar{\lambda}| < 1$ for stability. An arbitrary $A$ matrix provides **no guarantee** of this.

**Need:** A structured $A$ matrix that ensures stability by construction → **HiPPO**.

---

## 6. HiPPO (High-order Polynomial Projection Operator)

### Core Idea

Instead of remembering the entire past sequence, maintain a **compressed summary** using orthogonal polynomial projections.

$$c_k(t) \approx \int u(\tau) P_k(\tau) \, d\tau$$

where $P_k$ are **Legendre polynomials** (orthogonal on $[-1, 1]$).

### Memory as Orthogonal Components

| Component $c_k$ | Captures |
|---|---|
| $c_0$ | Average / broad topic |
| $c_1$ | Recent trend |
| $c_2, c_3, \ldots$ | Progressively finer short-term details |

- Each $P_k$ is orthogonal → memory channels **do not interfere**
- Like multiple clocks: slow clock = topic, medium = phrases, fast = connectors

### HiPPO Matrix

$$A_{\text{HiPPO}}[n,k] = \begin{cases} -\sqrt{(2n+1)(2k+1)} & \text{if } n > k \\ -(n+1) & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}$$

**Key property:** The eigenvalues of $A_{\text{HiPPO}}$ are **guaranteed stable** (real parts always negative).

---

## 7. S4: Structured State Spaces for Sequence Modeling (ICLR 2022)

### From HiPPO to S4

$$A_{S4} = A_{\text{HiPPO (normal form)}} + \text{low-rank correction } (PQ^*)$$

1. Decompose: $A_H = V \Lambda V^{-1}$ where $\Lambda$ is diagonal
2. Add low-rank: work with $\Lambda + \tilde{P}\tilde{Q}^*$ in the diagonal basis
3. **Woodbury identity** for efficient inversion:

$$(D - UV^T)^{-1} = D^{-1} + D^{-1}U(I - V^T D^{-1}U)^{-1}V^T D^{-1}$$

Since $D$ is diagonal → $D^{-1}$ is trivial → entire computation is efficient.

### S4 Pipeline

```
Input u → [FFT] → [Spectral filtering with kernel K(Λ + low-rank)] → [IFFT]
       → [Normalization / Gate] → [Residual + Head] → Output
```

- Can stack multiple S4 layers
- Decision head: LM head (next-token prediction), classification, etc.

### Performance

- **Long Range Arena:** S4 significantly outperforms BigBird, Reformer, Linear Transformers
- **Speed:** comparable to linear transformers, much better than quadratic attention
- **Memory:** constant-size state (A is time-invariant)

### Limitation of S4

The kernel $K$ is **data-independent** (A, B are fixed):

$$K_i = C\bar{A}^i \bar{B}$$

The dynamics do not adapt to the content of incoming tokens.

---

## 8. Mamba: Selective State Space Model (2024)

### Core Innovation — Selectivity

Make $A$, $B$, $C$ **input-dependent**:

$$B_t = W_B \cdot u_t, \quad C_t = W_C \cdot u_t, \quad \lambda_t = \text{softplus}(W_\lambda \cdot u_t)$$

$$A(u_t) = -\text{diag}(\lambda_t)$$

The negative sign is kept **outside** the diagonal matrix to ensure $\lambda_t > 0$ via softplus, guaranteeing stability ($e^{-\lambda_t \Delta_t} \in (0, 1)$).

### Gated Recurrence

$$x_{t+1} = \alpha_t \cdot x_t + \beta_t \cdot u_t$$

| Gate | Formula | Role |
|---|---|---|
| Forget gate $\alpha_t$ | $e^{-\lambda_t \Delta_t}$ | How much past state to retain |
| Update gate $\beta_t$ | $\frac{1 - \alpha_t}{\lambda_t} \cdot B_t$ | How much new input to incorporate |

- Larger $\lambda_t$ or $\Delta_t$ → faster forgetting on that channel for that token
- Resembles **GRU / LSTM** gating but derived from continuous SSM discretization

### Selective Scan — Why Linear Time

At each step, only look at the **previous** state:

$$x_t = \alpha_t \cdot x_{t-1} + \beta_t \cdot u_t$$

No need to attend to all previous tokens → $O(T)$ per sequence.

The kernel becomes **token-conditioned**:

$$K_{\text{Mamba}}(t, k) = C_t \left(\prod_{j=k+1}^{t} \alpha_j\right) \beta_k \cdot u_k$$

vs. S4's fixed kernel $K_i = C\bar{A}^i\bar{B}$.

### Linguistic Intuition

| Token | Expected Behavior |
|---|---|
| "not" | $\lambda_t$ spikes → suppress prior affirmative intent |
| "Paris" | $\lambda_t$ small for entity channel → persist location topic |
| "yesterday" | moderate $\beta_t$ → brief temporal context, then fast decay |
| "he" | state highlights channel binding back to "John" |

### Stability Guarantees

1. **Softplus** ensures $\lambda_t > 0$ → diagonal always negative → eigenvalues in stable region
2. **Step positivity:** $\Delta_t > 0$ always
3. **Bounded forget gate:** $\alpha_t = e^{-\lambda_t \Delta_t} \in (0, 1)$ → avoids explosion
4. **Per-step contraction** → healthy gradient flow

### Performance Results

| Metric | Result |
|---|---|
| Mamba-370M vs Pythia-410M | Mamba wins on zero-shot benchmarks |
| Mamba-1.4B / 2.8B | Matches transformers ~2× the size |
| Mamba-6.9B | Comparable to 13B GPT-NeoX |
| Training scaling | Linear with sequence length (vs. quadratic for attention) |
| Context length | Trains stably beyond 256K tokens (flash attention OOMs) |
| Inference throughput | 5× higher than transformers; scales linearly with batch size |

### Summary: S4 → Mamba

```
Continuous SSM → Discretize (ZOH) → Convolve → FFT optimize → S4 (stable, fast, but static)
                                                                     ↓
                                               Make A, B, C input-dependent → Mamba (selective)
```

---

## 9. When to Use What

| Use Case | Recommended |
|---|---|
| Short sequences, complex reasoning | Transformers (attention is fine) |
| Very long sequences (>100K tokens) | SSMs (Mamba) — linear scaling |
| Complex behavioral trajectories | SSMs — capture long-horizon drift |
| Standard NLP tasks | Transformers remain competitive |

**Key takeaway:** No one-size-fits-all. SSMs excel at long-range, streaming, or memory-constrained scenarios. Transformers dominate where complex token-to-token interactions matter and sequence length is manageable.
