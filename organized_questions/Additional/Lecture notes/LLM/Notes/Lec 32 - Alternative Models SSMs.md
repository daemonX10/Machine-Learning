# Lecture 32: Alternative Models — State Space Models (SSMs)

---

## From RWKV to SSMs

### RWKV Limitations (Recap)

| Limitation | Detail |
|-----------|--------|
| **Information bottleneck** | All history compressed into a single state vector with exponential decay |
| **Long-range recall** | Older tokens forgotten unless explicitly reinforced by new input |
| **State compression** | $s_t = \frac{\sum_i \alpha^{t-i} \beta_i v_i}{\sum_i \alpha^{t-i} \beta_i}$ — weighted average with exponential decay |
| **LRA weakness** | Fails on Pathfinder task (long-context reasoning) where S4 excels |

### RWKV Benchmark Summary

| Benchmark | RWKV Performance |
|-----------|-----------------|
| ARC, HellaSwag, LAMBADA, OpenBookQA | Comparable to Bloom/Pythia/OPT |
| Long Range Arena (Text, Retrieval) | Comparable |
| Long Range Arena (Pathfinder) | **Weak** — S4 significantly better |
| Instruction-tuned vs GPT-4 | Approaches on RTE but inconsistent |

---

## State Space Models (SSMs): The Continuous View

### Core Philosophy

> **SSMs treat language as a fluid, continuous process** — like speech or signal processing — rather than discrete token jumps.

| Model | View of Language |
|-------|-----------------|
| Transformer | Discrete: each token independently attends to others |
| RWKV | Discrete: bigram mixing with exponential decay |
| **SSM** | **Continuous**: meaning flows and evolves smoothly between tokens |

**Analogy**: 
- RWKV = Flipping through a photo album (discrete)
- SSM = Watching a movie (continuous)
- Transformer = Assembling a snowman (independent pieces)
- SSM = Pottery sculpting (continuous reshaping of clay)

---

### The SSM Equation

$$\frac{dx}{dt} = Ax(t) + Bu(t)$$

| Component | Meaning |
|-----------|---------|
| $x(t) \in \mathbb{R}^n$ | Hidden state (internal "meaning field") |
| $u(t)$ | Input signal (current token influence) |
| $A \in \mathbb{R}^{n \times n}$ | State transition matrix — how meaning **drifts autonomously** |
| $B$ | Input projection — how new tokens **nudge** the meaning field |

### Closed-Form Solution

$$x(t + \Delta) = e^{A\Delta} \cdot x(t) + \int_0^{\Delta} e^{A\tau} \cdot B \cdot u(t + \Delta - \tau) \, d\tau$$

**Properties**:
- **Infinitely differentiable** (no abrupt jumps)
- **Bounded rate of change**: $\|\dot{x}(t)\| \leq \|A\| \cdot \|x(t)\| + \|B\| \cdot \|u(t)\|$

### Derivation (Scalar Case)

1. Start with: $\dot{x} = ax + bu$
2. Multiply both sides by $e^{-at}$ (integrating factor)
3. Recognize: $\frac{d}{dt}[e^{-at} x(t)] = b \cdot e^{-at} u(t)$
4. Integrate from $t$ to $t + \Delta$
5. Solve for $x(t + \Delta)$:

$$x(t + \Delta) = e^{a\Delta} x(t) + \int_0^{\Delta} e^{a\tau} \cdot b \cdot u(t + \Delta - \tau) \, d\tau$$

For the **matrix case**, use Taylor expansion: $e^{A\Delta} = I + A\Delta + \frac{(A\Delta)^2}{2!} + \cdots$

---

## Understanding $\Delta$ (Delta) — Semantic Time Step

$\Delta$ controls **how long the internal state evolves before the next token arrives**.

| $\Delta$ Value | Interpretation | Analogy |
|---------------|----------------|---------|
| $\Delta = 1$ | Normal flow, steady semantic pace | Standard text |
| $\Delta > 1$ | Long pause, topic shift | Paragraph break, sentence boundary |
| $\Delta < 1$ | Rapid input, tokens arriving fast | Dense technical text, rapid speech |
| Variable $\Delta$ | Irregular, incoherent flow | Disjointed conversation |

**Example**: "John loves Mary who lives in New York City"
- Between "Mary" and "who": small $\Delta$ → smooth transition from object to relative clause subject
- After a period (full stop): large $\Delta$ → topic may shift, meaning drifts further

**In practice**: For text, $\Delta$ is typically kept at 1 (uniform pace). Variable $\Delta$ is more relevant for speech/audio signals.

### Special Cases

| Condition | Result |
|-----------|--------|
| **No input** ($u = 0$) | $x(t+\Delta) = e^{A\Delta} x(t)$ — pure exponential drift/decay |
| **No dynamics** ($A = 0$) | $x(t+\Delta) = x(t) + B \cdot u$ — simple accumulation |
| **Infinitesimal step** ($\Delta \to 0$) | Reduces to $\dot{x} = Ax + Bu$ (the original ODE) |

---

## Understanding $A$ — The State Transition Matrix

$A$ governs **how internal meaning drifts when no input arrives**.

### Eigendecomposition Interpretation

If $A$ is diagonalizable: $A = Q \Lambda Q^{-1}$ where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$

| Component | Meaning |
|-----------|---------|
| Eigenvector $q_i$ | A **semantic direction** (aspect of meaning) |
| Eigenvalue $\lambda_i$ | **Intensity** of that aspect's evolution |

Each eigenvalue $\lambda_i = a_i + b_i j$ (can be complex):

| Part | Effect | Linguistic Interpretation |
|------|--------|--------------------------|
| $a_i < 0$ (real, negative) | Exponential **decay** | Memory fades (e.g., "Mary" eventually forgotten) |
| $a_i = 0$ | **Persists** | Memory maintained (e.g., "John" stays in focus) |
| $a_i > 0$ | Exponential **growth** | Amplification (typically avoided for stability) |
| $b_i \neq 0$ (imaginary) | **Oscillation** | Aspect recurs (e.g., syntactic patterns like "who", "which") |

### Linguistic Example

| Eigenvector | Semantic Aspect | Eigenvalue Behavior |
|-------------|----------------|---------------------|
| $q_1$: Entity mode | Tracks proper nouns (Mary, City) | Small negative $a_1$ → slow decay |
| $q_2$: Predicate mode | Tracks verbs (loves, lives) | Medium negative $a_2$ → moderate decay |
| $q_3$: Syntactic rhythm | Clausal cues (who, which, while) | $b_3 \neq 0$ → oscillatory (recurs) |
| $q_4$: Function words | Determiners, prepositions (in, the) | Large negative $a_4$ → fast decay |

**Orchestra analogy**: Each eigenvector is an instrument; each eigenvalue controls whether that instrument crescendos, sustains, or fades. Together, they create the full semantic "music" of the sentence.

---

## Why Initialization of $A$ Matters

| Approach | Problem |
|----------|---------|
| Random initialization | No guarantee of diagonalizability → lose interpretability + continuity |
| **Structured initialization** | Guarantees: (1) easy exponentiation, (2) independent semantic vectors, (3) stable training |

### Initialization Methods

| Method | Description |
|--------|-------------|
| Distinct eigenvalues | Compute $\lambda_i$ directly, construct $A$ |
| Hermitian/Symmetric | $A = A^*$ guarantees real eigenvalues, orthogonal eigenvectors |
| Diagonal form | $A = \text{diag}(\lambda_1, \ldots, \lambda_n)$ — simplest (used by S4) |
| Spectral form | Full eigendecomposition $Q \Lambda Q^{-1}$ |
| **HiPPO** | Structured polynomial basis (used by S4 for long-range memory) |

### RWKV as a Special Case of SSM

$$A_{RWKV} = -w \cdot I$$

- $A$ is just a scalar decay times identity matrix
- No drift, no oscillation, no aspect-specific behavior
- The $\alpha = e^{A\Delta} = e^{-w}$ from RWKV is this trivial case

**Key insight**: Diagonalizability ensures $A$ acts like a set of **independent semantic resonators**.

---

## SSM Variants

| Model | $A$ Initialization | Key Feature |
|-------|-------------------|-------------|
| **S4** | Diagonal matrix of eigenvalues + HiPPO | Excels at Long Range Arena |
| **Mamba** | Structured diagonal init | Input-dependent $\Delta$ and $B$ (selective mechanism) |
| **RWKV** | $-wI$ (trivial) | Pure exponential decay, no aspect decomposition |

### Visual Comparison: Discrete vs Continuous State Evolution

| | RWKV (Discrete) | SSM (Continuous) |
|---|-----------------|------------------|
| State updates | Step function (jumps at each token) | Smooth curve (continuous drift between tokens) |
| "John" influence | Decays in discrete steps | Decays continuously, can re-intensify |
| Between tokens | Waits (does nothing) | **Drifts** (anticipates next input) |

---

## Key Takeaways

1. **SSMs model language as continuous flow**, not discrete jumps — meaning evolves smoothly via ODEs
2. **Delta** ($\Delta$) is the semantic time step controlling drift duration between tokens
3. **Matrix $A$** governs autonomous drift; its eigenstructure determines which semantic aspects persist, decay, or oscillate
4. **Initialization of $A$ is critical** — unlike Transformers, random init doesn't work; structured init (diagonal, HiPPO) ensures stability and interpretability
5. **RWKV is a trivial special case** of SSMs where $A = -wI$
6. **S4 and Mamba** are practical SSM realizations that achieve state-of-the-art on long-range tasks
7. SSMs are **very competitive** with Transformers on standard LLM benchmarks while being more efficient
