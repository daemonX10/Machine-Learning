# Lecture 31: Alternative Models — RWKV (Receptance Weighted Key Value)

---

## Motivation: Bridging RNNs and Transformers

| Model Family | Strength | Weakness |
|-------------|----------|----------|
| **RNNs/LSTMs** | $O(n)$ inference, constant memory | Sequential (no parallelism), vanishing gradients |
| **Transformers** | Parallelizable training, captures long-range deps | $O(n^2)$ attention complexity |
| **RWKV** | **Both**: Transformer-quality + RNN efficiency | Exponential decay limits long-range recall |

**RWKV** = Receptance Weighted Key Value (pronounced "Raku")  
Published: **EMNLP Findings 2023**

---

## Background: Evolution of Language Models

1. **N-gram LMs** → co-occurrence statistics  
2. **HMMs** → latent states, Markov assumption  
3. **LSTMs** → gated RNNs with forget/input/output gates  
4. **Transformers** → generalized skip-gram with Q/K/V, self-attention  
5. **RWKV** → blends RNN recurrence with Transformer-style parallelism

---

## RWKV Architecture

### Step 1: Token Shift (Bigram-Level Mixing)

Before computing K, V, R: blend current token $x_t$ with previous token $x_{t-1}$:

$$x_t^{mix} = \mu \cdot x_t + (1 - \mu) \cdot x_{t-1}$$

Where $\mu$ is a **learnable** mixing parameter (separate $\mu$ for K, V, R channels).

**Key Insight**: All $x_t$ values are available simultaneously → parallelizable (no RNN-style sequential dependency).

Then compute:

$$k_t = W_K \cdot x_t^{mix,K}, \quad v_t = W_V \cdot x_t^{mix,V}, \quad r_t = W_R \cdot x_t^{mix,R}$$

---

### Step 2: WKV Operation (Core Attention Mechanism)

#### Gates

| Gate | Formula | Analogy | Role |
|------|---------|---------|------|
| $\beta_t$ (Beta) | $e^{k_t}$ | Update gate | How strongly current token contributes |
| $\alpha^{t-i}$ (Alpha) | $e^{-w(t-i)}$ | Forget gate | Exponential decay for distance |

where $w > 0$ is a **learnable** decay parameter.

#### State Computation

$$a_t = \sum_{i=1}^{t} \alpha^{t-i} \cdot \beta_i \cdot v_i = \alpha \cdot a_{t-1} + \beta_t \cdot v_t$$

$$b_t = \sum_{i=1}^{t} \alpha^{t-i} \cdot \beta_i = \alpha \cdot b_{t-1} + \beta_t$$

$$s_t = \frac{a_t}{b_t}$$

#### Attention Weight (Omega)

$$\omega_{t,i} = \frac{\beta_i \cdot \alpha^{t-i}}{\sum_{j=1}^{t} \beta_j \cdot \alpha^{t-j}}$$

**Comparison with Transformer attention**:

| | Transformer | RWKV |
|---|------------|------|
| Attention weight | $\omega_{t,i} = \frac{e^{q_t \cdot k_i}}{\sum_j e^{q_t \cdot k_j}}$ | $\omega_{t,i} = \frac{e^{k_i} \cdot e^{-w(t-i)}}{\sum_j e^{k_j} \cdot e^{-w(t-j)}}$ |
| Interaction | Q-K **handshake** (dot product) | **No handshake** — just $\beta \cdot \alpha$ |
| Distance sensitivity | None (position via embeddings) | **Explicit exponential decay** |
| Complexity | $O(n^2)$ | $O(n)$ |

---

### Step 3: Receptance (R) — Output Gate

$$y_t = \sigma(r_t) \cdot \text{LayerNorm}(s_t)$$

- $\sigma(r_t)$: Sigmoid gate controlling **what percentage of $s_t$ to accept** in the output
- Analogous to LSTM's **output gate**
- $r_t = W_R \cdot x_t^{mix,R}$ — derived from the mixed token representation

#### Summary of Gate Analogies

| RWKV Component | LSTM Analog | Function |
|---------------|-------------|----------|
| $\alpha = e^{-w}$ | Forget gate | Controls exponential decay |
| $\beta = e^{k_t}$ | Input/Update gate | Controls new input contribution |
| $\sigma(r_t)$ | Output gate | Controls what to emit |

---

### Step 4: Channel Mixing (Analog of FFN)

After time mixing produces $y_t$, add residual:

$$y'_t = y_t + x_t \quad \text{(residual connection)}$$

Then perform channel mixing (analogous to Transformer's FFN between layers):

1. **Bigram mix again**: Blend $y'_t$ and $y'_{t-1}$ along two channels (with separate mixing parameters)
2. **Gate & Transform**:
   - Channel F: Linear transform $W_F$ (expansion)  
   - Channel G: Sigmoid gate $\sigma(\cdot)$
3. **Activation**: Apply **ReL-Square** activation: $\max(u_t, 0)^2$
4. **Apply gate**: $z_t = \sigma(x_t^G) \cdot \text{ReLSquare}(u_t)$
5. **Residual**: $y''_t = z_t + y'_t$

$y''_t$ becomes $x_t$ for the **next layer**.

---

### Full RWKV Block

```
Input x_t
    │
    ▼
┌─────────────────┐
│  Time Mixing     │  Token shift → K, V, R computation
│  (WKV Operation) │  Exponential decay attention → s_t
│  Receptance gate │  σ(r_t) · LayerNorm(s_t) → y_t
└────────┬────────┘
         │ + x_t (residual)
         ▼
┌─────────────────┐
│  Channel Mixing  │  Bigram blend → Gate + ReLSquare → z_t
│  (FFN analog)    │
└────────┬────────┘
         │ + y'_t (residual)
         ▼
    Output → next layer
```

Stack multiple RWKV blocks → LM head on top for auto-regressive generation.

---

### Key Properties

| Property | Detail |
|----------|--------|
| **No multi-head attention** | Unlike Transformers, RWKV uses stacked layers but no heads |
| **Infinite context (theoretical)** | No fixed context window; decay naturally handles relevance |
| **Parallelizable** | All tokens available simultaneously; no sequential dependency |
| **Markovian drift hypothesis** | Nearby tokens should naturally contribute more than distant ones |

---

## The Markovian Drift Hypothesis

RWKV's core scientific claim:

> **Language exhibits Markovian drift** — the influence of a token should decay exponentially with distance, unless reinforced by new input.

- Transformer: Learns arbitrary attention patterns (distant tokens can have high weight)
- RWKV: **Forces** nearby tokens to have higher base weight via $\alpha^{t-i}$, modulated by $\beta_i$

**Example**: In "John loves Mary":
- At "Mary": "loves" (nearby) naturally has higher weight
- "John" (farther) gets decayed unless $\beta_{John}$ is very high
- **Balance**: $\beta$ provides content-based importance, $\alpha$ enforces position-based decay

---

## Benchmark Results

### Standard NLP Benchmarks

| Benchmark | Task | RWKV vs Baselines |
|-----------|------|-------------------|
| **ARC** | Multiple-choice science reasoning | Comparable/better than Bloom, Pythia, OPT |
| **HellaSwag** | Common sense continuation | Comparable to best |
| **LAMBADA** | Predict final word (long context) | Comparable with more compute |
| **OpenBookQA** | Science reasoning | Comparable |

**Key claim**: RWKV achieves comparable accuracy with **less compute** (reaches target accuracy earlier in training FLOPs).

### Limitations
- **Long Range Arena**: Comparable on text/retrieval but **weak on Pathfinder** (long-context tasks)
- S4 (state space model) excels on LRA where RWKV struggles
- Fundamental bottleneck: **information compression** — all past tokens compressed into single state with exponential decay

### Ongoing Development
- RWKV versions continue to evolve (RWKV-7+)
- Active use in vision/image processing domains
- Goal: Challenge Transformer-based models with significantly less real compute
