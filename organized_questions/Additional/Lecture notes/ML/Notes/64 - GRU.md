# GRU — Gated Recurrent Unit

## Overview

- **Full Form:** Gated Recurrent Unit
- **What:** An RNN architecture (like Simple RNN and LSTM) for processing **sequential data**
- **Origin:** Introduced in **2014** (LSTM was 1997)
- **Why it exists:** LSTM is powerful but has a **complex architecture** with many parameters → slow training. GRU offers a **simpler alternative** with **comparable performance**.

---

## Why GRU When LSTM Exists?

| Issue with LSTM | How GRU Helps |
|----------------|---------------|
| 3 gates → many parameters | Only 2 gates → fewer parameters |
| 2 separate states (cell + hidden) | Single hidden state only |
| Slow training on large datasets | Faster training due to fewer computations |
| Complex architecture | Simpler architecture |

> **Key insight:** Despite being simpler, GRU's performance is **comparable** to LSTM.  
> On some datasets/tasks, GRU **outperforms** LSTM; on others, LSTM is better.  
> The choice often comes down to **empirical testing**.

---

## The Big Idea Behind GRU

### Compared to LSTM:

| Feature | LSTM | GRU |
|---------|------|-----|
| **Memory types** | Cell State (long-term) + Hidden State (short-term) | **Single** Hidden State handles both |
| **Number of gates** | 3 (Forget, Input, Output) | **2** (Reset, Update) |
| **Separate long-term path?** | Yes (cell state line) | No — single state carries everything |

### GRU's approach:
- Maintain **one hidden state** that carries both long-term and short-term context
- Use **two gates** (Reset and Update) to manipulate this single state
- Achieve the same goal — retaining long-range dependencies — with less overhead

---

## Setup: Vectors, Layers, and Operations

### All Key Quantities are Vectors

| Symbol | Name | Dimension |
|--------|------|-----------|
| $h_{t-1}$ | Previous hidden state | $n$ |
| $h_t$ | Current hidden state | $n$ |
| $x_t$ | Current input | $d$ (can differ from $n$) |
| $r_t$ | Reset gate | $n$ |
| $z_t$ | Update gate | $n$ |
| $\tilde{h}_t$ | Candidate hidden state | $n$ |

> **Rule:** All vectors except $x_t$ have the **same dimension** $n$ (= number of hidden units).

### Yellow Boxes = Neural Network Layers

Each yellow box in the diagram is a **fully connected layer** with:
- $n$ nodes (same across all layers in the cell)
- Activation: either **sigmoid** ($\sigma$) or **tanh**
- Its own weight matrix $W$ and bias $b$

### Pink Circles = Pointwise (Element-wise) Operations

| Symbol | Operation |
|--------|-----------|
| $\times$ | Pointwise (element-wise) multiplication |
| $+$ | Pointwise addition |
| $1 - $ | Element-wise subtraction from 1 |

**Example:**

$$[a, b, c] \odot [d, e, f] = [ad, be, cf]$$

$$[a, b, c] + [d, e, f] = [a+d, b+e, c+f]$$

---

## What the Hidden State Represents: An Intuition

The hidden state $h_t$ is the **memory** of the system — it encodes the context of everything processed so far.

### Example with the Story Analogy

Assume a 4-dimensional hidden state tracking: **Power**, **Conflict**, **Tragedy**, **Revenge**

| Story Event | $h_t$ Update |
|-------------|-------------|
| King Vikram introduced (powerful) | $[0.5, 0, 0, 0]$ — power context only |
| Enemy Kali introduced | $[1.0, 0.5, 0, 0]$ — power + conflict |
| War → Vikram killed | $[0.6, 0.7, 0.8, 0]$ — tragedy spikes |
| Vikram Jr. grows up, very strong | $[0.8, 0.5, 0.5, 0]$ — power restored |
| Vikram Jr. attacks, gets killed | $[0.8, 0.8, 0.9, 0.5]$ — tragedy + revenge |
| Grandson fights and kills Kali | $[0.7, 0.8, 0.3, 1.0]$ — revenge dominates |

At the end, the final $h_t = [0.7, 0.8, 0.3, 1.0]$ encodes: lots of power, lots of conflict, manageable tragedy, and full revenge → **happy ending**.

> The transitions $h_{t-1} \rightarrow h_t$ are exactly what the **Reset** and **Update** gates manage.

---

## The Two-Step Process: $h_{t-1} \rightarrow h_t$

### Step 1: Create Candidate Hidden State ($\tilde{h}_t$)

From $h_{t-1}$ (past memory) + $x_t$ (current input), propose a **candidate** new memory.

- This candidate is **heavily influenced by the current input**
- It represents what the memory **would** look like if the current input were very important

### Step 2: Balance Past and Candidate

- $h_t$ is computed as a **weighted combination** of $h_{t-1}$ and $\tilde{h}_t$
- The **Update Gate** ($z_t$) controls the balance:
  - High $z_t$ → favor candidate (current input is important)
  - Low $z_t$ → favor past memory (current input is not so important)

This prevents blindly trusting the current input and losing valuable historical context.

---

## Architecture: 4-Step Computation

### Step 1: Calculate Reset Gate ($r_t$)

$$\boxed{r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)}$$

| Component | Detail |
|-----------|--------|
| **Input** | Concatenation of $h_{t-1}$ (dim $n$) and $x_t$ (dim $d$) → dim $(n+d)$ |
| **Weights** | $W_r$: shape $n \times (n+d)$ |
| **Bias** | $b_r$: shape $n$ |
| **Activation** | Sigmoid → output ∈ $[0, 1]$ |
| **Output** | $r_t$: vector of dim $n$ |

**Purpose:** Decides which dimensions of $h_{t-1}$ to **reset** (suppress) before creating the candidate.

**How it works as a gate:**

| $r_t$ value | Meaning |
|-------------|---------|
| $0.0$ | Fully reset this dimension (ignore past info) |
| $0.5$ | Keep 50% of past info |
| $1.0$ | Fully retain this dimension (keep all past info) |

**Example:** After reading about Vikram Jr.'s introduction:
$$r_t = [0.8, 0.2, 0.1, 0.9]$$

- Power: retain 80% (still relevant — new powerful king)
- Conflict: retain only 20% (no current fighting)
- Tragedy: retain only 10% (old news)
- Revenge: retain 90% (revenge storyline is building)

---

### Step 2: Calculate Candidate Hidden State ($\tilde{h}_t$)

First, create the **modulated (reset) past memory**:

$$h_{t-1}' = r_t \odot h_{t-1}$$

**Example:** $h_{t-1} = [0.6, 0.6, 0.7, 0.1]$, $r_t = [0.8, 0.2, 0.1, 0.9]$

$$h_{t-1}' = [0.48, 0.12, 0.07, 0.09]$$

Then compute the candidate:

$$\boxed{\tilde{h}_t = \tanh(W_{\tilde{h}} \cdot [h_{t-1}', x_t] + b_{\tilde{h}})}$$

| Component | Detail |
|-----------|--------|
| **Input** | Concatenation of **reset** $h_{t-1}'$ and $x_t$ |
| **Weights** | $W_{\tilde{h}}$: shape $n \times (n+d)$ |
| **Bias** | $b_{\tilde{h}}$: shape $n$ |
| **Activation** | $\tanh$ → output ∈ $[-1, 1]$ |
| **Output** | $\tilde{h}_t$: candidate hidden state (dim $n$) |

**Note:** The reset gate controls **how much past memory influences** the candidate. This is different from LSTM where forget and input gates operate directly on the cell state.

---

### Step 3: Calculate Update Gate ($z_t$)

$$\boxed{z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)}$$

| Component | Detail |
|-----------|--------|
| **Input** | Concatenation of **original** $h_{t-1}$ and $x_t$ |
| **Weights** | $W_z$: shape $n \times (n+d)$ |
| **Bias** | $b_z$: shape $n$ |
| **Activation** | Sigmoid → output ∈ $[0, 1]$ |
| **Output** | $z_t$: vector of dim $n$ |

**Purpose:** Controls the **balance** between past memory ($h_{t-1}$) and candidate memory ($\tilde{h}_t$).

| $z_t$ value | Effect |
|-------------|--------|
| High (→ 1) | Trust the **candidate** more (current input is very important) |
| Low (→ 0) | Trust the **past memory** more (current input is less important) |

---

### Step 4: Compute Final Hidden State ($h_t$)

$$\boxed{h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t}$$

| Term | Meaning |
|------|---------|
| $(1 - z_t) \odot h_{t-1}$ | Portion of **past memory** retained |
| $z_t \odot \tilde{h}_t$ | Portion of **candidate (new) memory** incorporated |
| $h_t$ | Final hidden state for current time step |

> This is a **convex combination** — the two coefficients $(1 - z_t)$ and $z_t$ always sum to 1, ensuring a proper interpolation.

**Example:** $z_t = [0.3, 0.7, 0.8, 0.2]$

| Dimension | $(1-z_t)$ × past | $z_t$ × candidate | Effect |
|-----------|-------------------|--------------------|--------|
| Power | 0.7 × past | 0.3 × candidate | Mostly keep past |
| Conflict | 0.3 × past | 0.7 × candidate | Mostly new |
| Tragedy | 0.2 × past | 0.8 × candidate | Heavily new |
| Revenge | 0.8 × past | 0.2 × candidate | Mostly keep past |

---

## Complete GRU Equations Summary

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \quad \text{(Reset Gate)}$$

$$\tilde{h}_t = \tanh(W_{\tilde{h}} \cdot [r_t \odot h_{t-1}, x_t] + b_{\tilde{h}}) \quad \text{(Candidate Hidden State)}$$

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \quad \text{(Update Gate)}$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(Final Hidden State)}$$

---

## Parameter Count

Each gate/layer has a weight matrix of shape $n \times (n + d)$ and bias of shape $n$:

| Component | Weight Matrix | Bias | Parameters |
|-----------|---------------|------|------------|
| Reset Gate | $W_r$: $n \times (n+d)$ | $b_r$: $n$ | $n(n+d) + n$ |
| Candidate | $W_{\tilde{h}}$: $n \times (n+d)$ | $b_{\tilde{h}}$: $n$ | $n(n+d) + n$ |
| Update Gate | $W_z$: $n \times (n+d)$ | $b_z$: $n$ | $n(n+d) + n$ |

**Total parameters:**

$$\boxed{3 \times [n(n + d) + n] = 3n(n + d + 1)}$$

Compare with LSTM: $4n(n + d + 1)$ — GRU has **25% fewer parameters**.

---

## LSTM vs GRU: Key Differences

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Number of gates** | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Memory units** | Cell State + Hidden State | Hidden State only |
| **Parameter count** | $4n(n + d + 1)$ | $3n(n + d + 1)$ |
| **Computational cost** | Higher | Lower (faster to train) |
| **Architecture complexity** | More complex | Simpler |
| **Performance (complex tasks)** | Often slightly better | Comparable |
| **Performance (small data)** | Good | Can outperform LSTM |
| **Training speed** | Slower | Faster |
| **When to use** | Complex tasks, large data, need max accuracy | Limited resources, small data, first baseline |

### Practical Recommendation

1. **Start with GRU** — faster training, simpler architecture
2. **Try LSTM** if GRU results need improvement
3. **Empirical testing** is the only reliable way to choose — performance depends on the specific dataset and task
4. On some tasks GRU outperforms LSTM, on others the reverse is true

---

## Flow Diagram

```
h_{t-1}, x_t
     │
     ├──→ [σ] ──→ r_t (Reset Gate)
     │              │
     │         r_t ⊙ h_{t-1} = h'_{t-1} (modulated memory)
     │              │
     │         [h'_{t-1}, x_t] ──→ [tanh] ──→ h̃_t (Candidate)
     │
     ├──→ [σ] ──→ z_t (Update Gate)
     │              │
     │     ┌────────┴────────┐
     │     │                 │
     │  (1-z_t) ⊙ h_{t-1}  z_t ⊙ h̃_t
     │     │                 │
     │     └────────┬────────┘
     │              │
     │           [  +  ]
     │              │
     └──────────→  h_t (Final Hidden State)
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What GRU solves** | Same as LSTM (vanishing gradients, long-range dependencies) but with less complexity |
| **Core innovation** | Single hidden state + two gates achieves what LSTM does with two states + three gates |
| **Reset Gate ($r_t$)** | Decides which parts of past memory to reset before forming candidate |
| **Update Gate ($z_t$)** | Balances between past memory and candidate — convex combination |
| **Candidate ($\tilde{h}_t$)** | New memory proposal based on reset past + current input |
| **Final state** | $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ — interpolation of old and new |
| **Fewer params** | 25% fewer than LSTM → faster training |
| **Performance** | Comparable to LSTM; sometimes better, sometimes worse — must test empirically |
