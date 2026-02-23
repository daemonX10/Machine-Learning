# LSTM Part 2 — Architecture (The How)

## Overview

- This note covers the **internal architecture** of an LSTM cell — all three gates, their math, and how information flows
- Prerequisite: LSTM Part 1 (The What) — understanding of cell state, hidden state, and gate purposes

---

## Recap from Part 1

| Concept | Detail |
|---------|--------|
| **Long-term memory** | Cell State ($C_t$) — flows along the top line |
| **Short-term memory** | Hidden State ($h_t$) — flows along the bottom line |
| **Why complex?** | Need communication between long-term and short-term memory |
| **Processing** has 2 jobs | (1) Update cell state, (2) Calculate hidden state |

### Input / Output of an LSTM Cell

**Inputs** (at time step $t$):
- $C_{t-1}$ — previous cell state (long-term memory)
- $h_{t-1}$ — previous hidden state (short-term memory)
- $x_t$ — current input

**Outputs:**
- $C_t$ — current cell state
- $h_t$ — current hidden state

### Cell State Update involves two sub-operations:
1. **Remove** unnecessary information (Forget Gate)
2. **Add** new important information (Input Gate)

---

## Mathematical Building Blocks

### All States and Gates are Vectors

| Symbol | Name | Type |
|--------|------|------|
| $C_t$ | Cell State | Vector (dim = $n$) |
| $h_t$ | Hidden State | Vector (dim = $n$) |
| $x_t$ | Input | Vector (dim = $d$, can differ from $n$) |
| $f_t$ | Forget Gate output | Vector (dim = $n$) |
| $i_t$ | Input Gate output | Vector (dim = $n$) |
| $\tilde{C}_t$ | Candidate Cell State | Vector (dim = $n$) |
| $o_t$ | Output Gate output | Vector (dim = $n$) |

> **Critical Rule:** $C_t$, $h_t$, $f_t$, $i_t$, $\tilde{C}_t$, $o_t$ — all have the **exact same dimension** $n$ (= number of hidden units).  
> $x_t$ can have a different dimension $d$.

### Input Vector ($x_t$)

$x_t$ is a single word/token converted into a numerical vector using text vectorization (one-hot, TF-IDF, Word2Vec, etc.).

**Example:** For vocabulary = {good, night, bad} with one-hot encoding:

| Word | Vector |
|------|--------|
| good | $[1, 0, 0]$ |
| night | $[0, 1, 0]$ |
| bad | $[0, 0, 1]$ |

So if $d = 3$, then $x_t$ is a 3-dimensional vector.

---

### Pointwise Operations

These are **element-wise** operations on vectors of equal dimension:

**Pointwise Multiplication (Hadamard Product):**

$$[4, 5, 6] \odot [1, 2, 3] = [4, 10, 18]$$

**Pointwise Addition:**

$$[4, 5, 6] + [1, 2, 3] = [5, 7, 9]$$

---

### Neural Network Layers (Yellow Boxes)

Each yellow box in the LSTM diagram is a **fully connected neural network layer** with:
- A fixed number of nodes = $n$ (number of hidden units)
- An activation function (sigmoid $\sigma$ or $\tanh$)
- Its own weight matrix $W$ and bias $b$

> If you set $n = 3$ hidden units, then **all** neural network layers inside the LSTM cell have exactly 3 nodes.

---

## Gate 1: Forget Gate

### Purpose
Decides **what to remove** from the cell state (long-term memory) based on current input and previous hidden state.

### Architecture

The forget gate is a neural network layer with **sigmoid** activation:

$$\boxed{f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)}$$

Where:
- $[h_{t-1}, x_t]$ = concatenation of previous hidden state and current input → dimension $(n + d)$
- $W_f$ = weight matrix of shape $n \times (n + d)$
- $b_f$ = bias vector of shape $n \times 1$
- $\sigma$ = sigmoid activation (forces output ∈ [0, 1])
- $f_t$ = forget gate output vector of shape $n \times 1$

### How Forgetting Works

$$C_{t-1}' = f_t \odot C_{t-1}$$

The forget gate output $f_t$ acts as a **filter** on the previous cell state:

| $f_t$ value | Effect |
|-------------|--------|
| $1.0$ | **Keep** 100% of that dimension (gate fully open) |
| $0.5$ | **Keep** 50%, effectively halving the information |
| $0.0$ | **Remove** 100% of that dimension (gate fully closed) |

**Example:**

$$C_{t-1} = [4, 5, 6], \quad f_t = [0.5, 0.5, 0.5]$$
$$f_t \odot C_{t-1} = [2.0, 2.5, 3.0]$$

Half the information was retained. If $f_t = [0, 0, 0]$, everything is erased. If $f_t = [1, 1, 1]$, everything passes through unchanged.

---

## Gate 2: Input Gate

### Purpose
Decides **what new information to add** to the cell state.

### Two Sub-Steps

#### Step 1: Calculate Candidate Cell State ($\tilde{C}_t$)

A neural network layer with **tanh** activation proposes new values:

$$\boxed{\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)}$$

- $W_C$ = weight matrix, shape $n \times (n + d)$
- $b_C$ = bias vector, shape $n \times 1$
- Output $\tilde{C}_t$ ∈ $[-1, 1]$ (because of $\tanh$)

#### Step 2: Calculate Input Gate Filter ($i_t$)

A neural network layer with **sigmoid** activation decides which candidates to actually add:

$$\boxed{i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)}$$

- $W_i$ = weight matrix, shape $n \times (n + d)$
- $b_i$ = bias vector, shape $n \times 1$
- Output $i_t$ ∈ $[0, 1]$

#### Filtered Candidate:

$$i_t \odot \tilde{C}_t$$

Only the portions of the candidate deemed important by $i_t$ survive.

**Example:**

$$\tilde{C}_t = [4, 5, 6], \quad i_t = [1, 0, 1]$$
$$i_t \odot \tilde{C}_t = [4, 0, 6]$$

The second dimension was **blocked** (gate closed), the rest was **added**.

---

## Updating the Cell State

Combine the outputs of Forget Gate and Input Gate:

$$\boxed{C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t}$$

| Term | Meaning |
|------|---------|
| $f_t \odot C_{t-1}$ | What **remains** after forgetting |
| $i_t \odot \tilde{C}_t$ | What **new** information is added |
| $C_t$ | Updated long-term memory |

### Why This Solves the Vanishing Gradient Problem

In a vanilla RNN, $h_t$ undergoes repeated **multiplicative** transformations → gradients vanish.

In LSTM, the cell state update is **additive**:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

- If $f_t = [1, 1, 1]$ and $i_t = [0, 0, 0]$: $C_t = C_{t-1}$ → information passes through **unchanged**
- This creates a **gradient highway** — gradients can flow back through many time steps without vanishing
- The LSTM can choose to **preserve** information from step 1 all the way to step 100

---

## Gate 3: Output Gate

### Purpose
Decides the **current hidden state** ($h_t$) — which parts of the (updated) cell state to expose as output.

### Two Sub-Steps

#### Step 1: Calculate Output Gate Filter ($o_t$)

$$\boxed{o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)}$$

- $W_o$ = weight matrix, shape $n \times (n + d)$
- $b_o$ = bias vector, shape $n \times 1$
- Output $o_t$ ∈ $[0, 1]$

#### Step 2: Compute Hidden State ($h_t$)

$$\boxed{h_t = o_t \odot \tanh(C_t)}$$

- $\tanh(C_t)$ squashes the cell state values to $[-1, 1]$
- $o_t$ filters which dimensions of the cell state to expose

---

## Complete LSTM Equations Summary

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget Gate)}$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate)}$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(Candidate Cell State)}$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell State Update)}$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output Gate)}$$

$$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden State)}$$

---

## Parameter Count

Each gate has its own weight matrix of shape $n \times (n + d)$ and bias of shape $n$:

| Gate / Component | Weight Matrix | Bias | Parameters |
|------------------|---------------|------|------------|
| Forget Gate | $W_f$: $n \times (n+d)$ | $b_f$: $n$ | $n(n+d) + n$ |
| Input Gate | $W_i$: $n \times (n+d)$ | $b_i$: $n$ | $n(n+d) + n$ |
| Candidate Cell State | $W_C$: $n \times (n+d)$ | $b_C$: $n$ | $n(n+d) + n$ |
| Output Gate | $W_o$: $n \times (n+d)$ | $b_o$: $n$ | $n(n+d) + n$ |

**Total parameters:**

$$\boxed{4 \times [n(n + d) + n] = 4n(n + d + 1)}$$

Where $n$ = number of hidden units, $d$ = input dimension.

---

## Information Flow Diagram

```
                    C_{t-1} ──→ [×] ──────────→ [+] ──────→ C_t
                                 ↑                ↑              |
                                f_t          i_t ⊙ C̃_t         |
                                 ↑                ↑              ↓
                              ┌──────┐      ┌──────┐        [tanh]
                              │Forget│      │Input │           |
                              │ Gate │      │ Gate │           ↓
                              │  σ   │      │σ + tanh│    o_t ⊙ tanh(C_t)
                              └──┬───┘      └───┬──┘           |
                                 │              │              ↓
    h_{t-1} ────────────────────┼──────────────┤──────────→  h_t
                                 │              │
                              x_t            x_t
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Forget Gate** | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ — decides what to remove from $C_{t-1}$ |
| **Input Gate** | $i_t$ (sigmoid) + $\tilde{C}_t$ (tanh) — decides what new info to add |
| **Cell State Update** | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ — additive (gradient-friendly) |
| **Output Gate** | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ — decides what to expose |
| **Hidden State** | $h_t = o_t \odot \tanh(C_t)$ — filtered version of updated cell state |
| **Why it works** | Additive cell state update avoids vanishing gradients; gates learn to preserve/forget selectively |
| **Total params** | $4n(n + d + 1)$ |
