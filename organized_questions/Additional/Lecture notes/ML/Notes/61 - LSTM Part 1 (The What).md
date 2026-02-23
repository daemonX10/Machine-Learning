# LSTM Part 1 — The What (Long Short-Term Memory)

## Overview

- **Full Form:** Long Short-Term Memory
- **Origin:** Created to solve the inability of vanilla RNNs to handle **long sequences**
- **Root Cause:** Vanishing / Exploding Gradient Problem in RNNs
- **Importance:** Foundation for modern architectures — Transformers, Attention, LLMs all evolved from the LSTM lineage

---

## Quick Recap: Why RNNs Fall Short

### ANN Limitation

- Standard ANNs (fully connected input → output) process the **entire input at once**
- They cannot handle **sequential data** (text, time series) where **order matters**
- Feeding a whole sentence as one vector loses all positional / temporal information

### RNN Improvement

- RNNs introduce a **state** (hidden state $h_t$) — each hidden unit feeds back to itself and others at the **next time step**
- Words are fed **one at a time**, and the hidden state carries context forward

$$h_t = f(h_{t-1}, x_t)$$

| Time Step | Input | Hidden State Passed |
|-----------|-------|---------------------|
| $t=1$ | Word 1 | $h_1$ |
| $t=2$ | Word 2 | $h_2$ |
| $t=3$ | Word 3 | $h_3$ |
| ... | ... | ... |

### The Problem with Long Sequences

When the sequence length is large (e.g., 50+ words) and a decision at the end depends on an early word:

> **Example:** *"Maharashtra is a beautiful state... it has 25 cities... beautiful vegetation... the language spoken there is ____"*
>
> The answer ("Marathi") depends on "Maharashtra" — the **very first word**. But the RNN has already **forgotten** it.

**Cause:** The **vanishing gradient problem** — during backpropagation through many time steps, gradients shrink exponentially, so early inputs lose their influence.

---

## The Core Idea Behind LSTM

### The Analogy: A Story

Imagine you're listening to a long story and must decide if it's **good or bad** at the end (sentiment analysis). Your brain:

1. Processes the story **word by word** (sequential processing)
2. Maintains a **short-term context** — what's happening right now
3. Simultaneously builds a **long-term context** — what's important for the overall story
4. **Adds** important new information to long-term memory (e.g., new hero introduced)
5. **Removes** outdated information from long-term memory (e.g., hero dies → no longer the protagonist)
6. At the end, uses **long-term memory** to make the final decision

### The RNN's Single-Path Problem

In a vanilla RNN, there is **only one path** for retaining information about the past — the hidden state line:

$$h_t \longrightarrow h_{t+1} \longrightarrow h_{t+2} \longrightarrow \cdots$$

This single variable carries **both** short-term and long-term context, but mathematically it **cannot maintain both** — short-term context dominates, and long-term information fades.

### LSTM's Solution: Two Separate Paths

LSTM introduces a **second path** — a dedicated **long-term memory** channel:

| Component | Name | Purpose |
|-----------|------|---------|
| **Cell State** ($C_t$) | Long-Term Memory | Carries important information across many time steps |
| **Hidden State** ($h_t$) | Short-Term Memory | Carries recent/current context for immediate decisions |

> If something is marked as important early on and is never removed, it flows **all the way to the end** via the cell state — regardless of sequence length.

---

## RNN vs LSTM: Key Structural Differences

| Aspect | RNN | LSTM |
|--------|-----|------|
| **States** | 1 (hidden state only) | 2 (hidden state + cell state) |
| **Architecture complexity** | Simple | More complex |
| **Long-term memory** | Prone to vanishing gradients | Maintained via cell state |
| **Communication** | N/A | Short-term ↔ Long-term interaction via gates |

---

## The Three Gates

The complex internal architecture of LSTM exists to manage **communication between short-term and long-term memory**. This is done via **three gates**:

### 1. Forget Gate

- **Purpose:** Decides what to **remove** from long-term memory (cell state)
- **Based on:** Current input ($x_t$) + previous short-term context ($h_{t-1}$)
- **Example:** King Vikram dies → remove "Vikram" from long-term memory since he's no longer relevant

### 2. Input Gate

- **Purpose:** Decides what **new information to add** to long-term memory
- **Based on:** Current input ($x_t$) + previous short-term context ($h_{t-1}$)
- **Example:** Vikram Junior is introduced → add "Vikram Junior" to long-term memory

### 3. Output Gate

- **Purpose:** Decides the **final output** for the current time step
- **Based on:** Current long-term memory + current input context
- **Also:** Creates the **new short-term memory** (hidden state) for the next time step

---

## LSTM as a Black Box (I/O Summary)

### Inputs (3 things enter at time $t$):

| Input | Symbol | Description |
|-------|--------|-------------|
| Previous Cell State | $C_{t-1}$ | Long-term memory from last step |
| Previous Hidden State | $h_{t-1}$ | Short-term memory from last step |
| Current Input | $x_t$ | Current word/token |

### Outputs (2 things come out):

| Output | Symbol | Description |
|--------|--------|-------------|
| Updated Cell State | $C_t$ | Updated long-term memory for next step |
| Current Hidden State | $h_t$ | Short-term memory / output for next step |

### Internal Processing (2 operations):

1. **Update** the long-term memory ($C_{t-1} \rightarrow C_t$):
   - Remove irrelevant information (Forget Gate)
   - Add new important information (Input Gate)

2. **Create** the short-term memory / hidden state ($h_t$):
   - Based on the updated long-term memory + current context (Output Gate)

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What LSTM solves** | Vanishing/exploding gradient problem in RNNs for long sequences |
| **Core innovation** | Two separate memory paths: Cell State (long-term) + Hidden State (short-term) |
| **Number of gates** | 3 — Forget, Input, Output |
| **Forget Gate** | Removes irrelevant info from cell state |
| **Input Gate** | Adds new important info to cell state |
| **Output Gate** | Produces current output and creates next hidden state |
| **Key advantage** | Information can flow unchanged across arbitrarily long sequences via cell state |
