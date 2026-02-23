# Problems with RNN

## Overview

Simple RNNs suffer from two fundamental problems that limit their practical use:

| Problem | Root Cause | Effect |
|---------|-----------|--------|
| **Long-Term Dependency** | Vanishing Gradients | Cannot remember distant past inputs |
| **Unstable Training** | Exploding Gradients | Training diverges or stalls |

> These problems motivate the need for advanced architectures like **LSTM** and **GRU**.

---

## Problem 1: Long-Term Dependency

### What Is It?

In sequential data, the output at a given time step may depend on inputs from **many steps ago**. RNNs fail to capture these distant dependencies — they have **short-term memory**.

### Example: Next Word Prediction

> *"Marathi is spoken in Maharashtra. I visited many beautiful places there but could not enjoy it properly because I don't understand ____"*

- The correct answer is **"Marathi"**
- This word depends on information from the **very beginning** of the sentence
- RNN would fail because the relevant context is too far back in the sequence

| Dependency Type | Description | RNN Performance |
|----------------|-------------|-----------------|
| **Short-term** | Output depends on recent inputs | Works well |
| **Long-term** | Output depends on distant past inputs | Fails |

---

### Mathematical Explanation: Why Vanishing Gradients Cause This

During backpropagation through time (BPTT), the gradient of the loss w.r.t. $W_{in}$ involves computing:

$$\frac{\partial L}{\partial W_{in}} = \sum_{t=1}^{T} \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_T} \cdot \left( \prod_{k=t+1}^{T} \frac{\partial o_k}{\partial o_{k-1}} \right) \cdot \frac{\partial o_t}{\partial W_{in}}$$

The critical term is the **product of partial derivatives across time steps**:

$$\prod_{k=t+1}^{T} \frac{\partial o_k}{\partial o_{k-1}}$$

#### Computing Each Factor

For a simple RNN with $\tanh$ activation:

$$o_t = \tanh(x_t \cdot W_{in} + o_{t-1} \cdot W_h)$$

$$\frac{\partial o_t}{\partial o_{t-1}} = f'(\cdot) \cdot W_h$$

Where $f'(\cdot)$ is the derivative of $\tanh$, which lies in $[0, 1]$.

#### The Vanishing Effect

If $f'(\cdot) \cdot W_h$ has values **between 0 and 1**:

$$\left( f'(\cdot) \cdot W_h \right)^{100} \approx 0$$

| Number of Time Steps | Product $\prod$ | Gradient Contribution |
|---------------------|------------------|----------------------|
| Few (short-term) | Moderate | Significant |
| Many (long-term) | $\approx 0$ | **Negligible** |

> As the number of time steps grows, the **long-term gradient contributions vanish** → the network learns only from recent inputs.

---

### Solutions for Vanishing Gradients

| Solution | Description |
|----------|-------------|
| **Different activation function** | Use ReLU instead of $\tanh$ — derivative is not bounded to $(0, 1)$ |
| **Better weight initialization** | Initialize $W_h$ as an **identity matrix** — multiplying by identity preserves values |
| **Skip connections** | Allow gradients to flow through shortcut paths, bypassing many time steps |
| **Use LSTM / GRU** | Purpose-built architectures with gating mechanisms that preserve long-term information |

---

## Problem 2: Unstable Training (Exploding Gradients)

### What Is It?

The opposite of vanishing gradients — the gradient product becomes **extremely large**, causing:

- Weight updates that are too large
- Weights diverge to infinity ($\pm \infty$)
- Loss becomes `NaN`
- Model **cannot train at all**

### When Does It Happen?

The same product term causes the issue:

$$\prod_{k=t+1}^{T} f'(\cdot) \cdot W_h$$

If the values are **greater than 1** (e.g., using ReLU + large weights):

$$(\text{value} > 1)^{100} \rightarrow \infty$$

| Cause | Why |
|-------|-----|
| **Activation function** | ReLU derivative is always positive; repeated multiplication grows unbounded |
| **Large weight values** | $W_h > 1$ → product grows exponentially |
| **High learning rate** | Amplifies already-large gradient updates |

### Effects

$$\text{Exploding gradients} \Rightarrow \text{Huge weight updates} \Rightarrow W \to \pm \infty \Rightarrow \text{Training diverges}$$

---

### Solutions for Exploding Gradients

| Solution | Description |
|----------|-------------|
| **Gradient Clipping** | Cap gradients at a maximum threshold: if $\|g\| > \text{max}$, scale it down |
| **Lower learning rate** | Reduce the step size to prevent overshooting |
| **Use LSTM / GRU** | Gating mechanisms control gradient flow and prevent explosion |

#### Gradient Clipping

```python
# In Keras/TensorFlow
optimizer = keras.optimizers.Adam(clipvalue=1.0)
# or
optimizer = keras.optimizers.Adam(clipnorm=1.0)
```

- **`clipvalue`**: Clips each gradient component to $[-v, v]$
- **`clipnorm`**: Scales the gradient vector so its norm does not exceed the threshold

---

## Vanishing vs Exploding Gradients — Comparison

| Aspect | Vanishing Gradient | Exploding Gradient |
|--------|-------------------|-------------------|
| **Gradient magnitude** | $\to 0$ | $\to \infty$ |
| **Caused by** | $f'(\cdot) \cdot W_h < 1$ | $f'(\cdot) \cdot W_h > 1$ |
| **Symptom** | No learning, loss stagnates | Loss = NaN, weights diverge |
| **Effect on RNN** | Cannot learn long-term dependencies | Unstable / impossible training |
| **Common activation** | $\tanh$, sigmoid | ReLU (unbounded derivative) |
| **Fix** | LSTM/GRU, ReLU, skip connections, identity init | Gradient clipping, lower LR, LSTM/GRU |

---

## Why LSTM Is the Solution

Both problems motivate the move from simple RNN → LSTM:

- LSTM uses **gates** (forget, input, output) to control information flow
- The **cell state** provides a highway for gradients to flow across many time steps without vanishing
- Gating prevents uncontrolled gradient growth (exploding)
- LSTM can selectively **remember or forget** information over long sequences

---

## Summary

| Problem | Root Cause | Key Equation | Impact | Primary Fix |
|---------|-----------|--------------|--------|-------------|
| Long-term dependency | Vanishing gradients | $\prod (f' \cdot W_h)^T \to 0$ | Can't learn distant patterns | LSTM / GRU |
| Unstable training | Exploding gradients | $\prod (f' \cdot W_h)^T \to \infty$ | Training diverges | Gradient clipping + LSTM |
