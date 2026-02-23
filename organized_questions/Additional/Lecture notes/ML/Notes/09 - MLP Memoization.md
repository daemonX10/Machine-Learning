# Lecture 09 - MLP Memoization

## What is Memoization?

> **Memoization** is an optimization technique used to speed up programs by **storing results** of expensive function calls and **returning the cached result** when the same inputs occur again.

- Trade-off: uses **extra memory (space)** to **reduce time**
- Core technique in **Dynamic Programming**

---

## Fibonacci Example: Why Memoization Matters

### Naive Recursive Approach

```python
def fib(n):
    if n == 0 or n == 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

**Problem:** Enormous redundant computation.

For `fib(5)`, the call tree:
```
                    fib(5)
                   /      \
              fib(4)       fib(3)
             /    \        /    \
         fib(3)  fib(2) fib(2) fib(1)
        /    \    / \     / \
    fib(2) fib(1)...   ...
     / \
fib(1) fib(0)
```

- `fib(3)` computed **2 times**
- `fib(2)` computed **3 times**
- `fib(1)` computed **5 times**

**Time complexity:** $O(2^n)$ — exponential!

| Input ($n$) | Approximate Time |
|:---:|:---|
| 36 | ~5 seconds |
| 38 | ~17 seconds |
| 40 | ~1 minute |
| 45 | ~3 hours |
| 50 | ~2 months |

### Memoized Approach

```python
def fib_memo(n, memo={0: 0, 1: 1}):
    if n in memo:
        return memo[n]
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]
```

**Result:** `fib_memo(120)` runs in **microseconds** — same time as `fib_memo(38)`.

**Time complexity:** $O(n)$ — linear!

---

## Memoization in Backpropagation

### The Problem: Deep Networks Have Redundant Derivative Calculations

Consider a 4-layer network (Input → Hidden 1 → Hidden 2 → Output):

When computing gradients using chain rule, **deeper layers require derivatives already computed for later layers**.

### Example: Derivative with Two Hidden Layers

For a weight $w_{11}^1$ in the **first hidden layer**, the derivative involves:

$$\frac{\partial L}{\partial w_{11}^1}$$

This depends on outputs and derivatives from **all subsequent layers**, creating a chain:

$$\frac{\partial L}{\partial w_{11}^1} = \frac{\partial L}{\partial a_1^2} \cdot \frac{\partial a_1^2}{\partial z_1^2} \cdot \frac{\partial z_1^2}{\partial a_1^1} \cdot \frac{\partial a_1^1}{\partial z_1^1} \cdot \frac{\partial z_1^1}{\partial w_{11}^1}$$

### The Branching Problem

When a node's output feeds into **multiple nodes** in the next layer, the gradient must account for **all paths**:

```
                    ┌──→ Node 1 (Layer 2) ──→ Loss
  Node 1 (Layer 1) ─┤
                    └──→ Node 2 (Layer 2) ──→ Loss
```

Using the **multivariate chain rule**:

$$\frac{\partial L}{\partial a_1^1} = \frac{\partial L}{\partial a_1^2} \cdot \frac{\partial a_1^2}{\partial a_1^1} + \frac{\partial L}{\partial a_2^2} \cdot \frac{\partial a_2^2}{\partial a_1^1}$$

→ Sum over all paths from the node to the loss.

### Where Memoization Helps

When computing derivatives for layer 1 weights, **many sub-expressions are identical** to what was already computed for layer 2:

| Computed for Layer 2 | Reused in Layer 1 |
|:---|:---|
| $\frac{\partial L}{\partial a_1^2}$ | ✅ Same value |
| $\frac{\partial L}{\partial a_2^2}$ | ✅ Same value |
| $\frac{\partial a_1^2}{\partial z_1^2}$ | ✅ Same value |

**Without memoization:** Recompute everything from scratch → exponential blowup
**With memoization:** Store intermediate derivatives, reuse them → linear time

---

## Backpropagation = Chain Rule + Memoization

$$\boxed{\text{Backpropagation} = \underbrace{\text{Chain Rule}}_{\text{Mathematics}} + \underbrace{\text{Memoization}}_{\text{Computer Science}}}$$

| Component | Field | Role |
|:---|:---|:---|
| **Chain Rule** | Calculus | Provides the formula for computing gradients |
| **Memoization** | Dynamic Programming (CS) | Stores and reuses intermediate derivatives |

### How It Works in Practice

1. **Forward pass:** Compute and **store** all intermediate activations $(a^l, z^l)$
2. **Backward pass:** Start from the output layer, compute derivatives **layer by layer moving backward**
3. At each layer, **reuse** stored values from:
   - The forward pass (activations)
   - Previously computed derivatives from later layers

This is exactly what libraries like **TensorFlow** and **PyTorch** implement internally.

---

## Key Insight: Deeper Networks = More Savings

| Network Depth | Without Memoization | With Memoization |
|:---|:---|:---|
| 2 layers | Manageable | Helpful |
| 5 layers | Very slow | Essential |
| 10+ layers | Infeasible | Practical |

> As networks get **deeper**, the derivative expressions grow exponentially complex. Memoization transforms backpropagation from an infeasible computation into an efficient one.

---

## Summary

1. **Memoization** = Cache computed results, avoid redundant recalculation
2. In neural networks, **derivatives computed for later layers are reused in earlier layers**
3. Without memoization, training deep networks would be **exponentially slow**
4. Backpropagation achieves efficiency by combining **chain rule** (math) with **memoization** (CS)
5. Extra **memory** is used to store intermediate values, but **time** is dramatically reduced
