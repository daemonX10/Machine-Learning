# Lecture 06 - Problem with Perceptron

## Core Problem

**Perceptron can only classify linearly separable data.** It draws a single straight line (hyperplane) to separate classes and fails when the data requires a non-linear decision boundary.

## Demonstration with Logic Gates

| Logic Gate | Linearly Separable? | Perceptron Works? |
|:---:|:---:|:---:|
| **AND** | Yes | ✅ |
| **OR** | Yes | ✅ |
| **XOR** | No | ❌ |

### AND Gate

| $x_1$ | $x_2$ | Output |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

→ A single line can separate the class-1 point from class-0 points.

### OR Gate

| $x_1$ | $x_2$ | Output |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

→ A single line can separate the class-0 point from class-1 points.

### XOR Gate (The Problem)

| $x_1$ | $x_2$ | Output |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

→ **No single straight line can separate the two classes.** Output is 1 when inputs differ, 0 when they are the same — creating a non-linear pattern.

## Why Perceptron Fails on XOR

- Perceptron learns a linear decision boundary: $w_1 x_1 + w_2 x_2 + b = 0$
- XOR requires a **non-linear** boundary (you need at least two lines or a curve)
- No matter how long you train a perceptron on XOR data, it will **never converge**

## Code Demonstration

```python
from sklearn.linear_model import Perceptron

# Train perceptron on AND data → converges, clean boundary ✅
# Train perceptron on OR data  → converges, clean boundary ✅
# Train perceptron on XOR data → fails to find boundary    ❌
```

## TensorFlow Playground Verification

- **Linearly separable data** (simple clusters): Single perceptron with no hidden layers → converges quickly
- **XOR-like data** (spiral/circular): Single perceptron → **never converges**, no matter how many epochs

## Key Takeaway

> The perceptron's inability to handle non-linear relationships is the fundamental motivation for **Multi-Layer Perceptrons (MLPs)** — combining multiple perceptrons to capture non-linearity.

## What Comes Next

The solution: **Multi-Layer Perceptron (MLP)** — stack multiple perceptrons into layers to create non-linear decision boundaries.
