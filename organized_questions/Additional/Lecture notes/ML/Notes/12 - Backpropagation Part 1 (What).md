# Lecture 12 — Backpropagation Part 1: The What

## 1. Definition

> **Backpropagation** (short for *backward propagation of errors*) is an algorithm used to **train neural networks** — i.e., find optimal weights and biases.

Given a neural network and a loss function, backpropagation calculates the **gradient** (partial derivative) of the loss function **with respect to each weight and bias**.

**Simple definition:** Backpropagation finds the values of weights and biases that minimize the loss function for a given dataset.

---

## 2. Prerequisites

Two concepts must be understood first:

1. **Gradient Descent** — optimization algorithm to minimize loss
2. **Forward Propagation** — technique used by neural networks to make predictions

---

## 3. Setup Example

**Dataset:** CGPA, IQ → Package (regression)

| CGPA | IQ | Package (LPA) |
|---|---|---|
| 8 | 80 | 3 |
| 7 | 70 | 7 |
| 6 | 60 | 5 |
| 5 | 50 | 1 |

**Architecture:** Simple neural network with:
- Input layer: 2 neurons (CGPA, IQ)
- Hidden layer: 2 neurons
- Output layer: 1 neuron

**Activation:** Linear (everywhere — it's a regression problem)

**Parameter labeling:**

| Connection | Weight | Bias |
|---|---|---|
| Input₁ → Hidden₁ | $w_{11}$ | $b_{11}$ |
| Input₂ → Hidden₁ | $w_{12}$ | — |
| Input₁ → Hidden₂ | $w_{21}$ | $b_{12}$ |
| Input₂ → Hidden₂ | $w_{22}$ | — |
| Hidden₁ → Output | $w_{2_{11}}$ | $b_{21}$ |
| Hidden₂ → Output | $w_{2_{12}}$ | — |

**Total trainable parameters:** 9 (6 weights + 3 biases)

---

## 4. Backpropagation Algorithm — Step by Step

### Step 0: Initialize Weights & Biases

Choose an initialization strategy:
- Random values
- All weights = 1, all biases = 0 (used in this example for simplicity)
- Other techniques (Xavier, He, etc.)

### Step 1: Select Training Point

Pick one student from the dataset (e.g., Student 1: CGPA=8, IQ=80).

### Step 2: Forward Propagation → Prediction

Feed inputs through the network using current weights:

$$\hat{y} = f(W \cdot X + b)$$

Example: Prediction = 8 LPA (but actual = 3 LPA).

### Step 3: Calculate Loss

Using **MSE** (for regression):

$$L = (y - \hat{y})^2 = (3 - 8)^2 = 25$$

### Step 4: Update Weights & Biases (Gradient Descent)

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

$$b_{new} = b_{old} - \eta \cdot \frac{\partial L}{\partial b}$$

Apply this update rule for **all 9 parameters** (6 weights + 3 biases).

### Step 5: Repeat for Next Point

Move to Student 2 → forward propagation → loss → update weights → repeat for all students.

### Step 6: Multiple Epochs

Repeat the entire dataset pass multiple times until **convergence** (loss is minimized).

---

## 5. The Core Challenge — Computing Derivatives

The key calculation in Step 4 is $\frac{\partial L}{\partial w}$ for every parameter.

### Dependency Chain

Loss depends on $\hat{y}$, which depends on hidden layer outputs, which depend on weights and inputs:

$$L \xrightarrow{\text{depends on}} \hat{y} \xrightarrow{\text{depends on}} o_1, o_2, w_{2_{11}}, w_{2_{12}}, b_{21}$$

Each hidden output itself depends on:

$$o_1 \xrightarrow{\text{depends on}} w_{11}, w_{12}, b_{11}, x_1, x_2$$

### Chain Rule of Differentiation

Since weights are **indirectly** related to loss, we use the **chain rule**:

$$\frac{\partial L}{\partial w_{2_{11}}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_{2_{11}}}$$

For deeper layers:

$$\frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_1} \cdot \frac{\partial o_1}{\partial w_{11}}$$

---

## 6. Computing All 9 Derivatives (Regression, Linear Activation)

### Output Layer Derivatives

**Loss function:** $L = (y - \hat{y})^2$

$$\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y})$$

**Output computation:** $\hat{y} = w_{2_{11}} \cdot o_1 + w_{2_{12}} \cdot o_2 + b_{21}$

| Parameter | Derivative |
|---|---|
| $\frac{\partial L}{\partial w_{2_{11}}}$ | $-2(y - \hat{y}) \cdot o_1$ |
| $\frac{\partial L}{\partial w_{2_{12}}}$ | $-2(y - \hat{y}) \cdot o_2$ |
| $\frac{\partial L}{\partial b_{21}}$ | $-2(y - \hat{y}) \cdot 1$ |

### Hidden Layer Derivatives (via Chain Rule)

$$\frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial o_1} \cdot \frac{\partial o_1}{\partial w_{11}}$$

| Parameter | Derivative |
|---|---|
| $\frac{\partial L}{\partial w_{11}}$ | $-2(y - \hat{y}) \cdot w_{2_{11}} \cdot x_{i1}$ |
| $\frac{\partial L}{\partial w_{12}}$ | $-2(y - \hat{y}) \cdot w_{2_{11}} \cdot x_{i2}$ |
| $\frac{\partial L}{\partial b_{11}}$ | $-2(y - \hat{y}) \cdot w_{2_{11}} \cdot 1$ |
| $\frac{\partial L}{\partial w_{21}}$ | $-2(y - \hat{y}) \cdot w_{2_{12}} \cdot x_{i1}$ |
| $\frac{\partial L}{\partial w_{22}}$ | $-2(y - \hat{y}) \cdot w_{2_{12}} \cdot x_{i2}$ |
| $\frac{\partial L}{\partial b_{12}}$ | $-2(y - \hat{y}) \cdot w_{2_{12}} \cdot 1$ |

### Pattern Observed

- **Common factor** in all output-layer derivatives: $-2(y - \hat{y})$
- **Common factor** in hidden-layer derivatives: $-2(y - \hat{y}) \cdot w_{2_{\cdot}}$
- The final varying factor is the partial of the hidden neuron's output w.r.t. its own parameter

---

## 7. Complete Training Loop (Pseudocode)

```
Initialize all weights = 1, biases = 0

for epoch in range(num_epochs):      # outer loop
    for student in dataset:           # inner loop
        1. Forward propagation → ŷ
        2. Calculate loss L = (y - ŷ)²
        3. Compute all 9 partial derivatives
        4. Update all weights and biases:
           w_new = w_old - η * ∂L/∂w
    
    avg_loss = mean(all losses in this epoch)
    print(epoch, avg_loss)

# Convergence: stop when loss is minimized
```

---

## 8. Key Takeaways

1. Backpropagation = algorithm to find optimal weights/biases by propagating error **backward** through the network
2. Uses **chain rule** to compute gradients for parameters not directly connected to the loss
3. **Gradient descent** uses these gradients to update parameters iteratively
4. One pass through entire dataset = **one epoch**
5. Training continues for multiple epochs until convergence
