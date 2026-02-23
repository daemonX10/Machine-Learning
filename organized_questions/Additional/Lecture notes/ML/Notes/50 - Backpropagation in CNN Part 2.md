# Backpropagation in CNN — Part 2

## Overview

This covers how backpropagation works through the three key CNN-specific layers:

1. **Flatten Layer** — reshaping gradients back
2. **Max Pooling Layer** — routing gradients to max positions
3. **Convolution Layer** — computing weight/bias gradients via convolution

---

## CNN Architecture (Recap)

A simple CNN pipeline used for demonstration:

$$\text{Input } X \xrightarrow{\text{Conv}} Z_1 \xrightarrow{\text{ReLU}} A_1 \xrightarrow{\text{MaxPool}} P_1 \xrightarrow{\text{Flatten}} F \xrightarrow{\text{FC}} Z_2 \xrightarrow{\sigma} A_2 \xrightarrow{} L$$

| Symbol | Description |
|--------|-------------|
| $X$ | Input image (e.g., $6 \times 6$) |
| $W_1, b_1$ | Convolution filter weights and bias |
| $Z_1$ | Convolution output: $Z_1 = X * W_1 + b_1$ |
| $A_1$ | After ReLU: $A_1 = \text{ReLU}(Z_1)$ |
| $P_1$ | After Max Pooling on $A_1$ |
| $F$ | Flattened vector from $P_1$ |
| $W_2, b_2$ | Fully connected layer weights and bias |
| $Z_2$ | FC output: $Z_2 = W_2^T F + b_2$ |
| $A_2$ | After sigmoid: $A_2 = \sigma(Z_2)$ |
| $L$ | Loss (Binary Cross-Entropy) |

---

## Forward Propagation Equations

$$Z_1 = X * W_1 + b_1$$

$$A_1 = \text{ReLU}(Z_1)$$

$$P_1 = \text{MaxPool}(A_1)$$

$$F = \text{Flatten}(P_1)$$

$$Z_2 = W_2^T F + b_2$$

$$A_2 = \sigma(Z_2)$$

$$L = -[y \log(A_2) + (1-y)\log(1-A_2)]$$

---

## Backpropagation Goal

Update the convolution parameters using gradient descent:

$$W_1 = W_1 - \eta \cdot \frac{\partial L}{\partial W_1}$$

$$b_1 = b_1 - \eta \cdot \frac{\partial L}{\partial b_1}$$

### Full Chain Rule Path

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial F} \cdot \frac{\partial F}{\partial P_1} \cdot \frac{\partial P_1}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial W_1}$$

> From the FC layer backprop (Part 1), we already know:
> $$\frac{\partial L}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} = A_2 - y$$

---

## 1. Backpropagation Through the Flatten Layer

### Key Insight: No Trainable Parameters

The flatten layer has **no trainable parameters** — it simply reshapes a matrix into a vector.

### Forward

$$P_1 \; (2 \times 2) \xrightarrow{\text{Flatten}} F \; (4 \times 1)$$

### Backward

Simply **reshape** the gradient back to the original shape:

$$\frac{\partial L}{\partial F} \; (4 \times 1) \xrightarrow{\text{Reshape}} \frac{\partial L}{\partial P_1} \; (2 \times 2)$$

> No differentiation is needed — just reverse the reshape operation.

---

## 2. Backpropagation Through Max Pooling

### Key Insight: No Trainable Parameters

Max pooling also has **no trainable parameters**. Backpropagation reverses the max operation.

### Forward (Example)

Given a $4 \times 4$ input with $2 \times 2$ pooling:

$$A_1 = \begin{bmatrix} 1 & 2 & 3 & \mathbf{4} \\ 5 & 6 & 7 & \mathbf{8} \\ 9 & 10 & 11 & \mathbf{12} \\ 13 & 14 & 15 & \mathbf{16} \end{bmatrix} \xrightarrow{\text{MaxPool}} P_1 = \begin{bmatrix} 6 & 8 \\ 14 & 16 \end{bmatrix}$$

Each $2 \times 2$ window keeps only the **maximum** value.

### Backward: Route Gradients to Max Positions

Given incoming gradient:

$$\frac{\partial L}{\partial P_1} = \begin{bmatrix} x_1 & x_2 \\ x_3 & x_4 \end{bmatrix}$$

Reconstruct a $4 \times 4$ matrix where:
- **Max positions** receive the corresponding gradient value
- **All other positions** are set to **zero**

$$\frac{\partial L}{\partial A_1} = \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & x_1 & 0 & x_2 \\ 0 & 0 & 0 & 0 \\ 0 & x_3 & 0 & x_4 \end{bmatrix}$$

### Why This Works

- Only the **maximum element** in each window contributes to the output (and therefore to the loss)
- Non-maximum elements have **zero contribution** to the loss → gradient is zero
- The gradient flows **only through the path of the maximum value**

### Mathematical Notation

$$\frac{\partial L}{\partial A_1}(m, n) = \begin{cases} \frac{\partial L}{\partial P_1}(i, j) & \text{if } A_1(m, n) = \max(\text{window}_{i,j}) \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Backpropagation Through ReLU

At this point we have $\frac{\partial L}{\partial A_1}$. To get $\frac{\partial L}{\partial Z_1}$:

$$\frac{\partial A_1}{\partial Z_1}(m, n) = \begin{cases} 1 & \text{if } Z_1(m, n) > 0 \\ 0 & \text{if } Z_1(m, n) \leq 0 \end{cases}$$

$$\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \mathbb{1}[Z_1 > 0]$$

> Element-wise multiplication with the ReLU mask.

---

## 4. Backpropagation Through the Convolution Layer

### Setup (Simplified Example)

For clarity, assume:
- Input $X$: $3 \times 3$
- Filter $W_1$: $2 \times 2$
- Output $Z_1$: $2 \times 2$ (no padding, stride 1)

$$X = \begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{bmatrix}, \quad W_1 = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}$$

### Forward Convolution Equations

$$z_{11} = x_{11}w_{11} + x_{12}w_{12} + x_{21}w_{21} + x_{22}w_{22} + b_1$$

$$z_{12} = x_{12}w_{11} + x_{13}w_{12} + x_{22}w_{21} + x_{23}w_{22} + b_1$$

$$z_{21} = x_{21}w_{11} + x_{22}w_{12} + x_{31}w_{21} + x_{32}w_{22} + b_1$$

$$z_{22} = x_{22}w_{11} + x_{23}w_{12} + x_{32}w_{21} + x_{33}w_{22} + b_1$$

### Gradient w.r.t. Bias $b_1$

Since $b_1$ appears in **every** element of $Z_1$:

$$\frac{\partial L}{\partial b_1} = \sum_{i,j} \frac{\partial L}{\partial z_{ij}} \cdot \frac{\partial z_{ij}}{\partial b_1} = \sum_{i,j} \frac{\partial L}{\partial z_{ij}} \cdot 1$$

$$\boxed{\frac{\partial L}{\partial b_1} = \sum_{i,j} \frac{\partial L}{\partial Z_1}(i,j)}$$

> Simply **sum all elements** of the gradient matrix $\frac{\partial L}{\partial Z_1}$.

### Gradient w.r.t. Weights $W_1$

Using the chain rule and expanding for each weight:

$$\frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial z_{11}} \cdot x_{11} + \frac{\partial L}{\partial z_{12}} \cdot x_{12} + \frac{\partial L}{\partial z_{21}} \cdot x_{21} + \frac{\partial L}{\partial z_{22}} \cdot x_{22}$$

$$\frac{\partial L}{\partial w_{12}} = \frac{\partial L}{\partial z_{11}} \cdot x_{12} + \frac{\partial L}{\partial z_{12}} \cdot x_{13} + \frac{\partial L}{\partial z_{21}} \cdot x_{22} + \frac{\partial L}{\partial z_{22}} \cdot x_{23}$$

$$\frac{\partial L}{\partial w_{21}} = \frac{\partial L}{\partial z_{11}} \cdot x_{21} + \frac{\partial L}{\partial z_{12}} \cdot x_{22} + \frac{\partial L}{\partial z_{21}} \cdot x_{31} + \frac{\partial L}{\partial z_{22}} \cdot x_{32}$$

$$\frac{\partial L}{\partial w_{22}} = \frac{\partial L}{\partial z_{11}} \cdot x_{22} + \frac{\partial L}{\partial z_{12}} \cdot x_{23} + \frac{\partial L}{\partial z_{21}} \cdot x_{32} + \frac{\partial L}{\partial z_{22}} \cdot x_{33}$$

### The Key Pattern

> $\frac{\partial L}{\partial W_1}$ is the **convolution** of input $X$ with the upstream gradient $\frac{\partial L}{\partial Z_1}$

$$\boxed{\frac{\partial L}{\partial W_1} = X * \frac{\partial L}{\partial Z_1}}$$

This works because each weight element's gradient is essentially sliding $\frac{\partial L}{\partial Z_1}$ over the corresponding positions in $X$ — which is exactly a convolution operation.

---

## Complete Backpropagation Flow (Summary)

| Step | Layer | Operation | Has Trainable Params? |
|------|-------|-----------|-----------------------|
| 1 | FC Layer | Standard backprop (already computed): $\frac{\partial L}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} = A_2 - y$ | Yes ($W_2, b_2$) |
| 2 | Flatten | **Reshape** gradient from vector back to matrix | No |
| 3 | Max Pooling | **Route** gradients to max positions, zero elsewhere | No |
| 4 | ReLU | **Element-wise multiply** with $\mathbb{1}[Z_1 > 0]$ | No |
| 5 | Convolution (bias) | **Sum** all elements of $\frac{\partial L}{\partial Z_1}$ | Yes ($b_1$) |
| 6 | Convolution (weights) | **Convolve** input $X$ with $\frac{\partial L}{\partial Z_1}$ | Yes ($W_1$) |

---

## Key Takeaways

| Concept | Detail |
|---------|--------|
| **Flatten backprop** | Just reshape the gradient — no math needed |
| **MaxPool backprop** | Gradients only flow through max positions; all others get zero |
| **ReLU backprop** | Multiply gradient by binary mask ($1$ if positive, $0$ otherwise) |
| **Conv bias gradient** | Sum of all elements in the upstream gradient |
| **Conv weight gradient** | Convolution of input with upstream gradient: $X * \frac{\partial L}{\partial Z_1}$ |
| **General principle** | Layers without trainable parameters simply reverse their forward operation |
