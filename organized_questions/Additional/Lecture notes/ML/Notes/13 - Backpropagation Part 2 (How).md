# Lecture 13 — Backpropagation Part 2: The How

## 1. Overview

This lecture implements backpropagation **from scratch** in code on two problems:
1. **Regression** (CGPA, Resume Score → Package)
2. **Classification** (CGPA, Profile Score → Placed yes/no)

Then compares results with Keras.

---

## 2. Regression Example — From Scratch

### Dataset

| CGPA | Resume Score | Package (LPA) |
|---|---|---|
| 8 | 8 | 4 |
| 7 | 9 | 5 |
| 6 | 10 | 5 |
| 5 | 5 | 2 |

> IQ replaced with Resume Score to keep inputs in similar ranges — helps training.

### Architecture

- **Input layer:** 2 neurons (CGPA, Resume Score)
- **Hidden layer:** 2 neurons (linear activation)
- **Output layer:** 1 neuron (linear activation)
- **Loss function:** MSE → $L = (y - \hat{y})^2$
- **Total parameters:** 9

### Initialization

- All weights = 0.1
- All biases = 0

### Code Implementation — Key Functions

#### 1. `initialize_parameters(architecture)`

```python
# Input: architecture = [2, 2, 1]
# Output: dictionary of W1, b1, W2, b2 initialized
# W1 shape: (2, 2), b1 shape: (1, 2)
# W2 shape: (2, 1), b2 shape: (1, 1)
```

#### 2. `linear_forward(W, X, b)`

Computes single neuron output:

$$z = W \cdot X + b$$

```python
return np.dot(X, W) + b
```

#### 3. `forward_propagation(X, params)`

Chains linear_forward across layers. Returns:
- $\hat{y}$ (final prediction)
- $a_1$ (hidden layer activations — needed for weight updates)

#### 4. `update_parameters(params, y, y_hat, a1, X)`

Applies gradient descent using the 9 derivative formulas:

$$w_{new} = w_{old} + \eta \cdot 2(y - \hat{y}) \cdot (\text{appropriate factor})$$

> Note: The minus signs in the derivative and the gradient descent subtraction combine to give a **plus** sign.

**Learning rate used:** $\eta = 0.001$

### Training Loop

```python
params = initialize_parameters([2, 2, 1])

for epoch in range(num_epochs):
    losses = []
    for j in range(num_students):
        x = df.iloc[j, :2].values  # input
        y = df.iloc[j, 2]          # target
        
        y_hat, a1 = forward_propagation(x, params)
        
        update_parameters(params, y, y_hat, a1, x)
        
        loss = (y - y_hat) ** 2
        losses.append(loss)
    
    avg_loss = np.mean(losses)
    print(f"Epoch {epoch}: Loss = {avg_loss}")
```

### Results (5 epochs)

| Epoch | Average Loss |
|---|---|
| 1 | 3.25 |
| 2 | 2.08 |
| 3 | 1.47 |
| 4 | ~1.37 |
| 5 | ~1.34 |

### Keras Verification

```python
model = Sequential([
    Dense(2, activation='linear', input_dim=2),
    Dense(1, activation='linear')
])
# Set same initial weights (0.1) and biases (0)
# Set same learning rate (0.001)
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.001))
model.fit(X, y, epochs=5, batch_size=1)
```

Keras gives nearly identical loss values → validates the from-scratch implementation.

---

## 3. Classification Example — From Scratch

### Dataset

| CGPA | Profile Score | Placed |
|---|---|---|
| 8 | 8 | 1 |
| 7 | 9 | 1 |
| 6 | 6 | 0 |
| 5 | 5 | 0 |

### Key Changes from Regression

| Aspect | Regression | Classification |
|---|---|---|
| **Activation** | Linear (all layers) | **Sigmoid** (all layers) |
| **Loss function** | MSE: $(y - \hat{y})^2$ | **Binary Cross-Entropy**: $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| **Algorithm structure** | Identical | Identical |

### Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

$$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$

### Derivative Computation (Classification)

The chain rule now includes the sigmoid derivative at each layer.

#### Step 1: $\frac{\partial L}{\partial \hat{y}}$

$$\frac{\partial L}{\partial \hat{y}} = \frac{-(y - \hat{y})}{\hat{y}(1 - \hat{y})}$$

#### Step 2: $\frac{\partial \hat{y}}{\partial z_{\text{final}}}$

$$= \sigma(z) \cdot (1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

#### Combined (Steps 1 × 2):

$$\frac{\partial L}{\partial z_{\text{final}}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_{\text{final}}} = -(y - \hat{y})$$

> Beautiful simplification — the $\hat{y}(1-\hat{y})$ cancels out!

#### Output Layer Derivatives

| Parameter | Derivative |
|---|---|
| $\frac{\partial L}{\partial w_{2_{11}}}$ | $-(y - \hat{y}) \cdot o_1$ |
| $\frac{\partial L}{\partial w_{2_{12}}}$ | $-(y - \hat{y}) \cdot o_2$ |
| $\frac{\partial L}{\partial b_{21}}$ | $-(y - \hat{y})$ |

Compare with regression: $-2(y - \hat{y})$ replaced by $-(y - \hat{y})$.

#### Hidden Layer Derivatives

For hidden layer, additional sigmoid derivative terms appear:

$$\frac{\partial L}{\partial w_{11}} = -(y - \hat{y}) \cdot w_{2_{11}} \cdot o_1(1 - o_1) \cdot x_{i1}$$

$$\frac{\partial L}{\partial w_{12}} = -(y - \hat{y}) \cdot w_{2_{11}} \cdot o_1(1 - o_1) \cdot x_{i2}$$

$$\frac{\partial L}{\partial b_{11}} = -(y - \hat{y}) \cdot w_{2_{11}} \cdot o_1(1 - o_1)$$

Similarly for $w_{21}, w_{22}, b_{12}$ with $w_{2_{12}}$ and $o_2$ replacing $w_{2_{11}}$ and $o_1$.

### Code Changes

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def linear_forward(W, X, b):
    return sigmoid(np.dot(X, W) + b)  # sigmoid wraps the dot product

# Loss function changes to binary cross-entropy
loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
```

### Results

- Loss stabilizes around **0.6** — doesn't decrease further
- Keras produces **identical** behavior on this small dataset
- The data is too small and not linearly separable enough for convergence — not a code error

---

## 4. Summary of Derivative Formulas

### Regression (Linear Activation, MSE Loss)

| Layer | Parameter | Derivative |
|---|---|---|
| Output | $w_{2_{11}}$ | $-2(y - \hat{y}) \cdot o_1$ |
| Output | $w_{2_{12}}$ | $-2(y - \hat{y}) \cdot o_2$ |
| Output | $b_{21}$ | $-2(y - \hat{y})$ |
| Hidden | $w_{11}$ | $-2(y - \hat{y}) \cdot w_{2_{11}} \cdot x_1$ |
| Hidden | $w_{12}$ | $-2(y - \hat{y}) \cdot w_{2_{11}} \cdot x_2$ |
| Hidden | $b_{11}$ | $-2(y - \hat{y}) \cdot w_{2_{11}}$ |

### Classification (Sigmoid Activation, BCE Loss)

| Layer | Parameter | Derivative |
|---|---|---|
| Output | $w_{2_{11}}$ | $-(y - \hat{y}) \cdot o_1$ |
| Output | $w_{2_{12}}$ | $-(y - \hat{y}) \cdot o_2$ |
| Output | $b_{21}$ | $-(y - \hat{y})$ |
| Hidden | $w_{11}$ | $-(y - \hat{y}) \cdot w_{2_{11}} \cdot o_1(1-o_1) \cdot x_1$ |
| Hidden | $w_{12}$ | $-(y - \hat{y}) \cdot w_{2_{11}} \cdot o_1(1-o_1) \cdot x_2$ |
| Hidden | $b_{11}$ | $-(y - \hat{y}) \cdot w_{2_{11}} \cdot o_1(1-o_1)$ |

**Key difference:** Sigmoid adds the $o_k(1-o_k)$ term in hidden layer derivatives.
