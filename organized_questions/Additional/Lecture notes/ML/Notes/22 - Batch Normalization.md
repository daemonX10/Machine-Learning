# Lecture 22 — Batch Normalization

## 1. What Is Batch Normalization?

> **Batch Normalization** is an algorithmic method that makes neural network training **faster** and more **stable** by normalizing the activations of each layer using the mean and variance of the current mini-batch.

- Introduced in **2015** (Ioffe & Szegedy)
- Applied **layer-by-layer** on hidden layer outputs
- Optional per layer — you can choose which layers get it

---

## 2. Why Batch Normalization? (Two Reasons)

### Reason 1: Extension of Input Normalization

We normalize **inputs** to get symmetric loss contours → faster training. The same logic applies to **hidden layer outputs** (which are inputs to the next layer). Batch normalization extends this idea to every hidden layer.

### Reason 2: Internal Covariate Shift

#### Covariate Shift (External)
When the **input distribution changes** between training and test data (even though the X→Y relationship remains the same), model performance degrades and retraining is needed.

*Example:* Training a rose classifier with only red roses → fails on white/pink roses at test time.

#### Internal Covariate Shift
During training, weights are **constantly changing** → the output distribution of each hidden layer **shifts continuously** → subsequent layers receive inputs with unstable distributions → training becomes slow and unstable.

> **Batch Normalization fixes this** by ensuring the distribution at each layer's output is always normalized (mean ≈ 0, std ≈ 1), giving downstream layers a **stable foundation**.

---

## 3. How Batch Normalization Works

### Step-by-Step Process

Given a hidden layer with neurons processing a **mini-batch** of size $m$:

#### Step 1: Compute Weighted Sum
For neuron $j$, compute $z_j$ for all $m$ examples in the batch.

#### Step 2: Normalize (per neuron, across the batch)

Compute batch statistics:

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} z_j^{(i)}$$

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (z_j^{(i)} - \mu_B)^2$$

Normalize:

$$\hat{z}_j^{(i)} = \frac{z_j^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

- $\epsilon$ is a small constant (e.g., $10^{-8}$) to prevent division by zero
- After this step: mean ≈ 0, std ≈ 1

#### Step 3: Scale and Shift (Learnable Parameters)

$$\tilde{z}_j^{(i)} = \gamma \cdot \hat{z}_j^{(i)} + \beta$$

- $\gamma$ (scale) — initialized to **1**
- $\beta$ (shift) — initialized to **0**
- Both are **learnable parameters** updated via backpropagation
- **Each neuron** has its own $\gamma$ and $\beta$

#### Step 4: Apply Activation Function

$$a_j^{(i)} = g(\tilde{z}_j^{(i)})$$

### Where BN Is Applied

Two common placements:

| Option | Description |
|---|---|
| **Before activation** (more popular) | $z → \text{BN} → g(z) → a$ |
| After activation | $z → g(z) → \text{BN} → a$ |

---

## 4. Training vs Testing

### During Training
- Use **mini-batch mean** ($\mu_B$) and **mini-batch variance** ($\sigma_B^2$)
- Maintain an **exponentially weighted moving average (EWMA)** of $\mu$ and $\sigma$ across all batches

### During Testing / Inference
- Only **one sample** is available → can't compute batch statistics
- Use the **EWMA values** of $\mu$ and $\sigma$ accumulated during training

### Parameters Per BN Layer (per neuron)

| Parameter | Type | Count |
|---|---|---|
| $\gamma$ | Learnable (trainable) | 1 per neuron |
| $\beta$ | Learnable (trainable) | 1 per neuron |
| $\mu_{\text{EWMA}}$ | Non-learnable (running stat) | 1 per neuron |
| $\sigma_{\text{EWMA}}$ | Non-learnable (running stat) | 1 per neuron |

**Total per neuron:** 4 parameters (2 trainable + 2 non-trainable)

*Example:* Layer with 3 neurons → 12 additional parameters (6 trainable, 6 non-trainable)

---

## 5. Advantages of Batch Normalization

| # | Advantage | Explanation |
|---|---|---|
| 1 | **More stable training** | Wider range of valid hyperparameter values; less sensitivity to initialization |
| 2 | **Faster training** | Allows higher learning rates → fewer epochs to converge |
| 3 | **Mild regularization effect** | Randomness in batch $\mu$/$\sigma$ introduces noise → slight overfitting reduction (not a substitute for dropout) |
| 4 | **Reduces importance of weight initialization** | Smooths the loss surface → model converges from various starting points |

---

## 6. Keras Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

model = Sequential([
    Dense(3, activation='relu', input_dim=2),
    BatchNormalization(),       # BN after first hidden layer
    Dense(2, activation='relu'),
    BatchNormalization(),       # BN after second hidden layer
    Dense(1, activation='sigmoid')  # Output — no BN here
])
```

### Experimental Results (Circles Dataset)

| Metric | Without BN | With BN |
|---|---|---|
| Epochs to reach ~68% accuracy | ~70 epochs | ~20 epochs |
| Overall training accuracy | Lower | Higher |
| Convergence speed | Slower | **3–4× faster** |

---

## 7. Summary

1. **Batch Normalization** normalizes hidden layer outputs per mini-batch → stabilizes training
2. It combats **Internal Covariate Shift** — the constant distribution change at hidden layers
3. Two steps: **(a)** Normalize to zero mean/unit variance, **(b)** Scale and shift with learnable $\gamma$, $\beta$
4. During inference, use **running averages** of $\mu$ and $\sigma$ from training
5. Major benefits: **faster convergence, higher learning rates, stable training, mild regularization**
6. In Keras: just add `BatchNormalization()` layer after each hidden layer
