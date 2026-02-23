# Lecture 11 — Loss Functions in Deep Learning

## 1. What is a Loss Function?

> *"A loss function is a method of evaluating how well your algorithm/model fits your data."*

- A **mathematical function** of the model's parameters (weights & biases)
- **High loss** → model performing poorly; **Low loss** → model performing well
- Adjusting parameters changes the loss value
- Goal: find parameter values that **minimize** the loss function (using gradient descent)

> *"You can't improve what you can't measure."* — Peter Drucker

---

## 2. Loss Function vs Cost Function

| Aspect | Loss Function | Cost Function |
|---|---|---|
| **Scope** | Single training example | Entire training set (batch) |
| **Formula (MSE)** | $L = (y_i - \hat{y}_i)^2$ | $J = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ |
| **Alias** | Error function | — |

> **Interview tip:** These are **not** interchangeable — know the difference.

---

## 3. Taxonomy of Loss Functions

```
Loss Functions in Deep Learning
├── Regression
│   ├── Mean Squared Error (MSE)
│   ├── Mean Absolute Error (MAE)
│   └── Huber Loss
├── Classification
│   ├── Binary Cross-Entropy (log loss)
│   ├── Categorical Cross-Entropy
│   └── Hinge Loss (SVM)
├── Autoencoders → KL Divergence
├── GANs → Discriminator / Minimax GAN Loss
├── Object Detection → Focal Loss
└── Embeddings → Triplet Loss / Contrastive Loss
```

- Custom losses can be implemented in Keras via custom loss function API

---

## 4. Mean Squared Error (MSE)

### Formula

$$L_{MSE} = (y_i - \hat{y}_i)^2$$

$$J_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Also called: **Squared Loss**, **L2 Loss**

### Why Square?

Simple difference $(y - \hat{y})$ can be positive or negative → negatives cancel out positives when summed → misleading total error. Squaring makes everything positive.

### Quadratic Nature — Outlier Sensitivity

Because of the $(\cdot)^2$ term:
- 1 unit error → 1 unit loss
- 2 unit error → **4** unit loss
- 4 unit error → **16** unit loss

Points **far from the true value** get **disproportionately penalized**, causing drastic weight updates.

### Advantages

| # | Advantage |
|---|---|
| 1 | Easy to interpret |
| 2 | Always differentiable → gradient descent applies smoothly |
| 3 | Only **one** local minimum (convex) → guaranteed global optimum |

### Disadvantages

| # | Disadvantage |
|---|---|
| 1 | Units are **squared** (e.g., LPA²) — not directly interpretable |
| 2 | **Not robust to outliers** — outliers cause huge squared errors → large weight swings |

### Keras Usage

```python
# Output layer: linear activation
model.add(Dense(1, activation='linear'))

# Compile
model.compile(loss='mean_squared_error', optimizer='adam')
```

---

## 5. Mean Absolute Error (MAE)

### Formula

$$L_{MAE} = |y_i - \hat{y}_i|$$

$$J_{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

Also called: **L1 Loss**, **Absolute Loss**

### Advantages

| # | Advantage |
|---|---|
| 1 | Intuitive — absolute difference is easy to understand |
| 2 | Same units as $y$ (e.g., LPA, not LPA²) |
| 3 | **Robust to outliers** — no quadratic penalty; outliers aren't disproportionately punished |

### Disadvantages

| # | Disadvantage |
|---|---|
| 1 | **Not differentiable at 0** → requires sub-gradients → increases computational complexity |

### When to Use

- **No outliers** → use MSE
- **Outliers present** → use MAE

---

## 6. Huber Loss

### Formula

$$L_\delta = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

### Key Idea

- **Non-outlier points** → behaves like **MSE** (quadratic)
- **Outlier points** → behaves like **MAE** (linear)
- $\delta$ is a tunable hyperparameter that defines the threshold

### When to Use

Best when data has a **mix of normal points and outliers** — it interpolates between MSE and MAE.

---

## 7. Binary Cross-Entropy (BCE)

### Formula (Single Point)

$$L = -\left[y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y})\right]$$

### Cost Function (Full Dataset)

$$J = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]$$

Also called: **Log Loss**

### Setup Requirements

- Used for **binary classification** (2 classes: yes/no, 0/1)
- Output layer: **single neuron** with **sigmoid** activation

```python
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
```

### Worked Example

| Student | $y$ | $\hat{y}$ | Loss |
|---|---|---|---|
| 1 | 1 | 0.73 | $-1 \cdot \log(0.73) = 0.13$ |
| 2 | 0 | 0.25 | $-(1-0)\log(1-0.25) = -\log(0.75) = 0.12$ |

### Properties

| Advantage | Disadvantage |
|---|---|
| **Differentiable** — gradient descent applies easily | Multiple local minima possible |
| Well-established (from logistic regression theory) | Not intuitive at first glance |

---

## 8. Categorical Cross-Entropy (CCE)

### Formula (Single Point)

$$L = -\sum_{j=1}^{K} y_j \cdot \log(\hat{y}_j)$$

where $K$ = number of classes.

### Cost Function (Full Dataset)

$$J = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{K} y_{ij} \cdot \log(\hat{y}_{ij})$$

### Setup Requirements

- Used for **multi-class classification** (3+ classes)
- Output layer: **K neurons** (one per class) with **softmax** activation
- Target labels must be **one-hot encoded**

```python
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

### Softmax Function

$$\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$

- All outputs ∈ [0, 1]
- Outputs sum to 1 → interpretable as probabilities

### Worked Calculation

Given 3 classes, prediction = [0.2, 0.3, 0.5], true label = [1, 0, 0]:

$$L = -(1 \cdot \log(0.2) + 0 + 0) = -\log(0.2)$$

Only the **true class probability** contributes to the loss (others multiply by 0).

---

## 9. Sparse Categorical Cross-Entropy

| Aspect | Categorical CE | Sparse Categorical CE |
|---|---|---|
| Label encoding | One-hot (`[1,0,0]`) | Integer (`0, 1, 2`) |
| Math | Identical | Identical |
| Architecture | Same (softmax output) | Same (softmax output) |
| Speed | Slower (compute all logs) | **Faster** (only compute log for true class index) |

```python
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

> Use **sparse** when you have many classes — avoids one-hot encoding overhead.

---

## 10. Quick Reference — Which Loss to Use

| Problem Type | Condition | Loss Function | Output Activation |
|---|---|---|---|
| Regression | No outliers | MSE | Linear |
| Regression | Outliers present | MAE | Linear |
| Regression | Mix of outliers + normal | Huber Loss | Linear |
| Binary Classification | 2 classes | Binary Cross-Entropy | Sigmoid |
| Multi-class Classification | Few classes | Categorical Cross-Entropy | Softmax |
| Multi-class Classification | Many classes | Sparse Categorical Cross-Entropy | Softmax |
