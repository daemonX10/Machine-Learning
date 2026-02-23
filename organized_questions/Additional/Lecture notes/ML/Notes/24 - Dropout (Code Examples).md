# Lecture 24 — Dropout (Code Examples & Practical Tips)

## 1. Dropout in Regression

### Setup
- Synthetic non-linear data (noisy sinusoidal relationship)
- Architecture: 4 layers — 1 input, 2 hidden (128 units each, ReLU), 1 output (linear)
- Optimizer: Adam (lr = 0.01), Loss: MSE
- Trained for 500 epochs (intentionally long → to induce overfitting)

### Without Dropout

```python
model = Sequential([
    Dense(128, activation='relu', input_dim=1),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')
])
```

**Result:**
- Training loss: ~0.03 (very low)
- Test loss: ~0.2+ (much higher)
- Regression curve: **spiky**, overfits to every training point
- Large gap between training and validation loss curves → **overfitting**

### With Dropout (p = 0.2)

```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    Dense(128, activation='relu', input_dim=1),
    Dropout(0.2),               # Drop 20% of neurons
    Dense(128, activation='relu'),
    Dropout(0.2),               # Drop 20% of neurons
    Dense(1, activation='linear')
])
```

**Result:**
- Training loss slightly increased
- Test loss **decreased**
- Gap between training/test loss **reduced**
- Regression curve: **smoother**, captures the trend not the noise

---

## 2. Dropout in Classification

### Setup
- Synthetic 2D classification data (approximately linearly separable with some overlap)
- Same architecture with sigmoid output for binary classification
- Loss: binary crossentropy, Epochs: 500

### Without Dropout

- Training accuracy: ~92%
- Validation accuracy: stuck at ~74%
- Decision boundary: **jagged**, capturing small noise clusters
- Loss curves: training loss keeps dropping, validation loss increases → **clear overfitting**

### With Dropout (p = 0.2)

- Overfitting slightly reduced but not enough
- Still captures minor patterns in data

### With Dropout (p = 0.5) — Best Result

```python
model = Sequential([
    Dense(128, activation='relu', input_dim=2),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

**Result:**
- Decision boundary: **smooth, clean line** — proper generalization
- Validation accuracy improved to **~80%+** (was 74% without dropout)
- Training and validation loss curves **converge** (gap minimized)

---

## 3. Effect of Dropout Rate $p$

| $p$ Value | Behavior | Decision Boundary |
|---|---|---|
| **0.0** (no dropout) | Overfitting; captures all noise | Spiky, complex, jagged |
| **0.2** (low) | Mild regularization | Slightly smoother |
| **0.5** (medium) | Good regularization | Smooth, generalizes well |
| **0.7–0.8** (high) | **Underfitting**; too many neurons dropped | Too simple, loses patterns |

> **Sweet spot:** $p \in [0.2, 0.5]$

### Visual Comparison (Regression)

```
p=0.0:  ∿∿∿∿∿∿∿  (spiky, overfits every point)
p=0.2:  ∿∿∿∿      (slightly smoother)  
p=0.5:  ~~~        (smooth curve following trend)
p=0.8:  ___        (too smooth, underfits)
```

---

## 4. Practical Tips

### Tip 1: Adjust $p$ Based on Fit

| Observation | Action |
|---|---|
| Overfitting (train acc >> val acc) | **Increase** $p$ |
| Underfitting (both accuracies low) | **Decrease** $p$ |

### Tip 2: Start with the Last Layer

- Don't apply dropout to all layers immediately
- **Start by adding dropout only before the last hidden layer**
- Check if it helps → then add to more layers
- Many successful architectures only use dropout in the **final layers**

### Tip 3: Recommended Ranges

| Network Type | Recommended $p$ |
|---|---|
| **CNNs** (Convolutional) | ~0.5 (50%) works best |
| **RNNs** (Recurrent) | 0.2 to 0.5 |
| **Dense / Fully Connected** | 0.2 to 0.5 |

> Never apply dropout on the **output layer**.

---

## 5. Drawbacks of Dropout

| Problem | Explanation |
|---|---|
| **Slower convergence** | Network needs more epochs to train (sub-networks are weaker individually) |
| **Loss function changes each batch** | Different architecture each batch → loss surface shifts → gradient calculation becomes harder → debugging is more difficult |

---

## 6. Keras Dropout Implementation Summary

```python
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

model = Sequential([
    Dense(128, activation='relu', input_dim=n_features),
    Dropout(0.3),        # 30% neurons dropped
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # No dropout on output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=100,
          validation_data=(X_test, y_test))
```

**Key:** `Dropout(rate)` where `rate` = fraction of neurons to **drop** (not keep).

---

## 7. Summary

1. Dropout is a **regularization technique** — a form of implicit ensemble learning
2. Low $p$ → overfitting persists; High $p$ → underfitting; **Optimal: 0.2–0.5**
3. Start with dropout on the **last hidden layer**, then expand
4. For CNNs, $p = 0.5$ is commonly used; for RNNs, $p = 0.2\text{–}0.5$
5. Trade-off: slower convergence but better generalization
6. **Never** apply dropout on the output layer
