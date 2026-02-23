# MNIST Handwritten Digit Classification using ANN

## Problem Statement

- **Type**: Multi-class Classification (10 classes: digits 0–9)
- **Dataset**: MNIST — 70,000 grayscale images of handwritten digits
- **Goal**: Predict which digit (0–9) is in a given image
- **Image Size**: 28 × 28 pixels (784 pixels total)

---

## Dataset Overview

| Split | Images | Shape |
|---|---|---|
| Training | 60,000 | `(60000, 28, 28)` — 3D array |
| Test | 10,000 | `(10000, 28, 28)` — 3D array |

- Each image is a 28×28 NumPy array with pixel values ranging from **0 to 255**
- Labels are integers: 0, 1, 2, ..., 9
- Pre-loaded in Keras: no separate download needed

### Loading the Data

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

### Visualizing an Image

```python
import matplotlib.pyplot as plt
plt.imshow(X_train[0])  # Shows the first training image
```

---

## Data Preprocessing

### Normalize Pixel Values

> **Why normalize?** Neural networks converge faster when input values are in a similar small range. Pixel values (0–255) are too large and varied.

$$X_{normalized} = \frac{X}{255}$$

- Maximum value 255 → becomes 1.0
- Minimum value 0 → stays 0.0
- All pixels now in range $[0, 1]$

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

> **Note**: No train-test split needed — MNIST comes pre-split.

---

## Model Architecture

### The Flatten Problem

- Input data shape: `(28, 28)` — 2D array
- Dense layers expect: `(784,)` — 1D array
- **Solution**: Use `Flatten` layer to convert 2D → 1D

### Architecture

```
Input (28×28) → Flatten (784) → Dense(128, relu) → Dense(10, softmax)
```

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))       # 2D → 1D (no trainable params)
model.add(Dense(128, activation='relu'))        # Hidden layer
model.add(Dense(10, activation='softmax'))      # Output layer (10 classes)
```

### Why 10 Output Nodes?

> In multi-class classification, the output layer has **as many nodes as there are classes**. Each node outputs the probability of that class. The class with the highest probability is the prediction.

### Parameter Calculation

| Layer | Parameters | Calculation |
|---|---|---|
| Flatten | 0 | No training — just reshaping |
| Dense (hidden) | 100,480 | $(784 \times 128) + 128$ |
| Dense (output) | 1,290 | $(128 \times 10) + 10$ |
| **Total** | **101,770** | |

---

## Output Layer Activation: Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

- Outputs a **probability distribution** over all classes
- All outputs sum to 1
- Use softmax whenever output layer has **more than 1 node** for classification

---

## Loss Function: Sparse Categorical Crossentropy

```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

| Loss Function | When to Use |
|---|---|
| `sparse_categorical_crossentropy` | Labels are **integers** (0, 1, 2, ..., 9) |
| `categorical_crossentropy` | Labels are **one-hot encoded** |

> **Sparse** variant avoids the need to one-hot encode labels — simpler and equivalent results.

---

## Training

```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2
)
```

---

## Prediction

### Get Probabilities

```python
y_prob = model.predict(X_test)  # Shape: (10000, 10)
```

Each row contains 10 probabilities (one per digit class).

### Convert to Class Labels

```python
import numpy as np
y_pred = y_prob.argmax(axis=1)  # Index of max probability
```

### Evaluate

```python
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))  # ~97.2%
```

> With basic ANN: **~97%** accuracy. Traditional ML (Random Forest, SVM) typically achieves ~96-97%. CNNs can push this to **99%+**.

### Single Image Prediction

```python
# Predict digit for a single test image
result = model.predict(X_test[0].reshape(1, 28, 28))
predicted_digit = result.argmax(axis=1)[0]
```

---

## Model Improvement

### Improved Architecture (added hidden layer)

```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))     # Additional hidden layer
model.add(Dense(10, activation='softmax'))
```

- Increased epochs to 25
- Tracked accuracy with `metrics=['accuracy']` in compile

### Overfitting Observed

- Training accuracy reached ~100%
- Validation accuracy lagged behind (~97%)
- Gap indicates **overfitting**
- Solutions: regularization, dropout, early stopping

---

## Training Visualization

```python
# Loss Plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

# Accuracy Plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
```

---

## Summary: Multi-Class Classification with ANN

| Aspect | Configuration |
|---|---|
| Output nodes | **Equal to number of classes** (10) |
| Output activation | **Softmax** |
| Loss function | `sparse_categorical_crossentropy` (integer labels) |
| Prediction | `argmax` on probability vector |
| Flatten layer | Required when input is 2D (images) |
| Normalization | Divide by 255 for pixel data |

### Classification Type Comparison

| Type | Output Nodes | Activation | Loss |
|---|---|---|---|
| Binary | 1 | Sigmoid | `binary_crossentropy` |
| Multi-class | N (num classes) | Softmax | `sparse_categorical_crossentropy` |
| Regression | 1 | Linear | `mean_squared_error` |
