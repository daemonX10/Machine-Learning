# LeNet-5 Architecture

## Overview

- **Created by:** Yann LeCun (often called the "Father of CNNs")
- **Year:** 1998
- **Purpose:** Handwritten digit recognition for the US Postal Service (zip code recognition)
- **Name origin:** "LeNet-**5**" because it has **5 trainable layers**
- **Significance:** One of the **first CNN architectures** ever published — foundational to all modern CNNs

---

## Generic CNN Architecture Pattern

Every CNN follows this general structure:

```
Input Image
    ↓
[ Convolution Layer + Activation ] ──┐
[ Pooling Layer                  ]   │  Repeat N times
    ↓  ←─────────────────────────────┘
Flatten (3D → 1D)
    ↓
Fully Connected Layer(s)
    ↓
Output Layer (sigmoid / softmax)
```

### What Varies Between Architectures

| Hyperparameter | Options |
|---|---|
| Number of Conv + Pooling blocks | 1, 2, 3, ... |
| Number of filters per Conv layer | 6, 16, 32, 64, 128, ... |
| Filter size | 3×3, 5×5, 7×7, ... |
| Stride | 1, 2, ... |
| Padding | Valid (none), Same |
| Pooling type | Max pooling, Average pooling |
| Number of FC layers | 1, 2, 3, ... |
| Neurons per FC layer | 84, 120, 256, 512, ... |
| Activation function | tanh, ReLU, Leaky ReLU, ... |
| Regularization | Dropout, Batch Normalization |

---

## LeNet-5 Architecture (Detailed)

### Layer-by-Layer Breakdown

| # | Layer | Details | Output Shape |
|---|-------|---------|-------------|
| — | **Input** | Grayscale image | $32 \times 32 \times 1$ |
| 1 | **Conv1** | 6 filters, $5 \times 5$, stride 1, no padding | $28 \times 28 \times 6$ |
| — | *Activation* | tanh | $28 \times 28 \times 6$ |
| 1 | **AvgPool1** | $2 \times 2$, stride 2 | $14 \times 14 \times 6$ |
| 2 | **Conv2** | 16 filters, $5 \times 5$, stride 1, no padding | $10 \times 10 \times 16$ |
| — | *Activation* | tanh | $10 \times 10 \times 16$ |
| 2 | **AvgPool2** | $2 \times 2$, stride 2 | $5 \times 5 \times 16$ |
| — | **Flatten** | 3D → 1D | $400$ |
| 3 | **FC1** | 120 neurons | $120$ |
| 4 | **FC2** | 84 neurons | $84$ |
| 5 | **Output** | 10 neurons + Softmax | $10$ |

> **Note:** Conv + Pooling together count as **one layer** in LeNet-5's naming convention.

### Shape Computation

Using the formula: $\text{output} = \frac{n - f + 2p}{s} + 1$

| Step | Calculation | Result |
|------|------------|--------|
| Input | — | $32 \times 32 \times 1$ |
| After Conv1 | $\frac{32 - 5 + 0}{1} + 1 = 28$ | $28 \times 28 \times 6$ |
| After AvgPool1 | $\frac{28 - 2}{2} + 1 = 14$ | $14 \times 14 \times 6$ |
| After Conv2 | $\frac{14 - 5 + 0}{1} + 1 = 10$ | $10 \times 10 \times 16$ |
| After AvgPool2 | $\frac{10 - 2}{2} + 1 = 5$ | $5 \times 5 \times 16$ |
| Flatten | $5 \times 5 \times 16$ | $400$ |
| FC1 | $400 \to 120$ | $120$ |
| FC2 | $120 \to 84$ | $84$ |
| Output | $84 \to 10$ | $10$ |

### Parameter Count

| Layer | Parameters |
|-------|-----------|
| Conv1 | $(5 \times 5 \times 1) \times 6 + 6 = 156$ |
| Conv2 | $(5 \times 5 \times 6) \times 16 + 16 = 2{,}416$ |
| FC1 | $400 \times 120 + 120 = 48{,}120$ |
| FC2 | $120 \times 84 + 84 = 10{,}164$ |
| Output | $84 \times 10 + 10 = 850$ |
| **Total** | **~61,706** |

> Most parameters are in the **fully connected layers**, not the convolution layers.

---

## Architecture Diagram

```
[32×32×1]                          INPUT
    │
    ▼
[28×28×6]      CONV1: 6 filters, 5×5, tanh
    │
    ▼
[14×14×6]      AVGPOOL1: 2×2, stride 2
    │
    ▼
[10×10×16]     CONV2: 16 filters, 5×5, tanh
    │
    ▼
[5×5×16]       AVGPOOL2: 2×2, stride 2
    │
    ▼
[400]          FLATTEN
    │
    ▼
[120]          FC1 (tanh)
    │
    ▼
[84]           FC2 (tanh)
    │
    ▼
[10]           OUTPUT (Softmax)
```

---

## Keras Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

model = Sequential([
    # Layer 1: Conv + AvgPool
    Conv2D(6, kernel_size=(5, 5), strides=1, activation='tanh',
           input_shape=(32, 32, 1), padding='valid'),
    AveragePooling2D(pool_size=(2, 2), strides=2),

    # Layer 2: Conv + AvgPool
    Conv2D(16, kernel_size=(5, 5), strides=1, activation='tanh',
           padding='valid'),
    AveragePooling2D(pool_size=(2, 2), strides=2),

    # Flatten
    Flatten(),

    # Layer 3: FC
    Dense(120, activation='tanh'),

    # Layer 4: FC
    Dense(84, activation='tanh'),

    # Layer 5: Output
    Dense(10, activation='softmax')
])

model.summary()
```

---

## Key Design Choices

| Choice | LeNet-5 (1998) | Modern CNNs |
|--------|---------------|-------------|
| **Activation** | tanh | ReLU / variants |
| **Pooling** | Average Pooling | Max Pooling |
| **Depth** | 5 layers | 16–152+ layers |
| **Filters** | 6 → 16 | 64 → 128 → 256 → 512 |

### Common Pattern (Still Used Today)

As you go **deeper** in the network:
- **Spatial dimensions decrease** ($32 \to 28 \to 14 \to 10 \to 5$)
- **Number of filters increases** ($1 \to 6 \to 16$)

---

## Famous CNN Architectures (Chronological)

| Architecture | Year | Key Innovation |
|---|---|---|
| **LeNet-5** | 1998 | First practical CNN |
| **AlexNet** | 2012 | ReLU, Dropout, GPU training |
| **VGGNet** | 2014 | Very deep (16–19 layers), all 3×3 filters |
| **GoogLeNet/Inception** | 2014 | Inception modules |
| **ResNet** | 2015 | Skip connections, 152 layers |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What** | First practical CNN architecture for digit recognition |
| **Who** | Yann LeCun, 1998 |
| **Input** | $32 \times 32 \times 1$ grayscale |
| **Structure** | 2× (Conv + AvgPool) → Flatten → 2× FC → Softmax |
| **Activation** | tanh (ReLU didn't exist yet) |
| **Parameters** | ~61,706 (mostly in FC layers) |
| **Legacy** | Template for all future CNN architectures |
