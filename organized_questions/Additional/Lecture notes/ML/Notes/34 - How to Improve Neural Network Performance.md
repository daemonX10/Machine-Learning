# How to Improve Neural Network Performance

## Overview

Two main strategies to improve a neural network:
1. **Hyperparameter Tuning** — finding optimal values for configurable settings
2. **Solving Common Problems** — addressing issues that degrade performance

---

## Part 1: Hyperparameter Tuning

### Key Hyperparameters

| Hyperparameter | Description | How to Decide |
|---|---|---|
| Number of Hidden Layers | Depth of the network | Add layers until overfitting begins |
| Neurons per Layer | Width of each layer | Start high, reduce if overfitting |
| Learning Rate | Step size for gradient descent | Use schedulers or tuning |
| Optimizer | Algorithm for weight updates (Adam, SGD, RMSProp, etc.) | Experiment; covered under slow training |
| Batch Size | Samples per gradient update | Trade-off between speed and generalization |
| Activation Function | Non-linearity applied at each neuron | Depends on problem; helps with vanishing gradients |
| Epochs | Number of full passes over data | Use Early Stopping |

---

### 1. Number of Hidden Layers

- A single hidden layer with many neurons *can* approximate any function, but in practice:

> **Multiple layers with fewer neurons per layer >> Single layer with many neurons**

- **Why?** Deep networks use **Representation Learning**:
  - Early layers capture **primitive features** (lines, edges)
  - Middle layers combine primitives into **shapes**
  - Later layers form **complex patterns** (e.g., faces)

- **Rule:** Keep adding hidden layers until **overfitting** starts, then stop.
- **Benefit:** Enables **Transfer Learning** — reuse early layers (primitive feature extractors) from a model trained on one task for a similar task.

#### Transfer Learning Example
1. Train a face detection model on human faces
2. For a new task (e.g., monkey face detection), reuse the early layers and retrain only the later layers

---

### 2. Neurons per Layer

- **Input layer:** Fixed — equals the number of input features
- **Output layer:** Fixed — depends on the task:
  - Regression → 1 neuron
  - Binary classification → 1 neuron
  - Multi-class classification → number of classes
- **Hidden layers:**

#### Pyramid Structure (Older Approach)
Decrease neurons layer by layer:

```
Input → 64 → 32 → 16 → Output
```

**Logic:** Primitive features (many) get combined into fewer complex patterns.

#### Equal Neurons (Modern Finding)
Keep the same number of neurons in all hidden layers:

```
Input → 32 → 32 → 32 → Output
```

Both approaches give **similar results** in practice.

> **Key Rule:** Always use **sufficient** neurons — more is better than too few. Start with more and reduce only if overfitting occurs. If a layer has too few neurons, features are lost irreversibly.

---

### 3. Batch Size

| Approach | Pros | Cons |
|---|---|---|
| **Small batch** (e.g., 32) | Better generalization on test data | Slower training |
| **Large batch** (e.g., GPU max) | Faster training | Less stable; poorer generalization |

#### Warm-Up Learning Rate Strategy (for large batches)

1. Start with a **small learning rate** in early epochs
2. **Gradually increase** the learning rate as training progresses

$$\text{lr}(t) = \text{lr}_{\text{init}} \cdot f(t) \quad \text{where } f(t) \text{ increases over epochs}$$

This gives **fast training + good generalization** with large batches.

**Recommended approach:**
1. Try large batch + warm-up LR first
2. If it doesn't work, fall back to small batch size

---

### 4. Epochs

- **Rule:** Set epochs to a **large number** and use **Early Stopping**.
- Early Stopping monitors validation performance:
  - If no improvement for several consecutive epochs → stop training
  - Implemented as a **Keras Callback**
  - The algorithm decides when to stop — no manual tuning needed

---

## Part 2: Common Problems and Solutions

### Problem → Solution Map

| Problem | Solutions |
|---|---|
| **Vanishing / Exploding Gradients** | Weight Initialization, Activation Functions (ReLU), Batch Normalization, Gradient Clipping |
| **Insufficient Data** | Transfer Learning, Data Augmentation, Semi-supervised / Unsupervised Pretraining |
| **Slow Training** | Optimizers (Adam, RMSProp, etc.), Learning Rate Schedulers |
| **Overfitting** | Regularization (L1/L2), Dropout |

---

### Vanishing & Exploding Gradients

- **Cause:** During backpropagation with sigmoid activation, gradients shrink exponentially in earlier layers → weights stop updating → training stalls.
- **Solutions:**

| Technique | Description |
|---|---|
| **Weight Initialization** | Use smart init (Xavier, He) instead of random/zero init |
| **Activation Functions** | Replace Sigmoid with **ReLU** to prevent gradient shrinkage |
| **Batch Normalization** | Normalize layer inputs to stabilize gradient flow |
| **Gradient Clipping** | Cap gradient values to prevent explosion (specifically for exploding gradients) |

---

### Insufficient Data

| Technique | Description |
|---|---|
| **Transfer Learning** | Reuse pretrained models (e.g., VGG, ResNet) |
| **Data Augmentation** | Generate variations of existing data (flip, rotate, crop) |
| **Semi-supervised / Unsupervised Pretraining** | Leverage unlabeled data for initial learning |

---

### Slow Training

| Technique | Description |
|---|---|
| **Advanced Optimizers** | Adam, RMSProp, Adagrad — converge faster than vanilla SGD |
| **Learning Rate Schedulers** | Dynamically adjust LR across epochs for faster convergence |

---

### Overfitting

| Technique | Description |
|---|---|
| **Regularization** | L1/L2 penalties on weights (same as in traditional ML) |
| **Dropout** | Randomly deactivate neurons during training to prevent co-adaptation |

---

## Summary Roadmap

```
Improve Neural Network
├── Tune Hyperparameters
│   ├── Number of Hidden Layers → add until overfitting
│   ├── Neurons per Layer → start sufficient, reduce if needed
│   ├── Learning Rate → use schedulers
│   ├── Optimizer → experiment (Adam, RMSProp, SGD)
│   ├── Batch Size → small for generalization, large + warm-up for speed
│   ├── Activation Function → ReLU over Sigmoid
│   └── Epochs → large number + Early Stopping
│
└── Solve Problems
    ├── Vanishing/Exploding Gradients → Init, ReLU, BatchNorm, Clipping
    ├── Insufficient Data → Transfer Learning, Augmentation
    ├── Slow Training → Optimizers, LR Schedulers
    └── Overfitting → Regularization, Dropout
```
