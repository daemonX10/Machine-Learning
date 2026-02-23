# Lecture 20: Xavier (Glorot) and He Weight Initialization

## 1. Recap: What NOT to Do

| Strategy | Result |
|----------|--------|
| Zero initialization | No training (weights stuck at 0) |
| Constant non-zero | Symmetry problem → acts as single neuron → linear model |
| Small random values | Vanishing gradient (tanh/sigmoid) or very slow convergence (ReLU) |
| Large random values | Saturation → vanishing gradient (tanh/sigmoid) or exploding gradient (ReLU) |

**conclusion:** We need random values with a **variance that depends on network architecture**.

---

## 2. The Core Intuition

When computing a neuron's weighted sum:

$$z = \sum_{i=1}^{n} w_i x_i + b$$

- If $n$ (number of inputs) is **large** → many terms are added → $z$ can become very large or very small
- **Solution:** Scale each weight inversely with $n$ to keep $z$ in a reasonable range

> The variance of weights should be proportional to $\frac{1}{n}$ where $n$ is the number of inputs (fan-in).

**If fan-in is large** → make each weight smaller (to prevent $z$ from exploding)
**If fan-in is small** → weights can be larger (since fewer terms contribute to $z$)

This ensures activations remain in a **healthy range** across all layers, preventing both vanishing and exploding gradients.

---

## 3. Xavier / Glorot Initialization

> **Use with:** Sigmoid and Tanh activation functions

### Xavier Normal

Draw weights from a normal distribution:

$$W \sim \mathcal{N}\left(0, \quad \frac{1}{n_{\text{in}}}\right)$$

**Implementation:**

$$W = \text{np.random.randn}(n_{\text{in}}, n_{\text{out}}) \times \sqrt{\frac{1}{n_{\text{in}}}}$$

Where $n_{\text{in}}$ = fan-in = number of inputs to the neuron.

#### Alternate Formula (used in some implementations)

$$W \sim \mathcal{N}\left(0, \quad \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

$$W = \text{np.random.randn}(n_{\text{in}}, n_{\text{out}}) \times \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$$

Where $n_{\text{out}}$ = fan-out = number of outputs from the neuron.

### Xavier Uniform

Draw weights from a uniform distribution:

$$W \sim U\left[-\text{limit}, \; +\text{limit}\right]$$

$$\text{limit} = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$$

---

## 4. He (Kaiming) Initialization

> **Use with:** ReLU and its variants (Leaky ReLU, PReLU, ELU)

### He Normal

$$W \sim \mathcal{N}\left(0, \quad \frac{2}{n_{\text{in}}}\right)$$

**Implementation:**

$$W = \text{np.random.randn}(n_{\text{in}}, n_{\text{out}}) \times \sqrt{\frac{2}{n_{\text{in}}}}$$

The only difference from Xavier: **numerator is 2 instead of 1** — compensating for ReLU zeroing out ~50% of values.

### He Uniform

$$W \sim U\left[-\text{limit}, \; +\text{limit}\right]$$

$$\text{limit} = \sqrt{\frac{6}{n_{\text{in}}}}$$

---

## 5. When to Use Which

| Activation Function | Initialization | Formula (Normal) |
|--------------------|----------------|------------------|
| **Sigmoid** | Xavier / Glorot | $\text{std} = \sqrt{\frac{1}{n_{\text{in}}}}$ |
| **Tanh** | Xavier / Glorot | $\text{std} = \sqrt{\frac{1}{n_{\text{in}}}}$ |
| **ReLU** | He / Kaiming | $\text{std} = \sqrt{\frac{2}{n_{\text{in}}}}$ |
| **Leaky ReLU / PReLU / ELU** | He / Kaiming | $\text{std} = \sqrt{\frac{2}{n_{\text{in}}}}$ |

> These are **experimentally proven** heuristics — derived through rigorous experimentation and published as research papers.

---

## 6. Keras Implementation

### Setting Initialization Per Layer

```python
from tensorflow.keras.layers import Dense

# Xavier/Glorot (for tanh)
Dense(128, activation='tanh', kernel_initializer='glorot_normal')
Dense(128, activation='tanh', kernel_initializer='glorot_uniform')

# He (for ReLU)
Dense(128, activation='relu', kernel_initializer='he_normal')
Dense(128, activation='relu', kernel_initializer='he_uniform')
```

### Available Options in Keras

| Keras Name | Initialization | Best With |
|------------|---------------|-----------|
| `glorot_normal` | Xavier Normal | Tanh, Sigmoid |
| `glorot_uniform` | Xavier Uniform (Keras default) | Tanh, Sigmoid |
| `he_normal` | He Normal | ReLU and variants |
| `he_uniform` | He Uniform | ReLU and variants |

> **Keras default** is `glorot_uniform` — you should change it to `he_normal` or `he_uniform` when using ReLU.

### Full Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(10, activation='tanh', kernel_initializer='glorot_normal', input_shape=(2,)),
    Dense(10, activation='tanh', kernel_initializer='glorot_normal'),
    Dense(10, activation='tanh', kernel_initializer='glorot_normal'),
    Dense(10, activation='tanh', kernel_initializer='glorot_normal'),
    Dense(1,  activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_split=0.2)
```

---

## 7. Summary Formulas

### Normal Variants

| Method | Variance | Standard Deviation |
|--------|----------|--------------------|
| **Xavier Normal** | $\frac{1}{n_{\text{in}}}$ | $\sqrt{\frac{1}{n_{\text{in}}}}$ |
| **Xavier Normal (alt)** | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | $\sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$ |
| **He Normal** | $\frac{2}{n_{\text{in}}}$ | $\sqrt{\frac{2}{n_{\text{in}}}}$ |

### Uniform Variants

| Method | Range |
|--------|-------|
| **Xavier Uniform** | $\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \; +\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]$ |
| **He Uniform** | $\left[-\sqrt{\frac{6}{n_{\text{in}}}}, \; +\sqrt{\frac{6}{n_{\text{in}}}}\right]$ |

---

## 8. Key Interview Points

1. **Why not zero initialization?** → No gradient flow with ReLU/tanh; symmetry with sigmoid — all neurons become identical.

2. **Why not constant initialization?** → Symmetry problem — all neurons in a layer learn the same features, effectively reducing to a single neuron.

3. **Why not small random?** → Activations collapse to near-zero → vanishing gradient (tanh) or slow convergence (ReLU).

4. **Why not large random?** → Saturation in tanh/sigmoid → vanishing gradient; large gradients in ReLU → instability.

5. **Xavier vs He?** → Xavier uses $\frac{1}{n}$, He uses $\frac{2}{n}$. The factor of 2 compensates for ReLU zeroing ~50% of activations. Xavier for tanh/sigmoid, He for ReLU.

6. **Normal vs Uniform?** → Both work; choice is empirical. Keras default is `glorot_uniform`.
