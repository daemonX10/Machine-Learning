# Lecture 15 — Gradient Descent Variants (Batch vs Stochastic vs Mini-Batch)

## 1. Gradient Descent — Recap

> *"Gradient descent is arguably the most popular algorithm to perform optimization and the most common way to optimize neural networks."*

**Core Update Rule:**

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

- Minimizes the loss function by moving in the **opposite direction of the gradient**
- $\eta$ = learning rate = step size toward the minimum

---

## 2. Where GD Fits in Backpropagation

```
for epoch in range(num_epochs):
    for each point in data:          ← inner loop
        1. Forward propagation → ŷ
        2. Calculate loss
        3. ★ Update weights using GD  ← this step has 3 variants
    calc average loss for epoch
```

The **three variants** differ in **how much data** is used to compute the gradient before each weight update.

---

## 3. Batch Gradient Descent (Vanilla GD)

### Algorithm

```
for epoch in range(num_epochs):
    ŷ = forward_prop(entire_dataset, W)   # vectorized, all rows at once
    loss = compute_loss(y, ŷ)              # aggregate loss over all rows
    W = W - η * gradient(loss, W)          # single update per epoch
    print(epoch, loss)
```

### Key Properties

| Property | Value |
|---|---|
| Data used per update | **Entire dataset** |
| Weight updates per epoch | **1** (= num_epochs total) |
| Uses vectorization? | Yes (dot product, no inner loop) |
| Convergence path | **Smooth** — stable loss decrease |

### In Keras

```python
model.fit(X, y, epochs=10, batch_size=len(X_train))
# batch_size = N → Batch Gradient Descent
```

---

## 4. Stochastic Gradient Descent (SGD)

### Algorithm

```
for epoch in range(num_epochs):
    shuffle(data)                          # randomize order each epoch
    for i in range(num_rows):
        x_i = random_point()              # pick 1 random point
        ŷ_i = forward_prop(x_i, W)
        loss_i = compute_loss(y_i, ŷ_i)
        W = W - η * gradient(loss_i, W)   # update after EVERY point
    print(epoch, avg_loss)
```

### Key Properties

| Property | Value |
|---|---|
| Data used per update | **1 data point** |
| Weight updates per epoch | **N** (= num_rows × num_epochs total) |
| Uses vectorization? | No (loop over individual points) |
| Convergence path | **Noisy/spiky** — zigzag toward minimum |

### Shuffling

Data is shuffled before each epoch to **eliminate ordering bias** — ensures every epoch sees points in a different sequence.

### In Keras

```python
model.fit(X, y, epochs=10, batch_size=1)
# batch_size = 1 → Stochastic Gradient Descent
```

---

## 5. Mini-Batch Gradient Descent

### Algorithm

```
for epoch in range(num_epochs):
    shuffle(data)
    for batch in create_batches(data, batch_size=B):
        ŷ = forward_prop(batch, W)        # vectorized within batch
        loss = compute_loss(y_batch, ŷ)
        W = W - η * gradient(loss, W)     # update after each batch
    print(epoch, avg_loss)
```

### Key Properties

| Property | Value |
|---|---|
| Data used per update | **B data points** (batch) |
| Weight updates per epoch | **⌈N/B⌉** batches per epoch |
| Uses vectorization? | Yes (within each batch) |
| Convergence path | **Moderately smooth** — between batch & SGD |

### In Keras

```python
model.fit(X, y, epochs=10, batch_size=32)
# batch_size = B → Mini-Batch Gradient Descent
```

### Uneven Division

If dataset has 400 rows and batch_size = 150:
- Batch 1: 150 rows
- Batch 2: 150 rows
- Batch 3: **100 rows** (remainder)
- 3 weight updates per epoch

---

## 6. Comparison Table

| Aspect | Batch GD | SGD | Mini-Batch GD |
|---|---|---|---|
| **Data per update** | All $N$ rows | 1 row | $B$ rows |
| **Updates per epoch** | 1 | $N$ | $\lceil N/B \rceil$ |
| **Total updates** (E epochs) | $E$ | $E \times N$ | $E \times \lceil N/B \rceil$ |
| **Speed per epoch** | Fastest | Slowest | Middle |
| **Convergence speed** | Slowest | Fastest | Middle |
| **Loss curve** | Smooth | Very noisy | Moderately smooth |
| **Vectorization** | Full | None | Partial |
| **Memory** | High (entire dataset) | Low (1 point) | Moderate |
| **Local minima** | Can get stuck | Can escape (noisy jumps) | Moderate escape ability |
| **Solution quality** | Exact (global min if convex) | Approximate | Good balance |

---

## 7. Convergence Behavior — Visual

### Batch GD — Smooth Descent

```
Loss
  |╲
  | ╲
  |  ╲
  |   ╲___________
  +────────────────→ Epochs
```

### SGD — Noisy/Spiky Descent

```
Loss
  |╲ /╲
  | ╳  ╲ /╲
  |/    ╳  ╲ /╲
  |     ╲  ╳  ╲___
  +────────────────→ Epochs
```

### 2D Loss Surface Trajectory

- **Batch GD:** Straight smooth path toward minimum
- **SGD:** Drunk-walk path — zigzags but eventually reaches minimum vicinity
- **Mini-Batch:** Slightly wobbly but mostly direct path

---

## 8. SGD and Local Minima

### Problem with Batch GD

On non-convex loss surfaces with **multiple local minima**, Batch GD may converge to a **local minimum** and get stuck (smooth descent = no escape mechanism).

### SGD Advantage

SGD's **noisy updates** can cause the point to **jump out** of shallow local minima → reach the **global minimum**:

```
Loss surface:   ╲   ╱╲   ╱╲     ╱
                 ╲_╱  ╲_╱  ╲___╱
                 local  local  global
                  min    min    min
                  
SGD can jump over local minima due to noisy updates!
```

### SGD Disadvantage

Same noise that helps escape local minima means SGD **doesn't converge exactly** — gives an **approximate** solution near the true minimum.

---

## 9. Vectorization — Why Batch GD is Fast Per Epoch

**Batch GD** replaces the inner loop with a **matrix dot product**:

```python
# Loop version (slow):
for i in range(N):
    y_hat[i] = np.dot(X[i], W) + b

# Vectorized version (fast):
y_hat = np.dot(X, W) + b  # single operation, GPU-optimized
```

> **Vectorization** = replacing loops with matrix operations → orders of magnitude faster.

### Downside of Vectorization with Large Data

If $N = 10$ million rows, the entire $X$ matrix must fit in **RAM** — may cause memory overflow.

Mini-batch GD is the practical solution: vectorized computation within manageable batch sizes.

---

## 10. Batch Size — Practical Tips

### Why Powers of 2?

Batch sizes like 32, 64, 128, 256 are common because:
- RAM architecture handles **binary values** efficiently
- Powers of 2 optimize memory alignment and GPU utilization

> Not mandatory — any batch size works, but powers of 2 tend to be slightly faster.

### Common Defaults

| Framework | Default batch_size |
|---|---|
| Keras / TensorFlow | 32 |
| PyTorch | user-defined |

---

## 11. Keras `batch_size` Cheat Sheet

| `batch_size` Value | GD Variant | Updates/Epoch |
|---|---|---|
| `len(X_train)` or `N` | Batch GD | 1 |
| `1` | SGD | N |
| `32` (or any $1 < B < N$) | Mini-Batch GD | ⌈N/B⌉ |

---

## 12. Key Takeaways

1. **Mini-Batch GD** is the default in practice — best of both worlds
2. **Batch GD** is smooth but slow to converge and memory-heavy
3. **SGD** converges fast but noisily; can escape local minima
4. `batch_size` in Keras directly controls which variant is used
5. Use **powers of 2** for batch size for hardware efficiency
6. In deep learning, "SGD" often refers to mini-batch GD (confusing but standard)
