# Lecture 30: Nesterov Accelerated Gradient (NAG)

## 1. What is NAG?

**Nesterov Accelerated Gradient** is an optimization technique that improves upon SGD with Momentum by **reducing oscillations** around the minimum. It does this by computing the gradient at a **look-ahead position** rather than the current position.

> **One-line summary:** NAG = Momentum + "look before you leap"

---

## 2. Motivation: Momentum's Oscillation Problem

Momentum's biggest weakness is **overshooting**:
- At the minimum, accumulated velocity carries the optimizer past the optimal point
- It oscillates back and forth before settling down
- This wastes many epochs near the minimum

**NAG's fix:** Before taking the full step, peek ahead to see if you should slow down.

---

## 3. Mathematical Formulation

### Momentum (Recap)

$$v_t = \beta \cdot v_{t-1} + \eta \cdot \nabla L(w_t)$$
$$w_{t+1} = w_t - v_t$$

Gradient is computed at the **current position** $w_t$.

### NAG

**Step 1:** Compute the look-ahead position (apply momentum first)

$$w_{\text{look-ahead}} = w_t - \beta \cdot v_{t-1}$$

**Step 2:** Compute gradient at the look-ahead position

$$v_t = \beta \cdot v_{t-1} + \eta \cdot \nabla L(w_{\text{look-ahead}})$$

**Step 3:** Update weights

$$w_{t+1} = w_t - v_t$$

### The Key Difference

| | Momentum | NAG |
|---|----------|-----|
| Gradient computed at | Current position $w_t$ | Look-ahead position $w_{\text{look-ahead}}$ |
| Update strategy | Momentum + gradient applied **simultaneously** | Momentum applied **first**, then gradient at new position |

---

## 4. Geometric Intuition

### Momentum Update (at each step)
```
Current position → [momentum + gradient] → Next position
                   (both applied together at current point)
```

### NAG Update (at each step)
```
Current position → [momentum only] → Look-ahead position
Look-ahead position → [gradient at this point] → Final position
```

### Why This Helps

Consider a point near the minimum where momentum would cause overshooting:

**Momentum:**
1. Standing at point $A$ (near minimum)
2. Compute momentum: pushes **forward** (past minimum)
3. Compute gradient at $A$: also pushes **forward** (still before minimum)
4. Result: big overshoot ↗

**NAG:**
1. Standing at point $A$ (near minimum)
2. Apply momentum: jump to look-ahead point $B$ (past minimum)
3. Compute gradient at $B$: points **backward** (toward minimum)
4. Gradient **partially cancels** the momentum overshoot
5. Result: smaller overshoot ↗ then ↙ (dampened)

---

## 5. Comparison: Momentum vs NAG

### Animation Observations (Linear Regression Example)

| Metric | Batch GD | Momentum ($\beta=0.9$) | NAG |
|--------|----------|------------------------|-----|
| Epochs to converge | ~25-30 | ~50 (with oscillations) | Fewer |
| Overshooting | None | Significant | Reduced |
| Path to minimum | Smooth, direct | Fast approach, then oscillates | Fast approach, controlled landing |

### Tuning the Decay Factor

| $\beta$ | Momentum Behavior | NAG Behavior |
|---------|-------------------|--------------|
| 0.9 | Large oscillations | Moderately reduced oscillations |
| 0.8 | Reduced oscillations | Even smoother convergence |
| 0.5 | Moderate momentum | Smooth, ~8-9 epochs |

---

## 6. Advantages of NAG

1. **Reduces oscillations** near the minimum compared to momentum
2. **Faster convergence** — dampened oscillations mean fewer wasted epochs
3. **Anticipatory correction** — "looks ahead" and corrects before overshooting
4. All benefits of momentum are **retained** (speed, escaping local minima)

---

## 7. Disadvantage of NAG

> **NAG may get stuck in local minima.**

- Momentum's overshooting, while wasteful, can accidentally **push past shallow local minima**
- NAG dampens this behavior → the optimizer may not have enough velocity to escape local minima
- In non-convex landscapes with many local minima, this can be a limitation

```
Momentum: =====> (overshoots local min) =====> Global min ✓
NAG:      =====> ● (stops at local min) ✗
```

---

## 8. Keras Implementation

All three optimizers (SGD, Momentum, NAG) use the **same `SGD` class** with different parameters:

```python
from tensorflow.keras.optimizers import SGD

# Vanilla SGD
optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

# SGD with Momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=False)

# NAG (Nesterov Accelerated Gradient)
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
```

| Parameter | SGD | Momentum | NAG |
|-----------|-----|----------|-----|
| `momentum` | `0.0` | `> 0` (e.g., `0.9`) | `> 0` (e.g., `0.9`) |
| `nesterov` | `False` | `False` | **`True`** |

---

## 9. Summary Table

| Feature | SGD | Momentum | NAG |
|---------|-----|----------|-----|
| Gradient at | Current $w$ | Current $w$ | **Look-ahead** $w$ |
| Speed | Slow | Fast | Fast |
| Oscillation | None | High | **Reduced** |
| Local minima escape | Poor | Good | Moderate |
| Overshooting | No | Yes | **Dampened** |
| Use case | Simple problems | General use | When oscillations are a concern |

---

## 10. Key Takeaways

1. NAG = Momentum with one tweak: **compute gradient at the look-ahead position**
2. This "look before you leap" strategy **dampens oscillations** near minima
3. Trade-off: reduced overshooting but **may get stuck in local minima**
4. Implementation: simply set `nesterov=True` in Keras `SGD` optimizer
5. Both Momentum and NAG still have the limitation of a **single global learning rate** — addressed by AdaGrad, RMSProp, and Adam
