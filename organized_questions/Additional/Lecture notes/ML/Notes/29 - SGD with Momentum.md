# Lecture 29: SGD with Momentum

## 1. Types of Visualization Graphs

Three types of graphs used to visualize loss surfaces:

| Graph Type | Axes | Description |
|------------|------|-------------|
| **1D Loss Curve** | $w$ vs $L$ | Loss as a function of a single parameter |
| **3D Surface Plot** | $w_1$, $w_2$ vs $L$ | Loss surface over two parameters |
| **Contour Plot** | $w_1$ vs $w_2$ (color = $L$) | 2D top-down view of the 3D surface |

### Reading Contour Plots
- **Color encodes height** (the lost 3rd dimension): yellow/orange = high loss, blue/purple = low loss
- **Closely spaced contour lines** = steep slope
- **Widely spaced contour lines** = flat surface (plateau or saddle region)
- Same-colored contour lines = same altitude (loss value)

---

## 2. Convex vs Non-Convex Optimization

| Convex | Non-Convex |
|--------|-----------|
| Single global minimum | Multiple local minima |
| Simple bowl-shaped surface | Complex landscape |
| Easy to optimize | Hard to find global minimum |
| Rare in deep learning | **Common in deep learning** |

### Three Problems in Non-Convex Optimization

1. **Local Minima** — getting stuck in a sub-optimal minimum
2. **Saddle Points** — flat regions where gradient ≈ 0, training stalls
3. **High Curvature** — sharp bends that are difficult to navigate

---

## 3. Vanilla Gradient Descent Behavior

| Algorithm | Behavior | Convergence |
|-----------|----------|-------------|
| Batch GD | Smooth path to minimum | Slow, constant speed |
| Stochastic GD | Noisy/erratic path | Fast updates, high variance |
| Mini-batch GD | Moderately noisy | Balance of speed and stability |

---

## 4. What is Momentum?

### Intuition — Car Analogy
Imagine driving to an unknown destination, asking locals for directions:
- If **all 4 people** point the same way → you speed up confidently
- If **2 say forward, 2 say backward** → you go forward slowly

> **Momentum principle:** If past gradients consistently point in the same direction, **accelerate** in that direction.

### Physics Analogy
- A ball rolling downhill **gains momentum** — it accelerates as it descends
- Even at flat regions or small bumps, the accumulated momentum carries it forward

---

## 5. Mathematical Formulation

### Vanilla SGD (no momentum)

$$w_{t+1} = w_t - \eta \cdot \nabla L(w_t)$$

### SGD with Momentum

**Step 1:** Compute velocity (exponentially weighted average of gradients)

$$v_t = \beta \cdot v_{t-1} + \eta \cdot \nabla L(w_t)$$

**Step 2:** Update weights

$$w_{t+1} = w_t - v_t$$

Where:
- $v_t$ = velocity at time $t$ (accumulates gradient history)
- $\beta$ = momentum coefficient / decay factor $(0 < \beta < 1)$
- $\eta$ = learning rate
- $\nabla L(w_t)$ = gradient at current position
- $v_0 = 0$ (initial velocity)

### What the Velocity Term Captures

The update is made of **two components**:

$$w_{t+1} = w_t - \underbrace{\beta \cdot v_{t-1}}_{\text{past velocity (momentum)}} - \underbrace{\eta \cdot \nabla L(w_t)}_{\text{current gradient}}$$

- **Past velocity**: pushes in the historically dominant direction
- **Current gradient**: adjusts based on current slope

---

## 6. Role of β (Decay Factor)

| $\beta$ value | Behavior |
|--------------|----------|
| $\beta = 0$ | No momentum → **vanilla SGD** |
| $\beta = 0.5$ | Moderate momentum |
| **$\beta = 0.9$** | **Standard value** — averaging ~10 past velocities |
| $\beta = 1$ | Never decays → **infinite oscillation** (unusable) |

> EWMA averaging window = $\frac{1}{1-\beta}$ past updates
>
> For $\beta = 0.9$: averages over last ~10 velocity updates

### Sweet Spot
Typically $\beta \in [0.5, 0.99]$, with **0.9** being the most common default.

---

## 7. Benefits of Momentum

### Benefit 1: Faster Convergence (Speed)
- In elongated loss surfaces, vanilla GD oscillates vertically while making slow horizontal progress
- Momentum **dampens vertical oscillations** and **amplifies horizontal movement**
- Result: **much faster convergence**

```
Vanilla SGD:     ~~~~~~~~> (zig-zag, slow)
SGD + Momentum:  =======>  (smoother, fast)
```

### Benefit 2: Escaping Local Minima
- Accumulated momentum can carry the optimizer **past shallow local minima**
- Like a ball with enough velocity rolling through a small dip and continuing down

---

## 8. Disadvantage: Overshooting

> **The biggest strength of momentum is also its biggest weakness.**

- When the optimizer reaches the minimum, it has accumulated velocity
- It **overshoots** past the minimum, oscillates back and forth
- Eventually settles down as historical velocities decay, but this wastes epochs

```
Without momentum:  → → → → ● (arrives and stops)
With momentum:     → → → → → ● ← → ● ← ● (oscillates before settling)
```

### Why Overshooting Happens
- Momentum pushes in the historically dominant direction
- At the minimum, gradient reverses direction, but momentum still pushes forward
- Decay factor $\beta < 1$ gradually reduces old velocities → oscillations dampen over time

---

## 9. Visual Summary

### Contour Plot Comparison

| Vanilla SGD | SGD with Momentum |
|-------------|-------------------|
| Many zig-zag steps | Fewer, more direct steps |
| Slow to converge | Fast horizontal progress |
| Stops at first minimum | Can cross local minima |
| No overshooting | May overshoot and oscillate |

---

## 10. Key Takeaways

1. Momentum uses **EWMA of past gradients** to build velocity
2. The **single most important benefit** of momentum is **speed**
3. β = 0 → vanilla SGD; β = 0.9 → standard momentum; β = 1 → unstable
4. Advantages: faster convergence, escapes local minima, handles flat regions
5. Disadvantage: overshooting at the minimum → could be resolved by NAG
6. Momentum is the foundation for more advanced optimizers (NAG, Adam)
