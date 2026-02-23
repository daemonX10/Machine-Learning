# Perceptron Training (Perceptron Trick)

## 1. Goal of Training

Find the optimal values of **weights** ($w_0, w_1, w_2$) such that the line:

$$w_0 + w_1 x_1 + w_2 x_2 = 0$$

correctly separates the two classes in the data.

> Here $w_0$ = bias ($b$), $x_0 = 1$ always.

---

## 2. Positive and Negative Regions

Every line $ax + by + c = 0$ divides the plane into two regions:

- **Positive region**: points where $ax + by + c > 0$
- **Negative region**: points where $ax + by + c < 0$

### How to Identify Regions
1. Take any point $(x_0, y_0)$
2. Substitute into the line equation $ax_0 + by_0 + c$
3. If result > 0 → point is in the **positive** region
4. If result < 0 → point is in the **negative** region

---

## 3. Effect of Changing Coefficients

| Change | Effect on the Line |
|--------|-------------------|
| Change $c$ (bias) | Line moves **parallel** (up/down) |
| Change $a$ ($w_1$) | Line **rotates** |
| Change $b$ ($w_2$) | Line **rotates** |
| Change all together | **Combined transformation** (translation + rotation) |

---

## 4. The Perceptron Trick Algorithm

### Intuition
1. Start with a **random line** (random $w_0, w_1, w_2$)
2. Repeat many times (e.g., 1000 epochs):
   - Pick a **random point** from training data
   - Check if the point is **correctly classified** or **misclassified**
   - If **correctly classified** → do nothing
   - If **misclassified** → move the line toward that point by updating coefficients
3. After enough iterations, the line **converges** to a good position

### Two Misclassification Cases

| Case | Description | Action |
|------|-------------|--------|
| **Negative point in positive region** | Data says class 0, but point falls in positive region | **Add** point's coordinates to line coefficients |
| **Positive point in negative region** | Data says class 1, but point falls in negative region | **Subtract** point's coordinates from line coefficients |

### Update Rule (with Learning Rate)

For a misclassified point with coordinates $(x_0, x_1, x_2)$ where $x_0 = 1$:

**If negative point in positive region** (need to push line toward negative side):
$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + \eta \cdot \mathbf{x}$$

**If positive point in negative region** (need to push line toward positive side):
$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta \cdot \mathbf{x}$$

Where $\eta$ is the **learning rate** (typically small, e.g., 0.1, 0.01).

> The learning rate prevents large jumps — ensures small, incremental steps toward convergence.

### Simplified Single Rule

Using label encoding ($y = 1$ for positive, $y = 0$ for negative), everything simplifies to:

$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + \eta \cdot (y_i - \hat{y}_i) \cdot \mathbf{x}_i$$

Where:
- $y_i$ = actual label (0 or 1)
- $\hat{y}_i$ = predicted label (0 or 1)
- When **correct**: $y_i - \hat{y}_i = 0$ → no update
- When **wrong**: $y_i - \hat{y}_i = \pm 1$ → update happens

---

## 5. Complete Algorithm (Pseudocode)

```
Initialize w = [w₀, w₁, w₂] randomly
Set learning_rate = 0.1
Set epochs = 1000

FOR i = 1 to epochs:
    Randomly select a point (xᵢ, yᵢ) from training data
    
    Compute: z = w₀·1 + w₁·x₁ + w₂·x₂
    Predict: ŷ = 1 if z ≥ 0 else 0
    
    Update:
        w₀ = w₀ + η · (yᵢ - ŷᵢ) · 1
        w₁ = w₁ + η · (yᵢ - ŷᵢ) · x₁
        w₂ = w₂ + η · (yᵢ - ŷᵢ) · x₂

RETURN w
```

---

## 6. Python Implementation from Scratch

```python
import numpy as np

def perceptron(X, y, lr=0.1, epochs=1000):
    # X: (n_samples, n_features), y: (n_samples,) with 0/1 labels
    
    # Initialize weights: [bias, w1, w2]
    w = np.ones(3)
    lr = lr
    
    for _ in range(epochs):
        # Random point
        idx = np.random.randint(0, X.shape[0])
        
        # Prepend 1 for bias term: [1, x1, x2]
        x_i = np.insert(X[idx], 0, 1)
        
        # Predict
        z = np.dot(w, x_i)
        y_hat = 1 if z >= 0 else 0
        
        # Update weights
        w = w + lr * (y[idx] - y_hat) * x_i
    
    return w[0], w[1:]  # intercept, coefficients

# Usage
intercept, coef = perceptron(X, y)

# Plot the decision boundary
m = -coef[0] / coef[1]        # slope
c = -intercept / coef[1]       # y-intercept
```

---

## 7. Visualization: How the Line Moves

During training, the animation shows:
1. **Start**: Random line (poor classification)
2. **Each step**: When a misclassified point is selected, the line nudges toward it
3. **Correctly classified points**: Line stays still (no update)
4. **Convergence**: After enough iterations, line stabilizes at a good position

---

## 8. Key Takeaways

1. Training = finding $w$ and $b$ that define the best decision boundary
2. **Perceptron Trick**: randomly pick points, update line if misclassified
3. **Learning rate** ($\eta$) controls step size — prevents overshooting
4. Simplified update rule: $\mathbf{w} \leftarrow \mathbf{w} + \eta(y - \hat{y})\mathbf{x}$
5. Correctly classified points contribute **zero** update
6. The algorithm is a **heuristic** — works well in practice but has limitations (discussed in next lecture)
