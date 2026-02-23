# Lecture 01 — Monte Carlo Pi Estimation (Introduction)

## 1. Course Overview

- **Course**: Parallel Programming / High Performance Computing (HPC)
- **Goal**: Learn techniques to improve the performance of algorithms and code
- **Language**: Python — easy to learn but inherently slow (interpreted), making it an ideal candidate for demonstrating performance optimization
- **Tool options**: Kaggle notebooks, Jupyter Notebook (Anaconda), PyCharm IDE

---

## 2. Problem Statement — Estimating π

Estimate the value of $\pi$ using **Monte Carlo simulation** by randomly sampling points in a square that encloses a unit circle.

---

## 3. Geometric Setup

Consider a **circle of radius 1** centered at the origin, inscribed in a **square of side 2** (from $-1$ to $+1$ on both axes).

| Shape  | Formula          | Value |
|--------|------------------|-------|
| Square | $A_s = (2)^2$    | $4$   |
| Circle | $A_c = \pi r^2$  | $\pi$ |

---

## 4. The Monte Carlo Method

### Core Idea

1. Generate $N$ random points $(x, y)$ where $x, y \in [-1, +1]$
2. For each point, compute the squared distance to the origin:

$$r^2 = x^2 + y^2$$

3. If $r^2 \leq 1$, the point is **inside** the circle — count it as an inner point $I$
4. The ratio of inner points to total points approximates the ratio of areas:

$$\frac{I}{N} \approx \frac{A_c}{A_s} = \frac{\pi}{4}$$

5. Therefore:

$$\boxed{\pi \approx 4 \cdot \frac{I}{N}}$$

> **Note**: We use $r^2 \leq 1$ instead of $r \leq 1$ to avoid the unnecessary square root operation — the comparison result is identical near the boundary of 1.

### Why "Monte Carlo"?

The method uses **random sampling** (pseudorandom numbers) to simulate the coordinates of points and estimate a deterministic value — a hallmark of Monte Carlo methods.

---

## 5. Python Implementation

### Generating Random Numbers

```python
import random

x = random.uniform(-1, 1)  # random float in [-1, 1]
y = random.uniform(-1, 1)
```

- Python uses **pseudorandom** number generators (deterministic but statistically random)
- `random.uniform(a, b)` returns a float in the range $[a, b]$

### Checking if a Point is Inside the Circle

```python
r_squared = x**2 + y**2
if r_squared <= 1:
    print("Inside the circle")
else:
    print("Outside the circle")
```

- The `**` operator is Python's exponentiation operator

### Full Simulation

```python
import random
import time

n = 10_000_000  # total number of points
inner = 0        # count of points inside the circle

start_time = time.time()

for _ in range(n):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if x**2 + y**2 <= 1:
        inner += 1

end_time = time.time()

pi = 4 * inner / n
print("pi = %.5f  i = %d  n = %d" % (pi, inner, n))
print("Time: %.5f seconds" % (end_time - start_time))
```

### Python Syntax Notes

| Concept | Detail |
|---------|--------|
| Loop without index | `for _ in range(n):` — underscore discards the loop variable |
| Indentation | Python uses indentation (tabs/spaces) instead of braces for code blocks |
| String formatting | `"%.5f" % value` — C-style format strings with `%d` (int) and `%f` (float) |

---

## 6. Measuring Execution Time

```python
import time

start = time.time()  # Unix timestamp (seconds since Jan 1, 1970)
# ... computation ...
end = time.time()
elapsed = end - start
```

---

## 7. Python vs. Fortran Performance Comparison

| Language | Time (10M points) | Speedup |
|----------|-------------------|---------|
| Python   | ~24.78 seconds    | 1×      |
| Fortran  | ~0.32 seconds     | **~76×** |

- Python is an **interpreted** language → extremely slow for computation
- Fortran is a **compiled** language → near-optimal machine code
- **Course objective**: Bridge this performance gap using parallel programming techniques

---

## 8. Key Takeaways

1. **Monte Carlo simulation** estimates π by sampling random points and comparing the ratio of points inside a circle vs. total points
2. The formula $\pi \approx 4 \cdot I / N$ converges to π as $N \to \infty$
3. No square root is needed — comparing $r^2 \leq 1$ is equivalent and more efficient
4. Python is ~76× slower than Fortran for this computation — motivating the need for **parallel programming** and **performance optimization**
5. `time.time()` provides wall-clock timing using Unix timestamps
