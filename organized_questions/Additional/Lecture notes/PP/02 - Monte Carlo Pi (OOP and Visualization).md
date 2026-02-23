# Lecture 02 — Monte Carlo Pi (OOP and Visualization)

## 1. Recap

- Previous lecture: estimated π using Monte Carlo simulation in a flat script
- Python took ~24.78s for 10M points; Fortran took ~0.32s (~76× faster)
- This lecture: **refactor into OOP** and add **visualization**

---

## 2. Object-Oriented Refactoring

### Project Structure

```
project/
├── bc.py       # Classes: TicToc, FindPi
└── main.py     # Entry point
```

### The `if __name__ == "__main__"` Pattern

```python
if __name__ == "__main__":
    print("Running directly")
```

- When a file is run directly: `__name__` is `"__main__"` → code block executes
- When a file is imported: `__name__` is the module name → code block is **skipped**
- This allows a file to serve as both a module and a standalone script

### TicToc Class — Timing Utility

```python
import time

class TicToc:
    def __init__(self):
        self.t1 = 0
        self.t2 = 0

    def tick(self):
        self.t1 = time.time()

    def tock(self):
        self.t2 = time.time()
        return self.t2 - self.t1
```

| Method | Purpose |
|--------|---------|
| `__init__` | Constructor — initializes `t1`, `t2` to 0 |
| `tick()` | Records start time |
| `tock()` | Records end time, returns elapsed seconds |

- `self` in Python is equivalent to `this` in Java/C++ — provides access to instance attributes

### FindPi Class — Monte Carlo Simulation

```python
import random

class FindPi:
    def __init__(self):
        self.n = 0  # total points
        self.i = 0  # inner points (inside circle)

    def draw_points(self, n):
        for _ in range(n):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            r = x**2 + y**2
            self.n += 1
            if r <= 1:
                self.i += 1

    def value_of_pi(self):
        return 4 * self.i / self.n
```

### Main Script (main.py)

```python
from bc import TicToc, FindPi

if __name__ == "__main__":
    tt = TicToc()
    finding_pi = FindPi()

    tt.tick()
    finding_pi.draw_points(10_000_000)
    pi = finding_pi.value_of_pi()

    print("pi = %.8f  i = %d  n = %d  time = %.5f seconds"
          % (pi, finding_pi.i, finding_pi.n, tt.tock()))
```

---

## 3. Visualization with Matplotlib

### Scatter Plot of Random Points

```python
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# Generate random points
x, y = [], []
for _ in range(5000):
    x.append(random.uniform(-1, 1))
    y.append(random.uniform(-1, 1))

# Create figure and axes
fig, ax = plt.subplots()

# Plot points
ax.scatter(x, y, s=1, color='black')

# Add square (starting at (-1,-1), width=2, height=2)
ax.add_patch(Rectangle((-1, -1), 2, 2, alpha=0.2, color='yellow'))

# Add circle (center=(0,0), radius=1)
ax.add_patch(Circle((0, 0), 1, alpha=0.2, color='red'))

# Formatting
ax.axis('off')
ax.set_aspect('equal')
plt.show()
```

### Key Matplotlib Concepts

| Feature | Code | Purpose |
|---------|------|---------|
| Scatter plot | `ax.scatter(x, y, s=1)` | `s` controls marker size (in points²) |
| Rectangle patch | `Rectangle((x, y), w, h, alpha=0.2)` | Semi-transparent rectangle overlay |
| Circle patch | `Circle((cx, cy), r, alpha=0.2)` | Semi-transparent circle overlay |
| Equal aspect ratio | `ax.set_aspect('equal')` | Ensures the circle looks circular |
| Hide axes | `ax.axis('off')` | Removes axis lines and labels |

---

## 4. Introduction to Threading

### Creating and Starting a Thread

```python
from threading import Thread

thread = Thread(target=finding_pi.draw_points, args=(n,))
thread.start()
```

- `target` — the function the thread will execute
- `args` — tuple of arguments to pass (note the trailing comma for single-argument tuples)
- `thread.start()` — begins execution in a **separate thread**

### Synchronization Problem

When using threads, the **main thread does not wait** for the sub-thread to finish:

```python
thread.start()
# This executes immediately, before thread finishes!
pi = finding_pi.value_of_pi()  # ← only ~4000 points processed
```

### Observing Progress with a Loop

```python
import time

thread.start()
for _ in range(25):
    pi = finding_pi.value_of_pi()
    print("pi = %.8f  i = %d  n = %d" % (pi, finding_pi.i, finding_pi.n))
    time.sleep(1)
```

- The values of `i` and `n` increase each second as the thread progresses
- After ~25 seconds, the thread completes and final values stabilize

> **Key Insight**: The main process and sub-thread run concurrently, but without explicit synchronization, the main thread may read incomplete results.

---

## 5. Key Takeaways

1. **OOP structure**: Separating timing (`TicToc`) and simulation (`FindPi`) into classes promotes reusability and prepares the code for multi-threading
2. **`if __name__ == "__main__"`**: Essential pattern to prevent code execution during imports
3. **Matplotlib visualization** confirms the Monte Carlo setup — points uniformly distributed in the square, with ~78.5% falling inside the circle
4. **Threading** allows computation to run in the background, but requires **synchronization** to ensure results are complete before reading them
5. **Next lecture**: Thread synchronization with `join()`, multiple threads, race conditions
