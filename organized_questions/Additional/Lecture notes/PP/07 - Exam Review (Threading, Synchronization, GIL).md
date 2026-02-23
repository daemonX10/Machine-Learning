# Lecture 07 — Exam Review: Threading, Synchronization & GIL

## 1. Key Definitions

| Term | Definition |
|------|-----------|
| **Semaphore** | Synchronization primitive used instead of locks; allows **more than one thread** to access a shared resource. Initialized with an integer (default 1). If set to `n`, then `n` threads can hold the semaphore simultaneously. |
| **Global Interpreter Lock (GIL)** | A lock in CPython that allows **only one thread** to execute Python bytecode at a time. Threads appear concurrent but do not run truly in parallel. Bypassed using **Numba JIT** or **multiprocessing**. |
| **Embarrassingly Parallel** | A problem that can be parallelized with **little or no additional effort**. Sub-tasks are independent and require no communication (e.g., Monte Carlo simulation). |
| **Deadlock** | Two (or more) threads each hold a resource the other needs, and both wait indefinitely. Classic example: *Dining Philosophers* problem. |
| **Race Condition** | Multiple threads modify a shared variable simultaneously, producing **unpredictable results**. Avoided by eliminating shared mutable state or using synchronization primitives. |

---

## 2. Monte Carlo Pi with Quadrants & Threading

### 2.1 Problem Setup

Estimate $\pi$ using Monte Carlo simulation with **4 threads**, each operating in a separate quadrant of the unit circle.

$$\pi \approx 4 \times \frac{\text{inner points}}{\text{total points}}$$

### 2.2 Quadrant Coordinate Ranges

| Quadrant | Thread | $x$ range | $y$ range |
|----------|--------|-----------|-----------|
| Q1 | Thread 0 | $[0, 1]$ | $[0, 1]$ |
| Q2 | Thread 1 | $[-1, 0]$ | $[0, 1]$ |
| Q3 | Thread 2 | $[-1, 0]$ | $[-1, 0]$ |
| Q4 | Thread 3 | $[0, 1]$ | $[-1, 0]$ |

### 2.3 Implementation

```python
import random
from threading import Thread
from numba import jit

class MonteCarloPi:
    def __init__(self, n, limits):
        self.n = n
        self.limits = limits  # [[x_low, x_high], [y_low, y_high]]
        self.i = 0            # inner points count

    def throw_points(self):
        self.i = MonteCarloPi.throw_points_static(
            self.n,
            self.limits[0][0], self.limits[0][1],
            self.limits[1][0], self.limits[1][1]
        )

    @staticmethod
    @jit(nopython=True, nogil=True)
    def throw_points_static(n, x_lower, x_upper, y_lower, y_upper):
        i = 0
        for _ in range(n):
            x = random.uniform(x_lower, x_upper)
            y = random.uniform(y_lower, y_upper)
            if x**2 + y**2 <= 1:
                i += 1
        return i
```

### 2.4 Main Program — Creating Threads

```python
# Define quadrant limits
quadrants = [
    [[0, 1],  [0, 1]],   # Q1
    [[-1, 0], [0, 1]],   # Q2
    [[-1, 0], [-1, 0]],  # Q3
    [[0, 1],  [-1, 0]],  # Q4
]

# Create instances & threads
find_pies = [MonteCarloPi(100, quadrants[i]) for i in range(4)]
threads = [Thread(target=find_pies[i].throw_points) for i in range(4)]

for t in threads: t.start()
for t in threads: t.join()

# Aggregate results (avoids race condition)
inner = sum(fp.i for fp in find_pies)
total = sum(fp.n for fp in find_pies)
pi = 4 * inner / total
```

### 2.5 How Each Problem Is Avoided

| Problem | Solution |
|---------|----------|
| **Race Condition** | Each thread writes to its own instance (`self.i`); results are aggregated **after** `join()`. |
| **Deadlock** | No locks or semaphores are used — no circular waiting possible. |
| **GIL** | `@jit(nopython=True, nogil=True)` from **Numba** releases the GIL during computation. |

### 2.6 Verifying the GIL Problem

Using `htop`, threads under the GIL show only **one thread running (R)** at a time while others are **sleeping (S)**. After applying `@jit(nogil=True)`, **all threads run simultaneously**.

---

## 3. Brute Force Password Cracking

### 3.1 Problem Setup

Given an MD5-encrypted password, find the plaintext by trying all combinations of uppercase letters, lowercase letters, and digits (62 characters total), with password length = 8.

- Total combinations: $62^8 \approx 2.18 \times 10^{14}$

### 3.2 Helper Function (Given)

```python
import hashlib

def text_to_md5(text):
    return hashlib.md5(text.encode()).hexdigest()
```

### 3.3 Random Approach (Simple but Non-Deterministic)

```python
import string, random
from threading import Thread

class BruteForce:
    def __init__(self, password, length=8):
        self.pwd = password
        self.chars = string.ascii_letters + string.digits
        self.found = False
        self.length = length

    def random_combination(self):
        def text_to_md5(text):
            return hashlib.md5(text.encode()).hexdigest()

        while not self.found:
            generated = ''.join(
                self.chars[random.randint(0, len(self.chars) - 1)]
                for _ in range(self.length)
            )
            if text_to_md5(generated) == self.pwd:
                self.found = True
                print(f"Found: {generated}")
                return
```

**Limitation:** Random generation may repeat guesses and is **not guaranteed** to finish in bounded time.

### 3.4 Deterministic Approach — Nested Loops

Generate all index combinations using 8 nested loops:

```python
for a in range(len(self.chars)):
    for b in range(len(self.chars)):
        ...  # 8 levels deep
```

**Problem:** Storing all $62^8$ combinations in memory first is impossible — this is a **memory-bound** bottleneck.

### 3.5 Thread Distribution Strategy

With $N$ threads, assign combinations using strided indexing:

```python
# Thread t processes indices: t, t+N, t+2N, t+3N, ...
for i in range(0 + t, len(char_indices), num_threads):
    ...
```

### 3.6 GIL Limitation

`hashlib` is a third-party C extension — **Numba JIT cannot compile** code using it. Therefore, the GIL **cannot be released** with threads for this problem. Solution: switch to **multiprocessing** (covered in Lecture 08).

---

## 4. Key Takeaways

- **Semaphores** generalize locks to allow `n` concurrent accesses.
- The **GIL** prevents true thread parallelism in CPython; use `numba.jit(nogil=True)` for numerical code or **multiprocessing** for general code.
- **Embarrassingly parallel** problems (like Monte Carlo) are trivially parallelizable.
- Avoid **race conditions** by giving each thread its own data and aggregating afterward.
- **Deadlocks** arise from circular resource dependencies — avoid by not using multiple locks.
- For brute-force problems, **deterministic enumeration** is preferred over random guessing, but requires solving the **memory-bound** issue with generators (Lecture 08).
