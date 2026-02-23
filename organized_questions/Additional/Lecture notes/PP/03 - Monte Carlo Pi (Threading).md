# Lecture 03 — Monte Carlo Pi (Threading)

## 1. Recap

- Previous lecture built OOP classes (`TicToc`, `FindPi`) and introduced basic `threading.Thread`
- Problem: main thread doesn't wait for sub-thread to finish
- This lecture: **thread synchronization**, **race conditions**, **GIL**, and **Numba JIT**

---

## 2. Thread Synchronization with `join()`

### The Problem

Without synchronization, the main thread reads results immediately — before the worker thread finishes:

```python
thread = Thread(target=finding_pi.draw_points, args=(n,))
thread.start()
# Main thread continues immediately!
pi = finding_pi.value_of_pi()  # ← incomplete result (~4000 of 10M points)
```

### The Solution: `thread.join()`

```python
thread = Thread(target=finding_pi.draw_points, args=(n,))
thread.start()
thread.join()  # ← blocks until thread completes
pi = finding_pi.value_of_pi()  # ← correct result (all 10M points)
```

- `join()` makes the **main thread wait** until the sub-thread finishes
- Result: correct pi value using all 10M points (~23 seconds)

---

## 3. Multiple Threads

### Scaling to All CPU Cores

```python
import os
from threading import Thread
from bc import FindPi

n = 10_000_000
num_cpus = os.cpu_count()  # e.g., 8

# Create instances and threads
find_pis = [FindPi() for _ in range(num_cpus)]
threads = []

for i in range(num_cpus):
    t = Thread(target=find_pis[i].draw_points, args=(n,))
    threads.append(t)
    print("Started thread number %d" % i)

# Start all threads
for t in threads:
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

# Aggregate results
inner = sum(fp.i for fp in find_pis)
total = sum(fp.n for fp in find_pis)
pi = 4 * inner / total

print("pi = %.8f  inner = %d  total = %d" % (pi, inner, total))
```

> **Important**: Three separate loops — create, start, join — ensure all threads launch before any blocking occurs.

---

## 4. Race Condition

### What Happens with a Shared Instance

If all threads share **one** `FindPi` instance, the counter variable `i` gets corrupted:

| Expected total | Actual total | Lost points |
|---------------|--------------|-------------|
| 80,000,000    | ~53,000,000  | ~27,000,000 |

### Why It Happens

Incrementing `self.i += 1` is **not atomic** — it consists of three steps:

```
1. READ   → take current value of i (e.g., 3)
2. ADD    → compute i + 1 (e.g., 4)
3. WRITE  → assign new value back to i
```

When two threads interleave these steps:

```
Thread 0: READ  i → 3
Thread 1: READ  i → 3     (reads same stale value!)
Thread 0: ADD   → 4
Thread 1: ADD   → 4       (both compute 4)
Thread 0: WRITE i ← 4
Thread 1: WRITE i ← 4     (overwrites with same value!)
```

**Result**: Two increments but `i` only increased by 1. This is a **race condition**.

### Solution: Separate Instances

Each thread gets its **own `FindPi` instance** with private `i` and `n` variables. Results are aggregated after all threads finish:

```python
inner = 0
total = 0
for fp in find_pis:
    inner += fp.i
    total += fp.n
pi = 4 * inner / total
```

---

## 5. Global Interpreter Lock (GIL)

### The Problem

Even with correct multi-threading (no race condition), Python threads provide **no speedup**:

| Configuration | Time (10M pts each) |
|--------------|---------------------|
| 1 thread     | ~10 seconds         |
| 8 threads    | ~80 seconds         |

### What is the GIL?

The **Global Interpreter Lock** is a mutex in CPython that allows **only one thread to execute Python bytecode at a time**.

- Threads are scheduled by the OS across CPU cores
- But due to GIL, only one thread is truly *running* at any moment — others are *sleeping*
- Observable in `htop`: thread status column shows **R** (Running) for only 1 thread, **S** (Sleeping) for all others
- CPU utilization stays at ~12.5% on 8 cores (only 1 core effectively used)

### Consequence

Threading in Python **does not achieve true parallelism** for CPU-bound tasks. The total work is divided but executed serially under the GIL.

---

## 6. Solving GIL with Numba JIT

### What is Numba?

**Numba** is a JIT (Just-In-Time) compiler that translates Python code into optimized machine code (via LLVM), achieving speeds comparable to C/Fortran.

### Step 1: Extract the Atomic Computation

The core computation (the loop) must be a **static method** — no access to `self` (class state):

```python
from numba import jit

class FindPi:
    def __init__(self):
        self.n = 0
        self.i = 0

    @staticmethod
    @jit(nopython=True, nogil=True)
    def draw_points_static(n):
        i = 0
        total = 0
        for _ in range(n):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x**2 + y**2 <= 1:
                i += 1
            total += 1
        return i, total

    def draw_points(self, n):
        self.i, self.n = FindPi.draw_points_static(n)

    def value_of_pi(self):
        return 4 * self.i / self.n
```

### Step 2: Key Decorator Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `nopython` | `True` | Forces full compilation — no Python fallback. Errors if any Python objects are used |
| `nogil` | `True` | **Releases the GIL** during execution — enables true parallel execution |

### Step 3: The Interface Pattern

The `draw_points()` method wraps the static method to maintain the same external API. The static method operates on local variables only (no `self`), returns values, and the wrapper assigns them back.

### Results

| Approach | Time (8 threads × 10M points) | Speedup |
|----------|-------------------------------|---------|
| Pure Python threads | ~100+ seconds | 1× |
| Numba JIT + nogil | ~1 second | **~100×** |

### Verification

- `htop` shows all threads with status **R** (Running) simultaneously
- CPU utilization at **100%** across all cores
- Process reports ~600% CPU usage (8 logical processors fully utilized)
- First run includes JIT compilation overhead; subsequent runs are faster

### Gotcha: Variable Name Collision

Using `i` as both the inner-point counter and the loop variable causes silent bugs:

```python
# Bug: loop variable `i` overwrites counter `i`
for i in range(n):
    if ...:
        i += 1  # modifies loop variable, not counter!
```

**Fix**: Use `_` for the loop variable when the index isn't needed:

```python
for _ in range(n):
    if ...:
        i += 1  # correctly increments counter
```

---

## 7. Summary: Performance Comparison

| Method | Threads | GIL | Time | Relative |
|--------|---------|-----|------|----------|
| Pure Python (single) | 1 | N/A | ~24s | 1× |
| Pure Python (8 threads) | 8 | Locked | ~100+s | 0.24× (worse!) |
| Numba JIT + nogil | 8 | Released | ~1s | **~100×** |
| Fortran (compiled) | 1 | N/A | ~0.32s | ~75× |

---

## 8. Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **`thread.join()`** | Blocks the calling thread until the target thread finishes |
| **Race Condition** | Multiple threads modify shared state → corrupted results |
| **GIL** | CPython mutex — only one thread executes Python bytecode at a time |
| **Numba `@jit`** | JIT compiler that converts Python to optimized machine code |
| **`nopython=True`** | Ensures full compilation with no Python interpreter fallback |
| **`nogil=True`** | Releases the GIL, enabling true multi-core parallelism |
| **`@staticmethod`** | Decouples the method from class state — required for Numba JIT |
| **Atomic operation** | The smallest indivisible unit of computation (not interruptible) |

---

## 9. Homework

**Find the value of $e$ (Euler's number ≈ 2.718) using Monte Carlo simulation:**

1. Generate random real numbers $x_1, x_2, x_3, \ldots \in [0, 1]$
2. Accumulate their sum: $S = x_1 + x_2 + x_3 + \ldots$
3. Stop when $S > 1$, and record $n$ = number of terms added
4. Repeat this process many times
5. The **average value of $n$** converges to $e$

$$\boxed{e \approx \frac{1}{M} \sum_{k=1}^{M} n_k}$$

where $M$ is the number of trials and $n_k$ is the number of terms needed in trial $k$.
