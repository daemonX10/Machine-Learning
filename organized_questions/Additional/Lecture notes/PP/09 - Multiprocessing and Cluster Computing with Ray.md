# Lecture 09 — Multiprocessing & Cluster Computing with Ray

## 1. Multiprocessing: Monte Carlo Pi Revisited

### 1.1 Threads vs Processes — Memory Model

| Feature | Threads | Processes |
|---------|---------|-----------|
| Memory | **Shared** — all threads access the same RAM | **Separate** — each process has its own memory space |
| GIL | Limits true parallelism | No GIL issue — each process has its own interpreter |
| Communication | Direct variable access | Requires **shared variables** (`Array`, `Value`) |
| Overhead | Low (lightweight) | Higher (process creation cost) |

### 1.2 TicToc Timer Class

```python
import time

class TicToc:
    def __init__(self):
        self.t1 = 0
        self.t2 = 0

    def tic(self):
        self.t1 = time.time()

    def toc(self):
        self.t2 = time.time()
        return self.t2 - self.t1
```

### 1.3 FindPi Class for Multiprocessing

```python
import random

class FindPi:
    def __init__(self, n=0):
        self.n = n  # total points
        self.i = 0  # inner points

    def throw_points(self, nn, p, all_i, all_n):
        """
        nn    — number of points to throw
        p     — process index (for writing to shared arrays)
        all_i — shared array for inner point counts
        all_n — shared array for total point counts
        """
        for _ in range(nn):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            self.n += 1
            if x**2 + y**2 <= 1:
                self.i += 1
        # Write results to shared arrays
        all_i[p] = self.i
        all_n[p] = self.n
```

### 1.4 Shared Arrays Between Processes

```python
from multiprocessing import Process, Array

# Create shared arrays (type 'i' = integer, 'd' = double)
shared_i = Array('i', [0] * num_processors)  # inner points per process
shared_n = Array('i', [0] * num_processors)  # total points per process
```

Each process writes to its **own index** `p` — no race conditions.

### 1.5 Main Program with Performance Comparison

```python
import os
from multiprocessing import Process, Array

if __name__ == '__main__':
    total_points = 40_320_000  # divisible by 1 through 8

    for num_p in range(1, os.cpu_count() + 1):
        timer = TicToc()
        timer.tic()

        n = int(total_points / num_p)  # points per process
        find_pies = []
        processes = []

        shared_i = Array('i', [0] * num_p)
        shared_n = Array('i', [0] * num_p)

        for i in range(num_p):
            find_pies.append(FindPi())
            processes.append(
                Process(
                    target=find_pies[i].throw_points,
                    args=(n, i, shared_i, shared_n)
                )
            )

        for p in processes: p.start()
        for p in processes: p.join()

        # Aggregate results
        inner = sum(shared_i)
        total = sum(shared_n)
        pi = 4 * inner / total

        print(f"P={num_p}, pi={pi:.8f}, time={timer.toc():.8f}s")
```

### 1.6 Choosing a Divisible Number of Points

The total points must be divisible by 1, 2, 3, ..., 8 for fair comparison:

$$N = 1 \times 2 \times 3 \times 4 \times 5 \times 6 \times 7 \times 8 = 40{,}320$$

Use multiples like 4,032,000 or 40,320,000 by appending zeros.

### 1.7 Performance Observations

| CPUs | Expected Speedup | Actual Behavior |
|------|------------------|-----------------|
| 1 | Baseline | Baseline |
| 2 | ~2× | ~1.2–1.5× (overhead from process creation) |
| 4 | ~4× | ~2–3× |
| 8 | ~8× | Diminishing returns; may get **worse** beyond physical core count |

**Key insight:** Adding more CPUs does **not** guarantee proportional speedup due to:
- Process creation overhead
- OS scheduling overhead
- Shared resource contention
- Hyperthreading limitations (logical vs physical cores)

---

## 2. Cluster Computing with Ray

### 2.1 What is Ray?

**Ray** is a general-purpose distributed computing framework for Python. It enables:
- Running code across **multiple machines** (cluster computing)
- **Auto-scaling** — automatically uses available CPUs across the cluster
- Simple API — uses decorators to mark functions for remote execution

### 2.2 Setting Up a Cluster

#### Step 1: Start the Head Node

```bash
ray start --head --min-worker-port 12000
```

This outputs the cluster address (e.g., `10.42.0.94:6379`) and a password.

#### Step 2: Connect Worker Nodes

On each additional machine:

```bash
ray start --address 10.42.0.94:6379 --redis-password <password>
```

#### Step 3: Check Cluster Status

```bash
ray status
```

Shows total nodes, CPUs, and memory across the cluster.

### 2.3 Ray API — Key Functions

| Function / Decorator | Purpose |
|---------------------|---------|
| `ray.init(address='auto')` | Connect to an existing cluster |
| `@ray.remote` | Mark a function for distributed execution |
| `func.remote(args)` | Call a remote function (returns a future) |
| `ray.get(futures)` | Block and retrieve results from remote calls |
| `ray.nodes()` | List all nodes in the cluster |
| `ray.cluster_resources()` | Get cluster resources (CPUs, memory, etc.) |

### 2.4 Implementation with Ray

```python
import ray
import random
import os

def init_cluster():
    """Initialize connection to Ray cluster."""
    ray.init(address='auto')
    num_cpus = int(ray.cluster_resources()['CPU'])
    num_nodes = len(ray.nodes())
    print(f"Cluster: {num_nodes} nodes, {num_cpus} CPUs")
    return num_cpus

@ray.remote
def throw_points(n):
    """Run on any node in the cluster."""
    print(f"PID {os.getpid()} on {os.uname().nodename} started")
    i = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            i += 1
    print(f"PID {os.getpid()} on {os.uname().nodename} finished")
    return i

def main():
    import sys
    sys.stderr = open(os.devnull, 'w')  # suppress warnings

    num_cpus = init_cluster()
    n = 10_000_000  # points per CPU

    # Launch remote tasks — one per CPU in the cluster
    futures = [throw_points.remote(n) for _ in range(num_cpus)]

    # Collect results
    inner = ray.get(futures)  # blocks until all tasks complete

    pi = 4 * sum(inner) / (num_cpus * n)
    print(f"Estimated pi = {pi:.8f}")

if __name__ == '__main__':
    main()
```

### 2.5 How Ray Distributes Work

```
                     ┌─────────────┐
                     │  Head Node  │
                     │  (Desktop)  │
                     │   8 CPUs    │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
      ┌───────┴──────┐ ┌───┴────┐ ┌──────┴───────┐
      │  Worker 1    │ │ Worker │ │  Worker 2    │
      │  (Laptop 1)  │ │ (Head) │ │  (Laptop 2)  │
      │   8 CPUs     │ │ 8 CPUs │ │   8 CPUs     │
      └──────────────┘ └────────┘ └──────────────┘
                    Total: 24 CPUs
```

- Each `throw_points.remote(n)` call is scheduled on any available CPU across the cluster
- Ray handles serialization, communication, and result collection
- The slowest node determines total completion time

### 2.6 Key Differences: Multiprocessing vs Ray

| Feature | `multiprocessing` | `ray` |
|---------|-------------------|-------|
| **Scope** | Single machine | Multiple machines (cluster) |
| **Scaling** | Limited by local CPUs | Auto-scales across nodes |
| **Communication** | Shared memory (`Array`, `Value`) | Automatic serialization |
| **API** | `Process(target=..., args=...)` | `@ray.remote` + `.remote()` |
| **Result collection** | Manual (shared arrays + `join`) | `ray.get(futures)` |
| **Setup** | None | `ray start --head` on head + `ray start --address` on workers |

### 2.7 Auto-Scaling Behavior

Ray **automatically distributes tasks** across available resources:
- If you request 8 tasks and have 8 local CPUs → runs locally
- If you request 24 tasks and have 24 CPUs across 3 machines → distributes across all nodes
- No code changes needed — just add more nodes to the cluster

---

## 3. Practical Considerations

### 3.1 Network Quality

- Wireless connections are **not ideal** for cluster computing
- Use wired Ethernet for production clusters
- Communication latency affects overall performance

### 3.2 Monitoring with `htop`

```bash
htop                    # on each machine
# Press F4 to filter by process name (e.g., "ray" or "python")
```

Observe:
- **CPU column**: all processes should hit ~100%
- **Status column**: R = running, S = sleeping
- The slowest machine is the bottleneck

### 3.3 Process vs Thread Status in `htop`

| With Threads (GIL) | With Processes / Ray |
|--------------------|---------------------|
| 1 thread **R**unning, rest **S**leeping | Multiple processes **R**unning simultaneously |
| Single core utilized | All cores utilized |

---

## 4. Key Takeaways

- **Multiprocessing** on a single machine uses `Process` and `Array`/`Value` for inter-process communication.
- Adding CPUs yields **diminishing returns** — doubling CPUs does not halve execution time (Amdahl's Law).
- **Ray** extends parallelism across multiple machines with minimal code changes (`@ray.remote` decorator).
- `ray.get()` collects results from all remote tasks; it blocks until completion.
- Auto-scaling: Ray schedules tasks across all available CPUs in the cluster without manual distribution.
- Use a total number of points **divisible by all possible CPU counts** for fair performance comparisons.
- The **slowest node** in a heterogeneous cluster determines overall completion time.
