# Lecture 06: Load Balancing with Conditions

## 1. Generalizing the Dining Philosophers Problem

The Dining Philosophers solution can be generalized to **any** parallel programming problem by abstracting its components:

| Dining Philosophers | General Problem |
|---|---|
| Philosophers | **Users / Processes** |
| Chopsticks | **Resources** |
| Meals | **Jobs** |

### General Setup

- $n$ **users** (e.g., 10 processes)
- $r$ **resources** (e.g., 3 shared resources)
- $m$ **jobs** per user (e.g., 100 tasks to complete)

Each user must acquire a resource to complete one unit of work, then release it.

---

## 2. Initial Approach: Semaphores with Random Resource Selection

### Implementation

```python
import time
import random
from threading import Thread, Semaphore

class JobsAndUsers:
    def __init__(self, num_users, num_resources, job_size):
        self.n = num_users
        self.r = num_resources
        self.m = job_size
        self.resources = [Semaphore(1) for _ in range(num_resources)]
        self.jobs = [job_size for _ in range(num_users)]

    def user(self, i):
        while self.jobs[i] > 0:
            time.sleep(random.random() + 3)  # Sleep 3-4 seconds

            # Pick a RANDOM resource
            r = random.randint(0, self.r - 1)

            if self.resources[r].acquire(timeout=10):
                time.sleep(random.random() + 5)  # Do work (5-6 sec)
                self.jobs[i] -= 1
                self.resources[r].release()

if __name__ == "__main__":
    n, r, m = 10, 3, 100
    ju = JobsAndUsers(n, r, m)
    threads = [Thread(target=ju.user, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
```

### Problems with This Approach

1. **Unbalanced queues**: Random resource selection causes some resources to have 8-9 users waiting while others are completely free
2. **Fixed timeouts**: The `timeout=10` is arbitrary — too short means missed opportunities, too long means wasted waiting
3. **Timeout failures**: Users that can't acquire a resource within the timeout get nothing, even when other resources are free

---

## 3. From Semaphores to Conditions

### Why Conditions?

| Semaphore | Condition |
|---|---|
| Fixed `timeout` (e.g., 10 sec) | **Dynamic** waiting — no timeout needed |
| User waits a fixed time, then gives up | User waits until **notified** by the releasing thread |
| Cannot signal specific waiters | Can `notify()` or `notify_all()` waiting threads |

### Condition: Wait and Signal Pattern

A **Condition** supports two key operations:

- **`wait()`**: Block the current thread until notified (no timeout needed)
- **`notify(n=1)`**: Wake up $n$ threads waiting on this condition
- **`notify_all()`**: Wake up all waiting threads

```
User A finishes with resource → notify() → User B (waiting) wakes up and gets resource
```

> The wait time is **dynamic**: it could be nanoseconds or hours, determined by when the previous user finishes.

### Implementation with Conditions

```python
from threading import Thread, Condition

class JobsAndUsers:
    def __init__(self, num_users, num_resources, job_size):
        self.n = num_users
        self.r = num_resources
        self.m = job_size
        self.resources = [Condition() for _ in range(num_resources)]
        self.jobs = [job_size for _ in range(num_users)]

    def user(self, i):
        while self.jobs[i] > 0:
            time.sleep(random.random())

            r = random.randint(0, self.r - 1)

            # Acquire the underlying lock of the Condition
            if self.resources[r].acquire():
                time.sleep(random.random() + 5)
                self.jobs[i] -= 1

                # Notify one waiting user that resource is available
                self.resources[r].notify()
                self.resources[r].release()
```

**Key change**: Replace `Semaphore` → `Condition`, remove `timeout`, add `notify()` before `release()`.

### Result

- No more timeout failures
- Users wait as long as needed (dynamically)
- **Still has unbalanced queues** — because resource selection is still random

---

## 4. Load Balancing

### The Problem

With random resource selection, the waiting queues become **dramatically unbalanced**:

```
Resource 0: [User 8, User 2, User 1, User 0]  ← 4 waiting
Resource 1: [User 6]                            ← 1 waiting  
Resource 2: [User 9, User 3, User 4, User 7, User 5]  ← 5 waiting
```

Some resources are overloaded while others are idle.

### The Solution: Minimum Queue Selection

Instead of choosing a random resource, **choose the resource with the shortest waiting queue**:

```python
# Before (random):
r = random.randint(0, self.r - 1)

# After (load balanced):
r = self.q.index(min(self.q))
```

### Full Implementation with Load Balancing

```python
import time
import random
from threading import Thread, Condition
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class JobsAndUsers:
    def __init__(self, num_users, num_resources, job_size):
        self.n = num_users
        self.r = num_resources
        self.m = job_size
        self.resources = [Condition() for _ in range(num_resources)]
        self.jobs = [job_size for _ in range(num_users)]
        self.q = [0 for _ in range(num_resources)]  # Queue counters

    def user(self, i):
        while self.jobs[i] > 0:
            time.sleep(random.random() / 2)

            # Load balance: pick resource with shortest queue
            r = self.q.index(min(self.q))
            self.q[r] += 1  # Join queue

            if self.resources[r].acquire():
                time.sleep(random.random() / 2)
                self.jobs[i] -= 1

                self.q[r] -= 1  # Leave queue
                self.resources[r].notify()
                self.resources[r].release()

if __name__ == "__main__":
    n, r, m = 10, 3, 100
    ju = JobsAndUsers(n, r, m)
    threads = [Thread(target=ju.user, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    ju.draw()  # Visualize
    for t in threads:
        t.join()
```

### Queue Tracking

Maintain an array `self.q` of size $r$ (number of resources):
- **Increment** `self.q[r]` when a user enters the queue for resource $r$
- **Decrement** `self.q[r]` when a user finishes and releases resource $r$

The load balancer selects:

$$r^* = \arg\min_{r} \text{queue}[r] = \texttt{self.q.index(min(self.q))}$$

### Results

| Metric | Random Selection | Load Balanced |
|---|---|---|
| Max queue length | Up to $n$ (all users on one resource) | $\approx \lceil n/r \rceil$ |
| Job completion | Uneven across users | **Balanced** across users |
| Resource utilization | Some idle, some overloaded | **Even** distribution |
| Starvation | Possible | **Prevented** |

---

## 5. Visualization with Matplotlib

The lecture uses `FuncAnimation` to create real-time bar charts showing:

- **Left plot**: Number of users waiting in each resource's queue
- **Right plot**: Remaining jobs per user

```python
from matplotlib.animation import FuncAnimation

# In constructor:
self.fig, self.axes = plt.subplots(1, 2, tight_layout=True)

def init(self):
    self.axes_properties()
    self.axes[0].bar(range(self.r), self.q)       # Resource queues
    self.axes[1].bar(range(self.n), self.jobs)     # Jobs remaining

def update(self, _):
    self.axes_properties()
    self.axes[0].bar(range(self.r), self.q)
    self.axes[1].bar(range(self.n), self.jobs)
    if sum(self.jobs) == 0:
        plt.close()  # All jobs done

def draw(self):
    _ = FuncAnimation(self.fig, self.update, init_func=self.init,
                      save_count=10)
    plt.show()
```

---

## 6. Course Summary (Pre-Midterm)

| Problem | Solution | Primitives |
|---|---|---|
| **GIL** (Global Interpreter Lock) | JIT Compiler (`numba`) | `@jit` decorator |
| **Race Condition** | Atomic operations | `Lock`, `Semaphore`, `Condition` |
| **Synchronization** | Thread joining | `Thread.join()` |
| **Deadlock** | Resource ordering, check-and-release | Lock algorithms |
| **Load Balance** | Minimum-queue selection | `Condition` with wait/signal |

> After midterm: transition from **multi-threading** to **multi-processing**.

---

## Key Takeaways

- **Conditions** replace semaphores when you need **dynamic wait times** instead of fixed timeouts
- The **wait/signal (notify)** pattern: releasing thread signals waiting threads, enabling dynamic queuing
- **Load balancing** = always direct the next user to the resource with the **shortest queue**
- Formula: $r^* = \texttt{q.index(min(q))}$ where `q[i]` = number of users waiting for resource $i$
- Maximum balanced queue length is approximately $\lceil n / r \rceil$ (users divided by resources)
- The three-step progression: Semaphore (fixed timeout) → Condition (dynamic wait) → Load-balanced Condition (smart resource selection)
