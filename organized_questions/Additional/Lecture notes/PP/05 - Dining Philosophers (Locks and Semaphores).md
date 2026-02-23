# Lecture 05: Dining Philosophers (Locks and Semaphores)

## 1. Problem Description

The **Dining Philosophers Problem** is a classic concurrency problem that illustrates deadlock and resource contention.

### Setup

- **5 philosophers** sit around a **circular table**
- Each philosopher has a **plate** of food (spaghetti)
- **5 chopsticks** are placed between adjacent philosophers
- Philosophers are indexed $0, 1, 2, 3, 4$

### Rules

1. A philosopher can be in one of two states: **Thinking** or **Eating**
2. Initially, all philosophers are **thinking** for a random duration
3. To **eat**, a philosopher needs **both** adjacent chopsticks
4. A philosopher can only pick up **one chopstick at a time**
5. After eating, the philosopher puts down both chopsticks and resumes thinking

### Deadlock Scenario

If all philosophers simultaneously pick up their **left** chopstick:

$$\text{Philosopher } i \text{ holds chopstick } i, \text{ waits for chopstick } (i+1) \bmod 5$$

All are waiting for the next chopstick → **deadlock** (no progress possible).

---

## 2. Solution 1: Using Locks

### Chopstick Indexing (Circular Table)

For philosopher $i$:
- **First chopstick:** index $i$
- **Second chopstick:** index $j = (i + 1) \bmod n$

This modular arithmetic handles the wrap-around (philosopher 4 needs chopstick 0).

### Strategy: Check Before Acquiring

Instead of blindly acquiring a chopstick (which causes deadlock), **check its status first** using `locked()`:

1. Check if chopstick $i$ is free → if yes, acquire it
2. Sleep briefly (simulate real-world delay)
3. Check if chopstick $j$ is free → if yes, acquire it; **if not, release chopstick $i$**
4. If both acquired: eat, then release both

> Releasing the first chopstick when the second is unavailable **prevents deadlock**.

### Implementation

```python
import time
import random
from threading import Thread, Lock

class DiningPhilosophers:
    def __init__(self, num_philosophers=5, meal_size=7):
        self.meals = [meal_size for _ in range(num_philosophers)]
        self.chopsticks = [Lock() for _ in range(num_philosophers)]
        self.status = ['T' for _ in range(num_philosophers)]  # T=Thinking

    def philosopher(self, i):
        n = len(self.meals)
        j = (i + 1) % n  # Next chopstick index

        while self.meals[i] > 0:
            # Thinking
            self.status[i] = 'T'
            time.sleep(random.random())

            # Decide to eat — look for chopsticks
            self.status[i] = '_'  # Looking for chopsticks

            if not self.chopsticks[i].locked():
                self.chopsticks[i].acquire()
                time.sleep(random.random())

                if not self.chopsticks[j].locked():
                    self.chopsticks[j].acquire()

                    # Eating
                    self.status[i] = 'E'
                    time.sleep(random.random())
                    self.meals[i] -= 1

                    # Release both chopsticks
                    self.chopsticks[j].release()
                    self.chopsticks[i].release()
                else:
                    # Second chopstick not free — release first to avoid deadlock
                    self.chopsticks[i].release()

def main():
    n, m = 5, 7
    dp = DiningPhilosophers(n, m)
    threads = [Thread(target=dp.philosopher, args=(i,)) for i in range(n)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
```

### Key Observations with Locks

| Philosophers | Max Eating Simultaneously |
|---|---|
| 5 | 2 |
| 10 | 3 |

- With locks, philosophers **check and pass** if a chopstick is locked
- No waiting/queueing mechanism — a philosopher either gets the chopstick or gives up

---

## 3. Solution 2: Using Semaphores

### Semaphore Basics

A **semaphore** maintains an internal counter:

| Counter Value | Meaning |
|---|---|
| $> 0$ | Resource is available |
| $= 0$ | Resource is in use; new requesters must **wait** |

Operations:
- **acquire()**: Decrement counter. If counter would go negative → **block/wait**
- **release()**: Increment counter. Unblock a waiting thread

### Key Difference: Lock vs Semaphore

| Feature | Lock | Semaphore |
|---|---|---|
| Status check | `locked()` → check without waiting | No `locked()` method |
| Waiting | No built-in wait | `acquire(timeout=...)` waits up to N seconds |
| Concurrent users | Always 1 | Configurable via initial value |
| Initial value | N/A (binary) | `Semaphore(n)` → `n` concurrent users |

### Implementation Changes

```python
from threading import Thread, Semaphore

class DiningPhilosophers:
    def __init__(self, num_philosophers=5, meal_size=7):
        self.meals = [meal_size for _ in range(num_philosophers)]
        # Semaphore(1) = only 1 philosopher can hold each chopstick
        self.chopsticks = [Semaphore(1) for _ in range(num_philosophers)]

    def philosopher(self, i):
        n = len(self.meals)
        j = (i + 1) % n

        while self.meals[i] > 0:
            time.sleep(random.random())

            # Try to acquire first chopstick (wait up to 1 second)
            if self.chopsticks[i].acquire(timeout=1):
                # Try to acquire second chopstick
                if self.chopsticks[j].acquire(timeout=1):
                    # Eat
                    time.sleep(random.random())
                    self.meals[i] -= 1

                    # Release both
                    self.chopsticks[j].release()
                    self.chopsticks[i].release()
                else:
                    # Could not get second — release first
                    self.chopsticks[i].release()
```

### Semaphore with Higher Initial Values

```python
# Allow 2 philosophers to share each chopstick simultaneously
chopsticks = [Semaphore(2) for _ in range(n)]
```

With `Semaphore(2)`: up to **5 philosophers** can eat simultaneously (all of them).

> For the standard Dining Philosophers problem, the constraint is `Semaphore(1)` because each physical chopstick can only be held by one person.

### When to Use Higher Values

In real-world problems, resources may support multiple concurrent users:
- Database connection pool: `Semaphore(10)` → 10 concurrent connections
- Thread pool: `Semaphore(4)` → 4 concurrent worker threads
- API rate limiter: `Semaphore(100)` → 100 concurrent requests

---

## 4. Comparison Summary

| Aspect | Locks Approach | Semaphores Approach |
|---|---|---|
| Deadlock prevention | Check `locked()`, release if blocked | Use `timeout`, release if expired |
| Waiting behavior | Pass immediately if locked | Wait up to `timeout` seconds |
| Flexibility | Binary (locked/unlocked) | Counter-based (configurable concurrency) |
| Use case | Simple mutual exclusion | Resource pools, bounded concurrency |

---

## Key Takeaways

- The **Dining Philosophers** problem models competing processes needing multiple shared resources
- **Deadlock avoidance**: Always release acquired resources if you can't get all the resources you need
- **Circular indexing**: Use $j = (i+1) \bmod n$ for circular resource adjacency
- **Locks**: Good for simple check-and-pass; no built-in waiting
- **Semaphores**: Support timeout-based waiting and configurable concurrency levels
- The `timeout` argument in semaphores enables a **wait-and-retry** pattern instead of immediate pass/fail
- In real applications, choose the initial semaphore value based on how many processes your resource can serve concurrently
