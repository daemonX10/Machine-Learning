# Lecture 04: Race Conditions and Deadlocks

## 1. How Python Executes Code (CPU-Level Operations)

A single Python statement is broken into **multiple CPU-level operations** (bytecode instructions). Use [Godbolt Compiler Explorer](https://godbolt.org) to inspect them.

### Example: Variable Assignment

```python
x = 0
```

Produces two bytecode instructions:

| Step | Instruction    | Description                |
|------|---------------|----------------------------|
| 1    | `LOAD_CONST 0` | Generate the value `0`     |
| 2    | `STORE_NAME x`  | Store value in variable `x` |

### Example: Increment Operation

```python
x += 1
```

Produces **four** bytecode instructions:

| Step | Instruction       | Description                     |
|------|------------------|---------------------------------|
| 1    | `LOAD_NAME x`     | Get the current value of `x`    |
| 2    | `LOAD_CONST 1`    | Generate the value `1`          |
| 3    | `BINARY_ADD`      | Add the two values              |
| 4    | `STORE_NAME x`    | Store the result back in `x`    |

> **Key insight:** Even a simple `x += 1` involves multiple sub-operations: **Get → Change → Assign**.

---

## 2. Race Condition

### Definition

A **race condition** occurs when multiple threads access and modify shared data concurrently, and the final result depends on the unpredictable order (timing) in which threads execute.

### How It Happens

To modify a shared variable, a thread performs three sub-steps:

1. **Get** the current value
2. **Change** (compute new value)
3. **Assign** the new value back

If two threads interleave these steps, they can read **stale values**:

| Time | Thread 0       | Thread 1       | $x$  |
|------|---------------|----------------|------|
| $t_0$ | Get value → 0  |                | 0    |
| $t_1$ | Change → 1     |                | 0    |
| $t_2$ |                | Get value → 0  | 0    |
| $t_3$ | Assign → 1     |                | **1** |
| $t_4$ |                | Change → 1     | 1    |
| $t_5$ |                | Assign → 1     | **1** |

> We incremented $x$ **twice**, but the result is $x = 1$ instead of $x = 2$.

### Demonstration: Bank Account Example

```python
from threading import Thread, Lock

class BankAccount:
    def __init__(self):
        self.balance = 0

    def earn(self):
        for _ in range(1_000_000):
            self.balance += 1

    def spend(self):
        for _ in range(1_000_000):
            self.balance -= 1

    def get_balance(self):
        return self.balance

if __name__ == "__main__":
    account = BankAccount()

    t1 = Thread(target=account.earn)
    t2 = Thread(target=account.spend)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Balance: {account.get_balance()}")
    # Expected: 0, Actual: some non-zero value (e.g., -243315)
```

- With **1,000 iterations**: works fine (race condition unlikely)
- With **1,000,000 iterations**: produces **wrong results** (race condition manifests)

---

## 3. Solving Race Conditions with Locks

### Atomic Operations

The get-change-assign sequence must be treated as an **atomic operation** — non-dividable, meaning no other thread should interfere during execution.

### Using `Lock` from `threading`

```python
from threading import Thread, Lock

class BankAccount:
    def __init__(self):
        self.balance = 0
        self.lock = Lock()

    def earn(self):
        for _ in range(1_000_000):
            self.lock.acquire()     # Block other threads
            self.balance += 1
            self.lock.release()     # Allow other threads

    def spend(self):
        for _ in range(1_000_000):
            self.lock.acquire()
            self.balance -= 1
            self.lock.release()
```

| Without Lock | With Lock |
|---|---|
| Wrong result (e.g., `-243315`) | Correct result: `0` |
| Fast execution | Slower (threads must wait) |
| Unsafe | Thread-safe |

> Works correctly even with 10 million iterations.

### Lock Mechanism

1. Thread **acquires** the lock → other threads **wait**
2. Thread performs the atomic operation
3. Thread **releases** the lock → waiting threads can proceed

### Trade-off

- Locks make code **safer** but **slower**
- Use locks when you cannot restructure code to avoid shared mutable state

---

## 4. Mutex (Mutual Exclusion)

A **mutex** (mutual exclusion) is another name for a lock. The term emphasizes that only **one thread** can hold it at a time, ensuring exclusive access to a resource.

```python
mutex = Lock()
mutex.acquire()   # Enter critical section
# ... critical section ...
mutex.release()   # Exit critical section
```

---

## 5. Deadlock

### Definition

A **deadlock** occurs when two or more threads are each waiting for a resource held by the other, creating a circular dependency where none can proceed.

### Conditions for Deadlock

When multiple locks exist and threads acquire them in **different orders**:

| Step | Thread A (Bora) | Thread B (Burak) |
|------|----------------|-----------------|
| 1    | Acquires Lock 1 | Acquires Lock 2  |
| 2    | Waits for Lock 2 | Waits for Lock 1 |
| —    | **Deadlock!** Both waiting forever | |

### Demonstration

```python
from threading import Thread, Lock
import time

lock1 = Lock()
lock2 = Lock()

def bora(l1, l2):
    while True:
        l1.acquire()        # Get Lock 1
        time.sleep(0.5)
        l2.acquire()        # Try to get Lock 2
        # ... do work ...
        l1.release()
        l2.release()

def burak(l1, l2):
    while True:
        l2.acquire()        # Get Lock 2 first (opposite order!)
        time.sleep(0.5)
        l1.acquire()        # Try to get Lock 1
        # ... do work ...
        l2.release()
        l1.release()

t1 = Thread(target=bora, args=(lock1, lock2))
t2 = Thread(target=burak, args=(lock1, lock2))
t1.start()
t2.start()
```

- May work for a few iterations, then **freezes** (deadlock)
- Once in deadlock, the program hangs indefinitely

---

## 6. Dining Philosophers Problem (Preview)

A classic CS problem that illustrates deadlock:

- **5 philosophers** sit around a circular table
- **5 chopsticks** are placed between them
- Each philosopher alternates between **thinking** and **eating**
- To eat, a philosopher needs **both** adjacent chopsticks
- A philosopher can only pick up **one chopstick at a time**

### Deadlock Scenario

If all 5 philosophers simultaneously pick up their left chopstick, each is waiting for the right chopstick held by their neighbor → **deadlock**.

> Solution is covered in Lecture 05.

---

## Key Takeaways

| Concept | Description | Solution |
|---------|-------------|----------|
| **Race Condition** | Threads read stale shared data due to interleaved operations | Locks (atomic operations) |
| **Atomic Operation** | Non-dividable sequence that must complete without interruption | `Lock.acquire()` / `Lock.release()` |
| **Mutex** | Mutual exclusion — only one thread accesses the critical section | Same as Lock |
| **Deadlock** | Circular wait — threads block each other forever | Lock ordering, resource hierarchy (Lecture 05) |

- Race conditions may **not appear** during development with small data but **manifest in production** with large-scale operations
- Locks make code **slower but safer** — use them when threads share mutable state
- Having **multiple locks** introduces the risk of **deadlock**
