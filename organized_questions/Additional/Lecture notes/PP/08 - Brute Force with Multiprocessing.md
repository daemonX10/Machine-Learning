# Lecture 08 — Brute Force with Multiprocessing

## 1. Recap: The Memory-Bound Problem

From Lecture 07, the brute-force password cracker stored **all** index combinations in a Python list before processing. For 62 characters and 8-character passwords:

$$\text{Combinations} = 62^8 \approx 2.18 \times 10^{14}$$

Storing these in memory is **impossible** on a regular PC. The solution was **memory-bound** — we need to transform it into a **CPU-bound** problem.

---

## 2. Generators vs Regular Functions

### 2.1 The Problem with `return`

A regular function creates the **entire list in memory** before returning:

```python
def squares(n):
    return [i ** 2 for i in range(n)]
```

For `n = 10,000,000`, the list consumes ~80 MB of RAM, and there is a **delay** before any output appears.

### 2.2 The Solution: `yield` (Generators)

A generator **produces values one at a time**, keeping only one value in memory:

```python
def square_generator(n):
    for i in range(n):
        yield i ** 2
```

| | Regular Function | Generator |
|--|-----------------|-----------|
| **Keyword** | `return` | `yield` |
| **Memory** | Stores entire result | Stores one value at a time |
| **Startup** | Must compute all values first | Produces values immediately |
| **Use case** | Small datasets | Large/infinite sequences |

### 2.3 Memory Comparison

```python
import sys

regular = squares(10_000_000)
generated = square_generator(10_000_000)

print(sys.getsizeof(regular))    # ~89,095,160 bytes (~85 MB)
print(sys.getsizeof(generated))  # 112 bytes
```

---

## 3. Recursive Generator for Password Combinations

### 3.1 Tree Structure of Combinations

Password combinations form a **tree**:

```
Level 0: ""  (empty string)
Level 1: "a", "b", "c", ..., "9"
Level 2: "aa", "ab", ..., "a9", "ba", "bb", ..., "99"
  ...
Level 8: all 62^8 eight-character combinations
```

### 3.2 Recursive Generator Implementation

```python
def generate_text(n, chars):
    """Yields all combinations of length n from chars."""
    if n == 0:
        yield ""    # base case: empty string
    else:
        for pw in generate_text(n - 1, chars):  # recursive call
            for c in chars:
                yield pw + c   # append each character
```

**How it works:**
1. **Base case** (`n=0`): yield an empty string
2. **Recursive case**: get all strings of length `n-1`, append each character from the pool
3. Uses **no additional memory** — generates combinations on-the-fly

### 3.3 Usage

```python
import string

chars = string.ascii_letters + string.digits  # 62 characters
length = 8

for text in generate_text(length, chars):
    print(text)  # Output appears immediately, no memory buildup
```

---

## 4. Multiprocessing Solution

### 4.1 Why Multiprocessing Instead of Threading?

| Issue | Threading | Multiprocessing |
|-------|-----------|-----------------|
| **GIL** | Cannot bypass when using `hashlib` | Each process has its own interpreter — no GIL issue |
| **Memory** | Threads share memory (easy) | Processes have separate memory (need shared variables) |
| **`numba.jit`** | Cannot compile `hashlib` code | Not needed — GIL is irrelevant |

### 4.2 Shared Variables Between Processes

```python
from multiprocessing import Process, Value

# Shared integer variable (type 'i' = integer, 'd' = double)
flag = Value('i', 0)  # 0 = not found, 1 = found
```

### 4.3 Parallelization Strategy

**Fix the first character** and assign each starting character to a separate process:

```
Process 0 → starts with 'a', generates remaining 7 chars
Process 1 → starts with 'b', generates remaining 7 chars
...
Process 61 → starts with '9', generates remaining 7 chars
```

This creates **62 processes** (one per character in the pool).

### 4.4 Single Process Function

```python
import hashlib, os

def text_to_md5(text):
    return hashlib.md5(text.encode()).hexdigest()

def single_process(initial_text, chars, length, pwd, flag):
    """Each process runs this with a fixed first character."""
    for i in range(1, length + 1):
        for text in generate_text(i, chars):
            combined = initial_text + text

            # Check if another process found the password
            if flag.value == 1:
                break

            # Check if this combination matches
            if text_to_md5(combined) == pwd:
                flag.value = 1
                print(f"Password found: {combined}")
                return

        if flag.value == 1:
            break
```

### 4.5 Main Program

```python
from multiprocessing import Process, Value
import string

def main():
    chars = string.ascii_letters + string.digits
    length = 8
    pwd = "e4b872a851..."  # MD5 hash of target password

    processes = []
    flag = Value('i', 0)  # shared flag between processes

    # Create one process per starting character
    for c in chars:
        processes.append(
            Process(
                target=single_process,
                args=(c, chars, length - 1, pwd, flag)
                # length - 1 because first char is fixed
            )
        )

    for p in processes: p.start()
    for p in processes: p.join()

if __name__ == '__main__':
    main()
```

### 4.6 Process Communication

```python
# Inside single_process:
if flag.value == 1:   # check: did another process find it?
    break

flag.value = 1        # signal: I found the password!
```

### 4.7 Process IDs for Monitoring

```python
import os

os.getpid()    # current process ID
os.getppid()   # parent process ID (main thread)
```

Use `htop` with filter to monitor all processes:
```bash
htop  # Press F4, type filename to filter
```

---

## 5. Comparison: Memory-Bound vs CPU-Bound

| Approach | Memory Usage | Startup Time | GIL Issue |
|----------|-------------|--------------|-----------|
| List of all indices | $O(62^8)$ — impossible | Minutes+ | Yes (threads) |
| Generator + threads | $O(1)$ per combination | Immediate | Yes (`hashlib` blocks JIT) |
| Generator + multiprocessing | $O(1)$ per combination | Immediate | **No** — separate interpreters |

---

## 6. Key Takeaways

- **Generators** (`yield`) transform memory-bound problems into CPU-bound by producing values lazily.
- A **recursive generator** elegantly enumerates all combinations of length $n$ without storing them.
- **Multiprocessing** bypasses the GIL entirely since each process has its own Python interpreter.
- Use `multiprocessing.Value('i', 0)` to create **shared variables** between processes for coordination (e.g., a stop flag).
- Fix the first character and assign subtrees to processes — a simple and effective **parallelization strategy** for tree-structured search.
- Avoid excessive `print()` calls in parallel code — they consume significant resources and serialize output.
