# Python ML Interview Questions - Theory Questions

## Question 1

**Explain the difference between Python 2 and Python 3.**

### Definition
Python 2 and Python 3 are two major versions of Python. Python 3 (released 2008) is a redesign that is **not backward-compatible** with Python 2. Python 2 support ended in 2020; Python 3 is the standard for all modern development.

### Key Differences

| Feature | Python 2 | Python 3 |
|---------|----------|----------|
| Print | `print "Hello"` (statement) | `print("Hello")` (function) |
| Integer Division | `5/2 = 2` (floor) | `5/2 = 2.5` (true division) |
| Strings | ASCII by default (`str` + `unicode`) | Unicode by default (`str` + `bytes`) |
| Range | `range()` creates list; `xrange()` is generator | `range()` is generator (memory efficient) |
| Exception | `except Exception, e:` | `except Exception as e:` |

### Why Python 3 Matters for ML
- **True division** prevents subtle bugs in numerical computations
- **Unicode support** is critical for NLP and text processing
- **Memory-efficient range** handles large iterations without memory issues
- All modern ML libraries (NumPy, TensorFlow, PyTorch) require Python 3

### Interview Tip
Always mention that Python 3 is the only choice for production ML work today.

---

## Question 2

**How does Python manage memory?**

### Definition
Python manages memory automatically through a **private heap space**, **reference counting** for immediate deallocation, and a **cyclic garbage collector** for handling reference cycles.

### Core Mechanisms

**1. Reference Counting (Primary)**
- Every object has a counter tracking how many references point to it
- Counter increments when assigned, decrements when reference is removed
- When count reaches 0, memory is immediately freed

```python
x = [1, 2, 3]    # ref count = 1
y = x           # ref count = 2
del x           # ref count = 1
y = None        # ref count = 0 -> memory freed
```

**2. Cyclic Garbage Collector (Secondary)**
- Handles reference cycles that reference counting cannot detect
- Uses generational collection (Gen 0, 1, 2)
- New objects start in Gen 0; surviving objects are promoted
- Younger generations collected more frequently

```python
# Reference cycle example
a = []
b = []
a.append(b)
b.append(a)
# Both have ref count >= 1 but are unreachable
# Cyclic GC detects and cleans this
```

### Practical Relevance
- Understanding memory helps optimize large-scale ML data pipelines
- Use generators for memory-efficient data loading
- Avoid creating unnecessary reference cycles

---

## Question 3

**What is PEP 8 and why is it important?**

### Definition
PEP 8 is Python's official **style guide** that provides conventions for writing clean, readable, and consistent code.

### Key Guidelines

| Element | Convention |
|---------|------------|
| Indentation | 4 spaces |
| Line length | Max 79 characters |
| Functions/Variables | `lowercase_with_underscores` |
| Classes | `PascalCase` |
| Constants | `UPPERCASE_WITH_UNDERSCORES` |
| Imports | Separate lines; grouped (standard, third-party, local) |

### Why It Matters
1. **Readability**: Code is read more than written
2. **Consistency**: Team collaboration becomes seamless
3. **Reduced cognitive load**: Focus on logic, not parsing style
4. **Professionalism**: Industry standard expectation

### Tools
- **Linters**: `flake8`, `pylint` (detect violations)
- **Formatters**: `black`, `autopep8` (auto-fix)

---

## Question 4

**Describe how a dictionary works in Python.**

### Definition
A dictionary is a **mutable, unordered** (insertion-ordered in Python 3.7+) collection of **key-value pairs** implemented using a **hash table** for O(1) average lookup.

### Keys and Values
- **Keys**: Must be **immutable** and **hashable** (strings, numbers, tuples)
- **Values**: Can be any data type

### How Hash Tables Work
1. Key is passed through a **hash function** → produces hash value
2. Hash value determines memory location for storage
3. Lookup: hash the key again → directly access the value

### Code Example
```python
# Create dictionary
student = {"name": "Alice", "age": 21, "courses": ["Math", "CS"]}

# Access value
print(student["name"])          # Alice

# Add/modify
student["gpa"] = 3.8            # Add new key
student["age"] = 22             # Modify existing

# Check existence
if "courses" in student:
    print("Has courses")

# Iterate
for key, value in student.items():
    print(f"{key}: {value}")
```

### Interview Tip
Mention O(1) average time complexity for lookup, insertion, and deletion.

---

## Question 5

**What is list comprehension? Give an example.**

### Definition
List comprehension is a **concise, Pythonic syntax** for creating lists by applying an expression to each item in an iterable, optionally with filtering.

### Syntax
```python
new_list = [expression for item in iterable if condition]
```

### Example: Squares of even numbers
```python
# Traditional loop
result = []
for num in range(10):
    if num % 2 == 0:
        result.append(num ** 2)

# List comprehension (same result)
result = [num ** 2 for num in range(10) if num % 2 == 0]
# Output: [0, 4, 16, 36, 64]
```

### Why Use It
1. **Concise**: Single line vs multiple lines
2. **Faster**: Optimized at C level internally
3. **Readable**: Closer to mathematical set notation

### ML Use Cases
```python
# Data cleaning
cleaned = [text.strip().lower() for text in raw_data]

# Feature extraction
binary_features = [1 if val > threshold else 0 for val in values]
```

---

## Question 6

**Explain generators. How do they differ from list comprehensions?**

### Definition
Generators are **iterators** that produce values **lazily** (one at a time, on demand) rather than storing all values in memory at once.

### Key Differences

| Feature | List Comprehension | Generator Expression |
|---------|-------------------|---------------------|
| Syntax | `[expr for x in iter]` | `(expr for x in iter)` |
| Evaluation | Eager (all at once) | Lazy (on demand) |
| Memory | Stores entire list | Stores only state |
| Reusability | Can iterate multiple times | Single use |

### Code Example
```python
import sys

# List comprehension - high memory
list_comp = [i**2 for i in range(1_000_000)]
print(sys.getsizeof(list_comp))   # ~8 MB

# Generator expression - minimal memory
gen_exp = (i**2 for i in range(1_000_000))
print(sys.getsizeof(gen_exp))     # ~120 bytes
```

### Generator Function (using yield)
```python
def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1

for num in count_up_to(5):
    print(num)  # 0, 1, 2, 3, 4
```

### ML Use Case
- **Data loading**: Feed large datasets to models in batches (Keras/TensorFlow data generators)
- **Streaming data**: Process files line by line without loading entire file

---

## Question 7

**How does Python's garbage collection work?**

### Definition
Python uses **reference counting** as the primary mechanism and a **generational cyclic garbage collector** as secondary to handle reference cycles.

### Algorithm Steps

**Step 1: Reference Counting**
- Track reference count for each object
- Increment on assignment, decrement on deletion
- Free memory when count = 0

**Step 2: Cyclic Garbage Collector**
- Detect unreachable cycles (objects referencing each other)
- Uses 3 generations (0, 1, 2)
- Younger generations collected more frequently
- Surviving objects promoted to older generations

### Generational Collection
| Generation | Contains | Collection Frequency |
|------------|----------|---------------------|
| 0 | New objects | Most frequent |
| 1 | Survived Gen 0 | Less frequent |
| 2 | Long-lived objects | Rare |

### Interview Tip
Reference counting handles ~95% of garbage; cyclic GC handles the edge cases.

---

## Question 8

**What are decorators? Provide an example.**

### Definition
A decorator is a **function that wraps another function** to add functionality without modifying its source code. Uses the `@` syntax.

### How It Works
```python
@my_decorator
def func():
    pass

# Equivalent to:
func = my_decorator(func)
```

### Practical Example: Timing Decorator
```python
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

@timing_decorator
def process_data(n):
    time.sleep(1)  # Simulate work
    return "Done"

process_data(100)
# Output: process_data took 1.0023s
```

### Common Use Cases
- **Logging**: Log function calls
- **Timing**: Profile execution time
- **Authentication**: Check user permissions (Flask/Django)
- **Caching/Memoization**: Store results of expensive computations

---

## Question 9

**What is NumPy and how is it useful in ML?**

### Definition
NumPy (Numerical Python) is the **foundational library** for scientific computing, providing high-performance **multi-dimensional arrays (ndarray)** and mathematical functions.

### Why NumPy for ML

**1. Performance**
- Vectorized operations (no Python loops needed)
- Implemented in optimized C/Fortran code
- 10-100x faster than Python lists for numerical operations

**2. Data Representation**
- Dataset → 2D array (samples × features)
- Image → 3D array (height × width × channels)
- Model weights → ndarray

**3. Essential Operations**
```python
import numpy as np

# Linear algebra
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = a @ b                    # Matrix multiplication

# Statistics
mean = np.mean(a)
std = np.std(a)

# Random numbers
weights = np.random.randn(100, 50)  # Initialize neural network weights
```

### Interview Tip
NumPy is the foundation; Pandas, scikit-learn, TensorFlow all build on top of it.

---

## Question 10

**How does Scikit-learn fit into the ML workflow?**

### Definition
Scikit-learn is Python's **most popular library for traditional ML**, providing a consistent API for preprocessing, model training, evaluation, and selection.

### Role in ML Workflow

| Step | Scikit-learn Tools |
|------|-------------------|
| Preprocessing | `StandardScaler`, `OneHotEncoder`, `SimpleImputer` |
| Dimensionality Reduction | `PCA`, `TSNE` |
| Model Training | `LogisticRegression`, `RandomForest`, `SVM` |
| Model Evaluation | `accuracy_score`, `confusion_matrix`, `roc_auc_score` |
| Hyperparameter Tuning | `GridSearchCV`, `RandomizedSearchCV` |
| Pipelines | `Pipeline` (chain preprocessing + model) |

### Consistent API Pattern
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# All models follow: fit → predict
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Interview Tip
Scikit-learn is for traditional ML (not deep learning). For neural networks, use TensorFlow/PyTorch.

---

## Question 11

**Explain Matplotlib and Seaborn for data visualization.**

### Definition
**Matplotlib** is the foundational low-level plotting library; **Seaborn** is a high-level statistical visualization library built on Matplotlib.

### Comparison

| Feature | Matplotlib | Seaborn |
|---------|-----------|---------|
| Level | Low-level, verbose | High-level, concise |
| Purpose | Full customization | Statistical insights |
| Default Style | Basic | Beautiful defaults |
| Data Input | Arrays, lists | Pandas DataFrames |
| Best For | Custom, complex plots | EDA, statistical plots |

### Code Example
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Matplotlib - more control
plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Scatter Plot")
plt.show()

# Seaborn - one line for complex statistical plot
sns.boxplot(data=df, x="category", y="value", hue="group")
```

### Typical Workflow
Use Seaborn for quick EDA, then Matplotlib for final customization.

---

## Question 12

**What is TensorFlow and Keras? How do they relate?**

### Definition
**TensorFlow** is Google's low-level deep learning framework for tensor operations. **Keras** is the official high-level API for TensorFlow, making neural network building simple.

### Relationship
- Pre-2019: Keras was separate, backend-agnostic
- Post-2019: Keras is integrated as `tf.keras`
- TensorFlow = Engine (tensor ops, GPU acceleration)
- Keras = Steering wheel (simple model building)

### Comparison

| Feature | TensorFlow (low-level) | Keras (tf.keras) |
|---------|----------------------|-----------------|
| Control | Fine-grained | Abstracted |
| Use Case | Research, custom ops | Rapid prototyping |
| Complexity | Higher | Lower |

### Code Example (tf.keras)
```python
import tensorflow as tf
from tensorflow import keras

# Build model in few lines
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)
```

### Interview Tip
For most practitioners, `tf.keras` is the recommended approach—Keras simplicity with TensorFlow power.

