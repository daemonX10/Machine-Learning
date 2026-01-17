# TensorFlow Interview Questions - Theory Questions

## Question 1

**What is TensorFlow and who developed it?**

### Answer

**Definition**: TensorFlow is a free, open-source machine learning platform developed by the **Google Brain team**. It was open-sourced in November 2015.

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Computation Graph** | Mathematical operations represented as nodes, data (tensors) flow along edges |
| **Tensor** | Multi-dimensional arrays that flow through the graph |
| **Eager Execution** | Operations execute immediately (TF 2.x default) |
| **End-to-end Platform** | Covers training, deployment, and production ML |

### Python Code Example
```python
import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Basic tensor operations
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = tf.add(a, b)  # Element-wise addition

print(f"a + b = {c.numpy()}")  # Output: [5 7 9]
```

---

## Question 2

**What are the main features of TensorFlow?**

### Answer

### Key Features

| Feature | Description |
|---------|-------------|
| **Eager Execution** | Operations run immediately, easy debugging |
| **Graph Mode** | `tf.function` converts code to optimized graphs |
| **Auto Differentiation** | `tf.GradientTape` computes gradients automatically |
| **Keras Integration** | High-level API for building models |
| **Multi-Platform** | CPU, GPU, TPU, mobile, browser support |
| **Distributed Training** | `tf.distribute.Strategy` for scaling |
| **TensorBoard** | Visualization tools for training |

### Python Code Example
```python
import tensorflow as tf

# Feature 1: Eager execution (default in TF 2.x)
x = tf.constant([[1, 2], [3, 4]])
print(x.numpy())  # Executes immediately

# Feature 2: Graph mode for optimization
@tf.function
def fast_multiply(a, b):
    return tf.matmul(a, b)

# Feature 3: Auto differentiation
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2
grad = tape.gradient(y, x)  # dy/dx = 2x = 6
print(f"Gradient: {grad.numpy()}")
```

---

## Question 3

**Can you explain the concept of a computation graph in TensorFlow?**

### Answer

**Definition**: A computation graph is a directed acyclic graph (DAG) where:
- **Nodes** = Operations (add, multiply, layers)
- **Edges** = Tensors (data flowing between operations)

### Components

| Component | Role | Example |
|-----------|------|---------|
| **Node** | Represents operation | `tf.add`, `tf.matmul` |
| **Edge** | Data flow (tensor) | Weights, activations |
| **Input** | Entry point | Training data |
| **Output** | Result | Predictions |

### Python Code Example
```python
import tensorflow as tf

# Graph mode with tf.function
@tf.function
def compute_graph(a, b):
    """This function gets converted to a computation graph"""
    c = tf.add(a, b)      # Node: addition
    d = tf.multiply(c, 2)  # Node: multiplication
    return d              # Edges: tensors flow between nodes

# Execute
x = tf.constant(3.0)
y = tf.constant(4.0)
result = compute_graph(x, y)
print(f"Result: {result.numpy()}")  # (3+4)*2 = 14

# View graph structure
print(compute_graph.get_concrete_function(x, y).graph.as_graph_def())
```

### Advantages
- **Optimization**: Graph analyzed before execution
- **Portability**: Export to different platforms
- **Parallelism**: Operations can run in parallel

---

## Question 4

**What are Tensors in TensorFlow?**

### Answer

**Definition**: A Tensor is a multi-dimensional array with a uniform data type.

### Tensor Ranks

| Rank | Name | Example | Shape |
|------|------|---------|-------|
| 0 | Scalar | `5` | `()` |
| 1 | Vector | `[1, 2, 3]` | `(3,)` |
| 2 | Matrix | `[[1,2], [3,4]]` | `(2, 2)` |
| 3 | 3D Tensor | Batch of images | `(batch, height, width)` |
| 4 | 4D Tensor | Color images | `(batch, H, W, channels)` |

### Python Code Example
```python
import tensorflow as tf

# Scalar (rank 0)
scalar = tf.constant(5)
print(f"Scalar shape: {scalar.shape}")  # ()

# Vector (rank 1)
vector = tf.constant([1, 2, 3])
print(f"Vector shape: {vector.shape}")  # (3,)

# Matrix (rank 2)
matrix = tf.constant([[1, 2], [3, 4]])
print(f"Matrix shape: {matrix.shape}")  # (2, 2)

# 3D Tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor shape: {tensor_3d.shape}")  # (2, 2, 2)

# Key properties
print(f"dtype: {matrix.dtype}")  # int32
print(f"rank: {tf.rank(matrix).numpy()}")  # 2

# Variable (mutable tensor for model parameters)
var = tf.Variable([1.0, 2.0, 3.0])
var.assign([4.0, 5.0, 6.0])  # Can change values
```

---

## Question 5

**How does TensorFlow differ from other ML libraries?**

### Answer

### Comparison Table

| Feature | TensorFlow | PyTorch | Scikit-learn |
|---------|------------|---------|--------------|
| **Focus** | Production ML | Research | Classical ML |
| **Execution** | Eager + Graph | Eager | Imperative |
| **API** | Keras (high-level) | Explicit, Pythonic | fit/predict |
| **Deployment** | TF Serving, TFLite | TorchServe | Pickle |
| **Distributed** | `tf.distribute` | DDP | Limited |
| **Debugging** | Good (Eager mode) | Excellent | Simple |

### When to Use

```python
# TensorFlow: Production + Deep Learning
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Scikit-learn: Classical ML
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```

### Summary
- **TensorFlow**: Best for production deployment, scalable systems
- **PyTorch**: Best for research, rapid prototyping
- **Scikit-learn**: Best for classical ML algorithms

---

## Question 6

**What is a Session in TensorFlow?**

### Answer

**Definition**: `tf.Session` was a TensorFlow 1.x concept for executing computation graphs. **Deprecated in TF 2.x**.

### TF 1.x vs TF 2.x

| Aspect | TF 1.x (Session) | TF 2.x (Eager) |
|--------|-----------------|----------------|
| **Execution** | Deferred (build graph first) | Immediate |
| **Syntax** | `sess.run()` | Direct function call |
| **Debugging** | Difficult | Easy |
| **Data input** | `feed_dict` | Direct arguments |

### Code Comparison
```python
# TensorFlow 1.x style (legacy)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.add(a, b)

with tf.Session() as sess:
    result = sess.run(c, feed_dict={a: 5.0, b: 3.0})
    print(result)  # 8.0

# TensorFlow 2.x style (modern)
import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(3.0)
c = tf.add(a, b)
print(c.numpy())  # 8.0 - executes immediately!
```

---

## Question 7

**What is the difference between TensorFlow 1.x and TensorFlow 2.x?**

### Answer

### Key Differences

| Feature | TensorFlow 1.x | TensorFlow 2.x |
|---------|---------------|----------------|
| **Default Mode** | Graph execution | Eager execution |
| **Sessions** | Required | Not needed |
| **Placeholders** | Required for input | Not needed |
| **API** | Multiple competing APIs | Keras is standard |
| **Control Flow** | `tf.cond`, `tf.while_loop` | Python if/for |
| **Performance** | Graph mode | `tf.function` decorator |

### Python Code Example
```python
import tensorflow as tf

# TF 2.x: Clean, Pythonic code
# No sessions, no placeholders!

# Simple computation
x = tf.constant([1.0, 2.0, 3.0])
y = x * 2 + 1
print(y.numpy())  # Immediate result

# Control flow with Python
def compute(x):
    if x > 0:  # Regular Python if
        return x * 2
    else:
        return x

# Convert to graph for performance
@tf.function
def fast_compute(x):
    return x * 2 + 1

result = fast_compute(tf.constant(5.0))
print(result.numpy())
```

---

## Question 8

**How does TensorFlow handle automatic differentiation?**

### Answer

**Definition**: TensorFlow uses **reverse-mode automatic differentiation** via `tf.GradientTape` to compute gradients.

### How It Works
1. **Record**: Operations inside `GradientTape` are recorded
2. **Forward pass**: Compute loss
3. **Backward pass**: `tape.gradient()` computes gradients
4. **Update**: Optimizer updates weights

### Python Code Example
```python
import tensorflow as tf

# Model parameters
w = tf.Variable(2.0)
b = tf.Variable(1.0)

# Data
x = tf.constant([1.0, 2.0, 3.0, 4.0])
y_true = tf.constant([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1

# Optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Training step
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = w * x + b
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Compute gradients
    gradients = tape.gradient(loss, [w, b])
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, [w, b]))
    return loss

# Train
for epoch in range(100):
    loss = train_step()

print(f"w: {w.numpy():.4f}, b: {b.numpy():.4f}")  # ~2.0, ~1.0
```

---

## Question 9

**What is a Placeholder in TensorFlow?**

### Answer

**Definition**: `tf.placeholder` was a TF 1.x concept for declaring input variables. **Deprecated in TF 2.x**.

### Comparison

| TF 1.x (Placeholder) | TF 2.x (Modern) |
|---------------------|-----------------|
| Declare shape/dtype upfront | Pass data directly |
| Feed via `feed_dict` | Function arguments |
| Part of static graph | Eager execution |

### Code Evolution
```python
# TF 1.x: Placeholder (legacy)
# x = tf.placeholder(tf.float32, shape=[None, 784])
# result = sess.run(output, feed_dict={x: data})

# TF 2.x: Direct input (modern)
import tensorflow as tf

@tf.function
def model_forward(x):
    """x is just a function parameter, no placeholder needed"""
    return tf.nn.softmax(tf.matmul(x, weights) + bias)

# Call directly with data
data = tf.random.normal([32, 784])
output = model_forward(data)
```

### Summary
Placeholders are **obsolete** in TF 2.x. Use:
- Function parameters for inputs
- `tf.data.Dataset` for data pipelines

---

## Question 10

**Could you explain TensorFlow Lite and where it's used?**

### Answer

**Definition**: TensorFlow Lite (TFLite) is a lightweight version for **mobile and embedded devices**.

### Key Features

| Feature | Description |
|---------|-------------|
| **Small size** | Optimized for mobile (~1MB) |
| **Quantization** | Float32 → Int8 (4x smaller) |
| **On-device** | No server needed |
| **Low latency** | Fast inference |

### Conversion Workflow
```python
import tensorflow as tf

# 1. Train a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10)

# 2. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Apply quantization for smaller size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 3. Save the model
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Use Cases
- **Mobile apps**: Image classification, object detection
- **IoT devices**: Sensor data analysis
- **Microcontrollers**: Keyword spotting, gesture recognition

---

## Question 11

**What are the different data types supported by TensorFlow?**

### Answer

### Data Types Table

| Type | Bits | Use Case |
|------|------|----------|
| `tf.float32` | 32 | Default for ML (weights, inputs) |
| `tf.float64` | 64 | High precision scientific computing |
| `tf.float16` | 16 | Mixed precision training (faster) |
| `tf.bfloat16` | 16 | TPU optimization |
| `tf.int32` | 32 | Indices, labels |
| `tf.int64` | 64 | Large indices |
| `tf.bool` | 1 | Conditions, masks |
| `tf.string` | - | Text data |

### Python Code Example
```python
import tensorflow as tf

# Creating tensors with different dtypes
float_tensor = tf.constant([1.0, 2.0], dtype=tf.float32)
int_tensor = tf.constant([1, 2], dtype=tf.int32)
bool_tensor = tf.constant([True, False], dtype=tf.bool)

# Type casting
casted = tf.cast(int_tensor, tf.float32)

# Check dtype
print(f"dtype: {float_tensor.dtype}")  # <dtype: 'float32'>

# Mixed precision (for faster training on GPUs)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### Best Practices
- Use `tf.float32` for most cases
- Use `tf.float16` for faster GPU training
- Use `tf.int32` for indices and labels

