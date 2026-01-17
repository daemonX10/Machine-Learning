# Keras Interview Questions - Theory Questions

## Question 1

**What is Keras and how does it relate to TensorFlow?**

### Answer

**Definition**: Keras is a high-level deep learning API written in Python. It is the official high-level API for **TensorFlow 2.x** (accessed via `tf.keras`).

### Core Concepts

| Aspect | Description |
|--------|-------------|
| **Purpose** | Simplify neural network development |
| **Relationship** | Wrapper over TensorFlow, JAX, PyTorch |
| **Philosophy** | User-friendly, modular, extensible |
| **Access** | `from tensorflow import keras` |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

# Simple neural network in Keras
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Configure for training
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### Key Relationship with TensorFlow
- **tf.keras** = Keras integrated into TensorFlow
- Access TensorFlow features (distributed training, TensorBoard, deployment)
- Use TensorFlow's GPU/TPU acceleration

---

## Question 2

**Can you explain the concept of a deep learning framework?**

### Answer

**Definition**: A deep learning framework is a software library that provides building blocks for designing, training, and deploying neural networks.

### Key Features

| Feature | Description |
|---------|-------------|
| **Tensor Operations** | Efficient multi-dimensional array math |
| **Auto Differentiation** | Automatic gradient computation |
| **Pre-built Layers** | Dense, Conv2D, LSTM, etc. |
| **Hardware Acceleration** | GPU/TPU support |
| **Scalability** | Distributed training |

### Popular Frameworks

| Framework | Strengths |
|-----------|-----------|
| **TensorFlow** | Production, deployment |
| **PyTorch** | Research, flexibility |
| **Keras** | High-level API, simplicity |
| **JAX** | Scientific computing |

### Python Code Example
```python
# Framework handles complex operations automatically
import tensorflow as tf

# Automatic differentiation example
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2  # Forward pass

grad = tape.gradient(y, x)  # Framework computes gradient
print(f"dy/dx at x=3: {grad.numpy()}")  # Output: 6.0
```

---

## Question 3

**What are the core components of a Keras model?**

### Answer

### Core Components

| Component | Purpose | Examples |
|-----------|---------|----------|
| **Layers** | Build architecture | Dense, Conv2D, LSTM |
| **Optimizer** | Update weights | Adam, SGD, RMSprop |
| **Loss Function** | Measure error | CrossEntropy, MSE |
| **Metrics** | Monitor performance | Accuracy, F1, AUC |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

# 1. LAYERS - Define architecture
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 2. OPTIMIZER
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# 3. LOSS FUNCTION
loss = keras.losses.SparseCategoricalCrossentropy()

# 4. METRICS
metrics = ['accuracy']

# Combine in compile()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
```

### Best Practices
- Match loss to output activation (softmax → categorical_crossentropy)
- Start with Adam optimizer
- Track multiple metrics for imbalanced data

---

## Question 4

**Explain the difference between Sequential and Functional APIs in Keras.**

### Answer

### Comparison

| Aspect | Sequential API | Functional API |
|--------|----------------|----------------|
| **Structure** | Linear stack | Any topology |
| **Multiple I/O** | ❌ No | ✅ Yes |
| **Shared layers** | ❌ No | ✅ Yes |
| **Branching** | ❌ No | ✅ Yes |
| **Complexity** | Simple | Complex |

### Python Code Examples

```python
from tensorflow import keras
from tensorflow.keras import layers

# SEQUENTIAL API - Linear models
sequential_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# FUNCTIONAL API - Complex architectures
inputs = keras.Input(shape=(10,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

functional_model = keras.Model(inputs=inputs, outputs=outputs)
```

### When to Use Functional API
```python
# Multi-input model
input_a = keras.Input(shape=(32,), name='input_a')
input_b = keras.Input(shape=(64,), name='input_b')

x_a = layers.Dense(16)(input_a)
x_b = layers.Dense(16)(input_b)

# Merge branches
merged = layers.Concatenate()([x_a, x_b])
output = layers.Dense(1)(merged)

model = keras.Model(inputs=[input_a, input_b], outputs=output)
```

---

## Question 5

**What are activation functions and why are they important?**

### Answer

**Definition**: Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

### Common Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **ReLU** | max(0, x) | Hidden layers (default) |
| **Sigmoid** | 1/(1+e^-x) | Binary classification output |
| **Softmax** | e^xi / Σe^xj | Multi-class output |
| **Tanh** | (e^x - e^-x)/(e^x + e^-x) | RNN hidden layers |
| **LeakyReLU** | max(0.01x, x) | Prevent dead neurons |

### Python Code Example
```python
from tensorflow.keras import layers

# Different activation functions
model = keras.Sequential([
    # ReLU for hidden layers (most common)
    layers.Dense(64, activation='relu'),
    
    # LeakyReLU for preventing dead neurons
    layers.Dense(32),
    layers.LeakyReLU(alpha=0.1),
    
    # Output layer activations
    # Binary classification: sigmoid
    layers.Dense(1, activation='sigmoid'),
    
    # Multi-class classification: softmax
    # layers.Dense(10, activation='softmax'),
    
    # Regression: linear (no activation)
    # layers.Dense(1, activation='linear'),
])
```

### Why Important?
- **Without**: Network is just linear transformation
- **With**: Can approximate any function

---

## Question 6

**What is the purpose of the compile() method in Keras?**

### Answer

**Definition**: `compile()` configures the model for training by specifying optimizer, loss function, and metrics.

### Parameters

| Parameter | Purpose | Example |
|-----------|---------|---------|
| **optimizer** | Weight update algorithm | 'adam', 'sgd' |
| **loss** | Error measurement | 'mse', 'categorical_crossentropy' |
| **metrics** | Performance tracking | ['accuracy', 'auc'] |

### Python Code Example
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])

# Basic compile
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Advanced compile with custom settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanSquaredError(),
    metrics=[
        keras.metrics.MeanAbsoluteError(),
        keras.metrics.RootMeanSquaredError()
    ]
)
```

### Common Loss-Activation Pairs

| Task | Output Activation | Loss Function |
|------|-------------------|---------------|
| Binary classification | sigmoid | binary_crossentropy |
| Multi-class | softmax | categorical_crossentropy |
| Multi-class (int labels) | softmax | sparse_categorical_crossentropy |
| Regression | linear | mse |

---

## Question 7

**How does backpropagation work in Keras?**

### Answer

**Definition**: Backpropagation computes gradients of the loss with respect to weights by applying the chain rule backward through the network.

### Process

| Step | Action |
|------|--------|
| 1. Forward pass | Compute predictions |
| 2. Compute loss | Compare predictions to labels |
| 3. Backward pass | Compute gradients (chain rule) |
| 4. Update weights | Apply optimizer |

### Python Code Example
```python
import tensorflow as tf

# Keras handles backpropagation automatically
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10)  # Backprop happens here

# Manual backpropagation with GradientTape
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x, training=True)
        loss = tf.keras.losses.mse(y, predictions)
    
    # Backward pass - compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
```

### Key Points
- Keras abstracts backpropagation in `model.fit()`
- Use `tf.GradientTape` for custom training loops
- Gradients flow backward through all layers

---

## Question 8

**What is the difference between training and inference in Keras?**

### Answer

### Comparison

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Purpose** | Learn weights | Make predictions |
| **Dropout** | Active | Disabled |
| **BatchNorm** | Updates statistics | Uses learned stats |
| **Gradients** | Computed | Not needed |
| **Speed** | Slower | Faster |

### Python Code Example
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),  # Behaves differently
    keras.layers.BatchNormalization(),  # Behaves differently
    keras.layers.Dense(1)
])

# TRAINING MODE
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10)  # training=True

# INFERENCE MODE
predictions = model.predict(X_test)  # training=False (default)

# Explicit control
training_output = model(X_test, training=True)  # Dropout active
inference_output = model(X_test, training=False)  # Dropout inactive
```

### Key Differences
- **Dropout**: Randomly drops neurons during training only
- **BatchNorm**: Updates running statistics during training, uses them during inference

---

## Question 9

**What are regularization techniques available in Keras?**

### Answer

### Regularization Methods

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **L1/L2** | Penalize large weights | `kernel_regularizer` |
| **Dropout** | Random neuron dropping | `Dropout` layer |
| **BatchNorm** | Normalize activations | `BatchNormalization` |
| **Early Stopping** | Stop before overfit | Callback |
| **Data Augmentation** | Increase data variety | `ImageDataGenerator` |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers, regularizers

model = keras.Sequential([
    # L2 regularization on weights
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01),
                 input_shape=(784,)),
    
    # Dropout
    layers.Dropout(0.5),
    
    # Batch Normalization
    layers.BatchNormalization(),
    
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    
    layers.Dense(10, activation='softmax')
])

# Early stopping callback
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, validation_split=0.2, 
          epochs=100, callbacks=[early_stop])
```

---

## Question 10

**What is the role of callbacks in Keras?**

### Answer

**Definition**: Callbacks are functions called at specific points during training to customize behavior.

### Common Callbacks

| Callback | Purpose |
|----------|---------|
| `ModelCheckpoint` | Save model during training |
| `EarlyStopping` | Stop when no improvement |
| `TensorBoard` | Visualization logging |
| `ReduceLROnPlateau` | Reduce learning rate |
| `LearningRateScheduler` | Custom LR schedule |

### Python Code Example
```python
from tensorflow import keras

# Define callbacks
callbacks = [
    # Save best model
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        save_best_only=True,
        monitor='val_loss'
    ),
    
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    
    # Reduce LR when plateau
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    ),
    
    # TensorBoard
    keras.callbacks.TensorBoard(log_dir='./logs')
]

# Use in training
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=callbacks
)
```

### Custom Callback
```python
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            print("\n95% accuracy reached!")
            self.model.stop_training = True
```

