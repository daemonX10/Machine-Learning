# TensorFlow Interview Questions - General Questions

## Question 1

**What types of devices does TensorFlow support for computation?**

### Answer

### Supported Devices

| Device | Description | Use Case |
|--------|-------------|----------|
| **CPU** | Default, available everywhere | Small models, debugging |
| **GPU** | NVIDIA CUDA support | Training deep learning models |
| **TPU** | Google's custom hardware | Large-scale training |
| **Mobile** | TFLite on Android/iOS | On-device inference |
| **Browser** | TensorFlow.js | Web-based ML |
| **Microcontrollers** | TFLite Micro | Edge devices |

### Python Code Example
```python
import tensorflow as tf

# List available devices
devices = tf.config.list_physical_devices()
print("Available devices:", devices)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

# Place operations on specific device
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)

# Enable memory growth (prevents TF from taking all GPU memory)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## Question 2

**How do you create a simple neural network in TensorFlow?**

### Answer

### Methods to Build Models

| Method | Use Case |
|--------|----------|
| **Sequential** | Linear stack of layers |
| **Functional** | Complex architectures (multi-input/output) |
| **Subclassing** | Custom behavior |

### Python Code Example
```python
import tensorflow as tf

# Method 1: Sequential API (simplest)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
# model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Summary
model.summary()
```

### Functional API (for complex models)
```python
# Method 2: Functional API
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

---

## Question 3

**What is tf.function and why is it important?**

### Answer

**Definition**: `tf.function` is a decorator that converts Python functions into TensorFlow graphs for better performance.

### Benefits

| Benefit | Description |
|---------|-------------|
| **Speed** | Graph optimization and compilation |
| **Portability** | Export graphs without Python |
| **Parallelism** | Operations can run in parallel |

### Python Code Example
```python
import tensorflow as tf
import time

# Without tf.function (eager mode)
def eager_function(x):
    return x ** 2 + 2 * x + 1

# With tf.function (graph mode)
@tf.function
def graph_function(x):
    return x ** 2 + 2 * x + 1

# Benchmark
x = tf.random.normal([1000, 1000])

# Eager execution
start = time.time()
for _ in range(100):
    eager_function(x)
eager_time = time.time() - start

# Graph execution
start = time.time()
for _ in range(100):
    graph_function(x)
graph_time = time.time() - start

print(f"Eager: {eager_time:.4f}s")
print(f"Graph: {graph_time:.4f}s")
print(f"Speedup: {eager_time/graph_time:.2f}x")
```

### When to Use
- Production inference
- Training loops that run many times
- Exporting models

---

## Question 4

**How do you save and load models in TensorFlow?**

### Answer

### Saving Methods

| Method | Format | Use Case |
|--------|--------|----------|
| `model.save()` | SavedModel or H5 | Complete model |
| `model.save_weights()` | Checkpoint | Weights only |
| `tf.saved_model.save()` | SavedModel | Production deployment |

### Python Code Example
```python
import tensorflow as tf

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Method 1: Save entire model (recommended)
model.save('my_model')  # SavedModel format (directory)
model.save('my_model.h5')  # HDF5 format (single file)

# Load entire model
loaded_model = tf.keras.models.load_model('my_model')

# Method 2: Save weights only
model.save_weights('weights/my_weights')
model.load_weights('weights/my_weights')

# Method 3: Checkpoints during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/epoch_{epoch:02d}',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss'
)
# model.fit(..., callbacks=[checkpoint])
```

---

## Question 5

**What is TensorBoard and how do you use it?**

### Answer

**Definition**: TensorBoard is TensorFlow's visualization toolkit for monitoring training, viewing model graphs, and analyzing performance.

### Features

| Feature | Description |
|---------|-------------|
| **Scalars** | Track loss, accuracy over time |
| **Graphs** | Visualize model architecture |
| **Histograms** | Weight distributions |
| **Images** | View input/output images |
| **Profiler** | Performance analysis |

### Python Code Example
```python
import tensorflow as tf
import datetime

# Create log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # Log weight histograms
    write_graph=True,  # Log model graph
    write_images=True  # Log weight images
)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train with TensorBoard
# model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback])

# Custom logging with tf.summary
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
    tf.summary.scalar('custom_metric', 0.5, step=1)
```

### Launch TensorBoard
```bash
tensorboard --logdir logs/fit
# Open http://localhost:6006 in browser
```

---

## Question 6

**How do you handle datasets in TensorFlow?**

### Answer

**Definition**: `tf.data.Dataset` is TensorFlow's API for building efficient input pipelines.

### Key Methods

| Method | Description |
|--------|-------------|
| `from_tensor_slices()` | Create from NumPy/tensors |
| `batch()` | Group into batches |
| `shuffle()` | Randomize order |
| `prefetch()` | Load data while training |
| `map()` | Apply transformations |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# Create dataset from arrays
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randint(0, 2, 1000).astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Build efficient pipeline
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Overlap data loading with training

# Use in training
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
# model.fit(dataset, epochs=5)

# Data augmentation with map
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label

# image_dataset = image_dataset.map(augment)
```

---

## Question 7

**What are callbacks in TensorFlow/Keras?**

### Answer

**Definition**: Callbacks are functions called at specific points during training to extend functionality.

### Common Callbacks

| Callback | Purpose |
|----------|---------|
| `ModelCheckpoint` | Save model during training |
| `EarlyStopping` | Stop when metric stops improving |
| `TensorBoard` | Logging for visualization |
| `ReduceLROnPlateau` | Reduce learning rate |
| `LearningRateScheduler` | Custom LR schedule |

### Python Code Example
```python
import tensorflow as tf

# Checkpoint: save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Early stopping: prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Reduce learning rate when plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# TensorBoard logging
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')

# Use in training
callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]
# model.fit(X_train, y_train, validation_split=0.2, 
#           epochs=100, callbacks=callbacks)
```

### Custom Callback
```python
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy, stopping!")
            self.model.stop_training = True
```

---

## Question 8

**How do you implement transfer learning in TensorFlow?**

### Answer

**Definition**: Transfer learning uses a pre-trained model as a starting point for a new task.

### Steps
1. Load pre-trained model (without top layers)
2. Freeze base layers
3. Add new classification layers
4. Train on new data

### Python Code Example
```python
import tensorflow as tf

# Load pre-trained model (without classification layers)
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,  # Remove classification head
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Build new model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile with low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning (optional): unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all but last 20
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Question 9

**What is distributed training in TensorFlow?**

### Answer

**Definition**: Distributed training spreads model training across multiple GPUs or machines for faster training.

### Strategies

| Strategy | Description |
|----------|-------------|
| `MirroredStrategy` | Multiple GPUs, single machine |
| `MultiWorkerMirroredStrategy` | Multiple machines |
| `TPUStrategy` | Google TPUs |
| `ParameterServerStrategy` | Large-scale distributed |

### Python Code Example
```python
import tensorflow as tf

# Strategy: Single machine, multiple GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Build model within strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss='mse'
    )

# Train as usual - distribution happens automatically
# model.fit(X_train, y_train, epochs=10, batch_size=64*strategy.num_replicas_in_sync)

# For TPU
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)
```

### Key Points
- Batch size scales with number of devices
- Data is automatically sharded
- Gradients are synchronized

---

## Question 10

**How do you handle overfitting in TensorFlow models?**

### Answer

### Techniques

| Technique | Implementation |
|-----------|----------------|
| **Dropout** | `tf.keras.layers.Dropout` |
| **L1/L2 Regularization** | `kernel_regularizer` |
| **Early Stopping** | Callback |
| **Data Augmentation** | `tf.image` transforms |
| **Batch Normalization** | `tf.keras.layers.BatchNormalization` |

### Python Code Example
```python
import tensorflow as tf

# Model with regularization techniques
model = tf.keras.Sequential([
    # L2 regularization on weights
    tf.keras.layers.Dense(128, activation='relu', 
                          kernel_regularizer=tf.keras.regularizers.l2(0.01),
                          input_shape=(10,)),
    
    # Batch normalization
    tf.keras.layers.BatchNormalization(),
    
    # Dropout
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation for images
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])
```

### Summary
- Use **dropout** (0.2-0.5) after dense layers
- Add **L2 regularization** for weight penalty
- Implement **early stopping** to prevent overtraining
- Use **data augmentation** to increase training variety

