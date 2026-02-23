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



---

## Question 11

**Can you explain the difference between validation and test sets in the context of a Keras model?**

### Answer

**Definition**: Validation and test sets serve different purposes in model evaluation. The **validation set** is used during training to tune hyperparameters and monitor overfitting, while the **test set** is used only once at the end to provide an unbiased estimate of final model performance.

### Key Differences

| Aspect | Validation Set | Test Set |
|--------|---------------|----------|
| **When Used** | During training (each epoch) | After training is complete |
| **Purpose** | Tune hyperparameters, monitor overfitting | Final unbiased evaluation |
| **Influence on Model** | Indirectly (via early stopping, LR scheduling) | None |
| **Typical Split** | 10-20% of training data | 10-20% held out separately |
| **Frequency of Use** | Every epoch | Once |

### Python Code Example
```python
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Split data into train, validation, and test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Method 1: Explicit validation data
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=5)])

# Method 2: Automatic validation split
history = model.fit(X_train_full, y_train_full,
                    validation_split=0.2,
                    epochs=50)

# Final evaluation on TEST set (only once!)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### Interview Tips
- Never use the test set for hyperparameter tuning — that causes data leakage
- Use `validation_split` for quick experiments, explicit `validation_data` for reproducibility
- Monitor `val_loss` vs `train_loss` to detect overfitting

---

## Question 12

**What is the importance of data preprocessing in training Keras models?**

### Answer

**Definition**: Data preprocessing transforms raw data into a format that neural networks can efficiently learn from. Proper preprocessing is critical because neural networks are sensitive to the scale, distribution, and quality of input data.

### Common Preprocessing Techniques

| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| **Normalization** | Scale features to [0, 1] | Image pixel values |
| **Standardization** | Mean=0, Std=1 | Tabular data with varying scales |
| **One-Hot Encoding** | Convert categories to vectors | Categorical labels |
| **Tokenization** | Convert text to sequences | NLP tasks |
| **Handling Missing Values** | Fill or remove NaNs | Incomplete datasets |
| **Feature Scaling** | Uniform feature ranges | Mixed-scale features |

### Python Code Example
```python
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. NUMERICAL DATA - Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)         # Transform test

# 2. IMAGE DATA - Normalization
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0   # Scale to [0, 1]
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)       # Reshape for Conv2D

# 3. CATEGORICAL LABELS - One-Hot Encoding
y_train_onehot = keras.utils.to_categorical(y_train, num_classes=10)

# 4. TEXT DATA - Tokenization
tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)
X_train_seq = tokenizer.texts_to_sequences(train_texts)
X_train_padded = keras.preprocessing.sequence.pad_sequences(
    X_train_seq, maxlen=200
)

# 5. USING KERAS PREPROCESSING LAYERS (built into model)
preprocessing_model = keras.Sequential([
    keras.layers.Normalization(),  # Learns mean/std from data
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])
# Adapt normalization layer to training data
preprocessing_model.layers[0].adapt(X_train)
```

### Why It Matters
- **Faster convergence**: Scaled inputs help gradient descent converge faster
- **Numerical stability**: Prevents exploding/vanishing gradients
- **Equal feature importance**: Prevents features with large ranges from dominating
- **Model accuracy**: Clean, well-formatted data leads to better generalization

### Interview Tips
- Always fit preprocessors on training data only, then transform both train and test
- Keras provides built-in preprocessing layers (`Normalization`, `Rescaling`, `TextVectorization`) that can be embedded directly in the model
- Mention that preprocessing is part of the data pipeline and should be reproducible

---

## Question 13

**Can you describe the concept of hyperparameter tuning and its importance in Keras models?**

### Answer

**Definition**: Hyperparameter tuning is the process of finding the optimal configuration of parameters that are set before training (not learned from data). These include learning rate, number of layers, units per layer, batch size, dropout rate, etc.

### Hyperparameters vs Parameters

| Aspect | Hyperparameters | Parameters (Weights) |
|--------|----------------|---------------------|
| **Set By** | Engineer | Training process |
| **Examples** | Learning rate, batch size | Weights, biases |
| **When Set** | Before training | During training |
| **Tuning Method** | Search strategies | Gradient descent |

### Common Hyperparameters in Keras

| Hyperparameter | Typical Range | Impact |
|----------------|---------------|--------|
| **Learning rate** | 1e-5 to 1e-1 | Convergence speed/stability |
| **Batch size** | 16, 32, 64, 128 | Memory usage, generalization |
| **Number of layers** | 1-10+ | Model capacity |
| **Units per layer** | 32-1024 | Representation power |
| **Dropout rate** | 0.1-0.5 | Regularization strength |
| **Optimizer** | Adam, SGD, RMSprop | Training dynamics |

### Python Code Example
```python
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers

# Define a model-building function with tunable hyperparameters
def build_model(hp):
    model = keras.Sequential()
    
    # Tune number of layers
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(
            units=hp.Choice(f'units_{i}', [32, 64, 128, 256]),
            activation='relu'
        ))
        model.add(layers.Dropout(
            rate=hp.Float('dropout', 0.1, 0.5, step=0.1)
        ))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Use Keras Tuner for search
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    directory='tuner_results',
    project_name='keras_tuning'
)

tuner.search(X_train, y_train, validation_split=0.2, epochs=50)

# Get best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
best_model = tuner.get_best_models()[0]
print(f"Best learning rate: {best_hp.get('lr')}")
```

### Tuning Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Grid Search** | Try all combinations | Thorough | Expensive |
| **Random Search** | Random combinations | Efficient | May miss optimal |
| **Bayesian** | Informed search | Smart | Complex |
| **Hyperband** | Adaptive resource allocation | Fast | Requires early stopping |

### Interview Tips
- Keras Tuner (`keras-tuner` package) is the recommended tool for hyperparameter optimization in Keras
- Start with random search to narrow the range, then use Bayesian optimization for fine-tuning
- Always use a validation set for tuning — never the test set

---

## Question 14

**How does learning rate affect the training of a Keras model?**

### Answer

**Definition**: The learning rate controls the step size at which the optimizer updates model weights during gradient descent. It is arguably the most important hyperparameter in deep learning.

### Learning Rate Effects

| Learning Rate | Behavior | Outcome |
|---------------|----------|---------|
| **Too high** | Large weight updates, oscillation | Divergence, NaN loss |
| **Too low** | Tiny weight updates | Very slow convergence, stuck in local minima |
| **Just right** | Balanced updates | Smooth convergence to good minimum |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# 1. FIXED LEARNING RATE
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 2. LEARNING RATE SCHEDULING - Reduce on plateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # Multiply LR by 0.5
    patience=3,        # Wait 3 epochs with no improvement
    min_lr=1e-7
)

# 3. EXPONENTIAL DECAY SCHEDULE
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# 4. LEARNING RATE FINDER (manual approach)
import numpy as np

lrs = np.logspace(-6, -1, 100)
losses = []
for lr in lrs:
    model_tmp = keras.models.clone_model(model)
    model_tmp.compile(optimizer=keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy')
    history = model_tmp.fit(X_train, y_train, epochs=1, batch_size=128, verbose=0)
    losses.append(history.history['loss'][0])

# Plot losses vs learning rates to find optimal LR
# Best LR is typically one order of magnitude before the minimum loss

# 5. COSINE ANNEALING
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.01,
    decay_steps=10000
)
```

### Learning Rate Schedules

| Schedule | Description | Best For |
|----------|-------------|----------|
| **Constant** | Fixed throughout | Baseline experiments |
| **Step Decay** | Reduce at fixed intervals | Standard training |
| **Exponential Decay** | Continuous reduction | Smooth convergence |
| **Cosine Annealing** | Cosine-shaped decay | State-of-the-art results |
| **ReduceLROnPlateau** | Reduce when stuck | Adaptive training |
| **Warm-up** | Start low, increase, then decay | Transformer models |

### Interview Tips
- Start with Adam optimizer and `lr=0.001` as a baseline
- Use `ReduceLROnPlateau` callback for automatic adjustment
- Mention the learning rate finder technique to empirically determine the best initial LR
- Warm-up schedules are critical for large models and Transformers

---

## Question 15

**Explain how you would fine-tune a pre-trained model in Keras.**

### Answer

**Definition**: Fine-tuning is the process of taking a model pre-trained on a large dataset (e.g., ImageNet) and adapting it to a new, typically smaller, dataset by selectively unfreezing and retraining some layers.

### Fine-Tuning Strategy

| Phase | Action | Learning Rate |
|-------|--------|---------------|
| **1. Feature Extraction** | Freeze base, train new head | Higher (1e-3) |
| **2. Fine-Tuning** | Unfreeze top layers of base | Very low (1e-5) |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Load pre-trained model WITHOUT top classification layer
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Step 2: Freeze all layers in the base model
base_model.trainable = False

# Step 3: Add custom classification head
inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.resnet50.preprocess_input(inputs)
x = base_model(x, training=False)  # Keep BatchNorm in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Step 4: Train only the new head (feature extraction)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Step 5: Unfreeze top layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
    layer.trainable = False

# Step 6: Recompile with VERY low learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

### Available Pre-Trained Models in Keras

| Model | Parameters | Use Case |
|-------|-----------|----------|
| **VGG16/19** | 138M/144M | Simple baseline |
| **ResNet50** | 25M | General purpose |
| **InceptionV3** | 24M | Balanced accuracy/speed |
| **EfficientNet** | 5-66M | Best accuracy/efficiency |
| **MobileNetV2** | 3.4M | Mobile/edge deployment |

### Interview Tips
- Always freeze the base model first, train the head, then unfreeze top layers gradually
- Use a much lower learning rate for fine-tuning (10-100x smaller) to avoid destroying learned features
- Keep BatchNormalization layers in inference mode during fine-tuning by passing `training=False`
- Fine-tuning works best when your dataset is similar to the pre-training dataset

---

## Question 16

**What is the use of a grid search in hyperparameter optimization and can it be used with Keras?**

### Answer

**Definition**: Grid search is an exhaustive search method that evaluates all possible combinations of specified hyperparameter values to find the best configuration. It can be used with Keras via scikit-learn's `GridSearchCV` wrapper.

### How Grid Search Works

| Step | Action |
|------|--------|
| 1 | Define parameter grid (all values to try) |
| 2 | Generate all combinations |
| 3 | Train model for each combination with cross-validation |
| 4 | Evaluate and rank by validation score |
| 5 | Return best combination |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# Step 1: Define model-building function
def build_model(units=64, dropout_rate=0.2, learning_rate=0.001):
    model = keras.Sequential([
        layers.Dense(units, activation='relu', input_shape=(784,)),
        layers.Dropout(dropout_rate),
        layers.Dense(units // 2, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Step 2: Wrap Keras model for scikit-learn compatibility
keras_clf = KerasClassifier(
    model=build_model,
    epochs=20,
    batch_size=32,
    verbose=0
)

# Step 3: Define parameter grid
param_grid = {
    'model__units': [64, 128, 256],
    'model__dropout_rate': [0.2, 0.3, 0.5],
    'model__learning_rate': [0.001, 0.01],
    'batch_size': [32, 64]
}

# Step 4: Run GridSearchCV
grid_search = GridSearchCV(
    estimator=keras_clf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,     # Keras models should use n_jobs=1
    verbose=2
)
grid_result = grid_search.fit(X_train, y_train)

# Step 5: Results
print(f"Best accuracy: {grid_result.best_score_:.4f}")
print(f"Best params: {grid_result.best_params_}")
best_model = grid_result.best_estimator_.model_
```

### Grid Search vs Alternatives

| Method | Combinations Tested | Scalability | Best For |
|--------|-------------------|-------------|----------|
| **Grid Search** | All (exhaustive) | Poor for many params | Few hyperparameters |
| **Random Search** | Random subset | Good | Many hyperparameters |
| **Bayesian (Keras Tuner)** | Informed selection | Excellent | Complex search spaces |

### Interview Tips
- Grid search is exhaustive but computationally expensive — grows exponentially with parameters
- Use `scikeras.wrappers.KerasClassifier` (or `KerasRegressor`) to integrate Keras with scikit-learn
- For large search spaces, prefer Random Search or Keras Tuner's Bayesian optimization
- Always use cross-validation (`cv` parameter) to get robust results

---

## Question 17

**Explain how you would use data augmentation in Keras.**

### Answer

**Definition**: Data augmentation artificially increases the diversity of training data by applying random transformations (rotation, flipping, zooming, etc.) to existing samples. This helps reduce overfitting and improves model generalization, especially with limited training data.

### Common Augmentation Techniques

| Technique | Description | Parameter |
|-----------|-------------|-----------|
| **Rotation** | Random rotation | `rotation_range=20` |
| **Horizontal Flip** | Mirror image | `horizontal_flip=True` |
| **Width/Height Shift** | Translate image | `width_shift_range=0.2` |
| **Zoom** | Random zoom in/out | `zoom_range=0.2` |
| **Shear** | Slant transformation | `shear_range=0.2` |
| **Brightness** | Random brightness | `brightness_range=[0.8, 1.2]` |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# METHOD 1: ImageDataGenerator (classic approach)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.15,
    fill_mode='nearest'
)

# Fit the generator on training data
datagen.fit(X_train)

# Train with augmented data
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=50,
    steps_per_epoch=len(X_train) // 32
)

# METHOD 2: Keras Preprocessing Layers (modern, recommended)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
])

# Embed augmentation directly in the model
model = keras.Sequential([
    # Augmentation layers (active only during training)
    data_augmentation,
    
    # Model layers
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# METHOD 3: tf.data pipeline with augmentation
import tensorflow as tf

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(augment).batch(32).prefetch(tf.data.AUTOTUNE)
```

### Interview Tips
- Keras preprocessing layers are preferred over `ImageDataGenerator` (which is legacy)
- Augmentation layers are only active during `training=True` — inference is unaffected
- Always augment only the training set, never validation or test
- Data augmentation is most impactful when training data is limited

---

## Question 18

**How does Keras handle sequence data for tasks like text generation or translation?**

### Answer

**Definition**: Keras provides specialized recurrent layers (LSTM, GRU), embedding layers, and preprocessing utilities to handle variable-length sequence data for tasks like text generation, machine translation, and time series forecasting.

### Sequence Processing Pipeline

| Step | Tool | Purpose |
|------|------|---------|
| 1. Tokenization | `Tokenizer` / `TextVectorization` | Convert text to integer sequences |
| 2. Padding | `pad_sequences` | Ensure uniform input length |
| 3. Embedding | `Embedding` layer | Dense word representations |
| 4. Sequence Modeling | LSTM / GRU / Transformer | Learn sequence patterns |
| 5. Decoding | Dense + softmax | Generate output |

### Python Code Example — Text Generation
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Prepare text data
text = "your training corpus here..."
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
total_chars = len(tokenizer.word_index) + 1

# Create input sequences
encoded = tokenizer.texts_to_sequences([text])[0]
sequences = []
for i in range(len(encoded) - 40):
    sequences.append(encoded[i:i+41])
sequences = np.array(sequences)

X = sequences[:, :-1]  # Input: first 40 chars
y = sequences[:, -1]   # Target: next char

# Text generation model
model = keras.Sequential([
    layers.Embedding(total_chars, 64, input_length=40),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(128),
    layers.Dense(total_chars, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=50, batch_size=128)
```

### Sequence-to-Sequence (Translation) Model
```python
# Encoder
encoder_inputs = keras.Input(shape=(None,))
enc_emb = layers.Embedding(src_vocab_size, 256)(encoder_inputs)
encoder_lstm = layers.LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = keras.Input(shape=(None,))
dec_emb = layers.Embedding(tgt_vocab_size, 256)(decoder_inputs)
decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = layers.Dense(tgt_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Full model
seq2seq_model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
seq2seq_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

### Key Recurrent Layers

| Layer | Description | Best For |
|-------|-------------|----------|
| **SimpleRNN** | Basic recurrence | Short sequences |
| **LSTM** | Long short-term memory | Long-range dependencies |
| **GRU** | Gated recurrent unit | Faster LSTM alternative |
| **Bidirectional** | Process both directions | Classification tasks |

### Interview Tips
- Use `return_sequences=True` when stacking RNN layers (except the last)
- `return_state=True` is essential for seq2seq architectures to pass encoder states to decoder
- Modern Keras also supports Transformer-based approaches via `MultiHeadAttention` layer
- For production text tasks, use `TextVectorization` preprocessing layer instead of the legacy `Tokenizer`

---

## Question 19

**Explain the use of attention mechanisms in Keras models.**

### Answer

**Definition**: Attention mechanisms allow a model to selectively focus on the most relevant parts of the input sequence when producing each output element, rather than relying on a fixed-length context vector. This dramatically improves performance on long sequences.

### Types of Attention

| Type | Description | Use Case |
|------|-------------|----------|
| **Bahdanau (Additive)** | Learned alignment via MLP | Seq2seq models |
| **Luong (Multiplicative)** | Dot product attention | Efficient alignment |
| **Self-Attention** | Attend to own sequence | Transformers |
| **Multi-Head Attention** | Multiple parallel attention heads | Transformers, BERT, GPT |

### Python Code Example — Multi-Head Attention
```python
from tensorflow import keras
from tensorflow.keras import layers

# Transformer-style self-attention block
def transformer_block(embed_dim, num_heads, ff_dim, dropout=0.1):
    inputs = keras.Input(shape=(None, embed_dim))
    
    # Multi-Head Self-Attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads
    )(inputs, inputs)  # Self-attention: query=key=value=inputs
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization()(inputs + attn_output)  # Residual
    
    # Feed-Forward Network
    ff_output = layers.Dense(ff_dim, activation='relu')(out1)
    ff_output = layers.Dense(embed_dim)(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    out2 = layers.LayerNormalization()(out1 + ff_output)  # Residual
    
    return keras.Model(inputs, out2)

# Text classification with attention
maxlen = 200
vocab_size = 20000
embed_dim = 64
num_heads = 4

inputs = keras.Input(shape=(maxlen,))
x = layers.Embedding(vocab_size, embed_dim)(inputs)

# Apply attention
attn_output = layers.MultiHeadAttention(
    num_heads=num_heads, key_dim=embed_dim // num_heads
)(x, x)
x = layers.LayerNormalization()(x + attn_output)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Custom Attention Layer
```python
class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        # query: (batch, hidden_size) -> decoder hidden state
        # values: (batch, seq_len, hidden_size) -> encoder outputs
        query_expanded = tf.expand_dims(query, 1)
        
        # Alignment scores
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_expanded)))
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context = tf.reduce_sum(attention_weights * values, axis=1)
        return context, attention_weights
```

### Interview Tips
- `MultiHeadAttention` is the built-in Keras layer for attention (available since TF 2.4+)
- Self-attention means query, key, and value all come from the same input
- Attention eliminates the information bottleneck of fixed-size context vectors in seq2seq
- Transformer architecture is entirely built on multi-head self-attention — no recurrence needed

---

## Question 20

**What are the challenges associated with training very deep networks in Keras and how can you overcome them?**

### Answer

**Definition**: Very deep networks (dozens to hundreds of layers) face unique challenges including vanishing/exploding gradients, degradation problems, computational cost, and difficulty in optimization. Keras provides several tools and techniques to address these.

### Challenges and Solutions

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Vanishing Gradients** | Gradients shrink to ~0 in early layers | BatchNorm, residual connections, ReLU |
| **Exploding Gradients** | Gradients grow extremely large | Gradient clipping, proper initialization |
| **Degradation Problem** | Deeper ≠ better accuracy | Skip/residual connections |
| **Slow Training** | Many parameters, long epochs | GPU/TPU, mixed precision |
| **Overfitting** | Too many parameters | Dropout, regularization, data augmentation |
| **Memory Issues** | Large model doesn't fit in GPU | Gradient checkpointing, smaller batches |

### Python Code Example
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# SOLUTION 1: Residual Connections (Skip Connections)
def residual_block(x, filters):
    shortcut = x
    
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection: add input to output
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1)(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Build deep network with residual blocks
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
for _ in range(20):  # 20 residual blocks = very deep
    x = residual_block(x, 64)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# SOLUTION 2: Batch Normalization (applied above in residual_block)
# Normalizes activations, stabilizes training

# SOLUTION 3: Gradient Clipping
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0      # Clip gradients by norm
    # clipvalue=0.5    # Or clip by value
)

# SOLUTION 4: Proper Weight Initialization
layer = layers.Dense(256, activation='relu',
                     kernel_initializer='he_normal')  # He init for ReLU

# SOLUTION 5: Mixed Precision Training (speed + memory)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# SOLUTION 6: Learning Rate Warm-up
class WarmUpSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Interview Tips
- Residual connections (ResNet) solved the degradation problem — deeper networks can now perform at least as well as shallower ones
- Batch Normalization + He initialization + ReLU is the standard recipe for deep networks
- Mixed precision (`float16`) can cut memory usage and training time nearly in half
- Gradient clipping prevents exploding gradients, especially important for RNNs and Transformers

---

## Question 21

**What are some common issues you might face when working with Keras and how do you resolve them?**

### Answer

### Common Issues and Resolutions

| Issue | Symptoms | Resolution |
|-------|----------|------------|
| **Overfitting** | High train acc, low val acc | Dropout, regularization, more data |
| **Underfitting** | Low train and val acc | Larger model, more epochs, lower regularization |
| **NaN Loss** | Loss becomes NaN | Lower learning rate, check data for NaN, gradient clipping |
| **OOM Error** | GPU out of memory | Reduce batch size, mixed precision, smaller model |
| **Slow Training** | Epochs take too long | GPU acceleration, `tf.data` pipeline, reduce model size |
| **Shape Mismatch** | Dimension errors | Check `model.summary()`, verify input shapes |
| **Import Errors** | Version conflicts | Ensure compatible TF/Keras versions |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ISSUE 1: Overfitting → Use regularization
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(100,),
                 kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),           # Dropout for regularization
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

# ISSUE 2: NaN Loss → Check data and use gradient clipping
# Check for NaN/Inf in data
assert not np.any(np.isnan(X_train)), "Data contains NaN!"
assert not np.any(np.isinf(X_train)), "Data contains Inf!"

optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy')

# ISSUE 3: OOM → Reduce batch size or use mixed precision
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Use generator to load data in batches
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

# ISSUE 4: Shape Mismatch → Debug with summary
model.summary()  # Check expected shapes
print(f"Input shape: {X_train.shape}")
print(f"Output shape: {y_train.shape}")

# ISSUE 5: Slow Data Pipeline → Use tf.data
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

# ISSUE 6: Reproducibility → Set seeds
tf.random.set_seed(42)
np.random.seed(42)

# ISSUE 7: Model not learning → Check learning rate
# Use LearningRateScheduler or ReduceLROnPlateau
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                  restore_best_weights=True)
]
```

### Debugging Checklist
1. **Check data**: Look for NaN, correct shapes, proper normalization
2. **Check model**: `model.summary()` to verify architecture
3. **Check loss/activation pairing**: sigmoid→binary_crossentropy, softmax→categorical_crossentropy
4. **Start simple**: Get a small model working first, then scale up
5. **Monitor training**: Use TensorBoard or plot `history.history` curves

### Interview Tips
- Always start debugging with `model.summary()` and data inspection
- NaN loss is usually caused by too high a learning rate, bad data, or numerical instability
- OOM errors are best resolved by reducing batch size first before changing the model
- Mention systematic debugging approach: data → model → training → evaluation

---

## Question 22

**Describe the process of serving a Keras model using TensorFlow Serving.**

### Answer

**Definition**: TensorFlow Serving is a production-grade system for serving machine learning models via REST or gRPC APIs. It supports model versioning, batching, and hot-swapping models without downtime.

### Serving Pipeline

| Step | Action | Tool |
|------|--------|------|
| 1 | Train and save model | Keras `model.save()` |
| 2 | Export as SavedModel | TensorFlow SavedModel format |
| 3 | Deploy to TF Serving | Docker or native install |
| 4 | Send inference requests | REST API or gRPC |

### Python Code Example
```python
import tensorflow as tf
from tensorflow import keras
import json
import requests
import numpy as np

# Step 1: Train a Keras model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Step 2: Save as TF SavedModel (versioned directory)
export_path = 'saved_models/my_model/1'  # Version 1
model.save(export_path)

# Inspect the saved model
print(f"SavedModel saved to: {export_path}")
loaded = tf.saved_model.load(export_path)
print(f"Signatures: {list(loaded.signatures.keys())}")
```

### Deploy with Docker
```bash
# Step 3: Run TensorFlow Serving in Docker
docker pull tensorflow/serving

docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/saved_models/my_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  tensorflow/serving

# Model is now available at:
# REST: http://localhost:8501/v1/models/my_model:predict
# gRPC: localhost:8500
```

### Send Prediction Requests
```python
# Step 4: Send REST API request
import requests
import json
import numpy as np

# Prepare input data
data = json.dumps({
    "signature_name": "serving_default",
    "instances": X_test[:3].tolist()
})

# Send POST request
headers = {"content-type": "application/json"}
response = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    data=data,
    headers=headers
)

predictions = json.loads(response.text)['predictions']
print(f"Predictions: {np.argmax(predictions, axis=1)}")

# Check model status
status = requests.get('http://localhost:8501/v1/models/my_model')
print(f"Model status: {status.json()}")
```

### Model Versioning
```python
# Save version 2
model_v2.save('saved_models/my_model/2')

# Directory structure:
# saved_models/
#   my_model/
#     1/           <- version 1
#       saved_model.pb
#       variables/
#     2/           <- version 2 (auto-detected)
#       saved_model.pb
#       variables/
```

### Interview Tips
- TF Serving auto-detects new model versions and hot-swaps them without downtime
- SavedModel format is required (not `.h5`) — use `model.save('path/')` without extension
- REST API (port 8501) is simpler; gRPC (port 8500) is faster for production
- For Kubernetes deployments, TF Serving integrates with TFX pipelines
- Mention batching support for throughput optimization in production

---

## Question 23

**How does reinforcement learning work in Keras?**

### Answer

**Definition**: Reinforcement Learning (RL) trains an agent to make sequential decisions by maximizing cumulative reward through interaction with an environment. Keras is used to build the neural network components (policy networks, value networks, Q-networks) that drive RL algorithms.

### RL Components with Keras

| Component | Role | Keras Implementation |
|-----------|------|---------------------|
| **Agent** | Decision maker | Neural network model |
| **Environment** | World the agent interacts with | OpenAI Gym / Gymnasium |
| **Policy Network** | Maps states → actions | Dense/Conv model |
| **Value Network** | Estimates state value | Dense model |
| **Q-Network** | Estimates action values | Dense/Conv model (DQN) |

### Python Code Example — Deep Q-Network (DQN)
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random

# Build Q-Network
def build_q_network(state_size, action_size):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(state_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear')  # Q-value per action
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99       # Discount factor
        self.epsilon = 1.0      # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Main and target networks
        self.model = build_q_network(state_size, action_size)
        self.target_model = build_q_network(state_size, action_size)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([s[0][0] for s in batch])
        next_states = np.array([s[3][0] for s in batch])
        
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = reward
            if not done:
                target += self.gamma * np.max(next_q[i])
            current_q[i][action] = target
        
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training loop (with Gymnasium)
# import gymnasium as gym
# env = gym.make('CartPole-v1')
# agent = DQNAgent(state_size=4, action_size=2)
# for episode in range(500):
#     state = env.reset()[0].reshape(1, -1)
#     total_reward = 0
#     for step in range(500):
#         action = agent.act(state)
#         next_state, reward, done, _, _ = env.step(action)
#         next_state = next_state.reshape(1, -1)
#         agent.remember(state, action, reward, next_state, done)
#         agent.replay()
#         state = next_state
#         total_reward += reward
#         if done:
#             break
#     agent.update_target_model()
```

### Common RL Algorithms Using Keras

| Algorithm | Type | Description |
|-----------|------|-------------|
| **DQN** | Value-based | Q-learning with neural network |
| **Policy Gradient** | Policy-based | Directly optimize policy |
| **A2C/A3C** | Actor-Critic | Combines policy and value networks |
| **PPO** | Actor-Critic | Stable policy optimization |

### Interview Tips
- Keras handles only the neural network part — the RL algorithm logic is separate
- Use separate main and target networks for stable DQN training
- For production RL, consider libraries like TF-Agents, Stable-Baselines3, or RLlib that wrap Keras models
- Key challenge: RL requires millions of environment interactions — training is much slower than supervised learning

---

## Question 24

**Describe how you would install and set up Keras in a Python environment**

*Answer to be added.*

---

## Question 25

**What are some advantages of using Keras over other deep learning frameworks ?**

*Answer to be added.*

---

## Question 26

**How do you save and load models in Keras ?**

*Answer to be added.*

---

## Question 27

**What is the purpose of the Dense layer in Keras ?**

*Answer to be added.*

---

## Question 28

**How would you implement a Convolutional Neural Network in Keras ?**

*Answer to be added.*

---

## Question 29

**Can you describe how Recurrent Neural Networks are different and how to implement one in Keras ?**

*Answer to be added.*

---

## Question 30

**Explain the purpose of dropout layers and how to use them in Keras**

*Answer to be added.*

---

## Question 31

**How do you use Batch Normalization in a Keras model ?**

*Answer to be added.*

---

## Question 32

**Discuss how you would construct a residual network (ResNet) in Keras**

*Answer to be added.*

---

## Question 33

**Explain the role of optimizers in Keras**

*Answer to be added.*

---

## Question 34

**What is the purpose of a loss function in Keras and how do you select one?**

*Answer to be added.*

---

## Question 35

**Discuss the process of compiling a model in Keras**

*Answer to be added.*

---

## Question 36

**Discuss different strategies for finding the optimal batch size and number of epochs in Keras**

*Answer to be added.*

---

## Question 37

**Discuss the process of feature scaling and why it’s important for neural networks in Keras**

*Answer to be added.*

---

## Question 38

**How to incorporate transfer learning into Keras ?**

*Answer to be added.*

---

## Question 39

**How would you convert a Keras model to TensorFlow’s SavedModel format for deployment?**

*Answer to be added.*

---

## Question 40

**Discuss the use of Keras in mobile and edge devices**

*Answer to be added.*

---

## Question 41

**What are Generative Adversarial Networks (GANs) and how would you implement them using Keras ?**

*Answer to be added.*

---

## Question 42

**Discuss recent advancements in Keras , such as custom training loops**

*Answer to be added.*

---
## Question 43

**What is a custom layer in Keras and how would you implement one?**

**Answer:**

A custom layer extends `tf.keras.layers.Layer` when built-in layers don't meet your needs.

```python
import tensorflow as tf
from tensorflow.keras import layers

# === Basic Custom Layer ===
class LinearLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # Create weights (called once on first input)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# === Advanced: Layer with Training Behavior ===
class NoisyDense(layers.Layer):
    def __init__(self, units, noise_std=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.noise_std = noise_std
        self.dense = layers.Dense(units)
    
    def call(self, inputs, training=False):
        output = self.dense(inputs)
        if training:
            noise = tf.random.normal(tf.shape(output), stddev=self.noise_std)
            output += noise
        return output

# === Usage ===
model = tf.keras.Sequential([
    LinearLayer(64, name='custom_linear'),
    layers.ReLU(),
    NoisyDense(32),
    layers.Dense(10, activation='softmax')
])
```

| Method | Purpose | Required? |
|--------|---------|----------|
| `__init__` | Store configuration | Yes |
| `build(input_shape)` | Create weights | Yes (for learnable params) |
| `call(inputs)` | Forward pass logic | Yes |
| `get_config()` | Serialization | For model saving |

> **Interview Tip:** Use `build()` instead of `__init__` for weights so the layer infers input shape lazily. Always implement `get_config()` if you need to save/load the model.

---

## Question 44

**What is early stopping in Keras and how do you implement it?**

**Answer:**

Early stopping monitors a metric and stops training when it stops improving, preventing overfitting.

```python
import tensorflow as tf
from tensorflow.keras import callbacks

# === Basic Early Stopping ===
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',           # Metric to watch
    patience=10,                  # Epochs to wait after no improvement
    restore_best_weights=True,    # Revert to best weights
    min_delta=0.001,              # Minimum change to qualify as improvement
    mode='min',                   # 'min' for loss, 'max' for accuracy
    verbose=1
)

# === Combined with ModelCheckpoint ===
checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)

history = model.fit(
    X_train, y_train,
    epochs=500,                   # Set high; early stopping handles it
    validation_split=0.2,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

print(f"Stopped at epoch: {len(history.history['loss'])}")
print(f"Best val_loss: {min(history.history['val_loss']):.4f}")

# === Custom Early Stopping ===
class CustomEarlyStopping(callbacks.Callback):
    def __init__(self, patience=5, min_accuracy=0.95):
        super().__init__()
        self.patience = patience
        self.min_accuracy = min_accuracy
        self.wait = 0
        self.best_loss = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        
        if val_acc and val_acc >= self.min_accuracy:
            print(f"\nTarget accuracy reached: {val_acc:.4f}")
            self.model.stop_training = True
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
```

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| `patience` | 0 | 5-20 (higher for noisy metrics) |
| `min_delta` | 0 | 0.001 (ignore tiny improvements) |
| `restore_best_weights` | False | **Always True** |
| `monitor` | `val_loss` | Use validation metric |

> **Interview Tip:** Always set `restore_best_weights=True`. Without it, the model keeps the weights from the *last* epoch (which may be worse than the best). Set `epochs` high (e.g., 500) and let EarlyStopping decide when to stop.

---

## Question 45

**How do you implement a multi-output model in Keras?**

**Answer:**

Multi-output models predict multiple targets simultaneously using the Functional API.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# === Example: Predict Age (regression) + Gender (classification) from face ===
inputs = layers.Input(shape=(128, 128, 3), name='image_input')

# Shared backbone
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

# Branch 1: Age prediction (regression)
age_output = layers.Dense(64, activation='relu')(x)
age_output = layers.Dense(1, activation='linear', name='age_output')(age_output)

# Branch 2: Gender prediction (binary classification)
gender_output = layers.Dense(64, activation='relu')(x)
gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender_output)

# Branch 3: Ethnicity (multi-class)
ethnicity_output = layers.Dense(64, activation='relu')(x)
ethnicity_output = layers.Dense(5, activation='softmax', name='ethnicity_output')(ethnicity_output)

# Build model
model = models.Model(
    inputs=inputs,
    outputs=[age_output, gender_output, ethnicity_output]
)

# Compile with per-output loss and weights
model.compile(
    optimizer='adam',
    loss={
        'age_output': 'mse',
        'gender_output': 'binary_crossentropy',
        'ethnicity_output': 'categorical_crossentropy'
    },
    loss_weights={
        'age_output': 0.5,
        'gender_output': 1.0,
        'ethnicity_output': 1.0
    },
    metrics={
        'age_output': 'mae',
        'gender_output': 'accuracy',
        'ethnicity_output': 'accuracy'
    }
)

# Train
model.fit(
    X_train,
    {'age_output': y_age, 'gender_output': y_gender, 'ethnicity_output': y_ethnicity},
    epochs=50, batch_size=32
)

model.summary()
```

| Component | Purpose |
|-----------|--------|
| Shared backbone | Common feature extraction |
| Output branches | Task-specific heads |
| `loss_weights` | Balance losses between tasks |
| Named outputs | Map losses/metrics to outputs |

> **Interview Tip:** Multi-task learning often improves generalization because the shared backbone learns features useful for all tasks. Name your output layers for cleaner loss/metric configuration.

---

## Question 46

**Discuss the implementation of stateful LSTM networks in Keras.**

**Answer:**

Stateful LSTMs maintain hidden state across batches, unlike default (stateless) LSTMs that reset state after each batch.

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

# === Stateful LSTM ===
batch_size = 32
timesteps = 10
features = 5

model = models.Sequential([
    layers.LSTM(64, stateful=True, return_sequences=True,
                batch_input_shape=(batch_size, timesteps, features)),  # Must specify batch_size
    layers.LSTM(32, stateful=True),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# === Training: Manually Reset States ===
for epoch in range(50):
    model.reset_states()  # Reset at start of each sequence/epoch
    for batch_x, batch_y in data_generator:  # Must yield exact batch_size
        model.train_on_batch(batch_x, batch_y)

# === Prediction ===
model.reset_states()
for chunk in test_chunks:
    pred = model.predict(chunk, batch_size=batch_size)

# === Stateful vs Stateless Comparison ===
# Stateless (default): states reset after every batch
stateless = layers.LSTM(64, stateful=False)  # Default

# Stateful: states carry over between batches
stateful = layers.LSTM(64, stateful=True,
                       batch_input_shape=(batch_size, timesteps, features))

# === When to Use Stateful ===
# 1. Very long sequences split across batches
# 2. Real-time streaming predictions
# 3. When temporal continuity matters across batches

# === Key Requirements ===
# - batch_input_shape instead of input_shape
# - Manual model.reset_states()
# - Data must be ordered (no shuffling between batches!)
# - Each batch must have exactly batch_size samples
```

| Aspect | Stateless (Default) | Stateful |
|--------|-------------------|----------|
| State reset | After each batch | Manual `reset_states()` |
| Input shape | `input_shape=(T, F)` | `batch_input_shape=(B, T, F)` |
| Shuffling | Allowed | **Not allowed** between batches |
| Batch size | Flexible | **Fixed** |
| Use case | Standard sequence tasks | Very long sequences, streaming |

> **Interview Tip:** Stateful LSTMs are rarely needed. Use them only for sequences too long to fit in one batch or real-time streaming. Most tasks work fine with stateless LSTMs and longer sequence lengths.

---

## Question 47

**Explain how you can use Keras to implement a neural style transfer model.**

**Answer:**

```python
import tensorflow as tf
import numpy as np

# === Neural Style Transfer ===
# Combines content of one image with style of another

# 1. Load pre-trained VGG19
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# 2. Define content and style layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                'block4_conv1', 'block5_conv1']

def get_model(content_layers, style_layers):
    outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)

extractor = get_model(content_layers, style_layers)

# 3. Gram matrix for style representation
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    shape = tf.shape(tensor)
    num_locations = tf.cast(shape[1] * shape[2], tf.float32)
    return result / num_locations

# 4. Compute losses
def compute_loss(generated, content_target, style_targets,
                 content_weight=1e4, style_weight=1e-2):
    outputs = extractor(generated)
    n_style = len(style_layers)
    
    content_output = outputs[:len(content_layers)]
    style_outputs = outputs[len(content_layers):]
    
    # Content loss
    content_loss = tf.reduce_mean(
        [(tf.reduce_mean((c - t) ** 2))
         for c, t in zip(content_output, content_target)]
    )
    
    # Style loss
    style_loss = tf.reduce_mean(
        [(tf.reduce_mean((gram_matrix(s) - gram_matrix(t)) ** 2))
         for s, t in zip(style_outputs, style_targets)]
    )
    
    return content_weight * content_loss + style_weight * style_loss

# 5. Optimization loop
def style_transfer(content_image, style_image, epochs=1000, lr=0.02):
    # Initialize generated image from content
    generated = tf.Variable(content_image, dtype=tf.float32)
    
    content_target = extractor(content_image)[:len(content_layers)]
    style_targets = extractor(style_image)[len(content_layers):]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(generated, content_target, style_targets)
        
        grads = tape.gradient(loss, generated)
        optimizer.apply_gradients([(grads, generated)])
        generated.assign(tf.clip_by_value(generated, 0.0, 1.0))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.2f}")
    
    return generated
```

| Component | Purpose |
|-----------|--------|
| Content layers | Capture high-level structural features |
| Style layers | Capture textures, patterns, colors |
| Gram matrix | Statistical representation of style |
| Content loss | Preserve structure of content image |
| Style loss | Match texture statistics of style image |

> **Interview Tip:** Neural style transfer optimizes the **pixel values** of the generated image (not model weights). The Gram matrix captures correlations between feature maps, representing artistic style independent of spatial arrangement.

---

## Question 48

**Discuss strategies to identify the cause of a performance bottleneck in a Keras model**

*Answer to be added.*

---

