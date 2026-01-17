# Keras Interview Questions - General Questions

## Question 1

**How do you handle overfitting in Keras models?**

### Answer

### Overfitting Solutions

| Technique | Implementation |
|-----------|----------------|
| **Dropout** | `layers.Dropout(0.5)` |
| **L1/L2 Regularization** | `kernel_regularizer=l2(0.01)` |
| **Early Stopping** | `EarlyStopping` callback |
| **Data Augmentation** | Random transforms |
| **Reduce Model Complexity** | Fewer layers/neurons |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Model with regularization
model = keras.Sequential([
    layers.Dense(128, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01),
                 input_shape=(784,)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Early stopping
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

## Question 2

**How do you save and load a Keras model?**

### Answer

### Saving Options

| Method | Saves | Format |
|--------|-------|--------|
| `model.save()` | Full model | .h5 or SavedModel |
| `model.save_weights()` | Weights only | .h5 |
| `ModelCheckpoint` | During training | Configurable |

### Python Code Example
```python
from tensorflow import keras

# Build and train model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Save entire model
model.save('my_model.h5')  # HDF5 format
model.save('my_model')      # SavedModel format

# Load entire model
loaded_model = keras.models.load_model('my_model.h5')

# Save weights only
model.save_weights('weights.h5')

# Load weights (need same architecture)
new_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
new_model.load_weights('weights.h5')

# Checkpoint during training
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    save_best_only=True,
    monitor='val_loss'
)
```

---

## Question 3

**What is transfer learning and how do you implement it in Keras?**

### Answer

**Definition**: Transfer learning uses a pre-trained model as starting point for a new task.

### Steps
1. Load pre-trained model (without top layers)
2. Freeze base layers
3. Add new classification layers
4. Train on new data

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

# Load pre-trained model
base_model = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,  # Remove classification head
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add new layers
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train new layers
model.fit(X_train, y_train, epochs=10)

# Fine-tuning: unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Question 4

**How do you handle imbalanced datasets in Keras?**

### Answer

### Solutions

| Method | Implementation |
|--------|----------------|
| **Class weights** | `class_weight` in fit() |
| **Sample weights** | `sample_weight` in fit() |
| **Oversampling** | SMOTE, data augmentation |
| **Custom loss** | Focal loss |

### Python Code Example
```python
from tensorflow import keras
import numpy as np
from collections import Counter

# Calculate class weights
counter = Counter(y_train)
total = len(y_train)
class_weight = {
    0: total / (2 * counter[0]),  # Majority class
    1: total / (2 * counter[1])   # Minority class
}

# Method 1: Class weights
model.fit(
    X_train, y_train,
    class_weight=class_weight,
    epochs=10
)

# Method 2: Sample weights
sample_weights = np.ones(len(y_train))
sample_weights[y_train == 1] = 5.0  # Increase minority weight

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    epochs=10
)

# Method 3: Focal Loss (for severe imbalance)
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1 - p_t) ** self.gamma
        return self.alpha * focal_weight * bce

model.compile(optimizer='adam', loss=FocalLoss())
```

---

## Question 5

**What are the different types of layers available in Keras?**

### Answer

### Layer Categories

| Category | Layers | Use Case |
|----------|--------|----------|
| **Core** | Dense, Activation, Embedding | Basic building blocks |
| **Convolutional** | Conv2D, Conv1D, SeparableConv2D | Image processing |
| **Pooling** | MaxPooling2D, AveragePooling2D | Reduce dimensions |
| **Recurrent** | LSTM, GRU, SimpleRNN | Sequences |
| **Normalization** | BatchNormalization, LayerNormalization | Stabilize training |
| **Regularization** | Dropout, SpatialDropout2D | Prevent overfitting |
| **Attention** | MultiHeadAttention | Transformers |

### Python Code Example
```python
from tensorflow.keras import layers

# Dense layer
dense = layers.Dense(64, activation='relu')

# Convolutional layers
conv = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
pool = layers.MaxPooling2D(pool_size=(2, 2))

# Recurrent layers
lstm = layers.LSTM(64, return_sequences=True)
gru = layers.GRU(32)

# Normalization
batch_norm = layers.BatchNormalization()

# Dropout
dropout = layers.Dropout(0.5)

# Embedding (for text)
embedding = layers.Embedding(input_dim=10000, output_dim=128)

# Attention
attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)

# Complete CNN example
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

---

## Question 6

**How do you implement data augmentation in Keras?**

### Answer

### Augmentation Methods

| Method | Use Case |
|--------|----------|
| **Keras Layers** | In-model augmentation |
| **ImageDataGenerator** | Real-time augmentation |
| **tf.data** | Pipeline augmentation |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

# Method 1: Augmentation as layers (TF 2.x)
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

model = keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Method 2: ImageDataGenerator (legacy but still common)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
    rescale=1./255
)

# Flow from directory
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Use in training
# model.fit(train_generator, epochs=10)
```

---

## Question 7

**How do you create a custom layer in Keras?**

### Answer

### Custom Layer Structure
- `__init__`: Define parameters
- `build`: Create weights
- `call`: Define forward pass

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf

class CustomDense(keras.layers.Layer):
    """Custom fully connected layer"""
    
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        # Create trainable weights
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
        super().build(input_shape)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation)
        })
        return config

# Use custom layer
model = keras.Sequential([
    CustomDense(64, activation='relu', input_shape=(10,)),
    CustomDense(32, activation='relu'),
    CustomDense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## Question 8

**What is batch normalization and why is it used?**

### Answer

**Definition**: Batch normalization normalizes layer inputs to have zero mean and unit variance, stabilizing training.

### Benefits

| Benefit | Description |
|---------|-------------|
| **Faster training** | Allows higher learning rates |
| **Stability** | Reduces internal covariate shift |
| **Regularization** | Acts as mild regularizer |
| **Reduce sensitivity** | Less sensitive to initialization |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

# BatchNormalization placement
model = keras.Sequential([
    # Option 1: After activation
    layers.Dense(64),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    
    # Option 2: Before activation (recommended)
    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(10, activation='softmax')
])

# In CNN
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
```

### Key Points
- Different behavior in training vs inference
- Uses batch statistics during training
- Uses learned moving averages during inference

---

## Question 9

**How do you implement early stopping in Keras?**

### Answer

**Definition**: Early stopping stops training when validation metric stops improving, preventing overfitting.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `monitor` | Metric to track |
| `patience` | Epochs to wait |
| `restore_best_weights` | Return to best model |
| `min_delta` | Minimum improvement |

### Python Code Example
```python
from tensorflow import keras

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',        # Metric to monitor
    patience=5,                 # Wait 5 epochs
    min_delta=0.001,           # Minimum improvement
    restore_best_weights=True,  # Return best model
    verbose=1
)

# Use with model checkpoint
callbacks = [
    early_stopping,
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_loss'
    )
]

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train with early stopping
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,  # Max epochs
    callbacks=callbacks
)

# Check when training stopped
print(f"Stopped at epoch: {len(history.history['loss'])}")
```

---

## Question 10

**How do you use pre-trained word embeddings in Keras?**

### Answer

### Steps
1. Load pre-trained embeddings (GloVe, Word2Vec)
2. Create embedding matrix
3. Initialize Embedding layer with weights

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load GloVe embeddings
def load_glove(path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create embedding matrix
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    
    for word, i in word_index.items():
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
    
    return embedding_matrix

# word_index from tokenizer
# embedding_matrix = load_glove('glove.6B.100d.txt', word_index)

# Create embedding layer with pre-trained weights
vocab_size = 10000
embedding_dim = 100

embedding_layer = layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    # weights=[embedding_matrix],  # Pre-trained weights
    trainable=False  # Freeze embeddings
)

# Model for text classification
model = keras.Sequential([
    embedding_layer,
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
```
