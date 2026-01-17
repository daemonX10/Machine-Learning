# TensorFlow Interview Questions - Coding Questions

## Question 1

**How do you implement a custom layer in TensorFlow?**

### Solution
```python
import tensorflow as tf

class CustomDenseLayer(tf.keras.layers.Layer):
    """Custom dense layer with optional L2 regularization"""
    
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
    
    def build(self, input_shape):
        # Create trainable weights
        self.w = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        if self.use_bias:
            self.b = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
        super().build(input_shape)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.w)
        if self.use_bias:
            output = output + self.b
        if self.activation:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config

# Use custom layer
model = tf.keras.Sequential([
    CustomDenseLayer(64, activation='relu', input_shape=(10,)),
    CustomDenseLayer(32, activation='relu'),
    CustomDenseLayer(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
```

---

## Question 2

**Implement a custom training loop with GradientTape.**

### Solution
```python
import tensorflow as tf
import numpy as np

# Sample data
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1)).astype(np.float32)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Loss and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_loss.update_state(loss)
    train_accuracy.update_state(y, predictions)
    
    return loss

# Dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(1000).batch(32)

# Training loop
epochs = 10
for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    for x_batch, y_batch in dataset:
        train_step(x_batch, y_batch)
    
    print(f"Epoch {epoch+1}: Loss={train_loss.result():.4f}, "
          f"Accuracy={train_accuracy.result():.4f}")
```

---

## Question 3

**Implement a CNN for image classification.**

### Solution
```python
import tensorflow as tf

# CNN Model
model = tf.keras.Sequential([
    # First conv block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                           input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Second conv block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Third conv block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    
    # Flatten and dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Train
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## Question 4

**Implement an RNN/LSTM for sequence prediction.**

### Solution
```python
import tensorflow as tf
import numpy as np

# Generate sequence data
def generate_sequences(n_samples=1000, seq_length=10):
    X = np.random.randn(n_samples, seq_length, 1).astype(np.float32)
    y = np.sum(X, axis=1)  # Target: sum of sequence
    return X, y

X_train, y_train = generate_sequences()

# LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, 
                         input_shape=(10, 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Bidirectional LSTM
bi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        input_shape=(10, 1)
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1)
])

# GRU (faster alternative)
gru_model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True, input_shape=(10, 1)),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(1)
])
```

---

## Question 5

**Implement a custom loss function.**

### Solution
```python
import tensorflow as tf

# Method 1: Simple function
def custom_mse(y_true, y_pred):
    """Custom Mean Squared Error"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Method 2: Class-based (for configurable losses)
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        return tf.reduce_mean(alpha_weight * focal_weight * bce)
    
    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config

# Method 3: Weighted loss
def weighted_bce(class_weights):
    """Weighted Binary Cross Entropy"""
    def loss(y_true, y_pred):
        weights = y_true * class_weights[1] + (1 - y_true) * class_weights[0]
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(weights * bce)
    return loss

# Use custom loss
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=FocalLoss(gamma=2.0),  # or custom_mse
    metrics=['accuracy']
)
```

---

## Question 6

**Implement a custom metric.**

### Solution
```python
import tensorflow as tf

# Method 1: Simple function
def f1_score(y_true, y_pred):
    """F1 Score as a metric"""
    y_pred = tf.round(y_pred)
    
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

# Method 2: Class-based (accumulates over batches)
class F1Score(tf.keras.metrics.Metric):
    """Stateful F1 Score metric"""
    
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.round(tf.cast(y_pred, tf.float32))
        
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))
    
    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    
    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

# Use custom metric
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', F1Score()]
)
```

---

## Question 7

**Implement data augmentation for images.**

### Solution
```python
import tensorflow as tf

# Method 1: Using Keras layers (in model)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Include in model
model = tf.keras.Sequential([
    # Augmentation layers (only active during training)
    data_augmentation,
    
    # Model layers
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Method 2: Using tf.data.Dataset
def augment_image(image, label):
    """Apply augmentations to image"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    return image, label

# Apply to dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Method 3: Using ImageDataGenerator (legacy but still common)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1
)
```

---

## Question 8

**Implement an Autoencoder.**

### Solution
```python
import tensorflow as tf

# Simple Autoencoder
class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(latent_dim, activation='relu')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(784, activation='sigmoid'),
            tf.keras.layers.Reshape((28, 28))
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create model
autoencoder = Autoencoder(latent_dim=32)
autoencoder.compile(optimizer='adam', loss='mse')

# Load MNIST
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Train (input = output)
autoencoder.fit(X_train, X_train, 
                epochs=10, 
                batch_size=256,
                validation_data=(X_test, X_test))

# Use encoder for dimensionality reduction
encoded_imgs = autoencoder.encoder(X_test[:10])
decoded_imgs = autoencoder.decoder(encoded_imgs)
```

---

## Question 9

**Implement model with multiple inputs/outputs.**

### Solution
```python
import tensorflow as tf

# Multi-input model (e.g., image + metadata)
image_input = tf.keras.Input(shape=(224, 224, 3), name='image')
meta_input = tf.keras.Input(shape=(10,), name='metadata')

# Image branch
x1 = tf.keras.layers.Conv2D(32, 3, activation='relu')(image_input)
x1 = tf.keras.layers.MaxPooling2D()(x1)
x1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(x1)
x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)

# Metadata branch
x2 = tf.keras.layers.Dense(32, activation='relu')(meta_input)

# Combine
combined = tf.keras.layers.Concatenate()([x1, x2])
x = tf.keras.layers.Dense(64, activation='relu')(combined)

# Multiple outputs
class_output = tf.keras.layers.Dense(10, activation='softmax', name='class')(x)
score_output = tf.keras.layers.Dense(1, activation='linear', name='score')(x)

# Create model
model = tf.keras.Model(
    inputs=[image_input, meta_input],
    outputs=[class_output, score_output]
)

# Compile with different losses
model.compile(
    optimizer='adam',
    loss={
        'class': 'sparse_categorical_crossentropy',
        'score': 'mse'
    },
    loss_weights={'class': 1.0, 'score': 0.5},
    metrics={
        'class': 'accuracy',
        'score': 'mae'
    }
)

# Train
# model.fit(
#     {'image': X_images, 'metadata': X_meta},
#     {'class': y_class, 'score': y_score},
#     epochs=10
# )
```

---

## Question 10

**Implement learning rate scheduling.**

### Solution
```python
import tensorflow as tf
import numpy as np

# Method 1: Built-in schedules
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Method 2: Callback-based
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Method 3: Custom schedule function
def lr_schedule_fn(epoch, lr):
    """Custom learning rate schedule"""
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule_fn)

# Method 4: Warmup + decay
class WarmupDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.initial_lr * step / self.warmup_steps
        decay = self.initial_lr * tf.math.exp(-step / self.decay_steps)
        return tf.minimum(warmup, decay)

# Use with model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='mse'
)

# Or use callbacks during fit
# model.fit(X, y, epochs=50, callbacks=[reduce_lr, lr_scheduler])
```

