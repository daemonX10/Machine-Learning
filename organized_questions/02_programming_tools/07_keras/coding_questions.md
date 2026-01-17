# Keras Interview Questions - Coding Questions

## Question 1

**Implement a CNN for image classification.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers

# CNN for MNIST
model = keras.Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', 
                  input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Third conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load and preprocess MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## Question 2

**Implement an LSTM for sequence classification.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Sample: Sentiment classification
vocab_size = 10000
max_length = 200
embedding_dim = 128

model = keras.Sequential([
    # Embedding layer
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    
    # LSTM layers
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dropout(0.2),
    
    # Classifier
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
    num_words=vocab_size
)

# Pad sequences
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length)

# Train
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Bidirectional LSTM (better for many tasks)
bi_lstm_model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
])
```

---

## Question 3

**Implement a model with multiple inputs using Functional API.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers

# Multi-input model: Image + Metadata

# Image input branch
image_input = keras.Input(shape=(64, 64, 3), name='image')
x1 = layers.Conv2D(32, 3, activation='relu')(image_input)
x1 = layers.MaxPooling2D()(x1)
x1 = layers.Conv2D(64, 3, activation='relu')(x1)
x1 = layers.GlobalAveragePooling2D()(x1)

# Metadata input branch
meta_input = keras.Input(shape=(10,), name='metadata')
x2 = layers.Dense(32, activation='relu')(meta_input)
x2 = layers.Dense(16, activation='relu')(x2)

# Merge branches
merged = layers.Concatenate()([x1, x2])
x = layers.Dense(64, activation='relu')(merged)
x = layers.Dropout(0.3)(x)

# Output
output = layers.Dense(5, activation='softmax', name='class')(x)

# Create model
model = keras.Model(
    inputs=[image_input, meta_input],
    outputs=output
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Training with multiple inputs
# model.fit(
#     {'image': X_images, 'metadata': X_meta},
#     y_labels,
#     epochs=10
# )
```

---

## Question 4

**Implement a custom training loop in Keras.**

### Solution
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Loss and optimizer
loss_fn = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Metrics
train_loss = keras.metrics.Mean(name='train_loss')
train_acc = keras.metrics.BinaryAccuracy(name='train_accuracy')
val_loss = keras.metrics.Mean(name='val_loss')
val_acc = keras.metrics.BinaryAccuracy(name='val_accuracy')

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss.update_state(loss)
    train_acc.update_state(y, predictions)

@tf.function
def val_step(x, y):
    predictions = model(x, training=False)
    loss = loss_fn(y, predictions)
    
    val_loss.update_state(loss)
    val_acc.update_state(y, predictions)

# Training loop
epochs = 10
batch_size = 32

# Sample data
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 2, 1000).astype(np.float32)
X_val = np.random.randn(200, 10).astype(np.float32)
y_val = np.random.randint(0, 2, 200).astype(np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size)

for epoch in range(epochs):
    # Reset metrics
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
    
    # Training
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    
    # Validation
    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)
    
    print(f"Epoch {epoch+1}: "
          f"Loss={train_loss.result():.4f}, Acc={train_acc.result():.4f}, "
          f"Val_Loss={val_loss.result():.4f}, Val_Acc={val_acc.result():.4f}")
```

---

## Question 5

**Implement an Autoencoder.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers

# Encoder
encoder_inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(encoder_inputs)
x = layers.Dense(128, activation='relu')(x)
latent = layers.Dense(32, activation='relu')(x)  # Bottleneck

encoder = keras.Model(encoder_inputs, latent, name='encoder')

# Decoder
decoder_inputs = keras.Input(shape=(32,))
x = layers.Dense(128, activation='relu')(decoder_inputs)
x = layers.Dense(256, activation='relu')(x)
decoder_outputs = layers.Dense(784, activation='sigmoid')(x)

decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

# Autoencoder
autoencoder_inputs = keras.Input(shape=(784,))
encoded = encoder(autoencoder_inputs)
decoded = decoder(encoded)

autoencoder = keras.Model(autoencoder_inputs, decoded, name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mse')

# Train on MNIST
(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

autoencoder.fit(
    X_train, X_train,  # Input = Output
    epochs=20,
    batch_size=256,
    validation_data=(X_test, X_test)
)

# Use encoder for dimensionality reduction
encoded_imgs = encoder.predict(X_test[:100])
print(f"Encoded shape: {encoded_imgs.shape}")  # (100, 32)

# Reconstruct images
reconstructed = autoencoder.predict(X_test[:10])
```

---

## Question 6

**Implement learning rate scheduling.**

### Solution
```python
from tensorflow import keras
import numpy as np

# Method 1: ReduceLROnPlateau callback
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Reduce LR by half
    patience=3,        # Wait 3 epochs
    min_lr=1e-6,
    verbose=1
)

# Method 2: LearningRateScheduler callback
def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)

lr_scheduler = keras.callbacks.LearningRateScheduler(schedule)

# Method 3: Built-in schedules
initial_lr = 0.01

# Exponential decay
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.9
)

# Cosine decay
cosine_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=10000
)

# Use schedule with optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Method 4: Warmup schedule
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
    
    def __call__(self, step):
        warmup = self.initial_lr * step / self.warmup_steps
        decay = self.initial_lr * (1 - step / self.decay_steps)
        return tf.minimum(warmup, decay)

# Use
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
model.compile(optimizer=optimizer, loss='mse')
```

---

## Question 7

**Implement a Siamese network for similarity learning.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Shared encoder
def create_encoder(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return keras.Model(inputs, x, name='encoder')

# Create shared encoder
encoder = create_encoder((28, 28, 1))

# Siamese network
input_a = keras.Input(shape=(28, 28, 1), name='input_a')
input_b = keras.Input(shape=(28, 28, 1), name='input_b')

# Get embeddings (shared weights)
embedding_a = encoder(input_a)
embedding_b = encoder(input_b)

# Compute distance
distance = layers.Lambda(
    lambda x: tf.abs(x[0] - x[1])
)([embedding_a, embedding_b])

# Classifier
output = layers.Dense(1, activation='sigmoid')(distance)

siamese_model = keras.Model(
    inputs=[input_a, input_b],
    outputs=output
)

siamese_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Contrastive loss (alternative)
def contrastive_loss(y_true, y_pred, margin=1.0):
    """Contrastive loss for siamese networks"""
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Use: pairs of similar (1) and dissimilar (0) samples
# siamese_model.fit([X1, X2], y_similar, epochs=10)
```

---

## Question 8

**Implement model ensembling in Keras.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Create multiple models
def create_model(seed):
    np.random.seed(seed)
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train ensemble
n_models = 5
models = []

for i in range(n_models):
    model = create_model(seed=i)
    # model.fit(X_train, y_train, epochs=10, verbose=0)
    models.append(model)

# Ensemble prediction (averaging)
def ensemble_predict(models, X):
    predictions = [model.predict(X) for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

# Voting ensemble
def voting_predict(models, X):
    predictions = [np.argmax(model.predict(X), axis=1) for model in models]
    predictions = np.array(predictions)
    # Majority voting
    from scipy import stats
    ensemble_pred, _ = stats.mode(predictions, axis=0)
    return ensemble_pred.flatten()

# Keras ensemble model
def create_ensemble_model(models, input_shape):
    inputs = keras.Input(shape=input_shape)
    outputs = [model(inputs) for model in models]
    
    # Average predictions
    averaged = layers.Average()(outputs)
    
    ensemble = keras.Model(inputs=inputs, outputs=averaged)
    return ensemble

# Usage
# ensemble = create_ensemble_model(models, (10,))
# predictions = ensemble.predict(X_test)
```

---

## Question 9

**Implement attention mechanism for sequence models.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Simple self-attention layer
class SelfAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
    
    def call(self, x):
        # x shape: (batch, seq_len, features)
        score = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context = tf.reduce_sum(x * tf.expand_dims(attention_weights, -1), axis=1)
        return context

# Model with attention
vocab_size = 10000
embedding_dim = 128
max_length = 100

model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    layers.LSTM(64, return_sequences=True),
    SelfAttention(64),  # Custom attention
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Using built-in MultiHeadAttention
inputs = keras.Input(shape=(max_length,))
x = layers.Embedding(vocab_size, embedding_dim)(inputs)
x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

attention_model = keras.Model(inputs, outputs)
attention_model.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## Question 10

**Implement a GAN (Generative Adversarial Network).**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# Generator
def build_generator(latent_dim):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(784, activation='sigmoid'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Discriminator
def build_discriminator():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build GAN
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator()

discriminator.compile(
    optimizer=keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Combined model
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = keras.Model(gan_input, gan_output)
gan.compile(
    optimizer=keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy'
)

# Training
def train_gan(epochs, batch_size=128):
    (X_train, _), (_, _) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)
        
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D_loss={d_loss[0]:.4f}, G_loss={g_loss:.4f}")

# train_gan(epochs=1000)
```
