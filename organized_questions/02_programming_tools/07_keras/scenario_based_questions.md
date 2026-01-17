# Keras Interview Questions - Scenario-Based Questions

## Question 1

**Your model trains fine but predictions are poor in production. What could be wrong?**

### Answer

### Common Issues

| Issue | Solution |
|-------|----------|
| **Training mode in inference** | Use `model.predict()` not `model(x, training=True)` |
| **Different preprocessing** | Save and reuse preprocessing params |
| **Data distribution shift** | Monitor input distributions |
| **Missing normalization** | Ensure same scaling in production |

### Python Code Example
```python
from tensorflow import keras
import numpy as np

# Problem: Dropout/BatchNorm behavior
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1)
])

# WRONG: training=True in production
# bad_pred = model(X_test, training=True)  # Dropout active!

# CORRECT: Use predict() or training=False
good_pred = model.predict(X_test)  # training=False by default
# or
good_pred = model(X_test, training=False)

# Save preprocessing parameters
class ProductionPipeline:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-7)
    
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
    
    def load(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']

# Use same preprocessing
pipeline = ProductionPipeline()
pipeline.fit(X_train)
pipeline.save('preprocessing.npz')

# In production
prod_pipeline = ProductionPipeline()
prod_pipeline.load('preprocessing.npz')
X_prod_normalized = prod_pipeline.transform(X_production)
```

---

## Question 2

**Your model is too large for mobile deployment. How do you reduce its size?**

### Answer

### Size Reduction Techniques

| Technique | Size Reduction | Accuracy Impact |
|-----------|----------------|-----------------|
| **Quantization** | 2-4x | Minimal |
| **Pruning** | 2-10x | Minimal |
| **Knowledge Distillation** | Variable | Some loss |
| **Architecture** | Variable | Depends |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Original model
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 1. Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

original_size = len(model.to_json())
quantized_size = len(quantized_model)
print(f"Size reduction: {original_size/quantized_size:.1f}x")

# 2. Pruning during training
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

pruned_model = prune_low_magnitude(model, **pruning_params)
pruned_model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# 3. Knowledge distillation
class Distiller(keras.Model):
    def __init__(self, student, teacher, temperature=3):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
    
    def train_step(self, data):
        x, y = data
        
        # Teacher predictions (soft labels)
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Student predictions
            student_predictions = self.student(x, training=True)
            
            # Distillation loss
            soft_loss = keras.losses.KLDivergence()(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )
            
            # Hard label loss
            hard_loss = keras.losses.sparse_categorical_crossentropy(
                y, student_predictions
            )
            
            loss = soft_loss + hard_loss
        
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables)
        )
        
        return {'loss': loss}
```

---

## Question 3

**You need to train on data that doesn't fit in memory. What's your approach?**

### Answer

### Solutions

| Method | Use Case |
|--------|----------|
| **tf.data.Dataset** | Most flexible |
| **ImageDataGenerator** | Image data (legacy) |
| **Generators** | Custom data loading |
| **TFRecord** | Large datasets |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

# Method 1: tf.data.Dataset from generator
def data_generator(file_paths, batch_size):
    """Generator that yields batches"""
    while True:
        np.random.shuffle(file_paths)
        for path in file_paths:
            # Load data from file
            data = np.load(path)
            X, y = data['X'], data['y']
            
            for i in range(0, len(X), batch_size):
                yield X[i:i+batch_size], y[i:i+batch_size]

# Create dataset from generator
def create_dataset(file_paths, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(file_paths, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Method 2: Memory mapping for large files
class MemoryMappedDataset(keras.utils.Sequence):
    def __init__(self, data_path, batch_size):
        self.data = np.load(data_path, mmap_mode='r')  # Memory mapped
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.data['X']) // self.batch_size
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.data['X'][start:end], self.data['y'][start:end]

# Method 3: TFRecord for efficient storage
def create_tfrecord(X, y, filename):
    writer = tf.io.TFRecordWriter(filename)
    for i in range(len(X)):
        feature = {
            'X': tf.train.Feature(float_list=tf.train.FloatList(value=X[i])),
            'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[i]]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

def read_tfrecord(filename):
    def parse_fn(example):
        features = {
            'X': tf.io.FixedLenFeature([10], tf.float32),
            'y': tf.io.FixedLenFeature([1], tf.int64)
        }
        parsed = tf.io.parse_single_example(example, features)
        return parsed['X'], parsed['y'][0]
    
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_fn)
    return dataset

# Use with model
# model.fit(dataset, epochs=10, steps_per_epoch=1000)
```

---

## Question 4

**Your model accuracy plateaus early in training. How do you diagnose and fix this?**

### Answer

### Diagnostic Steps

| Check | What to Look For |
|-------|------------------|
| Learning rate | Too high/low |
| Loss function | Appropriate for task |
| Data quality | Label noise, imbalance |
| Architecture | Capacity issues |
| Gradients | Vanishing/exploding |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

# 1. Check if model can overfit on small batch
def overfit_test(model, X, y, epochs=100):
    """Model should reach ~100% on small batch"""
    X_small, y_small = X[:32], y[:32]
    
    model_copy = keras.models.clone_model(model)
    model_copy.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                       metrics=['accuracy'])
    
    history = model_copy.fit(X_small, y_small, epochs=epochs, verbose=0)
    
    if history.history['accuracy'][-1] < 0.95:
        print("❌ Cannot overfit - check model capacity or data")
    else:
        print("✅ Model can learn")

# 2. Learning rate finder
class LRFinder(keras.callbacks.Callback):
    def __init__(self, min_lr=1e-7, max_lr=10, steps=100):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps = steps
        self.losses = []
        self.lrs = []
    
    def on_train_begin(self, logs=None):
        self.step = 0
        self.factor = (self.max_lr / self.min_lr) ** (1 / self.steps)
        keras.backend.set_value(self.model.optimizer.lr, self.min_lr)
    
    def on_batch_end(self, batch, logs=None):
        self.step += 1
        lr = self.min_lr * (self.factor ** self.step)
        keras.backend.set_value(self.model.optimizer.lr, lr)
        
        self.losses.append(logs['loss'])
        self.lrs.append(lr)
        
        if self.step >= self.steps:
            self.model.stop_training = True
    
    def get_optimal_lr(self):
        min_loss_idx = np.argmin(self.losses)
        return self.lrs[min_loss_idx] / 10  # Use 1/10 of min

# 3. Gradient checking
@tf.function
def check_gradients(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(y, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    for var, grad in zip(model.trainable_variables, gradients):
        if grad is not None:
            grad_norm = tf.norm(grad).numpy()
            if grad_norm < 1e-7:
                print(f"⚠️ {var.name}: Vanishing gradient")
            elif grad_norm > 1000:
                print(f"⚠️ {var.name}: Exploding gradient")

# 4. Solutions
# Use learning rate warmup
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        return self.initial_lr * tf.minimum(1.0, step / self.warmup_steps)

# Use gradient clipping
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
```

---

## Question 5

**You need to explain your Keras model's predictions to stakeholders. How do you do it?**

### Answer

### Interpretability Techniques

| Technique | Use Case |
|-----------|----------|
| **Feature importance** | Tabular data |
| **Grad-CAM** | Image models |
| **SHAP** | Model-agnostic |
| **Attention visualization** | Sequence models |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

# 1. Grad-CAM for CNNs
def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    # Create model that outputs conv layer and predictions
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_class = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_class]
    
    # Gradients of top class w.r.t. conv output
    grads = tape.gradient(top_class_channel, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv outputs by importance
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# 2. Feature importance for Dense networks
def get_feature_importance(model, X, y, feature_names):
    baseline_loss = model.evaluate(X, y, verbose=0)[0]
    importances = []
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])
        
        permuted_loss = model.evaluate(X_permuted, y, verbose=0)[0]
        importance = permuted_loss - baseline_loss
        importances.append(importance)
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
    
    return importances

# 3. SHAP values (pip install shap)
import shap

def explain_with_shap(model, X_train, X_explain):
    # For Keras models
    explainer = shap.DeepExplainer(model, X_train[:100])
    shap_values = explainer.shap_values(X_explain)
    
    # Summary plot
    shap.summary_plot(shap_values, X_explain)
    
    # Force plot for single prediction
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][0],
        X_explain[0]
    )

# 4. Attention visualization
def visualize_attention(model, text, tokenizer):
    """For models with attention layers"""
    # Get attention weights from model
    attention_layer = model.get_layer('attention')
    
    attention_model = keras.Model(
        inputs=model.input,
        outputs=[attention_layer.output, model.output]
    )
    
    # Tokenize input
    tokens = tokenizer.texts_to_sequences([text])
    attention_weights, predictions = attention_model.predict(tokens)
    
    # Display attention weights
    words = text.split()
    for word, weight in zip(words, attention_weights[0]):
        print(f"{word}: {'█' * int(weight * 20)}")
```

---

## Question 6

**Your model works well on validation but poorly on real-world data. What's happening?**

### Answer

### Possible Causes

| Issue | Description |
|-------|-------------|
| **Data leakage** | Information from future/target in features |
| **Distribution shift** | Real data differs from training |
| **Overfitting to validation** | Hyperparameter tuning on val set |
| **Sample bias** | Training data not representative |

### Python Code Example
```python
from tensorflow import keras
import numpy as np
from scipy.stats import ks_2samp

# 1. Check for data drift
def detect_data_drift(training_data, production_data, threshold=0.05):
    drifted_features = []
    
    for i in range(training_data.shape[1]):
        stat, p_value = ks_2samp(
            training_data[:, i],
            production_data[:, i]
        )
        if p_value < threshold:
            drifted_features.append(i)
    
    if drifted_features:
        print(f"⚠️ Data drift in features: {drifted_features}")
    return drifted_features

# 2. Check for data leakage
def check_leakage(X_train, X_val, y_train, y_val):
    """Check if validation examples appear in training"""
    # Check for duplicate rows
    combined = np.vstack([X_train, X_val])
    unique_rows = np.unique(combined, axis=0)
    
    if len(unique_rows) < len(combined):
        print("⚠️ Possible data leakage: duplicate rows found")
    
    # Check correlation of features with target
    for i in range(X_train.shape[1]):
        corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
        if abs(corr) > 0.95:
            print(f"⚠️ Feature {i} has high correlation with target: {corr:.3f}")

# 3. Proper train/val/test split
from sklearn.model_selection import train_test_split

def proper_split(X, y):
    """Use separate test set that's never touched"""
    # First split: separate test set (hold out completely)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Second split: train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.18, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 4. Cross-validation for hyperparameter tuning
from sklearn.model_selection import KFold

def cross_validate_model(create_model_fn, X, y, n_folds=5):
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kfold.split(X):
        model = create_model_fn()
        model.fit(X[train_idx], y[train_idx], epochs=10, verbose=0)
        score = model.evaluate(X[val_idx], y[val_idx], verbose=0)[1]
        scores.append(score)
    
    print(f"CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores
```

---

## Question 7

**You need to serve a Keras model with low latency. What optimizations do you apply?**

### Answer

### Optimization Techniques

| Technique | Benefit |
|-----------|---------|
| **Batching** | Higher throughput |
| **Quantization** | Faster inference |
| **TF-TRT** | GPU optimization |
| **XLA compilation** | Graph optimization |
| **Model simplification** | Reduce complexity |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time

# 1. Quantization for faster inference
def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Full integer quantization
    def representative_data():
        for _ in range(100):
            yield [np.random.randn(1, 10).astype(np.float32)]
    
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    return converter.convert()

# 2. XLA compilation
@tf.function(jit_compile=True)
def fast_inference(model, x):
    return model(x, training=False)

# 3. Batch inference
def batch_predict(model, data, batch_size=256):
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        pred = model.predict(batch, verbose=0)
        predictions.append(pred)
    return np.vstack(predictions)

# 4. TensorFlow Serving optimized model
def export_for_serving(model, export_path):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 10], dtype=tf.float32, name='input')
    ])
    def serve(x):
        return {'predictions': model(x)}
    
    tf.saved_model.save(
        model, 
        export_path,
        signatures={'serving_default': serve}
    )

# 5. Benchmark
def benchmark_inference(model, data, n_runs=100):
    # Warmup
    _ = model.predict(data[:1], verbose=0)
    
    # Single inference
    start = time.time()
    for _ in range(n_runs):
        _ = model.predict(data[:1], verbose=0)
    single_time = (time.time() - start) / n_runs
    
    # Batch inference
    start = time.time()
    _ = model.predict(data, verbose=0)
    batch_time = time.time() - start
    
    print(f"Single inference: {single_time*1000:.2f}ms")
    print(f"Batch inference ({len(data)}): {batch_time*1000:.2f}ms")
    print(f"Per-sample in batch: {batch_time/len(data)*1000:.2f}ms")
```

---

## Question 8

**You're asked to implement A/B testing for two model versions. How do you approach this?**

### Answer

### A/B Testing Framework

| Component | Purpose |
|-----------|---------|
| **Traffic split** | Random user assignment |
| **Metrics** | Define success criteria |
| **Statistical test** | Determine significance |
| **Monitoring** | Track real-time performance |

### Python Code Example
```python
from tensorflow import keras
import numpy as np
from scipy import stats
import hashlib

class ABTestingFramework:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        
        # Metrics storage
        self.metrics_a = {'predictions': [], 'outcomes': []}
        self.metrics_b = {'predictions': [], 'outcomes': []}
    
    def assign_variant(self, user_id):
        """Deterministic assignment based on user_id"""
        hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        return 'A' if (hash_value % 100) < (self.split_ratio * 100) else 'B'
    
    def predict(self, user_id, features):
        """Route prediction to appropriate model"""
        variant = self.assign_variant(user_id)
        
        if variant == 'A':
            prediction = self.model_a.predict(features, verbose=0)
            return prediction, 'A'
        else:
            prediction = self.model_b.predict(features, verbose=0)
            return prediction, 'B'
    
    def log_outcome(self, variant, prediction, actual_outcome):
        """Log outcomes for analysis"""
        if variant == 'A':
            self.metrics_a['predictions'].append(prediction)
            self.metrics_a['outcomes'].append(actual_outcome)
        else:
            self.metrics_b['predictions'].append(prediction)
            self.metrics_b['outcomes'].append(actual_outcome)
    
    def analyze_results(self):
        """Statistical comparison of A vs B"""
        outcomes_a = np.array(self.metrics_a['outcomes'])
        outcomes_b = np.array(self.metrics_b['outcomes'])
        
        # Conversion rate comparison
        rate_a = np.mean(outcomes_a)
        rate_b = np.mean(outcomes_b)
        
        # Statistical significance (chi-square test)
        contingency = [
            [np.sum(outcomes_a), len(outcomes_a) - np.sum(outcomes_a)],
            [np.sum(outcomes_b), len(outcomes_b) - np.sum(outcomes_b)]
        ]
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        
        print(f"Model A rate: {rate_a:.4f} (n={len(outcomes_a)})")
        print(f"Model B rate: {rate_b:.4f} (n={len(outcomes_b)})")
        print(f"Relative improvement: {(rate_b - rate_a) / rate_a * 100:.2f}%")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            winner = 'B' if rate_b > rate_a else 'A'
            print(f"✅ Statistically significant. Model {winner} wins.")
        else:
            print("❌ Not statistically significant yet.")
        
        return {'rate_a': rate_a, 'rate_b': rate_b, 'p_value': p_value}

# Usage
ab_test = ABTestingFramework(model_a, model_b, split_ratio=0.5)

# Simulate serving
for user_id in range(1000):
    features = np.random.randn(1, 10).astype(np.float32)
    prediction, variant = ab_test.predict(user_id, features)
    
    # Simulate outcome (real outcome would come later)
    outcome = 1 if np.random.random() < 0.1 + 0.02 * (variant == 'B') else 0
    ab_test.log_outcome(variant, prediction, outcome)

# Analyze
results = ab_test.analyze_results()
```

---

## Question 9

**Your training is unstable with loss spikes. How do you stabilize it?**

### Answer

### Stabilization Techniques

| Technique | Implementation |
|-----------|----------------|
| **Gradient clipping** | `clipnorm`, `clipvalue` |
| **Learning rate warmup** | Gradual LR increase |
| **Batch normalization** | Normalize activations |
| **Proper initialization** | He/Glorot init |
| **Mixed precision fix** | Loss scaling |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf

# 1. Gradient clipping
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0,  # Clip gradient norm
    # clipvalue=0.5  # Or clip by value
)

# 2. Learning rate warmup
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        
        # Cosine decay phase
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_lr = self.initial_lr * 0.5 * (1 + tf.cos(np.pi * progress))
        
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

# 3. Proper weight initialization
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer='he_normal',  # For ReLU
                       input_shape=(10,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation='relu',
                       kernel_initializer='he_normal'),
    keras.layers.Dense(1)
])

# 4. Mixed precision with loss scaling
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

# Loss scaling is automatic, but can be manual:
optimizer = keras.mixed_precision.LossScaleOptimizer(
    keras.optimizers.Adam(),
    dynamic=True  # Automatically adjust loss scale
)

# 5. Gradient monitoring callback
class GradientMonitor(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        # Check for NaN loss
        if logs and np.isnan(logs.get('loss', 0)):
            print(f"⚠️ NaN loss detected at batch {batch}")
            self.model.stop_training = True

# 6. Complete stable training setup
def create_stable_model():
    model = keras.Sequential([
        keras.layers.Dense(128, kernel_initializer='he_normal', input_shape=(10,)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(64, kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        
        keras.layers.Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(
        learning_rate=WarmupCosineDecay(0.001, 1000, 10000),
        clipnorm=1.0
    )
    
    model.compile(optimizer=optimizer, loss='mse')
    return model
```

---

## Question 10

**You need to implement continual learning where the model learns from new data without forgetting old patterns. How?**

### Answer

### Continual Learning Strategies

| Strategy | Description |
|----------|-------------|
| **Elastic Weight Consolidation** | Penalize changes to important weights |
| **Experience Replay** | Mix old and new data |
| **Progressive Networks** | Add capacity for new tasks |
| **Knowledge Distillation** | Preserve old knowledge |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

# 1. Experience Replay
class ExperienceReplay:
    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.X_buffer = []
        self.y_buffer = []
    
    def add(self, X, y):
        self.X_buffer.extend(X)
        self.y_buffer.extend(y)
        
        # Keep buffer manageable
        if len(self.X_buffer) > self.buffer_size:
            self.X_buffer = self.X_buffer[-self.buffer_size:]
            self.y_buffer = self.y_buffer[-self.buffer_size:]
    
    def sample(self, n):
        indices = np.random.choice(len(self.X_buffer), min(n, len(self.X_buffer)))
        return np.array([self.X_buffer[i] for i in indices]), \
               np.array([self.y_buffer[i] for i in indices])

# 2. Elastic Weight Consolidation
class EWC:
    def __init__(self, model, X_old, y_old, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # Store old weights
        self.old_weights = [w.numpy() for w in model.trainable_weights]
        
        # Compute Fisher information (importance of each weight)
        self.fisher = self._compute_fisher(X_old, y_old)
    
    def _compute_fisher(self, X, y):
        fisher = []
        
        for _ in range(len(self.model.trainable_weights)):
            fisher.append(np.zeros_like(self.model.trainable_weights[_].numpy()))
        
        # Compute gradients for each sample
        for i in range(min(len(X), 100)):
            with tf.GradientTape() as tape:
                pred = self.model(X[i:i+1], training=True)
                loss = keras.losses.sparse_categorical_crossentropy(y[i:i+1], pred)
            
            grads = tape.gradient(loss, self.model.trainable_weights)
            
            for j, grad in enumerate(grads):
                if grad is not None:
                    fisher[j] += grad.numpy() ** 2
        
        # Normalize
        fisher = [f / len(X) for f in fisher]
        return fisher
    
    def ewc_loss(self):
        """Penalty for changing important weights"""
        ewc_loss = 0
        for i, (w, old_w, f) in enumerate(zip(
            self.model.trainable_weights, self.old_weights, self.fisher
        )):
            ewc_loss += tf.reduce_sum(f * (w - old_w) ** 2)
        return self.lambda_ewc * ewc_loss

# 3. Continual Learning Training
class ContinualLearner:
    def __init__(self, model):
        self.model = model
        self.replay = ExperienceReplay(buffer_size=5000)
        self.ewc = None
    
    def learn_task(self, X_new, y_new, epochs=10, replay_ratio=0.3):
        """Learn new task while remembering old"""
        
        # Add new data to replay buffer
        self.replay.add(list(X_new), list(y_new))
        
        for epoch in range(epochs):
            # Mix new data with replay
            X_replay, y_replay = self.replay.sample(int(len(X_new) * replay_ratio))
            
            X_combined = np.vstack([X_new, X_replay])
            y_combined = np.concatenate([y_new, y_replay])
            
            # Shuffle
            indices = np.random.permutation(len(X_combined))
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
            
            # Train with EWC if we have old task
            if self.ewc:
                # Custom training step with EWC loss
                self._train_with_ewc(X_combined, y_combined)
            else:
                self.model.fit(X_combined, y_combined, epochs=1, verbose=0)
        
        # Update EWC after learning task
        self.ewc = EWC(self.model, X_new, y_new)
    
    def _train_with_ewc(self, X, y):
        optimizer = self.model.optimizer
        
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            task_loss = keras.losses.sparse_categorical_crossentropy(y, predictions)
            task_loss = tf.reduce_mean(task_loss)
            
            ewc_loss = self.ewc.ewc_loss()
            total_loss = task_loss + ewc_loss
        
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

# Usage
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

learner = ContinualLearner(model)

# Learn task 1
learner.learn_task(X_task1, y_task1)

# Learn task 2 (without forgetting task 1)
learner.learn_task(X_task2, y_task2)
```
