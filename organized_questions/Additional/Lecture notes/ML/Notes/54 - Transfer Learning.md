# Transfer Learning

## Overview

- **Definition:** A machine learning technique that focuses on **storing knowledge gained from one problem and applying it to a different but related problem**
- **Core Idea:** Reuse a pretrained CNN model (trained on a large dataset like ImageNet) on your own custom dataset
- **Why It Matters:** Andrew Ng called transfer learning "the next big thing" after supervised learning — it's the #2 driver of ML in industry

---

## Why Transfer Learning?

| Problem with Training From Scratch | Transfer Learning Solution |
|------------------------------------|---------------------------|
| Need massive **labeled dataset** (costly to build) | Reuse knowledge from a model already trained on large data |
| Training takes **hours/days/weeks** | Skip most or all training — use pretrained weights |
| Building good architectures is **complex** | Leverage proven architectures (VGG, ResNet, etc.) |

### Real-Life Analogy

- Learning to ride a **bicycle** → makes learning a **motorcycle** easier (related skills transfer)
- Learning **violin** → makes learning **guitar** easier (shared understanding of musical notes)
- Transfer learning applies the same principle: knowledge from one domain transfers to a related domain

---

## How Transfer Learning Works

### CNN Architecture — Two Parts

Any CNN model (e.g., VGG16) has two distinct sections:

| Part | Name | Function |
|------|------|----------|
| **Part 1** | Convolution Base (Conv Base) | Feature extraction — extracts spatial patterns from images (edges → shapes → complex features) |
| **Part 2** | Fully Connected (FC) Layers + Output | Classification — maps extracted features to class predictions |

### What You Do in Transfer Learning

1. Take a pretrained model (e.g., VGG16 trained on ImageNet with 1000 classes)
2. **Remove** the FC layers (classification head)
3. **Keep** the Convolution Base (feature extractor)
4. **Add your own** FC layers + output layer matching your problem
5. **Freeze** the Conv Base weights (prevent retraining)
6. Train only the new FC layers on your own data

> **Key:** The convolution base already knows how to extract general image features. You only need to teach the model your specific classification task.

---

## Why Transfer Learning Works

### Feature Hierarchy in CNNs

| Layer Depth | Features Learned | Generality |
|-------------|-----------------|------------|
| **Early layers** | Primitive features: edges, gradients | **Very general** — same across all images |
| **Middle layers** | Shapes, textures, patterns | **General** — applicable to many tasks |
| **Deep layers** | Complex/task-specific features | **Task-specific** — depends on training data |

### The Logic

- ImageNet has ~1000 classes covering most real-world objects
- Primitive features (edges, shapes) are **universal** — no need to relearn them
- The Conv Base has already learned these → **don't reinvent the wheel**
- Only the final classification needs to change for your specific task

---

## Two Approaches to Transfer Learning

### 1. Feature Extraction

| Aspect | Detail |
|--------|--------|
| **What** | Freeze the **entire** Conv Base, replace only FC layers |
| **Conv Base** | Frozen (weights not updated during training) |
| **FC Layers** | Your custom layers — trained on your data |
| **When to Use** | Your task classes are **similar** to ImageNet classes (e.g., cat vs dog — animals already in ImageNet) |
| **Training Speed** | Fast — only FC layer weights are updated |

### 2. Fine-Tuning

| Aspect | Detail |
|--------|--------|
| **What** | Freeze early Conv layers, **unfreeze last few Conv layers** + replace FC layers |
| **Early Conv Layers** | Frozen (primitive features are universal) |
| **Last Conv Layers** | Unfrozen — retrained with your data |
| **FC Layers** | Your custom layers — trained on your data |
| **When to Use** | Your task is **different** from ImageNet classes (e.g., phone vs tablet — not in ImageNet) |
| **Training Speed** | Slower — more parameters to train |
| **Learning Rate** | Use a **very low learning rate** (e.g., 1e-5) to avoid destroying pretrained features |

### Decision Guide

```
Is your task similar to ImageNet classes?
├── YES → Use Feature Extraction (freeze all Conv layers)
└── NO  → Use Fine-Tuning (unfreeze last few Conv layers)
```

---

## Implementation in Keras

### Feature Extraction — Cat vs Dog Classifier

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Step 1: Load VGG16 Conv Base (without FC layers)
conv_base = VGG16(
    weights='imagenet',
    include_top=False,         # Exclude FC layers
    input_shape=(150, 150, 3)
)

# Step 2: Freeze the Conv Base
conv_base.trainable = False

# Step 3: Build new model
model = Sequential([
    conv_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Step 4: Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=val_data)
```

**Key parameters:**
- `weights='imagenet'` → use ImageNet pretrained weights
- `include_top=False` → exclude original FC layers
- `conv_base.trainable = False` → freeze all Conv Base weights

### Data Preprocessing

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Normalize pixel values to [0, 1]
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train/', target_size=(150, 150),
    batch_size=32, class_mode='binary'
)
val_generator = test_datagen.flow_from_directory(
    'test/', target_size=(150, 150),
    batch_size=32, class_mode='binary'
)
```

### Fine-Tuning — Unfreezing Last Conv Block

```python
from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Unfreeze only block5 layers, freeze everything else
conv_base.trainable = True
for layer in conv_base.layers:
    if 'block5' not in layer.name:
        layer.trainable = False

# Build model (same structure)
model = Sequential([
    conv_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Use low learning rate for fine-tuning
from tensorflow.keras.optimizers import RMSprop

model.compile(
    optimizer=RMSprop(learning_rate=1e-5),  # Very low LR
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(train_data, epochs=10, validation_data=val_data)
```

> **Important:** Use a low learning rate (e.g., `1e-5`) during fine-tuning to avoid destroying the pretrained features.

---

## Results Comparison — Cat vs Dog Classification

| Approach | Test Accuracy | Notes |
|----------|---------------|-------|
| Custom CNN (from scratch) | ~81% | Baseline |
| Feature Extraction (VGG16) | ~90% | +9% improvement, some overfitting |
| Feature Extraction + Data Augmentation | ~91.4% | Reduced overfitting |
| **Fine-Tuning (VGG16, block5 unfrozen)** | **~95.2%** | Best result — 14% above baseline |

### Handling Overfitting

- **Data Augmentation** — apply random transformations (rotation, flip, zoom) to training images
- **Dropout layers** — randomly deactivate neurons during training
- **Batch Normalization** — normalize activations between layers

```python
# Data augmentation example
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What** | Reuse knowledge from a model trained on one dataset for a different but related task |
| **Two Approaches** | Feature Extraction (freeze all Conv) and Fine-Tuning (unfreeze last Conv blocks) |
| **Feature Extraction** | Use when your task is similar to the pretrained model's task |
| **Fine-Tuning** | Use when your task differs significantly; unfreeze last few Conv layers |
| **Fine-Tuning LR** | Use very low learning rate (e.g., 1e-5) |
| **Typical Gains** | 81% (scratch) → 95% (fine-tuning) on Cat vs Dog |
| **Key Library** | `tensorflow.keras.applications` — VGG16, ResNet50, InceptionV3, etc. |
| **Key Parameter** | `include_top=False` to remove original FC head |
