# Data Augmentation in Deep Learning

## Overview

- **Definition:** A technique to artificially generate new training data from existing data by applying image transformations
- **Core Idea:** Apply transformations (rotation, flipping, zooming, shifting, shearing) to existing images to create new variations
- **Tool:** `ImageDataGenerator` from Keras

---

## Why Use Data Augmentation?

### Reason 1: Insufficient Data

- Deep learning models require **large amounts of data**
- In many domains (especially **medical imaging**), data collection is **expensive and difficult**
  - Example: Malaria detection — getting biological cell images of actual patients is costly
- Data augmentation lets you **generate more training samples** from limited data

### Reason 2: Reduce Overfitting

- If all training images share an accidental pattern (e.g., all cats looking left), the model may learn **spurious features**
- Augmentation introduces variety → model learns to **generalize** instead of memorizing
  - Example: Horizontal flip makes a left-looking cat also appear right-looking → model doesn't associate direction with class

| Reason | Description |
|--------|-------------|
| **Generate data** | Create new training samples when data is scarce |
| **Reduce overfitting** | Add variety to prevent learning spurious correlations |

---

## Common Augmentation Transformations

| Transformation | Parameter | Description |
|---------------|-----------|-------------|
| **Rotation** | `rotation_range` | Rotate image by up to N degrees |
| **Zoom** | `zoom_range` | Zoom in/out within a range |
| **Horizontal Flip** | `horizontal_flip=True` | Mirror image left-right |
| **Width Shift** | `width_shift_range` | Shift image horizontally |
| **Height Shift** | `height_shift_range` | Shift image vertically |
| **Shear** | `shear_range` | Apply shear transformation |
| **Brightness** | `brightness_range` | Adjust brightness |
| **Rescale** | `rescale=1./255` | Normalize pixel values to [0, 1] |

> **Vertical Flip** is generally **not used** for natural images (e.g., cats/dogs don't appear upside down in real life)

---

## Using ImageDataGenerator on a Single Image

### Step 1: Import and Load

```python
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Load image
img = image.load_img('path/to/image.jpg', target_size=(210, 210))
```

> `image.load_img()` returns a **PIL Image object** (not a NumPy array)

### Step 2: Create the Generator

```python
datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)
```

### Step 3: Convert Image and Generate

```python
import numpy as np

# Convert PIL image to NumPy array
img_array = image.img_to_array(img)     # Shape: (210, 210, 3)

# Add batch dimension
img_array = img_array.reshape(1, 210, 210, 3)  # Shape: (1, 210, 210, 3)

# Generate augmented images
i = 0
for batch in datagen.flow(img_array, batch_size=1, save_to_dir='output/', save_format='jpeg'):
    i += 1
    if i >= 10:    # Generate 10 images
        break
```

---

## Fill Mode Options

When transformations shift pixels, empty regions appear at the borders. The `fill_mode` parameter controls how to fill them:

| Fill Mode | Behavior | Example |
|-----------|----------|---------|
| `nearest` | Fill with nearest pixel values | Most commonly used |
| `reflect` | Mirror/reflect the border pixels | Edge pixels reflected back |
| `constant` | Fill with a constant value (black/zero) | Useful when background is dark |
| `wrap` | Wrap around to the opposite edge | Less commonly used |

---

## Using Data Augmentation During Training

### For Training Data (with augmentation)

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'data/train/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)
```

### For Validation/Test Data (NO augmentation)

```python
test_datagen = ImageDataGenerator(rescale=1./255)   # Only rescaling

validation_generator = test_datagen.flow_from_directory(
    'data/validation/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)
```

> **Never apply augmentation transformations to test/validation data** — only apply rescaling. Augmentation is strictly for training to increase data variety.

---

## Training a Model with Data Augmentation

Use `fit_generator` (or `fit` in newer Keras) instead of `model.fit(X, y)`:

```python
history = model.fit_generator(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)
```

### What Happens Internally

1. Original images are **not used directly** for training
2. Each epoch, the generator applies **random transformations** to original images on-the-fly
3. The model sees **different augmented versions** each epoch
4. This is equivalent to training on a **much larger, more diverse dataset**

---

## Practical Results: Cat vs Dog Classifier

### Setup

- Intentionally reduced training data: **only 10 cats + 10 dogs** (20 images total)
- Same CNN architecture used for both experiments

### Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D × 3 | 32, 32, 16 filters |
| MaxPooling × 3 | After each conv layer |
| Flatten | — |
| Dense | With dropout (0.5) |
| Sigmoid | Binary output |

### Results Comparison

| Metric | Without Augmentation | With Augmentation |
|--------|---------------------|-------------------|
| **Max Validation Accuracy** | ~57.8% | ~69% |
| **After extended training (100+ epochs)** | — | **~74%** |
| **Overfitting** | Severe | Reduced |

> With only 20 training images, data augmentation improved accuracy by **~16 percentage points**.

---

## Key Takeaways

| Aspect | Detail |
|--------|--------|
| **What it does** | Creates new training images via transformations on existing data |
| **When to use** | Limited data, overfitting problems |
| **Most useful in** | Medical imaging, specialized domains with scarce data |
| **Key class** | `keras.preprocessing.image.ImageDataGenerator` |
| **Training data** | Apply augmentation + rescaling |
| **Test/validation data** | Apply **only** rescaling — no augmentation |
| **How it trains** | `fit_generator()` with generators instead of `fit(X, y)` |
| **Common transforms** | Rotation, zoom, horizontal flip, shift, shear |
| **Vertical flip** | Usually avoided (unnatural for most objects) |
