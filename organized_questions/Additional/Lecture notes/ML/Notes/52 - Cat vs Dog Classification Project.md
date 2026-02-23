# Cat vs Dog Image Classification Project (CNN)

## Overview

- **Goal:** Build a CNN model that classifies an input image as either a **cat** or a **dog**
- **Dataset:** Kaggle — Dogs vs Cats
- **Environment:** Google Colab with **GPU runtime**
- **Framework:** TensorFlow / Keras

---

## Dataset Structure

```
data/
├── train/
│   ├── cats/     # ~12,500 images
│   └── dogs/     # ~12,500 images
└── test/
    ├── cats/     # Validation images
    └── dogs/     # Validation images
```

- Total training images: ~25,000
- Images are of **varying sizes** → must be resized to uniform dimensions

---

## Step 1: Setup (Google Colab)

### Enable GPU

> **Runtime → Change runtime type → Hardware accelerator → GPU**

Without GPU, training is extremely slow on image data.

### Download Dataset from Kaggle

```python
# Upload kaggle.json API key first
!pip install kaggle
!kaggle datasets download -d <dataset-path>

# Unzip
import zipfile
with zipfile.ZipFile('dogs-vs-cats.zip', 'r') as z:
    z.extractall('.')
```

---

## Step 2: Imports

```python
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
```

---

## Step 3: Load Data with Generators

### Why Use Generators?

- 25,000+ images **cannot fit in RAM** at once
- Generators load data in **batches** → only one batch in memory at a time
- Keras provides `tf.keras.utils.image_dataset_from_directory`

### Create Train and Validation Datasets

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='train/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='test/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `directory` | Path to folder | Folder containing class subfolders |
| `labels` | `'inferred'` | Automatically assigns labels from subfolder names |
| `label_mode` | `'int'` | Integer labels (0 = cat, 1 = dog) |
| `batch_size` | `32` | Number of images per batch |
| `image_size` | `(256, 256)` | All images resized to this dimension |

---

## Step 4: Normalize Pixel Values

Raw pixel values are 0–255. Normalize to [0, 1] for better training:

```python
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
```

> Without normalization, training results will be poor.

---

## Step 5: Build CNN Model (Version 1 — Baseline)

### Architecture

```python
model = Sequential()

# Conv Block 1
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv Block 2
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv Block 3
model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# FC Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### Architecture Summary

| Layer | Filters/Neurons | Output Shape |
|-------|----------------|--------------|
| Conv2D + ReLU | 32 filters, 3×3 | (254, 254, 32) |
| MaxPooling2D | 2×2 | (127, 127, 32) |
| Conv2D + ReLU | 64 filters, 3×3 | (125, 125, 64) |
| MaxPooling2D | 2×2 | (62, 62, 64) |
| Conv2D + ReLU | 128 filters, 3×3 | (60, 60, 128) |
| MaxPooling2D | 2×2 | (30, 30, 128) |
| Flatten | — | (115200,) |
| Dense + ReLU | 128 | (128,) |
| Dense + ReLU | 64 | (64,) |
| Dense + Sigmoid | 1 | (1,) |

**Total parameters:** ~4.8 million

---

## Step 6: Compile and Train

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

---

## Step 7: Analyze Results (Baseline)

### Training Curves

- **Training accuracy:** Steadily increases across epochs
- **Validation accuracy:** Fluctuates between **75–80%**, then stagnates
- **Gap between training and validation accuracy** → **overfitting**

### Loss Curves

- Training loss decreases consistently
- Validation loss **increases** after a few epochs → classic overfitting signal

---

## Step 8: Reduce Overfitting

### Techniques to Combat Overfitting

| Technique | Description |
|-----------|-------------|
| More data | Not always possible |
| **Data Augmentation** | Generate more data from existing (covered in next lecture) |
| **Batch Normalization** | Normalizes layer inputs, stabilizes training |
| **Dropout** | Randomly disables neurons during training |
| L1/L2 Regularization | Penalizes large weights |
| Reduce model complexity | Fewer layers/neurons |

### Improved Model (Version 2)

Add **BatchNormalization** after each Conv layer and **Dropout** in FC layers:

```python
model = Sequential()

# Conv Block 1
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv Block 2
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv Block 3
model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# FC Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

### V2 Results

- Validation accuracy improved to **~80%**
- Gap between training and validation curves **reduced**
- Overfitting is partially mitigated

---

## Step 9: Prediction on Unseen Data

```python
import cv2
import numpy as np

# Load and preprocess
test_img = cv2.imread('dog.jpg')
test_img = cv2.resize(test_img, (256, 256))
test_img = test_img / 255.0

# Add batch dimension: (1, 256, 256, 3)
test_input = np.expand_dims(test_img, axis=0)

# Predict
prediction = model.predict(test_input)
# Output: 1.0 → Dog, 0.0 → Cat
```

| Predicted Value | Class |
|----------------|-------|
| Close to 0 | Cat |
| Close to 1 | Dog |

> Labels are assigned alphabetically: Cat = 0, Dog = 1

---

## Complete Pipeline Summary

```
Kaggle Dataset
    ↓
image_dataset_from_directory (batch loading + resizing)
    ↓
Normalize (÷ 255)
    ↓
CNN Model (Conv2D → BatchNorm → MaxPool) × 3 → Flatten → Dense → Dropout → Sigmoid
    ↓
Compile (Adam + Binary Crossentropy)
    ↓
Train (model.fit with validation_data)
    ↓
Evaluate (accuracy/loss curves)
    ↓
Predict (resize → normalize → expand_dims → model.predict)
```

---

## Key Takeaways

| Aspect | Detail |
|--------|--------|
| **Dataset** | Kaggle Dogs vs Cats (~25K images) |
| **Input size** | 256 × 256 × 3 |
| **Architecture** | 3 Conv blocks (32→64→128) + FC layers |
| **Baseline accuracy** | ~75–80% validation |
| **After BatchNorm + Dropout** | ~80% validation, reduced overfitting |
| **Key preprocessing** | Resize all images to same dimension + normalize to [0, 1] |
| **Generator** | `image_dataset_from_directory` for memory-efficient batch loading |
| **Overfitting signals** | Training acc ↑ but validation acc flat; validation loss ↑ |
| **Next improvement** | Data augmentation (covered in separate lecture) |
