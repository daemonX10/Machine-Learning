# Pretrained Models and ImageNet

## Overview

- **Pretrained Model:** A CNN architecture that someone else has built and trained on a large dataset, which you can directly use for your own problem
- **Core Idea:** Instead of building and training your own CNN from scratch (costly + time-consuming), leverage models already trained on massive datasets
- **Key Dataset:** ImageNet — the dataset that fueled the deep learning revolution

---

## Why Use Pretrained Models?

| Problem | Explanation |
|---------|-------------|
| **Data Hungry** | Deep learning models need huge amounts of labeled data. Collecting and labeling images (e.g., 10,000+ images) is expensive and tedious |
| **Training Time** | Training on large datasets takes hours, days, or even weeks |
| **Financial Cost** | Labeling requires hiring people → significant expense |

> **Solution:** Use a pretrained model — skip the data collection and training entirely.

### Benefits of Pretrained Models

1. **No training needed** → no large dataset required (works even with zero or very little data)
2. **No training time** → use the model directly for inference

---

## ImageNet Dataset

### What Is ImageNet?

- A **visual database of images** built for computer vision research
- Created by **Fei-Fei Li** (Stanford) in collaboration with the creator of WordNet

### Key Statistics

| Property | Value |
|----------|-------|
| **Total Images** | ~14 million (1.4 crore) |
| **Categories** | ~20,000 classes |
| **Content** | Daily household items, animals, vehicles, objects |
| **Labeling** | Fully labeled — class name, breed info, visual description, bounding boxes |
| **Bounding Box Labels** | At least 1 million images with object localization annotations |

### How Was It Built?

- Used **crowdsourcing** via **Amazon Mechanical Turk**
- Workers labeled images (class identification + bounding box annotation)
- Similar to CAPTCHA-style verification programs

---

## ILSVRC — ImageNet Challenge

**Full Form:** ImageNet Large Scale Visual Recognition Challenge

### Challenge Setup

| Property | Value |
|----------|-------|
| **Started** | 2010 |
| **Dataset Used** | Subset of ImageNet — ~1 million images |
| **Classes** | 1,000 (reduced from 20,000) |
| **Goal** | Find the best image classification model |

### Timeline of Winning Models

| Year | Model | Error Rate | Type | Key Note |
|------|-------|------------|------|----------|
| 2010 | ML-based | 28% | Machine Learning | Manual feature extraction |
| 2011 | ML-based | 25% | Machine Learning | Incremental improvement |
| **2012** | **AlexNet** | **16.4%** | **Deep Learning (CNN)** | **Started the DL revolution** — 10%+ improvement over 2nd place |
| 2013 | ZFNet | 11.7% | CNN | Refined AlexNet concepts |
| 2014 | VGGNet | 7.3% | CNN | Very deep (16–19 layers), widely used |
| 2015 | GoogLeNet (Inception) | 6.7% | CNN | Inception modules |
| 2016 | ResNet | **3.5%** | CNN | Surpassed human-level performance (human error ≈ 5%) |

> **Key Insight:** As years progressed, models got deeper (more layers) → error rates dropped. ResNet surpassed human vision accuracy.

### AlexNet Architecture (2012 — The Breakthrough)

- Input: 227 × 227 color images
- Layer 1: 96 filters of 11×11, stride 4, with padding
- Max Pooling: 3×3, stride 2
- Layer 2: 256 filters of 5×5
- Max Pooling again
- Layer 3: 384 filters
- 3 Fully Connected layers: 9216 → 4096 → 4096 → 1000 (softmax)
- **Innovation:** Used ReLU activation function

---

## Common Pretrained Models in Keras

| Model | Size (MB) | Parameters (M) | Top-1 Accuracy | Top-5 Accuracy |
|-------|-----------|-----------------|----------------|----------------|
| VGG16 | 528 | 138.4 | ~71.3% | ~90.1% |
| VGG19 | 549 | 143.7 | ~71.3% | ~90.0% |
| ResNet50 | 98 | 25.6 | ~74.9% | ~92.1% |
| InceptionV3 | 92 | 23.9 | ~77.9% | ~93.7% |
| MobileNet | 16 | 4.3 | ~70.4% | ~89.5% |
| Xception | 88 | 22.9 | ~79.0% | ~94.5% |

### Accuracy Metrics

| Metric | Meaning |
|--------|---------|
| **Top-1 Accuracy** | Model's single best prediction is correct |
| **Top-5 Accuracy** | Correct class appears in the model's top 5 predictions |

---

## Using Pretrained Models in Keras — Code Example

### Universal Image Classifier with ResNet50

```python
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pretrained ResNet50 (trained on ImageNet)
model = ResNet50(weights='imagenet')

# Load and preprocess image
img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)       # Convert to batch format
x = preprocess_input(x)              # Apply ResNet50-specific preprocessing

# Predict
preds = model.predict(x)
results = decode_predictions(preds, top=3)[0]

for _, label, score in results:
    print(f"{label}: {score:.4f}")
```

### Example Results

| Input Image | Predictions |
|-------------|-------------|
| Dog | Labrador Retriever, Golden Retriever |
| Bread | French Loaf |
| Chair | Folding Chair, Rocking Chair |
| Tomato | Strawberry (confused), Hip |

---

## Key Takeaways

| Aspect | Detail |
|--------|--------|
| **What** | CNN models pre-trained on large datasets (typically ImageNet) |
| **Why** | Avoids need for large labeled datasets and long training times |
| **ImageNet** | 14M images, 20K classes — the foundation of modern computer vision |
| **ILSVRC** | Competition that drove CNN innovation (2010–2017) |
| **AlexNet (2012)** | Triggered the deep learning revolution |
| **ResNet (2016)** | Surpassed human-level accuracy (3.5% error vs 5% human) |
| **In Keras** | Available via `tensorflow.keras.applications` — VGG16, ResNet50, InceptionV3, MobileNet, etc. |
| **Next Topic** | Transfer Learning — adapting pretrained models to custom problems |
