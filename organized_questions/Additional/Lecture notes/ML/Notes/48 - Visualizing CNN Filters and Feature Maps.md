# Visualizing CNN Filters and Feature Maps

## Overview

- **Problem:** CNNs act as **black boxes** — input image goes in, prediction comes out, but what happens inside is opaque
- **Goal:** Visualize what a CNN "sees" at each layer to build intuition about how it processes images
- **Two things to visualize:**
  1. **Filters (Kernels):** The learned weight matrices themselves
  2. **Feature Maps:** The output produced when a filter convolves over an input

---

## Key Concept: Hierarchical Feature Extraction

CNNs learn features in a **hierarchy** — from simple to complex:

| Layer Depth | What It Detects | Example |
|---|---|---|
| **Early layers** (1–2) | Primitive features | Edges, color gradients, textures |
| **Middle layers** (3–5) | Parts/patterns | Eyes, nose, ears, wheels |
| **Deep layers** (6+) | High-level concepts | Faces, objects, scenes |

> As you go deeper, the **identity of the specific input** gets lost, and **general category-level features** emerge.

---

## What Are Filters vs Feature Maps?

| Term | Definition | Analogy |
|------|-----------|---------|
| **Filter (Kernel)** | The learnable weight matrix (e.g., $3 \times 3$) | The "question" the network asks |
| **Feature Map** | Output from convolving a filter over an image | The "answer" to that question |

```
Input Image  ──[Filter/Kernel]──►  Feature Map
  (what)         (what to look for)    (where it was found)
```

---

## Experiment Setup: VGG-16

VGG-16 is used because:
- It's a **pre-trained** model (trained on ImageNet — 1000 classes)
- Very **simple, uniform architecture** — all convolution layers use $3 \times 3$ filters
- Easy to inspect and understand

### VGG-16 Architecture (Quick Reference)

```
Input: 224 × 224 × 3
    ↓
Block 1: Conv(64) × 2 → MaxPool
Block 2: Conv(128) × 2 → MaxPool
Block 3: Conv(256) × 3 → MaxPool
Block 4: Conv(512) × 3 → MaxPool
Block 5: Conv(512) × 3 → MaxPool
    ↓
Flatten → Dense(4096) → Dense(4096) → Dense(1000)
```

- **13 convolutional layers** + **3 fully connected layers** = 16 layers
- Convolution layers at indices: 0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28

---

## Part 1: Visualizing Filters

### How to Extract Filters

```python
import torch
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained VGG-16
model = models.vgg16(pretrained=True)

# Get first conv layer's filters
# model.features[0] is the first Conv2d layer
filters, biases = model.features[0].weight.data, model.features[0].bias.data

# Normalize to [0, 1] for visualization
filters = (filters - filters.min()) / (filters.max() - filters.min())

# Plot first 18 out of 64 filters
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    # Each filter has 3 channels (RGB) — show each channel separately
    ax.imshow(filters[i, 0, :, :], cmap='gray')  # show first channel
    ax.axis('off')
plt.suptitle("First Conv Layer Filters (Channel 0)")
plt.show()
```

### What Do First-Layer Filters Look Like?

The first convolution layer (64 filters, $3 \times 3 \times 3$) learns:

| Filter Type | What It Looks Like | What It Detects |
|---|---|---|
| Plain color | Uniform gray/color block | Presence of a specific color |
| Horizontal gradient | Light-to-dark horizontally | Horizontal edges |
| Vertical gradient | Light-to-dark vertically | Vertical edges |
| Diagonal pattern | Angled light/dark | Diagonal edges |

> The exact filter patterns **depend on training** — different trained models may learn different filter arrangements.

---

## Part 2: Visualizing Feature Maps

### Approach

1. Take a **pre-trained model** (VGG-16)
2. Create a **truncated model** that outputs at a specific layer
3. Pass an image through it
4. Visualize the output (feature maps) at that layer

### Code: Feature Maps at a Specific Layer

```python
from torchvision import transforms
from PIL import Image

# Create a sub-model that outputs after layer N
layer_index = 0  # first conv layer
sub_model = torch.nn.Sequential(*list(model.features.children())[:layer_index + 1])

# Load and preprocess an image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open("sample_image.jpg")
input_tensor = transform(img).unsqueeze(0)  # add batch dimension

# Get feature maps
with torch.no_grad():
    feature_maps = sub_model(input_tensor)

# feature_maps shape: (1, 64, 224, 224) for first conv layer

# Plot feature maps
fig, axes = plt.subplots(8, 8, figsize=(16, 16))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[0, i, :, :].numpy(), cmap='gray')
    ax.axis('off')
plt.suptitle(f"Feature Maps after Layer {layer_index}")
plt.show()
```

---

## What Feature Maps Reveal at Each Depth

### Example: Human Face Image Through VGG-16

| Layer | What's Visible in Feature Maps |
|-------|-------------------------------|
| **Layer 1–2** (early conv) | Clear edges, textures. Image is still very recognizable. Beard, hair, facial outline all visible. |
| **Layer 5** (mid conv) | Face structure visible but losing detail. Focus shifts to general shapes — "there is a face here." |
| **Layer 9** (deeper conv) | Image broken into small abstract pieces. Individual features (eyes, nose) becoming isolated spots. |
| **Layer 13** (deep conv) | Specific spots/activations remain. Linear combination of these spots would reconstruct a face-like pattern. |
| **Layer 17** (last conv) | Highly abstract. Original identity completely lost. Only category-level information remains. |

### Visual Progression

```
Early Layers          Middle Layers         Deep Layers
┌──────────┐         ┌──────────┐         ┌──────────┐
│  Clear   │         │ General  │         │ Abstract │
│  edges,  │   →     │  shapes, │   →     │  spots,  │
│ textures │         │ outlines │         │ patterns │
└──────────┘         └──────────┘         └──────────┘
  "edges"             "there's            "is this a
                       a face"             human?"
```

---

## Why Does This Happen?

The CNN's end goal is **classification**, not reproduction:

1. It doesn't need to preserve the exact image — it needs to extract **discriminative features**
2. Early layers: detect **low-level patterns** (edges) that are common to many classes
3. Deep layers: combine low-level patterns into **class-specific signatures**
4. For a "human" class: face present → eyes present → specific proportions → human

$$\text{Deep Feature Map} = f(\text{combination of mid-level features}) = f(f(\text{edges + textures}))$$

> **Key Insight:** The network progressively **distills** the image from pixel-level detail to category-level semantics.

---

## Practical Takeaways

| Observation | Implication |
|---|---|
| Early filters look like edge detectors | CNNs rediscover basic image processing operations |
| Deeper layers lose spatial detail | Transfer learning works because early features are universal |
| Different filters in same layer detect different things | Each filter is a different "question" about the image |
| Feature maps become sparser with depth | Network becomes more selective about what it responds to |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What we visualized** | CNN filters (weights) and feature maps (outputs) at each layer |
| **Model used** | VGG-16 (pre-trained on ImageNet) |
| **Key finding** | Early layers → edges/textures; Deep layers → abstract, class-level features |
| **Why it matters** | Demystifies the CNN "black box"; builds intuition about feature hierarchy |
| **Practical use** | Debug models, understand what the network has learned, guide architecture design |
