# CNN vs Visual Cortex — The Famous Cat Experiment & History of CNN

## Part 1: Human Visual Processing Pathway

### The Visual Pathway

```
Light → Eye (Retina) → Optic Nerve → Thalamus (LGN) → Primary Visual Cortex (V1)
```

| Stage | Structure | Function |
|---|---|---|
| 1 | **Retina** | Light-sensitive sheet; converts light to electrochemical signals |
| 2 | **Optic Nerve** | Bundle of axons carrying signals from retina |
| 3 | **Lateral Geniculate Nucleus (LGN)** | Part of thalamus; preprocessing of visual signals |
| 4 | **Primary Visual Cortex (V1)** | Main area for visual information processing |

- Light hits the retina → converted to **electrochemical signals**
- Signals travel through the optic nerve to the **thalamus** (LGN) for preprocessing
- Then projected to the **primary visual cortex** (V1) at the back of the brain

---

## Part 2: Hubel & Wiesel's Cat Experiment (1958–1968)

### Experimental Setup

- Scientists: **David Hubel** and **Torsten Wiesel**
- Subjects: Cats (and later monkeys)
- Method:
  1. Cat anesthetized (semi-conscious — brain responds but body doesn't)
  2. Electrode inserted into the **visual cortex**
  3. Screen placed in front of cat's eyes
  4. Different oriented edge stimuli shown (bars at various angles)
  5. Neural responses recorded from individual cells

### Key Observation

When recording from a **single cell** in the visual cortex:

| Stimulus Orientation | Response |
|---|---|
| Horizontal edge | No response |
| Slightly tilted | Weak response |
| More tilted toward vertical | Stronger response |
| **Perfectly vertical** | **Maximum response** |
| Tilting back toward horizontal | Response decreases |
| Horizontal again | No response |

> **Conclusion**: Each individual cell in the visual cortex responds to a **specific orientation** of edges. Different cells respond to different orientations.

---

## Part 3: Findings — Two Types of Cells

### Simple Cells

| Property | Description |
|---|---|
| Receptive Field | **Small** — processes a small area of the visual field |
| Function | **Edge detection** at a specific orientation |
| Behavior | Orientation-specific (responds to only one type of edge) |
| Also called | Feature Detectors, Orientation Detectors |
| Principle | **Spatial invariance** — fixed position sensitivity |

- A vertical-detecting simple cell fires for vertical edges only, not horizontal
- For every type of edge orientation, there exists a dedicated simple cell

### Complex Cells

| Property | Description |
|---|---|
| Receptive Field | **Larger** — processes a bigger area |
| Function | **Higher-level feature detection** |
| Behavior | Combines outputs from multiple simple cells |
| Input | Receives processed information from simple cells |

### Hierarchical Processing

```
Simple Cells (edges) → Complex Cells (shapes) → Higher Complex Cells (objects)
```

1. **Simple cells** detect basic features: individual edges at specific orientations
2. **Complex cells** combine edges into shapes (e.g., hexagon from 6 edges)
3. Further layers detect increasingly **complex patterns**
4. Eventually → full image/object recognition

> **Key Insight**: All images are fundamentally made of **edges**. Even circles can be decomposed into small edge segments. Nature's approach: detect edges first → build up to complex features.

---

## Part 4: From Biology to CNNs

### Neocognitron (1980) — Kunihiko Fukushima

| Aspect | Detail |
|---|---|
| Creator | Kunihiko Fukushima (Japanese scientist) |
| Purpose | Japanese character pattern recognition |
| Architecture | S-cells and C-cells (inspired by simple and complex cells) |
| Principle | Hierarchical feature detection — simple → complex |
| Limitation | Not very effective in practice |

- **First computational model** inspired by visual cortex findings
- Rough inspiration for CNN architecture

### LeNet (1990s) — Yann LeCun

| Aspect | Detail |
|---|---|
| Creator | Yann LeCun |
| Innovation | Backpropagation + Convolution layers + Pooling layers |
| Application | Bank check/digit recognition |
| Significance | First practical CNN; performed very well |

- Introduced **backpropagation** for training
- Used **convolution** and **pooling** layers
- Kickstarted serious CNN research

### AlexNet (2012)

| Aspect | Detail |
|---|---|
| Event | Won the **ImageNet** competition in 2012 |
| Impact | Triggered the modern deep learning revolution |
| Significance | Proved CNNs scale to large image classification tasks |

---

## Summary: Biology → CNN Connection

| Biological Concept | CNN Equivalent |
|---|---|
| Simple cells (edge detectors) | **Convolutional filters** (detect edges, textures) |
| Complex cells (combine features) | **Deeper convolutional layers** (detect shapes, patterns) |
| Hierarchical processing | **Layer stacking** (low-level → high-level features) |
| Small receptive field | **Small kernel/filter size** (3×3, 5×5) |
| Large receptive field | **Deeper layers** see more of the input |
| Visual cortex | **Full CNN architecture** |

### Timeline

```
1958-1968: Hubel & Wiesel experiments (cats/monkeys)
     ↓
1980: Neocognitron (Fukushima) — first computational model
     ↓
1990s: LeNet (Yann LeCun) — first practical CNN
     ↓
2012: AlexNet — ImageNet winner → CNN revolution
     ↓
2012+: VGG, ResNet, Inception, etc.
```
