# Lecture 33: Multimodal Encoder Models

*Guest Lecture by Manish Gupta (Microsoft)*

---

## Vision & Language Tasks Overview

| Task | Description |
|------|-------------|
| **VQA** (Visual Question Answering) | Answer questions about an image |
| **VCR** (Visual Commonsense Reasoning) | Answer + provide rationale |
| **Referring Expressions** | Identify object described by text |
| **Image Retrieval** | Find image matching text query |
| **Document Understanding** | Extract KV pairs, classify documents |
| **Dense Prediction** | Pixel-level tasks (segmentation, depth estimation) |

---

## 1. ViT — Vision Transformer

### Architecture

1. **Tile image** into fixed-size patches (e.g., 16×16)
2. **Flatten & project** each patch via linear embedding
3. Add **position embeddings** (learnable)
4. Prepend **[CLS] token**
5. Feed through standard **Transformer encoder**
6. [CLS] output → classification head

$$\text{Input}: x \in \mathbb{R}^{H \times W \times C} \rightarrow \text{Patches}: x_p \in \mathbb{R}^{N \times (P^2 \cdot C)} \xrightarrow{E} z_0 \in \mathbb{R}^{N \times D}$$

- Trained on **ImageNet** (supervised)
- Scales well with data size and model size
- Foundation for all subsequent vision-language models

---

## 2. VisualBERT (2019) — Single-Tower

### Architecture
- **Single unified Transformer** processes both visual and text tokens
- Visual features: Extracted via **Faster R-CNN** (region-based CNN) → bounding box features
- Text: Standard BERT tokenization

```
[CLS] text_1 text_2 ... text_n [SEP] vis_1 vis_2 ... vis_m
                    ↓
           Shared Transformer Encoder
                    ↓
              [CLS] output
```

### Pre-training Objectives
1. **MLM** (Masked Language Modeling): Predict masked text tokens given visual context
2. **Sentence-Image Prediction**: Binary — does this text match this image?

### Training Data
- **MS COCO**: ~600K image-caption pairs

---

## 3. ViLBERT (2019) — Two-Tower with Cross-Attention

### Architecture
- **Separate Transformer streams** for vision and language
- Connected via **Co-Transformer layers** (cross-modal attention)

```
Visual Stream          Linguistic Stream
     │                       │
  Self-Attn               Self-Attn
     │                       │
     └──── Cross-Attn ───────┘   (Co-Transformer)
     ┌──── Cross-Attn ───────┐
     │                       │
  Self-Attn               Self-Attn
     │                       │
```

- Visual encoder: **Faster R-CNN** (same as VisualBERT)

### Pre-training Objectives
1. **MLM**: Masked language modeling
2. **Masked Multimodal Learning**: Mask some visual regions, predict their semantic class
3. **Image-Text Alignment**: Binary matching

### Training Data
- **Conceptual Captions**: ~3.3M image-text pairs (web-crawled)

### VisualBERT vs ViLBERT

| Feature | VisualBERT | ViLBERT |
|---------|-----------|---------|
| Architecture | Single-tower (shared) | Two-tower (cross-attention) |
| Interaction | Full self-attention over all tokens | Cross-attention between modalities |
| Complexity | Simpler | More complex but modular |
| Data | MS COCO (600K) | Conceptual Captions (3.3M) |

---

## 4. CLIP (OpenAI, 2021) — Contrastive Learning

### Architecture
- **Pure two-tower** model (no cross-attention)
- Image encoder: **ResNet** or **ViT**
- Text encoder: **12-layer Transformer** (BERT-style)
- No fusion — just contrastive alignment

### Contrastive Training (InfoNCE Loss)

For a batch of $N$ image-text pairs $(I_i, T_i)$:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{e^{\text{sim}(I_i, T_i)/\tau}}{\sum_{j=1}^{N} e^{\text{sim}(I_i, T_j)/\tau}} + \log \frac{e^{\text{sim}(T_i, I_i)/\tau}}{\sum_{j=1}^{N} e^{\text{sim}(T_i, I_j)/\tau}} \right]$$

- Diagonal pairs (matching) → high similarity
- Off-diagonal pairs (non-matching) → low similarity
- $\tau$: Temperature parameter

### Training Data
- **400M web image-text pairs** (WIT dataset)

### Key Result
- **Zero-shot** competitive with **supervised** baselines on ImageNet
- Enables open-vocabulary classification (no fixed label set)

---

## 5. LayoutLM — Document Understanding

### Problem
Documents are **multimodal**: text + visual layout + spatial position. Standard NLP models ignore layout.

### Architecture
- Combines: **Text tokens** + **2D position embeddings** (bounding box coordinates from OCR) + **Visual features**
- Visual encoder: **MaskEn backbone** with tiling

### Input Representation

```
[CLS] token_1 token_2 ... [SEP] image_patch_1 image_patch_2 ...
  │       │                         │
  │    text_emb + 2D_pos_emb     visual_emb + 2D_pos_emb
  │       │                         │
  └───────┴─────────────────────────┘
              Transformer Encoder
```

- 2D position embedding: Bounding box coordinates $(x_1, y_1, x_2, y_2)$ from OCR
- Special tokens ([CLS], [SEP]): Padded with $(0,0,0,0)$ box

### Pre-training Objectives
1. **Masked Vision-Language Modeling (MVLM)**: Mask text tokens, predict using both text and visual context
2. **Text-Image Matching**: Binary — does text belong to this document?
3. **Text-Image Alignment**: Given an unmasked token, predict if it was on a covered (masked) line or not → implicitly learns OCR

### Training Data
- **11M scanned documents** (self-supervised, no labeling needed)
- Uses **Microsoft OCR API** for text extraction

### Applications
- Key-Value extraction from forms
- Document VQA
- Contract parsing
- Receipt/invoice parsing

---

## 6. Vision Transformer Register Tokens (2024)

### Discovery: Attention Artifacts in ViTs

Empirical finding across multiple ViT models (DeiT, OpenCLIP, DINO):

- **2.37%** of tokens have **10x higher** output embedding norms than others
- These outlier tokens appear at **background patches** (semantically uninformative)
- Appear starting from **middle layers**, persist to final layer
- Only in **sufficiently large** models with **sufficient pre-training**
- Related to **attention sinks** phenomenon in text Transformers

### Solution: Register Tokens

- Add extra **learnable tokens** (registers) to the input sequence
- These act as "dustbins" that absorb spurious attention
- Train ViT models with registers included

**Result**: Adding register tokens during pre-training **improves accuracy** on dense prediction, object discovery, and other tasks.

---

## 7. DINOv2 / DINOv3 — Self-Supervised Vision Encoders

### Problem
ViT encoders trained for **classification** perform poorly on **dense prediction tasks** (segmentation, depth estimation, boundary detection).

### DINOv2 Architecture: Student-Teacher Framework

```
Image → Data Augmentation (global + local crops)
              │
    ┌─────────┴─────────┐
    │                   │
 Student (FS)      Teacher (FT)
 (masked input)    (unmasked input)
    │                   │
  [CLS] + patches   [CLS] + patches
```

- **Teacher**: Not separately trained — weights updated as **exponential moving average (EMA)** of student weights
- Teacher is **larger capacity** than student
- **17 billion images** used for training

### Pre-training Loss Functions

| Loss | Type | Description |
|------|------|-------------|
| **DINO Loss** | Image-level | Cross-entropy between student and teacher CLS tokens (across views) |
| **Patch Reconstruction** | Patch-level | Match student patch embeddings to teacher's unmasked patch embeddings |
| **Koleo Regularizer** | Regularization | Enforce diversity — nearest-neighbor patches should be far apart |

### The Dense Prediction Problem

As training progresses:
- **Classification accuracy** keeps improving ✅
- **Dense prediction accuracy** drops ❌

**Explanation**: CLS token absorbs all classification-relevant information, causing patch representations to lose diversity and spatial consistency.

**Evidence**: 
- CLS token becomes highly similar to foreground patches only
- Patch-to-patch similarity degrades (patches lose local consistency)

### DINOv3 Solution: Gram Anchoring

A 4th loss function applied in the **final few training epochs**:

1. Compute **Gram matrix** of patch representations: $G = X X^T$ (patch-patch correlation)
2. Compare student's Gram matrix to a **frozen teacher** from ~10K iterations earlier (when patch representations were still good)
3. Minimize difference → preserves patch-level spatial consistency

$$\mathcal{L}_{gram} = \|G_{student} - G_{teacher}^{(t - 10K)}\|$$

**Result**: After 1M+ iterations of standard training (dense prediction dropping), just **10-20K iterations** with Gram anchoring causes **dramatic improvement** in dense prediction accuracy.

### DINOv3 Model Sizes

Available as ViT models up to **7 billion parameters** (40 layers).

---

## 8. Video Encoders

### VideoClip
- Extension of CLIP for videos
- Contrastive learning: video encoder + text encoder
- Challenge: **Video-text alignment** is harder than image-text (speaker's face ≠ topic being discussed)
- Uses CNN or ViT-based frame encoders
- Videos processed as sequences of tiled frames

### ImageBind (Meta, 2023)
- **6 modalities**: Image, Video, Audio, Text, Depth, Thermal, IMU (sensor)
- All modalities mapped to a **shared embedding space**
- Training: Standard contrastive learning using **image-paired data** for each modality

| Modality Pair | Data Source |
|---------------|------------|
| Image-Text | Web crawled captions |
| Image-Depth | Depth estimation datasets |
| Image-Audio | Video frames + audio tracks |
| Image-Thermal | FLIR thermal imaging |
| Image-IMU | Sensor data paired with images |

**Key insight**: Image is the **central binding modality** — all other modalities are aligned via image pairs.

**Applications**: Cross-modal generation (audio of fire → image of fire), multimodal arithmetic (crane + waves = crane on beach).

---

## Summary: Evolution of Multimodal Encoders

| Model | Year | Architecture | Visual Encoder | Training Data | Key Innovation |
|-------|------|-------------|----------------|---------------|----------------|
| **ViT** | 2020 | Vision-only Transformer | Patch + linear proj | ImageNet | Pure Transformer for vision |
| **VisualBERT** | 2019 | Single-tower | Faster R-CNN | COCO (600K) | Unified V+L Transformer |
| **ViLBERT** | 2019 | Two-tower + cross-attn | Faster R-CNN | Conceptual Captions (3.3M) | Co-Transformer layers |
| **CLIP** | 2021 | Two-tower contrastive | ResNet/ViT | WIT (400M) | Zero-shot via contrastive |
| **LayoutLM** | 2020 | Single-tower + 2D pos | MaskEn + OCR | 11M scanned docs | Document layout understanding |
| **DINOv3** | 2025 | Student-teacher | ViT (up to 7B) | 17B images | Gram anchoring for dense prediction |
| **CLIP-Video** | 2021 | Two-tower contrastive | CNN/ViT frames | Video-text pairs | Video-text alignment |
| **ImageBind** | 2023 | Multi-tower contrastive | ViT | Multi-modal pairs | 6-modality shared space |
