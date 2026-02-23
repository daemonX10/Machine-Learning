# Lecture 17.1 — Multimodal Models I

> Focus: Vision + Language understanding (not generation). Generation covered in Part II.

---

## Vision & Language Tasks

| Task | Input | Output | Description |
|------|-------|--------|-------------|
| **Visual Question Answering (VQA)** | Image + Question | Answer (select or generate) | "Is there something to cut vegetables with?" |
| **Visual Commonsense Reasoning (VCR)** | Image + Question + Choices | Selected answer + Rationale | "Why is person 4 pointing at person 1?" |
| **Referring Expressions** | Image + Bounding box | Text description | Describe the highlighted region |
| **Caption-Based Image Retrieval** | Text caption | Ranked images | Find images matching the caption |
| **Multimodal Fake News Detection** | Image + Text | Binary classification | Is this multimodal content fake? |
| **Multimodal Hate Speech Detection** | Image + Text | Binary classification | Is this multimodal content hateful? |
| **Document Understanding** | Scanned document | Key-value pairs, QA | Extract structured info from scans |
| **Video QA** | Video + Question + Choices | Selected answer | Multiple-choice over video content |
| **Text-Video Retrieval** | Text query | Ranked videos | Find relevant videos |

---

## 1. Vision Transformer (ViT)

### Architecture

```
Image → Split into N fixed-size patches → Linear projection (per patch) → Add position embeddings
  → Prepend [CLS] token → Transformer Encoder → [CLS] embedding → MLP Head → Classification
```

1. **Patch embedding:** Image split into $P \times P$ patches; each patch linearly projected to $D$-dimensional embedding
2. **Position embeddings:** Added to each patch embedding
3. **Standard Transformer Encoder:** Multi-head self-attention + MLP + residual connections
4. **[CLS] token** output used for classification via MLP head

### Model Variants

| Model | Layers | Hidden Dim | Attention Heads | Parameters |
|-------|--------|------------|-----------------|------------|
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 16 | 632M |

- **Patch sizes:** ViT-L/16 uses 16×16 patches; ViT-L/14 uses 14×14 patches
- Smaller patches → longer sequence → higher accuracy but higher latency
- Pre-trained on: ImageNet-1K, ImageNet-21K, JFT-300M
- **Result:** ViT matches or exceeds ResNets on many image classification benchmarks

---

## 2. VisualBERT

### Architecture: Single-Tower Model

Text tokens and image tokens concatenated at the **input layer** → processed jointly through a standard Transformer encoder.

```
[CLS] [text tokens...] [SEP] [image tokens...] → Transformer Encoder
```

**Image tokens:** Objects detected by **Faster R-CNN** (not uniform patches). Each object region → feature vector.

**Input embeddings per position = Position Embedding + Segment Embedding + Token/Image Embedding**

- Segment embedding: distinguishes text segment vs. image segment
- Token embedding: standard token embedding for text; Faster R-CNN features for images

### Pre-training Data

- **MS COCO:** 120K images × 5 captions = 600K image-text pairs

### Pre-training Objectives

| Objective | Description |
|-----------|-------------|
| **Masked Language Modeling (MLM)** | Mask text tokens; predict using both unmasked text + all image tokens. Image tokens help predict word "tennis" by seeing a racket in the image |
| **Sentence-Image Prediction** | Binary classification: is this (image, caption) pair real or randomly paired? Half positive, half negative in each batch |

### Attention Visualization

At later layers, text tokens like "man" strongly attend to the image region containing a man → model learns **cross-modal alignment**.

---

## 3. ViLBERT (Vision-and-Language BERT)

### Architecture: Two-Tower Model with Co-Attention

Unlike VisualBERT (single tower), ViLBERT processes text and image in **separate streams**, then fuses via **Co-Transformer layers**.

```
Text tokens → Text Transformer layers ──→ Co-Attention ──→ More Text layers
Image tokens → Visual Transformer layers ─→ Co-Attention ──→ More Visual layers
```

### Co-Attention Mechanism

Standard self-attention uses Q, K, V from the same stream. Co-attention **crosses streams**:

| Stream | Query from | Keys & Values from |
|--------|-----------|-------------------|
| Visual stream | Visual | **Linguistic** |
| Linguistic stream | Linguistic | **Visual** |

This enables cross-modal information flow while maintaining modality-specific processing.

### Pre-training Data

- **Conceptual Captions:** 3M image-text pairs (scraped from web: alt-tags, captions under images)
- Automated collection → much larger than manually labeled MS COCO

### Pre-training Objectives

| Objective | Description |
|-----------|-------------|
| **Multimodal Alignment Prediction** | Predict if image and text are aligned (same as VisualBERT) |
| **Masked Multimodal Learning** | Extended from MLM: mask **both text and image** tokens. For masked image regions, predict distribution over object class labels (from Faster R-CNN) |

### Image Encoder

- Faster R-CNN pre-trained on **Visual Genome** dataset
- Detects salient objects → each object becomes an image token

---

## 4. CLIP (Contrastive Language-Image Pre-training)

### Architecture: Two-Tower with Contrastive Loss

```
Text → Text Encoder (12-layer Transformer) → Text Embedding (pooled)
Image → Image Encoder (ViT or ResNet) → Image Embedding (pooled)
                                              ↓
                                    Contrastive loss
```

### Contrastive Training

Given a batch of $N$ image-text pairs:
- $N$ positive pairs (correct image-text matches)
- $N^2 - N$ negative pairs (incorrect pairings)

**Objective:** Maximize cosine similarity for positive pairs; minimize for negatives.

$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}$$

Symmetric loss applied from both image→text and text→image directions.

### Models Used

| Component | Architecture |
|-----------|-------------|
| Text Encoder | 12-layer Transformer |
| Image Encoder | 5 ResNets **or** ViT-B/32, ViT-B/16, ViT-L/14 |

### Pre-training Data

- **WIT (Web Image Text):** 400M image-text pairs from the web
- Massive scale: 600K (VisualBERT) → 3M (ViLBERT) → **400M (CLIP)**

### Zero-Shot Classification

```
Test image → Image Encoder → Image embedding
Class labels → Text Encoder → Text embeddings (one per class)
→ Compute similarities → Predict class with highest similarity
```

**Result:** Zero-shot CLIP outperforms **fully supervised ResNet-50** on many datasets (Food-101, land use classification, object recognition, etc.). Tested on 30+ computer vision tasks.

---

## 5. LayoutLM (Visually Rich Document Understanding)

### Use Cases

- Key-value extraction from forms, invoices, receipts
- Document QA ("What is the ZIP code?")
- Document classification (invoices vs. ID documents vs. utility bills)
- Legal contract analysis

### LayoutLM V2 Architecture

Based on **UniLM V2** Transformer (base: 200M params, large: 426M params).

**Input representation (4 components):**

| Component | Description |
|-----------|-------------|
| **Segment embedding** | Image token vs. text token (+ masked/unmasked) |
| **1D Position embedding** | Standard sequential position |
| **2D Position embedding** | Bounding box coordinates ($x_{\min}, x_{\max}, y_{\min}, y_{\max}$, width, height) |
| **Token/Visual embedding** | Text: standard token embedding; Image: Mask R-CNN features |

**Document preprocessing:**
1. OCR (Microsoft Read API) extracts text from document scan
2. Some lines randomly **hidden** (covered) in the image
3. Remaining text tokenized with some tokens **masked** (standard MLM)

### Pre-training Objectives

| Objective | Description |
|-----------|-------------|
| **Masked Visual-Language Modeling** | Predict masked text tokens using surrounding text + image |
| **Text-Image Alignment** | Predict whether each text token belongs to a covered or uncovered line |
| **Text-Image Matching** | Binary: does this image match this text? |

**Pre-training data:** 11M scanned documents.

---

## 6. Video + Text: VideoCLIP

### Video as Data

A video = sequence of image frames = 3D tensor: (Height × Width × 3·num_frames).

Encoding approaches:
- **3D CNNs** (I3D, 3D ConvNets) — capture temporal structure
- **Transformer models** — frame-level ViT + temporal attention

### VideoCLIP Architecture

- **Text Encoder:** BERT-base-uncased
- **Video Encoder:** Frozen pre-trained CNN → frame embeddings → MLP projection → BERT-base Transformer

**Training:** Same contrastive loss as CLIP, but with (video, transcript) pairs.

**Alignment caveat:** Speech and visual content are often **temporally misaligned** (e.g., mentioning ingredients before showing them). Careful positive/negative pair construction needed.

**Pre-training data:** HowTo100M dataset (100M instructional video clips).

---

## 7. ImageBind (Six-Modality Model)

### Supported Modalities

| Modality | Description |
|----------|-------------|
| **Image** | Standard RGB images |
| **Text** | Natural language |
| **Audio** | Sound/speech signals |
| **Depth** | Per-pixel distance from camera (white = close, black = far) |
| **Thermal** | Infrared temperature maps (FLIR imaging) |
| **IMU** | Inertial Measurement Unit sensor data (accelerometer, gyroscope) |

### Key Insight

6 modalities → ${6 \choose 2} = 15$ possible pairwise datasets needed. But **images can bridge all modalities**:

> "An image of a beach reminds us of the sound of waves (audio), the texture of sand (depth), a breeze (thermal), and can inspire a poem (text)."

**Solution:** Train only **image-paired** data:

```
Image ↔ Text     (e.g., CLIP data)
Image ↔ Audio    (e.g., AudioSet)
Image ↔ Depth    (depth estimation datasets)
Image ↔ Thermal  (thermal imaging datasets)
Video ↔ IMU      (sensor datasets)
```

Images serve as the **binding modality** connecting all others.

### Architecture

- **Image/Video Encoder:** ViT-Huge (632M parameters)
- **Text Encoder:** OpenCLIP (302M parameters, frozen)
- **Other modalities:** Modality-specific encoders trained from scratch
- **Loss:** Symmetric InfoNCE contrastive loss (same as CLIP)

### Applications

| Application | Example |
|-------------|---------|
| **Cross-modal retrieval** | Audio of crackling fire → retrieve fire images, fire-related text, depth maps of fireplaces |
| **Embedding arithmetic** | Image of bird + audio of waves = bird on a beach |
| **Audio → Image generation** | Barking sound → generate dog image |
| **Zero-shot recognition** | Classify across modalities without task-specific training |

---

## Model Evolution Summary

| Model | Year | Architecture | Modalities | Pre-training Data | Key Innovation |
|-------|------|-------------|-----------|-------------------|----------------|
| **ViT** | 2020 | Single encoder | Image only | ImageNet | Transformer for images |
| **VisualBERT** | 2019 | Single tower | Image + Text | 600K (MS COCO) | Joint image-text encoding |
| **ViLBERT** | 2019 | Two tower + co-attention | Image + Text | 3M (Conceptual Captions) | Cross-modal attention |
| **CLIP** | 2021 | Two tower + contrastive | Image + Text | 400M (WIT) | Contrastive pre-training; zero-shot transfer |
| **LayoutLM V2** | 2020 | Transformer + 2D position | Document image + Text | 11M documents | Spatial layout understanding |
| **VideoCLIP** | 2021 | Two tower + contrastive | Video + Text | HowTo100M | Video-text contrastive learning |
| **ImageBind** | 2023 | Multi-encoder + contrastive | 6 modalities | Multiple datasets | Image as binding modality |

**Trend:** Single modality → two modalities → six+ modalities; small labeled data → massive web-scraped data; supervised → contrastive self-supervised.
