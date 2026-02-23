# LLMs 17.2 — Multimodal Models II (Text Generation from Multimodal Inputs)

## Overview

This lecture covers **multimodal text generation models** — models that accept multimodal inputs (images, video, audio) and generate text outputs. The focus is on architectures that combine vision encoders with language models.

---

## Use Cases

| Use Case | Description |
|---|---|
| **Image Captioning** | Generate textual descriptions of images |
| **Visual QA (VQA)** | Answer questions about images |
| **Visual Conversation** | Multi-turn dialogue grounded in images |
| **Scene Understanding** | Describe spatial relationships and context |
| **Knowledge-Grounded Description** | Generate descriptions using external knowledge |

---

## 1. Frozen (2021)

**Key Idea**: Keep LLM weights **frozen**; only train the vision encoder.

- Uses a frozen LLM (e.g., GPT-like) and trains a **vision encoder** to produce embeddings compatible with the LLM's input space
- **Interleaved image-text** input: images are converted to prefix tokens via the vision encoder
- Training: only the vision encoder is updated; LLM weights are completely frozen
- Advantage: leverages the full language capability of the pre-trained LLM without catastrophic forgetting

---

## 2. Flamingo (NeurIPS 2022)

**Architecture**:
- **Frozen vision encoder**: pre-trained ResNet (NFNet)
- **Perceiver Resampler**: converts variable-size image features → fixed-size output (constant number of visual tokens regardless of input image resolution)
- **Gated Cross-Attention Dense (GATED XATTN-DENSE) layers**: interleaved between frozen LLM layers; allow visual information to attend to text

**Model Sizes**: 3B, 9B, 80B parameters

**Training Data**:

| Dataset | Description |
|---|---|
| **M3W** | MultiModal Massive Web — interleaved image-text from web |
| **ALIGN** | Image-text pairs (1.8B noisy pairs) |
| **LTIP** | Long Text & Image Pairs |
| **VTP** | Video & Text Pairs |

**Key Contributions**:
- Few-shot learning on vision-language tasks without fine-tuning
- Handles interleaved image-text sequences of arbitrary length

---

## 3. BLIP (ICML 2022)

**Architecture**: Multimodal Mixture of Encoder-Decoder with **4 towers**:

| Tower | Type | Role |
|---|---|---|
| **Unimodal Image Encoder** | ViT | Encode images independently |
| **Unimodal Text Encoder** | BERT-based | Encode text independently |
| **Image-Grounded Text Encoder** | BERT + Cross-Attention | Fuse image-text representations |
| **Image-Grounded Text Decoder** | Causal LM with Cross-Attention | Generate text conditioned on images |

**Loss Functions**:

| Loss | Applied To |
|---|---|
| **ITC** (Image-Text Contrastive) | Unimodal encoders — align image and text in shared space |
| **ITM** (Image-Text Matching) | Image-grounded text encoder — binary classification of match |
| **LM** (Language Modeling) | Image-grounded text decoder — autoregressive text generation |

**Data Cleaning Pipeline**: Rigorous filtering of noisy web-scraped image-text pairs using a captioner + filter approach (CapFilt).

---

## 4. BLIP-2 (2023)

**Core Innovation**: **Q-Former** (Querying Transformer)

### Q-Former Architecture
- Contains an **image transformer** and a **text transformer** sharing self-attention layers
- Uses **learned query vectors** that interact with frozen image encoder features via cross-attention
- Produces a fixed number of output embeddings regardless of image resolution

### Three Loss Functions with Different Attention Masking

| Loss | Attention Strategy | Purpose |
|---|---|---|
| **ITC** | Separate towers (no cross-attention between queries and text) | Align image-text |
| **ITM** | Bi-directional attention (queries attend to text and vice versa) | Match image-text |
| **LM** (Text Gen) | Autoregressive masking (causal) | Generate text |

### Two-Stage Pre-training

| Stage | What's Frozen | What's Trained | Data |
|---|---|---|---|
| **Stage 1** | Image encoder | Q-Former | Image-text pairs |
| **Stage 2** | Image encoder + LLM | Q-Former (+ projection) | Image-text pairs |

**LLM Backends**: OPT, FlanT5

---

## 5. mPLUG (2022)

**Architecture**: Cross-Modal Skip-Connected Network

- **6-layer BERT base** encoder
- **Asymmetric cross-attention** between vision and language streams
- Skip connections across modalities to preserve modality-specific features

**Loss Functions** (4 total):
1. Image-Text Contrastive (ITC)
2. Image-Text Matching (ITM)
3. Masked Language Modeling (MLM)
4. Image-grounded text generation (prefix LM)

---

## 6. LLaVA (2023)

**Architecture**: Vision Encoder + Projector + LLM

| Component | Model |
|---|---|
| Vision Encoder | ViT-L/14 (from CLIP) |
| Projector | Linear layer (maps vision tokens to LLM space) |
| LLM | LLaMA |

### Two-Stage Training

| Stage | Frozen | Trained | Data |
|---|---|---|---|
| **Stage 1** (Pre-training) | Vision encoder + LLM | Projector only | CC3M 595K filtered image-text pairs |
| **Stage 2** (Instruction Tuning) | Vision encoder | Projector + LLM | 158K instruction-following data (generated via GPT-4) |

**Instruction Data Generation**: Used GPT-4 to create multi-turn visual conversations, detailed descriptions, and complex reasoning QA from image captions and bounding boxes.

---

## 7. Video-LLaMA

**Architecture** extends LLaVA to video + audio:

| Branch | Components |
|---|---|
| **Video** | Video Q-Former (frame-level features → temporal-aware embeddings) |
| **Audio** | Audio Spectrogram Transformer + Audio Q-Former |
| **LLM** | Vicuna (LLaMA fine-tuned) |

Processes video frames and audio spectrograms independently, then fuses via Q-Former outputs fed to the LLM.

---

## 8. MiniGPT-4

**Architecture**: ViT + Q-Former + Linear Projection + Vicuna

| Component | Source |
|---|---|
| Vision Encoder | ViT (from BLIP-2) |
| Q-Former | Pre-trained from BLIP-2 |
| Projection | Single linear layer |
| LLM | Vicuna (LLaMA) |

### Two-Stage Training (similar to LLaVA)

| Stage | Description |
|---|---|
| **Stage 1** | Train projection layer on large-scale image-text data |
| **Stage 2** | Fine-tune on curated high-quality instruction data |

---

## Architecture Comparison Table

| Model | Year | Vision Encoder | Bridge Module | LLM | Frozen LLM? |
|---|---|---|---|---|---|
| Frozen | 2021 | Trained encoder | Direct embedding | GPT-like | Yes |
| Flamingo | 2022 | NFNet (frozen) | Perceiver Resampler + Gated XATTN | Chinchilla | Yes |
| BLIP | 2022 | ViT | 4-tower encoder-decoder | — | N/A (unified) |
| BLIP-2 | 2023 | ViT (frozen) | Q-Former | OPT / FlanT5 | Yes |
| mPLUG | 2022 | ViT | Asymmetric Cross-Attention | BERT-based | No |
| LLaVA | 2023 | ViT-L/CLIP (frozen) | Linear Projector | LLaMA | Stage 1: Yes |
| Video-LLaMA | 2023 | ViT + AST | Video/Audio Q-Former | Vicuna | Yes |
| MiniGPT-4 | 2023 | ViT + Q-Former (frozen) | Linear Projection | Vicuna | Yes |

---

## Key Design Principles

1. **Freeze large components** (vision encoder, LLM) to preserve pre-trained knowledge
2. **Train lightweight bridges** (projectors, Q-Formers) to align modalities
3. **Two-stage training**: first align representations, then instruction-tune
4. **Q-Former** pattern: use learnable queries to extract fixed-size representations from variable inputs
5. **Instruction data generation via GPT-4**: scalable way to create high-quality multimodal instruction data
