# Lecture 34: Text Generation with Multimodal Inputs

## 1. Overview & Motivation

Multimodal text generation = taking **non-text inputs** (images, video, audio) and producing **text outputs** (captions, answers, descriptions).

### Key Applications

| Application | Description |
|---|---|
| Visual Conversation | Multi-turn dialogue grounded in images |
| Visual Question Answering (VQA) | Answer questions about an image |
| Scene Understanding | Describe objects, relationships, spatial layout |
| OCR / Document Understanding | Extract text from images |
| Image Captioning | Generate natural language descriptions |
| Video Summarization | Produce text summaries of video content |
| Speech-to-Text | Transcribe audio using multimodal models |

---

## 2. Frozen (2021)

**Core idea:** Keep the vision encoder and the language model **frozen**; only train a lightweight **projection layer** in between.

```
Image → [Frozen Vision Encoder] → visual tokens
                                      ↓
                              [Trainable Projection]
                                      ↓
Text prompt → [Frozen LM] → generated text
```

- Vision encoder: pre-trained ViT or similar
- Language model: pre-trained GPT-style decoder
- **Only the projection layer** (linear or small MLP) is trained
- Drastically reduces training cost

---

## 3. Flamingo (DeepMind, 2022)

### Architecture

| Component | Role |
|---|---|
| Vision Encoder (frozen) | Extracts per-image features |
| Perceiver Resampler | Compresses variable-length visual tokens → fixed-length set of latent vectors |
| Gated Cross-Attention | Injects visual information into the frozen LM layers |
| Frozen LM | Generates text output |

### Perceiver Resampler

- Uses a small set of **learnable query vectors**
- Cross-attends to the visual tokens to produce a **fixed number** of output tokens regardless of image resolution
- Solves the variable-length problem

### Gated Cross-Attention

Inserted between frozen LM layers:

$$y = x + \tanh(\alpha) \cdot \text{CrossAttn}(x, v)$$

- $x$: text hidden states
- $v$: visual latent vectors from perceiver resampler
- $\alpha$: learnable gating scalar (initialized to 0 so the LM is unperturbed at start)
- Allows **interleaved** image-text inputs (multiple images in a sequence)

---

## 4. BLIP (Salesforce, 2022)

### 4-Component Architecture

| Tower | Training Objective | Description |
|---|---|---|
| Image Encoder | — | ViT-based, produces visual features |
| Text Encoder | ITC (Image-Text Contrastive) | Aligns image & text in shared embedding space |
| Image-Grounded Text Encoder | ITM (Image-Text Matching) | Binary classification: does this text match this image? |
| Image-Grounded Text Decoder | LM (Language Modeling) | Causal generation conditioned on image |

### ITC Loss

$$\mathcal{L}_{\text{ITC}} = -\log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_j \exp(\text{sim}(v_i, t_j)/\tau)}$$

Contrastive loss aligning positive image-text pairs.

### Key Innovation: CapFilt

- **Cap**tioner generates synthetic captions
- **Filt**er removes noisy / incorrect captions
- Bootstraps high-quality training data from web-crawled images

---

## 5. BLIP-2 (Salesforce, 2023)

### Q-Former (Querying Transformer)

Central bridging module between frozen vision encoder and frozen LLM.

```
Frozen Image Encoder → image features
                            ↓
            Q-Former (learnable queries + cross-attention)
                            ↓
                    fixed-size output embeddings
                            ↓
                      [Frozen LLM] → text
```

### Three Attention Masks in Q-Former

| Stage | Mask Type | Objective |
|---|---|---|
| Image-Text Contrastive (ITC) | Unimodal (queries can't see text, text can't see queries) | Align representations |
| Image-Text Matching (ITM) | Bi-directional (queries ↔ text) | Binary match classification |
| Image-Grounded Text Generation | Causal (queries attend to image; text uses causal mask) | Generate text from image |

### Two-Stage Pre-training

1. **Stage 1:** Train Q-Former with frozen image encoder (ITC + ITM + LM objectives)
2. **Stage 2:** Connect Q-Former to frozen LLM via linear projection; train projection only

---

## 6. mPLUG (Alibaba)

### Asymmetric Co-Attention

- Instead of symmetric cross-attention, uses **asymmetric** attention where visual and textual branches have different attention patterns
- Visual branch attends to text differently than text attends to visual
- Reduces redundant computation while maintaining cross-modal alignment

---

## 7. LLaVA (Liu et al., 2023)

### Architecture (Simplest Effective Design)

```
Image → [Frozen CLIP ViT] → patch embeddings → [Linear Projection] → visual tokens
                                                                         ↓
Text instruction →                                              [Frozen LLM (Vicuna/LLaMA)]
                                                                         ↓
                                                                   generated response
```

| Component | Status |
|---|---|
| Vision Encoder (CLIP ViT-L/14) | Frozen |
| Projection Layer (linear / MLP) | **Trainable** |
| LLM (Vicuna / LLaMA) | Frozen |

### Training Data

- Image-caption pairs for projection alignment
- Visual QA data
- Conversational multimodal data
- First turn: randomly place image before or after text query (robustness + KV-cache efficiency)

### Why It Works

- Leverages strong pre-trained vision and language models
- Only needs to learn the mapping between their representation spaces
- Training is fast and data-efficient

---

## 8. Video-LLaMA

Extends multimodal generation to **video + audio**.

```
Video → frame sampling → [Frozen ViT per frame] → [Video Q-Former] → fixed video tokens
                                                                              ↓
Audio → spectrogram → [Audio Spectrogram Transformer] → [Audio Q-Former] → fixed audio tokens
                                                                              ↓
                                                                    [Frozen LLM] → text
```

| Branch | Encoder | Bottleneck |
|---|---|---|
| Visual | ViT (per frame) | Video Q-Former → fixed tokens |
| Audio | Audio Spectrogram Transformer (AST) | Audio Q-Former → fixed tokens |

- Q-Formers compress variable-length inputs to fixed size — critical for fitting within LLM context window (e.g., LLaMA-1 had 2K tokens)
- **Training:** pre-train video branch and audio branch separately, then train projectors jointly

---

## 9. Mini-GPT4

```
Image → [Frozen Q-Former (from BLIP-2)] → [Single Linear Layer] → [Frozen Vicuna / GPT-4]
```

- Only the **linear projection layer** is trained
- Leverages the strongest available frozen LLM
- Benefits automatically from LLM improvements

---

## 10. UI Agents (Multimodal Agentic Systems)

### Challenge

Use multimodal understanding to **interact with user interfaces** — book flights, install printers, navigate apps.

### Required Capabilities

1. **UI element recognition**: identify buttons, text fields, icons
2. **Task decomposition**: break "book a flight" into sub-steps
3. **Milestone tracking**: recognize when sub-goals are achieved
4. **Error handling**: deal with 404s, timeouts, unexpected states
5. **Backtracking / Reflection**: recognize failures, retry or adjust

### Training Approach

- Collect large datasets of **desktop/mobile screenshots** with labeled elements
- Element description, action prediction, QA about UI components
- Active research extending to **mobile screens** (more compact, complex layouts)

---

## 11. Evolution Summary

```
Frozen (2021)           → Proved frozen components + projection works
    ↓
Flamingo (2022)         → Gated cross-attention + perceiver resampler
    ↓
BLIP / BLIP-2 (2022-23) → Q-Former bridging + multi-objective training
    ↓
LLaVA (2023)            → Simplest architecture: linear projection only
    ↓
Video-LLaMA / Mini-GPT4 → Extensions to video, audio, stronger LLMs
    ↓
UI Agents               → Agentic multimodal interaction with interfaces
```

### Key Trend

Architectures became **simpler** over time — the core insight is that strong frozen encoders + strong frozen LLMs only need a thin trainable bridge.
