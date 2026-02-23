# Lecture 11 - Efficient LLMs (Part 01)

## Overview
This lecture covers GPU memory analysis during LLM training, techniques to reduce memory — activation recomputation, gradient accumulation — and the foundations of data parallelism.

---

## 1. Three Pillars of LLM Efficiency

| Pillar | Goal | Examples |
|--------|------|----------|
| **Efficient Training** | Scale pre-training across GPU clusters | Data/Tensor/Pipeline Parallelism |
| **Efficient Implementation** | Better throughput, same algorithm | Flash Attention, Paged Attention |
| **Efficient Design** | Better architectures | Grouped Query Attention, Mixture of Experts |

**Reference**: HuggingFace Ultra-Scale Playbook

---

## 2. GPU Memory Breakdown During Training

### Four Components of GPU Memory

```python
for batch in dataloader:
    batch = batch.cuda()         # Data on GPU
    loss = model(batch)          # Forward pass → activations stored
    loss.backward()              # Backward pass → gradients computed
    optimizer.step()             # Optimizer states updated
```

| Component | When Needed | Persists After Step? |
|-----------|-------------|---------------------|
| **Model Parameters** | Always | Yes |
| **Gradients** | Backward pass | Yes (space reserved) |
| **Optimizer States** | optimizer.step() | Yes |
| **Activations** | Forward pass (for backward) | No (cleared after backward) |

### Why Activations Must Be Stored
For a linear layer $y = Wx$:

$$\frac{\partial W}{\partial L} = \frac{\partial L}{\partial y} \cdot x^T$$

The input $x$ (activation) is needed during backpropagation to compute gradients of $W$.

---

## 3. Memory Calculation for Transformer Layer

### Parameter Count Per Layer

| Component | Parameters |
|-----------|-----------|
| Token embeddings | $V \times H$ |
| QKV matrices | $3(H^2 + H)$ |
| Output projection | $H^2 + H$ |
| Layer Norm (×2) | $2 \times 2H$ |
| MLP (H → 4H → H) | $4H^2 + 4H + 4H^2 + H$ |

**Total per layer**: $\approx 12H^2 + 13H$

**Total model**: $N = L \times (12H^2 + 13H) + V \times H + 2H$

### Memory for Weights, Gradients, Optimizer

#### FP32 Training
| Component | Memory |
|-----------|--------|
| Parameters | $4N$ bytes |
| Gradients | $4N$ bytes |
| Optimizer (Adam: $m_1, m_2$) | $8N$ bytes |
| **Total** | **$16N$ bytes** |

#### Mixed Precision Training
| Component | Precision | Memory |
|-----------|-----------|--------|
| Parameters (compute) | FP16 | $2N$ |
| Gradients (compute) | FP16 | $2N$ |
| Master weights (copy) | FP32 | $4N$ |
| Optimizer $m_1$ | FP32 | $4N$ |
| Optimizer $m_2$ | FP32 | $4N$ |
| **Total** | | **$16N$ bytes** |

> **Rule of thumb**: 1B parameters ≈ 16 GB GPU memory (for training)

### Why Mixed Precision?
1. **Faster computation** (FP16 ops are faster on modern GPUs)
2. **Reduced activation memory** (activations stored in FP16)

---

## 4. Activation Memory

### Per-Layer Activation Memory

| Component | Memory |
|-----------|--------|
| QKV inputs, MLP inputs, etc. | $\sim 34 \times B \times S \times H$ |
| Attention scores (softmax) | $\sim 5 \times B \times n_h \times S^2$ |

**Key observation**: Attention is $O(S^2)$ — quadratic in sequence length!

For an 8B model with long sequences, activation memory can reach **1500+ GB**.

---

## 5. Activation Recomputation (Gradient Checkpointing)

### Idea
Instead of storing all intermediate activations, save only **checkpoints** at layer boundaries. During backward pass, **recompute** internal activations from the checkpoint.

### Strategies

| Strategy | Memory Saved | Compute Overhead |
|----------|-------------|-----------------|
| **Full recomputation** | Store only layer outputs | 30–40% extra compute |
| **Selective recomputation** | Don't store attention activations (the $S^2$ culprit), keep MLP | Up to 70% memory reduction, only 2.7% extra compute |

### Example (8B Model, batch=1, seq=4096)

| Method | Activation Memory |
|--------|------------------|
| No recomputation | 97 GB |
| Selective | 17 GB |
| Full | ~1 GB |

---

## 6. Batch Size Strategy for Pre-Training

### Critical Batch Size
- **Early training** (high loss): Use **small batch sizes** → quick, noisy updates
- **Later training** (approaching optimum): Use **large batch sizes** → stable, high-quality gradients

Batch size measured in **tokens** (not sequences):

$$\text{Token batch size} = \text{Sequence length} \times \text{Number of sequences}$$

Typical starting batch: **16M tokens**, doubling at milestones up to **128M tokens**.

---

## 7. Gradient Accumulation

### Problem
Even with activation recomputation, large batch sizes don't fit in GPU memory.

### Solution
Process micro-batches sequentially, **accumulate gradients** before updating.

```python
optimizer.zero_grad()
for i in range(gradient_accumulation_steps):
    micro_batch = get_micro_batch(i)
    loss = model(micro_batch)
    loss.backward()              # Gradients accumulate (additive)
# Only update once after all micro-batches
optimizer.step()
```

### Key Property
Gradients are additive: $\nabla W = \nabla W_1 + \nabla W_2 + ... + \nabla W_k$

| Component | Stored | Recomputed per micro-batch |
|-----------|--------|---------------------------|
| Model parameters | ✓ | — |
| Gradients | ✓ (accumulated) | — |
| Optimizer states | ✓ | — |
| Activations | — | ✓ (per micro-batch, then cleared) |

---

## 8. Data Parallelism (DP)

### Setup
- Replicate entire model (parameters + gradients + optimizer) on each GPU
- Each GPU processes a different micro-batch

### Training Loop
```
1. Each GPU does forward + backward on its own micro-batch
2. Gradients are communicated via ALL-REDUCE (sum + broadcast)
3. Each GPU independently updates parameters → models stay in sync
```

### All-Reduce Operation
Each GPU has local gradients $g_i$. After all-reduce, every GPU has:

$$g = \frac{1}{N_{DP}} \sum_{i=1}^{N_{DP}} g_i$$

### Global Batch Size Formula

$$\text{Global Batch Size} = \text{MBS} \times \text{Grad Accum Steps} \times N_{DP}$$

### Example
| Parameter | Value |
|-----------|-------|
| Target global batch | 1024 sequences |
| Sequence length | 4096 |
| MBS (max per GPU) | 2 |
| Number of GPUs ($N_{DP}$) | 128 |
| Gradient Accumulation | $\frac{1024}{2 \times 128} = 4$ |

### Limitations of Data Parallelism
- **Assumption**: Entire model must fit on one GPU
- Communication overhead causes ~40% throughput drop at high GPU counts
- Scaling is not lossless

---

## 9. Summary of Techniques (Single GPU → Multi-GPU)

```
Problem: Model + activations don't fit in memory
    ↓
Tool 1: Activation Recomputation → 70% memory reduction, 2-3% compute cost
    ↓
Tool 2: Gradient Accumulation → Large effective batch size on limited memory
    ↓
Tool 3: Data Parallelism → Distribute micro-batches across GPUs
    ↓
Next: What if the model itself doesn't fit on one GPU? → ZeRO, Tensor Parallelism
```
