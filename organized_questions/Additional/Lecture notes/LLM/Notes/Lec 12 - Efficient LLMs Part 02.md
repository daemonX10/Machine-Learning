# Lecture 12 - Efficient LLMs (Part 02)

## Overview
This lecture covers ZeRO (Zero Redundancy Optimizer) stages 1–3 for sharding optimizer/gradients/parameters across GPUs, communication primitives (all-reduce, reduce-scatter, all-gather), and introduces tensor parallelism for the MLP block.

---

## 1. Recap: Data Parallelism

- **Same computation, different data** on each GPU
- Model replicated fully on every GPU
- Gradients communicated via **all-reduce**
- Communication cost: $Z$ (total parameter count)
- **Assumption**: Model fits on one GPU

---

## 2. Communication Primitives

### All-Reduce
- Aggregates (e.g., sums) tensors from all GPUs
- Result available on **all** GPUs

```python
dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
tensor = torch.ones(5) * (dist.get_rank() + 1)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# Result: [6, 6, 6, 6, 6] on all 3 GPUs (1+2+3)
```

### Reduce-Scatter
- **Reduces** (sums) across GPUs, then **scatters** different chunks to different GPUs
- Each GPU gets only $\frac{1}{N}$ of the aggregated result

### All-Gather
- Each GPU contributes its shard → every GPU gets the **full** concatenated tensor

---

## 3. Overlapping Communication and Computation

### Naive Approach
```
Forward → Backward → [IDLE: All-Reduce gradients] → Optimizer Step
```
GPUs sit idle during communication.

### Optimized Approach: Bucketed Gradient Communication
- Register **backward hooks**: as each parameter's gradient is computed, start communicating immediately
- **Bucket** gradients (e.g., per layer) to avoid too many small transfers
- Communication overlaps with backward computation

```python
for p in model.parameters():
    if p.requires_grad:
        p.register_post_accumulate_grad_hook(start_communication)
```

---

## 4. ZeRO (Zero Redundancy Optimizer)

### Memory Breakdown (Mixed Precision)

| Component | Per GPU | Notation |
|-----------|---------|----------|
| Parameters (FP16) | $2Z$ | Always needed |
| Gradients (FP16) | $2Z$ | For backward |
| Optimizer: master weights (FP32) | $4Z$ | |
| Optimizer: $m_1$ (FP32) | $4Z$ | |
| Optimizer: $m_2$ (FP32) | $4Z$ | |
| **Total optimizer** | **$KZ$ ($K=12$)** | |

### ZeRO Stage 1 (Shard Optimizer)

| What's sharded | Optimizer states |
|----------------|-----------------|
| Replicated | Parameters, Gradients |
| Memory per GPU | $2Z + 2Z + \frac{KZ}{N_{DP}}$ |

**Workflow**:
1. Forward + backward as usual (all GPUs have full model)
2. All-reduce gradients
3. Each GPU updates only its **shard** of optimizer states
4. Each GPU updates its shard of parameters
5. **All-gather** updated parameters → all GPUs in sync

**Communication**: $2Z$ (all-reduce gradients + all-gather params)

### ZeRO Stage 2 (Shard Optimizer + Gradients)

| What's sharded | Optimizer states, Gradients |
|----------------|---------------------------|
| Replicated | Parameters |
| Memory per GPU | $2Z + \frac{2Z}{N_{DP}} + \frac{KZ}{N_{DP}}$ |

**Key change**: Replace all-reduce with **reduce-scatter** for gradients
- Each GPU receives only the gradient shard it needs
- Transiently needs full gradient memory per layer during backward, but discards 2/3 after communication

**Communication**: $2Z$ (reduce-scatter gradients + all-gather params)

> In practice, always use ZeRO-2 over ZeRO-1 (same communication cost, more memory savings).

### ZeRO Stage 3 (Shard Everything)

| What's sharded | Optimizer states, Gradients, Parameters |
|----------------|---------------------------------------|
| Replicated | Nothing |
| Memory per GPU | $\frac{2Z + 2Z + KZ}{N_{DP}}$ |

**Key change**: Parameters gathered **on-the-fly** during forward and backward

**Workflow**:
1. **Pre-fetch** parameters for the next layer during computation
2. Forward through layer → discard parameters
3. Re-fetch parameters during backward
4. Reduce-scatter gradients
5. Update optimizer (local shard only)
6. No final all-gather needed (params stay sharded)

**Communication**: $3Z$ (gather params for forward + gather for backward + scatter gradients)

### Pre-fetching in ZeRO-3
```
Computing layer i → simultaneously fetching parameters of layer i+1
Reduces idle time from waiting on communication
```

### Comparison Table

| | Vanilla DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|-----------|--------|--------|--------|
| **Params replicated** | Full | Full | Full | Sharded |
| **Gradients** | Full → all-reduce | Full → all-reduce | Sharded → reduce-scatter | Sharded → reduce-scatter |
| **Optimizer** | Full | Sharded | Sharded | Sharded |
| **Memory (excl. activations)** | $(4+KZ)$ | $4Z + \frac{KZ}{N}$ | $2Z + \frac{(2+K)Z}{N}$ | $\frac{(4+K)Z}{N}$ |
| **Comm. cost** | $Z$ | $2Z$ | $2Z$ | $3Z$ |

### Example: 8B Model, 8 GPUs

| Configuration | Memory (excl. activations) |
|---------------|---------------------------|
| Vanilla DP | Does not fit (>80 GB) |
| ZeRO-1 | Reduced but still tight |
| ZeRO-2 | Better |
| ZeRO-3 | Fits within 80 GB ✓ |

**Common assumption across all ZeRO stages**: Activations for at least **one sequence** must fit on one GPU.

---

## 5. Combining DP with Gradient Accumulation

$$\text{Global Batch Size} = \text{MBS} \times \text{Grad Accum Steps} \times N_{DP}$$

Gradient accumulation is compatible with all ZeRO stages. Process micro-batches sequentially within each GPU while doing DP across GPUs.

---

## 6. Practical Usage

### PyTorch FSDP (Fully Sharded Data Parallel)
```python
from torch.distributed.fsdp import fully_shard
model = load_model()
for layer in model.layers:
    fully_shard(layer)  # Shard model across GPUs
fully_shard(model)
```

### HuggingFace SFT Trainer Config
```python
SFTConfig(
    per_device_train_batch_size=4,         # MBS
    gradient_accumulation_steps=4,          # Grad accum
    gradient_checkpointing=True,            # Activation recomputation
)
# Launch: accelerate launch --config_file config.yaml train.py
```

---

## 7. Introduction to Tensor Parallelism

### Motivation
- ZeRO handles parameter/gradient/optimizer memory
- But **activation memory** for even one sequence can exceed GPU capacity
- Especially attention: $O(S^2)$ scaling

### Key Idea
Split computation of a **single input** across multiple GPUs (different operations on same data).

| Parallelism | Data | Computation |
|-------------|------|-------------|
| Data Parallel | Different data | Same operations |
| Tensor Parallel | **Same data** | **Different operations** |

### MLP Block: Column-Wise Split (First Layer)

For $U = X^T W$ where $W$ has columns $[w_1, w_2, ..., w_{4H}]$:

Split $W$ **column-wise** across GPUs:
- GPU 1: $X^T [w_1, ..., w_H]$ → first $H$ neurons
- GPU 2: $X^T [w_{H+1}, ..., w_{2H}]$ → next $H$ neurons
- etc.

✅ Splits both **parameters** and **activations** along hidden dimension

### MLP Block: Row-Wise Split (Second Layer)

For the output layer $O = U^T M$:

Split $M$ **row-wise** to match the column split of input:
- Each GPU computes a **partial** output
- Final output = **sum** of all partial outputs → requires **all-reduce**

### Combined MLP Tensor Parallelism
```
Input X (same on all GPUs)
  → Column split on W₁ (no communication needed)
  → Activation function (local)
  → Row split on W₂ (no communication needed between layers)
  → All-Reduce to get final output
```

---

## 8. Summary

| Technique | What it Addresses | Communication Cost |
|-----------|------------------|-------------------|
| Gradient Accumulation | Batch size limits | None (sequential) |
| Data Parallelism | Training speed | $Z$ |
| ZeRO-1 | Optimizer memory | $2Z$ |
| ZeRO-2 | Optimizer + gradient memory | $2Z$ |
| ZeRO-3 | All static memory | $3Z$ |
| Tensor Parallelism | Activation memory | Exposed (sync points) |
