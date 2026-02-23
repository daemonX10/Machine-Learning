# LLMs 14.2 — Quantization, Pruning & Distillation

## 1. Why Model Compression?

### The Problem

- Model sizes growing exponentially over time
- Performance scales with parameters, but so do costs

### Inference Pain Points

| Concern | Impact |
|---|---|
| **GPU requirements** | Larger models need more/newer GPUs → fewer organizations can run them |
| **Latency** | Larger model = slower response; agentic workflows multiply this (multiple LLM calls per response) |
| **Inference cost** | Revenue per user must exceed serving cost; bigger model = higher cost per request |
| **Sustainability** | Massive energy consumption, data center cooling, carbon emissions |

### Real-World Example: GPT-4o vs. GPT-4o Mini

| Model | MMLU (accuracy) | MGSM | Cost (same workload) |
|---|---|---|---|
| GPT-4o | Higher | Higher | **$80** |
| GPT-4o Mini | 6 pts lower | 3 pts lower | **$3** |
| GPT-3.5 Turbo | Lower than Mini | Lower than Mini | More than Mini |

→ Slight accuracy drop, **96% cost reduction**.

---

## 2. Two Classes of Inference Optimization

| Class | Description | Examples |
|---|---|---|
| **Model Compression** | Reduce model size/precision | Quantization, Pruning, Distillation |
| **Efficient Engineering** | Same model, faster execution | Fused kernels (FlashAttention), hardware-specific optimizations |

> Efficient engineering is **lossless** (exact same output). Model compression is generally **lossy**.

---

## 3. Quantization

### Floating Point Representations

| Format | Sign | Exponent | Fraction | Total Bits | Use |
|---|---|---|---|---|---|
| **FP32** | 1 | 8 | 23 | 32 | Full precision (scientific computing) |
| **FP16** | 1 | 5 | 10 | 16 | Deep learning standard |
| **BF16** | 1 | **8** | **7** | 16 | Same range as FP32, less precision; trivial conversion (just truncate) |
| **INT8** | 1 | — | 7 | 8 | Quantized inference |
| **NF4** | — | — | — | 4 | QLoRA (normal-float quantization) |

> **BF16 advantage:** Same exponent bits as FP32 → same numerical range. Conversion = simple truncation. Used in mixed-precision training.

---

### Symmetric Quantization (FP32 → INT8)

**Step 1:** Find max absolute value:

$$\alpha = \max(|x_i|)$$

**Step 2:** Compute scale factor:

$$S = \frac{127}{\alpha}$$

**Step 3:** Quantize:

$$x_q = \text{round}(x \cdot S)$$

**Step 4:** Dequantize:

$$\hat{x} = \frac{x_q}{S}$$

**Quantization error:** $\epsilon = x - \hat{x} \neq 0$ (lossy)

> Each vector has its own $\alpha$ and $S$. For a matrix with 20,000 rows → 20,000 scale factors.

### Asymmetric Quantization

- Zero is **not** mapped to zero
- Different positive/negative ranges
- Requires additional constants (zero-point) alongside scale factor

---

### Post-Training Quantization (PTQ)

**Workflow:**

```
Train in FP16/FP32 → Compute scale factors → Quantize → INT8 Inference
```

**For weights/biases:** Fixed after training → compute scale factors once.

**For activations:** Vary with input → use a **calibration dataset** to estimate value ranges.

**Per-operation flow:**

```
FP16 input → Quantize → INT8 MatMul → Dequantize → FP16 output
         ↗ Quantize weights →  ↗
```

> Quantize before each operation, dequantize after. The **speedup from INT8 matmul** far exceeds the quantize/dequantize overhead.

### Hardware Speedup (H100 Example)

| Precision | TFLOPS |
|---|---|
| FP16 | ~1,979 |
| INT8 | **Much higher** (tensor ops) |

→ Same energy, **many more operations** with INT8.

---

### The Outlier Problem (Models > 2.7B Parameters)

**Observation:** INT8 quantization works well for models ≤ 2.7B, but **degrades significantly for larger models**.

**Cause:** **Outlier activations** — a few extreme values distort the entire quantization range.

```
Normal values: [-10, ..., 10]
Outlier:       [500]

→ α = 500
→ All normal values map to ~0 in INT8 → massive information loss
```

**Emergence:** Negligible outliers below 2.7B, but percentage increases sharply from 6B+.

### LLM.int8() — Solution

> *Dettmers et al., 2022*

**Key idea:** Separate outliers from normal values.

```
Matrix X:
  ├── Non-outlier columns → Quantize to INT8 → INT8 MatMul → Dequantize
  └── Outlier columns     → Keep in FP16    → FP16 MatMul → as-is
                                              ↓
                                        Add results together
```

1. Set a threshold for outliers
2. Separate outlier columns/rows
3. Quantize only non-outlier values
4. Compute outlier operations in original precision
5. Sum the results

→ Fixes the degradation for large models.

---

## 4. QLoRA — Quantization-Aware Training

> *Dettmers et al., 2023*

### Four Innovations

| Component | Description |
|---|---|
| **NF4 (4-bit Normal Float)** | Quantization buckets sized for Gaussian distribution (not uniform) |
| **Double Quantization** | Quantize the quantization constants themselves |
| **Paged Optimizers** | Use GPU ↔ CPU memory paging to handle memory overflow |
| **LoRA** | Low-rank adaptation for parameter-efficient fine-tuning |

### NF4 — Normal Float Quantization

- Standard quantization assumes **uniform** weight distribution → equally-spaced buckets
- Model weights actually follow a **Gaussian distribution**
- NF4 uses **unequally-spaced buckets** matching the Gaussian → each bucket has ~equal number of values
- **Result:** Better accuracy at 4-bit precision

### Double Quantization

- Quantization constants (scale factors) stored in FP32 → many constants for large models → significant memory
- **Solution:** Group constants into blocks → quantize the constants themselves
- Negligible accuracy impact, **large storage savings**

### Paged Optimizers

- If GPU memory insufficient → **page** optimizer states between GPU and CPU memory
- Similar to virtual memory paging in operating systems
- Enables fine-tuning 65B models on GPUs with only 48 GB

### QLoRA Computation Flow

$$y = W_{\text{BF16}} \cdot x + L_2 \cdot L_1 \cdot x$$

where:

$$W_{\text{BF16}} = \text{Dequant}_{\text{NF4}}\!\Big(\text{Dequant}_{\text{FP8}}\big(c_{\text{FP32}}^{(2)}, c_{\text{FP8}}^{(1)}\big), W_{\text{NF4}}\Big)$$

```
Double-quantized constants (FP32)
     ↓ dequantize
Quantization constants (FP8)
     ↓ dequantize
NF4 Weights → BF16 Weights → BF16 MatMul
                              + LoRA (BF16)
                              → BF16 Output
```

### QLoRA Results

| Quantization | Accuracy |
|---|---|
| Simple FP4 | Drop from BF16 baseline |
| **NF4** | Clear improvement over FP4 |
| **NF4 + Double Quantization** | Same accuracy as NF4 (but less storage) |
| **NF4 + DQ ≈ BF16 baseline** | ✓ Maintains half-precision performance |

### Memory Savings

| Setup | Memory for 65B Model |
|---|---|
| Full fine-tuning | ~780 GB |
| **QLoRA** | **~48 GB** |

---

## 5. Pruning

### Unstructured Pruning

**Magnitude Pruning:** Sort weights by absolute value, remove the smallest.

| % Pruned | Performance |
|---|---|
| 40% | Minimal degradation |
| 40-80% | Degradation, but recoverable with continued fine-tuning |
| 80%+ | Significant degradation |

**Wanda (Weights AND Activations):** Prune based on $|w| \times |a|$ (product of weight and activation magnitude), not just weight magnitude.

**Problem with Unstructured Pruning:** Setting values to zero doesn't help unless **hardware supports sparse operations**. The zeros still consume memory and compute time.

### Structured Pruning

Removes **structural components** aligned with hardware capabilities:

| Granularity | What's Removed |
|---|---|
| Layer-level | Entire Transformer layers |
| Attention head | Individual heads in multi-head attention |
| FFN dimensions | Hidden dimensions in feed-forward network |
| N:M sparsity | Keep N out of every M values (hardware-supported) |

### 2:4 Sparsity (NVIDIA A100+)

- For every block of 4 values, keep only 2 (50% sparsity)
- Use 2-bit indices to store which positions are non-zero
- **Hardware-accelerated** → actual speedup (unlike unstructured pruning)

| Precision | Dense TFLOPS | 2:4 Sparse TFLOPS |
|---|---|---|
| FP16 | X | **~2X** |

---

## 6. Knowledge Distillation

### Core Concept

Train a **smaller student** model to mimic a **larger teacher** model.

```
Input → Teacher (large, frozen) → Soft predictions q
Input → Student (small, trainable) → Predictions p
                                      ↓
              Loss = CrossEntropy(p, q)  or  KL(p ‖ q)
```

### Hard vs. Soft Targets

| Type | Target | Quality |
|---|---|---|
| **Hard targets** | argmax of teacher output (one-hot) | Easier but loses information |
| **Soft targets** | Full probability distribution from teacher | Better — captures inter-class relationships |

> Soft targets are **significantly better** than hard targets.

### Word-Level vs. Sequence-Level Distillation

**Word-level:** At each position, match student distribution to teacher distribution:

$$\mathcal{L}_{\text{word}} = \sum_{t} \text{CrossEntropy}\big(p_\theta(\cdot | y_{<t}), \; q_T(\cdot | y_{<t})\big)$$

**Sequence-level:** Match over entire output sequences:

$$\mathcal{L}_{\text{seq}} \approx -\log p_\theta(\hat{y})$$

where $\hat{y}$ = teacher's top beam search output (approximation since exact sequence marginalization is intractable).

### Distillation Variants

| Variant | Description |
|---|---|
| **Output-only** | Match teacher's output distribution |
| **Intermediate layers** | Also match hidden representations at intermediate layers |
| **KL Divergence** | Use KL instead of cross-entropy |
| **Reverse KL** | Student tries to be "mode-seeking" rather than "mean-seeking" |

### Pros and Cons

| Pros | Cons |
|---|---|
| Student can be **much smaller** → greatest potential latency reduction | Needs **training data** (often unavailable for proprietary models) |
| Flexible architecture (define student to meet latency target) | **Most expensive** compression method (teacher inference + student training) |
| Least lossy of all compression methods | Requires representative data distribution |

---

## 7. Self-Instruct — LLM-Based Data Generation for Distillation

> *Wang et al., 2022*

### The Paradigm

Use a large model to **generate training data** for fine-tuning a smaller model.

```
175 seed tasks (hand-crafted with 1 example each)
     ↓ few-shot prompting
Large Base LLM generates:
  → New instructions
  → Input-output pairs
     ↓ quality filtering
Synthetic instruct-tuning dataset
     ↓
Fine-tune smaller model
```

### Why This Works

- Base LLMs (not instruction-tuned) can do few-shot generation well
- The 405B model exists primarily to **generate high-quality synthetic data** for distilling smaller models
- The large model captures nuances that transfer to the student

### Modern Applications

- **Synthetic data generation** for domain-specific fine-tuning
- Build natural language interfaces for software
- Generate diverse training examples from a few seeds
- **Distillation is what makes building very large models commercially viable** — they produce data for cheaper smaller models

---

## 8. Summary Comparison

| Technique | Lossy? | Training Required? | Speedup Source | Cost |
|---|---|---|---|---|
| **Quantization** | Slightly | No (PTQ) or Yes (QAT) | Reduced precision → more ops/sec | Low |
| **Pruning** | Yes | Optional fine-tuning | Fewer parameters (if hardware supports sparsity) | Low |
| **Distillation** | Least lossy | Yes (full training) | Smaller architecture | **High** |

| Technique | Best For |
|---|---|
| **Quantization** | Quick deployment with minimal accuracy loss |
| **Pruning** | Hardware with sparsity support (NVIDIA A100+) |
| **Distillation** | Maximum compression with best quality retention |
