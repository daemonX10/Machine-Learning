# LLMs 14.1 — Parameter Efficient Fine-Tuning (PEFT)

## 1. Context: Pre-LLM vs. LLM Era

### Pre-LLM (Transfer Learning Era)

```
Unlabeled Data → Pre-training (Word2Vec, BERT, etc.)
                     ↓
Task-Specific Data → Full Fine-Tuning
```

### LLM Era

```
Unlabeled Data → Pre-training → Instruction Tuning → Alignment
                                        ↓
                              In-Context Learning / PEFT / Full Fine-Tuning
```

---

## 2. Why Not Just In-Context Learning?

| Problem | Explanation |
|---|---|
| **Lower accuracy** | Prompting cannot match fully fine-tuned smaller models on critical tasks |
| **Prompt sensitivity** | Tiny changes (missing preposition, example order) cause large performance swings |
| **Not model-portable** | Prompts optimized for one model (LLaMA 3) may not work for another (LLaMA 3.1) |
| **Unclear model reasoning** | Gap between what you say in prompt and what model understands; spurious correlations from few-shot examples |
| **Context window waste** | Long instructions + examples consume tokens → increased latency, reduced throughput, higher energy cost |

---

## 3. Why Not Full Fine-Tuning?

| Problem | Details |
|---|---|
| **Memory** | Need 12-20× trainable parameters in memory (optimizer states, gradients, activations). LLaMA 8B in fp16 = 16 GB weights, but **~96-320 GB total for training** |
| **Storage** | Each task checkpoint = full model copy (e.g., 350 GB per task) |
| **Overfitting** | 175B parameters + 2,000 examples → complete memorization, no generalization |
| **Catastrophic forgetting** | Fine-tuning on Task B destroys performance on Task A; world knowledge lost |
| **Serving multiple tasks** | Can't host 10-15 separate full LLMs efficiently |

**Example:** Fine-tuning GPT-3 (175B) requires ~$175B \times 16 = 2.8\text{ TB}$ just for weights in fp16, plus optimizer states.

---

## 4. PEFT — The Middle Ground

**Core idea:** Freeze most parameters, train only a **tiny subset** (~0.1-3.6% of total).

### Advantages

| Advantage | Explanation |
|---|---|
| **Lower memory** | Fewer trainable params → smaller optimizer states, gradients, temp memory |
| **Faster convergence** | Frozen world model + small trainable space forces task mapping, not relearning |
| **Cheaper hardware** | Can use older GPUs (V100, T4) instead of requiring A100/H100 |
| **Reduced overfitting** | Limited capacity → less memorization of small datasets |
| **No catastrophic forgetting** | Frozen backbone preserves world knowledge; generalizes to new domains |
| **Minimal storage** | Only save incremental weights (not full model copy) |

---

## 5. Prompt Tuning (Soft Prompting)

> *Lester et al., 2021*

### Concept

Replace human-crafted "hard prompts" with **learnable embedding vectors** prepended to the input.

```
[Soft Prompt Tokens (trainable)] + [Input Tokens] → Frozen LLM → Output
      ↑ only these are trained
```

### Key Details

- Trainable parameters: only the soft prompt embeddings (e.g., 5-20 tokens × embedding dim)
- **Orders of magnitude fewer** parameters than full fine-tuning
- Everything else (all LLM weights) is **frozen**

### Initialization

- **Random init:** May not converge or takes very long
- **Task-relevant word embeddings:** Initialize with embeddings of words related to the task → better convergence
- **Sweet spot:** ~20 tokens; beyond 20 tokens, no further improvement (for T5)

### Multitask Serving

```
                 ┌─ [Task A Soft Prompt] ─┐
Input batches →  │  [Task B Soft Prompt]  │ → Shared Frozen LLM → Outputs
                 └─ [Task C Soft Prompt] ─┘
```

- Host **one LLM**, swap soft prompts per task
- Can batch different tasks together → extremely efficient serving
- Easy to scale up/down

### Performance

- With **small models:** significant gap below full fine-tuning
- With **large models (11B+):** prompt tuning **matches** full fine-tuning
- Hard prompting (manual) has a large gap even at 11B scale

### Domain Generalization

- Prompt tuning generalizes **better** to out-of-domain data than full fine-tuning
- Example: Trained on SQuAD (Wikipedia) → **+12.5% improvement** on Books dataset vs. full fine-tuning

---

## 6. Prefix Tuning

> *Li & Liang, 2021 (contemporary with Prompt Tuning)*

### Key Difference from Prompt Tuning

| | Prompt Tuning | Prefix Tuning |
|---|---|---|
| Trainable tokens at | **Input embedding layer only** | **Every layer** of the Transformer |
| Parameters | Very few | More (but still small fraction) |

### Architecture

```
For each Transformer layer:
  [Prefix Embeddings (trainable)] → influence all subsequent tokens
  via attention mechanism
```

- In a decoder-only model: prefix comes first, every input word is influenced by prefix at every layer

### Reparameterization for Stable Training

Direct optimization of prefix embeddings causes **unstable training** (gradient explosion/underflow).

**Solution:** Use a two-stage parameterization:
1. Smaller embedding $F_{\theta'}$ (e.g., $|P| \times 200$)
2. MLP projects up to hidden dim (e.g., $200 \times 1024$)
3. After training: **fuse both** into a single prefix matrix (same final size)

### Results

- **0.1% of parameters** → comparable to full fine-tuning
- Evaluated on: table-to-text generation (GPT-2), summarization (BART)
- Good out-of-domain generalization (same as prompt tuning)

---

## 7. Adapters

> *Houlsby et al., 2019 — first popular PEFT technique*

### Architecture

New adapter layers inserted **inside each Transformer block**:

```
Input → Multi-Head Attention → Add & LayerNorm
                                      ↓
                              ┌─ Adapter Layer ─┐
                              │  Down-project    │  (hidden_dim → m)
                              │  NonLinearity    │
                              │  Up-project      │  (m → hidden_dim)
                              │  + Residual      │
                              └──────────────────┘
                                      ↓
      → Feed-Forward → Add & LayerNorm
                              ↓
                      ┌─ Adapter Layer ─┐
                      │  (same as above) │
                      └──────────────────┘
```

### Bottleneck Structure

$$\text{Params} = 2 \times d \times m \quad \text{(vs. } d \times d \text{ without bottleneck)}$$

- $d$ = hidden dimension (e.g., 1024), $m$ = bottleneck dimension (e.g., 24)
- **Example:** $2 \times 1024 \times 24 = 49{,}152$ vs. $1024 \times 1024 = 1{,}048{,}576$

### Residual Connection

Critical for stable training:
- At initialization (with small Gaussian weights), adapter output ≈ 0
- Residual bypass ensures model behaves like original LLM in epoch 1

### Results

- **3.6% of parameters** → matches full fine-tuning (on BERT)
- But prefix tuning achieves the same with only 0.1% → prefix tuning generally preferred

### Disadvantages

| Issue | Explanation |
|---|---|
| **Inference overhead** | New layers add latency (sequential computation) |
| **Architecture modification** | Harder to swap adapters at serving time |
| **Suboptimal parameter efficiency** | Later methods (prefix tuning, LoRA) achieve same results with fewer params |

---

## 8. LoRA (Low-Rank Adaptation)

> *Hu et al., 2021 — most popular PEFT technique*

### Theoretical Foundation

#### Intrinsic Dimensionality (2018)

- Flatten all model parameters into one vector $\theta \in \mathbb{R}^D$
- Fine-tuning: $\theta = \theta_0 + \Delta\theta$
- **Key insight:** For a random projection matrix $P \in \mathbb{R}^{D \times d}$, there exists a $\theta_d \in \mathbb{R}^d$ (where $d \ll D$) such that:

$$\theta_0 + P \cdot \theta_d \quad \text{achieves same performance as full fine-tuning}$$

- **Intrinsic dimensionality:** The smallest $d$ achieving 90% of full fine-tuning performance

#### Structure-Aware Intrinsic Dimensionality (2021)

- Instead of one giant vector, decompose **layer-wise**
- Add per-layer scaling factors $\lambda_i$
- **Result:** Much lower intrinsic dimensionality when structure is exploited
- **Finding:** As model size increases, intrinsic dimensionality **decreases** (larger models need fewer parameters per task)

### LoRA Core Idea

Decompose weight updates as a **low-rank matrix factorization**:

$$W' = W + \Delta W = W + B \cdot A$$

where:
- $W \in \mathbb{R}^{d \times d}$ — frozen pre-trained weights
- $A \in \mathbb{R}^{r \times d}$ — down-projection (trainable)
- $B \in \mathbb{R}^{d \times r}$ — up-projection (trainable)
- $r \ll d$ — rank (hyperparameter)

```
         ┌─── W (frozen) ──────┐
Input x → │                      │ → W·x + B·A·x = Output
         └─── A → B (trainable) ┘
```

### No Nonlinearity

- Unlike adapters, LoRA has **no activation function** between $A$ and $B$
- Reason: $\Delta W = BA$ must be a valid weight matrix that can be **fused** back into $W$ after training
- **Inference:** $W' = W + BA$ → zero inference overhead!

### Initialization

| Matrix | Initialization | Reason |
|---|---|---|
| $B$ | **Zero** | $BA = 0$ initially → model starts with original behavior |
| $A$ | **Gaussian** (zero mean, small variance) | Stable training; prevents sudden large updates |

### Which Weights to Modify?

Based on ablation studies:

| Matrices | Recommended |
|---|---|
| $W_Q$ (Query) | ✓ |
| $W_K$ (Key) | ✓ |
| $W_V$ (Value) | ✓ |
| $W_O$ (Output projection) | ✓ |
| Feed-forward layers | Usually not needed |

> Best practice: **Apply LoRA to Q, K, V, and O** in attention layers.

### Effect of Rank

- Performance rises quickly, saturates around $r = 4$, then slightly decreases
- **Rank acts as a proxy for intrinsic dimensionality:**
  - Easy task + strong base model → small $r$ (even $r = 1$)
  - Hard task + weak base model → larger $r$

### Results

| Method | Trainable Params | Performance |
|---|---|---|
| Full fine-tuning | 100% | Baseline |
| BitFit (biases only) | ~0.1% | Lower |
| Prompt tuning | ~0.01% | Good for large models |
| Prefix tuning | ~0.1% | Good |
| Adapters | ~3.6% | Matches FT (but latency) |
| **LoRA** | **~0.1-2%** | **Matches or exceeds FT** |

### LoRA vs. Adapters

| Feature | Adapters | LoRA |
|---|---|---|
| Architecture | New layers inside Transformer | Parallel low-rank matrices |
| Inference overhead | Yes (extra layers) | **None** (fuse $BA$ into $W$) |
| Nonlinearity | Yes | No |
| Swappability | Difficult | Can hot-swap at batch level |

---

## 9. LoRA Extensions

| Variant | Innovation |
|---|---|
| **QLoRA** | Quantized LoRA — 4-bit quantized base model + LoRA (drastically reduces memory) |
| **LongLoRA** | LoRA for long context lengths |
| **LoRA+** | Different learning rates for $A$ and $B$ matrices → faster convergence |
| **DoRA** | Automatically determines optimal rank $r$ without grid search |

---

## 10. Summary Comparison

| Technique | Trainable Location | Params | Inference Overhead | Key Feature |
|---|---|---|---|---|
| **Prompt Tuning** | Input embeddings only | Lowest | None | Best for multitask serving |
| **Prefix Tuning** | Every layer (prefix) | Low | None | Comparable to FT with 0.1% params |
| **Adapters** | New layers in Transformer | Medium | **Yes** | First popular PEFT; bottleneck design |
| **LoRA** | Parallel low-rank in attention | Low | **None** (fusable) | Most popular; theoretically grounded |
