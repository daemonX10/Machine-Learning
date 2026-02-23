# Lecture 19 — Knowledge Distillation (KD)

---

## 1. Overview & Motivation

- **Goal**: Compress a large **teacher** model into a smaller **student** model while retaining performance.
- Every major LLM release now ships a distilled variant (e.g., DeepSeek-R1 → distilled versions).
- KD is critical for **deployment on edge devices**, reducing inference cost, and meeting latency requirements.

---

## 2. Classic KD (Hinton et al., 2015)

### 2.1 Setup

| Component | Description |
|-----------|-------------|
| **Teacher** $T$ | Large pre-trained model (may be black-box or white-box) |
| **Student** $S$ | Smaller model to be trained |
| **Transfer data** | Labeled or unlabeled corpus used during distillation |

### 2.2 KD Loss

The student is trained on a **weighted combination** of two losses:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \hat{y}_S) + (1 - \alpha) \cdot \mathcal{L}_{\text{KD}}(p_T, p_S)$$

- $\mathcal{L}_{\text{CE}}$: Cross-entropy with ground-truth labels.
- $\mathcal{L}_{\text{KD}}$: Divergence between teacher and student output distributions (typically KL divergence).

### 2.3 Temperature Scaling

Soft targets are produced using a **temperature** parameter $\tau$:

$$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

| Temperature | Effect |
|-------------|--------|
| $\tau \to 0$ | Peaked distribution → higher **precision** |
| $\tau \to \infty$ | Uniform distribution → higher **recall** |

Higher temperature reveals **dark knowledge** — information in non-top logits about inter-class similarities.

---

## 3. Divergence Measures for KD

### 3.1 Forward KL Divergence

$$D_{\text{KL}}(P_T \| P_S) = \sum_x P_T(x) \log \frac{P_T(x)}{P_S(x)}$$

- **Mean-seeking**: Student tries to cover all modes of the teacher.
- Problem: Student overestimates low-probability regions (**overestimation**).

### 3.2 Reverse KL Divergence

$$D_{\text{KL}}(P_S \| P_T) = \sum_x P_S(x) \log \frac{P_S(x)}{P_T(x)}$$

- **Mode-seeking**: Student focuses on the teacher's dominant modes.
- Problem: Student ignores some modes (**underestimation**), but avoids overestimation.
- Used in **MiniLM** — showed reverse KLD fixes overestimation issues.

### 3.3 Jensen-Shannon Divergence (JSD)

$$\text{JSD}(P_T \| P_S) = \frac{1}{2} D_{\text{KL}}(P_T \| M) + \frac{1}{2} D_{\text{KL}}(P_S \| M), \quad M = \frac{P_T + P_S}{2}$$

- Symmetric, bounded, avoids extremes of forward/reverse KL.

### 3.4 Adaptive $\alpha$-$\beta$ Weighted KL (2025)

Dynamically weight forward and reverse KL components per training step.

---

## 4. Layer-wise Distillation (TinyBERT)

Instead of matching only the final output, match **intermediate representations**:

$$\mathcal{L}_{\text{layer}} = \sum_{l} \text{MSE}\big(W \cdot H_S^{(l)},\; H_T^{(\phi(l))}\big)$$

| What is Matched | Layer |
|-----------------|-------|
| Embedding layer output | Layer 0 |
| Hidden states | Intermediate layers via mapping $\phi$ |
| Attention matrices | Attention layers |
| Final logits (prediction layer) | Output |

- Requires a **linear projection** $W$ when teacher and student hidden dimensions differ.
- Student layer $l$ maps to teacher layer $\phi(l)$ (e.g., uniformly spaced).

---

## 5. Categories of Knowledge Distillation

### 5.1 White-box KD

- Full access to teacher's weights, hidden states, attention maps, logits.
- Enables layer-wise loss, attention transfer, etc.
- Example: TinyBERT, MiniLM.

### 5.2 Black-box KD

- Only access to teacher's **output** (API calls).
- Cannot access intermediate representations.
- Relies on output-level KD loss only.

### 5.3 Meta KD (Teaching the Teacher)

- Teacher adapts based on student performance — **bidirectional feedback loop**.

---

## 6. Generalized Knowledge Distillation (GKD)

**Key Insight**: Standard KD trains student on **teacher-generated** outputs, but the student will generate its **own** outputs at inference time — distribution mismatch (train-test mismatch).

### Solution: Student-Generated Output (SGO)

1. Let the student generate output sequences.
2. Score them using teacher's probability distribution.
3. Train on the student's own outputs weighted by teacher's scores.

This is analogous to **on-policy training** in RL.

---

## 7. DistilLM — Adaptive SGO with Replay Buffer

- Extends GKD with a **replay buffer** storing previously generated student samples.
- Adaptively selects between on-policy (fresh student samples) and off-policy (replay buffer) training.
- More sample-efficient than pure on-policy SGO.

---

## 8. Meta Distillation — Student-Aware Teacher

### 8.1 Teacher–Student Error Distillation (2022)

Three-stage process:

| Stage | Action |
|-------|--------|
| 1. Standard KD | Train student with usual KD loss from teacher |
| 2. Quiz evaluation | Give quiz data to student; measure error |
| 3. Teacher update | Backpropagate student's quiz error **through the student to the teacher** (end-to-end) |

- At stage 2: compute both KD loss (backpropagated to student only) and CE loss on quiz data (backpropagated end-to-end through student to teacher).
- After teacher update, repeat KD with the **updated teacher** on the **actual student model**.

### 8.2 MP-Distill — Student-Aware Meta Distillation (2024)

Four-stage process incorporating **collaboration + competition**:

**Stage 1 — Teacher Fine-tuning**: Fine-tune teacher on task-specific data (even LoRA fine-tuning helps).

**Stage 2 — Student Distillation**: Train student with three losses:
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{task}} + (1-\alpha) \cdot \mathcal{L}_{\text{KD}} + \beta \cdot \mathcal{L}_{\text{layer}}$$

**Stage 3 — Meta Teacher Training**: Train a simple MLP (meta teacher) using two losses:

| Loss | Formula Intuition | Purpose |
|------|------|---------|
| **Collaborative** | Maximize $\log p_T + \log p_S$ | Prefer both teacher and student |
| **Competitive** | Maximize $\log p_T - \log p_S$ | Prefer teacher over student |

**Stage 4 — Curriculum Learning with Policy Learner**:
- A **policy learner** (MLP) selects auxiliary tasks from a task bench.
- Student trains on selected task → takes quiz → error serves as **reward** for policy learner.
- Iterative loop: policy selects task → student trains → quiz → reward → policy updates.

**Key Result**: On GLUE, SuperGLUE, BigBench — student **beats teacher** on some tasks (counterintuitive). Curriculum learning also improves out-of-distribution generalization.

---

## 9. Empirical Insights on KD Drivers

| Factor | Finding |
|--------|---------|
| **Size gap** | Larger teacher–student size difference → more KD benefit |
| **Task-aware teacher** | Task-specific teacher always outperforms generic large teacher |
| **Noise tolerance** | Logit distribution tolerates Gaussian noise up to a threshold, then performance drops sharply |
| **Temperature** | ↑ temperature → ↑ recall; ↓ temperature → ↑ precision |

### Agreement vs Fidelity

- **Agreement**: Do teacher and student produce the same final answer?
- **Fidelity**: Do they follow the same reasoning steps?

It is possible to have high agreement but low fidelity (same answer, different reasoning) or vice versa.

---

## 10. Summary Table

| Method | Year | Type | Key Innovation |
|--------|------|------|----------------|
| Hinton KD | 2015 | White-box | Soft targets + temperature |
| TinyBERT | 2019 | White-box | Layer-wise distillation |
| MiniLM | 2020 | White-box | Reverse KLD |
| GKD | 2023 | Any | Student-generated outputs |
| DistilLM | 2024 | Any | Adaptive SGO + replay buffer |
| Teacher–Student Error Distillation | 2022 | White-box/Meta | Quiz-based teacher update |
| MP-Distill | 2024 | Meta | Collaborative + competitive losses + curriculum learning |
