# Lecture 36: Physics of LLMs — Knowledge Storage & Extraction

## 1. Physics of LLMs — Framework

### Why "Physics"?

Positioned between two extremes:

| Approach | Pros | Cons |
|---|---|---|
| **Statistical Learning Theory** | Rigorous (VC-dim, PAC bounds) | Unrealistic assumptions; only works for shallow networks; rarely connects to practice |
| **Ethology (behavioral study)** | Anyone can do it; works on large models | Subjective; prone to biases and false claims |
| **Physics of LLMs** | Controlled, repeatable, probing-enabled | Middle ground — empirical rigor with controlled synthetic data |

### Four Principles

1. **Decompose intelligence** into building blocks (knowledge, reasoning, language structure)
2. **Controlled study** with synthetic data — full control over environment, no data leakage
3. **Repeatability** — small models, vary hyperparameters systematically
4. **Probing** — inspect internal representations to understand *how* models work

---

## 2. Synthetic Data: BioS & BioR

### Setup

- **100K synthetic biographies** with 6 attributes per person:
  - Name, Birthday, Birth city, University, Company, ...
- **BioS**: generated entirely **programmatically** (fixed templates)
- **BioR**: rewritten by an LLM to add natural language variation

### Knowledge Extraction Task

Given a person's biography in pre-training data, can the model answer: *"What is [person]'s birthday?"*

---

## 3. Training Approaches for Knowledge Extraction

### Mixed Training (Pre-train + QA simultaneously)

- Works when QA data ratio ≈ 80%
- **Problem:** unrealistic — in practice you don't have QA data for all knowledge

### Pre-train then Fine-tune Pipeline

- Pre-train on biographies → fine-tune on QA
- **Fails without data augmentation** — model doesn't learn proper knowledge storage

### Data Augmentation (Critical for Success)

| Augmentation | Description | Effect |
|---|---|---|
| **Multiplicity** | Same biography rewritten in multiple ways | Forces model to store knowledge abstractly, not memorize surface patterns |
| **Permutation** | Shuffle sentence order within biography | Prevents positional memorization |
| **Full-name augmentation** | Use full name in different positions | Strengthens name↔attribute binding |

**All three combined** → model achieves near-perfect knowledge extraction after fine-tuning.

---

## 4. Probing: Where & How Knowledge Is Stored

### Probing Setup

Train a small **linear classifier** on intermediate hidden states to predict whether the model has correctly stored a particular attribute.

### Key Finding: Spurious vs. Proper Storage

| Without Augmentation | With Augmentation |
|---|---|
| Model memorizes **surface patterns** (e.g., position of birthday in text) | Model stores knowledge **by key** (person's name) |
| Probing shows high accuracy for position-based features | Probing shows high accuracy for name-based lookup |
| Fails on reworded / reordered queries | Generalizes to unseen query formats |

### Q-Probing

A variant of probing that validates whether the model's internal representation supports **query-based retrieval** (given a name, can it activate the correct attribute?).

---

## 5. Celebrity vs. Minority Data

### Finding

Knowledge about **frequently occurring entities** ("celebrities") helps the model extract knowledge about **rare entities** ("minorities").

- Training on celebrity data improves the model's general knowledge extraction pipeline
- The model learns extraction *skills* from celebrities that transfer to minorities
- **Encoder models (e.g., BERT):** not suitable for knowledge extraction — autoregressive models needed

---

## 6. Knowledge Manipulation (Introduction)

### Definition

Beyond simple extraction — the model must **combine, transform, or reason** over stored knowledge.

### Types of Knowledge Manipulation

| Task | Example |
|---|---|
| **Classification** | "Was Joe Biden born in an even year?" → extract birth year, determine even/odd |
| **Comparison** | "Who is older, person A or person B?" → extract both birth years, compare |
| **Inverse Search** | "Who was born on [date] in [city]?" → reverse lookup |
| **Partial Retrieval** | "What is [person]'s birth year?" (not full birthday) |
| **Dual Retrieval** | "Where was [person] born AND what company do they work for?" |

### Key Results

- Models can **extract** full attributes at ~100% accuracy
- **Partial retrieval** drops significantly (e.g., birth year: 20% accuracy — 80% drop)
- **Dual retrieval** shows variable performance; degrades when attributes have causal/spatial relationships
- **Classification** requires **chain-of-thought** — model must first extract, then reason. Without CoT, accuracy is very poor
- **Inverse search** is nearly **impossible** — near-zero accuracy even for GPT-3.5/4

### Chain of Thought Requirement

The model cannot directly manipulate knowledge in a single forward pass. It must:

1. **Extract** the relevant fact into its own generated context
2. **Reason** over the extracted fact

Without generating the intermediate step, the model fails — even if it "knows" the answer internally.

---

## 7. Summary of Key Findings

| Finding | Detail |
|---|---|
| Mix training is necessary | Pre-training alone insufficient for knowledge extraction |
| Data augmentation is critical | Multiplicity + permutation + full-name augmentation |
| Knowledge stored by key | With proper augmentation, model uses name-based lookup |
| Celebrity helps minority | Frequent entities improve extraction for rare ones |
| BERT-style models fail | Autoregressive models required for generation-based knowledge extraction |
| Partial/dual retrieval is hard | Significant accuracy drops even with perfect extraction capability |
| Classification needs CoT | Model must extract-then-reason; single-step manipulation fails |
| Inverse search ≈ impossible | Near-zero accuracy without explicit reverse data in training |
