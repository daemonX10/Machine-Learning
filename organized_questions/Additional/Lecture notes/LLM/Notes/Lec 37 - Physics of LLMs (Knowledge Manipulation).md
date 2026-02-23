# Lecture 37: Physics of LLMs — Knowledge Manipulation, Scaling Laws & Reasoning

## 1. Recap: Knowledge Storage Results

| Finding | Detail |
|---|---|
| Mix training required | Pre-train + QA for knowledge extraction |
| Data augmentation matters | Multiplicity, permutation, full-name |
| Probing reveals storage mechanism | Name-based key lookup after augmentation |
| Celebrity helps minority | Frequent entities transfer extraction skill to rare ones |
| Encoder models (BERT) unsuitable | Autoregressive models needed |

---

## 2. Knowledge Manipulation

### Definition

Knowledge manipulation = model must **extract** a fact, then **transform / combine / reason** over it before answering.

**Example:** "Was Joe Biden born in an even year?"
1. Extract: Joe Biden → born 1946
2. Reason: Is 1946 even? → Yes
3. Answer: Yes

### Failure Modes

| Failure Type | Description |
|---|---|
| Knows "even" but can't apply to 1946 | Reasoning gap |
| Can check 1946 is even, but can't link to Biden | Connection gap |
| Doesn't know what "even" means | Concept gap |
| Doesn't know Biden's birth year | Storage gap (Part 3.1) |
| Memorized the answer directly | Not true reasoning — just retrieval |

Even GPT-4 fails: asked "Was Donald Trump born on an even day?", it extracted June 14 correctly but classified it as odd.

---

## 3. Partial & Dual Retrieval

### Partial Retrieval

Ask for a **sub-component** of a stored attribute.

| Query | Full Extraction | Partial Extraction |
|---|---|---|
| "What is X's birthday?" | ~100% accuracy | — |
| "What is X's birth year?" | — | Can drop to ~20% accuracy |
| "What is X's birth month?" | — | ~82-85% accuracy |

**Key finding:** Even when the full attribute is perfectly extractable, **later tokens** (e.g., birth year after month+day) may still be very poor at partial retrieval.

### Dual Retrieval

Ask **two questions** in one query: "Where was X born AND what company does X work for?"

| Observation | Detail |
|---|---|
| Easy + Easy = Easy | If both individual tasks work well, dual retrieval also works |
| Order matters | If attributes have causal/spatial relationship, query order affects accuracy |
| Performance drops | Some combinations: 3.6% or 17% accuracy |
| Occasionally improves | Some attribute pairs synergize (e.g., name + city) |

---

## 4. Knowledge Classification & Comparison

### Classification Task

"Was [person] born in an even year?" → multi-class classification with modular arithmetic (to isolate reasoning from large-number arithmetic).

### Comparison Task

"Who is older, person A or person B?" → extract two birth years, subtract, compare.

### Results

| Setup | Accuracy |
|---|---|
| Without CoT examples | Significantly low — even for single-step manipulation |
| With CoT in training only (not at test time) | Still struggles — the hint must be present at test time too |
| With CoT at both train + test time | Performs well |

**Critical insight:** Having chain-of-thought in training data is **not sufficient** — the model needs CoT prompting at **test time** as well. This contradicts the general belief that CoT training alone teaches the model to reason internally.

### Pre-trained vs. QA Fine-tuned

The difference between pre-trained and QA fine-tuned models is **minimal** for knowledge manipulation tasks. The bottleneck is not storage — it's the manipulation itself.

---

## 5. Knowledge Inverse Search

### Task

Instead of "When was [person] born?" → ask "Who was born on [date]?"

### Results

| Model | Forward Search | Inverse Search |
|---|---|---|
| GPT-3.5 | 89% | 23% |
| GPT-4 | High | Low (similar drop) |
| Controlled GPT-2 | ~100% (forward) | ~0% (inverse) |

**Inverse search is nearly impossible** for LLMs — even state-of-the-art proprietary models.

### Why It Fails

LLMs store knowledge as **key → value** mappings (name → attributes). Inverting the lookup (attribute → name) requires a fundamentally different retrieval mechanism that autoregressive models don't naturally learn.

### Attempted Mitigations

| Technique | Result |
|---|---|
| Data augmentation (name at different positions) | Near-zero improvement |
| RAG-style context injection | Helps if relevant context is provided |
| **Line/paragraph numbering** in training data | Model can use positional indices as auxiliary lookup keys |
| Include reverse triples in training data | Works but impractical at scale |

### Proposed Turing Test

The authors propose that **inverse knowledge search** can serve as a Turing test to distinguish humans from contemporary AI — it's a task humans do naturally but LLMs fundamentally struggle with.

---

## 6. Knowledge Capacity Scaling Law

### Quantifying Knowledge

For a controlled dataset with known vocabulary and attribute distributions:

$$\text{Total Knowledge (bits)} = \sum_{\text{attributes}} \log_2(\text{number of possible values})$$

**Example:** Birthday with 12 months × 28 days × 200 years ≈ 60 bits per person.

### N-Exposure

$$\text{N-Exposure} = \text{number of times each knowledge triple is seen during training}$$

Different from epochs — the same knowledge can appear in multiple training examples.

### Universal Scaling Law: 2 Bits per Parameter

| Exposure | Capacity Ratio |
|---|---|
| 1000 exposures | **2 bits per parameter** (universal across architectures/sizes) |
| 100 exposures (under-trained) | ~1 bit per parameter |

This holds across:
- Different model sizes
- Different depths and widths
- Different data sizes and types
- Different hyperparameters

### Where Knowledge Is Stored

| Finding | Detail |
|---|---|
| Attention layers can store knowledge | Counterintuitive — previous work attributed storage only to MLP |
| Attention-only models (no MLP) | **Can** store knowledge |
| Gated MLPs (LLaMA, Mistral style) | **Underperform** simple GPT-2 MLPs for knowledge storage |
| GPT-2 simple MLP | Best capacity per parameter |

### Quantization Effects

| Precision | Effect on Capacity |
|---|---|
| FP32 → FP16 | No significant change |
| FP16 → INT8 | No significant change |
| INT8 → **INT4** | ~2× capacity drop |
| INT4 → INT2 | Severe degradation |

### Mixture of Experts

- With 32 experts, using only ~10% of total parameters during inference
- 1000 exposure: only ~30% capacity drop (robust)
- 100 exposure: ~50% capacity drop

### Junk Data Effects

| Setup | Impact |
|---|---|
| 100 exposure + junk data | **20× worse** |
| 1000 exposure + junk data | 3× worse (more robust) |
| **Mitigation:** Add domain names to data | Model can distinguish reliable vs. unreliable sources; recovers up to 10× |

---

## 7. Hidden Reasoning Process

### IGSM Dataset (Infinite Synthetic GSM)

Why not use GSM-8K?
- Too small
- Restrictive templates
- Data contamination risk
- Can't distinguish reasoning from memorization

**IGSM:** Infinite synthetic math problems with:
- Known dependency graphs between variables
- Controlled number of operations (reasoning steps)
- No common-sense reasoning confound

| Difficulty Level | Max Operations | Solution Templates |
|---|---|---|
| Medium | ≤ 15 | ~7 billion |
| Hard | ≤ 21 | > 15 trillion |

→ Brute-force memorization is impossible.

### Level 1 Reasoning (Topological Sort)

The model:
1. Identifies which variables are relevant to the question
2. Builds a dependency graph
3. Finds the **shortest reasoning path** via topological sort
4. Uses only necessary operations — near-zero unnecessary steps

**Evidence:** Probing shows the model can predict:
- $NA(a)$: whether parameter $a$ is necessary → ~99-100% accuracy
- $DEP(a, b)$: whether $a$ depends on $b$ → ~99-100% accuracy

### Level 2 Reasoning (Pre-Question Mental Computation)

**Remarkable finding:** The model builds the full dependency graph **before even seeing the question**.

| Level | Description |
|---|---|
| Level 0 | Brute-force enumeration (impossible given template space) |
| Level 1 | After question: identify relevant variables → topological sort → shortest path |
| Level 2 | **Before question:** mentally compute all-pair dependency graph → question just activates relevant nodes |

This is **opposite to human reasoning** — humans start from the question and work backward. LLMs build the complete graph first.

### V-Probing Method

```
[Problem text] → [Frozen pre-trained LLM with LoRA on embeddings] → [START] parameter_name [END]
                                                                         → [Trainable linear head] → True/False
```

### Where LLMs Fail at Reasoning

Two primary failure modes (even in GPT-4):
1. **Using unnecessary parameters** during the planning stage
2. **Using parameters not previously computed** (skipping dependency steps)

### Depth Matters More Than Size

| Model Configuration | Accuracy |
|---|---|
| Large params, 4 layers | 68-72% |
| Small params, 20 layers | 90-95% |

- Depth enables multi-step mental computation (dependency graph building)
- **Cannot be mitigated by CoT** — the depth is needed for the internal planning, not the generated output
- Complexity of mental reasoning scales **linearly with depth**

---

## 8. Language Structure (CFGs)

### Setup

Train LLMs on context-free grammars (CFGs) to study language understanding capability.

Requirements lacking in standard datasets (Penn Treebank, etc.):
- Ambiguity
- Hierarchical structure
- Complex parsing

### Results

| Capability | Accuracy |
|---|---|
| Generate strings following CFGs | ~100% |
| Generate **diverse** strings satisfying CFGs | High entropy |
| Match CFG probability distribution | Low KL divergence |

### Positional Embedding Finding

- **Absolute positional embeddings alone:** insufficient for CFG learning
- **Relative positional embeddings (RoPE, ALiBi):** necessary for learning hierarchical grammar structure

---

## 9. Summary & Future Directions

### Key Takeaways

| Area | Finding |
|---|---|
| Knowledge manipulation | Partial/dual retrieval hard; classification needs CoT; inverse search ≈ impossible |
| Scaling law | Universal 2 bits/param at 1000 exposure |
| Reasoning | LLMs exhibit Level 1 + Level 2 reasoning (topological sort, pre-question graph) |
| Failure modes | Unnecessary params or skipped dependencies |
| Architecture | Depth > size for reasoning; simple MLP > gated MLP for storage |
| Language structure | LLMs learn CFGs; need relative positional embeddings |

### Open Research Directions

1. Apply physics-of-LLMs methodology to **efficient models** (pruned, distilled)
2. Extend synthetic data generation to **multimodal scenarios**
3. Replicate findings for **low-resource languages**
4. Study **combined objectives** (reasoning + knowledge together, not in isolation)
5. Architecture suitability analysis (Part 4 of the tutorial series)
