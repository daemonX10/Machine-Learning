# LLMs 19 — Reasoning in LLMs

## Overview

This lecture defines reasoning, categorizes its types, examines reasoning techniques (forward/backward chaining, Chain-of-Thought), benchmarks for testing reasoning, and critically evaluates whether LLMs truly reason.

---

## What Is Reasoning?

**Definition**: Deriving a **new assertion** from existing assertions, or selecting an **action** from goals and knowledge.

### What Is NOT Reasoning

| Task | Why Not Reasoning |
|---|---|
| Entity Linking | Pattern matching, not inference |
| Generative sentence construction | Language fluency, not logical derivation |
| Extractive QA | Retrieval, not inference |
| Paraphrasing | Reformulation, not new knowledge |

---

## Types of Reasoning

### 1. Deduction
- **Direction**: General rule + specific fact → specific conclusion
- **Example**: "All metals conduct electricity" + "Copper is a metal" → "Copper conducts electricity"
- **Certainty**: Conclusions are **necessarily true** if premises are true

### 2. Induction
- **Direction**: Specific instances → general rule
- **Example**: "Swan 1 is white, Swan 2 is white, ..., Swan n is white" → "All swans are white"
- **Certainty**: Conclusions are **probably true** but can be falsified

### 3. Abduction
- **Direction**: Observation → best explanation
- **Example**: "The grass is wet" → "It probably rained"
- **Certainty**: Conclusions are **plausible** but not guaranteed

### 4. Defeasible Reasoning (Non-Monotonic)
- Conclusions can be **retracted** when new information arrives
- **Example**: "Birds can fly" + "Tweety is a bird" → "Tweety can fly" ... but "Tweety is a penguin" → retract "Tweety can fly"

### Mapping to Applications

| Reasoning Type | Application Domain |
|---|---|
| Deductive | Arithmetic, formal logic, theorem proving |
| Inductive | Statistical reasoning, pattern recognition |
| Abductive | Commonsense reasoning, diagnosis |
| Spatial | Understanding physical arrangements |
| Temporal | Sequence of events, causality |

---

## NLI and Reasoning

**Natural Language Inference** (entailment/contradiction/neutral) overlaps with reasoning, but:
- Not all entailments require reasoning
- Simple lexical overlap can solve some NLI without genuine inference

---

## Judea Pearl's 3 Ladders of Causation

| Ladder | Level | Operation | Example |
|---|---|---|---|
| **1. Association** | Seeing | $P(Y \mid X)$ | "Customers who buy X also buy Y" |
| **2. Intervention** | Doing | $P(Y \mid \text{do}(X))$ | "If we raise the price, what happens to sales?" |
| **3. Counterfactual** | Imagining | $P(Y_x \mid X', Y')$ | "Would the patient have survived had they received treatment?" |

**Key Insight**: Most ML models operate at Ladder 1 (correlation). True reasoning requires Ladder 2-3 (causation/counterfactuals).

**References**: Kahneman — *Thinking, Fast and Slow*; Pearl — *The Book of Why*

---

## Reasoning Techniques

### End-to-End Reasoning
- **Black box**: input → model → output (no intermediate steps visible)
- No interpretability of reasoning process
- All inference happens in latent space

### Forward Reasoning (Forward Chaining)
- **Step-by-step**: build from premises toward conclusion
- More interpretable but computationally expensive (must explore all paths)
- Example: Chain-of-Thought prompting

### Backward Reasoning (Backward Chaining)
- **Goal-directed**: start from hypothesis, decompose into sub-questions
- Tied to specific goal → harder to generalize
- Generate sub-queries → answer them → combine

### Comparison

| Aspect | End-to-End | Forward | Backward |
|---|---|---|---|
| Interpretability | None | High | High |
| Computation | Low | High (explores all paths) | Moderate |
| Generalizability | Model-dependent | Good | Poor (goal-specific) |
| Intermediate steps | Hidden | Explicit chain | Explicit sub-questions |

### Example: "Who is older, Obama or Michelle?"

**End-to-End**: Input → "Obama" (no explanation)

**Forward (CoT)**:
1. Obama born 1961
2. Michelle born 1964
3. 1961 < 1964
4. Answer: Obama

**Backward**:
1. Decompose → "When was Obama born?" → 1961
2. Decompose → "When was Michelle born?" → 1964
3. New question → "Which is earlier, 1961 or 1964?" → 1961
4. Answer: Obama

---

## Chain-of-Thought Prompting (NeurIPS 2022, Google Brain)

### Standard vs CoT Prompting

| Type | Prompt Example | Model Behavior |
|---|---|---|
| **Standard** | Q: ... A: 11 | Directly outputs answer (often wrong) |
| **CoT** | Q: ... A: Roger started with 5. 2 cans × 3 = 6. 5 + 6 = 11. The answer is 11 | Shows reasoning steps → correct answer |

### Results

| Model | Standard Prompting | Chain-of-Thought |
|---|---|---|
| GPT-3 175B | Baseline | Improved |
| PaLM 540B | Baseline | **57% on GSM-8K** (large bump) |

**Key Finding**: CoT is an emergent ability — benefits increase dramatically with model scale.

---

## Backward Reasoning: EntailmentWriter (AI2, 2022)

### Method
1. Take hypothesis (e.g., "Paper clip conducts electricity")
2. Decompose into sub-premises ("Paper clip is made of metal" + "Metals conduct electricity")
3. Recursively decompose sub-premises
4. Assign probability to each leaf node
5. **Propagate probabilities upward** (joint probability)
6. Prune low-probability branches
7. Final proof tree with probability score

### Results
- Outperformed direct answering on OBQA and QuaRTS datasets

---

## Reasoning Benchmarks

### Logical Reasoning
| Dataset | Type |
|---|---|
| **BABEL-15** | Synthetic logical statements |
| **LogiGLUE** | Logical reasoning challenge |
| **BABEL-16** | Defeasible deductive reasoning |
| **DEAL** | Defeasible reasoning |
| **d-NLI** | Defeasible NLI |

### Natural Language Inference
| Dataset | Year |
|---|---|
| SNLI | 2015 |
| MultiNLI | 2018 |
| SciTail / ScienceNLI | Recent |

### Multi-hop QA
| Dataset | Key Feature |
|---|---|
| **HotpotQA** (EMNLP 2018) | Multi-hop with supporting context |
| **MuSiQue** | Answer NOT in context (harder) |

### Commonsense Reasoning
| Dataset | Type |
|---|---|
| **OBQA** (OpenBookQA) | What-questions with facts + common knowledge |
| **SWAG / HellaSwag** | What-if / event completion |
| **PIQA** (AAAI 2020) | Physical/physics intuition |
| **CSQA2** (NAACL 2021) | Mixed commonsense |

### Complex Reasoning
| Dataset | Scope |
|---|---|
| **MMLU** | Multi-task (57 subjects) |
| **BIG-Bench** | Challenging multi-task |
| **GSM-8K** | Arithmetic word problems |

---

## Do LLMs Actually Reason? (Critical Analysis)

### Evidence Against Genuine Reasoning

#### 1. The Reversal Curse (ICLR 2024)
- LLMs trained on "A is B" **fail** to answer "B is A"
- **Example**: Trained on "Tom Cruise's mother is Mary Lee Pfeiffer" → Cannot answer "Who is Mary Lee Pfeiffer's son?"
- Tested on: GPT-3.5 Turbo, LLaMA-7B/30B/65B
- **Implication**: Models memorize directional associations, not underlying relationships

#### 2. Counterfactual Worlds (NAACL 2024, MIT)
Simple counterfactual tweaks cause drastic performance drops on GPT-4:

| Domain | Counterfactual Tweak | Effect |
|---|---|---|
| Arithmetic | Change base 10 → base 9 | Drastic accuracy drop |
| Code execution | Array index starts at 1 (not 0) | Drastic accuracy drop |
| Syntax | Change SVO → different word order | Significant drop |
| Logic | Modified logical rules | Drop |

**Implication**: Models rely on memorized patterns, not abstract reasoning rules.

#### 3. Alice in Wonderland (AIW) Test
**Question**: "Alice has $N$ brothers and $M$ sisters. How many sisters does Alice's brother have?" → Answer: $M + 1$ (including Alice)

| Model | AIW Correct Rate | MMLU Score |
|---|---|---|
| GPT-4o | ~6/10 (best) | High |
| GPT-4o mini | Lower | High |
| Claude 3 Opus | Lower | High |
| All other models | < 2/10 (on variations) | High |

**Key Finding**: Models score high on MMLU but fail this simple reasoning task. Performance drops further with minor variations of the question.

**Interesting**: When "she" was replaced with "female Alice" (making gender explicit), models performed better — suggesting reliance on surface cues.

#### 4. Planning Failures (NeurIPS 2023, Prof. Subbarao Kambhampati's group)

**Setup**: Translate **PDDL** (Planning Domain Definition Language) problems into natural language → ask LLMs to produce action sequences.

**Benchmark**: Blocks World (robot moves balls between rooms)

| Model | Correct Plans (Blocks World) | Mystery Blocks (abstracted names) |
|---|---|---|
| GPT-4o | Low | **28.3%** |
| GPT-o1 | Better but drops with plan length | — |
| Classical Planner | **100%** (deterministic) | **100%** |

**As plan length increases** → LLM accuracy drops sharply.

**Recommendation**: Use LLMs to **generate candidate plans**, then use classical planners to **verify and select optimal plans**.

---

## Key Philosophical Question

> Is reasoning even an NLP problem?

- **NLP** = processing language (syntax, semantics)
- **NLU** = understanding intent
- **Reasoning** = deriving new knowledge from existing knowledge

**Argument**: LLMs excel at NLP/NLU but reasoning may require fundamentally different capabilities. A language model processes and generates language — it doesn't necessarily perform logical inference over abstract structures.

> "We may be asking too much of a language model when we expect it to reason."

---

## Key Takeaways

1. **Reasoning ≠ pattern matching** — LLMs often exploit surface patterns rather than genuine logical inference
2. **Chain-of-Thought helps** but is not sufficient for true reasoning (fails on counterfactuals)
3. **Reversal curse** demonstrates directional memorization, not relational understanding
4. **Counterfactual tests** expose fragility of "reasoning" — tiny changes break performance
5. **Planning ability is limited** — LLMs cannot reliably produce executable plans
6. **LLMs as reasoning assistants** (not autonomous reasoners): generate candidates → verify with formal tools
7. Models excel at **knowledge extraction** (Ladder 1) but struggle with **intervention and counterfactual reasoning** (Ladders 2-3)
