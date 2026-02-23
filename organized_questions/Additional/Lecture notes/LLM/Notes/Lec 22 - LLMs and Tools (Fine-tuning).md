# Lecture 22 — LLMs and Tools: Incorporating Tools during Fine-tuning

---

## 1. Motivation: Why Do LLMs Need Tools?

| Limitation | Example |
|-----------|---------|
| **Arithmetic errors** | GPT-3 solves only ~20% of GSM8K (grade-school math) |
| **Knowledge cutoff** | Can't answer about events after training data |
| **No real-time access** | Can't check weather, stock prices, etc. |
| **Enterprise use cases** | Need to call internal APIs, databases |

**Key insight**: LLMs are good at **language understanding and generation**, but should delegate factual retrieval, computation, and real-time tasks to **external tools**.

---

## 2. GSM8K — Calculator Tool during Fine-tuning (Cobbe et al., 2021)

### 2.1 Dataset

- **8.5K** grade-school math word problems.
- Each problem requires 2–8 reasoning steps.
- Baseline GPT-3 (175B): only ~20% accuracy.

### 2.2 Tool-Augmented Training

During fine-tuning, annotate training data so the model learns when to **invoke a calculator**:

```
Q: Natalia sold 48 clips in April. She sold half as many in May. How many in total?
A: In April Natalia sold 48 clips. In May she sold 48/2 = <<48/2=24>> 24 clips.
   Total = <<48+24=72>> 72
```

- `<<expression=result>>` marks calculator invocations.
- During training, the model **skips next-token prediction** for the result portion — the calculator provides the exact answer.
- Model learns to: (1) decide when to use the calculator, (2) formulate the expression, (3) continue generation with the result.

### 2.3 Non-differentiability of Tool Calls

The calculator is **not differentiable** — gradients can't flow through it. Solution:
- Model predicts the **call expression** (differentiable).
- Skip loss computation on the **result tokens** produced by the tool.
- Resume normal next-token prediction loss after the result.

---

## 3. Verifier Model — Test-Time Scaling (Cobbe et al., 2021)

### 3.1 Best-of-N Sampling

At inference:
1. Sample $N$ complete solutions (e.g., $N = 100$) with **high temperature**.
2. Use a **verifier model** to score each solution.
3. Return the highest-scoring solution.

### 3.2 Verifier Training

| Aspect | Detail |
|--------|--------|
| **Training data** | For each training problem, sample 100 solutions from the fine-tuned model |
| **Labels** | Check final answer against ground truth → binary correct/incorrect |
| **Architecture** | Same as the generator model; predicts correctness score |
| **Supervision** | The verifier learns to distinguish good reasoning chains from bad ones |

### 3.3 Results

Adding a verifier with best-of-100 sampling significantly boosts accuracy. This was one of the **earliest papers hinting at test-time scaling** — allocating more compute at inference yields better results.

---

## 4. TALM — Tool Augmented Language Models

### 4.1 Setup

One level of abstraction above GSM8K — multiple tools for multiple tasks:

| Tool | Task Type |
|------|-----------|
| Calculator / Math tool | Arithmetic / math word problems |
| Web Search | Open-domain QA |
| Machine Translation tool | Translation tasks |

### 4.2 Generation Format

```
Input → [Tool: translator] [Args: "la tortuga", "es-en"] [Result] → "the turtle" → [Output] the turtle
```

Model learns to:
1. Decide **if** a tool is needed.
2. Select **which** tool.
3. Predict **arguments**.
4. Incorporate the **result** into its response.

### 4.3 Self-Play Bootstrapping (Low Data)

To reduce annotation cost:
1. Start with a **small** annotated subset $D'$ per tool.
2. Fine-tune the LLM on $D'$.
3. For every example in the full dataset, sample $N$ tool predictions with high temperature.
4. Execute tools, generate final outputs.
5. If final output matches ground truth → add to training data.
6. Repeat for multiple rounds.

**Result**: ~30 points improvement. But still requires **fixed set of tools** — can't add a new tool without retraining.

### 4.4 Data Requirements

| Task | Demos Needed |
|------|-------------|
| Natural Questions (simple search) | ~150 |
| Math QA (complex argument extraction) | ~2,000 |

More complex tool usage → more training data needed.

---

## 5. PAL — Program-Aided Language Models

### 5.1 Key Idea

Instead of verbal chain-of-thought, express reasoning as **Python code**:

```python
# Q: I have a chair, two potatoes, a cauliflower, a lettuce head,
#    two tables, a cabbage, two onions, and three fridges.
#    How many vegetables do I have?

vegetables_to_count = {
    'potatoes': 2,
    'cauliflower': 1,
    'lettuce head': 1,
    'cabbage': 1,
    'onions': 2
}
answer = sum(vegetables_to_count.values())  # = 7
```

- Natural language reasoning steps are written as **Python comments**.
- Executable code handles computation and logic.
- The Python interpreter executes the code and returns the answer.

### 5.2 Advantages over Chain-of-Thought

| Aspect | CoT | PAL |
|--------|-----|-----|
| Reasoning medium | Natural language | Python code + comments |
| Computation | LLM does math (error-prone) | Python interpreter (exact) |
| Training needed | None (few-shot prompting) | None (few-shot prompting) |
| Generalization | Limited to LLM's arithmetic ability | Anything Python can express |

### 5.3 Robustness to Large Numbers

- LLMs have memorized single-digit arithmetic (>50% of GSM8K is single-digit).
- When replacing single-digit with 6–7 digit numbers: CoT performance **drops drastically**, PAL performance **barely drops**.

### 5.4 Limitations

- Only works for tasks expressible as **programs** — cannot use web search, databases, etc.
- Requires a few-shot prompt per task type.

---

## 6. Toolformer (Schick et al., 2023)

### 6.1 Key Innovation

- **Task-agnostic**: Learns tool usage from **plain text corpora** (not task-specific data).
- Works during **pre-training** / continued pre-training — not restricted to a specific downstream task.
- At test time, generalizes to unseen **tasks** (but not unseen **tools**).

### 6.2 Tools Used

| Tool | Purpose |
|------|---------|
| QA System | Factual knowledge retrieval |
| Web Search | Real-time information |
| Calculator | Arithmetic |
| Calendar | Date/time operations |
| Translation | Multilingual operations |

### 6.3 Self-Supervised Annotation Pipeline

**Step 1 — Annotate**: For each text passage, use few-shot prompting to generate potential tool calls at various positions.

**Step 2 — Execute**: Run all generated tool calls to get results.

**Step 3 — Filter**: Check whether the tool call **reduces negative log-likelihood** of subsequent tokens:

$$\text{Filter criterion}: \quad \mathcal{L}_{\text{with tool}} < \min(\mathcal{L}_{\text{without tool}},\; \mathcal{L}_{\text{with question only}})$$

| Condition | Meaning |
|-----------|---------|
| $\mathcal{L}_{\text{without tool}}$ | NLL of predicting next tokens without any API call |
| $\mathcal{L}_{\text{with question only}}$ | NLL with the paraphrased question but no execution result (captures model's inherent knowledge) |
| $\mathcal{L}_{\text{with tool}}$ | NLL with full API call + execution result |

If the tool call + result **reduces** NLL below both baselines → keep as training example.

### 6.4 Results

- Toolformer (GPT-J, 6B) beats GPT-3 (175B) on math, temporal reasoning, and multilingual tasks.
- Exception: QA (Wikipedia-based) — GPT-3 has memorized Wikipedia.

### 6.5 Summary

| Property | Value |
|----------|-------|
| Task-specific? | No — trained on plain text |
| Human annotations? | Minimal — few-shot examples per tool |
| Generalizes to new tasks? | Yes |
| Generalizes to new tools? | **No** — fixed tool set during training |

---

## 7. Limitations Common to All Fine-tuning Approaches

| Limitation | Description |
|-----------|-------------|
| **Fixed tool set** | Cannot add new tools without retraining |
| **No generalization to unseen tools** | Model has no mechanism to learn about tools not seen during training |
| **Annotation cost** | Even with bootstrapping, requires per-tool annotated examples |
| **Enterprise deployment** | Organizations need custom tools — retraining for each is impractical |

**Next step** (Lecture 23): How to teach LLMs to use **any** external API given at test time → **function calling**.

---

## 8. Summary Table

| Method | Year | Tools | Training Data | Key Innovation | Limitation |
|--------|------|-------|---------------|----------------|------------|
| GSM8K | 2021 | Calculator | 8.5K annotated | Skip loss on tool output; verifier model | Single tool, single task |
| TALM | 2022 | Multiple fixed | Small subset + self-play | Multi-tool, bootstrapped training | Fixed tools, requires annotation |
| PAL | 2022 | Python interpreter | Few-shot prompts | Code-based reasoning | Only programmable tasks |
| Toolformer | 2023 | QA, Search, Calc, Calendar, Translate | Plain text (self-supervised) | Task-agnostic, NLL-based filtering | Fixed tools |
