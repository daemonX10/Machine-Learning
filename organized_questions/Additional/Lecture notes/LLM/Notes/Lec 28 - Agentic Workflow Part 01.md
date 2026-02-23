# Lecture 28: Agentic Workflow Part 01

## 1. Agentic vs Non-Agentic Workflows

| Aspect | Non-Agentic | Agentic |
|--------|------------|---------|
| Flow | Single LLM call, one-shot | Iterative loop with actions + feedback |
| Example | Prompt → essay | Outline → draft → critique → revise → finalize |
| Analogy (Andrew Ng) | Writing essay in one sitting | Writing with planning, drafting, reviewing |

### Evidence: HumanEval Coding Benchmark

| Setup | Accuracy |
|-------|----------|
| GPT-3.5 (non-agentic) | 48% |
| GPT-4 (non-agentic) | 67% |
| GPT-3.5 with agentic workflow | **>70%** |

**Key insight**: A weaker model with agentic scaffolding can outperform a stronger model used non-agentically.

### Intervenor
Student-teacher prompts for iterative code generation — an early agentic coding approach.

---

## 2. ReAct Framework

### Core Idea
Interleave **Reasoning** (thought) with **Acting** (tool use) in a unified loop.

### Loop Structure
```
Thought → Action → Observation → Thought → Action → Observation → ... → Finish
```

- **Thought**: Internal reasoning about what to do next
- **Action**: Execute a tool/API call from the action space
- **Observation**: Result returned from the action

### Action Space Design (HotpotQA Example)

| Action | Description |
|--------|-------------|
| `search[query]` | Search Wikipedia for a topic |
| `lookup[string]` | Find next occurrence of string in current page |
| `finish[answer]` | Return final answer |

### Comparison of Approaches

| Method | Reasoning | Actions | Grounding |
|--------|-----------|---------|-----------|
| Standard Prompting | ✗ | ✗ | ✗ |
| Chain-of-Thought (CoT) | ✓ | ✗ | ✗ (can hallucinate facts) |
| Act-only | ✗ | ✓ | ✓ (but no reasoning) |
| **ReAct** | **✓** | **✓** | **✓** |

### ReAct Advantages
- **Grounded reasoning**: Actions retrieve real information → less hallucination
- **Traceable**: Can see exactly where information came from
- **Debuggable**: Each step is observable

### ReAct Limitations

| Issue | Description |
|-------|-------------|
| **Cyclic behavior** | Gets stuck repeating same thought-action loop |
| **More reasoning errors** | Than pure CoT on some tasks |
| **Action space dependent** | Success heavily depends on action design |

**Practical mitigations**:
- Break after $N$ iterations
- Stop if same action predicted more than twice
- Use backoff strategies

### Backoff Strategies
- **CoT-SC → ReAct**: Start with CoT self-consistency; if confidence is low, switch to ReAct
- **ReAct → CoT-SC**: Start with ReAct; if stuck in loop, fall back to CoT self-consistency

### Model Size Effects
- Smaller models: better at direct answering than CoT/ReAct
- Larger models: benefit more from CoT and ReAct
- **Fine-tuning insight**: When fine-tuned on successful trajectories, ReAct/Act **clearly beat** CoT-based training

### Key Finding
> When information needed is **not in the model's memory**, ReAct is always better than CoT.

---

## 3. SWE-bench & SWE-agent

### SWE-bench
- **2,294 GitHub issues** from 12 Python repositories
- Input: issue description + full codebase (at commit before issue)
- Goal: generate a **pull request** that fixes the issue
- Evaluation: new test cases pass AND old test cases don't break

### Challenge
Codebase is too large to fit in LLM context → need **localization**.

### Approaches to Localization

| Approach | Method | SWE-bench % |
|----------|--------|-------------|
| RAG only | Retrieve top-k code snippets | 1.31% |
| Shell commands | ls, cat, grep as actions | Better than RAG |
| **SWE-agent** | LLM-friendly custom actions | **12.47%** |

### SWE-agent: LLM-Friendly Action Space

| Action | Description |
|--------|-------------|
| `open [file]` | Open file, show first 100 lines |
| `scroll_down` / `scroll_up` | Navigate within file |
| `goto [line]` | Jump to specific line |
| `search_file [query]` | Search within current file |
| `search_dir [query]` | Search within directory |
| `find_file [name]` | Locate file in repo |
| `edit [start:end]` | Replace lines start–end with new code |
| `create [file]` | Create new file |
| `submit` | Bundle all edits into a PR |

### Why "LLM-Friendly"?
- Files displayed in **100-line windows** (avoids long-context attention issues)
- Line numbers always shown (needed for `edit` action)
- Metadata lines: "255 lines above" / "180 lines below" (aids navigation)

### Prompt Structure for SWE-agent
1. **System prompt**: Describe role, action space, output format
2. **Few-shot examples**: Successful trajectories
3. **Current issue**: GitHub issue text + instructions
4. **ReAct loop**: Thought → Action → Observation → ...
5. **Submit**: Bundle all `edit` actions into final PR

---

## 4. Self-Refine

### Core Idea
Iteratively improve output using self-generated feedback — **no external tools needed**.

### Three Required Prompts

| Prompt | Purpose |
|--------|---------|
| **Generation** | Produce initial output at $t=0$ |
| **Feedback** | Evaluate output along specified dimensions |
| **Refine** | Improve output using feedback history |

### Algorithm
```
y_0 = M(generation_prompt, x)
for t = 1, 2, ...:
    feedback_t = M(feedback_prompt, x, y_{t-1})
    if stopping_criteria(feedback_t, t): break
    y_t = M(refine_prompt, x, {y_0, feedback_1, ..., y_{t-1}, feedback_t})
return y_t
```

### Critical Design Choice
**Refine receives ALL previous outputs + feedbacks** (not just the latest):
- Prevents cycling: e.g., iteration 1 fixes engagement but breaks informativeness, iteration 2 does the reverse
- Full history lets model avoid repeating past mistakes

### Example: Dialogue Response

| Stage | Output |
|-------|--------|
| **Initial** | "I'm sure it's a great way to socialize and stay active." |
| **Feedback** | Engagement: no info about table tennis. Understanding: lacks user context. |
| **Refined** | "That's great! It requires quick reflexes and hand-eye coordination. Have you played before?" |

### Example: Code Optimization

| Stage | Output |
|-------|--------|
| **Initial** | Brute-force loop: `sum += i` |
| **Feedback** | "Code is slow, uses brute force. Use formula." |
| **Refined** | `return n * (n + 1) // 2` |

### Results
- **Monotonic improvement** with more iterations
- Largest gain at iteration 1; diminishing returns after
- Use cost-benefit analysis to decide iteration count

---

## 5. Reflexion

### Extends Self-Refine + ReAct with Explicit Memory

### Three Components

| Component | Role | Can be... |
|-----------|------|-----------|
| **Actor** ($M_A$) | Generate output / take actions | LLM with single prompt |
| **Evaluator** ($M_E$) | Assess output quality | LLM, rule-based, unit tests, external reward |
| **Self-Reflector** ($M_{SR}$) | Analyze what went wrong, suggest fixes | LLM |

### Key Innovation: Evaluator ≠ Reflector
- **Self-Refine**: Single feedback prompt does both evaluation and reflection
- **Reflexion**: Separates them — evaluator gives binary/numeric signal, reflector provides textual analysis

### Memory Architecture

| Memory Type | Contents | Scope |
|-------------|----------|-------|
| **Short-term** | Current trajectory (thought-action-observation steps) | Single episode |
| **Long-term (Experience)** | Self-reflection texts from all past episodes | Across episodes |

### Algorithm
```
for trial = 1, 2, ...:
    trajectory = actor.run(task, experience_memory)
    reward = evaluator.evaluate(trajectory)
    if reward >= threshold: break
    reflection = self_reflector.reflect(trajectory, reward)
    experience_memory.append(reflection)
```

### Policy as (Weights + Memory)
- Unlike RL where policy updates model weights
- Reflexion keeps weights **frozen** — only **memory** changes between iterations
- Memory acts as a "learned" component of the policy

### Results (HotpotQA)

| Method | Accuracy |
|--------|----------|
| CoT only | ~0.30 |
| ReAct only | ~0.30 |
| CoT + Reflexion | Higher |
| **ReAct + Reflexion** | **Highest** |

- ReAct + Reflexion clearly dominates when evaluator signal is available
- **Caveat**: Evaluator in experiments uses ground truth (binary correct/incorrect) — not always available in practice

---

## 6. Summary

| Paper | Key Contribution | When to Use |
|-------|-----------------|-------------|
| **ReAct** | Reasoning + Acting in unified loop | When external information is needed |
| **Self-Refine** | Iterative self-improvement via feedback | When quality can be self-assessed |
| **Reflexion** | Memory-augmented self-improvement with separate evaluator | When external reward signal exists |

### Impact on the Ecosystem
- These papers set the **foundational components** for agentic workflows
- LangGraph and similar frameworks directly implement these patterns
- Every modern agentic system uses some combination of: ReAct loops, self-refinement, memory, evaluator feedback
